from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cadgen.config import load_settings
from cadgen.freecad_mcp import (
    FreeCADMCPError,
    FreeCADValidationError,
    assess_freecad_validation,
)
from cadgen.freecad_script import (
    ADJACENT_INTERFACE_POLICY_ID,
    GENERATOR_VERSION,
    VALIDATION_SCHEMA_VERSION,
    VALIDATOR_POLICY_DIGEST,
    VALIDATOR_POLICY_ID,
    build_freecad_script,
    geometry_payload,
)
from cadgen.schemas import GlobalSpec, Goal, IntentResult
from cadgen.state import StateEngine


def _result(evidence: dict) -> dict:
    return {
        "content": [
            {
                "type": "text",
                "text": "CADGEN_VALIDATION="
                + json.dumps(evidence, separators=(",", ":")),
            }
        ]
    }


def _evidence(module_ids: list[str]) -> dict:
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "validator_policy": {
            "policy_id": VALIDATOR_POLICY_ID,
            "policy_digest": VALIDATOR_POLICY_DIGEST,
            "generator_version": GENERATOR_VERSION,
            "validation_schema_version": VALIDATION_SCHEMA_VERSION,
        },
        "payload_digest": "digest",
        "state_id": "state",
        "module_ids": module_ids,
        "checks": {
            "assembly": {
                "passed": True,
                "bounds": {
                    "minimum": [0.0, -10.0, -10.0],
                    "maximum": [100.0, 10.0, 10.0],
                },
            },
            "outer_network": {"passed": True},
            "bore_network": {"passed": True},
            "modules": {module_id: {"passed": True} for module_id in module_ids},
            "centerlines": {module_id: {"passed": True} for module_id in module_ids},
            "module_errors": [],
            "assembly_errors": [],
            "non_adjacent_overlaps": [],
            "adjacent_interface_overlaps": [],
            "connection_failures": [],
            "terminal_bore_failures": [],
            "anchored_inlet_bore_failures": [],
            "termination_seal_failures": [],
            "wall_section_failures": [],
            "sampled_internal_section_count": len(module_ids),
            "minimum_authored_wall_thickness": 2.0,
            "declared_downstream_open_port_count": 1,
            "anchored_inlet_count": 1,
        },
        "passed": True,
    }


def _assess(evidence: dict) -> dict:
    return assess_freecad_validation(
        _result(evidence),
        expected_digest="digest",
        expected_state_id="state",
        expected_module_ids=evidence["module_ids"],
    )


def _adjacent_overlap() -> dict:
    return {
        "module_ids": ["M1", "M2"],
        "parent_module_id": "M1",
        "child_module_id": "M2",
        "parent_child_binding": {
            "child_input_name": "inlet",
            "parent_port_id": "M1.outlet",
        },
        "common_volume": 182.6882574752523,
        "outside_interface_volume": 0.0,
        "outside_interface_allowance": 1.826882574752523e-6,
        "interface_upstream_depth": 10.004,
        "interface_downstream_depth": 0.004,
        "minimum_outlet_forward_dot": 0.0,
        "policy": "resolver_local_interface_band",
        "policy_id": ADJACENT_INTERFACE_POLICY_ID,
        "policy_digest": VALIDATOR_POLICY_DIGEST,
        "generator_version": GENERATOR_VERSION,
        "common_bounds": {
            "minimum": [20.0, -10.0, -10.0],
            "maximum": [30.0, 10.0, 10.0],
        },
        "common_centroid": [25.0, 0.0, 0.0],
    }


def test_generated_v23_payload_and_script_bind_validator_policy():
    state = StateEngine(load_settings(Path("missing.env"))).initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[
                Goal(goal_id="G1", type="move", direction="+X", length=10.0)
            ],
            expected_open_ports=1,
            expected_open_ports_source="explicit",
        )
    )
    payload = geometry_payload(state)
    script = build_freecad_script(state)

    assert payload["validator_policy_id"] == VALIDATOR_POLICY_ID
    assert payload["validator_policy_digest"] == VALIDATOR_POLICY_DIGEST
    assert len(VALIDATOR_POLICY_DIGEST) == 64
    assert '"validator_policy": {' in script
    assert VALIDATOR_POLICY_DIGEST in script
    compile(script, "generated_freecad.py", "exec")


def test_policy_metadata_is_legacy_optional_but_strict_when_present():
    legacy = _evidence(["M1"])
    legacy.pop("validator_policy")
    legacy["checks"].pop("adjacent_interface_overlaps")
    assert _assess(legacy)["passed"] is True

    partial = _evidence(["M1"])
    partial["validator_policy"].pop("policy_digest")
    with pytest.raises(FreeCADMCPError, match="policy metadata"):
        _assess(partial)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda item: item.update(module_ids=["M2", "M1"]),
        lambda item: item["parent_child_binding"].update(parent_port_id="M2.outlet"),
        lambda item: item.update(policy_id="spoofed-policy"),
        lambda item: item.update(outside_interface_volume=1.0),
        lambda item: item.update(common_centroid=[200.0, 0.0, 0.0]),
    ],
)
def test_adjacent_interface_overlap_is_strictly_validated(mutate):
    evidence = _evidence(["M1", "M2"])
    evidence["checks"]["adjacent_interface_overlaps"] = [_adjacent_overlap()]
    assert _assess(evidence)["passed"] is True

    malformed = copy.deepcopy(evidence)
    mutate(malformed["checks"]["adjacent_interface_overlaps"][0])
    with pytest.raises(FreeCADMCPError):
        _assess(malformed)


def test_structured_module_error_preserves_text_and_rejects_partial_shape():
    evidence = _evidence(["M1"])
    evidence["checks"]["module_errors"] = [
        {
            "module_id": "M1",
            "error": "exact compact junction fillet failed: kernel text",
            "failure_code": "JUNCTION_OUTER_FILLET_FAILED",
            "stage": "fillet_compact_junction_material",
            "details": {
                "module_type": "junction",
                "surface": "outer",
                "radius": 2.0,
                "edge_center": [30.0, 0.0, 10.0],
                "edge_length": 0.1,
                "kernel_code": "NO_SUITABLE_EDGES",
            },
        }
    ]
    evidence["passed"] = False
    with pytest.raises(FreeCADValidationError) as caught:
        _assess(evidence)
    assert caught.value.evidence["checks"]["module_errors"][0]["error"].endswith(
        "kernel text"
    )

    partial = copy.deepcopy(evidence)
    partial["checks"]["module_errors"][0].pop("stage")
    with pytest.raises(FreeCADMCPError, match="incomplete"):
        _assess(partial)
