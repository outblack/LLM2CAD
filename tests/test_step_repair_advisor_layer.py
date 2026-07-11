from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

from cadgen.config import load_settings
from cadgen.gemini_client import GeminiClient, gemini_json_schema
from cadgen.prompts import (
    step_repair_advisor_prompt,
    step_repair_advisor_system_instruction,
)
from cadgen.schemas import (
    DiagnosticBinding,
    Fact,
    FieldOwnership,
    GeometryValidationAdvisorResponse,
    LLMUsage,
    StepRepairDiagnosticContext,
)


def _diagnostic_context() -> StepRepairDiagnosticContext:
    return StepRepairDiagnosticContext(
        binding=DiagnosticBinding(
            run_id="run-1",
            state_id="S1",
            state_digest="a" * 64,
            contract_digest="b" * 64,
            step_index=2,
            attempt_index=1,
            action_digest="c" * 64,
            failure_signature="d" * 64,
            evidence_digest="e" * 64,
            generator_version="cadgen02-freecad-v23",
            validator_schema_version=3,
            validator_policy_digest="f" * 64,
            repair_epoch=0,
        ),
        issue_ids=["STEP_0002_01_FREECAD_GEOMETRY_VALIDATION_FAILED"],
        current_state={"state_id": "S1", "open_port_ids": ["M1.out"]},
        immutable_goal_slice={"goal_id": "junction_1", "type": "branch"},
        rejected_draft={
            "target_port": "M1.out",
            "module": "junction",
            "params": {"max_hub_radius": 12.0},
        },
        resolved_action=None,
        implicated_modules=[{"module_id": "M1"}, {"module_id": "M2"}],
        failed_checks=[
            {
                "check_name": "non_adjacent_overlaps",
                "adjacent": True,
                "common_volume": 182.688,
            }
        ],
        passed_check_summary=["assembly", "bore_network"],
        facts=[
            Fact(
                evidence_id="E_OVERLAP_1",
                kind="measurement",
                statement="The measured overlap was invariant across trials.",
                data={"metric_span": 0.0},
            )
        ],
        field_ownership=[
            FieldOwnership(
                path="/params/max_hub_radius",
                owner="planner_authored",
                mutable_in_current_repair=True,
                reason="The junction variant exposes this authored field.",
            )
        ],
        parameter_trials=[
            {
                "path": "/params/max_hub_radius",
                "values": [12.0, 11.0, 15.0],
                "metric_values": [182.688, 182.688, 182.688],
                "finding": "invariant_over_tested_range",
            }
        ],
        deterministic_recommendations=[],
        allowed_strategy_kinds=["mode_change", "validator_review"],
    )


def test_step_repair_advisor_prompt_is_exactly_one_typed_context_json():
    context = _diagnostic_context()

    prompt = step_repair_advisor_prompt(context)

    assert prompt.startswith("{") and prompt.endswith("}")
    assert json.loads(prompt) == context.model_dump(mode="json")
    assert "Current immutable state" not in prompt


def test_step_repair_advisor_system_instruction_enforces_authority_and_citations():
    instruction = step_repair_advisor_system_instruction()

    assert "Cite only evidence_id values" in instruction
    assert "planner_authored" in instruction
    assert "Never approve geometry" in instruction
    assert "validator_policy_mismatch" in instruction
    assert "ordinary step planner remains the sole author" in instruction
    assert "Do not echo" in instruction
    assert "bounded parameter range" in instruction
    assert "chooses every final exact value" in instruction


def test_geometry_advisor_provider_schema_stays_in_safe_grammar_subset():
    schema = gemini_json_schema(GeometryValidationAdvisorResponse)
    encoded = json.dumps(schema, separators=(",", ":"))

    assert len(encoded.encode("utf-8")) < 3_000
    assert len(schema.get("$defs", {})) == 1
    assert '"anyOf"' not in encoded
    assert '"null"' not in encoded

    def assert_all_object_fields_required(value):
        if isinstance(value, list):
            for item in value:
                assert_all_object_fields_required(item)
            return
        if not isinstance(value, dict):
            return
        if value.get("type") == "object":
            assert set(value.get("required", [])) == set(value.get("properties", {}))
        for item in value.values():
            assert_all_object_fields_required(item)

    assert_all_object_fields_required(schema)


class _TinyDiagnosis(BaseModel):
    value: int


def test_step_repair_advisor_uses_same_usage_ledger_but_never_lineage():
    settings = replace(
        load_settings(Path("missing.env")),
        gemini_api_key="test-only",
        gemini_stateful=True,
        gemini_max_calls=5,
        gemini_max_total_tokens=100_000,
    )

    class FakeInteractions:
        def __init__(self):
            self.calls = []

        def create(self, **body):
            self.calls.append(body)
            return SimpleNamespace(
                id=f"advisor-{len(self.calls)}",
                status="completed",
                output_text='{"value":1}',
                usage=SimpleNamespace(
                    total_input_tokens=10,
                    total_cached_tokens=0,
                    total_output_tokens=3,
                    total_thought_tokens=1,
                    total_tool_use_tokens=0,
                    total_tokens=14,
                ),
            )

    interactions = FakeInteractions()
    client = GeminiClient.__new__(GeminiClient)
    client._settings = settings
    client._client = SimpleNamespace(interactions=interactions)
    client._lineages = {}
    client._usage = LLMUsage()

    for _ in range(2):
        result = client.stream_structured(
            "{}",
            _TinyDiagnosis,
            part="step_repair_advisor",
            thinking_level="medium",
        )
        assert result.value == 1

    assert len(interactions.calls) == 2
    for body in interactions.calls:
        assert body["model"] == settings.model_for("step_repair_advisor")
        assert body["store"] is False
        assert "previous_interaction_id" not in body
        assert body["generation_config"]["max_output_tokens"] == 4096
    assert client.lineage_snapshot() == {}
    assert client.usage_snapshot().calls == 2
