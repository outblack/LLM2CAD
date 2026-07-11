from __future__ import annotations

import ast
import asyncio
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from cadgen.config import load_settings
import cadgen.freecad_app as freecad_app
import cadgen.freecad_mcp as freecad_mcp
import cadgen.pipeline as pipeline
from cadgen.freecad_app import FreeCADLaunchError, ensure_freecad_open
from cadgen.freecad_mcp import FreeCADMCPError
from cadgen.freecad_script import build_freecad_script
from cadgen.gemini_client import GeminiInvalidRequestError
from cadgen.local_heuristic import infer_intent, plan_next_action
from cadgen.pipeline import run_pipeline
from cadgen.prompts import compact_planner_payload, step_planner_prompt
from cadgen.registry import validate_action
from cadgen.schemas import (
    ActionDraft,
    CorePlannerDecision,
    CorePlannerDecisionWire,
    GlobalSpec,
    Goal,
    IntentResult,
    ModuleRef,
    PipeState,
    PlannerDecision,
    Port,
    ResolvedAction,
)
from cadgen.state import StateEngine
from cadgen.static_validation import (
    StaticValidationError,
    build_final_critic_report,
    build_step_verification,
)
from cadgen.stream import ThinkingStream


def _successful_freecad_result(code: str) -> dict:
    if "CADGEN_READY=" in code:
        evidence = {"passed": True, "protocol": 1}
        prefix = "CADGEN_READY="
    elif "CADGEN_VALIDATION=" in code:
        payload_line = next(
            line
            for line in code.splitlines()
            if line.startswith("PAYLOAD = json.loads(")
        )
        payload_json = ast.literal_eval(payload_line[len("PAYLOAD = json.loads(") : -1])
        payload = json.loads(payload_json)
        digest_line = next(
            line for line in code.splitlines() if line.startswith("PAYLOAD_DIGEST = ")
        )
        digest = ast.literal_eval(digest_line.split(" = ", 1)[1])
        assignments = {}
        for name in (
            "CANDIDATE_DOCUMENT",
            "GENERATOR_VERSION",
            "RUN_ID",
            "ATTEMPT_ID",
        ):
            line = next(
                item for item in code.splitlines() if item.startswith(name + " = ")
            )
            assignments[name] = ast.literal_eval(line.split(" = ", 1)[1])
        module_ids = [module["id"] for module in payload["modules"]]
        sampled_module_ids = [
            module["id"]
            for module in payload["modules"]
            if module["type"] not in {"terminate", "cap_pipe"}
        ]
        evidence = {
            "schema_version": 3,
            "generator_version": assignments["GENERATOR_VERSION"],
            "run_id": assignments["RUN_ID"],
            "state_version": payload["state_version"],
            "attempt_id": assignments["ATTEMPT_ID"],
            "candidate_document": assignments["CANDIDATE_DOCUMENT"],
            "candidate_shape_fingerprints": {
                "PipeAssembly": "0" * 64,
                **{f"solid_{module_id}": "0" * 64 for module_id in module_ids},
            },
            "payload_digest": digest,
            "state_id": payload["state_id"],
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
                "modules": {
                    module["id"]: {"passed": True} for module in payload["modules"]
                },
                "centerlines": {
                    module["id"]: {"passed": True} for module in payload["modules"]
                },
                "module_errors": [],
                "assembly_errors": [],
                "non_adjacent_overlaps": [],
                "connection_failures": [],
                "terminal_bore_failures": [],
                "anchored_inlet_bore_failures": [],
                "termination_seal_failures": [],
                "wall_section_failures": [],
                "sampled_internal_section_count": len(sampled_module_ids),
                "sampled_internal_sections_by_module": {
                    module_id: 1 for module_id in sampled_module_ids
                },
                "required_internal_section_module_count": len(sampled_module_ids),
                "minimum_authored_wall_thickness": 2.0,
                "declared_downstream_open_port_count": len(payload["open_ports"]),
                "anchored_inlet_count": 1 if payload["modules"] else 0,
            },
            "passed": True,
        }
        prefix = "CADGEN_VALIDATION="
    elif "CADGEN_PUBLISH=" in code:
        meta_line = next(
            line for line in code.splitlines() if line.startswith("META = json.loads(")
        )
        meta_json = ast.literal_eval(meta_line[len("META = json.loads(") : -1])
        meta = json.loads(meta_json)
        artifact = Path(meta["fcstd_path"])
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"fake-fcstd")
        evidence = {"passed": True, "saved": True, **meta}
        prefix = "CADGEN_PUBLISH="
    else:
        raise AssertionError("unexpected FreeCAD script")
    return {
        "content": [
            {
                "type": "text",
                "text": prefix + json.dumps(evidence, separators=(",", ":")),
            }
        ]
    }


def _production_line_draft(
    *,
    target_port: str = "START",
    length: float = 20.0,
    goal_id: str = "G1",
) -> ActionDraft:
    """프로덕션 경계를 통과하는 최소 schema-v2 직선 action fixture."""

    return ActionDraft(
        target_port=target_port,
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=[goal_id],
        completed_goal_ids=[goal_id],
        params={
            "section_source": "inherit_target",
            "path_kind": "line",
            "length": length,
            "direction": (1.0, 0.0, 0.0),
        },
    )


def test_model_env_overrides(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("GEMINI_INTENT_MODEL", "gemini-3.5-pro")
    settings = load_settings(Path("missing.env"))

    assert settings.model_for("text") == "gemini-3.5-flash"
    assert settings.model_for("intent") == "gemini-3.5-pro"


def test_step_repair_advisor_has_dedicated_model_policy_and_safe_defaults(
    monkeypatch,
):
    settings = load_settings(Path("missing.env"))

    assert settings.model_for("intent") == "gemini-3.1-flash-lite"
    assert settings.model_for("step_planner") == "gemini-3.5-flash"
    assert settings.model_for("step_repair_advisor") == "gemini-3.1-pro-preview"
    assert (
        len(
            {
                settings.model_for("intent"),
                settings.model_for("step_planner"),
                settings.model_for("step_repair_advisor"),
            }
        )
        == 3
    )
    assert settings.step_repair_advisor_enabled is True
    assert settings.step_repair_advisor_required is True
    assert settings.step_repair_advisor_trigger_attempt == 1
    assert settings.step_repair_advisor_max_calls_per_step == 3
    assert settings.step_repair_advisor_max_signatures_per_step == 3
    assert settings.step_repair_advisor_probe_limit == 0
    assert settings.output_token_limit_for("step_repair_advisor") == 4096

    monkeypatch.setenv("GEMINI_STEP_REPAIR_ADVISOR_MODEL", "diagnostician-model")
    monkeypatch.setenv("CADGEN_STEP_REPAIR_ADVISOR_ENABLED", "false")
    monkeypatch.setenv("CADGEN_STEP_REPAIR_ADVISOR_REQUIRED", "false")
    monkeypatch.setenv("CADGEN_STEP_REPAIR_ADVISOR_TRIGGER_ATTEMPT", "3")
    monkeypatch.setenv("CADGEN_STEP_REPAIR_ADVISOR_MAX_CALLS_PER_STEP", "2")
    monkeypatch.setenv("CADGEN_STEP_REPAIR_ADVISOR_MAX_SIGNATURES_PER_STEP", "4")
    monkeypatch.setenv("CADGEN_STEP_REPAIR_ADVISOR_PROBE_LIMIT", "1")
    monkeypatch.setenv(
        "CADGEN_GEMINI_STEP_REPAIR_ADVISOR_MAX_OUTPUT_TOKENS",
        "3072",
    )
    overridden = load_settings(Path("missing.env"))

    assert overridden.model_for("step_repair_advisor") == "diagnostician-model"
    assert overridden.step_repair_advisor_enabled is False
    assert overridden.step_repair_advisor_required is False
    assert overridden.step_repair_advisor_trigger_attempt == 3
    assert overridden.step_repair_advisor_max_calls_per_step == 2
    assert overridden.step_repair_advisor_max_signatures_per_step == 4
    assert overridden.step_repair_advisor_probe_limit == 1
    assert overridden.output_token_limit_for("step_repair_advisor") == 3072


def test_skip_freecad_keeps_static_inverse_advisor_enabled():
    settings = load_settings(Path("missing.env"))

    skipped = settings.with_overrides(skip_freecad=True)

    assert skipped.step_repair_advisor_enabled is True
    assert skipped.freecad_mcp_enabled is False


def test_step_repair_advisor_rejects_invalid_limits():
    settings = load_settings(Path("missing.env"))

    with pytest.raises(ValueError, match="TRIGGER_ATTEMPT"):
        replace(settings, step_repair_advisor_trigger_attempt=0)
    with pytest.raises(ValueError, match="MAX_CALLS_PER_STEP"):
        replace(settings, step_repair_advisor_max_calls_per_step=-1)
    with pytest.raises(ValueError, match="MAX_SIGNATURES_PER_STEP"):
        replace(settings, step_repair_advisor_max_signatures_per_step=-1)
    with pytest.raises(ValueError, match="PROBE_LIMIT"):
        replace(settings, step_repair_advisor_probe_limit=-1)


def test_intent_output_token_limit_has_independent_default_and_override(monkeypatch):
    settings = load_settings(Path("missing.env"))
    assert settings.gemini_max_output_tokens == 16384
    assert settings.gemini_intent_max_output_tokens == 16384
    assert settings.output_token_limit_for("step_planner") == 16384
    assert settings.output_token_limit_for("intent") == 16384

    monkeypatch.setenv("CADGEN_GEMINI_INTENT_MAX_OUTPUT_TOKENS", "8192")
    overridden = load_settings(Path("missing.env"))
    assert overridden.output_token_limit_for("intent") == 8192
    assert overridden.output_token_limit_for("step_planner") == 16384


def test_retry_defaults_are_generous_and_global_call_cap_is_retry_aware():
    settings = load_settings(Path("missing.env"))

    assert settings.intent_repair_attempts == 3
    assert settings.step_repair_attempts == 6
    assert settings.gemini_history_max_turns == 32
    assert settings.gemini_history_token_threshold == 48_000
    minimum_full_replay_budget = (
        settings.intent_repair_attempts
        + 1
        + 2
        * (settings.final_repair_rounds + 1)
        * settings.max_iter
        * (settings.step_repair_attempts + 1)
        + (settings.final_repair_rounds + 1)
        * settings.max_iter
        * settings.step_repair_advisor_max_calls_per_step
        * 2
    )
    assert settings.gemini_max_calls >= minimum_full_replay_budget
    assert settings.gemini_max_total_tokens == 1_000_000


def test_freecad_mcp_timeout_env_override(monkeypatch):
    monkeypatch.setenv("CADGEN_FREECAD_MCP_TIMEOUT_SEC", "7.5")
    settings = load_settings(Path("missing.env"))

    assert settings.freecad_mcp_timeout_sec == 7.5


def test_gemini_request_timeout_env_override(monkeypatch):
    monkeypatch.setenv("CADGEN_GEMINI_REQUEST_TIMEOUT_SEC", "12.5")
    settings = load_settings(Path("missing.env"))

    assert settings.gemini_request_timeout_sec == 12.5


def test_step_mcp_env_override(monkeypatch):
    monkeypatch.setenv("CADGEN_FREECAD_STEP_MCP_ENABLED", "false")
    settings = load_settings(Path("missing.env"))

    assert settings.freecad_step_mcp_enabled is False


def test_invalid_boolean_and_nonpositive_limits_fail_configuration(monkeypatch):
    monkeypatch.setenv("CADGEN_FREECAD_MCP_ENABLED", "tru")
    with pytest.raises(ValueError, match="explicit boolean"):
        load_settings(Path("missing.env"))
    monkeypatch.delenv("CADGEN_FREECAD_MCP_ENABLED")

    with pytest.raises(ValueError, match="MAX_ITER"):
        load_settings(Path("missing.env")).with_overrides(max_iter=0)


def test_action_budget_uses_soft_baseline_and_explicit_override_is_hard(monkeypatch):
    monkeypatch.delenv("CADGEN_MAX_ITER", raising=False)
    monkeypatch.delenv("CADGEN_MAX_ITER_HARD_CEILING", raising=False)
    settings = load_settings(Path("missing.env"))

    assert settings.max_iter == 12
    assert settings.max_iter_hard_ceiling == 64
    assert settings.max_iter_is_hard_limit is False

    explicit = settings.with_overrides(max_iter=20)
    assert explicit.max_iter == 20
    assert explicit.max_iter_hard_ceiling == 20
    assert explicit.max_iter_is_hard_limit is True

    monkeypatch.setenv("CADGEN_MAX_ITER", "12")
    monkeypatch.setenv("CADGEN_MAX_ITER_HARD_CEILING", "8")
    with pytest.raises(ValueError, match="must not exceed"):
        load_settings(Path("missing.env"))


def test_korean_dry_run_generates_pipe_modules(tmp_path):
    settings = load_settings(Path("missing.env")).with_overrides(
        output_dir=tmp_path,
        skip_freecad=True,
    )
    prompt = "지름 20mm hollow pipe로 100mm 직진하고, 90도 위로 꺾은 다음 80mm 올라가게 만들어줘"

    report = run_pipeline(
        prompt,
        settings,
        dry_run=True,
        stream=ThinkingStream(enabled=False),
    )

    run_dir = Path(report.artifacts.output_dir)
    intent = json.loads((run_dir / "intent.json").read_text(encoding="utf-8"))
    actions = json.loads((run_dir / "actions.json").read_text(encoding="utf-8"))
    step_checks = json.loads(
        (run_dir / "step_verification.json").read_text(encoding="utf-8")
    )
    critic = json.loads((run_dir / "critic_report.json").read_text(encoding="utf-8"))
    run_report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))

    assert [goal["type"] for goal in intent["target_behavior"]] == [
        "move",
        "turn",
        "move",
    ]
    assert intent["target_behavior"][1]["angle"] == 90.0
    assert [action["module"] for action in actions] == [
        "straight_pipe",
        "bend_pipe",
        "straight_pipe",
    ]
    assert (run_dir / "step_verification.json").exists()
    assert (run_dir / "critic_report.json").exists()
    assert run_report["verification_status"] == "partial"
    assert run_report["status"] == "partial"
    assert run_report["critic_passed"] is True
    assert step_checks[0]["transition"]["produced_module_id"] == "M1"
    assert step_checks[0]["status"] == "passed"
    assert step_checks[0]["mcp_status"] == "skipped"
    assert step_checks[0]["skipped_mcp_reason"] == "Dry-run skips step FreeCAD MCP."
    assert critic["passed"] is True
    assert critic["view_requests"][0]["evidence_status"] == "pending"
    artifact_statuses = {item.name: item for item in report.artifact_statuses}
    assert artifact_statuses["run_report"].status == "available"
    assert artifact_statuses["critic_report"].status == "available"
    assert artifact_statuses["mcp_result"].status == "unavailable"


def test_four_port_prompt_sets_explicit_open_port_count():
    settings = load_settings(Path("missing.env"))
    intent = infer_intent(
        "Create a four-port manifold with four open ends and visible hollow rims.",
        settings,
    )

    assert intent.expected_open_ports == 4
    assert intent.expected_open_ports_source == "explicit"
    assert intent.target_behavior[0].type == "branch"
    assert intent.target_behavior[0].branch_count == 4
    assert intent.target_behavior[0].include_primary_outlet is False
    assert intent.target_behavior[0].required_outlet_vectors == [
        (-1.0, 0.0, 1.0),
        (-1.0, 0.0, -1.0),
        (1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0),
    ]


def test_goal_schema_preserves_explicit_junction_topology():
    goal = Goal(
        type="branch",
        branch_count=2,
        required_outlet_vectors=[(-1.0, 0.0, 1.0), (1.0, 0.0, 1.0)],
        include_primary_outlet=False,
        junction_style="smooth_hub",
    )

    payload = goal.model_dump(mode="json")

    assert payload["required_outlet_vectors"] == [
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ]
    assert payload["include_primary_outlet"] is False
    assert payload["junction_style"] == "smooth_hub"


def test_step_planner_prompt_uses_compact_catalog_not_full_state():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    state = engine.initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[
                Goal(type="turn", direction="+Z", angle=90.0),
                Goal(type="move", direction="+Z", length=80.0),
            ],
        )
    )
    bend = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="bend_pipe",
            params={"angle": 90.0, "turn_direction": "+Z"},
        ),
        state,
    )
    state = engine.apply_action(bend, state)
    leaked_sentinel = "DO_NOT_LEAK_FULL_PIPE_STATE"
    state = state.model_copy(
        update={
            "placed_modules": [
                *state.placed_modules,
                ModuleRef(
                    id="LEAKY",
                    type="straight_pipe",
                    geometry_id=leaked_sentinel,
                    params={
                        "debug_marker": leaked_sentinel,
                        "path_points": [leaked_sentinel],
                    },
                ),
            ],
            "used_ports": [leaked_sentinel],
            "action_history": [
                ResolvedAction(
                    action_id=leaked_sentinel,
                    target_port="OLD_PORT",
                    module="straight_pipe",
                    params={"debug_marker": leaked_sentinel},
                )
            ],
        }
    )

    prompt = step_planner_prompt(state)
    payload = compact_planner_payload(state)
    payload_json = json.dumps(payload, ensure_ascii=False)

    assert set(payload) == {
        "global_spec",
        "contract",
        "state_id",
        "contract_digest",
        "pending_goals",
        "open_ports",
        "graph",
        "spatial",
        "module_catalog",
    }
    assert "module_catalog" in payload
    assert "open_ports" in payload
    assert payload["pending_goals"][0]["type"] == "move"
    assert "angle" not in payload["pending_goals"][0]
    assert "notes" not in payload["pending_goals"][0]
    assert {module["id"] for module in payload["module_catalog"]} == {
        "route",
        "transition",
        "junction",
        "connect_ports",
        "terminate",
    }
    route_catalog = next(
        module for module in payload["module_catalog"] if module["id"] == "route"
    )
    assert "direction" in route_catalog["authored_params"]
    assert "length" in route_catalog["authored_params"]
    assert "section_source" in route_catalog["authored_params"]
    assert "start_position" not in route_catalog["authored_params"]
    assert "axis" not in route_catalog["authored_params"]
    for blocked_text in [
        "Selection rules",
        "move -> straight_pipe",
        "turn -> bend_pipe",
        "branch -> junction_pipe",
        "diameter_change -> reducer_pipe",
        "end -> cap_pipe",
        "connector -> connector_pipe",
    ]:
        assert blocked_text not in prompt
    for leaked_text in [
        leaked_sentinel,
        "placed_modules",
        "used_ports",
        "action_history",
        "path_points",
        "geometry_id",
    ]:
        assert leaked_text not in prompt
        assert leaked_text not in payload_json
    assert "placed_modules" not in prompt
    assert "action_history" not in prompt
    assert "path_points" not in prompt
    assert "module_catalog" in prompt
    assert "open_ports" in prompt


def test_compact_payload_includes_branch_port_provenance():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    state = engine.initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[
                Goal(type="branch", branch_count=2, branch_angles=[45.0, -45.0])
            ],
        )
    )
    branch = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="junction_pipe",
            params={"branch_count": 2, "branch_angles": [45.0, -45.0]},
        ),
        state,
    )
    state = engine.apply_action(branch, state)

    ports = compact_planner_payload(state)["open_ports"]
    by_name = {port["port_name"]: port for port in ports}

    assert by_name["out"]["source_module_id"] == "M1"
    assert by_name["out"]["port_role"] == "primary_outlet"
    assert by_name["out_1"]["source_module_id"] == "M1"
    assert by_name["out_1"]["port_role"] == "branch_outlet"
    assert by_name["out_2"]["source_module_id"] == "M1"
    assert by_name["out_2"]["port_role"] == "branch_outlet"


def test_compact_payload_marks_junction_outputs_dynamic():
    settings = load_settings(Path("missing.env"))
    state = StateEngine(settings).initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[
                Goal(
                    type="branch",
                    branch_count=4,
                    required_outlet_vectors=[
                        (-1.0, 0.0, 1.0),
                        (-1.0, 0.0, -1.0),
                        (1.0, 0.0, 1.0),
                        (1.0, 0.0, -1.0),
                    ],
                    include_primary_outlet=False,
                )
            ],
        )
    )

    junction_catalog = next(
        module
        for module in compact_planner_payload(state)["module_catalog"]
        if module["id"] == "junction"
    )

    assert junction_catalog["outputs"] == "dynamic"
    assert "outlets" in junction_catalog["authored_params"]
    assert "blend_radius" in junction_catalog["authored_params"]
    assert "branch_count" not in junction_catalog["authored_params"]


def test_resolver_ignores_llm_supplied_engine_owned_geometry():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    state = engine.initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[Goal(type="move", direction="+X", length=30.0)],
        )
    )

    action = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="straight_pipe",
            params={
                "length": 30.0,
                "start_position": (999.0, 999.0, 999.0),
                "axis": (0.0, 1.0, 0.0),
                "outer_diameter": 999.0,
                "wall_thickness": 100.0,
                "path_points": [(999.0, 999.0, 999.0)],
            },
        ),
        state,
    )

    assert action.params["start_position"] == (0.0, 0.0, 0.0)
    assert action.params["axis"] == (1.0, 0.0, 0.0)
    assert action.params["outer_diameter"] == 20.0
    assert action.params["wall_thickness"] == 2.0
    assert "path_points" not in action.params


def test_non_dry_run_pipeline_keeps_gemini_module_selection(monkeypatch, tmp_path):
    settings = load_settings(Path("missing.env")).with_overrides(
        output_dir=tmp_path,
        skip_freecad=True,
    )

    def fail_local_heuristic(unused_state):
        raise AssertionError("local heuristic should not run outside dry-run")

    class FakeGeminiClient:
        calls = []

        def __init__(self, unused_settings):
            self.settings = unused_settings

        def stream_structured(self, prompt, schema, *, part):
            self.calls.append((prompt, schema, part))
            if part == "intent":
                return IntentResult(
                    global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
                    target_behavior=[Goal(type="move", direction="+X", length=40.0)],
                )
            return PlannerDecision.model_validate(
                {
                    "catalog_schema_version": 2,
                    "target_port": "START",
                    "choice": {
                        "module": "route",
                        "params": {
                            "path_kind": "line",
                            "section_source": "inherit_target",
                            "length": 40.0,
                            "direction": [1.0, 0.0, 0.0],
                        },
                    },
                    "affected_goal_ids": ["G1"],
                    "completed_goal_ids": ["G1"],
                    "rationale": "Gemini chose the route primitive for this connection.",
                }
            )

    monkeypatch.setattr(pipeline, "plan_next_action", fail_local_heuristic)
    monkeypatch.setattr(pipeline, "GeminiClient", FakeGeminiClient)

    report = run_pipeline(
        "use the most suitable primitive for this small connection",
        settings,
        dry_run=False,
        stream=ThinkingStream(enabled=False),
    )
    actions = json.loads(
        Path(report.artifacts.actions_path).read_text(encoding="utf-8")
    )

    assert actions[0]["module"] == "route"
    step_calls = [call for call in FakeGeminiClient.calls if call[2] == "step_planner"]
    assert len(step_calls) == 1
    prompt, schema, part = step_calls[0]
    assert schema is CorePlannerDecisionWire
    assert part == "step_planner"
    assert '"id":"route"' in prompt
    assert "straight_pipe" not in prompt


def test_non_dry_run_planner_rejects_nonzero_consumes_goal_index():
    settings = load_settings(Path("missing.env"))
    state = StateEngine(settings).initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[Goal(type="move", direction="+X", length=40.0)],
        )
    )

    class FakeGemini:
        def stream_structured(self, prompt, schema, *, part):
            return ActionDraft(
                target_port="START",
                module="straight_pipe",
                params={"length": 40.0},
                consumes_goal_index=1,
            )

    with pytest.raises(ValueError, match="consumes_goal_index=0"):
        pipeline._plan_action(state, dry_run=False, gemini=FakeGemini())


def test_non_dry_run_planner_rejects_legacy_action_before_defaults_can_apply():
    """LLM 누락값이 legacy resolver 기본값으로 바뀌는 경로를 차단한다."""

    settings = load_settings(Path("missing.env"))
    state = StateEngine(settings).initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[Goal(type="move", direction="+X", length=40.0)],
        )
    )

    class FakeGemini:
        def stream_structured(self, prompt, schema, *, part):
            del prompt, schema, part
            return ActionDraft(
                target_port="START",
                module="straight_pipe",
                params={},
            )

    with pytest.raises(ValueError, match="catalog_schema_version=2"):
        pipeline._plan_action(state, dry_run=False, gemini=FakeGemini())


def test_non_dry_run_step_mcp_uses_fake_executor_before_final_mcp(
    monkeypatch, tmp_path
):
    settings = replace(
        load_settings(Path("missing.env")),
        output_dir=tmp_path,
        freecad_auto_open=False,
        require_freecad_app=False,
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=True,
        freecad_mcp_required=True,
    )
    calls = []

    class FakeGeminiClient:
        def __init__(self, unused_settings):
            self.settings = unused_settings

        def stream_structured(self, prompt, schema, *, part):
            if part == "intent":
                return IntentResult(
                    global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
                    target_behavior=[Goal(type="move", direction="+X", length=40.0)],
                    expected_open_ports=1,
                    expected_open_ports_source="derived",
                )
            return _production_line_draft(length=40.0)

    async def fake_execute(unused_settings, code):
        calls.append(code)
        return _successful_freecad_result(code)

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGeminiClient)
    monkeypatch.setattr(pipeline, "execute_freecad_code", fake_execute)

    report = run_pipeline(
        "make one straight pipe",
        settings,
        dry_run=False,
        stream=ThinkingStream(enabled=False),
    )
    step_checks = json.loads(
        Path(report.artifacts.step_verification_path).read_text(encoding="utf-8")
    )

    assert len(calls) == 2
    assert step_checks[0]["mcp_status"] == "passed"
    assert step_checks[0]["mcp_result_path"].endswith("step_mcp/step_1_attempt_1.json")
    assert report.verification_status == "passed"
    assert report.top_issues == []
    assert report.freecad_mcp_used is True


def test_pipeline_static_failure_writes_partial_artifacts_and_skips_mcp(
    monkeypatch,
    tmp_path,
):
    settings = replace(
        load_settings(Path("missing.env")),
        output_dir=tmp_path,
        freecad_auto_open=False,
        require_freecad_app=False,
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=True,
        freecad_mcp_required=False,
    )
    calls = []

    class FakeGeminiClient:
        def __init__(self, unused_settings):
            self.settings = unused_settings

        def stream_structured(self, prompt, schema, *, part):
            if part == "intent":
                return IntentResult(
                    global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
                    target_behavior=[
                        Goal(
                            type="branch",
                            direction="-X",
                            branch_count=2,
                            branch_angles=[45.0, -45.0],
                            include_primary_outlet=False,
                        )
                    ],
                    expected_open_ports=2,
                    expected_open_ports_source="derived",
                )
            return ActionDraft(
                target_port="START",
                module="junction",
                catalog_schema_version=2,
                affected_goal_ids=["G1"],
                completed_goal_ids=["G1"],
                params={
                    "section_source": "inherit_target",
                    "outlets": [
                        {
                            "role": "branch",
                            "axis": [0.70710678, 0.70710678, 0.0],
                            "length": 20.0,
                            "outer_diameter": 20.0,
                            "wall_thickness": 2.0,
                        },
                        {
                            "role": "branch",
                            "axis": [0.70710678, -0.70710678, 0.0],
                            "length": 20.0,
                            "outer_diameter": 20.0,
                            "wall_thickness": 2.0,
                        },
                    ],
                    "blend_mode": "hard",
                    "max_hub_radius": 12.0,
                },
            )

    async def fake_execute(unused_settings, code):
        calls.append(code)
        return {"ok": True}

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGeminiClient)
    monkeypatch.setattr(pipeline, "execute_freecad_code", fake_execute)

    with pytest.raises(StaticValidationError) as exc:
        run_pipeline(
            "make a left branch",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
        )

    report_path = Path(exc.value.artifact_path)
    run_dir = report_path.parent
    report = json.loads(report_path.read_text(encoding="utf-8"))
    steps = json.loads((run_dir / "step_verification.json").read_text(encoding="utf-8"))
    critic = json.loads((run_dir / "critic_report.json").read_text(encoding="utf-8"))

    assert calls == []
    assert report["status"] == "failed"
    assert report["failed_stage"] == "static_step_validation"
    assert report["top_issues"] == ["STEP_0001_01_BRANCH_DIRECTION_MISMATCH"]
    assert steps[0]["status"] == "failed"
    assert steps[0]["issues"][0]["issue_code"] == "BRANCH_DIRECTION_MISMATCH"
    assert critic["passed"] is False
    assert critic["issues"][0]["module_id"] == "M1"


def test_pipeline_unknown_target_fails_at_draft_boundary(
    monkeypatch,
    tmp_path,
):
    settings = replace(
        load_settings(Path("missing.env")),
        output_dir=tmp_path,
        freecad_auto_open=False,
        require_freecad_app=False,
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=True,
    )
    calls = []

    class FakeGeminiClient:
        def __init__(self, unused_settings):
            self.settings = unused_settings

        def stream_structured(self, prompt, schema, *, part):
            if part == "intent":
                return IntentResult(
                    global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
                    target_behavior=[Goal(type="move", direction="+X", length=20.0)],
                    expected_open_ports=1,
                    expected_open_ports_source="derived",
                )
            return _production_line_draft(target_port="MISSING")

    async def fake_execute(unused_settings, code):
        calls.append(code)
        return {"ok": True}

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGeminiClient)
    monkeypatch.setattr(pipeline, "execute_freecad_code", fake_execute)

    with pytest.raises(StaticValidationError) as exc:
        run_pipeline(
            "make one straight pipe",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
        )

    report = json.loads(Path(exc.value.artifact_path).read_text(encoding="utf-8"))

    assert calls == []
    assert report["failed_stage"] == "draft_validation"
    assert report["top_issues"] == ["STEP_0001_01_DRAFT_VALIDATION_FAILED"]


def test_required_final_mcp_allows_step_mcp_to_be_disabled(monkeypatch, tmp_path):
    settings = replace(
        load_settings(Path("missing.env")),
        output_dir=tmp_path,
        freecad_auto_open=False,
        require_freecad_app=False,
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=False,
        freecad_mcp_required=True,
    )

    class FakeGeminiClient:
        def __init__(self, unused_settings):
            self.settings = unused_settings

        def stream_structured(self, prompt, schema, *, part):
            if part == "intent":
                return IntentResult(
                    global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
                    target_behavior=[Goal(type="move", direction="+X", length=20.0)],
                    expected_open_ports=1,
                    expected_open_ports_source="derived",
                )
            return _production_line_draft()

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGeminiClient)
    calls = []

    async def fake_execute(unused_settings, code):
        del unused_settings
        calls.append(code)
        return _successful_freecad_result(code)

    monkeypatch.setattr(pipeline, "execute_freecad_code", fake_execute)

    report = run_pipeline(
        "make one straight pipe",
        settings,
        dry_run=False,
        stream=ThinkingStream(enabled=False),
    )

    assert report.status == "success"
    assert report.verification_status == "passed"
    assert len(calls) == 2  # one final validation and one publish transaction


def test_required_freecad_launch_failure_exposes_report_path(monkeypatch, tmp_path):
    settings = replace(
        load_settings(Path("missing.env")),
        output_dir=tmp_path,
        freecad_auto_open=True,
        require_freecad_app=True,
        freecad_mcp_enabled=False,
    )

    def fail_launch(unused_settings, unused_stream):
        raise FreeCADLaunchError("cannot launch")

    monkeypatch.setattr(pipeline, "ensure_freecad_open", fail_launch)

    with pytest.raises(StaticValidationError) as exc:
        run_pipeline(
            "make one straight pipe",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
        )

    report = json.loads(Path(exc.value.artifact_path).read_text(encoding="utf-8"))

    assert report["failed_stage"] == "freecad_launch"
    assert report["top_issues"] == ["FINAL_01_FREECAD_LAUNCH_FAILED"]


def test_pipeline_final_critic_failure_writes_report_and_skips_final_mcp(
    monkeypatch,
    tmp_path,
):
    settings = replace(
        load_settings(Path("missing.env")),
        output_dir=tmp_path,
        freecad_auto_open=False,
        require_freecad_app=False,
        freecad_mcp_enabled=False,
        freecad_step_mcp_enabled=False,
        freecad_mcp_required=False,
        final_repair_rounds=0,
    )
    calls = []
    planner_calls = 0

    class FakeGeminiClient:
        def __init__(self, unused_settings):
            self.settings = unused_settings

        def stream_structured(self, prompt, schema, *, part):
            nonlocal planner_calls
            if part == "intent":
                return IntentResult(
                    global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
                    target_behavior=[
                        Goal(type="branch", branch_count=1, branch_angles=[45.0]),
                        Goal(type="branch", branch_count=1, branch_angles=[45.0]),
                    ],
                    expected_open_ports=4,
                    expected_open_ports_source="explicit",
                )
            planner_calls += 1
            return ActionDraft(
                target_port="START" if planner_calls == 1 else "M1.out",
                module="junction",
                catalog_schema_version=2,
                affected_goal_ids=[f"G{planner_calls}"],
                completed_goal_ids=[f"G{planner_calls}"],
                params={
                    "section_source": "inherit_target",
                    "outlets": [
                        {
                            "role": "primary",
                            "axis": [1.0, 0.0, 0.0],
                            "length": 20.0,
                            "outer_diameter": 20.0,
                            "wall_thickness": 2.0,
                        },
                        {
                            "role": "branch",
                            "axis": [0.70710678, 0.70710678, 0.0],
                            "length": 20.0,
                            "outer_diameter": 20.0,
                            "wall_thickness": 2.0,
                        },
                    ],
                    "blend_mode": "hard",
                    "max_hub_radius": 12.0,
                },
            )

    async def fake_execute(unused_settings, code):
        calls.append(code)
        return {"ok": True}

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGeminiClient)
    monkeypatch.setattr(pipeline, "execute_freecad_code", fake_execute)

    with pytest.raises(StaticValidationError) as exc:
        run_pipeline(
            "make a manifold with four generated downstream outlets",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
        )

    report = json.loads(Path(exc.value.artifact_path).read_text(encoding="utf-8"))

    assert calls == []
    assert report["failed_stage"] == "final_critic"
    assert report["top_issues"] == ["FINAL_01_OPEN_PORT_COUNT_MISMATCH"]


def test_pipeline_planning_failure_writes_report(monkeypatch, tmp_path):
    settings = replace(
        load_settings(Path("missing.env")),
        output_dir=tmp_path,
        freecad_auto_open=False,
        require_freecad_app=False,
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=True,
    )

    class FakeGeminiClient:
        def __init__(self, unused_settings):
            self.settings = unused_settings

        def stream_structured(self, prompt, schema, *, part):
            if part == "intent":
                return IntentResult(
                    global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
                    target_behavior=[Goal(type="move", direction="+X", length=20.0)],
                    expected_open_ports=1,
                    expected_open_ports_source="derived",
                )
            return ActionDraft(
                target_port="START",
                module="straight_pipe",
                params={"length": 20.0, "direction": "+X"},
                consumes_goal_index=1,
            )

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGeminiClient)

    with pytest.raises(StaticValidationError) as exc:
        run_pipeline(
            "make one straight pipe",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
        )

    report = json.loads(Path(exc.value.artifact_path).read_text(encoding="utf-8"))

    assert report["failed_stage"] == "planning"
    assert report["top_issues"] == ["STEP_0001_01_PLANNING_FAILED"]
    assert report["static_error_count"] == 1
    critic = json.loads(
        Path(report["artifacts"]["critic_report_path"]).read_text(encoding="utf-8")
    )
    assert [issue["issue_code"] for issue in critic["issues"]] == ["PLANNING_FAILED"]
    artifact_statuses = {item["name"]: item for item in report["artifact_statuses"]}
    assert artifact_statuses["actions"]["status"] == "available"
    assert artifact_statuses["state"]["status"] == "available"
    assert artifact_statuses["critic_report"]["status"] == "available"
    assert artifact_statuses["freecad_script"]["status"] == "unavailable"
    assert artifact_statuses["freecad_script"]["producer_stage"] == "planning"
    assert artifact_statuses["freecad_script"]["blocking_issue_ids"][0] == (
        "STEP_0001_01_PLANNING_FAILED"
    )


def test_structured_planner_failure_resets_lineage_and_retries_with_full_catalog(
    monkeypatch, tmp_path
):
    settings = replace(
        load_settings(Path("missing.env")).with_overrides(
            output_dir=tmp_path,
            skip_freecad=True,
        ),
        step_repair_attempts=1,
    )

    class FakeGeminiClient:
        supports_interaction_controls = True
        step_prompts: list[str] = []
        step_thinking_levels: list[str] = []
        reset_calls: list[str] = []

        def __init__(self, unused_settings):
            del unused_settings
            self.lineage = False

        def stream_structured(self, prompt, schema, *, part, thinking_level="low"):
            del schema
            if part == "intent":
                return IntentResult(
                    global_spec=GlobalSpec(
                        outer_diameter=20.0,
                        wall_thickness=2.0,
                    ),
                    target_behavior=[Goal(type="move", direction="+X", length=20.0)],
                    expected_open_ports=1,
                    expected_open_ports_source="derived",
                )
            self.step_prompts.append(prompt)
            self.step_thinking_levels.append(thinking_level)
            if len(self.step_prompts) == 1:
                self.lineage = True
                raise pipeline.StructuredOutputError(
                    "step_planner",
                    json.dumps(
                        {
                            "catalog_schema_version": 2,
                            "target_port": "START",
                            "affected_goal_ids": ["G1"],
                            "completed_goal_ids": ["G1"],
                            "choice": {
                                "module": "route",
                                "params": {
                                    "path_kind": "line",
                                    "section_source": "inherit_target",
                                    "waypoints": [[20.0, 0.0, 0.0]],
                                },
                            },
                        }
                    ),
                    ValueError("line route requires length and direction"),
                )
            assert self.lineage is False
            return CorePlannerDecision.model_validate(
                {
                    "catalog_schema_version": 2,
                    "target_port": "START",
                    "affected_goal_ids": ["G1"],
                    "completed_goal_ids": ["G1"],
                    "choice": {
                        "module": "route",
                        "params": {
                            "path_kind": "line",
                            "section_source": "inherit_target",
                            "length": 20.0,
                            "direction": [1.0, 0.0, 0.0],
                        },
                    },
                }
            )

        def has_previous(self, part):
            return part == "step_planner" and self.lineage

        def reset_lineage(self, part):
            self.reset_calls.append(part)
            self.lineage = False

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGeminiClient)

    report = run_pipeline(
        "make one straight pipe",
        settings,
        dry_run=False,
        stream=ThinkingStream(enabled=False),
    )

    assert report.status == "partial"
    assert FakeGeminiClient.reset_calls[0] == "step_planner"
    assert len(FakeGeminiClient.step_prompts) == 2
    assert FakeGeminiClient.step_thinking_levels == ["low", "low"]
    assert "module_catalog" in FakeGeminiClient.step_prompts[1]
    assert "PLANNING_FAILED" in FakeGeminiClient.step_prompts[1]


def test_pipeline_recovers_from_invalid_planner_schema_without_spending_repair_attempt(
    monkeypatch, tmp_path
):
    settings = replace(
        load_settings(Path("missing.env")).with_overrides(
            output_dir=tmp_path,
            skip_freecad=True,
        ),
        step_repair_attempts=0,
        final_repair_rounds=0,
    )

    class FakeGeminiClient:
        supports_interaction_controls = True
        supports_numeric_literals = True
        supports_system_instruction = True
        step_calls = []
        reset_calls = []

        def __init__(self, unused_settings):
            del unused_settings

        def stream_structured(self, prompt, schema, *, part, **kwargs):
            del schema
            if part == "intent":
                return IntentResult(
                    global_spec=GlobalSpec(
                        outer_diameter=20.0,
                        wall_thickness=2.0,
                    ),
                    target_behavior=[
                        Goal(
                            goal_id="G1",
                            type="move",
                            direction="+X",
                            length=20.0,
                        )
                    ],
                    expected_open_ports=1,
                    expected_open_ports_source="derived",
                )
            self.step_calls.append((prompt, kwargs))
            if len(self.step_calls) == 1:
                raise GeminiInvalidRequestError(
                    "400 invalid_request: Request contains an invalid argument",
                    provider_code="invalid_request",
                )
            return CorePlannerDecision.model_validate(
                {
                    "catalog_schema_version": 2,
                    "target_port": "START",
                    "affected_goal_ids": ["G1"],
                    "completed_goal_ids": ["G1"],
                    "choice": {
                        "module": "route",
                        "params": {
                            "path_kind": "line",
                            "section_source": "inherit_target",
                            "length": 20.0,
                            "direction": [1.0, 0.0, 0.0],
                        },
                    },
                }
            )

        def has_previous(self, part):
            del part
            return False

        def reset_lineage(self, part):
            self.reset_calls.append(part)

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGeminiClient)

    report = run_pipeline(
        "make one 20 mm straight pipe",
        settings,
        dry_run=False,
        stream=ThinkingStream(enabled=False),
    )
    attempts = json.loads(
        Path(report.artifacts.action_attempts_path).read_text(encoding="utf-8")
    )

    assert report.status == "partial"
    assert report.repair_attempt_count == 0
    assert len(FakeGeminiClient.step_calls) == 2
    assert FakeGeminiClient.reset_calls == ["step_planner", "step_planner"]
    assert len(attempts) == 1
    assert attempts[0]["status"] == "accepted"
    assert attempts[0]["issue_codes"] == []
    assert len(FakeGeminiClient.step_calls[1][1]["numeric_literals"]) < len(
        FakeGeminiClient.step_calls[0][1]["numeric_literals"]
    )


def test_pipeline_invalid_v2_draft_writes_report_before_resolution(
    monkeypatch, tmp_path
):
    settings = replace(
        load_settings(Path("missing.env")),
        output_dir=tmp_path,
        freecad_auto_open=False,
        require_freecad_app=False,
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=True,
    )

    class FakeGeminiClient:
        def __init__(self, unused_settings):
            self.settings = unused_settings

        def stream_structured(self, prompt, schema, *, part):
            if part == "intent":
                return IntentResult(
                    global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
                    target_behavior=[Goal(type="turn", direction="+Z", angle=90.0)],
                    expected_open_ports=1,
                    expected_open_ports_source="derived",
                )
            return ActionDraft(
                target_port="START",
                module="route",
                catalog_schema_version=2,
                affected_goal_ids=["G1"],
                completed_goal_ids=["G1"],
                params={
                    "section_source": "inherit_target",
                    "path_kind": "circular_arc",
                    "bend_radius": "bad",
                    "sweep_angle": 90.0,
                    "plane_normal": (0.0, 1.0, 0.0),
                },
            )

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGeminiClient)

    with pytest.raises(StaticValidationError) as exc:
        run_pipeline(
            "make one bend",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
        )

    report = json.loads(Path(exc.value.artifact_path).read_text(encoding="utf-8"))

    assert report["failed_stage"] == "draft_validation"
    assert report["top_issues"] == ["STEP_0001_01_DRAFT_VALIDATION_FAILED"]


def test_pipeline_max_iter_failure_writes_report(tmp_path):
    settings = load_settings(Path("missing.env")).with_overrides(
        output_dir=tmp_path,
        max_iter=1,
        skip_freecad=True,
    )

    with pytest.raises(StaticValidationError) as exc:
        run_pipeline(
            "지름 20mm 파이프로 100mm 직진하고 90도 위로 꺾은 다음 80mm 올라가게 만들어줘",
            settings,
            dry_run=True,
            stream=ThinkingStream(enabled=False),
        )

    report = json.loads(Path(exc.value.artifact_path).read_text(encoding="utf-8"))

    assert report["failed_stage"] == "max_iter"
    assert report["top_issues"] == ["STEP_0002_01_MAX_ITER_REACHED"]


def test_required_step_mcp_execution_failure_writes_issue(monkeypatch, tmp_path):
    settings = replace(
        load_settings(Path("missing.env")),
        output_dir=tmp_path,
        freecad_auto_open=False,
        require_freecad_app=False,
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=True,
        freecad_mcp_required=True,
    )

    class FakeGeminiClient:
        def __init__(self, unused_settings):
            self.settings = unused_settings

        def stream_structured(self, prompt, schema, *, part):
            if part == "intent":
                return IntentResult(
                    global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
                    target_behavior=[Goal(type="move", direction="+X", length=20.0)],
                    expected_open_ports=1,
                    expected_open_ports_source="derived",
                )
            return _production_line_draft()

    async def fail_execute(unused_settings, code):
        raise RuntimeError("step mcp boom")

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGeminiClient)
    monkeypatch.setattr(pipeline, "execute_freecad_code", fail_execute)

    with pytest.raises(StaticValidationError) as exc:
        run_pipeline(
            "make one straight pipe",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
        )

    report = json.loads(Path(exc.value.artifact_path).read_text(encoding="utf-8"))

    assert report["failed_stage"] == "step_mcp"
    assert report["top_issues"] == ["STEP_0001_01_REQUIRED_STEP_MCP_FAILED"]


def test_required_step_publish_failure_writes_issue(monkeypatch, tmp_path):
    settings = replace(
        load_settings(Path("missing.env")),
        output_dir=tmp_path,
        freecad_auto_open=False,
        require_freecad_app=False,
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=True,
        freecad_mcp_required=True,
    )
    calls = 0

    class FakeGeminiClient:
        def __init__(self, unused_settings):
            self.settings = unused_settings

        def stream_structured(self, prompt, schema, *, part):
            if part == "intent":
                return IntentResult(
                    global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
                    target_behavior=[Goal(type="move", direction="+X", length=20.0)],
                    expected_open_ports=1,
                    expected_open_ports_source="derived",
                )
            return _production_line_draft()

    async def fake_execute(unused_settings, code):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("publish mcp boom")
        return _successful_freecad_result(code)

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGeminiClient)
    monkeypatch.setattr(pipeline, "execute_freecad_code", fake_execute)

    with pytest.raises(StaticValidationError) as exc:
        run_pipeline(
            "make one straight pipe",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
        )

    report = json.loads(Path(exc.value.artifact_path).read_text(encoding="utf-8"))

    assert calls == 2
    assert report["failed_stage"] == "step_mcp"
    assert report["top_issues"] == ["STEP_0001_01_REQUIRED_STEP_MCP_FAILED"]


def test_move_direction_updates_straight_pipe_axis():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(type="move", direction="+X", length=100.0),
            Goal(type="move", direction="+Z", length=80.0),
        ],
    )
    state = engine.initial_state(intent)

    first = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="straight_pipe",
            params={"length": 100.0, "direction": "+X"},
        ),
        state,
    )
    state = engine.apply_action(first, state)
    second = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="straight_pipe",
            params={"length": 80.0, "direction": "+Z"},
        ),
        state,
    )
    state = engine.apply_action(second, state)

    assert state.open_ports[0].position == (100.0, 0.0, 80.0)
    assert state.open_ports[0].axis == (0.0, 0.0, 1.0)


def test_initial_axis_only_uses_first_goal_move_direction():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)

    state = engine.initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[
                Goal(type="turn", direction="+Z", angle=90.0),
                Goal(type="move", direction="+Z", length=80.0),
            ],
        )
    )

    assert state.open_ports[0].axis == (1.0, 0.0, 0.0)


def test_freecad_script_uses_smooth_sweeps_and_validation_sentinel(tmp_path):
    settings = load_settings(Path("missing.env")).with_overrides(
        output_dir=tmp_path,
        skip_freecad=True,
    )
    report = run_pipeline(
        "지름 20mm 파이프로 100mm 직진하고 90도 위로 꺾은 다음 80mm 올라가게 만들어줘",
        settings,
        dry_run=True,
        stream=ThinkingStream(enabled=False),
    )

    script = Path(report.artifacts.freecad_script_path).read_text(encoding="utf-8")

    assert "makePipeShell" in script
    assert "PAYLOAD = json.loads(" in script
    assert "CADGEN_VALIDATION=" in script
    assert "CadGenCandidate_" in script
    compile(script, "generated_freecad.py", "exec")


def test_freecad_script_keeps_legacy_hard_junction_out_of_blended_hub_path():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                type="branch",
                required_outlet_vectors=[
                    (-1.0, 0.0, 1.0),
                    (-1.0, 0.0, -1.0),
                    (1.0, 0.0, 1.0),
                    (1.0, 0.0, -1.0),
                ],
                include_primary_outlet=False,
                junction_style="smooth_hub",
            )
        ],
    )
    state = engine.initial_state(intent)
    action = engine.resolve_action(
        ActionDraft(target_port="START", module="junction_pipe", params={}),
        state,
    )
    state = engine.apply_action(action, state)

    script = build_freecad_script(state)

    assert 'if params["blend_mode"] == "fillet":' in script
    assert "fillet_compact_junction_material" in script
    assert "Part.makeSphere" not in script
    assert 'migrated["blend_mode"] = "hard"' in script
    assert "make_junction" in script
    assert '"include_primary_outlet":false' in script
    assert '"trunk_end":' not in script
    assert '"branch_4_end":' in script
    compile(script, "generated_junction.py", "exec")


def test_connector_defaults_include_coupling_geometry():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    state = engine.initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[Goal(type="connector", length=20.0)],
        )
    )

    action = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="connector_pipe",
            params={"length": 20.0},
        ),
        state,
    )

    assert action.params["coupling_outer_diameter"] == 25.0
    assert action.params["sleeve_overlap"] == 5.0


def test_continuation_modules_inherit_target_port_diameter_after_reducer():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    state = engine.initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[
                Goal(type="diameter_change", diameter_out=12.0),
                Goal(type="move", length=30.0),
            ],
        )
    )
    reducer = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="reducer_pipe",
            params={"length": 20.0, "diameter_out": 12.0, "wall_thickness_out": 1.0},
        ),
        state,
    )
    state = engine.apply_action(reducer, state)

    straight = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="straight_pipe",
            params={"length": 30.0},
        ),
        state,
    )
    connector = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="connector_pipe",
            params={"length": 10.0},
        ),
        state,
    )
    bend = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="bend_pipe",
            params={"angle": 45.0, "turn_direction": "+Z"},
        ),
        state,
    )

    assert straight.params["outer_diameter"] == 12.0
    assert straight.params["wall_thickness"] == 1.0
    assert connector.params["outer_diameter"] == 12.0
    assert connector.params["wall_thickness"] == 1.0
    assert connector.params["coupling_outer_diameter"] == 15.0
    assert connector.params["sleeve_overlap"] == 3.0
    assert bend.params["outer_diameter"] == 12.0
    assert bend.params["wall_thickness"] == 1.0


def test_none_params_are_defaulted_before_state_application():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    state = engine.initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[Goal(type="move")],
        )
    )

    action = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="straight_pipe",
            params={"length": None, "outer_diameter": None},
        ),
        state,
    )
    state = engine.apply_action(action, state)

    assert action.params["length"] == 100.0
    assert action.params["outer_diameter"] == 20.0
    assert state.open_ports[0].position == (100.0, 0.0, 0.0)


def test_bend_segment_resolution_is_clamped_and_axis_uses_tangent():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    state = engine.initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[Goal(type="turn", direction="+Z", angle=45.0)],
        )
    )

    action = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="bend_pipe",
            params={"angle": 45.0, "turn_direction": "+Z", "segment_resolution": 0},
        ),
        state,
    )
    state = engine.apply_action(action, state)
    axis = state.open_ports[0].axis

    assert action.params["segment_resolution"] == 4
    assert len(state.placed_modules[0].params["path_points"]) == 5
    assert 0.0 < axis[0] < 1.0
    assert 0.0 < axis[2] < 1.0


def test_bad_segment_resolution_defaults_during_resolution():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    state = engine.initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[Goal(type="turn", direction="+Z", angle=90.0)],
        )
    )

    action = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="bend_pipe",
            params={"angle": 90.0, "turn_direction": "+Z", "segment_resolution": "bad"},
        ),
        state,
    )

    assert action.params["segment_resolution"] == 24


def test_open_end_keeps_an_open_port():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    state = engine.initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[Goal(type="end", end_type="open")],
        )
    )

    action = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="cap_pipe",
            params={"end_type": "open"},
        ),
        state,
    )
    state = engine.apply_action(action, state)

    assert len(state.open_ports) == 1
    assert state.open_ports[0].id == "M1.out"


def test_cap_default_thickness_exceeds_bore_extension():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    state = engine.initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=1.0),
            target_behavior=[Goal(type="end", end_type="cap")],
        )
    )

    action = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="cap_pipe",
            params={"end_type": "cap"},
        ),
        state,
    )

    assert action.params["cap_thickness"] > 2.0


def test_final_critic_rejects_open_end_marker_without_geometry():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[Goal(type="end", end_type="open")],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )
    state = engine.initial_state(intent)
    action = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="cap_pipe",
            params={"end_type": "open"},
        ),
        state,
    )
    state = engine.apply_action(action, state)

    critic = build_final_critic_report(intent, state, [])

    assert not critic.passed
    assert any(issue.issue_code == "NO_GEOMETRY_MODULES" for issue in critic.issues)


def test_validation_reports_bad_numeric_values_without_raising():
    settings = load_settings(Path("missing.env"))
    state = StateEngine(settings).initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[],
        )
    )

    result = validate_action(
        ResolvedAction(
            action_id="A1",
            target_port="START",
            module="bend_pipe",
            params={
                "angle": None,
                "bend_radius": "bad",
                "outer_diameter": 20.0,
                "wall_thickness": 2.0,
                "start_position": (0.0, 0.0, 0.0),
                "axis": (1.0, 0.0, 0.0),
                "out_axis": (0.0, 0.0, 1.0),
                "segment_resolution": 0,
            },
        ),
        state,
    )

    assert not result.valid
    assert any("Missing required param" in error for error in result.errors)
    assert any(
        "Invalid numeric param for bend_radius" in error for error in result.errors
    )
    assert any(
        "segment_resolution must be at least 4" in error for error in result.errors
    )


def test_validation_reports_bad_junction_branch_angles():
    settings = load_settings(Path("missing.env"))
    state = StateEngine(settings).initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[],
        )
    )

    result = validate_action(
        ResolvedAction(
            action_id="A1",
            target_port="START",
            module="junction_pipe",
            params={
                "branch_count": 2,
                "branch_angles": ["bad"],
                "outer_diameter": 20.0,
                "wall_thickness": 2.0,
                "start_position": (0.0, 0.0, 0.0),
                "axis": (1.0, 0.0, 0.0),
            },
        ),
        state,
    )

    assert not result.valid
    assert any("branch_angles[0]" in error for error in result.errors)


def test_validation_rejects_invalid_script_consumed_numeric_params():
    settings = load_settings(Path("missing.env"))
    state = StateEngine(settings).initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[],
        )
    )

    reducer = validate_action(
        ResolvedAction(
            action_id="A1",
            target_port="START",
            module="reducer_pipe",
            params={
                "length": 20.0,
                "diameter_in": 20.0,
                "diameter_out": 12.0,
                "wall_thickness_in": 2.0,
                "wall_thickness_out": -1.0,
                "start_position": (0.0, 0.0, 0.0),
                "axis": (1.0, 0.0, 0.0),
            },
        ),
        state,
    )
    connector = validate_action(
        ResolvedAction(
            action_id="A2",
            target_port="START",
            module="connector_pipe",
            params={
                "length": 20.0,
                "outer_diameter": 20.0,
                "wall_thickness": 2.0,
                "start_position": (0.0, 0.0, 0.0),
                "axis": (1.0, 0.0, 0.0),
                "coupling_outer_diameter": 10.0,
                "sleeve_overlap": -1.0,
            },
        ),
        state,
    )
    cap = validate_action(
        ResolvedAction(
            action_id="A3",
            target_port="START",
            module="cap_pipe",
            params={
                "end_type": "cap",
                "outer_diameter": 20.0,
                "wall_thickness": 2.0,
                "start_position": (0.0, 0.0, 0.0),
                "axis": (1.0, 0.0, 0.0),
                "cap_thickness": 0.0,
            },
        ),
        state,
    )

    assert not reducer.valid
    assert any(
        "wall_thickness_out must be non-negative" in error for error in reducer.errors
    )
    assert not connector.valid
    assert any(
        "coupling_outer_diameter must be at least outer_diameter" in error
        for error in connector.errors
    )
    assert any(
        "sleeve_overlap must be non-negative" in error for error in connector.errors
    )
    assert not cap.valid
    assert any(
        "cap_thickness must be greater than zero for capped ends" in error
        for error in cap.errors
    )
    assert any(
        "cap_thickness must be greater than bore extension" in error
        for error in cap.errors
    )


def test_junction_branch_count_creates_trunk_plus_branch_ports():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    state = engine.initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[
                Goal(type="branch", branch_count=2, branch_angles=[45.0, -45.0])
            ],
        )
    )

    action = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="junction_pipe",
            params={"branch_count": 2, "branch_angles": [45.0, -45.0]},
        ),
        state,
    )
    state = engine.apply_action(action, state)

    assert len(state.open_ports) == 3
    assert {port.id for port in state.open_ports} == {
        "M1.out",
        "M1.out_1",
        "M1.out_2",
    }
    script = build_freecad_script(state)
    assert "branch_1_end" in script
    assert "branch_2_end" in script


def test_junction_required_vectors_disable_primary_and_set_terminal_axes():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                type="branch",
                required_outlet_vectors=[
                    (-1.0, 0.0, 1.0),
                    (-1.0, 0.0, -1.0),
                    (1.0, 0.0, 1.0),
                    (1.0, 0.0, -1.0),
                ],
                include_primary_outlet=False,
                junction_style="smooth_hub",
            )
        ],
        expected_open_ports=4,
        expected_open_ports_source="explicit",
    )
    state = engine.initial_state(intent)

    action = engine.resolve_action(
        ActionDraft(
            target_port="START",
            module="junction_pipe",
            params={},
        ),
        state,
    )
    state = engine.apply_action(action, state)

    assert action.params["branch_count"] == 4
    assert action.params["include_primary_outlet"] is False
    assert len(state.open_ports) == 4
    assert {port.id for port in state.open_ports} == {
        "M1.out_1",
        "M1.out_2",
        "M1.out_3",
        "M1.out_4",
    }
    assert "out" not in state.placed_modules[0].ports
    assert "trunk_end" not in state.placed_modules[0].params
    assert state.open_ports[0].axis == pytest.approx((-0.70710678, 0.0, 0.70710678))


def test_junction_required_vectors_default_to_no_primary_when_omitted():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                type="branch",
                required_outlet_vectors=[(-1.0, 0.0, 1.0), (1.0, 0.0, 1.0)],
            )
        ],
        expected_open_ports=2,
        expected_open_ports_source="explicit",
    )
    state = engine.initial_state(intent)

    action = engine.resolve_action(
        ActionDraft(target_port="START", module="junction_pipe", params={}),
        state,
    )
    state = engine.apply_action(action, state)

    assert action.params["branch_count"] == 2
    assert action.params["include_primary_outlet"] is False
    assert {port.id for port in state.open_ports} == {"M1.out_1", "M1.out_2"}


def test_local_planner_derives_branch_count_from_required_vectors():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    state = engine.initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[
                Goal(
                    type="branch",
                    required_outlet_vectors=[
                        (-1.0, 0.0, 1.0),
                        (-1.0, 0.0, -1.0),
                        (1.0, 0.0, 1.0),
                    ],
                )
            ],
        )
    )

    draft = plan_next_action(state)

    assert draft.params["branch_count"] == 3
    assert draft.params["include_primary_outlet"] is None
    assert draft.params["required_outlet_vectors"] == [
        (-1.0, 0.0, 1.0),
        (-1.0, 0.0, -1.0),
        (1.0, 0.0, 1.0),
    ]


def test_local_fixture_planner_fails_closed_for_freeform_route_goal():
    settings = load_settings(Path("missing.env"))
    state = StateEngine(settings).initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[
                Goal(
                    goal_id="G1",
                    type="route",
                    path_kind="spline",
                    required_waypoints=[(20.0, 20.0, 10.0)],
                )
            ],
        )
    )

    with pytest.raises(ValueError, match="will not substitute unrelated cap"):
        plan_next_action(state)


def test_local_intent_fixture_rejects_unrepresentable_qualitative_spiral():
    settings = load_settings(Path("missing.env"))

    with pytest.raises(ValueError, match="cannot represent a qualitative freeform"):
        infer_intent(
            "Create a rising spiral coil followed by a spatial S-curve.",
            settings,
        )


def test_junction_vector_count_mismatch_fails_registry_validation():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    state = engine.initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[],
        )
    )

    action = engine.resolve_action(
        ActionDraft(
            target_port="START",
            module="junction_pipe",
            params={
                "branch_count": 3,
                "required_outlet_vectors": [(-1.0, 0.0, 1.0), (1.0, 0.0, 1.0)],
                "include_primary_outlet": False,
            },
        ),
        state,
    )

    result = validate_action(action, state)

    assert not result.valid
    assert any(
        "branch_count must match explicit outlet vector count" in error
        for error in result.errors
    )


def test_junction_invalid_and_conflicting_vectors_fail_registry_validation():
    settings = load_settings(Path("missing.env"))
    state = StateEngine(settings).initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[],
        )
    )

    invalid = validate_action(
        ResolvedAction(
            action_id="A1",
            target_port="START",
            module="junction_pipe",
            params={
                "branch_count": 2,
                "branch_angles": [45.0, -45.0],
                "outer_diameter": 20.0,
                "wall_thickness": 2.0,
                "start_position": (0.0, 0.0, 0.0),
                "axis": (1.0, 0.0, 0.0),
                "required_outlet_vectors": [(0.0, 0.0, 0.0), (float("inf"), 0.0, 1.0)],
            },
        ),
        state,
    )
    conflicting = validate_action(
        ResolvedAction(
            action_id="A2",
            target_port="START",
            module="junction_pipe",
            params={
                "branch_count": 2,
                "branch_angles": [45.0, -45.0],
                "outer_diameter": 20.0,
                "wall_thickness": 2.0,
                "start_position": (0.0, 0.0, 0.0),
                "axis": (1.0, 0.0, 0.0),
                "required_outlet_vectors": [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
                "outlet_vectors": [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)],
            },
        ),
        state,
    )

    assert not invalid.valid
    assert any("must not be a zero vector" in error for error in invalid.errors)
    assert any(
        "Invalid numeric param for required_outlet_vectors[1][0]" in error
        for error in invalid.errors
    )
    assert not conflicting.valid
    assert any(
        "outlet_vectors must match required_outlet_vectors" in error
        for error in conflicting.errors
    )


def test_junction_registry_canonicalizes_planner_vector_and_style_aliases():
    settings = load_settings(Path("missing.env"))
    state = StateEngine(settings).initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[],
        )
    )
    action = ResolvedAction(
        action_id="A1",
        target_port="START",
        module="junction_pipe",
        params={
            "branch_count": 2,
            "branch_angles": [],
            "outer_diameter": 20.0,
            "wall_thickness": 2.0,
            "start_position": (0.0, 0.0, 0.0),
            "axis": (1.0, 0.0, 0.0),
            "required_outlet_vectors": [
                {"direction": "upper-left"},
                {"vector": {"x": -1.0, "y": 0.0, "z": -1.0}},
            ],
            "junction_style": "smooth Y-shaped",
        },
    )

    result = validate_action(action, state)

    assert result.valid
    assert action.params["required_outlet_vectors"] == [
        (-1.0, 0.0, 1.0),
        (-1.0, 0.0, -1.0),
    ]
    assert action.params["junction_style"] == "smooth_hub"


def test_resolver_canonicalizes_vectors_before_primary_defaulting():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[Goal(type="branch")],
        expected_open_ports=2,
        expected_open_ports_source="explicit",
    )
    state = engine.initial_state(intent)

    action = engine.resolve_action(
        ActionDraft(
            target_port="START",
            module="junction_pipe",
            params={
                "required_outlet_vectors": ["upper-left", "lower-left"],
                "junction_style": "y_blend",
            },
        ),
        state,
    )
    state = engine.apply_action(action, state)

    assert action.params["branch_count"] == 2
    assert action.params["include_primary_outlet"] is False
    assert action.params["junction_style"] == "smooth_hub"
    assert {port.id for port in state.open_ports} == {"M1.out_1", "M1.out_2"}
    assert "out" not in state.placed_modules[0].ports


def test_static_step_validation_reports_branch_direction_mismatch():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                type="branch",
                direction="-X",
                branch_count=2,
                branch_angles=[45.0, -45.0],
            )
        ],
        expected_open_ports=3,
        expected_open_ports_source="derived",
    )
    before = engine.initial_state(intent)
    action = engine.resolve_action(
        ActionDraft(
            target_port=before.open_ports[0].id,
            module="junction_pipe",
            params={
                "branch_count": 2,
                "branch_angles": [45.0, -45.0],
                "direction": "+X",
            },
        ),
        before,
    )
    after = engine.apply_action(action, before)

    step = build_step_verification(before, action, after, intent, 1)

    assert step.status == "failed"
    assert any(issue.issue_code == "BRANCH_DIRECTION_MISMATCH" for issue in step.issues)


def test_static_step_validation_requires_opposite_input_axis():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[Goal(type="move", direction="+X", length=20.0)],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )
    before = engine.initial_state(intent)
    target = before.open_ports[0]
    action = ResolvedAction(
        action_id="A1",
        target_port=target.id,
        module="straight_pipe",
        params={
            "length": 20.0,
            "start_position": target.position,
            "axis": target.axis,
            "outer_diameter": 20.0,
            "wall_thickness": 2.0,
        },
    )
    bad_in = Port(
        id="M1.in",
        position=target.position,
        axis=target.axis,
        outer_diameter=20.0,
        wall_thickness=2.0,
    )
    out = Port(
        id="M1.out",
        position=(20.0, 0.0, 0.0),
        axis=target.axis,
        outer_diameter=20.0,
        wall_thickness=2.0,
    )
    after = before.model_copy(
        update={
            "state_id": "S1",
            "placed_modules": [
                ModuleRef(
                    id="M1",
                    type="straight_pipe",
                    params=action.params,
                    ports={"in": bad_in, "out": out},
                )
            ],
            "open_ports": [out],
            "used_ports": [target.id],
            "remaining_goals": [],
            "action_history": [action],
        }
    )

    step = build_step_verification(before, action, after, intent, 1)

    assert any(
        issue.issue_code == "MODULE_INPUT_AXIS_MISMATCH" for issue in step.issues
    )


def test_explicit_required_outlet_directions_must_each_be_covered():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                type="branch",
                branch_count=2,
                branch_angles=[45.0, -45.0],
                required_outlet_directions=["+X", "-X"],
            )
        ],
        expected_open_ports=3,
        expected_open_ports_source="derived",
    )
    before = engine.initial_state(intent)
    action = engine.resolve_action(
        ActionDraft(
            target_port=before.open_ports[0].id,
            module="junction_pipe",
            params={
                "branch_count": 2,
                "branch_angles": [45.0, -45.0],
                "direction": "+X",
                "required_outlet_directions": ["+X", "+Z"],
            },
        ),
        before,
    )
    after = engine.apply_action(action, before)

    step = build_step_verification(before, action, after, intent, 1)
    mismatch = next(
        issue
        for issue in step.issues
        if issue.issue_code == "BRANCH_DIRECTION_MISMATCH"
    )

    assert "-X" in mismatch.actual["missing_directions"]


def test_explicit_required_outlet_vectors_must_each_be_covered_distinctly():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                type="branch",
                branch_count=2,
                required_outlet_vectors=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
                include_primary_outlet=False,
            )
        ],
        expected_open_ports=2,
        expected_open_ports_source="explicit",
    )
    before = engine.initial_state(intent)
    action = engine.resolve_action(
        ActionDraft(
            target_port="START",
            module="junction_pipe",
            params={
                "branch_count": 2,
                "outlet_vectors": [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                "include_primary_outlet": False,
            },
        ),
        before,
    )
    after = engine.apply_action(action, before)

    step = build_step_verification(before, action, after, intent, 1)
    mismatch = next(
        issue for issue in step.issues if issue.issue_code == "BRANCH_VECTOR_MISMATCH"
    )

    assert step.status == "failed"
    assert mismatch.actual["matched_port_ids"] == ["M1.out_1"]
    assert mismatch.actual["missing_vectors"][0]["expected_vector"] == [0.0, 0.0, 1.0]


def test_static_step_validation_rejects_planner_conflicting_primary_policy():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                type="branch",
                branch_count=2,
                required_outlet_vectors=[(-1.0, 0.0, 1.0), (1.0, 0.0, 1.0)],
                include_primary_outlet=False,
            )
        ],
        expected_open_ports=2,
        expected_open_ports_source="explicit",
    )
    before = engine.initial_state(intent)
    action = engine.resolve_action(
        ActionDraft(
            target_port="START",
            module="junction_pipe",
            params={
                "branch_count": 2,
                "outlet_vectors": [(-1.0, 0.0, 1.0), (1.0, 0.0, 1.0)],
                "include_primary_outlet": True,
            },
        ),
        before,
    )
    after = engine.apply_action(action, before)

    step = build_step_verification(before, action, after, intent, 1)
    issue_codes = {issue.issue_code for issue in step.issues}
    critic = build_final_critic_report(intent, after, [step])

    assert step.status == "failed"
    assert "UNEXPECTED_PRIMARY_OUTLET" in issue_codes
    assert "JUNCTION_OUTPUT_COUNT_MISMATCH" in issue_codes
    assert "OPEN_PORT_DELTA_MISMATCH" in issue_codes
    assert any(
        suggestion.operation == "remove_unexpected_primary_outlet"
        for suggestion in critic.patch_suggestions
    )


def test_static_step_validation_reports_turn_direction_mismatch():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[Goal(type="turn", direction="+Z", angle=90.0)],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )
    before = engine.initial_state(intent)
    action = engine.resolve_action(
        ActionDraft(
            target_port=before.open_ports[0].id,
            module="bend_pipe",
            params={"angle": 90.0, "turn_direction": "-Z"},
        ),
        before,
    )
    after = engine.apply_action(action, before)

    step = build_step_verification(before, action, after, intent, 1)

    assert any(issue.issue_code == "TURN_DIRECTION_MISMATCH" for issue in step.issues)


def test_junction_direction_param_guides_branch_axes_and_passes_static_check():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[Goal(type="branch", direction="-X", branch_count=2)],
        expected_open_ports=3,
        expected_open_ports_source="derived",
    )
    before = engine.initial_state(intent)
    action = engine.resolve_action(
        ActionDraft(
            target_port=before.open_ports[0].id,
            module="junction_pipe",
            params={
                "branch_count": 2,
                "branch_angles": [45.0, -45.0],
                "direction": "-X",
            },
        ),
        before,
    )
    after = engine.apply_action(action, before)

    step = build_step_verification(before, action, after, intent, 1)

    assert step.status == "passed"
    assert all(issue.issue_code != "BRANCH_DIRECTION_MISMATCH" for issue in step.issues)


def test_final_critic_rejects_expected_open_port_mismatch():
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(type="branch", branch_count=2, branch_angles=[45.0, -45.0]),
            Goal(type="branch", branch_count=2, branch_angles=[45.0, -45.0]),
        ],
        expected_open_ports=4,
        expected_open_ports_source="explicit",
    )
    state = engine.initial_state(intent)
    first = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="junction_pipe",
            params={"branch_count": 2, "branch_angles": [45.0, -45.0]},
        ),
        state,
    )
    state = engine.apply_action(first, state)
    second = engine.resolve_action(
        ActionDraft(
            target_port=state.open_ports[0].id,
            module="junction_pipe",
            params={"branch_count": 2, "branch_angles": [45.0, -45.0]},
        ),
        state,
    )
    state = engine.apply_action(second, state)

    critic = build_final_critic_report(intent, state, [])

    assert not critic.passed
    assert any(
        issue.issue_code == "OPEN_PORT_COUNT_MISMATCH" for issue in critic.issues
    )


def test_final_critic_rejects_missing_terminal_vector_even_when_count_matches():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                type="branch",
                branch_count=2,
                required_outlet_vectors=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
                include_primary_outlet=False,
            )
        ],
        expected_open_ports=2,
        expected_open_ports_source="explicit",
    )
    open_ports = [
        Port(
            id="M1.out_1",
            position=(40.0, 0.0, 0.0),
            axis=(1.0, 0.0, 0.0),
            outer_diameter=20.0,
            wall_thickness=2.0,
        ),
        Port(
            id="M1.out_2",
            position=(80.0, 0.0, 0.0),
            axis=(1.0, 0.0, 0.0),
            outer_diameter=20.0,
            wall_thickness=2.0,
        ),
    ]
    state = PipeState(
        state_id="S1",
        global_spec=intent.global_spec,
        placed_modules=[
            ModuleRef(
                id="M1",
                type="junction_pipe",
                params={"include_primary_outlet": False},
            )
        ],
        open_ports=open_ports,
    )

    critic = build_final_critic_report(intent, state, [])
    mismatch = next(
        issue
        for issue in critic.issues
        if issue.issue_code == "FINAL_OUTLET_VECTOR_MISMATCH"
    )

    assert not critic.passed
    assert mismatch.actual["matched_port_ids"] == ["M1.out_1"]
    assert mismatch.actual["missing_vectors"][0]["expected_vector"] == [0.0, 0.0, 1.0]


def test_freecad_mcp_timeout_is_reported(monkeypatch):
    async def slow_execute(settings, code):
        await asyncio.sleep(1)
        return {"ok": True}

    monkeypatch.setattr(freecad_mcp, "_execute_freecad_code", slow_execute)
    settings = replace(load_settings(Path("missing.env")), freecad_mcp_timeout_sec=0.01)

    with pytest.raises(FreeCADMCPError, match="timed out"):
        asyncio.run(freecad_mcp.execute_freecad_code(settings, "pass"))


def test_freecad_running_detects_macos_app_when_pgrep_misses(monkeypatch):
    settings = load_settings(Path("missing.env"))

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["pgrep", "-x"]:
            return subprocess_result(returncode=1)
        if cmd[:2] == ["osascript", "-e"]:
            return subprocess_result(returncode=0, stdout="true\n")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(freecad_app.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(freecad_app.subprocess, "run", fake_run)

    assert freecad_app.is_freecad_running(settings)


def test_ensure_freecad_open_does_not_launch_when_macos_app_is_running(monkeypatch):
    settings = load_settings(Path("missing.env"))
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["pgrep", "-x"]:
            return subprocess_result(returncode=1)
        if cmd[:2] == ["osascript", "-e"]:
            return subprocess_result(returncode=0, stdout="true\n")
        if cmd[:2] == ["open", "-a"]:
            raise AssertionError(
                "open -a should not be called for an already-running app"
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(freecad_app.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(freecad_app.subprocess, "run", fake_run)

    opened = ensure_freecad_open(settings, ThinkingStream(enabled=False))

    assert opened is False
    assert not any(call[:2] == ["open", "-a"] for call in calls)


def test_ensure_freecad_open_confirms_launch_with_macos_app_probe(monkeypatch):
    settings = replace(load_settings(Path("missing.env")), freecad_open_timeout_sec=1.0)
    calls = []
    app_probe_count = 0
    monotonic_values = iter([0.0, 0.1])

    def fake_run(cmd, **kwargs):
        nonlocal app_probe_count
        calls.append(cmd)
        if cmd[:2] == ["pgrep", "-x"]:
            return subprocess_result(returncode=1)
        if cmd[:2] == ["osascript", "-e"]:
            app_probe_count += 1
            return subprocess_result(
                returncode=0,
                stdout="true\n" if app_probe_count >= 2 else "false\n",
            )
        if cmd[:2] == ["open", "-a"]:
            return subprocess_result(returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(freecad_app.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(freecad_app.subprocess, "run", fake_run)
    monkeypatch.setattr(freecad_app.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(freecad_app.time, "sleep", lambda unused_seconds: None)

    opened = ensure_freecad_open(settings, ThinkingStream(enabled=False))

    assert opened is True
    assert any(call[:2] == ["open", "-a"] for call in calls)
    assert app_probe_count == 2


def subprocess_result(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )
