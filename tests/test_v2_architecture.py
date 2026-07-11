from __future__ import annotations

import asyncio
import hashlib
import json
import math
import textwrap
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import BaseModel, Field, ValidationError, model_validator

import cadgen.pipeline as pipeline
from cadgen.config import load_settings
from cadgen.freecad_mcp import (
    FreeCADMCPError,
    FreeCADValidationError,
    _document_probe_code,
    assess_freecad_publish,
    assess_freecad_validation,
)
from cadgen.freecad_script import (
    GENERATOR_VERSION,
    build_freecad_publish_script,
    build_freecad_script,
    geometry_payload_digest,
)
from cadgen.freecad_script import candidate_document_name, published_document_name
from cadgen.gemini_client import (
    GeminiBudgetError,
    GeminiClient,
    GeminiInvalidRequestError,
    GeminiRequestError,
    HostContractValidationError,
    StructuredOutputError,
    StructuredOutputIncompleteError,
    gemini_json_schema,
    schema_keywords,
)
from cadgen.pipeline import run_pipeline
from cadgen.prompts import (
    compact_visual_module_map,
    compact_planner_payload,
    realized_terminal_topology,
    step_planner_prompt,
    step_planner_system_instruction,
)
from cadgen.registry import (
    llm_visible_modules,
    planner_catalog,
    validate_action,
    validate_draft,
)
from cadgen.schemas import (
    ActionDraft,
    ActionAttempt,
    AgendaRepairDirective,
    AssemblyBounds,
    BranchGoalOutletSpec,
    ComponentGoalSpec,
    ConnectPortsParamsV2,
    CorePlannerDecision,
    CorePlannerDecisionWire,
    GeometricConstraint,
    GlobalSpec,
    Goal,
    IntentResult,
    LLMUsage,
    PlannerDecision,
    Port,
    ProductionGoal,
    ProductionIntent,
    ResolvedAction,
    JunctionParamsV2,
    RouteParamsV2,
    VisualCriticIssue,
    VisualCriticResult,
)
from cadgen.state import StateEngine, _arc_points_from_plane
from cadgen.static_validation import (
    _arc_endpoint_tangents,
    _segment_distance,
    build_final_critic_report,
    build_step_verification,
)
from cadgen.stream import ThinkingStream
from cadgen.vector import canonical_circular_arc_frame, dot, rotate


def _settings(tmp_path: Path | None = None):
    settings = load_settings(Path("missing.env"))
    if tmp_path is not None:
        settings = settings.with_overrides(output_dir=tmp_path, skip_freecad=True)
    return settings


def _intent(*goals: Goal, expected_open_ports: int | None = 1) -> IntentResult:
    return IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=list(goals),
        expected_open_ports=expected_open_ports,
        expected_open_ports_source="explicit",
    )


def _line_action(
    target: str,
    goal_ids: list[str],
    *,
    completed: list[str],
    length: float = 20.0,
) -> ActionDraft:
    return ActionDraft(
        target_port=target,
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=goal_ids,
        completed_goal_ids=completed,
        params={
            "path_kind": "line",
            "section_source": "inherit_target",
            "length": length,
            "direction": (1.0, 0.0, 0.0),
        },
    )


def _validation_evidence(
    state,
    *,
    digest: str | None = None,
    run_id: str = "local",
    attempt_id: int = 1,
) -> dict:
    module_ids = [module.id for module in state.placed_modules]
    sampled_module_ids = [
        module.id
        for module in state.placed_modules
        if module.type not in {"terminate", "cap_pipe"}
    ]
    return {
        "schema_version": 3,
        "generator_version": GENERATOR_VERSION,
        "run_id": run_id,
        "state_version": state.state_version,
        "attempt_id": attempt_id,
        "candidate_document": candidate_document_name(
            state, run_id=run_id, attempt_id=attempt_id
        ),
        "candidate_shape_fingerprints": {
            "PipeAssembly": "0" * 64,
            **{f"solid_{module_id}": "0" * 64 for module_id in module_ids},
        },
        "payload_digest": digest or geometry_payload_digest(state),
        "state_id": state.state_id,
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
            "declared_downstream_open_port_count": len(state.open_ports),
            "anchored_inlet_count": 1 if state.placed_modules else 0,
        },
        "passed": True,
    }


def _text_result(prefix: str, evidence: dict) -> dict:
    return {
        "content": [
            {
                "type": "text",
                "text": prefix + json.dumps(evidence, separators=(",", ":")),
            }
        ]
    }


def test_production_catalog_has_six_orthogonal_families():
    expected = [
        "route",
        "transition",
        "junction",
        "connect_ports",
        "terminate",
        "inline_component",
    ]

    assert llm_visible_modules() == expected
    assert [entry["id"] for entry in planner_catalog()] == expected
    assert all(entry["schema_version"] == 2 for entry in planner_catalog())
    assert not {
        "straight_pipe",
        "bend_pipe",
        "junction_pipe",
        "reducer_pipe",
        "connector_pipe",
        "cap_pipe",
    } & set(llm_visible_modules())


def test_planner_schema_is_compact_and_uses_only_supported_gemini_keywords():
    schema = gemini_json_schema(PlannerDecision)
    encoded = json.dumps(schema, separators=(",", ":"))
    unsupported = {
        "const",
        "default",
        "discriminator",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "oneOf",
        "title",
    }

    assert not (schema_keywords(schema) & unsupported)
    assert len(encoded) < 9500
    assert "catalog_schema_version" in schema["properties"]
    assert "choice" in schema["properties"]


def test_planner_decision_rejects_module_parameter_mismatch():
    with pytest.raises(ValidationError):
        PlannerDecision.model_validate(
            {
                "catalog_schema_version": 2,
                "target_port": "START",
                "choice": {
                    "module": "terminate",
                    "params": {
                        "path_kind": "line",
                        "section_source": "inherit_target",
                        "length": 10.0,
                        "direction": [1.0, 0.0, 0.0],
                    },
                },
                "affected_goal_ids": ["G1"],
                "completed_goal_ids": ["G1"],
                "rationale": "wrong variant on purpose",
            }
        )


def test_v2_draft_requires_llm_authored_section_and_geometry():
    engine = StateEngine(_settings())
    state = engine.initial_state(_intent(Goal(goal_id="G1", type="move")))
    missing = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={"path_kind": "line"},
    )
    explicit_without_dimensions = missing.model_copy(
        update={"params": {"path_kind": "line", "section_source": "explicit"}}
    )

    assert "section_source must be" in " ".join(validate_draft(missing, state).errors)
    errors = " ".join(validate_draft(explicit_without_dimensions, state).errors)
    assert "outer_diameter and wall_thickness" in errors
    assert "Missing LLM-authored param: length" in errors


def test_explicit_inlet_section_cannot_hide_a_mating_mismatch():
    engine = StateEngine(_settings())
    state = engine.initial_state(_intent(Goal(goal_id="G1", type="move")))
    draft = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "path_kind": "line",
            "section_source": "explicit",
            "outer_diameter": 24.0,
            "wall_thickness": 2.0,
            "length": 20.0,
            "direction": (1.0, 0.0, 0.0),
        },
    )
    resolved = engine.resolve_action(draft, state)

    assert not validate_action(resolved, state).valid
    assert "inlet outer diameter" in " ".join(validate_action(resolved, state).errors)


def test_junction_draft_rejects_outlet_that_retraces_consumed_target_axis():
    engine = StateEngine(_settings())
    state = engine.initial_state(
        _intent(
            Goal(
                goal_id="G1",
                type="branch",
                required_outlet_vectors=[(-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                include_primary_outlet=False,
            ),
            expected_open_ports=2,
        )
    )
    draft = ActionDraft(
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
                    "axis": (-1.0, 0.0, 0.0),
                    "length": 30.0,
                    "outer_diameter": 20.0,
                    "wall_thickness": 2.0,
                },
                {
                    "role": "branch",
                    "axis": (0.0, 1.0, 0.0),
                    "length": 30.0,
                    "outer_diameter": 20.0,
                    "wall_thickness": 2.0,
                },
            ],
            "blend_mode": "fillet",
            "blend_radius": 2.0,
            "inner_blend_radius": 1.0,
            "max_hub_radius": 10.0,
        },
    )

    result = validate_draft(draft, state)

    assert result.valid is False
    assert "must not retrace the consumed target-port axis" in " ".join(result.errors)


def test_two_vector_branch_contract_reports_two_expected_outputs():
    engine = StateEngine(_settings())
    intent = _intent(
        Goal(
            goal_id="G1",
            type="branch",
            required_outlet_vectors=[(0.0, 1.0, 0.0), (0.0, -1.0, 0.0)],
            include_primary_outlet=False,
            junction_style="smooth_hub",
        ),
        expected_open_ports=2,
    )
    before = engine.initial_state(intent)
    draft = ActionDraft(
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
                    "axis": (0.0, 1.0, 0.0),
                    "length": 30.0,
                    "outer_diameter": 20.0,
                    "wall_thickness": 2.0,
                },
                {
                    "role": "branch",
                    "axis": (0.0, -1.0, 0.0),
                    "length": 30.0,
                    "outer_diameter": 20.0,
                    "wall_thickness": 2.0,
                },
            ],
            "blend_mode": "fillet",
            "blend_radius": 2.0,
            "inner_blend_radius": 1.0,
            "max_hub_radius": 10.0,
        },
    )
    resolved = engine.resolve_action(draft, before)
    after = engine.apply_action(resolved, before)
    verification = build_step_verification(before, resolved, after, intent, 1)

    assert "JUNCTION_OUTPUT_COUNT_MISMATCH" not in {
        issue.issue_code for issue in verification.issues
    }
    assert "OPEN_PORT_DELTA_MISMATCH" not in {
        issue.issue_code for issue in verification.issues
    }


def test_pipeline_repairs_only_rejected_step_and_commits_only_valid_action(
    monkeypatch, tmp_path
):
    settings = replace(
        _settings(tmp_path),
        step_repair_attempts=1,
        final_repair_rounds=0,
    )

    class FakeGemini:
        prompts: list[str] = []
        planner_calls = 0

        def __init__(self, unused_settings):
            self.settings = unused_settings

        def stream_structured(self, prompt, schema, *, part):
            if part == "intent":
                return _intent(
                    Goal(goal_id="G1", type="move", direction="+X", length=20.0)
                )
            self.prompts.append(prompt)
            self.planner_calls += 1
            if self.planner_calls == 1:
                return ActionDraft(
                    target_port="START",
                    module="route",
                    catalog_schema_version=2,
                    affected_goal_ids=["G1"],
                    completed_goal_ids=["G1"],
                    params={
                        "path_kind": "line",
                        "section_source": "inherit_target",
                        "length": 20.0,
                        "direction": (0.0, 0.0, 0.0),
                    },
                )
            return _line_action("START", ["G1"], completed=["G1"])

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGemini)
    report = run_pipeline(
        "choose and place a suitable pipe primitive",
        settings,
        dry_run=False,
        stream=ThinkingStream(enabled=False),
    )
    run_dir = Path(report.artifacts.output_dir)
    actions = json.loads((run_dir / "actions.json").read_text(encoding="utf-8"))
    attempts = json.loads(
        (run_dir / "action_attempts.json").read_text(encoding="utf-8")
    )
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))

    assert [attempt["status"] for attempt in attempts] == ["rejected", "accepted"]
    assert attempts[0]["issue_codes"] == ["DRAFT_VALIDATION_FAILED"]
    assert (
        attempts[0]["observations"][0]["suggestion"]["operation"]
        == "revise_action_draft"
    )
    assert len(actions) == 1
    assert actions[0]["module"] == "route"
    assert state["state_version"] == 1
    assert report.repair_attempt_count == 1
    assert "DRAFT_VALIDATION_FAILED" in FakeGemini.prompts[1]
    assert "rejected_attempt_history" in FakeGemini.prompts[1]
    assert "direction must not be a zero vector" in FakeGemini.prompts[1]
    assert '"failure_phase":"draft_validation"' in FakeGemini.prompts[1]
    assert "move ->" not in FakeGemini.prompts[1]


def test_identical_failure_stops_before_exhausting_large_repair_budget(
    monkeypatch, tmp_path
):
    settings = replace(
        _settings(tmp_path),
        step_repair_attempts=24,
        final_repair_rounds=0,
    )

    class FakeGemini:
        planner_calls = 0

        def __init__(self, unused_settings):
            del unused_settings

        def stream_structured(self, prompt, schema, *, part):
            del prompt, schema
            if part == "intent":
                return _intent(
                    Goal(goal_id="G1", type="move", direction="+X", length=20.0)
                )
            type(self).planner_calls += 1
            if self.planner_calls <= 24:
                return ActionDraft(
                    target_port="START",
                    module="route",
                    catalog_schema_version=2,
                    affected_goal_ids=["G1"],
                    completed_goal_ids=["G1"],
                    params={
                        "path_kind": "line",
                        "section_source": "inherit_target",
                        "length": 20.0,
                        "direction": (0.0, 0.0, 0.0),
                    },
                )
            return _line_action("START", ["G1"], completed=["G1"])

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGemini)
    with pytest.raises(pipeline.StaticValidationError) as caught:
        run_pipeline(
            "one LLM-selected straight run",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
        )

    report = json.loads(Path(caught.value.artifact_path).read_text(encoding="utf-8"))
    assert FakeGemini.planner_calls == pipeline.MAX_IDENTICAL_VALIDATOR_FAILURES
    assert report["repair_attempt_count"] == pipeline.MAX_IDENTICAL_VALIDATOR_FAILURES
    assert "further identical LLM calls were stopped" in report["summary"]


def test_resume_reuses_paid_drafts_and_never_resets_step_retry_budget(
    monkeypatch, tmp_path
):
    settings = replace(
        _settings(tmp_path),
        step_repair_attempts=1,
        final_repair_rounds=0,
    )
    intent = _intent(Goal(goal_id="G1", type="move", direction="+X", length=20.0))

    class FakeGemini:
        planner_calls = 0

        def __init__(self, unused_settings):
            del unused_settings

        def stream_structured(self, prompt, schema, *, part):
            del prompt, schema
            if part == "intent":
                return intent
            type(self).planner_calls += 1
            return ActionDraft(
                target_port="START",
                module="route",
                catalog_schema_version=2,
                affected_goal_ids=["G1"],
                completed_goal_ids=["G1"],
                params={
                    "path_kind": "line",
                    "section_source": "inherit_target",
                    "length": 20.0,
                    # An explicit zero direction remains invalid even though
                    # omission now means inherit the target-port tangent.
                    "direction": (0.0, 0.0, 0.0),
                },
            )

    real_validate_draft = pipeline.validate_draft
    validation_calls = 0

    def interrupt_after_paid_draft(draft, state):
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls in {1, 3}:
            raise KeyboardInterrupt("simulated crash after paid planner response")
        return real_validate_draft(draft, state)

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGemini)
    monkeypatch.setattr(
        pipeline,
        "validate_draft",
        interrupt_after_paid_draft,
    )

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            "one LLM-selected straight run",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
        )
    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    first_checkpoint = json.loads(
        (run_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert first_checkpoint["pending_draft"] is not None
    assert first_checkpoint["pending_draft_attempt_index"] == 1
    assert first_checkpoint["next_attempt_index"] == 1

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            "",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
            resume_dir=run_dir,
        )
    assert FakeGemini.planner_calls == 2

    with pytest.raises(pipeline.StaticValidationError):
        run_pipeline(
            "",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
            resume_dir=run_dir,
        )
    exhausted_checkpoint = json.loads(
        (run_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert exhausted_checkpoint["pending_draft"] is None
    assert exhausted_checkpoint["next_attempt_index"] == 3
    assert [
        (item["attempt_index"], item["status"])
        for item in exhausted_checkpoint["attempts"]
    ] == [(1, "rejected"), (2, "rejected")]
    assert FakeGemini.planner_calls == settings.step_repair_attempts + 1

    # Repeated resume remains exhausted; it cannot buy another planner call.
    with pytest.raises(pipeline.StaticValidationError):
        run_pipeline(
            "",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
            resume_dir=run_dir,
        )
    assert FakeGemini.planner_calls == settings.step_repair_attempts + 1


def test_one_llm_action_can_complete_multiple_goals_and_all_are_checked():
    engine = StateEngine(_settings())
    intent = _intent(
        Goal(goal_id="G1", type="move", direction="+X", length=20.0),
        Goal(goal_id="G2", type="route", notes="same physical run"),
    )
    before = engine.initial_state(intent)
    resolved = engine.resolve_action(
        _line_action("START", ["G1", "G2"], completed=["G1", "G2"]),
        before,
    )
    after = engine.apply_action(resolved, before)
    step = build_step_verification(before, resolved, after, intent, 1)

    assert after.remaining_goals == []
    assert [goal["goal_id"] for goal in step.transition.affected_goals] == ["G1", "G2"]
    assert step.status == "passed"


def test_one_goal_can_span_multiple_llm_actions_without_implicit_completion():
    engine = StateEngine(_settings())
    intent = _intent(Goal(goal_id="G1", type="route"))
    state0 = engine.initial_state(intent)
    action1 = engine.resolve_action(
        _line_action("START", ["G1"], completed=[], length=10.0), state0
    )
    state1 = engine.apply_action(action1, state0)
    action2 = engine.resolve_action(
        _line_action("M1.out", ["G1"], completed=["G1"], length=10.0), state1
    )
    state2 = engine.apply_action(action2, state1)

    assert [goal.goal_id for goal in state1.remaining_goals] == ["G1"]
    assert state2.remaining_goals == []
    assert [action.action_id for action in state2.action_history] == ["A1", "A2"]


def test_completed_move_goal_checks_cumulative_llm_authored_length():
    engine = StateEngine(_settings())
    intent = _intent(Goal(goal_id="G1", type="move", direction="+X", length=20.0))
    state0 = engine.initial_state(intent)
    first = engine.resolve_action(
        _line_action("START", ["G1"], completed=[], length=10.0), state0
    )
    state1 = engine.apply_action(first, state0)
    second = engine.resolve_action(
        _line_action("M1.out", ["G1"], completed=["G1"], length=10.0), state1
    )
    state2 = engine.apply_action(second, state1)
    valid_step = build_step_verification(state1, second, state2, intent, 2)

    assert "GOAL_LENGTH_MISMATCH" not in {
        issue.issue_code for issue in valid_step.issues
    }

    wrong = engine.resolve_action(
        _line_action("M1.out", ["G1"], completed=["G1"], length=5.0), state1
    )
    wrong_state = engine.apply_action(wrong, state1)
    wrong_step = build_step_verification(state1, wrong, wrong_state, intent, 2)
    assert "GOAL_LENGTH_MISMATCH" in {issue.issue_code for issue in wrong_step.issues}


def test_completed_turn_goal_checks_authored_sweep_angle():
    engine = StateEngine(_settings())
    intent = _intent(Goal(goal_id="G1", type="turn", direction="+Z", angle=90.0))
    before = engine.initial_state(intent)
    draft = ActionDraft(
        target_port="START",
        module="bend_pipe",
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={"angle": 45.0, "turn_direction": "+Z", "bend_radius": 30.0},
    )
    resolved = engine.resolve_action(draft, before)
    after = engine.apply_action(resolved, before)
    step = build_step_verification(before, resolved, after, intent, 1)

    assert "GOAL_TURN_ANGLE_MISMATCH" in {issue.issue_code for issue in step.issues}


def test_junction_then_connect_ports_forms_one_explicit_graph_cycle():
    settings = _settings()
    engine = StateEngine(settings)
    intent = _intent(
        Goal(goal_id="G1", type="branch", branch_count=2),
        Goal(goal_id="G2", type="connect"),
        expected_open_ports=0,
    )
    state0 = engine.initial_state(intent)
    junction = ActionDraft(
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
                    "axis": (0.0, 1.0, 0.0),
                    "length": 20.0,
                    "outer_diameter": 20.0,
                    "wall_thickness": 2.0,
                },
                {
                    "role": "branch",
                    "axis": (0.0, -1.0, 0.0),
                    "length": 20.0,
                    "outer_diameter": 20.0,
                    "wall_thickness": 2.0,
                },
            ],
            "blend_mode": "hard",
            "blend_radius": 3.0,
            "inner_blend_radius": 2.0,
            "max_hub_radius": 12.0,
        },
    )
    resolved_junction = engine.resolve_action(junction, state0)
    assert validate_action(resolved_junction, state0).valid
    state1 = engine.apply_action(resolved_junction, state0)
    connect = ActionDraft(
        target_port="M1.out_1",
        module="connect_ports",
        catalog_schema_version=2,
        affected_goal_ids=["G2"],
        completed_goal_ids=["G2"],
        params={
            "other_port_id": "M1.out_2",
            "path_kind": "spline",
            "section_source": "inherit_target",
            "waypoints": [(20.0, 30.0, 0.0), (20.0, -30.0, 0.0)],
            "interpolation": "bspline",
            "frenet": True,
            "minimum_curvature_radius": 10.1,
        },
    )
    resolved_connect = engine.resolve_action(connect, state1)
    assert validate_action(resolved_connect, state1).valid
    state2 = engine.apply_action(resolved_connect, state1)
    step = build_step_verification(state1, resolved_connect, state2, intent, 2)
    step = step.model_copy(
        update={"mcp_measurements": {"M2": {"minimum_curvature_radius": 20.0}}}
    )
    critic = build_final_critic_report(intent, state2, [step])

    assert step.status == "passed"
    assert len(step.transition.consumed_port_ids) == 2
    assert step.transition.produced_port_ids == []
    assert len(step.transition.connection_edge_ids) == 2
    assert state2.open_ports == []
    assert critic.passed is True


def test_connect_ports_derives_exact_endpoint_tangents_from_ports():
    settings = _settings()
    engine = StateEngine(settings)
    intent = _intent(
        Goal(goal_id="G1", type="branch", branch_count=2),
        Goal(goal_id="G2", type="connect"),
        expected_open_ports=0,
    )
    state0 = engine.initial_state(intent)
    # Reuse a legacy junction only to create two deterministic open ports.
    branch = engine.resolve_action(
        ActionDraft(
            target_port="START",
            module="junction_pipe",
            params={"branch_count": 2, "branch_angles": [90.0, -90.0]},
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
        ),
        state0,
    )
    state1 = engine.apply_action(branch, state0)
    draft = ActionDraft(
        target_port="M1.out_1",
        module="connect_ports",
        catalog_schema_version=2,
        affected_goal_ids=["G2"],
        completed_goal_ids=["G2"],
        params={
            "other_port_id": "M1.out_2",
            "path_kind": "spline",
            "section_source": "inherit_target",
            "waypoints": [(0.0, 30.0, 10.0), (0.0, -30.0, 10.0)],
            "interpolation": "bspline",
            "frenet": True,
            "minimum_curvature_radius": 10.1,
        },
    )
    resolved = engine.resolve_action(draft, state1)
    target = next(port for port in state1.open_ports if port.id == "M1.out_1")
    other = next(port for port in state1.open_ports if port.id == "M1.out_2")
    assert resolved.params["initial_tangent"] == pytest.approx(target.axis)
    assert resolved.params["final_tangent"] == pytest.approx(
        tuple(-value for value in other.axis)
    )
    corrupted_params = dict(resolved.params)
    corrupted_params["final_tangent"] = other.axis
    corrupted = resolved.model_copy(update={"params": corrupted_params})
    corrupted_result = validate_action(corrupted, state1)
    assert not corrupted_result.valid
    assert any(
        "final_tangent invariant mismatch" in error for error in corrupted_result.errors
    )
    after = engine.apply_action(resolved, state1)
    step = build_step_verification(state1, resolved, after, intent, 2)

    assert "CONNECT_END_TANGENT_MISMATCH" not in {
        issue.issue_code for issue in step.issues
    }


def test_generated_script_safely_quotes_json_and_preserves_spline_tangents():
    engine = StateEngine(_settings())
    intent = _intent(Goal(goal_id="G1", type="route"))
    before = engine.initial_state(intent)
    spline = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        rationale='유니코드 "quote" \\ slash',
        params={
            "path_kind": "spline",
            "section_source": "inherit_target",
            "waypoints": [(20.0, 0.0, 5.0), (40.0, 10.0, 10.0)],
            "final_tangent": (1.0, 0.0, 0.0),
            "interpolation": "bspline",
            "frenet": True,
            "minimum_curvature_radius": 10.1,
        },
    )
    state = engine.apply_action(engine.resolve_action(spline, before), before)
    state.placed_modules[0].params["metadata"] = {
        "label": '한글 "quote" \\ path',
        "enabled": True,
        "optional": None,
    }
    script = build_freecad_script(state)

    compile(script, "generated_freecad.py", "exec")
    assert "PAYLOAD = json.loads(" in script
    # The resolver overrides legacy planner kernel-policy values and the
    # generator turns endpoint tangent directions into explicit cubic handles.
    assert state.placed_modules[0].params["frenet"] is False
    assert "curve = Part.BezierCurve()" in script
    assert "vectors[index] + tangents[index] * handles[index]" in script
    assert "vectors[index + 1] - tangents[index + 1] * handles[index + 1]" in script
    assert "tangents[index].dot(chord_direction) >= 1.0 - 1e-10" in script
    assert "Part.makeLine(vectors[index], vectors[index + 1])" in script
    assert "Part.Wire(span_edges)" in script
    assert script.index(
        "centerline_checks[module_id] = centerline_check(module)"
    ) < script.index("shape, outer, bore = make_module_shape(module)")
    assert "endpoint_tangent_dots" in script
    assert "curvature_repair_hint" in script
    assert "minimum_radius_nearest_path_point_index" in script
    assert "boxes_disjoint" in script
    assert "left_box.XMax < right_box.XMin - MODELING_TOLERANCE" in script
    assert "sampled_minimum_radius" in script
    assert "optimized_handle_factors" in script
    assert "makePipeShell" in script


def test_freecad_validation_accepts_bound_evidence_and_rejects_spoofed_results():
    engine = StateEngine(_settings())
    before = engine.initial_state(_intent(Goal(goal_id="G1", type="move")))
    state = engine.apply_action(
        engine.resolve_action(_line_action("START", ["G1"], completed=["G1"]), before),
        before,
    )
    evidence = _validation_evidence(state)
    valid = _text_result("CADGEN_VALIDATION=", evidence)

    assert (
        assess_freecad_validation(
            valid,
            expected_digest=geometry_payload_digest(state),
            expected_state_id=state.state_id,
            expected_module_ids=["M1"],
            expected_open_port_count=1,
        )["passed"]
        is True
    )

    duplicate = {
        "content": [
            {
                "type": "text",
                "text": "\n".join(
                    [
                        "CADGEN_VALIDATION=" + json.dumps(evidence),
                        "CADGEN_VALIDATION=" + json.dumps(evidence),
                    ]
                ),
            }
        ]
    }
    with pytest.raises(FreeCADMCPError, match="exactly one"):
        assess_freecad_validation(
            duplicate,
            expected_digest=geometry_payload_digest(state),
            expected_state_id=state.state_id,
            expected_module_ids=["M1"],
        )
    stale = _text_result(
        "CADGEN_VALIDATION=", _validation_evidence(state, digest="stale")
    )
    with pytest.raises(FreeCADMCPError, match="digest mismatch"):
        assess_freecad_validation(
            stale,
            expected_digest=geometry_payload_digest(state),
            expected_state_id=state.state_id,
            expected_module_ids=["M1"],
        )
    with pytest.raises(FreeCADMCPError, match="Malformed"):
        assess_freecad_validation(
            {"content": [{"type": "text", "text": "CADGEN_VALIDATION={bad"}]},
            expected_digest=geometry_payload_digest(state),
            expected_state_id=state.state_id,
            expected_module_ids=["M1"],
        )
    with pytest.raises(FreeCADMCPError, match="isError=true"):
        assess_freecad_validation(
            {"isError": True, **valid},
            expected_digest=geometry_payload_digest(state),
            expected_state_id=state.state_id,
            expected_module_ids=["M1"],
        )
    with pytest.raises(FreeCADMCPError, match="execution failure"):
        assess_freecad_validation(
            {"content": [{"type": "text", "text": "Error executing code: NameError"}]},
            expected_digest=geometry_payload_digest(state),
            expected_state_id=state.state_id,
            expected_module_ids=["M1"],
        )


def test_freecad_evidence_dict_order_does_not_break_ten_module_runs():
    engine = StateEngine(_settings())
    intent = _intent(Goal(goal_id="G1", type="move", direction="+X", length=10.0))
    before = engine.initial_state(intent)
    state = engine.apply_action(
        engine.resolve_action(
            _line_action("START", ["G1"], completed=["G1"], length=10.0),
            before,
        ),
        before,
    )
    evidence = _validation_evidence(state)
    module_ids = [f"M{index}" for index in range(1, 11)]
    # The generated sentinel uses sort_keys=True, placing M10 between M1/M2.
    evidence["module_ids"] = module_ids
    evidence["checks"]["modules"] = {
        module_id: {"passed": True} for module_id in sorted(module_ids)
    }
    evidence["checks"]["centerlines"] = {
        module_id: {"passed": True} for module_id in sorted(module_ids)
    }

    assessed = assess_freecad_validation(
        _text_result("CADGEN_VALIDATION=", evidence),
        expected_digest=geometry_payload_digest(state),
        expected_state_id=state.state_id,
        expected_module_ids=module_ids,
        expected_open_port_count=len(state.open_ports),
    )
    assert assessed["module_ids"] == module_ids


def test_freecad_publish_requires_exact_digest_and_document():
    evidence = {
        "passed": True,
        "payload_digest": "abc",
        "published_document": "CadGenPipe_run_v1",
    }
    result = _text_result("CADGEN_PUBLISH=", evidence)

    assert (
        assess_freecad_publish(
            result,
            expected_digest="abc",
            expected_document="CadGenPipe_run_v1",
        )["passed"]
        is True
    )
    with pytest.raises(FreeCADMCPError, match="digest mismatch"):
        assess_freecad_publish(
            result,
            expected_digest="other",
            expected_document="CadGenPipe_run_v1",
        )


class _TinyResult(BaseModel):
    value: int


def _gemini_client_with_output(tmp_path: Path, output_text: str) -> GeminiClient:
    """실제 SDK 없이 구조화 응답 경계만 검증하는 테스트 클라이언트."""

    settings = replace(
        _settings(tmp_path),
        gemini_api_key="test-only",
        gemini_max_calls=5,
        gemini_max_total_tokens=50000,
        gemini_stateful=False,
    )

    class FakeInteractions:
        def create(self, **body):
            del body
            return SimpleNamespace(
                id="strict-json-test",
                status="completed",
                output_text=output_text,
                usage=SimpleNamespace(
                    total_input_tokens=10,
                    total_cached_tokens=0,
                    total_output_tokens=3,
                    total_thought_tokens=0,
                    total_tool_use_tokens=0,
                    total_tokens=13,
                ),
            )

    client = GeminiClient.__new__(GeminiClient)
    client._settings = settings
    client._client = SimpleNamespace(interactions=FakeInteractions())
    client._lineages = {}
    client._usage = LLMUsage()
    return client


_STRUCTURED_JSON_PATHS = [
    pytest.param("patch", None, id="ordinary-json"),
    pytest.param("step_planner", ["1"], id="numeric-literal-json"),
    pytest.param("intent", None, id="encoded-decimal-json"),
]


@pytest.mark.parametrize(("part", "numeric_literals"), _STRUCTURED_JSON_PATHS)
@pytest.mark.parametrize(
    ("raw_json", "expected_message"),
    [
        ('{"value":1,"value":2}', "duplicate JSON object key"),
        (
            '{"value":1,"metadata":{"name":"first","name":"second"}}',
            "duplicate JSON object key",
        ),
        ('{"value":NaN}', "non-standard JSON numeric constant"),
        ('{"value":Infinity}', "non-standard JSON numeric constant"),
        ('{"value":-Infinity}', "non-standard JSON numeric constant"),
        ('{"value":1e999}', "non-finite JSON number"),
    ],
)
def test_gemini_client_rejects_nonstandard_json_on_every_structured_path(
    tmp_path,
    part,
    numeric_literals,
    raw_json,
    expected_message,
):
    client = _gemini_client_with_output(tmp_path, raw_json)
    kwargs = (
        {"numeric_literals": numeric_literals} if numeric_literals is not None else {}
    )

    with pytest.raises(StructuredOutputError) as exc_info:
        client.stream_structured("test", _TinyResult, part=part, **kwargs)

    assert expected_message in str(exc_info.value.cause)


@pytest.mark.parametrize(("part", "numeric_literals"), _STRUCTURED_JSON_PATHS)
def test_gemini_client_applies_json_schema_before_pydantic_on_every_path(
    tmp_path,
    part,
    numeric_literals,
):
    # Pydantic 자체는 문자열 "1"을 정수로 coercion할 수 있다. 모든 경로에서
    # 먼저 JSON Schema를 적용해야 공급자에게 전달한 계약과 로컬 검증이 같다.
    client = _gemini_client_with_output(tmp_path, '{"value":"1"}')
    kwargs = (
        {"numeric_literals": numeric_literals} if numeric_literals is not None else {}
    )

    with pytest.raises(StructuredOutputError) as exc_info:
        client.stream_structured("test", _TinyResult, part=part, **kwargs)

    assert isinstance(exc_info.value.cause, JSONSchemaValidationError)


class _HostRelationalResult(BaseModel):
    lower: int
    upper: int

    @model_validator(mode="after")
    def validate_order(self):
        if self.lower >= self.upper:
            raise ValueError("lower must be below upper")
        return self


def test_provider_schema_pass_host_semantic_failure_has_distinct_error(tmp_path):
    client = _gemini_client_with_output(
        tmp_path,
        '{"lower":5,"upper":1}',
    )

    with pytest.raises(HostContractValidationError) as captured:
        client.stream_structured(
            "test",
            _HostRelationalResult,
            part="patch",
        )

    assert not isinstance(captured.value, StructuredOutputError)
    assert "lower must be below upper" in str(captured.value.cause)


def test_gemini_client_resends_controls_tracks_usage_and_reuses_lineage(tmp_path):
    settings = replace(
        _settings(tmp_path),
        gemini_api_key="test-only",
        gemini_max_calls=2,
        gemini_max_total_tokens=100000,
        gemini_stateful=True,
    )

    class FakeInteractions:
        def __init__(self):
            self.calls = []

        def create(self, **body):
            self.calls.append(body)
            index = len(self.calls)
            usage = SimpleNamespace(
                total_input_tokens=10,
                total_cached_tokens=2 if index == 2 else 0,
                total_output_tokens=3,
                total_thought_tokens=1,
                total_tool_use_tokens=0,
                total_tokens=14,
            )
            return SimpleNamespace(
                id=f"interaction-{index}",
                output_text='{"value":1}',
                usage=usage,
            )

    fake_interactions = FakeInteractions()
    client = GeminiClient.__new__(GeminiClient)
    client._settings = settings
    client._client = SimpleNamespace(interactions=fake_interactions)
    client._lineages = {}
    client._usage = LLMUsage()

    system = "stable planner policy"
    assert (
        client.stream_structured(
            "first",
            _TinyResult,
            part="step_planner",
            system_instruction=system,
        ).value
        == 1
    )
    assert (
        client.stream_structured(
            "second",
            _TinyResult,
            part="step_planner",
            system_instruction=system,
        ).value
        == 1
    )
    first, second = fake_interactions.calls

    assert "previous_interaction_id" not in first
    assert second["previous_interaction_id"] == "interaction-1"
    for body in (first, second):
        assert body["system_instruction"] == system
        assert body["response_format"]["mime_type"] == "application/json"
        assert (
            body["generation_config"]["max_output_tokens"]
            == settings.gemini_max_output_tokens
        )
        assert "temperature" not in body["generation_config"]
        assert "seed" not in body["generation_config"]
        assert "schema" in body["response_format"]
    assert client.usage_snapshot().calls == 2
    assert client.usage_snapshot().cached_tokens == 2
    assert client.policy_snapshot()["models"]["step_planner"] == (
        settings.model_for("step_planner")
    )
    with pytest.raises(GeminiBudgetError, match="call ceiling"):
        client.stream_structured("third", _TinyResult, part="step_planner")


def test_gemini_client_uses_part_specific_and_actual_budget_limited_output_caps(
    tmp_path,
):
    class FakeInteractions:
        def __init__(self):
            self.calls = []

        def create(self, **body):
            self.calls.append(body)
            return SimpleNamespace(
                id=f"interaction-{len(self.calls)}",
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

    def client_for(settings):
        fake = FakeInteractions()
        client = GeminiClient.__new__(GeminiClient)
        client._settings = settings
        client._client = SimpleNamespace(interactions=fake)
        client._lineages = {}
        client._usage = LLMUsage()
        return client, fake

    ample = replace(
        _settings(tmp_path),
        gemini_api_key="test-only",
        gemini_max_total_tokens=100000,
        gemini_max_output_tokens=4096,
        gemini_intent_max_output_tokens=16384,
    )
    ample_client, ample_fake = client_for(ample)
    assert (
        ample_client.stream_structured("intent", _TinyResult, part="intent").value == 1
    )
    assert (
        ample_fake.calls[0]["generation_config"]["max_output_tokens"]
        == ample.gemini_intent_max_output_tokens
    )

    constrained = replace(ample, gemini_max_total_tokens=3000)
    constrained_client, constrained_fake = client_for(constrained)
    schema = gemini_json_schema(_TinyResult, encode_decimals=True)
    expected_limit = constrained_client._request_output_limit(
        "intent",
        schema,
        None,
        part="intent",
    )
    assert 256 <= expected_limit < constrained.gemini_intent_max_output_tokens
    assert (
        constrained_client.stream_structured("intent", _TinyResult, part="intent").value
        == 1
    )
    assert (
        constrained_fake.calls[0]["generation_config"]["max_output_tokens"]
        == expected_limit
    )


@pytest.mark.parametrize(
    ("status", "expected_error"),
    [
        ("incomplete", StructuredOutputIncompleteError),
        ("budget_exceeded", StructuredOutputIncompleteError),
        ("failed", GeminiRequestError),
        ("cancelled", GeminiRequestError),
        ("in_progress", GeminiRequestError),
        ("requires_action", GeminiRequestError),
        ("future_status", GeminiRequestError),
    ],
)
def test_gemini_client_rejects_noncompleted_status_before_parse_or_lineage(
    tmp_path,
    status,
    expected_error,
):
    settings = replace(
        _settings(tmp_path),
        gemini_api_key="test-only",
        gemini_max_calls=5,
        gemini_max_total_tokens=50000,
    )

    class FakeInteractions:
        def __init__(self):
            self.calls = []

        def create(self, **body):
            self.calls.append(body)
            return SimpleNamespace(
                id="must-not-be-lineage",
                status=status,
                output_text='{"value":1,"partial_secret":"DO_NOT_ECHO"}',
                usage=SimpleNamespace(
                    total_input_tokens=10,
                    total_cached_tokens=0,
                    total_output_tokens=7,
                    total_thought_tokens=100,
                    total_tool_use_tokens=0,
                    total_tokens=117,
                ),
            )

    fake = FakeInteractions()
    client = GeminiClient.__new__(GeminiClient)
    client._settings = settings
    client._client = SimpleNamespace(interactions=fake)
    client._lineages = {}
    client._usage = LLMUsage()

    with pytest.raises(expected_error) as exc_info:
        client.stream_structured("intent", _TinyResult, part="intent")

    assert client.usage_snapshot().calls == 1
    assert client.usage_snapshot().total_tokens == 117
    assert client.lineage_snapshot() == {}
    assert "DO_NOT_ECHO" not in str(exc_info.value)
    if isinstance(exc_info.value, StructuredOutputIncompleteError):
        assert exc_info.value.status == status
        assert exc_info.value.output_tokens == 7
        assert exc_info.value.thought_tokens == 100
        assert (
            exc_info.value.output_limit
            == fake.calls[0]["generation_config"]["max_output_tokens"]
        )


@pytest.mark.parametrize(
    ("status_code", "provider_code", "expected_invalid_request"),
    [
        (400, "invalid_request", True),
        (400, "invalid_argument", True),
        (401, "invalid_request", False),
        (429, "invalid_request", False),
        (500, "invalid_argument", False),
    ],
)
def test_gemini_client_classifies_only_http_400_invalid_request_for_schema_fallback(
    tmp_path,
    status_code,
    provider_code,
    expected_invalid_request,
):
    settings = replace(
        _settings(tmp_path),
        gemini_api_key="test-only",
        gemini_max_calls=5,
        gemini_max_total_tokens=50000,
    )

    class ProviderError(Exception):
        def __init__(self):
            self.status_code = status_code
            self.body = {
                "error": {
                    "message": "Request contains an invalid argument.",
                    "code": provider_code,
                }
            }
            super().__init__(f"Error code: {status_code} - {self.body}")

    class FakeInteractions:
        def create(self, **body):
            del body
            raise ProviderError()

    client = GeminiClient.__new__(GeminiClient)
    client._settings = settings
    client._client = SimpleNamespace(interactions=FakeInteractions())
    client._lineages = {}
    client._usage = LLMUsage()

    with pytest.raises(GeminiRequestError) as exc_info:
        client.stream_structured("intent", _TinyResult, part="intent")

    assert isinstance(exc_info.value, GeminiInvalidRequestError) is (
        expected_invalid_request
    )
    if expected_invalid_request:
        assert exc_info.value.status_code == 400
        assert exc_info.value.provider_code == provider_code
    assert client.usage_snapshot().calls == 0
    assert client.lineage_snapshot() == {}


def test_post_call_total_budget_ceiling_precedes_incomplete_retry(tmp_path):
    settings = replace(
        _settings(tmp_path),
        gemini_api_key="test-only",
        gemini_max_total_tokens=2000,
    )

    class FakeInteractions:
        def create(self, **body):
            del body
            return SimpleNamespace(
                id="over-budget-incomplete",
                status="incomplete",
                output_text='{"value":',
                usage=SimpleNamespace(
                    total_input_tokens=1000,
                    total_cached_tokens=0,
                    total_output_tokens=500,
                    total_thought_tokens=700,
                    total_tool_use_tokens=0,
                    total_tokens=2200,
                ),
            )

    client = GeminiClient.__new__(GeminiClient)
    client._settings = settings
    client._client = SimpleNamespace(interactions=FakeInteractions())
    client._lineages = {}
    client._usage = LLMUsage()

    with pytest.raises(
        GeminiBudgetError, match="above the conservative request ceiling"
    ):
        client.stream_structured("x", _TinyResult, part="intent")
    assert client.usage_snapshot().calls == 1
    assert client.lineage_snapshot() == {}


def test_prompt_contains_no_deterministic_goal_to_module_mapping():
    state = StateEngine(_settings()).initial_state(
        _intent(Goal(goal_id="G1", type="turn", direction="+Z", angle=90.0))
    )
    prompt = step_planner_prompt(state)
    system = step_planner_system_instruction()

    assert "There is no keyword-driven system planner" in system
    assert "lead-in waypoints that continue/ease the incoming curvature" in system
    assert "Handle optimization cannot change a bend's curvature direction" in system
    assert "branch -> junction" not in system
    assert "Compact planning payload" in prompt
    for forbidden in (
        "move -> route",
        "turn -> route",
        "branch -> junction",
        "diameter_change -> transition",
        "end -> terminate",
    ):
        assert forbidden not in prompt


def test_production_intent_rejects_invalid_section_axis_and_duplicate_goal_ids():
    payload = {
        "global_spec": {
            "outer_diameter": 20.0,
            "wall_thickness": 2.0,
            "is_hollow": True,
            "units": "mm",
        },
        "start_position": [0.0, 0.0, 0.0],
        "start_axis": [0.0, 0.0, 0.0],
        "target_behavior": [
            {
                "goal_id": "G1",
                "depends_on_goal_ids": [],
                "allow_parallel": False,
                "type": "move",
                "direction": "+X",
                "length": 10.0,
            },
            {
                "goal_id": "G1",
                "depends_on_goal_ids": [],
                "allow_parallel": False,
                "type": "move",
                "direction": "+X",
                "length": 10.0,
            },
        ],
        "expected_open_ports": 1,
        "expected_open_ports_source": "explicit",
        "required_components": [],
        "hard_constraints": [],
        "geometric_constraints": [],
    }

    with pytest.raises(ValidationError):
        ProductionIntent.model_validate(payload)
    payload["start_axis"] = [1.0, 0.0, 0.0]
    payload["global_spec"]["wall_thickness"] = 10.0
    with pytest.raises(ValidationError, match="twice wall_thickness"):
        ProductionIntent.model_validate(payload)


def test_intent_schema_failure_gets_one_local_llm_repair():
    settings = replace(_settings(), step_repair_attempts=1)

    class FakeGemini:
        calls = []

        def stream_structured(self, prompt, schema, *, part):
            self.calls.append(prompt)
            if len(self.calls) == 1:
                raise StructuredOutputError(
                    "intent", '{"bad":true}', ValueError("missing global_spec")
                )
            return ProductionIntent.model_validate(
                {
                    "global_spec": {
                        "outer_diameter": 20.0,
                        "wall_thickness": 2.0,
                        "is_hollow": True,
                        "units": "mm",
                    },
                    "start_position": [0.0, 0.0, 0.0],
                    "start_axis": [1.0, 0.0, 0.0],
                    "target_behavior": [
                        {
                            "goal_id": "G1",
                            "depends_on_goal_ids": [],
                            "allow_parallel": False,
                            "type": "move",
                            "direction": "+X",
                            "length": 10.0,
                        }
                    ],
                    "expected_open_ports": 1,
                    "expected_open_ports_source": "explicit",
                    "required_components": [],
                    "hard_constraints": [],
                    "geometric_constraints": [],
                }
            )

    result = pipeline._extract_intent(
        "straight pipe",
        settings,
        dry_run=False,
        gemini=FakeGemini(),
    )

    assert result.target_behavior[0].length == 10.0
    assert len(FakeGemini.calls) == 2
    assert "Validation diagnostic" in FakeGemini.calls[1]


def test_final_agenda_repair_localizes_only_and_cannot_rewrite_goals():
    directive = AgendaRepairDirective(
        scope="agenda",
        rollback_step=1,
        target_issue_ids=["FINAL_01_EXAMPLE"],
        target_module_ids=["M1"],
        repair_hint="choose different geometry",
        rationale="localized final defect",
    )

    assert directive.rollback_step == 1
    assert "revised_goals" not in AgendaRepairDirective.model_fields
    with pytest.raises(ValidationError):
        AgendaRepairDirective.model_validate(
            {
                **directive.model_dump(mode="json"),
                "revised_goals": [{"goal_id": "G1_REPAIR", "type": "route"}],
            }
        )
    with pytest.raises(ValidationError):
        AgendaRepairDirective.model_validate(
            {**directive.model_dump(mode="json"), "scope": "step"}
        )


def test_required_component_cannot_silently_disappear_from_final_state():
    engine = StateEngine(_settings())
    intent = _intent(
        Goal(
            goal_id="G1",
            type="connector",
            direction="+X",
            length=20.0,
            component="flange",
        )
    ).model_copy(update={"required_components": ["flange"]})
    before = engine.initial_state(intent)
    draft = _line_action("START", ["G1"], completed=["G1"])
    resolved = engine.resolve_action(draft, before)
    after = engine.apply_action(resolved, before)
    step = build_step_verification(before, resolved, after, intent, 1)
    critic = build_final_critic_report(intent, after, [step])

    assert "REQUIRED_COMPONENTS_UNSATISFIED" in {
        issue.issue_code for issue in critic.issues
    }

    claimed = draft.model_copy(update={"satisfied_components": ["flange"]})
    assert not validate_draft(claimed, before).valid

    component_draft = ActionDraft(
        target_port="START",
        module="inline_component",
        catalog_schema_version=2,
        params={
            "section_source": "inherit_target",
            "component_type": "flange",
            "length": 20.0,
            "body_outer_diameter": 40.0,
            "body_start_offset": 0.0,
            "body_length": 4.0,
            "flange_bolt_count": 4,
            "flange_bolt_circle_diameter": 30.0,
            "flange_bolt_hole_diameter": 4.0,
            "flange_reference_axis": [0.0, 1.0, 0.0],
            "connector_type_out": "plain",
            "connector_gender_out": "neutral",
            "connector_standard_out": None,
        },
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
    )
    assert validate_draft(component_draft, before).valid
    component_action = engine.resolve_action(component_draft, before)
    component_state = engine.apply_action(component_action, before)
    component_step = build_step_verification(
        before, component_action, component_state, intent, 1
    )
    assert build_final_critic_report(intent, component_state, [component_step]).passed


def _prepared_checkpoint(tmp_path: Path, *, phase: str):
    run_dir = tmp_path / "resume_case"
    run_dir.mkdir()
    prompt = "make a 20 mm straight hollow pipe"
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    settings = _settings(tmp_path)
    engine = StateEngine(settings)
    intent = pipeline._bind_contract(
        prompt,
        _intent(Goal(goal_id="G1", type="move", direction="+X", length=20.0)),
    )
    previous = engine.initial_state(intent)
    draft = _line_action("START", ["G1"], completed=["G1"])
    action = engine.resolve_action(draft, previous)
    candidate = engine.apply_action(action, previous)
    step = build_step_verification(previous, action, candidate, intent, 1)
    evidence = _validation_evidence(candidate, run_id=run_dir.name)
    payload = {
        "phase": phase,
        "run_id": run_dir.name,
        "intent": intent.model_dump(mode="json"),
        "previous_state": previous.model_dump(mode="json"),
        "candidate_state": candidate.model_dump(mode="json"),
        "candidate_digest": geometry_payload_digest(candidate),
        "candidate_state_digest": pipeline._pipe_state_digest(candidate),
        "candidate_document": candidate_document_name(
            candidate, run_id=run_dir.name, attempt_id=1
        ),
        "published_document": published_document_name(candidate, run_id=run_dir.name),
        "previous_freecad_verified": False,
        "draft": draft.model_dump(mode="json"),
        "action": action.model_dump(mode="json"),
        "attempt_id": 1,
        "step_verification": step.model_dump(mode="json"),
        "actions": [],
        "step_verifications": [],
        "attempts": [],
        "committed_states": [previous.model_dump(mode="json")],
        "planner_lineage": {"step_planner": "interaction-before-crash"},
        "planner_schema_profiles": {"S0": "encoded"},
    }
    if phase == "PUBLISHED":
        payload["evidence"] = evidence
    checkpoint = run_dir / "checkpoint.json"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    return run_dir, checkpoint, settings, engine, previous, candidate


def test_resume_refuses_uncertain_prepared_candidate_without_mcp(tmp_path):
    run_dir, checkpoint, settings, engine, unused_previous, unused_candidate = (
        _prepared_checkpoint(tmp_path, phase="PREPARED")
    )
    del unused_previous, unused_candidate

    with pytest.raises(FreeCADMCPError, match="requires live FreeCAD MCP"):
        pipeline._load_resume_context(
            checkpoint,
            settings,
            engine,
            dry_run=True,
            run_dir=run_dir,
            expected_run_id=run_dir.name,
        )


def test_resume_rejects_legacy_higher_degree_branch_without_migration(tmp_path):
    run_dir, checkpoint, settings, engine, unused_previous, unused_candidate = (
        _prepared_checkpoint(tmp_path, phase="PREPARED")
    )
    del unused_previous, unused_candidate
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["intent"]["target_behavior"] = [
        {
            "goal_id": "legacy_three_outlet",
            "depends_on_goal_ids": [],
            "allow_parallel": False,
            "type": "branch",
            "branch_count": 2,
            "include_primary_outlet": True,
        }
    ]
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="non-binary branch goals and cannot be deterministically migrated",
    ):
        pipeline._load_resume_context(
            checkpoint,
            settings,
            engine,
            dry_run=False,
            run_dir=run_dir,
            expected_run_id=run_dir.name,
        )


def test_resume_rolls_forward_published_candidate_from_bound_evidence(tmp_path):
    run_dir, checkpoint, settings, engine, unused_previous, candidate = (
        _prepared_checkpoint(tmp_path, phase="PUBLISHED")
    )
    del unused_previous

    context = pipeline._load_resume_context(
        checkpoint,
        settings,
        engine,
        dry_run=True,
        run_dir=run_dir,
        expected_run_id=run_dir.name,
    )

    assert context.state.state_id == candidate.state_id
    assert [action["action_id"] for action in context.actions] == ["A1"]
    assert context.step_verifications[-1].mcp_status == "skipped"
    assert context.attempts[-1].status == "accepted"
    assert context.semantic_mcp_passed is False
    assert "step_planner" not in context.planner_lineage
    assert context.planner_schema_profiles == {"S0": "encoded"}


def test_resume_reexecutes_prepared_transaction_idempotently_when_mcp_is_live(
    monkeypatch, tmp_path
):
    run_dir, checkpoint, settings, engine, unused_previous, candidate = (
        _prepared_checkpoint(tmp_path, phase="PREPARED")
    )
    del unused_previous
    settings = replace(
        settings,
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=True,
        freecad_mcp_required=True,
    )
    calls = []

    def fake_transaction(unused_settings, state, **kwargs):
        calls.append((state.state_id, kwargs["attempt_id"]))
        evidence = _validation_evidence(state)
        validator = kwargs.get("evidence_validator")
        if validator is not None:
            validator(evidence)
        return {}, evidence, {}

    monkeypatch.setattr(pipeline, "_validate_and_publish_freecad", fake_transaction)
    context = pipeline._load_resume_context(
        checkpoint,
        settings,
        engine,
        dry_run=False,
        run_dir=run_dir,
        expected_run_id=run_dir.name,
    )

    assert calls == [(candidate.state_id, 1)]
    assert context.state.state_id == candidate.state_id
    assert context.semantic_mcp_passed is True
    assert context.mcp_result_path.endswith("recovery_mcp/step_1_attempt_1.json")
    assert context.freecad_validation_path.endswith(
        "recovery_mcp/step_1_validation_1.json"
    )
    assert context.freecad_document_path.endswith(
        f"pipe_v1_{geometry_payload_digest(candidate)[:12]}.FCStd"
    )
    assert "step_planner" not in context.planner_lineage


def test_committed_resume_reconciliation_restores_measurements(monkeypatch, tmp_path):
    run_dir, checkpoint, settings, engine, unused_previous, candidate = (
        _prepared_checkpoint(tmp_path, phase="PUBLISHED")
    )
    del unused_previous
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["actions"] = [payload["action"]]
    payload["step_verifications"] = [payload["step_verification"]]
    payload["committed_states"] = [
        *payload["committed_states"],
        candidate.model_dump(mode="json"),
    ]
    payload["phase"] = "COMMITTED"
    payload["state"] = candidate.model_dump(mode="json")
    payload["state_digest"] = pipeline._pipe_state_digest(candidate)
    payload["freecad_verified"] = True
    for field in (
        "candidate_state",
        "candidate_state_digest",
        "candidate_digest",
        "candidate_document",
        "evidence",
        "action",
        "draft",
        "step_verification",
    ):
        payload.pop(field, None)
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    settings = replace(
        settings,
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=True,
        freecad_mcp_required=True,
    )

    def fake_transaction(unused_settings, state, **kwargs):
        del unused_settings, kwargs
        evidence = _validation_evidence(state)
        evidence["checks"]["centerlines"]["M1"]["curve_length"] = 10.0
        return {}, evidence, {}

    monkeypatch.setattr(pipeline, "_validate_and_publish_freecad", fake_transaction)
    context = pipeline._load_resume_context(
        checkpoint,
        settings,
        engine,
        dry_run=False,
        run_dir=run_dir,
        expected_run_id=run_dir.name,
    )

    assert context.semantic_mcp_passed is True
    assert context.state.module_measurements["M1"]["centerline_length"] == 10.0
    assert context.step_verifications[-1].mcp_status == "passed"
    assert (
        context.step_verifications[-1].mcp_measurements["M1"]["centerline_length"]
        == 10.0
    )


def test_resume_semantic_rollback_clears_inflight_planner_lineage(
    monkeypatch, tmp_path
):
    run_dir, checkpoint, settings, engine, previous, unused_candidate = (
        _prepared_checkpoint(tmp_path, phase="PREPARED")
    )
    del unused_candidate
    settings = replace(
        settings,
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=True,
        freecad_mcp_required=False,
    )

    def reject_geometry(*args, **kwargs):
        del args, kwargs
        raise pipeline._FreeCADSemanticError("B-Rep validation failed")

    monkeypatch.setattr(pipeline, "_validate_and_publish_freecad", reject_geometry)
    context = pipeline._load_resume_context(
        checkpoint,
        settings,
        engine,
        dry_run=False,
        run_dir=run_dir,
        expected_run_id=run_dir.name,
    )

    assert context.state.state_id == previous.state_id
    assert context.attempts[-1].status == "rejected"
    assert context.semantic_mcp_passed is False
    assert context.mcp_result_path is None
    assert context.freecad_validation_path is None
    assert context.freecad_document_path is None
    assert context.next_attempt_index == 2
    assert "step_planner" not in context.planner_lineage


def test_resume_rejects_tampered_candidate_digest(tmp_path):
    run_dir, checkpoint, settings, engine, unused_previous, unused_candidate = (
        _prepared_checkpoint(tmp_path, phase="PUBLISHED")
    )
    del unused_previous, unused_candidate
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["candidate_digest"] = "tampered"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate digest"):
        pipeline._load_resume_context(
            checkpoint,
            settings,
            engine,
            dry_run=True,
            run_dir=run_dir,
            expected_run_id=run_dir.name,
        )


def test_resume_revalidates_valid_old_generator_digest_with_current_policy(
    monkeypatch, tmp_path
):
    run_dir, checkpoint, settings, engine, unused_previous, candidate = (
        _prepared_checkpoint(tmp_path, phase="PUBLISHED")
    )
    del unused_previous
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["candidate_digest"] = "a" * 64
    payload["candidate_document"] = "old-generator-candidate"
    payload["published_document"] = "old-generator-published"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    settings = replace(
        settings,
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=True,
        freecad_mcp_required=True,
    )
    calls = []

    def current_generator_transaction(unused_settings, state, **kwargs):
        del unused_settings
        calls.append((state.state_id, kwargs["attempt_id"]))
        evidence = _validation_evidence(state, run_id=run_dir.name)
        validator = kwargs.get("evidence_validator")
        if validator is not None:
            validator(evidence)
        return {}, evidence, {}

    monkeypatch.setattr(
        pipeline,
        "_validate_and_publish_freecad",
        current_generator_transaction,
    )
    context = pipeline._load_resume_context(
        checkpoint,
        settings,
        engine,
        dry_run=False,
        run_dir=run_dir,
        expected_run_id=run_dir.name,
    )

    assert calls == [(candidate.state_id, 1)]
    assert context.state.state_id == candidate.state_id
    assert context.semantic_mcp_passed is True
    assert context.attempts[-1].phase == "recovery_commit"


def test_committed_resume_replays_unversioned_freecad_rejection_before_new_llm_call(
    tmp_path,
):
    run_dir, checkpoint, settings, engine, previous, unused_candidate = (
        _prepared_checkpoint(tmp_path, phase="PREPARED")
    )
    del unused_candidate
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload.update(
        {
            "phase": "COMMITTED",
            "state": previous.model_dump(mode="json"),
            "state_digest": pipeline._pipe_state_digest(previous),
            "freecad_verified": False,
            "attempts": [
                ActionAttempt(
                    step_index=1,
                    attempt_index=1,
                    state_id=previous.state_id,
                    phase="freecad_semantic_validation",
                    status="rejected",
                    draft=payload["draft"],
                    issue_codes=["FREECAD_GEOMETRY_VALIDATION_FAILED"],
                    observations=[
                        {
                            "issue_code": "FREECAD_GEOMETRY_VALIDATION_FAILED",
                            "actual": {
                                "evidence": {
                                    "failed_checks": {"non_adjacent_overlaps": []}
                                }
                            },
                        }
                    ],
                ).model_dump(mode="json")
            ],
            "next_attempt_index": 2,
        }
    )
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    context = pipeline._load_resume_context(
        checkpoint,
        settings,
        engine,
        dry_run=True,
        run_dir=run_dir,
        expected_run_id=run_dir.name,
    )

    assert context.pending_draft is not None
    assert context.pending_draft_attempt_index == 2
    assert context.next_attempt_index == 2
    assert any(
        item.get("context_type") == "generator_migration_replay"
        for item in context.pending_repair_observations
    )


def test_resume_completed_committed_run_does_not_replay_actions(tmp_path):
    settings = _settings(tmp_path)
    first = run_pipeline(
        "20mm hollow pipe straight 20mm",
        settings,
        dry_run=True,
        stream=ThinkingStream(enabled=False),
    )
    run_dir = Path(first.artifacts.output_dir)
    before_actions = json.loads(
        Path(first.artifacts.actions_path).read_text(encoding="utf-8")
    )

    resumed = run_pipeline(
        "",
        settings,
        dry_run=True,
        stream=ThinkingStream(enabled=False),
        resume_dir=run_dir,
    )
    after_actions = json.loads(
        Path(resumed.artifacts.actions_path).read_text(encoding="utf-8")
    )

    assert after_actions == before_actions
    assert resumed.run_id == first.run_id
    assert resumed.verification_status == "partial"


def test_new_unverified_state_cannot_inherit_previous_mcp_pass(monkeypatch, tmp_path):
    settings = replace(
        _settings(tmp_path),
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=True,
        freecad_mcp_required=False,
        final_repair_rounds=0,
    )

    class FakeGemini:
        planner_calls = 0

        def __init__(self, unused_settings):
            self.settings = unused_settings

        def stream_structured(self, prompt, schema, *, part):
            if part == "intent":
                return _intent(
                    Goal(goal_id="G1", type="move", direction="+X", length=10.0),
                    Goal(goal_id="G2", type="move", direction="+X", length=10.0),
                )
            self.planner_calls += 1
            target = "START" if self.planner_calls == 1 else "M1.out"
            goal = f"G{self.planner_calls}"
            return _line_action(target, [goal], completed=[goal], length=10.0)

    transaction_calls = 0

    def fake_transaction(unused_settings, state, **kwargs):
        nonlocal transaction_calls
        transaction_calls += 1
        if transaction_calls == 1:
            evidence = _validation_evidence(state)
            validator = kwargs.get("evidence_validator")
            if validator is not None:
                validator(evidence)
            return {}, evidence, {}
        raise FreeCADMCPError("temporary MCP outage")

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGemini)
    monkeypatch.setattr(pipeline, "_validate_and_publish_freecad", fake_transaction)
    report = run_pipeline(
        "two LLM-selected straight runs",
        settings,
        dry_run=False,
        stream=ThinkingStream(enabled=False),
    )

    assert (
        transaction_calls == 2
    )  # step 1, then one outage trips the run-scoped breaker
    assert report.status == "partial"
    assert report.verification_status == "partial"


def test_report_never_advertises_evidence_from_an_older_state(monkeypatch, tmp_path):
    settings = replace(
        _settings(tmp_path),
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=False,
        freecad_mcp_required=False,
        freecad_capture_views=False,
        final_repair_rounds=0,
    )
    intent = _intent(
        Goal(
            goal_id="G1",
            type="diameter_change",
            direction="+X",
            diameter_out=30.0,
            wall_thickness_out=2.5,
            transition_length=10.0,
            offset=(0.0, 0.0, 0.0),
        ),
        Goal(
            goal_id="G2",
            depends_on_goal_ids=["G1"],
            type="move",
            direction="+X",
            length=10.0,
        ),
    )

    class FakeGemini:
        planner_calls = 0

        def __init__(self, unused_settings):
            del unused_settings

        def stream_structured(self, prompt, schema, *, part):
            del schema
            if part == "intent":
                return intent
            type(self).planner_calls += 1
            if type(self).planner_calls == 1:
                return ActionDraft(
                    target_port="START",
                    module="transition",
                    catalog_schema_version=2,
                    affected_goal_ids=["G1"],
                    completed_goal_ids=["G1"],
                    params={
                        "section_source": "inherit_target",
                        "diameter_out": 30.0,
                        "wall_thickness_out": 2.5,
                        "length": 10.0,
                        "offset": (0.0, 0.0, 0.0),
                    },
                )
            return _line_action(
                "M1.out",
                ["G2"],
                completed=["G2"],
                length=10.0,
            )

    transaction_calls = 0

    def fake_transaction(unused_settings, state, **kwargs):
        nonlocal transaction_calls
        del unused_settings
        transaction_calls += 1
        if transaction_calls == 1:
            evidence = _validation_evidence(
                state,
                run_id=kwargs["run_id"],
                attempt_id=kwargs["attempt_id"],
            )
            validator = kwargs.get("evidence_validator")
            if validator is not None:
                validator(evidence)
            return {}, evidence, {}
        raise FreeCADMCPError("final transport outage")

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGemini)
    monkeypatch.setattr(pipeline, "_validate_and_publish_freecad", fake_transaction)
    report = run_pipeline(
        "transition, then a straight run",
        settings,
        dry_run=False,
        stream=ThinkingStream(enabled=False),
    )

    assert transaction_calls == 2  # adaptive S1, then canonical final S2
    assert report.status == "partial"
    assert report.artifacts.mcp_result_path is None
    assert report.artifacts.freecad_validation_path is None
    assert report.artifacts.freecad_document_path is None
    assert report.artifacts.visual_evidence_paths == []
    evidence_statuses = {item.name: item.status for item in report.artifact_statuses}
    assert evidence_statuses["mcp_result"] == "unavailable"
    assert evidence_statuses["freecad_validation"] == "unavailable"
    assert evidence_statuses["freecad_document"] == "unavailable"


def test_production_intent_rejects_component_multiplicity_inside_structured_schema():
    payload = {
        "global_spec": {
            "outer_diameter": 20.0,
            "wall_thickness": 2.0,
            "is_hollow": True,
            "units": "mm",
        },
        "start_position": [0.0, 0.0, 0.0],
        "start_axis": [1.0, 0.0, 0.0],
        "target_behavior": [
            {
                "goal_id": "G1",
                "depends_on_goal_ids": [],
                "allow_parallel": False,
                "type": "move",
                "direction": "+X",
                "length": 10.0,
            }
        ],
        "expected_open_ports": 1,
        "expected_open_ports_source": "explicit",
        "required_components": ["flange"],
        "hard_constraints": [],
        "geometric_constraints": [],
    }

    with pytest.raises(ValidationError, match="multiplicity"):
        ProductionIntent.model_validate(payload)


def test_completed_accessory_is_removed_from_followup_schema_and_catalog():
    engine = StateEngine(_settings())
    intent = _intent(
        Goal(
            goal_id="G1",
            type="connector",
            direction="+X",
            length=20.0,
            component="flange",
        ),
        Goal(goal_id="G2", type="move", direction="+X", length=10.0),
    ).model_copy(update={"required_components": ["flange"]})
    state0 = engine.initial_state(intent)
    flange = ActionDraft(
        target_port="START",
        module="inline_component",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "section_source": "inherit_target",
            "component_type": "flange",
            "length": 20.0,
            "body_outer_diameter": 40.0,
            "body_start_offset": 0.0,
            "body_length": 4.0,
            "flange_bolt_count": 4,
            "flange_bolt_circle_diameter": 30.0,
            "flange_bolt_hole_diameter": 4.0,
            "flange_reference_axis": [0.0, 1.0, 0.0],
            "connector_type_out": "plain",
            "connector_gender_out": "neutral",
            "connector_standard_out": None,
        },
    )
    state1 = engine.apply_action(engine.resolve_action(flange, state0), state0)
    payload = compact_planner_payload(state1)

    assert pipeline._needs_inline_component_planner(state1) is False
    assert "inline_component" not in {
        entry["id"] for entry in payload["module_catalog"]
    }

    class FakeGemini:
        selected_schema = None

        def stream_structured(self, prompt, schema, *, part):
            del prompt, part
            self.selected_schema = schema
            return _line_action("M1.out", ["G2"], completed=["G2"], length=10.0)

    fake = FakeGemini()
    pipeline._plan_action(state1, dry_run=False, gemini=fake)
    assert fake.selected_schema is CorePlannerDecisionWire


def test_missing_usage_metadata_latches_future_paid_calls_closed(tmp_path):
    settings = replace(
        _settings(tmp_path),
        gemini_api_key="test-only",
        gemini_max_calls=10,
        gemini_max_total_tokens=50000,
    )

    class FakeInteractions:
        def create(self, **body):
            del body
            return SimpleNamespace(
                id="interaction-unmetered",
                output_text='{"value":1}',
                usage=None,
            )

    client = GeminiClient.__new__(GeminiClient)
    client._settings = settings
    client._client = SimpleNamespace(interactions=FakeInteractions())
    client._lineages = {}
    client._usage = LLMUsage()

    assert client.stream_structured("first", _TinyResult, part="intent").value == 1
    assert client.usage_snapshot().accounting_complete is False
    assert client.usage_snapshot().unmetered_calls == 1
    with pytest.raises(GeminiBudgetError, match="usage metadata"):
        client.stream_structured("second", _TinyResult, part="intent")


def test_one_action_cannot_double_count_two_length_goals():
    engine = StateEngine(_settings())
    state = engine.initial_state(
        _intent(
            Goal(goal_id="G1", type="move", direction="+X", length=20.0),
            Goal(goal_id="G2", type="move", direction="+X", length=20.0),
        )
    )
    draft = _line_action("START", ["G1", "G2"], completed=["G1", "G2"], length=20.0)

    result = validate_draft(draft, state)
    assert result.valid is False
    assert any("double-counting" in error for error in result.errors)


def test_goal_dependency_must_be_completed_in_an_earlier_transition():
    engine = StateEngine(_settings())
    state = engine.initial_state(
        _intent(
            Goal(goal_id="G1", type="move", direction="+X", length=10.0),
            Goal(
                goal_id="G2",
                depends_on_goal_ids=["G1"],
                type="route",
                terminal_position=(10.0, 0.0, 0.0),
            ),
        )
    )
    draft = _line_action("START", ["G1", "G2"], completed=["G1", "G2"], length=10.0)

    result = validate_draft(draft, state)
    assert result.valid is False
    assert any("incomplete dependencies" in error for error in result.errors)


def test_nonparallel_later_goal_cannot_complete_by_merely_affecting_first_goal():
    engine = StateEngine(_settings())
    intent = _intent(
        Goal(
            goal_id="G1",
            allow_parallel=False,
            type="move",
            direction="+X",
            length=10.0,
        ),
        Goal(
            goal_id="G2",
            allow_parallel=False,
            type="route",
            terminal_position=(10.0, 0.0, 0.0),
        ),
    )
    before = engine.initial_state(intent)
    draft = _line_action("START", ["G1", "G2"], completed=["G2"], length=10.0)

    draft_result = validate_draft(draft, before)
    resolved = engine.resolve_action(draft, before)
    action_result = validate_action(resolved, before)
    after = engine.apply_action(resolved, before)
    step = build_step_verification(before, resolved, after, intent, 1)

    assert draft_result.valid is False
    assert action_result.valid is False
    assert "GOAL_ORDER_BYPASS" in {issue.issue_code for issue in step.issues}
    assert [goal.goal_id for goal in after.remaining_goals] == ["G1"]


def _spline_route_action(
    *,
    goal_id: str = "G1",
    waypoints: list[tuple[float, float, float]] | None = None,
) -> ActionDraft:
    return ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=[goal_id],
        completed_goal_ids=[goal_id],
        params={
            "path_kind": "spline",
            "section_source": "inherit_target",
            "waypoints": waypoints or [(20.0, 0.0, 5.0), (40.0, 0.0, 0.0)],
        },
    )


def test_curved_extent_constraint_fails_closed_without_occ_bounds():
    engine = StateEngine(_settings())
    intent = _intent(
        Goal(
            goal_id="G1",
            type="route",
            path_kind="spline",
            terminal_position=(40.0, 0.0, 0.0),
        )
    ).model_copy(
        update={
            "geometric_constraints": [
                GeometricConstraint(
                    constraint_id="C1",
                    type="max_extent",
                    axis="+Y",
                    value=30.0,
                )
            ]
        }
    )
    before = engine.initial_state(intent)
    resolved = engine.resolve_action(_spline_route_action(), before)
    after = engine.apply_action(resolved, before)
    step = build_step_verification(before, resolved, after, intent, 1)
    critic = build_final_critic_report(intent, after, [step])

    assert "GEOMETRIC_CONSTRAINT_REQUIRES_FREECAD_BOUNDS" in {
        issue.issue_code for issue in critic.issues
    }

    bound_step = step.model_copy(
        update={
            "mcp_assembly_bounds": AssemblyBounds(
                minimum=(0.0, -10.0, -10.0),
                maximum=(40.0, 10.0, 10.0),
            ),
            "mcp_measurements": {"M1": {"minimum_curvature_radius": 20.0}},
        }
    )
    assert build_final_critic_report(intent, after, [bound_step]).passed


def test_generated_spline_uses_official_freecad_edge_curvature_api():
    engine = StateEngine(_settings())
    intent = _intent(
        Goal(
            goal_id="G1",
            type="route",
            terminal_position=(40.0, 0.0, 0.0),
        )
    )
    before = engine.initial_state(intent)
    state = engine.apply_action(
        engine.resolve_action(_spline_route_action(), before), before
    )
    script = build_freecad_script(state)

    assert "wire = make_path_wire(" in script
    assert "edge.curvatureAt(parameter)" in script
    assert "curve.curvatureAt(parameter)" not in script


def test_final_only_mcp_supplies_spline_length_before_final_critic(
    monkeypatch, tmp_path
):
    settings = replace(
        _settings(tmp_path),
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=False,
        freecad_mcp_required=False,
        freecad_capture_views=False,
        visual_validation_mode="off",
    )
    intent = _intent(
        Goal(
            goal_id="G1",
            type="route",
            path_kind="spline",
            terminal_position=(40.0, 0.0, 0.0),
        )
    ).model_copy(
        update={
            "geometric_constraints": [
                GeometricConstraint(
                    constraint_id="C1",
                    type="max_total_centerline_length",
                    value=100.0,
                )
            ]
        }
    )

    class FakeGemini:
        def __init__(self, unused_settings):
            del unused_settings

        def stream_structured(self, prompt, schema, *, part):
            del prompt, schema
            if part == "intent":
                return intent
            return _spline_route_action(waypoints=[(20.0, 0.0, 2.0), (40.0, 0.0, 0.0)])

    calls = []

    def fake_transaction(unused_settings, state, **kwargs):
        del unused_settings
        calls.append(state.state_id)
        evidence = _validation_evidence(state)
        for centerline in evidence["checks"]["centerlines"].values():
            centerline["curve_length"] = 45.0
            centerline["minimum_radius"] = 20.0
        validator = kwargs.get("evidence_validator")
        if validator is not None:
            validator(evidence)
        return {}, evidence, {}

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGemini)
    monkeypatch.setattr(pipeline, "_validate_and_publish_freecad", fake_transaction)
    report = run_pipeline(
        "LLM-authored spline under a total length constraint",
        settings,
        dry_run=False,
        stream=ThinkingStream(enabled=False),
    )

    assert calls == ["S1"]
    assert report.critic_passed is True
    step_payload = json.loads(
        Path(report.artifacts.step_verification_path).read_text(encoding="utf-8")
    )
    assert step_payload[-1]["mcp_measurements"]["M1"]["centerline_length"] == 45.0


def test_visual_target_step_enters_local_agenda_repair(monkeypatch, tmp_path):
    settings = replace(
        _settings(tmp_path),
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=False,
        freecad_mcp_required=False,
        freecad_capture_views=True,
        visual_validation_mode="final_required",
        final_repair_rounds=1,
    )
    intent = _intent(
        Goal(goal_id="G1", type="move", direction="+X", length=10.0),
        Goal(goal_id="G2", type="move", direction="+X", length=10.0),
    )

    class FakeGemini:
        planner_calls = 0

        def __init__(self, unused_settings):
            del unused_settings

        def stream_structured(self, prompt, schema, *, part):
            del prompt, schema
            if part == "intent":
                return intent
            if part == "patch":
                return AgendaRepairDirective(
                    scope="agenda",
                    rollback_step=2,
                    target_issue_ids=["STEP_0002_01_VISUAL_GEOMETRY_REJECTED_1"],
                    target_module_ids=["M2"],
                    repair_hint="replace the visually rejected second module",
                    rationale="Replace only the visually rejected second step.",
                )
            FakeGemini.planner_calls += 1
            target = "START" if FakeGemini.planner_calls == 1 else "M1.out"
            goal_id = "G1" if FakeGemini.planner_calls == 1 else "G2"
            return _line_action(target, [goal_id], completed=[goal_id], length=10.0)

    def fake_transaction(unused_settings, state, **kwargs):
        del unused_settings
        evidence = _validation_evidence(state)
        validator = kwargs.get("evidence_validator")
        if validator is not None:
            validator(evidence)
        return {}, evidence, {}

    async def fake_capture(unused_settings, output_dir, **kwargs):
        del unused_settings, kwargs
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "isometric.png"
        path.write_bytes(b"static-test-image")
        return [str(path)]

    review_calls = []

    def fake_visual(unused_gemini, state, paths, *, intent):
        del unused_gemini, paths, intent
        review_calls.append(state.state_id)
        if len(review_calls) == 1:
            return VisualCriticResult(
                state_id=state.state_id,
                payload_digest=geometry_payload_digest(state),
                evidence_sha256=[],
                passed=False,
                issues=[
                    VisualCriticIssue(
                        issue_code="VISIBLE_KINK",
                        module_ids=["M2"],
                        observation="The second run has a visible local kink.",
                        target_step=2,
                    )
                ],
            )
        return VisualCriticResult(
            state_id=state.state_id,
            payload_digest=geometry_payload_digest(state),
            evidence_sha256=[],
            passed=True,
            issues=[],
        )

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGemini)
    monkeypatch.setattr(pipeline, "_validate_and_publish_freecad", fake_transaction)
    monkeypatch.setattr(pipeline, "capture_freecad_views", fake_capture)
    monkeypatch.setattr(pipeline, "_visual_review", fake_visual)
    report = run_pipeline(
        "two LLM-authored pipe stages",
        settings,
        dry_run=False,
        stream=ThinkingStream(enabled=False),
    )

    assert report.status == "success"
    assert FakeGemini.planner_calls == 3
    assert review_calls == ["S2", "S2"]


def test_visual_review_receives_immutable_shape_contract(monkeypatch, tmp_path):
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                goal_id="rising_coil",
                type="route",
                path_kind="spline",
                required_waypoints=[
                    (30.0, 0.0, 10.0),
                    (0.0, 30.0, 20.0),
                ],
                notes="loose rising coil",
            )
        ],
        design_notes=["turning radius shrinks as the route rises"],
    )
    engine = StateEngine(_settings(tmp_path))
    before = engine.initial_state(intent)
    state = engine.apply_action(
        engine.resolve_action(
            _line_action(
                "START",
                ["rising_coil"],
                completed=["rising_coil"],
                length=10.0,
            ),
            before,
        ),
        before,
    )
    image_path = tmp_path / "isometric.png"
    image_path.write_bytes(b"visual-contract-test")
    captured = {}

    def fake_call(gemini, inputs, schema, **kwargs):
        del gemini, schema, kwargs
        captured["text"] = inputs[0]["text"]
        captured["view_label"] = inputs[1]["text"]
        return VisualCriticResult(
            state_id=state.state_id,
            payload_digest=geometry_payload_digest(state),
            evidence_sha256=[hashlib.sha256(image_path.read_bytes()).hexdigest()],
            passed=True,
            issues=[],
        )

    monkeypatch.setattr(pipeline, "_call_structured", fake_call)
    result = pipeline._visual_review(
        object(),
        state,
        [str(image_path)],
        intent=intent,
    )

    assert result.passed
    assert "rising_coil" in captured["text"]
    assert "loose rising coil" in captured["text"]
    assert "turning radius shrinks" in captured["text"]
    assert "visible fidelity" in captured["text"]
    assert "topology and internal-passage facts as authoritative" in captured["text"]
    assert "free_downstream_only_excludes_anchored_START" in captured["text"]
    assert '"physical_open_terminal_ids":["START","M1.out"]' in captured["text"]
    assert "Consumed or internal_mated_interface never means capped" in captured["text"]
    assert captured["view_label"] == "Evidence view 1: isometric camera."


def test_visual_module_map_distinguishes_physical_ends_from_mated_ports():
    intent = _intent(
        Goal(goal_id="G1", type="move", direction="+X", length=10.0),
        Goal(goal_id="G2", type="move", direction="+X", length=10.0),
    )
    engine = StateEngine(_settings())
    before = engine.initial_state(intent)
    first = engine.resolve_action(
        _line_action("START", ["G1"], completed=["G1"], length=10.0),
        before,
    )
    after_first = engine.apply_action(first, before)
    second = engine.resolve_action(
        _line_action("M1.out", ["G2"], completed=["G2"], length=10.0),
        after_first,
    )
    state = engine.apply_action(second, after_first)

    module_map = compact_visual_module_map(state)
    roles = {
        port["id"]: port["physical_role"]
        for module in module_map
        for port in module["ports"]
    }
    assert roles == {
        "M1.in": "anchored_START_open_terminal",
        "M1.out": "internal_mated_interface",
        "M2.in": "internal_mated_interface",
        "M2.out": "free_downstream_open_terminal",
    }
    assert all(
        "open_now" not in port for module in module_map for port in module["ports"]
    )
    assert realized_terminal_topology(state) == {
        "anchored_START_is_physical_open_inlet": True,
        "free_downstream_open_terminal_ids": ["M2.out"],
        "physical_open_terminal_ids": ["START", "M2.out"],
        "physical_open_terminal_count": 2,
        "sealed_terminal_modules": [],
    }


def test_visual_review_path_is_append_only_across_resumes(tmp_path):
    review_dir = tmp_path / "visual_review"
    review_dir.mkdir()
    (review_dir / "round_1.json").write_text("{}", encoding="utf-8")
    (review_dir / "round_3.json").write_text("{}", encoding="utf-8")
    (review_dir / "notes.json").write_text("{}", encoding="utf-8")

    assert pipeline._next_visual_review_path(tmp_path) == (review_dir / "round_4.json")


def test_view_capture_probe_activates_document_and_clears_selection():
    code = _document_probe_code("ExpectedDocument", "abc123", set_active=True)

    assert "App.setActiveDocument(doc.Name)" in code
    assert "Gui.Selection.clearSelection()" in code
    assert "Gui.activeDocument().activeView().fitAll()" in code


def test_local_repair_rejoins_later_boundary_and_replays_suffix_without_llm(
    monkeypatch, tmp_path
):
    settings = replace(
        _settings(tmp_path),
        max_iter=6,
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=False,
        freecad_mcp_required=False,
        freecad_capture_views=True,
        visual_validation_mode="final_required",
        final_repair_rounds=1,
    )
    intent = _intent(
        Goal(goal_id="G1", type="route"),
        Goal(goal_id="G2", type="route"),
        Goal(goal_id="G3", type="route"),
    )

    def arc_action(target, goal_id, plane_normal, _terminal_axis):
        return ActionDraft(
            target_port=target,
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=[goal_id],
            completed_goal_ids=[goal_id],
            params={
                "path_kind": "circular_arc",
                "section_source": "inherit_target",
                "bend_radius": 30.0,
                "sweep_angle": 90.0,
                "plane_normal": plane_normal,
            },
        )

    class FakeGemini:
        planner_calls = 0
        planner_prompts = []

        def __init__(self, unused_settings):
            del unused_settings

        def stream_structured(self, prompt, schema, *, part):
            del schema
            if part == "intent":
                return intent
            if part == "patch":
                return AgendaRepairDirective(
                    scope="agenda",
                    rollback_step=1,
                    target_issue_ids=["STEP_0001_01_VISUAL_GEOMETRY_REJECTED_1"],
                    target_module_ids=["M1"],
                    repair_hint=(
                        "detour locally, recover the old +X interface early, and "
                        "reuse the unaffected tail"
                    ),
                    rationale="Only the first module is visually defective.",
                )
            FakeGemini.planner_prompts.append(prompt)
            FakeGemini.planner_calls += 1
            call = FakeGemini.planner_calls
            if call <= 3:
                return _line_action(
                    ["START", "M1.out", "M2.out"][call - 1],
                    [f"G{call}"],
                    completed=[f"G{call}"],
                    length=10.0,
                )
            if call == 4:
                return arc_action("START", "G1", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
            if call == 5:
                return arc_action("M1.out", "G2", (0.0, 0.0, -1.0), (1.0, 0.0, 0.0))
            raise AssertionError("G3 should be replayed without another planner call")

    transaction_state_ids = []

    def fake_transaction(unused_settings, state, **kwargs):
        del unused_settings
        transaction_state_ids.append(state.state_id)
        evidence = _validation_evidence(state)
        validator = kwargs.get("evidence_validator")
        if validator is not None:
            validator(evidence)
        return {}, evidence, {}

    async def fake_capture(unused_settings, output_dir, **kwargs):
        del unused_settings, kwargs
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "isometric.png"
        path.write_bytes(b"static-test-image")
        return [str(path)]

    review_calls = []

    def fake_visual(unused_gemini, state, paths, *, intent):
        del unused_gemini, paths, intent
        review_calls.append(state.state_id)
        if len(review_calls) == 1:
            return VisualCriticResult(
                state_id=state.state_id,
                payload_digest=geometry_payload_digest(state),
                evidence_sha256=[],
                passed=False,
                issues=[
                    VisualCriticIssue(
                        issue_code="LOCAL_DEFECT",
                        module_ids=["M1"],
                        observation="The first module needs a local detour.",
                        target_step=1,
                    )
                ],
            )
        return VisualCriticResult(
            state_id=state.state_id,
            payload_digest=geometry_payload_digest(state),
            evidence_sha256=[],
            passed=True,
            issues=[],
        )

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGemini)
    monkeypatch.setattr(pipeline, "_validate_and_publish_freecad", fake_transaction)
    monkeypatch.setattr(pipeline, "capture_freecad_views", fake_capture)
    monkeypatch.setattr(pipeline, "_visual_review", fake_visual)

    report = run_pipeline(
        "repair locally and reuse the unaffected suffix",
        settings,
        dry_run=False,
        stream=ThinkingStream(enabled=False),
    )

    assert report.status == "success"
    assert FakeGemini.planner_calls == 5
    assert "translation_flexible" in FakeGemini.planner_prompts[3]
    assert "reuse_starts_at_original_step" in FakeGemini.planner_prompts[4]
    assert review_calls == ["S3", "S3"]
    assert transaction_state_ids[-1] == "S3"
    assert transaction_state_ids.count("S3") == 2
    run_dir = Path(report.artifacts.output_dir)
    attempts = json.loads(
        (run_dir / "action_attempts.json").read_text(encoding="utf-8")
    )
    actions = json.loads((run_dir / "actions.json").read_text(encoding="utf-8"))
    assert any(item["phase"] == "suffix_replay" for item in attempts)
    assert actions[2]["params"]["start_position"] == pytest.approx([60.0, 60.0, 0.0])
    assert actions[2]["params"]["length"] == 10.0


def test_preserved_suffix_round_trip_keeps_rejoin_interfaces_position_flexible():
    settings = _settings()
    intent = _intent(
        Goal(goal_id="G1", type="route"),
        Goal(goal_id="G2", type="route"),
        Goal(goal_id="G3", type="route"),
    )
    engine = StateEngine(settings)
    states = [engine.initial_state(intent)]
    drafts = []
    actions = []
    for index in range(1, 4):
        draft = _line_action(
            "START" if index == 1 else f"M{index - 1}.out",
            [f"G{index}"],
            completed=[f"G{index}"],
            length=10.0,
        )
        action = engine.resolve_action(draft, states[-1])
        drafts.append(draft)
        actions.append(action)
        states.append(engine.apply_action(action, states[-1]))

    preserved = pipeline._PreservedSuffix(
        repair_start_step=1,
        original_actions=actions,
        original_drafts=drafts,
        original_checkpoints=states,
        repair_hint="recover a reusable interface",
    )
    restored = pipeline._PreservedSuffix.from_payload(preserved.to_payload())

    assert restored is not None
    assert restored.repair_start_step == 1
    context = pipeline._suffix_rejoin_context(restored, states[0])
    assert context is not None
    first_port = context["candidate_interfaces"][0]["open_port_interfaces"][0]
    assert context["candidate_interfaces"][0]["translation_flexible"] is True
    assert "position" not in first_port
    assert "relative_position" in first_port


def test_freecad_artifact_path_is_absolute_versioned_and_digest_bound(tmp_path):
    engine = StateEngine(_settings())
    intent = _intent(Goal(goal_id="G1", type="move", direction="+X", length=10.0))
    before = engine.initial_state(intent)
    state = engine.apply_action(
        engine.resolve_action(
            _line_action("START", ["G1"], completed=["G1"], length=10.0),
            before,
        ),
        before,
    )
    path = pipeline._freecad_document_path(tmp_path / "mcp_result.json", state)

    assert path.is_absolute()
    assert path.name == f"pipe_v1_{geometry_payload_digest(state)[:12]}.FCStd"


def test_required_mcp_preflight_stops_before_gemini(monkeypatch, tmp_path):
    settings = replace(
        _settings(tmp_path),
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=True,
        freecad_mcp_required=True,
    )
    gemini_created = False

    async def unavailable_probe(*args, **kwargs):
        del args, kwargs
        raise FreeCADMCPError("RPC server is unavailable")

    class ForbiddenGemini:
        def __init__(self, unused_settings):
            del unused_settings
            nonlocal gemini_created
            gemini_created = True

    monkeypatch.setattr(pipeline, "probe_freecad_mcp", unavailable_probe)
    monkeypatch.setattr(pipeline, "GeminiClient", ForbiddenGemini)

    with pytest.raises(pipeline.StaticValidationError) as caught:
        run_pipeline(
            "must stop before a paid call",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
        )

    assert caught.value.stage == "mcp_preflight"
    assert gemini_created is False


def test_optional_mcp_preflight_trips_run_scoped_circuit_breaker(monkeypatch, tmp_path):
    settings = replace(
        _settings(tmp_path),
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=True,
        freecad_mcp_required=False,
    )

    async def unavailable_probe(*args, **kwargs):
        del args, kwargs
        raise FreeCADMCPError("RPC server is unavailable")

    class FakeGemini:
        def __init__(self, unused_settings):
            del unused_settings

        def stream_structured(self, prompt, schema, *, part):
            del prompt, schema
            if part == "intent":
                return _intent(
                    Goal(goal_id="G1", type="move", direction="+X", length=10.0)
                )
            return _line_action("START", ["G1"], completed=["G1"], length=10.0)

    def forbidden_transaction(*args, **kwargs):
        del args, kwargs
        raise AssertionError("circuit breaker should suppress MCP transactions")

    monkeypatch.setattr(pipeline, "probe_freecad_mcp", unavailable_probe)
    monkeypatch.setattr(pipeline, "GeminiClient", FakeGemini)
    monkeypatch.setattr(
        pipeline, "_validate_and_publish_freecad", forbidden_transaction
    )
    report = run_pipeline(
        "optional MCP outage",
        settings,
        dry_run=False,
        stream=ThinkingStream(enabled=False),
    )

    assert report.status == "partial"
    assert "RPC server is unavailable" in (report.freecad_mcp_error or "")


def test_route_required_waypoints_are_order_sensitive():
    engine = StateEngine(_settings())
    intent = _intent(
        Goal(
            goal_id="G1",
            type="route",
            required_waypoints=[(40.0, 0.0, 0.0), (20.0, 0.0, 5.0)],
            terminal_position=(40.0, 0.0, 0.0),
        )
    )
    before = engine.initial_state(intent)
    action = engine.resolve_action(_spline_route_action(), before)
    after = engine.apply_action(action, before)
    step = build_step_verification(before, action, after, intent, 1)

    assert "GOAL_ROUTE_WAYPOINT_ORDER_MISMATCH" in {
        issue.issue_code for issue in step.issues
    }


def test_production_end_contract_supports_plug_but_not_open_marker():
    move = ProductionGoal(
        goal_id="G1",
        depends_on_goal_ids=[],
        allow_parallel=False,
        type="move",
        direction="+X",
        length=20.0,
    )
    plug = ProductionGoal(
        goal_id="G2",
        depends_on_goal_ids=["G1"],
        allow_parallel=False,
        type="end",
        end_type="plug",
        termination_thickness=3.0,
    )
    intent = ProductionIntent(
        global_spec={
            "outer_diameter": 20.0,
            "wall_thickness": 2.0,
            "is_hollow": True,
            "units": "mm",
        },
        start_position=(0.0, 0.0, 0.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[move, plug],
        expected_open_ports=0,
        expected_open_ports_source="derived",
        required_components=[],
        hard_constraints=[],
        geometric_constraints=[],
    )

    assert intent.target_behavior[-1].end_type == "plug"
    with pytest.raises(ValidationError):
        ProductionGoal.model_validate(
            {
                **plug.model_dump(mode="json"),
                "end_type": "open",
            }
        )


def test_component_goal_spec_is_checked_against_inline_geometry():
    engine = StateEngine(_settings())
    intent = _intent(
        Goal(
            goal_id="G1",
            type="connector",
            length=20.0,
            component="flange",
            component_spec=ComponentGoalSpec(
                component_type="flange",
                flange_bolt_count=8,
            ),
        )
    ).model_copy(update={"required_components": ["flange"]})
    before = engine.initial_state(intent)
    draft = ActionDraft(
        target_port="START",
        module="inline_component",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "section_source": "inherit_target",
            "component_type": "flange",
            "length": 20.0,
            "body_outer_diameter": 40.0,
            "body_start_offset": 0.0,
            "body_length": 4.0,
            "flange_bolt_count": 4,
            "flange_bolt_circle_diameter": 30.0,
            "flange_bolt_hole_diameter": 4.0,
            "flange_reference_axis": (0.0, 1.0, 0.0),
            "connector_type_out": "plain",
            "connector_gender_out": "neutral",
            "connector_standard_out": None,
        },
    )
    action = engine.resolve_action(draft, before)
    after = engine.apply_action(action, before)
    step = build_step_verification(before, action, after, intent, 1)

    assert "GOAL_COMPONENT_DIMENSION_MISMATCH" in {
        issue.issue_code for issue in step.issues
    }


def test_required_outlets_are_typed_distinct_and_topology_consistent():
    outlets = [
        BranchGoalOutletSpec(
            axis=(0.0, 1.0, 0.0),
            length=30.0,
            outer_diameter=10.0,
            wall_thickness=1.0,
        ),
        BranchGoalOutletSpec(
            axis=(0.0, -1.0, 0.0),
            length=40.0,
            outer_diameter=12.0,
            wall_thickness=1.0,
        ),
    ]
    goal = ProductionGoal(
        goal_id="G1",
        depends_on_goal_ids=[],
        allow_parallel=False,
        type="branch",
        branch_count=2,
        required_outlets=outlets,
        include_primary_outlet=False,
    )
    intent = ProductionIntent(
        global_spec={
            "outer_diameter": 20.0,
            "wall_thickness": 2.0,
            "is_hollow": True,
            "units": "mm",
        },
        start_position=(0.0, 0.0, 0.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[goal],
        expected_open_ports=2,
        expected_open_ports_source="derived",
        required_components=[],
        hard_constraints=[],
        geometric_constraints=[],
    )

    assert len(intent.target_behavior[0].required_outlets) == 2
    with pytest.raises(ValidationError, match="parallel duplicates"):
        ProductionGoal.model_validate(
            {
                **goal.model_dump(mode="json"),
                "required_outlets": [
                    {"axis": [1.0, 0.0, 0.0]},
                    {"axis": [1.0, 0.001, 0.0]},
                ],
            }
        )


def test_production_junction_is_a_binary_split_basis():
    outlet = {
        "role": "branch",
        "axis": [0.0, 1.0, 0.0],
        "length": 30.0,
        "outer_diameter": 20.0,
        "wall_thickness": 2.0,
    }
    with pytest.raises(ValidationError):
        JunctionParamsV2.model_validate(
            {
                "section_source": "inherit_target",
                "outlets": [
                    outlet,
                    {**outlet, "axis": [0.0, -1.0, 0.0]},
                    {**outlet, "axis": [0.0, 0.0, 1.0]},
                ],
                "blend_mode": "hard",
                "max_hub_radius": 12.0,
            }
        )


def test_planner_junction_schema_rejects_two_primary_roles():
    outlet = {
        "role": "primary",
        "axis": [1.0, 0.0, 0.0],
        "length": 30.0,
        "outer_diameter": 20.0,
        "wall_thickness": 2.0,
    }

    with pytest.raises(ValidationError, match="at most one primary role"):
        CorePlannerDecision.model_validate(
            {
                "catalog_schema_version": 2,
                "target_port": "START",
                "choice": {
                    "module": "junction",
                    "params": {
                        "section_source": "inherit_target",
                        "outlets": [
                            outlet,
                            {**outlet, "axis": [0.0, 1.0, 0.0]},
                        ],
                        "blend_mode": "hard",
                        "max_hub_radius": 12.0,
                    },
                },
                "affected_goal_ids": ["G1"],
                "completed_goal_ids": ["G1"],
            }
        )


def test_branch_goal_with_primary_continuation_rejects_two_branch_roles():
    intent = _intent(
        Goal(
            goal_id="G1",
            type="branch",
            branch_count=1,
            required_outlet_directions=["+Z"],
            include_primary_outlet=True,
        ),
        expected_open_ports=2,
    )
    engine = StateEngine(_settings())
    before = engine.initial_state(intent)
    draft = ActionDraft(
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
                    "axis": [0.0, 0.0, 1.0],
                    "length": 30.0,
                    "outer_diameter": 20.0,
                    "wall_thickness": 2.0,
                },
                {
                    "role": "branch",
                    "axis": [0.0, 1.0, 0.0],
                    "length": 30.0,
                    "outer_diameter": 20.0,
                    "wall_thickness": 2.0,
                },
            ],
            "blend_mode": "hard",
            "max_hub_radius": 12.0,
        },
    )

    draft_result = validate_draft(draft, before)
    assert not draft_result.valid
    assert "expected primary=1, branch=1" in " ".join(draft_result.errors)

    resolved = engine.resolve_action(draft, before)
    assert not validate_action(resolved, before).valid

    after = engine.apply_action(resolved, before)
    step = build_step_verification(before, resolved, after, intent, 1)
    mismatch = next(
        issue
        for issue in step.issues
        if issue.issue_code == "JUNCTION_OUTLET_ROLE_MISMATCH"
    )
    assert step.status == "failed"
    assert mismatch.expected == {
        "primary_role_count": 1,
        "branch_role_count": 1,
        "include_primary_outlet": True,
    }
    assert mismatch.actual["outlet_roles"] == ["branch", "branch"]
    critic = build_final_critic_report(intent, after, [step])
    patch = next(
        suggestion
        for suggestion in critic.patch_suggestions
        if suggestion.operation == "repair_junction_outlet_roles"
    )
    assert patch.params["primary_role_count"] == 1
    assert patch.params["branch_role_count"] == 1


def _two_facing_port_state():
    engine = StateEngine(_settings())
    intent = _intent(Goal(goal_id="G1", type="connect"), expected_open_ports=0)
    state = engine.initial_state(intent)
    other = Port(
        id="P2",
        position=(20.0, 0.0, 0.0),
        axis=(-1.0, 0.0, 0.0),
        outer_diameter=20.0,
        wall_thickness=2.0,
    )
    state = state.model_copy(
        update={
            "open_ports": [state.open_ports[0], other],
            "open_port_ids": ["START", "P2"],
            "port_nodes": {"START": state.open_ports[0], "P2": other},
        }
    )
    return engine, intent, state


def test_connect_ports_line_runs_end_to_end_without_curve_fields():
    engine, intent, before = _two_facing_port_state()
    draft = ActionDraft(
        target_port="START",
        module="connect_ports",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "other_port_id": "P2",
            "path_kind": "line",
            "section_source": "inherit_target",
        },
    )
    action = engine.resolve_action(draft, before)
    assert validate_action(action, before).valid
    after = engine.apply_action(action, before)
    step = build_step_verification(before, action, after, intent, 1)

    assert step.status == "passed"
    assert after.open_ports == []


def test_connect_ports_rejects_collinear_arc_and_duplicate_waypoints():
    engine, unused_intent, state = _two_facing_port_state()
    del unused_intent
    duplicate_params = {
        "other_port_id": "P2",
        "path_kind": "spline",
        "section_source": "inherit_target",
        "waypoints": [(5.0, 5.0, 0.0), (5.0, 5.0, 0.0)],
        "interpolation": "bspline",
        "frenet": True,
        "minimum_curvature_radius": 11.0,
    }
    with pytest.raises(ValidationError, match="duplicates"):
        ConnectPortsParamsV2.model_validate(duplicate_params)

    arc = ActionDraft(
        target_port="START",
        module="connect_ports",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "other_port_id": "P2",
            "path_kind": "circular_arc",
            "section_source": "inherit_target",
            "waypoints": [(10.0, 0.0, 0.0)],
            "minimum_curvature_radius": 11.0,
        },
    )
    resolved = engine.resolve_action(arc, state)
    result = validate_action(resolved, state)
    assert not result.valid
    assert any("non-collinear" in error for error in result.errors)


def test_zero_curvature_evidence_becomes_a_finite_proven_lower_bound():
    measurements = pipeline._freecad_measurements(
        {
            "checks": {
                "centerlines": {
                    "M1": {
                        "passed": True,
                        "curve_length": 40.0,
                        "minimum_radius": None,
                        "zero_curvature": True,
                        "required_radius": 12.0,
                    }
                }
            }
        }
    )

    assert measurements["M1"] == {
        "centerline_length": 40.0,
        "minimum_curvature_radius": 12.0,
    }


def test_generator_digest_binds_version_and_modeling_tolerance():
    intent = _intent(Goal(goal_id="G1", type="move", direction="+X", length=10.0))
    first = StateEngine(replace(_settings(), modeling_tolerance=1e-4)).initial_state(
        intent
    )
    second = StateEngine(replace(_settings(), modeling_tolerance=2e-4)).initial_state(
        intent
    )

    assert GENERATOR_VERSION == "cadgen02-freecad-v24"
    assert geometry_payload_digest(first) != geometry_payload_digest(second)


def test_publish_always_rebuilds_same_name_document_from_candidate():
    engine = StateEngine(_settings())
    intent = _intent(Goal(goal_id="G1", type="move", direction="+X", length=10.0))
    before = engine.initial_state(intent)
    state = engine.apply_action(
        engine.resolve_action(
            _line_action("START", ["G1"], completed=["G1"], length=10.0),
            before,
        ),
        before,
    )
    script = build_freecad_publish_script(
        state,
        run_id="test",
        attempt_id=1,
        fcstd_path="/tmp/test.FCStd",
    )

    assert "Never trust digest metadata on a mutable GUI document" in script
    assert 'App.closeDocument(META["published_document"])' in script
    assert 'if "Deviation" in view.PropertiesList:' in script
    assert 'view.Deviation = META["view_deviation_percent"]' in script
    assert 'if "AngularDeflection" in view.PropertiesList:' in script
    assert 'view.AngularDeflection = META["view_angular_deflection_degrees"]' in script
    assert '"view_deviation_percent":0.05' in script
    assert '"view_angular_deflection_degrees":5.0' in script
    assert '"view_specular_color":[0.12,0.12,0.12]' in script
    assert '"view_shininess":0.25' in script
    assert 'if "ShapeAppearance" in view.PropertiesList:' in script
    assert "view.ShapeAppearance = tuple(materials)" in script
    assert "existing_assembly" not in script


def test_anchored_inlet_evidence_is_required_and_semantic():
    engine = StateEngine(_settings())
    intent = _intent(Goal(goal_id="G1", type="move", direction="+X", length=10.0))
    before = engine.initial_state(intent)
    state = engine.apply_action(
        engine.resolve_action(
            _line_action("START", ["G1"], completed=["G1"], length=10.0),
            before,
        ),
        before,
    )
    evidence = _validation_evidence(state)
    missing = json.loads(json.dumps(evidence))
    del missing["checks"]["anchored_inlet_bore_failures"]
    with pytest.raises(FreeCADMCPError, match="incomplete"):
        assess_freecad_validation(
            _text_result("CADGEN_VALIDATION=", missing),
            expected_digest=geometry_payload_digest(state),
            expected_state_id=state.state_id,
            expected_module_ids=["M1"],
            expected_open_port_count=1,
            expected_anchored_inlet_count=1,
        )

    failed = json.loads(json.dumps(evidence))
    failed["checks"]["anchored_inlet_bore_failures"] = [
        {"port_id": "START", "blocked_volume": 1.0}
    ]
    with pytest.raises(FreeCADValidationError, match="anchored_inlet"):
        assess_freecad_validation(
            _text_result("CADGEN_VALIDATION=", failed),
            expected_digest=geometry_payload_digest(state),
            expected_state_id=state.state_id,
            expected_module_ids=["M1"],
            expected_open_port_count=1,
            expected_anchored_inlet_count=1,
        )


def test_segment_distance_and_static_nonadjacent_collision_regressions():
    assert _segment_distance(
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (8.0, 0.0, 0.0),
    ) == pytest.approx(0.0)
    assert _segment_distance(
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (0.0, 3.0, 0.0),
        (10.0, 3.0, 0.0),
    ) == pytest.approx(3.0)
    assert _segment_distance(
        (-0.0005, 0.0, 0.0),
        (0.0005, 0.0, 0.0),
        (0.0, -0.0005, 0.0),
        (0.0, 0.0005, 0.0),
    ) == pytest.approx(0.0, abs=1e-15)

    engine = StateEngine(_settings())
    intent = _intent(
        Goal(goal_id="G1", type="move", direction="+X", length=10.0),
        Goal(goal_id="G2", type="move", direction="+X", length=10.0),
        Goal(goal_id="G3", type="route", direction="-X", length=20.0),
    )
    state0 = engine.initial_state(intent)
    action1 = engine.resolve_action(
        _line_action("START", ["G1"], completed=["G1"], length=10.0), state0
    )
    state1 = engine.apply_action(action1, state0)
    action2 = engine.resolve_action(
        _line_action("M1.out", ["G2"], completed=["G2"], length=10.0), state1
    )
    state2 = engine.apply_action(action2, state1)
    action3 = engine.resolve_action(
        ActionDraft(
            target_port="M2.out",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["G3"],
            completed_goal_ids=["G3"],
            params={
                "path_kind": "line",
                "section_source": "inherit_target",
                "length": 20.0,
                "direction": (-1.0, 0.0, 0.0),
            },
        ),
        state2,
    )
    state3 = engine.apply_action(action3, state2)
    step = build_step_verification(state2, action3, state3, intent, 3)
    assert "STATIC_COLLISION_REQUIRES_FREECAD" in {
        issue.issue_code for issue in step.issues
    }


def test_port_contract_uses_radius_scaled_rim_error_not_only_axis_dot():
    settings = _settings()
    engine = StateEngine(settings)
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=2000.0, wall_thickness=20.0),
        target_behavior=[Goal(goal_id="G1", type="move", direction="+X", length=20.0)],
        expected_open_ports=1,
        expected_open_ports_source="explicit",
    )
    before = engine.initial_state(intent)
    action = engine.resolve_action(
        _line_action("START", ["G1"], completed=["G1"], length=20.0),
        before,
    )
    after = engine.apply_action(action, before)
    angular_error = 0.01
    tilted_inlet_axis = (
        -math.cos(angular_error),
        math.sin(angular_error),
        0.0,
    )
    # cos(0.01) still clears the old fixed 0.9999 threshold, but at a 1000 mm
    # radius it moves the mating rim by roughly 10 mm.
    assert math.cos(angular_error) > 0.9999
    after.placed_modules[0].ports["in"].axis = tilted_inlet_axis
    after.port_nodes["M1.in"].axis = tilted_inlet_axis

    step = build_step_verification(before, action, after, intent, 1)
    codes = {issue.issue_code for issue in step.issues}

    assert "MODULE_INPUT_AXIS_MISMATCH" in codes
    assert "PORT_CONTRACT_MISMATCH" in codes


@pytest.mark.parametrize(
    ("modeling_tolerance", "diameter_delta", "expected_valid"),
    [(2e-4, 1.5e-4, True), (1e-5, 5e-5, False)],
)
def test_registry_section_matching_uses_state_modeling_tolerance(
    modeling_tolerance,
    diameter_delta,
    expected_valid,
):
    engine = StateEngine(replace(_settings(), modeling_tolerance=modeling_tolerance))
    before = engine.initial_state(
        _intent(Goal(goal_id="G1", type="move", direction="+X", length=20.0))
    )
    action = engine.resolve_action(
        ActionDraft(
            target_port="START",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params={
                "path_kind": "line",
                "section_source": "explicit",
                "outer_diameter": 20.0 + diameter_delta,
                "wall_thickness": 2.0,
                "length": 20.0,
                "direction": (1.0, 0.0, 0.0),
            },
        ),
        before,
    )

    assert validate_action(action, before).valid is expected_valid


@pytest.mark.parametrize(
    ("step_mcp_enabled", "expected_stage"),
    [
        (True, "freecad_semantic_validation"),
        (False, "freecad_semantic_validation"),
    ],
)
def test_freecad_measurement_contract_blocks_publish_before_commit(
    monkeypatch, tmp_path, step_mcp_enabled, expected_stage
):
    settings = replace(
        _settings(tmp_path),
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=step_mcp_enabled,
        freecad_mcp_required=False,
        freecad_capture_views=False,
        visual_validation_mode="off",
        step_repair_attempts=0,
        final_repair_rounds=0,
    )
    intent = _intent(
        Goal(
            goal_id="G1",
            type="route",
            path_kind="spline",
            length=50.0,
            terminal_position=(40.0, 0.0, 0.0),
        )
    )

    class FakeGemini:
        def __init__(self, unused_settings):
            del unused_settings

        def stream_structured(self, prompt, schema, *, part):
            del prompt, schema
            return (
                intent
                if part == "intent"
                else _spline_route_action(
                    waypoints=[(20.0, 0.0, 2.0), (40.0, 0.0, 0.0)]
                )
            )

    published = False

    def fake_transaction(unused_settings, state, **kwargs):
        nonlocal published
        del unused_settings
        evidence = _validation_evidence(state)
        for centerline in evidence["checks"]["centerlines"].values():
            centerline["curve_length"] = 45.0
            centerline["minimum_radius"] = 20.0
        validator = kwargs.get("evidence_validator")
        if validator is not None:
            validator(evidence)
        published = True
        return {}, evidence, {}

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGemini)
    monkeypatch.setattr(pipeline, "_validate_and_publish_freecad", fake_transaction)
    with pytest.raises(pipeline.StaticValidationError) as caught:
        run_pipeline(
            "spline with an exact 50 mm centerline",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
        )

    assert caught.value.stage == expected_stage
    assert published is False


def test_generated_validation_script_handles_hubs_and_accessory_port_envelopes():
    script = build_freecad_script(
        StateEngine(_settings()).initial_state(
            _intent(Goal(goal_id="G1", type="move", direction="+X", length=10.0))
        )
    )

    assert 'module["type"] == "junction" and local_name == "in"' in script
    assert "hub_radius + (outlet_length - hub_radius)" in script
    assert "check_outer = False" in script
    assert '"assembly_errors": assembly_errors' in script


def test_atomic_connector_goal_cannot_be_completed_by_a_later_route():
    engine = StateEngine(_settings())
    intent = _intent(
        Goal(
            goal_id="G1",
            type="connector",
            component="coupling",
            length=20.0,
        )
    ).model_copy(update={"required_components": ["coupling"]})
    state0 = engine.initial_state(intent)
    component = ActionDraft(
        target_port="START",
        module="inline_component",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=[],
        params={
            "section_source": "inherit_target",
            "component_type": "coupling",
            "length": 20.0,
            "body_outer_diameter": 30.0,
            "body_start_offset": 0.0,
            "body_length": 20.0,
            "connector_type_out": "plain",
            "connector_gender_out": "neutral",
            "connector_standard_out": None,
        },
    )
    result = validate_draft(component, state0)
    assert not result.valid
    assert any("atomic goal G1" in error for error in result.errors)

    # Even if invalid state were injected around the normal validator, the next
    # unrelated route cannot claim completion of the component goal.
    state1 = engine.apply_action(engine.resolve_action(component, state0), state0)
    unrelated = _line_action("M1.out", ["G1"], completed=["G1"], length=10.0)
    result = validate_draft(unrelated, state1)
    assert not result.valid
    assert any("requires component_type coupling" in error for error in result.errors)


def test_atomic_turn_goal_cannot_absorb_a_later_straight_route():
    engine = StateEngine(_settings())
    intent = _intent(Goal(goal_id="G1", type="turn", direction="+Y", angle=90.0))
    state0 = engine.initial_state(intent)
    arc = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=[],
        params={
            "path_kind": "circular_arc",
            "section_source": "inherit_target",
            "bend_radius": 30.0,
            "sweep_angle": 90.0,
            "plane_normal": [0.0, 0.0, 1.0],
        },
    )
    assert not validate_draft(arc, state0).valid


def _arc_route_step(required_waypoints, *, sweep_angle=90.0):
    engine = StateEngine(_settings())
    intent = _intent(
        Goal(
            goal_id="G1",
            type="route",
            path_kind="circular_arc",
            required_waypoints=required_waypoints,
        )
    )
    before = engine.initial_state(intent)
    action = engine.resolve_action(
        ActionDraft(
            target_port="START",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params={
                "path_kind": "circular_arc",
                "section_source": "inherit_target",
                "bend_radius": 100.0,
                "sweep_angle": sweep_angle,
                "plane_normal": [0.0, 0.0, 1.0],
            },
        ),
        before,
    )
    after = engine.apply_action(action, before)
    return build_step_verification(before, action, after, intent, 1)


def test_route_arc_derives_exact_terminal_tangent_outside_llm_numeric_vocabulary():
    engine = StateEngine(_settings())
    intent = _intent(
        Goal(
            goal_id="G1",
            type="turn",
            angle=30.0,
            bend_radius=30.0,
            plane_normal=(1.0, 0.0, 0.0),
        )
    ).model_copy(update={"start_axis": (0.0, 0.0, 1.0)})
    before = engine.initial_state(intent)
    draft = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "path_kind": "circular_arc",
            "section_source": "inherit_target",
            "bend_radius": 30.0,
            "sweep_angle": 30.0,
            "plane_normal": (1.0, 0.0, 0.0),
        },
    )

    assert validate_draft(draft, before).valid
    action = engine.resolve_action(draft, before)

    assert action.params["terminal_axis"] == pytest.approx(
        (0.0, -0.5, math.sqrt(3.0) / 2.0), abs=1e-12
    )
    assert validate_action(action, before).valid
    after = engine.apply_action(action, before)
    step = build_step_verification(before, action, after, intent, 1)
    assert not {
        "ROUTE_START_TANGENT_MISMATCH",
        "ROUTE_END_TANGENT_MISMATCH",
    } & {issue.issue_code for issue in step.issues}


def test_registry_rejects_corrupted_arc_with_parallel_plane_normal():
    engine = StateEngine(_settings())
    before = engine.initial_state(_intent(Goal(goal_id="G1", type="route")))
    action = ResolvedAction(
        action_id="A1",
        target_port="START",
        module="route",
        consumed_port_ids=["START"],
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "path_kind": "circular_arc",
            "section_source": "inherit_target",
            "outer_diameter": 20.0,
            "wall_thickness": 2.0,
            "start_position": (0.0, 0.0, 0.0),
            "axis": (1.0, 0.0, 0.0),
            "bend_radius": 30.0,
            "sweep_angle": 90.0,
            "plane_normal": (1.0, 0.0, 0.0),
            "terminal_axis": (0.0, 1.0, 0.0),
        },
    )

    result = validate_action(action, before)

    assert not result.valid
    assert any("canonical frame is invalid" in error for error in result.errors)


def test_arc_frame_rejects_numerically_near_parallel_plane_hint():
    with pytest.raises(ValueError, match="meaningfully non-parallel"):
        canonical_circular_arc_frame(
            (1.0, 0.0, 0.0),
            (1.0, 1e-8, 0.0),
            30.0,
        )


def test_registry_rejects_corrupted_resolver_owned_spline_tangent():
    engine = StateEngine(_settings())
    before = engine.initial_state(_intent(Goal(goal_id="G1", type="route")))
    draft = _spline_route_action()
    action = engine.resolve_action(draft, before)
    params = dict(action.params)
    params["initial_tangent"] = (0.0, 1.0, 0.0)
    corrupted = action.model_copy(update={"params": params})

    result = validate_action(corrupted, before)

    assert not result.valid
    assert any("initial_tangent invariant mismatch" in error for error in result.errors)


def test_old_journal_draft_drops_new_resolver_owned_fields_for_suffix_replay():
    engine = StateEngine(_settings())
    before = engine.initial_state(_intent(Goal(goal_id="G1", type="route")))
    draft = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "path_kind": "circular_arc",
            "section_source": "inherit_target",
            "bend_radius": 30.0,
            "sweep_angle": 90.0,
            "plane_normal": (0.0, 0.0, 1.0),
        },
    )
    action = engine.resolve_action(draft, before)
    old_payload = draft.model_dump(mode="json")
    old_payload["params"]["terminal_axis"] = [0.0, 1.0, 0.0]
    attempt = ActionAttempt(
        step_index=1,
        attempt_index=1,
        state_id="S0",
        phase="commit",
        status="accepted",
        draft=old_payload,
        resolved=action.model_dump(mode="json"),
    )

    recovered = pipeline._accepted_draft_for_action(action, [attempt])

    assert "terminal_axis" not in recovered.params
    assert validate_draft(recovered, before).valid


def test_swept_tube_uses_one_co_terminal_bore_sweep():
    engine = StateEngine(_settings())
    before = engine.initial_state(
        _intent(Goal(goal_id="G1", type="route", path_kind="circular_arc"))
    )
    action = engine.resolve_action(
        ActionDraft(
            target_port="START",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params={
                "path_kind": "circular_arc",
                "section_source": "inherit_target",
                "bend_radius": 40.0,
                "sweep_angle": 90.0,
                "plane_normal": [0.0, 0.0, 1.0],
            },
        ),
        before,
    )
    state = engine.apply_action(action, before)
    script = build_freecad_script(state)

    assert "bore = extend_bore_ends(" not in script
    assert "return outer.cut(bore), outer, bore" in script


def test_circular_arc_waypoint_uses_analytic_curve_not_chord_samples():
    step = _arc_route_step([(38.2683432365, 7.6120467489, 0.0)])
    assert "GOAL_ROUTE_WAYPOINT_MISMATCH" not in {
        issue.issue_code for issue in step.issues
    }


def test_circular_arc_rejects_waypoint_beyond_authored_sweep():
    step = _arc_route_step([(92.3879532511, 138.2683432365, 0.0)])
    assert "GOAL_ROUTE_WAYPOINT_MISMATCH" in {issue.issue_code for issue in step.issues}


def test_negative_arc_waypoints_preserve_traversal_order():
    ordered = [
        (38.2683432365, -7.6120467489, 0.0),
        (92.3879532511, -61.7316567635, 0.0),
    ]
    assert "GOAL_ROUTE_WAYPOINT_ORDER_MISMATCH" not in {
        issue.issue_code for issue in _arc_route_step(ordered, sweep_angle=-90.0).issues
    }
    assert "GOAL_ROUTE_WAYPOINT_ORDER_MISMATCH" in {
        issue.issue_code
        for issue in _arc_route_step(list(reversed(ordered)), sweep_angle=-90.0).issues
    }


@pytest.mark.parametrize(
    "sweep_angle",
    [
        10.0,
        90.0,
        180.0,
        270.0,
        357.0,
        358.0,
        359.0,
        -10.0,
        -90.0,
        -180.0,
        -270.0,
        -357.0,
        -358.0,
        -359.0,
    ],
)
def test_ordered_arc_tangents_remain_correct_for_major_and_near_full_arcs(
    sweep_angle,
):
    start_tangent = (1.0, 0.0, 0.0)
    normal = (0.0, 0.0, 1.0)
    points = _arc_points_from_plane(
        (0.0, 0.0, 0.0),
        start_tangent,
        normal,
        2000.0,
        sweep_angle,
    )

    derived = _arc_endpoint_tangents(points)

    assert derived is not None
    assert dot(derived[0], start_tangent) == pytest.approx(1.0, abs=1e-12)
    expected_end = rotate(
        start_tangent,
        normal,
        math.radians(sweep_angle),
    )
    assert dot(derived[1], expected_end) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("sweep_angle", [358.0, 359.0, -358.0, -359.0])
def test_near_full_route_uses_analytic_endpoint_tangents(sweep_angle):
    engine = StateEngine(_settings())
    intent = _intent(Goal(goal_id="G1", type="route", path_kind="circular_arc"))
    before = engine.initial_state(intent)
    action = engine.resolve_action(
        ActionDraft(
            target_port="START",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params={
                "path_kind": "circular_arc",
                "section_source": "inherit_target",
                "bend_radius": 2000.0,
                "sweep_angle": sweep_angle,
                "plane_normal": (0.0, 0.0, 1.0),
            },
        ),
        before,
    )

    assert validate_action(action, before).valid
    after = engine.apply_action(action, before)
    step = build_step_verification(before, action, after, intent, 1)
    assert not {
        "ROUTE_START_TANGENT_MISMATCH",
        "ROUTE_END_TANGENT_MISMATCH",
    } & {issue.issue_code for issue in step.issues}


@pytest.mark.parametrize(
    ("bend_radius", "expected_valid"),
    [(9.9999, False), (10.0, True), (10.00005, True)],
)
def test_analytic_arc_tube_accepts_horn_boundary_and_rejects_spindle(
    bend_radius,
    expected_valid,
):
    engine = StateEngine(_settings())
    before = engine.initial_state(
        _intent(Goal(goal_id="G1", type="route", path_kind="circular_arc"))
    )
    action = engine.resolve_action(
        ActionDraft(
            target_port="START",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params={
                "path_kind": "circular_arc",
                "section_source": "inherit_target",
                "bend_radius": bend_radius,
                "sweep_angle": 90.0,
                "plane_normal": (0.0, 0.0, 1.0),
            },
        ),
        before,
    )

    assert validate_action(action, before).valid is expected_valid


def test_eccentric_transition_direction_is_its_axis_not_diagonal_displacement():
    engine = StateEngine(_settings())
    intent = _intent(
        Goal(
            goal_id="G1",
            type="diameter_change",
            direction="+X",
            diameter_out=12.0,
            wall_thickness_out=1.5,
            transition_length=60.0,
            offset=(0.0, 6.0, 0.0),
        )
    )
    before = engine.initial_state(intent)
    action = engine.resolve_action(
        ActionDraft(
            target_port="START",
            module="transition",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params={
                "section_source": "inherit_target",
                "diameter_out": 12.0,
                "wall_thickness_out": 1.5,
                "length": 60.0,
                "offset": (0.0, 6.0, 0.0),
            },
        ),
        before,
    )

    assert validate_action(action, before).valid
    after = engine.apply_action(action, before)
    step = build_step_verification(before, action, after, intent, 1)
    assert "GOAL_TRANSITION_DIRECTION_MISMATCH" not in {
        issue.issue_code for issue in step.issues
    }


def test_gemini_schema_avoids_pathological_subnormal_float_bounds():
    class StrictBounds(BaseModel):
        positive_float: float = Field(gt=0.0)
        positive_integer: int = Field(gt=0)

    schema = gemini_json_schema(StrictBounds)
    properties = schema["properties"]
    assert properties["positive_float"]["minimum"] == 0.0
    assert properties["positive_integer"]["minimum"] == 1
    with pytest.raises(ValidationError):
        StrictBounds(positive_float=0.0, positive_integer=1)


def test_structured_freecad_semantic_error_text_is_not_transport_failure():
    engine = StateEngine(_settings())
    before = engine.initial_state(_intent(Goal(goal_id="G1", type="move")))
    state = engine.apply_action(
        engine.resolve_action(_line_action("START", ["G1"], completed=["G1"]), before),
        before,
    )
    evidence = _validation_evidence(state)
    evidence["checks"]["module_errors"] = [
        {"module_id": "M1", "error": "execution error while filleting"}
    ]
    evidence["passed"] = False

    with pytest.raises(FreeCADValidationError) as caught:
        assess_freecad_validation(
            _text_result("CADGEN_VALIDATION=", evidence),
            expected_digest=geometry_payload_digest(state),
            expected_state_id=state.state_id,
            expected_module_ids=["M1"],
        )
    assert caught.value.evidence == evidence


def test_failed_freecad_geometry_without_fingerprints_remains_semantic():
    engine = StateEngine(_settings())
    before = engine.initial_state(_intent(Goal(goal_id="G1", type="move")))
    state = engine.apply_action(
        engine.resolve_action(_line_action("START", ["G1"], completed=["G1"]), before),
        before,
    )
    evidence = _validation_evidence(state, run_id="run", attempt_id=7)
    evidence["candidate_shape_fingerprints"] = {}
    evidence["checks"]["module_errors"] = [
        {"module_id": "M1", "error": "BRep_API: command not done"}
    ]
    evidence["passed"] = False

    with pytest.raises(FreeCADValidationError) as caught:
        assess_freecad_validation(
            _text_result("CADGEN_VALIDATION=", evidence),
            expected_digest=geometry_payload_digest(state),
            expected_state_id=state.state_id,
            expected_module_ids=["M1"],
            expected_generator_version=GENERATOR_VERSION,
            expected_run_id="run",
            expected_state_version=state.state_version,
            expected_attempt_id=7,
            expected_candidate_document=candidate_document_name(
                state, run_id="run", attempt_id=7
            ),
        )

    assert caught.value.evidence == evidence


def test_freecad_evidence_binds_attempt_document_version_and_fingerprints():
    engine = StateEngine(_settings())
    before = engine.initial_state(_intent(Goal(goal_id="G1", type="move")))
    state = engine.apply_action(
        engine.resolve_action(_line_action("START", ["G1"], completed=["G1"]), before),
        before,
    )
    evidence = _validation_evidence(state, run_id="run", attempt_id=7)
    common = {
        "expected_digest": geometry_payload_digest(state),
        "expected_state_id": state.state_id,
        "expected_module_ids": ["M1"],
        "expected_generator_version": GENERATOR_VERSION,
        "expected_run_id": "run",
        "expected_state_version": state.state_version,
        "expected_attempt_id": 7,
        "expected_candidate_document": candidate_document_name(
            state, run_id="run", attempt_id=7
        ),
    }
    assert assess_freecad_validation(
        _text_result("CADGEN_VALIDATION=", evidence), **common
    )["passed"]

    stale = json.loads(json.dumps(evidence))
    stale["attempt_id"] = 6
    with pytest.raises(FreeCADMCPError, match="attempt_id mismatch"):
        assess_freecad_validation(_text_result("CADGEN_VALIDATION=", stale), **common)
    malformed = json.loads(json.dumps(evidence))
    malformed["candidate_shape_fingerprints"]["PipeAssembly"] = "not-a-hash"
    with pytest.raises(FreeCADMCPError, match="fingerprints"):
        assess_freecad_validation(
            _text_result("CADGEN_VALIDATION=", malformed), **common
        )
    spoofed = json.loads(json.dumps(evidence))
    spoofed["candidate_shape_fingerprints"]["unexpected_shape"] = "a" * 64
    with pytest.raises(FreeCADMCPError, match="fingerprints"):
        assess_freecad_validation(_text_result("CADGEN_VALIDATION=", spoofed), **common)


def test_cancelled_freecad_lock_waiter_does_not_leak_mutation_lock(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_calls(unused_settings, calls):
        code = calls[0][1]["code"]
        if code == "first":
            started.set()
            await release.wait()
        return [{"content": [{"type": "text", "text": code}]}]

    monkeypatch.setattr(pipeline, "execute_freecad_code", pipeline.execute_freecad_code)
    import cadgen.freecad_mcp as module

    monkeypatch.setattr(module, "_call_freecad_tools", fake_calls)

    async def scenario():
        settings = _settings()
        first = asyncio.create_task(module._execute_freecad_code(settings, "first"))
        await started.wait()
        waiter = asyncio.create_task(module._execute_freecad_code(settings, "second"))
        await asyncio.sleep(0.03)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        await first
        third = await asyncio.wait_for(
            module._execute_freecad_code(settings, "third"), timeout=0.5
        )
        assert third["content"][0]["text"] == "third"

    asyncio.run(scenario())


def test_v2_variant_models_reject_silently_ignored_authored_fields():
    with pytest.raises(ValidationError, match="inherit_target"):
        RouteParamsV2(
            path_kind="line",
            section_source="inherit_target",
            outer_diameter=20.0,
            wall_thickness=2.0,
            length=10.0,
            direction=(1.0, 0.0, 0.0),
        )
    with pytest.raises(ValidationError, match="curve parameters"):
        RouteParamsV2(
            path_kind="line",
            section_source="inherit_target",
            length=10.0,
            direction=(1.0, 0.0, 0.0),
            bend_radius=30.0,
        )

    hard = JunctionParamsV2(
        section_source="inherit_target",
        outlets=[
            {
                "role": "branch",
                "axis": (0.0, 1.0, 0.0),
                "length": 20.0,
                "outer_diameter": 20.0,
                "wall_thickness": 2.0,
            },
            {
                "role": "branch",
                "axis": (0.0, -1.0, 0.0),
                "length": 20.0,
                "outer_diameter": 20.0,
                "wall_thickness": 2.0,
            },
        ],
        blend_mode="hard",
        max_hub_radius=12.0,
    )
    assert hard.blend_radius is None
    with pytest.raises(ValidationError, match="must omit"):
        JunctionParamsV2.model_validate(
            {**hard.model_dump(mode="json"), "blend_radius": 3.0}
        )


def test_generated_script_uses_root_safe_junction_and_publish_fingerprint():
    script = build_freecad_script(
        StateEngine(_settings()).initial_state(
            _intent(Goal(goal_id="G1", type="move", length=10.0))
        )
    )
    assert "def make_junction(params, root_interface=False)" in script
    assert "inlet_end = start + inlet_axis * engagement" in script
    assert "position - axis * probe_length" in script
    assert "candidate_shape_fingerprints" in script
    assert "flow_bore_network" in script
    assert "seal_with_trim" in script
    assert "outer_network.cut(flow_bore_network).removeSplitter()" in script
    assert 'if MODULES[0]["type"] in ("junction", "junction_pipe")' in script


def test_generated_fillet_junction_uses_exact_radius_sphere_free_material_seams():
    script = build_freecad_script(
        StateEngine(_settings()).initial_state(
            _intent(Goal(goal_id="G1", type="move", length=10.0))
        )
    )

    assert "def compact_junction_seams" in script
    assert "def fillet_compact_junction_material" in script
    assert "current.makeFillet(exact_radius, [selected])" in script
    assert "topology progress" in script
    assert "valid_closed_single_solid" in script
    assert "Part.makeSphere" not in script
    assert "hub_candidates" not in script
    assert 'if params["blend_mode"] == "fillet":' in script
    assert 'float(params["max_hub_radius"])' in script

    # Exact authored fillet radii are never changed and there is no silent
    # sphere/hard-union/radius-shrink fallback. Each successful seam operation
    # re-queries the current material topology.
    fillet_body = script.split("def fillet_compact_junction_material", 1)[1].split(
        "def make_junction", 1
    )[0]
    assert "makeFillet(exact_radius, [selected])" in fillet_body
    assert "compact_junction_seams" in fillet_body
    assert "for factor" not in fillet_body
    assert "exact_radius *" not in fillet_body


def test_generated_junction_overlap_uses_local_non_llm_interface_band():
    script = build_freecad_script(
        StateEngine(_settings()).initial_state(
            _intent(Goal(goal_id="G1", type="move", length=10.0))
        )
    )

    overlap_body = script.split("    def adjacent_junction_interface_overlap", 1)[
        1
    ].split("\n    for index, left_id", 1)[0]
    assert 'params["max_hub_radius"]' not in overlap_body
    assert "max(outlet_outer_radii) + engagement + margin" in overlap_body
    assert "common_shape.cut(interface_band)" in overlap_body
    assert "if forward_dot < -1e-9:" in overlap_body
    assert '"policy": "resolver_local_interface_band"' in overlap_body
    assert '"adjacent_interface_overlaps": adjacent_interface_overlaps' in script


def test_local_junction_interface_band_accepts_only_forward_local_overlap():
    script = build_freecad_script(
        StateEngine(_settings()).initial_state(
            _intent(Goal(goal_id="G1", type="move", length=10.0))
        )
    )
    function_source = textwrap.dedent(
        "def adjacent_junction_interface_overlap"
        + script.split("    def adjacent_junction_interface_overlap", 1)[1].split(
            "\n    for index, left_id", 1
        )[0]
    )

    class FakeVector:
        def __init__(self, values):
            self.values = tuple(float(value) for value in values)

        @property
        def Length(self):
            return math.sqrt(sum(value * value for value in self.values))

        def dot(self, other):
            return sum(left * right for left, right in zip(self.values, other.values))

        def __mul__(self, scalar):
            return FakeVector(value * float(scalar) for value in self.values)

        def __sub__(self, other):
            return FakeVector(
                left - right for left, right in zip(self.values, other.values)
            )

    def fake_vector(values):
        return values if isinstance(values, FakeVector) else FakeVector(values)

    def fake_normalized(value):
        candidate = fake_vector(value)
        return FakeVector(
            component / candidate.Length for component in candidate.values
        )

    class FakePart:
        @staticmethod
        def makeCylinder(radius, length, start, axis):
            return {
                "radius": radius,
                "length": length,
                "start": start,
                "axis": axis,
            }

    class FakeCommon:
        Volume = 182.6882574752523

        def __init__(self, outside_volume):
            self.outside_volume = outside_volume
            self.cut_calls = 0

        def cut(self, unused_zone):
            del unused_zone
            self.cut_calls += 1
            return SimpleNamespace(Volume=self.outside_volume)

    namespace = {
        "MODELING_TOLERANCE": 1e-4,
        "Part": FakePart,
        "vector": fake_vector,
        "normalized": fake_normalized,
    }
    exec(function_source, namespace)
    classify = namespace["adjacent_junction_interface_overlap"]
    forward_child = {
        "id": "M2",
        "params": {
            "start_position": [30.0, 0.0, 0.0],
            "axis": [1.0, 0.0, 0.0],
            "outer_diameter": 20.0,
            "max_hub_radius": 1.0,
            "outlets": [
                {"axis": [1.0, 0.0, 0.0], "outer_diameter": 20.0},
                {"axis": [0.0, 1.0, 0.0], "outer_diameter": 20.0},
            ],
        },
    }

    local_common = FakeCommon(0.0)
    evidence = classify(local_common, forward_child, "M1")
    assert evidence is not None
    assert local_common.cut_calls == 1
    assert evidence["policy"] == "resolver_local_interface_band"
    assert evidence["interface_upstream_depth"] == pytest.approx(10.004)

    # max_hub_radius is authored by the LLM and must not change the allowance.
    enlarged_hub = json.loads(json.dumps(forward_child))
    enlarged_hub["params"]["max_hub_radius"] = 1000.0
    enlarged = classify(FakeCommon(0.0), enlarged_hub, "M1")
    assert enlarged is not None
    assert enlarged["interface_upstream_depth"] == pytest.approx(
        evidence["interface_upstream_depth"]
    )

    nonlocal_common = FakeCommon(1.0)
    assert classify(nonlocal_common, forward_child, "M1") is None
    assert nonlocal_common.cut_calls == 1

    backward_child = json.loads(json.dumps(forward_child))
    backward_child["params"]["outlets"][1]["axis"] = [-0.5, 1.0, 0.0]
    backward_common = FakeCommon(0.0)
    assert classify(backward_common, backward_child, "M1") is None
    assert backward_common.cut_calls == 0


@pytest.mark.parametrize(
    "engagement_edge_length",
    [0.0028284271323941136, 0.005656854363770622],
)
def test_compact_junction_seams_ignore_tolerance_scale_engagement_edge(
    engagement_edge_length,
):
    script = build_freecad_script(
        StateEngine(_settings()).initial_state(
            _intent(Goal(goal_id="G1", type="move", length=10.0))
        )
    )
    function_source = (
        "def compact_junction_seams"
        + script.split("def compact_junction_seams", 1)[1].split(
            "\ndef fillet_compact_junction_material", 1
        )[0]
    )

    class ShortEngagementEdge:
        Length = engagement_edge_length

    class FakeShape:
        Edges = [ShortEngagementEdge()]

        @staticmethod
        def ancestorsOfType(unused_edge, unused_face_type):
            del unused_edge, unused_face_type
            return [object(), object()]

    namespace = {
        "MODELING_TOLERANCE": 1e-4,
        "Part": SimpleNamespace(Face=object()),
        "cylinder_face_radius": lambda unused_face: 10.0,
    }
    exec(function_source, namespace)

    seams = namespace["compact_junction_seams"](
        FakeShape(),
        [10.0],
        SimpleNamespace(),
        [],
    )
    assert seams == []
    assert "edge.Length) <= minimum_authored_seam_length" in function_source


def test_resume_preserves_semantic_rejection_for_next_llm_repair(monkeypatch, tmp_path):
    run_dir, checkpoint, settings, engine, previous, candidate = _prepared_checkpoint(
        tmp_path, phase="PREPARED"
    )
    settings = replace(
        settings,
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=True,
        freecad_mcp_required=True,
    )
    evidence = _validation_evidence(candidate, run_id=run_dir.name)
    evidence["checks"]["module_errors"] = [
        {"module_id": "M1", "error": "invalid recovered fillet"}
    ]
    evidence["checks"]["centerlines"]["M1"] = {
        "passed": True,
        "minimum_radius": 22.0,
        "curvature_repair_hint": "spread the nearby direction change",
    }
    evidence["passed"] = False

    def reject_transaction(*args, **kwargs):
        del args, kwargs
        raise pipeline._FreeCADSemanticError("semantic rejection", evidence)

    monkeypatch.setattr(pipeline, "_validate_and_publish_freecad", reject_transaction)
    context = pipeline._load_resume_context(
        checkpoint,
        settings,
        engine,
        dry_run=False,
        run_dir=run_dir,
        expected_run_id=run_dir.name,
    )

    assert context.state.state_id == previous.state_id
    assert (
        context.pending_repair_observations[-1]["issue_code"] == "RECOVERY_ROLLED_BACK"
    )
    assert context.pending_repair_observations[-1]["module_id"] == "M1"
    assert (
        context.pending_repair_observations[-1]["actual"]["evidence"]["failed_checks"][
            "module_errors"
        ][0]["module_id"]
        == "M1"
    )
    assert (
        context.pending_repair_observations[-1]["actual"]["evidence"][
            "centerline_context"
        ]["M1"]["curvature_repair_hint"]
        == "spread the nearby direction change"
    )
