from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cadgen.pipeline as pipeline

from cadgen.config import load_settings
from cadgen.conflict_kernel import (
    candidate_digest,
    duplicate_candidate_certificate,
    rejected_candidate_match,
)
from cadgen.contract_core import (
    compile_centerline_program,
    preflight_and_realize_intent,
    structural_intent_issues,
    verify_centerline_program,
)
from cadgen.freecad_script import build_freecad_script
from cadgen.pipeline import _bind_contract, run_pipeline
from cadgen.primitive_compiler import compile_next_action
from cadgen.registry import validate_action, validate_draft
from cadgen.schemas import (
    ActionAttempt,
    GlobalSpec,
    Goal,
    IntentResult,
    StaticIssue,
)
from cadgen.state import StateEngine
from cadgen.static_validation import build_step_verification, has_errors
from cadgen.stream import ThinkingStream


def _closed_triangle_intent() -> IntentResult:
    goals: list[Goal] = []

    def append(goal: Goal) -> None:
        index = len(goals) + 1
        previous = goals[-1].goal_id if goals else None
        goals.append(
            goal.model_copy(
                update={
                    "goal_id": f"G{index}",
                    "depends_on_goal_ids": [previous] if previous else [],
                }
            )
        )

    for _ in range(3):
        append(Goal(type="route", path_kind="line", length=40.0))
        append(
            Goal(
                type="turn",
                angle=120.0,
                bend_radius=10.0,
                plane_normal=(0.0, 0.0, 1.0),
            )
        )
    append(Goal(type="connect", connection_target="start_anchor"))
    return IntentResult(
        global_spec=GlobalSpec(
            outer_diameter=20.0,
            wall_thickness=2.0,
            is_hollow=True,
        ),
        target_behavior=goals,
        start_axis=(1.0, 0.0, 0.0),
        expected_open_ports=0,
        expected_open_ports_source="explicit",
    )


def test_global_preflight_repairs_generic_closed_route_before_freecad() -> None:
    authored = _closed_triangle_intent()

    realized, ledger, result = preflight_and_realize_intent(
        "Make one ordinary closed triangular hollow pipe.",
        authored,
        modeling_tolerance=1e-4,
    )

    assert result.status == "adjusted"
    assert result.scale_factor > 1.0
    assert result.deviations
    assert all(item.reason_code for item in result.deviations)
    assert ledger.ledger_digest == result.ledger_digest
    program = compile_centerline_program(realized, modeling_tolerance=1e-4)
    assert program.closure_position_error is not None
    assert program.closure_position_error < 1e-3
    assert not verify_centerline_program(
        program,
        realized,
        modeling_tolerance=1e-4,
    )


def test_global_preflight_is_rigid_transform_invariant() -> None:
    xy = _closed_triangle_intent()
    xz_goals = [
        goal.model_copy(update={"plane_normal": (0.0, 1.0, 0.0)})
        if goal.plane_normal is not None
        else goal
        for goal in xy.target_behavior
    ]
    xz = xy.model_copy(update={"target_behavior": xz_goals})

    _xy_realized, _xy_ledger, xy_result = preflight_and_realize_intent(
        "closed route",
        xy,
        modeling_tolerance=1e-4,
    )
    _xz_realized, _xz_ledger, xz_result = preflight_and_realize_intent(
        "closed route",
        xz,
        modeling_tolerance=1e-4,
    )

    assert xy_result.status == xz_result.status == "adjusted"
    assert abs(xy_result.scale_factor - xz_result.scale_factor) < 1e-9


def test_source_plane_mismatch_returns_intent_authoring_feedback() -> None:
    intent = _closed_triangle_intent().model_copy(
        update={
            "start_axis": (1.0, 0.0, 0.0),
            "target_behavior": [
                goal.model_copy(update={"plane_normal": (0.0, 1.0, 0.0)})
                if goal.plane_normal is not None
                else goal
                for goal in _closed_triangle_intent().target_behavior
            ],
        }
    )

    issues = structural_intent_issues(
        "XY 평 면 위에 폐곡선을 만들어줘",
        intent,
        modeling_tolerance=1e-4,
    )

    assert any("XY plane" in item for item in issues)


def test_host_compiler_completes_terminal_turn_and_connect_in_one_arc() -> None:
    authored = _closed_triangle_intent()
    realized, _ledger, _result = preflight_and_realize_intent(
        "closed route",
        authored,
        modeling_tolerance=1e-4,
    )
    intent = _bind_contract("closed route", realized)
    settings = replace(
        load_settings(),
        freecad_mcp_enabled=False,
        freecad_step_mcp_enabled=False,
        freecad_mcp_required=False,
        freecad_capture_views=False,
        visual_validation_mode="off",
    )
    engine = StateEngine(settings)
    state = engine.initial_state(intent)

    while state.remaining_goals:
        draft = compile_next_action(state)
        assert draft is not None
        assert validate_draft(draft, state).valid
        action = engine.resolve_action(draft, state)
        assert validate_action(action, state).valid
        candidate = engine.apply_action(action, state)
        verification = build_step_verification(
            state,
            action,
            candidate,
            intent,
            candidate.state_version,
        )
        assert not has_errors(verification.issues)
        state = candidate

    assert len(state.open_ports) == 0
    assert state.reserved_start_anchor is None
    assert state.placed_modules[-1].type == "connect_ports"
    assert state.placed_modules[-1].params["path_kind"] == "circular_arc"
    assert set(state.action_history[-1].completed_goal_ids) == {"G6", "G7"}
    script = build_freecad_script(state)
    assert "make_composite_route_tube" in script
    assert "composite_centerline_sweep" in script
    compile(script, "generated_freecad_script.py", "exec")


def test_exact_rejected_candidate_is_a_nogood() -> None:
    intent = _bind_contract("closed route", _closed_triangle_intent())
    settings = replace(
        load_settings(),
        freecad_mcp_enabled=False,
        freecad_step_mcp_enabled=False,
        freecad_mcp_required=False,
        freecad_capture_views=False,
        visual_validation_mode="off",
    )
    state = StateEngine(settings).initial_state(intent)
    draft = compile_next_action(state)
    assert draft is not None
    action = StateEngine(settings).resolve_action(draft, state)
    attempt = ActionAttempt(
        step_index=1,
        attempt_index=1,
        state_id=state.state_id,
        phase="freecad_semantic_validation",
        status="rejected",
        resolved=action.model_dump(mode="json"),
        issue_codes=["FREECAD_GEOMETRY_VALIDATION_FAILED"],
    )

    matched = rejected_candidate_match(action, [attempt], state_id=state.state_id)

    assert matched is attempt
    assert candidate_digest(action) == candidate_digest(attempt.resolved or {})
    certificate = duplicate_candidate_certificate(action, attempt)
    assert certificate.proof_strength == "proved"
    assert "change_primitive" in certificate.allowed_routes


def test_pipeline_default_architecture_needs_no_numeric_step_llm(
    monkeypatch,
    tmp_path: Path,
) -> None:
    intent = _closed_triangle_intent()

    class IntentOnlyGemini:
        step_calls = 0

        def __init__(self, unused_settings):
            del unused_settings

        def stream_structured(self, prompt, schema, *, part, **kwargs):
            del prompt, schema, kwargs
            if part == "intent":
                return intent
            self.step_calls += 1
            raise AssertionError("host primitive compiler should own numeric actions")

        def reset_lineage(self, part):
            del part

    monkeypatch.setattr(pipeline, "GeminiClient", IntentOnlyGemini)
    settings = replace(
        load_settings(Path("missing.env")).with_overrides(
            output_dir=tmp_path,
            skip_freecad=True,
        ),
        primitive_compiler_enabled=True,
        conflict_search_enabled=True,
    )

    report = run_pipeline(
        "Create one ordinary closed triangular hollow pipe.",
        settings,
        dry_run=False,
        stream=ThinkingStream(enabled=False),
    )

    assert IntentOnlyGemini.step_calls == 0
    assert report.realization_status == "adjusted"
    assert report.deviation_count == 6
    assert Path(report.artifacts.constraint_ledger_path or "").is_file()
    assert Path(report.artifacts.global_preflight_path or "").is_file()
    assert Path(report.artifacts.search_events_path or "").is_file()


def test_repairable_validation_exhaustion_backjumps_instead_of_exiting(
    monkeypatch,
    tmp_path: Path,
) -> None:
    goals = [
        Goal(goal_id="G1", type="move", direction="+X", length=10.0),
        Goal(
            goal_id="G2",
            depends_on_goal_ids=["G1"],
            type="move",
            direction="+X",
            length=10.0,
        ),
    ]
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=goals,
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )

    class BackjumpGemini:
        planner_calls = 0

        def __init__(self, unused_settings):
            del unused_settings

        def stream_structured(self, prompt, schema, *, part, **kwargs):
            del prompt, schema, kwargs
            if part == "intent":
                return intent
            return compile_next_action(self.state)  # pragma: no cover

        def reset_lineage(self, part):
            del part

    # The restored first step is planned through the LLM boundary because it
    # receives causal repair context. Return the same semantically selected G1
    # primitive; the important change is the checkpoint/prefix repair epoch.
    def fake_stream(self, prompt, schema, *, part, **kwargs):
        del prompt, schema, kwargs
        if part == "intent":
            return intent
        BackjumpGemini.planner_calls += 1
        if BackjumpGemini.planner_calls == 1:
            return pipeline.ActionDraft(
                target_port="M1.out",
                module="route",
                params={
                    "path_kind": "line",
                    "section_source": "inherit_target",
                    "length": 10.0,
                    "direction": (1.0, 0.0, 0.0),
                },
                catalog_schema_version=2,
                affected_goal_ids=["G2"],
                completed_goal_ids=["G2"],
            )
        return pipeline.ActionDraft(
            target_port="START",
            module="route",
            params={
                "path_kind": "spline",
                "section_source": "inherit_target",
                "waypoint_frame": "relative_to_target",
                "waypoints": [(5.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
            },
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
        )

    BackjumpGemini.stream_structured = fake_stream
    monkeypatch.setattr(pipeline, "GeminiClient", BackjumpGemini)
    original_verifier = pipeline.build_step_verification
    first_goal_verifications = 0

    def injected_verifier(before, action, after, bound_intent, step_index, **kwargs):
        nonlocal first_goal_verifications
        result = original_verifier(
            before,
            action,
            after,
            bound_intent,
            step_index,
            **kwargs,
        )
        if "G1" in action.completed_goal_ids:
            first_goal_verifications += 1
        if "G2" in action.completed_goal_ids and first_goal_verifications < 2:
            issue = StaticIssue(
                issue_id=f"STEP_{step_index:04d}_INJECTED_GLOBAL_CONFLICT",
                severity="error",
                issue_code="STATIC_COLLISION_REQUIRES_BACKJUMP",
                check_name="global_clearance",
                message="Injected non-local conflict owned by the previous prefix.",
                step_index=step_index,
                action_id=action.action_id,
                module_id="M1",
            )
            return result.model_copy(
                update={"status": "failed", "issues": [*result.issues, issue]}
            )
        return result

    monkeypatch.setattr(pipeline, "build_step_verification", injected_verifier)
    settings = replace(
        load_settings(Path("missing.env")).with_overrides(
            output_dir=tmp_path,
            skip_freecad=True,
        ),
        primitive_compiler_enabled=True,
        conflict_search_enabled=True,
        step_repair_attempts=1,
        step_repair_advisor_enabled=False,
        step_repair_advisor_required=False,
        max_causal_backjumps=1,
        final_repair_rounds=0,
    )

    report = run_pipeline(
        "Create two serial straight pipe stages.",
        settings,
        dry_run=False,
        stream=ThinkingStream(enabled=False),
    )

    assert report.status == "partial"
    events = Path(report.artifacts.search_events_path or "").read_text(encoding="utf-8")
    assert "causal_backjump_scheduled" in events
    assert first_goal_verifications == 2


def test_unresolved_repair_is_checkpointed_as_paused_not_failed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[Goal(goal_id="G1", type="move", direction="+X", length=10.0)],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )

    class InvalidPlanner:
        def __init__(self, unused_settings):
            del unused_settings

        def stream_structured(self, prompt, schema, *, part, **kwargs):
            del prompt, schema, kwargs
            if part == "intent":
                return intent
            return pipeline.ActionDraft(
                target_port="MISSING",
                module="route",
                params={
                    "path_kind": "line",
                    "section_source": "inherit_target",
                    "length": 10.0,
                    "direction": (1.0, 0.0, 0.0),
                },
                catalog_schema_version=2,
                affected_goal_ids=["G1"],
                completed_goal_ids=["G1"],
            )

    monkeypatch.setattr(pipeline, "GeminiClient", InvalidPlanner)
    settings = replace(
        load_settings(Path("missing.env")).with_overrides(
            output_dir=tmp_path,
            skip_freecad=True,
        ),
        primitive_compiler_enabled=False,
        conflict_search_enabled=True,
        step_repair_attempts=0,
        max_causal_backjumps=0,
        final_repair_rounds=0,
    )

    try:
        run_pipeline(
            "Create one straight stage.",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
        )
    except pipeline.PipelinePausedError as exc:
        report = Path(exc.artifact_path).read_text(encoding="utf-8")
        assert '"status": "paused"' in report
        assert '"resume_command": "./run.sh --resume ' in report
        assert Path(exc.artifact_path).with_name("checkpoint.json").is_file()
    else:  # pragma: no cover - the invalid planner must never be accepted.
        raise AssertionError("recoverable validation exhaustion was not paused")
