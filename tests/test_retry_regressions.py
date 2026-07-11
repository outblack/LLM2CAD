from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import cadgen.pipeline as pipeline
from cadgen.config import load_settings
from cadgen.gemini_client import _decode_decimal_numbers
from cadgen.pipeline import (
    _is_transient_occ_failure,
    _maybe_request_step_repair_advice,
    _plan_action,
    _planner_repair_context,
    _sequential_position_issues,
    run_pipeline,
)
from cadgen.registry import validate_draft
from cadgen.schemas import (
    ActionAttempt,
    ActionDraft,
    GlobalSpec,
    Goal,
    IntentResult,
    StepRepairAdvice,
)
from cadgen.state import StateEngine
from cadgen.stream import ThinkingStream


def _settings(tmp_path: Path | None = None):
    settings = load_settings(Path("missing.env"))
    if tmp_path is not None:
        settings = settings.with_overrides(output_dir=tmp_path, skip_freecad=True)
    return settings


def _line_intent() -> IntentResult:
    return IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                goal_id="G1",
                type="move",
                direction="+X",
                length=20.0,
            )
        ],
        expected_open_ports=1,
        expected_open_ports_source="explicit",
    )


def _line_draft(*, include_direction: bool = True) -> ActionDraft:
    params = {
        "section_source": "inherit_target",
        "path_kind": "line",
        "length": 20.0,
    }
    if include_direction:
        params["direction"] = (1.0, 0.0, 0.0)
    return ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params=params,
    )


def test_encoded_decimal_p14_round_trips_and_scale_explosion_is_rejected():
    decoded = _decode_decimal_numbers({"k": "d", "c": 3_367_006_979_750_809, "p": 14})
    assert decoded == 33.67006979750809

    # The same legal coefficient with a shifted decimal place is about 10^13
    # times larger. It must cross JSON/Pydantic successfully but be rejected at
    # the state-scaled authored-geometry boundary before resolution.
    exploded = _decode_decimal_numbers({"k": "d", "c": 3_367_006_979_750_809, "p": 1})
    state = StateEngine(_settings()).initial_state(_line_intent())
    draft = _line_draft().model_copy(
        update={
            "params": {
                **_line_draft().params,
                "length": exploded,
            }
        }
    )

    result = validate_draft(draft, state)

    assert result.valid is False
    assert exploded / decoded == pytest.approx(10**13)
    assert any("state-scaled safety limit" in error for error in result.errors)
    assert any("decimal point" in error for error in result.errors)


def test_line_length_heading_and_terminal_position_conflict_is_intent_error():
    settings = _settings()
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_position=(0.0, 0.0, 0.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[
            Goal(
                goal_id="line_goal",
                type="route",
                path_kind="line",
                length=10.0,
                direction="+X",
                terminal_position=(5.0, 5.0, 0.0),
            )
        ],
    )

    issues = _sequential_position_issues(intent, settings)

    assert any("line pose is over-constrained" in issue for issue in issues)
    assert any(
        "terminal_position lies off its line heading" in issue for issue in issues
    )
    with pytest.raises(ValueError, match="line pose is over-constrained"):
        pipeline._validate_intent_safety("Create one line route.", intent, settings)


def test_independent_repair_advisor_is_bounded_and_forwarded_once_to_planner():
    state = StateEngine(_settings()).initial_state(_line_intent())
    observation = {
        "issue_code": "FREECAD_GEOMETRY_VALIDATION_FAILED",
        "check_name": "freecad_semantic_validation",
        "message": "FreeCAD rejected the speculative geometry.",
        "expected": {"freecad_checks": "all implicated checks pass"},
        "actual": {"error": "opaque repeated construction failure"},
        "suggestion": {
            "operation": "revise_freecad_geometry_inputs",
            "recommended_changes": [],
        },
    }
    rejected = [
        ActionAttempt(
            step_index=1,
            attempt_index=index,
            state_id=state.state_id,
            phase="freecad_semantic_validation",
            status="rejected",
            draft=_line_draft().model_dump(mode="json"),
            issue_codes=["FREECAD_GEOMETRY_VALIDATION_FAILED"],
            observations=[observation],
        )
        for index in (1, 2)
    ]
    repair_context = _planner_repair_context([observation], rejected, 1)
    assert any(
        item.get("context_type") == "causal_repair_envelope" for item in repair_context
    )

    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True
        supports_system_instruction = True
        supports_repair_advisor = True

        def __init__(self):
            self.calls: list[tuple[str, str, dict]] = []
            self.reset_calls: list[str] = []

        def has_previous(self, unused_part):
            del unused_part
            return False

        def reset_lineage(self, part):
            self.reset_calls.append(part)

        def stream_structured(self, prompt, schema, *, part, **kwargs):
            del schema
            self.calls.append((part, prompt, kwargs))
            if part == "parameter":
                return StepRepairAdvice(
                    diagnosis_class="validator_or_kernel",
                    candidate_fixable=True,
                    diagnosis="The prior evidence has no causal parameter recommendation.",
                    preserve=["target_port", "module", "affected_goal_ids"],
                    change=["only an evidence-backed planner-authored parameter"],
                    avoid=["length-only random walk"],
                    verification_target="The same deterministic check must pass.",
                    planner_instruction="Keep the causal envelope and change one proven input.",
                )
            if part == "step_planner":
                return _line_draft()
            raise AssertionError(f"unexpected structured part: {part}")

    gemini = FakeGemini()
    advisor = _maybe_request_step_repair_advice(
        gemini,
        state,
        repair_context,
        [observation],
        rejected,
        1,
    )

    assert advisor is not None
    assert advisor["context_type"] == "repair_advisor"
    assert [part for part, _prompt, _kwargs in gemini.calls] == ["parameter"]

    observations_with_advisor = [observation, advisor]
    forwarded_context = _planner_repair_context(
        observations_with_advisor,
        rejected,
        1,
    )
    assert (
        _maybe_request_step_repair_advice(
            gemini,
            state,
            forwarded_context,
            observations_with_advisor,
            rejected,
            1,
        )
        is None
    )

    planned = _plan_action(
        state,
        dry_run=False,
        gemini=gemini,
        repair_observations=forwarded_context,
    )

    assert planned.module == "route"
    assert [part for part, _prompt, _kwargs in gemini.calls].count("parameter") == 1
    planner_prompt = next(
        prompt for part, prompt, _kwargs in gemini.calls if part == "step_planner"
    )
    assert '"context_type":"repair_advisor"' in planner_prompt
    assert '"context_type":"causal_repair_envelope"' in planner_prompt


def test_rejection_journal_is_flushed_before_next_call_interrupt(monkeypatch, tmp_path):
    settings = replace(
        _settings(tmp_path),
        step_repair_attempts=2,
        final_repair_rounds=0,
    )

    class InterruptingGemini:
        planner_calls = 0

        def __init__(self, unused_settings):
            del unused_settings

        def stream_structured(self, prompt, schema, *, part):
            del prompt, schema
            if part == "intent":
                return _line_intent()
            type(self).planner_calls += 1
            if self.planner_calls == 1:
                return _line_draft().model_copy(
                    update={
                        "params": {
                            **_line_draft().params,
                            "direction": (0.0, 0.0, 0.0),
                        }
                    }
                )
            raise KeyboardInterrupt("interrupt immediately after the first rejection")

    monkeypatch.setattr(pipeline, "GeminiClient", InterruptingGemini)

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            "Create one line route.",
            settings,
            dry_run=False,
            stream=ThinkingStream(enabled=False),
        )

    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    external_attempts = json.loads(
        (run_dir / "action_attempts.json").read_text(encoding="utf-8")
    )
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))

    assert external_attempts == checkpoint["attempts"]
    assert len(external_attempts) == 1
    assert external_attempts[0]["status"] == "rejected"
    assert external_attempts[0]["issue_codes"] == ["DRAFT_VALIDATION_FAILED"]
    assert checkpoint["next_attempt_index"] == 2


def test_transient_occ_classifier_excludes_measured_overlap():
    transient = {
        "checks": {
            "module_errors": [
                {
                    "module_id": "M2",
                    "error": "15StdFail_NotDone BRep_API: command not done",
                }
            ],
            "assembly_errors": [],
            "non_adjacent_overlaps": [],
            "connection_failures": [],
            "terminal_bore_failures": [],
            "anchored_inlet_bore_failures": [],
            "termination_seal_failures": [],
            "wall_section_failures": [],
            "deterministic_constraint_failures": [],
        }
    }
    measured_overlap = json.loads(json.dumps(transient))
    measured_overlap["checks"]["non_adjacent_overlaps"] = [
        {
            "module_ids": ["M1", "M2"],
            "adjacent": False,
            "common_volume": 10.0,
            "allowed_volume": 0.01,
        }
    ]

    assert _is_transient_occ_failure(transient) is True
    assert _is_transient_occ_failure(measured_overlap) is False


def test_transient_occ_retries_identical_candidate_without_new_planner_call(
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
        freecad_capture_views=False,
        visual_validation_mode="off",
        step_repair_attempts=1,
        final_repair_rounds=0,
    )

    class FakeGemini:
        planner_calls = 0

        def __init__(self, unused_settings):
            del unused_settings

        def stream_structured(self, prompt, schema, *, part):
            del prompt, schema
            if part == "intent":
                return _line_intent()
            type(self).planner_calls += 1
            return _line_draft()

    transaction_states: list[str] = []

    def fake_transaction(unused_settings, state, **kwargs):
        del unused_settings
        transaction_states.append(state.state_id)
        if len(transaction_states) == 1:
            raise pipeline._FreeCADSemanticError(
                "opaque OCC construction failure",
                {
                    "checks": {
                        "module_errors": [
                            {
                                "module_id": "M1",
                                "error": "15StdFail_NotDone BRep_API: command not done",
                            }
                        ],
                        "assembly_errors": [],
                        "non_adjacent_overlaps": [],
                        "connection_failures": [],
                        "terminal_bore_failures": [],
                        "anchored_inlet_bore_failures": [],
                        "termination_seal_failures": [],
                        "wall_section_failures": [],
                        "deterministic_constraint_failures": [],
                    }
                },
            )

        module_ids = [module.id for module in state.placed_modules]
        evidence = {
            "checks": {
                "assembly": {
                    "passed": True,
                    "bounds": {
                        "minimum": [0.0, -10.0, -10.0],
                        "maximum": [20.0, 10.0, 10.0],
                    },
                },
                "centerlines": {
                    module_id: {
                        "passed": True,
                        "curve_length": 20.0,
                    }
                    for module_id in module_ids
                },
            }
        }
        validator = kwargs.get("evidence_validator")
        if validator is not None:
            validator(evidence)
        return {}, evidence, {}

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGemini)
    monkeypatch.setattr(pipeline, "_validate_and_publish_freecad", fake_transaction)

    report = run_pipeline(
        "Create one line route.",
        settings,
        dry_run=False,
        stream=ThinkingStream(enabled=False),
    )
    attempts = json.loads(
        Path(report.artifacts.action_attempts_path).read_text(encoding="utf-8")
    )

    assert report.status == "success"
    assert FakeGemini.planner_calls == 1
    assert len(transaction_states) == 2
    assert transaction_states[0] == transaction_states[1]
    assert [attempt["status"] for attempt in attempts] == ["accepted"]
    assert report.repair_attempt_count == 0
