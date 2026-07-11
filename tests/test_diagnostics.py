from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from cadgen.artifact_store import _artifact_paths
from cadgen.config import load_settings
from cadgen.diagnostics import (
    DiagnosticValidationError,
    bind_advisor_response,
    bind_diagnosis,
    build_diagnostic_case,
    diagnostic_case_digest,
    planner_directive_from_diagnosis,
    should_call_advisor,
    validate_diagnosis,
)
from cadgen.gemini_client import GeminiInvalidRequestError
import cadgen.pipeline as pipeline
from cadgen.pipeline import _run_step_geometry_diagnostician, run_pipeline
from cadgen.schemas import (
    ActionAttempt,
    ActionDraft,
    DiagnosticEvidenceUse,
    Fact,
    DiagnosticJournal,
    DiagnosticRecordRef,
    GlobalSpec,
    GeometryAdvisorRecommendationWire,
    GeometryValidationAdvisorResponse,
    Goal,
    IntentResult,
    LLMUsage,
    ParameterCausality,
    ParameterRangeRecommendation,
    RepairStrategy,
    StepRepairDiagnosis,
    StepRepairDiagnosisBody,
)
from cadgen.state import StateEngine
from cadgen.stream import ThinkingStream


def _state_and_draft():
    settings = load_settings(Path("missing.env"))
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[Goal(goal_id="G1", type="move", direction="+X", length=20.0)],
        expected_open_ports=1,
        expected_open_ports_source="explicit",
    )
    state = StateEngine(settings).initial_state(intent)
    draft = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "section_source": "inherit_target",
            "path_kind": "line",
            "length": 20.0,
            "direction": (1.0, 0.0, 0.0),
        },
    )
    return settings, state, draft


def _observation() -> dict:
    return {
        "issue_id": "STEP_0001_01_FREECAD_GEOMETRY_VALIDATION_FAILED",
        "issue_code": "FREECAD_GEOMETRY_VALIDATION_FAILED",
        "check_name": "freecad_semantic_validation",
        "expected": {"common_volume": 0.2},
        "actual": {
            "evidence": {
                "module_ids": ["M1", "M2"],
                "failed_checks": {
                    "non_adjacent_overlaps": [
                        {
                            "module_ids": ["M1", "M2"],
                            "adjacent": True,
                            "common_volume": 182.688,
                            "allowed_volume": 0.282,
                        }
                    ]
                },
            }
        },
        "suggestion": {
            "operation": "revise_freecad_geometry_inputs",
            "recommended_changes": [],
        },
    }


def _case(*, attempt_index: int = 1, recommendations=()):
    _settings, state, draft = _state_and_draft()
    observation = _observation()
    return build_diagnostic_case(
        run_id="run-1",
        state=state,
        step_index=1,
        attempt_index=attempt_index,
        repair_epoch=0,
        draft=draft,
        issues=[observation],
        evidence={"checks": observation["actual"]["evidence"]["failed_checks"]},
        deterministic_recommendations=recommendations,
    )


def _diagnosis(case, *, path: str = "/params/length"):
    return StepRepairDiagnosis(
        state_id=case.binding.state_id,
        contract_digest=case.binding.contract_digest,
        diagnostic_context_digest=diagnostic_case_digest(case),
        failure_signature=case.binding.failure_signature,
        issue_ids=case.issue_ids,
        diagnosis_class="candidate_parameter",
        disposition="retry_planner",
        confidence="medium",
        summary="A planner-authored field must materially affect the failed metric.",
        causal_chain=["The supplied failure evidence binds the rejected candidate."],
        evidence_uses=[
            DiagnosticEvidenceUse(
                evidence_id=case.facts[0].evidence_id,
                supports="The state and contract binding are authoritative.",
            )
        ],
        parameter_causality=[
            ParameterCausality(
                parameter_path=path,
                influence="direct",
                observed_metric_response="unchanged",
                directive="change",
                explanation="The replacement needs a materially different causal field.",
                evidence_ids=[case.facts[0].evidence_id],
            )
        ],
        strategies=[
            RepairStrategy(
                priority=1,
                kind="parameter_change",
                target_fields=[path],
                instruction="Change only the cited planner-authored field.",
                expected_effect="The failed metric must change materially.",
                verification_checks=["non_adjacent_overlaps"],
            )
        ],
        planner_instruction="Create a new candidate and pass every validator again.",
    )


def _diagnosis_body(case, *, path: str = "/params/length"):
    payload = _diagnosis(case, path=path).model_dump(
        mode="python",
        exclude={
            "protocol_version",
            "state_id",
            "contract_digest",
            "diagnostic_context_digest",
            "failure_signature",
            "issue_ids",
        },
    )
    return StepRepairDiagnosisBody.model_validate(payload)


def test_diagnostic_case_is_policy_bound_and_preserves_nested_overlap_ids():
    case = _case()

    assert case.binding.generator_version == "cadgen02-freecad-v24"
    assert len(case.binding.validator_policy_digest) == 64
    assert case.failed_checks[0]["check_name"] == "non_adjacent_overlaps"
    nested = json.dumps(case.failed_checks, ensure_ascii=False)
    assert "M1" in nested and "M2" in nested
    assert "<depth-truncated>" not in nested
    assert any(card.path == "/params/length" for card in case.field_ownership)


def test_advisor_rechecks_new_candidates_in_same_failure_family_until_call_cap():
    case = _case(attempt_index=1)
    assert should_call_advisor(case, DiagnosticJournal()) is True

    record = DiagnosticRecordRef(
        case_id="a" * 64,
        diagnostic_context_digest=diagnostic_case_digest(case),
        failure_signature=case.binding.failure_signature,
        state_id=case.binding.state_id,
        step_index=1,
        attempt_index=1,
        repair_epoch=0,
        status="complete",
        artifact_path="diagnosis.json",
    )
    # Candidate identity/digest deduplication belongs to the pipeline artifact
    # boundary. A later candidate with the same validator-level signature must
    # still be re-diagnosed because its parameters and accumulated trials differ.
    assert should_call_advisor(case, DiagnosticJournal(records=[record])) is True
    assert (
        should_call_advisor(
            case,
            DiagnosticJournal(
                records=[record],
                calls_by_step={"1:0": 1},
            ),
            max_calls_per_step=1,
        )
        is False
    )

    deterministic = _case(
        attempt_index=1,
        recommendations=({"parameter": "length", "operation": "increase"},),
    )
    assert should_call_advisor(deterministic, DiagnosticJournal()) is True
    repeated = deterministic.model_copy(
        update={
            "binding": deterministic.binding.model_copy(update={"attempt_index": 2})
        }
    )
    assert should_call_advisor(repeated, DiagnosticJournal()) is True


def test_quantitative_registry_failure_triggers_without_freecad_and_retries_protocol_failure():
    case = _case(
        attempt_index=1,
        recommendations=({"operation": "revise_resolved_geometry_inputs"},),
    )
    diagnostic = {
        "metric": "minimum_curvature_radius",
        "comparator": ">=",
        "required": 20.0,
        "actual": 14.2421347306,
        "gap": 5.7578652694,
    }
    case = case.model_copy(
        update={
            "failed_checks": [
                {
                    "check_name": "registry_validation",
                    "actual": {"validation_diagnostics": [diagnostic]},
                }
            ]
        }
    )

    assert should_call_advisor(
        case,
        DiagnosticJournal(),
        freecad_enabled=False,
        trigger_attempt=2,
        max_calls_per_step=2,
    )

    retryable_failure = DiagnosticRecordRef(
        case_id="a" * 64,
        diagnostic_context_digest=diagnostic_case_digest(case),
        failure_signature=case.binding.failure_signature,
        state_id=case.binding.state_id,
        step_index=case.binding.step_index,
        attempt_index=case.binding.attempt_index,
        repair_epoch=case.binding.repair_epoch,
        status="failed",
        failure_reason="binding_mismatch",
    )
    assert should_call_advisor(
        case.model_copy(
            update={"binding": case.binding.model_copy(update={"attempt_index": 2})}
        ),
        DiagnosticJournal(
            records=[retryable_failure],
            calls_by_step={"1:0": 1},
        ),
        freecad_enabled=False,
        max_calls_per_step=2,
    )


def test_host_rejects_resolver_owned_or_unknown_change_and_scopes_directive():
    case = _case()
    valid = validate_diagnosis(case, _diagnosis(case))
    directive = planner_directive_from_diagnosis(valid, case)
    assert directive["repair_scope"] == "params"
    assert directive["validation_authority"].startswith("This diagnosis cannot accept")

    with pytest.raises(DiagnosticValidationError, match="cannot change|unknown"):
        validate_diagnosis(case, _diagnosis(case, path="/resolved_action/end_position"))


def test_terminal_advisor_disposition_stops_blind_planner_retry():
    terminal = pipeline._terminal_geometry_diagnosis(
        [
            {
                "context_type": "step_geometry_diagnosis",
                "diagnosis_class": "immutable_contract_conflict",
                "disposition": "stop_contract_infeasible",
                "planner_instruction": (
                    "No planner-owned range has been observed to satisfy the bound."
                ),
            }
        ]
    )

    assert terminal is not None
    assert "additional blind LLM action retries were stopped" in terminal
    assert (
        pipeline._terminal_geometry_diagnosis(
            [
                {
                    "context_type": "step_geometry_diagnosis",
                    "disposition": "retry_planner",
                }
            ]
        )
        is None
    )


def test_host_binds_body_identity_and_only_forwards_evidence_traced_ranges():
    case = _case()
    range_fact = Fact(
        evidence_id="E_RANGE",
        kind="attempt_delta",
        statement="Tested values bound the next planner search.",
        data={"values": [15.0, 20.0, 30.0], "unit": "mm"},
    )
    case = case.model_copy(update={"facts": [*case.facts, range_fact]})
    body = _diagnosis_body(case).model_copy(
        update={
            "parameter_ranges": [
                ParameterRangeRecommendation(
                    path="/params/length",
                    lower=15.0,
                    upper=30.0,
                    preferred=20.0,
                    unit="mm",
                    classification="promising",
                    rationale="Stay inside the evidence-backed tested interval.",
                    evidence_ids=["E_RANGE"],
                )
            ]
        }
    )

    diagnosis = bind_diagnosis(case, body)

    assert diagnosis.state_id == case.binding.state_id
    assert diagnosis.contract_digest == case.binding.contract_digest
    assert diagnosis.issue_ids == case.issue_ids
    directive = planner_directive_from_diagnosis(diagnosis, case)
    assert directive["parameter_ranges"][0]["preferred"] == 20.0
    assert directive["planner_selection_authority"].startswith(
        "Ranges are advisory evidence only"
    )

    inferred = body.model_copy(
        update={
            "parameter_ranges": [
                body.parameter_ranges[0].model_copy(update={"preferred": 25.0})
            ]
        }
    )
    assert bind_diagnosis(case, inferred).parameter_ranges[0].preferred == 25.0

    unbounded = body.model_copy(
        update={
            "parameter_ranges": [
                body.parameter_ranges[0].model_copy(
                    update={"upper": 500.0, "preferred": 500.0}
                )
            ]
        }
    )
    with pytest.raises(DiagnosticValidationError, match="safety envelope"):
        bind_diagnosis(case, unbounded)


def test_provider_wire_is_bound_to_direction_and_range_for_parameter_planner():
    case = _case()
    range_fact = Fact(
        evidence_id="E_RANGE",
        kind="attempt_delta",
        statement="The attempted parameter values bound a safer next search.",
        data={"values": [15.0, 20.0, 30.0], "unit": "mm"},
    )
    case = case.model_copy(update={"facts": [*case.facts, range_fact]})
    response = GeometryValidationAdvisorResponse(
        diagnosis_class="candidate_parameter",
        disposition="retry_planner",
        confidence="high",
        summary="The planner-authored length is the supported causal field.",
        causal_chain=["The rejected trials support changing the owned length field."],
        evidence_ids=["E_RANGE"],
        recommendations=[
            GeometryAdvisorRecommendationWire(
                path="/params/length",
                action="increase",
                bound_mode="closed",
                lower_text="15.0",
                upper_text="30.0",
                preferred_text="25.0",
                unit="mm",
                classification="promising",
                rationale="Move inside the evidence-scaled search interval.",
                evidence_id="E_RANGE",
            )
        ],
        strategy_kind="parameter_change",
        strategy_instruction="Change the owned length field materially.",
        verification_checks=["non_adjacent_overlaps"],
        missing_evidence=[],
        planner_instruction="Author a different candidate and revalidate it.",
    )

    diagnosis = bind_advisor_response(case, response)
    directive = planner_directive_from_diagnosis(diagnosis, case)

    assert diagnosis.direction_guidance[0].direction == "increase"
    assert diagnosis.parameter_ranges[0].preferred == 25.0
    assert directive["direction_guidance"][0]["path"] == "/params/length"
    assert directive["advisor_explanation"].startswith("Author a different")
    assert "Select and author" in directive["planner_instruction"]


def test_fixed_required_waypoints_are_immutable_and_trials_align_insertions():
    settings = load_settings(Path("missing.env"))
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                goal_id="spline_goal",
                type="route",
                path_kind="spline",
                required_waypoints=[
                    (30.0, 0.0, 0.0),
                    (60.0, 20.0, 20.0),
                    (90.0, 0.0, 0.0),
                ],
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
            )
        ],
        expected_open_ports=1,
        expected_open_ports_source="explicit",
    )
    state = StateEngine(settings).initial_state(intent)

    def draft(points):
        return ActionDraft(
            target_port="START",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["spline_goal"],
            completed_goal_ids=["spline_goal"],
            params={
                "section_source": "inherit_target",
                "path_kind": "spline",
                "waypoint_frame": "relative_to_target",
                "waypoints": points,
            },
        )

    required = [
        (30.0, 0.0, 0.0),
        (60.0, 20.0, 20.0),
        (90.0, 0.0, 0.0),
    ]
    with_optional = [
        required[0],
        (45.0, 10.0, 10.0),
        required[1],
        required[2],
    ]

    def observation(actual):
        diagnostic = {
            "code": "SPLINE_CURVATURE_PREFLIGHT",
            "check_name": "spline_curvature_preflight",
            "metric": "minimum_curvature_radius",
            "comparator": ">=",
            "required": 20.0,
            "actual": actual,
            "gap": 20.0 - actual,
            "units": "mm",
            "implicated_parameter_paths": ["/params/waypoints"],
        }
        return {
            "issue_id": "STEP_0001_01_REGISTRY_VALIDATION_FAILED",
            "issue_code": "REGISTRY_VALIDATION_FAILED",
            "check_name": "registry_validation",
            "expected": {"required": 20.0},
            "actual": {"validation_diagnostics": [diagnostic]},
            "suggestion": {"quantitative_constraints": [diagnostic]},
        }

    first = draft(required)
    second = draft(with_optional)
    history = [
        ActionAttempt(
            step_index=1,
            attempt_index=index,
            state_id=state.state_id,
            phase="registry_validation",
            status="rejected",
            draft=item.model_dump(mode="json"),
            issue_codes=["REGISTRY_VALIDATION_FAILED"],
            observations=[observation(actual)],
        )
        for index, (item, actual) in enumerate(
            ((first, 14.2421347306), (second, 8.12187395585)), start=1
        )
    ]
    case = build_diagnostic_case(
        run_id="run-spline",
        state=state,
        step_index=1,
        attempt_index=2,
        repair_epoch=0,
        draft=second,
        issues=history[-1].observations,
        evidence={"observations": history[-1].observations},
        attempt_history=history,
    )

    ownership = {card.path: card for card in case.field_ownership}
    assert ownership["/params/waypoints/0"].owner == "goal_derived_immutable"
    assert ownership["/params/waypoints/1"].owner == "planner_authored"
    assert ownership["/params/waypoints/2"].owner == "goal_derived_immutable"
    assert ownership["/params/waypoints/0/0"].mutable_in_current_repair is False

    changed_paths = {trial["path"] for trial in case.parameter_trials}
    assert not any("/@required/" in path for path in changed_paths)
    optional_trial = next(
        trial for trial in case.parameter_trials if "/@optional/" in trial["path"]
    )
    assert optional_trial["metric_id"] == (
        "/validation_metrics/minimum_curvature_radius/actual"
    )
    assert optional_trial["metric_values"] == pytest.approx(
        [14.2421347306, 8.12187395585]
    )
    assert optional_trial["current_path"].startswith("/params/waypoints/")
    assert optional_trial["values"][0] is None


def test_attempt_sensitivity_marks_repeated_length_random_walk_invariant():
    _settings, state, draft = _state_and_draft()
    history = []
    for index, length in enumerate((15.0, 20.0, 30.0), start=1):
        trial_draft = draft.model_copy(
            update={"params": {**draft.params, "length": length}}
        )
        observation = _observation()
        history.append(
            ActionAttempt(
                step_index=1,
                attempt_index=index,
                state_id=state.state_id,
                phase="freecad_semantic_validation",
                status="rejected",
                draft=trial_draft.model_dump(mode="json"),
                issue_codes=["FREECAD_GEOMETRY_VALIDATION_FAILED"],
                observations=[observation],
            )
        )

    case = build_diagnostic_case(
        run_id="run-1",
        state=state,
        step_index=1,
        attempt_index=3,
        repair_epoch=0,
        draft=history[-1].draft,
        issues=history[-1].observations,
        evidence={"checks": _observation()["actual"]["evidence"]["failed_checks"]},
        attempt_history=history,
    )

    length_trial = next(
        trial for trial in case.parameter_trials if trial["path"] == "/params/length"
    )
    assert length_trial["values"] == [15.0, 20.0, 30.0]
    assert length_trial["metric_span"] == pytest.approx(0.0)
    assert length_trial["finding"] == "invariant_over_tested_range"


def test_pipeline_diagnostician_persists_then_reuses_artifact_without_new_call(
    tmp_path: Path,
):
    settings, state, draft = _state_and_draft()
    settings = replace(
        settings,
        output_dir=tmp_path,
        step_repair_advisor_enabled=True,
        freecad_mcp_enabled=True,
    )
    observation = _observation()
    attempt = ActionAttempt(
        step_index=1,
        attempt_index=1,
        state_id=state.state_id,
        phase="freecad_semantic_validation",
        status="rejected",
        draft=draft.model_dump(mode="json"),
        issue_codes=["FREECAD_GEOMETRY_VALIDATION_FAILED"],
        observations=[observation],
    )

    class FakeGemini:
        supports_interaction_controls = True
        supports_system_instruction = True
        supports_step_repair_advisor = True

        def __init__(self):
            self.calls = 0
            self._usage = LLMUsage()

        def reset_lineage(self, unused_part):
            del unused_part

        def usage_snapshot(self):
            return self._usage

        def stream_structured(self, prompt, unused_schema, *, part, **unused_kwargs):
            assert part == "step_repair_advisor"
            self.calls += 1
            self._usage = self._usage.model_copy(update={"calls": self.calls})
            from cadgen.schemas import StepRepairDiagnosticContext

            case = StepRepairDiagnosticContext.model_validate(json.loads(prompt))
            # The real provider returns only the semantic body. Host binding
            # must inject every digest/ID before the artifact is accepted.
            return _diagnosis_body(case)

    persisted = []

    def persist(observations, journal, operation):
        persisted.append((list(observations), journal, operation))

    gemini = FakeGemini()
    paths = _artifact_paths(tmp_path)
    observations, journal = _run_step_geometry_diagnostician(
        run_id="run-1",
        run_dir=tmp_path,
        paths=paths,
        state=state,
        step_index=1,
        observations=[observation],
        attempts=[attempt],
        settings=settings,
        gemini=gemini,
        journal=DiagnosticJournal(),
        stream=ThinkingStream(False),
        persist=persist,
    )

    assert gemini.calls == 1
    assert [item.status for item in journal.records] == ["complete"]
    assert sum(journal.calls_by_step.values()) == 1
    assert persisted[0][2] == "step_repair_advisor"
    assert persisted[-1][2] is None
    assert any(
        item.get("context_type") == "step_geometry_diagnosis" for item in observations
    )

    reused_observations, reused_journal = _run_step_geometry_diagnostician(
        run_id="run-1",
        run_dir=tmp_path,
        paths=paths,
        state=state,
        step_index=1,
        observations=[observation],
        attempts=[attempt],
        settings=settings,
        gemini=gemini,
        journal=journal,
        stream=ThinkingStream(False),
        persist=persist,
    )
    assert gemini.calls == 1
    assert reused_journal.cache_hit_count == 1
    assert any(
        item.get("context_type") == "step_geometry_diagnosis"
        for item in reused_observations
    )


def test_advisor_protocol_failure_retries_same_episode_without_spending_planner_slot(
    tmp_path: Path,
):
    settings, state, draft = _state_and_draft()
    settings = replace(
        settings,
        output_dir=tmp_path,
        step_repair_advisor_enabled=True,
        step_repair_advisor_required=True,
    )
    observation = _observation()
    attempt = ActionAttempt(
        step_index=1,
        attempt_index=1,
        state_id=state.state_id,
        phase="freecad_semantic_validation",
        status="rejected",
        draft=draft.model_dump(mode="json"),
        issue_codes=["FREECAD_GEOMETRY_VALIDATION_FAILED"],
        observations=[observation],
    )

    class FakeGemini:
        supports_step_repair_advisor = True

        def __init__(self):
            self.calls = 0
            self._usage = LLMUsage()

        def reset_lineage(self, unused_part):
            del unused_part

        def usage_snapshot(self):
            return self._usage

        def stream_structured(self, prompt, unused_schema, *, part, **unused_kwargs):
            assert part == "step_repair_advisor"
            self.calls += 1
            self._usage = self._usage.model_copy(update={"calls": self.calls})
            if self.calls == 1:
                raise GeminiInvalidRequestError(
                    "provider rejected the response grammar",
                    status_code=400,
                    provider_code="invalid_request",
                )
            from cadgen.schemas import StepRepairDiagnosticContext

            case = StepRepairDiagnosticContext.model_validate(json.loads(prompt))
            return _diagnosis_body(case)

    gemini = FakeGemini()
    observations, journal = _run_step_geometry_diagnostician(
        run_id="run-1",
        run_dir=tmp_path,
        paths=_artifact_paths(tmp_path),
        state=state,
        step_index=1,
        observations=[observation],
        attempts=[attempt],
        settings=settings,
        gemini=gemini,
        journal=DiagnosticJournal(),
        stream=ThinkingStream(False),
        persist=lambda *unused_args: None,
    )

    artifact = json.loads(
        next(tmp_path.glob("diagnostics/*_diagnosis.json")).read_text()
    )
    assert gemini.calls == 2
    assert sum(journal.calls_by_step.values()) == 1
    assert artifact["protocol_attempt_count"] == 2
    assert artifact["protocol_errors"][0]["failure_reason"] == "provider_error"
    assert any(
        item.get("context_type") == "step_geometry_diagnosis" for item in observations
    )
    assert not any(
        item.get("context_type") == "geometry_validation_advisor_unavailable"
        for item in observations
    )


def test_required_advisor_failure_degrades_to_evidence_only_replanning(tmp_path: Path):
    settings, state, draft = _state_and_draft()
    settings = replace(
        settings,
        output_dir=tmp_path,
        step_repair_advisor_enabled=True,
        step_repair_advisor_required=True,
    )
    observation = _observation()
    attempt = ActionAttempt(
        step_index=1,
        attempt_index=1,
        state_id=state.state_id,
        phase="freecad_semantic_validation",
        status="rejected",
        draft=draft.model_dump(mode="json"),
        issue_codes=["FREECAD_GEOMETRY_VALIDATION_FAILED"],
        observations=[observation],
    )

    class AlwaysFailingAdvisor:
        supports_step_repair_advisor = True

        def __init__(self):
            self.calls = 0
            self._usage = LLMUsage()

        def reset_lineage(self, unused_part):
            del unused_part

        def usage_snapshot(self):
            return self._usage

        def stream_structured(self, *unused_args, **unused_kwargs):
            self.calls += 1
            self._usage = self._usage.model_copy(update={"calls": self.calls})
            raise GeminiInvalidRequestError(
                "provider rejected the response grammar",
                status_code=400,
                provider_code="invalid_request",
            )

    gemini = AlwaysFailingAdvisor()
    observations, journal = _run_step_geometry_diagnostician(
        run_id="run-1",
        run_dir=tmp_path,
        paths=_artifact_paths(tmp_path),
        state=state,
        step_index=1,
        observations=[observation],
        attempts=[attempt],
        settings=settings,
        gemini=gemini,
        journal=DiagnosticJournal(),
        stream=ThinkingStream(False),
        persist=lambda *unused_args: None,
    )

    failure = json.loads(
        next(tmp_path.glob("diagnostics/*_advisor_failure.json")).read_text()
    )
    assert gemini.calls == 2
    assert sum(journal.calls_by_step.values()) == 1
    assert failure["protocol_attempt_count"] == 2
    assert any(
        item.get("context_type") == "geometry_validation_advisor_unavailable"
        and item.get("terminal") is False
        and item.get("fallback") == "deterministic_evidence_only"
        for item in observations
    )
    assert pipeline._terminal_geometry_diagnosis(observations) is None


def test_run_pipeline_uses_typed_diagnosis_between_rejection_and_second_planner(
    monkeypatch,
    tmp_path: Path,
):
    base, _state, draft = _state_and_draft()
    settings = replace(
        base,
        output_dir=tmp_path,
        freecad_auto_open=False,
        require_freecad_app=False,
        freecad_mcp_enabled=True,
        freecad_step_mcp_enabled=True,
        freecad_mcp_required=False,
        freecad_capture_views=False,
        visual_validation_mode="off",
        step_repair_attempts=1,
        step_repair_advisor_trigger_attempt=1,
        final_repair_rounds=0,
    )

    class FakeGemini:
        supports_interaction_controls = True
        supports_system_instruction = True
        supports_step_repair_advisor = True
        planner_calls = 0
        advisor_calls = 0

        def __init__(self, unused_settings):
            del unused_settings
            self._usage = LLMUsage()

        def has_previous(self, unused_part):
            del unused_part
            return False

        def reset_lineage(self, unused_part):
            del unused_part

        def usage_snapshot(self):
            return self._usage

        def restore_usage(self, usage):
            self._usage = usage

        def stream_structured(self, prompt, unused_schema, *, part, **unused_kwargs):
            self._usage = self._usage.model_copy(
                update={"calls": self._usage.calls + 1}
            )
            if part == "intent":
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
            if part == "step_planner":
                type(self).planner_calls += 1
                return draft
            if part == "step_repair_advisor":
                type(self).advisor_calls += 1
                from cadgen.schemas import StepRepairDiagnosticContext

                case = StepRepairDiagnosticContext.model_validate(json.loads(prompt))
                return _diagnosis(case)
            raise AssertionError(f"unexpected model part: {part}")

    transactions = 0

    def fake_transaction(unused_settings, state, **kwargs):
        nonlocal transactions
        del unused_settings
        transactions += 1
        module_ids = [module.id for module in state.placed_modules]
        if transactions == 1:
            evidence = {
                "checks": {
                    "module_errors": [],
                    "assembly_errors": [],
                    "non_adjacent_overlaps": [
                        {
                            "module_ids": module_ids,
                            "adjacent": False,
                            "common_volume": 10.0,
                            "allowed_volume": 0.01,
                        }
                    ],
                    "connection_failures": [],
                    "terminal_bore_failures": [],
                    "anchored_inlet_bore_failures": [],
                    "termination_seal_failures": [],
                    "wall_section_failures": [],
                    "deterministic_constraint_failures": [],
                }
            }
            pipeline._atomic_write_json(kwargs["validation_path"], evidence)
            raise pipeline._FreeCADSemanticError("measured overlap", evidence)
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
                    module_id: {"passed": True, "curve_length": 20.0}
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
        stream=ThinkingStream(False),
    )

    assert report.status == "success"
    assert FakeGemini.planner_calls == 2
    assert FakeGemini.advisor_calls == 1
    assert report.advisor_call_count == 1
    assert report.advisor_success_count == 1
    assert report.repair_attempt_count == 1
    checkpoint = json.loads(Path(report.artifacts.checkpoint_path).read_text())
    assert checkpoint["diagnostic_journal"]["records"][0]["status"] == "complete"


def test_quantitative_advisor_terminal_decision_stops_before_seven_attempts(
    monkeypatch,
    tmp_path: Path,
):
    settings = replace(
        load_settings(Path("missing.env")),
        output_dir=tmp_path,
        freecad_auto_open=False,
        require_freecad_app=False,
        freecad_mcp_enabled=False,
        freecad_step_mcp_enabled=False,
        freecad_mcp_required=False,
        freecad_capture_views=False,
        visual_validation_mode="off",
        step_repair_attempts=6,
        step_repair_advisor_enabled=True,
        step_repair_advisor_trigger_attempt=2,
        step_repair_advisor_max_calls_per_step=2,
        final_repair_rounds=0,
    )
    failing_draft = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "section_source": "inherit_target",
            "path_kind": "spline",
            "waypoint_frame": "relative_to_target",
            "waypoints": [
                (30.0, 0.0, 0.0),
                (60.0, 20.0, 20.0),
                (90.0, 0.0, 0.0),
            ],
        },
    )

    class FakeGemini:
        supports_interaction_controls = True
        supports_system_instruction = True
        supports_step_repair_advisor = True
        planner_calls = 0
        advisor_calls = 0

        def __init__(self, unused_settings):
            del unused_settings
            self._usage = LLMUsage()

        def has_previous(self, unused_part):
            del unused_part
            return False

        def reset_lineage(self, unused_part):
            del unused_part

        def usage_snapshot(self):
            return self._usage

        def restore_usage(self, usage):
            self._usage = usage

        def stream_structured(self, prompt, unused_schema, *, part, **unused_kwargs):
            self._usage = self._usage.model_copy(
                update={"calls": self._usage.calls + 1}
            )
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
                    expected_open_ports_source="explicit",
                )
            if part == "step_planner":
                type(self).planner_calls += 1
                return failing_draft
            if part == "step_repair_advisor":
                type(self).advisor_calls += 1
                from cadgen.schemas import StepRepairDiagnosticContext

                case = StepRepairDiagnosticContext.model_validate(json.loads(prompt))
                return StepRepairDiagnosisBody(
                    diagnosis_class="immutable_contract_conflict",
                    disposition="stop_contract_infeasible",
                    confidence="high",
                    summary=(
                        "The localized failure has no supported blind retry plan."
                    ),
                    causal_chain=[
                        "The structured validator evidence localizes the failure."
                    ],
                    evidence_uses=[
                        DiagnosticEvidenceUse(
                            evidence_id=case.facts[0].evidence_id,
                            supports="The rejection remains bound to this state.",
                        )
                    ],
                    planner_instruction=(
                        "Stop this step instead of spending the remaining attempts."
                    ),
                )
            raise AssertionError(f"unexpected model part: {part}")

    monkeypatch.setattr(pipeline, "GeminiClient", FakeGemini)

    with pytest.raises(pipeline.StaticValidationError) as captured:
        run_pipeline(
            "Create one line route.",
            settings,
            dry_run=False,
            stream=ThinkingStream(False),
        )

    report_path = Path(captured.value.artifact_path)
    report = json.loads(report_path.read_text())
    assert FakeGemini.planner_calls == 1
    assert FakeGemini.advisor_calls == 1
    assert report["repair_attempt_count"] == 1
    assert report["futile_retry_avoided_count"] == 1
    assert "additional blind LLM action retries were stopped" in report["summary"]


def test_diagnosis_artifact_rolls_forward_after_checkpoint_crash_without_recall(
    tmp_path: Path,
):
    settings, state, draft = _state_and_draft()
    settings = replace(
        settings,
        step_repair_advisor_enabled=True,
        freecad_mcp_enabled=True,
    )
    observation = _observation()
    attempt = ActionAttempt(
        step_index=1,
        attempt_index=1,
        state_id=state.state_id,
        phase="freecad_semantic_validation",
        status="rejected",
        draft=draft.model_dump(mode="json"),
        issue_codes=["FREECAD_GEOMETRY_VALIDATION_FAILED"],
        observations=[observation],
    )

    class FakeGemini:
        supports_interaction_controls = True
        supports_system_instruction = True
        supports_step_repair_advisor = True

        def __init__(self):
            self.calls = 0
            self._usage = LLMUsage()

        def reset_lineage(self, unused_part):
            del unused_part

        def usage_snapshot(self):
            return self._usage

        def restore_usage(self, usage):
            self._usage = usage

        def stream_structured(self, prompt, unused_schema, *, part, **unused_kwargs):
            assert part == "step_repair_advisor"
            self.calls += 1
            self._usage = self._usage.model_copy(update={"calls": self.calls})
            from cadgen.schemas import StepRepairDiagnosticContext

            return _diagnosis(
                StepRepairDiagnosticContext.model_validate(json.loads(prompt))
            )

    durable = []

    def crash_after_artifact(observations, journal, operation):
        durable.append((list(observations), journal, operation))
        if operation is None:
            raise KeyboardInterrupt("crash after diagnosis artifact")

    first = FakeGemini()
    with pytest.raises(KeyboardInterrupt, match="after diagnosis artifact"):
        _run_step_geometry_diagnostician(
            run_id="run-1",
            run_dir=tmp_path,
            paths=_artifact_paths(tmp_path),
            state=state,
            step_index=1,
            observations=[observation],
            attempts=[attempt],
            settings=settings,
            gemini=first,
            journal=DiagnosticJournal(),
            stream=ThinkingStream(False),
            persist=crash_after_artifact,
        )
    pending_journal = durable[0][1]
    assert pending_journal.records[0].status == "pending"
    assert first.calls == 1

    resumed = FakeGemini()
    persisted_after_resume = []
    observations, complete_journal = _run_step_geometry_diagnostician(
        run_id="run-1",
        run_dir=tmp_path,
        paths=_artifact_paths(tmp_path),
        state=state,
        step_index=1,
        observations=[observation],
        attempts=[attempt],
        settings=settings,
        gemini=resumed,
        journal=pending_journal,
        stream=ThinkingStream(False),
        persist=lambda *args: persisted_after_resume.append(args),
    )

    assert resumed.calls == 0
    assert resumed.usage_snapshot().calls == 1
    assert sum(complete_journal.calls_by_step.values()) == 1
    assert complete_journal.cache_hit_count == 1
    assert complete_journal.records[0].status == "complete"
    assert any(
        item.get("context_type") == "step_geometry_diagnosis" for item in observations
    )
