"""pipeline issue, critic evidence와 최종 실행 보고서를 조립한다."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from cadgen.run_artifact_store import _artifact_statuses
from cadgen.typed_data_models import (
    CriticReport,
    DiagnosticJournal,
    GenerationArtifacts,
    LLMUsage,
    RunReport,
    StaticIssue,
    StepVerification,
)
from cadgen.validation_issue_policy import error_count, top_issue_ids, warning_count


def _issue(
    step_index: int,
    code: str,
    message: str,
    *,
    phase: str,
    action_id: str | None = None,
    target_port: str | None = None,
    module_id: str | None = None,
    port_ids: list[str] | None = None,
    expected: dict[str, Any] | None = None,
    actual: dict[str, Any] | None = None,
    suggestion: dict[str, Any] | None = None,
) -> StaticIssue:
    """issue를 계산하거나 반환한다."""

    prefix = f"STEP_{step_index:04d}" if step_index > 0 else "FINAL"
    return StaticIssue(
        issue_id=f"{prefix}_01_{code}",
        severity="error",
        issue_code=code,
        check_name=phase,
        message=message,
        step_index=step_index if step_index > 0 else None,
        action_id=action_id,
        module_id=module_id,
        port_ids=port_ids or [],
        target_port_id=target_port,
        expected=expected or {},
        actual=actual or {},
        suggestion=suggestion or {},
    )

def _critic_with_issue(
    critic: CriticReport | None,
    issue: StaticIssue,
) -> CriticReport:
    """critic_with_issue를 계산하거나 반환한다."""

    if critic is None:
        issues = [issue]
        return CriticReport(
            passed=False,
            verification_status="failed",
            error_count=1,
            actual_open_ports=0,
            issues=issues,
            next_actions=[f"Fix {issue.issue_code}: {issue.message}"],
        )
    issues = [
        issue,
        *[item for item in critic.issues if item.issue_id != issue.issue_id],
    ]
    return critic.model_copy(
        update={
            "passed": False,
            "verification_status": "failed",
            "issues": issues,
            "error_count": error_count(issues),
            "warning_count": warning_count(issues),
            "next_actions": [
                f"Fix {issue.issue_code}: {issue.message}",
                *critic.next_actions,
            ],
        }
    )

def _attach_view_evidence(
    critic: CriticReport,
    paths: list[str],
) -> CriticReport:
    """attach_view_evidence를 계산하거나 반환한다."""

    path_by_view = {Path(path).stem: path for path in paths}
    views = []
    for request in critic.view_requests:
        path = path_by_view.get(request.view_id)
        if path:
            views.append(
                request.model_copy(
                    update={
                        "evidence_status": "available",
                        "evidence_path": path,
                        "unavailable_reason": None,
                    }
                )
            )
        else:
            views.append(
                request.model_copy(
                    update={
                        "evidence_status": "unavailable",
                        "unavailable_reason": "The requested MCP view was not captured.",
                    }
                )
            )
    return critic.model_copy(update={"view_requests": views})

def _make_report(
    run_id: str,
    dry_run: bool,
    freecad_opened: bool,
    mcp_used: bool,
    mcp_error: str | None,
    artifacts: GenerationArtifacts,
    step_verifications: list[StepVerification],
    critic: CriticReport | None,
    *,
    status: str,
    verification_status: str,
    summary: str,
    failed_stage: str | None = None,
    skipped_mcp_reason: str | None = None,
    gemini: Any = None,
    repair_attempt_count: int = 0,
) -> RunReport:
    """새 객체 또는 구조를 생성한다."""

    issues = (
        critic.issues
        if critic
        else [issue for step in step_verifications for issue in step.issues]
    )
    usage = LLMUsage()
    if gemini is not None and hasattr(gemini, "usage_snapshot"):
        usage = gemini.usage_snapshot()
    llm_policy: dict[str, Any] = {}
    if gemini is not None and hasattr(gemini, "policy_snapshot"):
        llm_policy = dict(gemini.policy_snapshot())
    action_budget_policy = (
        getattr(gemini, "_cadgen_action_budget_policy", None)
        if gemini is not None
        else None
    )
    if isinstance(action_budget_policy, dict):
        llm_policy["action_budget"] = copy.deepcopy(action_budget_policy)
    (
        intent_attempt_count,
        intent_repair_count,
        intent_protocol_retry_count,
    ) = _intent_attempt_stats(artifacts)
    (
        intent_advisor_call_count,
        intent_advisor_success_count,
        intent_advisor_failure_count,
        intent_futile_retry_avoided_count,
    ) = _intent_diagnostic_stats(artifacts)
    diagnostic_journal, advisor_artifact_paths = _diagnostic_report_stats(artifacts)
    advisor_call_count = sum(diagnostic_journal.calls_by_step.values())
    advisor_success_count = sum(
        record.status == "complete" for record in diagnostic_journal.records
    )
    advisor_failure_count = sum(
        record.status == "failed" for record in diagnostic_journal.records
    )
    artifacts = artifacts.model_copy(
        update={"advisor_artifact_paths": advisor_artifact_paths}
    )
    realization_status = "not_run"
    deviation_count = 0
    if artifacts.global_preflight_path:
        try:
            preflight_payload = json.loads(
                Path(artifacts.global_preflight_path).read_text(encoding="utf-8")
            )
            if isinstance(preflight_payload, dict):
                realization_status = str(preflight_payload.get("status") or "not_run")
                deviations = preflight_payload.get("deviations")
                deviation_count = len(deviations) if isinstance(deviations, list) else 0
        except (OSError, json.JSONDecodeError):
            realization_status = "not_run"
    policy_deviation_count = sum(
        1
        for issue in issues
        if issue.severity == "warning"
        and (issue.suggestion or {}).get("disposition")
        == "accepted_with_deviation"
    )
    deviation_count += policy_deviation_count
    if status == "success" and (
        realization_status == "adjusted" or policy_deviation_count > 0
    ):
        status = "success_with_deviations"
    return RunReport(
        run_id=run_id,
        status=status,  # type: ignore[arg-type]
        realization_status=realization_status,  # type: ignore[arg-type]
        deviation_count=deviation_count,
        failed_stage=failed_stage,
        dry_run=dry_run,
        freecad_opened=freecad_opened,
        freecad_mcp_used=mcp_used,
        freecad_mcp_error=mcp_error,
        verification_status=verification_status,  # type: ignore[arg-type]
        static_error_count=error_count(issues),
        static_warning_count=warning_count(issues),
        critic_passed=critic.passed if critic else None,
        skipped_mcp_reason=skipped_mcp_reason,
        top_issues=top_issue_ids(issues, limit=1),
        llm_usage=usage,
        llm_policy=llm_policy,
        intent_attempt_count=intent_attempt_count,
        intent_repair_count=intent_repair_count,
        intent_protocol_retry_count=intent_protocol_retry_count,
        intent_advisor_call_count=intent_advisor_call_count,
        intent_advisor_success_count=intent_advisor_success_count,
        intent_advisor_failure_count=intent_advisor_failure_count,
        intent_futile_retry_avoided_count=(intent_futile_retry_avoided_count),
        action_repair_count=repair_attempt_count,
        repair_attempt_count=repair_attempt_count,
        step_repair_advisor_count=(
            advisor_call_count or _repair_advice_count(artifacts)
        ),
        advisor_call_count=advisor_call_count,
        advisor_success_count=advisor_success_count,
        advisor_failure_count=advisor_failure_count,
        advisor_cache_hit_count=diagnostic_journal.cache_hit_count,
        advisor_artifact_paths=advisor_artifact_paths,
        futile_retry_avoided_count=(diagnostic_journal.futile_retry_avoided_count),
        artifacts=artifacts,
        artifact_statuses=_artifact_statuses(
            artifacts,
            failed_stage=failed_stage,
            issues=issues,
        ),
        summary=summary,
    )

def _intent_attempt_stats(artifacts: GenerationArtifacts) -> tuple[int, int, int]:
    """Intent journal에서 실제 호출 수와 의미 교정 횟수를 계산한다."""

    if not artifacts.intent_attempts_path:
        return 0, 0, 0
    try:
        payload = json.loads(
            Path(artifacts.intent_attempts_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return 0, 0, 0
    if not isinstance(payload, list):
        return 0, 0, 0
    attempts = [item for item in payload if isinstance(item, dict)]
    semantic_calls = sum(
        item.get("consumes_semantic_budget") is True for item in attempts
    )
    protocol_retries = sum(
        item.get("phase") in {"structured_output", "provider_schema_negotiation"}
        and item.get("will_retry", item.get("status") == "schema_retry") is True
        for item in attempts
    )
    return len(attempts), max(0, semantic_calls - 1), protocol_retries

def _intent_diagnostic_stats(
    artifacts: GenerationArtifacts,
) -> tuple[int, int, int, int]:
    """intent_diagnostic_stats를 계산하거나 반환한다."""

    if not artifacts.intent_diagnostics_path:
        return 0, 0, 0, 0
    try:
        payload = json.loads(
            Path(artifacts.intent_diagnostics_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return 0, 0, 0, 0
    if not isinstance(payload, list):
        return 0, 0, 0, 0
    records = [item for item in payload if isinstance(item, dict)]
    calls = sum(
        max(0, int(item.get("advisor_protocol_attempts", 0) or 0)) for item in records
    )
    successes = sum(item.get("advisor_status") == "complete" for item in records)
    failures = sum(
        item.get("advisor_status") in {"failed", "degraded_fallback"}
        for item in records
    )
    avoided = sum(
        item.get("terminal_reason")
        in {"stop_futile_retry", "identical_intent_failure_stagnation"}
        for item in records
    )
    return calls, successes, failures, avoided

def _repair_advice_count(artifacts: GenerationArtifacts) -> int:
    """repair_advice_count를 계산하거나 반환한다."""

    if not artifacts.repair_advice_path:
        return 0
    try:
        payload = json.loads(
            Path(artifacts.repair_advice_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return 0
    return len(payload) if isinstance(payload, list) else 0

def _diagnostic_report_stats(
    artifacts: GenerationArtifacts,
) -> tuple[DiagnosticJournal, list[str]]:
    """port 관련 diagnostic_report_stats 처리를 한다."""

    payload: Any = None
    candidates = [
        artifacts.diagnostics_index_path,
        artifacts.checkpoint_path,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            loaded = json.loads(Path(candidate).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if candidate == artifacts.checkpoint_path and isinstance(loaded, dict):
            loaded = loaded.get("diagnostic_journal")
        if isinstance(loaded, dict):
            payload = loaded
            break
    try:
        journal = DiagnosticJournal.model_validate(payload or {})
    except ValueError:
        journal = DiagnosticJournal()
    diagnostic_dir = Path(artifacts.output_dir) / "diagnostics"
    artifact_paths = sorted(
        str(path) for path in diagnostic_dir.glob("*.json") if path.name != "index.json"
    )
    return journal, artifact_paths

def _clear_state_bound_evidence(
    artifacts: GenerationArtifacts,
) -> GenerationArtifacts:
    """clear_state_bound_evidence를 계산하거나 반환한다."""

    return artifacts.model_copy(
        update={
            "mcp_result_path": None,
            "freecad_validation_path": None,
            "freecad_document_path": None,
            "visual_evidence_paths": [],
        }
    )
