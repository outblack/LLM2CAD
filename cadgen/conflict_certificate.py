"""실패 원인 라우팅, 동일 후보 재제출 차단, 탐색 이벤트 기록을 담당한다.

검증 오류를 ``ConflictCertificate``로 통일하고, 이미 거절된 geometry digest는
nogood로 묶어 FreeCAD/LLM에 반복 제출하지 않는다. 탐색 이벤트는 append-only
JSONL로 남겨 중단 후에도 재현 가능한 감사 기록을 만든다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from cadgen.stable_content_hash import stable_digest
from cadgen.typed_data_models import (
    ActionAttempt,
    ConflictCertificate,
    PipeState,
    ResolvedAction,
    StaticIssue,
)


def candidate_digest(action: ResolvedAction | dict[str, Any]) -> str:
    """action_id를 제외한 geometry/contract 입력만 해시한다."""

    payload = (
        action.model_dump(mode="json")
        if isinstance(action, ResolvedAction)
        else dict(action)
    )
    payload.pop("action_id", None)
    return stable_digest(payload)


def pipe_state_digest(state: PipeState) -> str:
    """복구 prefix 비교용 상태 콘텐츠 digest를 계산한다."""

    return stable_digest(
        {
            "contract_digest": state.contract_digest,
            "modeling_tolerance": state.modeling_tolerance,
            "placed_modules": [
                module.model_dump(mode="json") for module in state.placed_modules
            ],
            "open_ports": [port.model_dump(mode="json") for port in state.open_ports],
            "reserved_start_anchor": (
                state.reserved_start_anchor.model_dump(mode="json")
                if state.reserved_start_anchor is not None
                else None
            ),
            "remaining_goal_ids": [goal.goal_id for goal in state.remaining_goals],
        }
    )


def rejected_candidate_match(
    action: ResolvedAction,
    attempts: list[ActionAttempt],
    *,
    state_id: str,
    state_digest: str | None = None,
) -> ActionAttempt | None:
    """동일 state에서 이미 거절된 후보와 일치하면 그 attempt를 반환한다."""

    digest = candidate_digest(action)
    for attempt in reversed(attempts):
        if (
            attempt.status == "rejected"
            and (
                attempt.state_digest == state_digest
                if attempt.state_digest is not None and state_digest is not None
                else attempt.state_id == state_id
            )
            and isinstance(attempt.resolved, dict)
            and candidate_digest(attempt.resolved) == digest
        ):
            return attempt
    return None


def duplicate_candidate_certificate(
    action: ResolvedAction,
    prior: ActionAttempt,
) -> ConflictCertificate:
    """이미 거절된 후보 재제출을 금지하는 ConflictCertificate를 만든다."""

    digest = candidate_digest(action)
    evidence = {
        "candidate_digest": digest,
        "prior_step": prior.step_index,
        "prior_attempt": prior.attempt_index,
        "prior_phase": prior.phase,
        "prior_issue_codes": prior.issue_codes,
    }
    return ConflictCertificate(
        certificate_id=f"conflict-{stable_digest(evidence)[:16]}",
        conflict_type="local_geometry",
        failed_predicate="candidate_digest not in rejected_nogoods",
        proof_strength="proved",
        primitive_ids=[action.module],
        candidate_digest=digest,
        evidence_digest=stable_digest(evidence),
        causal_decision_ids=[action.action_id],
        earliest_backjump_step=prior.step_index,
        mutable_fields=["primitive", "variant", "causal_prefix"],
        allowed_routes=["change_primitive", "backjump", "probe"],
        message=(
            "This exact state-bound geometry was already rejected; replay cannot "
            "produce new evidence and is forbidden."
        ),
    )


def issue_certificate(
    issue: StaticIssue,
    *,
    candidate: ResolvedAction | None = None,
) -> ConflictCertificate:
    """legacy StaticIssue를 공통 실패 대수 ConflictCertificate로 변환한다."""

    code = issue.issue_code.upper()
    check = issue.check_name.lower()
    if "PROVIDER" in code or code == "PLANNING_FAILED":
        conflict_type = "provider"
        routes = ["retry_protocol", "retry_infrastructure"]
        proof = "unknown"
    elif "SCHEMA" in code or "PROTOCOL" in code:
        conflict_type = "protocol"
        routes = ["retry_protocol"]
        proof = "proved"
    elif "FREECAD" in code or "MCP" in code or "kernel" in check:
        conflict_type = "backend"
        routes = ["retry_infrastructure", "probe", "backjump"]
        proof = "independently_measured"
    elif "COLLISION" in code or "CLEARANCE" in code:
        conflict_type = "clearance"
        routes = ["probe", "backjump", "change_primitive"]
        proof = "proved" if issue.expected and issue.actual else "heuristic"
    elif "TOPOLOGY" in code or "PORT" in code:
        conflict_type = "topology"
        routes = ["change_primitive", "backjump"]
        proof = "proved"
    elif issue.step_index is None:
        conflict_type = "intent_contract"
        routes = ["repair_intent", "proven_infeasible"]
        proof = "proved"
    else:
        conflict_type = "local_geometry"
        routes = ["reauthor_current", "change_primitive", "backjump"]
        proof = "heuristic"
    candidate_hash = candidate_digest(candidate) if candidate is not None else None
    evidence = {
        "issue": issue.model_dump(mode="json"),
        "candidate_digest": candidate_hash,
    }
    return ConflictCertificate(
        certificate_id=f"conflict-{stable_digest(evidence)[:16]}",
        conflict_type=conflict_type,  # type: ignore[arg-type]
        failed_predicate=issue.issue_code,
        proof_strength=proof,  # type: ignore[arg-type]
        constraint_ids=[issue.check_name],
        primitive_ids=[candidate.module] if candidate is not None else [],
        candidate_digest=candidate_hash,
        evidence_digest=stable_digest(evidence),
        causal_decision_ids=[candidate.action_id] if candidate is not None else [],
        earliest_backjump_step=issue.step_index,
        mutable_fields=list((issue.suggestion or {}).get("parameter_errors") or []),
        allowed_routes=routes,  # type: ignore[arg-type]
        message=issue.message,
    )


def append_search_event(path: Path, event: dict[str, Any]) -> None:
    """버전 붙은 탐색 이벤트를 기존 기록을 덮지 않고 내구성 있게 추가한다."""

    payload = {
        "schema_version": 1,
        **event,
    }
    payload["event_digest"] = stable_digest(payload)
    encoded = (
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "append_search_event",
    "candidate_digest",
    "duplicate_candidate_certificate",
    "issue_certificate",
    "pipe_state_digest",
    "rejected_candidate_match",
]
