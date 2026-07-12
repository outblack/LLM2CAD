"""검증 순서를 보존하며 결정론적 ``StaticIssue`` 식별자를 생성한다."""

from __future__ import annotations

from typing import Any

from cadgen.typed_data_models import (
    IssueSeverity,
    StateTransition,
    StaticIssue,
)


def append_issue(
    issues: list[StaticIssue],
    issue_code: str,
    severity: IssueSeverity,
    check_name: str,
    message: str,
    *,
    transition: StateTransition | None = None,
    module_id: str | None = None,
    port_ids: list[str] | None = None,
    target_port_id: str | None = None,
    expected: dict[str, Any] | None = None,
    actual: dict[str, Any] | None = None,
    suggestion: dict[str, Any] | None = None,
) -> None:
    """현재 issue 순서에 기반한 식별자로 목록에 issue를 추가한다."""

    step_index = transition.step_index if transition is not None else None
    prefix = f"STEP_{step_index:04d}" if step_index is not None else "FINAL"
    issues.append(
        StaticIssue(
            issue_id=f"{prefix}_{len(issues) + 1:02d}_{issue_code}",
            severity=severity,
            issue_code=issue_code,
            check_name=check_name,
            message=message,
            step_index=step_index,
            action_id=transition.action_id if transition is not None else None,
            module_id=module_id
            or (transition.produced_module_id if transition is not None else None),
            port_ids=port_ids or [],
            target_port_id=target_port_id
            or (
                transition.target_port_before.get("id")
                if transition is not None
                else None
            ),
            consumed_goal_index=(
                transition.consumed_goal_index if transition is not None else None
            ),
            expected=expected or {},
            actual=actual or {},
            suggestion=suggestion or {},
        )
    )


__all__ = ["append_issue"]
