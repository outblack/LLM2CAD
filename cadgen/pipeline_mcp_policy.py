"""파이프라인의 단계별 FreeCAD MCP 실행 정책을 결정한다.

오케스트레이터가 설정ㆍactionㆍ정적 검증 결과를 해석하는 규칙만 소유하며,
MCP 호출이나 산출물 기록 같은 부작용은 수행하지 않는다.
"""

from __future__ import annotations

from cadgen.runtime_settings import Settings
from cadgen.typed_data_models import PipeState, ResolvedAction, StepVerification


def should_run_step_mcp(settings: Settings, dry_run: bool) -> bool:
    """일반 step MCP 검증이 활성화되었는지 반환한다."""

    return (
        not dry_run
        and settings.freecad_mcp_enabled
        and settings.freecad_step_mcp_enabled
    )


def requires_progress_mcp(
    settings: Settings,
    dry_run: bool,
    state: PipeState,
    action: ResolvedAction,
) -> bool:
    """부분 spline 진행을 FreeCAD에서 확인해야 하는지 반환한다."""

    if dry_run or not settings.freecad_mcp_enabled:
        return False
    if action.params.get("path_kind") != "spline":
        return False
    completed = set(action.completed_goal_ids)
    for goal in state.remaining_goals:
        if (
            goal.goal_id in set(action.affected_goal_ids)
            and goal.goal_id not in completed
            and goal.length is not None
        ):
            return True
    return False


def requires_risk_mcp(
    settings: Settings,
    dry_run: bool,
    action: ResolvedAction,
    step: StepVerification,
) -> bool:
    """기하 위험도가 높은 action에 FreeCAD 검증이 필요한지 반환한다."""

    if dry_run or not settings.freecad_mcp_enabled:
        return False
    if action.module in {
        "transition",
        "junction",
        "connect_ports",
        "terminate",
        "inline_component",
    }:
        return True
    if action.module == "route" and action.params.get("path_kind") == "spline":
        return True
    return any(
        issue.issue_code == "STATIC_COLLISION_REQUIRES_FREECAD" for issue in step.issues
    )


def step_mcp_skip_reason(settings: Settings, dry_run: bool) -> str | None:
    """step MCP를 건너뛰는 설정상의 이유를 반환한다."""

    if dry_run:
        return "Dry-run skips step FreeCAD MCP."
    if not settings.freecad_mcp_enabled:
        return "FreeCAD MCP is disabled."
    if not settings.freecad_step_mcp_enabled:
        return "Step FreeCAD MCP is disabled."
    return None


def final_mcp_skip_reason(settings: Settings, dry_run: bool) -> str | None:
    """최종 MCP를 건너뛰는 설정상의 이유를 반환한다."""

    if dry_run:
        return "Dry-run skips final FreeCAD MCP."
    if not settings.freecad_mcp_enabled:
        return "FreeCAD MCP is disabled."
    return None


__all__ = [
    "final_mcp_skip_reason",
    "requires_progress_mcp",
    "requires_risk_mcp",
    "should_run_step_mcp",
    "step_mcp_skip_reason",
]
