"""정적 검증 issue의 enforcement와 보고용 집계를 정의한다."""

from __future__ import annotations

from cadgen.typed_data_models import StaticIssue


# These checks compare a physically valid candidate with implementation detail
# authored by the intent/step LLM (exact pose, direction, style, or dimensions).
# In ``physical_only`` mode they remain visible deviations but do not overrule
# the LLM's realizable CAD choice.  Interface, graph-integrity, collision,
# clearance, bore, and solid-validity codes are deliberately absent.
PHYSICAL_ONLY_ADVISORY_CODES = frozenset(
    {
        "BRANCH_ANGLE_MISMATCH",
        "BRANCH_DIRECTION_MISMATCH",
        "BRANCH_LENGTH_MISMATCH",
        "BRANCH_OUTLET_DETAIL_MISMATCH",
        "BRANCH_PLANE_MISMATCH",
        "BRANCH_SECTION_MISMATCH",
        "BRANCH_STYLE_MISMATCH",
        "BRANCH_VECTOR_MISMATCH",
        "GOAL_BEND_RADIUS_MISMATCH",
        "GOAL_COMPONENT_DIMENSION_MISMATCH",
        "GOAL_COMPONENT_GEOMETRY_MISMATCH",
        "GOAL_CONNECTOR_DIRECTION_MISMATCH",
        "GOAL_CONNECT_WAYPOINT_MISMATCH",
        "GOAL_DIAMETER_MISMATCH",
        "GOAL_JUNCTION_DIMENSION_MISMATCH",
        "GOAL_LENGTH_MISMATCH",
        "GOAL_LENGTH_REQUIRES_FREECAD",
        "GOAL_ROUTE_CURVATURE_CONTRACT_MISMATCH",
        "GOAL_ROUTE_DIRECTION_MISMATCH",
        "GOAL_ROUTE_PATH_KIND_MISMATCH",
        "GOAL_ROUTE_TERMINAL_AXIS_MISMATCH",
        "GOAL_ROUTE_TERMINAL_POSITION_MISMATCH",
        "GOAL_ROUTE_WAYPOINT_MISMATCH",
        "GOAL_ROUTE_WAYPOINT_ORDER_MISMATCH",
        "GOAL_TERMINATION_THICKNESS_MISMATCH",
        "GOAL_TRANSITION_DIRECTION_MISMATCH",
        "GOAL_TRANSITION_LENGTH_MISMATCH",
        "GOAL_TRANSITION_OFFSET_MISMATCH",
        "GOAL_TRANSITION_WALL_MISMATCH",
        "GOAL_TURN_ANGLE_MISMATCH",
        "GOAL_TURN_PLANE_MISMATCH",
        "GOAL_TURN_SIGNED_ANGLE_MISMATCH",
        "JUNCTION_OUTLET_ROLE_MISMATCH",
        "JUNCTION_OUTPUT_COUNT_MISMATCH",
        "MOVE_DIRECTION_MISMATCH",
        "OPEN_PORT_DELTA_MISMATCH",
        "STEP_GOAL_LENGTH_EVIDENCE_MISSING",
        "STEP_GOAL_LENGTH_MISMATCH",
        "GEOMETRIC_CONSTRAINT_REQUIRES_FREECAD_LENGTH",
        "GEOMETRIC_CONSTRAINT_REQUIRES_FREECAD_BOUNDS",
        "SPLINE_CURVATURE_REQUIRES_FREECAD",
        "TURN_DIRECTION_MISMATCH",
        "UNEXPECTED_PRIMARY_OUTLET",
    }
)

_MATERIAL_RELATIVE_DEVIATION = 0.10
_COMPARABLE_METRIC_KEYS = frozenset(
    {
        "angle",
        "bend_radius",
        "centerline_length",
        "diameter",
        "length",
        "offset",
        "outer_diameter",
        "radius",
        "sweep_angle",
        "thickness",
        "transition_length",
        "wall_thickness",
    }
)


def _material_metric_deviation(issue: StaticIssue) -> bool:
    """Return true only when an issue proves a large comparable metric delta."""

    tolerance = float(issue.expected.get("tolerance", 0.0) or 0.0)
    for key in _COMPARABLE_METRIC_KEYS:
        expected = issue.expected.get(key)
        actual = issue.actual.get(key)
        if not isinstance(expected, (int, float)) or isinstance(expected, bool):
            continue
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            continue
        allowed = max(tolerance, abs(float(expected)) * _MATERIAL_RELATIVE_DEVIATION)
        if abs(float(actual) - float(expected)) > allowed:
            return True
    return False


def apply_validation_enforcement(
    issues: list[StaticIssue],
    enforcement: str,
) -> list[StaticIssue]:
    """물리 blocker와 LLM plan-fidelity 편차를 enforcement에 맞게 분리한다."""

    if enforcement == "strict":
        return issues
    if enforcement != "physical_only":
        raise ValueError("validation_enforcement must be physical_only or strict")
    result: list[StaticIssue] = []
    for issue in issues:
        if (
            issue.severity == "error"
            and issue.issue_code in PHYSICAL_ONLY_ADVISORY_CODES
            and not _material_metric_deviation(issue)
        ):
            suggestion = dict(issue.suggestion or {})
            suggestion.update(
                {
                    "validation_enforcement": "physical_only",
                    "disposition": "accepted_with_deviation",
                    "reason": (
                        "LLM plan-fidelity mismatch; the deterministic checks did "
                        "not prove a disconnected, self-intersecting, or invalid CAD solid"
                    ),
                }
            )
            issue = issue.model_copy(
                update={"severity": "warning", "suggestion": suggestion}
            )
        result.append(issue)
    return result


def has_errors(issues: list[StaticIssue]) -> bool:
    """commit을 막는 error가 하나라도 있는지 반환한다."""

    return any(issue.severity == "error" for issue in issues)


def error_count(issues: list[StaticIssue]) -> int:
    """error 심각도의 issue 수를 반환한다."""

    return sum(1 for issue in issues if issue.severity == "error")


def warning_count(issues: list[StaticIssue]) -> int:
    """warning 심각도의 issue 수를 반환한다."""

    return sum(1 for issue in issues if issue.severity == "warning")


def top_issue_ids(issues: list[StaticIssue], limit: int = 5) -> list[str]:
    """보고서에 표시할 상위 error 식별자를 반환한다."""

    return [issue.issue_id for issue in issues if issue.severity == "error"][:limit]


__all__ = [
    "PHYSICAL_ONLY_ADVISORY_CODES",
    "apply_validation_enforcement",
    "error_count",
    "has_errors",
    "top_issue_ids",
    "warning_count",
]
