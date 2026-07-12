"""상태 전이와 최종 CAD 계약을 FreeCAD 전후에 결정론적으로 검사한다.

의도ㆍ이전/다음 상태ㆍ행동ㆍ실측 증거를 입력받아 ``StaticIssue``와 critic을 반환한다.
오류는 구체적인 관측으로 기록하며 검증기 자체가 형상을 고치지는 않는다.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

from cadgen.typed_data_models import (
    CriticReport,
    CriticViewRequest,
    Direction,
    IntentResult,
    IssueSeverity,
    PatchSuggestion,
    PipeState,
    Port,
    ResolvedAction,
    StateTransition,
    StaticIssue,
    StepVerification,
)
from cadgen.vector3_math import (
    add,
    canonical_circular_arc_frame,
    circular_rim_mismatch,
    cross,
    direction_to_vector,
    dot,
    length,
    mul,
    normalize,
    rotate,
    sub,
    vec,
)
from cadgen.validation_issue_policy import (
    PHYSICAL_ONLY_ADVISORY_CODES as _PHYSICAL_ONLY_ADVISORY_CODES,
    apply_validation_enforcement,
    error_count,
    has_errors,
    top_issue_ids,
    warning_count,
)
from cadgen.static_geometry_metrics import (  # noqa: F401
    _analytic_route_arc_tangents,
    _arc_endpoint_tangents,
    _circumradius,
    _collision_envelope_reliable,
    _connection_contract_invalid,
    _connection_interface_metrics,
    _direction_score,
    _find_connectable_port,
    _find_port,
    _goal_path_points,
    _include_primary_outlet,
    _is_start_anchor_bootstrap_transition,
    _match_vectors_to_ports,
    _module_centerline_length,
    _module_centerline_points,
    _module_collision_segments,
    _module_envelope_radius,
    _module_primary_displacement,
    _module_spatial_samples,
    _module_turn_angle,
    _near,
    _normalize_vector_list,
    _point_segment_distance,
    _point_to_circular_arc_projection,
    _point_to_goal_path_projection,
    _point_to_polyline_distance,
    _point_to_polyline_projection,
    _port_role,
    _required_outlet_vectors,
    _same_direction,
    _segment_distance,
    _segment_has_endpoint,
    _three_point_arc_length,
    _vectors_json,
)
from cadgen.static_issue_builder import append_issue as _append_issue
from cadgen.static_final_validators import (  # noqa: F401
    _validate_conservative_collision,
    _validate_final_goal_lengths,
    _validate_final_graph,
    _validate_final_spline_curvature,
    _validate_geometric_constraints,
    _validate_intra_module_clearance,
)
from cadgen.static_transition_validators import (  # noqa: F401
    _validate_graph_transition,
    _validate_module_connection,
    _validate_route_continuity,
)
from cadgen.static_goal_validators import (  # noqa: F401
    _validate_branch_goal_direction,
    _validate_goal_completion,
    _validate_move_goal_direction,
    _validate_turn_goal_direction,
)

VECTOR_TOLERANCE = 1e-4
PARALLEL_DOT_THRESHOLD = 0.9999
BRANCH_DIRECTION_DOT_THRESHOLD = 0.35
EXPLICIT_VECTOR_DOT_THRESHOLD = 0.9999



class StaticValidationError(RuntimeError):
    """정적 검증 오류 때문에 현재 단계를 안전하게 진행할 수 없음을 나타낸다."""

    def __init__(
        self,
        stage: str,
        artifact_path: str,
        issues: list[StaticIssue],
    ) -> None:
        """정적 검증 실패 예외를 이슈와 함께 초기화한다."""

        self.stage = stage
        self.artifact_path = artifact_path
        self.issues = issues
        issue_ids = top_issue_ids(issues)
        issue_text = ", ".join(issue_ids) if issue_ids else "no issue ids"
        super().__init__(
            f"{stage} failed with {error_count(issues)} validation error(s). "
            f"Report: {artifact_path}. Top issues: {issue_text}"
        )


class CriticValidationError(StaticValidationError):
    """최종 critic이 하나 이상의 차단 오류를 보고했음을 나타낸다."""

    pass


def build_step_verification(
    before_state: PipeState,
    action: ResolvedAction,
    after_state: PipeState,
    intent: IntentResult,
    step_index: int,
    *,
    mcp_required: bool = False,
    mcp_status: str = "skipped",
    mcp_result_path: str | None = None,
    mcp_error: str | None = None,
    skipped_mcp_reason: str | None = None,
    validation_enforcement: str = "strict",
) -> StepVerification:
    """한 speculative 전이의 구조와 계약 검사를 묶어 단계 검증 결과를 만든다."""

    transition = build_state_transition(
        before_state,
        action,
        after_state,
        step_index,
    )
    issues = validate_step_checkpoint(
        before_state,
        action,
        after_state,
        intent,
        transition,
    )
    issues = apply_validation_enforcement(issues, validation_enforcement)
    return StepVerification(
        transition=transition,
        status="failed" if has_errors(issues) else "passed",
        issues=issues,
        mcp_status=mcp_status,  # type: ignore[arg-type]
        mcp_required=mcp_required,
        mcp_result_path=mcp_result_path,
        mcp_error=mcp_error,
        skipped_mcp_reason=skipped_mcp_reason,
    )


def build_state_transition(
    before_state: PipeState,
    action: ResolvedAction,
    after_state: PipeState,
    step_index: int,
) -> StateTransition:
    """전후 상태 차이에서 생성 모듈ㆍ포트ㆍedgeㆍ목표 변경 기록을 계산한다."""

    target_port = _find_port(action.target_port, before_state.open_ports)
    before_module_ids = {module.id for module in before_state.placed_modules}
    produced_modules = [
        module
        for module in after_state.placed_modules
        if module.id not in before_module_ids
    ]
    produced_module_id = produced_modules[0].id if len(produced_modules) == 1 else None
    before_open_ids = [port.id for port in before_state.open_ports]
    after_open_ids = [port.id for port in after_state.open_ports]
    before_open_set = set(before_open_ids)
    after_open_set = set(after_open_ids)
    consumed_goal_index = 0
    consumed_goal_model = None
    affected = set(action.affected_goal_ids)
    affected_goal_models = [
        goal for goal in before_state.remaining_goals if goal.goal_id in affected
    ]
    if affected:
        for index, goal in enumerate(before_state.remaining_goals):
            if goal.goal_id in affected:
                consumed_goal_index = index
                consumed_goal_model = goal
                break
    elif before_state.remaining_goals:
        consumed_goal_model = before_state.remaining_goals[0]
    consumed_goal = (
        consumed_goal_model.model_dump(mode="json", exclude_none=True)
        if consumed_goal_model is not None
        else None
    )
    return StateTransition(
        step_index=step_index,
        consumed_goal_index=consumed_goal_index,
        consumed_goal=consumed_goal,
        affected_goals=[
            goal.model_dump(mode="json", exclude_none=True)
            for goal in affected_goal_models
        ],
        state_before_id=before_state.state_id,
        state_after_id=after_state.state_id,
        action_id=action.action_id,
        module=action.module,
        target_port_before=(
            target_port.model_dump(mode="json") if target_port is not None else {}
        ),
        produced_module_id=produced_module_id,
        produced_port_ids=[
            port_id for port_id in after_open_ids if port_id not in before_open_set
        ],
        removed_port_ids=[
            port_id
            for port_id in [
                *before_open_ids,
                *(
                    [before_state.reserved_start_anchor.id]
                    if before_state.reserved_start_anchor is not None
                    else []
                ),
            ]
            if port_id
            not in {
                *after_open_set,
                *(
                    [after_state.reserved_start_anchor.id]
                    if after_state.reserved_start_anchor is not None
                    else []
                ),
            }
        ],
        consumed_port_ids=list(action.consumed_port_ids or [action.target_port]),
        connection_edge_ids=[
            edge.edge_id
            for edge in after_state.connection_edges
            if edge.edge_id
            not in {item.edge_id for item in before_state.connection_edges}
        ],
        affected_goal_ids=list(action.affected_goal_ids),
        completed_goal_ids=list(action.completed_goal_ids),
        satisfied_components=list(action.satisfied_components),
        open_port_ids_before=before_open_ids,
        open_port_ids_after=after_open_ids,
    )


def validate_step_checkpoint(
    before_state: PipeState,
    action: ResolvedAction,
    after_state: PipeState,
    intent: IntentResult,
    transition: StateTransition,
) -> list[StaticIssue]:
    """commit 전 전이의 그래프ㆍ목표ㆍ기하 불변식을 검사해 이슈 목록을 반환한다."""

    issues: list[StaticIssue] = []
    target_port = _find_port(action.target_port, before_state.open_ports)
    produced_module = _produced_module(before_state, after_state)
    goals = transition.affected_goals or (
        [transition.consumed_goal] if transition.consumed_goal else [{}]
    )
    _validate_goal_order_and_dependencies(issues, before_state, action, transition)
    _validate_atomic_goal_binding(issues, action, transition)
    _validate_monotone_step_constraints(issues, after_state, intent, transition)

    if target_port is None:
        _append_issue(
            issues,
            "TARGET_PORT_MISSING",
            "error",
            "target_port_open",
            f"Target port is not open: {action.target_port}",
            transition=transition,
            target_port_id=action.target_port,
            expected={"target_port": "open"},
            actual={"open_ports": transition.open_port_ids_before},
        )
        return issues

    produced_count = _produced_module_count(before_state, after_state)
    if produced_count != 1:
        _append_issue(
            issues,
            "MODULE_COUNT_DELTA",
            "error",
            "module_delta",
            "Each action must add exactly one module.",
            transition=transition,
            expected={"added_modules": 1},
            actual={"added_modules": produced_count},
        )

    if produced_module is not None:
        _validate_module_connection(
            issues,
            transition,
            action,
            target_port,
            produced_module,
            after_state.modeling_tolerance,
        )
        _validate_capability_coverage(
            issues,
            transition,
            action,
            produced_module,
            goals,
        )
        _validate_affected_component_multiplicity(
            issues,
            transition,
            before_state,
            produced_module,
            goals,
        )
        for goal in goals:
            _validate_move_goal_direction(
                issues,
                transition,
                target_port,
                produced_module,
                goal,
            )
            _validate_turn_goal_direction(
                issues,
                transition,
                produced_module,
                goal,
            )
            _validate_branch_goal_direction(
                issues,
                transition,
                action,
                target_port,
                produced_module,
                goal,
            )
            _validate_goal_completion(
                issues,
                transition,
                action,
                before_state,
                target_port,
                produced_module,
                goal,
            )
        _validate_graph_transition(
            issues,
            transition,
            action,
            before_state,
            after_state,
            produced_module,
        )
        _validate_route_continuity(
            issues,
            transition,
            action,
            before_state,
            target_port,
            produced_module,
        )
        _validate_intra_module_clearance(
            issues,
            transition,
            produced_module,
        )
        _validate_conservative_collision(
            issues,
            transition,
            before_state,
            produced_module,
        )

    return issues


def validate_step_mcp_evidence(
    intent: IntentResult,
    state: PipeState,
    action: ResolvedAction,
    transition: StateTransition,
    step_verifications: list[StepVerification],
) -> list[StaticIssue]:
    """validate_step_mcp_evidence 관련 계약을 검증한다."""

    issues: list[StaticIssue] = []
    _validate_geometric_constraints(issues, state, step_verifications)
    completed = set(action.completed_goal_ids)
    for goal in intent.target_behavior:
        if (
            goal.goal_id not in completed
            or goal.type not in {"route", "connector"}
            or goal.length is None
        ):
            continue
        modules = [
            module
            for historical_action, module in zip(
                state.action_history, state.placed_modules
            )
            if goal.goal_id in historical_action.affected_goal_ids
        ]
        if goal.type == "connector" and goal.component is not None:
            modules = [
                module
                for module in modules
                if module.type == "inline_component"
                and module.params.get("component_type") == goal.component
            ]
        missing_splines = [
            module.id
            for module in modules
            if module.params.get("path_kind") == "spline"
            and "centerline_length" not in state.module_measurements.get(module.id, {})
        ]
        if missing_splines:
            _append_issue(
                issues,
                "STEP_GOAL_LENGTH_EVIDENCE_MISSING",
                "error",
                "step_mcp_goal_length",
                "FreeCAD did not return the spline length needed before commit.",
                transition=transition,
                expected={"goal_id": goal.goal_id, "length": goal.length},
                actual={"module_ids": missing_splines},
            )
            continue
        actual_length = sum(
            state.module_measurements.get(module.id, {}).get(
                "centerline_length", _module_centerline_length(module)
            )
            for module in modules
        )
        expected_length = float(goal.length)
        tolerance = max(VECTOR_TOLERANCE, expected_length * 1e-3)
        if abs(actual_length - expected_length) > tolerance:
            _append_issue(
                issues,
                "STEP_GOAL_LENGTH_MISMATCH",
                "error",
                "step_mcp_goal_length",
                "Digest-bound FreeCAD length does not complete the immutable goal.",
                transition=transition,
                expected={
                    "goal_id": goal.goal_id,
                    "centerline_length": expected_length,
                    "tolerance": tolerance,
                },
                actual={"centerline_length": actual_length},
            )
    _validate_final_spline_curvature(issues, state, step_verifications)
    return issues


def _validate_monotone_step_constraints(
    issues: list[StaticIssue],
    state: PipeState,
    intent: IntentResult,
    transition: StateTransition,
) -> None:
    """step 제약이 단조 증가하는지 검사한다."""

    for constraint in intent.geometric_constraints:
        if constraint.type == "max_module_count" and constraint.value is not None:
            actual = len(state.placed_modules)
            if actual > int(round(constraint.value)):
                _append_issue(
                    issues,
                    "STEP_MAX_MODULE_COUNT_EXCEEDED",
                    "error",
                    "step_geometric_constraint",
                    "The accepted candidate already exceeds the monotone module limit.",
                    transition=transition,
                    expected={
                        "constraint_id": constraint.constraint_id,
                        "max_module_count": int(round(constraint.value)),
                    },
                    actual={"module_count": actual},
                )
        elif (
            constraint.type == "max_total_centerline_length"
            and constraint.value is not None
            and not any(
                module.params.get("path_kind") == "spline"
                for module in state.placed_modules
            )
        ):
            actual_length = sum(
                _module_centerline_length(module) for module in state.placed_modules
            )
            if actual_length > float(constraint.value) + VECTOR_TOLERANCE:
                _append_issue(
                    issues,
                    "STEP_MAX_CENTERLINE_LENGTH_EXCEEDED",
                    "error",
                    "step_geometric_constraint",
                    "The candidate already exceeds the monotone centerline limit.",
                    transition=transition,
                    expected={
                        "constraint_id": constraint.constraint_id,
                        "maximum": float(constraint.value),
                    },
                    actual={"centerline_length": actual_length},
                )


def _validate_goal_order_and_dependencies(
    issues: list[StaticIssue],
    before_state: PipeState,
    action: ResolvedAction,
    transition: StateTransition,
) -> None:
    """goal 순서와 의존성이 유효한지 검사한다."""

    if not before_state.remaining_goals:
        return
    affected = set(action.affected_goal_ids)
    completed = set(action.completed_goal_ids)
    completed_history = {
        goal_id
        for historical_action in before_state.action_history
        for goal_id in historical_action.completed_goal_ids
    }
    goals_by_id = {
        goal.goal_id: goal
        for goal in before_state.remaining_goals
        if goal.goal_id is not None
    }
    for index, goal in enumerate(before_state.remaining_goals):
        if goal.goal_id not in affected:
            continue
        missing_dependencies = {
            dependency_id
            for dependency_id in set(goal.depends_on_goal_ids) - completed_history
            if not _same_action_dependency_is_provable(
                dependent_goal=goal,
                dependency_goal=goals_by_id.get(dependency_id),
                completed_goal_ids=completed,
                action=action,
            )
        }
        if missing_dependencies:
            _append_issue(
                issues,
                "GOAL_DEPENDENCY_BYPASS",
                "error",
                "goal_order",
                "An affected goal has dependencies that were not completed in an earlier transition.",
                transition=transition,
                expected={"completed_dependencies": sorted(goal.depends_on_goal_ids)},
                actual={
                    "goal_id": goal.goal_id,
                    "missing_dependencies": sorted(missing_dependencies),
                },
            )
        bypassed = [
            str(prior.goal_id)
            for prior in before_state.remaining_goals[:index]
            if prior.goal_id is not None and prior.goal_id not in completed
        ]
        if not goal.allow_parallel and bypassed:
            _append_issue(
                issues,
                "GOAL_ORDER_BYPASS",
                "error",
                "goal_order",
                "A non-parallel goal advanced before the earlier pending goal completed.",
                transition=transition,
                expected={"complete_prior_goal_ids": bypassed},
                actual={"advanced_goal_id": goal.goal_id},
            )


def _same_action_dependency_is_provable(
    *,
    dependent_goal: Any,
    dependency_goal: Any,
    completed_goal_ids: set[str],
    action: ResolvedAction,
) -> bool:
    """same_action_dependency_is_provable 동일/증명 가능 여부를 판정한다."""

    if dependency_goal is None or dependency_goal.goal_id not in completed_goal_ids:
        return False
    return bool(
        dependent_goal.type == "connect"
        and dependent_goal.connection_target == "start_anchor"
        and dependency_goal.type == "turn"
        and dependent_goal.goal_id in completed_goal_ids
        and action.module == "connect_ports"
        and action.params.get("path_kind") == "circular_arc"
    )


def _validate_atomic_goal_binding(
    issues: list[StaticIssue],
    action: ResolvedAction,
    transition: StateTransition,
) -> None:
    """goal 바인딩이 원자적인지 검사한다."""

    atomic_types = {
        "turn",
        "branch",
        "diameter_change",
        "connect",
        "end",
        "connector",
    }
    completed = set(action.completed_goal_ids)
    for goal in transition.affected_goals:
        goal_type = goal.get("type")
        if goal_type not in atomic_types:
            continue
        goal_id = goal.get("goal_id")
        component_mismatch = (
            goal_type == "connector"
            and goal.get("component") is not None
            and action.params.get("component_type") != goal.get("component")
        )
        if goal_id not in completed or component_mismatch:
            _append_issue(
                issues,
                "ATOMIC_GOAL_ACTION_MISMATCH",
                "error",
                "atomic_goal_binding",
                "An indivisible topology/component goal must be completed by its matching action.",
                transition=transition,
                expected={
                    "goal_id": goal_id,
                    "component_type": goal.get("component"),
                    "completed_in_same_action": True,
                },
                actual={
                    "module": action.module,
                    "component_type": action.params.get("component_type"),
                    "completed_goal_ids": sorted(completed),
                },
            )


def _validate_affected_component_multiplicity(
    issues: list[StaticIssue],
    transition: StateTransition,
    before_state: PipeState,
    produced_module: Any,
    goals: list[dict[str, Any]],
) -> None:
    """영향 컴포넌트 개수 제약을 검사한다."""

    for goal in goals:
        if goal.get("type") != "connector" or goal.get("component") is None:
            continue
        goal_id = goal.get("goal_id")
        component = str(goal["component"])
        matching_module_ids = [
            module.id
            for historical_action, module in zip(
                before_state.action_history, before_state.placed_modules
            )
            if goal_id in historical_action.affected_goal_ids
            and module.type == "inline_component"
            and module.params.get("component_type") == component
        ]
        if (
            produced_module.type == "inline_component"
            and produced_module.params.get("component_type") == component
        ):
            matching_module_ids.append(produced_module.id)
        if len(matching_module_ids) > 1:
            _append_issue(
                issues,
                "GOAL_COMPONENT_MULTIPLICITY_MISMATCH",
                "error",
                "goal_component_multiplicity",
                "One connector goal may own exactly one matching inline component instance.",
                transition=transition,
                module_id=produced_module.id,
                expected={"component": component, "count": 1},
                actual={
                    "component": component,
                    "count": len(matching_module_ids),
                    "module_ids": matching_module_ids,
                },
            )


def _validate_capability_coverage(
    issues: list[StaticIssue],
    transition: StateTransition,
    action: ResolvedAction,
    produced_module: Any,
    goals: list[dict[str, Any]],
) -> None:
    """카탈로그 capability가 goal을 덮는지 검사한다."""

    outputs = [
        port for name, port in produced_module.ports.items() if name.startswith("out")
    ]
    inlet = produced_module.ports.get("in") or produced_module.ports.get("in_a")
    observed: set[str] = set()
    if len(outputs) > 1:
        observed.add("branches")
    if len(action.consumed_port_ids) == 2 and not outputs:
        observed.add("closes_two_ports")
    if len(action.consumed_port_ids) == 1 and not outputs:
        observed.add("seals_port")
    if inlet is not None and any(
        abs(port.outer_diameter - inlet.outer_diameter) > VECTOR_TOLERANCE
        or abs(port.wall_thickness - inlet.wall_thickness) > VECTOR_TOLERANCE
        for port in outputs
    ):
        observed.add("changes_section")
    component_type = produced_module.params.get("component_type")
    if component_type is not None:
        observed.add(f"inline_component:{component_type}")

    covered: set[str] = set()
    for goal in goals:
        goal_type = goal.get("type")
        if goal_type == "branch":
            covered.add("branches")
            if (
                goal.get("branch_outer_diameter") is not None
                or goal.get("branch_wall_thickness") is not None
                or any(
                    outlet.get("outer_diameter") is not None
                    or outlet.get("wall_thickness") is not None
                    for outlet in (goal.get("required_outlets") or [])
                    if isinstance(outlet, dict)
                )
            ):
                covered.add("changes_section")
        elif goal_type == "diameter_change":
            covered.add("changes_section")
        elif goal_type == "connect":
            covered.add("closes_two_ports")
        elif goal_type == "end" and goal.get("end_type") in {"cap", "plug"}:
            covered.add("seals_port")
        elif goal_type == "connector" and goal.get("component") is not None:
            covered.add(f"inline_component:{goal['component']}")

    invented = sorted(observed - covered)
    if invented:
        _append_issue(
            issues,
            "UNREQUESTED_GEOMETRY_CAPABILITY",
            "error",
            "goal_capability_coverage",
            "The action introduced observable special geometry not owned by an affected goal.",
            transition=transition,
            module_id=produced_module.id,
            expected={"covered_capabilities": sorted(covered)},
            actual={"uncovered_capabilities": invented},
        )


def build_final_critic_report(
    intent: IntentResult,
    state: PipeState,
    step_verifications: list[StepVerification],
    *,
    skipped_mcp_reason: str | None = None,
    validation_enforcement: str = "strict",
) -> CriticReport:
    """누적 단계 증거와 최종 상태를 계약 전체에 대조해 최종 판정을 만든다."""

    issues = [issue for step in step_verifications for issue in step.issues]
    final_issues: list[StaticIssue] = []

    if state.remaining_goals:
        _append_issue(
            final_issues,
            "REMAINING_GOALS",
            "error",
            "final_goal_exhaustion",
            "Final state still has unconsumed goals.",
            expected={"remaining_goals": 0},
            actual={"remaining_goals": len(state.remaining_goals)},
            suggestion={"operation": "increase_max_iter_or_replan"},
        )

    if not any(_module_generates_geometry(module) for module in state.placed_modules):
        _append_issue(
            final_issues,
            "NO_GEOMETRY_MODULES",
            "error",
            "final_geometry_presence",
            "Final state has no module that generates outer pipe geometry.",
            expected={"geometry_module_count": ">0"},
            actual={
                "placed_modules": [
                    {"id": module.id, "type": module.type, "params": module.params}
                    for module in state.placed_modules
                ]
            },
            suggestion={"operation": "add_geometry_module_before_open_end_marker"},
        )

    _validate_final_graph(final_issues, state)
    _validate_geometric_constraints(final_issues, state, step_verifications)
    _validate_final_goal_lengths(final_issues, intent, state, step_verifications)
    _validate_final_spline_curvature(final_issues, state, step_verifications)

    satisfied_components = [
        str(module.params.get("component_type"))
        for module in state.placed_modules
        if module.type == "inline_component"
        and module.geometry_id
        and module.params.get("component_type")
    ]
    required_component_counts = Counter(intent.required_components)
    satisfied_component_counts = Counter(satisfied_components)
    missing_components = {
        component: required_count - satisfied_component_counts[component]
        for component, required_count in required_component_counts.items()
        if satisfied_component_counts[component] < required_count
    }
    if missing_components:
        _append_issue(
            final_issues,
            "REQUIRED_COMPONENTS_UNSATISFIED",
            "error",
            "final_component_contract",
            "One or more user-required components have no matching geometry-producing inline component.",
            expected={"required_components": intent.required_components},
            actual={
                "satisfied_component_counts": dict(
                    sorted(satisfied_component_counts.items())
                ),
                "missing_components": missing_components,
            },
            suggestion={"operation": "replan_required_components"},
        )
    excess_components = {
        component: satisfied_count - required_component_counts[component]
        for component, satisfied_count in satisfied_component_counts.items()
        if satisfied_count > required_component_counts[component]
    }
    if excess_components:
        _append_issue(
            final_issues,
            "COMPONENT_MULTIPLICITY_MISMATCH",
            "error",
            "final_component_contract",
            "The final geometry contains more inline component instances than the immutable contract.",
            expected={
                "required_component_counts": dict(
                    sorted(required_component_counts.items())
                )
            },
            actual={
                "satisfied_component_counts": dict(
                    sorted(satisfied_component_counts.items())
                ),
                "excess_components": excess_components,
            },
            suggestion={"operation": "rollback_excess_component"},
        )

    actual_open_ports = len(state.open_ports)
    if (
        intent.expected_open_ports is None
        or intent.expected_open_ports_source == "unknown"
    ):
        _append_issue(
            final_issues,
            "EXPECTED_OPEN_PORTS_UNKNOWN",
            "warning",
            "final_open_port_contract",
            "Intent did not provide a trusted expected open-port count.",
            expected={"expected_open_ports": "explicit_or_derived"},
            actual={
                "actual_open_ports": actual_open_ports,
                "provided_expected_open_ports": intent.expected_open_ports,
                "source": intent.expected_open_ports_source,
            },
            suggestion={"operation": "improve_intent_extraction"},
        )
    elif actual_open_ports != intent.expected_open_ports:
        _append_issue(
            final_issues,
            "OPEN_PORT_COUNT_MISMATCH",
            "error",
            "final_open_port_count",
            "Final open-port count does not match the intent.",
            port_ids=[port.id for port in state.open_ports],
            expected={"expected_open_ports": intent.expected_open_ports},
            actual={"actual_open_ports": actual_open_ports},
            suggestion={"operation": "replan_terminal_ports"},
        )

    for group in _duplicate_open_port_groups(state.open_ports):
        _append_issue(
            final_issues,
            "DUPLICATE_OPEN_PORT_POSITION",
            "error",
            "final_duplicate_open_ports",
            "Multiple terminal open ports occupy the same position with parallel axes.",
            port_ids=group["port_ids"],
            expected={"unique_terminal_positions": True, "tolerance": VECTOR_TOLERANCE},
            actual=group,
            suggestion={"operation": "deduplicate_or_move_terminal_port"},
        )

    required_vectors = _intent_required_outlet_vectors(intent)
    if required_vectors:
        assignment = _match_vectors_to_ports(required_vectors, state.open_ports)
        if assignment["missing_vectors"]:
            _append_issue(
                final_issues,
                "FINAL_OUTLET_VECTOR_MISMATCH",
                "error",
                "final_terminal_vector_coverage",
                "Final open terminals do not cover every required outlet vector.",
                port_ids=[port.id for port in state.open_ports],
                expected={
                    "expected_vectors": _vectors_json(required_vectors),
                    "threshold": EXPLICIT_VECTOR_DOT_THRESHOLD,
                    "distinct_ports": True,
                },
                actual=assignment,
                suggestion={
                    "operation": "replan_terminal_vectors",
                    "required_outlet_vectors": _vectors_json(required_vectors),
                },
            )

    issues.extend(final_issues)
    issues = apply_validation_enforcement(issues, validation_enforcement)
    errors = error_count(issues)
    warnings = warning_count(issues)
    return CriticReport(
        passed=errors == 0,
        verification_status="passed" if errors == 0 else "failed",
        error_count=errors,
        warning_count=warnings,
        expected_open_ports=intent.expected_open_ports,
        actual_open_ports=actual_open_ports,
        expected_open_ports_source=intent.expected_open_ports_source,
        issues=issues,
        view_requests=_view_requests(state, issues),
        patch_suggestions=_patch_suggestions(issues),
        skipped_mcp_reason=skipped_mcp_reason,
        next_actions=_next_actions(issues),
    )












def _intent_required_outlet_vectors(
    intent: IntentResult,
) -> list[tuple[float, float, float]]:
    """intent에서 요구된 outlet 벡터를 읽는다."""

    vectors: list[tuple[float, float, float]] = []
    for goal in intent.target_behavior:
        explicit_vectors = _normalize_vector_list(
            goal.required_outlet_vectors
            or [item.axis for item in goal.required_outlets]
        )
        if explicit_vectors:
            vectors.extend(explicit_vectors)
        else:
            vectors.extend(
                direction_to_vector(direction)
                for direction in goal.required_outlet_directions
            )
    return vectors


def _produced_module(before_state: PipeState, after_state: PipeState) -> Any | None:
    """전이가 생성한 모듈 참조를 반환한다."""

    before_module_ids = {module.id for module in before_state.placed_modules}
    modules = [
        module
        for module in after_state.placed_modules
        if module.id not in before_module_ids
    ]
    if len(modules) != 1:
        return None
    return modules[0]


def _produced_module_count(before_state: PipeState, after_state: PipeState) -> int:
    """전이가 생성한 모듈 개수를 센다."""

    before_module_ids = {module.id for module in before_state.placed_modules}
    return sum(
        1 for module in after_state.placed_modules if module.id not in before_module_ids
    )


def _parallel(a: Any, b: Any) -> bool:
    """두 벡터가 평행한지 판정한다."""

    return abs(dot(normalize(vec(a)), normalize(vec(b)))) >= PARALLEL_DOT_THRESHOLD


def _duplicate_open_port_groups(ports: list[Port]) -> list[dict[str, Any]]:
    """중복 열린 port 그룹을 탐지한다."""

    groups: list[dict[str, Any]] = []
    used: set[str] = set()
    for index, port in enumerate(ports):
        if port.id in used:
            continue
        group = [port]
        for other in ports[index + 1 :]:
            if _near(port.position, other.position) and _parallel(
                port.axis, other.axis
            ):
                group.append(other)
        if len(group) > 1:
            used.update(item.id for item in group)
            groups.append(
                {
                    "port_ids": [item.id for item in group],
                    "position": list(group[0].position),
                    "axes": [list(item.axis) for item in group],
                    "tolerance": VECTOR_TOLERANCE,
                }
            )
    return groups


def _module_generates_geometry(module: Any) -> bool:
    """모듈이 실제 형상을 생성하는지 판정한다."""

    if module.type == "cap_pipe" and module.params.get("end_type") == "open":
        return False
    return module.type in {
        "straight_pipe",
        "connector_pipe",
        "bend_pipe",
        "junction_pipe",
        "reducer_pipe",
        "cap_pipe",
        "route",
        "transition",
        "junction",
        "connect_ports",
        "terminate",
        "inline_component",
    }


def _view_requests(
    state: PipeState,
    issues: list[StaticIssue],
) -> list[CriticViewRequest]:
    """critic용 뷰 요청 목록을 구성한다."""

    target_modules = sorted(
        {issue.module_id for issue in issues if issue.module_id}
        or {module.id for module in state.placed_modules}
    )
    target_ports = sorted(
        {port_id for issue in issues for port_id in issue.port_ids}
        or {port.id for port in state.open_ports}
    )
    views = [
        ("front", "front orthographic"),
        ("right", "right orthographic"),
        ("top", "top orthographic"),
        ("isometric", "isometric overview"),
    ]
    return [
        CriticViewRequest(
            view_id=view_id,  # type: ignore[arg-type]
            camera=camera,
            target_module_ids=target_modules,
            target_port_ids=target_ports,
            purpose="Visually inspect topology, branch direction, open rims, and seams.",
            required=False,
            evidence_status="pending",
            unavailable_reason=None,
        )
        for view_id, camera in views
    ]


def _patch_suggestions(issues: list[StaticIssue]) -> list[PatchSuggestion]:
    """실패 기반 패치 제안을 만든다."""

    suggestions: list[PatchSuggestion] = []
    for issue in issues:
        if issue.severity != "error":
            continue
        operation = {
            "BRANCH_DIRECTION_MISMATCH": "adjust_junction_branch_direction",
            "BRANCH_VECTOR_MISMATCH": "assign_required_outlet_vectors",
            "UNEXPECTED_PRIMARY_OUTLET": "remove_unexpected_primary_outlet",
            "JUNCTION_OUTPUT_COUNT_MISMATCH": "repair_junction_output_count",
            "OPEN_PORT_DELTA_MISMATCH": "repair_open_port_transition",
            "JUNCTION_OUTLET_ROLE_MISMATCH": "repair_junction_outlet_roles",
            "FINAL_OUTLET_VECTOR_MISMATCH": "replan_terminal_vectors",
            "OPEN_PORT_COUNT_MISMATCH": "replan_terminal_ports",
            "DUPLICATE_OPEN_PORT_POSITION": "deduplicate_or_move_terminal_port",
        }.get(issue.issue_code, "review_static_issue")
        suggestions.append(
            PatchSuggestion(
                suggestion_id=f"PATCH_{len(suggestions) + 1:03d}_{issue.issue_code}",
                target_module_id=issue.module_id,
                target_port_ids=issue.port_ids,
                issue_ids=[issue.issue_id],
                operation=operation,
                params=issue.suggestion,
                rationale=issue.message,
            )
        )
    return suggestions


def _next_actions(issues: list[StaticIssue]) -> list[str]:
    """다음 권장 행동 목록을 만든다."""

    if not has_errors(issues):
        return ["Proceed to final FreeCAD MCP execution."]
    actions = []
    for issue in issues:
        if issue.severity == "error":
            actions.append(f"Fix {issue.issue_code}: {issue.message}")
    return actions[:5]
