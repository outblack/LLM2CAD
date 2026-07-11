"""상태 전이와 최종 CAD 계약을 FreeCAD 전후에 결정론적으로 검사한다.

의도ㆍ이전/다음 상태ㆍ행동ㆍ실측 증거를 입력받아 ``StaticIssue``와 critic을 반환한다.
오류는 구체적인 관측으로 기록하며 검증기 자체가 형상을 고치지는 않는다.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

from cadgen.schemas import (
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
from cadgen.vector import (
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
    """Re-check evidence-dependent monotone/goal contracts before commit."""

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
    """Permit the one atomic dependency pair measured by a closure arc."""

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


def _connection_interface_metrics(left: Port, right: Port) -> dict[str, float]:
    position_error = length(sub(vec(left.position), vec(right.position)))
    alignment = -dot(normalize(vec(left.axis)), normalize(vec(right.axis)))
    clamped_alignment = max(-1.0, min(1.0, alignment))
    return {
        "position_error": position_error,
        "anti_parallel_axis_dot": alignment,
        "axis_angle_error": math.acos(clamped_alignment),
        "od_error": abs(left.outer_diameter - right.outer_diameter),
        "id_error": abs(left.inner_diameter - right.inner_diameter),
        "wall_error": abs(left.wall_thickness - right.wall_thickness),
        "outer_rim_error": circular_rim_mismatch(
            position_error,
            left.outer_diameter / 2.0,
            right.outer_diameter / 2.0,
            alignment,
        ),
        "inner_rim_error": circular_rim_mismatch(
            position_error,
            left.inner_diameter / 2.0,
            right.inner_diameter / 2.0,
            alignment,
        ),
    }


def _connection_contract_invalid(
    edge: Any,
    metrics: dict[str, float],
    tolerance: float,
) -> bool:
    return bool(
        metrics["position_error"] > tolerance
        or metrics["anti_parallel_axis_dot"] < PARALLEL_DOT_THRESHOLD
        or metrics["od_error"] > tolerance
        or metrics["id_error"] > tolerance
        or metrics["wall_error"] > tolerance
        or metrics["outer_rim_error"] > tolerance
        or metrics["inner_rim_error"] > tolerance
        or not edge.connector_type_match
        or not edge.connector_gender_match
        or not edge.connector_standard_match
    )


def _validate_final_graph(issues: list[StaticIssue], state: PipeState) -> None:
    module_ids = {module.id for module in state.placed_modules}
    port_ids = set(state.port_nodes)
    derived_open_ids = [port.id for port in state.open_ports]
    edge_ids = [edge.edge_id for edge in state.connection_edges]
    incidence_pairs = [
        (edge.module_id, edge.port_id) for edge in state.module_incidence_edges
    ]
    if state.reserved_start_anchor is not None:
        _append_issue(
            issues,
            "FINAL_RESERVED_START_ANCHOR_UNCONSUMED",
            "error",
            "final_port_graph",
            "The first module inlet is still reserved for a pending START-seam closure.",
            port_ids=[state.reserved_start_anchor.id],
            expected={"reserved_start_anchor": None},
            actual={
                "reserved_start_anchor": state.reserved_start_anchor.model_dump(
                    mode="json"
                )
            },
        )
    if len(edge_ids) != len(set(edge_ids)) or len(incidence_pairs) != len(
        set(incidence_pairs)
    ):
        _append_issue(
            issues,
            "FINAL_GRAPH_DUPLICATE_EDGE",
            "error",
            "final_port_graph",
            "Graph edge identifiers and module-port incidence pairs must be unique.",
            actual={
                "connection_edge_ids": edge_ids,
                "incidence_pairs": incidence_pairs,
            },
        )
    expected_incidence = {
        (module.id, port.id)
        for module in state.placed_modules
        for port in module.ports.values()
    }
    if set(incidence_pairs) != expected_incidence:
        _append_issue(
            issues,
            "FINAL_GRAPH_INCIDENCE_COVERAGE_MISMATCH",
            "error",
            "final_port_graph",
            "Every persisted module port must have exactly one incidence edge.",
            expected={"incidence_pairs": sorted(expected_incidence)},
            actual={"incidence_pairs": sorted(set(incidence_pairs))},
        )
    for edge in state.connection_edges:
        left = state.port_nodes.get(edge.port_a_id)
        right = state.port_nodes.get(edge.port_b_id)
        if left is None or right is None:
            continue
        metrics = _connection_interface_metrics(left, right)
        if _connection_contract_invalid(edge, metrics, state.modeling_tolerance):
            _append_issue(
                issues,
                "FINAL_PORT_CONTRACT_MISMATCH",
                "error",
                "final_port_contract",
                "A committed mating interface exceeds its physical rim or section tolerance.",
                port_ids=[edge.port_a_id, edge.port_b_id],
                expected={
                    "modeling_tolerance": state.modeling_tolerance,
                    "anti_parallel_axis_dot": PARALLEL_DOT_THRESHOLD,
                },
                actual={
                    "stored_edge": edge.model_dump(mode="json"),
                    "recomputed_interface": metrics,
                },
            )
    connection_degree = Counter(
        port_id
        for edge in state.connection_edges
        for port_id in (edge.port_a_id, edge.port_b_id)
    )
    open_ids = set(derived_open_ids)
    invalid_degrees = {
        port_id: connection_degree[port_id]
        for port_id in port_ids
        if connection_degree[port_id] != (0 if port_id in open_ids else 1)
    }
    if invalid_degrees:
        _append_issue(
            issues,
            "FINAL_GRAPH_PORT_DEGREE_MISMATCH",
            "error",
            "final_port_graph",
            "Open ports must be unconnected and every consumed port must connect exactly once.",
            expected={"open_degree": 0, "consumed_degree": 1},
            actual={"invalid_port_degrees": invalid_degrees},
        )
    coincident_open_consumed = []
    for open_port in state.open_ports:
        for port_id, port in state.port_nodes.items():
            if port_id in open_ids or port_id == open_port.id:
                continue
            if _near(open_port.position, port.position):
                coincident_open_consumed.append(
                    {
                        "open_port_id": open_port.id,
                        "consumed_port_id": port_id,
                        "position": list(open_port.position),
                    }
                )
    if coincident_open_consumed:
        _append_issue(
            issues,
            "FINAL_OPEN_PORT_REENTERS_CONSUMED_PORT",
            "error",
            "final_port_graph",
            "An open terminal coincides with a consumed graph port without an explicit connection.",
            actual={"coincident_ports": coincident_open_consumed},
        )
    if state.open_port_ids != derived_open_ids:
        _append_issue(
            issues,
            "FINAL_OPEN_PORT_GRAPH_VIEW_MISMATCH",
            "error",
            "final_port_graph",
            "The persisted open-port view disagrees with the graph-derived view.",
            expected={"open_port_ids": derived_open_ids},
            actual={"open_port_ids": state.open_port_ids},
        )

    adjacency: dict[str, set[str]] = {
        **{f"module:{module_id}": set() for module_id in module_ids},
        **{f"port:{port_id}": set() for port_id in port_ids},
    }
    invalid_edges: list[dict[str, str]] = []
    for edge in state.module_incidence_edges:
        module_node = f"module:{edge.module_id}"
        port_node = f"port:{edge.port_id}"
        if module_node not in adjacency or port_node not in adjacency:
            invalid_edges.append(
                {"kind": "incidence", "left": edge.module_id, "right": edge.port_id}
            )
            continue
        adjacency[module_node].add(port_node)
        adjacency[port_node].add(module_node)
    for edge in state.connection_edges:
        left = f"port:{edge.port_a_id}"
        right = f"port:{edge.port_b_id}"
        if left not in adjacency or right not in adjacency:
            invalid_edges.append(
                {"kind": "connection", "left": edge.port_a_id, "right": edge.port_b_id}
            )
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
    if invalid_edges:
        _append_issue(
            issues,
            "FINAL_GRAPH_DANGLING_EDGE",
            "error",
            "final_port_graph",
            "The port graph contains edges that reference missing nodes.",
            actual={"invalid_edges": invalid_edges},
        )

    if not adjacency:
        return
    unvisited = set(adjacency)
    components = 0
    while unvisited:
        components += 1
        stack = [unvisited.pop()]
        while stack:
            node = stack.pop()
            neighbors = adjacency[node] & unvisited
            unvisited.difference_update(neighbors)
            stack.extend(neighbors)
    if state.placed_modules and components != 1:
        _append_issue(
            issues,
            "FINAL_GRAPH_DISCONNECTED",
            "error",
            "final_port_graph",
            "The committed module/port graph is not one connected network.",
            expected={"connected_components": 1},
            actual={"connected_components": components},
        )

    edge_count = len(state.module_incidence_edges) + len(state.connection_edges)
    cycle_rank = edge_count - len(adjacency) + components
    expected_cycles = sum(
        1 for module in state.placed_modules if module.type == "connect_ports"
    )
    if cycle_rank != expected_cycles:
        _append_issue(
            issues,
            "FINAL_GRAPH_CYCLE_RANK_MISMATCH",
            "error",
            "final_port_graph",
            "Graph cycle rank does not match the number of explicit two-port closures.",
            expected={"cycle_rank": expected_cycles},
            actual={"cycle_rank": cycle_rank},
        )


def _validate_geometric_constraints(
    issues: list[StaticIssue],
    state: PipeState,
    step_verifications: list[StepVerification],
) -> None:
    if not state.geometric_constraints:
        return
    samples: list[tuple[tuple[float, float, float], float]] = []
    total_centerline_length = 0.0
    measured_lengths = {
        module_id: values["centerline_length"]
        for step in step_verifications
        for module_id, values in step.mcp_measurements.items()
        if "centerline_length" in values
    }
    final_bounds = next(
        (
            step.mcp_assembly_bounds
            for step in reversed(step_verifications)
            if step.transition.state_after_id == state.state_id
            and step.mcp_assembly_bounds is not None
        ),
        None,
    )
    for module in state.placed_modules:
        samples.extend(_module_spatial_samples(module))
        total_centerline_length += measured_lengths.get(
            module.id, _module_centerline_length(module)
        )

    for constraint in state.geometric_constraints:
        if constraint.type == "max_module_count":
            actual = len(state.placed_modules)
            limit = int(round(float(constraint.value or 0.0)))
            passed = actual <= limit
            actual_payload = {"module_count": actual}
            expected_payload = {"maximum_module_count": limit}
        elif constraint.type == "max_total_centerline_length":
            unverifiable = [
                module.id
                for module in state.placed_modules
                if module.params.get("path_kind") == "spline"
                and module.id not in measured_lengths
            ]
            if unverifiable:
                _append_issue(
                    issues,
                    "GEOMETRIC_CONSTRAINT_REQUIRES_FREECAD_LENGTH",
                    "error",
                    "final_geometric_constraint",
                    "Spline length requires digest-bound FreeCAD curve measurement.",
                    expected={"constraint_id": constraint.constraint_id},
                    actual={"unverified_module_ids": unverifiable},
                )
                continue
            actual = total_centerline_length
            limit = float(constraint.value or 0.0)
            passed = actual <= limit + VECTOR_TOLERANCE
            actual_payload = {"total_centerline_length": actual}
            expected_payload = {"maximum_total_centerline_length": limit}
        elif constraint.type == "max_extent":
            axis_index = {"X": 0, "Y": 1, "Z": 2}[str(constraint.axis)[-1]]
            if final_bounds is None:
                _append_issue(
                    issues,
                    "GEOMETRIC_CONSTRAINT_REQUIRES_FREECAD_BOUNDS",
                    "error",
                    "final_geometric_constraint",
                    "Physical assembly extent requires digest-bound FreeCAD BoundBox evidence.",
                    expected={"constraint_id": constraint.constraint_id},
                    actual={
                        "unverified_module_ids": [
                            module.id for module in state.placed_modules
                        ]
                    },
                )
                continue
            if final_bounds is not None:
                actual = (
                    final_bounds.maximum[axis_index] - final_bounds.minimum[axis_index]
                )
            else:
                lows = [point[axis_index] - radius for point, radius in samples]
                highs = [point[axis_index] + radius for point, radius in samples]
                actual = max(highs, default=0.0) - min(lows, default=0.0)
            limit = float(constraint.value or 0.0)
            passed = actual <= limit + VECTOR_TOLERANCE
            actual_payload = {"extent": actual, "axis": constraint.axis}
            expected_payload = {"maximum_extent": limit, "axis": constraint.axis}
        else:
            minimum = constraint.minimum or (0.0, 0.0, 0.0)
            maximum = constraint.maximum or (0.0, 0.0, 0.0)
            violations = []
            if final_bounds is None:
                _append_issue(
                    issues,
                    "GEOMETRIC_CONSTRAINT_REQUIRES_FREECAD_BOUNDS",
                    "error",
                    "final_geometric_constraint",
                    "Physical assembly bounds require digest-bound FreeCAD BoundBox evidence.",
                    expected={"constraint_id": constraint.constraint_id},
                    actual={
                        "unverified_module_ids": [
                            module.id for module in state.placed_modules
                        ]
                    },
                )
                continue
            if final_bounds is not None:
                for index, axis_name in enumerate(("X", "Y", "Z")):
                    if (
                        final_bounds.minimum[index] < minimum[index] - VECTOR_TOLERANCE
                        or final_bounds.maximum[index]
                        > maximum[index] + VECTOR_TOLERANCE
                    ):
                        violations.append(
                            {
                                "axis": axis_name,
                                "actual_minimum": final_bounds.minimum[index],
                                "actual_maximum": final_bounds.maximum[index],
                            }
                        )
            else:
                for point, radius in samples:
                    for index, axis_name in enumerate(("X", "Y", "Z")):
                        if (
                            point[index] - radius < minimum[index] - VECTOR_TOLERANCE
                            or point[index] + radius > maximum[index] + VECTOR_TOLERANCE
                        ):
                            violations.append(
                                {
                                    "axis": axis_name,
                                    "point": list(point),
                                    "radius": radius,
                                }
                            )
            passed = not violations
            actual_payload = {"violations": violations[:20]}
            expected_payload = {
                "minimum": list(minimum),
                "maximum": list(maximum),
            }
        if not passed:
            _append_issue(
                issues,
                "GEOMETRIC_CONSTRAINT_VIOLATION",
                "error",
                "final_geometric_constraint",
                f"Geometric constraint {constraint.constraint_id} was violated.",
                expected={
                    "constraint_id": constraint.constraint_id,
                    "type": constraint.type,
                    **expected_payload,
                },
                actual=actual_payload,
                suggestion={
                    "operation": "replan_constraint",
                    "constraint_id": constraint.constraint_id,
                },
            )


def _validate_final_goal_lengths(
    issues: list[StaticIssue],
    intent: IntentResult,
    state: PipeState,
    step_verifications: list[StepVerification],
) -> None:
    measured_lengths = {
        module_id: values["centerline_length"]
        for step in step_verifications
        for module_id, values in step.mcp_measurements.items()
        if "centerline_length" in values
    }
    for goal in intent.target_behavior:
        if goal.type not in {"route", "connector"} or goal.length is None:
            continue
        modules = [
            module
            for action, module in zip(state.action_history, state.placed_modules)
            if goal.goal_id in action.affected_goal_ids
        ]
        if not modules:
            continue
        if goal.type == "connector" and goal.component is not None:
            matching_components = [
                module
                for module in modules
                if module.type == "inline_component"
                and module.params.get("component_type") == goal.component
            ]
            if len(matching_components) != 1:
                # Component multiplicity is reported by the dedicated final
                # contract validator; do not double-count unrelated approach
                # geometry as the accessory's authored length.
                continue
            modules = matching_components
        unmeasured_splines = [
            module.id
            for module in modules
            if module.params.get("path_kind") == "spline"
            and module.id not in measured_lengths
        ]
        if unmeasured_splines:
            _append_issue(
                issues,
                "GOAL_LENGTH_REQUIRES_FREECAD",
                "error",
                "final_goal_length",
                "Spline route length requires digest-bound FreeCAD curve measurements.",
                module_id=unmeasured_splines[0],
                expected={"goal_id": goal.goal_id, "length": goal.length},
                actual={"unmeasured_module_ids": unmeasured_splines},
            )
            continue
        actual_length = sum(
            measured_lengths.get(module.id, _module_centerline_length(module))
            for module in modules
        )
        expected_length = float(goal.length)
        tolerance = max(VECTOR_TOLERANCE, expected_length * 1e-3)
        if abs(actual_length - expected_length) > tolerance:
            _append_issue(
                issues,
                "GOAL_LENGTH_MISMATCH",
                "error",
                "final_goal_length",
                "The completed route does not realize its required centerline length.",
                module_id=modules[-1].id,
                expected={
                    "goal_id": goal.goal_id,
                    "centerline_length": expected_length,
                    "tolerance": tolerance,
                },
                actual={"centerline_length": actual_length},
            )


def _validate_final_spline_curvature(
    issues: list[StaticIssue],
    state: PipeState,
    step_verifications: list[StepVerification],
) -> None:
    measured_radii = {
        module_id: values["minimum_curvature_radius"]
        for step in step_verifications
        for module_id, values in step.mcp_measurements.items()
        if "minimum_curvature_radius" in values
    }
    for module in state.placed_modules:
        if module.params.get("path_kind") != "spline":
            continue
        required = module.params.get("minimum_curvature_radius")
        if required is None:
            continue
        actual = measured_radii.get(module.id)
        if actual is None:
            _append_issue(
                issues,
                "SPLINE_CURVATURE_REQUIRES_FREECAD",
                "error",
                "final_spline_curvature",
                "Spline curvature requires digest-bound FreeCAD curve evidence.",
                module_id=module.id,
                expected={"minimum_curvature_radius": float(required)},
                actual={"measurement": None},
            )
        elif actual + VECTOR_TOLERANCE < float(required):
            _append_issue(
                issues,
                "SPLINE_CURVATURE_TOO_TIGHT",
                "error",
                "final_spline_curvature",
                "FreeCAD measured a spline radius below the authored minimum.",
                module_id=module.id,
                expected={"minimum_curvature_radius": float(required)},
                actual={"minimum_curvature_radius": actual},
            )


def _module_centerline_points(module: Any) -> list[tuple[float, float, float]]:
    raw_points = module.params.get("path_points")
    if isinstance(raw_points, list) and len(raw_points) >= 2:
        try:
            return [vec(point) for point in raw_points]
        except (TypeError, ValueError):
            return []
    inlet = module.ports.get("in") or module.ports.get("in_a")
    outlet = module.ports.get("out")
    if inlet is not None and outlet is not None:
        return [vec(inlet.position), vec(outlet.position)]
    return [vec(port.position) for port in module.ports.values()]


def _validate_conservative_collision(
    issues: list[StaticIssue],
    transition: StateTransition,
    before_state: PipeState,
    produced_module: Any,
) -> None:
    adjacent_bindings: dict[str, list[tuple[float, float, float]]] = {}
    for binding in produced_module.input_bindings.values():
        if "." not in binding:
            continue
        bound_port = before_state.port_nodes.get(binding)
        if bound_port is not None:
            adjacent_bindings.setdefault(binding.split(".", 1)[0], []).append(
                vec(bound_port.position)
            )
    produced_segments = _module_collision_segments(produced_module)
    collisions = []
    uncertain_collisions = []
    for existing in before_state.placed_modules:
        hit_existing = False
        for left_start, left_end, left_radius, left_label in produced_segments:
            for (
                right_start,
                right_end,
                right_radius,
                right_label,
            ) in _module_collision_segments(existing):
                if any(
                    _segment_has_endpoint(left_start, left_end, binding_position)
                    and _segment_has_endpoint(right_start, right_end, binding_position)
                    for binding_position in adjacent_bindings.get(existing.id, [])
                ):
                    # The pair incident to the shared port intentionally fuses.
                    # Other segment pairs are still checked for a later re-entry.
                    continue
                separation = _segment_distance(
                    left_start, left_end, right_start, right_end
                )
                clearance = separation - left_radius - right_radius
                if clearance < -VECTOR_TOLERANCE:
                    target = (
                        collisions
                        if _collision_envelope_reliable(produced_module)
                        and _collision_envelope_reliable(existing)
                        else uncertain_collisions
                    )
                    target.append(
                        {
                            "module_ids": [produced_module.id, existing.id],
                            "segments": [left_label, right_label],
                            "candidate_segment": [
                                list(left_start),
                                list(left_end),
                            ],
                            "existing_segment": [
                                list(right_start),
                                list(right_end),
                            ],
                            "centerline_separation": separation,
                            "required_separation": left_radius + right_radius,
                            "clearance": clearance,
                        }
                    )
                    hit_existing = True
                    break
            if hit_existing:
                break
    if collisions:
        _append_issue(
            issues,
            "STATIC_NONADJACENT_COLLISION",
            "error",
            "conservative_collision",
            "The new module's conservative solid envelope intersects a non-adjacent module.",
            transition=transition,
            module_id=produced_module.id,
            expected={"minimum_clearance": 0.0, "freecad_boolean_authoritative": True},
            actual={"collisions": collisions[:8]},
        )
    if uncertain_collisions:
        _append_issue(
            issues,
            "STATIC_COLLISION_REQUIRES_FREECAD",
            "warning",
            "conservative_collision",
            "A curved/tapered envelope may intersect; FreeCAD Boolean evidence is authoritative.",
            transition=transition,
            module_id=produced_module.id,
            actual={"candidates": uncertain_collisions[:8]},
        )


def _validate_intra_module_clearance(
    issues: list[StaticIssue],
    transition: StateTransition,
    produced_module: Any,
) -> None:
    """Flag control-polyline evidence of a possible self-overlapping tube.

    This is deliberately a broad phase: an interpolating B-spline is not its
    control polyline.  A digest-bound FreeCAD check samples the actual curve and
    is authoritative.  The broad phase makes risky freeform routes trigger that
    check and gives the LLM localized segment evidence when it fails.
    """

    if produced_module.params.get("path_kind") != "spline":
        return
    segments = _module_collision_segments(produced_module)
    if len(segments) < 3:
        return
    outer_radius = _module_envelope_radius(produced_module)
    required_separation = 2.0 * outer_radius
    minimum_local_arc_gap = math.pi * outer_radius
    segment_lengths = [length(sub(end, start)) for start, end, *_ in segments]
    cumulative = [0.0]
    for segment_length in segment_lengths:
        cumulative.append(cumulative[-1] + segment_length)

    candidates = []
    for left_index, left in enumerate(segments):
        for right_index in range(left_index + 2, len(segments)):
            right = segments[right_index]
            intervening_arc = cumulative[right_index] - cumulative[left_index + 1]
            if intervening_arc <= minimum_local_arc_gap:
                continue
            separation = _segment_distance(left[0], left[1], right[0], right[1])
            if separation + VECTOR_TOLERANCE >= required_separation:
                continue
            candidates.append(
                {
                    "segments": [left[3], right[3]],
                    "centerline_separation": separation,
                    "required_separation": required_separation,
                    "intervening_centerline_length": intervening_arc,
                }
            )
    if candidates:
        _append_issue(
            issues,
            "STATIC_SELF_CLEARANCE_REQUIRES_FREECAD",
            "warning",
            "intra_module_clearance",
            "A freeform route's non-adjacent control-polyline segments may place "
            "the tube inside its own outer diameter; actual B-spline evidence is "
            "required.",
            transition=transition,
            module_id=produced_module.id,
            expected={
                "minimum_centerline_separation": required_separation,
                "freecad_curve_sampling_authoritative": True,
            },
            actual={"candidates": candidates[:8]},
        )


def _collision_envelope_reliable(module: Any) -> bool:
    del module
    # Capsule distances are a broad-phase only: rounded end caps can overlap
    # while the actual flat-ended cylinders are disjoint. FreeCAD Boolean
    # evidence, not this approximation, decides collision validity.
    return False


def _segment_has_endpoint(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    point: tuple[float, float, float],
) -> bool:
    return _near(start, point) or _near(end, point)


def _module_collision_segments(
    module: Any,
) -> list[
    tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        float,
        str,
    ]
]:
    params = module.params
    if module.type == "junction":
        start = vec(params["start_position"])
        return [
            (
                start,
                vec(outlet["end_position"]),
                float(outlet["outer_diameter"]) / 2.0,
                f"outlet_{index}",
            )
            for index, outlet in enumerate(params.get("outlets") or [])
        ]
    if module.type == "inline_component":
        start = vec(params["start_position"])
        axis = normalize(vec(params["axis"]))
        total = float(params["length"])
        body_start = float(params["body_start_offset"])
        body_end = body_start + float(params["body_length"])
        pipe_radius = float(params["outer_diameter"]) / 2.0
        body_radius = float(params["body_outer_diameter"]) / 2.0
        segments = []
        for label, low, high, radius in (
            ("neck_in", 0.0, body_start, pipe_radius),
            ("body", body_start, body_end, body_radius),
            ("neck_out", body_end, total, pipe_radius),
        ):
            if high - low > VECTOR_TOLERANCE:
                segments.append(
                    (
                        add(start, mul(axis, low)),
                        add(start, mul(axis, high)),
                        radius,
                        label,
                    )
                )
        if params.get("actuator_axis") is not None:
            actuator_axis = normalize(vec(params["actuator_axis"]))
            center = add(
                start,
                mul(axis, body_start + float(params["body_length"]) / 2.0),
            )
            height = float(params["actuator_height"])
            origin = sub(center, mul(actuator_axis, height * 0.15))
            segments.append(
                (
                    origin,
                    add(origin, mul(actuator_axis, height * 1.15)),
                    float(params["actuator_diameter"]) / 2.0,
                    "actuator",
                )
            )
        return segments
    points = _module_centerline_points(module)
    radius = _module_envelope_radius(module)
    return [
        (left, right, radius, f"segment_{index}")
        for index, (left, right) in enumerate(zip(points, points[1:]))
        if length(sub(right, left)) > VECTOR_TOLERANCE
    ]


def _segment_distance(
    first_start: tuple[float, float, float],
    first_end: tuple[float, float, float],
    second_start: tuple[float, float, float],
    second_end: tuple[float, float, float],
) -> float:
    u = sub(first_end, first_start)
    v = sub(second_end, second_start)
    w = sub(first_start, second_start)
    a = dot(u, u)
    b = dot(u, v)
    c = dot(v, v)
    d = dot(u, w)
    e = dot(v, w)
    denominator = a * c - b * b
    degenerate_squared = 1e-24
    if a <= degenerate_squared and c <= degenerate_squared:
        return length(w)
    if a <= degenerate_squared:
        first_parameter = 0.0
        second_parameter = max(0.0, min(1.0, e / c))
    elif c <= degenerate_squared:
        second_parameter = 0.0
        first_parameter = max(0.0, min(1.0, -d / a))
    elif denominator <= 64.0 * math.ulp(1.0) * a * c:
        return min(
            _point_segment_distance(point, start, end)
            for point, start, end in (
                (first_start, second_start, second_end),
                (first_end, second_start, second_end),
                (second_start, first_start, first_end),
                (second_end, first_start, first_end),
            )
        )
    else:
        first_parameter = max(0.0, min(1.0, (b * e - c * d) / denominator))
        second_parameter = (b * first_parameter + e) / c
        if second_parameter < 0.0:
            second_parameter = 0.0
            first_parameter = max(0.0, min(1.0, -d / a))
        elif second_parameter > 1.0:
            second_parameter = 1.0
            first_parameter = max(0.0, min(1.0, (b - d) / a))
    closest_first = add(first_start, mul(u, first_parameter))
    closest_second = add(second_start, mul(v, second_parameter))
    return length(sub(closest_first, closest_second))


def _point_segment_distance(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> float:
    segment = sub(end, start)
    denominator = dot(segment, segment)
    if denominator <= 1e-12:
        return length(sub(point, start))
    parameter = max(
        0.0,
        min(1.0, dot(sub(point, start), segment) / denominator),
    )
    return length(sub(point, add(start, mul(segment, parameter))))


def _module_centerline_length(module: Any) -> float:
    raw_points = module.params.get("path_points")
    if isinstance(raw_points, list) and len(raw_points) >= 2:
        try:
            points = [vec(point) for point in raw_points]
        except (TypeError, ValueError):
            return 0.0
        path_kind = module.params.get("path_kind")
        if path_kind == "circular_arc" and len(points) >= 3:
            if (
                module.params.get("bend_radius") is not None
                and module.params.get("sweep_angle") is not None
            ):
                return float(module.params["bend_radius"]) * abs(
                    math.radians(float(module.params["sweep_angle"]))
                )
            exact_arc_length = _three_point_arc_length(
                points[0], points[len(points) // 2], points[-1]
            )
            if exact_arc_length is not None:
                return exact_arc_length
            radius = _circumradius(points[0], points[len(points) // 2], points[-1])
            if math.isfinite(radius) and radius > VECTOR_TOLERANCE:
                middle = points[len(points) // 2]
                chord_a = length(sub(middle, points[0]))
                chord_b = length(sub(points[-1], middle))
                angle_a = 2.0 * math.asin(min(1.0, chord_a / (2.0 * radius)))
                angle_b = 2.0 * math.asin(min(1.0, chord_b / (2.0 * radius)))
                return radius * (angle_a + angle_b)
        return sum(length(sub(right, left)) for left, right in zip(points, points[1:]))
    inlet = module.ports.get("in") or module.ports.get("in_a")
    if inlet is None:
        return 0.0
    outlets = [port for name, port in module.ports.items() if name.startswith("out")]
    return sum(length(sub(vec(port.position), vec(inlet.position))) for port in outlets)


def _three_point_arc_length(
    start: tuple[float, float, float],
    middle: tuple[float, float, float],
    end: tuple[float, float, float],
) -> float | None:
    a = sub(middle, start)
    b = sub(end, start)
    normal_raw = cross(a, b)
    normal_squared = dot(normal_raw, normal_raw)
    if normal_squared <= 1e-12:
        return None
    center = add(
        start,
        mul(
            add(
                mul(cross(b, normal_raw), dot(a, a)),
                mul(cross(normal_raw, a), dot(b, b)),
            ),
            1.0 / (2.0 * normal_squared),
        ),
    )
    radius_vectors = [sub(point, center) for point in (start, middle, end)]
    radius = length(radius_vectors[0])
    if radius <= VECTOR_TOLERANCE:
        return None

    def angle_sum(normal: tuple[float, float, float]) -> float:
        total = 0.0
        for left, right in zip(radius_vectors, radius_vectors[1:]):
            signed = math.atan2(
                dot(normal, cross(left, right)),
                dot(left, right),
            )
            total += signed % (2.0 * math.pi)
        return total

    normal = normalize(normal_raw)
    candidates = [angle_sum(normal), angle_sum(mul(normal, -1.0))]
    valid = [value for value in candidates if value <= 2.0 * math.pi + 1e-9]
    if not valid:
        return None
    return radius * min(valid)


def _module_spatial_samples(
    module: Any,
) -> list[tuple[tuple[float, float, float], float]]:
    radius = _module_envelope_radius(module)
    samples = [(point, radius) for point in _module_centerline_points(module)]
    if module.type == "junction" and module.params.get("start_position") is not None:
        samples.append(
            (
                vec(module.params["start_position"]),
                max(radius, float(module.params.get("max_hub_radius", 0.0))),
            )
        )
    if module.type == "terminate" and module.params.get("start_position") is not None:
        start = vec(module.params["start_position"])
        axis = normalize(vec(module.params["axis"]))
        direction = 1.0 if module.params.get("termination_type") == "cap" else -1.0
        end = add(
            start,
            mul(axis, direction * float(module.params.get("thickness", 0.0))),
        )
        samples.append((end, radius))
    if (
        module.type == "inline_component"
        and module.params.get("actuator_axis") is not None
    ):
        start = vec(module.params["start_position"])
        pipe_axis = normalize(vec(module.params["axis"]))
        actuator_axis = normalize(vec(module.params["actuator_axis"]))
        height = float(module.params["actuator_height"])
        body_start = add(
            start,
            mul(pipe_axis, float(module.params["body_start_offset"])),
        )
        center = add(
            body_start,
            mul(pipe_axis, float(module.params["body_length"]) / 2.0),
        )
        origin = sub(center, mul(actuator_axis, height * 0.15))
        end = add(origin, mul(actuator_axis, height * 1.15))
        actuator_radius = float(module.params["actuator_diameter"]) / 2.0
        samples.extend(((origin, actuator_radius), (end, actuator_radius)))
    return samples


def _module_envelope_radius(module: Any) -> float:
    values = [
        module.params.get("outer_diameter"),
        module.params.get("diameter_in"),
        module.params.get("diameter_out"),
        module.params.get("body_outer_diameter"),
        module.params.get("union_ring_outer_diameter"),
        module.params.get("actuator_diameter"),
        (
            float(module.params.get("max_hub_radius")) * 2.0
            if module.params.get("max_hub_radius") is not None
            else None
        ),
    ]
    for outlet in module.params.get("outlets", []):
        if isinstance(outlet, dict):
            values.append(outlet.get("outer_diameter"))
    numeric = [float(value) for value in values if value is not None]
    return max(numeric, default=0.0) / 2.0


def has_errors(issues: list[StaticIssue]) -> bool:
    """이슈 목록에 commit을 막는 error가 하나라도 있는지 반환한다."""

    return any(issue.severity == "error" for issue in issues)


def error_count(issues: list[StaticIssue]) -> int:
    """error 심각도의 이슈 수를 센다."""

    return sum(1 for issue in issues if issue.severity == "error")


def warning_count(issues: list[StaticIssue]) -> int:
    """warning 심각도의 이슈 수를 센다."""

    return sum(1 for issue in issues if issue.severity == "warning")


def top_issue_ids(issues: list[StaticIssue], limit: int = 5) -> list[str]:
    """보고서에 표시할 상위 error 식별자를 제한된 개수로 반환한다."""

    return [issue.issue_id for issue in issues if issue.severity == "error"][:limit]


def _validate_module_connection(
    issues: list[StaticIssue],
    transition: StateTransition,
    action: ResolvedAction,
    target_port: Port,
    produced_module: Any,
    tolerance: float,
) -> None:
    in_port = produced_module.ports.get("in") or produced_module.ports.get("in_a")
    if in_port is None:
        _append_issue(
            issues,
            "MODULE_INPUT_PORT_MISSING",
            "error",
            "module_connection",
            "Produced module has no input port.",
            transition=transition,
            module_id=produced_module.id,
            expected={"port": "in"},
            actual={"ports": list(produced_module.ports)},
        )
        return

    if not _near(in_port.position, target_port.position, tolerance):
        _append_issue(
            issues,
            "MODULE_INPUT_POSITION_MISMATCH",
            "error",
            "module_connection",
            "Produced module input position does not match target port position.",
            transition=transition,
            module_id=produced_module.id,
            port_ids=[in_port.id, target_port.id],
            expected={"target_position": list(target_port.position)},
            actual={"input_position": list(in_port.position)},
            suggestion={
                "operation": "move_module_start",
                "target_port": target_port.id,
            },
        )

    expected_in_axis = tuple(-value for value in target_port.axis)
    axis_alignment = dot(
        normalize(vec(in_port.axis)),
        normalize(vec(expected_in_axis)),
    )
    outer_rim_error = circular_rim_mismatch(
        0.0,
        in_port.outer_diameter / 2.0,
        target_port.outer_diameter / 2.0,
        axis_alignment,
    )
    if axis_alignment < PARALLEL_DOT_THRESHOLD or outer_rim_error > tolerance:
        _append_issue(
            issues,
            "MODULE_INPUT_AXIS_MISMATCH",
            "error",
            "module_connection",
            "Produced module input axis is not opposite the target port axis.",
            transition=transition,
            module_id=produced_module.id,
            port_ids=[in_port.id, target_port.id],
            expected={"input_axis": list(expected_in_axis)},
            actual={
                "input_axis": list(in_port.axis),
                "axis_alignment": axis_alignment,
                "outer_rim_error": outer_rim_error,
                "modeling_tolerance": tolerance,
            },
            suggestion={
                "operation": "align_module_axis",
                "target_port": target_port.id,
            },
        )

    if action.target_port not in transition.removed_port_ids:
        _append_issue(
            issues,
            "TARGET_PORT_NOT_CONSUMED",
            "error",
            "open_port_transition",
            "Target port should be consumed by the applied action.",
            transition=transition,
            module_id=produced_module.id,
            target_port_id=action.target_port,
            expected={"removed_port_ids_contains": action.target_port},
            actual={"removed_port_ids": transition.removed_port_ids},
        )


def _validate_graph_transition(
    issues: list[StaticIssue],
    transition: StateTransition,
    action: ResolvedAction,
    before_state: PipeState,
    after_state: PipeState,
    produced_module: Any,
) -> None:
    derived_open_ids = [port.id for port in after_state.open_ports]
    if after_state.open_port_ids != derived_open_ids:
        _append_issue(
            issues,
            "OPEN_PORT_GRAPH_VIEW_MISMATCH",
            "error",
            "port_graph_consistency",
            "open_port_ids does not match the derived open_ports view.",
            transition=transition,
            expected={"open_port_ids": derived_open_ids},
            actual={"open_port_ids": after_state.open_port_ids},
        )
    consumed = list(action.consumed_port_ids or [action.target_port])
    if len(consumed) != len(set(consumed)):
        _append_issue(
            issues,
            "DUPLICATE_CONSUMED_PORT",
            "error",
            "port_graph_transition",
            "An action cannot consume the same open port twice.",
            transition=transition,
            port_ids=consumed,
        )
    missing_removed = sorted(set(consumed) - set(transition.removed_port_ids))
    if missing_removed:
        _append_issue(
            issues,
            "CONSUMED_PORT_NOT_REMOVED",
            "error",
            "port_graph_transition",
            "Every consumed port must leave the open-port set.",
            transition=transition,
            port_ids=missing_removed,
            expected={"removed": consumed},
            actual={"removed": transition.removed_port_ids},
        )
    incidence_ids = {
        edge.port_id
        for edge in after_state.module_incidence_edges
        if edge.module_id == produced_module.id
    }
    module_port_ids = {port.id for port in produced_module.ports.values()}
    if incidence_ids != module_port_ids:
        _append_issue(
            issues,
            "MODULE_INCIDENCE_MISMATCH",
            "error",
            "port_graph_incidence",
            "Module incidence edges do not cover exactly the module ports.",
            transition=transition,
            module_id=produced_module.id,
            expected={"port_ids": sorted(module_port_ids)},
            actual={"port_ids": sorted(incidence_ids)},
        )
    new_edges = [
        edge
        for edge in after_state.connection_edges
        if edge.edge_id in set(transition.connection_edge_ids)
    ]
    start_anchor_bootstrap = _is_start_anchor_bootstrap_transition(
        before_state,
        after_state,
        action,
        produced_module,
    )
    start_anchor_bootstrap_required = bool(
        before_state.state_version == 0
        and action.target_port == "START"
        and any(
            goal.type == "connect" and goal.connection_target == "start_anchor"
            for goal in before_state.remaining_goals
        )
    )
    if start_anchor_bootstrap_required and not start_anchor_bootstrap:
        _append_issue(
            issues,
            "START_ANCHOR_BOOTSTRAP_MISMATCH",
            "error",
            "port_graph_transition",
            "A closed-loop contract must replace virtual START with the first module inlet as its reserved seam anchor.",
            transition=transition,
            module_id=produced_module.id,
            expected={
                "reserved_start_anchor": produced_module.ports.get("in").id
                if produced_module.ports.get("in") is not None
                else "module.in",
                "start_connection_edge": False,
            },
            actual={
                "reserved_start_anchor": (
                    after_state.reserved_start_anchor.id
                    if after_state.reserved_start_anchor is not None
                    else None
                ),
                "input_bindings": produced_module.input_bindings,
                "START_in_port_nodes": "START" in after_state.port_nodes,
            },
        )
    expected_new_edges = 0 if start_anchor_bootstrap_required else len(consumed)
    if len(new_edges) != expected_new_edges:
        _append_issue(
            issues,
            "CONNECTION_EDGE_COUNT_MISMATCH",
            "error",
            "port_graph_connection",
            "Each consumed physical port must create one mating connection edge; "
            "the construction-only START cursor is replaced by the reserved first inlet.",
            transition=transition,
            expected={"new_connection_edges": expected_new_edges},
            actual={"new_connection_edges": len(new_edges)},
        )
    for produced_port_id in transition.produced_port_ids:
        produced_port = after_state.port_nodes.get(produced_port_id)
        if produced_port is None:
            continue
        collisions = [
            port.id
            for port in before_state.port_nodes.values()
            if _near(produced_port.position, port.position)
        ]
        if collisions:
            _append_issue(
                issues,
                "OPEN_PORT_REENTERS_EXISTING_PORT",
                "error",
                "port_graph_transition",
                "A new open terminal coincides with an existing graph port without connect_ports.",
                transition=transition,
                port_ids=[produced_port_id, *collisions],
                actual={"position": list(produced_port.position)},
                suggestion={"operation": "use_connect_ports_or_replan_route"},
            )
    for edge in new_edges:
        left = after_state.port_nodes.get(edge.port_a_id)
        right = after_state.port_nodes.get(edge.port_b_id)
        if left is None or right is None:
            continue
        metrics = _connection_interface_metrics(left, right)
        if _connection_contract_invalid(
            edge,
            metrics,
            after_state.modeling_tolerance,
        ):
            _append_issue(
                issues,
                "PORT_CONTRACT_MISMATCH",
                "error",
                "port_contract",
                "Mating ports violate position, axis, or section compatibility.",
                transition=transition,
                port_ids=[edge.port_a_id, edge.port_b_id],
                expected={
                    "position_tolerance": after_state.modeling_tolerance,
                    "anti_parallel_axis_dot": PARALLEL_DOT_THRESHOLD,
                    "section_tolerance": after_state.modeling_tolerance,
                    "maximum_rim_error": after_state.modeling_tolerance,
                },
                actual={
                    "stored_edge": edge.model_dump(mode="json"),
                    "recomputed_interface": metrics,
                },
            )
    if action.module == "connect_ports":
        if len(consumed) != 2 or transition.produced_port_ids:
            _append_issue(
                issues,
                "CONNECT_PORTS_TOPOLOGY_MISMATCH",
                "error",
                "connect_ports_topology",
                "connect_ports must consume two distinct ports and produce no open port.",
                transition=transition,
                expected={"consumed": 2, "produced": 0},
                actual={
                    "consumed": len(consumed),
                    "produced": len(transition.produced_port_ids),
                },
            )


def _is_start_anchor_bootstrap_transition(
    before_state: PipeState,
    after_state: PipeState,
    action: ResolvedAction,
    produced_module: Any,
) -> bool:
    """Recognize the one legal replacement of virtual START by a real inlet."""

    inlet = produced_module.ports.get("in")
    anchor = after_state.reserved_start_anchor
    return bool(
        before_state.state_version == 0
        and action.target_port == "START"
        and before_state.reserved_start_anchor is None
        and inlet is not None
        and anchor is not None
        and anchor.id == inlet.id
        and "START" not in after_state.port_nodes
        and "START" not in produced_module.input_bindings.values()
        and any(
            goal.type == "connect" and goal.connection_target == "start_anchor"
            for goal in before_state.remaining_goals
        )
    )


def _validate_route_continuity(
    issues: list[StaticIssue],
    transition: StateTransition,
    action: ResolvedAction,
    before_state: PipeState,
    target_port: Port,
    produced_module: Any,
) -> None:
    if action.module not in {"route", "connect_ports"}:
        return
    if action.module == "connect_ports" and action.params.get("path_kind") == "seam":
        # A seam is a topology-only closure between already coincident,
        # tangent-compatible physical ports. Registry validation proves those
        # predicates; there is intentionally no zero-length centerline edge.
        return
    points = action.params.get("path_points") or produced_module.params.get(
        "path_points"
    )
    if not isinstance(points, list) or len(points) < 2:
        _append_issue(
            issues,
            "ROUTE_PATH_MISSING",
            "error",
            "route_continuity",
            "A route action must resolve to at least two centerline points.",
            transition=transition,
        )
        return
    tolerance = before_state.modeling_tolerance
    coincident_segments = [
        index
        for index, (left, right) in enumerate(zip(points, points[1:]), start=1)
        if length(sub(vec(right), vec(left))) <= tolerance
    ]
    if coincident_segments:
        _append_issue(
            issues,
            "ROUTE_DEGENERATE_SEGMENT",
            "error",
            "route_continuity",
            "Route centerline contains coincident consecutive points.",
            transition=transition,
            expected={"minimum_segment_length": tolerance},
            actual={"segment_indices": coincident_segments},
        )
        return
    path_kind = action.params.get("path_kind")
    arc_tangents = None
    if path_kind == "circular_arc":
        arc_tangents = (
            _analytic_route_arc_tangents(action.params)
            if action.module == "route"
            else _arc_endpoint_tangents(points)
        )
    if path_kind == "circular_arc" and arc_tangents is None:
        _append_issue(
            issues,
            "ROUTE_ARC_DEGENERATE",
            "error",
            "route_continuity",
            "A circular arc requires three non-collinear centerline points.",
            transition=transition,
            actual={"path_points": [list(vec(point)) for point in points]},
        )
        return
    if arc_tangents is not None:
        start_tangent = arc_tangents[0]
    elif path_kind == "spline" and action.params.get("initial_tangent") is not None:
        start_tangent = normalize(vec(action.params["initial_tangent"]))
    else:
        start_tangent = normalize(sub(vec(points[1]), vec(points[0])))
    start_dot = dot(start_tangent, normalize(vec(target_port.axis)))
    start_rim_error = circular_rim_mismatch(
        0.0,
        target_port.outer_diameter / 2.0,
        target_port.outer_diameter / 2.0,
        start_dot,
    )
    if start_dot < PARALLEL_DOT_THRESHOLD or start_rim_error > tolerance:
        _append_issue(
            issues,
            "ROUTE_START_TANGENT_MISMATCH",
            "error",
            "route_continuity",
            "Route centerline is not tangent to the selected target port.",
            transition=transition,
            port_ids=[target_port.id],
            expected={
                "dot": PARALLEL_DOT_THRESHOLD,
                "maximum_rim_error": tolerance,
            },
            actual={
                "dot": round(start_dot, 12),
                "outer_rim_error": start_rim_error,
            },
        )
    if arc_tangents is not None:
        end_tangent = arc_tangents[1]
    elif path_kind == "spline" and action.params.get("final_tangent") is not None:
        end_tangent = normalize(vec(action.params["final_tangent"]))
    else:
        end_tangent = normalize(sub(vec(points[-1]), vec(points[-2])))
    out_port = produced_module.ports.get("out")
    if out_port is not None:
        end_dot = dot(end_tangent, normalize(vec(out_port.axis)))
        end_rim_error = circular_rim_mismatch(
            0.0,
            out_port.outer_diameter / 2.0,
            out_port.outer_diameter / 2.0,
            end_dot,
        )
        if end_dot < PARALLEL_DOT_THRESHOLD or end_rim_error > tolerance:
            _append_issue(
                issues,
                "ROUTE_END_TANGENT_MISMATCH",
                "error",
                "route_continuity",
                "Route centerline terminal tangent does not match its output port.",
                transition=transition,
                port_ids=[out_port.id],
                expected={
                    "dot": PARALLEL_DOT_THRESHOLD,
                    "maximum_rim_error": tolerance,
                },
                actual={
                    "dot": round(end_dot, 12),
                    "outer_rim_error": end_rim_error,
                },
            )
    if action.module == "connect_ports":
        other_id = action.params.get("other_port_id")
        other_port = _find_connectable_port(str(other_id), before_state)
        if other_port is not None:
            expected_end = tuple(-value for value in other_port.axis)
            end_dot = dot(end_tangent, normalize(vec(expected_end)))
            end_rim_error = circular_rim_mismatch(
                0.0,
                other_port.outer_diameter / 2.0,
                other_port.outer_diameter / 2.0,
                end_dot,
            )
            if end_dot < PARALLEL_DOT_THRESHOLD or end_rim_error > tolerance:
                _append_issue(
                    issues,
                    "CONNECT_END_TANGENT_MISMATCH",
                    "error",
                    "route_continuity",
                    "connect_ports must enter the second open port opposite its outward axis.",
                    transition=transition,
                    port_ids=[other_port.id],
                    expected={
                        "dot": PARALLEL_DOT_THRESHOLD,
                        "maximum_rim_error": tolerance,
                    },
                    actual={
                        "dot": round(end_dot, 12),
                        "outer_rim_error": end_rim_error,
                    },
                )
        if arc_tangents is not None:
            for label, authored, derived in (
                (
                    "initial_tangent",
                    action.params.get("initial_tangent"),
                    start_tangent,
                ),
                ("final_tangent", action.params.get("final_tangent"), end_tangent),
            ):
                authored_dot = (
                    dot(normalize(vec(authored)), derived)
                    if authored is not None
                    else 1.0
                )
                authored_rim_error = circular_rim_mismatch(
                    0.0,
                    target_port.outer_diameter / 2.0,
                    target_port.outer_diameter / 2.0,
                    authored_dot,
                )
                if authored is not None and (
                    authored_dot < PARALLEL_DOT_THRESHOLD
                    or authored_rim_error > tolerance
                ):
                    _append_issue(
                        issues,
                        "CONNECT_ARC_TANGENT_MISMATCH",
                        "error",
                        "route_continuity",
                        "Authored connect_ports arc tangent disagrees with its three-point arc.",
                        transition=transition,
                        expected={"parameter": label, "dot": PARALLEL_DOT_THRESHOLD},
                        actual={
                            "parameter": label,
                            "dot": authored_dot,
                            "outer_rim_error": authored_rim_error,
                        },
                    )
    # Waypoint circumcircles are not a sound bound for an interpolating
    # B-spline's curvature.  Circular arcs have an exact static radius; spline
    # curvature is checked from the digest-bound FreeCAD edge at final review.
    if path_kind != "circular_arc":
        return
    minimum_required = action.params.get("minimum_curvature_radius")
    if minimum_required is None or len(points) < 3:
        return
    minimum_actual = min(
        (
            _circumradius(vec(a), vec(b), vec(c))
            for a, b, c in zip(points, points[1:], points[2:])
        ),
        default=float("inf"),
    )
    if minimum_actual + VECTOR_TOLERANCE < float(minimum_required):
        _append_issue(
            issues,
            "ROUTE_CURVATURE_TOO_TIGHT",
            "error",
            "route_curvature",
            "Resolved route violates its LLM-authored minimum curvature radius.",
            transition=transition,
            expected={"minimum_curvature_radius": float(minimum_required)},
            actual={"minimum_sampled_radius": minimum_actual},
        )


def _circumradius(a: Any, b: Any, c: Any) -> float:
    ab = length(sub(b, a))
    bc = length(sub(c, b))
    ca = length(sub(a, c))
    cross_size = length(
        (
            (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
            (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
        )
    )
    if cross_size <= 1e-9:
        return float("inf")
    return (ab * bc * ca) / (2.0 * cross_size)


def _arc_endpoint_tangents(
    points: list[Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if len(points) < 3:
        return None
    start = vec(points[0])
    middle = vec(points[len(points) // 2])
    end = vec(points[-1])
    a = sub(middle, start)
    b = sub(end, start)
    normal_raw = cross(a, b)
    normal_squared = dot(normal_raw, normal_raw)
    if normal_squared <= 1e-12:
        return None
    numerator = add(
        mul(cross(b, normal_raw), dot(a, a)),
        mul(cross(normal_raw, a), dot(b, b)),
    )
    center = add(start, mul(numerator, 1.0 / (2.0 * normal_squared)))
    radius_vectors = [sub(point, center) for point in (start, middle, end)]

    def directed_sweep(normal: tuple[float, float, float]) -> float:
        total = 0.0
        for left, right in zip(radius_vectors, radius_vectors[1:]):
            angle = math.atan2(
                dot(normal, cross(left, right)),
                dot(left, right),
            )
            total += angle % math.tau
        return total

    base_normal = normalize(normal_raw)
    candidates = [base_normal, mul(base_normal, -1.0)]
    valid = [
        (directed_sweep(candidate), candidate)
        for candidate in candidates
        if directed_sweep(candidate) <= math.tau + 1e-9
    ]
    if not valid:
        return None
    _, normal = min(valid, key=lambda item: item[0])
    start_tangent = normalize(cross(normal, radius_vectors[0]))
    end_tangent = normalize(cross(normal, radius_vectors[-1]))
    return start_tangent, end_tangent


def _analytic_route_arc_tangents(
    params: dict[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Return the resolver-owned analytic endpoint tangents for a route arc."""

    try:
        _normal, start_tangent, end_tangent = canonical_circular_arc_frame(
            vec(params["axis"]),
            vec(params["plane_normal"]),
            float(params["sweep_angle"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return start_tangent, end_tangent


def _validate_branch_goal_direction(
    issues: list[StaticIssue],
    transition: StateTransition,
    action: ResolvedAction,
    target_port: Port,
    produced_module: Any,
    goal: dict[str, Any],
) -> None:
    if goal.get("type") != "branch":
        return
    topology_outputs = [
        port_name
        for port_name in produced_module.ports
        if _port_role(port_name) in {"primary_outlet", "branch_outlet"}
    ]
    if len(topology_outputs) < 2:
        _append_issue(
            issues,
            "BRANCH_TOPOLOGY_NOT_PRODUCED",
            "error",
            "branch_topology_compatibility",
            "The chosen action did not produce the multi-outlet topology claimed for the branch goal.",
            transition=transition,
            module_id=produced_module.id,
            expected={"minimum_output_count": 2},
            actual={"module": action.module, "output_count": len(topology_outputs)},
        )
        return
    branch_count = int(
        goal.get("branch_count")
        or len(goal.get("required_outlets") or [])
        or len(goal.get("required_outlet_vectors") or [])
        or len(goal.get("required_outlet_directions") or [])
        or action.params.get("branch_count")
        or 2
    )
    include_primary = _include_primary_outlet(action.params, goal)
    if produced_module.type == "junction":
        outlet_roles = [
            outlet.get("role")
            for outlet in (produced_module.params.get("outlets") or [])
            if isinstance(outlet, dict)
        ]
        expected_primary_count = int(include_primary)
        actual_primary_count = outlet_roles.count("primary")
        actual_branch_count = outlet_roles.count("branch")
        if (
            actual_primary_count != expected_primary_count
            or actual_branch_count != branch_count
            or len(outlet_roles) != expected_primary_count + branch_count
        ):
            _append_issue(
                issues,
                "JUNCTION_OUTLET_ROLE_MISMATCH",
                "error",
                "junction_outlet_role_contract",
                "Junction outlet roles do not match the immutable branch goal.",
                transition=transition,
                module_id=produced_module.id,
                expected={
                    "primary_role_count": expected_primary_count,
                    "branch_role_count": branch_count,
                    "include_primary_outlet": include_primary,
                },
                actual={
                    "outlet_roles": outlet_roles,
                    "primary_role_count": actual_primary_count,
                    "branch_role_count": actual_branch_count,
                },
                suggestion={
                    "operation": "repair_junction_outlet_roles",
                    "primary_role_count": expected_primary_count,
                    "branch_role_count": branch_count,
                },
            )
    branch_ports = [
        port
        for port_name, port in produced_module.ports.items()
        if _port_role(port_name) == "branch_outlet"
    ]
    primary_port = produced_module.ports.get("out")
    expected_output_count = branch_count + (1 if include_primary else 0)
    actual_output_count = len(
        [
            port_name
            for port_name in produced_module.ports
            if _port_role(port_name) in {"primary_outlet", "branch_outlet"}
        ]
    )
    if actual_output_count != expected_output_count:
        _append_issue(
            issues,
            "JUNCTION_OUTPUT_COUNT_MISMATCH",
            "error",
            "junction_output_count",
            "Junction produced a different number of terminal outputs than requested.",
            transition=transition,
            module_id=produced_module.id,
            port_ids=[port.id for port in branch_ports]
            + ([primary_port.id] if primary_port else []),
            expected={
                "branch_count": branch_count,
                "include_primary_outlet": include_primary,
                "produced_open_ports": expected_output_count,
            },
            actual={"produced_open_ports": actual_output_count},
            suggestion={"operation": "repair_junction_output_count"},
        )
    expected_after_open_count = (
        len(transition.open_port_ids_before) - 1 + expected_output_count
    )
    if len(transition.open_port_ids_after) != expected_after_open_count:
        _append_issue(
            issues,
            "OPEN_PORT_DELTA_MISMATCH",
            "error",
            "junction_open_port_delta",
            "Junction open-port delta does not match consumed target and produced outputs.",
            transition=transition,
            module_id=produced_module.id,
            expected={
                "after_open_count": expected_after_open_count,
                "formula": "before_open_count - 1 + produced_open_ports",
            },
            actual={
                "before_open_count": len(transition.open_port_ids_before),
                "after_open_count": len(transition.open_port_ids_after),
                "removed_port_ids": transition.removed_port_ids,
                "produced_port_ids": transition.produced_port_ids,
            },
            suggestion={"operation": "repair_open_port_transition"},
        )
    if include_primary is False and primary_port is not None:
        _append_issue(
            issues,
            "UNEXPECTED_PRIMARY_OUTLET",
            "error",
            "junction_primary_outlet_policy",
            "Junction retained a primary outlet even though the branch goal defines explicit terminals.",
            transition=transition,
            module_id=produced_module.id,
            port_ids=[primary_port.id],
            expected={"include_primary_outlet": False, "primary_outlet": None},
            actual={
                "primary_outlet": primary_port.id,
                "axis": list(primary_port.axis),
                "position": list(primary_port.position),
            },
            suggestion={"operation": "remove_unexpected_primary_outlet"},
        )

    if goal.get("length") is not None:
        expected_length = float(goal["length"])
        tolerance = max(VECTOR_TOLERANCE, expected_length * 1e-3)
        wrong_lengths = [
            {
                "port_id": port.id,
                "length": length(sub(vec(port.position), vec(target_port.position))),
            }
            for port in branch_ports
            if abs(
                length(sub(vec(port.position), vec(target_port.position)))
                - expected_length
            )
            > tolerance
        ]
        if wrong_lengths:
            _append_issue(
                issues,
                "BRANCH_LENGTH_MISMATCH",
                "error",
                "branch_length",
                "One or more branch arms have the wrong authored length.",
                transition=transition,
                module_id=produced_module.id,
                expected={"length": expected_length, "tolerance": tolerance},
                actual={"wrong_lengths": wrong_lengths},
            )

    required_branch_od = goal.get("branch_outer_diameter")
    required_branch_wall = goal.get("branch_wall_thickness")
    if required_branch_od is not None or required_branch_wall is not None:
        wrong_sections = []
        for port in branch_ports:
            if (
                required_branch_od is not None
                and abs(port.outer_diameter - float(required_branch_od))
                > VECTOR_TOLERANCE
            ) or (
                required_branch_wall is not None
                and abs(port.wall_thickness - float(required_branch_wall))
                > VECTOR_TOLERANCE
            ):
                wrong_sections.append(
                    {
                        "port_id": port.id,
                        "outer_diameter": port.outer_diameter,
                        "wall_thickness": port.wall_thickness,
                    }
                )
        if wrong_sections:
            _append_issue(
                issues,
                "BRANCH_SECTION_MISMATCH",
                "error",
                "branch_section",
                "One or more branch outlets have the wrong section.",
                transition=transition,
                module_id=produced_module.id,
                expected={
                    "outer_diameter": required_branch_od,
                    "wall_thickness": required_branch_wall,
                },
                actual={"wrong_sections": wrong_sections},
            )

    branch_angles = [float(value) for value in goal.get("branch_angles") or []]
    if branch_angles:
        normal_raw = goal.get("branch_plane_normal")
        normal = normalize(vec(normal_raw)) if normal_raw is not None else None
        inlet_axis = normalize(vec(target_port.axis))
        actual_angles = []
        out_of_plane = []
        if normal is not None and abs(dot(normal, inlet_axis)) > 1e-3:
            out_of_plane.append(
                {"port_id": target_port.id, "normal_dot": dot(normal, inlet_axis)}
            )
        for port in branch_ports:
            outlet_axis = normalize(vec(port.axis))
            cosine = max(-1.0, min(1.0, dot(inlet_axis, outlet_axis)))
            if normal is None:
                directed_angle = math.degrees(math.acos(cosine))
                # 일반적인 Y/branch 각도는 유동 방향 화살표가 아니라 main
                # centerline과 branch centerline 사이의 작은 선각이다. START
                # 쪽 junction처럼 outlet 축이 inlet 진행축의 반대쪽을 향하면
                # directed angle은 135°지만 설계 branch angle은 45°다.
                actual_angles.append(min(directed_angle, 180.0 - directed_angle))
            else:
                plane_dot = dot(normal, outlet_axis)
                if abs(plane_dot) > 1e-3:
                    out_of_plane.append({"port_id": port.id, "normal_dot": plane_dot})
                sine = dot(normal, cross(inlet_axis, outlet_axis))
                actual_angles.append(math.degrees(math.atan2(sine, cosine)))
        if out_of_plane:
            _append_issue(
                issues,
                "BRANCH_PLANE_MISMATCH",
                "error",
                "branch_angle_assignment",
                "The inlet and branch outlet axes must lie in the authored branch plane.",
                transition=transition,
                module_id=produced_module.id,
                expected={"branch_plane_normal": normal_raw, "max_abs_dot": 1e-3},
                actual={"out_of_plane": out_of_plane},
            )
        expected_angles = (
            branch_angles
            if normal is not None
            else [abs(value) for value in branch_angles]
        )
        available = list(enumerate(actual_angles))
        mismatches = []
        tolerance = 0.5
        for expected_angle in expected_angles:
            ranked = sorted(
                available,
                key=lambda item: abs(item[1] - expected_angle),
            )
            if not ranked or abs(ranked[0][1] - expected_angle) > tolerance:
                mismatches.append(expected_angle)
                continue
            available = [item for item in available if item[0] != ranked[0][0]]
        if mismatches or len(actual_angles) != len(expected_angles):
            _append_issue(
                issues,
                "BRANCH_ANGLE_MISMATCH",
                "error",
                "branch_angle_assignment",
                "Branch outlet angles do not match the distinct authored angle set.",
                transition=transition,
                module_id=produced_module.id,
                expected={"branch_angles": branch_angles, "tolerance_deg": tolerance},
                actual={"branch_angles": actual_angles, "unmatched": mismatches},
            )

    expected_style = goal.get("junction_style")
    if expected_style is not None:
        actual_style = action.params.get("blend_mode")
        if actual_style is None:
            actual_style = action.params.get("junction_style")
        expected_modes = {
            "smooth_hub": {"fillet", "smooth_hub"},
            "hard_fuse": {"hard", "hard_fuse"},
        }[expected_style]
        if actual_style not in expected_modes:
            _append_issue(
                issues,
                "BRANCH_STYLE_MISMATCH",
                "error",
                "branch_blend_style",
                "Junction blend geometry does not match the branch style contract.",
                transition=transition,
                module_id=produced_module.id,
                expected={"junction_style": expected_style},
                actual={"blend_mode": actual_style},
            )

    required_vectors = _required_outlet_vectors(action.params, goal)
    if required_vectors:
        assignment = _match_vectors_to_ports(required_vectors, branch_ports)
        required_outlets = list(goal.get("required_outlets") or [])
        if required_outlets:
            ports_by_id = {port.id: port for port in branch_ports}
            detail_failures = []
            for match in assignment["matched"]:
                index = int(match["vector_index"])
                if index >= len(required_outlets):
                    continue
                contract = required_outlets[index]
                port = ports_by_id[match["port_id"]]
                actual_length = length(
                    sub(vec(port.position), vec(target_port.position))
                )
                for field_name, actual_value in (
                    ("length", actual_length),
                    ("outer_diameter", port.outer_diameter),
                    ("wall_thickness", port.wall_thickness),
                ):
                    expected_value = contract.get(field_name)
                    if (
                        expected_value is not None
                        and abs(float(actual_value) - float(expected_value))
                        > VECTOR_TOLERANCE
                    ):
                        detail_failures.append(
                            {
                                "outlet_index": index,
                                "port_id": port.id,
                                "field": field_name,
                                "expected": expected_value,
                                "actual": actual_value,
                            }
                        )
            if detail_failures:
                _append_issue(
                    issues,
                    "BRANCH_OUTLET_DETAIL_MISMATCH",
                    "error",
                    "branch_outlet_contract",
                    "One or more distinct outlet dimensions differ from the immutable goal.",
                    transition=transition,
                    module_id=produced_module.id,
                    expected={"required_outlets": required_outlets},
                    actual={"failures": detail_failures},
                )
        if assignment["missing_vectors"]:
            _append_issue(
                issues,
                "BRANCH_VECTOR_MISMATCH",
                "error",
                "branch_vector_assignment",
                "Branch goal vectors are not represented by distinct branch outlet ports.",
                transition=transition,
                module_id=produced_module.id,
                port_ids=[port.id for port in branch_ports],
                target_port_id=target_port.id,
                expected={
                    "expected_vectors": _vectors_json(required_vectors),
                    "threshold": EXPLICIT_VECTOR_DOT_THRESHOLD,
                    "distinct_ports": True,
                },
                actual=assignment,
                suggestion={
                    "operation": "assign_required_outlet_vectors",
                    "required_outlet_vectors": _vectors_json(required_vectors),
                    "target_port": target_port.id,
                },
            )
        return

    explicit_directions = list(goal.get("required_outlet_directions") or [])
    single_direction = goal.get("direction") if not explicit_directions else None
    required_directions = explicit_directions or (
        [single_direction] if single_direction else []
    )
    if not required_directions:
        return
    required_count = len(required_directions) if explicit_directions else branch_count
    matched: list[str] = []
    missing_directions: list[str] = []
    rejected: list[dict[str, Any]] = []
    if explicit_directions:
        available_ports = list(branch_ports)
        for direction in required_directions:
            best_port: Port | None = None
            best_score = -1.0
            for port in available_ports:
                score = _direction_score(target_port, port, direction)
                if score > best_score:
                    best_score = score
                    best_port = port
            if best_port is not None and best_score >= EXPLICIT_VECTOR_DOT_THRESHOLD:
                matched.append(best_port.id)
                available_ports = [
                    port for port in available_ports if port.id != best_port.id
                ]
            else:
                missing_directions.append(direction)
        for port in branch_ports:
            scores = [
                _direction_score(target_port, port, direction)
                for direction in required_directions
            ]
            best_score = max(scores) if scores else -1.0
            if port.id not in matched:
                rejected.append(
                    {
                        "port_id": port.id,
                        "position": list(port.position),
                        "axis": list(port.axis),
                        "best_dot": round(best_score, 4),
                    }
                )
    else:
        for port in branch_ports:
            scores = [
                _direction_score(target_port, port, direction)
                for direction in required_directions
                if direction is not None
            ]
            best_score = max(scores) if scores else -1.0
            if best_score >= BRANCH_DIRECTION_DOT_THRESHOLD:
                matched.append(port.id)
            else:
                rejected.append(
                    {
                        "port_id": port.id,
                        "position": list(port.position),
                        "axis": list(port.axis),
                        "best_dot": round(best_score, 4),
                    }
                )

    if len(matched) < required_count or missing_directions:
        _append_issue(
            issues,
            "BRANCH_DIRECTION_MISMATCH",
            "error",
            "branch_direction",
            "Branch goal direction is not represented by enough branch outlet ports.",
            transition=transition,
            module_id=produced_module.id,
            port_ids=[port.id for port in branch_ports],
            target_port_id=target_port.id,
            expected={
                "required_directions": required_directions,
                "required_match_count": required_count,
                "threshold": (
                    EXPLICIT_VECTOR_DOT_THRESHOLD
                    if explicit_directions
                    else BRANCH_DIRECTION_DOT_THRESHOLD
                ),
            },
            actual={
                "matched_port_ids": matched,
                "matched_count": len(matched),
                "missing_directions": missing_directions,
                "rejected_ports": rejected,
            },
            suggestion={
                "operation": "adjust_junction_branch_direction",
                "direction": required_directions[0],
                "target_port": target_port.id,
            },
        )


def _validate_move_goal_direction(
    issues: list[StaticIssue],
    transition: StateTransition,
    target_port: Port,
    produced_module: Any,
    goal: dict[str, Any],
) -> None:
    if goal.get("type") != "move" or not goal.get("direction"):
        return
    out_port = produced_module.ports.get("out")
    if out_port is None:
        return
    score = _direction_score(target_port, out_port, goal["direction"])
    if score < PARALLEL_DOT_THRESHOLD:
        _append_issue(
            issues,
            "MOVE_DIRECTION_MISMATCH",
            "error",
            "move_direction",
            "Move goal direction is not represented by the produced outlet.",
            transition=transition,
            module_id=produced_module.id,
            port_ids=[out_port.id],
            target_port_id=target_port.id,
            expected={
                "direction": goal["direction"],
                "threshold": PARALLEL_DOT_THRESHOLD,
            },
            actual={
                "out_port": out_port.id,
                "position": list(out_port.position),
                "axis": list(out_port.axis),
                "dot": round(score, 4),
            },
            suggestion={
                "operation": "adjust_move_direction",
                "direction": goal["direction"],
            },
        )


def _validate_turn_goal_direction(
    issues: list[StaticIssue],
    transition: StateTransition,
    produced_module: Any,
    goal: dict[str, Any],
) -> None:
    if goal.get("type") != "turn" or not goal.get("direction"):
        return
    out_port = produced_module.ports.get("out")
    if out_port is None:
        return
    direction_vector = direction_to_vector(goal["direction"])
    score = dot(normalize(vec(out_port.axis)), direction_vector)
    if score < PARALLEL_DOT_THRESHOLD:
        _append_issue(
            issues,
            "TURN_DIRECTION_MISMATCH",
            "error",
            "turn_direction",
            "Turn goal direction is not represented by the produced outlet axis.",
            transition=transition,
            module_id=produced_module.id,
            port_ids=[out_port.id],
            expected={
                "direction": goal["direction"],
                "threshold": PARALLEL_DOT_THRESHOLD,
            },
            actual={
                "out_axis": list(out_port.axis),
                "dot": round(score, 4),
            },
            suggestion={
                "operation": "adjust_turn_direction",
                "direction": goal["direction"],
            },
        )


def _validate_goal_completion(
    issues: list[StaticIssue],
    transition: StateTransition,
    action: ResolvedAction,
    before_state: PipeState,
    target_port: Port,
    produced_module: Any,
    goal: dict[str, Any],
) -> None:
    goal_id = goal.get("goal_id")
    if not goal_id or goal_id not in set(action.completed_goal_ids):
        return
    modules = [
        module
        for historical_action, module in zip(
            before_state.action_history,
            before_state.placed_modules,
        )
        if goal_id in historical_action.affected_goal_ids
    ]
    modules.append(produced_module)
    goal_type = goal.get("type")

    if goal_type == "connect":
        if produced_module.type != "connect_ports":
            _append_issue(
                issues,
                "GOAL_CONNECT_MODULE_MISMATCH",
                "error",
                "goal_completion",
                "A completed connect goal must be realized by connect_ports.",
                transition=transition,
                module_id=produced_module.id,
                expected={"module": "connect_ports"},
                actual={"module": produced_module.type},
            )
        anchor = before_state.reserved_start_anchor
        other_port_id = action.params.get("other_port_id")
        connection_target = goal.get("connection_target", "another_open_port")
        if connection_target == "start_anchor":
            anchor_consumed = bool(
                anchor is not None
                and other_port_id == anchor.id
                and anchor.id in action.consumed_port_ids
                and anchor.id in transition.removed_port_ids
                and anchor.id in produced_module.input_bindings.values()
            )
            if not anchor_consumed:
                _append_issue(
                    issues,
                    "GOAL_START_ANCHOR_NOT_CONSUMED",
                    "error",
                    "goal_completion",
                    "The START-seam closure did not consume and mate the reserved first inlet.",
                    transition=transition,
                    module_id=produced_module.id,
                    port_ids=[anchor.id] if anchor is not None else [],
                    expected={
                        "connection_target": "start_anchor",
                        "reserved_port_id": anchor.id if anchor is not None else None,
                    },
                    actual={
                        "other_port_id": other_port_id,
                        "consumed_port_ids": action.consumed_port_ids,
                        "removed_port_ids": transition.removed_port_ids,
                        "input_bindings": produced_module.input_bindings,
                    },
                )
        elif anchor is not None and other_port_id == anchor.id:
            _append_issue(
                issues,
                "GOAL_UNDECLARED_START_ANCHOR_CLOSURE",
                "error",
                "goal_completion",
                "A normal two-open-port connect goal cannot consume the reserved START inlet.",
                transition=transition,
                module_id=produced_module.id,
                port_ids=[anchor.id],
                expected={"connection_target": "another_open_port"},
                actual={"other_port_id": other_port_id},
            )

    if goal_type in {"move", "route", "connector"} and goal.get("length") is not None:
        direction = goal.get("direction")
        matching_connector_modules = [
            module
            for module in modules
            if module.type == "inline_component"
            and module.params.get("component_type") == goal.get("component")
        ]
        has_unmeasured_spline = any(
            module.params.get("path_kind") == "spline" for module in modules
        )
        if goal_type == "connector" and goal.get("component") is not None:
            actual_length = (
                float(matching_connector_modules[0].params["length"])
                if len(matching_connector_modules) == 1
                and matching_connector_modules[0].params.get("length") is not None
                else None
            )
        elif goal_type in {"route", "connector"} and has_unmeasured_spline:
            actual_length = None
        elif goal_type in {"route", "connector"}:
            actual_length = sum(_module_centerline_length(module) for module in modules)
        elif direction is not None:
            direction_vector = direction_to_vector(direction)
            actual_length = sum(
                dot(_module_primary_displacement(module), direction_vector)
                for module in modules
            )
        else:
            actual_length = sum(
                length(_module_primary_displacement(module)) for module in modules
            )
        expected_length = float(goal["length"])
        tolerance = max(VECTOR_TOLERANCE, expected_length * 1e-3)
        if (
            actual_length is not None
            and abs(actual_length - expected_length) > tolerance
        ):
            _append_issue(
                issues,
                "GOAL_LENGTH_MISMATCH",
                "error",
                "goal_completion",
                "The actions completing this goal do not realize its required length.",
                transition=transition,
                module_id=produced_module.id,
                expected={"length": expected_length, "tolerance": tolerance},
                actual={"length": actual_length, "goal_id": goal_id},
            )

    if goal_type == "route":
        required_path_kind = goal.get("path_kind")
        if required_path_kind is not None:
            actual_path_kinds = sorted(
                {
                    str(module.params.get("path_kind"))
                    for module in modules
                    if module.params.get("path_kind") is not None
                }
            )
            if actual_path_kinds != [str(required_path_kind)]:
                _append_issue(
                    issues,
                    "GOAL_ROUTE_PATH_KIND_MISMATCH",
                    "error",
                    "goal_completion",
                    "Completed route does not preserve the requested path kind.",
                    transition=transition,
                    module_id=produced_module.id,
                    expected={"path_kind": required_path_kind},
                    actual={"path_kinds": actual_path_kinds},
                )
        required_curvature = goal.get("minimum_curvature_radius")
        if required_curvature is not None:
            curvature_failures = []
            for module in modules:
                if module.params.get("path_kind") == "circular_arc":
                    actual_radius = module.params.get("bend_radius")
                elif module.params.get("path_kind") == "spline":
                    actual_radius = module.params.get("minimum_curvature_radius")
                else:
                    actual_radius = None
                if actual_radius is not None and (
                    float(actual_radius) + VECTOR_TOLERANCE < float(required_curvature)
                ):
                    curvature_failures.append(
                        {"module_id": module.id, "authored_radius": actual_radius}
                    )
            if curvature_failures:
                _append_issue(
                    issues,
                    "GOAL_ROUTE_CURVATURE_CONTRACT_MISMATCH",
                    "error",
                    "goal_completion",
                    "The action weakened the route's immutable minimum curvature.",
                    transition=transition,
                    module_id=produced_module.id,
                    expected={"minimum_curvature_radius": required_curvature},
                    actual={"failures": curvature_failures},
                )
        route_points = _goal_path_points(modules)
        if not route_points:
            _append_issue(
                issues,
                "GOAL_ROUTE_DISCONTINUOUS",
                "error",
                "goal_completion",
                "Modules assigned to one route goal do not form one continuous chain.",
                transition=transition,
                module_id=produced_module.id,
                actual={"module_ids": [module.id for module in modules]},
            )
        direction = goal.get("direction")
        if direction is not None and len(route_points) >= 2:
            displacement = sub(route_points[-1], route_points[0])
            direction_dot = (
                dot(normalize(displacement), direction_to_vector(direction))
                if length(displacement) > VECTOR_TOLERANCE
                else -1.0
            )
            if direction_dot < PARALLEL_DOT_THRESHOLD:
                _append_issue(
                    issues,
                    "GOAL_ROUTE_DIRECTION_MISMATCH",
                    "error",
                    "goal_completion",
                    "Completed route displacement does not follow its required direction.",
                    transition=transition,
                    module_id=produced_module.id,
                    expected={"direction": direction, "dot": PARALLEL_DOT_THRESHOLD},
                    actual={"dot": direction_dot},
                )
        missing_waypoints = []
        out_of_order_waypoints = []
        previous_progress = -1.0
        waypoint_frame = goal.get("waypoint_frame") or "global"
        waypoint_origin = (
            vec(route_points[0])
            if waypoint_frame == "relative_to_target"
            else (0.0, 0.0, 0.0)
        )
        for waypoint in goal.get("required_waypoints") or []:
            authored_point = vec(waypoint)
            point = (
                add(waypoint_origin, authored_point)
                if waypoint_frame == "relative_to_target"
                else authored_point
            )
            distance, progress = _point_to_goal_path_projection(point, modules)
            if distance > VECTOR_TOLERANCE * 10.0:
                missing_waypoints.append(
                    {
                        "waypoint": list(authored_point),
                        "waypoint_frame": waypoint_frame,
                        "resolved_global_waypoint": list(point),
                        "distance": distance,
                    }
                )
            elif progress + VECTOR_TOLERANCE < previous_progress:
                out_of_order_waypoints.append(
                    {
                        "waypoint": list(authored_point),
                        "waypoint_frame": waypoint_frame,
                        "resolved_global_waypoint": list(point),
                        "progress": progress,
                        "previous_progress": previous_progress,
                    }
                )
            previous_progress = max(previous_progress, progress)
        if missing_waypoints:
            _append_issue(
                issues,
                "GOAL_ROUTE_WAYPOINT_MISMATCH",
                "error",
                "goal_completion",
                "Completed route does not pass through every required waypoint.",
                transition=transition,
                module_id=produced_module.id,
                expected={
                    "required_waypoints": goal.get("required_waypoints"),
                    "waypoint_frame": waypoint_frame,
                },
                actual={"missing_waypoints": missing_waypoints},
            )
        if out_of_order_waypoints:
            _append_issue(
                issues,
                "GOAL_ROUTE_WAYPOINT_ORDER_MISMATCH",
                "error",
                "goal_completion",
                "Required route waypoints occur in the wrong traversal order.",
                transition=transition,
                module_id=produced_module.id,
                expected={
                    "required_waypoints": goal.get("required_waypoints"),
                    "waypoint_frame": waypoint_frame,
                },
                actual={"out_of_order_waypoints": out_of_order_waypoints},
            )
        out_port = produced_module.ports.get("out")
        terminal_position = goal.get("terminal_position")
        if terminal_position is not None and (
            out_port is None or not _near(out_port.position, terminal_position)
        ):
            _append_issue(
                issues,
                "GOAL_ROUTE_TERMINAL_POSITION_MISMATCH",
                "error",
                "goal_completion",
                "Completed route terminal position does not match its contract.",
                transition=transition,
                module_id=produced_module.id,
                expected={"terminal_position": terminal_position},
                actual={
                    "terminal_position": (
                        list(out_port.position) if out_port is not None else None
                    )
                },
            )
        terminal_axis = goal.get("terminal_axis")
        if terminal_axis is not None and (
            out_port is None or not _same_direction(out_port.axis, terminal_axis)
        ):
            _append_issue(
                issues,
                "GOAL_ROUTE_TERMINAL_AXIS_MISMATCH",
                "error",
                "goal_completion",
                "Completed route terminal axis does not match its contract.",
                transition=transition,
                module_id=produced_module.id,
                expected={"terminal_axis": terminal_axis},
                actual={
                    "terminal_axis": list(out_port.axis)
                    if out_port is not None
                    else None
                },
            )

    if goal_type == "branch":
        for field_name in ("blend_radius", "inner_blend_radius", "max_hub_radius"):
            expected_value = goal.get(field_name)
            if expected_value is None:
                continue
            actual_value = produced_module.params.get(field_name)
            if (
                actual_value is None
                or abs(float(actual_value) - float(expected_value)) > VECTOR_TOLERANCE
            ):
                _append_issue(
                    issues,
                    "GOAL_JUNCTION_DIMENSION_MISMATCH",
                    "error",
                    "goal_completion",
                    "Junction geometry changed an immutable authored dimension.",
                    transition=transition,
                    module_id=produced_module.id,
                    expected={field_name: expected_value},
                    actual={field_name: actual_value},
                )

    if goal_type == "connector" and goal.get("component") is not None:
        expected_component = str(goal["component"])
        matching_modules = [
            module
            for module in modules
            if module.type == "inline_component"
            and module.params.get("component_type") == expected_component
        ]
        if len(matching_modules) != 1:
            _append_issue(
                issues,
                "GOAL_COMPONENT_GEOMETRY_MISMATCH",
                "error",
                "goal_completion",
                "A completed connector goal must contain exactly one matching inline component instance.",
                transition=transition,
                module_id=produced_module.id,
                expected={"component": expected_component},
                actual={
                    "module_types": [module.type for module in modules],
                    "component_types": [
                        module.params.get("component_type") for module in modules
                    ],
                    "matching_module_ids": [module.id for module in matching_modules],
                },
            )
        component_spec = goal.get("component_spec")
        if component_spec is not None and len(matching_modules) == 1:
            component_module = matching_modules[0]
            mismatches = []
            for field_name, expected_value in component_spec.items():
                if field_name == "component_type" or expected_value is None:
                    continue
                actual_value = component_module.params.get(field_name)
                if isinstance(expected_value, (list, tuple)):
                    matches = actual_value is not None and _same_direction(
                        actual_value, expected_value
                    )
                elif isinstance(expected_value, (int, float)) and not isinstance(
                    expected_value, bool
                ):
                    try:
                        matches = (
                            abs(float(actual_value) - float(expected_value))
                            <= VECTOR_TOLERANCE
                        )
                    except (TypeError, ValueError):
                        matches = False
                else:
                    matches = actual_value == expected_value
                if not matches:
                    mismatches.append(
                        {
                            "field": field_name,
                            "expected": expected_value,
                            "actual": actual_value,
                        }
                    )
            if mismatches:
                _append_issue(
                    issues,
                    "GOAL_COMPONENT_DIMENSION_MISMATCH",
                    "error",
                    "goal_completion",
                    "Inline component geometry changed explicit user-authored details.",
                    transition=transition,
                    module_id=component_module.id,
                    expected={"component_spec": component_spec},
                    actual={"mismatches": mismatches},
                )
        direction = goal.get("direction")
        if direction is not None:
            displacement = _module_primary_displacement(produced_module)
            score = (
                dot(normalize(displacement), direction_to_vector(direction))
                if length(displacement) > VECTOR_TOLERANCE
                else -1.0
            )
            if score < PARALLEL_DOT_THRESHOLD:
                _append_issue(
                    issues,
                    "GOAL_CONNECTOR_DIRECTION_MISMATCH",
                    "error",
                    "goal_completion",
                    "Inline connector displacement has the wrong direction.",
                    transition=transition,
                    module_id=produced_module.id,
                    expected={"direction": direction},
                    actual={"dot": score},
                )

    if goal_type == "turn" and goal.get("angle") is not None:
        expected_angle = abs(float(goal["angle"]))
        actual_angle = sum(_module_turn_angle(module) for module in modules)
        tolerance = max(0.1, expected_angle * 1e-3)
        if abs(actual_angle - expected_angle) > tolerance:
            _append_issue(
                issues,
                "GOAL_TURN_ANGLE_MISMATCH",
                "error",
                "goal_completion",
                "The actions completing this turn do not realize its required angle.",
                transition=transition,
                module_id=produced_module.id,
                expected={"angle": expected_angle, "tolerance": tolerance},
                actual={"angle": actual_angle, "goal_id": goal_id},
            )
        if goal.get("direction") is None and goal.get("plane_normal") is not None:
            signed_angles = [
                float(value)
                for module in modules
                for value in [
                    module.params.get(
                        "sweep_angle",
                        module.params.get("angle"),
                    )
                ]
                if value is not None
            ]
            actual_signed_angle = sum(signed_angles) if signed_angles else None
            expected_signed_angle = float(goal["angle"])
            if (
                actual_signed_angle is None
                or abs(actual_signed_angle - expected_signed_angle) > tolerance
            ):
                _append_issue(
                    issues,
                    "GOAL_TURN_SIGNED_ANGLE_MISMATCH",
                    "error",
                    "goal_completion",
                    "The direction-free turn must preserve its signed right-hand sweep.",
                    transition=transition,
                    module_id=produced_module.id,
                    expected={
                        "signed_angle": expected_signed_angle,
                        "plane_normal": goal.get("plane_normal"),
                        "tolerance": tolerance,
                    },
                    actual={
                        "signed_angle": actual_signed_angle,
                        "goal_id": goal_id,
                    },
                )
        required_radius = goal.get("bend_radius")
        actual_radius = produced_module.params.get("bend_radius")
        if required_radius is not None and (
            actual_radius is None
            or abs(float(actual_radius) - float(required_radius)) > VECTOR_TOLERANCE
        ):
            _append_issue(
                issues,
                "GOAL_BEND_RADIUS_MISMATCH",
                "error",
                "goal_completion",
                "Completed turn uses the wrong bend radius.",
                transition=transition,
                module_id=produced_module.id,
                expected={"bend_radius": required_radius},
                actual={"bend_radius": actual_radius},
            )
        required_plane = goal.get("plane_normal")
        actual_plane = produced_module.params.get("plane_normal")
        if required_plane is not None and (
            actual_plane is None or not _same_direction(actual_plane, required_plane)
        ):
            _append_issue(
                issues,
                "GOAL_TURN_PLANE_MISMATCH",
                "error",
                "goal_completion",
                "Completed turn uses the wrong bend plane.",
                transition=transition,
                module_id=produced_module.id,
                expected={"plane_normal": required_plane},
                actual={"plane_normal": actual_plane},
            )

    if goal_type == "diameter_change" and goal.get("diameter_out") is not None:
        inlet_port = produced_module.ports.get("in")
        out_port = produced_module.ports.get("out")
        actual_diameter = out_port.outer_diameter if out_port is not None else None
        if (
            actual_diameter is None
            or abs(actual_diameter - float(goal["diameter_out"])) > VECTOR_TOLERANCE
        ):
            _append_issue(
                issues,
                "GOAL_DIAMETER_MISMATCH",
                "error",
                "goal_completion",
                "The completed diameter-change goal has the wrong output section.",
                transition=transition,
                module_id=produced_module.id,
                expected={"diameter_out": float(goal["diameter_out"])},
                actual={"diameter_out": actual_diameter, "goal_id": goal_id},
            )
        if (
            inlet_port is not None
            and out_port is not None
            and (
                abs(inlet_port.outer_diameter - out_port.outer_diameter)
                <= VECTOR_TOLERANCE
                and abs(inlet_port.wall_thickness - out_port.wall_thickness)
                <= VECTOR_TOLERANCE
            )
        ):
            _append_issue(
                issues,
                "GOAL_DIAMETER_TRANSITION_IS_NOOP",
                "error",
                "goal_completion",
                "Diameter-change geometry does not change outer diameter or wall thickness.",
                transition=transition,
                module_id=produced_module.id,
                actual={
                    "inlet": [inlet_port.outer_diameter, inlet_port.wall_thickness],
                    "outlet": [out_port.outer_diameter, out_port.wall_thickness],
                },
            )
        required_direction = goal.get("direction")
        if required_direction is not None:
            transition_axis = produced_module.params.get("axis")
            direction_score = (
                dot(
                    normalize(vec(transition_axis)),
                    direction_to_vector(required_direction),
                )
                if transition_axis is not None
                else -1.0
            )
            if direction_score < PARALLEL_DOT_THRESHOLD:
                _append_issue(
                    issues,
                    "GOAL_TRANSITION_DIRECTION_MISMATCH",
                    "error",
                    "goal_completion",
                    "Diameter transition axis has the wrong direction.",
                    transition=transition,
                    module_id=produced_module.id,
                    expected={"direction": required_direction},
                    actual={"dot": direction_score},
                )
        required_wall = goal.get("wall_thickness_out")
        actual_wall = out_port.wall_thickness if out_port is not None else None
        if required_wall is not None and (
            actual_wall is None
            or abs(actual_wall - float(required_wall)) > VECTOR_TOLERANCE
        ):
            _append_issue(
                issues,
                "GOAL_TRANSITION_WALL_MISMATCH",
                "error",
                "goal_completion",
                "Completed transition has the wrong outlet wall thickness.",
                transition=transition,
                module_id=produced_module.id,
                expected={"wall_thickness_out": required_wall},
                actual={"wall_thickness_out": actual_wall},
            )
        required_length = goal.get("transition_length")
        actual_length = produced_module.params.get("length")
        if required_length is not None and (
            actual_length is None
            or abs(float(actual_length) - float(required_length)) > VECTOR_TOLERANCE
        ):
            _append_issue(
                issues,
                "GOAL_TRANSITION_LENGTH_MISMATCH",
                "error",
                "goal_completion",
                "Completed transition has the wrong axial length.",
                transition=transition,
                module_id=produced_module.id,
                expected={"transition_length": required_length},
                actual={"transition_length": actual_length},
            )
        required_offset = goal.get("offset")
        actual_offset = produced_module.params.get("offset")
        if required_offset is not None and (
            actual_offset is None or not _near(required_offset, actual_offset)
        ):
            _append_issue(
                issues,
                "GOAL_TRANSITION_OFFSET_MISMATCH",
                "error",
                "goal_completion",
                "Completed transition has the wrong eccentric offset.",
                transition=transition,
                module_id=produced_module.id,
                expected={"offset": required_offset},
                actual={"offset": actual_offset},
            )

    if goal_type == "connect" and (
        len(action.consumed_port_ids) != 2 or transition.produced_port_ids
    ):
        _append_issue(
            issues,
            "GOAL_CONNECT_TOPOLOGY_MISMATCH",
            "error",
            "goal_completion",
            "A completed connect goal must consume two ports and leave no outlet.",
            transition=transition,
            module_id=produced_module.id,
            expected={"consumed_ports": 2, "produced_ports": 0},
            actual={
                "consumed_ports": len(action.consumed_port_ids),
                "produced_ports": len(transition.produced_port_ids),
                "goal_id": goal_id,
            },
        )
    if goal_type == "connect" and goal.get("required_waypoints"):
        previous_progress = -1.0
        failures = []
        for waypoint in goal["required_waypoints"]:
            distance, progress = _point_to_goal_path_projection(vec(waypoint), modules)
            if (
                distance > VECTOR_TOLERANCE * 10.0
                or progress + VECTOR_TOLERANCE < previous_progress
            ):
                failures.append(
                    {
                        "waypoint": waypoint,
                        "distance": distance,
                        "progress": progress,
                    }
                )
            previous_progress = max(previous_progress, progress)
        if failures:
            _append_issue(
                issues,
                "GOAL_CONNECT_WAYPOINT_MISMATCH",
                "error",
                "goal_completion",
                "connect goal waypoints are missing or out of order.",
                transition=transition,
                module_id=produced_module.id,
                expected={"required_waypoints": goal["required_waypoints"]},
                actual={"failures": failures},
            )

    if goal_type == "end":
        end_type = goal.get("end_type")
        produced_count = len(transition.produced_port_ids)
        actual_termination = action.params.get("termination_type")
        if actual_termination is None and action.module == "cap_pipe":
            actual_termination = action.params.get("end_type")
        valid = (end_type == "open" and produced_count >= 1) or (
            end_type in {"cap", "plug"}
            and produced_count == 0
            and len(action.consumed_port_ids) == 1
            and actual_termination == end_type
        )
        if not valid:
            _append_issue(
                issues,
                "GOAL_END_TOPOLOGY_MISMATCH",
                "error",
                "goal_completion",
                "The completed end goal does not match its open/capped contract.",
                transition=transition,
                module_id=produced_module.id,
                expected={"end_type": end_type},
                actual={
                    "produced_ports": produced_count,
                    "goal_id": goal_id,
                    "module": action.module,
                    "termination_type": actual_termination,
                },
            )
        required_thickness = goal.get("termination_thickness")
        actual_thickness = action.params.get("thickness")
        if actual_thickness is None and action.module == "cap_pipe":
            actual_thickness = action.params.get("cap_thickness")
        if required_thickness is not None and (
            actual_thickness is None
            or abs(float(actual_thickness) - float(required_thickness))
            > VECTOR_TOLERANCE
        ):
            _append_issue(
                issues,
                "GOAL_TERMINATION_THICKNESS_MISMATCH",
                "error",
                "goal_completion",
                "Termination geometry changed the immutable authored thickness.",
                transition=transition,
                module_id=produced_module.id,
                expected={"termination_thickness": required_thickness},
                actual={"thickness": actual_thickness},
            )


def _module_primary_displacement(module: Any) -> tuple[float, float, float]:
    in_port = module.ports.get("in") or module.ports.get("in_a")
    out_port = module.ports.get("out")
    if in_port is None or out_port is None:
        return (0.0, 0.0, 0.0)
    return sub(vec(out_port.position), vec(in_port.position))


def _goal_path_points(modules: list[Any]) -> list[tuple[float, float, float]]:
    result: list[tuple[float, float, float]] = []
    for module in modules:
        raw_points = module.params.get("path_points")
        if isinstance(raw_points, list) and len(raw_points) >= 2:
            points = [vec(point) for point in raw_points]
        else:
            inlet = module.ports.get("in") or module.ports.get("in_a")
            outlet = module.ports.get("out")
            points = (
                [vec(inlet.position), vec(outlet.position)]
                if inlet is not None and outlet is not None
                else []
            )
        if result and points and not _near(result[-1], points[0]):
            return []
        for point in points:
            if not result or not _near(result[-1], point):
                result.append(point)
    return result


def _point_to_polyline_distance(
    point: tuple[float, float, float],
    polyline: list[tuple[float, float, float]],
) -> float:
    return _point_to_polyline_projection(point, polyline)[0]


def _point_to_polyline_projection(
    point: tuple[float, float, float],
    polyline: list[tuple[float, float, float]],
) -> tuple[float, float]:
    if not polyline:
        return float("inf"), 0.0
    if len(polyline) == 1:
        return length(sub(point, polyline[0])), 0.0
    best = float("inf")
    best_progress = 0.0
    cumulative = 0.0
    for start, end in zip(polyline, polyline[1:]):
        segment = sub(end, start)
        squared = dot(segment, segment)
        segment_length = math.sqrt(squared)
        if squared <= 1e-18:
            candidate = length(sub(point, start))
            parameter = 0.0
        else:
            parameter = max(0.0, min(1.0, dot(sub(point, start), segment) / squared))
            projection = add(start, mul(segment, parameter))
            candidate = length(sub(point, projection))
        if candidate < best:
            best = candidate
            best_progress = cumulative + parameter * segment_length
        cumulative += segment_length
    return best, best_progress


def _point_to_goal_path_projection(
    point: tuple[float, float, float],
    modules: list[Any],
) -> tuple[float, float]:
    """Project onto authored geometry without treating curves as their chords.

    Circular routes have an analytic center/radius/sweep representation.  A
    spline's exact shape belongs to FreeCAD, so deterministic validation requires
    every immutable waypoint to be present in the LLM-authored interpolation
    points instead of pretending the control-point polyline is the spline.
    """

    best_distance = float("inf")
    best_progress = 0.0
    cumulative = 0.0
    for module in modules:
        raw_points = module.params.get("path_points")
        points = (
            [vec(candidate) for candidate in raw_points]
            if isinstance(raw_points, list) and len(raw_points) >= 2
            else _goal_path_points([module])
        )
        kind = module.params.get("path_kind")
        if (
            module.type == "route"
            and kind == "circular_arc"
            and module.params.get("bend_radius") is not None
            and module.params.get("sweep_angle") is not None
            and module.params.get("plane_normal") is not None
            and module.params.get("axis") is not None
        ):
            local_distance, local_progress, path_length = (
                _point_to_circular_arc_projection(
                    point,
                    vec(module.params["start_position"]),
                    normalize(vec(module.params["axis"])),
                    normalize(vec(module.params["plane_normal"])),
                    float(module.params["bend_radius"]),
                    float(module.params["sweep_angle"]),
                )
            )
        elif kind in {"spline", "circular_arc"}:
            # connect_ports arcs expose their required middle waypoint directly;
            # splines interpolate every authored point.  Match those points
            # exactly and leave continuous curve measurements to FreeCAD MCP.
            distances = [length(sub(point, candidate)) for candidate in points]
            if distances:
                point_index = min(range(len(distances)), key=distances.__getitem__)
                local_distance = distances[point_index]
                local_progress = sum(
                    length(sub(right, left))
                    for left, right in zip(points, points[1 : point_index + 1])
                )
            else:
                local_distance = float("inf")
                local_progress = 0.0
            path_length = sum(
                length(sub(right, left)) for left, right in zip(points, points[1:])
            )
        else:
            local_distance, local_progress = _point_to_polyline_projection(
                point, points
            )
            path_length = sum(
                length(sub(right, left)) for left, right in zip(points, points[1:])
            )
        if local_distance < best_distance:
            best_distance = local_distance
            best_progress = cumulative + local_progress
        cumulative += path_length
    return best_distance, best_progress


def _point_to_circular_arc_projection(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    tangent: tuple[float, float, float],
    plane_normal: tuple[float, float, float],
    radius: float,
    sweep_angle: float,
) -> tuple[float, float, float]:
    normal = normalize(plane_normal)
    radial = normalize(cross(tangent, normal))
    signed_radius = radius if sweep_angle >= 0.0 else -radius
    center = sub(start, mul(radial, signed_radius))
    start_radius = sub(start, center)
    relative = sub(point, center)
    plane_offset = dot(relative, normal)
    planar = sub(relative, mul(normal, plane_offset))
    sweep = math.radians(sweep_angle)
    total_length = abs(sweep) * radius
    if length(planar) <= 1e-15:
        endpoint = start
        return length(sub(point, endpoint)), 0.0, total_length

    theta = math.atan2(
        dot(normal, cross(start_radius, planar)),
        dot(start_radius, planar),
    )
    if sweep >= 0.0:
        if theta < 0.0:
            theta += math.tau
        clamped_theta = max(0.0, min(sweep, theta))
    else:
        if theta > 0.0:
            theta -= math.tau
        clamped_theta = min(0.0, max(sweep, theta))
    nearest = add(
        center,
        rotate(start_radius, normal, clamped_theta),
    )
    return (
        length(sub(point, nearest)),
        abs(clamped_theta) * radius,
        total_length,
    )


def _module_turn_angle(module: Any) -> float:
    if module.params.get("sweep_angle") is not None:
        return abs(float(module.params["sweep_angle"]))
    if module.params.get("angle") is not None:
        return abs(float(module.params["angle"]))
    in_port = module.ports.get("in") or module.ports.get("in_a")
    out_port = module.ports.get("out")
    if in_port is None or out_port is None:
        return 0.0
    incoming = tuple(-value for value in in_port.axis)
    cosine = max(
        -1.0, min(1.0, dot(normalize(vec(incoming)), normalize(vec(out_port.axis))))
    )
    return math.degrees(math.acos(cosine))


def _include_primary_outlet(params: dict[str, Any], goal: dict[str, Any]) -> bool:
    if goal.get("include_primary_outlet") is not None:
        return bool(goal["include_primary_outlet"])
    if params.get("include_primary_outlet") is not None:
        return bool(params["include_primary_outlet"])
    return not bool(_required_outlet_vectors(params, goal))


def _required_outlet_vectors(
    params: dict[str, Any],
    goal: dict[str, Any],
) -> list[tuple[float, float, float]]:
    del params
    raw_vectors = goal.get("required_outlet_vectors") or [
        item.get("axis")
        for item in (goal.get("required_outlets") or [])
        if isinstance(item, dict)
    ]
    return _normalize_vector_list(raw_vectors)


def _intent_required_outlet_vectors(
    intent: IntentResult,
) -> list[tuple[float, float, float]]:
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


def _normalize_vector_list(raw_vectors: Any) -> list[tuple[float, float, float]]:
    if not isinstance(raw_vectors, list):
        return []
    vectors: list[tuple[float, float, float]] = []
    for raw_vector in raw_vectors:
        try:
            candidate = vec(raw_vector)
        except (TypeError, ValueError):
            continue
        if length(candidate) <= VECTOR_TOLERANCE:
            continue
        vectors.append(normalize(candidate))
    return vectors


def _match_vectors_to_ports(
    required_vectors: list[tuple[float, float, float]],
    ports: list[Port],
) -> dict[str, Any]:
    available = sorted(ports, key=lambda port: port.id)
    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for vector_index, required_vector in enumerate(required_vectors):
        ranked = sorted(
            (
                (
                    dot(normalize(vec(port.axis)), required_vector),
                    port,
                )
                for port in available
            ),
            key=lambda item: (-item[0], item[1].id),
        )
        if ranked and ranked[0][0] >= EXPLICIT_VECTOR_DOT_THRESHOLD:
            score, port = ranked[0]
            matched.append(
                {
                    "vector_index": vector_index,
                    "expected_vector": list(required_vector),
                    "port_id": port.id,
                    "dot": round(score, 4),
                    "axis": list(port.axis),
                    "position": list(port.position),
                }
            )
            available = [
                candidate for candidate in available if candidate.id != port.id
            ]
        else:
            best_score = ranked[0][0] if ranked else None
            missing.append(
                {
                    "vector_index": vector_index,
                    "expected_vector": list(required_vector),
                    "best_dot": round(best_score, 4)
                    if best_score is not None
                    else None,
                }
            )
    rejected = []
    for port in available:
        scores = [
            dot(normalize(vec(port.axis)), required_vector)
            for required_vector in required_vectors
        ]
        rejected.append(
            {
                "port_id": port.id,
                "position": list(port.position),
                "axis": list(port.axis),
                "best_dot": round(max(scores), 4) if scores else None,
            }
        )
    return {
        "expected_vectors": _vectors_json(required_vectors),
        "matched_port_ids": [item["port_id"] for item in matched],
        "matched": matched,
        "missing_vectors": missing,
        "rejected_ports": rejected,
        "threshold": EXPLICIT_VECTOR_DOT_THRESHOLD,
    }


def _vectors_json(vectors: list[tuple[float, float, float]]) -> list[list[float]]:
    return [[round(float(component), 6) for component in vector] for vector in vectors]


def _direction_score(target_port: Port, port: Port, direction: Direction) -> float:
    direction_vector = direction_to_vector(direction)
    displacement = sub(vec(port.position), vec(target_port.position))
    if length(displacement) > VECTOR_TOLERANCE:
        return dot(normalize(displacement), direction_vector)
    return dot(normalize(vec(port.axis)), direction_vector)


def _append_issue(
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


def _find_port(port_id: str, ports: list[Port]) -> Port | None:
    for port in ports:
        if port.id == port_id:
            return port
    return None


def _find_connectable_port(port_id: str, state: PipeState) -> Port | None:
    port = _find_port(port_id, state.open_ports)
    if port is not None:
        return port
    anchor = state.reserved_start_anchor
    if anchor is not None and anchor.id == port_id:
        return anchor
    return None


def _produced_module(before_state: PipeState, after_state: PipeState) -> Any | None:
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
    before_module_ids = {module.id for module in before_state.placed_modules}
    return sum(
        1 for module in after_state.placed_modules if module.id not in before_module_ids
    )


def _near(a: Any, b: Any, tolerance: float = VECTOR_TOLERANCE) -> bool:
    return length(sub(vec(a), vec(b))) <= tolerance


def _parallel(a: Any, b: Any) -> bool:
    return abs(dot(normalize(vec(a)), normalize(vec(b)))) >= PARALLEL_DOT_THRESHOLD


def _same_direction(a: Any, b: Any) -> bool:
    return dot(normalize(vec(a)), normalize(vec(b))) >= PARALLEL_DOT_THRESHOLD


def _port_role(port_name: str) -> str:
    if port_name.startswith("out_"):
        return "branch_outlet"
    if port_name == "out":
        return "primary_outlet"
    return "other"


def _duplicate_open_port_groups(ports: list[Port]) -> list[dict[str, Any]]:
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
    if not has_errors(issues):
        return ["Proceed to final FreeCAD MCP execution."]
    actions = []
    for issue in issues:
        if issue.severity == "error":
            actions.append(f"Fix {issue.issue_code}: {issue.message}")
    return actions[:5]
