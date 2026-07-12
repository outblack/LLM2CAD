"""최종 PipeState의 graph, 치수, 곡률과 충돌 규칙을 검증한다."""

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
from cadgen.static_geometry_metrics import (
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

# Shared thresholds (kept identical to static_geometry_validator).
VECTOR_TOLERANCE = 1e-4
PARALLEL_DOT_THRESHOLD = 0.9999
BRANCH_DIRECTION_DOT_THRESHOLD = 0.35
EXPLICIT_VECTOR_DOT_THRESHOLD = 0.9999
from cadgen.static_issue_builder import append_issue as _append_issue

def _validate_final_graph(issues: list[StaticIssue], state: PipeState) -> None:
    """최종 조립 그래프 불변식을 검증한다."""

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
    """intent 기하 제약을 최종 상태에서 검증한다."""

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
    """최종 goal 길이가 계약을 만족하는지 검사한다."""

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
    """최종 spline 곡률 하한을 검사한다."""

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

def _validate_conservative_collision(
    issues: list[StaticIssue],
    transition: StateTransition,
    before_state: PipeState,
    produced_module: Any,
) -> None:
    """보수적 비인접 충돌/clearance를 검사한다."""

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
    """validate_intra_module_clearance 관련 계약을 검증한다."""

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

