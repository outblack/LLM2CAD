"""단일 action의 module 접합, graph 전이와 route 연속성을 검증한다."""

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

def _validate_module_connection(
    issues: list[StaticIssue],
    transition: StateTransition,
    action: ResolvedAction,
    target_port: Port,
    produced_module: Any,
    tolerance: float,
) -> None:
    """단일 모듈 연결 계약을 검증한다."""

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
    """상태 전이 그래프 불변식을 검증한다."""

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

def _validate_route_continuity(
    issues: list[StaticIssue],
    transition: StateTransition,
    action: ResolvedAction,
    before_state: PipeState,
    target_port: Port,
    produced_module: Any,
) -> None:
    """route 연속성(위치·접선)을 검증한다."""

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

