"""move, turn, branch와 전체 goal 완료 계약을 검증한다."""

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

def _validate_branch_goal_direction(
    issues: list[StaticIssue],
    transition: StateTransition,
    action: ResolvedAction,
    target_port: Port,
    produced_module: Any,
    goal: dict[str, Any],
) -> None:
    """분기 goal 방향 계약을 검증한다."""

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
    """직선 move goal 방향을 검증한다."""

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
    """turn goal 평면·각 계약을 검증한다."""

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
    """goal 완료 표시가 실제 결과와 맞는지 검사한다."""

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
            # When host overrode an impossible world direction to keep port
            # continuity, treat residual heading error as a warning — mating
            # solid is still valid; a later turn can realize the world heading.
            severity = "error"
            if produced_module.params.get("direction_overridden_to_port"):
                severity = "warning"
            if direction_dot < PARALLEL_DOT_THRESHOLD:
                _append_issue(
                    issues,
                    "GOAL_ROUTE_DIRECTION_MISMATCH",
                    severity,
                    "goal_completion",
                    "Completed route displacement does not follow its required direction.",
                    transition=transition,
                    module_id=produced_module.id,
                    expected={"direction": direction, "dot": PARALLEL_DOT_THRESHOLD},
                    actual={
                        "dot": direction_dot,
                        "direction_overridden_to_port": bool(
                            produced_module.params.get("direction_overridden_to_port")
                        ),
                    },
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

