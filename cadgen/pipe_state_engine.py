"""LLM 행동을 결정론적인 파이프 모듈ㆍ포트 그래프 전이로 해석한다.

``ActionDraft``와 현재 ``PipeState``를 입력받아 resolved action과 다음 상태를 만든다.
종속 좌표만 계산하며, 실패한 행동 대신 다른 설계나 primitive를 선택하지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from cadgen.primitive_action_catalog import canonicalize_junction_params, filter_draft_params
from cadgen.runtime_settings import Settings
from cadgen.geometry_safety_policy import minimum_spline_curvature_radius
from cadgen.typed_data_models import (
    ActionDraft,
    ConnectionEdge,
    IntentResult,
    ModuleRef,
    ModuleIncidenceEdge,
    PipeState,
    Port,
    ResolvedAction,
)
from cadgen.vector3_math import (
    Vector,
    add,
    arc_points,
    canonical_circular_arc_frame,
    circular_rim_mismatch,
    choose_perpendicular_axis,
    direction_to_vector,
    dot,
    cross,
    length,
    mul,
    normalize,
    rotate,
    sub,
    vec,
)


@dataclass
class StateEngine:
    """행동의 종속 기하를 계산하고 불변 ``PipeState`` 전이를 생성한다."""

    settings: Settings

    def initial_state(self, intent: IntentResult) -> PipeState:
        """검증된 intent를 START 포트 하나가 있는 초기 상태로 변환한다."""

        start_axis = normalize(vec(intent.start_axis))
        goals = [
            goal if goal.goal_id else goal.model_copy(update={"goal_id": f"G{index}"})
            for index, goal in enumerate(intent.target_behavior, start=1)
        ]
        start_port = Port(
            id="START",
            position=intent.start_position,
            axis=start_axis,
            outer_diameter=intent.global_spec.outer_diameter,
            wall_thickness=intent.global_spec.wall_thickness,
        )
        return PipeState(
            state_id="S0",
            state_version=0,
            contract_digest=intent.contract_digest,
            modeling_tolerance=self.settings.modeling_tolerance,
            global_spec=intent.global_spec,
            expected_open_ports=intent.expected_open_ports,
            expected_open_ports_source=intent.expected_open_ports_source,
            required_components=list(intent.required_components),
            hard_constraints=list(intent.hard_constraints),
            geometric_constraints=[
                item.model_copy(deep=True) for item in intent.geometric_constraints
            ],
            design_notes=list(intent.design_notes),
            module_measurements={},
            placed_modules=[],
            open_ports=[start_port],
            port_nodes={start_port.id: start_port},
            open_port_ids=[start_port.id],
            used_ports=[],
            remaining_goals=goals,
            action_history=[],
        )

    def resolve_action(self, draft: ActionDraft, state: PipeState) -> ResolvedAction:
        """LLM 독립값을 보존하며 대상 포트에 종속된 파라미터를 계산한다."""

        target = self._find_port(draft.target_port, state)
        params = filter_draft_params(draft.module, _without_none(draft.params))
        action_id = f"A{state.state_version + 1}"
        global_spec = state.global_spec
        params.setdefault("start_position", target.position)
        params.setdefault("axis", target.axis)

        if draft.module in {"straight_pipe", "connector_pipe"}:
            if params.get("direction"):
                params["axis"] = direction_to_vector(
                    params.get("direction"),
                    default=target.axis,
                )
            params.setdefault(
                "length", 10.0 if draft.module == "connector_pipe" else 100.0
            )
            params.setdefault("outer_diameter", target.outer_diameter)
            params.setdefault("wall_thickness", target.wall_thickness)
            if draft.module == "connector_pipe":
                params.setdefault(
                    "coupling_outer_diameter", target.outer_diameter * 1.25
                )
                params.setdefault("sleeve_overlap", target.outer_diameter * 0.25)
        elif draft.module == "bend_pipe":
            out_axis = self._resolve_bend_out_axis(
                params.get("turn_direction"), target.axis
            )
            params.setdefault("angle", 90.0)
            params.setdefault("bend_radius", self.settings.default_bend_radius)
            params.setdefault("outer_diameter", target.outer_diameter)
            params.setdefault("wall_thickness", target.wall_thickness)
            params.setdefault("out_axis", out_axis)
            params.setdefault("segment_resolution", 24)
            params["segment_resolution"] = max(
                4,
                _int_or_default(params["segment_resolution"], 24),
            )
        elif draft.module == "junction_pipe":
            goal = (
                state.remaining_goals[0]
                if state.remaining_goals and state.remaining_goals[0].type == "branch"
                else None
            )
            if goal is not None:
                _backfill_junction_goal_params(params, goal)
            params = canonicalize_junction_params(params)
            explicit_vectors = _explicit_outlet_vectors(params)
            if explicit_vectors and "branch_count" not in params:
                params["branch_count"] = len(explicit_vectors)
            params.setdefault("branch_count", 2)
            params.setdefault("branch_angles", [45.0, -45.0])
            params.setdefault("outer_diameter", target.outer_diameter)
            params.setdefault("wall_thickness", target.wall_thickness)
            params.setdefault("blend_radius", target.outer_diameter * 0.6)
            params.setdefault("junction_style", "smooth_hub")
            if "include_primary_outlet" not in params:
                params["include_primary_outlet"] = not bool(explicit_vectors)
        elif draft.module == "reducer_pipe":
            params.setdefault("length", 50.0)
            params.setdefault("diameter_in", target.outer_diameter)
            params.setdefault(
                "diameter_out", params.get("diameter_out") or global_spec.outer_diameter
            )
            params.setdefault("wall_thickness_in", target.wall_thickness)
            params.setdefault("wall_thickness_out", global_spec.wall_thickness)
        elif draft.module == "cap_pipe":
            params.setdefault("end_type", "cap")
            params.setdefault("outer_diameter", target.outer_diameter)
            params.setdefault("wall_thickness", target.wall_thickness)
            params.setdefault("cap_thickness", max(target.wall_thickness * 1.5, 2.5))
        elif draft.module == "route":
            _resolve_section(params, target)
            kind = params["path_kind"]
            if kind == "line":
                # PAPR: mating is by construction. The open-port outward axis is
                # the only legal line heading at this attachment. A world-frame
                # goal.direction that fights the port (e.g. -X on a diagonal Y
                # primary) must not rewrite the module axis — that caused
                # MODULE_INPUT_AXIS_MISMATCH / PORT_CONTRACT_MISMATCH thrash.
                port_axis = normalize(vec(target.axis))
                authored = params.get("direction")
                if authored is not None:
                    desired = normalize(vec(authored))
                    if abs(dot(desired, port_axis) - 1.0) <= 1e-3 or (
                        dot(desired, port_axis) >= 0.85
                    ):
                        params["direction"] = desired
                        params["axis"] = desired
                    else:
                        params["direction"] = port_axis
                        params["axis"] = port_axis
                        params["direction_overridden_to_port"] = True
                        params["authored_direction"] = desired
                else:
                    params["direction"] = port_axis
                    params["axis"] = port_axis
            elif kind == "circular_arc":
                plane_normal, _start_tangent, terminal_tangent = (
                    canonical_circular_arc_frame(
                        normalize(vec(params["axis"])),
                        vec(params["plane_normal"]),
                        float(params["sweep_angle"]),
                    )
                )
                params["plane_normal"] = plane_normal
                params["terminal_axis"] = terminal_tangent
            elif kind == "spline":
                params["initial_tangent"] = normalize(vec(params["axis"]))
                waypoint_frame = params.get("waypoint_frame", "global")
                authored_waypoints = [vec(point) for point in params["waypoints"]]
                if waypoint_frame == "relative_to_target":
                    params["waypoints"] = [
                        add(target.position, offset) for offset in authored_waypoints
                    ]
                elif waypoint_frame == "global":
                    params["waypoints"] = authored_waypoints
                else:
                    raise ValueError(
                        "spline waypoint_frame must be global or relative_to_target"
                    )
                # Downstream geometry and evidence always use canonical global
                # points; the immutable goal retains the authored frame.
                params["waypoint_frame"] = "global"
                affected_goal_ids = set(draft.affected_goal_ids)
                route_goals = [
                    goal
                    for goal in state.remaining_goals
                    if goal.goal_id in affected_goal_ids and goal.type == "route"
                ]
                terminal_goal = route_goals[-1] if route_goals else None
                if (
                    terminal_goal is not None
                    and terminal_goal.terminal_axis is not None
                ):
                    final_tangent = vec(terminal_goal.terminal_axis)
                elif (
                    terminal_goal is not None
                    and len(terminal_goal.required_waypoints) >= 2
                ):
                    final_tangent = sub(
                        vec(terminal_goal.required_waypoints[-1]),
                        vec(terminal_goal.required_waypoints[-2]),
                    )
                elif (
                    terminal_goal is not None
                    and len(terminal_goal.required_waypoints) == 1
                ):
                    final_point = vec(terminal_goal.required_waypoints[0])
                    final_tangent = (
                        final_point
                        if terminal_goal.waypoint_frame == "relative_to_target"
                        else sub(final_point, target.position)
                    )
                else:
                    final_tangent = sub(
                        vec(params["waypoints"][-1]),
                        vec(params["waypoints"][-2]),
                    )
                # The model must provide a schema-complete hint, but the actual
                # outlet direction is dependent on the immutable terminal-axis
                # contract or, when absent, the final required waypoint chord.
                params["final_tangent"] = normalize(final_tangent)
                # These are kernel-policy fields, not planner degrees of
                # freedom.  Always overwrite legacy/LLM values so S-curves use
                # the corrected circular-section sweep frame consistently.
                params["interpolation"] = "bspline"
                # A circular profile is invariant under axial rotation, while
                # a Frenet frame is singular at spline inflections (common in
                # S-curves). Use OCC's corrected sweep frame by default.
                params["frenet"] = False
                required_goal_radii = [
                    float(goal.minimum_curvature_radius)
                    for goal in state.remaining_goals
                    if goal.goal_id in set(draft.affected_goal_ids)
                    and goal.minimum_curvature_radius is not None
                ]
                authored_bound = params.get("minimum_curvature_radius")
                params["minimum_curvature_radius"] = minimum_spline_curvature_radius(
                    float(params["outer_diameter"]),
                    state.modeling_tolerance,
                    max(
                        [
                            *required_goal_radii,
                            *(
                                [float(authored_bound)]
                                if authored_bound is not None
                                else []
                            ),
                        ],
                        default=None,
                    ),
                    enforcement=self.settings.validation_enforcement,  # type: ignore[arg-type]
                )
        elif draft.module == "transition":
            _resolve_section(params, target)
            params.setdefault("diameter_in", params["outer_diameter"])
            params.setdefault("wall_thickness_in", params["wall_thickness"])
            params.setdefault("wall_thickness_out", params["wall_thickness"])
            params.setdefault("offset", (0.0, 0.0, 0.0))
            params["offset"] = vec(params["offset"])
            params.setdefault("connector_type_out", "plain")
            params.setdefault("connector_gender_out", "neutral")
            params.setdefault("connector_standard_out", None)
        elif draft.module == "junction":
            _resolve_section(params, target)
            params["outlets"] = [dict(outlet) for outlet in params["outlets"]]
            for outlet in params["outlets"]:
                outlet.setdefault("connector_type", "plain")
                outlet.setdefault("connector_gender", "neutral")
                outlet.setdefault("connector_standard", None)
            params["branch_count"] = sum(
                1 for outlet in params["outlets"] if outlet.get("role") == "branch"
            )
            params["include_primary_outlet"] = any(
                outlet.get("role") == "primary" for outlet in params["outlets"]
            )
            params["outlet_vectors"] = [
                normalize(vec(outlet["axis"])) for outlet in params["outlets"]
            ]
        elif draft.module == "connect_ports":
            _resolve_section(params, target)
            other = self._find_connectable_port(str(params["other_port_id"]), state)
            params["end_position"] = other.position
            params["end_axis"] = other.axis
            params["waypoints"] = [vec(point) for point in params.get("waypoints", [])]
            if params.get("path_kind") == "circular_arc":
                try:
                    radius, traversal_normal, positive_sweep = (
                        _three_point_arc_geometry(
                            target.position,
                            params["waypoints"][0],
                            other.position,
                        )
                    )
                except ValueError:
                    # Keep the invalid authored midpoint intact so registry
                    # validation can return its normal repair diagnostic.
                    radius = traversal_normal = positive_sweep = None
                if radius is not None and traversal_normal is not None:
                    params["bend_radius"] = radius
                    affected = set(draft.affected_goal_ids)
                    turn_goal = next(
                        (
                            goal
                            for goal in state.remaining_goals
                            if goal.goal_id in affected
                            and goal.type == "turn"
                            and goal.plane_normal is not None
                        ),
                        None,
                    )
                    if turn_goal is not None:
                        goal_normal = normalize(vec(turn_goal.plane_normal))
                        params["plane_normal"] = goal_normal
                        params["sweep_angle"] = (
                            positive_sweep
                            if dot(goal_normal, traversal_normal) >= 0.0
                            else -positive_sweep
                        )
                    else:
                        params["plane_normal"] = traversal_normal
                        params["sweep_angle"] = positive_sweep
            if params.get("path_kind") in {"circular_arc", "spline"}:
                required_goal_radii = [
                    float(goal.minimum_curvature_radius)
                    for goal in state.remaining_goals
                    if goal.goal_id in set(draft.affected_goal_ids)
                    and goal.minimum_curvature_radius is not None
                ]
                authored_bound = params.get("minimum_curvature_radius")
                authored_minimum = max(
                    [
                        *required_goal_radii,
                        *(
                            [float(authored_bound)]
                            if authored_bound is not None
                            else []
                        ),
                    ],
                    default=None,
                )
                if params.get("path_kind") == "circular_arc":
                    # An analytic circular torus supports the horn boundary
                    # R == outer profile radius.  Do not apply the larger
                    # freeform-spline visual reserve to an exact circular arc.
                    params["minimum_curvature_radius"] = max(
                        float(params["outer_diameter"]) / 2.0,
                        float(authored_minimum or 0.0),
                    )
                else:
                    params["minimum_curvature_radius"] = (
                        minimum_spline_curvature_radius(
                            float(params["outer_diameter"]),
                            state.modeling_tolerance,
                            authored_minimum,
                            enforcement=self.settings.validation_enforcement,  # type: ignore[arg-type]
                        )
                    )
            if params.get("path_kind") == "spline":
                params["interpolation"] = "bspline"
                params["frenet"] = False
                params["initial_tangent"] = normalize(vec(params["axis"]))
                params["final_tangent"] = normalize(
                    mul(normalize(vec(other.axis)), -1.0)
                )
        elif draft.module == "terminate":
            _resolve_section(params, target)
        elif draft.module == "inline_component":
            _resolve_section(params, target)
            if params.get("actuator_axis") is not None:
                params["actuator_axis"] = normalize(vec(params["actuator_axis"]))
            if params.get("flange_reference_axis") is not None:
                params["flange_reference_axis"] = normalize(
                    vec(params["flange_reference_axis"])
                )

        return ResolvedAction(
            action_id=action_id,
            target_port=draft.target_port,
            module=draft.module,
            params=params,
            consumed_port_ids=(
                [draft.target_port, str(params["other_port_id"])]
                if draft.module == "connect_ports"
                else [draft.target_port]
            ),
            affected_goal_ids=draft.affected_goal_ids,
            completed_goal_ids=draft.completed_goal_ids,
            satisfied_components=draft.satisfied_components,
        )

    def apply_action(self, action: ResolvedAction, state: PipeState) -> PipeState:
        """resolved action을 새 모듈ㆍ연결ㆍ열린 포트가 반영된 다음 상태로 적용한다."""

        target = self._find_port(action.target_port, state)
        module_id = f"M{state.state_version + 1}"
        module, new_ports = self._create_module_ref(module_id, action, target, state)
        start_anchor_bootstrap = self._is_start_anchor_bootstrap(action, state)
        if start_anchor_bootstrap:
            # START is a construction cursor, not a second physical port in a
            # closed loop. Keep the first module inlet unbound and reserve that
            # real module port for the eventual two-ended closure primitive.
            module.input_bindings.pop("in", None)
        consumed_ids = set(action.consumed_port_ids or [target.id])
        open_ports = [port for port in state.open_ports if port.id not in consumed_ids]
        open_ports.extend(new_ports)
        completed_ids = set(action.completed_goal_ids)
        if completed_ids:
            remaining_goals = [
                goal
                for goal in state.remaining_goals
                if goal.goal_id not in completed_ids
            ]
        elif not action.affected_goal_ids:
            # Compatibility path for dry-run/v1 fixtures only.
            remaining_goals = list(state.remaining_goals[1:])
        else:
            remaining_goals = list(state.remaining_goals)
        next_version = state.state_version + 1
        next_state_id = f"S{next_version}"
        port_nodes = dict(state.port_nodes)
        if not port_nodes:
            port_nodes.update({port.id: port for port in state.open_ports})
        if start_anchor_bootstrap:
            port_nodes.pop("START", None)
        port_nodes.update({port.id: port for port in module.ports.values()})
        reserved_start_anchor = state.reserved_start_anchor
        if start_anchor_bootstrap:
            reserved_start_anchor = module.ports.get("in")
            if reserved_start_anchor is None:
                raise ValueError(
                    "start-anchor bootstrap module must expose one inlet port"
                )
        elif (
            reserved_start_anchor is not None
            and reserved_start_anchor.id in consumed_ids
        ):
            reserved_start_anchor = None
        incidence = [
            *state.module_incidence_edges,
            *[
                ModuleIncidenceEdge(module_id=module_id, port_id=port.id)
                for port in module.ports.values()
            ],
        ]
        connection_edges = list(state.connection_edges)
        for local_name, bound_port_id in module.input_bindings.items():
            module_port = module.ports[local_name]
            connection_edges.append(
                _connection_edge(
                    action,
                    len(connection_edges) + 1,
                    bound_port_id,
                    module_port,
                    port_nodes[bound_port_id],
                )
            )
        return PipeState(
            state_id=next_state_id,
            state_version=next_version,
            contract_digest=state.contract_digest,
            modeling_tolerance=state.modeling_tolerance,
            global_spec=state.global_spec,
            expected_open_ports=state.expected_open_ports,
            expected_open_ports_source=state.expected_open_ports_source,
            required_components=list(state.required_components),
            hard_constraints=list(state.hard_constraints),
            geometric_constraints=[
                item.model_copy(deep=True) for item in state.geometric_constraints
            ],
            design_notes=list(state.design_notes),
            module_measurements={
                module_id: dict(values)
                for module_id, values in state.module_measurements.items()
            },
            placed_modules=[*state.placed_modules, module],
            open_ports=open_ports,
            reserved_start_anchor=reserved_start_anchor,
            port_nodes=port_nodes,
            connection_edges=connection_edges,
            module_incidence_edges=incidence,
            open_port_ids=[port.id for port in open_ports],
            used_ports=[*state.used_ports, *action.consumed_port_ids],
            remaining_goals=remaining_goals,
            action_history=[*state.action_history, action],
        )

    def _find_port(self, port_id: str, state: PipeState) -> Port:
        """상태의 열린 port 중에서 ID로 port를 찾는다."""

        for port in state.open_ports:
            if port.id == port_id:
                return port
        raise ValueError(f"Open port not found: {port_id}")

    def _find_connectable_port(self, port_id: str, state: PipeState) -> Port:
        """연결 가능한 상대 port를 상태·파라미터에서 찾는다."""

        for port in state.open_ports:
            if port.id == port_id:
                return port
        anchor = state.reserved_start_anchor
        if anchor is not None and anchor.id == port_id:
            return anchor
        raise ValueError(f"Connectable port not found: {port_id}")

    def _is_start_anchor_bootstrap(
        self,
        action: ResolvedAction,
        state: PipeState,
    ) -> bool:
        """start anchor 예약 bootstrap 단계인지 판정한다."""

        return bool(
            state.state_version == 0
            and action.target_port == "START"
            and action.module not in {"terminate", "cap_pipe", "connect_ports"}
            and state.reserved_start_anchor is None
            and any(
                goal.type == "connect" and goal.connection_target == "start_anchor"
                for goal in state.remaining_goals
            )
        )

    def _create_module_ref(
        self,
        module_id: str,
        action: ResolvedAction,
        target: Port,
        state: PipeState,
    ) -> tuple[ModuleRef, list[Port]]:
        """배치된 모듈 참조 객체를 생성한다."""

        params = dict(action.params)
        start = vec(params["start_position"])
        axis = normalize(vec(params["axis"]))
        ports: dict[str, Port] = {}
        new_ports: list[Port] = []
        input_bindings: dict[str, str] = {}

        if action.module == "connect_ports":
            section_outer = float(params.get("outer_diameter", target.outer_diameter))
            section_wall = float(params.get("wall_thickness", target.wall_thickness))
            in_a = Port(
                id=f"{module_id}.in_a",
                position=start,
                axis=mul(axis, -1.0),
                outer_diameter=section_outer,
                wall_thickness=section_wall,
                connector_type=target.connector_type,
                connector_gender=_mating_gender(target.connector_gender),
                connector_standard=target.connector_standard,
            )
            end_position = vec(params["end_position"])
            end_axis = normalize(vec(params["end_axis"]))
            other_target = self._find_connectable_port(
                str(params["other_port_id"]), state
            )
            in_b = Port(
                id=f"{module_id}.in_b",
                position=end_position,
                axis=mul(end_axis, -1.0),
                outer_diameter=section_outer,
                wall_thickness=section_wall,
                connector_type=other_target.connector_type,
                connector_gender=_mating_gender(other_target.connector_gender),
                connector_standard=other_target.connector_standard,
            )
            ports.update({"in_a": in_a, "in_b": in_b})
            input_bindings = {
                "in_a": action.target_port,
                "in_b": str(params["other_port_id"]),
            }
        else:
            inlet_outer = float(
                params.get(
                    "diameter_in", params.get("outer_diameter", target.outer_diameter)
                )
            )
            inlet_wall = float(
                params.get(
                    "wall_thickness_in",
                    params.get("wall_thickness", target.wall_thickness),
                )
            )
            in_port = Port(
                id=f"{module_id}.in",
                position=start,
                axis=mul(axis, -1.0),
                outer_diameter=inlet_outer,
                wall_thickness=inlet_wall,
                connector_type=target.connector_type,
                connector_gender=_mating_gender(target.connector_gender),
                connector_standard=target.connector_standard,
            )
            ports["in"] = in_port
            input_bindings = {"in": action.target_port}

        if action.module in {"straight_pipe", "connector_pipe"}:
            end = add(start, mul(axis, float(params["length"])))
            params["end_position"] = end
            out_port = Port(
                id=f"{module_id}.out",
                position=end,
                axis=axis,
                outer_diameter=float(params["outer_diameter"]),
                wall_thickness=float(params["wall_thickness"]),
            )
            ports["out"] = out_port
            new_ports.append(out_port)
        elif action.module == "bend_pipe":
            out_axis = normalize(vec(params["out_axis"]))
            points = arc_points(
                start,
                axis,
                out_axis,
                float(params["bend_radius"]),
                float(params["angle"]),
                int(params.get("segment_resolution", 12)),
            )
            params["path_points"] = points
            params["end_position"] = points[-1]
            # The polyline is only a legacy dry-run sampling aid.  Compute the
            # analytic terminal tangent in the authored turn plane so the port
            # frame is neither chord-resolution dependent nor incorrectly
            # snapped to the desired direction for a partial-angle bend.
            plane_axis = cross(axis, out_axis)
            if length(plane_axis) <= 1e-9:
                plane_axis = choose_perpendicular_axis(axis)
            final_axis = normalize(
                rotate(
                    axis,
                    plane_axis,
                    math.radians(max(1.0, min(abs(float(params["angle"])), 180.0))),
                )
            )
            params["out_axis"] = final_axis
            out_port = Port(
                id=f"{module_id}.out",
                position=points[-1],
                axis=final_axis,
                outer_diameter=float(params["outer_diameter"]),
                wall_thickness=float(params["wall_thickness"]),
            )
            ports["out"] = out_port
            new_ports.append(out_port)
        elif action.module == "reducer_pipe":
            end = add(start, mul(axis, float(params["length"])))
            params["end_position"] = end
            out_port = Port(
                id=f"{module_id}.out",
                position=end,
                axis=axis,
                outer_diameter=float(params["diameter_out"]),
                wall_thickness=float(params["wall_thickness_out"]),
            )
            ports["out"] = out_port
            new_ports.append(out_port)
        elif action.module == "junction_pipe":
            trunk_length = float(params.get("length", target.outer_diameter * 2.0))
            include_primary = bool(params.get("include_primary_outlet", True))
            if include_primary:
                trunk_end = add(start, mul(axis, trunk_length))
                params["trunk_end"] = trunk_end
                out_port = Port(
                    id=f"{module_id}.out",
                    position=trunk_end,
                    axis=axis,
                    outer_diameter=float(params["outer_diameter"]),
                    wall_thickness=float(params["wall_thickness"]),
                )
                ports["out"] = out_port
                new_ports.append(out_port)
            branch_axes = self._branch_axes(axis, params)
            params["outlet_vectors"] = branch_axes
            for index, branch_axis in enumerate(branch_axes, start=1):
                branch_end = add(start, mul(branch_axis, trunk_length))
                params[f"branch_{index}_end"] = branch_end
                branch_port = Port(
                    id=f"{module_id}.out_{index}",
                    position=branch_end,
                    axis=branch_axis,
                    outer_diameter=float(params["outer_diameter"]),
                    wall_thickness=float(params["wall_thickness"]),
                )
                ports[f"out_{index}"] = branch_port
                new_ports.append(branch_port)
        elif action.module == "cap_pipe":
            params["end_position"] = start
            if params.get("end_type") == "open":
                out_port = Port(
                    id=f"{module_id}.out",
                    position=start,
                    axis=axis,
                    outer_diameter=float(params["outer_diameter"]),
                    wall_thickness=float(params["wall_thickness"]),
                )
                ports["out"] = out_port
                new_ports.append(out_port)
        elif action.module == "route":
            kind = params["path_kind"]
            if kind == "line":
                route_axis = normalize(vec(params["direction"]))
                points = [start, add(start, mul(route_axis, float(params["length"])))]
                final_axis = route_axis
            elif kind == "circular_arc":
                points = _arc_points_from_plane(
                    start,
                    axis,
                    normalize(vec(params["plane_normal"])),
                    float(params["bend_radius"]),
                    float(params["sweep_angle"]),
                )
                final_axis = normalize(vec(params["terminal_axis"]))
            else:
                points = [start, *[vec(point) for point in params["waypoints"]]]
                final_axis = normalize(vec(params["final_tangent"]))
            params["path_points"] = points
            params["end_position"] = points[-1]
            params["out_axis"] = final_axis
            out_port = Port(
                id=f"{module_id}.out",
                position=points[-1],
                axis=final_axis,
                outer_diameter=float(params["outer_diameter"]),
                wall_thickness=float(params["wall_thickness"]),
            )
            ports["out"] = out_port
            new_ports.append(out_port)
        elif action.module == "transition":
            end = add(
                add(start, mul(axis, float(params["length"]))), vec(params["offset"])
            )
            params["end_position"] = end
            out_port = Port(
                id=f"{module_id}.out",
                position=end,
                axis=axis,
                outer_diameter=float(params["diameter_out"]),
                wall_thickness=float(params["wall_thickness_out"]),
                connector_type=str(params["connector_type_out"]),
                connector_gender=params["connector_gender_out"],
                connector_standard=params.get("connector_standard_out"),
            )
            ports["out"] = out_port
            new_ports.append(out_port)
        elif action.module == "junction":
            resolved_outlets: list[dict[str, Any]] = []
            branch_index = 0
            primary_seen = False
            for outlet in params["outlets"]:
                resolved_outlet = dict(outlet)
                outlet_axis = normalize(vec(outlet["axis"]))
                outlet_end = add(start, mul(outlet_axis, float(outlet["length"])))
                resolved_outlet["axis"] = outlet_axis
                resolved_outlet["end_position"] = outlet_end
                role = outlet["role"]
                if role == "primary" and not primary_seen:
                    port_name = "out"
                    primary_seen = True
                    params["trunk_end"] = outlet_end
                else:
                    branch_index += 1
                    port_name = f"out_{branch_index}"
                    params[f"branch_{branch_index}_end"] = outlet_end
                out_port = Port(
                    id=f"{module_id}.{port_name}",
                    position=outlet_end,
                    axis=outlet_axis,
                    outer_diameter=float(outlet["outer_diameter"]),
                    wall_thickness=float(outlet["wall_thickness"]),
                    connector_type=str(outlet["connector_type"]),
                    connector_gender=outlet["connector_gender"],
                    connector_standard=outlet.get("connector_standard"),
                )
                ports[port_name] = out_port
                new_ports.append(out_port)
                resolved_outlets.append(resolved_outlet)
            params["outlets"] = resolved_outlets
        elif action.module == "connect_ports":
            params["path_points"] = (
                []
                if params.get("path_kind") == "seam"
                else [
                    start,
                    *[vec(point) for point in params["waypoints"]],
                    vec(params["end_position"]),
                ]
            )
        elif action.module == "terminate":
            params["end_position"] = start
        elif action.module == "inline_component":
            end = add(start, mul(axis, float(params["length"])))
            params["end_position"] = end
            out_port = Port(
                id=f"{module_id}.out",
                position=end,
                axis=axis,
                outer_diameter=float(params["outer_diameter"]),
                wall_thickness=float(params["wall_thickness"]),
                connector_type=str(params["connector_type_out"]),
                connector_gender=params["connector_gender_out"],
                connector_standard=params.get("connector_standard_out"),
            )
            ports["out"] = out_port
            new_ports.append(out_port)

        module = ModuleRef(
            id=module_id,
            type=action.module,
            schema_version=2
            if action.module
            in {
                "route",
                "transition",
                "junction",
                "connect_ports",
                "terminate",
                "inline_component",
            }
            else 1,
            geometry_id=(
                None
                if action.module == "connect_ports"
                and params.get("path_kind") == "seam"
                else f"solid_{module_id}"
            ),
            params=params,
            ports=ports,
            input_bindings=input_bindings,
        )
        return module, new_ports

    def _resolve_bend_out_axis(
        self,
        turn_direction: str | None,
        current_axis: Vector,
    ) -> Vector:
        """굽힘 출구 축 방향을 결정론적으로 계산한다."""

        current = normalize(current_axis)
        fallback = (0.0, 0.0, 1.0) if abs(current[2]) < 0.85 else (1.0, 0.0, 0.0)
        out_axis = direction_to_vector(turn_direction, default=fallback)
        out_axis = normalize(out_axis)
        if abs(dot(current, out_axis)) > 0.98:
            out_axis = normalize(fallback)
        if abs(dot(current, out_axis)) > 0.98:
            out_axis = choose_perpendicular_axis(current)
        return out_axis

    def _branch_axes(self, axis: Vector, params: dict[str, Any]) -> list[Vector]:
        """분기 outlet 축 집합을 계산한다."""

        explicit_vectors = _explicit_outlet_vectors(params)
        branch_count = int(params.get("branch_count") or len(explicit_vectors) or 2)
        if branch_count <= 0:
            return []
        if explicit_vectors:
            return explicit_vectors[:branch_count]

        directions = list(params.get("required_outlet_directions") or [])
        candidates: list[Vector] = []
        if directions:
            candidates.extend(
                normalize(direction_to_vector(direction, default=axis))
                for direction in directions[:branch_count]
            )
            if len(candidates) >= branch_count:
                return candidates

        raw_angles = params.get("branch_angles") or [45.0, -45.0]
        base_axis = normalize(
            direction_to_vector(params.get("direction"), default=axis)
        )
        side_axis = choose_perpendicular_axis(base_axis)
        for angle in raw_angles[: max(0, branch_count - len(candidates))]:
            angle_rad = math.radians(float(angle))
            along = mul(base_axis, math.cos(angle_rad))
            side = mul(side_axis, math.sin(angle_rad))
            candidates.append(normalize(add(along, side)))
        while len(candidates) < branch_count:
            spread_angle = math.radians(360.0 * len(candidates) / branch_count)
            along = mul(base_axis, math.cos(spread_angle))
            side = mul(side_axis, math.sin(spread_angle))
            candidates.append(normalize(add(along, side)))
        return candidates


def _resolve_section(params: dict[str, Any], target: Port) -> None:
    """단면 상속/명시 파라미터를 실제 단면 값으로 해석한다."""

    source = params.get("section_source")
    if source == "inherit_target":
        params["outer_diameter"] = target.outer_diameter
        params["wall_thickness"] = target.wall_thickness
        return
    if source == "explicit":
        params["outer_diameter"] = float(params["outer_diameter"])
        params["wall_thickness"] = float(params["wall_thickness"])
        return
    raise ValueError("section_source must be explicitly selected")


def _three_point_arc_geometry(
    start: Vector,
    middle: Vector,
    end: Vector,
) -> tuple[float, Vector, float]:
    """three_point_arc_geometry를 계산하거나 반환한다."""

    start = vec(start)
    middle = vec(middle)
    end = vec(end)
    chord_a = sub(middle, start)
    chord_b = sub(end, start)
    normal_raw = cross(chord_a, chord_b)
    normal_squared = dot(normal_raw, normal_raw)
    if normal_squared <= 1e-12:
        raise ValueError(
            "circular_arc connect_ports start, midpoint, and end must be non-collinear"
        )
    numerator = add(
        mul(cross(chord_b, normal_raw), dot(chord_a, chord_a)),
        mul(cross(normal_raw, chord_a), dot(chord_b, chord_b)),
    )
    center = add(start, mul(numerator, 1.0 / (2.0 * normal_squared)))
    radii = [sub(point, center) for point in (start, middle, end)]
    radius = length(radii[0])

    def positive_sweep(normal: Vector) -> float:
        """양수 치수 목록을 수집하거나 검사한다."""

        total = 0.0
        for left, right in zip(radii, radii[1:]):
            total += (
                math.atan2(
                    dot(normal, cross(left, right)),
                    dot(left, right),
                )
                % math.tau
            )
        return total

    candidates = [normalize(normal_raw), mul(normalize(normal_raw), -1.0)]
    valid = [
        (positive_sweep(normal), normal)
        for normal in candidates
        if positive_sweep(normal) <= math.tau + 1e-9
    ]
    if not valid:
        raise ValueError("circular_arc connect_ports has no finite directed sweep")
    sweep_radians, traversal_normal = min(valid, key=lambda item: item[0])
    return radius, traversal_normal, math.degrees(sweep_radians)


def _connection_edge(
    action: ResolvedAction,
    edge_index: int,
    bound_port_id: str,
    module_port: Port,
    bound_port: Port,
) -> ConnectionEdge:
    """두 port 사이의 연결 간선 레코드를 만든다."""

    position_error = length(sub(vec(bound_port.position), vec(module_port.position)))
    alignment = -dot(
        normalize(vec(bound_port.axis)),
        normalize(vec(module_port.axis)),
    )
    clamped_alignment = max(-1.0, min(1.0, alignment))
    outer_radius_a = bound_port.outer_diameter / 2.0
    outer_radius_b = module_port.outer_diameter / 2.0
    inner_radius_a = bound_port.inner_diameter / 2.0
    inner_radius_b = module_port.inner_diameter / 2.0
    return ConnectionEdge(
        edge_id=f"E{edge_index}",
        port_a_id=bound_port_id,
        port_b_id=module_port.id,
        action_id=action.action_id,
        position_error=position_error,
        anti_parallel_axis_dot=alignment,
        axis_angle_error=math.acos(clamped_alignment),
        od_error=abs(bound_port.outer_diameter - module_port.outer_diameter),
        id_error=abs(bound_port.inner_diameter - module_port.inner_diameter),
        wall_error=abs(bound_port.wall_thickness - module_port.wall_thickness),
        outer_rim_error=circular_rim_mismatch(
            position_error,
            outer_radius_a,
            outer_radius_b,
            alignment,
        ),
        inner_rim_error=circular_rim_mismatch(
            position_error,
            inner_radius_a,
            inner_radius_b,
            alignment,
        ),
        connector_type_match=(bound_port.connector_type == module_port.connector_type),
        connector_gender_match=_connector_genders_compatible(
            bound_port.connector_gender, module_port.connector_gender
        ),
        connector_standard_match=(
            bound_port.connector_standard == module_port.connector_standard
        ),
        engagement=0.0,
    )


def _mating_gender(gender: str) -> str:
    """커넥터 mating gender를 정규화한다."""

    return {"male": "female", "female": "male"}.get(gender, "neutral")


def _connector_genders_compatible(left: str, right: str) -> bool:
    """두 커넥터 gender가 결합 가능한지 검사한다."""

    return _mating_gender(left) == right


def _arc_points_from_plane(
    start: Vector,
    tangent: Vector,
    plane_normal: Vector,
    radius: float,
    sweep_angle: float,
    segments: int = 32,
) -> list[Vector]:
    """평면 법선 기반 원호 표본점을 생성한다."""

    tangent = normalize(tangent)
    normal = normalize(plane_normal)
    raw_radial = cross(tangent, normal)
    if length(raw_radial) <= 1e-9:
        raise ValueError("plane_normal must not be parallel to route tangent")
    radial = normalize(raw_radial)
    signed_radius = radius if sweep_angle >= 0 else -radius
    center = sub(start, mul(radial, signed_radius))
    start_radius = sub(start, center)
    count = max(8, int(segments * abs(sweep_angle) / 90.0))
    return [
        add(
            center,
            rotate(
                start_radius,
                normal,
                math.radians(sweep_angle) * index / count,
            ),
        )
        for index in range(count + 1)
    ]


def _without_none(params: dict[str, Any]) -> dict[str, Any]:
    """None 값을 제거한 dict를 반환한다."""

    return {key: value for key, value in params.items() if value is not None}


def _backfill_junction_goal_params(params: dict[str, Any], goal: Any) -> None:
    """junction goal 누락 파라미터를 채운다."""

    if goal.branch_count is not None:
        params.setdefault("branch_count", goal.branch_count)
    if goal.branch_angles:
        params.setdefault("branch_angles", list(goal.branch_angles))
    if goal.required_outlet_directions:
        params.setdefault(
            "required_outlet_directions", list(goal.required_outlet_directions)
        )
    if goal.required_outlet_vectors:
        params.setdefault("required_outlet_vectors", list(goal.required_outlet_vectors))
        params.setdefault("outlet_vectors", list(goal.required_outlet_vectors))
    if goal.include_primary_outlet is not None:
        params.setdefault("include_primary_outlet", goal.include_primary_outlet)
    if goal.junction_style is not None:
        params.setdefault("junction_style", goal.junction_style)
    if goal.direction is not None:
        params.setdefault("direction", goal.direction)


def _explicit_outlet_vectors(params: dict[str, Any]) -> list[Vector]:
    """명시된 outlet 방향 벡터 목록을 정규화한다."""

    raw_vectors = (
        params.get("outlet_vectors") or params.get("required_outlet_vectors") or []
    )
    vectors: list[Vector] = []
    if not isinstance(raw_vectors, list):
        return vectors
    for raw_vector in raw_vectors:
        try:
            candidate = normalize(vec(raw_vector))
        except (TypeError, ValueError):
            continue
        if length(candidate) > 0:
            vectors.append(candidate)
    return vectors


def _int_or_default(value: Any, default: int) -> int:
    """정수 변환에 실패하면 기본값을 반환한다."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default
