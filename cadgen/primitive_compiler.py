"""Compile LLM-selected intent primitives into host-owned action geometry.

The semantic LLM has already chosen the ordered goal/primitive program in
``IntentResult``.  This compiler therefore never invents a new topology.  It
only binds the next selected goal to the current construction port and derives
resolver-owned vectors, section inheritance and closure waypoints.  Unsupported
or genuinely ambiguous topology returns ``None`` so the bounded search layer may
ask an LLM to choose among discrete alternatives.
"""

from __future__ import annotations

from cadgen.schemas import ActionDraft, Goal, PipeState, Port
import math

from cadgen.vector import (
    add,
    canonical_circular_arc_frame,
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


def _goal_ids(goal: Goal) -> list[str]:
    return [goal.goal_id] if goal.goal_id else []


def _single_target(state: PipeState) -> Port | None:
    return state.open_ports[0] if len(state.open_ports) == 1 else None


def _route_draft(goal: Goal, target: Port) -> ActionDraft | None:
    affected = _goal_ids(goal)
    if goal.type == "move" or (goal.type == "route" and goal.path_kind == "line"):
        if goal.length is None:
            return None
        params: dict[str, object] = {
            "path_kind": "line",
            "section_source": "inherit_target",
            "length": float(goal.length),
        }
        if goal.direction is not None:
            params["direction"] = direction_to_vector(goal.direction)
        return ActionDraft(
            target_port=target.id,
            module="route",
            params=params,
            catalog_schema_version=2,
            affected_goal_ids=affected,
            completed_goal_ids=affected,
            rationale="Host compiler bound the LLM-selected line primitive.",
        )

    is_arc = goal.type == "turn" or (
        goal.type == "route" and goal.path_kind == "circular_arc"
    )
    if is_arc:
        if goal.angle is None or goal.bend_radius is None or goal.plane_normal is None:
            return None
        return ActionDraft(
            target_port=target.id,
            module="route",
            params={
                "path_kind": "circular_arc",
                "section_source": "inherit_target",
                "bend_radius": float(goal.bend_radius),
                "sweep_angle": float(goal.angle),
                "plane_normal": tuple(float(value) for value in goal.plane_normal),
            },
            catalog_schema_version=2,
            affected_goal_ids=affected,
            completed_goal_ids=affected,
            rationale="Host compiler derived the frame of the LLM-selected arc.",
        )

    if goal.type == "route" and goal.path_kind == "spline":
        if len(goal.required_waypoints) < 2:
            return None
        return ActionDraft(
            target_port=target.id,
            module="route",
            params={
                "path_kind": "spline",
                "section_source": "inherit_target",
                "waypoint_frame": goal.waypoint_frame or "global",
                "waypoints": [
                    tuple(float(value) for value in point)
                    for point in goal.required_waypoints
                ],
            },
            catalog_schema_version=2,
            affected_goal_ids=affected,
            completed_goal_ids=affected,
            rationale="Host compiler bound the LLM-selected spline anchors.",
        )
    return None


def _connect_draft(goal: Goal, target: Port, state: PipeState) -> ActionDraft | None:
    if goal.connection_target == "start_anchor":
        other = state.reserved_start_anchor
    else:
        other = next((port for port in state.open_ports if port.id != target.id), None)
    if other is None:
        return None
    affected = _goal_ids(goal)
    delta = sub(vec(other.position), vec(target.position))
    distance = length(delta)
    terminal_tangent = normalize(mul(vec(other.axis), -1.0))
    initial_tangent = normalize(vec(target.axis))
    if distance <= state.modeling_tolerance:
        if dot(initial_tangent, terminal_tangent) < 1.0 - 1e-7:
            return None
        params: dict[str, object] = {
            "other_port_id": other.id,
            "path_kind": "seam",
            "section_source": "inherit_target",
        }
    else:
        chord = normalize(delta)
        if (
            dot(initial_tangent, chord) >= 1.0 - 1e-7
            and dot(terminal_tangent, chord) >= 1.0 - 1e-7
        ):
            params = {
                "other_port_id": other.id,
                "path_kind": "line",
                "section_source": "inherit_target",
            }
        else:
            handle = distance / 3.0
            first = add(vec(target.position), mul(initial_tangent, handle))
            second = sub(vec(other.position), mul(terminal_tangent, handle))
            if length(sub(second, first)) <= state.modeling_tolerance:
                midpoint = mul(add(vec(target.position), vec(other.position)), 0.5)
                waypoints = [first, midpoint, second]
            else:
                waypoints = [first, second]
            params = {
                "other_port_id": other.id,
                "path_kind": "spline",
                "section_source": "inherit_target",
                "waypoints": waypoints,
            }
    return ActionDraft(
        target_port=target.id,
        module="connect_ports",
        params=params,
        catalog_schema_version=2,
        affected_goal_ids=affected,
        completed_goal_ids=affected,
        rationale="Host compiler synthesized the closure for the LLM-selected connect primitive.",
    )


def _terminal_arc_connect_draft(
    state: PipeState,
    target: Port,
) -> ActionDraft | None:
    """Fuse a final authored turn and START closure into one analytic arc."""

    if len(state.remaining_goals) < 2 or state.reserved_start_anchor is None:
        return None
    turn = state.remaining_goals[0]
    connect = state.remaining_goals[1]
    if not (
        turn.type == "turn"
        and turn.angle is not None
        and turn.bend_radius is not None
        and turn.plane_normal is not None
        and connect.type == "connect"
        and connect.connection_target == "start_anchor"
        and connect.depends_on_goal_ids
        and turn.goal_id in connect.depends_on_goal_ids
    ):
        return None
    try:
        normal, tangent, terminal = canonical_circular_arc_frame(
            vec(target.axis),
            vec(turn.plane_normal),
            float(turn.angle),
        )
    except ValueError:
        return None
    radius = float(turn.bend_radius)
    signed_radius = radius if float(turn.angle) >= 0.0 else -radius
    radial = normalize(cross(tangent, normal))
    center = sub(vec(target.position), mul(radial, signed_radius))
    start_radius = sub(vec(target.position), center)
    endpoint = add(
        center,
        rotate(start_radius, normal, math.radians(float(turn.angle))),
    )
    anchor = state.reserved_start_anchor
    expected_terminal = normalize(mul(vec(anchor.axis), -1.0))
    if (
        length(sub(endpoint, vec(anchor.position))) > state.modeling_tolerance
        or dot(terminal, expected_terminal) < 1.0 - 1e-7
    ):
        return None
    midpoint = add(
        center,
        rotate(start_radius, normal, math.radians(float(turn.angle)) / 2.0),
    )
    affected = [
        goal_id for goal_id in (turn.goal_id, connect.goal_id) if goal_id is not None
    ]
    return ActionDraft(
        target_port=target.id,
        module="connect_ports",
        params={
            "other_port_id": anchor.id,
            "path_kind": "circular_arc",
            "section_source": "inherit_target",
            "waypoints": [midpoint],
        },
        catalog_schema_version=2,
        affected_goal_ids=affected,
        completed_goal_ids=affected,
        rationale=(
            "Host compiler fused the LLM-selected terminal turn and closure "
            "into one analytic two-port arc."
        ),
    )


def compile_next_action(state: PipeState) -> ActionDraft | None:
    """Compile one unambiguous LLM-selected goal into a schema-v2 action."""

    if not state.remaining_goals:
        return None
    goal = state.remaining_goals[0]
    target = _single_target(state)
    if target is None:
        return None

    terminal_arc = _terminal_arc_connect_draft(state, target)
    if terminal_arc is not None:
        return terminal_arc

    route = _route_draft(goal, target)
    if route is not None:
        return route
    if goal.type == "connect":
        return _connect_draft(goal, target, state)
    if goal.type == "diameter_change":
        if goal.diameter_out is None or goal.transition_length is None:
            return None
        affected = _goal_ids(goal)
        return ActionDraft(
            target_port=target.id,
            module="transition",
            params={
                "section_source": "inherit_target",
                "diameter_out": float(goal.diameter_out),
                "wall_thickness_out": (
                    float(goal.wall_thickness_out)
                    if goal.wall_thickness_out is not None
                    else None
                ),
                "length": float(goal.transition_length),
                "offset": tuple(
                    float(value) for value in (goal.offset or (0.0, 0.0, 0.0))
                ),
            },
            catalog_schema_version=2,
            affected_goal_ids=affected,
            completed_goal_ids=affected,
            rationale="Host compiler bound the LLM-selected transition primitive.",
        )
    if goal.type == "end" and goal.end_type in {"cap", "plug"}:
        if goal.termination_thickness is None:
            return None
        affected = _goal_ids(goal)
        return ActionDraft(
            target_port=target.id,
            module="terminate",
            params={
                "section_source": "inherit_target",
                "termination_type": goal.end_type,
                "thickness": float(goal.termination_thickness),
            },
            catalog_schema_version=2,
            affected_goal_ids=affected,
            completed_goal_ids=affected,
            rationale="Host compiler bound the LLM-selected termination primitive.",
        )
    return None


__all__ = ["compile_next_action"]
