"""LLM이 고른 intent primitive를 host 소유 action geometry로 컴파일한다.

Intent Agent가 이미 순서 있는 goal/primitive program을 선택했으므로, 이 모듈은
새 topology를 만들지 않는다. 다음 goal을 현재 시공 port에 결합하고 접선·단면
상속·폐합 waypoint·junction outlet만 결정론적으로 계산한다.

PAPR (2026-07-12):
- open port가 여러 개여도 primary/축 정렬로 유일하면 host가 선택한다.
- branch goal은 junction draft로 host compile한다.
- 지원하지 않거나 모호한 이산 선택만 ``None``을 반환한다 (연속 수치 LLM 위임 아님).
"""

from __future__ import annotations

import math
from typing import Any

from cadgen.port_algebra import select_construction_port
from cadgen.primitive_action_catalog import SUPPORTED_INLINE_COMPONENTS
from cadgen.typed_data_models import ActionDraft, Goal, PipeState, Port
from cadgen.vector3_math import (
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

Vector3 = tuple[float, float, float]
OutletDict = dict[str, object]

_CATALOG_V2 = 2
_SECTION_INHERIT = "inherit_target"
_RETRACE_DOT = 0.999
_COLLINEAR_DOT_TOL = 1e-6
_TERMINAL_TANGENT_DOT = 1.0 - 1e-7


# ---------------------------------------------------------------------------
# Small builders
# ---------------------------------------------------------------------------


def _goal_ids(goal: Goal) -> list[str]:
    """goal_id가 있으면 단일 원소 목록으로, 없으면 빈 목록으로 반환한다."""

    return [goal.goal_id] if goal.goal_id else []


def _v2_draft(
    *,
    target: Port,
    module: str,
    params: dict[str, object],
    goal: Goal | None = None,
    goal_ids: list[str] | None = None,
    rationale: str,
) -> ActionDraft:
    """schema-v2 ActionDraft를 공통 규칙으로 만든다."""

    affected = goal_ids if goal_ids is not None else (_goal_ids(goal) if goal else [])
    return ActionDraft(
        target_port=target.id,
        module=module,
        params=params,
        catalog_schema_version=_CATALOG_V2,
        affected_goal_ids=affected,
        completed_goal_ids=list(affected),
        rationale=rationale,
        authorship="host_compile",
    )


def _outlet(
    *,
    role: str,
    axis: Vector3,
    length: float,
    outer_diameter: float,
    wall_thickness: float,
) -> OutletDict:
    """junction outlet 파라미터 dict를 만든다."""

    return {
        "role": role,
        "axis": axis,
        "length": float(length),
        "outer_diameter": float(outer_diameter),
        "wall_thickness": float(wall_thickness),
    }


def _primary_trunk_outlet(
    trunk: Vector3,
    target: Port,
    length: float,
) -> OutletDict:
    """trunk 정렬 primary outlet을 만든다."""

    return _outlet(
        role="primary",
        axis=trunk,
        length=length,
        outer_diameter=float(target.outer_diameter),
        wall_thickness=float(target.wall_thickness),
    )


def _default_branch_plane(trunk: Vector3) -> Vector3:
    """trunk에 수직인 기본 분기 평면을 고른다."""

    reference = (0.0, 0.0, 1.0) if abs(trunk[2]) < 0.9 else (0.0, 1.0, 0.0)
    plane = cross(trunk, reference)
    if length(plane) <= 1e-12:
        plane = cross(trunk, (1.0, 0.0, 0.0))
    return normalize(plane)


def _section_defaults(goal: Goal, target: Port) -> tuple[float, float, float]:
    """branch 기본 외경·두께·길이를 반환한다."""

    od = float(
        goal.branch_outer_diameter
        if goal.branch_outer_diameter is not None
        else target.outer_diameter
    )
    wall = float(
        goal.branch_wall_thickness
        if goal.branch_wall_thickness is not None
        else target.wall_thickness
    )
    length_mm = float(goal.length) if goal.length is not None else 40.0
    # FreeCAD hub checks require arms to extend beyond the hub envelope.
    min_arm = max(od * 3.5, 24.0)
    length_mm = max(length_mm, min_arm)
    return od, wall, length_mm


def _include_primary(goal: Goal) -> bool:
    """branch goal이 primary outlet을 포함하는지 판정한다."""

    if goal.include_primary_outlet is not None:
        return bool(goal.include_primary_outlet)
    return not bool(goal.required_outlet_vectors or goal.required_outlets)


# ---------------------------------------------------------------------------
# Goal → draft compilers
# ---------------------------------------------------------------------------


def _route_draft(goal: Goal, target: Port) -> ActionDraft | None:
    """line/arc/spline route goal을 schema-v2 ActionDraft로 변환한다."""

    if goal.type == "move" or (goal.type == "route" and goal.path_kind == "line"):
        if goal.length is None:
            return None
        params: dict[str, object] = {
            "path_kind": "line",
            "section_source": _SECTION_INHERIT,
            "length": float(goal.length),
        }
        # Only author a world-frame direction when it continues the open port.
        # Misaligned absolute directions (common after diagonal Y primaries) are
        # dropped so resolve uses port-frame continuity by construction.
        if goal.direction is not None:
            desired = normalize(vec(direction_to_vector(goal.direction)))
            port_axis = normalize(vec(target.axis))
            if dot(desired, port_axis) >= 0.85:
                params["direction"] = desired
            # else: omit — pipe_state_engine inherits target.axis
        return _v2_draft(
            target=target,
            module="route",
            params=params,
            goal=goal,
            rationale="Host compiler bound the LLM-selected line primitive.",
        )

    is_arc = goal.type == "turn" or (
        goal.type == "route" and goal.path_kind == "circular_arc"
    )
    if is_arc:
        if goal.angle is None or goal.bend_radius is None or goal.plane_normal is None:
            return None
        return _v2_draft(
            target=target,
            module="route",
            params={
                "path_kind": "circular_arc",
                "section_source": _SECTION_INHERIT,
                "bend_radius": float(goal.bend_radius),
                "sweep_angle": float(goal.angle),
                "plane_normal": tuple(float(value) for value in goal.plane_normal),
            },
            goal=goal,
            rationale="Host compiler derived the frame of the LLM-selected arc.",
        )

    if goal.type == "route" and goal.path_kind == "spline":
        if len(goal.required_waypoints) < 2:
            return None
        return _v2_draft(
            target=target,
            module="route",
            params={
                "path_kind": "spline",
                "section_source": _SECTION_INHERIT,
                "waypoint_frame": goal.waypoint_frame or "global",
                "waypoints": [
                    tuple(float(value) for value in point)
                    for point in goal.required_waypoints
                ],
            },
            goal=goal,
            rationale="Host compiler bound the LLM-selected spline anchors.",
        )
    return None


def _connect_draft(goal: Goal, target: Port, state: PipeState) -> ActionDraft | None:
    """두 port 사이 폐합을 seam/line/spline connect draft로 합성한다."""

    if goal.connection_target == "start_anchor":
        other = state.reserved_start_anchor
    else:
        other = next((port for port in state.open_ports if port.id != target.id), None)
    if other is None:
        return None

    delta = sub(vec(other.position), vec(target.position))
    distance = length(delta)
    base = {
        "other_port_id": other.id,
        "section_source": _SECTION_INHERIT,
    }

    if distance <= state.modeling_tolerance:
        return _v2_draft(
            target=target,
            module="connect_ports",
            params={**base, "path_kind": "seam"},
            goal=goal,
            rationale="Host compiler sealed co-located open ports.",
        )

    target_axis = normalize(vec(target.axis))
    collinear = abs(abs(dot(normalize(delta), target_axis)) - 1.0) <= _COLLINEAR_DOT_TOL
    if collinear:
        return _v2_draft(
            target=target,
            module="connect_ports",
            params={**base, "path_kind": "line"},
            goal=goal,
            rationale="Host compiler closed ports with a collinear line segment.",
        )

    # Non-collinear: host-authored midpoint so FreeCAD/spline has a real control
    # point (empty waypoints were a common FreeCAD thrash source).
    chord = normalize(delta)
    mid = mul(add(vec(target.position), vec(other.position)), 0.5)
    lateral = cross(target_axis, chord)
    if length(lateral) > 1e-6:
        # Slight offset off the chord in the local bend plane.
        mid = add(mid, mul(normalize(lateral), distance * 0.12))
    return _v2_draft(
        target=target,
        module="connect_ports",
        params={**base, "path_kind": "spline", "waypoints": [mid]},
        goal=goal,
        rationale="Host compiler closed ports with a smooth connector.",
    )


def _terminal_arc_connect_draft(state: PipeState, target: Port) -> ActionDraft | None:
    """마지막 turn+start_anchor connect를 하나의 analytic arc connect로 합친다."""

    if len(state.remaining_goals) < 2:
        return None
    turn, connect = state.remaining_goals[0], state.remaining_goals[1]
    if turn.type != "turn" or connect.type != "connect":
        return None
    if connect.connection_target != "start_anchor":
        return None
    anchor = state.reserved_start_anchor
    if anchor is None:
        return None
    if turn.angle is None or turn.bend_radius is None or turn.plane_normal is None:
        return None

    tangent = normalize(vec(target.axis))
    sweep = float(turn.angle)
    radius = float(turn.bend_radius)
    try:
        normal, initial, terminal = canonical_circular_arc_frame(
            tangent,
            vec(turn.plane_normal),
            sweep,
        )
    except ValueError:
        return None

    radial = normalize(cross(initial, normal))
    signed_radius = radius if sweep >= 0.0 else -radius
    center = sub(vec(target.position), mul(radial, signed_radius))
    start_radius = sub(vec(target.position), center)
    endpoint = add(center, rotate(start_radius, normal, math.radians(sweep)))
    expected_terminal = normalize(mul(vec(anchor.axis), -1.0))
    if (
        length(sub(endpoint, vec(anchor.position))) > state.modeling_tolerance
        or dot(terminal, expected_terminal) < _TERMINAL_TANGENT_DOT
    ):
        return None

    midpoint = add(center, rotate(start_radius, normal, math.radians(sweep) / 2.0))
    affected = [gid for gid in (turn.goal_id, connect.goal_id) if gid is not None]
    return _v2_draft(
        target=target,
        module="connect_ports",
        params={
            "other_port_id": anchor.id,
            "path_kind": "circular_arc",
            "section_source": _SECTION_INHERIT,
            "waypoints": [midpoint],
        },
        goal_ids=affected,
        rationale=(
            "Host compiler fused the LLM-selected terminal turn and closure "
            "into one analytic two-port arc."
        ),
    )


def _build_branch_outlets(
    goal: Goal,
    target: Port,
    *,
    include_primary: bool,
    default_od: float,
    default_wall: float,
    default_length: float,
    trunk: Vector3,
) -> list[OutletDict] | None:
    """branch goal 표현식에서 outlet 목록을 만든다. 실패 시 None."""

    outlets: list[OutletDict] = []

    if goal.required_outlets:
        for index, outlet in enumerate(goal.required_outlets):
            axis = normalize(vec(outlet.axis))
            role = "primary" if include_primary and index == 0 else "branch"
            if include_primary and index == 0 and abs(dot(axis, trunk)) < 0.2:
                role = "branch"
            outlets.append(
                _outlet(
                    role=role,
                    axis=axis,
                    length=float(
                        outlet.length if outlet.length is not None else default_length
                    ),
                    outer_diameter=float(
                        outlet.outer_diameter
                        if outlet.outer_diameter is not None
                        else default_od
                    ),
                    wall_thickness=float(
                        outlet.wall_thickness
                        if outlet.wall_thickness is not None
                        else default_wall
                    ),
                )
            )
        if include_primary and not any(item["role"] == "primary" for item in outlets):
            outlets.insert(0, _primary_trunk_outlet(trunk, target, default_length))
        return outlets

    if goal.required_outlet_vectors:
        if include_primary:
            outlets.append(_primary_trunk_outlet(trunk, target, default_length))
        for item in goal.required_outlet_vectors:
            outlets.append(
                _outlet(
                    role="branch",
                    axis=normalize(vec(item)),
                    length=default_length,
                    outer_diameter=default_od,
                    wall_thickness=default_wall,
                )
            )
        return outlets

    if goal.required_outlet_directions:
        if include_primary:
            outlets.append(_primary_trunk_outlet(trunk, target, default_length))
        for direction in goal.required_outlet_directions:
            outlets.append(
                _outlet(
                    role="branch",
                    axis=normalize(vec(direction_to_vector(direction))),
                    length=default_length,
                    outer_diameter=default_od,
                    wall_thickness=default_wall,
                )
            )
        return outlets

    if goal.branch_angles:
        plane = (
            normalize(vec(goal.branch_plane_normal))
            if goal.branch_plane_normal is not None
            else _default_branch_plane(trunk)
        )
        if include_primary:
            outlets.append(_primary_trunk_outlet(trunk, target, default_length))
        for angle in goal.branch_angles:
            outlets.append(
                _outlet(
                    role="branch",
                    axis=normalize(rotate(trunk, plane, math.radians(float(angle)))),
                    length=default_length,
                    outer_diameter=default_od,
                    wall_thickness=default_wall,
                )
            )
        return outlets

    return None


def _normalize_binary_outlets(
    outlets: list[OutletDict],
    *,
    include_primary: bool,
    trunk: Vector3,
) -> list[OutletDict] | None:
    """binary Y 계약(outlet 2개)과 non-retrace 축 규칙을 적용한다."""

    if len(outlets) != 2:
        primaries = [item for item in outlets if item["role"] == "primary"]
        branches = [item for item in outlets if item["role"] == "branch"]
        if include_primary and primaries and branches:
            outlets = [primaries[0], branches[0]]
        elif len(outlets) > 2:
            outlets = outlets[:2]
            if not include_primary:
                outlets[0]["role"] = "branch"
            outlets[1]["role"] = "branch"
        else:
            return None

    reverse_trunk = mul(trunk, -1.0)
    for outlet in outlets:
        axis = normalize(vec(outlet["axis"]))  # type: ignore[arg-type]
        if dot(axis, reverse_trunk) > _RETRACE_DOT:
            return None
        outlet["axis"] = axis
    return outlets


def _branch_draft(goal: Goal, target: Port, state: PipeState) -> ActionDraft | None:
    """branch goal을 schema-v2 junction draft로 컴파일한다."""

    del state  # reserved for future multi-port branch binding
    if goal.type != "branch":
        return None

    include_primary = _include_primary(goal)
    default_od, default_wall, default_length = _section_defaults(goal, target)
    trunk = normalize(vec(target.axis))
    outlets = _build_branch_outlets(
        goal,
        target,
        include_primary=include_primary,
        default_od=default_od,
        default_wall=default_wall,
        default_length=default_length,
        trunk=trunk,
    )
    if outlets is None:
        return None
    outlets = _normalize_binary_outlets(
        outlets, include_primary=include_primary, trunk=trunk
    )
    if outlets is None:
        return None

    # Honor the LLM/intent style contract. smooth_hub must compile to fillet
    # (not hard) or static validation raises BRANCH_STYLE_MISMATCH and the
    # host would otherwise replay the same illegal draft forever.
    params = _junction_style_params(goal, default_od=default_od)
    params["section_source"] = _SECTION_INHERIT
    # Guarantee arm length exceeds hub so OCC fuse has clear free ends.
    max_hub = float(params.get("max_hub_radius") or default_od)
    min_len = max(max_hub * 2.5, default_od * 3.5, 24.0)
    for outlet in outlets:
        length_val = float(outlet.get("length") or 0.0)
        if length_val < min_len:
            outlet["length"] = min_len
    params["outlets"] = outlets
    return _v2_draft(
        target=target,
        module="junction",
        params=params,
        goal=goal,
        rationale="Host compiler bound the LLM-selected binary junction primitive.",
    )


def _junction_style_params(goal: Goal, *, default_od: float) -> dict[str, object]:
    """Map intent junction_style to catalog-legal blend params.

    - smooth_hub → blend_mode=fillet + positive radii (required by catalog)
    - hard_fuse / default hard → blend_mode=hard and no blend radii
    FreeCAD may still fall back to a hard Boolean solid if fillet OCC fails;
    static style contract is satisfied by the authored blend_mode.

    Hub sizing is intentionally compact (near pipe OD). Oversized hubs
    (e.g. max(od*1.25, 12) on OD=12 → 15) plus acute branch angles were a
    common FreeCAD ``JUNCTION_RAW_MATERIAL_INVALID`` source.
    """

    style = goal.junction_style or "smooth_hub"
    # Prefer compact hubs unless the intent explicitly asks for a larger one.
    default_hub = max(default_od * 0.85, default_od * 0.5 + 2.0, 6.0)
    max_hub = float(
        goal.max_hub_radius if goal.max_hub_radius is not None else default_hub
    )
    # Never let an implicit hub dwarf the pipe OD by a large factor.
    if goal.max_hub_radius is None:
        max_hub = min(max_hub, max(default_od * 1.05, 8.0))
    if style == "hard_fuse":
        return {
            "blend_mode": "hard",
            "max_hub_radius": max_hub,
        }

    # smooth_hub (default for product prompts)
    outer_blend = float(
        goal.blend_radius
        if goal.blend_radius is not None
        else min(max(default_od * 0.2, 1.5), max_hub * 0.45)
    )
    inner_blend = float(
        goal.inner_blend_radius
        if goal.inner_blend_radius is not None
        else min(max(outer_blend * 0.55, 1.0), outer_blend)
    )
    # Keep radii inside hub bound (catalog / FreeCAD checks).
    outer_blend = min(outer_blend, max_hub)
    inner_blend = min(inner_blend, max_hub)
    return {
        "blend_mode": "fillet",
        "blend_radius": outer_blend,
        "inner_blend_radius": inner_blend,
        "max_hub_radius": max_hub,
    }


def _transition_draft(goal: Goal, target: Port) -> ActionDraft | None:
    """diameter_change goal을 transition draft로 컴파일한다."""

    if goal.diameter_out is None or goal.transition_length is None:
        return None
    return _v2_draft(
        target=target,
        module="transition",
        params={
            "section_source": _SECTION_INHERIT,
            "diameter_out": float(goal.diameter_out),
            "wall_thickness_out": (
                float(goal.wall_thickness_out)
                if goal.wall_thickness_out is not None
                else None
            ),
            "length": float(goal.transition_length),
            "offset": tuple(float(value) for value in (goal.offset or (0.0, 0.0, 0.0))),
        },
        goal=goal,
        rationale="Host compiler bound the LLM-selected transition primitive.",
    )


def _terminate_draft(goal: Goal, target: Port) -> ActionDraft | None:
    """cap/plug end goal만 terminate 모듈로 컴파일한다. open end는 None."""

    if goal.type != "end":
        return None
    if goal.end_type not in {"cap", "plug"}:
        return None
    if goal.termination_thickness is None:
        return None
    return _v2_draft(
        target=target,
        module="terminate",
        params={
            "section_source": _SECTION_INHERIT,
            "termination_type": goal.end_type,
            "thickness": float(goal.termination_thickness),
        },
        goal=goal,
        rationale="Host compiler bound the LLM-selected termination primitive.",
    )


def _stable_perpendicular(axis: Vector3) -> Vector3:
    """pipe axis에 수직인 안정 기준 벡터를 고른다."""

    unit = normalize(axis)
    for candidate in ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)):
        projected = cross(unit, candidate)
        if length(projected) > 1e-6:
            return normalize(projected)
    raise ValueError("cannot build perpendicular to axis")


def _clamp_flange_annulus(
    dims: dict[str, float | int],
    *,
    outer_diameter: float,
    wall_thickness: float,
) -> dict[str, float | int]:
    """Intent override 후에도 bolt annulus가 FreeCAD mid-wall sample을 피하도록 보정."""

    fixed = _flange_host_dimensions(
        outer_diameter=outer_diameter, wall_thickness=wall_thickness
    )
    od = float(outer_diameter)
    wall = float(wall_thickness)
    bore_r = max(od - 2.0 * wall, 1.0) * 0.5
    body = float(dims.get("body_outer_diameter") or fixed["body_outer_diameter"])
    hole = float(dims.get("flange_bolt_hole_diameter") or fixed["flange_bolt_hole_diameter"])
    circle = float(
        dims.get("flange_bolt_circle_diameter") or fixed["flange_bolt_circle_diameter"]
    )
    length = float(dims.get("length") or fixed["length"])
    # Enforce catalog annulus: circle-hole > od, circle+hole < body.
    if circle - hole <= od or circle + hole >= body:
        return fixed
    wall_r = (body * 0.5 + bore_r) * 0.5
    bolt_r = circle * 0.5
    if abs(bolt_r - wall_r) < hole * 0.5 + 1.0:
        # Mid-wall sample sits inside bolt holes — recompute safe plate.
        return fixed
    dims = dict(dims)
    dims.setdefault("body_start_offset", 0.0)
    dims["body_outer_diameter"] = body
    dims["flange_bolt_circle_diameter"] = circle
    dims["flange_bolt_hole_diameter"] = hole
    dims["length"] = length
    dims["body_length"] = float(dims.get("body_length") or length)
    dims.setdefault("flange_bolt_count", fixed["flange_bolt_count"])
    return dims


def _flange_host_dimensions(
    *,
    outer_diameter: float,
    wall_thickness: float,
) -> dict[str, float | int]:
    """Catalog + FreeCAD mid-wall sampling을 동시에 만족하는 flange 치수.

    FreeCAD ``component_body_mid`` 는 r=(body_r+bore_r)/2 에서 wall을 샘플한다.
    볼트 홀 annulus가 이 반지름과 겹치면 missing_wall_material 가 난다.
    """

    od = float(outer_diameter)
    wall = float(wall_thickness)
    bore = max(od - 2.0 * wall, 1.0)
    bore_r = bore * 0.5
    hole = max(2.0, min(od * 0.18, 3.5))
    body = max(od * 2.8, od + 18.0, 24.0)
    # Place bolt PCD outside the mid-wall sample radius.
    for _ in range(4):
        body_r = body * 0.5
        wall_r = (body_r + bore_r) * 0.5
        bolt_r = max(wall_r + hole * 0.5 + 1.5, (od + hole) * 0.5 + 0.75)
        if bolt_r + hole * 0.5 >= body_r - 1.0:
            body = 2.0 * (bolt_r + hole * 0.5 + 2.5)
            continue
        circle = 2.0 * bolt_r
        if circle - hole <= od or circle + hole >= body:
            body = max(body * 1.15, circle + hole + 4.0)
            continue
        length = max(wall * 4.0, od * 0.55, 8.0)
        return {
            "length": length,
            "body_outer_diameter": body,
            "body_start_offset": 0.0,
            "body_length": length,
            "flange_bolt_count": 4,
            "flange_bolt_circle_diameter": circle,
            "flange_bolt_hole_diameter": hole,
        }
    # Fallback conservative plate
    length = max(wall * 4.0, od * 0.55, 8.0)
    body = max(od * 3.2, 30.0)
    circle = max(od + 6.0, body * 0.72)
    hole = max(2.0, min(od * 0.15, 3.0))
    return {
        "length": length,
        "body_outer_diameter": body,
        "body_start_offset": 0.0,
        "body_length": length,
        "flange_bolt_count": 4,
        "flange_bolt_circle_diameter": circle,
        "flange_bolt_hole_diameter": hole,
    }


def _connector_draft(goal: Goal, target: Port) -> ActionDraft | None:
    """connector goal을 inline_component draft로 host 컴파일한다.

    Intent가 고른 component kind + (optional) component_spec 을 존중하되,
    연속 치수(body OD, bolt PCD, reference axis)는 host가 소유한다.
    """

    if goal.type != "connector":
        return None
    component = str(goal.component or "")
    if component not in SUPPORTED_INLINE_COMPONENTS:
        return None
    if goal.component_spec is not None and goal.component_spec.component_type != component:
        return None

    od = float(target.outer_diameter)
    wall = float(target.wall_thickness)
    spec = goal.component_spec
    params: dict[str, object] = {
        "section_source": _SECTION_INHERIT,
        "component_type": component,
        "connector_type_out": "plain",
        "connector_gender_out": "neutral",
        "connector_standard_out": None,
    }

    if component == "flange":
        dims = _flange_host_dimensions(outer_diameter=od, wall_thickness=wall)
        if spec is not None:
            if spec.body_outer_diameter is not None:
                dims["body_outer_diameter"] = float(spec.body_outer_diameter)
            if spec.body_length is not None:
                dims["body_length"] = float(spec.body_length)
                dims["length"] = float(spec.body_length)
            if spec.body_start_offset is not None:
                dims["body_start_offset"] = float(spec.body_start_offset)
            if spec.flange_bolt_count is not None:
                dims["flange_bolt_count"] = int(spec.flange_bolt_count)
            if spec.flange_bolt_circle_diameter is not None:
                dims["flange_bolt_circle_diameter"] = float(
                    spec.flange_bolt_circle_diameter
                )
            if spec.flange_bolt_hole_diameter is not None:
                dims["flange_bolt_hole_diameter"] = float(
                    spec.flange_bolt_hole_diameter
                )
        if goal.length is not None and goal.length > 0:
            dims["length"] = float(goal.length)
            dims["body_length"] = float(goal.length)
            dims["body_start_offset"] = 0.0
        # Re-solve PCD/body if intent overrides broke FreeCAD mid-wall invariants.
        dims = _clamp_flange_annulus(dims, outer_diameter=od, wall_thickness=wall)
        params.update(dims)
        if spec is not None and spec.flange_reference_axis is not None:
            params["flange_reference_axis"] = normalize(vec(spec.flange_reference_axis))
        else:
            params["flange_reference_axis"] = _stable_perpendicular(vec(target.axis))
    elif component == "coupling":
        length = float(goal.length) if goal.length else max(od * 1.2, wall * 6.0, 12.0)
        body = (
            float(spec.body_outer_diameter)
            if spec is not None and spec.body_outer_diameter is not None
            else max(od * 1.35, od + 4.0)
        )
        params.update(
            {
                "length": length,
                "body_outer_diameter": max(body, od + 2.0),
                "body_start_offset": 0.0,
                "body_length": length,
            }
        )
    elif component == "union":
        length = float(goal.length) if goal.length else max(od * 1.6, wall * 8.0, 16.0)
        body = (
            float(spec.body_outer_diameter)
            if spec is not None and spec.body_outer_diameter is not None
            else max(od * 1.25, od + 3.0)
        )
        ring_od = (
            float(spec.union_ring_outer_diameter)
            if spec is not None and spec.union_ring_outer_diameter is not None
            else max(body * 1.15, body + 2.0)
        )
        ring_len = (
            float(spec.union_ring_length)
            if spec is not None and spec.union_ring_length is not None
            else max(length * 0.2, wall * 2.0, 3.0)
        )
        # Union body must sit between two necks.
        body_len = max(length * 0.45, ring_len * 2.0, 6.0)
        body_len = min(body_len, length * 0.7)
        offset = max((length - body_len) * 0.5, wall)
        if offset + body_len >= length:
            offset = max(length * 0.15, wall)
            body_len = length - 2.0 * offset
        params.update(
            {
                "length": length,
                "body_outer_diameter": body,
                "body_start_offset": offset,
                "body_length": body_len,
                "union_ring_outer_diameter": max(ring_od, body + 1.0),
                "union_ring_length": min(ring_len, body_len * 0.4),
            }
        )
    elif component == "valve":
        length = float(goal.length) if goal.length else max(od * 2.0, wall * 10.0, 20.0)
        body = (
            float(spec.body_outer_diameter)
            if spec is not None and spec.body_outer_diameter is not None
            else max(od * 1.4, od + 4.0)
        )
        act_d = (
            float(spec.actuator_diameter)
            if spec is not None and spec.actuator_diameter is not None
            else max(od * 0.8, 6.0)
        )
        act_h = (
            float(spec.actuator_height)
            if spec is not None and spec.actuator_height is not None
            else max(od * 1.2, 10.0)
        )
        body_len = max(length * 0.4, od, 8.0)
        body_len = min(body_len, length * 0.65)
        offset = max((length - body_len) * 0.5, wall)
        if offset + body_len >= length:
            offset = max(length * 0.15, wall)
            body_len = length - 2.0 * offset
        if spec is not None and spec.actuator_axis is not None:
            act_axis = normalize(vec(spec.actuator_axis))
        else:
            act_axis = _stable_perpendicular(vec(target.axis))
        params.update(
            {
                "length": length,
                "body_outer_diameter": body,
                "body_start_offset": offset,
                "body_length": body_len,
                "actuator_diameter": act_d,
                "actuator_height": act_h,
                "actuator_axis": act_axis,
            }
        )
    else:
        return None

    return _v2_draft(
        target=target,
        module="inline_component",
        params=params,
        goal=goal,
        rationale=(
            "Host compiler bound the LLM-selected inline "
            f"{component} connector primitive."
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_next_action(state: PipeState) -> ActionDraft | None:
    """모호하지 않은 다음 LLM 선택 goal을 schema-v2 action으로 컴파일한다."""

    if not state.remaining_goals:
        return None
    goal = state.remaining_goals[0]
    target = select_construction_port(state, goal)
    if target is None:
        return None

    # Priority: fused terminal arc > route/turn/move > typed specials.
    for candidate in (
        _terminal_arc_connect_draft(state, target),
        _route_draft(goal, target),
    ):
        if candidate is not None:
            return candidate

    if goal.type == "connect":
        return _connect_draft(goal, target, state)
    if goal.type == "branch":
        return _branch_draft(goal, target, state)
    if goal.type == "diameter_change":
        return _transition_draft(goal, target)
    if goal.type == "end":
        return _terminate_draft(goal, target)
    if goal.type == "connector":
        return _connector_draft(goal, target)
    return None


def compile_as_many_as_possible(state: PipeState, engine: Any) -> list[ActionDraft]:
    """현재 상태에서 host가 이어서 컴파일 가능한 draft 접두를 반환한다.

    실제 commit은 호출자가 한 스텝씩 검증한다. 이 함수는 계획 가능성 진단용이다.
    ``engine`` 은 ``StateEngine`` 인스턴스다.
    """

    from cadgen.primitive_action_catalog import validate_action, validate_draft

    cursor = state
    drafts: list[ActionDraft] = []
    limit = max(1, len(state.remaining_goals) + 2)
    for _ in range(limit):
        draft = compile_next_action(cursor)
        if draft is None or not validate_draft(draft, cursor).valid:
            break
        resolved = engine.resolve_action(draft, cursor)
        if not validate_action(resolved, cursor).valid:
            break
        drafts.append(draft)
        cursor = engine.apply_action(resolved, cursor)
        if not cursor.remaining_goals:
            break
    return drafts


__all__ = ["compile_as_many_as_possible", "compile_next_action"]
