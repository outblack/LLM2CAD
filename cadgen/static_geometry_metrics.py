"""정적 검증기가 공유하는 순수 기하 측정과 조회 함수를 제공한다.

이 모듈은 ``StaticIssue``를 만들지 않으며 상태를 변경하지 않는다. 검증 규칙은
``static_detail_validators``가 측정값을 해석해 issue로 변환한다.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

from cadgen.typed_data_models import (
    Direction,
    Goal,
    IntentResult,
    PipeState,
    Port,
    ResolvedAction,
    StateTransition,
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

VECTOR_TOLERANCE = 1e-4
PARALLEL_DOT_THRESHOLD = 0.9999
BRANCH_DIRECTION_DOT_THRESHOLD = 0.35
EXPLICIT_VECTOR_DOT_THRESHOLD = 0.9999


def _connection_interface_metrics(left: Port, right: Port) -> dict[str, float]:
    """연결 인터페이스 거리·정렬 메트릭을 계산한다."""

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
    """연결 계약 위반 여부를 판정한다."""

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

def _module_centerline_points(module: Any) -> list[tuple[float, float, float]]:
    """모듈 중심선 표본점을 반환한다."""

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

def _collision_envelope_reliable(module: Any) -> bool:
    """충돌 envelope 계산이 신뢰 가능한지 판정한다."""

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
    """선분이 지정 끝점을 포함하는지 본다."""

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
    """모듈 충돌 검사 선분 목록을 만든다."""

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
    """두 유한 선분 사이 최소 거리를 계산한다."""

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
    """점과 선분 사이 최소 거리를 계산한다."""

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
    """모듈 중심선 길이를 합산한다."""

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
    """세 점 원호의 호의 길이를 계산한다."""

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
        """각도를 합산한다."""

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
    """모듈 공간 표본점을 생성한다."""

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
    """모듈 스윕 envelope 반경을 추정한다."""

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

def _is_start_anchor_bootstrap_transition(
    before_state: PipeState,
    after_state: PipeState,
    action: ResolvedAction,
    produced_module: Any,
) -> bool:
    """is_start_anchor_bootstrap_transition 조건 여부를 판정한다."""

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

def _circumradius(a: Any, b: Any, c: Any) -> float:
    """세 점의 외접원 반경을 계산한다."""

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
    """원호 끝점 접선 방향을 계산한다."""

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
        """부호 있는 스윕각을 계산한다."""

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
    """analytic_route_arc_tangents를 계산하거나 반환한다."""

    try:
        _normal, start_tangent, end_tangent = canonical_circular_arc_frame(
            vec(params["axis"]),
            vec(params["plane_normal"]),
            float(params["sweep_angle"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return start_tangent, end_tangent

def _module_primary_displacement(module: Any) -> tuple[float, float, float]:
    """모듈 주 변위 벡터를 계산한다."""

    in_port = module.ports.get("in") or module.ports.get("in_a")
    out_port = module.ports.get("out")
    if in_port is None or out_port is None:
        return (0.0, 0.0, 0.0)
    return sub(vec(out_port.position), vec(in_port.position))

def _goal_path_points(modules: list[Any]) -> list[tuple[float, float, float]]:
    """goal 경로 표본점을 모은다."""

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
    """점에서 polyline까지 최소 거리를 계산한다."""

    return _point_to_polyline_projection(point, polyline)[0]

def _point_to_polyline_projection(
    point: tuple[float, float, float],
    polyline: list[tuple[float, float, float]],
) -> tuple[float, float]:
    """점에서 polyline으로의 투영 정보를 계산한다."""

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
    """point_to_goal_path_projection를 계산하거나 반환한다."""

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
    """점에서 원호로의 투영 정보를 계산한다."""

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
    """모듈 회전각(도)을 계산한다."""

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
    """primary outlet 포함 여부를 판정한다."""

    if goal.get("include_primary_outlet") is not None:
        return bool(goal["include_primary_outlet"])
    if params.get("include_primary_outlet") is not None:
        return bool(params["include_primary_outlet"])
    return not bool(_required_outlet_vectors(params, goal))

def _required_outlet_vectors(
    params: dict[str, Any],
    goal: dict[str, Any],
) -> list[tuple[float, float, float]]:
    """필수 outlet 방향 벡터를 수집한다."""

    del params
    raw_vectors = goal.get("required_outlet_vectors") or [
        item.get("axis")
        for item in (goal.get("required_outlets") or [])
        if isinstance(item, dict)
    ]
    return _normalize_vector_list(raw_vectors)

def _normalize_vector_list(raw_vectors: Any) -> list[tuple[float, float, float]]:
    """벡터 목록을 단위 벡터로 정규화한다."""

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
    """요구 벡터를 실제 port에 최적으로 대응시킨다."""

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
    """벡터 목록을 JSON 친화 형식으로 직렬화한다."""

    return [[round(float(component), 6) for component in vector] for vector in vectors]

def _direction_score(target_port: Port, port: Port, direction: Direction) -> float:
    """두 방향의 정렬 점수를 계산한다."""

    direction_vector = direction_to_vector(direction)
    displacement = sub(vec(port.position), vec(target_port.position))
    if length(displacement) > VECTOR_TOLERANCE:
        return dot(normalize(displacement), direction_vector)
    return dot(normalize(vec(port.axis)), direction_vector)

def _find_port(port_id: str, ports: list[Port]) -> Port | None:
    """port ID로 상태 내 port를 찾는다."""

    for port in ports:
        if port.id == port_id:
            return port
    return None

def _find_connectable_port(port_id: str, state: PipeState) -> Port | None:
    """연결 가능한 상대 port를 찾는다."""

    port = _find_port(port_id, state.open_ports)
    if port is not None:
        return port
    anchor = state.reserved_start_anchor
    if anchor is not None and anchor.id == port_id:
        return anchor
    return None

def _near(a: Any, b: Any, tolerance: float = VECTOR_TOLERANCE) -> bool:
    """두 값이 허용 오차 안에서 가까운지 비교한다."""

    return length(sub(vec(a), vec(b))) <= tolerance

def _same_direction(a: Any, b: Any) -> bool:
    """두 벡터가 같은 방향인지 판정한다."""

    return dot(normalize(vec(a)), normalize(vec(b))) >= PARALLEL_DOT_THRESHOLD

def _port_role(port_name: str) -> str:
    """port 역할 토큰을 정규화한다."""

    if port_name.startswith("out_"):
        return "branch_outlet"
    if port_name == "out":
        return "primary_outlet"
    return "other"

