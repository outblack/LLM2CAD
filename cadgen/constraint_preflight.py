"""소스 근거 제약 ledger와 전역 중심선 사전검증(preflight)을 담당한다.

Gemini·FreeCAD와 독립적으로, 수락된 semantic intent를 typed ledger로 바꾸고
serial line/arc 중심선 프로그램의 폐합·곡률·clearance를 첫 action 전에 검사한다.
일반 구동 치수가 속이 빈 스윕과 충돌하면 기본 정책이 각도·topology·단면을
보존하는 최소 균일 scale을 계산하고 모든 편차를 deviation ledger에 기록한다.
지원 밖 기하는 가능/불가로 단정하지 않고 ``unknown``으로 남긴다.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Iterable, Literal

from cadgen.stable_content_hash import stable_digest
from cadgen.typed_data_models import (
    ConflictCertificate,
    ConstraintDeviation,
    ConstraintLedger,
    ConstraintRecord,
    GlobalPreflightResult,
    Goal,
    IntentResult,
)
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


PREFLIGHT_METHOD_VERSION = "global-centerline-preflight/1"

_STRICT_MARKERS = re.compile(
    r"(?:정확히|반드시|절대로?|엄격(?:하게|히)?|오차\s*없이|"
    r"\bexactly\b|\bmust\b|\bstrict(?:ly)?\b|\bwithout\s+deviation\b)",
    re.IGNORECASE,
)
_ALL_DIMENSIONS_STRICT = re.compile(
    r"(?:모든|전체)\s*(?:수치|치수|길이|반경)[^.;\n]{0,30}"
    r"(?:정확히|반드시|엄격|오차\s*없이)|"
    r"(?:all|every)\s+(?:numeric\s+)?dimensions?[^.;\n]{0,30}"
    r"(?:exact|strict|without\s+deviation)",
    re.IGNORECASE,
)
_NUMBER_TOKEN = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])")
_SOURCE_PLANE = re.compile(
    r"\b(?P<plane>XY|XZ|YZ)\s*(?:plane|평\s*면)",
    re.IGNORECASE,
)


def _digest(payload: Any) -> str:
    """페이로드의 안정적인 SHA-256 digest를 반환한다."""

    return stable_digest(payload)


def _constraint_id(goal: Goal, index: int, field: str) -> str:
    """goal 필드 기반 제약 ID 문자열을 만든다."""

    return f"{goal.goal_id or f'goal_{index + 1}'}.{field}"


def _strict_numeric_values(prompt: str) -> list[float]:
    """prompt에서 엄격 치수로 표시된 숫자를 수집한다."""

    if _ALL_DIMENSIONS_STRICT.search(prompt):
        return [float(match.group(0)) for match in _NUMBER_TOKEN.finditer(prompt)]
    values: list[float] = []
    for marker in _STRICT_MARKERS.finditer(prompt):
        start = max(0, marker.start() - 48)
        end = min(len(prompt), marker.end() + 48)
        for number in _NUMBER_TOKEN.finditer(prompt[start:end]):
            values.append(float(number.group(0)))
    return values


def _matches_numeric(value: float, candidates: Iterable[float]) -> bool:
    """값이 후보 숫자 집합에 허용 오차로 일치하는지 본다."""

    return any(
        abs(float(value) - float(candidate)) <= max(1e-9, abs(float(value)) * 1e-9)
        for candidate in candidates
    )


def _source_numeric_span(prompt: str, value: float) -> str | None:
    """소스 prompt에서 숫자 토큰의 위치 범위를 찾는다."""

    for match in _NUMBER_TOKEN.finditer(prompt):
        try:
            candidate = float(match.group(0))
        except ValueError:  # pragma: no cover - regex accepts numeric tokens only.
            continue
        if not _matches_numeric(value, [candidate]):
            continue
        start = max(0, match.start() - 32)
        end = min(len(prompt), match.end() + 32)
        return prompt[start:end].strip()
    return None


def build_constraint_ledger(
    prompt: str,
    intent: IntentResult,
    *,
    modeling_tolerance: float,
) -> ConstraintLedger:
    """수락된 intent를 소스 근거 typed 제약 ledger로 투영한다."""

    strict_values = _strict_numeric_values(prompt)
    constraints: list[ConstraintRecord] = [
        ConstraintRecord(
            constraint_id="section.outer_diameter",
            constraint_type="circular_section_outer_diameter",
            source_field="global_spec.outer_diameter",
            source_span=_source_numeric_span(
                prompt, float(intent.global_spec.outer_diameter)
            ),
            priority="safety",
            relation="exact",
            value=float(intent.global_spec.outer_diameter),
            relaxable=False,
            tolerance=modeling_tolerance,
            variable_ids=["section.outer_diameter"],
        ),
        ConstraintRecord(
            constraint_id="section.wall_thickness",
            constraint_type="positive_hollow_wall",
            source_field="global_spec.wall_thickness",
            source_span=_source_numeric_span(
                prompt, float(intent.global_spec.wall_thickness)
            ),
            priority="safety",
            relation="exact",
            value=float(intent.global_spec.wall_thickness),
            relaxable=False,
            tolerance=modeling_tolerance,
            variable_ids=["section.wall_thickness"],
        ),
        ConstraintRecord(
            constraint_id="topology.connected",
            constraint_type="single_connected_pipe_network",
            priority="topology",
            relation="exact",
            value=True,
            relaxable=False,
            tolerance=0.0,
            variable_ids=["topology.graph"],
        ),
        ConstraintRecord(
            constraint_id="geometry.no_self_intersection",
            constraint_type="positive_swept_envelope_clearance",
            priority="safety",
            relation="minimum",
            value=0.0,
            relaxable=False,
            tolerance=modeling_tolerance,
            variable_ids=["centerline.program"],
        ),
    ]
    if intent.expected_open_ports is not None:
        constraints.append(
            ConstraintRecord(
                constraint_id="topology.expected_open_ports",
                constraint_type="open_port_count",
                priority="topology",
                relation="exact",
                value=int(intent.expected_open_ports),
                relaxable=False,
                tolerance=0.0,
                variable_ids=["topology.open_ports"],
            )
        )

    dimension_fields = (
        "length",
        "angle",
        "bend_radius",
        "diameter_out",
        "wall_thickness_out",
        "transition_length",
        "termination_thickness",
        "minimum_curvature_radius",
    )
    for index, goal in enumerate(intent.target_behavior):
        goal_id = goal.goal_id or f"goal_{index + 1}"
        constraints.append(
            ConstraintRecord(
                constraint_id=f"{goal_id}.primitive",
                constraint_type="primitive_choice",
                source_goal_id=goal_id,
                source_field="type/path_kind",
                priority="topology",
                relation="exact",
                value={"type": goal.type, "path_kind": goal.path_kind},
                relaxable=False,
                tolerance=0.0,
                variable_ids=[f"{goal_id}.primitive"],
            )
        )
        for field_name in dimension_fields:
            value = getattr(goal, field_name)
            if value is None:
                continue
            constraints.append(
                ConstraintRecord(
                    constraint_id=_constraint_id(goal, index, field_name),
                    constraint_type="authored_driving_dimension",
                    source_goal_id=goal_id,
                    source_field=f"target_behavior[{index}].{field_name}",
                    source_span=_source_numeric_span(prompt, float(value)),
                    priority="driving",
                    relation="exact",
                    value=float(value),
                    relaxable=(
                        not _matches_numeric(float(value), strict_values)
                        and field_name in {"length", "bend_radius"}
                    ),
                    tolerance=modeling_tolerance,
                    variable_ids=[f"{goal_id}.{field_name}"],
                )
            )

    for constraint in intent.geometric_constraints:
        constraints.append(
            ConstraintRecord(
                constraint_id=f"geometric.{constraint.constraint_id}",
                constraint_type=constraint.type,
                source_field=f"geometric_constraints.{constraint.constraint_id}",
                priority="driving",
                relation="maximum",
                value=constraint.model_dump(mode="json", exclude_none=True),
                relaxable=False,
                tolerance=modeling_tolerance,
                variable_ids=["centerline.program"],
            )
        )

    payload = [item.model_dump(mode="json") for item in constraints]
    return ConstraintLedger(
        ledger_digest=_digest({"schema_version": 1, "constraints": payload}),
        constraints=constraints,
    )


@dataclass(frozen=True)
class CenterlinePrimitive:
    """CenterlinePrimitive 데이터 모델이다."""

    primitive_id: str
    goal_id: str
    goal_index: int
    kind: Literal["line", "circular_arc"]
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    start_tangent: tuple[float, float, float]
    end_tangent: tuple[float, float, float]
    points: tuple[tuple[float, float, float], ...]
    radius: float | None = None
    sweep_angle: float | None = None

    def payload(self) -> dict[str, Any]:
        """중심선 primitive의 직렬화 페이로드를 반환한다."""

        return {
            "primitive_id": self.primitive_id,
            "goal_id": self.goal_id,
            "goal_index": self.goal_index,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "start_tangent": self.start_tangent,
            "end_tangent": self.end_tangent,
            "points": self.points,
            "radius": self.radius,
            "sweep_angle": self.sweep_angle,
        }


@dataclass(frozen=True)
class CenterlineProgram:
    """CenterlineProgram 데이터 모델이다."""

    primitives: tuple[CenterlinePrimitive, ...]
    closed: bool
    closure_position_error: float | None
    closure_tangent_error_degrees: float | None
    digest: str


class _UnsupportedProgram(ValueError):
    """UnsupportedProgram 데이터 모델이다."""

    pass


def _arc_points(
    start: tuple[float, float, float],
    tangent: tuple[float, float, float],
    normal: tuple[float, float, float],
    radius: float,
    sweep_angle: float,
    *,
    outer_radius: float,
    modeling_tolerance: float,
) -> tuple[tuple[float, float, float], ...]:
    """원호 중심선 표본점을 생성한다."""

    theta = math.radians(float(sweep_angle))
    radial = normalize(cross(tangent, normal))
    signed_radius = radius if sweep_angle >= 0.0 else -radius
    center = sub(start, mul(radial, signed_radius))
    start_radius = sub(start, center)
    # Every returned point lies on the analytic arc.  A detected point-pair
    # overlap is therefore a sound conflict witness; sparse sampling may only
    # cause UNKNOWN/missed conflicts, never a fabricated proof.
    target_sagitta = max(modeling_tolerance * 8.0, outer_radius * 5e-4)
    if target_sagitta >= radius:
        max_step = math.pi / 12.0
    else:
        max_step = 2.0 * math.acos(max(-1.0, 1.0 - target_sagitta / radius))
    segments = max(4, min(720, int(math.ceil(abs(theta) / max(max_step, 1e-6)))))
    return tuple(
        add(center, rotate(start_radius, normal, theta * (index / segments)))
        for index in range(segments + 1)
    )


def compile_centerline_program(
    intent: IntentResult,
    *,
    modeling_tolerance: float,
) -> CenterlineProgram:
    """지원되는 serial 일정 단면 중심선 프로그램을 컴파일한다."""

    try:
        position = vec(intent.start_position)
        heading = normalize(vec(intent.start_axis))
    except ValueError as exc:
        raise _UnsupportedProgram(f"invalid start pose: {exc}") from exc

    start_position = position
    start_heading = heading
    outer_radius = float(intent.global_spec.outer_diameter) / 2.0
    primitives: list[CenterlinePrimitive] = []
    closed = False

    for index, goal in enumerate(intent.target_behavior):
        goal_id = goal.goal_id or f"goal_{index + 1}"
        if index > 0:
            previous_id = intent.target_behavior[index - 1].goal_id
            if goal.allow_parallel or (
                goal.depends_on_goal_ids and previous_id not in goal.depends_on_goal_ids
            ):
                raise _UnsupportedProgram(
                    f"{goal_id} leaves the uniquely serial degree-2 domain"
                )

        if goal.type == "move" or (goal.type == "route" and goal.path_kind == "line"):
            if goal.length is None:
                raise _UnsupportedProgram(f"{goal_id} line has no authored length")
            line_heading = (
                direction_to_vector(goal.direction)
                if goal.direction is not None
                else heading
            )
            # A route line is a tangent continuation.  Explicit non-tangent
            # headings belong in a turn primitive and must not be hidden here.
            if (
                goal.type == "route"
                and dot(normalize(line_heading), heading) < 1.0 - 1e-7
            ):
                raise _UnsupportedProgram(
                    f"{goal_id} has a non-tangent explicit line heading"
                )
            line_heading = normalize(line_heading)
            endpoint = add(position, mul(line_heading, float(goal.length)))
            primitives.append(
                CenterlinePrimitive(
                    primitive_id=f"P{len(primitives) + 1}",
                    goal_id=goal_id,
                    goal_index=index,
                    kind="line",
                    start=position,
                    end=endpoint,
                    start_tangent=line_heading,
                    end_tangent=line_heading,
                    points=(position, endpoint),
                )
            )
            position = endpoint
            heading = line_heading
            continue

        is_arc = goal.type == "turn" or (
            goal.type == "route" and goal.path_kind == "circular_arc"
        )
        if is_arc:
            sweep = goal.angle
            radius = goal.bend_radius
            normal_hint = goal.plane_normal
            if sweep is None or radius is None or normal_hint is None:
                raise _UnsupportedProgram(
                    f"{goal_id} arc requires angle, bend_radius and plane_normal"
                )
            try:
                normal, initial, terminal = canonical_circular_arc_frame(
                    heading,
                    vec(normal_hint),
                    float(sweep),
                )
            except ValueError as exc:
                raise _UnsupportedProgram(f"{goal_id} arc frame: {exc}") from exc
            points = _arc_points(
                position,
                initial,
                normal,
                float(radius),
                float(sweep),
                outer_radius=outer_radius,
                modeling_tolerance=modeling_tolerance,
            )
            endpoint = points[-1]
            primitives.append(
                CenterlinePrimitive(
                    primitive_id=f"P{len(primitives) + 1}",
                    goal_id=goal_id,
                    goal_index=index,
                    kind="circular_arc",
                    start=position,
                    end=endpoint,
                    start_tangent=initial,
                    end_tangent=terminal,
                    points=points,
                    radius=float(radius),
                    sweep_angle=float(sweep),
                )
            )
            position = endpoint
            heading = terminal
            continue

        if goal.type == "connect" and goal.connection_target == "start_anchor":
            closed = True
            continue
        if goal.type == "end":
            continue
        # Constant-section inline connectors are straight axial spans and can
        # participate in the same global preflight when length is authored.
        if goal.type == "connector" and goal.length is not None:
            endpoint = add(position, mul(heading, float(goal.length)))
            primitives.append(
                CenterlinePrimitive(
                    primitive_id=f"P{len(primitives) + 1}",
                    goal_id=goal_id,
                    goal_index=index,
                    kind="line",
                    start=position,
                    end=endpoint,
                    start_tangent=heading,
                    end_tangent=heading,
                    points=(position, endpoint),
                )
            )
            position = endpoint
            continue
        raise _UnsupportedProgram(
            f"{goal_id} ({goal.type}/{goal.path_kind}) is outside global preflight v1"
        )

    if not primitives:
        raise _UnsupportedProgram("intent contains no line/arc centerline primitives")
    position_error = length(sub(position, start_position)) if closed else None
    tangent_error = None
    if closed:
        cosine = max(-1.0, min(1.0, dot(normalize(heading), start_heading)))
        tangent_error = math.degrees(math.acos(cosine))
    payload = {
        "primitives": [item.payload() for item in primitives],
        "closed": closed,
        "closure_position_error": position_error,
        "closure_tangent_error_degrees": tangent_error,
    }
    return CenterlineProgram(
        primitives=tuple(primitives),
        closed=closed,
        closure_position_error=position_error,
        closure_tangent_error_degrees=tangent_error,
        digest=_digest(payload),
    )


def _segment_distance(
    a0: tuple[float, float, float],
    a1: tuple[float, float, float],
    b0: tuple[float, float, float],
    b1: tuple[float, float, float],
) -> float:
    """두 유한 3차원 선분 사이의 정확한 최소 유클리드 거리를 계산한다."""

    u = sub(a1, a0)
    v = sub(b1, b0)
    w = sub(a0, b0)
    aa = dot(u, u)
    bb = dot(u, v)
    cc = dot(v, v)
    dd = dot(u, w)
    ee = dot(v, w)
    determinant = aa * cc - bb * bb
    small = 1e-15
    s_den = determinant
    t_den = determinant
    if determinant < small:
        s_num = 0.0
        s_den = 1.0
        t_num = ee
        t_den = cc
    else:
        s_num = bb * ee - cc * dd
        t_num = aa * ee - bb * dd
        if s_num < 0.0:
            s_num = 0.0
            t_num = ee
            t_den = cc
        elif s_num > s_den:
            s_num = s_den
            t_num = ee + bb
            t_den = cc
    if t_num < 0.0:
        t_num = 0.0
        if -dd < 0.0:
            s_num = 0.0
        elif -dd > aa:
            s_num = s_den
        else:
            s_num = -dd
            s_den = aa
    elif t_num > t_den:
        t_num = t_den
        if -dd + bb < 0.0:
            s_num = 0.0
        elif -dd + bb > aa:
            s_num = s_den
        else:
            s_num = -dd + bb
            s_den = aa
    sc = 0.0 if abs(s_num) < small else s_num / max(s_den, small)
    tc = 0.0 if abs(t_num) < small else t_num / max(t_den, small)
    delta = sub(add(w, mul(u, sc)), mul(v, tc))
    return length(delta)


_AxisAlignedBounds = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    float,
]
_BoundedSegment = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    _AxisAlignedBounds,
]

# AABB는 실제 거리 계산을 생략할 수 있는지 판정하는 보조 하한일 뿐이다.
# 큰 전역 좌표에서 뺄셈 오차가 하한을 잘못 키우지 않도록 좌표 규모에 비례한
# 여유와 여러 ULP 중 큰 값을 사용한다. 이 여유는 충돌 판정 임계값에는 더하지
# 않고 하한에서만 빼므로, 성능상 보수적일 수는 있어도 기존 충돌을 놓치지 않는다.
_AABB_ROUNDOFF_ULPS = 256.0
_AABB_RELATIVE_ROUNDOFF = 1e-12
_AABB_DIAGONAL_ROUNDOFF_FACTOR = math.sqrt(3.0)


def _axis_aligned_bounds(
    points: Iterable[tuple[float, float, float]],
) -> _AxisAlignedBounds:
    """점 집합의 AABB와 반올림 여유 계산에 쓸 좌표 규모를 반환한다."""

    iterator = iter(points)
    try:
        first = next(iterator)
    except StopIteration as exc:  # pragma: no cover - primitive은 항상 두 점 이상이다.
        raise ValueError("cannot build bounds for an empty point sequence") from exc

    minimum = [float(first[0]), float(first[1]), float(first[2])]
    maximum = list(minimum)
    coordinate_scale = max(1.0, *(abs(component) for component in minimum))
    for point in iterator:
        for axis in range(3):
            component = float(point[axis])
            minimum[axis] = min(minimum[axis], component)
            maximum[axis] = max(maximum[axis], component)
            coordinate_scale = max(coordinate_scale, abs(component))
    return (
        (minimum[0], minimum[1], minimum[2]),
        (maximum[0], maximum[1], maximum[2]),
        coordinate_scale,
    )


def _aabb_distance_lower_bound(
    left: _AxisAlignedBounds,
    right: _AxisAlignedBounds,
) -> float:
    """두 AABB 사이 실제 거리보다 크지 않은 보수적 하한을 계산한다."""

    left_minimum, left_maximum, left_scale = left
    right_minimum, right_maximum, right_scale = right
    coordinate_scale = max(1.0, left_scale, right_scale)
    roundoff_margin = max(
        _AABB_ROUNDOFF_ULPS * math.ulp(coordinate_scale),
        coordinate_scale * _AABB_RELATIVE_ROUNDOFF,
    )
    axis_gaps = []
    for axis in range(3):
        raw_gap = max(
            0.0,
            right_minimum[axis] - left_maximum[axis],
            left_minimum[axis] - right_maximum[axis],
        )
        axis_gaps.append(max(0.0, raw_gap - roundoff_margin))

    # hypot 자체의 마지막 반올림도 하한을 위로 밀지 않도록 한 번 더 감산한다.
    return max(
        0.0,
        math.hypot(axis_gaps[0], axis_gaps[1], axis_gaps[2])
        - _AABB_DIAGONAL_ROUNDOFF_FACTOR * roundoff_margin,
    )


def _bounded_segments(primitive: CenterlinePrimitive) -> tuple[_BoundedSegment, ...]:
    """기존 polyline 순서 그대로 각 segment와 AABB를 묶는다."""

    return tuple(
        (start, end, _axis_aligned_bounds((start, end)))
        for start, end in zip(primitive.points, primitive.points[1:])
    )


def _polyline_distance(
    left: CenterlinePrimitive,
    right: CenterlinePrimitive,
    *,
    left_segments: tuple[_BoundedSegment, ...] | None = None,
    right_segments: tuple[_BoundedSegment, ...] | None = None,
) -> float:
    """정확한 segment 거리를 유지하며 AABB로 이길 수 없는 쌍만 생략한다."""

    left_candidates = left_segments or _bounded_segments(left)
    right_candidates = right_segments or _bounded_segments(right)
    minimum = math.inf
    for a0, a1, left_bounds in left_candidates:
        for b0, b1, right_bounds in right_candidates:
            if minimum < math.inf and (
                _aabb_distance_lower_bound(left_bounds, right_bounds) >= minimum
            ):
                # 같은 거리도 기존 min()에서 선행 segment가 유지되므로 생략해도
                # closest/tie 순서가 바뀌지 않는다. 보수 여유 안의 쌍은 계산한다.
                continue
            minimum = min(minimum, _segment_distance(a0, a1, b0, b1))
    return minimum


def _certificate(
    *,
    conflict_type: Literal["local_geometry", "closure", "clearance", "topology"],
    predicate: str,
    message: str,
    constraint_ids: list[str],
    primitive_ids: list[str],
    program_digest: str,
    measured: float | None = None,
    required: float | None = None,
    units: str | None = None,
    proof_strength: Literal["proved", "heuristic", "unknown"] = "proved",
    mutable_fields: list[str] | None = None,
    earliest_backjump_step: int | None = None,
) -> ConflictCertificate:
    """preflight 실패용 ConflictCertificate를 구성한다."""

    gap = None
    if measured is not None and required is not None:
        gap = max(0.0, float(required) - float(measured))
    payload = {
        "type": conflict_type,
        "predicate": predicate,
        "constraints": constraint_ids,
        "primitives": primitive_ids,
        "program": program_digest,
        "measured": measured,
        "required": required,
    }
    return ConflictCertificate(
        certificate_id=f"conflict-{_digest(payload)[:16]}",
        conflict_type=conflict_type,
        failed_predicate=predicate,
        proof_strength=proof_strength,
        constraint_ids=constraint_ids,
        primitive_ids=primitive_ids,
        candidate_digest=program_digest,
        evidence_digest=_digest(payload),
        measured=measured,
        required=required,
        gap=gap,
        units=units,
        causal_decision_ids=primitive_ids,
        earliest_backjump_step=earliest_backjump_step,
        mutable_fields=mutable_fields or [],
        allowed_routes=[
            "relax_driving_constraint",
            "backjump",
            "change_primitive",
        ],
        message=message,
    )


def verify_centerline_program(
    program: CenterlineProgram,
    intent: IntentResult,
    *,
    modeling_tolerance: float,
) -> list[ConflictCertificate]:
    """컴파일된 전체 경로에서 증명ㆍ추정된 모든 충돌을 반환한다."""

    conflicts: list[ConflictCertificate] = []
    outer_radius = float(intent.global_spec.outer_diameter) / 2.0
    clearance_margin = max(modeling_tolerance * 10.0, outer_radius * 1e-6)
    required_pair_distance = 2.0 * outer_radius + clearance_margin
    required_arc_radius = outer_radius + clearance_margin

    for primitive in program.primitives:
        if primitive.kind != "circular_arc" or primitive.radius is None:
            continue
        if primitive.radius + 1e-12 < required_arc_radius:
            conflicts.append(
                _certificate(
                    conflict_type="local_geometry",
                    predicate="centerline_radius > outer_profile_radius",
                    message=(
                        f"{primitive.goal_id} is at or inside the horn-torus "
                        "regularity boundary for a hollow sweep."
                    ),
                    constraint_ids=[
                        "geometry.no_self_intersection",
                        f"{primitive.goal_id}.bend_radius",
                        "section.outer_diameter",
                    ],
                    primitive_ids=[primitive.primitive_id],
                    program_digest=program.digest,
                    measured=primitive.radius,
                    required=required_arc_radius,
                    units="mm",
                    mutable_fields=[f"{primitive.goal_id}.bend_radius"],
                    earliest_backjump_step=primitive.goal_index + 1,
                )
            )

    count = len(program.primitives)
    primitive_bounds = tuple(
        _axis_aligned_bounds(primitive.points) for primitive in program.primitives
    )
    # primitive AABB에서 걸러지지 않은 가까운 쌍에 대해서만 segment AABB를
    # 만든다. 한 primitive이 여러 가까운 쌍에 참여할 때는 같은 순서의 묶음을
    # 재사용하며, 멀리 떨어진 대부분의 arc에는 segment 객체를 만들지 않는다.
    bounded_segments: dict[int, tuple[_BoundedSegment, ...]] = {}
    for left_index, left in enumerate(program.primitives):
        for right_index in range(left_index + 1, count):
            right = program.primitives[right_index]
            if right_index == left_index + 1:
                continue
            if program.closed and left_index == 0 and right_index == count - 1:
                continue
            if (
                _aabb_distance_lower_bound(
                    primitive_bounds[left_index],
                    primitive_bounds[right_index],
                )
                + 1e-12
                >= required_pair_distance
            ):
                # 보수적인 AABB 하한만으로 기존 `distance + 1e-12` 통과가
                # 증명된 쌍은 exact segment 순회를 시작하지 않는다.
                continue
            left_segments = bounded_segments.get(left_index)
            if left_segments is None:
                left_segments = _bounded_segments(left)
                bounded_segments[left_index] = left_segments
            right_segments = bounded_segments.get(right_index)
            if right_segments is None:
                right_segments = _bounded_segments(right)
                bounded_segments[right_index] = right_segments
            distance = _polyline_distance(
                left,
                right,
                left_segments=left_segments,
                right_segments=right_segments,
            )
            if distance + 1e-12 >= required_pair_distance:
                continue
            proof_strength: Literal["proved", "heuristic", "unknown"] = (
                "proved" if left.kind == right.kind == "line" else "heuristic"
            )
            conflicts.append(
                _certificate(
                    conflict_type="clearance",
                    predicate="non_adjacent_centerline_distance >= tube_diameter + margin",
                    message=(
                        f"Non-adjacent primitives {left.primitive_id} and "
                        f"{right.primitive_id} have insufficient swept-envelope clearance."
                    ),
                    constraint_ids=[
                        "geometry.no_self_intersection",
                        f"{left.goal_id}.primitive",
                        f"{right.goal_id}.primitive",
                    ],
                    primitive_ids=[left.primitive_id, right.primitive_id],
                    program_digest=program.digest,
                    measured=distance,
                    required=required_pair_distance,
                    units="mm",
                    proof_strength=proof_strength,
                    mutable_fields=[
                        f"{left.goal_id}.length",
                        f"{right.goal_id}.length",
                    ],
                    earliest_backjump_step=min(left.goal_index, right.goal_index) + 1,
                )
            )

    if program.closed:
        position_required = max(modeling_tolerance * 10.0, outer_radius * 1e-6)
        tangent_required = 1e-4
        if (program.closure_position_error or 0.0) > position_required:
            conflicts.append(
                _certificate(
                    conflict_type="closure",
                    predicate="closed_loop_endpoint_position_error <= tolerance",
                    message="The authored serial program does not reach the start anchor.",
                    constraint_ids=["topology.expected_open_ports"],
                    primitive_ids=[
                        item.primitive_id for item in program.primitives[-2:]
                    ],
                    program_digest=program.digest,
                    measured=float(program.closure_position_error or 0.0),
                    required=position_required,
                    units="mm",
                    proof_strength="proved",
                    earliest_backjump_step=1,
                )
            )
        if (program.closure_tangent_error_degrees or 0.0) > tangent_required:
            conflicts.append(
                _certificate(
                    conflict_type="closure",
                    predicate="closed_loop_terminal_tangent_matches_start",
                    message="The authored serial program reaches the seam with a tangent mismatch.",
                    constraint_ids=["topology.expected_open_ports"],
                    primitive_ids=[
                        item.primitive_id for item in program.primitives[-2:]
                    ],
                    program_digest=program.digest,
                    measured=float(program.closure_tangent_error_degrees or 0.0),
                    required=tangent_required,
                    units="degree",
                    proof_strength="proved",
                    earliest_backjump_step=1,
                )
            )
    return conflicts


def _scalable_goal_fields(intent: IntentResult) -> list[tuple[int, str, float]]:
    """균일 scale이 적용 가능한 goal 필드를 열거한다."""

    result: list[tuple[int, str, float]] = []
    for index, goal in enumerate(intent.target_behavior):
        if goal.type in {"move", "connector"} or (
            goal.type == "route" and goal.path_kind == "line"
        ):
            if goal.length is not None:
                result.append((index, "length", float(goal.length)))
        if goal.type == "turn" or (
            goal.type == "route" and goal.path_kind == "circular_arc"
        ):
            if goal.bend_radius is not None:
                result.append((index, "bend_radius", float(goal.bend_radius)))
    return result


def _scale_intent(intent: IntentResult, factor: float) -> IntentResult:
    """intent 치수에 균일 scale을 적용한 사본을 만든다."""

    goals = list(intent.target_behavior)
    for index, field_name, authored in _scalable_goal_fields(intent):
        goals[index] = goals[index].model_copy(
            update={field_name: float(authored) * float(factor)}
        )
    return intent.model_copy(update={"target_behavior": goals})


def _replace_goal_fields(
    intent: IntentResult,
    values: dict[tuple[int, str], float],
) -> IntentResult:
    """지정 goal 필드 값을 교체한 intent를 반환한다."""

    goals = list(intent.target_behavior)
    for (index, field_name), value in values.items():
        goals[index] = goals[index].model_copy(update={field_name: float(value)})
    return intent.model_copy(update={"target_behavior": goals})


def _solve_3x3(
    matrix: list[list[float]],
    vector: list[float],
) -> list[float] | None:
    """부분 피벗이 있는 정규화 3x3 선형계를 푼다."""

    augmented = [
        [float(matrix[row][column]) for column in range(3)] + [float(vector[row])]
        for row in range(3)
    ]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-18:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][3] for row in range(3)]


def _solve_closure_dimensions(
    intent: IntentResult,
    *,
    modeling_tolerance: float,
) -> IntentResult | None:
    """완화 가능 치수를 serial 폐합 방정식에 투영한다."""

    base_program = compile_centerline_program(
        intent,
        modeling_tolerance=modeling_tolerance,
    )
    if not base_program.closed:
        return intent
    if (base_program.closure_tangent_error_degrees or 0.0) > 1e-4:
        return None
    residual = list(base_program.primitives[-1].end)
    start = list(intent.start_position)
    residual = [value - start[index] for index, value in enumerate(residual)]
    if math.sqrt(sum(value * value for value in residual)) <= max(
        modeling_tolerance * 10.0,
        float(intent.global_spec.outer_diameter) * 5e-7,
    ):
        return intent

    variables = _scalable_goal_fields(intent)
    if not variables:
        return None
    keys = [(index, field_name) for index, field_name, _value in variables]
    authored = [value for _index, _field_name, value in variables]
    columns: list[tuple[float, float, float]] = []
    for index, field_name, authored_value in variables:
        delta = max(1.0, abs(authored_value) * 1e-6)
        perturbed = _replace_goal_fields(
            intent,
            {(index, field_name): authored_value + delta},
        )
        perturbed_program = compile_centerline_program(
            perturbed,
            modeling_tolerance=modeling_tolerance,
        )
        endpoint = perturbed_program.primitives[-1].end
        base_endpoint = base_program.primitives[-1].end
        columns.append(
            tuple((endpoint[axis] - base_endpoint[axis]) / delta for axis in range(3))
        )

    outer_radius = float(intent.global_spec.outer_diameter) / 2.0
    clearance_margin = max(modeling_tolerance * 10.0, outer_radius * 1e-6)
    lower_bounds = [
        (
            outer_radius + clearance_margin
            if field_name == "bend_radius"
            else max(modeling_tolerance * 10.0, 1e-8)
        )
        for _index, field_name, _value in variables
    ]
    # A relative least-change norm naturally assigns most closure correction to
    # a long authored return span instead of distorting every short segment.
    weights = [
        max(abs(value), outer_radius) * (0.35 if field_name == "bend_radius" else 1.0)
        for (_index, field_name, value) in variables
    ]
    fixed: dict[int, float] = {}
    solution = list(authored)
    for _active_round in range(len(variables) + 1):
        free = [index for index in range(len(variables)) if index not in fixed]
        if not free:
            return None
        fixed_effect = [0.0, 0.0, 0.0]
        for variable_index, value in fixed.items():
            for axis in range(3):
                fixed_effect[axis] += columns[variable_index][axis] * (
                    value - authored[variable_index]
                )
        target = [-residual[axis] - fixed_effect[axis] for axis in range(3)]
        matrix = [[0.0 for _ in range(3)] for _ in range(3)]
        for variable_index in free:
            weight_squared = weights[variable_index] ** 2
            column = columns[variable_index]
            for row in range(3):
                for column_index in range(3):
                    matrix[row][column_index] += (
                        weight_squared * column[row] * column[column_index]
                    )
        trace = sum(matrix[index][index] for index in range(3))
        regularization = max(trace, 1.0) * 1e-12
        for axis in range(3):
            matrix[axis][axis] += regularization
        dual = _solve_3x3(matrix, target)
        if dual is None:
            return None
        proposed = list(solution)
        for variable_index in free:
            proposed[variable_index] = authored[variable_index] + (
                weights[variable_index] ** 2
                * sum(columns[variable_index][axis] * dual[axis] for axis in range(3))
            )
        violated = [
            variable_index
            for variable_index in free
            if proposed[variable_index] < lower_bounds[variable_index]
        ]
        if violated:
            # Bind the most severe normalized lower-bound violation, then solve
            # the reduced convex projection again.
            selected = max(
                violated,
                key=lambda item: (lower_bounds[item] - proposed[item])
                / max(lower_bounds[item], 1e-12),
            )
            fixed[selected] = lower_bounds[selected]
            solution[selected] = lower_bounds[selected]
            continue
        solution = proposed
        break
    else:  # pragma: no cover - finite active-set loop above must terminate.
        return None

    realized = _replace_goal_fields(
        intent,
        {key: value for key, value in zip(keys, solution)},
    )
    check = compile_centerline_program(
        realized,
        modeling_tolerance=modeling_tolerance,
    )
    position_error = float(check.closure_position_error or 0.0)
    tolerance = max(
        modeling_tolerance * 10.0,
        float(intent.global_spec.outer_diameter) * 5e-7,
    )
    if position_error > tolerance:
        return None
    return realized


def _has_unrelaxable_scalable_field(ledger: ConstraintLedger) -> bool:
    """완화 불가능한 스케일 필드가 있는지 검사한다."""

    return any(
        record.source_field
        and record.source_field.endswith((".length", ".bend_radius"))
        and record.priority == "driving"
        and not record.relaxable
        for record in ledger.constraints
    )


def _violates_global_upper_bound(intent: IntentResult, factor: float) -> bool:
    """명시 공간 상한 하에서 균일 확장이 위반인지 검사한다."""

    if factor <= 1.0 + 1e-12:
        return False
    return any(
        item.type in {"max_extent", "max_total_centerline_length", "bounding_box"}
        for item in intent.geometric_constraints
    )


def _minimum_uniform_scale(
    intent: IntentResult,
    *,
    modeling_tolerance: float,
    max_scale: float,
) -> tuple[float, IntentResult, CenterlineProgram, list[ConflictCertificate]] | None:
    """지원 충돌을 해소하는 최소 중심선 scale을 찾는다."""

    low = 1.0
    high = 1.0
    realized = intent
    program = compile_centerline_program(
        realized, modeling_tolerance=modeling_tolerance
    )
    conflicts = verify_centerline_program(
        program,
        realized,
        modeling_tolerance=modeling_tolerance,
    )
    # Uniform scaling preserves a solved zero-residual closure.  A non-zero
    # closure needs the affine projection above and cannot be fixed by scale.
    if any(item.conflict_type == "closure" for item in conflicts):
        return None
    while conflicts and high < max_scale:
        high = min(max_scale, max(high * 1.25, high + 0.05))
        if _violates_global_upper_bound(intent, high):
            return None
        realized = _scale_intent(intent, high)
        program = compile_centerline_program(
            realized,
            modeling_tolerance=modeling_tolerance,
        )
        conflicts = verify_centerline_program(
            program,
            realized,
            modeling_tolerance=modeling_tolerance,
        )
    if conflicts:
        return None
    # Monotone within this supported uniform-scaling domain.  A fixed iteration
    # count provides deterministic convergence without prompt-specific grids.
    for _ in range(48):
        middle = (low + high) / 2.0
        candidate = _scale_intent(intent, middle)
        candidate_program = compile_centerline_program(
            candidate,
            modeling_tolerance=modeling_tolerance,
        )
        candidate_conflicts = verify_centerline_program(
            candidate_program,
            candidate,
            modeling_tolerance=modeling_tolerance,
        )
        if candidate_conflicts:
            low = middle
        else:
            high = middle
            realized = candidate
            program = candidate_program
            conflicts = candidate_conflicts
    return high, realized, program, conflicts


def _deviations(
    authored: IntentResult,
    realized: IntentResult,
    ledger: ConstraintLedger,
) -> list[ConstraintDeviation]:
    """원본 대비 실현 치수 편차 목록을 만든다."""

    records = {item.constraint_id: item for item in ledger.constraints}
    result: list[ConstraintDeviation] = []
    for index, field_name, authored_value in _scalable_goal_fields(authored):
        realized_value = float(getattr(realized.target_behavior[index], field_name))
        if math.isclose(authored_value, realized_value, rel_tol=1e-12, abs_tol=1e-12):
            continue
        goal = authored.target_behavior[index]
        constraint_id = _constraint_id(goal, index, field_name)
        record = records.get(constraint_id)
        absolute = abs(realized_value - authored_value)
        result.append(
            ConstraintDeviation(
                deviation_id=f"deviation-{_digest([constraint_id, authored_value, realized_value])[:16]}",
                constraint_id=constraint_id,
                goal_id=goal.goal_id,
                field_path=f"target_behavior[{index}].{field_name}",
                authored_value=authored_value,
                realized_value=realized_value,
                absolute_change=absolute,
                relative_change=absolute / max(abs(authored_value), 1e-12),
                reason_code="UNIFORM_CENTERLINE_SCALE_FOR_SWEEP_SAFETY",
                reason=(
                    "Minimum uniform centerline scale required to preserve the "
                    "LLM-selected primitive sequence, angles, topology and pipe "
                    "section while satisfying curvature and non-local clearance."
                ),
                priority=record.priority if record is not None else "driving",
            )
        )
    return result


def preflight_and_realize_intent(
    prompt: str,
    intent: IntentResult,
    *,
    modeling_tolerance: float,
    feasibility_mode: Literal["best_effort", "strict", "off"] = "best_effort",
    max_uniform_scale: float = 8.0,
) -> tuple[IntentResult, ConstraintLedger, GlobalPreflightResult]:
    """전역 preflight를 실행하고 exact 또는 조정된 intent를 반환한다."""

    ledger = build_constraint_ledger(
        prompt,
        intent,
        modeling_tolerance=modeling_tolerance,
    )
    if feasibility_mode == "off":
        result = GlobalPreflightResult(
            method_version=PREFLIGHT_METHOD_VERSION,
            status="unknown",
            ledger_digest=ledger.ledger_digest,
            notes=["Global preflight disabled by configuration."],
        )
        return (
            intent.model_copy(
                update={"constraint_ledger": ledger, "global_preflight": result}
            ),
            ledger,
            result,
        )

    try:
        authored_program = compile_centerline_program(
            intent,
            modeling_tolerance=modeling_tolerance,
        )
    except _UnsupportedProgram as exc:
        result = GlobalPreflightResult(
            method_version=PREFLIGHT_METHOD_VERSION,
            status="unknown",
            ledger_digest=ledger.ledger_digest,
            notes=[str(exc)],
        )
        return (
            intent.model_copy(
                update={"constraint_ledger": ledger, "global_preflight": result}
            ),
            ledger,
            result,
        )

    conflicts = verify_centerline_program(
        authored_program,
        intent,
        modeling_tolerance=modeling_tolerance,
    )
    if not conflicts:
        result = GlobalPreflightResult(
            method_version=PREFLIGHT_METHOD_VERSION,
            status="exact",
            ledger_digest=ledger.ledger_digest,
            authored_program_digest=authored_program.digest,
            realized_program_digest=authored_program.digest,
        )
        return (
            intent.model_copy(
                update={"constraint_ledger": ledger, "global_preflight": result}
            ),
            ledger,
            result,
        )

    proved = [item for item in conflicts if item.proof_strength == "proved"]
    if feasibility_mode == "strict" or _has_unrelaxable_scalable_field(ledger):
        status: Literal["infeasible", "unknown"] = "infeasible" if proved else "unknown"
        result = GlobalPreflightResult(
            method_version=PREFLIGHT_METHOD_VERSION,
            status=status,
            ledger_digest=ledger.ledger_digest,
            authored_program_digest=authored_program.digest,
            realized_program_digest=authored_program.digest,
            conflicts=conflicts,
            notes=[
                "Driving dimensions are strict; no automatic contract revision was authorized."
            ],
        )
        return (
            intent.model_copy(
                update={"constraint_ledger": ledger, "global_preflight": result}
            ),
            ledger,
            result,
        )

    closure_realized = _solve_closure_dimensions(
        intent,
        modeling_tolerance=modeling_tolerance,
    )
    if closure_realized is None:
        status = (
            "infeasible"
            if any(
                item.conflict_type == "closure" and item.proof_strength == "proved"
                for item in conflicts
            )
            else "unknown"
        )
        result = GlobalPreflightResult(
            method_version=PREFLIGHT_METHOD_VERSION,
            status=status,
            ledger_digest=ledger.ledger_digest,
            authored_program_digest=authored_program.digest,
            realized_program_digest=authored_program.digest,
            conflicts=conflicts,
            notes=[
                "The bounded affine closure projection found no positive dimension realization."
            ],
        )
        return (
            intent.model_copy(
                update={"constraint_ledger": ledger, "global_preflight": result}
            ),
            ledger,
            result,
        )

    solution = _minimum_uniform_scale(
        closure_realized,
        modeling_tolerance=modeling_tolerance,
        max_scale=max_uniform_scale,
    )
    if solution is None:
        status = (
            "infeasible"
            if proved
            and all(item.conflict_type in {"closure", "topology"} for item in proved)
            else "unknown"
        )
        result = GlobalPreflightResult(
            method_version=PREFLIGHT_METHOD_VERSION,
            status=status,
            ledger_digest=ledger.ledger_digest,
            authored_program_digest=authored_program.digest,
            realized_program_digest=authored_program.digest,
            conflicts=conflicts,
            notes=[
                "The supported minimal uniform relaxation could not certify a realization."
            ],
        )
        return (
            intent.model_copy(
                update={"constraint_ledger": ledger, "global_preflight": result}
            ),
            ledger,
            result,
        )

    factor, realized, realized_program, remaining = solution
    deviations = _deviations(intent, realized, ledger)
    result = GlobalPreflightResult(
        method_version=PREFLIGHT_METHOD_VERSION,
        status="adjusted",
        ledger_digest=ledger.ledger_digest,
        authored_program_digest=authored_program.digest,
        realized_program_digest=realized_program.digest,
        scale_factor=factor,
        deviations=deviations,
        conflicts=conflicts,
        notes=[
            "Authored conflicts are retained as evidence; the realized program passed the same verifier."
        ],
    )
    realized = realized.model_copy(
        update={
            "constraint_ledger": ledger,
            "global_preflight": result,
            "design_notes": [
                *realized.design_notes,
                (
                    f"Global preflight applied a {factor:.9g}x uniform centerline "
                    "scale; see global_preflight.json for source-bound deviations."
                ),
            ],
        }
    )
    return realized, ledger, result


def structural_intent_issues(
    prompt: str,
    intent: IntentResult,
    *,
    modeling_tolerance: float,
) -> list[str]:
    """수치 완화로 고칠 수 없는 구조적 작성 오류를 찾는다."""

    try:
        program = compile_centerline_program(
            intent,
            modeling_tolerance=modeling_tolerance,
        )
    except _UnsupportedProgram:
        return []
    issues: list[str] = []
    plane_match = _SOURCE_PLANE.search(prompt)
    if plane_match is not None:
        plane = plane_match.group("plane").upper()
        excluded_axis = {"XY": 2, "XZ": 1, "YZ": 0}[plane]
        coordinates = [
            point[excluded_axis]
            for primitive in program.primitives
            for point in (primitive.start, primitive.end)
        ]
        span = max(coordinates) - min(coordinates)
        allowed = max(
            modeling_tolerance * 10.0,
            float(intent.global_spec.outer_diameter) * 1e-6,
        )
        if span > allowed:
            issues.append(
                f"source requires the centerline in the {plane} plane, but the "
                f"compiled primitive program spans {span:.9g} mm along the "
                f"excluded {'XYZ'[excluded_axis]} axis. Re-author start_axis and "
                "every turn plane normal/sign in that source plane; do not rotate "
                "the requested design into a different coordinate plane"
            )

    structural_threshold = max(
        modeling_tolerance * 20.0,
        float(intent.global_spec.outer_diameter) * 1e-7,
    )
    conflicts = verify_centerline_program(
        program,
        intent,
        modeling_tolerance=modeling_tolerance,
    )
    for conflict in conflicts:
        if (
            conflict.conflict_type == "clearance"
            and conflict.proof_strength == "proved"
            and conflict.measured is not None
            and conflict.measured <= structural_threshold
        ):
            issues.append(
                "global centerline preflight proved a structural non-adjacent "
                f"crossing/overlap between {conflict.primitive_ids}: measured "
                f"distance {conflict.measured:.9g} mm, required at least "
                f"{float(conflict.required or 0.0):.9g} mm. Uniform dimension "
                "relaxation cannot separate coincident centerlines. Reinterpret "
                "the right/left turn signs, plane normal, or primitive topology "
                "while preserving the user's numeric values and shape identity"
            )
    return issues


__all__ = [
    "CenterlinePrimitive",
    "CenterlineProgram",
    "PREFLIGHT_METHOD_VERSION",
    "build_constraint_ledger",
    "compile_centerline_program",
    "preflight_and_realize_intent",
    "structural_intent_issues",
    "verify_centerline_program",
]
