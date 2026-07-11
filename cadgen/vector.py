"""파이프 좌표계에 쓰는 3차원 벡터와 원호 프레임 계산을 제공한다.

유한한 좌표ㆍ각도ㆍ축을 입력받아 결정론적인 벡터와 표본점을 반환한다.
영벡터나 퇴화한 회전면처럼 정의되지 않는 계산은 명시적으로 실패한다.
"""

from __future__ import annotations

import math
from typing import Final, Iterable

Vector = tuple[float, float, float]
ARC_PLANE_MIN_SINE = 1e-6

# 방향 토큰 표는 좌표 변환 때마다 새로 만들 필요가 없는 불변 데이터다.
# 모듈에서 한 번만 구성해 반복적인 작은 객체 할당을 없앤다.
_DIRECTION_VECTORS: Final[dict[str, Vector]] = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
    "UP": (0.0, 0.0, 1.0),
    "DOWN": (0.0, 0.0, -1.0),
    "RIGHT": (0.0, 1.0, 0.0),
    "LEFT": (0.0, -1.0, 0.0),
    "FORWARD": (1.0, 0.0, 0.0),
    "BACK": (-1.0, 0.0, 0.0),
}

# 임의 벡터와 안정적으로 외적할 기준축을 우선순위대로 보관한다.
_PERPENDICULAR_AXIS_CANDIDATES: Final[tuple[Vector, ...]] = (
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
)


def vec(values: Iterable[float]) -> Vector:
    """세 좌표를 표준 float 벡터 형태로 변환한다."""

    x, y, z = values
    return (float(x), float(y), float(z))


def add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a: Vector, scale: float) -> Vector:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(a: Vector) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Vector) -> Vector:
    """영벡터와 비유한 벡터를 거부하고 같은 방향의 단위 벡터를 반환한다."""

    size = length(a)
    if not math.isfinite(size) or size <= 1e-12:
        raise ValueError("cannot normalize a zero-length or non-finite vector")
    return (a[0] / size, a[1] / size, a[2] / size)


def direction_to_vector(
    direction: str | None,
    default: Vector | None = None,
) -> Vector:
    """방향 토큰을 단위 벡터로 바꾼다.

    기본 동작은 fail-closed다. 알 수 없거나 비어 있는 토큰을 임의의 +X로
    치환하지 않는다. legacy dry-run처럼 명시적인 대체 규칙이 필요한 호출부만
    ``default``를 전달해야 한다.
    """

    if not direction:
        if default is None:
            raise ValueError("direction token is required")
        return default
    normalized = direction.strip().upper()
    vector = _DIRECTION_VECTORS.get(normalized)
    if vector is not None:
        return vector
    if default is not None:
        return default
    raise ValueError(f"unknown direction token: {direction!r}")


def rotate(v: Vector, axis: Vector, angle_rad: float) -> Vector:
    """Rodrigues 공식으로 벡터를 지정 축 주위에서 회전한다."""

    k = normalize(axis)
    cos_t = math.cos(angle_rad)
    sin_t = math.sin(angle_rad)
    term1 = mul(v, cos_t)
    term2 = mul(cross(k, v), sin_t)
    term3 = mul(k, dot(k, v) * (1.0 - cos_t))
    return add(add(term1, term2), term3)


def canonical_circular_arc_frame(
    inlet_tangent: Vector,
    plane_normal_hint: Vector,
    sweep_angle_degrees: float,
) -> tuple[Vector, Vector, Vector]:
    """일관된 평면 법선과 원호 양 끝의 해석적 접선을 반환한다.

    planner는 굽힘 평면 힌트와 부호 있는 회전각만 선택한다. 정확한 직교화와
    종단 접선은 resolver가 계산하므로, LLM이 연속성 검사를 맞추기 위해
    무리수 사인ㆍ코사인 벡터를 직접 작성할 필요가 없다.
    """

    start_tangent = normalize(inlet_tangent)
    hint = normalize(plane_normal_hint)
    projected_normal = sub(hint, mul(start_tangent, dot(hint, start_tangent)))
    if length(projected_normal) <= ARC_PLANE_MIN_SINE:
        raise ValueError(
            "plane_normal hint must be meaningfully non-parallel to the inlet tangent"
        )
    plane_normal = normalize(projected_normal)
    terminal_tangent = normalize(
        rotate(
            start_tangent,
            plane_normal,
            math.radians(float(sweep_angle_degrees)),
        )
    )
    return plane_normal, start_tangent, terminal_tangent


def circular_rim_mismatch(
    position_error: float,
    radius_a: float,
    radius_b: float,
    alignment_cosine: float,
) -> float:
    """두 원형 접속면 테두리 사이의 보수적인 위치 불일치 상한을 계산한다.

    ``alignment_cosine``은 접속 규약을 적용한 뒤의 코사인이다. 경로 접선은
    평행, 바깥쪽 포트 축은 반평행을 기준으로 한다. 회전 오차는 두 축을
    일치시키는 최소 회전에서 큰 반지름의 점이 이동할 수 있는 최대 거리이며,
    중심과 반지름 오차는 삼각부등식으로 더한다.
    """

    values = (position_error, radius_a, radius_b, alignment_cosine)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("circular interface values must be finite")
    if position_error < 0.0 or radius_a < 0.0 or radius_b < 0.0:
        raise ValueError("circular interface distances must be non-negative")
    cosine = max(-1.0, min(1.0, float(alignment_cosine)))
    angle = math.acos(cosine)
    rotational_error = (
        2.0 * max(float(radius_a), float(radius_b)) * math.sin(angle / 2.0)
    )
    return (
        float(position_error)
        + abs(float(radius_a) - float(radius_b))
        + rotational_error
    )


def choose_perpendicular_axis(v: Vector) -> Vector:
    """입력 벡터와 안정적으로 직교하는 결정론적 단위축을 선택한다."""

    base = normalize(v)
    for candidate in _PERPENDICULAR_AXIS_CANDIDATES:
        if abs(dot(base, candidate)) < 0.85:
            return normalize(cross(base, candidate))
    return (0.0, 1.0, 0.0)


def arc_points(
    start: Vector,
    in_axis: Vector,
    out_axis: Vector,
    radius: float,
    angle_deg: float,
    segments: int,
) -> list[Vector]:
    """두 접선 방향과 반지름으로 원호의 균등 각도 표본점을 만든다."""

    u = normalize(in_axis)
    v = normalize(out_axis)
    theta = math.radians(max(1.0, min(abs(angle_deg), 180.0)))
    plane_axis = cross(u, v)
    if length(plane_axis) < 1e-6:
        plane_axis = choose_perpendicular_axis(u)
        if dot(u, v) > 0:
            v = rotate(u, plane_axis, theta)
    else:
        plane_axis = normalize(plane_axis)
    radius_vec = normalize(cross(u, plane_axis))
    center = sub(start, mul(radius_vec, radius))

    points = []
    for i in range(segments + 1):
        t = theta * (i / segments)
        points.append(add(center, mul(rotate(radius_vec, plane_axis, t), radius)))
    return points
