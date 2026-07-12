"""FreeCAD 독립적인 중심선 기하 분석을 제공한다.

생성 커널을 호출하지 않고 production C1 spline과 같은 결정론적 표본화를
수행한다. 안전 정책 임계값은 ``geometry_safety_policy``가 소유하고, 이 모듈은
작성된 중심선의 측정값만 반환한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from cadgen.vector3_math import add, cross, length, mul, normalize, sub, vec


@dataclass(frozen=True)
class C1SplinePrediction:
    """FreeCAD와 같은 C1 cubic 구성으로 계산한 kernel 독립 측정값이다."""

    minimum_radius: float
    handle_factors: tuple[float, ...]
    curve_length: float
    polyline_length: float
    minimum_chord: float
    critical_span_index: int | None
    critical_t: float | None
    critical_position: tuple[float, float, float] | None


def predict_c1_spline(
    points: list[tuple[float, float, float]],
    initial_tangent: tuple[float, float, float],
    final_tangent: tuple[float, float, float] | None,
    *,
    modeling_tolerance: float,
) -> C1SplinePrediction:
    """OCC 호출 전에 production spline의 곡률과 길이를 동일하게 예측한다.

    waypoint와 endpoint tangent는 LLM 소유 입력이고, Bezier handle factor는
    FreeCAD 생성기와 동일한 고정 후보ㆍ2-pass coordinate descent로 결정한다.
    따라서 실패가 확실한 후보를 유료 MCP/B-Rep 호출 전에 거부할 수 있다.
    """

    tolerance = float(modeling_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("modeling_tolerance must be finite and positive")
    canonical_points = [vec(point) for point in points]
    if len(canonical_points) < 2:
        raise ValueError("spline path needs at least two points")
    chords = [
        length(sub(right, left))
        for left, right in zip(canonical_points, canonical_points[1:])
    ]
    if any(chord <= tolerance for chord in chords):
        raise ValueError("spline path contains coincident or near-coincident waypoints")

    tangents = [normalize(vec(initial_tangent))]
    for index in range(1, len(canonical_points) - 1):
        bisector = add(
            normalize(sub(canonical_points[index], canonical_points[index - 1])),
            normalize(sub(canonical_points[index + 1], canonical_points[index])),
        )
        if length(bisector) <= 1e-9:
            raise ValueError("spline waypoint forms a 180-degree cusp")
        tangents.append(normalize(bisector))
    tangents.append(
        normalize(vec(final_tangent))
        if final_tangent is not None
        else normalize(sub(canonical_points[-1], canonical_points[-2]))
    )

    local_scales = [chords[0]]
    local_scales.extend(
        min(chords[index - 1], chords[index])
        for index in range(1, len(canonical_points) - 1)
    )
    local_scales.append(chords[-1])

    def evaluate(
        factors: list[float],
        *,
        samples_per_span: int,
        measure_length: bool = False,
    ) -> tuple[
        float,
        float,
        int | None,
        float | None,
        tuple[float, float, float] | None,
    ]:
        """handle factor로 곡률 최솟값과 선택적 곡선 길이를 표본 측정한다."""

        handles = [factor * scale for factor, scale in zip(factors, local_scales)]
        maximum_curvature = 0.0
        sampled_length = 0.0
        critical_span_index: int | None = None
        critical_t: float | None = None
        critical_position: tuple[float, float, float] | None = None
        for index in range(len(canonical_points) - 1):
            p0 = canonical_points[index]
            p1 = add(p0, mul(tangents[index], handles[index]))
            p3 = canonical_points[index + 1]
            p2 = sub(p3, mul(tangents[index + 1], handles[index + 1]))
            previous_position: tuple[float, float, float] | None = None
            for sample_index in range(samples_per_span):
                parameter = sample_index / float(samples_per_span - 1)
                complement = 1.0 - parameter
                first_derivative = add(
                    add(
                        mul(sub(p1, p0), 3.0 * complement * complement),
                        mul(sub(p2, p1), 6.0 * complement * parameter),
                    ),
                    mul(sub(p3, p2), 3.0 * parameter * parameter),
                )
                second_derivative = add(
                    mul(add(sub(p2, mul(p1, 2.0)), p0), 6.0 * complement),
                    mul(add(sub(p3, mul(p2, 2.0)), p1), 6.0 * parameter),
                )
                position = add(
                    add(
                        mul(p0, complement**3),
                        mul(p1, 3.0 * complement * complement * parameter),
                    ),
                    add(
                        mul(p2, 3.0 * complement * parameter * parameter),
                        mul(p3, parameter**3),
                    ),
                )
                speed = length(first_derivative)
                if speed <= tolerance:
                    return 0.0, 0.0, index, parameter, position
                curvature = length(cross(first_derivative, second_derivative)) / (
                    speed**3
                )
                if curvature > maximum_curvature:
                    maximum_curvature = curvature
                    critical_span_index = index
                    critical_t = parameter
                    critical_position = position
                if measure_length:
                    if previous_position is not None:
                        sampled_length += length(sub(position, previous_position))
                    previous_position = position
        minimum_radius = 1.0 / maximum_curvature if maximum_curvature > 1e-12 else 1e30
        return (
            minimum_radius,
            sampled_length,
            critical_span_index,
            critical_t,
            critical_position,
        )

    factors = [0.4] * len(canonical_points)
    candidates = (0.4, 0.35, 0.45, 0.3, 0.5, 0.25, 0.55)
    for _pass_index in range(2):
        for node_index in range(len(factors)):
            best_factor = factors[node_index]
            best_score, *_unused = evaluate(factors, samples_per_span=33)
            for candidate_factor in candidates:
                candidate = list(factors)
                candidate[node_index] = candidate_factor
                candidate_score, *_unused = evaluate(
                    candidate,
                    samples_per_span=33,
                )
                if candidate_score > best_score * (1.0 + 1e-9):
                    best_score = candidate_score
                    best_factor = candidate_factor
            factors[node_index] = best_factor

    (
        minimum_radius,
        curve_length,
        critical_span_index,
        critical_t,
        critical_position,
    ) = evaluate(
        factors,
        samples_per_span=257,
        measure_length=True,
    )
    return C1SplinePrediction(
        minimum_radius=minimum_radius,
        handle_factors=tuple(float(factor) for factor in factors),
        curve_length=curve_length,
        polyline_length=sum(chords),
        minimum_chord=min(chords),
        critical_span_index=critical_span_index,
        critical_t=critical_t,
        critical_position=critical_position,
    )


__all__ = ["C1SplinePrediction", "predict_c1_spline"]
