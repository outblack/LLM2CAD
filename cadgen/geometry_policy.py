"""FreeCAD 커널 안정성에 필요한 공통 기하 안전 정책을 계산한다.

파이프 단면ㆍ허용 오차ㆍ사용자 하한을 입력받아 결정론적 곡률 하한을 반환한다.
유효하지 않은 치수는 추정값으로 대체하지 않고 ``ValueError``로 거부한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from cadgen.vector import add, cross, length, mul, normalize, sub, vec


# OCC pipe sweeps become numerically fragile when the centerline curvature
# radius barely exceeds the circular profile radius, especially on spatial
# inflections.  A merely non-singular sweep can still look pinched and amplify
# C1 curvature changes into visible ribs.  Keep one additional tube radius as a
# general visual-quality/construction reserve unless the user authored a larger
# bound.  Qualitative relative waypoint skeletons may be uniformly expanded by
# the separately capped safety policy below; fixed/user coordinates are never
# silently rewritten.
SPLINE_REGULARITY_MARGIN_FRACTION = 1.0

# A qualitative LLM skeleton should already have broadly plausible proportions.
# Larger corrections would create an unexpectedly huge part and hide a poor
# route contract, so return it to intent repair instead of silently magnifying it.
MAX_QUALITATIVE_WAYPOINT_SAFETY_SCALE = 4.0

# Three-point arc reconstruction performs several cross products and divisions
# before returning a radius.  Treat only a handful of representational ULPs as
# exact equality; the much larger modeling tolerance must never excuse a real
# negative profile clearance.
# A three-point OCC circle fit followed by edge-center/radius extraction crosses
# several normalized-vector, transform, and square-root operations.  The exact
# horn boundary has been observed to return roughly 16 binary64 ULPs below the
# authored radius after that round trip.  Keep a bounded 64-ULP representation
# band: it covers accumulated floating-point evaluation error while remaining
# many orders of magnitude smaller than any modeling tolerance or physical
# clearance, so a genuinely spindle/self-intersecting sweep is still rejected.
CIRCULAR_SWEEP_EQUALITY_ULPS = 64


CircularSweepRadiusClassification = Literal[
    "regular",
    "horn_boundary",
    "self_intersecting",
]


@dataclass(frozen=True)
class CircularSweepRadiusAssessment:
    """Classify a circular-profile sweep from its exact radial clearance.

    A circular centerline swept by an outer profile radius forms a ring torus
    above the boundary, a horn torus at equality, and a self-intersecting
    spindle torus below it.  Modeling tolerance is deliberately absent here:
    it must not turn a genuinely negative clearance into accepted geometry.
    """

    centerline_radius: float
    outer_profile_radius: float
    raw_radial_clearance: float
    radial_clearance: float
    equality_roundoff_band: float
    classification: CircularSweepRadiusClassification

    @property
    def supported_by_analytic_torus(self) -> bool:
        return self.classification != "self_intersecting"


def classify_circular_sweep_radius(
    centerline_radius: float,
    outer_profile_radius: float,
) -> CircularSweepRadiusAssessment:
    """Return the exact ring/horn/spindle classification for a circular sweep."""

    centerline = float(centerline_radius)
    profile = float(outer_profile_radius)
    if not math.isfinite(centerline) or centerline <= 0.0:
        raise ValueError("centerline_radius must be finite and positive")
    if not math.isfinite(profile) or profile <= 0.0:
        raise ValueError("outer_profile_radius must be finite and positive")
    raw_clearance = centerline - profile
    equality_roundoff_band = CIRCULAR_SWEEP_EQUALITY_ULPS * max(
        math.ulp(centerline),
        math.ulp(profile),
    )
    clearance = 0.0 if abs(raw_clearance) <= equality_roundoff_band else raw_clearance
    if clearance < 0.0:
        classification: CircularSweepRadiusClassification = "self_intersecting"
    elif clearance == 0.0:
        classification = "horn_boundary"
    else:
        classification = "regular"
    return CircularSweepRadiusAssessment(
        centerline_radius=centerline,
        outer_profile_radius=profile,
        raw_radial_clearance=raw_clearance,
        radial_clearance=clearance,
        equality_roundoff_band=equality_roundoff_band,
        classification=classification,
    )


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


def minimum_spline_curvature_radius(
    outer_diameter: float,
    modeling_tolerance: float,
    authored_minimum: float | None = None,
) -> float:
    """단면과 허용 오차에서 sweep에 필요한 최소 중심선 곡률 반경을 계산한다."""

    diameter = float(outer_diameter)
    tolerance = float(modeling_tolerance)
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("outer_diameter must be finite and positive")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("modeling_tolerance must be finite and positive")
    tube_radius = diameter / 2.0
    regularity_margin = max(
        tolerance * 10.0,
        tube_radius * SPLINE_REGULARITY_MARGIN_FRACTION,
    )
    return max(
        tube_radius + regularity_margin,
        float(authored_minimum or 0.0),
    )
