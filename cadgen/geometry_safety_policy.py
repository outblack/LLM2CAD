"""FreeCAD 커널 안정성에 필요한 공통 기하 안전 정책을 계산한다.

파이프 단면ㆍ허용 오차ㆍ사용자 하한을 입력받아 결정론적 곡률 하한을 반환한다.
유효하지 않은 치수는 추정값으로 대체하지 않고 ``ValueError``로 거부한다.
순수 C1 spline 측정은 ``cadgen.geometry_analysis``에 있으며 기존 import
호환성을 위해 이 모듈에서도 다시 노출한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from cadgen.geometry_analysis import C1SplinePrediction, predict_c1_spline
from cadgen.vector3_math import add, cross, length, mul, normalize, sub, vec  # noqa: F401


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
    """원형 단면 스윕을 반경 여유 기준으로 분류한다."""

    centerline_radius: float
    outer_profile_radius: float
    raw_radial_clearance: float
    radial_clearance: float
    equality_roundoff_band: float
    classification: CircularSweepRadiusClassification

    @property
    def supported_by_analytic_torus(self) -> bool:
        """자기교차 spindle이 아니면 해석적 torus 생성을 허용한다."""

        return self.classification != "self_intersecting"


def classify_circular_sweep_radius(
    centerline_radius: float,
    outer_profile_radius: float,
) -> CircularSweepRadiusAssessment:
    """원형 스윕의 ring/horn/spindle 분류를 정확히 계산한다."""

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


def minimum_spline_curvature_radius(
    outer_diameter: float,
    modeling_tolerance: float,
    authored_minimum: float | None = None,
    *,
    enforcement: Literal["physical_only", "strict"] = "strict",
) -> float:
    """Return the hard spline sweep radius for the selected enforcement policy.

    ``physical_only`` blocks the non-singular sweep boundary only. ``strict``
    retains the historical extra tube-radius reserve for visual regularity.
    An authored minimum always remains authoritative in either mode.
    """

    diameter = float(outer_diameter)
    tolerance = float(modeling_tolerance)
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("outer_diameter must be finite and positive")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("modeling_tolerance must be finite and positive")
    if enforcement not in {"physical_only", "strict"}:
        raise ValueError("enforcement must be physical_only or strict")
    tube_radius = diameter / 2.0
    kernel_margin = max(
        tolerance * 10.0,
        CIRCULAR_SWEEP_EQUALITY_ULPS * math.ulp(tube_radius),
    )
    regularity_margin = (
        max(kernel_margin, tube_radius * SPLINE_REGULARITY_MARGIN_FRACTION)
        if enforcement == "strict"
        else kernel_margin
    )
    return max(
        tube_radius + regularity_margin,
        float(authored_minimum or 0.0),
    )

