"""검증 대상 ``PipeState``를 FreeCAD 후보ㆍ게시 스크립트로 직렬화한다.

타입이 확정된 모듈/포트 그래프를 입력받아 digest가 포함된 Python 코드를 출력한다.
상태 불변식이나 B-Rep 검사가 맞지 않으면 임의 형상을 만들지 않고 실패한다.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from cadgen.geometry_policy import CIRCULAR_SWEEP_EQUALITY_ULPS
from cadgen.schemas import PipeState


GENERATOR_VERSION = "cadgen02-freecad-v24"
VALIDATION_SCHEMA_VERSION = 3
VALIDATOR_POLICY_ID = "cadgen02-freecad-validator-v24"
ADJACENT_INTERFACE_POLICY_ID = "resolver-local-interface-band-v1"
_VALIDATOR_POLICY_SPEC = {
    "policy_id": VALIDATOR_POLICY_ID,
    "generator_version": GENERATOR_VERSION,
    "validation_schema_version": VALIDATION_SCHEMA_VERSION,
    "structured_module_error_schema_version": 1,
    "circular_arc_construction": {
        "method": "analytic_torus_segment_outer_minus_bore",
        "minimum_centerline_radius": "outer_profile_radius",
        "horn_boundary_supported": True,
        "self_intersecting_spindle_rejected": True,
        "equality_roundoff_policy": "scale_aware_ulps",
        "equality_ulps": CIRCULAR_SWEEP_EQUALITY_ULPS,
    },
    "adjacent_interface_overlap": {
        "policy_id": ADJACENT_INTERFACE_POLICY_ID,
        "engagement_tolerance_multiplier": 20.0,
        "engagement_radius_multiplier": 1e-4,
        "margin_tolerance_multiplier": 20.0,
        "margin_radius_multiplier": 1e-6,
        "outside_volume_relative_allowance": 1e-8,
        "minimum_outlet_forward_dot": -1e-9,
    },
}
VALIDATOR_POLICY_DIGEST = hashlib.sha256(
    json.dumps(
        _VALIDATOR_POLICY_SPEC,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()
PUBLISHED_VIEW_DEVIATION_PERCENT = 0.05
PUBLISHED_VIEW_ANGULAR_DEFLECTION_DEGREES = 5.0
PUBLISHED_VIEW_SPECULAR_COLOR = (0.12, 0.12, 0.12)
PUBLISHED_VIEW_SHININESS = 0.25


def physical_root_port_payload(state: PipeState) -> dict[str, Any] | None:
    """현재 열려 있는 구성 루트 접속면이 있으면 payload로 반환한다.

    일반적인 열린 체인은 가상 START cursor를 유지한다. 아직 닫히지 않은
    폐곡선은 첫 모듈의 실제 inlet으로 이를 대체한다. 저장된 축은 바깥쪽을
    향하므로 FreeCAD 루트 검사는 반대인 안쪽 방향을 사용한다. connect_ports가
    예약 inlet을 소비하면 두 루트 표현 모두 사라지고 anchored inlet은 0이 된다.
    """

    start = state.port_nodes.get("START")
    if start is not None:
        return start.model_dump(mode="json")
    anchor = state.reserved_start_anchor
    if anchor is None:
        return None
    payload = anchor.model_dump(mode="json")
    payload["axis"] = [-float(component) for component in anchor.axis]
    return payload


def anchored_inlet_count(state: PipeState) -> int:
    """현재 상태에서 기대되는 FreeCAD 루트 inlet 수를 반환한다."""

    return int(physical_root_port_payload(state) is not None)


def geometry_payload(state: PipeState) -> dict[str, Any]:
    """형상 생성에 필요한 상태 부분만 안정적인 JSON payload로 만든다."""

    return {
        "generator_version": GENERATOR_VERSION,
        "validator_policy_id": VALIDATOR_POLICY_ID,
        "validator_policy_digest": VALIDATOR_POLICY_DIGEST,
        "state_id": state.state_id,
        "state_version": state.state_version,
        "contract_digest": state.contract_digest,
        "modeling_tolerance": state.modeling_tolerance,
        "root_port": physical_root_port_payload(state),
        "modules": [
            module.model_dump(mode="json")
            for module in state.placed_modules
            if not (
                module.type == "connect_ports"
                and module.params.get("path_kind") == "seam"
            )
        ],
        "topological_modules": [
            module.model_dump(mode="json")
            for module in state.placed_modules
            if module.type == "connect_ports"
            and module.params.get("path_kind") == "seam"
        ],
        "open_ports": [port.model_dump(mode="json") for port in state.open_ports],
        "connection_edges": [
            edge.model_dump(mode="json") for edge in state.connection_edges
        ],
    }


def geometry_payload_digest(state: PipeState) -> str:
    """정규화된 형상 payload의 SHA-256 식별자를 계산한다."""

    return _geometry_payload_digest(geometry_payload(state))


def _geometry_payload_digest(payload: dict[str, Any]) -> str:
    """이미 만든 형상 payload를 다시 생성하지 않고 digest로 변환한다."""

    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def candidate_document_name(
    state: PipeState,
    *,
    run_id: str = "local",
    attempt_id: int = 1,
) -> str:
    """실행ㆍ상태ㆍ시도ㆍpayload digest가 결합된 후보 문서 이름을 만든다."""

    digest = geometry_payload_digest(state)
    return _candidate_document_name(
        state,
        run_id=run_id,
        attempt_id=attempt_id,
        digest=digest,
    )


def _candidate_document_name(
    state: PipeState,
    *,
    run_id: str,
    attempt_id: int,
    digest: str,
) -> str:
    """계산된 digest를 재사용해 후보 FreeCAD 문서 이름을 만든다."""

    safe_run_id = re.sub(r"[^A-Za-z0-9_]", "_", run_id)[:48] or "local"
    return f"CadGenCandidate_{safe_run_id}_{state.state_version}_{attempt_id}_{digest[:12]}"


def published_document_name(state: PipeState, *, run_id: str = "local") -> str:
    """실행과 상태 버전으로 식별되는 최종 게시 문서 이름을 만든다."""

    safe_run_id = re.sub(r"[^A-Za-z0-9_]", "_", run_id)[:48] or "local"
    return f"CadGenPipe_{safe_run_id}_v{state.state_version}"


def build_freecad_script(
    state: PipeState,
    *,
    run_id: str = "local",
    attempt_id: int = 1,
    modeling_tolerance: float | None = None,
) -> str:
    """후보 B-Rep 생성과 자체 검증을 수행하는 독립 FreeCAD 스크립트를 만든다."""

    if modeling_tolerance is None:
        modeling_tolerance = state.modeling_tolerance
    if abs(float(modeling_tolerance) - state.modeling_tolerance) > 1e-15:
        raise ValueError("modeling_tolerance differs from the immutable PipeState")
    # 스크립트 본문, 후보 이름과 검증 digest가 모두 같은 payload를 사용한다.
    # 큰 모듈 그래프를 단계 하나에서 여러 번 직렬화하지 않도록 한 번만 만든다.
    payload = geometry_payload(state)
    digest = _geometry_payload_digest(payload)
    payload_json_literal = repr(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    validator_policy_json_literal = repr(
        json.dumps(_VALIDATOR_POLICY_SPEC, ensure_ascii=True, separators=(",", ":"))
    )
    candidate_name = _candidate_document_name(
        state,
        run_id=run_id,
        attempt_id=attempt_id,
        digest=digest,
    )
    template = r'''# Generated by cadgen02. Candidate build only; publishing is a separate commit phase.
import json
import math
import hashlib
import FreeCAD as App
import Part

PAYLOAD = json.loads(__PAYLOAD_JSON__)
MODULES = PAYLOAD["modules"]
CANDIDATE_DOCUMENT = __CANDIDATE_NAME__
PAYLOAD_DIGEST = __PAYLOAD_DIGEST__
VALIDATION_SCHEMA_VERSION = __VALIDATION_SCHEMA_VERSION__
MODELING_TOLERANCE = __MODELING_TOLERANCE__
GENERATOR_VERSION = __GENERATOR_VERSION__
VALIDATOR_POLICY = json.loads(__VALIDATOR_POLICY_JSON__)
VALIDATOR_POLICY_ID = __VALIDATOR_POLICY_ID__
VALIDATOR_POLICY_DIGEST = __VALIDATOR_POLICY_DIGEST__
RUN_ID = __RUN_ID__
ATTEMPT_ID = __ATTEMPT_ID__
SPLINE_HANDLE_FACTOR_CACHE = {}


def vector(values):
    return App.Vector(float(values[0]), float(values[1]), float(values[2]))


def shape_fingerprint(shape):
    raw = shape.exportBrepToString()
    if not isinstance(raw, str) or not raw:
        raise ValueError("candidate B-Rep serialization is unavailable")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def vector_json(value):
    return [float(value.x), float(value.y), float(value.z)]


def normalized(value):
    result = App.Vector(value.x, value.y, value.z)
    if result.Length <= 1e-12:
        raise ValueError("zero-length direction")
    result.normalize()
    return result


def segment_distance(first_start, first_end, second_start, second_end):
    u = first_end - first_start
    v = second_end - second_start
    w = first_start - second_start
    a = u.dot(u)
    b = u.dot(v)
    c = v.dot(v)
    d = u.dot(w)
    e = v.dot(w)
    denominator = a * c - b * b
    epsilon = 1e-24
    if a <= epsilon and c <= epsilon:
        return float(w.Length)
    if a <= epsilon:
        first_parameter = 0.0
        second_parameter = max(0.0, min(1.0, e / c))
    elif c <= epsilon:
        second_parameter = 0.0
        first_parameter = max(0.0, min(1.0, -d / a))
    elif denominator <= 64.0 * math.ulp(1.0) * a * c:
        def point_segment(point, start, end):
            segment = end - start
            squared = segment.dot(segment)
            if squared <= epsilon:
                return float((point - start).Length)
            parameter = max(
                0.0,
                min(1.0, (point - start).dot(segment) / squared),
            )
            return float((point - (start + segment * parameter)).Length)
        return min(
            point_segment(first_start, second_start, second_end),
            point_segment(first_end, second_start, second_end),
            point_segment(second_start, first_start, first_end),
            point_segment(second_end, first_start, first_end),
        )
    else:
        first_parameter = max(
            0.0,
            min(1.0, (b * e - c * d) / denominator),
        )
        second_parameter = (b * first_parameter + e) / c
        if second_parameter < 0.0:
            second_parameter = 0.0
            first_parameter = max(0.0, min(1.0, -d / a))
        elif second_parameter > 1.0:
            second_parameter = 1.0
            first_parameter = max(0.0, min(1.0, (b - d) / a))
    closest_first = first_start + u * first_parameter
    closest_second = second_start + v * second_parameter
    return float((closest_first - closest_second).Length)


def perpendicular(value):
    axis = normalized(value)
    candidates = [App.Vector(0, 0, 1), App.Vector(0, 1, 0), App.Vector(1, 0, 0)]
    for candidate in candidates:
        result = axis.cross(candidate)
        if result.Length > 1e-9:
            return normalized(result)
    raise ValueError("could not derive perpendicular component axis")


def safe_radius(value):
    result = float(value)
    if result <= MODELING_TOLERANCE:
        raise ValueError("radius is below modeling tolerance")
    return result


def inner_radius(outer_diameter, wall_thickness):
    return safe_radius(float(outer_diameter) / 2.0 - float(wall_thickness))


def make_circle_wire(radius, center, normal):
    return Part.Wire([Part.makeCircle(safe_radius(radius), center, normalized(normal))])


def make_spline_wire(
    points,
    initial_tangent=None,
    final_tangent=None,
    return_handle_factors=False,
):
    """Build a waypoint-interpolating, endpoint-tangent-safe cubic spline.

    OCC's global ``BSplineCurve.interpolate`` can introduce tight endpoint
    loops when endpoint tangent constraints are combined with chord-length
    parameterization.  A chain of cubic Bezier spans is the equivalent clamped
    piecewise B-spline representation, while making every waypoint and endpoint
    tangent explicit.  Shared node handles use one direction and magnitude on
    both sides, so joins are C1 rather than merely visually close.
    """
    vectors = [vector(point) for point in points]
    if len(vectors) < 2:
        raise ValueError("spline path needs at least two points")
    chords = [float((right - left).Length) for left, right in zip(vectors, vectors[1:])]
    if any(length <= MODELING_TOLERANCE for length in chords):
        raise ValueError("spline path contains coincident or near-coincident waypoints")

    tangents = [
        normalized(vector(initial_tangent))
        if initial_tangent is not None
        else normalized(vectors[1] - vectors[0])
    ]
    for index in range(1, len(vectors) - 1):
        incoming = normalized(vectors[index] - vectors[index - 1])
        outgoing = normalized(vectors[index + 1] - vectors[index])
        bisector = incoming + outgoing
        if bisector.Length <= 1e-9:
            raise ValueError(
                "spline waypoint forms a 180-degree cusp; add separated turning waypoints"
            )
        tangents.append(normalized(bisector))
    tangents.append(
        normalized(vector(final_tangent))
        if final_tangent is not None
        else normalized(vectors[-1] - vectors[-2])
    )

    # The LLM owns waypoint placement and endpoint directions. Handle magnitude
    # is a dependent kernel parameter: optimize it deterministically for the
    # largest sampled minimum curvature radius while keeping every handle local
    # to adjacent chord scale. This avoids both a one-size-fits-all tension and
    # case-specific waypoint rewriting.
    local_scales = [chords[0]]
    local_scales.extend(
        min(chords[index - 1], chords[index])
        for index in range(1, len(vectors) - 1)
    )
    local_scales.append(chords[-1])
    cache_key = (
        tuple(
            tuple(round(float(component), 12) for component in vector_json(point))
            for point in vectors
        ),
        tuple(
            tuple(round(float(component), 12) for component in vector_json(tangent))
            for tangent in tangents
        ),
    )

    def sampled_minimum_radius(factors):
        handles = [
            float(factor) * float(scale)
            for factor, scale in zip(factors, local_scales)
        ]
        maximum_curvature = 0.0
        for index in range(len(vectors) - 1):
            p0 = vectors[index]
            p1 = p0 + tangents[index] * handles[index]
            p3 = vectors[index + 1]
            p2 = p3 - tangents[index + 1] * handles[index + 1]
            for sample_index in range(33):
                parameter = sample_index / 32.0
                complement = 1.0 - parameter
                first_derivative = (
                    (p1 - p0) * (3.0 * complement * complement)
                    + (p2 - p1) * (6.0 * complement * parameter)
                    + (p3 - p2) * (3.0 * parameter * parameter)
                )
                second_derivative = (
                    (p2 - p1 * 2.0 + p0) * (6.0 * complement)
                    + (p3 - p2 * 2.0 + p1) * (6.0 * parameter)
                )
                speed = float(first_derivative.Length)
                if speed <= MODELING_TOLERANCE:
                    return 0.0
                curvature = float(
                    first_derivative.cross(second_derivative).Length
                ) / (speed ** 3)
                maximum_curvature = max(maximum_curvature, curvature)
        return (
            1.0 / maximum_curvature
            if maximum_curvature > 1e-12
            else 1e30
        )

    factors = SPLINE_HANDLE_FACTOR_CACHE.get(cache_key)
    if factors is None:
        factors = [0.4] * len(vectors)
        candidate_factors = (0.4, 0.35, 0.45, 0.3, 0.5, 0.25, 0.55)
        # Two deterministic coordinate-descent passes let endpoint and adjacent
        # node handles co-adapt without a combinatorial search.
        for _pass_index in range(2):
            for node_index in range(len(factors)):
                best_factor = factors[node_index]
                best_score = sampled_minimum_radius(factors)
                for candidate_factor in candidate_factors:
                    candidate = list(factors)
                    candidate[node_index] = candidate_factor
                    score = sampled_minimum_radius(candidate)
                    if score > best_score * (1.0 + 1e-9):
                        best_score = score
                        best_factor = candidate_factor
                factors[node_index] = best_factor
        SPLINE_HANDLE_FACTOR_CACHE[cache_key] = list(factors)
    handles = [
        float(factor) * float(scale)
        for factor, scale in zip(factors, local_scales)
    ]

    span_edges = []
    for index in range(len(vectors) - 1):
        chord_direction = normalized(vectors[index + 1] - vectors[index])
        if (
            tangents[index].dot(chord_direction) >= 1.0 - 1e-10
            and tangents[index + 1].dot(chord_direction) >= 1.0 - 1e-10
        ):
            # A cubic Bezier whose two endpoint tangents are exactly collinear
            # with its chord is analytically a straight span.  Keeping that
            # span as a degree-3 curve can make OCC's pipe builder fail with
            # NCollection_IndexedDataMap::FindFromKey/MakeSolid even though
            # every sampled curvature check is valid.  Canonicalize the
            # degenerate representation, not the LLM-authored path geometry.
            span_edges.append(Part.makeLine(vectors[index], vectors[index + 1]))
            continue
        curve = Part.BezierCurve()
        curve.setPoles([
            vectors[index],
            vectors[index] + tangents[index] * handles[index],
            vectors[index + 1] - tangents[index + 1] * handles[index + 1],
            vectors[index + 1],
        ])
        span_edges.append(curve.toBSpline().toShape())
    # Preserve each C1 Bezier/B-spline span as an edge in one continuous wire.
    # OCC's corrected-frame pipe builder can fail at a spatial inflection when
    # the same spans are merged into one high-multiplicity B-spline edge. The
    # shared tangent/handle construction above already proves C1 continuity;
    # edge boundaries give OCC stable frame restart points at the waypoints.
    wire = Part.Wire(span_edges)
    if return_handle_factors:
        return wire, [float(factor) for factor in factors]
    return wire


def make_path_wire(points, path_kind="spline", initial_tangent=None, final_tangent=None):
    vectors = [vector(point) for point in points]
    if len(vectors) < 2:
        raise ValueError("path needs at least two points")
    if path_kind == "line":
        return Part.makePolygon(vectors)
    if path_kind == "circular_arc":
        if len(vectors) < 3:
            raise ValueError("arc path needs at least three sampled points")
        middle = vectors[len(vectors) // 2]
        return Part.Wire([Part.Arc(vectors[0], middle, vectors[-1]).toShape()])
    return make_spline_wire(points, initial_tangent, final_tangent)


def circular_sweep_radius_evidence(centerline_radius, outer_profile_radius):
    centerline = float(centerline_radius)
    profile = float(outer_profile_radius)
    if not math.isfinite(centerline) or centerline <= 0.0:
        raise ValueError("centerline radius must be finite and positive")
    if not math.isfinite(profile) or profile <= 0.0:
        raise ValueError("outer profile radius must be finite and positive")
    raw_clearance = centerline - profile
    equality_ulps = int(
        VALIDATOR_POLICY["circular_arc_construction"]["equality_ulps"]
    )
    roundoff_band = equality_ulps * max(
        math.ulp(centerline),
        math.ulp(profile),
    )
    clearance = 0.0 if abs(raw_clearance) <= roundoff_band else raw_clearance
    if clearance < 0.0:
        classification = "self_intersecting"
    elif clearance == 0.0:
        classification = "horn_boundary"
    else:
        classification = "regular"
    return {
        "centerline_radius": centerline,
        "canonical_centerline_radius": (
            profile if classification == "horn_boundary" else centerline
        ),
        "outer_profile_radius": profile,
        "raw_radial_clearance": raw_clearance,
        "radial_clearance": clearance,
        "equality_roundoff_band": roundoff_band,
        "classification": classification,
        "supported": classification != "self_intersecting",
        "construction_method": (
            "analytic_torus_segment_outer_minus_bore"
        ),
    }


def analytic_circular_arc_frame(points):
    wire = make_path_wire(points, "circular_arc")
    edges = list(wire.Edges)
    if len(edges) != 1:
        raise ValueError("analytic circular arc requires exactly one edge")
    edge = edges[0]
    first = float(edge.FirstParameter)
    last = float(edge.LastParameter)
    start = edge.valueAt(first)
    tangent = normalized(edge.tangentAt(first))
    try:
        center = edge.Curve.Center
    except Exception as exc:
        raise ValueError("circular arc has no analytic circle center") from exc
    radial_vector = start - center
    centerline_radius = float(radial_vector.Length)
    if centerline_radius <= MODELING_TOLERANCE:
        raise ValueError("circular arc centerline radius is degenerate")
    radial_axis = normalized(radial_vector)
    directed_normal = normalized(radial_axis.cross(tangent))
    sweep_degrees = math.degrees(float(edge.Length) / centerline_radius)
    if not (0.0 < sweep_degrees < 360.0):
        raise ValueError("analytic torus sweep magnitude must be in (0, 360)")
    return {
        "center": center,
        "radial_axis": radial_axis,
        "tangent_axis": tangent,
        "directed_normal": directed_normal,
        "centerline_radius": centerline_radius,
        "sweep_degrees": sweep_degrees,
    }


def rigid_frame_transform(shape, origin, x_axis, y_axis, z_axis):
    matrix = App.Matrix()
    matrix.A11 = float(x_axis.x)
    matrix.A12 = float(y_axis.x)
    matrix.A13 = float(z_axis.x)
    matrix.A14 = float(origin.x)
    matrix.A21 = float(x_axis.y)
    matrix.A22 = float(y_axis.y)
    matrix.A23 = float(z_axis.y)
    matrix.A24 = float(origin.y)
    matrix.A31 = float(x_axis.z)
    matrix.A32 = float(y_axis.z)
    matrix.A33 = float(z_axis.z)
    matrix.A34 = float(origin.z)
    matrix.A44 = 1.0
    transformed = shape.copy()
    transformed.transformShape(matrix, True)
    return transformed


def make_analytic_torus_segment(frame, profile_radius):
    profile = safe_radius(profile_radius)
    assessment = circular_sweep_radius_evidence(
        frame["centerline_radius"],
        profile,
    )
    if not assessment["supported"]:
        raise ValueError(
            "circular sweep profile is self-intersecting: radial_clearance="
            + str(assessment["raw_radial_clearance"])
        )
    local = Part.makeTorus(
        float(assessment["canonical_centerline_radius"]),
        profile,
        App.Vector(0.0, 0.0, 0.0),
        App.Vector(0.0, 0.0, 1.0),
        -180.0,
        180.0,
        float(frame["sweep_degrees"]),
    )
    return rigid_frame_transform(
        local,
        frame["center"],
        frame["radial_axis"],
        frame["tangent_axis"],
        frame["directed_normal"],
    )


def make_analytic_circular_tube(points, outer_radius, bore_radius):
    frame = analytic_circular_arc_frame(points)
    outer_assessment = circular_sweep_radius_evidence(
        frame["centerline_radius"],
        outer_radius,
    )
    if not outer_assessment["supported"]:
        raise ValueError(
            "circular arc centerline radius is smaller than the outer profile "
            "radius; classification=self_intersecting"
        )
    canonical_frame = dict(frame)
    canonical_frame["raw_centerline_radius"] = frame["centerline_radius"]
    canonical_frame["centerline_radius"] = outer_assessment[
        "canonical_centerline_radius"
    ]
    outer = make_analytic_torus_segment(canonical_frame, outer_radius)
    bore = make_analytic_torus_segment(canonical_frame, bore_radius)
    return outer.cut(bore), outer, bore


def make_sweep_solid(
    points,
    radius,
    path_kind="spline",
    frenet=False,
    initial_tangent=None,
    final_tangent=None,
):
    wire = make_path_wire(points, path_kind, initial_tangent, final_tangent)
    first = vector(points[0])
    tangent = (
        normalized(vector(initial_tangent))
        if initial_tangent is not None
        else normalized(vector(points[1]) - first)
    )
    profile = make_circle_wire(radius, first, tangent)
    return wire.makePipeShell([profile], True, bool(frenet))


def make_swept_tube(
    points,
    outer_radius,
    bore_radius,
    path_kind="spline",
    frenet=False,
    initial_tangent=None,
    final_tangent=None,
):
    if initial_tangent is None or final_tangent is None:
        tangent_wire = make_path_wire(
            points, path_kind, initial_tangent, final_tangent
        )
        if not tangent_wire.Edges:
            raise ValueError("path wire has no edges for endpoint tangents")
        first_edge = tangent_wire.Edges[0]
        last_edge = tangent_wire.Edges[-1]
        if initial_tangent is None:
            initial_tangent = vector_json(
                normalized(first_edge.tangentAt(first_edge.FirstParameter))
            )
        if final_tangent is None:
            final_tangent = vector_json(
                normalized(last_edge.tangentAt(last_edge.LastParameter))
            )
    outer = make_sweep_solid(
        points, outer_radius, path_kind, frenet, initial_tangent, final_tangent
    )
    bore = make_sweep_solid(
        points, bore_radius, path_kind, frenet, initial_tangent, final_tangent
    )
    # Keep the inner path as one co-terminal sweep.  Fusing tiny endpoint
    # cylinders onto an exact curved sweep creates tangential Boolean seams in
    # OCC (SelfIntersect/TooSmallEdge), even though the authored arc itself is
    # valid.  A co-terminal outer-minus-inner cut is open, closed-shell valid,
    # and preserves the analytic endpoint tangents for line, arc, and spline.
    return outer.cut(bore), outer, bore


def composite_route_edges(modules):
    """Build one ordered OCC edge list from solved degree-2 route modules."""
    edges = []
    for module in modules:
        params = module["params"]
        points = params.get("path_points") or []
        if len(points) < 2:
            raise ValueError(
                "composite route module has no resolved centerline: "
                + str(module.get("id"))
            )
        kind = params.get("path_kind", "line")
        vectors = [vector(point) for point in points]
        if kind == "line":
            edges.append(Part.makeLine(vectors[0], vectors[-1]))
        elif kind == "circular_arc":
            middle = vectors[len(vectors) // 2]
            edges.append(Part.Arc(vectors[0], middle, vectors[-1]).toShape())
        elif kind == "spline":
            spline = make_spline_wire(
                points,
                params.get("initial_tangent"),
                params.get("final_tangent"),
            )
            edges.extend(list(spline.Edges))
        else:
            raise ValueError("unsupported composite route path_kind: " + str(kind))
    return edges


def make_composite_route_tube(modules):
    """Sweep one outer and bore profile over a maximal constant-section route."""
    if not modules:
        raise ValueError("composite route requires at least one module")
    sections = {
        (
            float(module["params"]["outer_diameter"]),
            float(module["params"]["wall_thickness"]),
        )
        for module in modules
    }
    if len(sections) != 1:
        raise ValueError("composite route requires one constant circular section")
    outer_diameter, wall_thickness = next(iter(sections))
    edges = composite_route_edges(modules)
    wire = Part.Wire(edges)
    if not wire.Edges:
        raise ValueError("composite route wire has no edges")
    first_edge = wire.Edges[0]
    first = first_edge.valueAt(first_edge.FirstParameter)
    tangent = normalized(first_edge.tangentAt(first_edge.FirstParameter))
    outer_radius = outer_diameter / 2.0
    bore_radius = inner_radius(outer_diameter, wall_thickness)
    outer_profile = make_circle_wire(outer_radius, first, tangent)
    bore_profile = make_circle_wire(bore_radius, first, tangent)
    outer = wire.makePipeShell([outer_profile], True, False)
    bore = wire.makePipeShell([bore_profile], True, False)
    material = outer.cut(bore).removeSplitter()
    if not valid_closed_single_solid(material):
        raise ValueError("composite centerline sweep did not produce one valid tube solid")
    return material, outer, bore


def make_cylinder_tube(start_values, end_values, outer_radius, bore_radius):
    start = vector(start_values)
    end = vector(end_values)
    delta = end - start
    height = delta.Length
    if height <= MODELING_TOLERANCE:
        raise ValueError("zero-length tube")
    axis = normalized(delta)
    outer = Part.makeCylinder(outer_radius, height, start, axis)
    extension = max(MODELING_TOLERANCE * 20.0, bore_radius * 1e-4)
    bore = Part.makeCylinder(
        bore_radius,
        height + extension * 2.0,
        start - axis * extension,
        axis,
    )
    return outer.cut(bore), outer, bore


def make_loft_tube(params):
    start = vector(params["start_position"])
    axis = normalized(vector(params["axis"]))
    offset = vector(params.get("offset", [0.0, 0.0, 0.0]))
    axial_length = float(params["length"])
    outer_start = float(params["diameter_in"]) / 2.0
    outer_end = float(params["diameter_out"]) / 2.0
    inner_start = inner_radius(
        params["diameter_in"], params["wall_thickness_in"]
    )
    inner_end = inner_radius(
        params["diameter_out"], params["wall_thickness_out"]
    )
    outer_profiles = []
    bore_profiles = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        # Quintic smootherstep has zero first and second derivative at both
        # mating profiles.  The intermediate loft sections therefore ease the
        # radial/eccentric change more gently than a cubic smoothstep while
        # preserving the exact authored inlet and outlet sections.
        blend = fraction ** 3 * (
            10.0 - 15.0 * fraction + 6.0 * fraction * fraction
        )
        center = (
            start
            + axis * (axial_length * fraction)
            + offset * blend
        )
        outer_radius = outer_start + (outer_end - outer_start) * blend
        bore_radius = inner_start + (inner_end - inner_start) * blend
        outer_profiles.append(make_circle_wire(outer_radius, center, axis))
        bore_profiles.append(make_circle_wire(bore_radius, center, axis))
    outer = Part.makeLoft(outer_profiles, True, False)
    bore = Part.makeLoft(bore_profiles, True, False)
    return outer.cut(bore), outer, bore


def fuse_shapes(shapes):
    result = None
    for shape in shapes:
        if shape is None:
            continue
        result = shape if result is None else result.fuse(shape)
        if result is not None:
            result = result.removeSplitter()
    if result is None:
        raise ValueError("no shape was produced")
    return result


def valid_closed_single_solid(shape):
    return bool(
        shape is not None
        and not shape.isNull()
        and shape.isValid()
        and shape.isClosed()
        and len(shape.Solids) == 1
        and float(shape.Volume) > MODELING_TOLERANCE ** 3
    )


def cylinder_face_radius(face):
    surface = face.Surface
    if surface.__class__.__name__ != "Cylinder":
        return None
    return float(surface.Radius)


def compact_junction_seams(shape, allowed_radii, center, terminal_points):
    """Find only the unblended Cylinder↔Cylinder seams in one junction material."""
    maximum_radius = max(float(value) for value in allowed_radii)
    radius_tolerance = max(
        MODELING_TOLERANCE * 10.0,
        maximum_radius * 1e-7,
    )
    terminal_tolerance = max(
        MODELING_TOLERANCE * 50.0,
        maximum_radius * 1e-4,
    )
    # ``make_junction`` adds a tolerance-scale inlet engagement so OCC can fuse
    # the child to its parent robustly.  That tiny helper cylinder can leave a
    # short Cylinder↔Cylinder edge near the outer rim.  It is construction
    # topology, not the authored arm seam, and attempting the exact user fillet
    # on it produces the characteristic "no suitable edges" failure.  Derive
    # the cutoff from the same engagement policy instead of from the authored
    # blend radius, which must never be silently changed.
    engagement_scale = max(
        MODELING_TOLERANCE * 20.0,
        maximum_radius * 1e-4,
    )
    minimum_authored_seam_length = max(
        MODELING_TOLERANCE * 80.0,
        engagement_scale * 4.0,
    )

    def radius_is_allowed(value):
        return any(
            abs(float(value) - float(expected)) <= radius_tolerance
            for expected in allowed_radii
        )

    result = []
    for edge in shape.Edges:
        try:
            faces = shape.ancestorsOfType(edge, Part.Face)
            if len(faces) != 2:
                continue
            radii = [cylinder_face_radius(face) for face in faces]
            if any(value is None or not radius_is_allowed(value) for value in radii):
                continue
            if float(edge.Length) <= minimum_authored_seam_length:
                continue
            parameter = (
                float(edge.FirstParameter) + float(edge.LastParameter)
            ) / 2.0
            sample_points = [vertex.Point for vertex in edge.Vertexes]
            sample_points.append(edge.valueAt(parameter))
            if any(
                (point - terminal).Length <= terminal_tolerance
                for point in sample_points
                for terminal in terminal_points
            ):
                continue
            point = edge.CenterOfMass
            result.append((
                (
                    float((point - center).Length),
                    float(edge.Length),
                    float(point.x),
                    float(point.y),
                    float(point.z),
                ),
                edge,
                [float(value) for value in radii],
            ))
        except Exception:
            continue
    result.sort(key=lambda item: item[0])
    return result


def fillet_compact_junction_material(
    material,
    outer_radii,
    bore_radii,
    blend_radius,
    inner_blend_radius,
    center,
    terminal_points,
):
    """Apply exact radii one raw material seam at a time, with topology re-query."""
    current = material
    for label, allowed_radii, exact_radius in (
        ("outer", outer_radii, float(blend_radius)),
        ("inner", bore_radii, float(inner_blend_radius)),
    ):
        initial = compact_junction_seams(
            current,
            allowed_radii,
            center,
            terminal_points,
        )
        if not initial:
            raise ValueError(
                "no compact junction " + label + " Cylinder-Cylinder seams found"
            )
        remaining = len(initial)
        for _index in range(remaining):
            seams = compact_junction_seams(
                current,
                allowed_radii,
                center,
                terminal_points,
            )
            if not seams:
                break
            before = len(seams)
            _sort_key, selected, adjacent_radii = seams[0]
            point = selected.CenterOfMass
            try:
                candidate = current.makeFillet(exact_radius, [selected]).removeSplitter()
            except Exception as exc:
                raise ValueError(
                    "exact compact junction fillet failed: "
                    + json.dumps({
                        "surface": label,
                        "radius": exact_radius,
                        "edge_center": vector_json(point),
                        "edge_length": float(selected.Length),
                        "adjacent_cylinder_radii": adjacent_radii,
                        "error": str(exc)[:240],
                    }, separators=(",", ":"))
                )
            if not valid_closed_single_solid(candidate):
                raise ValueError(
                    "exact compact junction fillet produced an invalid solid: "
                    + json.dumps({
                        "surface": label,
                        "radius": exact_radius,
                        "edge_center": vector_json(point),
                        "edge_length": float(selected.Length),
                        "adjacent_cylinder_radii": adjacent_radii,
                    }, separators=(",", ":"))
                )
            current = candidate
            after = len(compact_junction_seams(
                current,
                allowed_radii,
                center,
                terminal_points,
            ))
            if after >= before:
                raise ValueError(
                    "compact junction sequential fillet made no topology progress: "
                    + json.dumps({
                        "surface": label,
                        "radius": exact_radius,
                        "before": before,
                        "after": after,
                    }, separators=(",", ":"))
                )
        unresolved = compact_junction_seams(
            current,
            allowed_radii,
            center,
            terminal_points,
        )
        if unresolved:
            raise ValueError(
                "compact junction left unresolved " + label + " seams: "
                + str(len(unresolved))
            )
    return current


def make_junction(params, root_interface=False):
    start = vector(params["start_position"])
    outer_parts = []
    bore_parts = []
    outer_radius_in = float(params["outer_diameter"]) / 2.0
    bore_radius_in = inner_radius(params["outer_diameter"], params["wall_thickness"])
    engagement = max(MODELING_TOLERANCE * 20.0, outer_radius_in * 1e-4)
    inlet_axis = normalized(vector(params["axis"]))
    inlet_start = start
    inlet_end = start + inlet_axis * engagement
    if not root_interface:
        inlet_start = start - inlet_axis * engagement
        inlet_end = start
    _, inlet_outer, inlet_bore = make_cylinder_tube(
        vector_json(inlet_start),
        vector_json(inlet_end),
        outer_radius_in,
        bore_radius_in,
    )
    outer_parts.append(inlet_outer)
    bore_parts.append(inlet_bore)
    terminal_points = [inlet_start]
    section_contracts = [
        (
            float(params["outer_diameter"]),
            float(params["wall_thickness"]),
        )
    ]
    for outlet in params["outlets"]:
        _, outlet_outer, outlet_bore = make_cylinder_tube(
            params["start_position"],
            outlet["end_position"],
            float(outlet["outer_diameter"]) / 2.0,
            inner_radius(outlet["outer_diameter"], outlet["wall_thickness"]),
        )
        outer_parts.append(outlet_outer)
        bore_parts.append(outlet_bore)
        terminal_points.append(vector(outlet["end_position"]))
        section_contracts.append(
            (
                float(outlet["outer_diameter"]),
                float(outlet["wall_thickness"]),
            )
        )
    if params["blend_mode"] == "fillet":
        if root_interface:
            raise ValueError(
                "a filleted junction requires a positive upstream route before "
                "the hub; it cannot be placed directly on START"
            )
        blend_radius = float(params["blend_radius"])
        inner_blend_radius = float(params["inner_blend_radius"])
        max_hub_radius = float(params["max_hub_radius"])
        outer_radii = [
            outer_diameter / 2.0
            for outer_diameter, _wall in section_contracts
        ]
        bore_radii = [
            inner_radius(outer_diameter, wall)
            for outer_diameter, wall in section_contracts
        ]
        if blend_radius > max_hub_radius + MODELING_TOLERANCE:
            raise ValueError(
                "exact outer blend_radius exceeds max_hub_radius: required "
                + str(blend_radius)
                + ", maximum "
                + str(max_hub_radius)
            )
        if inner_blend_radius > max_hub_radius + MODELING_TOLERANCE:
            raise ValueError(
                "exact inner_blend_radius exceeds max_hub_radius: required "
                + str(inner_blend_radius)
                + ", maximum "
                + str(max_hub_radius)
            )
        # Keep the local envelope equal to the authored arm cylinders. A sphere
        # would create the blob explicitly forbidden by compact Y-manifold
        # requests. OCC is most stable when the complete hollow material is made
        # first and each raw Cylinder-Cylinder seam receives the exact authored
        # radius individually, with topology re-queried after every success.
        candidate_outer = fuse_shapes(outer_parts)
        candidate_bore = fuse_shapes(bore_parts)
        candidate_shape = candidate_outer.cut(candidate_bore).removeSplitter()
        if not valid_closed_single_solid(candidate_shape):
            raise ValueError("raw compact junction is not one valid closed solid")
        candidate_shape = fillet_compact_junction_material(
            candidate_shape,
            outer_radii,
            bore_radii,
            blend_radius,
            inner_blend_radius,
            start,
            terminal_points,
        )
        return candidate_shape, candidate_outer, candidate_bore

    outer = fuse_shapes(outer_parts)
    bore = fuse_shapes(bore_parts)
    shape = outer.cut(bore).removeSplitter()
    return shape, outer, bore


def make_termination(params):
    start = vector(params["start_position"])
    axis = normalized(vector(params["axis"]))
    radius = float(params["outer_diameter"]) / 2.0
    thickness = float(params["thickness"])
    if params["termination_type"] == "plug":
        radius = inner_radius(params["outer_diameter"], params["wall_thickness"])
        start = start - axis * thickness
    shape = Part.makeCylinder(radius, thickness, start, axis)
    return shape, shape, None


def make_inline_component(params):
    start = vector(params["start_position"])
    axis = normalized(vector(params["axis"]))
    length = float(params["length"])
    pipe_radius = float(params["outer_diameter"]) / 2.0
    bore_radius = inner_radius(params["outer_diameter"], params["wall_thickness"])
    body_radius = float(params["body_outer_diameter"]) / 2.0
    body_start = start + axis * float(params["body_start_offset"])
    outer = Part.makeCylinder(pipe_radius, length, start, axis)
    body = Part.makeCylinder(body_radius, float(params["body_length"]), body_start, axis)
    outer = outer.fuse(body).removeSplitter()
    component_type = params["component_type"]
    if component_type == "flange":
        radial_a = normalized(vector(params["flange_reference_axis"]))
        radial_b = normalized(axis.cross(radial_a))
        bolt_radius = float(params["flange_bolt_hole_diameter"]) / 2.0
        bolt_circle = float(params["flange_bolt_circle_diameter"]) / 2.0
        bolt_cutters = []
        for index in range(int(params["flange_bolt_count"])):
            angle = 2.0 * math.pi * index / int(params["flange_bolt_count"])
            radial = radial_a * math.cos(angle) + radial_b * math.sin(angle)
            center = body_start + radial * bolt_circle
            bolt_cutters.append(
                Part.makeCylinder(
                    bolt_radius,
                    float(params["body_length"]) + MODELING_TOLERANCE * 4.0,
                    center - axis * MODELING_TOLERANCE * 2.0,
                    axis,
                )
            )
        outer = outer.cut(fuse_shapes(bolt_cutters)).removeSplitter()
    elif component_type == "union":
        ring_length = float(params["union_ring_length"])
        ring_radius = float(params["union_ring_outer_diameter"]) / 2.0
        ring_a = Part.makeCylinder(ring_radius, ring_length, body_start, axis)
        ring_b = Part.makeCylinder(
            ring_radius,
            ring_length,
            body_start + axis * (float(params["body_length"]) - ring_length),
            axis,
        )
        outer = outer.fuse(ring_a).fuse(ring_b).removeSplitter()
    elif component_type == "valve":
        actuator_axis = normalized(vector(params["actuator_axis"]))
        actuator_origin = (
            body_start
            + axis * (float(params["body_length"]) / 2.0)
            - actuator_axis * (float(params["actuator_height"]) * 0.15)
        )
        actuator = Part.makeCylinder(
            float(params["actuator_diameter"]) / 2.0,
            float(params["actuator_height"]) * 1.15,
            actuator_origin,
            actuator_axis,
        )
        outer = outer.fuse(actuator).removeSplitter()
    extension = max(MODELING_TOLERANCE * 20.0, bore_radius * 1e-4)
    bore = Part.makeCylinder(
        bore_radius,
        length + extension * 2.0,
        start - axis * extension,
        axis,
    )
    return outer.cut(bore).removeSplitter(), outer, bore


def make_module_shape(module):
    params = module["params"]
    kind = module["type"]
    if kind in ("straight_pipe", "connector_pipe"):
        outer_radius = float(params.get("coupling_outer_diameter", params["outer_diameter"])) / 2.0
        return make_cylinder_tube(
            params["start_position"],
            params["end_position"],
            outer_radius,
            inner_radius(params["outer_diameter"], params["wall_thickness"]),
        )
    if kind == "bend_pipe":
        return make_analytic_circular_tube(
            params["path_points"],
            float(params["outer_diameter"]) / 2.0,
            inner_radius(params["outer_diameter"], params["wall_thickness"]),
        )
    if kind == "reducer_pipe":
        return make_loft_tube(params)
    if kind == "junction_pipe":
        legacy_outlets = []
        if params.get("include_primary_outlet", True) and params.get("trunk_end") is not None:
            legacy_outlets.append({
                "end_position": params["trunk_end"],
                "outer_diameter": params["outer_diameter"],
                "wall_thickness": params["wall_thickness"],
            })
        for key, value in params.items():
            if key.startswith("branch_") and key.endswith("_end"):
                legacy_outlets.append({
                    "end_position": value,
                    "outer_diameter": params["outer_diameter"],
                    "wall_thickness": params["wall_thickness"],
                })
        migrated = dict(params)
        migrated["outlets"] = legacy_outlets
        migrated["blend_mode"] = "hard"
        migrated["blend_radius"] = max(float(params.get("blend_radius", 1.0)), MODELING_TOLERANCE)
        migrated["inner_blend_radius"] = migrated["blend_radius"]
        migrated["max_hub_radius"] = max(migrated["blend_radius"], float(params["outer_diameter"]) / 2.0)
        return make_junction(
            migrated,
            "START" in (module.get("input_bindings") or {}).values(),
        )
    if kind == "cap_pipe":
        if params.get("end_type") == "open":
            raise ValueError("legacy open-end marker has no geometry")
        migrated = dict(params)
        migrated["termination_type"] = "cap"
        migrated["thickness"] = params["cap_thickness"]
        return make_termination(migrated)
    if kind == "route":
        path_kind = params["path_kind"]
        if path_kind == "circular_arc":
            return make_analytic_circular_tube(
                params["path_points"],
                float(params["outer_diameter"]) / 2.0,
                inner_radius(
                    params["outer_diameter"],
                    params["wall_thickness"],
                ),
            )
        if path_kind == "line":
            path_kind = "line"
        else:
            path_kind = "spline"
        return make_swept_tube(
            params["path_points"],
            float(params["outer_diameter"]) / 2.0,
            inner_radius(params["outer_diameter"], params["wall_thickness"]),
            path_kind,
            bool(params.get("frenet", False)),
            params.get("initial_tangent", params.get("axis")),
            params.get("final_tangent", params.get("terminal_axis")),
        )
    if kind == "transition":
        return make_loft_tube(params)
    if kind == "junction":
        return make_junction(
            params,
            "START" in (module.get("input_bindings") or {}).values(),
        )
    if kind == "connect_ports":
        if params["path_kind"] == "circular_arc":
            return make_analytic_circular_tube(
                params["path_points"],
                float(params["outer_diameter"]) / 2.0,
                inner_radius(
                    params["outer_diameter"],
                    params["wall_thickness"],
                ),
            )
        return make_swept_tube(
            params["path_points"],
            float(params["outer_diameter"]) / 2.0,
            inner_radius(params["outer_diameter"], params["wall_thickness"]),
            params["path_kind"],
            bool(params.get("frenet", False)),
            params.get("initial_tangent"),
            params.get("final_tangent"),
        )
    if kind == "terminate":
        return make_termination(params)
    if kind == "inline_component":
        return make_inline_component(params)
    raise ValueError("unsupported module type: " + str(kind))


def module_circular_sweep_evidence(module):
    params = module["params"]
    is_circular = (
        module["type"] == "bend_pipe"
        or params.get("path_kind") == "circular_arc"
    )
    if not is_circular:
        return None
    frame = analytic_circular_arc_frame(params["path_points"])
    assessment = circular_sweep_radius_evidence(
        frame["centerline_radius"],
        float(params["outer_diameter"]) / 2.0,
    )
    authored_normal = params.get("plane_normal")
    signed_sweep = params.get("sweep_angle", params.get("angle"))
    normal_dot = None
    if authored_normal is not None:
        normal_dot = float(
            frame["directed_normal"].dot(normalized(vector(authored_normal)))
        )
    return {
        **assessment,
        "derived_sweep_magnitude_degrees": float(frame["sweep_degrees"]),
        "authored_signed_sweep_degrees": (
            float(signed_sweep) if signed_sweep is not None else None
        ),
        "directed_normal": vector_json(frame["directed_normal"]),
        "authored_plane_normal_dot": normal_dot,
        "analytic_frame_origin": vector_json(frame["center"]),
        "analytic_frame_radial_axis": vector_json(frame["radial_axis"]),
        "analytic_frame_tangent_axis": vector_json(frame["tangent_axis"]),
    }


def centerline_check(module):
    params = module["params"]
    curve_length = None
    circular_sweep = None
    try:
        circular_sweep = module_circular_sweep_evidence(module)
    except Exception as exc:
        return {
            "passed": False,
            "required_radius": params.get("minimum_curvature_radius"),
            "circular_sweep_error": str(exc),
        }
    try:
        if params.get("path_points"):
            curve_length = float(
                make_path_wire(
                    params["path_points"],
                    params.get("path_kind", "line"),
                    params.get("initial_tangent"),
                    params.get("final_tangent"),
                ).Length
            )
        elif module["type"] == "junction":
            curve_length = sum(float(item["length"]) for item in params["outlets"])
        elif module["type"] == "transition":
            curve_length = float(
                (vector(params["end_position"]) - vector(params["start_position"])).Length
            )
        elif params.get("length") is not None:
            curve_length = float(params["length"])
        elif module["type"] == "terminate":
            curve_length = 0.0
    except Exception as exc:
        return {
            "passed": False,
            "required_radius": params.get("minimum_curvature_radius"),
            "curve_length_error": str(exc),
        }
    required_raw = params.get("minimum_curvature_radius")
    required = float(required_raw) if required_raw is not None else None
    path_kind = params.get("path_kind")
    if path_kind not in {"spline", "circular_arc"}:
        actual = (
            float(params.get("bend_radius", required))
            if required is not None
            else None
        )
        return {
            "passed": (
                (required is None or actual + MODELING_TOLERANCE >= required)
                and (
                    circular_sweep is None
                    or bool(circular_sweep.get("supported"))
                )
            ),
            "required_radius": required,
            "minimum_radius": actual,
            "curve_length": curve_length,
            "minimum_nonlocal_distance": None,
            "required_self_clearance": None,
            "circular_sweep": circular_sweep,
        }
    try:
        optimized_handle_factors = None
        if path_kind == "spline":
            wire, optimized_handle_factors = make_spline_wire(
                params["path_points"],
                params.get("initial_tangent"),
                params.get("final_tangent"),
                return_handle_factors=True,
            )
        else:
            wire = make_path_wire(
                params["path_points"],
                path_kind,
                params.get("initial_tangent"),
                params.get("final_tangent"),
            )
        edges = list(wire.Edges)
        if not edges:
            raise ValueError("curved path wire has no edges")
        curvatures = []
        sampled_points = []
        sampled_locations = []
        join_tangent_dots = []
        samples_per_edge = (
            257
            if path_kind == "circular_arc"
            else max(
                257,
                min(1025, 64 * (len(params["path_points"]) - 1) + 1),
            )
        )
        for edge_index, edge in enumerate(edges):
            first = float(edge.FirstParameter)
            last = float(edge.LastParameter)
            if edge_index:
                previous = edges[edge_index - 1]
                previous_tangent = normalized(
                    previous.tangentAt(previous.LastParameter)
                )
                current_tangent = normalized(edge.tangentAt(first))
                join_tangent_dots.append(
                    float(previous_tangent.dot(current_tangent))
                )
            for sample_index in range(samples_per_edge):
                if edge_index and sample_index == 0:
                    continue
                parameter = first + (last - first) * sample_index / (
                    samples_per_edge - 1
                )
                curvature = abs(float(edge.curvatureAt(parameter)))
                if not math.isfinite(curvature):
                    raise ValueError("non-finite curve curvature sample")
                curvatures.append(curvature)
                sampled_points.append(edge.valueAt(parameter))
                sampled_locations.append([edge_index, parameter])
        maximum = max(curvatures or [0.0])
        zero_curvature = maximum <= 1e-12
        minimum_radius = None if zero_curvature else 1.0 / maximum
        worst_curvature_index = (
            max(range(len(curvatures)), key=curvatures.__getitem__)
            if curvatures
            else None
        )
        minimum_radius_location = None
        minimum_radius_nearest_path_point_index = None
        curvature_repair_hint = None
        if worst_curvature_index is not None:
            worst_point = sampled_points[worst_curvature_index]
            authored_points = [vector(point) for point in params["path_points"]]
            minimum_radius_nearest_path_point_index = min(
                range(len(authored_points)),
                key=lambda index: float((authored_points[index] - worst_point).Length),
            )
            minimum_radius_location = {
                "position": vector_json(worst_point),
                "edge_parameter": sampled_locations[worst_curvature_index],
                "sample_index": worst_curvature_index,
            }
            if path_kind == "circular_arc":
                curvature_repair_hint = (
                    "The circular arc is tighter than its required curvature bound. "
                    "Increase the independent bend radius or, for connect_ports, "
                    "move the midpoint farther from the endpoint chord."
                )
            elif minimum_radius_nearest_path_point_index == 0:
                curvature_repair_hint = (
                    "The curvature peak is near the spline inlet. Replace or move "
                    "non-required inlet-side waypoints so the turn begins over a "
                    "longer chord. Do not add closely spaced points: this spline "
                    "shortens handles to the adjacent chord and would tighten the "
                    "curve. Preserve only genuinely required anchors."
                )
            elif minimum_radius_nearest_path_point_index == len(authored_points) - 1:
                curvature_repair_hint = (
                    "The curvature peak is near the spline outlet. Replace or move "
                    "non-required outlet-side waypoints so the final direction change "
                    "uses a longer chord. Do not append closely spaced lead-out "
                    "points: they shorten cubic handles and increase curvature. "
                    "If the terminal tangent was not user-authored, remove that "
                    "Intent constraint instead of freezing it."
                )
            else:
                curvature_repair_hint = (
                    "The curvature peak is near authored path point index "
                    + str(minimum_radius_nearest_path_point_index)
                    + ". Move or remove nearby implementation points and spread the "
                    "turn over longer chords. Preserve genuinely required anchors, "
                    "but do not add clustered points because shorter adjacent chords "
                    "make this cubic construction tighter."
                )
        curvature_passed = (
            required is None
            or zero_curvature
            or minimum_radius + MODELING_TOLERANCE >= required
        )

        segment_lengths = [
            float((right - left).Length)
            for left, right in zip(sampled_points, sampled_points[1:])
        ]
        cumulative = [0.0]
        for segment_length in segment_lengths:
            cumulative.append(cumulative[-1] + segment_length)
        outer_radius = float(params["outer_diameter"]) / 2.0
        required_self_clearance = 2.0 * outer_radius + MODELING_TOLERANCE
        local_arc_exclusion = math.pi * outer_radius
        minimum_nonlocal_distance = None
        closest_pair = None
        for left_index in range(len(sampled_points) - 1):
            for right_index in range(left_index + 2, len(sampled_points) - 1):
                intervening_arc = (
                    cumulative[right_index] - cumulative[left_index + 1]
                )
                if intervening_arc <= local_arc_exclusion:
                    continue
                separation = segment_distance(
                    sampled_points[left_index],
                    sampled_points[left_index + 1],
                    sampled_points[right_index],
                    sampled_points[right_index + 1],
                )
                if (
                    minimum_nonlocal_distance is None
                    or separation < minimum_nonlocal_distance
                ):
                    minimum_nonlocal_distance = separation
                    closest_pair = {
                        "segment_indices": [left_index, right_index],
                        "edge_parameters": [
                            sampled_locations[left_index],
                            sampled_locations[right_index],
                        ],
                        "intervening_centerline_length": intervening_arc,
                    }
        self_clearance_passed = (
            minimum_nonlocal_distance is None
            or minimum_nonlocal_distance + MODELING_TOLERANCE
            >= required_self_clearance
        )
        minimum_join_tangent_dot = min(join_tangent_dots or [1.0])
        tangent_continuity_passed = minimum_join_tangent_dot >= 1.0 - 1e-9
        endpoint_tangent_dots = None
        endpoint_tangency_passed = True
        if path_kind == "spline":
            actual_initial = normalized(
                edges[0].tangentAt(edges[0].FirstParameter)
            )
            actual_final = normalized(
                edges[-1].tangentAt(edges[-1].LastParameter)
            )
            expected_initial = normalized(vector(params["initial_tangent"]))
            expected_final = normalized(vector(params["final_tangent"]))
            endpoint_tangent_dots = {
                "initial": float(actual_initial.dot(expected_initial)),
                "final": float(actual_final.dot(expected_final)),
            }
            endpoint_tangency_passed = all(
                value >= 1.0 - 1e-9
                for value in endpoint_tangent_dots.values()
            )
        return {
            "passed": (
                curvature_passed
                and self_clearance_passed
                and tangent_continuity_passed
                and endpoint_tangency_passed
                and (
                    circular_sweep is None
                    or bool(circular_sweep.get("supported"))
                )
            ),
            "required_radius": required,
            "minimum_radius": minimum_radius,
            "minimum_radius_location": minimum_radius_location,
            "minimum_radius_nearest_path_point_index": (
                minimum_radius_nearest_path_point_index
            ),
            "curvature_repair_hint": curvature_repair_hint,
            "zero_curvature": zero_curvature,
            "sample_count": len(curvatures),
            "curvature_method": "dense_piecewise_curve_sampling",
            "curvature_proof": "sampled_not_global_extremum",
            "curve_length": curve_length,
            "optimized_handle_factors": optimized_handle_factors,
            "minimum_join_tangent_dot": minimum_join_tangent_dot,
            "tangent_continuity_passed": tangent_continuity_passed,
            "endpoint_tangent_dots": endpoint_tangent_dots,
            "endpoint_tangency_passed": endpoint_tangency_passed,
            "minimum_nonlocal_distance": minimum_nonlocal_distance,
            "required_self_clearance": required_self_clearance,
            "self_clearance_passed": self_clearance_passed,
            "self_clearance_closest_pair": closest_pair,
            "self_clearance_method": "dense_nonlocal_segment_sampling",
            "circular_sweep": circular_sweep,
        }
    except Exception as exc:
        return {
            "passed": False,
            "required_radius": required,
            "error": str(exc),
            "circular_sweep": circular_sweep,
        }


def shape_check(shape):
    bounds = shape.BoundBox
    result = {
        "is_null": bool(shape.isNull()),
        "is_valid": bool(shape.isValid()),
        "is_closed": bool(shape.isClosed()),
        "solid_count": len(shape.Solids),
        "volume": float(shape.Volume),
        "bounds": {
            "minimum": [float(bounds.XMin), float(bounds.YMin), float(bounds.ZMin)],
            "maximum": [float(bounds.XMax), float(bounds.YMax), float(bounds.ZMax)],
        },
        "bop_errors": [],
        "shape_check_mode": "topology_without_global_bop_argument_analysis",
    }
    try:
        # ``check(True)`` invokes OCC's global BOP argument analyzer. Repeating
        # it for every module plus the outer, bore, and final networks is both
        # redundant with the explicit overlap/clearance probes below and has
        # been observed to crash or stall FreeCAD on valid high-order B-spline
        # sweep surfaces. ``isValid`` plus topology check(False) remains strict
        # about the B-Rep itself; the specialized checks cover nonlocal tube
        # self-clearance and inter-module intersection.
        raw = shape.check(False)
        if raw:
            result["bop_errors"] = [str(item) for item in raw]
    except Exception as exc:
        result["bop_errors"] = ["check_exception: " + str(exc)]
    result["passed"] = (
        not result["is_null"]
        and result["is_valid"]
        and result["is_closed"]
        and result["solid_count"] == 1
        and result["volume"] > MODELING_TOLERANCE ** 3
        and not result["bop_errors"]
    )
    return result


def sample_hollow_section(
    shape, position, axis, outer_radius, bore_radius, *, check_outer=True
):
    axis = normalized(axis)
    radial_a = perpendicular(axis)
    radial_b = normalized(axis.cross(radial_a))
    wall_radius = (float(outer_radius) + float(bore_radius)) / 2.0
    outside_radius = float(outer_radius) + max(
        MODELING_TOLERANCE * 20.0,
        (float(outer_radius) - float(bore_radius)) * 0.25,
    )
    failures = []
    for sample_index in range(8):
        angle = 2.0 * math.pi * sample_index / 8.0
        radial = radial_a * math.cos(angle) + radial_b * math.sin(angle)
        wall_point = position + radial * wall_radius
        if not bool(shape.isInside(wall_point, MODELING_TOLERANCE, True)):
            failures.append({
                "sample": sample_index,
                "kind": "missing_wall_material",
            })
        if check_outer:
            outside_point = position + radial * outside_radius
            if bool(shape.isInside(outside_point, MODELING_TOLERANCE, True)):
                failures.append({
                    "sample": sample_index,
                    "kind": "material_outside_outer_radius",
                })
    if bool(shape.isInside(position, MODELING_TOLERANCE, True)):
        failures.append({"sample": "center", "kind": "blocked_bore"})
    return failures


def kernel_failure_code(error):
    """Return a stable, coarse OCC code without parsing provider prose upstream."""
    lowered = str(error).lower()
    for marker, code in (
        ("no suitable edges", "NO_SUITABLE_EDGES"),
        ("command not done", "COMMAND_NOT_DONE"),
        ("makepipeshell", "MAKE_PIPE_SHELL_FAILED"),
        ("makefillet", "MAKE_FILLET_FAILED"),
        ("brep_api", "BREP_API_FAILED"),
        ("brepalgoapi", "BREP_BOOLEAN_FAILED"),
        ("boolean", "BOOLEAN_OPERATION_FAILED"),
        ("occ", "OCC_EXCEPTION"),
        ("topods", "TOPODS_CONVERSION_FAILED"),
    ):
        if marker in lowered:
            return code
    return "UNKNOWN_KERNEL_ERROR"


def embedded_failure_details(error):
    marker = str(error).find(": ")
    if marker < 0:
        return {}
    encoded = str(error)[marker + 2:]
    if not encoded.startswith("{"):
        return {}
    try:
        value = json.loads(encoded)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def module_failure_record(module, module_id, error):
    """Keep the original error while exposing stable failure semantics."""
    error = str(error)
    lowered = error.lower()
    details = {"module_type": str(module.get("type", "unknown"))}
    failure_code = "MODULE_SHAPE_CONSTRUCTION_FAILED"
    stage = "make_module_shape"

    if "a filleted junction requires a positive upstream route" in lowered:
        failure_code = "JUNCTION_FILLET_REQUIRES_UPSTREAM_ROUTE"
        stage = "junction_precondition"
    elif "blend_radius exceeds max_hub_radius" in lowered:
        failure_code = "JUNCTION_BLEND_RADIUS_EXCEEDS_HUB"
        stage = "junction_precondition"
        details["surface"] = "inner" if "inner_blend_radius" in lowered else "outer"
        try:
            required_text = error.split("required ", 1)[1]
            required_text, maximum_text = required_text.split(", maximum ", 1)
            details["required_radius"] = float(required_text)
            details["maximum_radius"] = float(maximum_text)
        except Exception:
            pass
    elif "raw compact junction is not one valid closed solid" in lowered:
        failure_code = "JUNCTION_RAW_MATERIAL_INVALID"
        stage = "junction_material_construction"
    elif "no compact junction " in lowered and "seams found" in lowered:
        failure_code = "JUNCTION_FILLET_SEAM_NOT_FOUND"
        stage = "fillet_compact_junction_material"
        details["surface"] = "inner" if " junction inner " in lowered else "outer"
        details["kernel_code"] = "NO_SUITABLE_EDGES"
    elif "exact compact junction fillet failed:" in lowered:
        parsed = embedded_failure_details(error)
        details.update(parsed)
        surface = str(details.get("surface", "unknown"))
        failure_code = (
            "JUNCTION_INNER_FILLET_FAILED"
            if surface == "inner"
            else "JUNCTION_OUTER_FILLET_FAILED"
        )
        stage = "fillet_compact_junction_material"
        details["kernel_code"] = kernel_failure_code(details.get("error", error))
    elif "exact compact junction fillet produced an invalid solid:" in lowered:
        failure_code = "JUNCTION_FILLET_INVALID_SOLID"
        stage = "fillet_compact_junction_material"
        details.update(embedded_failure_details(error))
    elif "compact junction sequential fillet made no topology progress:" in lowered:
        failure_code = "JUNCTION_FILLET_NO_TOPOLOGY_PROGRESS"
        stage = "fillet_compact_junction_material"
        details.update(embedded_failure_details(error))
    elif "compact junction left unresolved " in lowered and " seams:" in lowered:
        failure_code = "JUNCTION_FILLET_UNRESOLVED_SEAMS"
        stage = "fillet_compact_junction_material"
        details["surface"] = "inner" if "unresolved inner" in lowered else "outer"
        try:
            details["unresolved_count"] = int(error.rsplit(": ", 1)[1])
        except Exception:
            pass
    elif any(marker in lowered for marker in (
        "no suitable edges",
        "command not done",
        "makepipeshell",
        "makefillet",
        "brep_api",
        "brepalgoapi",
        "boolean",
        "occ",
        "topods",
    )):
        failure_code = "OCC_KERNEL_OPERATION_FAILED"
        stage = "make_module_shape"
        details["kernel_code"] = kernel_failure_code(error)
        details["kernel_error"] = error[:240]

    return {
        "module_id": module_id,
        "error": error,
        "failure_code": failure_code,
        "stage": stage,
        "details": details,
    }


if CANDIDATE_DOCUMENT in App.listDocuments():
    App.closeDocument(CANDIDATE_DOCUMENT)
doc = App.newDocument(CANDIDATE_DOCUMENT)

module_shapes = {}
module_outer_shapes = {}
module_bore_shapes = {}
module_checks = {}
centerline_checks = {}
module_errors = []
for module in MODULES:
    module_id = module.get("id")
    if module_id is not None:
        try:
            # Centerline diagnostics do not depend on a successful solid sweep.
            # Evaluate them first so OCC construction failures still report the
            # actual curvature, tangency, and nonlocal-clearance evidence needed
            # for a useful repair instead of only MakePipeShell::MakeSolid.
            centerline_checks[module_id] = centerline_check(module)
        except Exception as exc:
            centerline_checks[module_id] = {
                "passed": False,
                "error": "centerline validation failed: " + str(exc),
            }
    try:
        shape, outer, bore = make_module_shape(module)
        obj = doc.addObject("Part::Feature", module["geometry_id"])
        obj.Label = module["id"] + " " + module["type"]
        if obj.ViewObject is not None:
            # Candidate documents are validation-only. Hide before assigning a
            # high-order shape so Coin/OCC does not start asynchronous meshing
            # for every overlapping module while Boolean work is still active.
            obj.ViewObject.Visibility = False
        obj.Shape = shape
        obj.addProperty("App::PropertyString", "CadGenModuleId")
        obj.CadGenModuleId = module["id"]
        obj.addProperty("App::PropertyString", "CadGenPayloadDigest")
        obj.CadGenPayloadDigest = PAYLOAD_DIGEST
        if module["type"] == "inline_component":
            obj.addProperty("App::PropertyString", "CadGenComponentType")
            obj.CadGenComponentType = str(module["params"]["component_type"])
        module_shapes[module["id"]] = shape
        module_outer_shapes[module["id"]] = outer
        if bore is not None:
            module_bore_shapes[module["id"]] = bore
        module_check = shape_check(shape)
        circular_sweep = module_circular_sweep_evidence(module)
        if circular_sweep is not None:
            module_check["circular_sweep"] = circular_sweep
            module_check["construction_method"] = circular_sweep[
                "construction_method"
            ]
        module_checks[module["id"]] = module_check
    except Exception as exc:
        error = str(exc)
        module_errors.append(module_failure_record(module, module_id, error))
        if module_id is not None:
            module_checks[module_id] = {
                "passed": False,
                "error": error,
            }

assembly = None
assembly_check = {
    "is_null": True,
    "is_valid": False,
    "is_closed": False,
    "solid_count": 0,
    "volume": 0.0,
    "bop_errors": ["assembly not built"],
    "passed": False,
}
overlaps = []
adjacent_interface_overlaps = []
assembly_errors = []
backend_fallbacks = []
outer_network_check = {"passed": False, "reason": "outer network not built"}
bore_network_check = {"passed": False, "reason": "bore network not built"}
if not module_errors and module_shapes:
    outer_network = None
    flow_bore_network = None
    composite_route_modules = (
        list(MODULES)
        if MODULES
        and all(
            module.get("type") in ("route", "connect_ports")
            and module.get("params", {}).get("path_kind")
            in ("line", "circular_arc", "spline")
            for module in MODULES
        )
        else []
    )
    if composite_route_modules:
        try:
            assembly, outer_network, flow_bore_network = make_composite_route_tube(
                composite_route_modules
            )
            assembly_check = shape_check(assembly)
            assembly_check["construction_method"] = "composite_centerline_sweep"
            assembly_check["source_module_ids"] = [
                module["id"] for module in composite_route_modules
            ]
            outer_network_check = shape_check(outer_network)
            outer_network_check["construction_method"] = (
                "composite_centerline_outer_sweep"
            )
            bore_network_check = shape_check(flow_bore_network)
            bore_network_check["passed"] = bool(
                bore_network_check.get("passed")
                and int(bore_network_check.get("solid_count", 0)) == 1
            )
            bore_network_check["construction_method"] = (
                "composite_centerline_bore_sweep"
            )
        except Exception as exc:
            # A backend failure is not a geometry conclusion.  Preserve the
            # same solved GeometryIR and deterministically fall back to the
            # legacy fuse portfolio below.
            backend_fallbacks.append(
                {"stage": "composite_centerline_sweep", "error": str(exc)}
            )
            assembly = None
            outer_network = None
            flow_bore_network = None
    try:
        if outer_network is None:
            outer_network = fuse_shapes(list(module_outer_shapes.values()))
            outer_network_check = shape_check(outer_network)
            outer_network_check["construction_method"] = "module_outer_fuse_fallback"
    except Exception as exc:
        assembly_errors.append({"stage": "outer_network_fuse", "error": str(exc)})
    if module_bore_shapes and flow_bore_network is None:
        try:
            bore_network = fuse_shapes(list(module_bore_shapes.values()))
            flow_bore_network = bore_network
            for module in MODULES:
                if module["type"] not in ("terminate", "cap_pipe"):
                    continue
                seal = module_shapes.get(module["id"])
                if seal is not None:
                    params = module["params"]
                    start = vector(params["start_position"])
                    axis = normalized(vector(params["axis"]))
                    bore_radius = inner_radius(
                        params["outer_diameter"], params["wall_thickness"]
                    )
                    termination_type = params.get("termination_type", "cap")
                    thickness = float(
                        params.get("thickness", params.get("cap_thickness", 0.0))
                    )
                    trim = max(MODELING_TOLERANCE * 50.0, bore_radius * 1e-3)
                    low = -trim if termination_type == "cap" else -thickness - trim
                    high = thickness + trim if termination_type == "cap" else trim
                    seal_with_trim = seal.fuse(
                        Part.makeCylinder(
                            bore_radius,
                            high - low,
                            start + axis * low,
                            axis,
                        )
                    )
                    flow_bore_network = flow_bore_network.cut(
                        seal_with_trim
                    ).removeSplitter()
            bore_network_check = shape_check(flow_bore_network)
            bore_network_check["passed"] = bool(
                bore_network_check.get("passed")
                and int(bore_network_check.get("solid_count", 0)) == 1
            )
        except Exception as exc:
            assembly_errors.append({"stage": "bore_network_fuse", "error": str(exc)})
    try:
        if outer_network is None:
            raise ValueError("outer network is unavailable")
        if assembly is None:
            assembly = (
                outer_network.cut(flow_bore_network).removeSplitter()
                if flow_bore_network is not None
                else outer_network
            )
            assembly_check = shape_check(assembly)
            assembly_check["construction_method"] = "module_boolean_fallback"
        if flow_bore_network is not None:
            # ``assembly`` is defined immediately above as
            # ``outer_network.cut(flow_bore_network)``. Re-running a global
            # common() between those coincident high-order B-spline surfaces is
            # logically redundant and can trap OCC's face/face intersector for
            # minutes. Shape validity, a one-solid bore network, terminal probes,
            # and dense internal annular-section samples provide independent
            # evidence without repeating that pathological Boolean.
            bore_network_check["assembly_intrusion_volume"] = 0.0
            bore_network_check["assembly_intrusion_allowance"] = max(
                MODELING_TOLERANCE ** 3,
                float(flow_bore_network.Volume) * 1e-8,
            )
            bore_network_check["intrusion_check_method"] = (
                "outer_minus_bore_construction_invariant"
            )
            bore_network_check["intrusion_module_ids"] = []
            bore_network_check["passed"] = bool(bore_network_check.get("passed"))
    except Exception as exc:
        assembly_errors.append({"stage": "assembly_global_bore_cut", "error": str(exc)})
    module_ids = list(module_shapes)
    adjacency = set()
    adjacent_junction_children = {}
    for module in MODULES:
        for bound in (module.get("input_bindings") or {}).values():
            if "." in bound:
                parent_id = bound.split(".", 1)[0]
                pair = tuple(sorted((module["id"], parent_id)))
                adjacency.add(pair)
                if module["type"] == "junction":
                    adjacent_junction_children[pair] = (module, parent_id)

    def adjacent_junction_interface_overlap(common_shape, child, parent_id):
        """Return evidence only when overlap is confined to the mating interface.

        A transverse junction arm necessarily intersects the immediately upstream
        tube near the shared port.  The admissible region is a resolver-derived
        cylindrical band around that port; it never depends on the LLM-authored
        max_hub_radius.  Backward-pointing outlets are deliberately ineligible,
        and any material outside the local band remains a collision.
        """
        params = child["params"]
        # Keep the classifier self-contained so its deterministic policy can be
        # unit-tested without a FreeCAD document or hidden process globals.
        policy = __ADJACENT_INTERFACE_POLICY__
        policy_digest = __VALIDATOR_POLICY_DIGEST__
        policy_generator_version = __GENERATOR_VERSION__
        start = vector(params["start_position"])
        inlet_axis = normalized(vector(params["axis"]))
        inlet_outer_radius = float(params["outer_diameter"]) / 2.0
        outlets = params.get("outlets") or []
        if not outlets:
            return None
        outlet_outer_radii = []
        minimum_forward_dot = 1.0
        for outlet in outlets:
            outlet_axis = normalized(vector(outlet["axis"]))
            forward_dot = float(outlet_axis.dot(inlet_axis))
            minimum_forward_dot = min(minimum_forward_dot, forward_dot)
            # A local interface allowance must never legitimize a branch whose
            # centerline travels back into the upstream module.
            if forward_dot < -1e-9:
                return None
            outlet_outer_radii.append(float(outlet["outer_diameter"]) / 2.0)
        engagement = max(
            MODELING_TOLERANCE * float(policy["engagement_tolerance_multiplier"]),
            inlet_outer_radius * float(policy["engagement_radius_multiplier"]),
        )
        margin = max(
            MODELING_TOLERANCE * float(policy["margin_tolerance_multiplier"]),
            inlet_outer_radius * float(policy["margin_radius_multiplier"]),
        )
        upstream_depth = max(outlet_outer_radii) + engagement + margin
        downstream_depth = engagement + margin
        interface_band = Part.makeCylinder(
            inlet_outer_radius + margin,
            upstream_depth + downstream_depth,
            start - inlet_axis * upstream_depth,
            inlet_axis,
        )
        try:
            outside_shape = common_shape.cut(interface_band)
            outside_volume = float(outside_shape.Volume)
        except Exception:
            # Failure to prove locality is not permission to ignore an overlap.
            return None
        common_volume = float(common_shape.Volume)
        outside_allowance = max(
            MODELING_TOLERANCE ** 3,
            common_volume * float(policy["outside_volume_relative_allowance"]),
        )
        if outside_volume > outside_allowance:
            return None
        evidence = {
            "module_ids": [parent_id, child["id"]],
            "parent_module_id": parent_id,
            "child_module_id": child["id"],
            "common_volume": common_volume,
            "outside_interface_volume": outside_volume,
            "outside_interface_allowance": outside_allowance,
            "interface_upstream_depth": upstream_depth,
            "interface_downstream_depth": downstream_depth,
            "minimum_outlet_forward_dot": minimum_forward_dot,
            "policy": "resolver_local_interface_band",
            "policy_id": policy["policy_id"],
            "policy_digest": policy_digest,
            "generator_version": policy_generator_version,
        }
        parent_bindings = [
            (str(input_name), str(bound_port))
            for input_name, bound_port in (child.get("input_bindings") or {}).items()
            if isinstance(bound_port, str)
            and bound_port.startswith(str(parent_id) + ".")
        ]
        if len(parent_bindings) == 1:
            input_name, bound_port = parent_bindings[0]
            evidence["parent_child_binding"] = {
                "child_input_name": input_name,
                "parent_port_id": bound_port,
            }
        # Bounds and centroid are diagnostic aids, not permission criteria.  A
        # kernel that cannot expose either still has to pass the volume/locality
        # proof above; malformed values are never emitted.
        try:
            bounds = common_shape.BoundBox
            minimum = [
                float(bounds.XMin),
                float(bounds.YMin),
                float(bounds.ZMin),
            ]
            maximum = [
                float(bounds.XMax),
                float(bounds.YMax),
                float(bounds.ZMax),
            ]
            if (
                all(math.isfinite(value) for value in minimum + maximum)
                and all(low <= high for low, high in zip(minimum, maximum))
            ):
                evidence["common_bounds"] = {
                    "minimum": minimum,
                    "maximum": maximum,
                }
        except Exception:
            pass
        try:
            centroid = vector_json(common_shape.CenterOfMass)
            if all(math.isfinite(value) for value in centroid):
                evidence["common_centroid"] = centroid
        except Exception:
            pass
        return evidence

    for index, left_id in enumerate(module_ids):
        for right_id in module_ids[index + 1:]:
            pair = tuple(sorted((left_id, right_id)))
            # OCC Boolean common()은 고차 spline solid에서 가장 비싼 검사다.
            # AABB가 modeling tolerance까지 고려해 분리되어 있으면 실제 solid도
            # 겹칠 수 없으므로 정확도를 잃지 않고 Boolean 후보에서 제외한다.
            left_box = module_shapes[left_id].BoundBox
            right_box = module_shapes[right_id].BoundBox
            boxes_disjoint = (
                left_box.XMax < right_box.XMin - MODELING_TOLERANCE
                or right_box.XMax < left_box.XMin - MODELING_TOLERANCE
                or left_box.YMax < right_box.YMin - MODELING_TOLERANCE
                or right_box.YMax < left_box.YMin - MODELING_TOLERANCE
                or left_box.ZMax < right_box.ZMin - MODELING_TOLERANCE
                or right_box.ZMax < left_box.ZMin - MODELING_TOLERANCE
            )
            if boxes_disjoint:
                continue
            allowed_volume = MODELING_TOLERANCE ** 3
            try:
                common_shape = module_shapes[left_id].common(module_shapes[right_id])
                volume = float(common_shape.Volume)
            except Exception as exc:
                overlaps.append({"module_ids": [left_id, right_id], "error": str(exc)})
                continue
            if volume > allowed_volume:
                junction_relation = adjacent_junction_children.get(pair)
                if junction_relation is not None:
                    child, parent_id = junction_relation
                    interface_evidence = adjacent_junction_interface_overlap(
                        common_shape,
                        child,
                        parent_id,
                    )
                    if interface_evidence is not None:
                        adjacent_interface_overlaps.append(interface_evidence)
                        continue
                overlaps.append({
                    "module_ids": [left_id, right_id],
                    "adjacent": pair in adjacency,
                    "common_volume": volume,
                    "allowed_volume": allowed_volume,
                })
    if assembly is not None:
        try:
            assembly_obj = doc.addObject("Part::Feature", "PipeAssembly")
            if assembly_obj.ViewObject is not None:
                assembly_obj.ViewObject.Visibility = False
            assembly_obj.Shape = assembly
            assembly_obj.addProperty("App::PropertyString", "CadGenPayloadDigest")
            assembly_obj.CadGenPayloadDigest = PAYLOAD_DIGEST
            assembly_obj.addProperty("App::PropertyString", "CadGenStateId")
            assembly_obj.CadGenStateId = PAYLOAD["state_id"]
            for module_id in module_ids:
                module_view = doc.getObject("solid_" + module_id).ViewObject
                if module_view is not None:
                    module_view.Visibility = False
        except Exception as exc:
            assembly_errors.append({"stage": "assembly_object", "error": str(exc)})

try:
    doc.recompute()
except Exception as exc:
    assembly_errors.append({"stage": "document_recompute", "error": str(exc)})

connection_failures = []
for edge in PAYLOAD.get("connection_edges", []):
    if (
        float(edge.get("position_error", 0.0)) > MODELING_TOLERANCE
        or float(edge.get("anti_parallel_axis_dot", 0.0)) < 0.9999
        or float(edge.get("od_error", 0.0)) > MODELING_TOLERANCE
        or float(edge.get("id_error", 0.0)) > MODELING_TOLERANCE
        or float(edge.get("wall_error", 0.0)) > MODELING_TOLERANCE
        or float(edge.get("outer_rim_error", float("inf"))) > MODELING_TOLERANCE
        or float(edge.get("inner_rim_error", float("inf"))) > MODELING_TOLERANCE
        or not bool(edge.get("connector_type_match", False))
        or not bool(edge.get("connector_gender_match", False))
        or not bool(edge.get("connector_standard_match", False))
    ):
        connection_failures.append(edge)

terminal_bore_failures = []
anchored_inlet_bore_failures = []
termination_seal_failures = []
if assembly is not None:
    for port in PAYLOAD.get("open_ports", []):
        try:
            position = vector(port["position"])
            axis = normalized(vector(port["axis"]))
            bore_radius = inner_radius(
                port["outer_diameter"], port["wall_thickness"]
            )
            probe_radius = max(MODELING_TOLERANCE * 10.0, bore_radius * 0.35)
            probe_length = max(MODELING_TOLERANCE * 50.0, bore_radius * 0.1)
            probe = Part.makeCylinder(
                probe_radius,
                probe_length,
                position - axis * probe_length,
                axis,
            )
            blocked_volume = float(assembly.common(probe).Volume)
            allowance = max(MODELING_TOLERANCE ** 3, float(probe.Volume) * 1e-5)
            if blocked_volume > allowance:
                terminal_bore_failures.append({
                    "port_id": port["id"],
                    "blocked_volume": blocked_volume,
                    "allowance": allowance,
                })
        except Exception as exc:
            terminal_bore_failures.append({"port_id": port.get("id"), "error": str(exc)})
    root_port = PAYLOAD.get("root_port")
    root_is_open_bore = bool(
        root_port
        and MODULES
        and MODULES[0]["type"] not in ("terminate", "cap_pipe")
    )
    if root_is_open_bore:
        try:
            position = vector(root_port["position"])
            axis = normalized(vector(root_port["axis"]))
            bore_radius = inner_radius(
                root_port["outer_diameter"], root_port["wall_thickness"]
            )
            probe_radius = max(MODELING_TOLERANCE * 10.0, bore_radius * 0.35)
            probe_length = max(MODELING_TOLERANCE * 50.0, bore_radius * 0.1)
            if MODULES[0]["type"] in ("junction", "junction_pipe"):
                first_params = MODULES[0]["params"]
                outer_radius = float(first_params["outer_diameter"]) / 2.0
                engagement = max(
                    MODELING_TOLERANCE * 20.0,
                    outer_radius * 1e-4,
                )
                probe_length = min(probe_length, engagement * 0.75)
            probe = Part.makeCylinder(
                probe_radius,
                probe_length,
                position,
                axis,
            )
            blocked_volume = float(assembly.common(probe).Volume)
            allowance = max(MODELING_TOLERANCE ** 3, float(probe.Volume) * 1e-5)
            if blocked_volume > allowance:
                anchored_inlet_bore_failures.append({
                    "port_id": root_port["id"],
                    "blocked_volume": blocked_volume,
                    "allowance": allowance,
                })
        except Exception as exc:
            anchored_inlet_bore_failures.append({
                "port_id": root_port.get("id"), "error": str(exc)
            })
    for module in MODULES:
        if module["type"] != "terminate":
            continue
        params = module["params"]
        try:
            position = vector(params["start_position"])
            axis = normalized(vector(params["axis"]))
            radius = inner_radius(params["outer_diameter"], params["wall_thickness"])
            probe_radius = max(MODELING_TOLERANCE * 10.0, radius * 0.35)
            probe_length = min(
                float(params["thickness"]) * 0.5,
                max(MODELING_TOLERANCE * 50.0, radius * 0.1),
            )
            origin = (
                position
                if params["termination_type"] == "cap"
                else position - axis * probe_length
            )
            probe = Part.makeCylinder(probe_radius, probe_length, origin, axis)
            filled_volume = float(assembly.common(probe).Volume)
            required_volume = float(probe.Volume) * 0.99
            if filled_volume + MODELING_TOLERANCE ** 3 < required_volume:
                termination_seal_failures.append({
                    "module_id": module["id"],
                    "filled_volume": filled_volume,
                    "required_volume": required_volume,
                })
        except Exception as exc:
            termination_seal_failures.append({
                "module_id": module.get("id"), "error": str(exc)
            })

wall_section_failures = []
sampled_internal_section_count = 0
sampled_internal_sections_by_module = {}
for module in MODULES:
    if module["type"] in ("terminate", "cap_pipe"):
        continue
    shape = module_shapes.get(module["id"])
    if shape is None:
        wall_section_failures.append({
            "module_id": module.get("id"), "error": "module shape is missing"
        })
        continue
    for local_name, port in (module.get("ports") or {}).items():
        if module["type"] == "junction" and local_name == "in":
            # The inlet plane is the hub center, not an annular free section;
            # intentional branch bores invalidate a simple ring occupancy test.
            continue
        try:
            position = vector(port["position"])
            axis = normalized(vector(port["axis"]))
            radial_a = perpendicular(axis)
            radial_b = normalized(axis.cross(radial_a))
            outer_radius = float(port["outer_diameter"]) / 2.0
            bore_radius = inner_radius(
                port["outer_diameter"], port["wall_thickness"]
            )
            wall_radius = (outer_radius + bore_radius) / 2.0
            outside_radius = outer_radius + max(
                MODELING_TOLERANCE * 20.0,
                float(port["wall_thickness"]) * 0.25,
            )
            check_outer = True
            if module["type"] == "inline_component":
                params = module["params"]
                body_start = float(params["body_start_offset"])
                body_end = body_start + float(params["body_length"])
                if local_name == "in" and body_start <= MODELING_TOLERANCE:
                    check_outer = False
                if local_name == "out" and abs(
                    body_end - float(params["length"])
                ) <= MODELING_TOLERANCE:
                    check_outer = False
            failed_samples = []
            for sample_index in range(8):
                angle = 2.0 * math.pi * sample_index / 8.0
                radial = radial_a * math.cos(angle) + radial_b * math.sin(angle)
                wall_point = position + radial * wall_radius
                outside_point = position + radial * outside_radius
                if not bool(shape.isInside(wall_point, MODELING_TOLERANCE, True)):
                    failed_samples.append({
                        "sample": sample_index,
                        "kind": "missing_wall_material",
                    })
                if check_outer and bool(
                    shape.isInside(outside_point, MODELING_TOLERANCE, True)
                ):
                    failed_samples.append({
                        "sample": sample_index,
                        "kind": "material_outside_outer_radius",
                    })
            if bool(shape.isInside(position, MODELING_TOLERANCE, True)):
                failed_samples.append({"sample": "center", "kind": "blocked_bore"})
            if failed_samples:
                wall_section_failures.append({
                    "module_id": module["id"],
                    "port": local_name,
                    "authored_wall_thickness": float(port["wall_thickness"]),
                    "failures": failed_samples,
                })
        except Exception as exc:
            wall_section_failures.append({
                "module_id": module.get("id"),
                "port": local_name,
                "error": str(exc),
            })

for module in MODULES:
    module_id = module.get("id")
    shape = module_shapes.get(module_id)
    if shape is None:
        continue
    params = module["params"]
    kind = module["type"]
    try:
        candidates = []
        if kind in ("route", "connect_ports", "bend_pipe"):
            points = [vector(point) for point in params.get("path_points", [])]
            if len(points) >= 2:
                if params.get("path_kind") == "spline" and len(points) > 2:
                    wire = make_spline_wire(
                        params["path_points"],
                        params.get("initial_tangent"),
                        params.get("final_tangent"),
                    )
                    edges = list(wire.Edges)
                    edge_lengths = [float(edge.Length) for edge in edges]
                    total_length = sum(edge_lengths)
                    for index in range(1, 8):
                        target_length = total_length * index / 8.0
                        traversed = 0.0
                        edge = edges[-1]
                        local_fraction = 1.0
                        for candidate, edge_length in zip(edges, edge_lengths):
                            if target_length <= traversed + edge_length:
                                edge = candidate
                                local_fraction = (
                                    (target_length - traversed) / edge_length
                                    if edge_length > MODELING_TOLERANCE
                                    else 0.0
                                )
                                break
                            traversed += edge_length
                        parameter = float(edge.FirstParameter) + (
                            float(edge.LastParameter) - float(edge.FirstParameter)
                        ) * local_fraction
                        candidates.append((
                            "spline_sample_" + str(index),
                            edge.valueAt(parameter),
                            edge.tangentAt(parameter),
                            float(params["outer_diameter"]) / 2.0,
                            inner_radius(
                                params["outer_diameter"], params["wall_thickness"]
                            ),
                        ))
                else:
                    stride = max(1, int(math.ceil((len(points) - 1) / 16.0)))
                    for index in range(0, len(points) - 1, stride):
                        candidates.append((
                            "path_segment_" + str(index),
                            (points[index] + points[index + 1]) * 0.5,
                            points[index + 1] - points[index],
                            float(params["outer_diameter"]) / 2.0,
                            inner_radius(
                                params["outer_diameter"], params["wall_thickness"]
                            ),
                        ))
        elif kind == "transition":
            start = vector(params["start_position"])
            axis = normalized(vector(params["axis"]))
            offset = vector(params.get("offset", [0.0, 0.0, 0.0]))
            axial_length = float(params["length"])
            for index in range(1, 8):
                fraction = index / 8.0
                # 형상 생성과 동일한 quintic smootherstep을 사용해야 probe의
                # 예상 중심/단면이 실제 loft와 같은 위치를 가리킨다.
                blend = fraction ** 3 * (
                    10.0 - 15.0 * fraction + 6.0 * fraction * fraction
                )
                outer_diameter = (
                    float(params["diameter_in"]) * (1.0 - blend)
                    + float(params["diameter_out"]) * blend
                )
                wall = (
                    float(params["wall_thickness_in"]) * (1.0 - blend)
                    + float(params["wall_thickness_out"]) * blend
                )
                candidates.append((
                    "transition_" + str(index),
                    start + axis * (axial_length * fraction) + offset * blend,
                    axis,
                    outer_diameter / 2.0,
                    inner_radius(outer_diameter, wall),
                ))
        elif kind == "junction":
            start = vector(params["start_position"])
            for index, outlet in enumerate(params.get("outlets", [])):
                end = vector(outlet["end_position"])
                outlet_length = float((end - start).Length)
                hub_radius = float(params["max_hub_radius"])
                if outlet_length <= hub_radius + MODELING_TOLERANCE:
                    raise ValueError("junction outlet does not extend beyond hub")
                for distance in (
                    hub_radius + (outlet_length - hub_radius) * 0.35,
                    hub_radius + (outlet_length - hub_radius) * 0.75,
                ):
                    fraction = distance / outlet_length
                    candidates.append((
                        "outlet_" + str(index) + "_" + str(fraction),
                        start + (end - start) * fraction,
                        end - start,
                        float(outlet["outer_diameter"]) / 2.0,
                        inner_radius(
                            outlet["outer_diameter"], outlet["wall_thickness"]
                        ),
                    ))
        elif kind == "inline_component":
            start = vector(params["start_position"])
            axis = normalized(vector(params["axis"]))
            body_center = (
                start
                + axis * (
                    float(params["body_start_offset"])
                    + float(params["body_length"]) / 2.0
                )
            )
            candidates.append((
                "component_body_mid",
                body_center,
                axis,
                float(params["body_outer_diameter"]) / 2.0,
                inner_radius(
                    params["outer_diameter"], params["wall_thickness"]
                ),
            ))
        for label, position, axis, outer_radius, bore_radius in candidates:
            sampled_internal_section_count += 1
            sampled_internal_sections_by_module[module_id] = (
                sampled_internal_sections_by_module.get(module_id, 0) + 1
            )
            failures = sample_hollow_section(
                shape, position, axis, outer_radius, bore_radius
            )
            if failures:
                wall_section_failures.append({
                    "module_id": module_id,
                    "section": label,
                    "failures": failures,
                })
    except Exception as exc:
        wall_section_failures.append({
            "module_id": module_id,
            "section": "internal_sampling",
            "error": str(exc),
        })

authored_walls = []
for module in MODULES:
    params = module["params"]
    for key in ("wall_thickness", "wall_thickness_in", "wall_thickness_out"):
        if params.get(key) is not None:
            authored_walls.append(float(params[key]))
    for outlet in params.get("outlets", []):
        if outlet.get("wall_thickness") is not None:
            authored_walls.append(float(outlet["wall_thickness"]))
minimum_wall = min(authored_walls or [0.0])
required_internal_section_module_ids = [
    module["id"]
    for module in MODULES
    if module["type"] not in ("terminate", "cap_pipe")
]
unsampled_internal_section_module_ids = [
    module_id
    for module_id in required_internal_section_module_ids
    if sampled_internal_sections_by_module.get(module_id, 0) <= 0
]
if unsampled_internal_section_module_ids:
    wall_section_failures.append({
        "module_ids": unsampled_internal_section_module_ids,
        "error": "one or more geometry-bearing modules have no internal wall sample",
    })
checks = {
    "assembly": assembly_check,
    "outer_network": outer_network_check,
    "bore_network": bore_network_check,
    "modules": module_checks,
    "centerlines": centerline_checks,
    "module_errors": module_errors,
    "assembly_errors": assembly_errors,
    "backend_fallbacks": backend_fallbacks,
    "non_adjacent_overlaps": overlaps,
    "adjacent_interface_overlaps": adjacent_interface_overlaps,
    "connection_failures": connection_failures,
    "terminal_bore_failures": terminal_bore_failures,
    "anchored_inlet_bore_failures": anchored_inlet_bore_failures,
    "termination_seal_failures": termination_seal_failures,
    "wall_section_failures": wall_section_failures,
    "sampled_internal_section_count": sampled_internal_section_count,
    "sampled_internal_sections_by_module": sampled_internal_sections_by_module,
    "required_internal_section_module_count": len(
        required_internal_section_module_ids
    ),
    "minimum_authored_wall_thickness": minimum_wall,
    "declared_downstream_open_port_count": len(PAYLOAD.get("open_ports", [])),
    "anchored_inlet_count": 1 if (
        PAYLOAD.get("root_port")
        and MODULES
        and MODULES[0]["type"] not in ("terminate", "cap_pipe")
    ) else 0,
}
passed = (
    bool(assembly_check.get("passed"))
    and bool(outer_network_check.get("passed"))
    and bool(bore_network_check.get("passed"))
    and not module_errors
    and not assembly_errors
    and all(item.get("passed") for item in module_checks.values())
    and all(item.get("passed") for item in centerline_checks.values())
    and not overlaps
    and not connection_failures
    and not terminal_bore_failures
    and not anchored_inlet_bore_failures
    and not termination_seal_failures
    and not wall_section_failures
    and math.isfinite(minimum_wall)
    and minimum_wall > 0.0
    and not unsampled_internal_section_module_ids
)
validation = {
    "schema_version": VALIDATION_SCHEMA_VERSION,
    "generator_version": GENERATOR_VERSION,
    "validator_policy": {
        "policy_id": VALIDATOR_POLICY_ID,
        "policy_digest": VALIDATOR_POLICY_DIGEST,
        "generator_version": GENERATOR_VERSION,
        "validation_schema_version": VALIDATION_SCHEMA_VERSION,
    },
    "run_id": RUN_ID,
    "state_id": PAYLOAD["state_id"],
    "state_version": PAYLOAD["state_version"],
    "attempt_id": ATTEMPT_ID,
    "candidate_document": CANDIDATE_DOCUMENT,
    "candidate_shape_fingerprints": {
        obj.Name: shape_fingerprint(obj.Shape)
        for obj in doc.Objects
        if hasattr(obj, "Shape") and not obj.Shape.isNull()
    },
    "payload_digest": PAYLOAD_DIGEST,
    "module_ids": [module["id"] for module in MODULES],
    "freecad_version": App.Version(),
    "checks": checks,
    "passed": passed,
}
print("CADGEN_VALIDATION=" + json.dumps(validation, sort_keys=True, separators=(",", ":")))
'''
    replacements = {
        "__PAYLOAD_JSON__": payload_json_literal,
        "__CANDIDATE_NAME__": repr(candidate_name),
        "__PAYLOAD_DIGEST__": repr(digest),
        "__VALIDATION_SCHEMA_VERSION__": str(VALIDATION_SCHEMA_VERSION),
        "__MODELING_TOLERANCE__": repr(float(modeling_tolerance)),
        "__GENERATOR_VERSION__": repr(GENERATOR_VERSION),
        "__VALIDATOR_POLICY_JSON__": validator_policy_json_literal,
        "__VALIDATOR_POLICY_ID__": repr(VALIDATOR_POLICY_ID),
        "__VALIDATOR_POLICY_DIGEST__": repr(VALIDATOR_POLICY_DIGEST),
        "__ADJACENT_INTERFACE_POLICY__": repr(
            _VALIDATOR_POLICY_SPEC["adjacent_interface_overlap"]
        ),
        "__RUN_ID__": repr(run_id),
        "__ATTEMPT_ID__": str(int(attempt_id)),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    return template


def build_freecad_publish_script(
    state: PipeState,
    *,
    run_id: str,
    attempt_id: int,
    fcstd_path: str | None = None,
    candidate_shape_fingerprints: dict[str, str] | None = None,
) -> str:
    """검증된 fingerprint를 재확인하고 버전 문서를 저장하는 게시 코드를 만든다."""

    return _build_freecad_publish_script(
        state,
        run_id=run_id,
        attempt_id=attempt_id,
        fcstd_path=fcstd_path,
        candidate_shape_fingerprints=candidate_shape_fingerprints,
        payload_digest=geometry_payload_digest(state),
    )


def _build_freecad_publish_script(
    state: PipeState,
    *,
    run_id: str,
    attempt_id: int,
    fcstd_path: str | None,
    candidate_shape_fingerprints: dict[str, str] | None,
    payload_digest: str,
) -> str:
    """트랜잭션에서 이미 검증한 digest를 재사용해 게시 코드를 만든다."""

    digest = payload_digest
    candidate = _candidate_document_name(
        state,
        run_id=run_id,
        attempt_id=attempt_id,
        digest=digest,
    )
    published = published_document_name(state, run_id=run_id)
    payload = {
        "candidate_document": candidate,
        "candidate_prefix": candidate.rsplit(
            f"_{state.state_version}_{attempt_id}_{digest[:12]}", 1
        )[0]
        + "_",
        "published_document": published,
        "published_prefix": published.rsplit("_v", 1)[0] + "_v",
        "payload_digest": digest,
        "state_id": state.state_id,
        "state_version": state.state_version,
        "fcstd_path": fcstd_path,
        "candidate_shape_fingerprints": candidate_shape_fingerprints or {},
        "view_deviation_percent": PUBLISHED_VIEW_DEVIATION_PERCENT,
        "view_angular_deflection_degrees": (PUBLISHED_VIEW_ANGULAR_DEFLECTION_DEGREES),
        "view_specular_color": PUBLISHED_VIEW_SPECULAR_COLOR,
        "view_shininess": PUBLISHED_VIEW_SHININESS,
    }
    template = r"""# Generated by cadgen02. Idempotent publish phase.
import json
import os
import hashlib
import FreeCAD as App

META = json.loads(__META_JSON__)


def shape_fingerprint(shape):
    raw = shape.exportBrepToString()
    if not isinstance(raw, str) or not raw:
        raise ValueError("candidate B-Rep serialization is unavailable")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


candidate = App.getDocument(META["candidate_document"])
candidate_assembly = candidate.getObject("PipeAssembly")
if candidate_assembly is None:
    raise RuntimeError("candidate assembly is missing")
if getattr(candidate_assembly, "CadGenPayloadDigest", "") != META["payload_digest"]:
    raise RuntimeError("candidate digest mismatch")
expected_fingerprints = META.get("candidate_shape_fingerprints") or {}
if not expected_fingerprints:
    raise RuntimeError("validated candidate fingerprints are missing")
actual_fingerprints = {
    obj.Name: shape_fingerprint(obj.Shape)
    for obj in candidate.Objects
    if hasattr(obj, "Shape") and not obj.Shape.isNull()
}
if actual_fingerprints != expected_fingerprints:
    raise RuntimeError("candidate B-Rep changed after validation")
if META["published_document"] in App.listDocuments():
    # Never trust digest metadata on a mutable GUI document.  Rebuild every
    # publish from the just-validated candidate so manual shape edits cannot be
    # mistaken for an idempotent same-digest result.
    App.closeDocument(META["published_document"])
published = App.newDocument(META["published_document"])
for source in candidate.Objects:
    if not hasattr(source, "Shape") or source.Shape.isNull():
        continue
    target = published.addObject("Part::Feature", source.Name)
    target.Label = source.Label
    if target.ViewObject is not None:
        target.ViewObject.Visibility = False
    target.Shape = source.Shape.copy()
    target.addProperty("App::PropertyString", "CadGenPayloadDigest")
    target.CadGenPayloadDigest = META["payload_digest"]
    if hasattr(source, "CadGenModuleId"):
        target.addProperty("App::PropertyString", "CadGenModuleId")
        target.CadGenModuleId = source.CadGenModuleId
    if hasattr(source, "CadGenComponentType"):
        target.addProperty("App::PropertyString", "CadGenComponentType")
        target.CadGenComponentType = source.CadGenComponentType
    if source.Name == "PipeAssembly":
        target.addProperty("App::PropertyString", "CadGenStateId")
        target.CadGenStateId = META["state_id"]
published.recompute()
published = App.getDocument(META["published_document"])
published.recompute()
for obj in published.Objects:
    if hasattr(obj, "Shape") and not obj.Shape.isNull():
        obj.ViewObject.Visibility = obj.Name == "PipeAssembly"
        if obj.Name == "PipeAssembly":
            view = obj.ViewObject
            if "Shaded" in view.listDisplayModes():
                view.DisplayMode = "Shaded"
            # FreeCAD's defaults (0.2% / 28.65 degrees) visibly facet long,
            # highly curved pipes.  These bounded object-local settings keep
            # 640 px review renders smooth without the runaway meshing cost of
            # FreeCAD's absolute minimum 0.01% / 1 degree values.
            if "Deviation" in view.PropertiesList:
                view.Deviation = META["view_deviation_percent"]
            if "AngularDeflection" in view.PropertiesList:
                view.AngularDeflection = META["view_angular_deflection_degrees"]
            # A neutral semimatte material keeps tessellation highlights from
            # masquerading as surface defects while preserving silhouette and
            # curvature cues. ShapeAppearance is the real FreeCAD 1.x material
            # list; ShapeMaterial is retained only as a compatibility fallback.
            if "ShapeAppearance" in view.PropertiesList:
                materials = list(view.ShapeAppearance)
                if not materials:
                    materials = [App.Material()]
                for material in materials:
                    material.SpecularColor = tuple(META["view_specular_color"])
                    material.Shininess = META["view_shininess"]
                view.ShapeAppearance = tuple(materials)
            elif hasattr(view, "ShapeMaterial"):
                material = view.ShapeMaterial
                material.SpecularColor = tuple(META["view_specular_color"])
                material.Shininess = META["view_shininess"]
                view.ShapeMaterial = material
if META.get("fcstd_path"):
    published.saveAs(META["fcstd_path"])
check = published.getObject("PipeAssembly")
saved = bool(
    META.get("fcstd_path")
    and os.path.isfile(META["fcstd_path"])
    and os.path.getsize(META["fcstd_path"]) > 0
)
passed = (
    check is not None
    and getattr(check, "CadGenPayloadDigest", "") == META["payload_digest"]
    and saved
)
print("CADGEN_PUBLISH=" + json.dumps({"passed": passed, "saved": saved, **META}, sort_keys=True, separators=(",", ":")))
if META["published_document"] in App.listDocuments():
    for document_name in list(App.listDocuments()):
        if document_name.startswith(META["published_prefix"]) and document_name != META["published_document"]:
            App.closeDocument(document_name)
    for document_name in list(App.listDocuments()):
        if document_name.startswith(META["candidate_prefix"]):
            App.closeDocument(document_name)
"""
    return template.replace(
        "__META_JSON__",
        repr(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
    )


def build_freecad_candidate_cleanup_script(
    state: PipeState,
    *,
    run_id: str,
    attempt_id: int,
) -> str:
    """특정 실행 시도의 후보 문서만 닫는 정리 스크립트를 만든다."""

    return _build_freecad_candidate_cleanup_script(
        state,
        run_id=run_id,
        attempt_id=attempt_id,
        payload_digest=geometry_payload_digest(state),
    )


def _build_freecad_candidate_cleanup_script(
    state: PipeState,
    *,
    run_id: str,
    attempt_id: int,
    payload_digest: str,
) -> str:
    """트랜잭션의 digest로 해당 후보 문서 이름만 정확히 정리한다."""

    candidate = _candidate_document_name(
        state,
        run_id=run_id,
        attempt_id=attempt_id,
        digest=payload_digest,
    )
    return f"""import json\nimport FreeCAD as App\nname = {candidate!r}\nif name in App.listDocuments():\n    App.closeDocument(name)\nprint("CADGEN_CLEANUP=" + json.dumps({{"candidate_document": name, "closed": name not in App.listDocuments()}}, sort_keys=True, separators=(",", ":")))\n"""
