"""Intent geometry contracts: sequential heading/position and safety preflight.

Extracted for readability. Behavior matches the host intent preflight contract.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

from cadgen.stable_content_hash import stable_digest
from cadgen.runtime_settings import Settings
from cadgen.geometry_analysis import predict_c1_spline
from cadgen.geometry_safety_policy import minimum_spline_curvature_radius
from cadgen.constraint_preflight import structural_intent_issues
from cadgen.typed_data_models import Goal, IntentResult
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

class _IntentSafetyValidationError(ValueError):
    """A parsed intent failed one or more deterministic semantic checks."""

    def __init__(self, diagnostics: list[str]):
        self.diagnostics = list(diagnostics)
        super().__init__("; ".join(self.diagnostics))

_MM_NUMBER_TEXT = (
    r"[+\-−]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+\-]?\d+)?"
)

_EXPLICIT_MM_VALUE = re.compile(
    r"(?<![\d.,])"
    rf"({_MM_NUMBER_TEXT})"
    r"\s*(?:mm|㎜|밀리미터)(?![A-Za-z])",
    re.IGNORECASE,
)

_EXPLICIT_MM_RANGE = re.compile(
    rf"(?<![\d.,])(?P<start>{_MM_NUMBER_TEXT})\s*"
    r"(?:(?:mm|㎜|밀리미터)\s*)?"
    r"(?:[–—~〜]|-(?!\s*[+\-−])|\bto\b|부터)\s*"
    rf"(?P<end>{_MM_NUMBER_TEXT})\s*(?:mm|㎜|밀리미터)(?![A-Za-z])",
    re.IGNORECASE,
)

_EXPLICIT_DEGREE_RANGE = re.compile(
    rf"(?<![\d.,])(?P<start>{_MM_NUMBER_TEXT})\s*"
    r"(?:(?:degrees?|deg|°|도)\s*)?"
    r"(?:[–—~〜]|-(?!\s*[+\-−])|\bto\b|부터)\s*"
    rf"(?P<end>{_MM_NUMBER_TEXT})\s*(?:degrees?|deg|°|도)",
    re.IGNORECASE,
)

_ANY_NUMERIC_VALUE = re.compile(
    r"(?<![\d.,])"
    r"([+\-−]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+\-]?\d+)?)"
    r"(?![\d.,])"
)

_VECTOR_NUMBER = r"[+\-−]?(?:(?:\d+)(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?"

_EXPLICIT_VECTOR3 = re.compile(
    rf"[\[(]\s*(?P<x>{_VECTOR_NUMBER})\s*,\s*"
    rf"(?P<y>{_VECTOR_NUMBER})\s*,\s*"
    rf"(?P<z>{_VECTOR_NUMBER})\s*[\])]"
)

_PROPORTIONAL_VECTOR_PREFIX = re.compile(
    r"(?:proportional\s+to|parallel\s+(?:to|with)|"
    r"positive\s+(?:scalar\s+)?multiple\s+of)"
    r"\s*(?:(?:the\s+)?(?:(?:global|local)\s+)?"
    r"(?:vector|axis|direction)\s*)?$",
    re.IGNORECASE,
)

_PROPORTIONAL_VECTOR_SUFFIX = re.compile(
    r"^\s*(?:(?:에|와|과)\s*)?(?:비례|평행)",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class _ProportionalDirectionContract:
    value: tuple[float, float, float]
    role: str

@dataclass(frozen=True)
class _IntentDirectionCandidate:
    semantic_id: str
    role: str
    value: tuple[float, float, float]

@dataclass(frozen=True)
class _ExplicitMMRange:
    """사용자가 직접 작성한 포괄적 millimeter 구간이다."""

    authored_start: float
    authored_end: float
    span: tuple[int, int]

    @property
    def minimum(self) -> float:
        return min(self.authored_start, self.authored_end)

    @property
    def maximum(self) -> float:
        return max(self.authored_start, self.authored_end)

@dataclass(frozen=True)
class _ExplicitAngleRange:
    """main centerline을 기준으로 사용자가 직접 쓴 acute 각도 범위다."""

    minimum: float
    maximum: float
    span: tuple[int, int]


@dataclass(frozen=True)
class _PrimaryContinuationLengthContract:
    """A primary branch run whose final subsegment is included in its total."""

    total_length: float
    final_segment_length: float
    total_span: tuple[int, int]
    final_segment_span: tuple[int, int]
    span: tuple[int, int]


class _PrimaryContinuationLengthCandidate(NamedTuple):
    branch_goal_id: str
    total_length: float
    components: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class _SourceRoleMeasurement:
    role: str
    value: float
    span: tuple[int, int]
    relation: str = "exact"
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class _SourceNominalMeasurement:
    value: float
    minimum: float
    maximum: float
    span: tuple[int, int]


SOURCE_MEASUREMENT_CONTRACT_VERSION = "source-measurement-contract/1"


@dataclass(frozen=True)
class SourceMeasurementContract:
    """One immutable interpretation shared by intent authoring and validation."""

    version: str
    prompt_sha256: str
    explicit_mm_values: tuple[float, ...]
    exact_mm_values: tuple[float, ...]
    standalone_exact_mm_values: tuple[float, ...]
    ranges: tuple[_ExplicitMMRange, ...]
    nominal_measurements: tuple[_SourceNominalMeasurement, ...]
    role_measurements: tuple[_SourceRoleMeasurement, ...]
    primary_continuation_lengths: tuple[_PrimaryContinuationLengthContract, ...]
    contract_digest: str

    @property
    def role_values(self) -> dict[str, tuple[float, ...]]:
        result: dict[str, list[float]] = {}
        for measurement in self.role_measurements:
            result.setdefault(measurement.role, []).append(measurement.value)
        return {role: tuple(values) for role, values in result.items()}

    def assert_matches_prompt(self, prompt: str) -> None:
        actual = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if actual != self.prompt_sha256:
            raise ValueError(
                "SourceMeasurementContract does not match prompt: "
                f"expected_sha256={self.prompt_sha256}, actual_sha256={actual}"
            )

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "contract_digest": self.contract_digest,
            "role_bindings": {
                role: list(values) for role, values in self.role_values.items()
            },
            "exact_mm_values": list(self.exact_mm_values),
            "standalone_exact_mm_values": list(self.standalone_exact_mm_values),
            "inclusive_ranges": [
                [source_range.minimum, source_range.maximum]
                for source_range in self.ranges
            ],
            "nominal_mm_values": [
                {
                    "value": item.value,
                    "minimum": item.minimum,
                    "maximum": item.maximum,
                }
                for item in self.nominal_measurements
            ],
            "role_claims": [
                {
                    "role": item.role,
                    "value": item.value,
                    "relation": item.relation,
                    "minimum": item.minimum,
                    "maximum": item.maximum,
                }
                for item in self.role_measurements
            ],
            "primary_continuation_lengths": [
                {
                    "total_length": item.total_length,
                    "final_segment_length": item.final_segment_length,
                    "preceding_component_sum": (
                        item.total_length - item.final_segment_length
                    ),
                }
                for item in self.primary_continuation_lengths
            ],
        }

    def content_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "prompt_sha256": self.prompt_sha256,
            "explicit_mm_values": list(self.explicit_mm_values),
            "exact_mm_values": list(self.exact_mm_values),
            "standalone_exact_mm_values": list(self.standalone_exact_mm_values),
            "ranges": [
                {
                    "authored_start": item.authored_start,
                    "authored_end": item.authored_end,
                    "minimum": item.minimum,
                    "maximum": item.maximum,
                    "source_span": list(item.span),
                }
                for item in self.ranges
            ],
            "nominal_measurements": [
                {
                    "value": item.value,
                    "minimum": item.minimum,
                    "maximum": item.maximum,
                    "source_span": list(item.span),
                }
                for item in self.nominal_measurements
            ],
            "role_measurements": [
                {
                    "role": item.role,
                    "value": item.value,
                    "relation": item.relation,
                    "minimum": item.minimum,
                    "maximum": item.maximum,
                    "source_span": list(item.span),
                }
                for item in self.role_measurements
            ],
            "primary_continuation_lengths": [
                {
                    "total_length": item.total_length,
                    "final_segment_length": item.final_segment_length,
                    "preceding_component_sum": (
                        item.total_length - item.final_segment_length
                    ),
                    "total_span": list(item.total_span),
                    "final_segment_span": list(item.final_segment_span),
                    "source_span": list(item.span),
                }
                for item in self.primary_continuation_lengths
            ],
        }

    def to_artifact_payload(self) -> dict[str, Any]:
        return {
            **self.content_payload(),
            "contract_digest": self.contract_digest,
        }

_FOUR_CORNER_TERMINAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:upper|top)[\s-]*left|좌상|왼쪽\s*(?:위|상단)",
        r"(?:lower|bottom)[\s-]*left|좌하|왼쪽\s*(?:아래|하단)",
        r"(?:upper|top)[\s-]*right|우상|오른쪽\s*(?:위|상단)",
        r"(?:lower|bottom)[\s-]*right|우하|오른쪽\s*(?:아래|하단)",
    )
)

_EXPLICIT_FOUR_PORT = re.compile(
    r"(?:\bfour[\s-]*port\b|\b4[\s-]*port\b|"
    r"\b(?:all\s+)?four\s+(?:branch\s+)?ends?\b|"
    r"4\s*(?:개\s*)?(?:포트|개구|끝단))",
    re.IGNORECASE,
)

_SMOOTH_JUNCTION_REQUEST = re.compile(
    r"(?:smooth(?:ly)?[^.;\n]{0,45}(?:Y[\s-]*junction|junction|branches)|"
    r"(?:Y[\s-]*junction|junction)[^.;\n]{0,45}smooth|"
    r"no\s+sharp\s+Boolean\s+intersections?|"
    r"부드러운?[^.;\n]{0,30}(?:Y\s*분기|접합|분기))",
    re.IGNORECASE,
)

_DIFFERENT_BRANCH_LENGTHS = re.compile(
    r"(?:different|unequal|asymmetric|slightly\s+different)"
    r"[^.;\n]{0,40}branch\s+lengths?",
    re.IGNORECASE,
)

_JUNCTION_WIDTH_REFERENCE = re.compile(
    rf"\bjunction\s+width\b[^.;\n]{{0,60}}?"
    rf"(?P<value>{_MM_NUMBER_TEXT})\s*(?:mm|㎜|밀리미터)",
    re.IGNORECASE,
)

_MEASUREMENT_NUMBER = (
    r"(?P<value>[+\-−]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+\-]?\d+)?)"
)

_MEASUREMENT_UNIT = r"(?:mm|㎜|밀리미터)"

_APPROXIMATE_MM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"(?:\babout\b|\baround\b|\broughly\b|\bapproximately\b|\bapprox\.?|약|대략)"
        rf"\s*(?P<value>{_MM_NUMBER_TEXT})\s*{_MEASUREMENT_UNIT}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<value>{_MM_NUMBER_TEXT})\s*{_MEASUREMENT_UNIT}\s*(?:정도|내외)",
        re.IGNORECASE,
    ),
)

SOURCE_NOMINAL_RELATIVE_TOLERANCE = 0.05

_PRIMARY_CONTINUATION_WITH_FINAL_SEGMENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"(?:Y\s*분기(?:의|에서)?\s*)?(?:주|메인|주요)\s*경로"
        rf"[^.;\n]{{0,100}}?(?:총\s*)?"
        rf"(?P<total>{_MM_NUMBER_TEXT})\s*{_MEASUREMENT_UNIT}"
        rf"[^.;\n]{{0,100}}?(?:마지막|최종)\s*"
        rf"(?P<final>{_MM_NUMBER_TEXT})\s*{_MEASUREMENT_UNIT}\s*(?:의\s*)?구간"
        rf"(?=[^.;\n]{{0,100}}?(?:외경|직경|두께|테이퍼)[^.;\n]{{0,60}}?"
        rf"(?:줄|늘|변경|감소|증가|테이퍼))",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:main|primary)\s+(?:route|path|run)"
        rf"[^.;\n]{{0,100}}?(?:total(?:s|ing)?|overall|continue(?:s|d|ing)?|for)?\s*"
        rf"(?P<total>{_MM_NUMBER_TEXT})\s*{_MEASUREMENT_UNIT}"
        rf"[^.;\n]{{0,100}}?(?:last|final)\s*"
        rf"(?P<final>{_MM_NUMBER_TEXT})\s*{_MEASUREMENT_UNIT}\s*(?:segment|section|run)"
        rf"(?=[^.;\n]{{0,100}}?(?:taper|reduc|increas|change|diameter|wall\s+thickness))",
        re.IGNORECASE,
    ),
)

_ANCHORED_MM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "diameter_in_reference",
        re.compile(
            rf"(?:외경|직경)(?:은|는|이|가|을|를)?\s*"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}\s*(?:에서|로부터)",
            re.IGNORECASE,
        ),
    ),
    (
        "wall_thickness_in_reference",
        re.compile(
            rf"두께(?:은|는|이|가|을|를)?\s*"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}\s*(?:에서|로부터)",
            re.IGNORECASE,
        ),
    ),
    (
        "diameter_out",
        re.compile(
            rf"(?:외경|직경)(?:은|는|이|가|을|를)?\s*"
            rf"{_MM_NUMBER_TEXT}\s*{_MEASUREMENT_UNIT}\s*(?:에서|로부터)\s*"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}\s*(?:로|까지)"
            rf"(?=[^.;\n]{{0,50}}?(?:줄|늘|변경|감소|증가))",
            re.IGNORECASE,
        ),
    ),
    (
        "wall_thickness_out",
        re.compile(
            rf"두께(?:은|는|이|가|을|를)?\s*"
            rf"{_MM_NUMBER_TEXT}\s*{_MEASUREMENT_UNIT}\s*(?:에서|로부터)\s*"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}\s*(?:로|까지)"
            rf"(?=[^.;\n]{{0,50}}?(?:줄|늘|변경|감소|증가))",
            re.IGNORECASE,
        ),
    ),
    (
        "diameter_in_reference",
        re.compile(
            rf"(?:outer\s+diameter|outside\s+diameter|\bOD\b)"
            rf"[^.;\n]{{0,50}}?\bfrom\s*(?:about\s+|roughly\s+|approximately\s+)?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "wall_thickness_in_reference",
        re.compile(
            rf"(?:wall\s+thickness|\bWT\b)"
            rf"[^.;\n]{{0,50}}?\bfrom\s*(?:about\s+|roughly\s+|approximately\s+)?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "route_rise",
        re.compile(
            rf"\b(?:rise|rises|rising)\b[^.;\n]{{0,35}}?\bby\s*"
            rf"(?:about\s+|roughly\s+|approximately\s+)?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "diameter_out",
        re.compile(
            rf"(?:외경|직경)(?:을|를)\s*{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}"
            rf"(?=(?:(?!(?:외경|직경)).){{0,60}}?(?:로\s*)?"
            rf"(?:줄|늘|변경|감소|증가))",
            re.IGNORECASE,
        ),
    ),
    (
        "wall_thickness_out",
        re.compile(
            rf"두께(?:를|을)\s*{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}"
            rf"(?=(?:(?!두께).){{0,50}}?(?:로\s*)?"
            rf"(?:줄|늘|변경|감소|증가))",
            re.IGNORECASE,
        ),
    ),
    (
        "diameter_out",
        re.compile(
            rf"(?:reduce|decrease|increase|change)[^.;\n]{{0,60}}?"
            rf"(?:outer\s+diameter|outside\s+diameter|\bOD\b)"
            rf"[^.;\n]{{0,40}}?\bto\s*{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "wall_thickness_out",
        re.compile(
            rf"(?:reduce|decrease|increase|change)[^.;\n]{{0,80}}?"
            rf"(?:wall\s+thickness|\bWT\b)"
            rf"[^.;\n]{{0,40}}?\bto\s*{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "transition_length",
        re.compile(
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}\s*(?:의\s*)?구간에서"
            rf"(?=[\s\S]{{0,100}}(?:줄|늘|변경|외경|직경|두께))",
            re.IGNORECASE,
        ),
    ),
    (
        "transition_length",
        re.compile(
            rf"(?:transition|taper|reducer)[^.;\n]{{0,50}}?"
            rf"(?:length|over)\s*(?:of\s*)?{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "connector_length",
        re.compile(
            rf"(?:길이|length(?:\s+of)?)\s*{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}"
            rf"\s*(?:의\s*)?(?:coupling|커플링|flange|플랜지|union|유니온|valve|밸브)",
            re.IGNORECASE,
        ),
    ),
    (
        "connector_length",
        re.compile(
            rf"(?:coupling|커플링|flange|플랜지|union|유니온|valve|밸브)"
            rf"[^.;\n]{{0,30}}?(?:길이|length)\s*{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "junction_blend_radius",
        re.compile(
            rf"(?:outer(?:[-\s]+surface)?[-\s]+blend\s+radius|"
            rf"외부\s*블렌드\s*(?:반경|반지름))\s*(?:of\s+|[:=]\s*)?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "junction_inner_blend_radius",
        re.compile(
            rf"(?:inner(?:[-\s]+bore)?[-\s]+blend\s+radius|"
            rf"내부(?:\s*보어)?\s*블렌드\s*(?:반경|반지름))\s*"
            rf"(?:of\s+|[:=]\s*)?{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "junction_max_hub_radius",
        re.compile(
            rf"(?:maximum|max)\s+hub\s+radius\s*(?:of\s+|[:=]\s*)?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "straight_length",
        re.compile(
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}\s*(?:만큼\s*)?"
            rf"(?:직진|straight(?:\s+(?:run|section))?)",
            re.IGNORECASE,
        ),
    ),
    (
        "global_inner_diameter",
        re.compile(
            rf"(?:내경|내부\s*직경|inner\s+diameter|inside\s+diameter|"
            rf"\bID\b|bore\s+diameter)"
            rf"\s*(?:(?:은|는|이|가|:|=)\s*|of\s+)?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "global_outer_diameter",
        re.compile(
            rf"(?:외경|외부\s*직경|outer\s+diameter|outside\s+diameter|\bOD\b)"
            rf"\s*(?:(?:은|는|이|가|:|=)\s*|of\s+)?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "global_outer_diameter",
        re.compile(
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}\s*"
            rf"(?:의\s*)?(?:pipe|tube)\s+(?:outer\s+|outside\s+)?"
            rf"(?:diameter|OD\b|직경|외경)",
            re.IGNORECASE,
        ),
    ),
    (
        "global_wall_thickness",
        re.compile(
            rf"(?:두께|wall\s+thickness|\bWT\b)"
            rf"\s*(?:(?:은|는|이|가|:|=)\s*|of\s+)?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "global_wall_thickness",
        re.compile(
            rf"(?:all(?:\s+[A-Za-z0-9-]+){{0,4}}\s+ends?|"
            rf"every\s+(?:open\s+)?end|"
            rf"각\s*(?:개방\s*)?(?:끝단|단부)|모든\s*(?:개방\s*)?(?:끝|끝단))"
            rf"[^.;\n]{{0,60}}?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}\s*"
            rf"(?:의\s*)?(?:두께|wall\s+thickness|\bWT\b)",
            re.IGNORECASE,
        ),
    ),
)

_SINGLETON_MM_ROLES = {
    "global_outer_diameter",
    "global_inner_diameter",
    "global_wall_thickness",
}

_GOAL_LENGTH_FIELDS = (
    "length",
    "bend_radius",
    "blend_radius",
    "inner_blend_radius",
    "max_hub_radius",
    "diameter_out",
    "wall_thickness_out",
    "transition_length",
    "branch_outer_diameter",
    "branch_wall_thickness",
    "termination_thickness",
    "minimum_curvature_radius",
)

_COMPONENT_LENGTH_FIELDS = (
    "body_outer_diameter",
    "body_start_offset",
    "body_length",
    "flange_bolt_circle_diameter",
    "flange_bolt_hole_diameter",
    "union_ring_outer_diameter",
    "union_ring_length",
    "actuator_diameter",
    "actuator_height",
)


def _validate_intent_safety(
    prompt: str,
    intent: IntentResult,
    settings: Settings,
    *,
    source_measurement_contract: SourceMeasurementContract | None = None,
) -> None:
    """Reject internally valid intents that cannot preserve/model the request.

    This is deliberately a validator, not a repairer.  The LLM must return a
    corrected immutable contract on the next intent attempt.
    """

    issues: list[str] = []

    # Waypoints unambiguously select a spline when path_kind was omitted. Keep
    # an explicitly incompatible path kind fatal, but do not reject otherwise
    # complete geometry for missing redundant metadata. The local copy ensures
    # that the normal spline-curvature preflight below still runs.
    canonical_goals = list(intent.target_behavior)
    for goal_index, goal in enumerate(canonical_goals):
        if goal.type == "route" and goal.required_waypoints and goal.path_kind is None:
            goal = goal.model_copy(update={"path_kind": "spline"})
            canonical_goals[goal_index] = goal
        elif (
            goal.type == "route"
            and goal.required_waypoints
            and goal.path_kind != "spline"
        ):
            goal_label = goal.goal_id or f"target_behavior[{goal_index}]"
            issues.append(
                f"{goal_label} required_waypoints require path_kind=spline; "
                f"actual path_kind={goal.path_kind!r}. Do not leave the path kind "
                "implicit because waypoint curvature must be preflighted before "
                "the immutable contract is accepted"
            )
    intent = intent.model_copy(update={"target_behavior": canonical_goals})
    undersized = [
        (path, value)
        for path, value in _positive_intent_dimensions(intent)
        if value <= settings.modeling_tolerance
    ]
    if undersized:
        rendered = ", ".join(path for path, _value in undersized[:12])
        issues.append(
            "these physical dimension fields must exceed modeling_tolerance "
            f"{settings.modeling_tolerance:.12g} mm: {rendered}. Do not reuse "
            "the rejected numeric spelling; restore the concise value from the "
            "user request"
        )

    source_measurement_contract = (
        source_measurement_contract or build_source_measurement_contract(prompt)
    )
    source_measurement_contract.assert_matches_prompt(prompt)
    anchored_values = {
        role: list(values)
        for role, values in source_measurement_contract.role_values.items()
    }
    candidate_values = _anchored_intent_values(intent)
    primary_length_contracts = list(
        source_measurement_contract.primary_continuation_lengths
    )
    # An inclusive total is represented by the sum of its component fields and
    # must not also be demanded as a standalone field. Other exact values stay
    # in the multiplicity guard, including role-bound values, so an unrelated
    # repeated measurement cannot borrow an already-owned candidate field.
    explicit_values = list(source_measurement_contract.standalone_exact_mm_values)
    preserved_values = list(_intent_metric_values(intent))
    # A taper's authored `from` section is inherited graph state, not a second
    # independent field on diameter_change. Add that derived evidence only when
    # the source text contains a typed input-section reference.
    for role in (
        "diameter_in_reference",
        "wall_thickness_in_reference",
        "global_inner_diameter",
    ):
        if role in anchored_values:
            preserved_values.extend(candidate_values.get(role, []))
    missing_values: list[float] = []
    for value in explicit_values:
        match_index = next(
            (
                index
                for index, candidate in enumerate(preserved_values)
                if _same_metric_value(value, candidate)
            ),
            None,
        )
        if match_index is None:
            missing_values.append(value)
        else:
            # Preserve source multiplicity: one typed value cannot account for
            # two independent measurements authored by the user.
            preserved_values.pop(match_index)
    if missing_values:
        issues.append(
            "intent lost or altered explicit millimeter values from the user "
            f"request: {missing_values}"
        )

    # A source range is one inclusive contract, not two exact dimensions.  The
    # LLM may select any typed physical measurement inside it; the resolver and
    # later geometry checks still operate on that explicit chosen value.
    range_candidates = _intent_range_candidate_values(intent)
    for source_range in source_measurement_contract.ranges:
        if not any(
            source_range.minimum <= candidate <= source_range.maximum
            for candidate in range_candidates
        ):
            issues.append(
                "intent has no typed physical dimension inside explicit "
                "millimeter range "
                f"[{source_range.minimum}, {source_range.maximum}]"
            )

    # ``each branch ... 85–100 mm long``은 범위 안의 숫자 하나만 어디엔가
    # 존재하면 되는 계약이 아니다. START가 대표하는 첫 terminal arm과 각
    # binary junction의 최종 branch outlet 모두가 독립 length를 가져야 한다.
    for source_range in _explicit_branch_length_ranges(prompt):
        arm_lengths = _terminal_arm_length_contracts(intent)
        missing_arm_indexes = [
            index for index, value in enumerate(arm_lengths) if value is None
        ]
        outside = [
            {"arm_index": index, "length": value}
            for index, value in enumerate(arm_lengths)
            if value is not None
            and not (source_range.minimum <= value <= source_range.maximum)
        ]
        if missing_arm_indexes or outside:
            issues.append(
                "each-branch length range requires one typed terminal-arm length "
                f"inside [{source_range.minimum}, {source_range.maximum}] mm for "
                "START plus every final branch outlet; "
                f"actual={arm_lengths}, missing_arm_indexes={missing_arm_indexes}, "
                f"outside={outside}. Use branch outlet_contract mode=outlets when "
                "individual outlet lengths must be authored"
            )
        elif _DIFFERENT_BRANCH_LENGTHS.search(prompt):
            distinct = {
                round(float(value), 9) for value in arm_lengths if value is not None
            }
            if len(distinct) < 2:
                issues.append(
                    "the user requested different branch lengths, but every typed "
                    f"terminal-arm length is {arm_lengths[0]:.6g} mm; select at least "
                    "two distinct values inside the authored range"
                )

    for angle_range in _explicit_branch_angle_ranges(prompt):
        try:
            actual_angles = _main_axis_terminal_arm_angles(intent)
        except ValueError as exc:
            issues.append(
                "could not derive terminal-arm axes for the explicit branch-angle "
                f"range: {exc}"
            )
            continue
        outside_angles = [
            {"arm_index": index, "angle_degrees": angle}
            for index, angle in enumerate(actual_angles)
            if not (angle_range.minimum - 1e-6 <= angle <= angle_range.maximum + 1e-6)
        ]
        if outside_angles:
            issues.append(
                "terminal-arm axes violate the explicit acute branch-angle range "
                f"[{angle_range.minimum}, {angle_range.maximum}] degrees from the "
                f"main axis: actual={actual_angles}, outside={outside_angles}. "
                "Encode the main-axis relation in terminal outlet vectors/start "
                "arm heading; do not copy a main-axis angle into an incompatible "
                "inlet-relative branch_angles field"
            )

    claims_by_role: dict[str, list[_SourceRoleMeasurement]] = {}
    for claim in source_measurement_contract.role_measurements:
        claims_by_role.setdefault(claim.role, []).append(claim)
    for role, claims in claims_by_role.items():
        missing_for_role = _ordered_missing_role_claims(
            claims,
            candidate_values.get(role, []),
        )
        if missing_for_role:
            issues.append(
                f"intent moved or altered source measurements bound to {role}: "
                f"{missing_for_role}"
            )

    issues.extend(
        _primary_continuation_length_issues(
            primary_length_contracts,
            intent,
        )
    )

    source_vector_contracts = _explicit_vector3_contracts(prompt)
    source_vectors = [
        value for value, is_direction in source_vector_contracts if not is_direction
    ]
    source_directions = _explicit_proportional_direction_contracts(prompt)
    candidate_vectors = _intent_vector_values(intent)
    missing_vectors: list[list[float]] = []
    for source_vector in source_vectors:
        match_index = next(
            (
                index
                for index, candidate in enumerate(candidate_vectors)
                if all(
                    _same_metric_value(source_vector[axis], candidate[axis])
                    for axis in range(3)
                )
            ),
            None,
        )
        if match_index is None:
            missing_vectors.append([float(value) for value in source_vector])
        else:
            candidate_vectors.pop(match_index)
    if missing_vectors:
        issues.append(
            "intent lost or altered explicit XYZ vectors from the user request: "
            f"{missing_vectors[:12]}"
        )

    # Direction ratios such as ``proportional to (+2,-1,+1)`` may be authored
    # at any positive magnitude.  Validate those against typed intent axes and
    # waypoint chords rather than incorrectly requiring the ratio tuple to be
    # copied as an absolute coordinate or terminal_axis field.
    candidate_directions = _intent_direction_candidates(intent)
    missing_directions: list[dict[str, Any]] = []
    for source_direction in source_directions:
        match_index = next(
            (
                index
                for index, candidate in enumerate(candidate_directions)
                if _direction_roles_compatible(
                    source_direction.role,
                    candidate.role,
                )
                and _positive_parallel_direction(
                    source_direction.value,
                    candidate.value,
                )
            ),
            None,
        )
        if match_index is None:
            missing_directions.append(
                {
                    "role": source_direction.role,
                    "vector": [float(value) for value in source_direction.value],
                }
            )
        else:
            candidate_directions.pop(match_index)
    if missing_directions:
        issues.append(
            "intent lost or altered explicit XYZ vectors from the user request "
            "that were authored as proportional directions: "
            f"{missing_directions[:12]}"
        )

    issues.extend(_sequential_heading_issues(intent, settings))
    issues.extend(_branch_successor_spline_issues(intent, settings))
    issues.extend(_sequential_position_issues(intent, settings))
    issues.extend(_branch_angle_vector_issues(intent))

    if _EXPLICIT_FOUR_PORT.search(prompt):
        if intent.expected_open_ports != 3:
            issues.append(
                "an explicit four-port manifold rooted at START requires exactly "
                "3 generated downstream open ports (START is the fourth physical port)"
            )
        # expected_open_ports_source is provenance metadata. The enforced
        # count above is the actual topology contract.

    if _SMOOTH_JUNCTION_REQUEST.search(prompt):
        wrong_styles = [
            goal.goal_id or f"target_behavior[{index}]"
            for index, goal in enumerate(intent.target_behavior)
            if goal.type == "branch"
            and goal.junction_style not in {None, "smooth_hub"}
        ]
        if wrong_styles:
            issues.append(
                "the user explicitly requested smooth Y-junctions/no sharp Boolean "
                "intersections; every branch goal must set "
                f"junction_style='smooth_hub'. Conflicting goals: {wrong_styles}"
            )

    width_references = [
        _metric_number(match.group("value"))
        for match in _JUNCTION_WIDTH_REFERENCE.finditer(prompt)
    ]
    copied_widths = [
        {
            "goal_id": goal.goal_id or f"target_behavior[{index}]",
            "max_hub_radius": float(goal.max_hub_radius),
        }
        for index, goal in enumerate(intent.target_behavior)
        if goal.type == "branch"
        and goal.max_hub_radius is not None
        and any(
            _same_metric_value(float(goal.max_hub_radius), width)
            for width in width_references
        )
    ]
    if copied_widths:
        issues.append(
            "junction width/pipe diameter is a full transverse size, not a hub "
            "radius; do not copy the source width directly into max_hub_radius. "
            f"width_references={width_references}, invalid_mappings={copied_widths}. "
            "Leave non-authored hub/blend radii unset for action planning"
        )

    # A branch placed directly on START must not emit an outlet back over the
    # consumed inlet ray. Apart from being semantically duplicative, that exact
    # topology is unstable in OCC's junction Boolean. Force the intent model to
    # author a positive-length run from a named terminal into the junction.
    first_goal = intent.target_behavior[0] if intent.target_behavior else None
    if first_goal is not None and first_goal.type == "branch":
        if all(pattern.search(prompt) for pattern in _FOUR_CORNER_TERMINAL_PATTERNS):
            issues.append(
                "a four-corner terminal manifold must start with a positive-length "
                "route from the anchored remote START arm to the first junction; "
                "the first goal cannot place that junction directly on START"
            )
        try:
            start_axis = normalize(vec(intent.start_axis))
        except ValueError:
            start_axis = None
        if start_axis is not None:
            reverse_vectors: list[tuple[float, float, float]] = []
            for raw_vector in first_goal.required_outlet_vectors:
                reverse_vectors.append(normalize(vec(raw_vector)))
            for outlet in first_goal.required_outlets:
                reverse_vectors.append(normalize(vec(outlet.axis)))
            for direction in first_goal.required_outlet_directions:
                reverse_vectors.append(direction_to_vector(direction))
            if any(
                dot(start_axis, candidate) <= -0.999 for candidate in reverse_vectors
            ):
                issues.append(
                    "the first branch recreates the anchored START arm with an "
                    "outlet opposite start_axis; add a positive-length route from "
                    "START to the junction and keep the START terminal out of all "
                    "downstream outlet contracts"
                )

    # Whole-program structural mistakes are intent-authoring errors, not
    # step-local numeric repair problems.  Only conflicts that continuous host
    # relaxation cannot fix (wrong source plane or coincident non-adjacent
    # centerlines) are returned to the LLM here.  Curvature, ordinary clearance
    # deficits and closure dimensions are solved after intent acceptance by the
    # deterministic ContractCore.
    issues.extend(
        structural_intent_issues(
            prompt,
            intent,
            modeling_tolerance=settings.modeling_tolerance,
        )
    )

    if issues:
        raise _IntentSafetyValidationError(issues)


def _predicted_c1_spline_minimum_radius(
    offsets: list[tuple[float, float, float]],
    initial_tangent: tuple[float, float, float],
    final_tangent: tuple[float, float, float] | None,
    *,
    modeling_tolerance: float,
) -> tuple[float, list[float]]:
    """Mirror the FreeCAD spline handle optimizer for intent preflight.

    Relative qualitative anchors are LLM-authored, but their feasibility is a
    dependent geometric calculation. Running the same scale-aware cubic model
    here prevents an impossible immutable route contract from reaching the
    paid step-repair loop.
    """

    try:
        prediction = predict_c1_spline(
            [(0.0, 0.0, 0.0), *[vec(point) for point in offsets]],
            normalize(initial_tangent),
            normalize(final_tangent) if final_tangent is not None else None,
            modeling_tolerance=modeling_tolerance,
        )
    except ValueError:
        return 0.0, []
    return prediction.minimum_radius, list(prediction.handle_factors)


def _sequential_heading_issues(
    intent: IntentResult,
    settings: Settings,
) -> list[str]:
    """Find linear-prefix heading contracts that no connected pipe can realize.

    Intent extraction is LLM-authored, but a straight run cannot change heading
    and a circular turn's endpoint separation is fixed by its sweep magnitude.
    Catching contradictions here gives the intent model a chance to repair the
    immutable agenda instead of making the step planner retry an impossible goal.
    Once topology forks or a freeform route has no terminal tangent, there is no
    single heading to simulate and this deliberately stops rather than guessing.
    """

    try:
        current = normalize(vec(intent.start_axis))
    except ValueError:
        return ["start_axis must be a finite non-zero sequential heading"]

    issues: list[str] = []
    current_outer_diameter = float(intent.global_spec.outer_diameter)
    for index, goal in enumerate(intent.target_behavior):
        if index > 0:
            previous_goal_id = intent.target_behavior[index - 1].goal_id
            if goal.allow_parallel or (
                goal.depends_on_goal_ids
                and previous_goal_id not in goal.depends_on_goal_ids
            ):
                # This is no longer a unique serial centerline, so one heading
                # cannot be propagated without guessing which branch is meant.
                break
        goal_label = goal.goal_id or f"target_behavior[{index}]"
        if goal.type == "move" and goal.direction is not None:
            desired = direction_to_vector(goal.direction)
            alignment = dot(current, desired)
            if alignment < 0.9999:
                issues.append(
                    f"sequential heading contradiction at {goal_label}: move.direction="
                    f"{goal.direction} is a straight-run heading but the incoming "
                    f"heading is {[round(value, 6) for value in current]} "
                    f"(dot={alignment:.6g}); align start_axis/the preceding outlet "
                    "or represent the heading change as a turn"
                )
            current = desired
            continue

        if goal.type == "turn" and goal.angle is not None:
            desired = (
                direction_to_vector(goal.direction)
                if goal.direction is not None
                else None
            )
            plane_normal = None
            if goal.plane_normal is not None:
                try:
                    plane_normal = normalize(vec(goal.plane_normal))
                except ValueError:
                    issues.append(
                        f"{goal_label}.plane_normal must be a finite non-zero vector"
                    )

            if desired is not None:
                cosine = max(-1.0, min(1.0, dot(current, desired)))
                outlet_separation = math.degrees(math.acos(cosine))
                sweep = abs(float(goal.angle)) % 360.0
                required_separation = min(sweep, 360.0 - sweep)
                direction_tolerance = math.degrees(math.acos(0.9999))
                tolerance = max(
                    direction_tolerance,
                    0.1,
                    required_separation * 1e-3,
                )
                if abs(outlet_separation - required_separation) > tolerance:
                    issues.append(
                        f"sequential heading contradiction at {goal_label}: incoming "
                        f"heading {[round(value, 6) for value in current]}, requested "
                        f"turn.direction={goal.direction}, and turn.angle={goal.angle:g} "
                        f"cannot all hold; those headings are {outlet_separation:.6g} "
                        f"degrees apart but the sweep requires {required_separation:.6g} "
                        "degrees. A turn direction is the outlet tangent, not a prose "
                        "label for the bend's location"
                    )
                if plane_normal is not None:
                    inlet_plane_error = abs(dot(current, plane_normal))
                    outlet_plane_error = abs(dot(desired, plane_normal))
                    if max(inlet_plane_error, outlet_plane_error) > 1e-4:
                        issues.append(
                            f"sequential heading contradiction at {goal_label}: "
                            "turn.plane_normal must be perpendicular to both the "
                            "incoming and outlet tangents; absolute dot products are "
                            f"{inlet_plane_error:.6g} and {outlet_plane_error:.6g}"
                        )
                current = desired
                continue

            if plane_normal is None:
                issues.append(
                    f"{goal_label} must provide either an exact cardinal outlet "
                    "direction or a signed bend plane"
                )
                break
            inlet_plane_error = abs(dot(current, plane_normal))
            if inlet_plane_error > 1e-4:
                issues.append(
                    f"sequential heading contradiction at {goal_label}: "
                    "turn.plane_normal must be perpendicular to the incoming tangent; "
                    f"incoming heading is {[round(value, 6) for value in current]} "
                    f"and absolute dot product is {inlet_plane_error:.6g}"
                )
                break
            current = normalize(
                rotate(current, plane_normal, math.radians(float(goal.angle)))
            )
            continue

        if goal.type == "route":
            if goal.waypoint_frame == "relative_to_target" and goal.required_waypoints:
                first_offset = vec(goal.required_waypoints[0])
                axial = dot(first_offset, current)
                total = length(first_offset)
                lateral = math.sqrt(max(0.0, total * total - axial * axial))
                required_radius = minimum_spline_curvature_radius(
                    current_outer_diameter,
                    settings.modeling_tolerance,
                    goal.minimum_curvature_radius,
                    enforcement=settings.validation_enforcement,  # type: ignore[arg-type]
                )
                if axial <= settings.modeling_tolerance:
                    issues.append(
                        f"{goal_label} relative first waypoint must advance along "
                        "the incoming tangent before the lateral freeform turn; "
                        f"axial projection is {axial:.6g} mm for incoming heading "
                        f"{[round(value, 6) for value in current]} and first offset "
                        f"{[round(value, 6) for value in first_offset]}. Make that "
                        "offset a positive parallel multiple of the incoming heading"
                    )
                if lateral > settings.modeling_tolerance:
                    implied_entry_radius = (total * total) / (2.0 * lateral)
                    if (
                        implied_entry_radius + settings.modeling_tolerance
                        < required_radius
                    ):
                        issues.append(
                            f"{goal_label} relative first waypoint is too close/off-axis "
                            "for a regular tangent entry: its circular-entry radius is "
                            f"{implied_entry_radius:.6g} mm but at least "
                            f"{required_radius:.6g} mm is required. Move the first "
                            "anchor farther along the incoming tangent before turning"
                        )
                try:
                    if goal.terminal_axis is not None:
                        final_tangent = normalize(vec(goal.terminal_axis))
                    elif len(goal.required_waypoints) == 1:
                        final_tangent = normalize(first_offset)
                    else:
                        final_tangent = normalize(
                            sub(
                                vec(goal.required_waypoints[-1]),
                                vec(goal.required_waypoints[-2]),
                            )
                        )
                    predicted_radius = None
                    predicted_curve_length: float | None = None
                    polyline_lower_bound: float | None = None
                    handle_factors: list[float] = []
                    if len(goal.required_waypoints) >= 2:
                        prediction = predict_c1_spline(
                            [
                                (0.0, 0.0, 0.0),
                                *[vec(point) for point in goal.required_waypoints],
                            ],
                            current,
                            final_tangent,
                            modeling_tolerance=settings.modeling_tolerance,
                        )
                        predicted_radius = prediction.minimum_radius
                        predicted_curve_length = prediction.curve_length
                        polyline_lower_bound = prediction.polyline_length
                        handle_factors = list(prediction.handle_factors)
                except ValueError:
                    issues.append(
                        f"{goal_label} relative waypoint chain contains a zero-length "
                        "tangent or 180-degree cusp; redistribute the anchors"
                    )
                    break
                if goal.length is not None and polyline_lower_bound is not None:
                    expected_length = float(goal.length)
                    tolerance = max(
                        settings.modeling_tolerance * 10.0,
                        expected_length * 1e-3,
                    )
                    if polyline_lower_bound > expected_length + tolerance:
                        issues.append(
                            f"{goal_label} route.length={expected_length:.6g} mm is "
                            "mathematically shorter than its ordered required-waypoint "
                            f"polyline lower bound {polyline_lower_bound:.6g} mm; no "
                            "spline can satisfy both. Remove model-invented waypoints "
                            "for an ordinary line arm, or choose a source-allowed "
                            "length and anchor chain whose calculated curve length agrees"
                        )
                    elif (
                        predicted_curve_length is not None
                        and abs(predicted_curve_length - expected_length) > tolerance
                    ):
                        issues.append(
                            f"{goal_label} deterministic spline centerline length is "
                            f"{predicted_curve_length:.6g} mm but route.length requires "
                            f"{expected_length:.6g}±{tolerance:.6g} mm. Revise the "
                            "model-authored anchors/terminal tangent or choose a "
                            "source-allowed route length before freezing the Intent"
                        )
                if (
                    predicted_radius is not None
                    and predicted_radius + settings.modeling_tolerance < required_radius
                ):
                    natural_radius: float | None = None
                    if (
                        goal.terminal_axis is not None
                        and len(goal.required_waypoints) >= 2
                    ):
                        try:
                            natural_radius, _natural_factors = (
                                _predicted_c1_spline_minimum_radius(
                                    [vec(point) for point in goal.required_waypoints],
                                    current,
                                    None,
                                    modeling_tolerance=settings.modeling_tolerance,
                                )
                            )
                        except ValueError:
                            natural_radius = None
                    natural_hint = ""
                    if (
                        natural_radius is not None
                        and natural_radius + settings.modeling_tolerance
                        >= required_radius
                    ):
                        natural_hint = (
                            " The same anchors with no terminal_axis contract use "
                            "their natural final chord and predict "
                            f"{natural_radius:.6g} mm; if terminal_axis was not "
                            "explicitly authored by the user, omit it instead of "
                            "adding short lead-out waypoints."
                        )
                    issues.append(
                        f"{goal_label} direct required-anchor realization predicts "
                        "a minimum curvature radius of "
                        f"{predicted_radius:.6g} mm but "
                        f"at least {required_radius:.6g} mm is required after "
                        f"deterministic handle optimization (factors "
                        f"{[round(value, 3) for value in handle_factors]}). "
                        "Redistribute or add well-separated relative anchors so each "
                        "direction change is spread over more distance" + natural_hint
                    )
                # With no explicit terminal-axis contract, the supported spline
                # uses the final waypoint chord as its natural downstream heading.
                # Propagating it lets the next qualitative segment receive the
                # same entry-feasibility check instead of stopping at this route.
                current = final_tangent
                continue
            if goal.path_kind == "line":
                line_headings: list[tuple[str, tuple[float, float, float]]] = []
                if goal.direction is not None:
                    line_headings.append(
                        ("direction", direction_to_vector(goal.direction))
                    )
                if goal.terminal_axis is not None:
                    try:
                        line_headings.append(
                            ("terminal_axis", normalize(vec(goal.terminal_axis)))
                        )
                    except ValueError:
                        issues.append(
                            f"{goal_label}.terminal_axis must be a finite non-zero heading"
                        )
                for field_name, desired in line_headings:
                    alignment = dot(current, desired)
                    if alignment < 0.9999:
                        issues.append(
                            f"sequential heading contradiction at {goal_label}: line "
                            f"route {field_name} cannot mate to incoming heading "
                            f"{[round(value, 6) for value in current]} without a turn"
                        )
                continue
            if goal.terminal_axis is not None:
                try:
                    current = normalize(vec(goal.terminal_axis))
                except ValueError:
                    return [
                        *issues,
                        f"{goal_label}.terminal_axis must be a finite non-zero heading",
                    ]
                continue
            break

        if goal.type in {"diameter_change", "connector"}:
            if goal.direction is not None:
                desired = direction_to_vector(goal.direction)
                alignment = dot(current, desired)
                if alignment < 0.9999:
                    issues.append(
                        f"sequential heading contradiction at {goal_label}: "
                        f"{goal.type}.direction={goal.direction} cannot change the "
                        f"incoming axial heading {[round(value, 6) for value in current]}"
                    )
            if goal.type == "diameter_change" and goal.diameter_out is not None:
                current_outer_diameter = float(goal.diameter_out)
            continue

        if goal.type in {"branch", "connect", "end"}:
            break

    return issues


def _branch_outlet_heading_candidates(
    incoming: tuple[float, float, float],
    goal: Goal,
) -> list[tuple[float, float, float]]:
    """Return every heading a branch contract may expose to its successors.

    This mirrors the authored outlet surface without deciding which outlet a
    later planner action must consume.  A successor contract is rejected only
    when *all* of these candidates fail, so the preflight cannot silently make
    a topology choice on the LLM's behalf.
    """

    candidates: list[tuple[float, float, float]] = []

    def append(raw: tuple[float, float, float]) -> None:
        try:
            candidate = normalize(vec(raw))
        except ValueError:
            return
        if not any(dot(candidate, existing) >= 1.0 - 1e-9 for existing in candidates):
            candidates.append(candidate)

    if goal.include_primary_outlet is not False:
        append(incoming)
    for raw in goal.required_outlet_vectors:
        append(vec(raw))
    for outlet in goal.required_outlets:
        append(vec(outlet.axis))
    for direction in goal.required_outlet_directions:
        append(direction_to_vector(direction))

    # Some legacy branch contracts express their outlet fan with only signed
    # angles.  Reproduce that generic construction so those candidates receive
    # the same feasibility check as explicit outlet vectors.
    if not candidates and goal.branch_angles:
        base = (
            direction_to_vector(goal.direction)
            if goal.direction is not None
            else incoming
        )
        try:
            base = normalize(base)
            if goal.branch_plane_normal is not None:
                normal = normalize(vec(goal.branch_plane_normal))
                for angle in goal.branch_angles:
                    append(rotate(base, normal, math.radians(float(angle))))
            else:
                side = choose_perpendicular_axis(base)
                for angle in goal.branch_angles:
                    radians = math.radians(float(angle))
                    append(
                        add(
                            mul(base, math.cos(radians)),
                            mul(side, math.sin(radians)),
                        )
                    )
        except ValueError:
            pass
    return candidates


def _serial_heading_before_goal(
    intent: IntentResult,
    stop_index: int,
) -> tuple[float, float, float] | None:
    """Propagate only the unambiguous serial heading before ``stop_index``."""

    try:
        current = normalize(vec(intent.start_axis))
    except ValueError:
        return None
    for index, goal in enumerate(intent.target_behavior[:stop_index]):
        if index > 0:
            previous_id = intent.target_behavior[index - 1].goal_id
            if goal.allow_parallel or (
                goal.depends_on_goal_ids and previous_id not in goal.depends_on_goal_ids
            ):
                return None
        try:
            if goal.type == "move" and goal.direction is not None:
                current = direction_to_vector(goal.direction)
            elif goal.type == "turn" and goal.angle is not None:
                if goal.direction is not None:
                    current = direction_to_vector(goal.direction)
                elif goal.plane_normal is not None:
                    current = normalize(
                        rotate(
                            current,
                            normalize(vec(goal.plane_normal)),
                            math.radians(float(goal.angle)),
                        )
                    )
                else:
                    return None
            elif goal.type == "route":
                if goal.terminal_axis is not None:
                    current = normalize(vec(goal.terminal_axis))
                elif len(goal.required_waypoints) >= 2:
                    current = normalize(
                        sub(
                            vec(goal.required_waypoints[-1]),
                            vec(goal.required_waypoints[-2]),
                        )
                    )
                elif goal.direction is not None:
                    current = direction_to_vector(goal.direction)
                elif goal.path_kind != "line":
                    return None
            elif goal.type in {"diameter_change", "connector"}:
                if goal.direction is not None:
                    current = direction_to_vector(goal.direction)
            elif goal.type in {"branch", "connect", "end"}:
                return None
        except ValueError:
            return None
    return normalize(current)


def _branch_successor_spline_issues(
    intent: IntentResult,
    settings: Settings,
) -> list[str]:
    """Preflight fixed spline anchors immediately downstream of a branch.

    The ordinary sequential simulator intentionally stops at a fork because it
    must not guess an outlet.  That previously let an infeasible immutable
    spline contract reach the step retry loop.  Here every authored outlet is
    evaluated independently and the contract is returned to the Intent LLM
    only when no possible successor heading can satisfy it.
    """

    goals_by_id = {
        goal.goal_id: (index, goal)
        for index, goal in enumerate(intent.target_behavior)
        if goal.goal_id is not None
    }
    issues: list[str] = []
    current_outer_diameter = float(intent.global_spec.outer_diameter)
    diameter_by_index: list[float] = []
    for goal in intent.target_behavior:
        diameter_by_index.append(current_outer_diameter)
        if goal.type == "diameter_change" and goal.diameter_out is not None:
            current_outer_diameter = float(goal.diameter_out)

    for route_index, route in enumerate(intent.target_behavior):
        if not (
            route.type == "route"
            and route.path_kind == "spline"
            and route.waypoint_frame == "relative_to_target"
            and route.required_waypoints
        ):
            continue
        branch_refs = [
            goals_by_id[goal_id]
            for goal_id in route.depends_on_goal_ids
            if goal_id in goals_by_id and goals_by_id[goal_id][1].type == "branch"
        ]
        if len(branch_refs) != 1:
            continue
        branch_index, branch = branch_refs[0]
        incoming = _serial_heading_before_goal(intent, branch_index)
        if incoming is None:
            continue
        headings = _branch_outlet_heading_candidates(incoming, branch)
        if not headings:
            continue

        required_radius = minimum_spline_curvature_radius(
            diameter_by_index[route_index],
            settings.modeling_tolerance,
            route.minimum_curvature_radius,
            enforcement=settings.validation_enforcement,  # type: ignore[arg-type]
        )
        points = [
            (0.0, 0.0, 0.0),
            *[vec(point) for point in route.required_waypoints],
        ]
        try:
            final_tangent = (
                normalize(vec(route.terminal_axis))
                if route.terminal_axis is not None
                else normalize(sub(points[-1], points[-2]))
            )
        except ValueError:
            continue

        evaluations: list[dict[str, Any]] = []
        any_feasible = False
        for heading in headings:
            first_offset = points[1]
            axial = dot(first_offset, heading)
            try:
                prediction = predict_c1_spline(
                    points,
                    heading,
                    final_tangent,
                    modeling_tolerance=settings.modeling_tolerance,
                )
                radius = float(prediction.minimum_radius)
            except ValueError:
                radius = 0.0
            feasible = (
                axial > settings.modeling_tolerance
                and radius + settings.modeling_tolerance >= required_radius
            )
            any_feasible = any_feasible or feasible
            evaluations.append(
                {
                    "incoming_heading": [round(value, 6) for value in heading],
                    "first_waypoint_axial_projection": round(axial, 6),
                    "predicted_minimum_radius": round(radius, 6),
                    "feasible": feasible,
                }
            )
        if any_feasible:
            continue
        route_label = route.goal_id or f"target_behavior[{route_index}]"
        branch_label = branch.goal_id or f"target_behavior[{branch_index}]"
        issues.append(
            f"{route_label} fixed required-anchor spline is infeasible after every "
            f"authored outlet of {branch_label}: required minimum curvature radius "
            f"is {required_radius:.6g} mm; outlet evaluations={evaluations}. The "
            "system will not choose an outlet or patch coordinates. Re-author the "
            "LLM-inferred relative waypoint contract so at least one permitted "
            "branch heading has positive entry advance and passes the calculated "
            "curvature bound; preserve any coordinates explicitly supplied by the user"
        )
    return issues


def _sequential_position_issues(
    intent: IntentResult,
    settings: Settings,
) -> list[str]:
    """Integrate the uniquely serial linear prefix and reject impossible poses.

    A line with a fixed inlet, heading and length has exactly one endpoint.  If
    Intent also freezes a different terminal_position, no amount of step-level
    replanning can satisfy it.  This preflight returns that contradiction to the
    Intent author before the contract becomes immutable.
    """

    try:
        position = vec(intent.start_position)
        heading = normalize(vec(intent.start_axis))
    except ValueError:
        return ["start_position/start_axis must define a finite serial pose"]

    issues: list[str] = []
    for index, goal in enumerate(intent.target_behavior):
        if index > 0:
            previous_goal_id = intent.target_behavior[index - 1].goal_id
            if goal.allow_parallel or (
                goal.depends_on_goal_ids
                and previous_goal_id not in goal.depends_on_goal_ids
            ):
                break
        label = goal.goal_id or f"target_behavior[{index}]"
        if goal.type == "move":
            if goal.direction is not None:
                heading = direction_to_vector(goal.direction)
            if goal.length is not None:
                position = add(position, mul(heading, float(goal.length)))
            continue
        if goal.type == "route" and goal.path_kind == "line":
            route_heading = (
                direction_to_vector(goal.direction)
                if goal.direction is not None
                else heading
            )
            if goal.terminal_axis is not None:
                try:
                    terminal_heading = normalize(vec(goal.terminal_axis))
                except ValueError:
                    terminal_heading = route_heading
            else:
                terminal_heading = route_heading
            predicted = (
                add(position, mul(route_heading, float(goal.length)))
                if goal.length is not None
                else None
            )
            if goal.terminal_position is not None:
                terminal = vec(goal.terminal_position)
                delta = sub(terminal, position)
                distance = length(delta)
                tolerance = max(
                    settings.modeling_tolerance * 10.0,
                    max(distance, float(goal.length or 0.0), 1.0) * 1e-6,
                )
                if predicted is not None:
                    endpoint_error = length(sub(predicted, terminal))
                    if endpoint_error > tolerance:
                        issues.append(
                            f"{label} line pose is over-constrained: start "
                            f"{[round(value, 6) for value in position]} + length "
                            f"{float(goal.length):.6g} * heading "
                            f"{[round(value, 6) for value in route_heading]} gives "
                            f"{[round(value, 6) for value in predicted]}, not "
                            f"terminal_position {[round(value, 6) for value in terminal]} "
                            f"(error {endpoint_error:.6g} mm). Revise the Intent's "
                            "length/heading/terminal position together"
                        )
                if distance > tolerance:
                    alignment = dot(normalize(delta), route_heading)
                    if alignment < 0.9999:
                        issues.append(
                            f"{label} terminal_position lies off its line heading: "
                            f"displacement {[round(value, 6) for value in delta]} has "
                            f"alignment {alignment:.6g} with heading "
                            f"{[round(value, 6) for value in route_heading]}"
                        )
                position = terminal
            elif predicted is not None:
                position = predicted
            heading = terminal_heading
            continue
        if (
            goal.type == "route"
            and goal.path_kind == "spline"
            and goal.required_waypoints
        ):
            endpoint = vec(goal.required_waypoints[-1])
            if goal.waypoint_frame == "relative_to_target":
                endpoint = add(position, endpoint)
            if goal.terminal_position is not None:
                terminal = vec(goal.terminal_position)
                tolerance = max(settings.modeling_tolerance * 10.0, 1e-6)
                endpoint_error = length(sub(endpoint, terminal))
                if endpoint_error > tolerance:
                    issues.append(
                        f"{label} final required waypoint and terminal_position "
                        f"differ by {endpoint_error:.6g} mm"
                    )
                position = terminal
            else:
                position = endpoint
            if goal.terminal_axis is not None:
                try:
                    heading = normalize(vec(goal.terminal_axis))
                except ValueError:
                    pass
            continue
        # Circular arcs, turns and branches need additional topology/frame
        # choices. Stop at that boundary instead of guessing a downstream pose.
        break
    return issues


def _branch_angle_vector_issues(intent: IntentResult) -> list[str]:
    """branch angle과 전역 outlet 축이 같은 작은 선각을 뜻하는지 검사한다.

    branch 전후의 축은 방향 화살표이지만 사용자가 말하는 Y 각도는 보통 두
    centerline 사이의 acute line angle이다. 직렬 primary 축이 Goal에 없으면
    다음 명시 move/line에서 다시 heading을 잡고, 모르는 값을 추측하지 않는다.
    """

    try:
        current: tuple[float, float, float] | None = normalize(vec(intent.start_axis))
    except ValueError:
        return []
    issues: list[str] = []
    for index, goal in enumerate(intent.target_behavior):
        label = goal.goal_id or f"target_behavior[{index}]"
        if goal.type == "move" and goal.direction is not None:
            current = direction_to_vector(goal.direction)
            continue
        if goal.type == "route":
            try:
                if goal.terminal_axis is not None:
                    current = normalize(vec(goal.terminal_axis))
                elif len(goal.required_waypoints) >= 2:
                    current = normalize(
                        sub(
                            vec(goal.required_waypoints[-1]),
                            vec(goal.required_waypoints[-2]),
                        )
                    )
                elif goal.direction is not None:
                    current = direction_to_vector(goal.direction)
                elif goal.path_kind == "line":
                    current = None
            except ValueError:
                current = None
            continue
        if goal.type == "turn" and current is not None and goal.angle is not None:
            try:
                if goal.direction is not None:
                    current = direction_to_vector(goal.direction)
                elif goal.plane_normal is not None:
                    current = normalize(
                        rotate(
                            current,
                            normalize(vec(goal.plane_normal)),
                            math.radians(float(goal.angle)),
                        )
                    )
            except ValueError:
                current = None
            continue
        if goal.type != "branch":
            continue

        vectors: list[tuple[float, float, float]] = []
        vectors.extend(vec(value) for value in goal.required_outlet_vectors)
        vectors.extend(vec(value.axis) for value in goal.required_outlets)
        vectors.extend(
            direction_to_vector(value) for value in goal.required_outlet_directions
        )
        if current is not None and goal.branch_angles and vectors:
            actual_angles = []
            for outlet_axis in vectors:
                cosine = max(-1.0, min(1.0, dot(current, normalize(outlet_axis))))
                directed = math.degrees(math.acos(cosine))
                actual_angles.append(min(directed, 180.0 - directed))
            expected_angles = [abs(float(value)) for value in goal.branch_angles]
            available = list(actual_angles)
            unmatched: list[float] = []
            for expected in expected_angles:
                if not available:
                    unmatched.append(expected)
                    continue
                best_index = min(
                    range(len(available)),
                    key=lambda item: abs(available[item] - expected),
                )
                if abs(available[best_index] - expected) > 0.5:
                    unmatched.append(expected)
                else:
                    available.pop(best_index)
            if unmatched or available:
                issues.append(
                    f"{label} branch_angles conflict with its outlet axes: "
                    f"expected acute angles {expected_angles}, actual "
                    f"{[round(value, 6) for value in actual_angles]}, tolerance "
                    "0.5 degree. Select one source-allowed angle per outlet and "
                    "make each vector mathematically consistent with it"
                )
        current = None
        if goal.include_primary_outlet is False:
            break
    return issues


def _positive_intent_dimensions(intent: IntentResult) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = [
        ("global_spec.outer_diameter", float(intent.global_spec.outer_diameter)),
        ("global_spec.wall_thickness", float(intent.global_spec.wall_thickness)),
    ]
    for goal_index, goal in enumerate(intent.target_behavior):
        prefix = f"target_behavior[{goal_index}]"
        for field_name in _GOAL_LENGTH_FIELDS:
            value = getattr(goal, field_name)
            if value is not None:
                values.append((f"{prefix}.{field_name}", float(value)))
        if goal.offset is not None:
            magnitude = math.sqrt(
                sum(float(component) ** 2 for component in goal.offset)
            )
            if magnitude > 0.0:
                values.append((f"{prefix}.offset_magnitude", magnitude))
        for outlet_index, outlet in enumerate(goal.required_outlets):
            for field_name in ("length", "outer_diameter", "wall_thickness"):
                value = getattr(outlet, field_name)
                if value is not None:
                    values.append(
                        (
                            f"{prefix}.required_outlets[{outlet_index}].{field_name}",
                            float(value),
                        )
                    )
        if goal.component_spec is not None:
            for field_name in _COMPONENT_LENGTH_FIELDS:
                value = getattr(goal.component_spec, field_name)
                if value is not None and float(value) > 0.0:
                    values.append(
                        (f"{prefix}.component_spec.{field_name}", float(value))
                    )
    for constraint_index, constraint in enumerate(intent.geometric_constraints):
        if constraint.type != "max_module_count" and constraint.value is not None:
            values.append(
                (
                    f"geometric_constraints[{constraint_index}].value",
                    float(constraint.value),
                )
            )
        if (
            constraint.type == "bounding_box"
            and constraint.minimum is not None
            and constraint.maximum is not None
        ):
            for axis_index, (minimum, maximum) in enumerate(
                zip(constraint.minimum, constraint.maximum)
            ):
                values.append(
                    (
                        f"geometric_constraints[{constraint_index}]"
                        f".bounding_box_extent[{axis_index}]",
                        float(maximum) - float(minimum),
                    )
                )
    return values


def _metric_number(raw: str) -> float:
    """쉼표와 유니코드 minus를 정규화해 유한한 실수로 변환한다."""

    return float(raw.replace(",", "").replace("−", "-"))


def _explicit_mm_ranges(text: str) -> list[_ExplicitMMRange]:
    """``85–100 mm``처럼 단위를 공유하는 명시적 범위를 추출한다."""

    result: list[_ExplicitMMRange] = []
    for match in _EXPLICIT_MM_RANGE.finditer(text):
        authored_start = _metric_number(match.group("start"))
        authored_end = _metric_number(match.group("end"))
        if not (math.isfinite(authored_start) and math.isfinite(authored_end)):
            continue
        result.append(
            _ExplicitMMRange(
                authored_start=authored_start,
                authored_end=authored_end,
                span=match.span(),
            )
        )
    return result


def _explicit_branch_length_ranges(text: str) -> list[_ExplicitMMRange]:
    """``each branch ... 85–100 mm long``에 귀속된 범위만 반환한다."""

    result: list[_ExplicitMMRange] = []
    for source_range in _explicit_mm_ranges(text):
        left = max(
            text.rfind(".", 0, source_range.span[0]),
            text.rfind("\n", 0, source_range.span[0]),
            text.rfind(";", 0, source_range.span[0]),
        )
        sentence_end_candidates = [
            position
            for marker in (".", "\n", ";")
            for position in [text.find(marker, source_range.span[1])]
            if position >= 0
        ]
        right = min(sentence_end_candidates) if sentence_end_candidates else len(text)
        context = text[left + 1 : right]
        if re.search(
            r"\b(?:each|every|all)\s+branch(?:es)?\b", context, re.IGNORECASE
        ) and re.search(
            r"\b(?:long|lengths?)\b",
            context,
            re.IGNORECASE,
        ):
            result.append(source_range)
    return result


def _explicit_branch_angle_ranges(text: str) -> list[_ExplicitAngleRange]:
    """branch angle과 main axis가 같은 clause에 있는 degree 범위를 찾는다."""

    result: list[_ExplicitAngleRange] = []
    for match in _EXPLICIT_DEGREE_RANGE.finditer(text):
        left = max(
            text.rfind(".", 0, match.start()),
            text.rfind("\n", 0, match.start()),
            text.rfind(";", 0, match.start()),
        )
        sentence_end_candidates = [
            position
            for marker in (".", "\n", ";")
            for position in [text.find(marker, match.end())]
            if position >= 0
        ]
        right = min(sentence_end_candidates) if sentence_end_candidates else len(text)
        context = text[left + 1 : right]
        if not re.search(r"\b(?:branch|arm)\s+angles?\b", context, re.IGNORECASE):
            continue
        if not re.search(r"\bmain\s+(?:axis|centerline)\b", context, re.IGNORECASE):
            continue
        start = _metric_number(match.group("start"))
        end = _metric_number(match.group("end"))
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        result.append(
            _ExplicitAngleRange(
                minimum=min(start, end),
                maximum=max(start, end),
                span=match.span(),
            )
        )
    return result


def _main_axis_terminal_arm_angles(intent: IntentResult) -> list[float]:
    """중앙 직선 축선과 모든 물리 terminal arm 축선의 acute 각도를 계산한다."""

    branch_indexes = [
        index
        for index, goal in enumerate(intent.target_behavior)
        if goal.type == "branch"
    ]
    main_axis: tuple[float, float, float] | None = None
    if len(branch_indexes) >= 2:
        for goal in intent.target_behavior[branch_indexes[0] + 1 : branch_indexes[-1]]:
            try:
                if goal.type == "move" and goal.direction is not None:
                    main_axis = direction_to_vector(goal.direction)
                    break
                if goal.type == "route" and goal.path_kind == "line":
                    if goal.direction is not None:
                        main_axis = direction_to_vector(goal.direction)
                        break
                    if goal.terminal_axis is not None:
                        main_axis = normalize(vec(goal.terminal_axis))
                        break
            except ValueError:
                continue
    if main_axis is None:
        main_axis = normalize(vec(intent.start_axis))

    arm_axes: list[tuple[float, float, float]] = []
    first_branch_index = (
        branch_indexes[0] if branch_indexes else len(intent.target_behavior)
    )
    prefix = intent.target_behavior[:first_branch_index]
    root_axis: tuple[float, float, float] | None = None
    for goal in reversed(prefix):
        try:
            if goal.type == "route":
                if goal.terminal_axis is not None:
                    root_axis = normalize(vec(goal.terminal_axis))
                elif len(goal.required_waypoints) >= 2:
                    root_axis = normalize(
                        sub(
                            vec(goal.required_waypoints[-1]),
                            vec(goal.required_waypoints[-2]),
                        )
                    )
                elif (
                    len(goal.required_waypoints) == 1
                    and goal.waypoint_frame == "relative_to_target"
                ):
                    root_axis = normalize(vec(goal.required_waypoints[0]))
                elif goal.direction is not None:
                    root_axis = direction_to_vector(goal.direction)
            elif goal.type == "move" and goal.direction is not None:
                root_axis = direction_to_vector(goal.direction)
        except ValueError:
            root_axis = None
        if root_axis is not None:
            break
    arm_axes.append(root_axis or normalize(vec(intent.start_axis)))

    for goal in intent.target_behavior:
        if goal.type != "branch":
            continue
        arm_axes.extend(vec(value) for value in goal.required_outlet_vectors)
        arm_axes.extend(vec(outlet.axis) for outlet in goal.required_outlets)
        arm_axes.extend(
            direction_to_vector(value) for value in goal.required_outlet_directions
        )

    result: list[float] = []
    for arm_axis in arm_axes:
        cosine = max(
            -1.0,
            min(1.0, abs(dot(normalize(main_axis), normalize(arm_axis)))),
        )
        result.append(math.degrees(math.acos(cosine)))
    return result


def _terminal_arm_length_contracts(intent: IntentResult) -> list[float | None]:
    """START arm과 각 최종 branch outlet의 독립 중심선 길이를 순서대로 모은다."""

    first_branch_index = next(
        (
            index
            for index, goal in enumerate(intent.target_behavior)
            if goal.type == "branch"
        ),
        len(intent.target_behavior),
    )
    root_length = 0.0
    has_root_length = False
    for goal in intent.target_behavior[:first_branch_index]:
        if goal.type in {"move", "route"} and goal.length is not None:
            root_length += float(goal.length)
            has_root_length = True
        elif (
            goal.type == "turn"
            and goal.bend_radius is not None
            and goal.angle is not None
        ):
            root_length += math.radians(abs(float(goal.angle))) * float(
                goal.bend_radius
            )
            has_root_length = True
    result: list[float | None] = [root_length if has_root_length else None]

    for goal in intent.target_behavior:
        if goal.type != "branch":
            continue
        described_count = (
            goal.branch_count
            or len(goal.required_outlets)
            or len(goal.required_outlet_vectors)
            or len(goal.required_outlet_directions)
        )
        if goal.required_outlets:
            result.extend(
                float(outlet.length) if outlet.length is not None else None
                for outlet in goal.required_outlets
            )
        elif goal.length is not None:
            result.extend(float(goal.length) for _index in range(described_count))
        else:
            result.extend(None for _index in range(described_count))
    return result


def _explicit_mm_values(text: str) -> list[float]:
    """명시적 mm 값과 범위의 양 끝점을 원문 순서대로 반환한다."""

    ranges = _explicit_mm_ranges(text)
    events: list[tuple[int, int, list[float]]] = [
        (
            source_range.span[0],
            0,
            [source_range.authored_start, source_range.authored_end],
        )
        for source_range in ranges
    ]
    for match in _EXPLICIT_MM_VALUE.finditer(text):
        if any(
            match.start() < source_range.span[1] and source_range.span[0] < match.end()
            for source_range in ranges
        ):
            continue
        value = _metric_number(match.group(1))
        if math.isfinite(value):
            events.append((match.start(), 1, [value]))
    values: list[float] = []
    for _offset, _kind, event_values in sorted(events):
        values.extend(event_values)
    return values


def _exact_mm_contract_values(text: str) -> list[float]:
    """범위와 동일 전역 속성의 반복을 제외한 exact 수치 계약을 만든다.

    근사 표현(``about 68 mm`` 등)은 임의 허용오차를 만들어 값을 바꾸지
    않고 68 mm라는 nominal 계약으로 유지한다. 반면 명시적 범위만 그
    경계 안의 선택을 허용한다.
    """

    range_spans = [source_range.span for source_range in _explicit_mm_ranges(text)]
    duplicate_singleton_spans: set[tuple[int, int]] = set()
    seen_singletons: list[tuple[str, float]] = []
    for _offset, span, role, value in _anchored_mm_matches(text):
        if role not in _SINGLETON_MM_ROLES:
            continue
        if any(
            prior_role == role and _same_metric_value(prior_value, value)
            for prior_role, prior_value in seen_singletons
        ):
            duplicate_singleton_spans.add(span)
        else:
            seen_singletons.append((role, value))

    result: list[float] = []
    for match in _EXPLICIT_MM_VALUE.finditer(text):
        if any(
            match.start() < right and left < match.end() for left, right in range_spans
        ):
            continue
        if match.span(1) in duplicate_singleton_spans:
            continue
        value = _metric_number(match.group(1))
        if math.isfinite(value):
            result.append(value)
    return result


def _explicit_primary_continuation_length_contracts(
    text: str,
) -> list[_PrimaryContinuationLengthContract]:
    """Extract totals that explicitly include a named final subsegment.

    Phrases such as ``주 경로 80 mm, 마지막 30 mm 구간`` describe one
    relational contract.  The 80 mm value must equal the sum of the primary
    continuation components; it must not be copied into a second independent
    route field next to the 30 mm transition.
    """

    contracts: list[_PrimaryContinuationLengthContract] = []
    claimed_spans: list[tuple[int, int]] = []
    for pattern in _PRIMARY_CONTINUATION_WITH_FINAL_SEGMENT_PATTERNS:
        for match in pattern.finditer(text):
            if any(
                match.start() < right and left < match.end()
                for left, right in claimed_spans
            ):
                continue
            total = _metric_number(match.group("total"))
            final = _metric_number(match.group("final"))
            if not math.isfinite(total) or not math.isfinite(final):
                continue
            contracts.append(
                _PrimaryContinuationLengthContract(
                    total_length=total,
                    final_segment_length=final,
                    total_span=match.span("total"),
                    final_segment_span=match.span("final"),
                    span=match.span(),
                )
            )
            claimed_spans.append(match.span())
    return contracts


def _exact_mm_contract_values_excluding_spans(
    text: str,
    excluded_spans: list[tuple[int, int]],
) -> list[float]:
    """Return exact contracts except values owned by relational validators.

    The ordinary multiplicity contract remains intact. Only a scalar that is
    provably represented by a checked relationship, such as an inclusive total,
    is removed from the standalone-value requirement.
    """

    range_spans = [source_range.span for source_range in _explicit_mm_ranges(text)]
    duplicate_singleton_spans: set[tuple[int, int]] = set()
    seen_singletons: list[tuple[str, float]] = []
    for _offset, span, role, value in _anchored_mm_matches(text):
        if role not in _SINGLETON_MM_ROLES:
            continue
        if any(
            prior_role == role and _same_metric_value(prior_value, value)
            for prior_role, prior_value in seen_singletons
        ):
            duplicate_singleton_spans.add(span)
        else:
            seen_singletons.append((role, value))

    result: list[float] = []
    for match in _EXPLICIT_MM_VALUE.finditer(text):
        span = match.span(1)
        if any(span[0] < right and left < span[1] for left, right in range_spans):
            continue
        if span in duplicate_singleton_spans:
            continue
        if any(span[0] < right and left < span[1] for left, right in excluded_spans):
            continue
        value = _metric_number(match.group(1))
        if math.isfinite(value):
            result.append(value)
    return result


def _approximate_mm_measurements(text: str) -> list[_SourceNominalMeasurement]:
    """Extract explicitly approximate scalar measurements as bounded nominals."""

    result: list[_SourceNominalMeasurement] = []
    claimed_spans: list[tuple[int, int]] = []
    for pattern in _APPROXIMATE_MM_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span("value")
            if any(span[0] < right and left < span[1] for left, right in claimed_spans):
                continue
            value = _metric_number(match.group("value"))
            if not math.isfinite(value):
                continue
            delta = abs(value) * SOURCE_NOMINAL_RELATIVE_TOLERANCE
            result.append(
                _SourceNominalMeasurement(
                    value=value,
                    minimum=value - delta,
                    maximum=value + delta,
                    span=span,
                )
            )
            claimed_spans.append(span)
    return result


def build_source_measurement_contract(prompt: str) -> SourceMeasurementContract:
    """Interpret source measurements exactly once for all downstream consumers."""

    ranges = tuple(_explicit_mm_ranges(prompt))
    primary_lengths = tuple(
        _explicit_primary_continuation_length_contracts(prompt)
    )
    nominal_measurements = tuple(_approximate_mm_measurements(prompt))
    nominal_by_span = {item.span: item for item in nominal_measurements}
    role_measurements: list[_SourceRoleMeasurement] = []
    singleton_values: dict[str, list[float]] = {}
    for _offset, span, role, value in _anchored_mm_matches(prompt):
        seen = singleton_values.setdefault(role, [])
        if role in _SINGLETON_MM_ROLES and any(
            _same_metric_value(value, previous) for previous in seen
        ):
            continue
        seen.append(value)
        nominal = nominal_by_span.get(span)
        role_measurements.append(
            _SourceRoleMeasurement(
                role=role,
                value=value,
                span=span,
                relation="nominal" if nominal is not None else "exact",
                minimum=nominal.minimum if nominal is not None else value,
                maximum=nominal.maximum if nominal is not None else value,
            )
        )

    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    explicit_values = tuple(_explicit_mm_values(prompt))
    exact_values = tuple(_exact_mm_contract_values(prompt))
    standalone_values = tuple(
        _exact_mm_contract_values_excluding_spans(
            prompt,
            [
                *[item.total_span for item in primary_lengths],
                *[item.span for item in nominal_measurements],
            ],
        )
    )
    contract = SourceMeasurementContract(
        version=SOURCE_MEASUREMENT_CONTRACT_VERSION,
        prompt_sha256=prompt_sha256,
        explicit_mm_values=explicit_values,
        exact_mm_values=exact_values,
        standalone_exact_mm_values=standalone_values,
        ranges=ranges,
        nominal_measurements=nominal_measurements,
        role_measurements=tuple(role_measurements),
        primary_continuation_lengths=primary_lengths,
        contract_digest="",
    )
    return replace(contract, contract_digest=stable_digest(contract.content_payload()))


def _goal_centerline_length_component(goal: Goal) -> tuple[str, float] | None:
    if goal.type in {"move", "route"} and goal.length is not None:
        return ("length", float(goal.length))
    if goal.type == "diameter_change" and goal.transition_length is not None:
        return ("transition_length", float(goal.transition_length))
    if goal.type == "connector" and goal.length is not None:
        return ("length", float(goal.length))
    if goal.type == "turn" and goal.angle is not None and goal.bend_radius is not None:
        return (
            "arc_length",
            math.radians(abs(float(goal.angle))) * float(goal.bend_radius),
        )
    return None


def _primary_continuation_length_candidates(
    intent: IntentResult,
) -> list[_PrimaryContinuationLengthCandidate]:
    """Measure each dependency-ordered primary continuation after a branch."""

    candidates: list[_PrimaryContinuationLengthCandidate] = []
    for branch_index, branch in enumerate(intent.target_behavior):
        if branch.type != "branch" or not _effective_primary_outlet(branch):
            continue
        previous_goal_id = branch.goal_id
        components: list[tuple[str, float]] = []
        for goal in intent.target_behavior[branch_index + 1 :]:
            if (
                goal.depends_on_goal_ids
                and previous_goal_id not in goal.depends_on_goal_ids
            ):
                break
            component = _goal_centerline_length_component(goal)
            if component is not None:
                field_name, value = component
                components.append((f"{goal.goal_id or goal.type}.{field_name}", value))
            previous_goal_id = goal.goal_id
        if components:
            candidates.append(
                _PrimaryContinuationLengthCandidate(
                    branch_goal_id=branch.goal_id or f"target_behavior[{branch_index}]",
                    total_length=sum(value for _field, value in components),
                    components=tuple(components),
                )
            )
    return candidates


def _effective_primary_outlet(goal: Goal) -> bool:
    """Mirror the compiler's effective primary-outlet default."""

    if goal.include_primary_outlet is not None:
        return bool(goal.include_primary_outlet)
    return not bool(goal.required_outlet_vectors or goal.required_outlets)


def _primary_continuation_length_issues(
    contracts: list[_PrimaryContinuationLengthContract],
    intent: IntentResult,
) -> list[str]:
    if not contracts:
        return []
    candidates = _primary_continuation_length_candidates(intent)
    issues: list[str] = []
    for contract in contracts:
        matching_total = [
            candidate
            for candidate in candidates
            if _same_metric_value(candidate.total_length, contract.total_length)
        ]
        matching_final = [
            candidate
            for candidate in matching_total
            if any(
                field.endswith(".transition_length")
                and _same_metric_value(value, contract.final_segment_length)
                for field, value in candidate.components
            )
        ]
        if matching_final:
            continue
        rendered = [
            {
                "branch_goal_id": candidate.branch_goal_id,
                "total_length": candidate.total_length,
                "components": dict(candidate.components),
            }
            for candidate in candidates
        ]
        if not matching_total:
            issues.append(
                "primary continuation total length does not equal the authored "
                f"{contract.total_length:.6g} mm inclusive total; actual={rendered}. "
                f"The final {contract.final_segment_length:.6g} mm transition is "
                "part of that total, so author preceding straight length as "
                f"total-final ({contract.total_length - contract.final_segment_length:.6g} mm), "
                "not as the full total"
            )
        else:
            issues.append(
                "primary continuation total length is correct, but its authored "
                f"final {contract.final_segment_length:.6g} mm subsegment is not a "
                f"diameter_change.transition_length; actual={rendered}"
            )
    return issues


def _protected_source_measurement_bindings(
    prompt: str,
    intent: IntentResult,
    *,
    source_measurement_contract: SourceMeasurementContract | None = None,
) -> dict[str, Any]:
    """Return already-correct source bindings that a repair must not regress."""

    source_measurement_contract = (
        source_measurement_contract or build_source_measurement_contract(prompt)
    )
    source_measurement_contract.assert_matches_prompt(prompt)
    anchored_values = {
        role: list(values)
        for role, values in source_measurement_contract.role_values.items()
    }
    candidate_values = _anchored_intent_values(intent)
    claims_by_role: dict[str, list[_SourceRoleMeasurement]] = {}
    for claim in source_measurement_contract.role_measurements:
        claims_by_role.setdefault(claim.role, []).append(claim)
    protected_roles = {
        role: [claim.value for claim in claims]
        for role, claims in claims_by_role.items()
        if not _ordered_missing_role_claims(
            claims,
            candidate_values.get(role, []),
        )
    }
    protected_totals: list[dict[str, Any]] = []
    candidates = _primary_continuation_length_candidates(intent)
    for contract in source_measurement_contract.primary_continuation_lengths:
        for candidate in candidates:
            if not _same_metric_value(candidate.total_length, contract.total_length):
                continue
            if not any(
                field.endswith(".transition_length")
                and _same_metric_value(value, contract.final_segment_length)
                for field, value in candidate.components
            ):
                continue
            protected_totals.append(
                {
                    "total_length": contract.total_length,
                    "final_segment_length": contract.final_segment_length,
                    "branch_goal_id": candidate.branch_goal_id,
                    "components": dict(candidate.components),
                }
            )
            break
    return {
        "source_measurement_contract_digest": (
            source_measurement_contract.contract_digest
        ),
        "role_bindings": protected_roles,
        "primary_continuation_totals": protected_totals,
    }


def _intent_range_candidate_values(intent: IntentResult) -> list[float]:
    """범위 계약과 비교 가능한 양의 typed 물리 치수만 수집한다."""

    return [
        value
        for _path, value in _positive_intent_dimensions(intent)
        if math.isfinite(value) and value > 0.0
    ]


def _explicit_vector3_contracts(
    text: str,
) -> list[tuple[tuple[float, float, float], bool]]:
    """Return source tuples with exact-coordinate vs direction-ratio roles."""

    result: list[tuple[tuple[float, float, float], bool]] = []
    for match in _EXPLICIT_VECTOR3.finditer(text):
        vector_value = tuple(
            float(match.group(name).replace("−", "-")) for name in ("x", "y", "z")
        )
        if all(math.isfinite(value) for value in vector_value):
            prefix = text[max(0, match.start() - 96) : match.start()]
            suffix = text[match.end() : min(len(text), match.end() + 48)]
            is_direction = bool(
                _PROPORTIONAL_VECTOR_PREFIX.search(prefix)
                or _PROPORTIONAL_VECTOR_SUFFIX.search(suffix)
            )
            result.append((vector_value, is_direction))
    return result


def _explicit_proportional_direction_contracts(
    text: str,
) -> list[_ProportionalDirectionContract]:
    """Extract proportional vector ratios with their authored semantic role."""

    result: list[_ProportionalDirectionContract] = []
    for match in _EXPLICIT_VECTOR3.finditer(text):
        prefix = text[max(0, match.start() - 192) : match.start()]
        suffix = text[match.end() : min(len(text), match.end() + 64)]
        if not (
            _PROPORTIONAL_VECTOR_PREFIX.search(prefix)
            or _PROPORTIONAL_VECTOR_SUFFIX.search(suffix)
        ):
            continue
        value = tuple(
            float(match.group(name).replace("−", "-")) for name in ("x", "y", "z")
        )
        if not all(math.isfinite(component) for component in value):
            continue
        # Restrict role words to the current clause. A prior sentence may name
        # START or a branch and must not relabel a later spline heading.
        local_prefix = re.split(r"[.;\n]", prefix)[-1]
        local_suffix = re.split(r"[.;\n]", suffix)[0]
        context = f"{local_prefix[-128:]} {local_suffix[:48]}".lower()
        if re.search(
            r"\bplane[-\s]*normal\b|\bnormal\s+(?:of|to)\s+(?:the\s+)?plane\b|"
            r"평면[^.;\n]{0,24}법선|법선[^.;\n]{0,24}평면",
            context,
        ):
            role = "plane_normal"
        elif re.search(
            r"\b(?:branch|arm|terminal\s+branch|branch\s+outlet)\b|"
            r"분기|가지|암\b",
            context,
        ):
            role = "branch_outlet"
        elif re.search(
            r"\b(?:actuator|flange|component)\b|액추에이터|플랜지",
            context,
        ):
            role = "component_axis"
        elif re.search(
            r"(?:\bstart\b|\bstarting\b|\binlet\b)[^.;\n]{0,48}\baxis\b|"
            r"\bstart_axis\b|시작[^.;\n]{0,32}(?:축|방향)",
            context,
        ):
            role = "start_axis"
        elif re.search(
            r"\b(?:outlet|terminal|final)\s+(?:axis|heading|direction)\b|"
            r"\b(?:spline|route|bend|turn)\b|\b(?:last|final)\s+chord\b|"
            r"출구|말단|최종[^.;\n]{0,24}(?:축|방향)|마지막[^.;\n]{0,24}현",
            context,
        ):
            role = "goal_terminal"
        else:
            # Ambiguous proportional directions may bind to a typed generated
            # terminal, but never borrow START or a plane normal.
            role = "generated_terminal"
        result.append(_ProportionalDirectionContract(value=value, role=role))
    return result


def _intent_vector_values(
    intent: IntentResult,
) -> list[tuple[float, float, float]]:
    result = [vec(intent.start_position), vec(intent.start_axis)]
    for goal in intent.target_behavior:
        for value in (
            goal.plane_normal,
            goal.branch_plane_normal,
            goal.offset,
            goal.terminal_position,
            goal.terminal_axis,
        ):
            if value is not None:
                result.append(vec(value))
        result.extend(vec(value) for value in goal.required_waypoints)
        result.extend(vec(value) for value in goal.required_outlet_vectors)
        result.extend(vec(outlet.axis) for outlet in goal.required_outlets)
        if goal.component_spec is not None:
            for value in (
                goal.component_spec.flange_reference_axis,
                goal.component_spec.actuator_axis,
            ):
                if value is not None:
                    result.append(vec(value))
    return result


def _intent_direction_candidates(
    intent: IntentResult,
) -> list[_IntentDirectionCandidate]:
    """Collect one role-labelled candidate per physical authored direction."""

    result = [
        _IntentDirectionCandidate(
            semantic_id="START.axis",
            role="start_axis",
            value=vec(intent.start_axis),
        )
    ]
    for goal_index, goal in enumerate(intent.target_behavior):
        goal_key = goal.goal_id or f"goal_{goal_index}"
        for field_name, value in (
            ("plane_normal", goal.plane_normal),
            ("branch_plane_normal", goal.branch_plane_normal),
        ):
            if value is not None:
                result.append(
                    _IntentDirectionCandidate(
                        semantic_id=f"{goal_key}.{field_name}",
                        role="plane_normal",
                        value=vec(value),
                    )
                )

        outlet_values: list[tuple[float, float, float]] = []
        outlet_values.extend(vec(value) for value in goal.required_outlet_vectors)
        outlet_values.extend(vec(outlet.axis) for outlet in goal.required_outlets)
        outlet_values.extend(
            direction_to_vector(value) for value in goal.required_outlet_directions
        )
        result.extend(
            _IntentDirectionCandidate(
                semantic_id=f"{goal_key}.branch_outlet[{index}]",
                role="branch_outlet",
                value=value,
            )
            for index, value in enumerate(outlet_values)
        )

        terminal_value: tuple[float, float, float] | None = None
        if goal.type == "route":
            if goal.terminal_axis is not None:
                terminal_value = vec(goal.terminal_axis)
            elif len(goal.required_waypoints) >= 2:
                terminal_value = sub(
                    vec(goal.required_waypoints[-1]),
                    vec(goal.required_waypoints[-2]),
                )
            elif (
                len(goal.required_waypoints) == 1
                and goal.waypoint_frame == "relative_to_target"
            ):
                terminal_value = vec(goal.required_waypoints[0])
            elif goal.direction is not None:
                terminal_value = direction_to_vector(goal.direction)
        elif (
            goal.type
            in {
                "move",
                "turn",
                "diameter_change",
                "connector",
            }
            and goal.direction is not None
        ):
            terminal_value = direction_to_vector(goal.direction)
        if terminal_value is not None:
            result.append(
                _IntentDirectionCandidate(
                    semantic_id=f"{goal_key}.terminal_heading",
                    role="goal_terminal",
                    value=terminal_value,
                )
            )

        if goal.component_spec is not None:
            for field_name, value in (
                (
                    "flange_reference_axis",
                    goal.component_spec.flange_reference_axis,
                ),
                ("actuator_axis", goal.component_spec.actuator_axis),
            ):
                if value is not None:
                    result.append(
                        _IntentDirectionCandidate(
                            semantic_id=f"{goal_key}.component.{field_name}",
                            role="component_axis",
                            value=vec(value),
                        )
                    )
    return result


def _direction_roles_compatible(source_role: str, candidate_role: str) -> bool:
    if source_role == "generated_terminal":
        return candidate_role in {
            "goal_terminal",
            "branch_outlet",
            "component_axis",
        }
    if source_role == "goal_terminal":
        # A branch outlet is also a generated terminal. Source text that says
        # only "outlet direction" cannot distinguish a route terminal from a
        # branch terminal, so enforce the vector without inventing that role.
        return candidate_role in {"goal_terminal", "branch_outlet"}
    return source_role == candidate_role


def _positive_parallel_direction(
    expected: tuple[float, float, float],
    candidate: tuple[float, float, float],
) -> bool:
    try:
        expected_unit = normalize(vec(expected))
        candidate_unit = normalize(vec(candidate))
    except ValueError:
        return False
    return dot(expected_unit, candidate_unit) >= 1.0 - 1e-6


def _anchored_mm_matches(
    text: str,
) -> list[tuple[int, tuple[int, int], str, float]]:
    """역할과 원문 span을 보존한 high-confidence 측정값을 추출한다."""

    matches: list[tuple[int, tuple[int, int], str, float]] = []
    claimed_spans: list[tuple[int, int]] = []
    for role, pattern in _ANCHORED_MM_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span("value")
            if any(span[0] < right and left < span[1] for left, right in claimed_spans):
                continue
            value = _metric_number(match.group("value"))
            if not math.isfinite(value):
                continue
            claimed_spans.append(span)
            matches.append((span[0], span, role, value))
    return sorted(matches)


def _anchored_mm_values(text: str) -> dict[str, list[float]]:
    """Extract only high-confidence source measurement roles.

    Specific change/transition patterns run before generic global-section
    patterns.  A numeric source span is claimed once, preventing a phrase such
    as ``reduce outer diameter`` from being interpreted as both an initial and
    an output diameter. Repeated mentions of the same global section property
    collapse to one semantic contract, while contradictory values remain.
    """

    result: dict[str, list[float]] = {}
    for _offset, _span, role, value in _anchored_mm_matches(text):
        role_values = result.setdefault(role, [])
        if role in _SINGLETON_MM_ROLES and any(
            _same_metric_value(value, previous) for previous in role_values
        ):
            continue
        role_values.append(value)
    return result


def _anchored_intent_values(intent: IntentResult) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {
        "global_outer_diameter": [float(intent.global_spec.outer_diameter)],
        "global_inner_diameter": [
            float(intent.global_spec.outer_diameter)
            - 2.0 * float(intent.global_spec.wall_thickness)
        ],
        "global_wall_thickness": [float(intent.global_spec.wall_thickness)],
        "straight_length": [],
        "connector_length": [],
        "transition_length": [],
        "diameter_out": [],
        "wall_thickness_out": [],
        "diameter_in_reference": [],
        "wall_thickness_in_reference": [],
        "route_rise": [],
        "junction_blend_radius": [],
        "junction_inner_blend_radius": [],
        "junction_max_hub_radius": [],
    }
    current_outer_diameter = float(intent.global_spec.outer_diameter)
    current_wall_thickness = float(intent.global_spec.wall_thickness)
    for goal in intent.target_behavior:
        if goal.type in {"move", "route"} and goal.length is not None:
            result["straight_length"].append(float(goal.length))
        if goal.type == "connector" and goal.length is not None:
            result["connector_length"].append(float(goal.length))
        if goal.type == "branch":
            for role, value in (
                ("junction_blend_radius", goal.blend_radius),
                ("junction_inner_blend_radius", goal.inner_blend_radius),
                ("junction_max_hub_radius", goal.max_hub_radius),
            ):
                if value is not None:
                    result[role].append(float(value))
        if goal.type == "diameter_change":
            result["diameter_in_reference"].append(current_outer_diameter)
            result["wall_thickness_in_reference"].append(current_wall_thickness)
            for role, value in (
                ("transition_length", goal.transition_length),
                ("diameter_out", goal.diameter_out),
                ("wall_thickness_out", goal.wall_thickness_out),
            ):
                if value is not None:
                    result[role].append(float(value))
            if goal.diameter_out is not None:
                current_outer_diameter = float(goal.diameter_out)
            if goal.wall_thickness_out is not None:
                current_wall_thickness = float(goal.wall_thickness_out)
        if (
            goal.type == "route"
            and goal.path_kind == "spline"
            and goal.waypoint_frame == "relative_to_target"
            and goal.required_waypoints
        ):
            result["route_rise"].append(float(goal.required_waypoints[-1][2]))
    return result


def _ordered_missing_values(
    expected: list[float],
    candidates: list[float],
) -> list[float]:
    """Return values that cannot map to distinct role-compatible fields.

    Candidate values may include inferred intermediate goals, so source values
    need only form an ordered subsequence.  Each candidate is consumed at most
    once, preserving multiplicity without forcing a brittle one-to-one count.
    """

    missing: list[float] = []
    start = 0
    for value in expected:
        match_index = next(
            (
                index
                for index in range(start, len(candidates))
                if _same_metric_value(value, candidates[index])
            ),
            None,
        )
        if match_index is None:
            missing.append(value)
        else:
            start = match_index + 1
    return missing


def _ordered_missing_role_claims(
    claims: list[_SourceRoleMeasurement],
    candidates: list[float],
) -> list[float]:
    """Match exact/nominal role claims without allowing cross-role borrowing."""

    missing: list[float] = []
    start = 0
    for claim in claims:
        minimum = claim.minimum if claim.minimum is not None else claim.value
        maximum = claim.maximum if claim.maximum is not None else claim.value
        match_index = next(
            (
                index
                for index in range(start, len(candidates))
                if (
                    minimum - 1e-9
                    <= float(candidates[index])
                    <= maximum + 1e-9
                )
            ),
            None,
        )
        if match_index is None:
            missing.append(claim.value)
        else:
            start = match_index + 1
    return missing


def _intent_metric_values(intent: IntentResult) -> list[float]:
    alias_paths: set[str] = set()
    for goal_index, goal in enumerate(intent.target_behavior):
        if (
            goal.type == "connector"
            and goal.length is not None
            and goal.component_spec is not None
            and goal.component_spec.body_length is not None
            and _same_metric_value(goal.length, goal.component_spec.body_length)
        ):
            # One physical authored length is represented in both the connector
            # goal and its same-span body detail.  It must count once when source
            # measurement multiplicity is checked.
            alias_paths.add(f"target_behavior[{goal_index}].component_spec.body_length")
    values = [
        value
        for path, value in _positive_intent_dimensions(intent)
        if path not in alias_paths
    ]
    values.extend(float(component) for component in intent.start_position)
    text_contracts = [
        contract
        for contract in intent.hard_constraints
        if contract.startswith("unsupported:")
    ]
    for goal in intent.target_behavior:
        if goal.offset is not None:
            values.extend(float(component) for component in goal.offset)
        if goal.terminal_position is not None:
            values.extend(float(component) for component in goal.terminal_position)
        for waypoint in goal.required_waypoints:
            values.extend(float(component) for component in waypoint)
    for constraint in intent.geometric_constraints:
        if constraint.minimum is not None:
            values.extend(float(component) for component in constraint.minimum)
        if constraint.maximum is not None:
            values.extend(float(component) for component in constraint.maximum)
    for text in text_contracts:
        values.extend(_explicit_mm_values(text))
    return [value for value in values if math.isfinite(value)]


def _same_metric_value(left: float, right: float) -> bool:
    tolerance = max(1e-9, abs(left) * 1e-9)
    return abs(left - right) <= tolerance
