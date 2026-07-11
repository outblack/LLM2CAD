"""의도, 행동, 포트 그래프, 검증 증거와 실행 보고서의 타입 계약을 정의한다.

외부 JSON과 내부 Python 데이터를 입력받아 엄격한 Pydantic 모델로 변환한다.
추가 필드, 비유한 수와 모순된 조합은 조용히 수용하지 않고 검증 오류로 거부한다.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypeAliasType


class StrictModel(BaseModel):
    """추가 필드와 NaN/Infinity를 허용하지 않는 모든 계약 모델의 기반이다."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ProviderWireModel(StrictModel):
    """Marker for DTOs whose host parser must equal the advertised JSON Schema."""

    provider_wire_contract: ClassVar[bool] = True


Direction = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
ConnectionTarget = Literal["another_open_port", "start_anchor"]
WaypointFrame = Literal["global", "relative_to_target"]
WaypointScalePolicy = Literal["fixed", "uniform_expand_for_safety"]
Vector3 = tuple[float, float, float]
# Reused only at structured-output boundaries.  A named alias makes Pydantic
# emit one compact $defs entry while values still validate as ordinary tuples.
LLMVector3 = TypeAliasType("V3", tuple[float, float, float])
IssueSeverity = Literal["info", "warning", "error"]
VerificationStatus = Literal["not_run", "passed", "failed", "partial"]
MCPStatus = Literal["skipped", "passed", "failed", "unavailable"]
OpenPortExpectationSource = Literal["explicit", "derived", "unknown"]
InlineComponentKind = Literal["flange", "coupling", "union", "valve"]


class GlobalSpec(StrictModel):
    """전체 파이프 설계가 시작할 기본 원형 단면 계약이다."""

    outer_diameter: float = Field(default=20.0, gt=0)
    wall_thickness: float = Field(default=2.0, ge=0)
    is_hollow: bool = True
    units: str = "mm"


class GeometricConstraint(StrictModel):
    """최종 상태에서 결정론적으로 검사할 수 있는 수치 공간 제약이다."""

    constraint_id: str
    type: Literal[
        "max_extent",
        "max_module_count",
        "max_total_centerline_length",
        "bounding_box",
    ]
    axis: Direction | None = None
    value: float | None = Field(default=None, gt=0)
    minimum: Vector3 | None = None
    maximum: Vector3 | None = None

    @model_validator(mode="after")
    def validate_constraint(self) -> "GeometricConstraint":
        if self.type == "max_extent" and (self.axis is None or self.value is None):
            raise ValueError("max_extent requires axis and value")
        if (
            self.type in {"max_module_count", "max_total_centerline_length"}
            and self.value is None
        ):
            raise ValueError(f"{self.type} requires value")
        if self.type == "max_module_count" and self.value is not None:
            if abs(self.value - round(self.value)) > 1e-9:
                raise ValueError("max_module_count value must be an integer")
        if self.type == "bounding_box":
            if self.minimum is None or self.maximum is None:
                raise ValueError("bounding_box requires minimum and maximum")
            if not _finite_vector(self.minimum) or not _finite_vector(self.maximum):
                raise ValueError("bounding_box vectors must be finite")
            if any(low >= high for low, high in zip(self.minimum, self.maximum)):
                raise ValueError("bounding_box minimum must be below maximum")
        return self


class IntentMaxExtentConstraint(StrictModel):
    """Provider wire form whose type structurally requires axis and value."""

    constraint_id: str
    type: Literal["max_extent"]
    axis: Direction
    value: float


class IntentMaxModuleCountConstraint(StrictModel):
    constraint_id: str
    type: Literal["max_module_count"]
    value: int = Field(ge=1)


class IntentMaxCenterlineLengthConstraint(StrictModel):
    constraint_id: str
    type: Literal["max_total_centerline_length"]
    value: float


class IntentBoundingBoxConstraint(StrictModel):
    constraint_id: str
    type: Literal["bounding_box"]
    minimum: LLMVector3
    maximum: LLMVector3


IntentGeometricConstraint = Annotated[
    IntentMaxExtentConstraint
    | IntentMaxModuleCountConstraint
    | IntentMaxCenterlineLengthConstraint
    | IntentBoundingBoxConstraint,
    Field(discriminator="type"),
]


class ComponentGoalSpec(StrictModel):
    """User-authored accessory dimensions that must survive later planning."""

    component_type: InlineComponentKind
    body_outer_diameter: float | None = Field(default=None, gt=0)
    body_start_offset: float | None = Field(default=None, ge=0)
    body_length: float | None = Field(default=None, gt=0)
    flange_bolt_count: int | None = Field(default=None, ge=3, le=32)
    flange_bolt_circle_diameter: float | None = Field(default=None, gt=0)
    flange_bolt_hole_diameter: float | None = Field(default=None, gt=0)
    flange_reference_axis: Vector3 | None = None
    union_ring_outer_diameter: float | None = Field(default=None, gt=0)
    union_ring_length: float | None = Field(default=None, gt=0)
    actuator_diameter: float | None = Field(default=None, gt=0)
    actuator_height: float | None = Field(default=None, gt=0)
    actuator_axis: Vector3 | None = None

    @model_validator(mode="after")
    def validate_component_contract(self) -> "ComponentGoalSpec":
        flange_values = (
            self.flange_bolt_count,
            self.flange_bolt_circle_diameter,
            self.flange_bolt_hole_diameter,
            self.flange_reference_axis,
        )
        union_values = (self.union_ring_outer_diameter, self.union_ring_length)
        actuator_values = (
            self.actuator_diameter,
            self.actuator_height,
            self.actuator_axis,
        )
        if self.component_type != "flange" and any(
            value is not None for value in flange_values
        ):
            raise ValueError("flange fields require component_type=flange")
        if self.component_type != "union" and any(
            value is not None for value in union_values
        ):
            raise ValueError("union fields require component_type=union")
        if self.component_type != "valve" and any(
            value is not None for value in actuator_values
        ):
            raise ValueError("actuator fields require component_type=valve")
        for label, value in (
            ("flange_reference_axis", self.flange_reference_axis),
            ("actuator_axis", self.actuator_axis),
        ):
            if value is not None and (
                not _finite_vector(value) or _vector_size(value) <= 1e-12
            ):
                raise ValueError(f"{label} must be a finite non-zero vector")
        return self


class BranchGoalOutletSpec(StrictModel):
    """의도 단계에서 요구하는 한 분기 outlet의 축ㆍ길이ㆍ단면 계약이다."""

    axis: Vector3
    length: float | None = Field(default=None, gt=0)
    outer_diameter: float | None = Field(default=None, gt=0)
    wall_thickness: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_outlet_contract(self) -> "BranchGoalOutletSpec":
        if not _finite_vector(self.axis) or _vector_size(self.axis) <= 1e-12:
            raise ValueError("branch outlet axis must be a finite non-zero vector")
        if (
            self.outer_diameter is not None
            and self.wall_thickness is not None
            and self.outer_diameter <= 2.0 * self.wall_thickness
        ):
            raise ValueError("branch outlet diameter must exceed twice wall thickness")
        return self


class Goal(StrictModel):
    """런타임 상태가 보존하는 하나의 불변 설계 목표와 의존성이다."""

    goal_id: str | None = None
    depends_on_goal_ids: list[str] = Field(default_factory=list)
    allow_parallel: bool = False
    type: Literal[
        "move",
        "turn",
        "route",
        "branch",
        "diameter_change",
        "connect",
        "end",
        "connector",
    ]
    direction: Direction | None = None
    path_kind: Literal["line", "circular_arc", "spline"] | None = None
    length: float | None = None
    angle: float | None = None
    bend_radius: float | None = None
    plane_normal: Vector3 | None = None
    branch_count: int | None = None
    branch_angles: list[float] = Field(default_factory=list)
    branch_plane_normal: Vector3 | None = None
    required_outlet_directions: list[Direction] = Field(default_factory=list)
    required_outlet_vectors: list[tuple[float, float, float]] = Field(
        default_factory=list
    )
    required_outlets: list[BranchGoalOutletSpec] = Field(default_factory=list)
    include_primary_outlet: bool | None = None
    junction_style: Literal["hard_fuse", "smooth_hub"] | None = None
    blend_radius: float | None = Field(default=None, gt=0)
    inner_blend_radius: float | None = Field(default=None, gt=0)
    max_hub_radius: float | None = Field(default=None, gt=0)
    diameter_out: float | None = None
    wall_thickness_out: float | None = None
    transition_length: float | None = None
    offset: Vector3 | None = None
    branch_outer_diameter: float | None = None
    branch_wall_thickness: float | None = None
    end_type: Literal["open", "cap", "plug"] | None = None
    termination_thickness: float | None = Field(default=None, gt=0)
    component: str | None = None
    component_spec: ComponentGoalSpec | None = None
    required_waypoints: list[Vector3] = Field(default_factory=list)
    waypoint_frame: WaypointFrame | None = None
    waypoint_scale_policy: WaypointScalePolicy | None = None
    waypoint_safety_scale: float | None = Field(default=None, ge=1.0)
    terminal_position: Vector3 | None = None
    terminal_axis: Vector3 | None = None
    minimum_curvature_radius: float | None = Field(default=None, gt=0)
    connection_target: ConnectionTarget = "another_open_port"
    notes: str | None = None

    @model_validator(mode="after")
    def validate_goal_fields(self) -> "Goal":
        _validate_goal_field_compatibility(self)
        if self.waypoint_frame is not None and not self.required_waypoints:
            raise ValueError("waypoint_frame requires required_waypoints")
        if self.waypoint_scale_policy is not None and not self.required_waypoints:
            raise ValueError("waypoint_scale_policy requires required_waypoints")
        if self.waypoint_frame == "global" and self.waypoint_scale_policy not in {
            None,
            "fixed",
        }:
            raise ValueError("global waypoints require waypoint_scale_policy=fixed")
        if self.component_spec is not None and (
            self.type != "connector"
            or self.component != self.component_spec.component_type
        ):
            raise ValueError(
                "component_spec requires a connector goal with the same component type"
            )
        return self


ConstraintPriority = Literal["safety", "topology", "driving", "preference"]
ConstraintRelation = Literal["exact", "minimum", "maximum", "range", "derived"]
ProofStrength = Literal["proved", "independently_measured", "heuristic", "unknown"]
PreflightStatus = Literal["exact", "adjusted", "infeasible", "unknown"]


class ConstraintRecord(StrictModel):
    """Source-bound design constraint used by deterministic preflight.

    The LLM may interpret the source into a goal, but it does not decide whether
    this record is geometrically satisfiable or silently weaken it later.
    """

    constraint_id: str
    constraint_type: str
    source_goal_id: str | None = None
    source_field: str | None = None
    source_span: str | None = None
    priority: ConstraintPriority
    relation: ConstraintRelation
    value: Any = None
    relaxable: bool = False
    tolerance: float | None = Field(default=None, ge=0)
    variable_ids: list[str] = Field(default_factory=list)


class ConstraintDeviation(StrictModel):
    """Auditable difference between the authored and verified realization."""

    deviation_id: str
    constraint_id: str
    goal_id: str | None = None
    field_path: str
    authored_value: float
    realized_value: float
    absolute_change: float = Field(ge=0)
    relative_change: float = Field(ge=0)
    reason_code: str
    reason: str
    priority: ConstraintPriority = "driving"


class ConflictCertificate(StrictModel):
    """One versioned, machine-checkable reason a realization was rejected."""

    certificate_id: str
    conflict_type: Literal[
        "provider",
        "protocol",
        "intent_contract",
        "topology",
        "local_geometry",
        "closure",
        "clearance",
        "backend",
        "resource",
    ]
    failed_predicate: str
    proof_strength: ProofStrength
    constraint_ids: list[str] = Field(default_factory=list)
    primitive_ids: list[str] = Field(default_factory=list)
    candidate_digest: str | None = None
    evidence_digest: str | None = None
    measured: float | None = None
    required: float | None = None
    gap: float | None = Field(default=None, ge=0)
    units: str | None = None
    causal_decision_ids: list[str] = Field(default_factory=list)
    earliest_backjump_step: int | None = Field(default=None, ge=1)
    mutable_fields: list[str] = Field(default_factory=list)
    allowed_routes: list[
        Literal[
            "retry_protocol",
            "reauthor_current",
            "change_primitive",
            "probe",
            "backjump",
            "repair_intent",
            "retry_infrastructure",
            "relax_driving_constraint",
            "proven_infeasible",
        ]
    ] = Field(default_factory=list)
    message: str


class ConstraintLedger(StrictModel):
    """Content-addressed projection of the accepted semantic contract."""

    schema_version: int = 1
    ledger_digest: str
    constraints: list[ConstraintRecord]


class GlobalPreflightResult(StrictModel):
    """Tri-state feasibility result plus an explicit best-effort realization."""

    method_version: str
    status: PreflightStatus
    ledger_digest: str
    authored_program_digest: str | None = None
    realized_program_digest: str | None = None
    scale_factor: float = Field(default=1.0, gt=0)
    deviations: list[ConstraintDeviation] = Field(default_factory=list)
    conflicts: list[ConflictCertificate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class IntentResult(StrictModel):
    """검증된 전역 단면, 목표 agenda와 최종 포트 계약을 묶는다."""

    global_spec: GlobalSpec
    target_behavior: list[Goal]
    start_position: Vector3 = (0.0, 0.0, 0.0)
    start_axis: Vector3 = (1.0, 0.0, 0.0)
    expected_open_ports: int | None = Field(default=None, ge=0)
    expected_open_ports_source: OpenPortExpectationSource = "unknown"
    required_components: list[str] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)
    geometric_constraints: list[GeometricConstraint] = Field(default_factory=list)
    design_notes: list[str] = Field(default_factory=list)
    constraint_ledger: ConstraintLedger | None = None
    global_preflight: GlobalPreflightResult | None = None
    prompt_sha256: str | None = None
    contract_digest: str | None = None


class ProductionGlobalSpec(StrictModel):
    """프로덕션 intent가 반드시 작성해야 하는 양의 중공 원형 단면이다."""

    outer_diameter: float = Field(gt=0)
    wall_thickness: float = Field(gt=0)
    is_hollow: Literal[True]
    units: Literal["mm"]

    @model_validator(mode="after")
    def validate_section(self) -> "ProductionGlobalSpec":
        if self.outer_diameter <= 2.0 * self.wall_thickness:
            raise ValueError("outer_diameter must exceed twice wall_thickness")
        return self


class ProductionGoal(StrictModel):
    """프로덕션 목표의 타입별 필드 조합과 측정 가능한 완료 조건을 정의한다."""

    goal_id: str = Field(min_length=1)
    depends_on_goal_ids: list[str]
    allow_parallel: bool
    type: Literal[
        "move",
        "turn",
        "route",
        "branch",
        "diameter_change",
        "connect",
        "end",
        "connector",
    ]
    direction: Direction | None = None
    path_kind: Literal["line", "circular_arc", "spline"] | None = None
    length: float | None = None
    angle: float | None = None
    bend_radius: float | None = Field(default=None, gt=0)
    plane_normal: Vector3 | None = None
    branch_count: int | None = None
    branch_angles: list[float] = Field(default_factory=list)
    branch_plane_normal: Vector3 | None = None
    required_outlet_directions: list[Direction] = Field(default_factory=list)
    required_outlet_vectors: list[Vector3] = Field(default_factory=list)
    required_outlets: list[BranchGoalOutletSpec] = Field(default_factory=list)
    include_primary_outlet: bool | None = None
    junction_style: Literal["hard_fuse", "smooth_hub"] | None = None
    blend_radius: float | None = Field(default=None, gt=0)
    inner_blend_radius: float | None = Field(default=None, gt=0)
    max_hub_radius: float | None = Field(default=None, gt=0)
    diameter_out: float | None = None
    wall_thickness_out: float | None = Field(default=None, gt=0)
    transition_length: float | None = Field(default=None, gt=0)
    offset: Vector3 | None = None
    branch_outer_diameter: float | None = Field(default=None, gt=0)
    branch_wall_thickness: float | None = Field(default=None, gt=0)
    end_type: Literal["cap", "plug"] | None = None
    termination_thickness: float | None = Field(default=None, gt=0)
    component: str | None = None
    component_spec: ComponentGoalSpec | None = None
    required_waypoints: list[Vector3] = Field(default_factory=list)
    waypoint_frame: WaypointFrame | None = None
    waypoint_scale_policy: WaypointScalePolicy | None = None
    terminal_position: Vector3 | None = None
    terminal_axis: Vector3 | None = None
    minimum_curvature_radius: float | None = Field(default=None, gt=0)
    connection_target: ConnectionTarget = "another_open_port"
    notes: str | None = None

    @model_validator(mode="after")
    def validate_goal_contract(self) -> "ProductionGoal":
        if self.goal_id != self.goal_id.strip():
            raise ValueError(
                "goal_id must be non-empty and must not contain surrounding whitespace"
            )
        if len(self.depends_on_goal_ids) != len(set(self.depends_on_goal_ids)):
            raise ValueError("depends_on_goal_ids must be unique")
        if any(
            not dependency.strip() or dependency != dependency.strip()
            for dependency in self.depends_on_goal_ids
        ):
            raise ValueError(
                "depends_on_goal_ids must contain non-empty IDs without surrounding whitespace"
            )
        _validate_goal_field_compatibility(self)
        if self.length is not None and self.length <= 0:
            raise ValueError("goal length must be greater than zero")
        if self.angle is not None and not (0.0 < abs(self.angle) < 360.0):
            raise ValueError("goal angle magnitude must be in (0, 360)")
        if self.branch_count is not None and self.branch_count < 1:
            raise ValueError("branch_count must be at least one")
        if self.diameter_out is not None and self.diameter_out <= 0:
            raise ValueError("diameter_out must be greater than zero")
        for label, outer, wall in (
            (
                "diameter_change",
                self.diameter_out,
                self.wall_thickness_out,
            ),
            (
                "branch",
                self.branch_outer_diameter,
                self.branch_wall_thickness,
            ),
        ):
            if outer is not None and wall is not None and outer <= 2.0 * wall:
                raise ValueError(
                    f"{label} outer diameter must exceed twice wall thickness"
                )
        if self.type == "move" and (self.direction is None or self.length is None):
            raise ValueError("move goal requires direction and length")
        if self.type == "turn" and (
            self.angle is None or (self.direction is None and self.plane_normal is None)
        ):
            raise ValueError(
                "turn goal requires angle plus either a cardinal terminal direction "
                "or a signed plane_normal"
            )
        if self.type == "branch" and not (
            self.branch_count
            or self.required_outlet_directions
            or self.required_outlet_vectors
            or self.required_outlets
        ):
            raise ValueError(
                "branch goal requires an outlet count or outlet directions"
            )
        if self.type == "branch":
            outlet_representations = sum(
                bool(values)
                for values in (
                    self.required_outlet_directions,
                    self.required_outlet_vectors,
                    self.required_outlets,
                )
            )
            if outlet_representations > 1:
                raise ValueError("use exactly one explicit outlet-axis representation")
            explicit_counts = [
                count
                for count in (
                    self.branch_count,
                    len(self.required_outlet_directions) or None,
                    len(self.required_outlet_vectors) or None,
                    len(self.required_outlets) or None,
                )
                if count is not None
            ]
            if len(set(explicit_counts)) > 1:
                raise ValueError(
                    "branch_count and explicit outlet contracts must have matching multiplicity"
                )
            outlet_count = explicit_counts[0] if explicit_counts else 0
            include_primary = (
                self.include_primary_outlet
                if self.include_primary_outlet is not None
                else not bool(self.required_outlet_vectors or self.required_outlets)
            )
            if outlet_count + int(include_primary) != 2:
                raise ValueError(
                    "branch goal must represent exactly one binary split with two total outlets"
                )
            if len(self.required_outlet_directions) != len(
                set(self.required_outlet_directions)
            ):
                raise ValueError("required_outlet_directions must be distinct")
            for label, vectors in (
                ("required_outlet_vectors", self.required_outlet_vectors),
                (
                    "required_outlets",
                    [outlet.axis for outlet in self.required_outlets],
                ),
            ):
                for vector in vectors:
                    if not _finite_vector(vector) or _vector_size(vector) <= 1e-12:
                        raise ValueError(
                            f"{label} axes must be finite non-zero vectors"
                        )
                normalized = [
                    tuple(
                        round(float(component) / _vector_size(vector), 9)
                        for component in vector
                    )
                    for vector in vectors
                ]
                if len(normalized) != len(set(normalized)):
                    raise ValueError(f"{label} axes must be distinct")
                for index, left in enumerate(vectors):
                    left_size = _vector_size(left)
                    for right in vectors[index + 1 :]:
                        score = sum(
                            float(a) * float(b) for a, b in zip(left, right)
                        ) / (left_size * _vector_size(right))
                        if score > 0.999:
                            raise ValueError(
                                f"{label} axes must not be parallel duplicates"
                            )
        if self.type == "diameter_change" and self.diameter_out is None:
            raise ValueError("diameter_change goal requires diameter_out")
        if self.type == "connector" and self.component is None:
            raise ValueError(
                "connector goal requires an explicit supported or unsupported component id"
            )
        if self.type == "end" and self.end_type not in {"cap", "plug"}:
            raise ValueError(
                "production end goals represent physical cap/plug geometry; open terminals belong in expected_open_ports"
            )
        if self.component_spec is not None and (
            self.type != "connector"
            or self.component != self.component_spec.component_type
        ):
            raise ValueError(
                "component_spec requires a connector goal with the same component type"
            )
        for label, value in (
            ("terminal_position", self.terminal_position),
            ("terminal_axis", self.terminal_axis),
        ):
            if value is not None and not _finite_vector(value):
                raise ValueError(f"{label} must be finite")
        if self.terminal_axis is not None and _vector_size(self.terminal_axis) <= 1e-12:
            raise ValueError("terminal_axis must be non-zero")
        if any(not _finite_vector(point) for point in self.required_waypoints):
            raise ValueError("required_waypoints must be finite")
        if self.waypoint_frame is not None and not self.required_waypoints:
            raise ValueError("waypoint_frame requires required_waypoints")
        if self.waypoint_scale_policy is not None and not self.required_waypoints:
            raise ValueError("waypoint_scale_policy requires required_waypoints")
        if self.waypoint_frame == "global" and self.waypoint_scale_policy not in {
            None,
            "fixed",
        }:
            raise ValueError("global waypoints require waypoint_scale_policy=fixed")
        if self.branch_plane_normal is not None and (
            not _finite_vector(self.branch_plane_normal)
            or _vector_size(self.branch_plane_normal) <= 1e-12
        ):
            raise ValueError("branch_plane_normal must be a finite non-zero vector")
        for label, value in (
            ("plane_normal", self.plane_normal),
            ("offset", self.offset),
        ):
            if value is not None and not _finite_vector(value):
                raise ValueError(f"{label} must be finite")
        if self.plane_normal is not None and _vector_size(self.plane_normal) <= 1e-12:
            raise ValueError("plane_normal must be non-zero")
        if self.type == "route" and not (
            self.length is not None
            or self.direction is not None
            or self.required_waypoints
            or self.terminal_position is not None
            or self.terminal_axis is not None
        ):
            raise ValueError(
                "route goal requires at least one measurable path contract"
            )
        return self

    def to_goal(self) -> Goal:
        return Goal.model_validate(self.model_dump())


def _validate_goal_field_compatibility(goal: Any) -> None:
    allowed_by_type = {
        "move": {"direction", "length"},
        "turn": {"direction", "angle", "bend_radius", "plane_normal"},
        "route": {
            "direction",
            "path_kind",
            "length",
            "required_waypoints",
            "waypoint_frame",
            "waypoint_scale_policy",
            "terminal_position",
            "terminal_axis",
            "minimum_curvature_radius",
        },
        "branch": {
            "direction",
            "length",
            "branch_count",
            "branch_angles",
            "branch_plane_normal",
            "required_outlet_directions",
            "required_outlet_vectors",
            "required_outlets",
            "include_primary_outlet",
            "junction_style",
            "blend_radius",
            "inner_blend_radius",
            "max_hub_radius",
            "branch_outer_diameter",
            "branch_wall_thickness",
        },
        "diameter_change": {
            "direction",
            "diameter_out",
            "wall_thickness_out",
            "transition_length",
            "offset",
        },
        "connect": {"required_waypoints", "connection_target"},
        "end": {"end_type", "termination_thickness"},
        "connector": {"direction", "length", "component", "component_spec"},
    }
    all_contract_fields = {
        "direction",
        "path_kind",
        "length",
        "angle",
        "bend_radius",
        "plane_normal",
        "branch_count",
        "branch_angles",
        "branch_plane_normal",
        "required_outlet_directions",
        "required_outlet_vectors",
        "required_outlets",
        "include_primary_outlet",
        "junction_style",
        "blend_radius",
        "inner_blend_radius",
        "max_hub_radius",
        "diameter_out",
        "wall_thickness_out",
        "transition_length",
        "offset",
        "branch_outer_diameter",
        "branch_wall_thickness",
        "end_type",
        "termination_thickness",
        "component",
        "component_spec",
        "required_waypoints",
        "waypoint_frame",
        "waypoint_scale_policy",
        "terminal_position",
        "terminal_axis",
        "minimum_curvature_radius",
        "connection_target",
    }
    allowed = allowed_by_type[goal.type]
    for field_name in all_contract_fields - allowed:
        value = getattr(goal, field_name)
        if field_name == "connection_target" and value == "another_open_port":
            # The backward-compatible default is inert outside connect goals.
            continue
        if value is not None and value != []:
            raise ValueError(f"{field_name} is not valid for goal type {goal.type}")


class ProductionIntent(StrictModel):
    """프로덕션 의도의 순서ㆍ토폴로지ㆍ부품 배수를 엄격히 검증한다."""

    global_spec: ProductionGlobalSpec
    start_position: Vector3
    start_axis: Vector3
    target_behavior: list[ProductionGoal] = Field(min_length=1)
    expected_open_ports: int = Field(ge=0)
    expected_open_ports_source: Literal["explicit", "derived"]
    required_components: list[str]
    hard_constraints: list[str]
    geometric_constraints: list[GeometricConstraint]
    design_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contract(self) -> "ProductionIntent":
        if not _finite_vector(self.start_position):
            raise ValueError("start_position must contain only finite values")
        if (
            not _finite_vector(self.start_axis)
            or _vector_size(self.start_axis) <= 1e-12
        ):
            raise ValueError("start_axis must be non-zero")
        goal_ids = [goal.goal_id for goal in self.target_behavior]
        if len(goal_ids) != len(set(goal_ids)):
            raise ValueError("goal_id values must be unique")
        if self.target_behavior[0].type == "end":
            raise ValueError(
                "a production pipe agenda cannot terminate the anchored START before creating a hollow run"
            )
        start_anchor_goals = [
            goal
            for goal in self.target_behavior
            if goal.type == "connect" and goal.connection_target == "start_anchor"
        ]
        if len(start_anchor_goals) > 1:
            raise ValueError(
                "a production pipe agenda may consume the anchored START seam only once"
            )
        if start_anchor_goals and self.target_behavior[0] is start_anchor_goals[0]:
            raise ValueError(
                "a start_anchor connection requires prior hollow-run geometry"
            )
        seen: set[str] = set()
        for goal in self.target_behavior:
            unknown = set(goal.depends_on_goal_ids) - set(goal_ids)
            if unknown:
                raise ValueError(
                    f"goal {goal.goal_id} has unknown dependencies: {sorted(unknown)}"
                )
            if goal.goal_id in goal.depends_on_goal_ids:
                raise ValueError(f"goal {goal.goal_id} cannot depend on itself")
            if not set(goal.depends_on_goal_ids).issubset(seen):
                raise ValueError(
                    f"goal {goal.goal_id} dependencies must appear earlier in target_behavior"
                )
            seen.add(goal.goal_id)
        open_port_count = 1
        for goal in self.target_behavior:
            if goal.type == "branch":
                branch_count = (
                    goal.branch_count
                    or len(goal.required_outlet_vectors)
                    or len(goal.required_outlet_directions)
                    or len(goal.required_outlets)
                )
                include_primary = (
                    goal.include_primary_outlet
                    if goal.include_primary_outlet is not None
                    else not bool(goal.required_outlet_vectors or goal.required_outlets)
                )
                open_port_count += branch_count + int(include_primary) - 1
            elif goal.type == "connect":
                open_port_count -= 1 if goal.connection_target == "start_anchor" else 2
            elif goal.type == "end":
                open_port_count -= 1
            if open_port_count < 0:
                raise ValueError(
                    f"goal {goal.goal_id} consumes more open terminals than its prefix can produce"
                )
        if open_port_count != self.expected_open_ports:
            raise ValueError(
                "expected_open_ports conflicts with target_behavior topology: "
                f"expected={self.expected_open_ports}, derived={open_port_count}"
            )
        required_components = Counter(self.required_components)
        connector_components = Counter(
            goal.component
            for goal in self.target_behavior
            if goal.type == "connector" and goal.component is not None
        )
        if required_components != connector_components:
            raise ValueError(
                "required_components multiplicity must match connector goals: "
                f"required={dict(required_components)}, "
                f"connector_goals={dict(connector_components)}"
            )
        return self

    def to_intent_result(self) -> IntentResult:
        return IntentResult(
            global_spec=GlobalSpec.model_validate(self.global_spec.model_dump()),
            start_position=self.start_position,
            start_axis=self.start_axis,
            target_behavior=[goal.to_goal() for goal in self.target_behavior],
            expected_open_ports=self.expected_open_ports,
            expected_open_ports_source=self.expected_open_ports_source,
            required_components=self.required_components,
            hard_constraints=self.hard_constraints,
            geometric_constraints=self.geometric_constraints,
            design_notes=self.design_notes,
        )


class _LLMProductionGoalBase(StrictModel):
    """Fields common to every Gemini-authored production goal."""

    goal_id: str
    depends_on_goal_ids: list[str]
    allow_parallel: bool
    notes: str | None = None


class IntentGlobalSpec(StrictModel):
    """JSON-schema-only wire; section relations are domain semantics."""

    outer_diameter: float
    wall_thickness: float
    is_hollow: Literal[True]
    units: Literal["mm"]


class _IntentComponentSpecBase(StrictModel):
    body_outer_diameter: float | None = None
    body_start_offset: float | None = None
    body_length: float | None = None


class IntentFlangeComponentSpec(_IntentComponentSpecBase):
    component_type: Literal["flange"]
    flange_bolt_count: int | None = Field(default=None, ge=3, le=32)
    flange_bolt_circle_diameter: float | None = None
    flange_bolt_hole_diameter: float | None = None
    flange_reference_axis: LLMVector3 | None = None


class IntentCouplingComponentSpec(_IntentComponentSpecBase):
    component_type: Literal["coupling"]


class IntentUnionComponentSpec(_IntentComponentSpecBase):
    component_type: Literal["union"]
    union_ring_outer_diameter: float | None = None
    union_ring_length: float | None = None


class IntentValveComponentSpec(_IntentComponentSpecBase):
    component_type: Literal["valve"]
    actuator_diameter: float | None = None
    actuator_height: float | None = None
    actuator_axis: LLMVector3 | None = None


IntentComponentSpec = Annotated[
    IntentFlangeComponentSpec
    | IntentCouplingComponentSpec
    | IntentUnionComponentSpec
    | IntentValveComponentSpec,
    Field(discriminator="component_type"),
]


class IntentBranchOutletSpec(StrictModel):
    axis: LLMVector3
    length: float | None = None
    outer_diameter: float | None = None
    wall_thickness: float | None = None


class IntentMoveGoal(_LLMProductionGoalBase):
    type: Literal["move"]
    direction: Direction
    length: float


class IntentTurnCardinalContract(StrictModel):
    """A turn whose outlet tangent is one exact global cardinal direction."""

    mode: Literal["cardinal"]
    direction: Direction = Field(
        description="Exact cardinal outlet tangent after the bend.",
    )


class IntentTurnSignedPlaneContract(StrictModel):
    """A non-cardinal turn represented by its independent signed bend plane."""

    mode: Literal["signed_plane"]
    plane_normal: LLMVector3 = Field(
        description=(
            "Bend-plane normal perpendicular to the incoming tangent; the signed "
            "angle and this normal determine the exact outlet tangent."
        ),
    )


IntentTurnOrientationContract = Annotated[
    IntentTurnCardinalContract | IntentTurnSignedPlaneContract,
    Field(discriminator="mode"),
]


class IntentTurnGoal(_LLMProductionGoalBase):
    type: Literal["turn"]
    orientation: IntentTurnOrientationContract
    angle: float = Field(
        description=(
            "Signed right-hand sweep. In signed_plane mode the system derives the "
            "outlet tangent from this angle and orientation.plane_normal."
        )
    )
    bend_radius: float | None = None

    def to_production_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="python", exclude={"orientation"})
        if isinstance(self.orientation, IntentTurnCardinalContract):
            payload["direction"] = self.orientation.direction
        elif isinstance(self.orientation, IntentTurnSignedPlaneContract):
            payload["plane_normal"] = self.orientation.plane_normal
        else:  # pragma: no cover - the discriminated union is closed above.
            raise TypeError(
                "unsupported turn orientation contract: "
                f"{type(self.orientation).__name__}"
            )
        return payload


class IntentRouteLengthContract(StrictModel):
    mode: Literal["length"]
    length: float


class IntentRouteDirectionContract(StrictModel):
    mode: Literal["direction"]
    direction: Direction


class IntentRouteWaypointsContract(StrictModel):
    mode: Literal["waypoints"]
    waypoint_frame: WaypointFrame = Field(
        description=(
            "Use relative_to_target for LLM-invented freeform shape offsets; use "
            "global only when the user explicitly supplied global coordinates."
        )
    )
    waypoint_scale_policy: Literal["fixed"] = Field(
        description=(
            "Waypoints are immutable LLM-authored geometry. Curvature failure "
            "returns a diagnostic and requires a new LLM-authored contract."
        )
    )
    required_waypoints: list[LLMVector3] = Field(min_length=1)


class IntentRouteTerminalPositionContract(StrictModel):
    mode: Literal["terminal_position"]
    terminal_position: LLMVector3


class IntentRouteTerminalAxisContract(StrictModel):
    mode: Literal["terminal_axis"]
    terminal_axis: LLMVector3


IntentRouteGeometryContract = Annotated[
    IntentRouteLengthContract
    | IntentRouteDirectionContract
    | IntentRouteWaypointsContract
    | IntentRouteTerminalPositionContract
    | IntentRouteTerminalAxisContract,
    Field(discriminator="mode"),
]


class IntentRouteGoal(_LLMProductionGoalBase):
    type: Literal["route"]
    path_kind: Literal["line", "circular_arc", "spline"] | None = None
    geometry_contracts: list[IntentRouteGeometryContract] = Field(
        min_length=1,
        max_length=5,
        description=(
            "One or more distinct measurable path contracts. Freeform routes "
            "must include a waypoints contract with LLM-designed shape anchors."
        ),
    )
    minimum_curvature_radius: float | None = None

    def to_production_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="python", exclude={"geometry_contracts"})
        for contract in self.geometry_contracts:
            values = contract.model_dump(mode="python", exclude={"mode"})
            overlap = set(values) & set(payload)
            if overlap:  # pragma: no cover - unique modes make this unreachable.
                raise ValueError(
                    "duplicate route geometry contract fields: "
                    + ", ".join(sorted(overlap))
                )
            payload.update(values)
        return payload


class IntentBranchCountContract(StrictModel):
    mode: Literal["count"]
    branch_count: int = Field(ge=1, le=2)


class IntentBranchDirectionsContract(StrictModel):
    mode: Literal["directions"]
    required_outlet_directions: list[Direction] = Field(min_length=1, max_length=2)


class IntentBranchVectorsContract(StrictModel):
    mode: Literal["vectors"]
    required_outlet_vectors: list[LLMVector3] = Field(min_length=1, max_length=2)


class IntentBranchOutletsContract(StrictModel):
    mode: Literal["outlets"]
    required_outlets: list[IntentBranchOutletSpec] = Field(
        min_length=1,
        max_length=2,
    )


IntentBranchOutletContract = Annotated[
    IntentBranchCountContract
    | IntentBranchDirectionsContract
    | IntentBranchVectorsContract
    | IntentBranchOutletsContract,
    Field(discriminator="mode"),
]


class IntentBranchGoal(_LLMProductionGoalBase):
    type: Literal["branch"]
    direction: Direction | None = None
    length: float | None = None
    branch_angles: list[float] = Field(default_factory=list)
    branch_plane_normal: LLMVector3 | None = None
    outlet_contract: IntentBranchOutletContract
    include_primary_outlet: bool
    junction_style: Literal["hard_fuse", "smooth_hub"] | None = None
    blend_radius: float | None = None
    inner_blend_radius: float | None = None
    max_hub_radius: float | None = None
    branch_outer_diameter: float | None = None
    branch_wall_thickness: float | None = None

    def to_production_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="python", exclude={"outlet_contract"})
        contract = self.outlet_contract
        if isinstance(contract, IntentBranchCountContract):
            payload["branch_count"] = contract.branch_count
        elif isinstance(contract, IntentBranchDirectionsContract):
            payload["required_outlet_directions"] = contract.required_outlet_directions
        elif isinstance(contract, IntentBranchVectorsContract):
            payload["required_outlet_vectors"] = contract.required_outlet_vectors
        elif isinstance(contract, IntentBranchOutletsContract):
            payload["required_outlets"] = [
                outlet.model_dump(mode="python") for outlet in contract.required_outlets
            ]
        else:  # pragma: no cover - the discriminated union is closed above.
            raise TypeError(
                f"unsupported branch outlet contract: {type(contract).__name__}"
            )
        return payload


class IntentDiameterChangeGoal(_LLMProductionGoalBase):
    type: Literal["diameter_change"]
    direction: Direction | None = None
    diameter_out: float
    wall_thickness_out: float | None = None
    transition_length: float
    offset: LLMVector3 | None = None


class IntentConnectGoal(_LLMProductionGoalBase):
    type: Literal["connect"]
    required_waypoints: list[LLMVector3] = Field(default_factory=list)
    connection_target: ConnectionTarget = "another_open_port"


class IntentEndGoal(_LLMProductionGoalBase):
    type: Literal["end"]
    end_type: Literal["cap", "plug"]
    termination_thickness: float | None = None


class IntentConnectorGoal(_LLMProductionGoalBase):
    type: Literal["connector"]
    direction: Direction | None = None
    length: float | None = None
    component: str
    component_spec: IntentComponentSpec | None = None


LLMProductionGoal = Annotated[
    IntentMoveGoal
    | IntentTurnGoal
    | IntentRouteGoal
    | IntentBranchGoal
    | IntentDiameterChangeGoal
    | IntentConnectGoal
    | IntentEndGoal
    | IntentConnectorGoal,
    Field(discriminator="type"),
]


class LLMProductionIntent(ProviderWireModel):
    """Structurally narrow intent schema used only at the Gemini boundary.

    Conversion deliberately re-validates through ``ProductionIntent`` instead
    of duplicating its graph, topology, component, and geometric invariants.
    """

    global_spec: IntentGlobalSpec
    start_position: LLMVector3
    start_axis: LLMVector3
    target_behavior: list[LLMProductionGoal] = Field(min_length=1)
    expected_open_ports: int = Field(ge=0)
    expected_open_ports_source: Literal["explicit", "derived"]
    required_components: list[str]
    hard_constraints: list[str]
    geometric_constraints: list[IntentGeometricConstraint]
    design_notes: list[str] = Field(default_factory=list)

    def to_production_intent(self) -> ProductionIntent:
        payload = self.model_dump(mode="python")
        payload["target_behavior"] = [
            goal.to_production_payload()
            if isinstance(goal, (IntentTurnGoal, IntentRouteGoal, IntentBranchGoal))
            else goal.model_dump(mode="python")
            for goal in self.target_behavior
        ]
        return ProductionIntent.model_validate(payload)

    def to_intent_result(self) -> IntentResult:
        return self.to_production_intent().to_intent_result()


class LLMIntentJSONEnvelope(ProviderWireModel):
    """Minimal provider grammar used only if the full intent schema is rejected."""

    intent_json: str = Field(
        description=(
            "One complete LLMProductionIntent JSON object serialized as a JSON "
            "string. The host strictly parses and validates this inner object."
        )
    )


class Port(StrictModel):
    """위치ㆍ외향 축ㆍ단면ㆍconnector 속성을 가진 타입이 있는 연결점이다."""

    id: str
    position: Vector3
    axis: Vector3
    outer_diameter: float
    wall_thickness: float
    connector_type: str = "plain"
    connector_gender: Literal["neutral", "male", "female"] = "neutral"
    connector_standard: str | None = None

    @model_validator(mode="after")
    def validate_port_geometry(self) -> "Port":
        if not _finite_vector(self.position):
            raise ValueError("port position must contain only finite values")
        if not _finite_vector(self.axis) or _vector_size(self.axis) <= 1e-12:
            raise ValueError("port axis must be a finite non-zero vector")
        if self.outer_diameter <= 0 or self.wall_thickness <= 0:
            raise ValueError("port section dimensions must be positive")
        if self.outer_diameter <= 2.0 * self.wall_thickness:
            raise ValueError("port outer diameter must exceed twice wall thickness")
        return self

    @property
    def inner_diameter(self) -> float:
        return max(0.0, self.outer_diameter - 2.0 * self.wall_thickness)


class ConnectionEdge(StrictModel):
    """두 물리 포트의 결합과 위치ㆍ축ㆍ단면 오차 측정값을 기록한다."""

    edge_id: str
    port_a_id: str
    port_b_id: str
    action_id: str
    position_error: float = 0.0
    anti_parallel_axis_dot: float = 1.0
    axis_angle_error: float = 0.0
    od_error: float = 0.0
    id_error: float = 0.0
    wall_error: float = 0.0
    outer_rim_error: float = 0.0
    inner_rim_error: float = 0.0
    connector_type_match: bool = True
    connector_gender_match: bool = True
    connector_standard_match: bool = True
    engagement: float = 0.0


class ModuleIncidenceEdge(StrictModel):
    module_id: str
    port_id: str


class ModuleRef(StrictModel):
    """배치된 primitive의 파라미터, 생성 형상과 로컬 포트를 참조한다."""

    id: str
    type: str
    schema_version: int = 1
    geometry_id: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    ports: dict[str, Port] = Field(default_factory=dict)
    input_bindings: dict[str, str] = Field(default_factory=dict)


class ActionDraft(StrictModel):
    """LLM 또는 dry-run planner가 제안한 아직 commit되지 않은 행동이다."""

    target_port: str
    module: Literal[
        "straight_pipe",
        "bend_pipe",
        "junction_pipe",
        "reducer_pipe",
        "connector_pipe",
        "cap_pipe",
        "route",
        "transition",
        "junction",
        "connect_ports",
        "terminate",
        "inline_component",
    ]
    params: dict[str, Any] = Field(default_factory=dict)
    consumes_goal_index: int = 0
    catalog_schema_version: int = 1
    affected_goal_ids: list[str] = Field(default_factory=list)
    completed_goal_ids: list[str] = Field(default_factory=list)
    satisfied_components: list[str] = Field(default_factory=list)
    rationale: str | None = None


class JunctionOutlet(StrictModel):
    role: Literal["primary", "branch"]
    axis: Vector3
    length: float = Field(gt=0)
    outer_diameter: float = Field(gt=0)
    wall_thickness: float = Field(gt=0)


def _validate_junction_outlet_roles(outlets: list[JunctionOutlet]) -> None:
    """목표와 무관하게 성립하는 이진 junction 역할 불변식을 검사한다."""

    primary_count = sum(outlet.role == "primary" for outlet in outlets)
    if primary_count > 1:
        raise ValueError("junction outlets may contain at most one primary role")


SectionSource = Literal["inherit_target", "explicit"]


def _validate_section_contract(
    section_source: SectionSource,
    outer_diameter: float | None,
    wall_thickness: float | None,
) -> None:
    if section_source == "inherit_target":
        if outer_diameter is not None or wall_thickness is not None:
            raise ValueError(
                "inherit_target must omit outer_diameter and wall_thickness"
            )
        return
    if outer_diameter is None or wall_thickness is None:
        raise ValueError(
            "explicit section_source requires outer_diameter and wall_thickness"
        )
    if outer_diameter <= wall_thickness * 2.0:
        raise ValueError("outer_diameter must exceed twice wall_thickness")


class RouteParamsV2(StrictModel):
    path_kind: Literal["line", "circular_arc", "spline"]
    section_source: SectionSource
    outer_diameter: float | None = Field(default=None, gt=0)
    wall_thickness: float | None = Field(default=None, gt=0)
    length: float | None = Field(default=None, gt=0)
    direction: Vector3 | None = None
    bend_radius: float | None = Field(default=None, gt=0)
    sweep_angle: float | None = None
    plane_normal: Vector3 | None = None
    terminal_axis: Vector3 | None = None
    waypoint_frame: WaypointFrame | None = None
    waypoints: list[Vector3] = Field(default_factory=list)
    initial_tangent: Vector3 | None = None
    final_tangent: Vector3 | None = None
    interpolation: Literal["bspline"] | None = None
    frenet: bool | None = None
    minimum_curvature_radius: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_variant(self) -> "RouteParamsV2":
        _validate_section_contract(
            self.section_source, self.outer_diameter, self.wall_thickness
        )
        for label, value in (
            ("direction", self.direction),
            ("plane_normal", self.plane_normal),
            ("terminal_axis", self.terminal_axis),
            ("initial_tangent", self.initial_tangent),
            ("final_tangent", self.final_tangent),
        ):
            if value is not None and (
                not _finite_vector(value) or _vector_size(value) <= 1e-12
            ):
                raise ValueError(f"{label} must be a finite non-zero vector")
        if any(not _finite_vector(point) for point in self.waypoints):
            raise ValueError("waypoints must contain only finite vectors")
        if any(
            _vector_size(tuple(b - a for a, b in zip(left, right))) <= 1e-12
            for left, right in zip(self.waypoints, self.waypoints[1:])
        ):
            raise ValueError("waypoints must not contain consecutive duplicates")
        if self.path_kind == "line":
            if self.length is None:
                raise ValueError("line route requires length")
            if (
                self.bend_radius is not None
                or self.sweep_angle is not None
                or self.plane_normal is not None
                or self.terminal_axis is not None
                or self.waypoint_frame is not None
                or self.waypoints
                or self.initial_tangent is not None
                or self.final_tangent is not None
                or self.interpolation is not None
                or self.frenet is not None
                or self.minimum_curvature_radius is not None
            ):
                raise ValueError("line route does not accept curve parameters")
        if self.path_kind == "circular_arc" and any(
            value is None
            for value in (
                self.bend_radius,
                self.sweep_angle,
                self.plane_normal,
            )
        ):
            raise ValueError(
                "circular_arc route requires bend_radius, sweep_angle, and plane_normal"
            )
        if self.path_kind == "circular_arc" and (
            self.length is not None
            or self.direction is not None
            or self.waypoint_frame is not None
            or self.waypoints
            or self.initial_tangent is not None
            or self.final_tangent is not None
            or self.interpolation is not None
            or self.frenet is not None
            or self.minimum_curvature_radius is not None
        ):
            raise ValueError(
                "circular_arc route does not accept line/spline parameters"
            )
        if self.path_kind == "circular_arc" and self.sweep_angle is not None:
            if abs(self.sweep_angle) <= 1e-6 or abs(self.sweep_angle) >= 360.0:
                raise ValueError(
                    "circular_arc sweep_angle magnitude must be in (0, 360)"
                )
        if self.path_kind == "spline" and (
            len(self.waypoints) < 2
            or self.final_tangent is None
            or self.interpolation is None
            or self.frenet is None
            or self.minimum_curvature_radius is None
        ):
            raise ValueError(
                "spline route requires waypoints, final tangent, interpolation, frenet, and minimum curvature"
            )
        if self.path_kind == "spline" and any(
            value is not None
            for value in (
                self.length,
                self.direction,
                self.bend_radius,
                self.sweep_angle,
                self.plane_normal,
                self.terminal_axis,
            )
        ):
            raise ValueError("spline route does not accept line/arc parameters")
        return self


class _PlannerRouteBase(StrictModel):
    """Fields shared by the structurally distinct planner route variants.

    ``RouteParamsV2`` remains the runtime/public compatibility model.  The
    planner uses the narrower models below so its JSON Schema can express the
    path-kind contract before a response reaches the runtime validators.
    """

    section_source: SectionSource
    outer_diameter: float | None = Field(default=None, gt=0)
    wall_thickness: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_section(self) -> "_PlannerRouteBase":
        _validate_section_contract(
            self.section_source, self.outer_diameter, self.wall_thickness
        )
        return self


class RouteLine(_PlannerRouteBase):
    path_kind: Literal["line"]
    length: float = Field(gt=0)
    direction: LLMVector3 | None = Field(
        default=None,
        description=(
            "Omit for a tangent continuation inherited from target_port.axis; "
            "provide only for an immutable explicit route-direction contract."
        ),
    )

    @model_validator(mode="after")
    def validate_direction(self) -> "RouteLine":
        if self.direction is not None and (
            not _finite_vector(self.direction) or _vector_size(self.direction) <= 1e-12
        ):
            raise ValueError("direction must be a finite non-zero vector")
        return self


class RouteArc(_PlannerRouteBase):
    path_kind: Literal["circular_arc"]
    bend_radius: float = Field(
        gt=0,
        description="Independent centerline bend radius in millimeters.",
    )
    sweep_angle: float = Field(
        description=(
            "Signed bend angle in degrees; use right-hand rotation about plane_normal."
        )
    )
    plane_normal: LLMVector3 = Field(
        description=(
            "Bend-plane normal hint, not parallel to the inlet; the resolver "
            "orthogonalizes it and derives the terminal tangent."
        )
    )

    @model_validator(mode="after")
    def validate_arc(self) -> "RouteArc":
        if (
            not _finite_vector(self.plane_normal)
            or _vector_size(self.plane_normal) <= 1e-12
        ):
            raise ValueError("plane_normal must be a finite non-zero vector")
        if abs(self.sweep_angle) <= 1e-6 or abs(self.sweep_angle) >= 360.0:
            raise ValueError("circular_arc sweep_angle magnitude must be in (0, 360)")
        return self


class RouteSpline(_PlannerRouteBase):
    path_kind: Literal["spline"]
    waypoint_frame: WaypointFrame = Field(
        description=(
            "global for explicit global XYZ points; relative_to_target for XYZ "
            "offsets that the resolver translates from the selected port."
        )
    )
    waypoints: list[LLMVector3] = Field(
        min_length=2,
        description=(
            "Points or offsets in waypoint_frame, excluding the current inlet; "
            "the last item is terminal."
        ),
    )

    @model_validator(mode="after")
    def validate_spline(self) -> "RouteSpline":
        if any(not _finite_vector(point) for point in self.waypoints):
            raise ValueError("waypoints must contain only finite vectors")
        if any(
            _vector_size(tuple(b - a for a, b in zip(left, right))) <= 1e-12
            for left, right in zip(self.waypoints, self.waypoints[1:])
        ):
            raise ValueError("waypoints must not contain consecutive duplicates")
        return self


PlannerRouteParamsV2 = Annotated[
    RouteLine | RouteArc | RouteSpline,
    Field(discriminator="path_kind"),
]


class TransitionParamsV2(StrictModel):
    section_source: SectionSource
    outer_diameter: float | None = Field(default=None, gt=0)
    wall_thickness: float | None = Field(default=None, gt=0)
    diameter_out: float = Field(gt=0)
    wall_thickness_out: float | None = Field(default=None, gt=0)
    length: float = Field(gt=0)
    offset: Vector3 | None = None

    @model_validator(mode="after")
    def validate_vectors(self) -> "TransitionParamsV2":
        _validate_section_contract(
            self.section_source, self.outer_diameter, self.wall_thickness
        )
        if self.offset is not None and not _finite_vector(self.offset):
            raise ValueError("offset must contain only finite values")
        return self


class JunctionParamsV2(StrictModel):
    section_source: SectionSource
    outer_diameter: float | None = Field(default=None, gt=0)
    wall_thickness: float | None = Field(default=None, gt=0)
    # Production uses a binary split basis.  Higher-degree manifolds are
    # composed from multiple verified 1->2 transitions instead of one fragile
    # multi-cylinder Boolean at a shared origin.
    outlets: list[JunctionOutlet] = Field(min_length=2, max_length=2)
    blend_mode: Literal["hard", "fillet"]
    blend_radius: float | None = Field(default=None, gt=0)
    inner_blend_radius: float | None = Field(default=None, gt=0)
    max_hub_radius: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_vectors(self) -> "JunctionParamsV2":
        _validate_section_contract(
            self.section_source, self.outer_diameter, self.wall_thickness
        )
        _validate_junction_outlet_roles(self.outlets)
        if self.blend_mode == "fillet" and (
            self.blend_radius is None or self.inner_blend_radius is None
        ):
            raise ValueError("fillet junction requires outer and inner blend radii")
        if self.blend_mode == "hard" and (
            self.blend_radius is not None or self.inner_blend_radius is not None
        ):
            raise ValueError("hard junction must omit unused blend radii")
        for index, outlet in enumerate(self.outlets):
            if not _finite_vector(outlet.axis) or _vector_size(outlet.axis) <= 1e-12:
                raise ValueError(
                    f"outlets[{index}].axis must be a finite non-zero vector"
                )
        return self


class _PlannerJunctionBase(StrictModel):
    section_source: SectionSource
    outer_diameter: float | None = Field(default=None, gt=0)
    wall_thickness: float | None = Field(default=None, gt=0)
    outlets: list[JunctionOutlet] = Field(min_length=2, max_length=2)
    max_hub_radius: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_common_junction(self) -> "_PlannerJunctionBase":
        _validate_section_contract(
            self.section_source, self.outer_diameter, self.wall_thickness
        )
        _validate_junction_outlet_roles(self.outlets)
        for index, outlet in enumerate(self.outlets):
            if not _finite_vector(outlet.axis) or _vector_size(outlet.axis) <= 1e-12:
                raise ValueError(
                    f"outlets[{index}].axis must be a finite non-zero vector"
                )
        return self


class JunctionHard(_PlannerJunctionBase):
    blend_mode: Literal["hard"]


class JunctionFillet(_PlannerJunctionBase):
    blend_mode: Literal["fillet"]
    blend_radius: float = Field(gt=0)
    inner_blend_radius: float = Field(gt=0)


PlannerJunctionParamsV2 = Annotated[
    JunctionHard | JunctionFillet,
    Field(discriminator="blend_mode"),
]


class ConnectPortsParamsV2(StrictModel):
    other_port_id: str
    path_kind: Literal["seam", "line", "circular_arc", "spline"]
    section_source: SectionSource
    outer_diameter: float | None = Field(default=None, gt=0)
    wall_thickness: float | None = Field(default=None, gt=0)
    waypoints: list[Vector3] = Field(default_factory=list)
    initial_tangent: Vector3 | None = None
    final_tangent: Vector3 | None = None
    interpolation: Literal["bspline"] | None = None
    frenet: bool | None = None
    minimum_curvature_radius: float | None = Field(default=None, gt=0)
    bend_radius: float | None = Field(default=None, gt=0)
    sweep_angle: float | None = None
    plane_normal: Vector3 | None = None

    @model_validator(mode="after")
    def validate_variant(self) -> "ConnectPortsParamsV2":
        _validate_section_contract(
            self.section_source, self.outer_diameter, self.wall_thickness
        )
        if any(not _finite_vector(point) for point in self.waypoints):
            raise ValueError("waypoints must contain only finite vectors")
        if self.plane_normal is not None and (
            not _finite_vector(self.plane_normal)
            or _vector_size(self.plane_normal) <= 1e-12
        ):
            raise ValueError("plane_normal must be a finite non-zero vector")
        if any(
            _vector_size(tuple(b - a for a, b in zip(left, right))) <= 1e-12
            for left, right in zip(self.waypoints, self.waypoints[1:])
        ):
            raise ValueError("waypoints must not contain consecutive duplicates")
        if self.path_kind == "seam":
            if self.waypoints or any(
                value is not None
                for value in (
                    self.initial_tangent,
                    self.final_tangent,
                    self.interpolation,
                    self.frenet,
                    self.minimum_curvature_radius,
                    self.bend_radius,
                    self.sweep_angle,
                    self.plane_normal,
                )
            ):
                raise ValueError(
                    "seam connect_ports represents coincident compatible ports and accepts no path geometry"
                )
            return self
        if self.path_kind == "line":
            if self.waypoints:
                raise ValueError("line connect_ports does not accept waypoints")
            if any(
                value is not None
                for value in (
                    self.initial_tangent,
                    self.final_tangent,
                    self.interpolation,
                    self.frenet,
                    self.minimum_curvature_radius,
                    self.bend_radius,
                    self.sweep_angle,
                    self.plane_normal,
                )
            ):
                raise ValueError(
                    "line connect_ports derives its chord tangent and does not accept curve parameters"
                )
            return self
        if self.path_kind == "circular_arc":
            if len(self.waypoints) != 1:
                raise ValueError(
                    "circular_arc connect_ports requires exactly one arc waypoint"
                )
            if any(
                value is not None
                for value in (
                    self.initial_tangent,
                    self.final_tangent,
                    self.interpolation,
                    self.frenet,
                )
            ):
                raise ValueError(
                    "circular_arc connect_ports derives its tangents and does not accept spline parameters"
                )
            if self.minimum_curvature_radius is None:
                raise ValueError(
                    "circular_arc connect_ports requires minimum_curvature_radius"
                )
            return self
        if any(
            value is not None
            for value in (self.bend_radius, self.sweep_angle, self.plane_normal)
        ):
            raise ValueError(
                "spline connect_ports does not accept circular-arc parameters"
            )
        if (
            self.interpolation != "bspline"
            or self.frenet is None
            or self.minimum_curvature_radius is None
        ):
            raise ValueError(
                "curved connect_ports requires interpolation, frenet, and minimum_curvature_radius"
            )
        if not self.waypoints:
            raise ValueError("curved connect_ports requires at least one waypoint")
        return self


class _PlannerConnectPortsBase(StrictModel):
    other_port_id: str
    section_source: SectionSource
    outer_diameter: float | None = Field(default=None, gt=0)
    wall_thickness: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_section(self) -> "_PlannerConnectPortsBase":
        _validate_section_contract(
            self.section_source, self.outer_diameter, self.wall_thickness
        )
        return self


class ConnectPortsLine(_PlannerConnectPortsBase):
    path_kind: Literal["line"]


class ConnectPortsArc(_PlannerConnectPortsBase):
    path_kind: Literal["circular_arc"]
    waypoints: list[LLMVector3] = Field(min_length=1, max_length=1)

    @model_validator(mode="after")
    def validate_arc_waypoint(self) -> "ConnectPortsArc":
        if any(not _finite_vector(point) for point in self.waypoints):
            raise ValueError("waypoints must contain only finite vectors")
        return self


class ConnectPortsSpline(_PlannerConnectPortsBase):
    path_kind: Literal["spline"]
    waypoints: list[LLMVector3] = Field(
        min_length=1,
        description=(
            "Interior global XYZ points only; both endpoint tangents are derived from the ports."
        ),
    )

    @model_validator(mode="after")
    def validate_spline(self) -> "ConnectPortsSpline":
        if any(not _finite_vector(point) for point in self.waypoints):
            raise ValueError("waypoints must contain only finite vectors")
        if any(
            _vector_size(tuple(b - a for a, b in zip(left, right))) <= 1e-12
            for left, right in zip(self.waypoints, self.waypoints[1:])
        ):
            raise ValueError("waypoints must not contain consecutive duplicates")
        return self


PlannerConnectPortsParamsV2 = Annotated[
    ConnectPortsLine | ConnectPortsArc | ConnectPortsSpline,
    Field(discriminator="path_kind"),
]


def _finite_vector(value: Vector3) -> bool:
    return all(math.isfinite(float(component)) for component in value)


def _vector_size(value: Vector3) -> float:
    return math.sqrt(sum(float(component) ** 2 for component in value))


class TerminateParamsV2(StrictModel):
    section_source: SectionSource
    outer_diameter: float | None = Field(default=None, gt=0)
    wall_thickness: float | None = Field(default=None, gt=0)
    termination_type: Literal["cap", "plug"]
    thickness: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_section(self) -> "TerminateParamsV2":
        _validate_section_contract(
            self.section_source, self.outer_diameter, self.wall_thickness
        )
        return self


class InlineComponentParamsV2(StrictModel):
    section_source: SectionSource
    outer_diameter: float | None = Field(default=None, gt=0)
    wall_thickness: float | None = Field(default=None, gt=0)
    component_type: InlineComponentKind
    length: float = Field(gt=0)
    body_outer_diameter: float = Field(gt=0)
    body_start_offset: float = Field(ge=0)
    body_length: float = Field(gt=0)
    flange_bolt_count: int | None = Field(default=None, ge=3, le=32)
    flange_bolt_circle_diameter: float | None = Field(default=None, gt=0)
    flange_bolt_hole_diameter: float | None = Field(default=None, gt=0)
    flange_reference_axis: Vector3 | None = None
    union_ring_outer_diameter: float | None = Field(default=None, gt=0)
    union_ring_length: float | None = Field(default=None, gt=0)
    actuator_diameter: float | None = Field(default=None, gt=0)
    actuator_height: float | None = Field(default=None, gt=0)
    actuator_axis: Vector3 | None = None
    connector_type_out: str
    connector_gender_out: Literal["neutral", "male", "female"]
    connector_standard_out: str | None

    @model_validator(mode="after")
    def validate_component(self) -> "InlineComponentParamsV2":
        _validate_section_contract(
            self.section_source, self.outer_diameter, self.wall_thickness
        )
        body_end = self.body_start_offset + self.body_length
        if body_end > self.length + 1e-9:
            raise ValueError(
                "component body must remain inside the authored axial length"
            )
        if (
            self.outer_diameter is not None
            and self.body_outer_diameter <= self.outer_diameter
        ):
            raise ValueError("body_outer_diameter must exceed the mating pipe diameter")
        actuator_values = (
            self.actuator_diameter,
            self.actuator_height,
            self.actuator_axis,
        )
        if self.component_type == "valve" and any(
            value is None for value in actuator_values
        ):
            raise ValueError("valve requires actuator dimensions and actuator_axis")
        if self.component_type != "valve" and any(
            value is not None for value in actuator_values
        ):
            raise ValueError("actuator parameters are valid only for valve")
        flange_values = (
            self.flange_bolt_count,
            self.flange_bolt_circle_diameter,
            self.flange_bolt_hole_diameter,
            self.flange_reference_axis,
        )
        if self.component_type == "flange" and any(
            value is None for value in flange_values
        ):
            raise ValueError(
                "flange requires authored bolt count, circle, and hole diameter"
            )
        if self.component_type != "flange" and any(
            value is not None for value in flange_values
        ):
            raise ValueError("flange bolt parameters are valid only for flange")
        if self.flange_reference_axis is not None and (
            not _finite_vector(self.flange_reference_axis)
            or _vector_size(self.flange_reference_axis) <= 1e-12
        ):
            raise ValueError("flange_reference_axis must be a finite non-zero vector")
        union_values = (self.union_ring_outer_diameter, self.union_ring_length)
        if self.component_type == "union" and any(
            value is None for value in union_values
        ):
            raise ValueError("union requires authored ring diameter and length")
        if self.component_type != "union" and any(
            value is not None for value in union_values
        ):
            raise ValueError("union ring parameters are valid only for union")
        if self.actuator_axis is not None and (
            not _finite_vector(self.actuator_axis)
            or _vector_size(self.actuator_axis) <= 1e-12
        ):
            raise ValueError("actuator_axis must be a finite non-zero vector")
        if self.component_type == "flange" and not (
            self.body_start_offset <= 1e-9 or abs(body_end - self.length) <= 1e-9
        ):
            raise ValueError("flange collar must touch one authored axial end")
        if self.component_type == "coupling" and not (
            self.body_start_offset <= 1e-9
            and abs(self.body_length - self.length) <= 1e-9
        ):
            raise ValueError("coupling sleeve must span the authored axial length")
        if self.component_type in {"union", "valve"} and not (
            self.body_start_offset > 1e-9 and body_end < self.length - 1e-9
        ):
            raise ValueError(
                f"{self.component_type} body must lie between two pipe necks"
            )
        return self


PlannerParamsV2 = (
    PlannerRouteParamsV2
    | TransitionParamsV2
    | PlannerJunctionParamsV2
    | PlannerConnectPortsParamsV2
    | TerminateParamsV2
    | InlineComponentParamsV2
)


class RouteChoice(StrictModel):
    module: Literal["route"]
    params: PlannerRouteParamsV2


class TransitionChoice(StrictModel):
    module: Literal["transition"]
    params: TransitionParamsV2


class JunctionChoice(StrictModel):
    module: Literal["junction"]
    params: PlannerJunctionParamsV2


class ConnectPortsChoice(StrictModel):
    module: Literal["connect_ports"]
    params: PlannerConnectPortsParamsV2


class TerminateChoice(StrictModel):
    module: Literal["terminate"]
    params: TerminateParamsV2


class InlineComponentChoice(StrictModel):
    module: Literal["inline_component"]
    params: InlineComponentParamsV2


PlannerChoice = Annotated[
    RouteChoice
    | TransitionChoice
    | JunctionChoice
    | ConnectPortsChoice
    | TerminateChoice
    | InlineComponentChoice,
    Field(discriminator="module"),
]
CorePlannerChoice = Annotated[
    RouteChoice
    | TransitionChoice
    | JunctionChoice
    | ConnectPortsChoice
    | TerminateChoice,
    Field(discriminator="module"),
]


class PlannerDecisionBase(StrictModel):
    """schema-v2 primitive 선택과 목표 진행 주장을 공통 형식으로 묶는다."""

    catalog_schema_version: Literal[2]
    target_port: str
    affected_goal_ids: list[str] = Field(min_length=1)
    completed_goal_ids: list[str]
    rationale: str | None = None

    @property
    def module(self) -> str:
        return self.choice.module  # type: ignore[attr-defined,no-any-return]

    @property
    def params(self) -> PlannerParamsV2:
        return self.choice.params  # type: ignore[attr-defined,no-any-return]

    def to_action_draft(self) -> ActionDraft:
        params = self.params.model_dump(mode="json", exclude_none=True)
        if isinstance(self.params, InlineComponentParamsV2):
            params["connector_standard_out"] = self.params.connector_standard_out
        elif isinstance(self.params, _PlannerJunctionBase):
            params["outlets"] = [
                outlet.model_dump(mode="json", exclude_none=True)
                for outlet in self.params.outlets
            ]
        return ActionDraft(
            target_port=self.target_port,
            module=self.module,
            params=params,
            catalog_schema_version=self.catalog_schema_version,
            affected_goal_ids=self.affected_goal_ids,
            completed_goal_ids=self.completed_goal_ids,
            rationale=self.rationale,
        )


class CorePlannerDecision(PlannerDecisionBase):
    """부품이 필요하지 않은 상태에서 다섯 core primitive 중 하나를 선택한다."""

    choice: CorePlannerChoice


class PlannerDecision(PlannerDecisionBase):
    """필요할 때 inline component까지 포함하는 전체 primitive 선택 응답이다."""

    choice: PlannerChoice


class _PlannerWireSection(StrictModel):
    """Planner wire always inherits the selected target-port section."""

    section_source: Literal["inherit_target"]


class PlannerRouteLineWire(_PlannerWireSection):
    path_kind: Literal["line"]
    length: float
    direction: LLMVector3 | None = None


class PlannerRouteArcWire(_PlannerWireSection):
    path_kind: Literal["circular_arc"]
    bend_radius: float
    sweep_angle: float
    plane_normal: LLMVector3


class PlannerRouteSplineWire(_PlannerWireSection):
    path_kind: Literal["spline"]
    waypoint_frame: WaypointFrame
    waypoints: list[LLMVector3] = Field(min_length=2)


PlannerRouteWire = Annotated[
    PlannerRouteLineWire | PlannerRouteArcWire | PlannerRouteSplineWire,
    Field(discriminator="path_kind"),
]


class PlannerTransitionWire(_PlannerWireSection):
    diameter_out: float
    wall_thickness_out: float | None = None
    length: float
    offset: LLMVector3 | None = None


class PlannerJunctionOutletWire(StrictModel):
    role: Literal["primary", "branch"]
    axis: LLMVector3
    length: float
    outer_diameter: float
    wall_thickness: float


class _PlannerJunctionWireBase(_PlannerWireSection):
    outlets: list[PlannerJunctionOutletWire] = Field(min_length=2, max_length=2)
    max_hub_radius: float


class PlannerJunctionHardWire(_PlannerJunctionWireBase):
    blend_mode: Literal["hard"]


class PlannerJunctionFilletWire(_PlannerJunctionWireBase):
    blend_mode: Literal["fillet"]
    blend_radius: float
    inner_blend_radius: float


PlannerJunctionWire = Annotated[
    PlannerJunctionHardWire | PlannerJunctionFilletWire,
    Field(discriminator="blend_mode"),
]


class PlannerConnectLineWire(_PlannerWireSection):
    path_kind: Literal["line"]
    other_port_id: str


class PlannerConnectArcWire(_PlannerWireSection):
    path_kind: Literal["circular_arc"]
    other_port_id: str
    waypoints: list[LLMVector3] = Field(min_length=1, max_length=1)


class PlannerConnectSplineWire(_PlannerWireSection):
    path_kind: Literal["spline"]
    other_port_id: str
    waypoints: list[LLMVector3] = Field(min_length=1)


PlannerConnectWire = Annotated[
    PlannerConnectLineWire | PlannerConnectArcWire | PlannerConnectSplineWire,
    Field(discriminator="path_kind"),
]


class PlannerTerminateWire(_PlannerWireSection):
    termination_type: Literal["cap", "plug"]
    thickness: float


class _PlannerInlineWireBase(_PlannerWireSection):
    length: float
    body_outer_diameter: float
    body_start_offset: float
    body_length: float
    connector_type_out: str
    connector_gender_out: Literal["neutral", "male", "female"]
    connector_standard_out: str | None


class PlannerFlangeWire(_PlannerInlineWireBase):
    component_type: Literal["flange"]
    flange_bolt_count: int = Field(ge=3, le=32)
    flange_bolt_circle_diameter: float
    flange_bolt_hole_diameter: float
    flange_reference_axis: LLMVector3


class PlannerCouplingWire(_PlannerInlineWireBase):
    component_type: Literal["coupling"]


class PlannerUnionWire(_PlannerInlineWireBase):
    component_type: Literal["union"]
    union_ring_outer_diameter: float
    union_ring_length: float


class PlannerValveWire(_PlannerInlineWireBase):
    component_type: Literal["valve"]
    actuator_diameter: float
    actuator_height: float
    actuator_axis: LLMVector3


PlannerInlineWire = Annotated[
    PlannerFlangeWire | PlannerCouplingWire | PlannerUnionWire | PlannerValveWire,
    Field(discriminator="component_type"),
]


class PlannerRouteChoiceWire(StrictModel):
    module: Literal["route"]
    params: PlannerRouteWire


class PlannerTransitionChoiceWire(StrictModel):
    module: Literal["transition"]
    params: PlannerTransitionWire


class PlannerJunctionChoiceWire(StrictModel):
    module: Literal["junction"]
    params: PlannerJunctionWire


class PlannerConnectChoiceWire(StrictModel):
    module: Literal["connect_ports"]
    params: PlannerConnectWire


class PlannerTerminateChoiceWire(StrictModel):
    module: Literal["terminate"]
    params: PlannerTerminateWire


class PlannerInlineChoiceWire(StrictModel):
    module: Literal["inline_component"]
    params: PlannerInlineWire


PlannerChoiceWire = Annotated[
    PlannerRouteChoiceWire
    | PlannerTransitionChoiceWire
    | PlannerJunctionChoiceWire
    | PlannerConnectChoiceWire
    | PlannerTerminateChoiceWire
    | PlannerInlineChoiceWire,
    Field(discriminator="module"),
]
CorePlannerChoiceWire = Annotated[
    PlannerRouteChoiceWire
    | PlannerTransitionChoiceWire
    | PlannerJunctionChoiceWire
    | PlannerConnectChoiceWire
    | PlannerTerminateChoiceWire,
    Field(discriminator="module"),
]


class PlannerDecisionWireBase(ProviderWireModel):
    catalog_schema_version: Literal[2]
    target_port: str
    affected_goal_ids: list[str] = Field(min_length=1)
    completed_goal_ids: list[str]
    rationale: str | None = None

    def to_action_draft(self) -> ActionDraft:
        params = self.choice.params.model_dump(  # type: ignore[attr-defined]
            mode="json",
            exclude_none=True,
        )
        if isinstance(
            self.choice.params,  # type: ignore[attr-defined]
            _PlannerInlineWireBase,
        ):
            params["connector_standard_out"] = self.choice.params.connector_standard_out  # type: ignore[attr-defined]
        return ActionDraft(
            target_port=self.target_port,
            module=self.choice.module,  # type: ignore[attr-defined]
            params=params,
            catalog_schema_version=self.catalog_schema_version,
            affected_goal_ids=self.affected_goal_ids,
            completed_goal_ids=self.completed_goal_ids,
            rationale=self.rationale,
        )


class CorePlannerDecisionWire(PlannerDecisionWireBase):
    choice: CorePlannerChoiceWire


class PlannerDecisionWire(PlannerDecisionWireBase):
    choice: PlannerChoiceWire


class AgendaRepairDirective(StrictModel):
    """최종 오류를 고칠 rollback 단계와 국소 repair 범위를 지정한다."""

    scope: Literal["agenda"]
    rollback_step: int = Field(ge=1)
    target_issue_ids: list[str] = Field(min_length=1, max_length=8)
    target_module_ids: list[str] = Field(default_factory=list, max_length=8)
    repair_hint: str = Field(min_length=1, max_length=800)
    rationale: str

    @model_validator(mode="after")
    def validate_localization(self) -> "AgendaRepairDirective":
        for label, values in (
            ("target_issue_ids", self.target_issue_ids),
            ("target_module_ids", self.target_module_ids),
        ):
            if len(values) != len(set(values)) or any(
                not value.strip() or value != value.strip() for value in values
            ):
                raise ValueError(f"{label} must contain unique, non-empty, trimmed IDs")
        return self


class AgendaRepairDirectiveWire(ProviderWireModel):
    """Provider-only shape; localization semantics are bound by the host."""

    scope: Literal["agenda"]
    rollback_step: int = Field(ge=1)
    target_issue_ids: list[str] = Field(min_length=1, max_length=8)
    target_module_ids: list[str] = Field(default_factory=list, max_length=8)
    repair_hint: str
    rationale: str


class VisualCriticIssue(StrictModel):
    issue_code: str
    module_ids: list[str] = Field(min_length=1)
    observation: str
    target_step: int | None = None


class VisualCriticResult(StrictModel):
    """digest 결합 화면에 대한 구조화된 통과 여부와 시각 이슈를 담는다."""

    state_id: str
    payload_digest: str
    evidence_sha256: list[str]
    passed: bool
    issues: list[VisualCriticIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "VisualCriticResult":
        if self.passed and self.issues:
            raise ValueError("a passing visual result must not contain issues")
        if not self.passed and not self.issues:
            raise ValueError("a failing visual result must localize at least one issue")
        return self


class VisualCriticResultWire(ProviderWireModel):
    """Provider-only visual result before passed/issues semantic binding."""

    state_id: str
    payload_digest: str
    evidence_sha256: list[str]
    passed: bool
    issues: list[VisualCriticIssue] = Field(default_factory=list)


class ResolvedAction(StrictModel):
    """대상 상태에 맞춰 모든 resolver 소유 파라미터가 계산된 행동이다."""

    action_id: str
    action_type: Literal["ADD_MODULE"] = "ADD_MODULE"
    target_port: str
    module: str
    params: dict[str, Any] = Field(default_factory=dict)
    consumed_port_ids: list[str] = Field(default_factory=list)
    affected_goal_ids: list[str] = Field(default_factory=list)
    completed_goal_ids: list[str] = Field(default_factory=list)
    satisfied_components: list[str] = Field(default_factory=list)


class ValidationDiagnostic(StrictModel):
    """검증 실패의 수치 계약과 기하적 발생 위치를 기계 판독 가능하게 담는다."""

    code: str
    check_name: str
    evaluator_id: str
    evaluator_version: str
    calculation_method: str
    metric: str
    comparator: Literal[">=", ">", "<=", "<", "=="]
    required: float
    actual: float
    gap: float = Field(ge=0)
    ratio: float | None = Field(default=None, ge=0)
    units: str | None = None
    modeling_tolerance: float = Field(gt=0)
    critical_span_index: int | None = Field(default=None, ge=0)
    critical_t: float | None = Field(default=None, ge=0, le=1)
    critical_span_endpoints: tuple[Vector3, Vector3] | None = None
    critical_location: Vector3 | None = None
    handle_factors: list[float] = Field(default_factory=list)
    curve_length: float | None = Field(default=None, ge=0)
    polyline_length: float | None = Field(default=None, ge=0)
    minimum_chord: float | None = Field(default=None, gt=0)
    implicated_parameter_paths: list[str] = Field(default_factory=list)


class ValidationResult(StrictModel):
    """registry 검사의 통과 여부와 구체적인 오류 목록을 담는다."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    diagnostics: list[ValidationDiagnostic] = Field(default_factory=list)


class PipeState(StrictModel):
    """한 시점의 모듈ㆍ포트 그래프, 남은 목표와 행동 이력을 모두 담는다."""

    state_id: str
    state_version: int = 0
    contract_digest: str | None = None
    modeling_tolerance: float = Field(default=1e-4, gt=0)
    global_spec: GlobalSpec
    expected_open_ports: int | None = None
    expected_open_ports_source: OpenPortExpectationSource = "unknown"
    required_components: list[str] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)
    geometric_constraints: list[GeometricConstraint] = Field(default_factory=list)
    design_notes: list[str] = Field(default_factory=list)
    module_measurements: dict[str, dict[str, float]] = Field(default_factory=dict)
    placed_modules: list[ModuleRef] = Field(default_factory=list)
    open_ports: list[Port] = Field(default_factory=list)
    reserved_start_anchor: Port | None = None
    port_nodes: dict[str, Port] = Field(default_factory=dict)
    connection_edges: list[ConnectionEdge] = Field(default_factory=list)
    module_incidence_edges: list[ModuleIncidenceEdge] = Field(default_factory=list)
    open_port_ids: list[str] = Field(default_factory=list)
    used_ports: list[str] = Field(default_factory=list)
    remaining_goals: list[Goal] = Field(default_factory=list)
    action_history: list[ResolvedAction] = Field(default_factory=list)


class StaticIssue(StrictModel):
    """검증 위치, 기대/실제 증거와 수정 제안을 가진 단일 관측이다."""

    issue_id: str
    severity: IssueSeverity
    issue_code: str
    check_name: str
    message: str
    step_index: int | None = None
    action_id: str | None = None
    module_id: str | None = None
    port_ids: list[str] = Field(default_factory=list)
    target_port_id: str | None = None
    consumed_goal_index: int | None = None
    expected: dict[str, Any] = Field(default_factory=dict)
    actual: dict[str, Any] = Field(default_factory=dict)
    suggestion: dict[str, Any] = Field(default_factory=dict)


class StateTransition(StrictModel):
    """한 행동 전후의 모듈ㆍ포트ㆍedgeㆍ목표 차이를 감사 가능하게 기록한다."""

    step_index: int
    consumed_goal_index: int
    consumed_goal: dict[str, Any] | None = None
    affected_goals: list[dict[str, Any]] = Field(default_factory=list)
    state_before_id: str
    state_after_id: str
    action_id: str
    module: str
    target_port_before: dict[str, Any]
    produced_module_id: str | None = None
    produced_port_ids: list[str] = Field(default_factory=list)
    removed_port_ids: list[str] = Field(default_factory=list)
    consumed_port_ids: list[str] = Field(default_factory=list)
    connection_edge_ids: list[str] = Field(default_factory=list)
    affected_goal_ids: list[str] = Field(default_factory=list)
    completed_goal_ids: list[str] = Field(default_factory=list)
    satisfied_components: list[str] = Field(default_factory=list)
    open_port_ids_before: list[str] = Field(default_factory=list)
    open_port_ids_after: list[str] = Field(default_factory=list)


class AssemblyBounds(StrictModel):
    """FreeCAD가 측정한 전체 조립체의 최소ㆍ최대 XYZ 경계다."""

    minimum: Vector3
    maximum: Vector3

    @model_validator(mode="after")
    def validate_bounds(self) -> "AssemblyBounds":
        if not _finite_vector(self.minimum) or not _finite_vector(self.maximum):
            raise ValueError("assembly bounds must contain only finite values")
        if any(low > high for low, high in zip(self.minimum, self.maximum)):
            raise ValueError("assembly bounds minimum must not exceed maximum")
        return self


class StepVerification(StrictModel):
    """상태 전이의 정적 이슈와 선택적 FreeCAD 실측 상태를 묶는다."""

    transition: StateTransition
    status: Literal["passed", "failed"]
    issues: list[StaticIssue] = Field(default_factory=list)
    mcp_status: MCPStatus = "skipped"
    mcp_required: bool = False
    mcp_result_path: str | None = None
    mcp_error: str | None = None
    skipped_mcp_reason: str | None = None
    freecad_validation_path: str | None = None
    mcp_measurements: dict[str, dict[str, float]] = Field(default_factory=dict)
    mcp_assembly_bounds: AssemblyBounds | None = None


class ActionAttempt(StrictModel):
    """수락 또는 거절된 행동 시도와 다음 repair용 관측을 기록한다."""

    step_index: int
    attempt_index: int
    state_id: str
    state_digest: str | None = None
    phase: str
    status: Literal["rejected", "accepted"]
    draft: dict[str, Any] | None = None
    resolved: dict[str, Any] | None = None
    issue_codes: list[str] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)


class IntentRepairAdvice(StrictModel):
    """검증된 intent 후보를 다시 작성하게 하는 비실행형 진단 지시다.

    Advisor는 intent를 직접 만들거나 승인하지 않는다. 이 객체는 현재 후보에서
    보존할 요구와 다시 작성할 필드만 제한하며, 수정 후보는 동일한 결정론적
    intent 검증을 처음부터 다시 통과해야 한다.
    """

    diagnosis_class: Literal[
        "candidate_contract_error",
        "candidate_topology_error",
        "unsupported_user_requirement",
        "validator_policy_mismatch",
        "insufficient_evidence",
    ]
    disposition: Literal[
        "retry_intent",
        "stop_contract_infeasible",
        "escalate_validator_review",
        "stop_futile_retry",
    ]
    candidate_fixable: bool
    summary: str
    causal_chain: list[str] = Field(min_length=1, max_length=6)
    preserve_requirements: list[str] = Field(default_factory=list, max_length=12)
    change_fields: list[str] = Field(default_factory=list, max_length=12)
    avoid: list[str] = Field(default_factory=list, max_length=12)
    intent_instruction: str

    @model_validator(mode="after")
    def validate_advice(self) -> "IntentRepairAdvice":
        for label, value in (
            ("summary", self.summary),
            ("intent_instruction", self.intent_instruction),
        ):
            if not value.strip():
                raise ValueError(f"intent repair advice {label} must not be blank")
        for label, values in (
            ("causal_chain", self.causal_chain),
            ("preserve_requirements", self.preserve_requirements),
            ("change_fields", self.change_fields),
            ("avoid", self.avoid),
        ):
            if any(not str(value).strip() for value in values):
                raise ValueError(
                    f"intent repair advice {label} must not contain blank values"
                )
            if len(values) != len(set(values)):
                raise ValueError(
                    f"intent repair advice {label} must contain unique values"
                )

        if self.disposition == "retry_intent":
            if not self.candidate_fixable:
                raise ValueError("retry_intent requires candidate_fixable=true")
            if not self.change_fields:
                raise ValueError("retry_intent requires at least one change field")
        elif self.disposition == "stop_contract_infeasible":
            if self.diagnosis_class != "unsupported_user_requirement":
                raise ValueError(
                    "stop_contract_infeasible requires unsupported_user_requirement"
                )
            if self.candidate_fixable:
                raise ValueError(
                    "stop_contract_infeasible requires candidate_fixable=false"
                )
        elif self.candidate_fixable:
            raise ValueError(f"{self.disposition} requires candidate_fixable=false")
        return self


class IntentRepairAdviceWire(ProviderWireModel):
    """Provider-safe intent diagnosis body before host semantic validation."""

    diagnosis_class: Literal[
        "candidate_contract_error",
        "candidate_topology_error",
        "unsupported_user_requirement",
        "validator_policy_mismatch",
        "insufficient_evidence",
    ]
    disposition: Literal[
        "retry_intent",
        "stop_contract_infeasible",
        "escalate_validator_review",
        "stop_futile_retry",
    ]
    candidate_fixable: bool
    summary: str
    causal_chain: list[str] = Field(min_length=1, max_length=6)
    preserve_requirements: list[str] = Field(default_factory=list, max_length=12)
    change_fields: list[str] = Field(default_factory=list, max_length=12)
    avoid: list[str] = Field(default_factory=list, max_length=12)
    intent_instruction: str


class StepRepairAdvice(StrictModel):
    """독립 진단 모델이 planner에 넘기는 비실행형 교정 지시다.

    이 객체는 CAD 상태나 action을 직접 변경할 권한이 없다. 실제 후보는
    기존 step planner schema를 다시 통과해야 한다.
    """

    diagnosis_class: Literal[
        "planner_parameter",
        "planner_topology",
        "contract_infeasible",
        "validator_or_kernel",
        "infrastructure",
        "unknown",
    ]
    candidate_fixable: bool
    diagnosis: str
    preserve: list[str] = Field(default_factory=list)
    change: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    verification_target: str
    planner_instruction: str

    @model_validator(mode="after")
    def validate_advice(self) -> "StepRepairAdvice":
        text_fields = (
            self.diagnosis,
            self.verification_target,
            self.planner_instruction,
        )
        if any(not value.strip() for value in text_fields):
            raise ValueError("repair advice text fields must not be blank")
        for name, values in (
            ("preserve", self.preserve),
            ("change", self.change),
            ("avoid", self.avoid),
        ):
            if len(values) > 12 or any(not str(value).strip() for value in values):
                raise ValueError(f"repair advice {name} must contain 0..12 labels")
        return self


class StepRepairAdviceWire(ProviderWireModel):
    diagnosis_class: Literal[
        "planner_parameter",
        "planner_topology",
        "contract_infeasible",
        "validator_or_kernel",
        "infrastructure",
        "unknown",
    ]
    candidate_fixable: bool
    diagnosis: str
    preserve: list[str] = Field(default_factory=list)
    change: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    verification_target: str
    planner_instruction: str


_SHA256_PATTERN = r"^[0-9a-f]{64}$"


def _diagnostic_nonblank(value: str, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} must not be blank")
    return value


def _diagnostic_unique_nonblank(values: list[str], label: str) -> list[str]:
    if any(not str(value).strip() for value in values):
        raise ValueError(f"{label} must not contain blank values")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must contain unique values")
    return values


def _validate_diagnostic_pointer(path: str, label: str) -> str:
    """Validate the RFC 6901 shape used by diagnostic field ownership.

    Ownership cards may use ``*`` for an array slot, but otherwise paths are
    ordinary JSON pointers.  Empty path segments are rejected because they are
    ambiguous in planner-facing repair directives.
    """

    if not path.startswith("/") or path == "/":
        raise ValueError(f"{label} must be a non-root JSON pointer")
    for token in path[1:].split("/"):
        if not token:
            raise ValueError(f"{label} must not contain empty path segments")
        index = 0
        while index < len(token):
            if token[index] == "~":
                if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
                    raise ValueError(f"{label} contains an invalid JSON pointer escape")
                index += 2
                continue
            index += 1
    return path


def _validate_diagnostic_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must contain only finite numbers")
    if isinstance(value, dict):
        for child in value.values():
            _validate_diagnostic_finite_tree(child, label)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_diagnostic_finite_tree(child, label)


DiagnosticFactKind = Literal[
    "measurement",
    "relationship",
    "validator_policy",
    "parameter_effect",
    "attempt_delta",
    "immutable_contract",
    "passed_check",
    "kernel_error",
]
DiagnosticOwner = Literal[
    "user_immutable",
    "goal_derived_immutable",
    "planner_authored",
    "resolver_owned",
    "validator_policy",
    "downstream_state_sensitive",
]
RepairStrategyKind = Literal[
    "parameter_change",
    "mode_change",
    "primitive_change",
    "topology_change",
    "rollback_earlier_step",
    "evidence_probe",
    "validator_review",
    "kernel_review",
    "infrastructure_retry",
    "stop_futile_retry",
]


class DiagnosticBinding(StrictModel):
    """Immutable provenance binding for one rejected transition candidate."""

    protocol_version: Literal[1] = 1
    run_id: str
    state_id: str
    state_digest: str = Field(pattern=_SHA256_PATTERN)
    contract_digest: str = Field(pattern=_SHA256_PATTERN)
    step_index: int = Field(ge=0)
    attempt_index: int = Field(ge=0)
    action_digest: str = Field(pattern=_SHA256_PATTERN)
    failure_signature: str = Field(pattern=_SHA256_PATTERN)
    evidence_digest: str = Field(pattern=_SHA256_PATTERN)
    generator_version: str
    validator_schema_version: int = Field(ge=1)
    validator_policy_digest: str = Field(pattern=_SHA256_PATTERN)
    repair_epoch: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_binding_labels(self) -> "DiagnosticBinding":
        for label, value in (
            ("run_id", self.run_id),
            ("state_id", self.state_id),
            ("generator_version", self.generator_version),
        ):
            _diagnostic_nonblank(value, label)
        return self


class Fact(StrictModel):
    """One evidence-addressable fact supplied to the diagnostician."""

    evidence_id: str
    kind: DiagnosticFactKind
    statement: str
    data: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fact(self) -> "Fact":
        _diagnostic_nonblank(self.evidence_id, "evidence_id")
        _diagnostic_nonblank(self.statement, "statement")
        _validate_diagnostic_finite_tree(self.data, "fact data")
        return self


# The design document originally used DiagnosticFact.  Keep both public names
# so callers can migrate without maintaining two subtly different contracts.
DiagnosticFact = Fact


class FieldOwnership(StrictModel):
    """Authority and mutability of one planner-visible JSON pointer."""

    path: str
    owner: DiagnosticOwner
    mutable_in_current_repair: bool
    reason: str

    @model_validator(mode="after")
    def validate_ownership(self) -> "FieldOwnership":
        _validate_diagnostic_pointer(self.path, "field ownership path")
        _diagnostic_nonblank(self.reason, "field ownership reason")
        if self.mutable_in_current_repair and self.owner != "planner_authored":
            raise ValueError(
                "only planner_authored fields may be mutable in the current repair"
            )
        return self


class ParameterCausality(StrictModel):
    """Evidence-backed causal assessment for one candidate parameter."""

    parameter_path: str
    influence: Literal["direct", "conditional", "non_causal", "unproven"]
    observed_metric_response: Literal[
        "improved",
        "worsened",
        "unchanged",
        "not_comparable",
    ]
    directive: Literal[
        "change",
        "keep",
        "avoid_repeating",
        "change_mode_first",
        "collect_probe",
        "unknown",
    ]
    explanation: str
    evidence_ids: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_causality(self) -> "ParameterCausality":
        _validate_diagnostic_pointer(self.parameter_path, "parameter_path")
        _diagnostic_nonblank(self.explanation, "causality explanation")
        _diagnostic_unique_nonblank(self.evidence_ids, "causality evidence_ids")
        return self


class RepairStrategy(StrictModel):
    """A bounded, non-executing repair strategy ranked by the diagnostician."""

    priority: int = Field(ge=1, le=3)
    kind: RepairStrategyKind
    target_fields: list[str] = Field(default_factory=list, max_length=8)
    instruction: str
    expected_effect: str
    verification_checks: list[str] = Field(min_length=1, max_length=8)
    risks: list[str] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def validate_strategy(self) -> "RepairStrategy":
        _diagnostic_nonblank(self.instruction, "strategy instruction")
        _diagnostic_nonblank(self.expected_effect, "strategy expected_effect")
        _diagnostic_unique_nonblank(self.target_fields, "strategy target_fields")
        _diagnostic_unique_nonblank(
            self.verification_checks, "strategy verification_checks"
        )
        _diagnostic_unique_nonblank(self.risks, "strategy risks")
        for path in self.target_fields:
            _validate_diagnostic_pointer(path, "strategy target field")
        if self.kind in {"parameter_change", "mode_change"} and not self.target_fields:
            raise ValueError(f"{self.kind} requires at least one target field")
        return self


class DiagnosticEvidenceUse(StrictModel):
    evidence_id: str
    supports: str

    @model_validator(mode="after")
    def validate_evidence_use(self) -> "DiagnosticEvidenceUse":
        _diagnostic_nonblank(self.evidence_id, "evidence_id")
        _diagnostic_nonblank(self.supports, "evidence support")
        return self


class StepRepairDiagnosticContext(StrictModel):
    """Typed, digest-bound and failure-specific advisor input."""

    # This value participates in the context digest.  Changing the provider
    # transport or its host binding therefore invalidates stale resumable
    # failure artifacts instead of suppressing the upgraded advisor call.
    advisor_protocol_version: Literal[2] = 2
    binding: DiagnosticBinding
    issue_ids: list[str] = Field(min_length=1, max_length=8)
    current_state: dict[str, Any]
    immutable_goal_slice: dict[str, Any]
    rejected_draft: dict[str, Any]
    resolved_action: dict[str, Any] | None = None
    implicated_modules: list[dict[str, Any]] = Field(default_factory=list)
    failed_checks: list[dict[str, Any]] = Field(min_length=1)
    passed_check_summary: list[str] = Field(default_factory=list)
    facts: list[Fact] = Field(min_length=1)
    field_ownership: list[FieldOwnership] = Field(min_length=1)
    parameter_trials: list[dict[str, Any]] = Field(default_factory=list)
    deterministic_recommendations: list[dict[str, Any]] = Field(default_factory=list)
    allowed_strategy_kinds: list[RepairStrategyKind] = Field(min_length=1)

    @property
    def ownership(self) -> list[FieldOwnership]:
        """Compatibility spelling for early Phase-1 callers."""

        return self.field_ownership

    @model_validator(mode="after")
    def validate_context(self) -> "StepRepairDiagnosticContext":
        _diagnostic_unique_nonblank(self.issue_ids, "diagnostic issue_ids")
        _diagnostic_unique_nonblank(self.passed_check_summary, "passed_check_summary")
        _diagnostic_unique_nonblank(
            list(self.allowed_strategy_kinds), "allowed_strategy_kinds"
        )
        evidence_ids = [fact.evidence_id for fact in self.facts]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("diagnostic facts must have unique evidence IDs")
        ownership_paths = [card.path for card in self.field_ownership]
        if len(ownership_paths) != len(set(ownership_paths)):
            raise ValueError("field ownership paths must be unique")
        for label, value in (
            ("current_state", self.current_state),
            ("immutable_goal_slice", self.immutable_goal_slice),
            ("rejected_draft", self.rejected_draft),
            ("implicated_modules", self.implicated_modules),
            ("failed_checks", self.failed_checks),
            ("parameter_trials", self.parameter_trials),
            ("deterministic_recommendations", self.deterministic_recommendations),
        ):
            _validate_diagnostic_finite_tree(value, label)
        return self


class ParameterRangeRecommendation(StrictModel):
    """An evidence-traceable search range, never a planner-authored value.

    Bounds are nullable because a one-sided inequality or an unresolved search
    surface can still be useful to the ordinary planner.  The host validates
    every supplied number against the cited structured evidence before the
    recommendation is allowed into planner context.
    """

    path: str
    lower: float | None = None
    upper: float | None = None
    preferred: float | None = None
    unit: str | None = None
    classification: Literal["feasible", "promising", "avoid", "unresolved"]
    rationale: str
    evidence_ids: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_parameter_range(self) -> "ParameterRangeRecommendation":
        _validate_diagnostic_pointer(self.path, "parameter range path")
        _diagnostic_nonblank(self.rationale, "parameter range rationale")
        _diagnostic_unique_nonblank(self.evidence_ids, "parameter range evidence_ids")
        if self.unit is not None:
            _diagnostic_nonblank(self.unit, "parameter range unit")
        values = [
            value
            for value in (self.lower, self.upper, self.preferred)
            if value is not None
        ]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("parameter range values must be finite")
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower > self.upper
        ):
            raise ValueError("parameter range lower must not exceed upper")
        if self.preferred is not None:
            if self.lower is not None and self.preferred < self.lower:
                raise ValueError("parameter range preferred must be at least lower")
            if self.upper is not None and self.preferred > self.upper:
                raise ValueError("parameter range preferred must not exceed upper")
        return self


class GeometryAdvisorRecommendationWire(StrictModel):
    """One compact field recommendation accepted by Gemini structured output."""

    path: str
    action: Literal[
        "increase",
        "decrease",
        "replace",
        "keep",
        "avoid",
        "collect_probe",
        "none",
    ]
    bound_mode: Literal["none", "lower", "upper", "closed"]
    lower_text: str
    upper_text: str
    preferred_text: str
    unit: str
    classification: Literal["feasible", "promising", "avoid", "unresolved"]
    rationale: str
    evidence_id: str


class GeometryValidationAdvisorResponse(ProviderWireModel):
    """Small provider wire DTO for the independent validation agent."""

    diagnosis_class: Literal[
        "candidate_parameter",
        "candidate_variant_or_topology",
        "immutable_contract_conflict",
        "validator_policy_mismatch",
        "kernel_operation_failure",
        "infrastructure_failure",
        "insufficient_evidence",
    ]
    disposition: Literal[
        "retry_planner",
        "change_primitive_or_mode",
        "collect_more_evidence",
        "escalate_validator_review",
        "escalate_kernel_review",
        "retry_infrastructure",
        "stop_contract_infeasible",
        "stop_futile_retry",
    ]
    confidence: Literal["low", "medium", "high"]
    summary: str
    causal_chain: list[str] = Field(min_length=1, max_length=6)
    evidence_ids: list[str] = Field(min_length=1, max_length=12)
    recommendations: list[GeometryAdvisorRecommendationWire] = Field(max_length=8)
    strategy_kind: Literal[
        "none",
        "parameter_change",
        "mode_change",
        "primitive_change",
        "topology_change",
        "rollback_earlier_step",
        "evidence_probe",
        "validator_review",
        "kernel_review",
        "infrastructure_retry",
        "stop_futile_retry",
    ]
    strategy_instruction: str
    verification_checks: list[str] = Field(max_length=8)
    missing_evidence: list[str] = Field(max_length=8)
    planner_instruction: str


class ParameterDirectionRecommendation(StrictModel):
    """Evidence-backed direction supplied to the ordinary parameter planner."""

    path: str
    direction: Literal["increase", "decrease", "replace", "keep", "avoid", "none"]
    rationale: str
    evidence_ids: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_direction(self) -> "ParameterDirectionRecommendation":
        _validate_diagnostic_pointer(self.path, "parameter direction path")
        _diagnostic_nonblank(self.rationale, "parameter direction rationale")
        _diagnostic_unique_nonblank(self.evidence_ids, "direction evidence IDs")
        return self


class StepRepairDiagnosisBody(StrictModel):
    """LLM-authored diagnosis content without host-owned identity fields.

    The model is deliberately not asked to copy state IDs or SHA-256 digests.
    The host binds this body to the exact diagnostic case after structured
    generation, eliminating a non-semantic source of advisor failures.
    """

    diagnosis_class: Literal[
        "candidate_parameter",
        "candidate_variant_or_topology",
        "immutable_contract_conflict",
        "validator_policy_mismatch",
        "kernel_operation_failure",
        "infrastructure_failure",
        "insufficient_evidence",
    ]
    disposition: Literal[
        "retry_planner",
        "change_primitive_or_mode",
        "collect_more_evidence",
        "escalate_validator_review",
        "escalate_kernel_review",
        "retry_infrastructure",
        "stop_contract_infeasible",
        "stop_futile_retry",
    ]
    confidence: Literal["low", "medium", "high"]
    summary: str
    causal_chain: list[str] = Field(min_length=1, max_length=6)
    evidence_uses: list[DiagnosticEvidenceUse] = Field(min_length=1, max_length=12)
    parameter_causality: list[ParameterCausality] = Field(default_factory=list)
    direction_guidance: list[ParameterDirectionRecommendation] = Field(
        default_factory=list, max_length=12
    )
    parameter_ranges: list[ParameterRangeRecommendation] = Field(
        default_factory=list, max_length=8
    )
    strategies: list[RepairStrategy] = Field(default_factory=list, max_length=3)
    missing_evidence: list[str] = Field(default_factory=list, max_length=8)
    planner_instruction: str

    @model_validator(mode="after")
    def validate_diagnosis_body_shape(self) -> "StepRepairDiagnosisBody":
        _diagnostic_nonblank(self.summary, "diagnosis summary")
        _diagnostic_nonblank(self.planner_instruction, "planner_instruction")
        _diagnostic_unique_nonblank(self.causal_chain, "causal_chain")
        _diagnostic_unique_nonblank(self.missing_evidence, "missing_evidence")
        evidence_ids = [use.evidence_id for use in self.evidence_uses]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_uses must reference each evidence ID once")
        paths = [item.parameter_path for item in self.parameter_causality]
        if len(paths) != len(set(paths)):
            raise ValueError(
                "a parameter path may not carry contradictory causality directives"
            )
        range_paths = [item.path for item in self.parameter_ranges]
        if len(range_paths) != len(set(range_paths)):
            raise ValueError(
                "a parameter path may carry at most one range recommendation"
            )
        direction_paths = [item.path for item in self.direction_guidance]
        if len(direction_paths) != len(set(direction_paths)):
            raise ValueError("a parameter path may carry at most one direction")
        priorities = [strategy.priority for strategy in self.strategies]
        if len(priorities) != len(set(priorities)):
            raise ValueError("repair strategy priorities must be unique")
        if self.disposition in {"retry_planner", "change_primitive_or_mode"}:
            if not self.strategies:
                raise ValueError(f"{self.disposition} requires a repair strategy")
        geometry_kinds = {
            "parameter_change",
            "mode_change",
            "primitive_change",
            "topology_change",
            "rollback_earlier_step",
        }
        if self.diagnosis_class == "infrastructure_failure" and any(
            strategy.kind in geometry_kinds for strategy in self.strategies
        ):
            raise ValueError(
                "infrastructure_failure must not include a geometry repair strategy"
            )
        return self


class StepRepairDiagnosis(StepRepairDiagnosisBody):
    """Host-bound causal diagnosis; it is never an executable CAD action."""

    protocol_version: Literal[1] = 1
    state_id: str
    contract_digest: str = Field(pattern=_SHA256_PATTERN)
    diagnostic_context_digest: str = Field(pattern=_SHA256_PATTERN)
    failure_signature: str = Field(pattern=_SHA256_PATTERN)
    issue_ids: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_bound_diagnosis(self) -> "StepRepairDiagnosis":
        _diagnostic_nonblank(self.state_id, "state_id")
        _diagnostic_unique_nonblank(self.issue_ids, "diagnosis issue_ids")
        return self


class DiagnosticRecordRef(StrictModel):
    case_id: str = Field(pattern=_SHA256_PATTERN)
    diagnostic_context_digest: str = Field(pattern=_SHA256_PATTERN)
    failure_signature: str = Field(pattern=_SHA256_PATTERN)
    state_id: str
    step_index: int = Field(ge=0)
    attempt_index: int = Field(ge=0)
    repair_epoch: int = Field(ge=0)
    status: Literal["pending", "complete", "failed", "skipped"]
    artifact_path: str | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_record(self) -> "DiagnosticRecordRef":
        _diagnostic_nonblank(self.state_id, "state_id")
        if self.artifact_path is not None:
            _diagnostic_nonblank(self.artifact_path, "artifact_path")
        if self.failure_reason is not None:
            _diagnostic_nonblank(self.failure_reason, "failure_reason")
        return self


class DiagnosticJournal(StrictModel):
    records: list[DiagnosticRecordRef] = Field(default_factory=list)
    calls_by_step: dict[str, int] = Field(default_factory=dict)
    cache_hit_count: int = Field(default=0, ge=0)
    futile_retry_avoided_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_journal(self) -> "DiagnosticJournal":
        case_ids = [record.case_id for record in self.records]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("diagnostic journal case IDs must be unique")
        for key, count in self.calls_by_step.items():
            _diagnostic_nonblank(key, "diagnostic journal step key")
            if count < 0:
                raise ValueError("diagnostic journal call counts must be non-negative")
        return self


class LLMUsage(StrictModel):
    """호출 수와 입력ㆍ출력ㆍthinkingㆍcache 토큰의 누적 사용량이다."""

    calls: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    tool_use_tokens: int = 0
    total_tokens: int = 0
    accounting_complete: bool = True
    unmetered_calls: int = 0


class CriticViewRequest(StrictModel):
    view_id: Literal["front", "right", "top", "isometric"]
    camera: str
    target_module_ids: list[str] = Field(default_factory=list)
    target_port_ids: list[str] = Field(default_factory=list)
    purpose: str
    required: bool = True
    evidence_status: Literal["unavailable", "pending", "available"] = "unavailable"
    evidence_path: str | None = None
    unavailable_reason: str | None = None


class PatchSuggestion(StrictModel):
    suggestion_id: str
    target_module_id: str | None = None
    target_port_ids: list[str] = Field(default_factory=list)
    issue_ids: list[str] = Field(default_factory=list)
    operation: str
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str


class CriticReport(StrictModel):
    """최종 계약 판정, 이슈, 화면 요청과 후속 조치를 제공한다."""

    passed: bool
    verification_status: Literal["passed", "failed"]
    error_count: int = 0
    warning_count: int = 0
    expected_open_ports: int | None = None
    actual_open_ports: int = 0
    expected_open_ports_source: OpenPortExpectationSource = "unknown"
    issues: list[StaticIssue] = Field(default_factory=list)
    view_requests: list[CriticViewRequest] = Field(default_factory=list)
    patch_suggestions: list[PatchSuggestion] = Field(default_factory=list)
    skipped_mcp_reason: str | None = None
    next_actions: list[str] = Field(default_factory=list)


class GenerationArtifacts(StrictModel):
    """한 실행에서 생성될 모든 감사ㆍCAD 파일의 경로를 묶는다."""

    run_id: str
    output_dir: str
    prompt_path: str
    intent_path: str
    intent_attempts_path: str | None = None
    intent_diagnostics_path: str | None = None
    actions_path: str
    state_path: str
    freecad_script_path: str
    report_path: str
    step_verification_path: str | None = None
    critic_report_path: str | None = None
    mcp_result_path: str | None = None
    action_attempts_path: str | None = None
    repair_advice_path: str | None = None
    diagnostics_index_path: str | None = None
    constraint_ledger_path: str | None = None
    global_preflight_path: str | None = None
    search_events_path: str | None = None
    advisor_artifact_paths: list[str] = Field(default_factory=list)
    checkpoint_path: str | None = None
    freecad_validation_path: str | None = None
    freecad_document_path: str | None = None
    visual_evidence_paths: list[str] = Field(default_factory=list)


ArtifactFileStatus = Literal["available", "unavailable", "partial", "stale"]


class ArtifactStatus(StrictModel):
    """산출물 하나의 가용성, 생산 단계와 차단 원인을 기록한다."""

    name: str
    path: str | None = None
    status: ArtifactFileStatus
    producer_stage: str | None = None
    blocking_issue_ids: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None


class RunReport(StrictModel):
    """실행 상태, 검증 수준, artifact 가용성과 LLM 사용량의 최종 요약이다."""

    run_id: str
    status: Literal[
        "success",
        "success_with_deviations",
        "partial",
        "paused",
        "failed",
    ] = "success"
    realization_status: PreflightStatus | Literal["not_run"] = "not_run"
    deviation_count: int = 0
    pause_reason: str | None = None
    resume_command: str | None = None
    recovery_state: dict[str, Any] = Field(default_factory=dict)
    failed_stage: str | None = None
    dry_run: bool
    freecad_opened: bool
    freecad_mcp_used: bool
    freecad_mcp_error: str | None = None
    verification_status: VerificationStatus = "not_run"
    static_error_count: int = 0
    static_warning_count: int = 0
    critic_passed: bool | None = None
    skipped_mcp_reason: str | None = None
    top_issues: list[str] = Field(default_factory=list)
    llm_usage: LLMUsage = Field(default_factory=LLMUsage)
    llm_policy: dict[str, Any] = Field(default_factory=dict)
    intent_attempt_count: int = 0
    intent_repair_count: int = 0
    intent_protocol_retry_count: int = 0
    intent_advisor_call_count: int = 0
    intent_advisor_success_count: int = 0
    intent_advisor_failure_count: int = 0
    intent_futile_retry_avoided_count: int = 0
    action_repair_count: int = 0
    repair_attempt_count: int = 0
    step_repair_advisor_count: int = 0
    advisor_call_count: int = 0
    advisor_success_count: int = 0
    advisor_failure_count: int = 0
    advisor_cache_hit_count: int = 0
    advisor_artifact_paths: list[str] = Field(default_factory=list)
    futile_retry_avoided_count: int = 0
    artifacts: GenerationArtifacts
    artifact_statuses: list[ArtifactStatus] = Field(default_factory=list)
    summary: str
