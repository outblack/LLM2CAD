"""생성 가능한 primitive catalog와 파라미터 소유권ㆍ유효성 규칙을 정의한다.

LLM draft 또는 resolved action과 현재 상태를 입력받아 명시적 검증 결과를 반환한다.
검증기는 누락값을 채우거나 대체 primitive를 선택하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from cadgen.geometry_policy import predict_c1_spline
from cadgen.schemas import ActionDraft, PipeState, ResolvedAction, ValidationResult
from cadgen.vector import (
    canonical_circular_arc_frame,
    circular_rim_mismatch,
    dot,
    length as vector_length,
    normalize,
    sub,
    vec,
)


SUPPORTED_INLINE_COMPONENTS: tuple[str, ...] = (
    "flange",
    "coupling",
    "union",
    "valve",
)

# A general geometry-safety bound for reducer-like lofts. It is intentionally
# independent of nominal size: both OD and bore center/radius movement must fit
# inside this half-angle, preventing a positive-but-near-zero length from being
# accepted as a "smooth" transition.
MAX_TRANSITION_HALF_ANGLE_DEGREES = 30.0


@dataclass(frozen=True)
class ModuleSpec:
    """primitive의 입출력 차수, 파라미터와 LLM 노출 정책을 정의한다."""

    module_id: str
    required_params: tuple[str, ...]
    optional_params: tuple[str, ...]
    input_count: int
    output_count: int | str
    llm_visible: bool = False
    use_when: str = ""
    not_when: str = ""
    invariants: tuple[str, ...] = ()


MODULE_REGISTRY: dict[str, ModuleSpec] = {
    "straight_pipe": ModuleSpec(
        "straight_pipe",
        ("length", "outer_diameter", "wall_thickness", "start_position", "axis"),
        ("segment_resolution",),
        1,
        1,
    ),
    "bend_pipe": ModuleSpec(
        "bend_pipe",
        (
            "angle",
            "bend_radius",
            "outer_diameter",
            "wall_thickness",
            "start_position",
            "axis",
            "out_axis",
        ),
        ("segment_resolution", "turn_direction"),
        1,
        1,
    ),
    "junction_pipe": ModuleSpec(
        "junction_pipe",
        (
            "branch_count",
            "branch_angles",
            "outer_diameter",
            "wall_thickness",
            "start_position",
            "axis",
        ),
        (
            "blend_radius",
            "include_primary_outlet",
            "required_outlet_directions",
            "required_outlet_vectors",
            "outlet_vectors",
            "junction_style",
        ),
        1,
        "dynamic",
    ),
    "reducer_pipe": ModuleSpec(
        "reducer_pipe",
        (
            "length",
            "diameter_in",
            "diameter_out",
            "wall_thickness_in",
            "wall_thickness_out",
            "start_position",
            "axis",
        ),
        ("transition_type",),
        1,
        1,
    ),
    "connector_pipe": ModuleSpec(
        "connector_pipe",
        ("length", "outer_diameter", "wall_thickness", "start_position", "axis"),
        ("connection_type", "coupling_outer_diameter", "sleeve_overlap"),
        1,
        1,
    ),
    "cap_pipe": ModuleSpec(
        "cap_pipe",
        ("end_type", "outer_diameter", "wall_thickness", "start_position", "axis"),
        ("cap_thickness",),
        1,
        0,
    ),
    "route": ModuleSpec(
        "route",
        (
            "path_kind",
            "section_source",
            "outer_diameter",
            "wall_thickness",
            "start_position",
            "axis",
        ),
        (
            "length",
            "direction",
            "bend_radius",
            "sweep_angle",
            "plane_normal",
            "terminal_axis",
            "waypoint_frame",
            "waypoints",
            "initial_tangent",
            "final_tangent",
            "interpolation",
            "frenet",
            "minimum_curvature_radius",
            "path_points",
            "end_position",
            "out_axis",
        ),
        1,
        1,
        True,
        "Create one continuous line, watertight analytic elbow, or freeform spline run.",
        "Do not use for branching, diameter transition, loop closure, or termination.",
        ("one inlet and one outlet", "continuous centerline", "constant section"),
    ),
    "transition": ModuleSpec(
        "transition",
        (
            "section_source",
            "diameter_in",
            "diameter_out",
            "wall_thickness_in",
            "wall_thickness_out",
            "length",
            "offset",
            "start_position",
            "axis",
        ),
        (
            "connector_type_out",
            "connector_gender_out",
            "connector_standard_out",
            "end_position",
        ),
        1,
        1,
        True,
        "Change pipe diameter or wall thickness, concentrically or eccentrically.",
        "Do not use for a constant-section run or branch.",
        ("one inlet and one outlet", "positive wall", "continuous bore"),
    ),
    "junction": ModuleSpec(
        "junction",
        (
            "section_source",
            "outlets",
            "blend_mode",
            "max_hub_radius",
            "outer_diameter",
            "wall_thickness",
            "start_position",
            "axis",
        ),
        (
            "blend_radius",
            "inner_blend_radius",
            "include_primary_outlet",
            "branch_count",
            "outlet_vectors",
        ),
        1,
        "dynamic",
        True,
        "Split one inlet into exactly two explicitly described outlets.",
        "Do not use for a single continuation or to merge two existing ports.",
        ("one inlet", "exactly two outlets", "continuous bore", "bounded hub"),
    ),
    "connect_ports": ModuleSpec(
        "connect_ports",
        (
            "other_port_id",
            "path_kind",
            "section_source",
            "outer_diameter",
            "wall_thickness",
            "start_position",
            "axis",
            "end_position",
            "end_axis",
        ),
        (
            "waypoints",
            "initial_tangent",
            "final_tangent",
            "interpolation",
            "frenet",
            "minimum_curvature_radius",
            "path_points",
        ),
        2,
        0,
        True,
        "Join two distinct existing open ports to close a loop or merge branches.",
        "Do not use when only one endpoint exists or when an outlet must remain open.",
        ("two distinct inlets", "no new open port", "compatible endpoint sections"),
    ),
    "terminate": ModuleSpec(
        "terminate",
        (
            "section_source",
            "termination_type",
            "thickness",
            "outer_diameter",
            "wall_thickness",
            "start_position",
            "axis",
        ),
        (),
        1,
        0,
        True,
        "Seal one open port with an explicit cap or plug.",
        "Do not create a module for an end that should remain open.",
        ("one consumed inlet", "sealed terminal", "no outlet"),
    ),
    "inline_component": ModuleSpec(
        "inline_component",
        (
            "section_source",
            "component_type",
            "length",
            "body_outer_diameter",
            "body_start_offset",
            "body_length",
            "connector_type_out",
            "connector_gender_out",
            "outer_diameter",
            "wall_thickness",
            "start_position",
            "axis",
        ),
        (
            "actuator_diameter",
            "actuator_height",
            "connector_standard_out",
            "end_position",
        ),
        1,
        1,
        True,
        "Place one explicit inline flange, coupling, union, or valve body.",
        "Do not use as a plain route, diameter transition, branch, closure, or cap.",
        (
            "one inlet and one outlet",
            "continuous bore",
            "component identity bound to generated geometry",
        ),
    ),
}

MODULE_DRAFT_PARAMS: dict[str, tuple[str, ...]] = {
    "straight_pipe": ("length", "direction", "segment_resolution"),
    "bend_pipe": ("angle", "turn_direction", "bend_radius", "segment_resolution"),
    "junction_pipe": (
        "branch_count",
        "branch_angles",
        "blend_radius",
        "length",
        "direction",
        "required_outlet_directions",
        "required_outlet_vectors",
        "outlet_vectors",
        "include_primary_outlet",
        "junction_style",
    ),
    "reducer_pipe": (
        "length",
        "diameter_out",
        "wall_thickness_out",
        "transition_type",
    ),
    "connector_pipe": (
        "length",
        "direction",
        "connection_type",
        "coupling_outer_diameter",
        "sleeve_overlap",
    ),
    "cap_pipe": ("end_type", "cap_thickness"),
    "route": (
        "path_kind",
        "section_source",
        "outer_diameter",
        "wall_thickness",
        "length",
        "direction",
        "bend_radius",
        "sweep_angle",
        "plane_normal",
        "waypoint_frame",
        "waypoints",
        "final_tangent",
        "interpolation",
        "frenet",
        "minimum_curvature_radius",
    ),
    "transition": (
        "section_source",
        "outer_diameter",
        "wall_thickness",
        "diameter_out",
        "wall_thickness_out",
        "length",
        "offset",
    ),
    "junction": (
        "section_source",
        "outer_diameter",
        "wall_thickness",
        "outlets",
        "blend_mode",
        "blend_radius",
        "inner_blend_radius",
        "max_hub_radius",
    ),
    "connect_ports": (
        "other_port_id",
        "path_kind",
        "section_source",
        "outer_diameter",
        "wall_thickness",
        "waypoints",
        "interpolation",
        "frenet",
        "minimum_curvature_radius",
    ),
    "terminate": (
        "section_source",
        "outer_diameter",
        "wall_thickness",
        "termination_type",
        "thickness",
    ),
    "inline_component": (
        "section_source",
        "outer_diameter",
        "wall_thickness",
        "component_type",
        "length",
        "body_outer_diameter",
            "body_start_offset",
            "body_length",
            "flange_bolt_count",
            "flange_bolt_circle_diameter",
            "flange_bolt_hole_diameter",
            "flange_reference_axis",
            "union_ring_outer_diameter",
            "union_ring_length",
            "actuator_diameter",
            "actuator_height",
            "actuator_axis",
        "connector_type_out",
        "connector_gender_out",
        "connector_standard_out",
    ),
}


def available_modules() -> list[str]:
    """legacy를 포함해 registry에 등록된 모든 primitive ID를 반환한다."""

    return list(MODULE_REGISTRY)


def llm_visible_modules() -> list[str]:
    """프로덕션 LLM 선택이 허용된 primitive ID만 반환한다."""

    return [module_id for module_id, spec in MODULE_REGISTRY.items() if spec.llm_visible]


def planner_catalog() -> list[dict[str, Any]]:
    """프로덕션 planner에 허용된 schema-v2 primitive 설명만 반환한다."""

    catalog = []
    for module_id, spec in MODULE_REGISTRY.items():
        if not spec.llm_visible:
            continue
        entry = {
            "id": module_id,
            "schema_version": 2,
            "inputs": spec.input_count,
            "outputs": spec.output_count,
            "use_when": spec.use_when,
            "not_when": spec.not_when,
            # The inlet section is already immutable state on target_port.
            # Production planning must inherit it; a diameter/wall change is a
            # transition output, not an explicit mismatch at a mating face.
            "authored_params": [
                parameter
                for parameter in draft_params_for(module_id)
                if parameter not in {"outer_diameter", "wall_thickness"}
            ],
            "invariants": list(spec.invariants),
        }
        variants = _planner_variant_contracts(module_id)
        if variants:
            entry["variants"] = variants
        catalog.append(entry)
    return catalog


def _planner_variant_contracts(module_id: str) -> dict[str, list[str]]:
    if module_id == "route":
        return {
            "line": ["length", "direction"],
            "circular_arc": [
                "bend_radius",
                "signed sweep_angle",
                "bend-plane normal hint; resolver orthogonalizes it",
                "terminal tangent derived analytically",
            ],
            "spline": [
                "2+ waypoints",
                "waypoint_frame=global or relative_to_target",
                "initial tangent derived from target port",
                "final_tangent",
                "resolver-owned cubic spline/corrected-frame sweep policy",
                "minimum_curvature_radius derived from section and immutable goal",
            ],
        }
    if module_id == "connect_ports":
        return {
            "line": ["direct chord; no curve fields"],
            "circular_arc": [
                "exactly 1 non-collinear midpoint",
                "endpoint tangents derived from the three-point arc",
                "minimum_curvature_radius resolver-derived",
            ],
            "spline": [
                "1+ waypoints",
                "endpoint tangents derived from the two ports",
                "resolver-owned cubic spline/corrected-frame sweep policy",
                "minimum_curvature_radius resolver-derived",
            ],
        }
    if module_id == "junction":
        return {
            "hard": [
                "omit blend_radius and inner_blend_radius",
                "author max_hub_radius",
                "every outlet.length > max_hub_radius",
            ],
            "fillet": [
                "author blend_radius and inner_blend_radius",
                "author max_hub_radius",
                "every outlet.length > max_hub_radius",
            ],
        }
    if module_id == "inline_component":
        return {
            "flange": [
                "flange_bolt_count",
                "flange_bolt_circle_diameter",
                "flange_bolt_hole_diameter",
                "flange_reference_axis perpendicular to pipe",
                "outer_diameter + flange_bolt_hole_diameter < flange_bolt_circle_diameter",
                "flange_bolt_circle_diameter + flange_bolt_hole_diameter < body_outer_diameter",
                "body_start_offset=0 or body touches the output axial end",
            ],
            "coupling": [
                "body_start_offset=0",
                "body_length=length",
            ],
            "union": [
                "union_ring_outer_diameter > body_outer_diameter",
                "2*union_ring_length <= body_length",
                "body between necks",
            ],
            "valve": [
                "actuator_diameter",
                "actuator_height",
                "actuator_axis perpendicular to pipe",
                "body between necks",
            ],
        }
    return {}


def get_module_spec(module_id: str) -> ModuleSpec:
    """지정 primitive의 registry 계약을 반환한다."""

    return MODULE_REGISTRY[module_id]


def draft_params_for(module_id: str) -> tuple[str, ...]:
    """해당 primitive에서 planner가 작성할 수 있는 파라미터 이름을 반환한다."""

    return MODULE_DRAFT_PARAMS[module_id]


def resolver_owned_params_for(module_id: str) -> tuple[str, ...]:
    """해당 primitive에서 resolver만 계산할 수 있는 파라미터 이름을 반환한다."""

    spec = MODULE_REGISTRY[module_id]
    draft_params = set(draft_params_for(module_id))
    resolved_params = [*spec.required_params, *spec.optional_params]
    return tuple(param for param in resolved_params if param not in draft_params)


def filter_draft_params(module_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """LLM 입력에서 primitive별 authored 파라미터만 보존한다."""

    allowed = set(draft_params_for(module_id))
    return {key: value for key, value in params.items() if key in allowed}


def canonicalize_junction_params(params: dict[str, Any]) -> dict[str, Any]:
    """legacy junction 표현을 검증 가능한 표준 벡터와 style로 정규화한다."""

    result = dict(params)
    for key in ("required_outlet_vectors", "outlet_vectors"):
        if key in result and result[key] is not None:
            result[key] = _canonical_vector_list(result[key])
    if result.get("junction_style") is not None:
        result["junction_style"] = _canonical_junction_style(result["junction_style"])
    return result


def validate_draft(draft: ActionDraft, state: PipeState) -> ValidationResult:
    """Validate an LLM decision without choosing or repairing it."""
    errors: list[str] = []
    spec = MODULE_REGISTRY.get(draft.module)
    if spec is None:
        return ValidationResult(valid=False, errors=[f"Unknown module: {draft.module}"])
    if not spec.llm_visible:
        return ValidationResult(
            valid=False,
            errors=[f"Legacy module is not accepted from the production planner: {draft.module}"],
        )
    if draft.catalog_schema_version != 2:
        errors.append("catalog_schema_version must be 2")

    open_ids = {port.id for port in state.open_ports}
    target_port = next(
        (port for port in state.open_ports if port.id == draft.target_port), None
    )
    if draft.target_port not in open_ids:
        errors.append(f"Target port is not open: {draft.target_port}")
    elif target_port is not None and target_port.connector_type != "plain":
        if (
            draft.module != "inline_component"
            or draft.params.get("component_type") != target_port.connector_type
        ):
            errors.append(
                "A non-plain target port requires matching inline_component mating geometry"
            )

    pending_ids = {
        goal.goal_id for goal in state.remaining_goals if goal.goal_id is not None
    }
    affected = set(draft.affected_goal_ids)
    completed = set(draft.completed_goal_ids)
    if not affected:
        errors.append("affected_goal_ids must not be empty")
    unknown_affected = sorted(affected - pending_ids)
    if unknown_affected:
        errors.append(f"affected_goal_ids are not pending: {unknown_affected}")
    if not completed.issubset(affected):
        errors.append("completed_goal_ids must be a subset of affected_goal_ids")
    _validate_goal_order_claims(affected, completed, state, errors)
    _validate_completion_claims(completed, state, errors)
    _validate_atomic_goal_binding(
        draft.module,
        draft.params,
        affected,
        completed,
        state,
        errors,
    )
    if draft.satisfied_components:
        errors.append(
            "satisfied_components claims are not accepted; component evidence must "
            "come from an inline_component module"
        )

    allowed = set(draft_params_for(draft.module))
    unexpected = sorted(set(draft.params) - allowed)
    if unexpected:
        errors.append(f"Unexpected authored params for {draft.module}: {unexpected}")
    params = draft.params
    section_source = params.get("section_source")
    if section_source not in {"inherit_target", "explicit"}:
        errors.append("section_source must be inherit_target or explicit")
    if section_source == "explicit":
        if params.get("outer_diameter") is None or params.get("wall_thickness") is None:
            errors.append(
                "explicit section_source requires outer_diameter and wall_thickness"
            )
    elif section_source == "inherit_target" and (
        params.get("outer_diameter") is not None
        or params.get("wall_thickness") is not None
    ):
        errors.append(
            "inherit_target must omit outer_diameter and wall_thickness"
        )

    if draft.module == "route":
        kind = params.get("path_kind")
        if kind == "line":
            _require_authored(params, ("length", "direction"), errors)
            line_length = _float_param(params, "length", errors)
            if line_length is not None and line_length <= 0.0:
                errors.append("line route length must be greater than zero")
            _validate_vector_value(params.get("direction"), "direction", errors)
            if any(
                params.get(name) is not None
                for name in (
                    "bend_radius",
                    "sweep_angle",
                    "plane_normal",
                    "terminal_axis",
                    "waypoint_frame",
                    "initial_tangent",
                    "final_tangent",
                    "interpolation",
                    "frenet",
                    "minimum_curvature_radius",
                )
            ) or bool(params.get("waypoints")):
                errors.append("line route does not accept curve parameters")
        elif kind == "circular_arc":
            _require_authored(
                params,
                ("bend_radius", "sweep_angle", "plane_normal"),
                errors,
            )
            bend_radius = _float_param(params, "bend_radius", errors)
            sweep_angle = _float_param(params, "sweep_angle", errors)
            if bend_radius is not None and bend_radius <= 0.0:
                errors.append("circular_arc bend_radius must be greater than zero")
            if sweep_angle is not None and not (0.0 < abs(sweep_angle) < 360.0):
                errors.append(
                    "circular_arc sweep_angle magnitude must be in (0, 360)"
                )
            _validate_vector_value(params.get("plane_normal"), "plane_normal", errors)
            if any(
                params.get(name) is not None
                for name in (
                    "length",
                    "direction",
                    "waypoint_frame",
                    "initial_tangent",
                    "final_tangent",
                    "interpolation",
                    "frenet",
                    "minimum_curvature_radius",
                )
            ) or bool(params.get("waypoints")):
                errors.append(
                    "circular_arc route does not accept line/spline parameters"
                )
        elif kind == "spline":
            _require_authored(
                params,
                ("waypoints",),
                errors,
            )
            if len(params.get("waypoints") or []) < 2:
                errors.append("spline route requires at least two waypoints")
            _vector_list_param(params, "waypoints", errors, allow_zero=True)
            if params.get("final_tangent") is not None:
                _validate_vector_value(
                    params.get("final_tangent"), "final_tangent", errors
                )
            if params.get("waypoint_frame", "global") not in {
                "global",
                "relative_to_target",
            }:
                errors.append(
                    "spline waypoint_frame must be global or relative_to_target"
                )
            elif target_port is not None and params.get("waypoints"):
                first = _canonical_vector(params["waypoints"][0])
                if first is not None:
                    resolved_first = (
                        tuple(
                            float(target_port.position[index]) + first[index]
                            for index in range(3)
                        )
                        if params.get("waypoint_frame") == "relative_to_target"
                        else first
                    )
                    separation = vector_length(
                        tuple(
                            resolved_first[index]
                            - float(target_port.position[index])
                            for index in range(3)
                        )
                    )
                    if separation <= state.modeling_tolerance:
                        errors.append(
                            "the first spline waypoint must not coincide with the target port"
                        )
            if any(
                params.get(name) is not None
                for name in (
                    "length",
                    "direction",
                    "bend_radius",
                    "sweep_angle",
                    "plane_normal",
                    "terminal_axis",
                )
            ):
                errors.append("spline route does not accept line/arc parameters")
        else:
            errors.append("route path_kind must be line, circular_arc, or spline")
    elif draft.module == "transition":
        _require_authored(
            params,
            (
                "diameter_out",
                "length",
            ),
            errors,
        )
        if params.get("wall_thickness_out") is not None:
            wall_out = _float_param(params, "wall_thickness_out", errors)
            if wall_out is not None and wall_out <= 0.0:
                errors.append("wall_thickness_out must be greater than zero")
        if params.get("offset") is not None:
            _validate_vector_value(
                params.get("offset"), "offset", errors, allow_zero=True
            )
        if target_port is not None:
            diameter_out = _float_param(params, "diameter_out", errors)
            transition_length = _float_param(params, "length", errors)
            wall_out = _float_param(params, "wall_thickness_out", errors)
            offset = _canonical_vector(params.get("offset")) or (0.0, 0.0, 0.0)
            _validate_transition_taper_geometry(
                diameter_in=float(target_port.outer_diameter),
                wall_in=float(target_port.wall_thickness),
                diameter_out=diameter_out,
                wall_out=(
                    wall_out
                    if wall_out is not None
                    else float(target_port.wall_thickness)
                ),
                axial_length=transition_length,
                offset=offset,
                errors=errors,
            )
    elif draft.module == "junction":
        _require_authored(
            params,
            (
                "outlets",
                "blend_mode",
                "max_hub_radius",
            ),
            errors,
        )
        if params.get("blend_mode") == "fillet":
            _require_authored(
                params, ("blend_radius", "inner_blend_radius"), errors
            )
        elif params.get("blend_mode") == "hard" and (
            params.get("blend_radius") is not None
            or params.get("inner_blend_radius") is not None
        ):
            errors.append("hard junction must omit unused blend radii")
        outlets = params.get("outlets")
        if not isinstance(outlets, list) or len(outlets) != 2:
            errors.append(
                "junction outlets must contain exactly two entries; "
                "compose additional binary junctions for higher-degree branching"
            )
        else:
            for index, outlet in enumerate(outlets):
                if not isinstance(outlet, dict):
                    errors.append(f"outlets[{index}] must be an object")
                    continue
                outlet_axis = _validate_vector_value(
                    outlet.get("axis"), f"outlets[{index}].axis", errors
                )
                if outlet_axis is not None and target_port is not None:
                    alignment = dot(
                        normalize(outlet_axis),
                        normalize(tuple(float(value) for value in target_port.axis)),
                    )
                    if alignment <= -0.999:
                        errors.append(
                            f"outlets[{index}].axis must not retrace the consumed "
                            "target-port axis; route to the junction first or choose "
                            "a forward/diverging outlet"
                        )
    elif draft.module == "connect_ports":
        _require_authored(
            params,
            (
                "other_port_id",
                "path_kind",
            ),
            errors,
        )
        other = params.get("other_port_id")
        if other == draft.target_port:
            errors.append("connect_ports requires two distinct open ports")
        elif other not in open_ids:
            errors.append(f"Second target port is not open: {other}")
        waypoints = _vector_list_param(
            params, "waypoints", errors, allow_zero=True
        )
        if params.get("path_kind") == "line":
            if waypoints:
                errors.append("line connect_ports does not accept waypoints")
            if any(
                params.get(name) is not None
                for name in (
                    "initial_tangent",
                    "final_tangent",
                    "interpolation",
                    "frenet",
                    "minimum_curvature_radius",
                )
            ):
                errors.append("line connect_ports does not accept curve parameters")
        elif params.get("path_kind") == "circular_arc":
            _require_authored(
                params,
                ("waypoints",),
                errors,
            )
            if len(waypoints) != 1:
                errors.append(
                    "circular_arc connect_ports requires exactly one arc waypoint"
                )
            if any(
                params.get(name) is not None
                for name in (
                    "initial_tangent",
                    "final_tangent",
                    "interpolation",
                    "frenet",
                )
            ):
                errors.append(
                    "circular_arc connect_ports does not accept spline parameters"
                )
            minimum_radius = _float_param(
                params, "minimum_curvature_radius", errors
            )
            if (
                minimum_radius is not None
                and target_port is not None
                and minimum_radius
                <= target_port.outer_diameter / 2.0 + state.modeling_tolerance
            ):
                errors.append(
                    "connect_ports minimum_curvature_radius must exceed the pipe outer radius by the modeling tolerance"
                )
        else:
            _require_authored(
                params,
                ("waypoints",),
                errors,
            )
            if not waypoints:
                errors.append("curved connect_ports requires at least one waypoint")
            minimum_radius = _float_param(
                params, "minimum_curvature_radius", errors
            )
            if (
                minimum_radius is not None
                and target_port is not None
                and minimum_radius
                <= target_port.outer_diameter / 2.0 + state.modeling_tolerance
            ):
                errors.append(
                    "connect_ports minimum_curvature_radius must exceed the pipe outer radius by the modeling tolerance"
                )
    elif draft.module == "terminate":
        _require_authored(params, ("termination_type", "thickness"), errors)
    elif draft.module == "inline_component":
        _require_authored(
            params,
            (
                "component_type",
                "length",
                "body_outer_diameter",
                "body_start_offset",
                "body_length",
                "connector_type_out",
                "connector_gender_out",
                "connector_standard_out",
            ),
            errors,
            allow_none={"connector_standard_out"},
        )
        component_type = params.get("component_type")
        if component_type not in SUPPORTED_INLINE_COMPONENTS:
            errors.append(
                "inline_component component_type must be one of "
                f"{list(SUPPORTED_INLINE_COMPONENTS)}"
            )
        actuator_diameter = params.get("actuator_diameter")
        actuator_height = params.get("actuator_height")
        actuator_axis = params.get("actuator_axis")
        if component_type == "valve" and (
            actuator_diameter is None
            or actuator_height is None
            or actuator_axis is None
        ):
            errors.append("valve requires actuator dimensions and actuator_axis")
        if component_type != "valve" and (
            actuator_diameter is not None
            or actuator_height is not None
            or actuator_axis is not None
        ):
            errors.append("actuator parameters are valid only for valve")
        if actuator_axis is not None:
            _validate_vector_value(actuator_axis, "actuator_axis", errors)
        flange_values = (
            params.get("flange_bolt_count"),
            params.get("flange_bolt_circle_diameter"),
            params.get("flange_bolt_hole_diameter"),
            params.get("flange_reference_axis"),
        )
        if component_type == "flange" and any(value is None for value in flange_values):
            errors.append("flange requires authored bolt count, circle, and hole diameter")
        if component_type != "flange" and any(value is not None for value in flange_values):
            errors.append("flange bolt parameters are valid only for flange")
        union_values = (
            params.get("union_ring_outer_diameter"),
            params.get("union_ring_length"),
        )
        if component_type == "union" and any(value is None for value in union_values):
            errors.append("union requires authored ring diameter and length")
        if component_type != "union" and any(value is not None for value in union_values):
            errors.append("union ring parameters are valid only for union")
        connector_type = params.get("connector_type_out")
        if connector_type not in {"plain", component_type}:
            errors.append(
                "inline_component connector_type_out must be plain or match component_type"
            )
        if connector_type == "plain":
            _validate_plain_connector_fields(
                params,
                "connector_type_out",
                "connector_gender_out",
                "connector_standard_out",
                errors,
            )
        elif any(
            goal.goal_id not in completed
            for goal in state.remaining_goals
            if goal.goal_id is not None
        ):
            errors.append(
                "a non-plain inline_component output is allowed only as the final authored terminal"
            )
        body_offset = _float_param(params, "body_start_offset", errors)
        body_length = _float_param(params, "body_length", errors)
        component_length = _float_param(params, "length", errors)
        if (
            target_port is not None
            and target_port.connector_type != "plain"
            and body_offset is not None
            and body_offset > 1e-9
        ):
            errors.append(
                "a non-plain inlet connector requires component geometry touching the input end"
            )
        if connector_type != "plain" and (
            component_type not in {"flange", "coupling"}
            or body_offset is None
            or body_length is None
            or component_length is None
            or abs(body_offset + body_length - component_length) > 1e-9
        ):
            errors.append(
                "a non-plain output connector requires flange/coupling geometry touching the output end"
            )

    return ValidationResult(valid=not errors, errors=errors)


def _require_authored(
    params: dict[str, Any],
    names: tuple[str, ...],
    errors: list[str],
    *,
    allow_none: set[str] | None = None,
) -> None:
    allow_none = allow_none or set()
    for name in names:
        if name not in params or (params[name] is None and name not in allow_none):
            errors.append(f"Missing LLM-authored param: {name}")


def validate_action(action: ResolvedAction, state: PipeState) -> ValidationResult:
    """파생값까지 포함된 행동이 현재 포트와 primitive 불변식을 지키는지 검사한다."""

    errors: list[str] = []
    spec = MODULE_REGISTRY.get(action.module)
    if spec is None:
        return ValidationResult(valid=False, errors=[f"Unknown module: {action.module}"])

    open_port_ids = {port.id for port in state.open_ports}
    target_port = next(
        (port for port in state.open_ports if port.id == action.target_port),
        None,
    )
    if action.target_port not in open_port_ids:
        errors.append(f"Target port is not open: {action.target_port}")
    elif target_port is not None and target_port.connector_type != "plain":
        if (
            action.module != "inline_component"
            or action.params.get("component_type") != target_port.connector_type
        ):
            errors.append(
                "A non-plain target port requires matching inline_component mating geometry"
            )
    _validate_completion_claims(set(action.completed_goal_ids), state, errors)
    affected = set(action.affected_goal_ids)
    completed = set(action.completed_goal_ids)
    _validate_goal_order_claims(affected, completed, state, errors)
    if action.module in {
        "route",
        "transition",
        "junction",
        "connect_ports",
        "terminate",
        "inline_component",
    }:
        _validate_atomic_goal_binding(
            action.module,
            action.params,
            affected,
            completed,
            state,
            errors,
        )

    for param in spec.required_params:
        if param not in action.params or action.params[param] is None:
            errors.append(f"Missing required param for {action.module}: {param}")

    params: dict[str, Any] = action.params
    if action.module == "junction_pipe":
        params = canonicalize_junction_params(params)
        action.params.update(params)
    length = _float_param(params, "length", errors)
    if length is not None and length <= 0:
        errors.append("length must be greater than zero")

    outer_diameter_key = _first_present(params, "outer_diameter", "diameter_in")
    wall_thickness_key = _first_present(params, "wall_thickness", "wall_thickness_in")
    outer_diameter = (
        _float_param(params, outer_diameter_key, errors) if outer_diameter_key else None
    )
    wall_thickness = (
        _float_param(params, wall_thickness_key, errors) if wall_thickness_key else None
    )
    if outer_diameter is not None and wall_thickness is not None:
        if outer_diameter <= wall_thickness * 2:
            errors.append("outer diameter must be greater than twice wall thickness")
        if action.module in {
            "route",
            "transition",
            "junction",
            "connect_ports",
            "terminate",
            "inline_component",
        } and wall_thickness <= 0:
            errors.append("hollow pipe wall thickness must be greater than zero")
        if action.module in {
            "route",
            "transition",
            "junction",
            "connect_ports",
            "terminate",
            "inline_component",
        } and target_port is not None:
            if abs(outer_diameter - target_port.outer_diameter) > state.modeling_tolerance:
                errors.append("authored inlet outer diameter does not match target port")
            if abs(wall_thickness - target_port.wall_thickness) > state.modeling_tolerance:
                errors.append("authored inlet wall thickness does not match target port")

    if action.module == "bend_pipe":
        angle = _float_param(params, "angle", errors)
        if angle is not None and (angle <= 0 or angle > 180):
            errors.append("bend angle must be in (0, 180]")
        bend_radius = _float_param(params, "bend_radius", errors)
        if bend_radius is not None and bend_radius <= 0:
            errors.append("bend radius must be greater than zero")
        outer_radius = (outer_diameter or 0.0) / 2.0
        if (
            bend_radius is not None
            and outer_radius > 0
            and bend_radius <= outer_radius + state.modeling_tolerance
        ):
            errors.append(
                "bend radius must exceed the outer pipe radius by the modeling tolerance"
            )
        segment_resolution = _int_param(params, "segment_resolution", errors)
        if segment_resolution is not None and segment_resolution < 4:
            errors.append("segment_resolution must be at least 4")

    if action.module == "junction_pipe":
        branch_count = _int_param(params, "branch_count", errors)
        if branch_count is not None and branch_count < 1:
            errors.append("branch_count must be at least 1")
        branch_angles = params.get("branch_angles")
        outlet_vectors = _vector_list_param(params, "outlet_vectors", errors)
        required_vectors = _vector_list_param(params, "required_outlet_vectors", errors)
        vector_count = len(outlet_vectors or required_vectors)
        if outlet_vectors and required_vectors and not _same_vector_list(
            outlet_vectors,
            required_vectors,
        ):
            errors.append("outlet_vectors must match required_outlet_vectors when both are provided")
        if vector_count == 0 and (not isinstance(branch_angles, list) or not branch_angles):
            errors.append("branch_angles must be a non-empty list")
        elif isinstance(branch_angles, list):
            for index, angle in enumerate(branch_angles):
                numeric_angle = _float_value(angle, errors, f"branch_angles[{index}]")
                if numeric_angle is not None and not math.isfinite(numeric_angle):
                    errors.append(f"branch_angles[{index}] must be finite")
        if branch_count is not None and vector_count and branch_count != vector_count:
            errors.append("branch_count must match explicit outlet vector count")
        include_primary = params.get("include_primary_outlet")
        if include_primary is not None and not isinstance(include_primary, bool):
            errors.append("include_primary_outlet must be a boolean")
        junction_style = params.get("junction_style")
        if junction_style is not None and junction_style not in {"hard_fuse", "smooth_hub"}:
            errors.append("junction_style must be hard_fuse or smooth_hub")
        blend_radius = _float_param(params, "blend_radius", errors)
        if blend_radius is not None and blend_radius <= 0:
            errors.append("blend_radius must be greater than zero")

    if action.module == "reducer_pipe":
        diameter_out = _float_param(params, "diameter_out", errors)
        wall_thickness_out = _float_param(params, "wall_thickness_out", errors)
        diameter_in = _float_param(params, "diameter_in", errors)
        wall_thickness_in = _float_param(params, "wall_thickness_in", errors)
        for key, value in [
            ("diameter_in", diameter_in),
            ("diameter_out", diameter_out),
        ]:
            if value is not None and value <= 0:
                errors.append(f"{key} must be greater than zero")
        for key, value in [
            ("wall_thickness_in", wall_thickness_in),
            ("wall_thickness_out", wall_thickness_out),
        ]:
            if value is not None and value < 0:
                errors.append(f"{key} must be non-negative")
        if diameter_out is not None and wall_thickness_out is not None:
            if diameter_out <= wall_thickness_out * 2:
                errors.append("output diameter must be greater than twice output wall thickness")

    if action.module == "connector_pipe":
        coupling_outer = _float_param(params, "coupling_outer_diameter", errors)
        sleeve_overlap = _float_param(params, "sleeve_overlap", errors)
        if coupling_outer is not None and coupling_outer <= 0:
            errors.append("coupling_outer_diameter must be greater than zero")
        if (
            coupling_outer is not None
            and outer_diameter is not None
            and coupling_outer < outer_diameter
        ):
            errors.append("coupling_outer_diameter must be at least outer_diameter")
        if sleeve_overlap is not None and sleeve_overlap < 0:
            errors.append("sleeve_overlap must be non-negative")

    if action.module == "cap_pipe":
        cap_thickness = _float_param(params, "cap_thickness", errors)
        if cap_thickness is not None and cap_thickness < 0:
            errors.append("cap_thickness must be non-negative")
        if params.get("end_type") == "cap" and cap_thickness is not None and cap_thickness <= 0:
            errors.append("cap_thickness must be greater than zero for capped ends")
        if params.get("end_type") == "cap" and cap_thickness is not None and cap_thickness <= 2.0:
            errors.append("cap_thickness must be greater than bore extension for capped ends")

    if action.module == "route":
        kind = params.get("path_kind")
        if kind == "line":
            route_length = _float_param(params, "length", errors)
            if route_length is not None and route_length <= 0:
                errors.append("route length must be greater than zero")
            _validate_vector_value(params.get("direction"), "direction", errors)
        elif kind == "circular_arc":
            radius = _float_param(params, "bend_radius", errors)
            angle = _float_param(params, "sweep_angle", errors)
            if radius is not None and outer_diameter is not None:
                if radius <= outer_diameter / 2.0 + state.modeling_tolerance:
                    errors.append(
                        "bend_radius must exceed the outer pipe radius by the modeling tolerance"
                    )
            if angle is not None and (abs(angle) <= 1e-6 or abs(angle) >= 360.0):
                errors.append("sweep_angle magnitude must be in (0, 360)")
            _validate_vector_value(params.get("plane_normal"), "plane_normal", errors)
            _validate_vector_value(params.get("terminal_axis"), "terminal_axis", errors)
            try:
                plane_normal, actual_start_tangent, expected_terminal = (
                    canonical_circular_arc_frame(
                        tuple(float(value) for value in params["axis"]),
                        tuple(float(value) for value in params["plane_normal"]),
                        float(angle),
                    )
                )
                route_axis = normalize(tuple(float(value) for value in params["axis"]))
                authored_plane = normalize(
                    tuple(float(value) for value in params["plane_normal"])
                )
                if dot(plane_normal, authored_plane) < 1.0 - 1e-12:
                    errors.append(
                        "resolved plane_normal is not the canonical orthogonalized bend-plane normal"
                    )
                outer_radius = float(outer_diameter or 0.0) / 2.0
                start_rim_error = circular_rim_mismatch(
                    0.0,
                    outer_radius,
                    outer_radius,
                    dot(route_axis, actual_start_tangent),
                )
                if start_rim_error > state.modeling_tolerance:
                    errors.append(
                        "resolved circular-arc start tangent does not match the inlet axis"
                    )
                authored_terminal = normalize(
                    tuple(float(value) for value in params["terminal_axis"])
                )
                terminal_rim_error = circular_rim_mismatch(
                    0.0,
                    outer_radius,
                    outer_radius,
                    dot(expected_terminal, authored_terminal),
                )
                if terminal_rim_error > state.modeling_tolerance:
                    errors.append(
                        "resolved terminal_axis invariant mismatch: expected "
                        f"{[round(value, 12) for value in expected_terminal]}, "
                        f"actual {[round(value, 12) for value in authored_terminal]}, "
                        f"rim_error={terminal_rim_error:.12g}, "
                        f"tolerance={state.modeling_tolerance:.12g}"
                    )
            except (KeyError, TypeError):
                # Missing/malformed values already receive field-specific errors above.
                pass
            except ValueError as exc:
                errors.append(
                    "resolved circular_arc canonical frame is invalid: " + str(exc)
                )
        elif kind == "spline":
            waypoints = _vector_list_param(
                params, "waypoints", errors, allow_zero=True
            )
            if len(waypoints) < 2:
                errors.append("spline route requires at least two valid waypoints")
            _validate_vector_value(params.get("initial_tangent"), "initial_tangent", errors)
            _validate_vector_value(params.get("final_tangent"), "final_tangent", errors)
            try:
                expected_initial = normalize(
                    tuple(float(value) for value in params["axis"])
                )
                resolved_initial = normalize(
                    tuple(float(value) for value in params["initial_tangent"])
                )
                if dot(expected_initial, resolved_initial) < 1.0 - 1e-12:
                    errors.append(
                        "resolved spline initial_tangent invariant mismatch: it must "
                        "equal the selected inlet axis"
                    )
            except (KeyError, TypeError, ValueError):
                # Field-level validation above reports missing or malformed vectors.
                pass
            try:
                affected_goal_ids = set(action.affected_goal_ids)
                route_goals = [
                    goal
                    for goal in state.remaining_goals
                    if goal.goal_id in affected_goal_ids and goal.type == "route"
                ]
                terminal_goal = route_goals[-1] if route_goals else None
                if (
                    terminal_goal is not None
                    and terminal_goal.terminal_axis is not None
                ):
                    expected_final = normalize(vec(terminal_goal.terminal_axis))
                elif (
                    terminal_goal is not None
                    and len(terminal_goal.required_waypoints) >= 2
                ):
                    expected_final = normalize(
                        sub(
                            vec(terminal_goal.required_waypoints[-1]),
                            vec(terminal_goal.required_waypoints[-2]),
                        )
                    )
                elif (
                    terminal_goal is not None
                    and len(terminal_goal.required_waypoints) == 1
                ):
                    final_point = vec(terminal_goal.required_waypoints[0])
                    expected_final = normalize(
                        final_point
                        if terminal_goal.waypoint_frame == "relative_to_target"
                        else sub(final_point, vec(params["start_position"]))
                    )
                else:
                    expected_final = normalize(sub(waypoints[-1], waypoints[-2]))
                resolved_final = normalize(vec(params["final_tangent"]))
                if dot(expected_final, resolved_final) < 1.0 - 1e-12:
                    errors.append(
                        "resolved spline final_tangent invariant mismatch: it must "
                        "equal the immutable terminal axis or final required "
                        "waypoint chord"
                    )
            except (IndexError, KeyError, TypeError, ValueError):
                # Field-level validation above reports malformed route geometry.
                pass
            minimum_radius = _float_param(params, "minimum_curvature_radius", errors)
            if minimum_radius is not None and outer_diameter is not None:
                if (
                    minimum_radius
                    <= outer_diameter / 2.0 + state.modeling_tolerance
                ):
                    errors.append(
                        "minimum_curvature_radius must exceed the outer pipe radius by the modeling tolerance"
                    )
            # FreeCAD 생성기와 동일한 cubic/handle 정책을 kernel 없이 계산한다.
            # 실패가 확실한 waypoint 후보에 MCP 토큰과 B-Rep 시간을 쓰지 않고,
            # expected/actual 수치를 그대로 다음 planner repair에 돌려준다.
            if minimum_radius is not None:
                try:
                    prediction = predict_c1_spline(
                        [
                            vec(params["start_position"]),
                            *[vec(point) for point in params["waypoints"]],
                        ],
                        vec(params["initial_tangent"]),
                        vec(params["final_tangent"]),
                        modeling_tolerance=state.modeling_tolerance,
                    )
                    if (
                        prediction.minimum_radius + state.modeling_tolerance
                        < minimum_radius
                    ):
                        errors.append(
                            "spline curvature preflight failed: "
                            f"expected minimum_radius>={minimum_radius:.12g} mm; "
                            f"actual minimum_radius={prediction.minimum_radius:.12g} mm; "
                            f"minimum_chord={prediction.minimum_chord:.12g} mm; "
                            f"curve_length={prediction.curve_length:.12g} mm. "
                            "Replace or move non-required waypoints with a few broad, "
                            "well-separated points; do not add closely spaced points, "
                            "because they shorten cubic handles and increase curvature."
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append("spline curvature preflight could not run: " + str(exc))
            if params.get("interpolation") != "bspline":
                errors.append("spline route interpolation must be bspline")
            if not isinstance(params.get("frenet"), bool):
                errors.append("spline route frenet must be a boolean")
            if params.get("waypoint_frame", "global") != "global":
                errors.append(
                    "resolved spline waypoint_frame must be canonical global"
                )
        else:
            errors.append("route path_kind must be line, circular_arc, or spline")

    if action.module == "transition":
        diameter_out = _float_param(params, "diameter_out", errors)
        wall_out = _float_param(params, "wall_thickness_out", errors)
        if diameter_out is not None and wall_out is not None:
            if diameter_out <= 2.0 * wall_out:
                errors.append("output diameter must be greater than twice output wall thickness")
            if wall_out <= 0:
                errors.append("output wall thickness must be greater than zero")
        _validate_vector_value(params.get("offset"), "offset", errors, allow_zero=True)
        _validate_transition_taper_geometry(
            diameter_in=outer_diameter,
            wall_in=wall_thickness,
            diameter_out=diameter_out,
            wall_out=wall_out,
            axial_length=length,
            offset=_canonical_vector(params.get("offset")),
            errors=errors,
        )
        try:
            axis = normalize(_canonical_vector(params.get("axis")) or (0.0, 0.0, 0.0))
            offset = _canonical_vector(params.get("offset"))
            if length is not None and offset is not None:
                axial_offset = sum(
                    offset[index] * axis[index] for index in range(3)
                )
                if abs(axial_offset) > 1e-6:
                    errors.append(
                        "transition offset must be perpendicular to the pipe axis; use length for axial displacement"
                    )
                displacement = tuple(
                    axis[index] * length + offset[index] for index in range(3)
                )
                if math.sqrt(sum(value * value for value in displacement)) <= 1e-6:
                    errors.append("transition start and end profiles must not coincide")
                if sum(displacement[index] * axis[index] for index in range(3)) <= 1e-6:
                    errors.append("transition must make positive axial progress")
        except ValueError:
            errors.append("transition axis must be a finite non-zero vector")
        _validate_plain_connector_fields(
            params,
            "connector_type_out",
            "connector_gender_out",
            "connector_standard_out",
            errors,
        )

    if action.module == "junction":
        if params.get("blend_mode") not in {"hard", "fillet"}:
            errors.append("junction blend_mode must be hard or fillet")
        for key in ("blend_radius", "inner_blend_radius", "max_hub_radius"):
            value = _float_param(params, key, errors)
            if value is not None and value <= 0:
                errors.append(f"{key} must be greater than zero")
        blend_radius = _float_param(params, "blend_radius", errors)
        inner_blend_radius = _float_param(params, "inner_blend_radius", errors)
        max_hub = _float_param(params, "max_hub_radius", errors)
        if blend_radius is not None and max_hub is not None and blend_radius > max_hub:
            errors.append("blend_radius must not exceed max_hub_radius")
        if inner_blend_radius is not None and max_hub is not None and inner_blend_radius > max_hub:
            errors.append("inner_blend_radius must not exceed max_hub_radius")
        outlets = params.get("outlets")
        if isinstance(outlets, list):
            if len(outlets) != 2:
                errors.append(
                    "junction outlets must contain exactly two entries; "
                    "compose additional binary junctions for higher-degree branching"
                )
            primary_count = sum(
                1
                for outlet in outlets
                if isinstance(outlet, dict) and outlet.get("role") == "primary"
            )
            if primary_count > 1:
                errors.append("junction outlets may contain at most one primary role")
            normalized_axes: list[tuple[float, float, float]] = []
            for index, outlet in enumerate(outlets):
                if not isinstance(outlet, dict):
                    errors.append(f"outlets[{index}] must be an object")
                    continue
                axis_value = _validate_vector_value(
                    outlet.get("axis"), f"outlets[{index}].axis", errors
                )
                if axis_value is not None:
                    normalized_axis = normalize(axis_value)
                    normalized_axes.append(normalized_axis)
                    if target_port is not None and dot(
                        normalized_axis,
                        normalize(
                            tuple(float(value) for value in target_port.axis)
                        ),
                    ) <= -0.999:
                        errors.append(
                            f"outlets[{index}].axis must not retrace the consumed "
                            "target-port axis"
                        )
                length_value = _float_value(
                    outlet.get("length"), errors, f"outlets[{index}].length"
                )
                od_value = _float_value(
                    outlet.get("outer_diameter"),
                    errors,
                    f"outlets[{index}].outer_diameter",
                )
                wall_value = _float_value(
                    outlet.get("wall_thickness"),
                    errors,
                    f"outlets[{index}].wall_thickness",
                )
                if length_value is not None and length_value <= 0:
                    errors.append(f"outlets[{index}].length must be greater than zero")
                if od_value is not None and wall_value is not None and od_value <= 2 * wall_value:
                    errors.append(
                        f"outlets[{index}] outer diameter must exceed twice wall thickness"
                    )
                if wall_value is not None and wall_value <= 0:
                    errors.append(
                        f"outlets[{index}].wall_thickness must be greater than zero"
                    )
                if (
                    length_value is not None
                    and max_hub is not None
                    and length_value <= max_hub + 1e-6
                ):
                    errors.append(
                        f"outlets[{index}].length must extend beyond max_hub_radius"
                    )
                _validate_plain_connector_fields(
                    outlet,
                    "connector_type",
                    "connector_gender",
                    "connector_standard",
                    errors,
                    label=f"outlets[{index}]",
                )
            for left_index, left in enumerate(normalized_axes):
                for right_index, right in enumerate(normalized_axes[left_index + 1 :], start=left_index + 1):
                    if sum(a * b for a, b in zip(left, right)) > 0.999:
                        errors.append(
                            "junction outlet axes must be distinct: "
                            f"outlets[{left_index}] and outlets[{right_index}]"
                        )

    if action.module == "connect_ports":
        path_kind = params.get("path_kind")
        if path_kind not in {"line", "circular_arc", "spline"}:
            errors.append("connect_ports path_kind must be line, circular_arc, or spline")
        other_id = params.get("other_port_id")
        other_port = next((port for port in state.open_ports if port.id == other_id), None)
        if other_id == action.target_port:
            errors.append("connect_ports requires distinct ports")
        if other_port is None:
            errors.append(f"Second target port is not open: {other_id}")
        elif outer_diameter is not None and wall_thickness is not None:
            if abs(other_port.outer_diameter - outer_diameter) > state.modeling_tolerance:
                errors.append("connect_ports endpoint outer diameters are incompatible")
            if abs(other_port.wall_thickness - wall_thickness) > state.modeling_tolerance:
                errors.append("connect_ports endpoint wall thicknesses are incompatible")
        waypoints = _vector_list_param(
            params, "waypoints", errors, allow_zero=True
        )
        if path_kind == "line":
            if waypoints:
                errors.append("line connect_ports does not accept waypoints")
            if any(
                params.get(name) is not None
                for name in (
                    "initial_tangent",
                    "final_tangent",
                    "interpolation",
                    "frenet",
                    "minimum_curvature_radius",
                )
            ):
                errors.append("line connect_ports does not accept curve parameters")
        elif path_kind == "circular_arc":
            if len(waypoints) != 1:
                errors.append(
                    "circular_arc connect_ports requires exactly one arc waypoint"
                )
            if any(
                params.get(name) is not None
                for name in (
                    "initial_tangent",
                    "final_tangent",
                    "interpolation",
                    "frenet",
                )
            ):
                errors.append(
                    "circular_arc connect_ports does not accept spline parameters"
                )
            minimum_radius = _float_param(
                params, "minimum_curvature_radius", errors
            )
            if (
                minimum_radius is not None
                and outer_diameter is not None
                and minimum_radius
                <= outer_diameter / 2.0 + state.modeling_tolerance
            ):
                errors.append(
                    "connect_ports minimum_curvature_radius must exceed the pipe outer radius by the modeling tolerance"
                )
        else:
            _validate_vector_value(
                params.get("initial_tangent"), "initial_tangent", errors
            )
            _validate_vector_value(
                params.get("final_tangent"), "final_tangent", errors
            )
            if params.get("interpolation") != "bspline":
                errors.append("curved connect_ports interpolation must be bspline")
            if not isinstance(params.get("frenet"), bool):
                errors.append("curved connect_ports frenet must be a boolean")
            if not waypoints:
                errors.append("curved connect_ports requires at least one valid waypoint")
            minimum_radius = _float_param(
                params, "minimum_curvature_radius", errors
            )
            if (
                minimum_radius is not None
                and outer_diameter is not None
                and minimum_radius
                <= outer_diameter / 2.0 + state.modeling_tolerance
            ):
                errors.append(
                    "connect_ports minimum_curvature_radius must exceed the pipe outer radius by the modeling tolerance"
                )
        if target_port is not None and other_port is not None:
            if path_kind == "spline":
                try:
                    initial = normalize(
                        tuple(float(value) for value in params["initial_tangent"])
                    )
                    final = normalize(
                        tuple(float(value) for value in params["final_tangent"])
                    )
                    expected_initial = normalize(
                        tuple(float(value) for value in target_port.axis)
                    )
                    expected_final = tuple(
                        -value
                        for value in normalize(
                            tuple(float(value) for value in other_port.axis)
                        )
                    )
                    if dot(initial, expected_initial) < 1.0 - 1e-12:
                        errors.append(
                            "resolved connect_ports initial_tangent invariant mismatch"
                        )
                    if dot(final, expected_final) < 1.0 - 1e-12:
                        errors.append(
                            "resolved connect_ports final_tangent invariant mismatch"
                        )
                except (KeyError, TypeError, ValueError):
                    # Field-level validation above reports malformed tangents.
                    pass
            path_points = [
                tuple(float(value) for value in target_port.position),
                *waypoints,
                tuple(float(value) for value in other_port.position),
            ]
            if any(
                math.sqrt(sum((b - a) ** 2 for a, b in zip(left, right)))
                <= 1e-6
                for left, right in zip(path_points, path_points[1:])
            ):
                errors.append("connect_ports path contains coincident consecutive points")
            if path_kind == "circular_arc" and len(path_points) == 3:
                first = tuple(
                    path_points[1][index] - path_points[0][index]
                    for index in range(3)
                )
                second = tuple(
                    path_points[2][index] - path_points[0][index]
                    for index in range(3)
                )
                cross_value = (
                    first[1] * second[2] - first[2] * second[1],
                    first[2] * second[0] - first[0] * second[2],
                    first[0] * second[1] - first[1] * second[0],
                )
                if math.sqrt(sum(value * value for value in cross_value)) <= 1e-6:
                    errors.append(
                        "circular_arc connect_ports start, midpoint, and end must be non-collinear"
                    )

    if action.module == "terminate":
        if params.get("termination_type") not in {"cap", "plug"}:
            errors.append("termination_type must be cap or plug")
        thickness = _float_param(params, "thickness", errors)
        if thickness is not None and thickness <= 0:
            errors.append("termination thickness must be greater than zero")

    if action.module == "inline_component":
        component_type = params.get("component_type")
        if component_type not in SUPPORTED_INLINE_COMPONENTS:
            errors.append("unsupported inline component_type")
        body_outer = _float_param(params, "body_outer_diameter", errors)
        body_offset = _float_param(params, "body_start_offset", errors)
        body_length = _float_param(params, "body_length", errors)
        if body_outer is not None and outer_diameter is not None:
            if body_outer <= outer_diameter:
                errors.append("component body_outer_diameter must exceed pipe diameter")
        if body_offset is not None and body_offset < 0:
            errors.append("component body_start_offset must be non-negative")
        if body_length is not None and body_length <= 0:
            errors.append("component body_length must be greater than zero")
        if (
            length is not None
            and body_offset is not None
            and body_length is not None
            and body_offset + body_length > length + 1e-9
        ):
            errors.append("component body must remain inside its axial length")
        if length is not None and body_offset is not None and body_length is not None:
            body_end = body_offset + body_length
            if component_type == "flange" and not (
                body_offset <= 1e-9 or abs(body_end - length) <= 1e-9
            ):
                errors.append("flange collar must touch one axial end")
            if component_type == "coupling" and not (
                body_offset <= 1e-9 and abs(body_length - length) <= 1e-9
            ):
                errors.append("coupling sleeve must span the axial length")
            if component_type in {"union", "valve"} and not (
                body_offset > 1e-9 and body_end < length - 1e-9
            ):
                errors.append(f"{component_type} body must lie between two pipe necks")
        actuator_diameter = _float_param(params, "actuator_diameter", errors)
        actuator_height = _float_param(params, "actuator_height", errors)
        actuator_axis = params.get("actuator_axis")
        if component_type == "valve":
            if actuator_diameter is None or actuator_height is None or actuator_axis is None:
                errors.append("valve requires positive actuator dimensions and axis")
            axis_value = _validate_vector_value(actuator_axis, "actuator_axis", errors)
            pipe_axis = _canonical_vector(params.get("axis"))
            if axis_value is not None and pipe_axis is not None:
                normalized_actuator = normalize(axis_value)
                normalized_pipe = normalize(pipe_axis)
                if abs(sum(a * b for a, b in zip(normalized_actuator, normalized_pipe))) > 1e-3:
                    errors.append("valve actuator_axis must be perpendicular to pipe axis")
        elif actuator_diameter is not None or actuator_height is not None or actuator_axis is not None:
            errors.append("actuator parameters are valid only for valve")
        flange_count = _int_param(params, "flange_bolt_count", errors)
        flange_circle = _float_param(params, "flange_bolt_circle_diameter", errors)
        flange_hole = _float_param(params, "flange_bolt_hole_diameter", errors)
        flange_reference = params.get("flange_reference_axis")
        if component_type == "flange":
            if flange_count is None or not 3 <= flange_count <= 32:
                errors.append("flange_bolt_count must be in [3, 32]")
            if (
                flange_circle is None
                or flange_hole is None
                or body_outer is None
                or outer_diameter is None
                or flange_hole <= 0
                or flange_circle - flange_hole <= outer_diameter
                or flange_circle + flange_hole >= body_outer
            ):
                errors.append("flange bolt circle/holes must fit inside the flange annulus")
            reference_value = _validate_vector_value(
                flange_reference, "flange_reference_axis", errors
            )
            pipe_axis = _canonical_vector(params.get("axis"))
            if reference_value is not None and pipe_axis is not None:
                if abs(sum(a * b for a, b in zip(normalize(reference_value), normalize(pipe_axis)))) > 1e-3:
                    errors.append("flange_reference_axis must be perpendicular to pipe axis")
        elif (
            flange_count is not None
            or flange_circle is not None
            or flange_hole is not None
            or flange_reference is not None
        ):
            errors.append("flange bolt parameters are valid only for flange")
        ring_outer = _float_param(params, "union_ring_outer_diameter", errors)
        ring_length = _float_param(params, "union_ring_length", errors)
        if component_type == "union":
            if (
                ring_outer is None
                or body_outer is None
                or ring_outer <= body_outer
                or ring_length is None
                or body_length is None
                or ring_length * 2.0 > body_length
            ):
                errors.append("union rings must exceed the body and fit twice inside body_length")
        elif ring_outer is not None or ring_length is not None:
            errors.append("union ring parameters are valid only for union")
        connector_type = params.get("connector_type_out")
        if connector_type not in {"plain", component_type}:
            errors.append(
                "inline_component connector_type_out must be plain or match component_type"
            )
        if connector_type == "plain":
            _validate_plain_connector_fields(
                params,
                "connector_type_out",
                "connector_gender_out",
                "connector_standard_out",
                errors,
            )
        elif any(
            goal.goal_id not in completed
            for goal in state.remaining_goals
            if goal.goal_id is not None
        ):
            errors.append(
                "a non-plain inline_component output is allowed only as the final authored terminal"
            )
        if (
            target_port is not None
            and target_port.connector_type != "plain"
            and body_offset is not None
            and body_offset > 1e-9
        ):
            errors.append(
                "a non-plain inlet connector requires component geometry touching the input end"
            )
        if connector_type != "plain" and (
            component_type not in {"flange", "coupling"}
            or body_offset is None
            or body_length is None
            or length is None
            or abs(body_offset + body_length - length) > 1e-9
        ):
            errors.append(
                "a non-plain output connector requires flange/coupling geometry touching the output end"
            )

    return ValidationResult(valid=not errors, errors=errors)


def _validate_completion_claims(
    completed_goal_ids: set[str],
    state: PipeState,
    errors: list[str],
) -> None:
    groups: dict[str, list[str]] = {}
    for goal in state.remaining_goals:
        if goal.goal_id not in completed_goal_ids:
            continue
        if goal.type in {"move", "route", "connector"} and goal.length is not None:
            groups.setdefault("linear_length", []).append(str(goal.goal_id))
        if goal.type == "turn" and goal.angle is not None:
            groups.setdefault("turn_angle", []).append(str(goal.goal_id))
        if goal.type == "branch":
            groups.setdefault("branch_topology", []).append(str(goal.goal_id))
        if goal.type == "diameter_change":
            groups.setdefault("diameter_change", []).append(str(goal.goal_id))
        if goal.type == "connect":
            groups.setdefault("connect_topology", []).append(str(goal.goal_id))
        if goal.type == "end":
            groups.setdefault("end_topology", []).append(str(goal.goal_id))
        if goal.type == "connector" and goal.component is not None:
            groups.setdefault("component_instance", []).append(str(goal.goal_id))
    for group, goal_ids in groups.items():
        if len(goal_ids) > 1:
            errors.append(
                f"one action cannot complete multiple {group} goals without "
                f"double-counting geometry: {goal_ids}"
            )


def _validate_atomic_goal_binding(
    module: str,
    params: dict[str, Any],
    affected_goal_ids: set[str],
    completed_goal_ids: set[str],
    state: PipeState,
    errors: list[str],
) -> None:
    """Bind one-shot topology/component goals to the action that realizes them.

    Route and move goals may intentionally span several LLM actions.  The goals
    below describe an indivisible topology or fitting operation; letting a later
    unrelated action complete them would make goal accounting lie about the CAD.
    """

    atomic_types = {
        "turn",
        "branch",
        "diameter_change",
        "connect",
        "end",
        "connector",
    }
    for goal in state.remaining_goals:
        goal_id = goal.goal_id
        if goal_id is None or goal_id not in affected_goal_ids:
            continue
        if goal.type not in atomic_types:
            continue
        if goal_id not in completed_goal_ids:
            errors.append(
                f"atomic goal {goal_id} ({goal.type}) must be completed by the "
                "same action that affects it"
            )
        if (
            goal.type == "connector"
            and goal.component is not None
            and params.get("component_type") != goal.component
        ):
            errors.append(
                f"connector goal {goal_id} requires component_type "
                f"{goal.component}, not {params.get('component_type')}"
            )
        if goal.type == "branch":
            if module != "junction":
                errors.append(
                    f"branch goal {goal_id} must be realized by junction, not {module}"
                )
                continue
            expected_branch_count = (
                goal.branch_count
                or len(goal.required_outlets)
                or len(goal.required_outlet_vectors)
                or len(goal.required_outlet_directions)
            )
            if goal.include_primary_outlet is not None:
                expected_primary_count = int(goal.include_primary_outlet)
            elif expected_branch_count in {1, 2}:
                # 구형/테스트 Goal에는 명시 플래그가 없을 수 있으므로 이진
                # junction의 총 outlet 수(2)에서 역할 수를 결정한다.
                expected_primary_count = 2 - expected_branch_count
            else:
                expected_primary_count = int(bool(params.get("include_primary_outlet")))
            outlets = params.get("outlets")
            roles = []
            if isinstance(outlets, list):
                roles = [
                    outlet.get("role")
                    for outlet in outlets
                    if isinstance(outlet, dict)
                ]
            actual_primary_count = roles.count("primary")
            actual_branch_count = roles.count("branch")
            if (
                actual_primary_count != expected_primary_count
                or actual_branch_count != expected_branch_count
                or len(roles) != expected_primary_count + expected_branch_count
            ):
                errors.append(
                    f"junction outlet roles conflict with branch goal {goal_id}: "
                    f"expected primary={expected_primary_count}, "
                    f"branch={expected_branch_count}; actual roles={roles}"
                )


def _validate_goal_order_claims(
    affected_goal_ids: set[str],
    completed_goal_ids: set[str],
    state: PipeState,
    errors: list[str],
) -> None:
    completed_history = {
        goal_id
        for historical_action in state.action_history
        for goal_id in historical_action.completed_goal_ids
    }
    for index, goal in enumerate(state.remaining_goals):
        goal_id = goal.goal_id
        if goal_id is None or goal_id not in affected_goal_ids:
            continue
        missing_dependencies = set(goal.depends_on_goal_ids) - completed_history
        if missing_dependencies:
            errors.append(
                f"goal {goal_id} has incomplete dependencies: {sorted(missing_dependencies)}"
            )
        if goal.allow_parallel:
            continue
        bypassed = [
            str(prior.goal_id)
            for prior in state.remaining_goals[:index]
            if prior.goal_id is not None
            and prior.goal_id not in completed_goal_ids
        ]
        if bypassed:
            errors.append(
                f"goal {goal_id} cannot bypass earlier pending goals {bypassed}"
            )


def _validate_plain_connector_fields(
    params: dict[str, Any],
    type_key: str,
    gender_key: str,
    standard_key: str,
    errors: list[str],
    *,
    label: str = "output connector",
) -> None:
    if params.get(type_key) != "plain":
        errors.append(f"{label} must be plain; use inline_component for a fitting")
        return
    if params.get(gender_key) != "neutral":
        errors.append(f"{label} plain connector gender must be neutral")
    if params.get(standard_key) is not None:
        errors.append(f"{label} plain connector standard must be null")


def _validate_vector_value(
    value: Any,
    label: str,
    errors: list[str],
    *,
    allow_zero: bool = False,
) -> tuple[float, float, float] | None:
    vector = _canonical_vector(value)
    if vector is None:
        errors.append(f"{label} must be a 3D vector")
        return None
    if not allow_zero and math.sqrt(sum(component * component for component in vector)) <= 1e-9:
        errors.append(f"{label} must not be a zero vector")
        return None
    return vector


def _first_present(params: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if params.get(key) is not None:
            return key
    return None


def _validate_transition_taper_geometry(
    *,
    diameter_in: float | None,
    wall_in: float | None,
    diameter_out: float | None,
    wall_out: float | None,
    axial_length: float | None,
    offset: tuple[float, float, float] | None,
    errors: list[str],
) -> None:
    values = (
        diameter_in,
        wall_in,
        diameter_out,
        wall_out,
        axial_length,
    )
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return
    diameter_in = float(diameter_in)
    wall_in = float(wall_in)
    diameter_out = float(diameter_out)
    wall_out = float(wall_out)
    axial_length = float(axial_length)
    if axial_length <= 0.0:
        return
    bore_in = diameter_in / 2.0 - wall_in
    bore_out = diameter_out / 2.0 - wall_out
    if min(bore_in, bore_out) <= 0.0:
        return
    offset_length = vector_length(offset or (0.0, 0.0, 0.0))
    outer_lateral_change = offset_length + abs(diameter_out - diameter_in) / 2.0
    bore_lateral_change = offset_length + abs(bore_out - bore_in)
    governing_change = max(outer_lateral_change, bore_lateral_change)
    half_angle = math.degrees(math.atan2(governing_change, axial_length))
    if half_angle > MAX_TRANSITION_HALF_ANGLE_DEGREES + 1e-9:
        errors.append(
            "transition taper is too abrupt: governing OD/bore half-angle "
            f"{half_angle:.6g}° exceeds the general "
            f"{MAX_TRANSITION_HALF_ANGLE_DEGREES:g}° limit; increase length, "
            "reduce section change, or reduce eccentric offset"
        )


def _float_param(
    params: dict[str, Any],
    key: str,
    errors: list[str],
) -> float | None:
    value = params.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        errors.append(f"Invalid numeric param for {key}: {value!r}")
        return None
    if not math.isfinite(result):
        errors.append(f"Invalid numeric param for {key}: {value!r}")
        return None
    return result


def _float_value(
    value: Any,
    errors: list[str],
    label: str,
) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        errors.append(f"Invalid numeric param for {label}: {value!r}")
        return None
    if not math.isfinite(result):
        errors.append(f"Invalid numeric param for {label}: {value!r}")
        return None
    return result


def _canonical_vector_list(value: Any) -> Any:
    if isinstance(value, (str, dict)):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return value
    canonical: list[Any] = []
    for item in items:
        converted = _canonical_vector(item)
        canonical.append(converted if converted is not None else item)
    return canonical


def _canonical_vector(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, str):
        return _named_vector(value)
    if isinstance(value, dict):
        for key in ("vector", "axis", "direction", "name", "label"):
            if key in value:
                nested = _canonical_vector(value[key])
                if nested is not None:
                    return nested
        if all(key in value for key in ("x", "y", "z")):
            return _numeric_vector((value["x"], value["y"], value["z"]))
        if all(key in value for key in ("X", "Y", "Z")):
            return _numeric_vector((value["X"], value["Y"], value["Z"]))
        return None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return _numeric_vector(value)
    return None


def _numeric_vector(value: Any) -> tuple[float, float, float] | None:
    try:
        x, y, z = value
        return (float(x), float(y), float(z))
    except (TypeError, ValueError):
        return None


def _named_vector(value: str) -> tuple[float, float, float] | None:
    key = _normalized_token(value)
    table = {
        "upperleft": (-1.0, 0.0, 1.0),
        "leftupper": (-1.0, 0.0, 1.0),
        "upleft": (-1.0, 0.0, 1.0),
        "lowerleft": (-1.0, 0.0, -1.0),
        "leftlower": (-1.0, 0.0, -1.0),
        "downleft": (-1.0, 0.0, -1.0),
        "upperright": (1.0, 0.0, 1.0),
        "rightupper": (1.0, 0.0, 1.0),
        "upright": (1.0, 0.0, 1.0),
        "lowerright": (1.0, 0.0, -1.0),
        "rightlower": (1.0, 0.0, -1.0),
        "downright": (1.0, 0.0, -1.0),
        "+x": (1.0, 0.0, 0.0),
        "-x": (-1.0, 0.0, 0.0),
        "+y": (0.0, 1.0, 0.0),
        "-y": (0.0, -1.0, 0.0),
        "+z": (0.0, 0.0, 1.0),
        "-z": (0.0, 0.0, -1.0),
        "up": (0.0, 0.0, 1.0),
        "down": (0.0, 0.0, -1.0),
        "left": (-1.0, 0.0, 0.0),
        "right": (1.0, 0.0, 0.0),
    }
    return table.get(key)


def _canonical_junction_style(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    key = _normalized_token(value)
    smooth_aliases = {
        "smooth",
        "smoothhub",
        "smoothy",
        "smoothyblend",
        "smoothyshaped",
        "blended",
        "blend",
        "yblend",
        "yshaped",
    }
    hard_aliases = {
        "hard",
        "hardfuse",
        "boolean",
        "raw",
        "straightfuse",
    }
    if key in smooth_aliases:
        return "smooth_hub"
    if key in hard_aliases:
        return "hard_fuse"
    return value


def _normalized_token(value: str) -> str:
    raw = value.strip().lower()
    if raw in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        return raw
    return "".join(character for character in raw if character.isalnum())


def _vector_list_param(
    params: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    allow_zero: bool = False,
) -> list[tuple[float, float, float]]:
    value = params.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f"{key} must be a list of 3D vectors")
        return []
    vectors: list[tuple[float, float, float]] = []
    for index, raw_vector in enumerate(value):
        if not isinstance(raw_vector, (list, tuple)) or len(raw_vector) != 3:
            errors.append(f"{key}[{index}] must be a 3D vector")
            continue
        vector_values: list[float] = []
        for component_index, component in enumerate(raw_vector):
            numeric = _float_value(
                component,
                errors,
                f"{key}[{index}][{component_index}]",
            )
            if numeric is not None:
                vector_values.append(numeric)
        if len(vector_values) != 3:
            continue
        if not all(math.isfinite(component) for component in vector_values):
            errors.append(f"{key}[{index}] components must be finite")
            continue
        if (
            not allow_zero
            and math.sqrt(sum(component * component for component in vector_values))
            <= 1e-9
        ):
            errors.append(f"{key}[{index}] must not be a zero vector")
            continue
        vectors.append((vector_values[0], vector_values[1], vector_values[2]))
    return vectors


def _same_vector_list(
    left: list[tuple[float, float, float]],
    right: list[tuple[float, float, float]],
) -> bool:
    if len(left) != len(right):
        return False
    for left_vector, right_vector in zip(left, right):
        if _normalized_dot(left_vector, right_vector) < 0.999:
            return False
    return True


def _normalized_dot(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    left_size = math.sqrt(sum(component * component for component in left))
    right_size = math.sqrt(sum(component * component for component in right))
    if left_size <= 1e-9 or right_size <= 1e-9:
        return -1.0
    return sum(
        left_component * right_component
        for left_component, right_component in zip(left, right)
    ) / (left_size * right_size)


def _int_param(
    params: dict[str, Any],
    key: str,
    errors: list[str],
) -> int | None:
    value = params.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        errors.append(f"Invalid integer param for {key}: {value!r}")
        return None
