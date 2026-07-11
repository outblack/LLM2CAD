from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import ValidationError

from cadgen.gemini_client import (
    GeminiInvalidRequestError,
    GeminiRequestError,
    MAX_STRUCTURED_NUMBER_LITERAL_BYTES,
    _decode_decimal_numbers,
    gemini_json_schema,
)
from cadgen.config import load_settings
from cadgen.pipeline import (
    PLANNER_PREFERRED_NUMBER_LITERAL_BYTES,
    _PlannerStagnationError,
    _compact_freecad_failure_evidence,
    _freecad_repair_contract,
    _plan_action,
    _planner_numeric_literals,
    _planner_repair_context,
    _rebind_replay_draft,
)
from cadgen.registry import validate_action, validate_draft
from cadgen.schemas import (
    ActionDraft,
    ActionAttempt,
    CorePlannerDecision,
    GeometricConstraint,
    GlobalSpec,
    Goal,
    IntentResult,
    PipeState,
    PlannerDecision,
    Port,
    RouteParamsV2,
)
from cadgen.state import StateEngine
from cadgen.static_validation import build_step_verification
from cadgen.vector import direction_to_vector


def _planner_route_variants() -> dict[str, dict]:
    schema = gemini_json_schema(PlannerDecision)
    params_schema = schema["$defs"]["RouteChoice"]["properties"]["params"]
    variants: dict[str, dict] = {}
    for branch in params_schema["anyOf"]:
        name = branch["$ref"].rsplit("/", 1)[-1]
        variant = schema["$defs"][name]
        path_kind = variant["properties"]["path_kind"]["enum"][0]
        variants[path_kind] = variant
    return variants


def test_unknown_direction_token_does_not_silently_become_positive_x():
    """알 수 없는 방향은 기본 축으로 바뀌지 않고 즉시 거절되어야 한다."""

    with pytest.raises(ValueError, match="unknown direction token"):
        direction_to_vector("SIDEWAYS")

    assert direction_to_vector("SIDEWAYS", default=(0.0, 1.0, 0.0)) == (
        0.0,
        1.0,
        0.0,
    )


@pytest.mark.parametrize("bad_length", [-10.0, float("nan"), float("inf")])
def test_direct_v2_line_draft_rejects_nonpositive_or_nonfinite_length(bad_length):
    """resume/test-double 입력도 typed Gemini 경계와 같은 수치 제약을 적용한다."""

    state = StateEngine(load_settings(Path("missing.env"))).initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[
                Goal(goal_id="G1", type="move", direction="+X", length=20.0)
            ],
        )
    )
    draft = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "section_source": "inherit_target",
            "path_kind": "line",
            "length": bad_length,
            "direction": (1.0, 0.0, 0.0),
        },
    )

    result = validate_draft(draft, state)
    assert not result.valid
    assert any("length" in error for error in result.errors)


def test_spline_waypoint_may_pass_through_global_origin():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=10.0, wall_thickness=1.0),
        start_position=(-20.0, 0.0, 0.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[
            Goal(
                goal_id="G1",
                type="route",
                path_kind="spline",
                required_waypoints=[(0.0, 0.0, 0.0)],
            )
        ],
    )
    engine = StateEngine(load_settings(Path("missing.env")))
    state = engine.initial_state(intent)
    draft = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "section_source": "inherit_target",
            "path_kind": "spline",
            "waypoints": [(0.0, 0.0, 0.0), (20.0, 10.0, 0.0)],
            "final_tangent": (1.0, 0.0, 0.0),
        },
    )

    assert validate_draft(draft, state).valid
    action = engine.resolve_action(draft, state)
    assert action.params["interpolation"] == "bspline"
    assert action.params["frenet"] is False
    assert action.params["minimum_curvature_radius"] == pytest.approx(10.0)
    assert validate_action(action, state).valid


def test_relative_spline_waypoints_resolve_from_selected_target_port():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=10.0, wall_thickness=1.0),
        target_behavior=[
            Goal(goal_id="G1", type="move", direction="+X", length=40.0),
            Goal(
                goal_id="G2",
                depends_on_goal_ids=["G1"],
                type="route",
                path_kind="spline",
                waypoint_frame="relative_to_target",
                required_waypoints=[(20.0, 0.0, 0.0), (40.0, 10.0, 10.0)],
            ),
        ],
    )
    engine = StateEngine(load_settings(Path("missing.env")))
    state0 = engine.initial_state(intent)
    line = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "section_source": "inherit_target",
            "path_kind": "line",
            "length": 40.0,
            "direction": (1.0, 0.0, 0.0),
        },
    )
    state1 = engine.apply_action(engine.resolve_action(line, state0), state0)
    spline = ActionDraft(
        target_port="M1.out",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G2"],
        completed_goal_ids=["G2"],
        params={
            "section_source": "inherit_target",
            "path_kind": "spline",
            "waypoint_frame": "relative_to_target",
            "waypoints": [(20.0, 0.0, 0.0), (40.0, 10.0, 10.0)],
            "final_tangent": (1.0, 1.0, 1.0),
        },
    )

    assert validate_draft(spline, state1).valid
    action = engine.resolve_action(spline, state1)
    assert action.params["waypoint_frame"] == "global"
    assert action.params["waypoints"] == [
        (60.0, 0.0, 0.0),
        (80.0, 10.0, 10.0),
    ]
    assert action.params["final_tangent"] == pytest.approx(
        (2.0 / math.sqrt(6.0), 1.0 / math.sqrt(6.0), 1.0 / math.sqrt(6.0))
    )
    corrupted = action.model_copy(
        update={"params": {**action.params, "final_tangent": (1.0, 0.0, 0.0)}}
    )
    assert any(
        "final_tangent invariant mismatch" in error
        for error in validate_action(corrupted, state1).errors
    )
    state2 = engine.apply_action(action, state1)
    verification = build_step_verification(state1, action, state2, intent, 2)
    assert not {
        "GOAL_ROUTE_WAYPOINT_MISMATCH",
        "GOAL_ROUTE_WAYPOINT_ORDER_MISMATCH",
    } & {issue.issue_code for issue in verification.issues}


def test_relative_route_goal_keeps_first_action_origin_when_spanning_actions():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=10.0, wall_thickness=1.0),
        target_behavior=[
            Goal(goal_id="G0", type="move", direction="+X", length=10.0),
            Goal(
                goal_id="G1",
                depends_on_goal_ids=["G0"],
                type="route",
                waypoint_frame="relative_to_target",
                required_waypoints=[(20.0, 0.0, 0.0), (40.0, 0.0, 0.0)],
            ),
        ],
    )
    engine = StateEngine(load_settings(Path("missing.env")))
    state = engine.initial_state(intent)

    def line(target: str, goal_id: str, length: float, *, completed: bool):
        return ActionDraft(
            target_port=target,
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=[goal_id],
            completed_goal_ids=[goal_id] if completed else [],
            params={
                "section_source": "inherit_target",
                "path_kind": "line",
                "length": length,
                "direction": (1.0, 0.0, 0.0),
            },
        )

    first = engine.resolve_action(line("START", "G0", 10.0, completed=True), state)
    state = engine.apply_action(first, state)
    second = engine.resolve_action(line("M1.out", "G1", 20.0, completed=False), state)
    state = engine.apply_action(second, state)
    third = engine.resolve_action(line("M2.out", "G1", 20.0, completed=True), state)
    after = engine.apply_action(third, state)
    verification = build_step_verification(state, third, after, intent, 3)

    assert not {
        "GOAL_ROUTE_WAYPOINT_MISMATCH",
        "GOAL_ROUTE_WAYPOINT_ORDER_MISMATCH",
    } & {issue.issue_code for issue in verification.issues}


def test_relative_first_waypoint_cannot_coincide_with_target_port():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=10.0, wall_thickness=1.0),
        target_behavior=[
            Goal(
                goal_id="G1",
                type="route",
                path_kind="spline",
                waypoint_frame="relative_to_target",
                required_waypoints=[(0.0, 0.0, 0.0), (10.0, 5.0, 5.0)],
            )
        ],
    )
    state = StateEngine(load_settings(Path("missing.env"))).initial_state(intent)
    draft = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "section_source": "inherit_target",
            "path_kind": "spline",
            "waypoint_frame": "relative_to_target",
            "waypoints": [(0.0, 0.0, 0.0), (10.0, 5.0, 5.0)],
            "final_tangent": (1.0, 1.0, 1.0),
        },
    )

    result = validate_draft(draft, state)
    assert not result.valid
    assert any("must not coincide" in error for error in result.errors)


def test_suffix_replay_translates_global_waypoints_but_not_relative_offsets():
    base = {
        "section_source": "inherit_target",
        "path_kind": "spline",
        "waypoints": [[10.0, 20.0, 30.0], [20.0, 30.0, 40.0]],
        "final_tangent": [1.0, 0.0, 0.0],
    }
    relative = ActionDraft(
        target_port="OLD",
        module="route",
        catalog_schema_version=2,
        params={**base, "waypoint_frame": "relative_to_target"},
    )
    global_draft = relative.model_copy(
        deep=True,
        update={"params": {**base, "waypoint_frame": "global"}},
    )

    rebound_relative = _rebind_replay_draft(
        relative, {"OLD": "NEW"}, (100.0, -50.0, 5.0)
    )
    rebound_global = _rebind_replay_draft(
        global_draft, {"OLD": "NEW"}, (100.0, -50.0, 5.0)
    )

    assert rebound_relative.params["waypoints"] == base["waypoints"]
    assert rebound_global.params["waypoints"] == [
        [110.0, -30.0, 35.0],
        [120.0, -20.0, 45.0],
    ]


def test_self_crossing_spline_control_path_requires_freecad_clearance_check():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=10.0, wall_thickness=1.0),
        start_position=(-20.0, -20.0, 0.0),
        start_axis=(1.0, 1.0, 0.0),
        target_behavior=[
            Goal(
                goal_id="G1",
                type="route",
                path_kind="spline",
                required_waypoints=[
                    (20.0, 20.0, 0.0),
                    (-20.0, 20.0, 0.0),
                    (20.0, -20.0, 0.0),
                ],
            )
        ],
    )
    engine = StateEngine(load_settings(Path("missing.env")))
    before = engine.initial_state(intent)
    draft = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "section_source": "inherit_target",
            "path_kind": "spline",
            "waypoints": [
                (20.0, 20.0, 0.0),
                (-20.0, 20.0, 0.0),
                (20.0, -20.0, 0.0),
            ],
            "final_tangent": (1.0, -1.0, 0.0),
            "interpolation": "bspline",
            "frenet": True,
            "minimum_curvature_radius": 6.0,
        },
    )
    action = engine.resolve_action(draft, before)
    after = engine.apply_action(action, before)
    step = build_step_verification(before, action, after, intent, 1)

    issue = next(
        item
        for item in step.issues
        if item.issue_code == "STATIC_SELF_CLEARANCE_REQUIRES_FREECAD"
    )
    assert issue.severity == "warning"
    assert issue.actual["candidates"][0]["centerline_separation"] == pytest.approx(
        0.0
    )


def test_transition_derives_concentric_offset_and_preserves_wall_by_default():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                goal_id="G1",
                type="diameter_change",
                diameter_out=12.0,
                transition_length=40.0,
            )
        ],
    )
    engine = StateEngine(load_settings(Path("missing.env")))
    state = engine.initial_state(intent)
    draft = ActionDraft(
        target_port="START",
        module="transition",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "section_source": "inherit_target",
            "diameter_out": 12.0,
            "length": 40.0,
        },
    )

    assert validate_draft(draft, state).valid
    action = engine.resolve_action(draft, state)
    assert action.params["wall_thickness_out"] == 2.0
    assert action.params["offset"] == (0.0, 0.0, 0.0)
    assert validate_action(action, state).valid


def test_transition_rejects_positive_but_near_step_taper_length():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                goal_id="G1",
                type="diameter_change",
                diameter_out=8.0,
                transition_length=0.001,
            )
        ],
    )
    engine = StateEngine(load_settings(Path("missing.env")))
    state = engine.initial_state(intent)
    draft = ActionDraft(
        target_port="START",
        module="transition",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "section_source": "inherit_target",
            "diameter_out": 8.0,
            "length": 0.001,
        },
    )

    result = validate_draft(draft, state)
    assert not result.valid
    assert any("taper is too abrupt" in error for error in result.errors)


def test_gemini_route_schema_structurally_separates_path_kinds():
    variants = _planner_route_variants()

    assert set(variants) == {"line", "circular_arc", "spline"}
    assert set(variants["line"]["required"]) == {
        "section_source",
        "path_kind",
        "length",
        "direction",
    }
    assert set(variants["line"]["properties"]) == {
        "section_source",
        "path_kind",
        "length",
        "direction",
    }

    assert set(variants["circular_arc"]["required"]) == {
        "section_source",
        "path_kind",
        "bend_radius",
        "sweep_angle",
        "plane_normal",
    }
    assert not {
        "length",
        "direction",
        "waypoints",
        "initial_tangent",
        "final_tangent",
        "terminal_axis",
    } & set(variants["circular_arc"]["properties"])

    assert set(variants["spline"]["required"]) == {
        "section_source",
        "path_kind",
        "waypoint_frame",
        "waypoints",
    }
    assert not {
        "length",
        "direction",
        "bend_radius",
        "sweep_angle",
        "plane_normal",
        "terminal_axis",
        "initial_tangent",
        "final_tangent",
    } & set(variants["spline"]["properties"])
    assert not {
        "interpolation",
        "frenet",
        "minimum_curvature_radius",
    } & set(variants["spline"]["properties"])
    assert all(variant["additionalProperties"] is False for variant in variants.values())


def test_connect_ports_and_junction_variants_are_structural_at_gemini_boundary():
    schema = gemini_json_schema(PlannerDecision)

    connect_params = schema["$defs"]["ConnectPortsChoice"]["properties"]["params"]
    connect_variants = {}
    for branch in connect_params["anyOf"]:
        variant = schema["$defs"][branch["$ref"].rsplit("/", 1)[-1]]
        kind = variant["properties"]["path_kind"]["enum"][0]
        connect_variants[kind] = variant

    assert set(connect_variants) == {"line", "circular_arc", "spline"}
    assert set(connect_variants["line"]["properties"]) == {
        "other_port_id",
        "section_source",
        "path_kind",
    }
    assert set(connect_variants["circular_arc"]["required"]) == {
        "other_port_id",
        "section_source",
        "path_kind",
        "waypoints",
    }
    assert not {
        "initial_tangent",
        "final_tangent",
        "interpolation",
        "frenet",
        "minimum_curvature_radius",
    } & set(connect_variants["circular_arc"]["properties"])
    assert set(connect_variants["spline"]["required"]) == {
        "other_port_id",
        "section_source",
        "path_kind",
        "waypoints",
    }
    assert not {
        "interpolation",
        "frenet",
        "minimum_curvature_radius",
    } & set(connect_variants["spline"]["properties"])
    assert not {"initial_tangent", "final_tangent"} & set(
        connect_variants["spline"]["properties"]
    )

    junction_params = schema["$defs"]["JunctionChoice"]["properties"]["params"]
    junction_variants = {}
    for branch in junction_params["anyOf"]:
        variant = schema["$defs"][branch["$ref"].rsplit("/", 1)[-1]]
        mode = variant["properties"]["blend_mode"]["enum"][0]
        junction_variants[mode] = variant

    assert set(junction_variants) == {"hard", "fillet"}
    assert not {"blend_radius", "inner_blend_radius"} & set(
        junction_variants["hard"]["properties"]
    )
    assert {"blend_radius", "inner_blend_radius"} <= set(
        junction_variants["fillet"]["required"]
    )


def test_gemini_planner_forces_target_section_inheritance_without_repeated_od_wall():
    schema = gemini_json_schema(PlannerDecision)
    for definition in schema["$defs"].values():
        properties = definition.get("properties", {})
        if "section_source" not in properties:
            continue
        assert properties["section_source"] == {
            "type": "string",
            "enum": ["inherit_target"],
        }
        assert "outer_diameter" not in properties
        assert "wall_thickness" not in properties

    invalid = {
        "catalog_schema_version": 2,
        "target_port": "START",
        "affected_goal_ids": ["G1"],
        "completed_goal_ids": ["G1"],
        "choice": {
            "module": "route",
            "params": {
                "path_kind": "line",
                "section_source": "inherit_target",
                "outer_diameter": 20.0,
                "length": 80.0,
                "direction": [1.0, 0.0, 0.0],
            },
        },
    }
    with pytest.raises(JSONSchemaValidationError):
        Draft202012Validator(schema).validate(invalid)


def test_gemini_planner_floats_use_bounded_decimal_objects_at_llm_boundary():
    schema = gemini_json_schema(PlannerDecision, encode_decimals=True)
    route_choice = schema["$defs"]["RouteChoice"]
    params = route_choice["properties"]["params"]
    line_ref = next(
        branch["$ref"]
        for branch in params["anyOf"]
        if branch["$ref"].endswith("/RouteLine")
    )
    line = schema["$defs"][line_ref.rsplit("/", 1)[-1]]
    vector = schema["$defs"]["V3"]

    assert line["properties"]["length"] == {"$ref": "#/$defs/CDP"}
    assert all(
        item == {"$ref": "#/$defs/CD"}
        for item in vector["prefixItems"]
    )
    assert schema["$defs"]["CDP"]["properties"]["p"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 9,
    }

    decoded = _decode_decimal_numbers(
        {
            "catalog_schema_version": 2,
            "target_port": "START",
            "affected_goal_ids": ["G1"],
            "completed_goal_ids": ["G1"],
            "choice": {
                "module": "route",
                "params": {
                    "path_kind": "line",
                    "section_source": "inherit_target",
                    "length": {"k": "d", "c": 80, "p": 0},
                    "direction": [
                        {"k": "d", "c": 1, "p": 0},
                        {"k": "d", "c": 0, "p": 0},
                        {"k": "d", "c": 0, "p": 0},
                    ],
                },
            },
        }
    )
    decision = PlannerDecision.model_validate(decoded)
    assert decision.to_action_draft().params["length"] == 80.0
    assert decision.to_action_draft().params["direction"] == [1.0, 0.0, 0.0]


def test_planner_numeric_vocabulary_preserves_contract_and_safe_derived_values():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(goal_id="G1", type="move", direction="+X", length=80.0),
            Goal(
                goal_id="G2",
                depends_on_goal_ids=["G1"],
                type="diameter_change",
                diameter_out=12.0,
                wall_thickness_out=1.5,
                transition_length=40.0,
            ),
        ],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )
    state = StateEngine(load_settings(Path("missing.env"))).initial_state(intent)

    literals = _planner_numeric_literals(state)

    assert {"-1", "0", "1", "1.5", "12", "20", "40", "80"} <= set(
        literals
    )
    assert {"2.25", "25"} <= set(literals)
    assert len(literals) <= 96


def test_planner_numeric_vocabulary_preserves_later_dependency_ready_parallel_goal():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                goal_id="G0",
                type="move",
                direction="+X",
                length=10.0,
            ),
            Goal(
                goal_id="G1",
                type="move",
                direction="+X",
                length=1000.125,
            ),
            Goal(
                goal_id="G2",
                depends_on_goal_ids=["G0"],
                allow_parallel=True,
                type="move",
                direction="+Y",
                length=1222.375,
            ),
            # Later sequential goals can sometimes share one compatible action
            # with earlier goals. Keep their values too instead of preselecting
            # a narrower action sequence at the schema boundary.
            Goal(
                goal_id="G3",
                depends_on_goal_ids=["G0"],
                type="move",
                direction="+Z",
                length=1333.625,
            ),
        ],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )
    engine = StateEngine(load_settings(Path("missing.env")))
    state = engine.initial_state(intent)
    state = engine.apply_action(
        engine.resolve_action(
            ActionDraft(
                target_port="START",
                module="straight_pipe",
                params={"length": 10.0, "direction": "+X"},
                affected_goal_ids=["G0"],
                completed_goal_ids=["G0"],
            ),
            state,
        ),
        state,
    )

    literals = set(_planner_numeric_literals(state))

    assert {"-360", "-1", "0", "0.5", "1", "2", "90", "360"} <= literals
    assert {"1000.125", "1222.375", "1333.625"} <= literals
    assert len(literals) <= 96


def test_planner_numeric_vocabulary_preserves_immutable_geometric_constraint():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(goal_id="G1", type="move", direction="+X", length=80.0)
        ],
        geometric_constraints=[
            GeometricConstraint(
                constraint_id="C1",
                type="max_extent",
                axis="+X",
                value=137.625,
            )
        ],
    )
    state = StateEngine(load_settings(Path("missing.env"))).initial_state(intent)

    assert "137.625" in _planner_numeric_literals(state)


def test_planner_numeric_vocabulary_keeps_distinct_mandatory_float_spellings():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                goal_id="G1",
                type="move",
                direction="+X",
                length=100_000_000.1,
            ),
            Goal(
                goal_id="G2",
                allow_parallel=True,
                type="move",
                direction="+Y",
                length=100_000_000.2,
            ),
            Goal(
                goal_id="G3",
                allow_parallel=True,
                type="move",
                direction="+Z",
                length=1.000000001,
            ),
            Goal(
                goal_id="G4",
                allow_parallel=True,
                type="move",
                direction="-Z",
                length=1.000000002,
            ),
        ],
    )
    state = StateEngine(load_settings(Path("missing.env"))).initial_state(intent)

    literals = set(_planner_numeric_literals(state))

    assert {
        "100000000.1",
        "100000000.2",
        "1.000000001",
        "1.000000002",
    } <= literals


def test_planner_numeric_vocabulary_reserves_active_section_derived_values():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=1000.0, wall_thickness=10.0),
        target_behavior=[
            Goal(
                goal_id="G1",
                type="connector",
                direction="+X",
                length=100.0,
                component="coupling",
            )
        ],
        expected_open_ports=1,
        expected_open_ports_source="derived",
        required_components=["coupling"],
    )
    state = StateEngine(load_settings(Path("missing.env"))).initial_state(intent)

    literals = set(_planner_numeric_literals(state))

    assert {"1250", "1500", "2000", "3000"} <= literals


def test_planner_numeric_vocabulary_never_truncates_large_active_spline_goal():
    coordinate_values = [float(index) for index in range(1, 151)]
    waypoints = [
        tuple(coordinate_values[index : index + 3])
        for index in range(0, len(coordinate_values), 3)
    ]
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                goal_id="G1",
                type="route",
                path_kind="spline",
                required_waypoints=waypoints,
                terminal_position=waypoints[-1],
                terminal_axis=(1.0, 0.0, 0.0),
                minimum_curvature_radius=10.0,
            )
        ],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )
    state = StateEngine(load_settings(Path("missing.env"))).initial_state(intent)

    with pytest.raises(ValueError, match="provider-safe maximum is 96"):
        _planner_numeric_literals(state)


def test_planner_numeric_vocabulary_preserves_nonfirst_targetable_port():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(goal_id="G1", type="move", direction="+X", length=80.0)
        ],
    )
    state = StateEngine(load_settings(Path("missing.env"))).initial_state(intent)
    ports = [state.open_ports[0]]
    for index in range(1, 10):
        ports.append(
            Port(
                id=f"P{index}",
                position=(100.125 + index, index * 0.125, index * -0.25),
                axis=(1.0, 0.0, 0.0),
                outer_diameter=20.0,
                wall_thickness=2.0,
            )
        )
    state = state.model_copy(
        update={
            "open_ports": ports,
            "open_port_ids": [port.id for port in ports],
            "port_nodes": {port.id: port for port in ports},
        }
    )

    literals = set(_planner_numeric_literals(state))

    assert {"109.125", "1.125", "-2.25"} <= literals
    assert len(literals) <= 96


def test_planner_numeric_vocabulary_secondary_tail_keeps_provider_headroom():
    values = [float(index) + 0.125 for index in range(1, 7)]
    waypoints = [
        tuple(values[index : index + 3])
        for index in range(0, len(values), 3)
    ]
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                goal_id="G1",
                type="route",
                path_kind="spline",
                required_waypoints=waypoints,
                terminal_position=waypoints[-1],
                terminal_axis=(1.0, 0.0, 0.0),
                minimum_curvature_radius=10.0,
            )
        ],
    )
    state = StateEngine(load_settings(Path("missing.env"))).initial_state(intent)
    resumed_state = PipeState.model_validate_json(state.model_dump_json())

    first = _planner_numeric_literals(state)
    second = _planner_numeric_literals(state)
    resumed = _planner_numeric_literals(resumed_state)
    mandatory = _planner_numeric_literals(state, include_optional=False)

    payload_bytes = len(
        json.dumps(first, separators=(",", ":")).encode("utf-8")
    )

    assert len(first) < 96
    assert payload_bytes <= PLANNER_PREFERRED_NUMBER_LITERAL_BYTES
    assert payload_bytes < MAX_STRUCTURED_NUMBER_LITERAL_BYTES
    assert first[: len(mandatory)] == mandatory
    assert first == second == resumed
    assert resumed_state.model_dump(mode="json") == state.model_dump(mode="json")
    assert {"1.125", "6.125"} <= set(first)


def _diagonal_post_route_state() -> PipeState:
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                goal_id="route_to_first_junction",
                type="route",
                length=30.0,
                minimum_curvature_radius=30.0,
            ),
            Goal(
                goal_id="left_y_junction",
                depends_on_goal_ids=["route_to_first_junction"],
                type="branch",
                required_outlet_vectors=[(-1.0, 0.0, -1.0)],
                include_primary_outlet=True,
                junction_style="smooth_hub",
                max_hub_radius=25.0,
            ),
            Goal(
                goal_id="central_passage",
                depends_on_goal_ids=["left_y_junction"],
                type="route",
                length=40.0,
                minimum_curvature_radius=30.0,
            ),
            Goal(
                goal_id="right_y_junction",
                depends_on_goal_ids=["central_passage"],
                type="branch",
                required_outlet_vectors=[
                    (1.0, 0.0, 1.0),
                    (1.0, 0.0, -1.0),
                ],
                include_primary_outlet=False,
                junction_style="smooth_hub",
                max_hub_radius=25.0,
            ),
        ],
        start_axis=(1.0, 0.0, -1.0),
        expected_open_ports=3,
        expected_open_ports_source="derived",
    )
    engine = StateEngine(load_settings(Path("missing.env")))
    initial = engine.initial_state(intent)
    route = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["route_to_first_junction"],
        completed_goal_ids=["route_to_first_junction"],
        params={
            "section_source": "inherit_target",
            "path_kind": "line",
            "length": 30.0,
            "direction": (0.7071, 0.0, -0.7071),
        },
    )
    return engine.apply_action(engine.resolve_action(route, initial), initial)


def _left_junction_decision() -> CorePlannerDecision:
    return CorePlannerDecision.model_validate(
        {
            "catalog_schema_version": 2,
            "target_port": "M1.out",
            "affected_goal_ids": ["left_y_junction"],
            "completed_goal_ids": ["left_y_junction"],
            "choice": {
                "module": "junction",
                "params": {
                    "section_source": "inherit_target",
                    "outlets": [
                        {
                            "role": "primary",
                            "axis": [1.0, 0.0, 0.0],
                            "length": 30.0,
                            "outer_diameter": 20.0,
                            "wall_thickness": 2.0,
                        },
                        {
                            "role": "branch",
                            "axis": [-1.0, 0.0, -1.0],
                            "length": 30.0,
                            "outer_diameter": 20.0,
                            "wall_thickness": 2.0,
                        },
                    ],
                    "max_hub_radius": 25.0,
                    "blend_mode": "fillet",
                    "blend_radius": 5.0,
                    "inner_blend_radius": 2.0,
                },
            },
        }
    )


def test_planner_invalid_request_retries_with_mandatory_only_numeric_schema():
    state = _diagonal_post_route_state()

    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True
        supports_system_instruction = True

        def __init__(self):
            self.calls = []
            self.reset_calls = []

        def has_previous(self, part):
            del part
            return False

        def reset_lineage(self, part):
            self.reset_calls.append(part)

        def stream_structured(self, prompt, schema, **kwargs):
            self.calls.append((prompt, schema, kwargs))
            if len(self.calls) == 1:
                raise GeminiInvalidRequestError(
                    "400 invalid_request: Request contains an invalid argument",
                    provider_code="invalid_request",
                )
            return _left_junction_decision()

    gemini = FakeGemini()
    action = _plan_action(state, dry_run=False, gemini=gemini)

    preferred = gemini.calls[0][2]["numeric_literals"]
    mandatory = gemini.calls[1][2]["numeric_literals"]
    assert action.module == "junction"
    assert len(gemini.calls) == 2
    assert gemini.reset_calls == ["step_planner"]
    assert mandatory == _planner_numeric_literals(state, include_optional=False)
    assert len(mandatory) < len(preferred)
    assert {
        "-1",
        "0",
        "1",
        "2",
        "20",
        "25",
        "30",
        "40",
        "21.213203435596427",
        "-21.213203435596427",
    } <= set(mandatory)
    assert "module_catalog" in gemini.calls[1][0]


def test_planner_repeated_invalid_request_falls_back_to_encoded_decimals():
    state = _diagonal_post_route_state()

    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True
        supports_system_instruction = True

        def __init__(self):
            self.calls = []
            self.reset_calls = []

        def has_previous(self, part):
            del part
            return False

        def reset_lineage(self, part):
            self.reset_calls.append(part)

        def stream_structured(self, prompt, schema, **kwargs):
            self.calls.append((prompt, schema, kwargs))
            if len(self.calls) < 3:
                raise GeminiInvalidRequestError(
                    "400 invalid_request: Request contains an invalid argument",
                    provider_code="invalid_request",
                )
            return _left_junction_decision()

    gemini = FakeGemini()
    action = _plan_action(state, dry_run=False, gemini=gemini)

    assert action.module == "junction"
    assert len(gemini.calls) == 3
    assert gemini.reset_calls == ["step_planner", "step_planner"]
    assert "numeric_literals" in gemini.calls[0][2]
    assert "numeric_literals" in gemini.calls[1][2]
    assert "numeric_literals" not in gemini.calls[2][2]
    assert "bounded decimal-object representation" in gemini.calls[2][0]
    assert "bounded decimal object" in gemini.calls[2][2]["system_instruction"]

    second_action = _plan_action(state, dry_run=False, gemini=gemini)
    assert second_action.module == "junction"
    assert len(gemini.calls) == 4
    assert "numeric_literals" not in gemini.calls[3][2]
    assert "bounded decimal-object representation" in gemini.calls[3][0]


def test_repeated_validator_failure_resets_lineage_and_replans_with_exact_decimals():
    state = _diagonal_post_route_state()
    attempts = [
        ActionAttempt(
            step_index=2,
            attempt_index=index,
            state_id=state.state_id,
            phase="registry_validation",
            status="rejected",
            resolved={
                "target_port": "M1.out",
                "module": "route",
                "params": {"sweep_angle": float(index * 30)},
                "affected_goal_ids": ["left_y_junction"],
                "completed_goal_ids": ["left_y_junction"],
            },
            issue_codes=["REGISTRY_VALIDATION_FAILED"],
            observations=[
                {
                    "actual": {
                        "errors": [
                            "analytic terminal tangent mismatch: expected 0.866025, "
                            f"attempt {index}"
                        ]
                    }
                }
            ],
        )
        for index in range(1, 4)
    ]
    observations = [
        {
            "issue_code": "REGISTRY_VALIDATION_FAILED",
            "message": "the same geometric invariant still fails",
            "expected": {"analytic_tangent": [0.0, -0.5, 0.866025403784]},
            "actual": {"terminal_axis": [0.0, 0.5, 0.75]},
        }
    ]
    repair_context = _planner_repair_context(observations, attempts, 2)

    assert repair_context[0]["context_type"] == "planner_stagnation"
    assert repair_context[0]["repeat_count"] == 3
    assert repair_context[0]["schema_strategy"] == "encoded"

    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True
        supports_system_instruction = True

        def __init__(self):
            self.has_lineage = True
            self.calls = []
            self.reset_calls = []

        def has_previous(self, part):
            assert part == "step_planner"
            return self.has_lineage

        def reset_lineage(self, part):
            self.reset_calls.append(part)
            self.has_lineage = False

        def stream_structured(self, prompt, schema, **kwargs):
            self.calls.append((prompt, schema, kwargs))
            return _left_junction_decision()

    gemini = FakeGemini()
    action = _plan_action(
        state,
        dry_run=False,
        gemini=gemini,
        repair_observations=repair_context,
    )

    assert action.module == "junction"
    assert gemini.reset_calls == ["step_planner"]
    assert len(gemini.calls) == 1
    prompt, _schema, kwargs = gemini.calls[0]
    assert "module_catalog" in prompt
    assert "planner_stagnation" in prompt
    assert "numeric_literals" not in kwargs
    assert "bounded decimal-object representation" in prompt

    fourth = attempts[-1].model_copy(update={"attempt_index": 4})
    continued_context = _planner_repair_context(
        observations, [*attempts, fourth], 2
    )
    assert all(
        item.get("context_type") != "planner_stagnation"
        for item in continued_context
    )

    # Different geometry is allowed to use the remaining bounded retry budget;
    # a shared issue code alone is not proof that the planner made no progress.
    changing = [
        attempts[-1].model_copy(
            update={
                "attempt_index": index,
                "resolved": {
                    **(attempts[-1].resolved or {}),
                    "params": {"sweep_angle": float(index * 30)},
                },
                "observations": [
                    {
                        "actual": {
                            "errors": [
                                "analytic terminal tangent mismatch: "
                                f"expected 0.866025, attempt {index}"
                            ]
                        }
                    }
                ],
            }
        )
        for index in range(4, 7)
    ]
    _planner_repair_context(observations, [*attempts, *changing], 2)

    identical = [
        attempts[-1].model_copy(update={"attempt_index": index})
        for index in range(1, 7)
    ]
    with pytest.raises(_PlannerStagnationError, match="repeated 6 times"):
        _planner_repair_context(observations, identical, 2)


def test_freecad_repair_feedback_preserves_required_and_measured_radius():
    raw = {
        "checks": {
            "assembly": {"passed": False},
            "centerlines": {
                "M1": {
                    "curvature_method": "dense_piecewise_curve_sampling",
                    "curvature_proof": "sampled_not_global_extremum",
                    "curvature_repair_hint": "spread the turn around point 1",
                    "curve_length": 40.5,
                    "endpoint_tangency_passed": True,
                    "endpoint_tangent_dots": {"initial": 1.0, "final": 1.0},
                    "minimum_join_tangent_dot": 1.0,
                    "minimum_nonlocal_distance": 30.9,
                    "minimum_radius": 4.242640687,
                    "minimum_radius_location": {
                        "sample_index": 256,
                        "position": [10.0, 0.0, 0.0],
                    },
                    "minimum_radius_nearest_path_point_index": 1,
                    "optimized_handle_factors": [0.45, 0.3, 0.4],
                    "passed": False,
                    "required_radius": 20.0,
                    "required_self_clearance": 20.0001,
                }
            },
        }
    }

    summary = _compact_freecad_failure_evidence(raw)
    centerline = summary["centerline_context"]["M1"]
    expected, actual, recommendations = _freecad_repair_contract(
        summary,
        module_path_kinds={"M1": "spline"},
    )

    assert centerline["required_radius"] == 20.0
    assert centerline["minimum_radius"] == pytest.approx(4.242640687)
    assert centerline["minimum_radius_location"]["position"] == [10.0, 0.0, 0.0]
    assert expected["minimum_curvature_radius_by_module"] == {"M1": 20.0}
    assert actual["measured_minimum_curvature_radius_by_module"]["M1"] == pytest.approx(
        4.242640687
    )
    assert recommendations[0]["nearest_path_point_index"] == 1
    assert recommendations[0]["nearest_waypoint_index"] == 0
    assert recommendations[0]["parameters"] == ["waypoints"]
    assert "length" not in recommendations[0]["parameters"]


def test_rollback_repair_epoch_ignores_rejections_before_last_accept():
    base = ActionAttempt(
        step_index=2,
        attempt_index=1,
        state_id="S1",
        phase="registry_validation",
        status="rejected",
        draft={"target_port": "M1.out", "module": "route", "params": {"length": 20.0}},
        issue_codes=["REGISTRY_VALIDATION_FAILED"],
        observations=[{"actual": {"errors": ["same error"]}}],
    )
    old_rejections = [
        base.model_copy(update={"attempt_index": index}) for index in range(1, 7)
    ]
    accepted = base.model_copy(
        update={"attempt_index": 7, "status": "accepted", "observations": []}
    )
    new_epoch = base.model_copy(update={"attempt_index": 1})

    context = _planner_repair_context(
        [{"issue_code": "REGISTRY_VALIDATION_FAILED"}],
        [*old_rejections, accepted, new_epoch],
        2,
    )

    history = next(
        item for item in context if item.get("context_type") == "rejected_attempt_history"
    )
    assert len(history["attempts"]) == 1


def test_different_registry_error_details_do_not_trigger_stagnation():
    state = _diagonal_post_route_state()
    errors = [
        "bend_radius must exceed the outer pipe radius",
        "plane_normal hint must not be parallel to the inlet tangent",
        "output section is incompatible with target port",
    ]
    attempts = [
        ActionAttempt(
            step_index=2,
            attempt_index=index,
            state_id=state.state_id,
            phase="registry_validation",
            status="rejected",
            resolved={
                "target_port": "M1.out",
                "module": "route",
                "affected_goal_ids": ["left_y_junction"],
            },
            issue_codes=["REGISTRY_VALIDATION_FAILED"],
            observations=[{"actual": {"errors": [error]}}],
        )
        for index, error in enumerate(errors, start=1)
    ]

    context = _planner_repair_context(
        [{"issue_code": "REGISTRY_VALIDATION_FAILED"}], attempts, 2
    )

    assert all(item.get("context_type") != "planner_stagnation" for item in context)


@pytest.mark.parametrize(
    "request_error",
    [
        GeminiRequestError("authentication or network failure"),
        GeminiInvalidRequestError(
            "429 invalid_request-shaped rate limit",
            status_code=429,
            provider_code="invalid_request",
        ),
    ],
)
def test_planner_does_not_schema_retry_unrelated_request_failure(request_error):
    state = _diagonal_post_route_state()

    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True
        supports_system_instruction = True

        def __init__(self):
            self.calls = 0

        def has_previous(self, part):
            del part
            return False

        def stream_structured(self, prompt, schema, **kwargs):
            del prompt, schema, kwargs
            self.calls += 1
            raise request_error

    gemini = FakeGemini()
    with pytest.raises(GeminiRequestError):
        _plan_action(state, dry_run=False, gemini=gemini)

    assert gemini.calls == 1


def test_planner_schema_fallback_is_bounded_when_all_profiles_are_rejected():
    state = _diagonal_post_route_state()

    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True
        supports_system_instruction = True

        def __init__(self):
            self.calls = 0

        def has_previous(self, part):
            del part
            return False

        def reset_lineage(self, part):
            del part

        def stream_structured(self, prompt, schema, **kwargs):
            del prompt, schema, kwargs
            self.calls += 1
            raise GeminiInvalidRequestError(
                "400 invalid_request: Request contains an invalid argument",
                provider_code="invalid_request",
            )

    gemini = FakeGemini()
    with pytest.raises(
        GeminiInvalidRequestError,
        match="preferred numeric enum, mandatory-only enum, and encoded-decimal",
    ):
        _plan_action(state, dry_run=False, gemini=gemini)

    assert gemini.calls == 3


def test_planner_numeric_vocabulary_count_boundary_is_stable_at_96_on_resume():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                goal_id=f"G{value}",
                type="move",
                direction="+X",
                length=float(value),
            )
            for value in range(3, 85)
        ],
    )
    state = StateEngine(load_settings(Path("missing.env"))).initial_state(intent)
    resumed_state = PipeState.model_validate_json(state.model_dump_json())

    first = _planner_numeric_literals(state)
    resumed = _planner_numeric_literals(resumed_state)

    assert len(first) == 96
    assert (
        len(json.dumps(first, separators=(",", ":")).encode("utf-8"))
        <= MAX_STRUCTURED_NUMBER_LITERAL_BYTES
    )
    assert first == resumed


def test_planner_numeric_vocabulary_97_mandatory_values_use_encoded_schema():
    coordinates = [1000.125 + index for index in range(69)]
    waypoints = [
        tuple(coordinates[index : index + 3])
        for index in range(0, len(coordinates), 3)
    ]
    goals = [
        Goal(
            goal_id="G1",
            type="route",
            path_kind="spline",
            required_waypoints=waypoints,
            terminal_position=waypoints[-1],
            terminal_axis=(1.0, 0.0, 0.0),
            minimum_curvature_radius=10.0,
        ),
        Goal(
            goal_id="G2",
            allow_parallel=True,
            type="move",
            direction="+X",
            length=3000.125,
        ),
        Goal(
            goal_id="G3",
            allow_parallel=True,
            type="move",
            direction="+Y",
            length=3001.125,
        ),
    ]
    state = StateEngine(load_settings(Path("missing.env"))).initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=goals,
        )
    )

    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True

        def __init__(self):
            self.calls = []

        def has_previous(self, part):
            del part
            return False

        def stream_structured(self, prompt, schema, **kwargs):
            del schema
            self.calls.append((prompt, kwargs))
            return ActionDraft(
                target_port="START",
                module="route",
                catalog_schema_version=2,
                affected_goal_ids=["G1"],
                completed_goal_ids=["G1"],
                params={
                    "section_source": "inherit_target",
                    "path_kind": "spline",
                    "waypoints": waypoints,
                    "final_tangent": (1.0, 0.0, 0.0),
                    "interpolation": "bspline",
                    "frenet": True,
                    "minimum_curvature_radius": 10.0,
                },
            )

    gemini = FakeGemini()
    action = _plan_action(state, dry_run=False, gemini=gemini)

    assert action.module == "route"
    assert len(gemini.calls) == 1
    assert "numeric_literals" not in gemini.calls[0][1]
    assert "bounded decimal-object representation" in gemini.calls[0][0]


def test_small_spline_uses_encoded_schema_before_angle_literals_can_leak_into_xyz():
    waypoints = [(25.0, 0.0, 10.0), (55.0, 20.0, 30.0)]
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                goal_id="G1",
                type="route",
                path_kind="spline",
                required_waypoints=waypoints,
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
                minimum_curvature_radius=20.0,
            )
        ],
    )
    state = StateEngine(load_settings(Path("missing.env"))).initial_state(intent)

    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True

        def __init__(self):
            self.calls = []

        def has_previous(self, part):
            del part
            return False

        def stream_structured(self, prompt, schema, **kwargs):
            del schema
            self.calls.append((prompt, kwargs))
            return ActionDraft(
                target_port="START",
                module="route",
                catalog_schema_version=2,
                affected_goal_ids=["G1"],
                completed_goal_ids=["G1"],
                params={
                    "section_source": "inherit_target",
                    "path_kind": "spline",
                    "waypoint_frame": "relative_to_target",
                    "waypoints": waypoints,
                    "interpolation": "bspline",
                    "frenet": True,
                    "minimum_curvature_radius": 20.0,
                },
            )

    gemini = FakeGemini()
    action = _plan_action(state, dry_run=False, gemini=gemini)

    assert action.module == "route"
    assert len(gemini.calls) == 1
    assert "numeric_literals" not in gemini.calls[0][1]
    assert "bounded decimal-object representation" in gemini.calls[0][0]


def test_spline_encoded_repairs_reuse_lineage_until_stagnation_reset():
    waypoints = [(25.0, 0.0, 10.0), (55.0, 20.0, 30.0)]
    state = StateEngine(load_settings(Path("missing.env"))).initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[
                Goal(
                    goal_id="G1",
                    type="route",
                    path_kind="spline",
                    required_waypoints=waypoints,
                    waypoint_frame="relative_to_target",
                    waypoint_scale_policy="fixed",
                    minimum_curvature_radius=20.0,
                )
            ],
        )
    )

    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True

        def __init__(self):
            self.calls = []
            self.lineage = False
            self.reset_calls = []

        def has_previous(self, part):
            assert part == "step_planner"
            return self.lineage

        def reset_lineage(self, part):
            self.reset_calls.append(part)
            self.lineage = False

        def stream_structured(self, prompt, schema, **kwargs):
            del schema
            self.calls.append((prompt, kwargs))
            self.lineage = True
            return ActionDraft(
                target_port="START",
                module="route",
                catalog_schema_version=2,
                affected_goal_ids=["G1"],
                completed_goal_ids=["G1"],
                params={
                    "section_source": "inherit_target",
                    "path_kind": "spline",
                    "waypoint_frame": "relative_to_target",
                    "waypoints": waypoints,
                    "interpolation": "bspline",
                    "frenet": True,
                    "minimum_curvature_radius": 20.0,
                },
            )

    geometry_failure = {
        "check_name": "freecad_semantic_validation",
        "expected": {"minimum_radius": 20.0},
        "actual": {
            "centerline_context": {
                "M1": {"required_radius": 20.0, "minimum_radius": 10.0}
            }
        },
    }
    gemini = FakeGemini()

    _plan_action(state, dry_run=False, gemini=gemini)
    _plan_action(
        state,
        dry_run=False,
        gemini=gemini,
        repair_observations=[geometry_failure],
    )
    _plan_action(
        state,
        dry_run=False,
        gemini=gemini,
        repair_observations=[
            {
                "context_type": "planner_stagnation",
                "repeat_count": 3,
                "schema_strategy": "encoded",
            },
            geometry_failure,
        ],
    )

    assert gemini.reset_calls == ["step_planner"]
    assert len(gemini.calls) == 3
    first_prompt, first_kwargs = gemini.calls[0]
    second_prompt, second_kwargs = gemini.calls[1]
    third_prompt, third_kwargs = gemini.calls[2]
    assert "module_catalog" in first_prompt
    assert "Compact planning payload:" in first_prompt
    assert "bounded decimal-object representation" in first_prompt
    assert "numeric_literals" not in first_kwargs
    assert second_prompt.startswith("Repair your immediately previous action")
    assert "module_catalog" not in second_prompt
    assert "Compact planning payload:" not in second_prompt
    assert "bounded decimal-object representation" in second_prompt
    assert "numeric_literals" not in second_kwargs
    assert "module_catalog" in third_prompt
    assert "Compact planning payload:" in third_prompt
    assert "planner_stagnation" in third_prompt
    assert "bounded decimal-object representation" in third_prompt
    assert "numeric_literals" not in third_kwargs


def test_resolved_spline_curvature_is_rejected_before_freecad_call():
    """원본 manifold의 23.57 mm 후보는 registry에서 수치와 함께 거부한다."""

    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=24.0, wall_thickness=3.5),
        target_behavior=[
            Goal(
                goal_id="arm",
                type="route",
                path_kind="spline",
                required_waypoints=[
                    (30.0, 0.0, 0.0),
                    (60.0, 15.0, 5.0),
                    (90.0, 30.0, 10.0),
                ],
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
                terminal_axis=(1.0, 0.0, 0.0),
                minimum_curvature_radius=35.0,
            )
        ],
        expected_open_ports=1,
    )
    settings = load_settings(Path("missing.env"))
    engine = StateEngine(settings)
    state = engine.initial_state(intent)
    resolved = engine.resolve_action(
        ActionDraft(
            target_port="START",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["arm"],
            completed_goal_ids=["arm"],
            params={
                "section_source": "inherit_target",
                "path_kind": "spline",
                "waypoint_frame": "relative_to_target",
                "waypoints": [
                    (30.0, 0.0, 0.0),
                    (60.0, 15.0, 5.0),
                    (90.0, 30.0, 10.0),
                ],
                "final_tangent": (1.0, 0.0, 0.0),
            },
        ),
        state,
    )

    result = validate_action(resolved, state)

    assert not result.valid
    diagnostic = " ".join(result.errors)
    assert "spline curvature preflight failed" in diagnostic
    assert "expected minimum_radius>=35" in diagnostic
    assert "actual minimum_radius=23.5709865616" in diagnostic
    assert "do not add closely spaced points" in diagnostic


def test_planner_numeric_vocabulary_rejects_unencodable_mandatory_value():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                goal_id="G1",
                type="move",
                direction="+X",
                length=1_000_000_001.0,
            )
        ],
    )
    state = StateEngine(load_settings(Path("missing.env"))).initial_state(intent)

    with pytest.raises(ValueError, match="outside the provider-safe ±1e9 range"):
        _planner_numeric_literals(state)


def test_planner_numeric_vocabulary_oversized_mandatory_bytes_use_encoded_schema():
    state = StateEngine(load_settings(Path("missing.env"))).initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[
                Goal(
                    goal_id=f"G{index}",
                    allow_parallel=True,
                    type="move",
                    direction="+X",
                    length=100_000_000.1 + index * 0.001,
                )
                for index in range(25)
            ],
        )
    )

    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True

        def __init__(self):
            self.calls = []

        def has_previous(self, part):
            del part
            return False

        def stream_structured(self, prompt, schema, **kwargs):
            del schema
            self.calls.append((prompt, kwargs))
            return ActionDraft(
                target_port="START",
                module="route",
                catalog_schema_version=2,
                affected_goal_ids=["G0"],
                completed_goal_ids=["G0"],
                params={
                    "section_source": "inherit_target",
                    "path_kind": "line",
                    "length": 100_000_000.1,
                    "direction": (1.0, 0.0, 0.0),
                },
            )

    gemini = FakeGemini()
    action = _plan_action(state, dry_run=False, gemini=gemini)

    assert action.module == "route"
    assert len(gemini.calls) == 1
    assert "numeric_literals" not in gemini.calls[0][1]
    assert "bounded decimal-object representation" in gemini.calls[0][0]


def test_actual_planner_numeric_literal_schema_rejects_raw_float_and_bad_line():
    literals = ["-1", "0", "1", "20", "80"]
    schema = gemini_json_schema(PlannerDecision, number_literals=literals)
    valid = {
        "catalog_schema_version": 2,
        "target_port": "START",
        "affected_goal_ids": ["G1"],
        "completed_goal_ids": ["G1"],
        "choice": {
            "module": "route",
            "params": {
                "path_kind": "line",
                "section_source": "inherit_target",
                "length": "80",
                "direction": ["1", "0", "0"],
            },
        },
    }

    Draft202012Validator(schema).validate(valid)
    parsed = PlannerDecision.model_validate(valid)
    assert parsed.to_action_draft().params["length"] == 80.0

    raw_float = {
        **valid,
        "choice": {
            **valid["choice"],
            "params": {**valid["choice"]["params"], "length": 80.0},
        },
    }
    with pytest.raises(JSONSchemaValidationError):
        Draft202012Validator(schema).validate(raw_float)

    invalid_line = {
        **valid,
        "choice": {
            **valid["choice"],
            "params": {
                "path_kind": "line",
                "section_source": "inherit_target",
                "waypoints": [["80", "0", "0"]],
            },
        },
    }
    with pytest.raises(JSONSchemaValidationError):
        Draft202012Validator(schema).validate(invalid_line)


@pytest.mark.parametrize("decision_schema", [CorePlannerDecision, PlannerDecision])
def test_planner_provider_schema_uses_one_bounded_numeric_enum(decision_schema):
    literals = [str(index) for index in range(96)]
    schema = gemini_json_schema(decision_schema, number_literals=literals)
    Draft202012Validator.check_schema(schema)

    numeric_definitions = [
        definition
        for definition in schema["$defs"].values()
        if definition == {"type": "string", "enum": literals}
    ]

    def raw_number_nodes(value):
        if isinstance(value, list):
            return sum(raw_number_nodes(item) for item in value)
        if not isinstance(value, dict):
            return 0
        return int(value.get("type") == "number") + sum(
            raw_number_nodes(item) for item in value.values()
        )

    assert len(numeric_definitions) == 1
    assert numeric_definitions[0]["enum"] == literals
    assert raw_number_nodes(schema) == 0

    valid = {
        "catalog_schema_version": 2,
        "target_port": "START",
        "affected_goal_ids": ["G1"],
        "completed_goal_ids": ["G1"],
        "choice": {
            "module": "route",
            "params": {
                "path_kind": "line",
                "section_source": "inherit_target",
                "length": "80",
                "direction": ["1", "0", "0"],
            },
        },
    }
    Draft202012Validator(schema).validate(valid)
    decision_schema.model_validate(valid)

    invalid = {
        **valid,
        "choice": {
            **valid["choice"],
            "params": {**valid["choice"]["params"], "length": 80.0},
        },
    }
    with pytest.raises(JSONSchemaValidationError):
        Draft202012Validator(schema).validate(invalid)


def test_numeric_literal_schema_rejects_more_than_provider_safe_maximum():
    with pytest.raises(ValueError, match="provider-safe maximum of 96"):
        gemini_json_schema(
            CorePlannerDecision,
            number_literals=[str(index) for index in range(97)],
        )


def test_numeric_literal_schema_rejects_oversized_serialized_enum():
    with pytest.raises(ValueError, match="serialized size of 512 bytes"):
        gemini_json_schema(
            CorePlannerDecision,
            number_literals=[
                f"{index}.123456789012345" for index in range(40)
            ],
        )


def test_encoded_decimal_decoder_rejects_precision_loss_and_bad_scale():
    assert _decode_decimal_numbers({"k": "d", "c": 15, "p": 1}) == 1.5
    with pytest.raises(ValueError, match="invalid encoded decimal"):
        _decode_decimal_numbers({"k": "d", "c": 1 << 53, "p": 0})
    with pytest.raises(ValueError, match="invalid encoded decimal"):
        _decode_decimal_numbers({"k": "d", "c": 15, "p": 10})


def test_invalid_line_reports_only_selected_module_and_path_variant():
    with pytest.raises(ValidationError) as captured:
        PlannerDecision.model_validate(
            {
                "catalog_schema_version": 2,
                "target_port": "START",
                "choice": {
                    "module": "route",
                    "params": {
                        "path_kind": "line",
                        "section_source": "inherit_target",
                        "waypoints": [[80.0, 0.0, 0.0]],
                    },
                },
                "affected_goal_ids": ["G1"],
                "completed_goal_ids": ["G1"],
            }
        )

    errors = captured.value.errors(include_url=False)
    locations = {error["loc"] for error in errors}
    assert locations == {
        ("choice", "route", "params", "line", "length"),
        ("choice", "route", "params", "line", "direction"),
        ("choice", "route", "params", "line", "waypoints"),
    }


def test_route_discriminators_keep_valid_decision_conversion_compatible():
    decision = PlannerDecision.model_validate(
        {
            "catalog_schema_version": 2,
            "target_port": "START",
            "choice": {
                "module": "route",
                "params": {
                    "path_kind": "line",
                    "section_source": "inherit_target",
                    "length": 80.0,
                    "direction": [1.0, 0.0, 0.0],
                },
            },
            "affected_goal_ids": ["G1"],
            "completed_goal_ids": ["G1"],
        }
    )

    draft = decision.to_action_draft()
    assert draft.module == "route"
    assert draft.params == {
        "section_source": "inherit_target",
        "path_kind": "line",
        "length": 80.0,
        "direction": [1.0, 0.0, 0.0],
    }

    # The existing runtime/public construction API remains available to code
    # that validates resolved route parameters directly.
    runtime = RouteParamsV2.model_validate(draft.params)
    assert runtime.length == 80.0
    assert runtime.direction == (1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("path_kind", "params", "unexpected_field"),
    [
        (
            "circular_arc",
            {
                "bend_radius": 30.0,
                "sweep_angle": 90.0,
                "plane_normal": [0.0, 0.0, 1.0],
                "length": 10.0,
            },
            "length",
        ),
        (
            "spline",
                {
                    "waypoint_frame": "global",
                    "waypoints": [[10.0, 2.0, 0.0], [20.0, 5.0, 0.0]],
                    "direction": [1.0, 0.0, 0.0],
                },
            "direction",
        ),
    ],
)
def test_arc_and_spline_forbid_other_variant_fields(
    path_kind: str, params: dict, unexpected_field: str
):
    with pytest.raises(ValidationError) as captured:
        PlannerDecision.model_validate(
            {
                "catalog_schema_version": 2,
                "target_port": "START",
                "choice": {
                    "module": "route",
                    "params": {
                        "path_kind": path_kind,
                        "section_source": "inherit_target",
                        **params,
                    },
                },
                "affected_goal_ids": ["G1"],
                "completed_goal_ids": ["G1"],
            }
        )

    locations = {error["loc"] for error in captured.value.errors(include_url=False)}
    assert locations == {
        ("choice", "route", "params", path_kind, unexpected_field)
    }
