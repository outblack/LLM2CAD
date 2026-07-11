from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import ValidationError

from cadgen.gemini_client import gemini_json_schema
from cadgen.schemas import LLMProductionIntent, ProductionIntent


def _exact_prompt_payload() -> dict:
    return {
        "global_spec": {
            "outer_diameter": 20.0,
            "wall_thickness": 2.0,
            "is_hollow": True,
            "units": "mm",
        },
        "start_position": [0.0, 0.0, 0.0],
        "start_axis": [1.0, 0.0, 0.0],
        "target_behavior": [
            {
                "goal_id": "G1",
                "depends_on_goal_ids": [],
                "allow_parallel": False,
                "type": "move",
                "direction": "+X",
                "length": 80.0,
            },
            {
                "goal_id": "G2",
                "depends_on_goal_ids": ["G1"],
                "allow_parallel": False,
                "type": "connector",
                "length": 30.0,
                "component": "coupling",
                "component_spec": {
                    "component_type": "coupling",
                    "body_length": 30.0,
                },
            },
            {
                "goal_id": "G3",
                "depends_on_goal_ids": ["G2"],
                "allow_parallel": False,
                "type": "diameter_change",
                "diameter_out": 12.0,
                "wall_thickness_out": 1.5,
                "transition_length": 40.0,
            },
            {
                "goal_id": "G4",
                "depends_on_goal_ids": ["G3"],
                "allow_parallel": False,
                "type": "move",
                "direction": "+X",
                "length": 60.0,
            },
            {
                "goal_id": "G5",
                "depends_on_goal_ids": ["G4"],
                "allow_parallel": False,
                "type": "end",
                "end_type": "cap",
            },
        ],
        "expected_open_ports": 0,
        "expected_open_ports_source": "derived",
        "required_components": ["coupling"],
        "hard_constraints": [],
        "geometric_constraints": [],
        "design_notes": [],
    }


def _intent_goal_schemas() -> dict[str, dict]:
    schema = gemini_json_schema(LLMProductionIntent)
    items = schema["properties"]["target_behavior"]["items"]
    variants: dict[str, dict] = {}
    for branch in items["anyOf"]:
        name = branch["$ref"].rsplit("/", 1)[-1]
        variant = schema["$defs"][name]
        goal_type = variant["properties"]["type"]["enum"][0]
        variants[goal_type] = variant
    return variants


def _branch_payload(outlet_contract: dict) -> dict:
    payload = _exact_prompt_payload()
    payload["target_behavior"] = [
        {
            "goal_id": "B1",
            "depends_on_goal_ids": [],
            "allow_parallel": False,
            "type": "branch",
            "direction": "+X",
            "outlet_contract": outlet_contract,
            "include_primary_outlet": False,
            "junction_style": "smooth_hub",
        }
    ]
    payload["expected_open_ports"] = 2
    payload["required_components"] = []
    return payload


def test_connector_schema_exposes_only_compatible_contract_fields():
    connector = _intent_goal_schemas()["connector"]

    assert connector["additionalProperties"] is False
    assert {"component", "type", "goal_id"} <= set(connector["required"])
    assert not {"offset", "path_kind", "plane_normal"} & set(
        connector["properties"]
    )

    payload = _exact_prompt_payload()
    payload["target_behavior"][1].update(
        {
            "offset": [0.0, 0.0, 0.0],
            "path_kind": "line",
            "plane_normal": [0.0, 0.0, 1.0],
        }
    )
    with pytest.raises(ValidationError) as captured:
        LLMProductionIntent.model_validate(payload)

    assert {error["loc"] for error in captured.value.errors(include_url=False)} == {
        ("target_behavior", 1, "connector", "offset"),
        ("target_behavior", 1, "connector", "path_kind"),
        ("target_behavior", 1, "connector", "plane_normal"),
    }


def test_diameter_change_schema_structurally_requires_output_diameter():
    diameter_change = _intent_goal_schemas()["diameter_change"]
    assert {"diameter_out", "transition_length"} <= set(
        diameter_change["required"]
    )

    payload = _exact_prompt_payload()
    del payload["target_behavior"][2]["diameter_out"]
    with pytest.raises(ValidationError) as captured:
        LLMProductionIntent.model_validate(payload)

    assert {error["loc"] for error in captured.value.errors(include_url=False)} == {
        ("target_behavior", 2, "diameter_change", "diameter_out")
    }


def _turn_payload(orientation: dict) -> dict:
    payload = _exact_prompt_payload()
    payload["target_behavior"] = [
        {
            "goal_id": "T1",
            "depends_on_goal_ids": [],
            "allow_parallel": False,
            "type": "turn",
            "orientation": orientation,
            "angle": 90.0,
            "bend_radius": 30.0,
        }
    ]
    payload["expected_open_ports"] = 1
    payload["required_components"] = []
    return payload


def test_turn_schema_structurally_requires_one_orientation_contract():
    schema = gemini_json_schema(LLMProductionIntent)
    turn = _intent_goal_schemas()["turn"]

    assert "orientation" in turn["required"]
    assert not {"direction", "plane_normal"} & set(turn["properties"])
    variants = [
        schema["$defs"][item["$ref"].rsplit("/", 1)[-1]]
        for item in turn["properties"]["orientation"]["anyOf"]
    ]
    assert {variant["properties"]["mode"]["enum"][0] for variant in variants} == {
        "cardinal",
        "signed_plane",
    }
    assert all(variant["additionalProperties"] is False for variant in variants)

    missing = _turn_payload({"mode": "signed_plane"})
    with pytest.raises(ValidationError):
        LLMProductionIntent.model_validate(missing)

    legacy = _turn_payload({"mode": "cardinal", "direction": "+Z"})
    legacy["target_behavior"][0]["plane_normal"] = [0.0, 1.0, 0.0]
    with pytest.raises(ValidationError):
        LLMProductionIntent.model_validate(legacy)


@pytest.mark.parametrize(
    ("orientation", "expected_direction", "expected_plane"),
    [
        ({"mode": "cardinal", "direction": "+Z"}, "+Z", None),
        (
            {"mode": "signed_plane", "plane_normal": [0.0, 1.0, 0.0]},
            None,
            (0.0, 1.0, 0.0),
        ),
    ],
)
def test_turn_orientation_contract_flattens_losslessly(
    orientation, expected_direction, expected_plane
):
    llm_intent = LLMProductionIntent.model_validate(_turn_payload(orientation))
    turn = llm_intent.to_production_intent().target_behavior[0]

    assert turn.direction == expected_direction
    assert turn.plane_normal == expected_plane


def _route_payload(geometry_contracts: list[dict]) -> dict:
    payload = _exact_prompt_payload()
    payload["target_behavior"] = [
        {
            "goal_id": "R1",
            "depends_on_goal_ids": [],
            "allow_parallel": False,
            "type": "route",
            "path_kind": "spline",
            "geometry_contracts": geometry_contracts,
            "minimum_curvature_radius": 20.0,
        }
    ]
    payload["expected_open_ports"] = 1
    payload["required_components"] = []
    return payload


def test_route_schema_structurally_requires_measurable_geometry_contracts():
    schema = gemini_json_schema(LLMProductionIntent)
    route = _intent_goal_schemas()["route"]

    assert "geometry_contracts" in route["required"]
    assert not {
        "length",
        "direction",
        "required_waypoints",
        "terminal_position",
        "terminal_axis",
    } & set(route["properties"])
    contract_items = route["properties"]["geometry_contracts"]["items"]
    variants = [
        schema["$defs"][item["$ref"].rsplit("/", 1)[-1]]
        for item in contract_items["anyOf"]
    ]
    assert {variant["properties"]["mode"]["enum"][0] for variant in variants} == {
        "length",
        "direction",
        "waypoints",
        "terminal_position",
        "terminal_axis",
    }

    missing = _route_payload([])
    with pytest.raises(ValidationError):
        LLMProductionIntent.model_validate(missing)

    duplicate = _route_payload(
        [
            {"mode": "direction", "direction": "+X"},
            {"mode": "direction", "direction": "+Z"},
        ]
    )
    with pytest.raises(ValidationError, match="must be unique"):
        LLMProductionIntent.model_validate(duplicate)


def test_route_geometry_contracts_flatten_losslessly():
    payload = _route_payload(
        [
            {
                "mode": "waypoints",
                "waypoint_frame": "global",
                "waypoint_scale_policy": "fixed",
                "required_waypoints": [[20.0, 20.0, 10.0], [0.0, 40.0, 30.0]],
            },
            {
                "mode": "terminal_position",
                "terminal_position": [-20.0, 20.0, 50.0],
            },
            {"mode": "terminal_axis", "terminal_axis": [-1.0, 0.0, 1.0]},
        ]
    )
    route = LLMProductionIntent.model_validate(
        payload
    ).to_production_intent().target_behavior[0]

    assert route.path_kind == "spline"
    assert route.waypoint_scale_policy == "fixed"
    assert route.required_waypoints == [
        (20.0, 20.0, 10.0),
        (0.0, 40.0, 30.0),
    ]

    missing_policy = _route_payload(
        [
            {
                "mode": "waypoints",
                "waypoint_frame": "relative_to_target",
                "required_waypoints": [[20.0, 0.0, 10.0]],
            }
        ]
    )
    with pytest.raises(ValidationError):
        LLMProductionIntent.model_validate(missing_policy)
    assert route.waypoint_frame == "global"
    assert route.terminal_position == (-20.0, 20.0, 50.0)
    assert route.terminal_axis == (-1.0, 0.0, 1.0)


def test_branch_schema_requires_one_discriminated_outlet_contract():
    schema = gemini_json_schema(LLMProductionIntent)
    branch = _intent_goal_schemas()["branch"]

    assert {"outlet_contract", "include_primary_outlet"} <= set(branch["required"])
    assert not {
        "branch_count",
        "required_outlet_directions",
        "required_outlet_vectors",
        "required_outlets",
    } & set(branch["properties"])

    refs = branch["properties"]["outlet_contract"]["anyOf"]
    variants = [schema["$defs"][item["$ref"].rsplit("/", 1)[-1]] for item in refs]
    assert {variant["properties"]["mode"]["enum"][0] for variant in variants} == {
        "count",
        "directions",
        "vectors",
        "outlets",
    }
    assert all(variant["additionalProperties"] is False for variant in variants)


@pytest.mark.parametrize(
    ("contract", "legacy_field"),
    [
        ({"mode": "count", "branch_count": 2}, "branch_count"),
        (
            {
                "mode": "directions",
                "required_outlet_directions": ["+Y", "-Y"],
            },
            "required_outlet_directions",
        ),
        (
            {
                "mode": "vectors",
                "required_outlet_vectors": [[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            },
            "required_outlet_vectors",
        ),
        (
            {
                "mode": "outlets",
                "required_outlets": [
                    {"axis": [-1.0, 0.0, 1.0], "length": 40.0},
                    {"axis": [1.0, 0.0, 1.0], "length": 60.0},
                ],
            },
            "required_outlets",
        ),
    ],
)
def test_branch_outlet_contract_modes_flatten_losslessly(contract, legacy_field):
    llm_intent = LLMProductionIntent.model_validate(_branch_payload(contract))
    production = llm_intent.to_production_intent()
    branch = production.target_behavior[0]

    assert branch.include_primary_outlet is False
    populated = {
        field
        for field in (
            "branch_count",
            "required_outlet_directions",
            "required_outlet_vectors",
            "required_outlets",
        )
        if getattr(branch, field) not in (None, [])
    }
    assert populated == {legacy_field}
    if legacy_field != "branch_count":
        assert branch.branch_count is None


def test_branch_boundary_rejects_legacy_mixed_and_zero_vector_contracts():
    legacy = _branch_payload(
        {
            "mode": "vectors",
            "required_outlet_vectors": [[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        }
    )
    legacy["target_behavior"][0]["required_outlet_directions"] = ["+Y", "-Y"]
    with pytest.raises(ValidationError):
        LLMProductionIntent.model_validate(legacy)

    mixed = _branch_payload(
        {
            "mode": "vectors",
            "required_outlet_vectors": [[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            "required_outlet_directions": ["+Y", "-Y"],
        }
    )
    with pytest.raises(ValidationError):
        LLMProductionIntent.model_validate(mixed)

    zero = _branch_payload(
        {
            "mode": "vectors",
            "required_outlet_vectors": [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
        }
    )
    with pytest.raises(ValidationError, match="finite non-zero"):
        LLMProductionIntent.model_validate(zero)


def test_existing_flat_production_branch_contract_remains_compatible_and_safe():
    payload = _branch_payload({"mode": "count", "branch_count": 2})
    payload["target_behavior"][0].pop("outlet_contract")
    payload["target_behavior"][0]["required_outlet_vectors"] = [
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ]

    production = ProductionIntent.model_validate(payload)
    assert len(production.target_behavior[0].required_outlet_vectors) == 2

    payload["target_behavior"][0]["required_outlet_vectors"][0] = [0.0, 0.0, 0.0]
    with pytest.raises(ValidationError, match="finite non-zero"):
        ProductionIntent.model_validate(payload)


def test_two_llm_authored_binary_goals_represent_four_total_open_ends():
    payload = _branch_payload(
        {
            "mode": "vectors",
            "required_outlet_vectors": [[-1.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        }
    )
    payload["target_behavior"][0]["goal_id"] = "left_binary_y"
    payload["target_behavior"].append(
        {
            "goal_id": "right_binary_y",
            "depends_on_goal_ids": ["left_binary_y"],
            "allow_parallel": False,
            "type": "branch",
            "direction": "+X",
            "outlet_contract": {
                "mode": "vectors",
                "required_outlet_vectors": [
                    [1.0, 0.0, 1.0],
                    [1.0, 0.0, -1.0],
                ],
            },
            "include_primary_outlet": False,
            "junction_style": "smooth_hub",
        }
    )
    payload["expected_open_ports"] = 3

    llm_intent = LLMProductionIntent.model_validate(payload)
    production = llm_intent.to_production_intent()

    assert production.expected_open_ports == 3
    assert [goal.goal_id for goal in production.target_behavior] == [
        "left_binary_y",
        "right_binary_y",
    ]
    assert all(
        len(goal.required_outlet_vectors) == 2
        and goal.include_primary_outlet is False
        for goal in production.target_behavior
    )


def test_higher_degree_llm_branch_is_rejected_without_auto_decomposition():
    payload = _branch_payload(
        {
            "mode": "vectors",
            "required_outlet_vectors": [[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        }
    )
    payload["target_behavior"][0]["include_primary_outlet"] = True
    payload["expected_open_ports"] = 3

    with pytest.raises(ValidationError, match="exactly one binary split"):
        LLMProductionIntent.model_validate(payload)


def test_exact_user_prompt_contract_converts_through_existing_runtime_validators():
    llm_intent = LLMProductionIntent.model_validate(_exact_prompt_payload())
    production = llm_intent.to_production_intent()
    intent = llm_intent.to_intent_result()

    assert isinstance(production, ProductionIntent)
    assert [goal.type for goal in production.target_behavior] == [
        "move",
        "connector",
        "diameter_change",
        "move",
        "end",
    ]
    assert intent.target_behavior[0].length == 80.0
    assert intent.target_behavior[1].component == "coupling"
    assert intent.target_behavior[2].transition_length == 40.0
    assert intent.target_behavior[2].diameter_out == 12.0
    assert intent.target_behavior[2].wall_thickness_out == 1.5
    assert intent.target_behavior[3].length == 60.0
    assert intent.target_behavior[4].end_type == "cap"
    assert intent.expected_open_ports == 0


def test_source_numeric_literal_schema_blocks_raw_float_corruption():
    literals = ["-1", "0", "1", "1.5", "2", "12", "20", "30", "40", "60", "80"]
    schema = gemini_json_schema(
        LLMProductionIntent,
        number_literals=literals,
    )
    number_definition = next(
        definition
        for definition in schema["$defs"].values()
        if definition == {"type": "string", "enum": literals}
    )
    assert number_definition["enum"] == literals

    def encode_floats(value):
        if isinstance(value, float):
            return str(int(value)) if value.is_integer() else str(value)
        if isinstance(value, list):
            return [encode_floats(item) for item in value]
        if isinstance(value, dict):
            return {key: encode_floats(item) for key, item in value.items()}
        return value

    encoded = encode_floats(_exact_prompt_payload())
    Draft202012Validator(schema).validate(encoded)
    parsed = LLMProductionIntent.model_validate(encoded)
    assert parsed.target_behavior[2].wall_thickness_out == 1.5

    encoded["target_behavior"][2]["wall_thickness_out"] = 1.5e-121
    with pytest.raises(JSONSchemaValidationError):
        Draft202012Validator(schema).validate(encoded)
