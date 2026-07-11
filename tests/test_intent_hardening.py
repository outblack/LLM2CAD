from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

import cadgen.pipeline as pipeline
from cadgen.config import load_settings
from cadgen.gemini_client import (
    GeminiBudgetError,
    GeminiConfigError,
    GeminiInvalidRequestError,
    GeminiRequestError,
    HostContractValidationError,
    StructuredOutputIncompleteError,
)
from cadgen.prompts import intent_prompt, intent_system_instruction
from cadgen.schemas import (
    BranchGoalOutletSpec,
    GlobalSpec,
    Goal,
    IntentRepairAdvice,
    IntentRepairAdviceWire,
    IntentResult,
    LLMIntentJSONEnvelope,
    LLMProductionIntent,
    ProductionIntent,
)


PROMPT = (
    "외경 20 mm, 두께 2 mm의 중공 파이프를 +X로 80 mm 직진시키고, "
    "길이 30 mm coupling을 설치한다. 이어서 40 mm 구간에서 외경을 "
    "12 mm, 두께를 1.5 mm로 줄인 뒤 60 mm 직진시키고 cap으로 닫는다."
)


def _settings():
    return replace(
        load_settings(Path("missing.env")),
        step_repair_attempts=1,
    )


def _production_intent(
    *,
    wall_thickness_out: float = 1.5,
    transition_length: float | None = 40.0,
    hard_constraints: list[str] | None = None,
) -> ProductionIntent:
    return ProductionIntent.model_validate(
        {
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
                    "goal_id": "initial_straight_section",
                    "depends_on_goal_ids": [],
                    "allow_parallel": False,
                    "type": "move",
                    "direction": "+X",
                    "length": 80.0,
                },
                {
                    "goal_id": "coupling_installation",
                    "depends_on_goal_ids": ["initial_straight_section"],
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
                    "goal_id": "diameter_reduction",
                    "depends_on_goal_ids": ["coupling_installation"],
                    "allow_parallel": False,
                    "type": "diameter_change",
                    "diameter_out": 12.0,
                    "wall_thickness_out": wall_thickness_out,
                    "transition_length": transition_length,
                },
                {
                    "goal_id": "reduced_straight_section",
                    "depends_on_goal_ids": ["diameter_reduction"],
                    "allow_parallel": False,
                    "type": "move",
                    "direction": "+X",
                    "length": 60.0,
                },
                {
                    "goal_id": "terminal_cap",
                    "depends_on_goal_ids": ["reduced_straight_section"],
                    "allow_parallel": False,
                    "type": "end",
                    "end_type": "cap",
                },
            ],
            "expected_open_ports": 0,
            "expected_open_ports_source": "derived",
            "required_components": ["coupling"],
            "hard_constraints": hard_constraints or [],
            "geometric_constraints": [],
            "design_notes": [],
        }
    )


def _llm_intent_envelope() -> LLMIntentJSONEnvelope:
    wire = LLMProductionIntent.model_validate(
        {
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
                    "goal_id": "initial_straight_section",
                    "depends_on_goal_ids": [],
                    "allow_parallel": False,
                    "type": "move",
                    "direction": "+X",
                    "length": 80.0,
                },
                {
                    "goal_id": "coupling_installation",
                    "depends_on_goal_ids": ["initial_straight_section"],
                    "allow_parallel": False,
                    "type": "connector",
                    "length": 30.0,
                    "component": "coupling",
                },
                {
                    "goal_id": "diameter_reduction",
                    "depends_on_goal_ids": ["coupling_installation"],
                    "allow_parallel": False,
                    "type": "diameter_change",
                    "diameter_out": 12.0,
                    "wall_thickness_out": 1.5,
                    "transition_length": 40.0,
                },
                {
                    "goal_id": "final_straight_section",
                    "depends_on_goal_ids": ["diameter_reduction"],
                    "allow_parallel": False,
                    "type": "move",
                    "direction": "+X",
                    "length": 60.0,
                },
                {
                    "goal_id": "end_cap",
                    "depends_on_goal_ids": ["final_straight_section"],
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
    )
    return LLMIntentJSONEnvelope(intent_json=wire.model_dump_json())


def test_intent_safety_rejects_tiny_dimension_and_lost_explicit_mm_values():
    corrupted = _production_intent(
        wall_thickness_out=1.5e-121,
        transition_length=None,
    ).to_intent_result()

    with pytest.raises(ValueError) as exc_info:
        pipeline._validate_intent_safety(PROMPT, corrupted, _settings())

    message = str(exc_info.value)
    assert "modeling_tolerance" in message
    assert "wall_thickness_out" in message
    assert "40.0" in message
    assert "1.5" in message


def _heading_intent(
    *,
    turn_direction: str | None,
    turn_angle: float,
    plane_normal: tuple[float, float, float] | None = None,
) -> ProductionIntent:
    return ProductionIntent.model_validate(
        {
            "global_spec": {
                "outer_diameter": 20.0,
                "wall_thickness": 2.0,
                "is_hollow": True,
                "units": "mm",
            },
            "start_position": [0.0, 0.0, 0.0],
            "start_axis": [0.0, 0.0, 1.0],
            "target_behavior": [
                {
                    "goal_id": "vertical_straight",
                    "depends_on_goal_ids": [],
                    "allow_parallel": False,
                    "type": "move",
                    "direction": "+Z",
                    "length": 40.0,
                },
                {
                    "goal_id": "bend",
                    "depends_on_goal_ids": ["vertical_straight"],
                    "allow_parallel": False,
                    "type": "turn",
                    "direction": turn_direction,
                    "angle": turn_angle,
                    "bend_radius": 30.0,
                    "plane_normal": plane_normal,
                },
            ],
            "expected_open_ports": 1,
            "expected_open_ports_source": "derived",
            "required_components": [],
            "hard_constraints": [],
            "geometric_constraints": [],
            "design_notes": [],
        }
    )


def test_intent_safety_rejects_nonzero_turn_with_unchanged_outlet_heading():
    intent = _heading_intent(turn_direction="+Z", turn_angle=30.0).to_intent_result()

    with pytest.raises(
        ValueError, match="sequential heading contradiction"
    ) as exc_info:
        pipeline._validate_intent_safety(
            "Create a vertical straight followed by a bend.", intent, _settings()
        )

    message = str(exc_info.value)
    assert "turn.direction=+Z" in message
    assert "turn.angle=30" in message
    assert "0 degrees apart" in message


def test_intent_safety_accepts_heading_consistent_right_angle_turn():
    intent = _heading_intent(turn_direction="-Y", turn_angle=90.0).to_intent_result()

    pipeline._validate_intent_safety(
        "Create a vertical straight followed by a right-angle bend.",
        intent,
        _settings(),
    )


def test_intent_safety_rejects_too_close_off_axis_relative_spline_entry():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[
            Goal(goal_id="move", type="move", direction="+X", length=40.0),
            Goal(
                goal_id="turn",
                depends_on_goal_ids=["move"],
                type="turn",
                direction="+Z",
                angle=90.0,
                bend_radius=30.0,
            ),
            Goal(
                goal_id="coil",
                depends_on_goal_ids=["turn"],
                type="route",
                path_kind="spline",
                waypoint_frame="relative_to_target",
                required_waypoints=[(10.0, 0.0, 5.0), (30.0, 20.0, 20.0)],
            ),
        ],
    )

    with pytest.raises(ValueError, match="circular-entry radius"):
        pipeline._validate_intent_safety("Create a rising coil.", intent, _settings())

    safe = intent.model_copy(deep=True)
    safe.target_behavior[2] = safe.target_behavior[2].model_copy(
        update={
            "required_waypoints": [
                (0.0, 0.0, 30.0),
                (36.0, 24.0, 60.0),
            ]
        }
    )
    pipeline._validate_intent_safety("Create a rising coil.", safe, _settings())


def test_intent_safety_rejects_curvature_infeasible_relative_spline_chain():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(0.0, 0.0, 1.0),
        target_behavior=[
            Goal(
                goal_id="coil",
                type="route",
                path_kind="spline",
                waypoint_frame="relative_to_target",
                required_waypoints=[
                    (5.0, 0.0, 10.0),
                    (20.0, 30.0, 20.0),
                    (-1.0, 40.0, 30.0),
                    (-1.0, 15.0, 40.0),
                ],
            ),
        ],
    )

    with pytest.raises(ValueError, match="predicts a minimum curvature radius"):
        pipeline._validate_intent_safety("Create a rising coil.", intent, _settings())


def test_sequential_heading_check_stops_at_parallel_goal_boundary():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[
            Goal(goal_id="main", type="move", direction="+X", length=20.0),
            Goal(
                goal_id="parallel",
                type="move",
                direction="+Y",
                length=20.0,
                allow_parallel=True,
            ),
        ],
    )

    assert pipeline._sequential_heading_issues(intent, _settings()) == []


def test_branch_successor_preflight_rejects_latest_fixed_spline_contract():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[
            Goal(goal_id="inlet", type="move", direction="+X", length=40.0),
            Goal(
                goal_id="first_junction",
                depends_on_goal_ids=["inlet"],
                type="branch",
                required_outlet_directions=["+Y"],
                include_primary_outlet=True,
            ),
            Goal(
                goal_id="central_spline",
                depends_on_goal_ids=["first_junction"],
                type="route",
                path_kind="spline",
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
                required_waypoints=[
                    (30.0, 0.0, 0.0),
                    (60.0, 20.0, 20.0),
                    (90.0, 0.0, 0.0),
                ],
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"central_spline fixed required-anchor spline is infeasible.*14\.242135",
    ):
        pipeline._validate_intent_safety(
            "Create a Y branch followed by a qualitative spline.",
            intent,
            _settings(),
        )


def test_branch_successor_preflight_accepts_when_one_authored_outlet_is_feasible():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[
            Goal(goal_id="inlet", type="move", direction="+X", length=40.0),
            Goal(
                goal_id="first_junction",
                depends_on_goal_ids=["inlet"],
                type="branch",
                required_outlet_directions=["+Y"],
                include_primary_outlet=True,
            ),
            Goal(
                goal_id="central_spline",
                depends_on_goal_ids=["first_junction"],
                type="route",
                path_kind="spline",
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
                required_waypoints=[
                    (30.0, 0.0, 0.0),
                    (60.0, 0.0, 0.0),
                    (90.0, 0.0, 0.0),
                ],
            ),
        ],
    )

    assert pipeline._branch_successor_spline_issues(intent, _settings()) == []


def test_relative_spline_curvature_uses_current_post_transition_diameter():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(0.0, 0.0, 1.0),
        target_behavior=[
            Goal(
                goal_id="grow",
                type="diameter_change",
                diameter_out=40.0,
                transition_length=50.0,
            ),
            Goal(
                goal_id="route",
                depends_on_goal_ids=["grow"],
                type="route",
                path_kind="spline",
                waypoint_frame="relative_to_target",
                required_waypoints=[
                    (0.0, 0.0, 25.0),
                    (30.0, 20.0, 50.0),
                ],
            ),
        ],
    )

    with pytest.raises(ValueError, match="at least 40"):
        pipeline._validate_intent_safety("Grow, then curve.", intent, _settings())


def test_qualitative_relative_spline_is_not_silently_scaled():
    original_points = [
        (7.5, 0.0, 15.0),
        (30.0, 45.0, 30.0),
        (-1.5, 60.0, 45.0),
        (-1.5, 22.5, 60.0),
    ]
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(0.0, 0.0, 1.0),
        target_behavior=[
            Goal(
                goal_id="coil",
                type="route",
                path_kind="spline",
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="uniform_expand_for_safety",
                required_waypoints=original_points,
            )
        ],
    )

    canonical = pipeline._canonicalize_dependent_intent_geometry(intent, _settings())
    goal = canonical.target_behavior[0]
    assert goal.waypoint_safety_scale is None
    assert goal.required_waypoints == original_points
    with pytest.raises(
        ValueError,
        match="predicts a minimum curvature radius.*at least",
    ):
        pipeline._validate_intent_safety(
            "Create a rising coil.", canonical, _settings()
        )
    assert (
        pipeline._canonicalize_dependent_intent_geometry(
            canonical, _settings()
        ).model_dump()
        == canonical.model_dump()
    )


def test_fixed_or_downstream_absolute_relative_waypoints_are_not_scaled():
    unsafe_points = [
        (5.0, 0.0, 10.0),
        (20.0, 30.0, 20.0),
        (-1.0, 40.0, 30.0),
        (-1.0, 15.0, 40.0),
    ]
    fixed = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(0.0, 0.0, 1.0),
        target_behavior=[
            Goal(
                goal_id="fixed",
                type="route",
                path_kind="spline",
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
                required_waypoints=unsafe_points,
            )
        ],
    )
    downstream_absolute = fixed.model_copy(
        deep=True,
        update={
            "target_behavior": [
                fixed.target_behavior[0].model_copy(
                    update={"waypoint_scale_policy": "uniform_expand_for_safety"}
                ),
                Goal(
                    goal_id="absolute_end",
                    depends_on_goal_ids=["fixed"],
                    type="route",
                    path_kind="line",
                    terminal_position=(200.0, 0.0, 100.0),
                ),
            ]
        },
    )

    for intent in (fixed, downstream_absolute):
        canonical = pipeline._canonicalize_dependent_intent_geometry(
            intent, _settings()
        )
        assert canonical.target_behavior[0].waypoint_safety_scale is None
        assert canonical.target_behavior[0].required_waypoints == unsafe_points


def test_excessive_qualitative_scale_is_returned_to_intent_repair():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(0.0, 0.0, 1.0),
        target_behavior=[
            Goal(
                goal_id="overscaled",
                type="route",
                path_kind="spline",
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="uniform_expand_for_safety",
                required_waypoints=[
                    (0.0, 5.0, 5.0),
                    (20.0, 40.0, 20.0),
                    (0.0, 80.0, 40.0),
                    (-1.0, 40.0, 60.0),
                    (15.0, 10.0, 80.0),
                ],
            )
        ],
    )

    canonical = pipeline._canonicalize_dependent_intent_geometry(intent, _settings())
    assert canonical.target_behavior[0].waypoint_safety_scale is None
    with pytest.raises(ValueError, match="predicts a minimum curvature radius"):
        pipeline._validate_intent_safety("Create a broad coil.", canonical, _settings())


def test_authored_route_length_disables_uniform_waypoint_scaling():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(0.0, 0.0, 1.0),
        target_behavior=[
            Goal(
                goal_id="fixed",
                type="route",
                path_kind="spline",
                length=100.0,
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="uniform_expand_for_safety",
                required_waypoints=[
                    (5.0, 0.0, 10.0),
                    (20.0, 30.0, 20.0),
                    (-1.0, 40.0, 30.0),
                    (-1.0, 15.0, 40.0),
                ],
            )
        ],
    )

    canonical = pipeline._canonicalize_dependent_intent_geometry(intent, _settings())
    assert canonical.target_behavior[0].waypoint_safety_scale is None
    with pytest.raises(ValueError, match="predicts a minimum curvature radius"):
        pipeline._validate_intent_safety(
            "Create a 100 mm coil.", canonical, _settings()
        )


def test_signed_turn_plane_hint_is_canonicalized_against_incoming_heading():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(1.0, 1.0, 0.0),
        target_behavior=[
            Goal(
                goal_id="bend",
                type="turn",
                angle=40.0,
                plane_normal=(0.0, 1.0, 1.0),
            )
        ],
    )

    canonical = pipeline._canonicalize_dependent_intent_geometry(intent, _settings())
    normal = canonical.target_behavior[0].plane_normal
    assert normal is not None
    incoming = (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0)
    assert sum(a * b for a, b in zip(incoming, normal)) == pytest.approx(0.0)
    pipeline._validate_intent_safety("Create an outward bend.", canonical, _settings())


def test_intent_safety_represents_noncardinal_turn_by_signed_plane_not_llm_axis():
    intent = _heading_intent(
        turn_direction=None,
        turn_angle=30.0,
        plane_normal=(1.0, 0.0, 0.0),
    ).to_intent_result()

    pipeline._validate_intent_safety(
        "Create a vertical straight followed by an explicit 30-degree elbow.",
        intent,
        _settings(),
    )


def test_intent_safety_rejects_turn_plane_parallel_to_incoming_heading():
    intent = _heading_intent(turn_direction="-Y", turn_angle=90.0).to_intent_result()
    goals = list(intent.target_behavior)
    goals[1] = goals[1].model_copy(update={"plane_normal": (0.0, 0.0, 1.0)})
    intent = intent.model_copy(update={"target_behavior": goals})

    with pytest.raises(ValueError, match="plane_normal must be perpendicular"):
        pipeline._validate_intent_safety(
            "Create a vertical straight followed by a right-angle bend.",
            intent,
            _settings(),
        )


@pytest.mark.parametrize(
    "goal",
    [
        Goal(
            goal_id="transition",
            type="diameter_change",
            direction="+Y",
            diameter_out=12.0,
            wall_thickness_out=1.5,
            transition_length=20.0,
        ),
        Goal(
            goal_id="coupling",
            type="connector",
            direction="+Y",
            length=20.0,
            component="coupling",
        ),
    ],
)
def test_intent_safety_rejects_axial_modules_that_try_to_change_heading(goal):
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(0.0, 0.0, 1.0),
        target_behavior=[goal],
    )

    with pytest.raises(ValueError, match="cannot change the incoming axial heading"):
        pipeline._validate_intent_safety(
            "Create an axial fitting.", intent, _settings()
        )


def test_intent_safety_checks_line_direction_before_terminal_axis():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(0.0, 0.0, 1.0),
        target_behavior=[
            Goal(
                goal_id="bad_line",
                type="route",
                path_kind="line",
                direction="+Y",
                length=20.0,
                terminal_axis=(0.0, 1.0, 0.0),
            )
        ],
    )

    with pytest.raises(ValueError, match="line route direction cannot mate"):
        pipeline._validate_intent_safety("Create a line route.", intent, _settings())


def test_intent_heading_tolerance_matches_static_direction_validator():
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[
            Goal(
                goal_id="near_right_angle",
                type="turn",
                direction="+Y",
                angle=89.5,
            )
        ],
    )

    pipeline._validate_intent_safety(
        "Create a near-right-angle bend.", intent, _settings()
    )


def test_intent_extraction_repairs_schema_valid_contract_without_lineage_reset():
    class FakeGemini:
        def __init__(self):
            self.calls: list[str] = []
            self.reset_calls: list[str] = []

        def stream_structured(self, prompt, schema, *, part):
            del schema, part
            self.calls.append(prompt)
            if len(self.calls) == 1:
                return _production_intent(
                    wall_thickness_out=1.5e-121,
                    transition_length=None,
                )
            return _production_intent()

        def has_previous(self, part):
            return part == "intent" and bool(self.calls)

        def reset_lineage(self, part):
            self.reset_calls.append(part)

    gemini = FakeGemini()
    result = pipeline._extract_intent(
        PROMPT,
        _settings(),
        dry_run=False,
        gemini=gemini,
    )

    transition = result.target_behavior[2]
    assert transition.wall_thickness_out == 1.5
    assert transition.transition_length == 40.0
    assert len(gemini.calls) == 2
    assert gemini.reset_calls == []
    assert "Validation diagnostic" in gemini.calls[1]
    assert "prior intent was schema-valid" in gemini.calls[1]
    assert "preserving every unaffected topology" in gemini.calls[1]
    assert "modeling_tolerance" in gemini.calls[1]
    assert "User request:" in gemini.calls[1]
    assert PROMPT in gemini.calls[1]
    assert "[20.0, 2.0, 80.0, 30.0, 40.0, 12.0, 1.5, 60.0]" in (gemini.calls[1])


def test_intent_scope_rejection_runs_advisor_then_reenters_authoring_loop(tmp_path):
    class FakeGemini:
        supports_intent_repair_advisor = True
        supports_system_instruction = True
        supports_numeric_literals = True

        def __init__(self):
            self.parts: list[str] = []
            self.prompts: list[str] = []
            self.intent_calls = 0

        def stream_structured(self, prompt, unused_schema, **kwargs):
            part = kwargs["part"]
            self.parts.append(part)
            self.prompts.append(prompt)
            if part == "intent_repair_advisor":
                return IntentRepairAdvice(
                    diagnosis_class="candidate_contract_error",
                    disposition="retry_intent",
                    candidate_fixable=True,
                    summary="The candidate invented an unsupported hard constraint.",
                    causal_chain=[
                        "The source request contains no such requirement.",
                        "hard_constraints therefore blocks scope validation.",
                    ],
                    preserve_requirements=["all source-authored measurements"],
                    change_fields=["/hard_constraints"],
                    avoid=["dropping any source-authored requirement"],
                    intent_instruction=(
                        "Remove only the model-invented constraint and preserve the "
                        "complete agenda."
                    ),
                )
            self.intent_calls += 1
            return _production_intent(
                hard_constraints=(
                    ["unsupported:model-invented decorative loop"]
                    if self.intent_calls == 1
                    else []
                )
            )

        def reset_lineage(self, unused_part):
            return None

    attempts: list[dict] = []
    diagnostics: list[dict] = []
    attempts_path = tmp_path / "intent_attempts.json"
    diagnostics_path = tmp_path / "intent_diagnostics.json"
    gemini = FakeGemini()

    result = pipeline._extract_intent(
        PROMPT,
        _settings(),
        dry_run=False,
        gemini=gemini,
        attempt_journal=attempts,
        attempt_journal_path=attempts_path,
        diagnostic_journal=diagnostics,
        diagnostic_journal_path=diagnostics_path,
    )

    assert result.hard_constraints == []
    assert gemini.parts == ["intent", "intent_repair_advisor", "intent"]
    assert [item["status"] for item in attempts] == ["rejected", "accepted"]
    assert attempts[0]["phase"] == "intent_scope"
    assert attempts[0]["issue_codes"] == ["UNSUPPORTED_HARD_CONSTRAINT"]
    assert attempts[0]["advisor_status"] == "complete"
    assert attempts[0]["advisor_disposition"] == "retry_intent"
    assert attempts[1]["scope_validated"] is True
    assert diagnostics[0]["rejected_candidate"]["hard_constraints"] == [
        "unsupported:model-invented decorative loop"
    ]
    assert diagnostics[0]["advisor"]["disposition"] == "retry_intent"
    assert diagnostics[0]["will_retry"] is True
    retry_prompt = gemini.prompts[-1]
    assert "UNSUPPORTED_HARD_CONSTRAINT" in retry_prompt
    assert "model-invented decorative loop" in retry_prompt
    assert "independent_advisor" in retry_prompt
    assert json.loads(attempts_path.read_text(encoding="utf-8")) == attempts
    assert json.loads(diagnostics_path.read_text(encoding="utf-8")) == diagnostics


def test_explicit_closed_loop_capability_gap_is_diagnosed_without_blind_retry(
    tmp_path,
):
    class FakeGemini:
        supports_intent_repair_advisor = True
        supports_system_instruction = True
        supports_numeric_literals = True

        def __init__(self):
            self.parts: list[str] = []

        def stream_structured(self, unused_prompt, unused_schema, **kwargs):
            part = kwargs["part"]
            self.parts.append(part)
            if part == "intent":
                return _production_intent(
                    hard_constraints=[
                        "unsupported:마지막 구간은 시작 구간과 연결한 폐곡선이어야 한다"
                    ]
                )
            return IntentRepairAdvice(
                diagnosis_class="unsupported_user_requirement",
                disposition="stop_contract_infeasible",
                candidate_fixable=False,
                summary=(
                    "The anchored catalog cannot close one unbranched front back "
                    "onto START without deleting the requested topology."
                ),
                causal_chain=[
                    "The request explicitly requires one closed loop.",
                    "The catalog begins with one construction front.",
                    "connect requires two distinct open fronts.",
                ],
                preserve_requirements=["closed-loop path"],
                change_fields=[],
                avoid=["silently deleting the closed-loop requirement"],
                intent_instruction="Do not retry until the catalog supports closure.",
            )

        def reset_lineage(self, unused_part):
            raise AssertionError("a terminal capability diagnosis must not reset")

    attempts: list[dict] = []
    diagnostics: list[dict] = []
    gemini = FakeGemini()

    with pytest.raises(
        pipeline._IntentScopeValidationError,
        match="UNSUPPORTED_HARD_CONSTRAINT",
    ):
        pipeline._extract_intent(
            PROMPT + " 마지막 구간은 시작 구간과 연결한 폐곡선이어야 한다.",
            replace(_settings(), intent_repair_attempts=5),
            dry_run=False,
            gemini=gemini,
            attempt_journal=attempts,
            diagnostic_journal=diagnostics,
            diagnostic_journal_path=tmp_path / "intent_diagnostics.json",
        )

    assert gemini.parts == ["intent", "intent_repair_advisor"]
    assert [item["status"] for item in attempts] == ["rejected"]
    assert attempts[0]["will_retry"] is False
    assert attempts[0]["terminal_reason"] == "stop_contract_infeasible"
    assert diagnostics[0]["advisor"]["diagnosis_class"] == (
        "unsupported_user_requirement"
    )
    assert diagnostics[0]["rejected_candidate"]["hard_constraints"] == [
        "unsupported:마지막 구간은 시작 구간과 연결한 폐곡선이어야 한다"
    ]


def test_pipeline_preserves_scope_issue_and_reports_intent_advisor_audit(
    monkeypatch,
    tmp_path,
):
    class FakeGemini:
        supports_intent_repair_advisor = True
        supports_system_instruction = True
        supports_numeric_literals = True

        def __init__(self):
            self.parts: list[str] = []

        def stream_structured(self, unused_prompt, unused_schema, **kwargs):
            part = kwargs["part"]
            self.parts.append(part)
            if part == "intent":
                return _production_intent(
                    hard_constraints=[
                        "unsupported:마지막 구간은 시작 구간과 연결한 폐곡선이어야 한다"
                    ]
                )
            return IntentRepairAdvice(
                diagnosis_class="unsupported_user_requirement",
                disposition="stop_contract_infeasible",
                candidate_fixable=False,
                summary="The current catalog cannot realize closure to START.",
                causal_chain=[
                    "One unbranched construction front cannot satisfy connect."
                ],
                preserve_requirements=["closed-loop path"],
                change_fields=[],
                avoid=["dropping the closed-loop requirement"],
                intent_instruction="Stop until a closure primitive is available.",
            )

        def reset_lineage(self, unused_part):
            return None

    fake = FakeGemini()
    monkeypatch.setattr(pipeline, "GeminiClient", lambda unused_settings: fake)
    settings = replace(
        _settings().with_overrides(output_dir=tmp_path, skip_freecad=True),
        intent_repair_attempts=5,
        stream_thinking_summary=False,
    )

    with pytest.raises(pipeline.StaticValidationError) as exc_info:
        pipeline.run_pipeline(
            PROMPT + " 마지막 구간은 시작 구간과 연결한 폐곡선이어야 한다.",
            settings,
        )

    report_path = Path(exc_info.value.artifact_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    attempts = json.loads(
        (report_path.parent / "intent_attempts.json").read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        (report_path.parent / "intent_diagnostics.json").read_text(encoding="utf-8")
    )

    assert fake.parts == ["intent", "intent_repair_advisor"]
    assert report["failed_stage"] == "intent_scope"
    assert report["top_issues"] == ["FINAL_01_UNSUPPORTED_HARD_CONSTRAINT"]
    assert report["intent_attempt_count"] == 1
    assert report["intent_repair_count"] == 0
    assert report["intent_advisor_call_count"] == 1
    assert report["intent_advisor_success_count"] == 1
    assert report["intent_advisor_failure_count"] == 0
    assert report["artifacts"]["intent_diagnostics_path"].endswith(
        "intent_diagnostics.json"
    )
    assert attempts[0]["status"] == "rejected"
    assert attempts[0]["phase"] == "intent_scope"
    assert diagnostics[0]["advisor"]["disposition"] == ("stop_contract_infeasible")


def test_identical_intent_and_scope_failure_stops_after_one_lineage_reset():
    class FakeGemini:
        supports_intent_repair_advisor = True
        supports_system_instruction = True
        supports_numeric_literals = True

        def __init__(self):
            self.parts: list[str] = []
            self.reset_calls: list[str] = []

        def stream_structured(self, unused_prompt, unused_schema, **kwargs):
            part = kwargs["part"]
            self.parts.append(part)
            if part == "intent":
                return _production_intent(
                    hard_constraints=["unsupported:model-invented constraint"]
                )
            return IntentRepairAdvice(
                diagnosis_class="candidate_contract_error",
                disposition="retry_intent",
                candidate_fixable=True,
                summary="Retry by changing the hard constraint mapping.",
                causal_chain=["The same candidate field blocks intent scope."],
                preserve_requirements=["all user-authored requirements"],
                change_fields=["/hard_constraints"],
                avoid=["repeating the identical candidate"],
                intent_instruction="Return a materially changed complete intent.",
            )

        def reset_lineage(self, part):
            self.reset_calls.append(part)

    gemini = FakeGemini()
    attempts: list[dict] = []
    diagnostics: list[dict] = []

    with pytest.raises(pipeline._IntentScopeValidationError):
        pipeline._extract_intent(
            PROMPT,
            replace(_settings(), intent_repair_attempts=8),
            dry_run=False,
            gemini=gemini,
            attempt_journal=attempts,
            diagnostic_journal=diagnostics,
        )

    assert gemini.parts == [
        "intent",
        "intent_repair_advisor",
        "intent",
        "intent_repair_advisor",
        "intent",
    ]
    assert gemini.reset_calls == ["intent"]
    assert len(attempts) == 3
    assert [item["exact_failure_repeat_count"] for item in attempts] == [1, 2, 3]
    assert attempts[-1]["will_retry"] is False
    assert attempts[-1]["terminal_reason"] == ("identical_intent_failure_stagnation")
    assert diagnostics[-1]["terminal_reason"] == ("identical_intent_failure_stagnation")


def _noncardinal_straight_candidate(*, repaired: bool) -> IntentResult:
    straight_after_turn = (
        Goal(
            goal_id="G3",
            depends_on_goal_ids=["G2"],
            type="route",
            path_kind="line",
            length=40.0,
        )
        if repaired
        else Goal(
            goal_id="G3",
            depends_on_goal_ids=["G2"],
            type="move",
            direction="+X",
            length=40.0,
        )
    )
    return IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[
            Goal(goal_id="G1", type="move", direction="+X", length=40.0),
            Goal(
                goal_id="G2",
                depends_on_goal_ids=["G1"],
                type="turn",
                angle=60.0,
                plane_normal=(0.0, 0.0, 1.0),
                bend_radius=20.0,
            ),
            straight_after_turn,
        ],
        expected_open_ports=1,
        expected_open_ports_source="derived",
        hard_constraints=(
            [] if repaired else ["unsupported:latent candidate constraint"]
        ),
    )


def test_advisor_host_rejections_feed_secondary_reviewer_and_author_retry():
    class FakeGemini:
        supports_intent_repair_advisor = True
        supports_intent_repair_reviewer = True
        supports_system_instruction = True
        supports_numeric_literals = True

        def __init__(self):
            self.parts: list[str] = []
            self.prompts: list[str] = []
            self.intent_calls = 0

        def stream_structured(self, prompt, unused_schema, **kwargs):
            part = kwargs["part"]
            self.parts.append(part)
            self.prompts.append(prompt)
            if part == "intent":
                self.intent_calls += 1
                return _noncardinal_straight_candidate(repaired=self.intent_calls > 1)
            if part == "intent_repair_advisor":
                return IntentRepairAdvice(
                    diagnosis_class="unsupported_user_requirement",
                    disposition="stop_contract_infeasible",
                    candidate_fixable=False,
                    summary="Incorrectly focused on a latent candidate field.",
                    causal_chain=["A latent hard constraint was present."],
                    preserve_requirements=["all measurements"],
                    change_fields=[],
                    avoid=["retry"],
                    intent_instruction="Stop.",
                )
            return IntentRepairAdvice(
                diagnosis_class="candidate_contract_error",
                disposition="retry_intent",
                candidate_fixable=True,
                summary="The current heading failure is repairable.",
                causal_chain=[
                    "A cardinal move cannot follow a non-cardinal turn tangent."
                ],
                preserve_requirements=["both 40 mm straight lengths", "60 degree turn"],
                change_fields=["/target_behavior/2"],
                avoid=["resetting the straight to +X"],
                intent_instruction=(
                    "Use a length-only line route that inherits the incoming tangent."
                ),
            )

        def reset_lineage(self, unused_part):
            return None

    gemini = FakeGemini()
    attempts: list[dict] = []
    diagnostics: list[dict] = []
    result = pipeline._extract_intent(
        "Create two 40 mm straights separated by a 60 degree turn.",
        _settings(),
        dry_run=False,
        gemini=gemini,
        attempt_journal=attempts,
        diagnostic_journal=diagnostics,
    )

    assert result.target_behavior[-1].type == "route"
    assert gemini.parts == [
        "intent",
        "intent_repair_advisor",
        "intent_repair_advisor",
        "intent_repair_reviewer",
        "intent",
    ]
    second_advisor_prompt = json.loads(gemini.prompts[2])
    assert len(second_advisor_prompt["prior_advisor_rejections"]) == 1
    assert (
        second_advisor_prompt["prior_advisor_rejections"][0]["host_rejection"]["code"]
        == "CURRENT_BLOCKER_REQUIRES_REPAIR"
    )
    assert attempts[0]["will_retry"] is True
    assert attempts[0]["advisor_source"] == "secondary_reviewer"
    assert diagnostics[0]["advisor_fallback_used"] is True
    assert [item["status"] for item in diagnostics[0]["advisor_attempts"]] == [
        "host_rejected",
        "host_rejected",
        "accepted",
    ]


def test_required_advisor_chain_failure_degrades_to_evidence_only_author_retry():
    class FakeGemini:
        supports_intent_repair_advisor = True
        supports_intent_repair_reviewer = True
        supports_system_instruction = True
        supports_numeric_literals = True

        def __init__(self):
            self.parts: list[str] = []
            self.intent_calls = 0

        def stream_structured(self, unused_prompt, unused_schema, **kwargs):
            part = kwargs["part"]
            self.parts.append(part)
            if part == "intent":
                self.intent_calls += 1
                return _noncardinal_straight_candidate(repaired=self.intent_calls > 1)
            raise GeminiRequestError(f"{part} unavailable")

        def reset_lineage(self, unused_part):
            return None

    gemini = FakeGemini()
    attempts: list[dict] = []
    diagnostics: list[dict] = []
    result = pipeline._extract_intent(
        "Create two 40 mm straights separated by a 60 degree turn.",
        replace(_settings(), intent_repair_advisor_required=True),
        dry_run=False,
        gemini=gemini,
        attempt_journal=attempts,
        diagnostic_journal=diagnostics,
    )

    assert result.target_behavior[-1].type == "route"
    assert gemini.parts == [
        "intent",
        "intent_repair_advisor",
        "intent_repair_advisor",
        "intent_repair_reviewer",
        "intent",
    ]
    assert attempts[0]["will_retry"] is True
    assert attempts[0]["terminal_reason"] is None
    assert attempts[0]["advisor_source"] == "deterministic_evidence_only"
    assert attempts[0]["advisor_fallback_used"] is True
    assert diagnostics[0]["advisor_status"] == "degraded_fallback"
    assert diagnostics[0]["advisor_required"] is True


def _terminal_capability_advice() -> IntentRepairAdvice:
    return IntentRepairAdvice(
        diagnosis_class="unsupported_user_requirement",
        disposition="stop_contract_infeasible",
        candidate_fixable=False,
        summary="The current catalog cannot realize the requirement.",
        causal_chain=["The current issue reports an unsupported capability."],
        preserve_requirements=["the unsupported requirement"],
        change_fields=[],
        avoid=["dropping the requirement"],
        intent_instruction="Preserve the blocking requirement.",
    )


def test_mixed_scope_issues_cannot_grant_advisor_terminal_authority():
    validation_details = [
        {
            "issue_code": "NON_BINARY_BRANCH_CONTRACT",
            "actual": {"nonbinary_branch_goal_ids": ["branch"]},
        },
        {
            "issue_code": "UNSUPPORTED_HARD_CONSTRAINT",
            "actual": {"source_provenance_complete": True},
        },
    ]

    with pytest.raises(
        pipeline._IntentAdvisorAuthorityError,
        match="only current unsupported capability issues",
    ) as exc_info:
        pipeline._validate_intent_repair_advice_authority(
            _terminal_capability_advice(),
            validation_details,
        )

    assert exc_info.value.code == "CURRENT_BLOCKER_REQUIRES_REPAIR"


def test_unsupported_scope_requires_deterministic_source_provenance_to_stop():
    validation_details = [
        {
            "issue_code": "UNSUPPORTED_HARD_CONSTRAINT",
            "actual": {
                "hard_constraints": ["unsupported:model-invented constraint"],
                "source_provenance_complete": False,
            },
        }
    ]

    with pytest.raises(pipeline._IntentAdvisorAuthorityError) as exc_info:
        pipeline._validate_intent_repair_advice_authority(
            _terminal_capability_advice(),
            validation_details,
        )

    assert exc_info.value.code == "TERMINAL_SOURCE_PROVENANCE_REQUIRED"


@pytest.mark.parametrize(
    "advisor_error",
    [GeminiBudgetError("advisor budget"), GeminiConfigError("advisor config")],
    ids=["budget", "config"],
)
def test_advisor_capacity_failure_skips_reviewer_and_retries_author(
    advisor_error,
):
    class FakeGemini:
        supports_intent_repair_advisor = True
        supports_intent_repair_reviewer = True
        supports_system_instruction = True
        supports_numeric_literals = True

        def __init__(self):
            self.parts: list[str] = []
            self.intent_calls = 0

        def stream_structured(self, unused_prompt, unused_schema, **kwargs):
            part = kwargs["part"]
            self.parts.append(part)
            if part == "intent":
                self.intent_calls += 1
                return _noncardinal_straight_candidate(repaired=self.intent_calls > 1)
            raise advisor_error

        def reset_lineage(self, unused_part):
            return None

    gemini = FakeGemini()
    attempts: list[dict] = []
    diagnostics: list[dict] = []
    result = pipeline._extract_intent(
        "Create two 40 mm straights separated by a 60 degree turn.",
        _settings(),
        dry_run=False,
        gemini=gemini,
        attempt_journal=attempts,
        diagnostic_journal=diagnostics,
    )

    assert result.target_behavior[-1].type == "route"
    assert gemini.parts == ["intent", "intent_repair_advisor", "intent"]
    assert attempts[0]["advisor_source"] == "deterministic_evidence_only"
    assert diagnostics[0]["advisor_attempts"][0]["status"] == "provider_failed"


def test_advisor_call_reserve_bypasses_advisors_for_author_retry():
    class FakeGemini:
        supports_intent_repair_advisor = True
        supports_intent_repair_reviewer = True
        supports_system_instruction = True
        supports_numeric_literals = True

        def __init__(self):
            self.parts: list[str] = []
            self.intent_calls = 0

        def remaining_call_budget(self):
            return 1

        def stream_structured(self, unused_prompt, unused_schema, **kwargs):
            part = kwargs["part"]
            self.parts.append(part)
            assert part == "intent"
            self.intent_calls += 1
            return _noncardinal_straight_candidate(repaired=self.intent_calls > 1)

        def reset_lineage(self, unused_part):
            return None

    gemini = FakeGemini()
    diagnostics: list[dict] = []
    result = pipeline._extract_intent(
        "Create two 40 mm straights separated by a 60 degree turn.",
        _settings(),
        dry_run=False,
        gemini=gemini,
        diagnostic_journal=diagnostics,
    )

    assert result.target_behavior[-1].type == "route"
    assert gemini.parts == ["intent", "intent"]
    assert diagnostics[0]["advisor_source"] == "deterministic_evidence_only"
    assert diagnostics[0]["advisor_attempts"][0]["status"] == (
        "skipped_author_call_reserve"
    )


def test_last_author_attempt_does_not_spend_advisor_calls():
    class FakeGemini:
        supports_intent_repair_advisor = True
        supports_intent_repair_reviewer = True
        supports_system_instruction = True
        supports_numeric_literals = True

        def __init__(self):
            self.parts: list[str] = []

        def stream_structured(self, unused_prompt, unused_schema, **kwargs):
            part = kwargs["part"]
            self.parts.append(part)
            assert part == "intent"
            return _noncardinal_straight_candidate(repaired=False)

        def reset_lineage(self, unused_part):
            return None

    gemini = FakeGemini()
    attempts: list[dict] = []
    diagnostics: list[dict] = []
    with pytest.raises(pipeline._IntentSafetyValidationError):
        pipeline._extract_intent(
            "Create two 40 mm straights separated by a 60 degree turn.",
            replace(_settings(), intent_repair_attempts=0),
            dry_run=False,
            gemini=gemini,
            attempt_journal=attempts,
            diagnostic_journal=diagnostics,
        )

    assert gemini.parts == ["intent"]
    assert attempts[0]["advisor_status"] == "skipped_no_author_repair_budget"
    assert attempts[0]["terminal_reason"] == "intent_repair_budget_exhausted"


def test_wire_host_rejection_preserves_raw_response_for_advisor_retry():
    class FakeGemini:
        supports_intent_repair_advisor = True
        supports_intent_repair_reviewer = True
        supports_system_instruction = True
        supports_numeric_literals = True

        def __init__(self):
            self.parts: list[str] = []
            self.prompts: list[str] = []
            self.intent_calls = 0
            self.advisor_calls = 0

        def stream_structured(self, prompt, unused_schema, **kwargs):
            part = kwargs["part"]
            self.parts.append(part)
            self.prompts.append(prompt)
            if part == "intent":
                self.intent_calls += 1
                return _noncardinal_straight_candidate(repaired=self.intent_calls > 1)
            assert part == "intent_repair_advisor"
            self.advisor_calls += 1
            if self.advisor_calls == 1:
                return IntentRepairAdviceWire(
                    diagnosis_class="candidate_contract_error",
                    disposition="retry_intent",
                    candidate_fixable=False,
                    summary="Relationally invalid retry response.",
                    causal_chain=["The response omitted a valid repair scope."],
                    preserve_requirements=["all source requirements"],
                    change_fields=[],
                    avoid=["repeating the candidate"],
                    intent_instruction="Retry.",
                )
            return IntentRepairAdvice(
                diagnosis_class="candidate_contract_error",
                disposition="retry_intent",
                candidate_fixable=True,
                summary="Repair the current heading contract.",
                causal_chain=["The straight direction contradicts its inlet."],
                preserve_requirements=["all source requirements"],
                change_fields=["/target_behavior/2"],
                avoid=["repeating the cardinal direction"],
                intent_instruction="Use a route that inherits the inlet heading.",
            )

        def reset_lineage(self, unused_part):
            return None

    gemini = FakeGemini()
    result = pipeline._extract_intent(
        "Create two 40 mm straights separated by a 60 degree turn.",
        _settings(),
        dry_run=False,
        gemini=gemini,
    )

    assert result.target_behavior[-1].type == "route"
    retry_context = json.loads(gemini.prompts[2])
    rejected_response = retry_context["prior_advisor_rejections"][0][
        "rejected_response"
    ]
    assert rejected_response["disposition"] == "retry_intent"
    assert rejected_response["candidate_fixable"] is False


def test_disabled_advisor_records_deterministic_evidence_fallback():
    class FakeGemini:
        supports_system_instruction = True
        supports_numeric_literals = True

        def __init__(self):
            self.parts: list[str] = []
            self.intent_calls = 0

        def stream_structured(self, unused_prompt, unused_schema, **kwargs):
            part = kwargs["part"]
            self.parts.append(part)
            assert part == "intent"
            self.intent_calls += 1
            return _noncardinal_straight_candidate(repaired=self.intent_calls > 1)

        def reset_lineage(self, unused_part):
            return None

    diagnostics: list[dict] = []
    result = pipeline._extract_intent(
        "Create two 40 mm straights separated by a 60 degree turn.",
        replace(_settings(), intent_repair_advisor_enabled=False),
        dry_run=False,
        gemini=FakeGemini(),
        diagnostic_journal=diagnostics,
    )

    assert result.target_behavior[-1].type == "route"
    assert diagnostics[0]["advisor_status"] == "disabled"
    assert diagnostics[0]["advisor_source"] == "deterministic_evidence_only"
    assert diagnostics[0]["advisor_fallback_used"] is True


def test_unsupported_advisor_client_records_deterministic_evidence_fallback():
    class FakeGemini:
        supports_system_instruction = True
        supports_numeric_literals = True

        def __init__(self):
            self.intent_calls = 0

        def stream_structured(self, unused_prompt, unused_schema, **kwargs):
            assert kwargs["part"] == "intent"
            self.intent_calls += 1
            return _noncardinal_straight_candidate(repaired=self.intent_calls > 1)

        def reset_lineage(self, unused_part):
            return None

    diagnostics: list[dict] = []
    result = pipeline._extract_intent(
        "Create two 40 mm straights separated by a 60 degree turn.",
        _settings(),
        dry_run=False,
        gemini=FakeGemini(),
        diagnostic_journal=diagnostics,
    )

    assert result.target_behavior[-1].type == "route"
    assert diagnostics[0]["advisor_status"] == "unsupported_by_client"
    assert diagnostics[0]["advisor_source"] == "deterministic_evidence_only"
    assert diagnostics[0]["advisor_fallback_used"] is True


def test_semantic_exhaustion_reports_validator_issue_instead_of_generic_error(
    monkeypatch,
    tmp_path,
):
    class FakeGemini:
        supports_system_instruction = True
        supports_numeric_literals = True

        def stream_structured(self, unused_prompt, unused_schema, **kwargs):
            assert kwargs["part"] == "intent"
            return _noncardinal_straight_candidate(repaired=False)

        def reset_lineage(self, unused_part):
            return None

    monkeypatch.setattr(pipeline, "GeminiClient", lambda unused_settings: FakeGemini())
    settings = replace(
        _settings().with_overrides(output_dir=tmp_path, skip_freecad=True),
        intent_repair_attempts=1,
        stream_thinking_summary=False,
    )

    with pytest.raises(pipeline.StaticValidationError) as exc_info:
        pipeline.run_pipeline(
            "Create two 40 mm straights separated by a 60 degree turn.",
            settings,
        )

    report = json.loads(Path(exc_info.value.artifact_path).read_text(encoding="utf-8"))
    assert report["failed_stage"] == "intent_semantic_validation"
    assert report["top_issues"] == ["FINAL_01_INTENT_SAFETY_CONTRACT"]
    assert report["intent_attempt_count"] == 2
    assert report["intent_repair_count"] == 1


def test_host_contract_exhaustion_preserves_actual_validator_issue(
    monkeypatch,
    tmp_path,
):
    class FakeGemini:
        supports_system_instruction = True
        supports_numeric_literals = True

        def stream_structured(self, unused_prompt, unused_schema, **kwargs):
            assert kwargs["part"] == "intent"
            raise HostContractValidationError(
                "intent",
                {"candidate": "provider-wire-valid"},
                ValueError("host relational contract mismatch"),
            )

        def reset_lineage(self, unused_part):
            return None

    monkeypatch.setattr(pipeline, "GeminiClient", lambda unused_settings: FakeGemini())
    settings = replace(
        _settings().with_overrides(output_dir=tmp_path, skip_freecad=True),
        intent_repair_attempts=1,
        stream_thinking_summary=False,
    )

    with pytest.raises(pipeline.StaticValidationError) as exc_info:
        pipeline.run_pipeline("Create a valid hollow pipe.", settings)

    report = json.loads(Path(exc_info.value.artifact_path).read_text(encoding="utf-8"))
    assert report["failed_stage"] == "intent_semantic_validation"
    assert report["top_issues"] == ["FINAL_01_INTENT_STRUCTURED_OR_HOST_CONTRACT"]
    critic = json.loads(
        (Path(exc_info.value.artifact_path).parent / "critic_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert critic["issues"][0]["actual"]["error_type"] == (
        "HostContractValidationError"
    )
    diagnostics = json.loads(
        (
            Path(exc_info.value.artifact_path).parent / "intent_diagnostics.json"
        ).read_text(encoding="utf-8")
    )
    assert len(diagnostics) == 2
    assert diagnostics[-1]["issue_codes"] == ["INTENT_STRUCTURED_OR_HOST_CONTRACT"]


def test_budget_after_semantic_rejection_preserves_last_validator_issue(
    monkeypatch,
    tmp_path,
):
    class FakeGemini:
        supports_system_instruction = True
        supports_numeric_literals = True

        def __init__(self):
            self.intent_calls = 0

        def stream_structured(self, unused_prompt, unused_schema, **kwargs):
            assert kwargs["part"] == "intent"
            self.intent_calls += 1
            if self.intent_calls == 1:
                return _noncardinal_straight_candidate(repaired=False)
            raise GeminiBudgetError("global call ceiling reached")

        def reset_lineage(self, unused_part):
            return None

    monkeypatch.setattr(pipeline, "GeminiClient", lambda unused_settings: FakeGemini())
    settings = replace(
        _settings().with_overrides(output_dir=tmp_path, skip_freecad=True),
        intent_repair_attempts=2,
        intent_repair_advisor_enabled=False,
        stream_thinking_summary=False,
    )

    with pytest.raises(pipeline.StaticValidationError) as exc_info:
        pipeline.run_pipeline(
            "Create two 40 mm straights separated by a 60 degree turn.",
            settings,
        )

    report = json.loads(Path(exc_info.value.artifact_path).read_text(encoding="utf-8"))
    assert report["failed_stage"] == "intent_semantic_validation"
    assert report["top_issues"] == ["FINAL_01_INTENT_SAFETY_CONTRACT"]


def test_incomplete_intent_is_discarded_and_retried_as_fresh_llm_output():
    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True
        supports_numeric_schema_modes = True

        def __init__(self):
            self.calls = []
            self.reset_calls = []

        def stream_structured(
            self,
            prompt,
            schema,
            *,
            part,
            thinking_level,
            numeric_schema_mode,
        ):
            del schema
            assert numeric_schema_mode == "plain"
            self.calls.append((prompt, part, thinking_level))
            if len(self.calls) == 1:
                raise StructuredOutputIncompleteError(
                    "intent",
                    status="incomplete",
                    output_limit=16384,
                    output_tokens=350,
                    thought_tokens=16000,
                )
            return _production_intent()

        def reset_lineage(self, part):
            self.reset_calls.append(part)

    gemini = FakeGemini()
    result = pipeline._extract_intent(
        PROMPT,
        _settings(),
        dry_run=False,
        gemini=gemini,
    )

    assert result.target_behavior[0].length == 80.0
    assert len(gemini.calls) == 2
    assert [call[2] for call in gemini.calls] == ["low", "low"]
    assert gemini.reset_calls == ["intent"]
    retry_prompt = gemini.calls[1][0]
    assert PROMPT in retry_prompt
    assert "The prior attempt was discarded" in retry_prompt
    assert "Generate a new complete intent JSON object" in retry_prompt
    assert "Repair the previous intent JSON" not in retry_prompt
    assert "outer_diameter" in retry_prompt
    assert "20.0" in retry_prompt
    assert "thought_tokens=16000" in retry_prompt


def test_spline_semantic_repair_escalates_reasoning_with_plain_schema_sticky(
    tmp_path,
):
    def candidate(points):
        return IntentResult(
            global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
            target_behavior=[
                Goal(
                    goal_id="freeform",
                    type="route",
                    path_kind="spline",
                    required_waypoints=points,
                    waypoint_frame="relative_to_target",
                    waypoint_scale_policy="fixed",
                )
            ],
            expected_open_ports=1,
            expected_open_ports_source="derived",
        )

    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True
        supports_numeric_schema_modes = True
        supports_system_instruction = True

        def __init__(self):
            self.calls = []
            self.reset_calls = []

        def stream_structured(self, prompt, schema, **kwargs):
            del schema
            self.calls.append((prompt, kwargs))
            if len(self.calls) == 1:
                return candidate([(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)])
            return candidate([(30.0, 0.0, 0.0), (60.0, 30.0, 0.0)])

        def reset_lineage(self, part):
            self.reset_calls.append(part)

    gemini = FakeGemini()
    journal = []
    journal_path = tmp_path / "intent_attempts.json"
    result = pipeline._extract_intent(
        "Create a qualitative spatial spline.",
        _settings(),
        dry_run=False,
        gemini=gemini,
        attempt_journal=journal,
        attempt_journal_path=journal_path,
    )

    assert result.target_behavior[0].required_waypoints[-1] == (60.0, 30.0, 0.0)
    assert len(gemini.calls) == 2
    assert gemini.calls[0][1]["thinking_level"] == "low"
    assert gemini.calls[0][1]["numeric_schema_mode"] == "plain"
    assert "numeric_literals" not in gemini.calls[0][1]
    assert gemini.calls[1][1]["thinking_level"] == "medium"
    assert gemini.calls[1][1]["numeric_schema_mode"] == "plain"
    assert "numeric_literals" not in gemini.calls[1][1]
    assert "bounded decimal-object representation" not in gemini.calls[1][0]
    assert "Validation diagnostic history" in gemini.calls[1][0]
    assert gemini.reset_calls == []
    persisted = json.loads(journal_path.read_text(encoding="utf-8"))
    assert [item["status"] for item in persisted] == ["rejected", "accepted"]
    assert persisted[0]["phase"] == "semantic_validation"
    assert persisted[0]["parsed_intent"] is True
    assert "minimum curvature radius" in persisted[0]["diagnostic"]
    assert persisted[0]["candidate_digest"]
    assert persisted[1]["candidate_digest"]


def test_persistent_incomplete_intent_exhausts_llm_attempts_without_fallback(
    monkeypatch,
):
    settings = replace(_settings(), intent_repair_attempts=2)

    def forbidden_fallback(*args, **kwargs):
        del args, kwargs
        raise AssertionError("production must not call deterministic intent fallback")

    monkeypatch.setattr(pipeline, "infer_intent", forbidden_fallback)

    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True
        supports_numeric_schema_modes = True

        def __init__(self):
            self.calls = []
            self.reset_calls = []

        def stream_structured(
            self,
            prompt,
            schema,
            *,
            part,
            thinking_level,
            numeric_schema_mode,
        ):
            del schema
            assert numeric_schema_mode == "plain"
            self.calls.append((prompt, part, thinking_level))
            raise StructuredOutputIncompleteError(
                "intent",
                status="incomplete",
                output_limit=16384,
                output_tokens=200,
                thought_tokens=16100,
            )

        def reset_lineage(self, part):
            self.reset_calls.append(part)

    gemini = FakeGemini()
    with pytest.raises(StructuredOutputIncompleteError):
        pipeline._extract_intent(
            PROMPT,
            settings,
            dry_run=False,
            gemini=gemini,
        )

    assert len(gemini.calls) == 3
    assert gemini.reset_calls == ["intent", "intent"]
    assert all(call[2] == "low" for call in gemini.calls)
    assert all(PROMPT in call[0] for call in gemini.calls)
    assert all("DO_NOT_ECHO" not in call[0] for call in gemini.calls)


def test_wire_success_domain_failure_uses_semantic_repair_not_protocol_retry():
    def wire(*, inverted_box: bool) -> LLMProductionIntent:
        return LLMProductionIntent.model_validate(
            {
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
                    }
                ],
                "expected_open_ports": 1,
                "expected_open_ports_source": "derived",
                "required_components": [],
                "hard_constraints": [],
                "geometric_constraints": [
                    {
                        "constraint_id": "box",
                        "type": "bounding_box",
                        "minimum": (
                            [10.0, 10.0, 10.0]
                            if inverted_box
                            else [-10.0, -10.0, -10.0]
                        ),
                        "maximum": [0.0, 0.0, 0.0]
                        if inverted_box
                        else [100.0, 10.0, 10.0],
                    }
                ],
                "design_notes": [],
            }
        )

    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True

        def __init__(self):
            self.calls = []

        def stream_structured(self, prompt, unused_schema, **kwargs):
            self.calls.append((prompt, kwargs))
            return wire(inverted_box=len(self.calls) == 1)

        def reset_lineage(self, unused_part):
            raise AssertionError("first semantic correction should keep lineage")

    journal = []
    result = pipeline._extract_intent(
        "Create an 80 mm straight pipe inside the authored bounds.",
        _settings(),
        dry_run=False,
        gemini=FakeGemini(),
        attempt_journal=journal,
    )

    assert result.target_behavior[0].length == 80.0
    assert [item["phase"] for item in journal] == [
        "semantic_validation",
        "semantic_validation",
    ]
    assert journal[0]["provider_response_received"] is True
    assert journal[0]["parsed_intent"] is False
    assert journal[0]["consumes_semantic_budget"] is True
    assert journal[0]["structured_retry_attempt"] == 0
    assert "bounding_box" in journal[0]["diagnostic"]


def test_intent_plain_numeric_schema_does_not_depend_on_enum_capacity(
    monkeypatch,
):
    monkeypatch.setattr(
        pipeline,
        "_intent_numeric_literals",
        lambda prompt, settings: (_ for _ in ()).throw(
            AssertionError("plain intent schema must not build a numeric enum")
        ),
    )

    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True
        supports_numeric_schema_modes = True

        def __init__(self):
            self.calls = []

        def stream_structured(self, prompt, schema, **kwargs):
            del schema
            self.calls.append((prompt, kwargs))
            return _production_intent()

    gemini = FakeGemini()
    result = pipeline._extract_intent(
        PROMPT,
        _settings(),
        dry_run=False,
        gemini=gemini,
    )

    assert result.target_behavior[0].length == 80.0
    assert len(gemini.calls) == 1
    assert "numeric_literals" not in gemini.calls[0][1]
    assert gemini.calls[0][1]["numeric_schema_mode"] == "plain"
    assert "bounded decimal-object representation" not in gemini.calls[0][0]


def test_intent_invalid_plain_request_falls_back_to_host_validated_envelope():
    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True
        supports_numeric_schema_modes = True

        def __init__(self):
            self.calls = []
            self.reset_calls = []

        def stream_structured(self, prompt, schema, **kwargs):
            self.calls.append((prompt, schema, kwargs))
            if len(self.calls) == 1:
                raise GeminiInvalidRequestError(
                    "400 invalid_argument: response schema could not compile",
                    provider_code="invalid_argument",
                )
            return _llm_intent_envelope()

        def reset_lineage(self, part):
            self.reset_calls.append(part)

    gemini = FakeGemini()
    result = pipeline._extract_intent(
        PROMPT,
        _settings(),
        dry_run=False,
        gemini=gemini,
    )

    assert result.target_behavior[0].length == 80.0
    assert len(gemini.calls) == 2
    assert gemini.calls[0][1] is LLMProductionIntent
    assert gemini.calls[1][1] is LLMIntentJSONEnvelope
    assert gemini.calls[0][2]["numeric_schema_mode"] == "plain"
    assert gemini.calls[1][2]["numeric_schema_mode"] == "plain"
    assert "numeric_literals" not in gemini.calls[0][2]
    assert "numeric_literals" not in gemini.calls[1][2]
    assert gemini.reset_calls == ["intent"]
    assert "Set intent_json to a JSON-encoded string" in gemini.calls[1][0]


def test_intent_schema_fallback_does_not_consume_zero_semantic_repair_budget():
    """스키마 컴파일 거절은 LLM 의미 오류가 아니므로 교정 횟수를 쓰지 않는다."""

    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True
        supports_numeric_schema_modes = True

        def __init__(self):
            self.calls = []
            self.reset_calls = []

        def stream_structured(self, prompt, schema, **kwargs):
            self.calls.append((prompt, schema, kwargs))
            if len(self.calls) == 1:
                raise GeminiInvalidRequestError(
                    "400 invalid_argument: response schema could not compile",
                    provider_code="invalid_argument",
                )
            return _llm_intent_envelope()

        def reset_lineage(self, part):
            self.reset_calls.append(part)

    gemini = FakeGemini()
    result = pipeline._extract_intent(
        PROMPT,
        replace(_settings(), intent_repair_attempts=0),
        dry_run=False,
        gemini=gemini,
    )

    assert result.target_behavior[0].length == 80.0
    assert len(gemini.calls) == 2
    assert gemini.reset_calls == ["intent"]
    assert gemini.calls[0][2]["numeric_schema_mode"] == "plain"
    assert gemini.calls[1][2]["numeric_schema_mode"] == "plain"


def test_schema_negotiation_does_not_consume_malformed_output_retry_budget():
    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_literals = True
        supports_numeric_schema_modes = True

        def __init__(self):
            self.calls = []
            self.reset_calls = []

        def stream_structured(self, prompt, schema, **kwargs):
            self.calls.append((prompt, schema, kwargs))
            if len(self.calls) == 1:
                raise GeminiInvalidRequestError(
                    "400 invalid_argument: response schema could not compile",
                    provider_code="invalid_argument",
                )
            if len(self.calls) == 2:
                return LLMIntentJSONEnvelope(intent_json='{"global_spec":')
            return _llm_intent_envelope()

        def reset_lineage(self, part):
            self.reset_calls.append(part)

    gemini = FakeGemini()
    result = pipeline._extract_intent(
        PROMPT,
        _settings(),
        dry_run=False,
        gemini=gemini,
    )

    assert result.target_behavior[0].length == 80.0
    assert [call[1] for call in gemini.calls] == [
        LLMProductionIntent,
        LLMIntentJSONEnvelope,
        LLMIntentJSONEnvelope,
    ]
    assert all(call[2]["numeric_schema_mode"] == "plain" for call in gemini.calls)
    assert all("numeric_literals" not in call[2] for call in gemini.calls)
    assert "The prior attempt was discarded" in gemini.calls[2][0]
    assert gemini.reset_calls == ["intent", "intent"]


def test_all_intent_provider_schema_profiles_rejected_are_fully_journaled():
    class FakeGemini:
        supports_interaction_controls = True
        supports_numeric_schema_modes = True

        def __init__(self):
            self.calls = []
            self.reset_calls = []

        def stream_structured(self, prompt, schema, **kwargs):
            self.calls.append((prompt, schema, kwargs))
            raise GeminiInvalidRequestError(
                "400 invalid_argument: response schema could not compile",
                provider_code="invalid_argument",
            )

        def reset_lineage(self, part):
            self.reset_calls.append(part)

    journal = []
    gemini = FakeGemini()
    with pytest.raises(GeminiInvalidRequestError):
        pipeline._extract_intent(
            PROMPT,
            replace(_settings(), intent_repair_attempts=0),
            dry_run=False,
            gemini=gemini,
            attempt_journal=journal,
        )

    assert [call[1] for call in gemini.calls] == [
        LLMProductionIntent,
        LLMIntentJSONEnvelope,
    ]
    assert [item["status"] for item in journal] == [
        "schema_retry",
        "schema_rejected",
    ]
    assert [item["schema_retry_attempt"] for item in journal] == [1, 2]
    assert all(item["consumes_semantic_budget"] is False for item in journal)


def test_intent_repair_diagnostic_does_not_echo_malformed_raw_json():
    error = pipeline.StructuredOutputError(
        "intent",
        '{"wall_thickness_out": 1.500000000000000000000000',
        ValueError("truncated"),
    )

    diagnostic = pipeline._intent_repair_diagnostic(error)

    assert "wall_thickness_out" not in diagnostic
    assert "1.500000" not in diagnostic
    assert diagnostic == "structured output was invalid: ValueError"


def test_incomplete_intent_diagnostic_is_metadata_only():
    error = StructuredOutputIncompleteError(
        "intent",
        status="incomplete",
        output_limit=16384,
        output_tokens=200,
        thought_tokens=16100,
    )

    diagnostic = pipeline._intent_repair_diagnostic(error)

    assert error.raw_text == ""
    assert "raw=" not in str(error)
    assert "status=incomplete" in diagnostic
    assert "output_limit=16384" in diagnostic
    assert "partial" not in diagnostic


def test_intent_prompt_routes_non_geometric_appearance_to_design_notes():
    prompt = intent_prompt(
        "Create a hollow manifold with a brushed matte metal finish."
    )
    system = intent_system_instruction()

    assert "material, surface finish, color, rendering, camera, and" in system
    assert "soft visual preferences in design_notes" in system
    assert "brushed metal or matte metal" in system
    assert "design must be rejected when" in system
    assert "Never let a visual preference replace or" in system
    assert "Exact helices/threads, non-circular ducts" in system
    assert "`unsupported:<verbatim>` in hard_constraints" in system
    assert "exactly one binary Y junction" in system
    assert "expected_open_ports=N-1" in system
    assert "Canonical four-end example" in system
    assert "include_primary_outlet=true" in system
    assert "Never emit upper-left again" in system
    assert "User request:" in prompt
    assert "Authoring rules:" not in prompt


def test_intent_safety_rejects_root_branch_that_recreates_start_arm():
    damaged = ProductionIntent.model_validate(
        {
            "global_spec": {
                "outer_diameter": 20.0,
                "wall_thickness": 2.0,
                "is_hollow": True,
                "units": "mm",
            },
            "start_position": [0.0, 0.0, 0.0],
            "start_axis": [1.0, 0.0, -1.0],
            "target_behavior": [
                {
                    "goal_id": "bad_root_y",
                    "depends_on_goal_ids": [],
                    "allow_parallel": False,
                    "type": "branch",
                    "required_outlet_vectors": [
                        [-1.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0],
                    ],
                    "include_primary_outlet": False,
                    "junction_style": "smooth_hub",
                }
            ],
            "expected_open_ports": 2,
            "expected_open_ports_source": "explicit",
            "required_components": [],
            "hard_constraints": [],
            "geometric_constraints": [],
            "design_notes": [],
        }
    ).to_intent_result()

    with pytest.raises(ValueError, match="recreates the anchored START arm"):
        pipeline._validate_intent_safety(
            "four-port manifold with an upper-left START arm",
            damaged,
            _settings(),
        )


def test_intent_safety_requires_run_before_first_y_for_four_named_remote_ends():
    damaged = ProductionIntent.model_validate(
        {
            "global_spec": {
                "outer_diameter": 20.0,
                "wall_thickness": 2.0,
                "is_hollow": True,
                "units": "mm",
            },
            "start_position": [0.0, 0.0, 0.0],
            "start_axis": [1.0, 0.0, -1.0],
            "target_behavior": [
                {
                    "goal_id": "left_y",
                    "depends_on_goal_ids": [],
                    "allow_parallel": False,
                    "type": "branch",
                    "required_outlet_vectors": [[-1.0, 0.0, -1.0]],
                    "include_primary_outlet": True,
                    "junction_style": "smooth_hub",
                },
                {
                    "goal_id": "central_run",
                    "depends_on_goal_ids": ["left_y"],
                    "allow_parallel": False,
                    "type": "route",
                    "length": 40.0,
                },
                {
                    "goal_id": "right_y",
                    "depends_on_goal_ids": ["central_run"],
                    "allow_parallel": False,
                    "type": "branch",
                    "required_outlet_vectors": [
                        [1.0, 0.0, 1.0],
                        [1.0, 0.0, -1.0],
                    ],
                    "include_primary_outlet": False,
                    "junction_style": "smooth_hub",
                },
            ],
            "expected_open_ports": 3,
            "expected_open_ports_source": "explicit",
            "required_components": [],
            "hard_constraints": [],
            "geometric_constraints": [],
            "design_notes": [],
        }
    ).to_intent_result()

    prompt = (
        "Create four open ends at upper-left, lower-left, upper-right, and lower-right."
    )
    with pytest.raises(ValueError, match="must start with a positive-length route"):
        pipeline._validate_intent_safety(prompt, damaged, _settings())


def test_explicit_mm_value_preserved_in_fail_closed_text_contract_is_not_lost():
    intent = _production_intent(
        hard_constraints=["unsupported:minimum 7 mm clearance"],
    ).to_intent_result()

    pipeline._validate_intent_safety(
        PROMPT + " 최소 7 mm clearance도 요구한다.",
        intent,
        _settings(),
    )


def test_explicit_mm_parser_handles_attached_korean_grouping_and_unicode_minus():
    assert pipeline._explicit_mm_values(
        "외경20mm 두께1.5mm로 1,000 mm −40㎜ 60 밀리미터로 90도 DN20"
    ) == [20.0, 1.5, 1000.0, -40.0, 60.0]


def test_explicit_mm_parser_preserves_both_shared_unit_range_endpoints():
    assert pipeline._explicit_mm_values(
        "Each branch should be approximately 85–100 mm long."
    ) == [85.0, 100.0]
    assert pipeline._explicit_mm_values("Keep it between 85 mm to 100 mm.") == [
        85.0,
        100.0,
    ]


def test_intent_safety_accepts_derived_inner_diameter_and_repeated_wall_reference():
    prompt = (
        "Use a uniform outer diameter of 24 mm, inner diameter of 17 mm, and "
        "wall thickness of 3.5 mm throughout. All branch ends must show "
        "the 3.5 mm wall thickness."
    )
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=24.0, wall_thickness=3.5),
        target_behavior=[Goal(type="move", direction="+X", length=40.0)],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )

    assert pipeline._anchored_mm_values(prompt) == {
        "global_outer_diameter": [24.0],
        "global_inner_diameter": [17.0],
        "global_wall_thickness": [3.5],
    }
    assert pipeline._exact_mm_contract_values(prompt) == [24.0, 17.0, 3.5]
    pipeline._validate_intent_safety(prompt, intent, _settings())


def test_intent_safety_rejects_inner_diameter_inconsistent_with_od_and_wall():
    prompt = (
        "Use outer diameter 24 mm, inner diameter 16 mm, and wall thickness "
        "3.5 mm throughout."
    )
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=24.0, wall_thickness=3.5),
        target_behavior=[Goal(type="move", direction="+X", length=40.0)],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )

    with pytest.raises(ValueError, match="global_inner_diameter"):
        pipeline._validate_intent_safety(prompt, intent, _settings())


def test_intent_safety_treats_explicit_mm_range_as_inclusive_choice():
    prompt = "Make the branch approximately 85–100 mm long."
    inside = IntentResult(
        global_spec=GlobalSpec(outer_diameter=24.0, wall_thickness=3.5),
        target_behavior=[Goal(type="move", direction="+X", length=92.0)],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )
    outside = inside.model_copy(
        update={
            "target_behavior": [
                inside.target_behavior[0].model_copy(update={"length": 110.0})
            ]
        }
    )

    pipeline._validate_intent_safety(prompt, inside, _settings())
    with pytest.raises(ValueError, match="inside explicit millimeter range"):
        pipeline._validate_intent_safety(prompt, outside, _settings())


def test_waypoint_route_cannot_hide_spline_from_intent_curvature_preflight():
    """원본 manifold 회귀: null path_kind가 불가능한 좌표를 통과시키면 안 된다."""

    prompt = (
        "Create a smooth branch approximately 90 mm long with a minimum "
        "centerline bend radius of 35 mm."
    )
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=24.0, wall_thickness=3.5),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[
            Goal(
                goal_id="arm_1_route",
                type="route",
                path_kind=None,
                length=90.0,
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
        expected_open_ports_source="derived",
    )

    with pytest.raises(ValueError) as captured:
        pipeline._validate_intent_safety(prompt, intent, _settings())

    diagnostic = str(captured.value)
    assert "required_waypoints require path_kind=spline" in diagnostic
    assert "minimum curvature radius of 23.571" in diagnostic
    assert "no terminal_axis contract" in diagnostic
    assert "42.9073 mm" in diagnostic


def test_original_manifold_semantics_reject_missing_arm_lengths_and_conflicts():
    prompt = (
        "Create a four-port manifold. Each branch should be approximately "
        "85–100 mm long, with slightly different branch lengths. Use smooth "
        "Y-junctions and no sharp Boolean intersections."
    )
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=24.0, wall_thickness=3.5),
        target_behavior=[
            Goal(
                goal_id="start_arm",
                type="route",
                path_kind="line",
                direction="+X",
                length=90.0,
            ),
            Goal(
                goal_id="left_y",
                depends_on_goal_ids=["start_arm"],
                type="branch",
                branch_angles=[40.0],
                required_outlet_vectors=[(-1.0, 0.0, -1.0)],
                include_primary_outlet=True,
            ),
            Goal(
                goal_id="center",
                depends_on_goal_ids=["left_y"],
                type="move",
                direction="+X",
                length=68.0,
            ),
            Goal(
                goal_id="right_y",
                depends_on_goal_ids=["center"],
                type="branch",
                branch_angles=[40.0, 40.0],
                required_outlet_vectors=[
                    (1.0, 0.0, 1.0),
                    (1.0, 0.0, -1.0),
                ],
                include_primary_outlet=False,
            ),
        ],
        expected_open_ports=3,
        expected_open_ports_source="derived",
    )

    with pytest.raises(ValueError) as captured:
        pipeline._validate_intent_safety(prompt, intent, _settings())

    diagnostic = str(captured.value)
    assert "actual=[90.0, None, None, None]" in diagnostic
    assert diagnostic.count("branch_angles conflict with its outlet axes") == 2
    assert "expected_open_ports_source must be 'explicit'" in diagnostic
    assert "junction_style='smooth_hub'" in diagnostic


def test_original_manifold_semantics_accept_complete_terminal_arm_contracts():
    prompt = (
        "Create a four-port manifold. Each branch should be approximately "
        "85–100 mm long, with slightly different branch lengths. Use smooth "
        "Y-junctions and no sharp Boolean intersections."
    )
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=24.0, wall_thickness=3.5),
        target_behavior=[
            Goal(
                goal_id="start_arm",
                type="route",
                path_kind="line",
                direction="+X",
                length=85.0,
            ),
            Goal(
                goal_id="left_y",
                depends_on_goal_ids=["start_arm"],
                type="branch",
                branch_angles=[45.0],
                required_outlets=[
                    BranchGoalOutletSpec(axis=(-1.0, 0.0, -1.0), length=90.0)
                ],
                include_primary_outlet=True,
                junction_style="smooth_hub",
            ),
            Goal(
                goal_id="center",
                depends_on_goal_ids=["left_y"],
                type="move",
                direction="+X",
                length=68.0,
            ),
            Goal(
                goal_id="right_y",
                depends_on_goal_ids=["center"],
                type="branch",
                branch_angles=[45.0, 45.0],
                required_outlets=[
                    BranchGoalOutletSpec(axis=(1.0, 0.0, 1.0), length=95.0),
                    BranchGoalOutletSpec(axis=(1.0, 0.0, -1.0), length=100.0),
                ],
                include_primary_outlet=False,
                junction_style="smooth_hub",
            ),
        ],
        expected_open_ports=3,
        expected_open_ports_source="explicit",
    )

    pipeline._validate_intent_safety(prompt, intent, _settings())


def test_main_axis_branch_angle_range_checks_every_physical_terminal_arm():
    prompt = (
        "Create a four-port manifold with each branch 85–100 mm long and branch "
        "angles around 35–45 degrees from the main axis."
    )
    base = IntentResult(
        global_spec=GlobalSpec(outer_diameter=24.0, wall_thickness=3.5),
        start_axis=(1.0, 0.0, 1.0),
        target_behavior=[
            Goal(
                goal_id="start_arm",
                type="route",
                path_kind="line",
                length=85.0,
                terminal_axis=(1.0, 0.0, 1.0),
            ),
            Goal(
                goal_id="left_y",
                depends_on_goal_ids=["start_arm"],
                type="branch",
                required_outlets=[
                    BranchGoalOutletSpec(axis=(-1.0, 0.0, 1.0), length=90.0)
                ],
                include_primary_outlet=True,
            ),
            Goal(
                goal_id="center",
                depends_on_goal_ids=["left_y"],
                type="move",
                direction="+X",
                length=68.0,
            ),
            Goal(
                goal_id="right_y",
                depends_on_goal_ids=["center"],
                type="branch",
                required_outlets=[
                    BranchGoalOutletSpec(axis=(1.0, 0.0, 1.0), length=95.0),
                    BranchGoalOutletSpec(axis=(1.0, 0.0, -1.0), length=100.0),
                ],
                include_primary_outlet=False,
            ),
        ],
        expected_open_ports=3,
        expected_open_ports_source="explicit",
    )

    pipeline._validate_intent_safety(prompt, base, _settings())

    straight_start = base.model_copy(
        update={
            "start_axis": (1.0, 0.0, 0.0),
            "target_behavior": [
                base.target_behavior[0].model_copy(
                    update={"terminal_axis": (1.0, 0.0, 0.0)}
                ),
                *base.target_behavior[1:],
            ],
        }
    )
    with pytest.raises(ValueError, match="terminal-arm axes violate") as captured:
        pipeline._validate_intent_safety(prompt, straight_start, _settings())
    assert "'arm_index': 0" in str(captured.value)


def test_junction_width_reference_is_not_reused_as_hub_radius():
    prompt = (
        "Use outer diameter 24 mm. Keep the junction width close to the "
        "24 mm pipe diameter."
    )
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=24.0, wall_thickness=3.5),
        target_behavior=[
            Goal(goal_id="lead", type="move", direction="+X", length=40.0),
            Goal(
                goal_id="split",
                depends_on_goal_ids=["lead"],
                type="branch",
                required_outlet_vectors=[(1.0, 1.0, 0.0)],
                include_primary_outlet=True,
                max_hub_radius=24.0,
            ),
        ],
        expected_open_ports=2,
    )

    assert pipeline._exact_mm_contract_values(prompt) == [24.0]
    with pytest.raises(ValueError, match="not a hub radius"):
        pipeline._validate_intent_safety(prompt, intent, _settings())


def test_explicit_xyz_vectors_feed_numeric_grammar_and_contract_audit():
    prompt = "Route through (30, 0, 0) and then (110, -80, 55)."

    assert pipeline._explicit_vector3_values(prompt) == [
        (30.0, 0.0, 0.0),
        (110.0, -80.0, 55.0),
    ]
    literals = pipeline._intent_numeric_literals(prompt, _settings())
    assert {"110", "-110", "80", "-80", "55", "-55"} <= set(literals)


def test_explicit_waypoint_vector_cannot_be_silently_rewritten():
    prompt = "Use waypoint offsets (30, 0, 0), then (60, 30, 0)."
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                goal_id="route",
                type="route",
                path_kind="spline",
                required_waypoints=[(30.0, 0.0, 0.0), (60.0, 30.0, 0.0)],
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
            )
        ],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )

    pipeline._validate_intent_safety(prompt, intent, _settings())

    changed = intent.target_behavior[0].model_copy(
        update={"required_waypoints": [(30.0, 0.0, 0.0), (60.0, 25.0, 0.0)]}
    )
    with pytest.raises(ValueError, match="explicit XYZ vectors"):
        pipeline._validate_intent_safety(
            prompt,
            intent.model_copy(update={"target_behavior": [changed]}),
            _settings(),
        )


def test_proportional_heading_vector_accepts_positive_parallel_route_chord():
    prompt = "The spline outlet heading is proportional to (+2, -1, +1)."
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=2.0, wall_thickness=0.2),
        target_behavior=[
            Goal(
                goal_id="route",
                type="route",
                path_kind="spline",
                required_waypoints=[(20.0, 0.0, 0.0), (40.0, -10.0, 10.0)],
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
            )
        ],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )

    pipeline._validate_intent_safety(prompt, intent, _settings())

    wrong_chord = intent.target_behavior[0].model_copy(
        update={"required_waypoints": [(20.0, 0.0, 0.0), (40.0, 10.0, 10.0)]}
    )
    with pytest.raises(ValueError, match="proportional directions"):
        pipeline._validate_intent_safety(
            prompt,
            intent.model_copy(update={"target_behavior": [wrong_chord]}),
            _settings(),
        )


def test_proportional_outlet_heading_cannot_borrow_interior_spline_chord():
    prompt = "The spline outlet heading is proportional to (0, 1, 0)."
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=2.0, wall_thickness=0.2),
        target_behavior=[
            Goal(
                goal_id="route",
                type="route",
                path_kind="spline",
                required_waypoints=[
                    (20.0, 0.0, 0.0),
                    (20.0, 20.0, 0.0),
                    (40.0, 20.0, 0.0),
                ],
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
            )
        ],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )

    with pytest.raises(ValueError, match="proportional directions"):
        pipeline._validate_intent_safety(prompt, intent, _settings())


def test_proportional_outlet_heading_cannot_borrow_turn_plane_normal():
    prompt = "The bend outlet heading is proportional to (0, 1, 0)."
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[
            Goal(
                goal_id="bend",
                type="turn",
                direction="-Z",
                plane_normal=(0.0, 1.0, 0.0),
                angle=90.0,
                bend_radius=30.0,
            )
        ],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )

    with pytest.raises(ValueError, match="proportional directions"):
        pipeline._validate_intent_safety(prompt, intent, _settings())


def test_proportional_branch_axis_cannot_borrow_start_axis_for_count_only_branch():
    prompt = "The final terminal branch axis is proportional to (1, 0, 0)."
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[
            Goal(
                goal_id="junction",
                type="branch",
                branch_count=2,
                include_primary_outlet=False,
            )
        ],
        expected_open_ports=2,
        expected_open_ports_source="derived",
    )

    with pytest.raises(ValueError, match="proportional directions"):
        pipeline._validate_intent_safety(prompt, intent, _settings())


def test_proportional_heading_accepts_cardinal_turn_direction():
    prompt = "The bend outlet heading is proportional to (0, 1, 0)."
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[
            Goal(
                goal_id="bend",
                type="turn",
                direction="+Y",
                angle=90.0,
                bend_radius=30.0,
            )
        ],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )

    pipeline._validate_intent_safety(prompt, intent, _settings())


def test_proportional_branch_axes_accept_cardinal_outlet_directions():
    prompt = (
        "The first branch arm axis is proportional to (0, 1, 0), and the "
        "second branch arm axis is proportional to (0, -1, 0)."
    )
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                goal_id="junction",
                type="branch",
                required_outlet_directions=["+Y", "-Y"],
                include_primary_outlet=False,
            )
        ],
        expected_open_ports=2,
        expected_open_ports_source="derived",
    )

    pipeline._validate_intent_safety(prompt, intent, _settings())


def test_one_terminal_heading_representation_cannot_satisfy_two_source_roles():
    prompt = (
        "One outlet heading is proportional to (0, 1, 0). "
        "A second outlet heading is proportional to (0, 1, 0)."
    )
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=2.0, wall_thickness=0.2),
        target_behavior=[
            Goal(
                goal_id="route",
                type="route",
                path_kind="spline",
                required_waypoints=[(20.0, 0.0, 0.0), (20.0, 20.0, 0.0)],
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
                terminal_axis=(0.0, 1.0, 0.0),
            )
        ],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )

    with pytest.raises(ValueError, match="proportional directions"):
        pipeline._validate_intent_safety(prompt, intent, _settings())


def test_exact_waypoint_vector_cannot_borrow_matching_derived_chord():
    prompt = "Use the exact relative waypoint offset (40, -20, 20)."
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=2.0, wall_thickness=0.2),
        target_behavior=[
            Goal(
                goal_id="route",
                type="route",
                path_kind="spline",
                required_waypoints=[(100.0, 0.0, 0.0), (140.0, -20.0, 20.0)],
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
            )
        ],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )

    with pytest.raises(ValueError, match="explicit XYZ vectors"):
        pipeline._validate_intent_safety(prompt, intent, _settings())


def test_explicit_vector_contract_classifies_only_proportional_language_as_direction():
    assert pipeline._explicit_vector3_contracts(
        "Visit (40, 0, 0), then point proportional to (+2, -1, +1); "
        "follow an axis proportional to the vector (3, 2, 1); "
        "축 (1, 1, 0)에 비례한다."
    ) == [
        ((40.0, 0.0, 0.0), False),
        ((2.0, -1.0, 1.0), True),
        ((3.0, 2.0, 1.0), True),
        ((1.0, 1.0, 0.0), True),
    ]


def test_modeling_tolerance_boundary_is_fail_closed():
    settings = _settings()
    at_tolerance = _production_intent(
        wall_thickness_out=settings.modeling_tolerance,
    ).to_intent_result()
    above_tolerance = _production_intent(
        wall_thickness_out=math.nextafter(settings.modeling_tolerance, math.inf),
    ).to_intent_result()

    with pytest.raises(ValueError, match="modeling_tolerance"):
        pipeline._validate_intent_safety("pipe", at_tolerance, settings)
    pipeline._validate_intent_safety("pipe", above_tolerance, settings)


def test_goal_notes_cannot_hide_a_lost_typed_dimension():
    damaged = _production_intent(transition_length=None).to_intent_result()
    goals = list(damaged.target_behavior)
    goals[2] = goals[2].model_copy(update={"notes": "transition is 40 mm"})
    damaged = damaged.model_copy(update={"target_behavior": goals})

    with pytest.raises(ValueError, match="40.0"):
        pipeline._validate_intent_safety(PROMPT, damaged, _settings())


def test_intent_safety_rejects_dimensions_swapped_between_semantic_fields():
    damaged = _production_intent().to_intent_result()
    goals = list(damaged.target_behavior)
    goals[0] = goals[0].model_copy(update={"length": 20.0})
    damaged = damaged.model_copy(
        update={
            "global_spec": damaged.global_spec.model_copy(
                update={"outer_diameter": 80.0}
            ),
            "target_behavior": goals,
        }
    )

    with pytest.raises(ValueError) as exc_info:
        pipeline._validate_intent_safety(PROMPT, damaged, _settings())

    message = str(exc_info.value)
    assert "global_outer_diameter" in message
    assert "straight_length" in message


def test_intent_safety_rejects_connector_and_transition_length_swap():
    damaged = _production_intent().to_intent_result()
    goals = list(damaged.target_behavior)
    goals[1] = goals[1].model_copy(update={"length": 40.0})
    goals[2] = goals[2].model_copy(update={"transition_length": 30.0})
    damaged = damaged.model_copy(update={"target_behavior": goals})

    with pytest.raises(ValueError) as exc_info:
        pipeline._validate_intent_safety(PROMPT, damaged, _settings())

    message = str(exc_info.value)
    assert "connector_length" in message
    assert "transition_length" in message


def test_intent_safety_preserves_repeated_source_measurement_multiplicity():
    intent = _production_intent().to_intent_result()

    with pytest.raises(ValueError, match="lost or altered"):
        pipeline._validate_intent_safety(
            PROMPT + " 추가로 분류되지 않은 기준 20 mm를 요구한다.",
            intent,
            _settings(),
        )


def test_intent_safety_does_not_misclassify_korean_initial_section_wording():
    prompt = "외경을 20 mm로 하고 두께를 2 mm로 한 파이프를 +X로 80 mm 직진시킨다."
    intent = _production_intent().to_intent_result()

    anchors = pipeline._anchored_mm_values(prompt)

    assert "diameter_out" not in anchors
    assert "wall_thickness_out" not in anchors
    pipeline._validate_intent_safety(prompt, intent, _settings())


def test_english_from_to_change_anchors_the_destination_not_source_value():
    anchors = pipeline._anchored_mm_values(
        "reduce outer diameter from 20 mm to 12 mm and "
        "change wall thickness from 2 mm to 1.5 mm"
    )

    assert anchors["diameter_in_reference"] == [20.0]
    assert anchors["wall_thickness_in_reference"] == [2.0]
    assert anchors["diameter_out"] == [12.0]
    assert anchors["wall_thickness_out"] == [1.5]


def test_junction_blend_measurements_bind_to_typed_branch_fields_in_order():
    prompt = (
        "First use outer blend radius 5 mm, inner-bore blend radius 3 mm, and "
        "maximum hub radius 25 mm. Later use outer blend radius 4 mm, "
        "inner-bore blend radius 2.5 mm, and maximum hub radius 20 mm."
    )

    assert pipeline._anchored_mm_values(prompt) == {
        "junction_blend_radius": [5.0, 4.0],
        "junction_inner_blend_radius": [3.0, 2.5],
        "junction_max_hub_radius": [25.0, 20.0],
    }


def test_missing_junction_inner_blend_reports_its_typed_role():
    prompt = (
        "Use outer blend radius 5 mm, inner-bore blend radius 3 mm, and "
        "maximum hub radius 25 mm."
    )
    goal = Goal(
        goal_id="junction",
        type="branch",
        required_outlet_vectors=[(1.0, 1.0, 0.0), (1.0, -1.0, 0.0)],
        include_primary_outlet=False,
        junction_style="smooth_hub",
        blend_radius=5.0,
        inner_blend_radius=3.0,
        max_hub_radius=25.0,
    )
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[goal],
        expected_open_ports=2,
        expected_open_ports_source="derived",
    )

    pipeline._validate_intent_safety(prompt, intent, _settings())

    damaged = intent.model_copy(
        update={
            "target_behavior": [goal.model_copy(update={"inner_blend_radius": None})]
        }
    )
    with pytest.raises(ValueError, match="junction_inner_blend_radius"):
        pipeline._validate_intent_safety(prompt, damaged, _settings())


def test_from_section_measurements_bind_to_inherited_graph_state():
    prompt = (
        "Reduce outer diameter from 20 mm to 12 mm and change wall thickness "
        "from 2 mm to 1.5 mm."
    )
    intent = _production_intent().to_intent_result()

    pipeline._validate_intent_safety(prompt, intent, _settings())

    damaged = intent.model_copy(
        update={
            "global_spec": intent.global_spec.model_copy(
                update={"outer_diameter": 18.0}
            )
        }
    )
    with pytest.raises(ValueError, match="diameter_in_reference"):
        pipeline._validate_intent_safety(prompt, damaged, _settings())


def test_route_rise_binds_to_final_relative_z_not_incidental_waypoint():
    prompt = "Create a spatial spline that should rise by about 35 mm."
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        target_behavior=[
            Goal(
                goal_id="rise",
                type="route",
                path_kind="spline",
                required_waypoints=[
                    (40.0, 0.0, 0.0),
                    (80.0, 20.0, 35.0),
                ],
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
            )
        ],
        expected_open_ports=1,
        expected_open_ports_source="derived",
    )

    pipeline._validate_intent_safety(prompt, intent, _settings())

    damaged_goal = intent.target_behavior[0].model_copy(
        update={
            "required_waypoints": [
                (40.0, 0.0, 35.0),
                (80.0, 20.0, 25.0),
            ]
        }
    )
    with pytest.raises(ValueError, match="route_rise"):
        pipeline._validate_intent_safety(
            prompt,
            intent.model_copy(update={"target_behavior": [damaged_goal]}),
            _settings(),
        )


def test_intent_numeric_vocabulary_is_closed_under_sign_for_spatial_routes():
    literals = pipeline._intent_numeric_literals(
        "Use a 70 mm plan radius, a 48 mm upper radius, and rise 85 mm.",
        _settings(),
    )

    assert {"70", "-70", "48", "-48", "85", "-85"} <= set(literals)
    assert pipeline._numeric_literal_schema_fits(literals)


def test_component_detail_alias_cannot_fake_a_second_source_measurement():
    intent = _production_intent().to_intent_result()

    with pytest.raises(ValueError, match="lost or altered"):
        pipeline._validate_intent_safety(
            PROMPT + " 추가로 분류되지 않은 기준 30 mm를 요구한다.",
            intent,
            _settings(),
        )


def test_two_y_manifold_prompt_has_a_schema_valid_safe_intent_realization():
    prompt = Path("prompts/complex_two_y_manifold.txt").read_text()
    assert [
        (contract.value, contract.role)
        for contract in pipeline._explicit_proportional_direction_contracts(prompt)
    ] == [
        ((1.0, -1.0, 1.0), "branch_outlet"),
        ((2.0, -1.0, 1.0), "goal_terminal"),
        ((1.0, 1.0, 1.0), "branch_outlet"),
        ((1.0, -1.0, -1.0), "branch_outlet"),
    ]
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_position=(0.0, 0.0, 0.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[
            Goal(
                goal_id="inlet",
                type="move",
                direction="+X",
                length=70.0,
            ),
            Goal(
                goal_id="first_y",
                depends_on_goal_ids=["inlet"],
                type="branch",
                required_outlets=[
                    {
                        "axis": (1.0, -1.0, 1.0),
                        "length": 85.0,
                        "outer_diameter": 20.0,
                        "wall_thickness": 2.0,
                    }
                ],
                include_primary_outlet=True,
                junction_style="smooth_hub",
                blend_radius=5.0,
                inner_blend_radius=3.0,
                max_hub_radius=25.0,
            ),
            Goal(
                goal_id="central_spline",
                depends_on_goal_ids=["first_y"],
                type="route",
                path_kind="spline",
                required_waypoints=[
                    (40.0, 0.0, 0.0),
                    (75.0, 30.0, 25.0),
                    (110.0, 55.0, 55.0),
                    (150.0, 20.0, 80.0),
                    (190.0, 0.0, 100.0),
                ],
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
            ),
            Goal(
                goal_id="taper",
                depends_on_goal_ids=["central_spline"],
                type="diameter_change",
                diameter_out=16.0,
                wall_thickness_out=1.5,
                transition_length=45.0,
            ),
            Goal(
                goal_id="second_y",
                depends_on_goal_ids=["taper"],
                type="branch",
                required_outlets=[
                    {
                        "axis": (1.0, 1.0, 1.0),
                        "length": 95.0,
                        "outer_diameter": 16.0,
                        "wall_thickness": 1.5,
                    },
                    {
                        "axis": (1.0, -1.0, -1.0),
                        "length": 115.0,
                        "outer_diameter": 16.0,
                        "wall_thickness": 1.5,
                    },
                ],
                include_primary_outlet=False,
                junction_style="smooth_hub",
                blend_radius=4.0,
                inner_blend_radius=2.5,
                max_hub_radius=20.0,
            ),
        ],
        expected_open_ports=3,
        expected_open_ports_source="explicit",
    )

    pipeline._validate_intent_safety(prompt, intent, _settings())


def test_serial_coil_prompt_has_a_schema_valid_safe_intent_realization():
    prompt = Path("prompts/complex_serial_coil.txt").read_text()
    coil = [
        (0.0, 5.0, 20.0),
        (0.0, 20.0, 35.0),
        (0.0, 40.0, 40.0),
        (35.0, 90.0, 45.0),
        (70.0, 100.0, 50.0),
        (110.0, 80.0, 55.0),
        (130.0, 40.0, 60.0),
        (110.0, 0.0, 65.0),
        (70.0, -15.0, 70.0),
        (35.0, 0.0, 75.0),
        (20.0, 40.0, 80.0),
        (35.0, 75.0, 80.0),
        (70.0, 90.0, 85.0),
    ]
    s_curve = [
        (35.0, 15.0, 5.0),
        (70.0, 50.0, 10.0),
        (110.0, 70.0, 20.0),
        (150.0, 30.0, 25.0),
        (190.0, -20.0, 35.0),
    ]
    escape = [
        (40.0, -50.0, 10.0),
        (70.0, -70.0, 10.0),
        (100.0, -60.0, 15.0),
        (130.0, -30.0, 20.0),
        (160.0, 0.0, 25.0),
        (200.0, 0.0, 25.0),
    ]
    assert pipeline._explicit_vector3_values(prompt) == [*coil, *s_curve, *escape]
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=18.0, wall_thickness=2.0),
        start_position=(0.0, 0.0, 0.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[
            Goal(
                goal_id="inlet",
                type="move",
                direction="+X",
                length=55.0,
            ),
            Goal(
                goal_id="bend",
                depends_on_goal_ids=["inlet"],
                type="turn",
                direction="+Z",
                angle=90.0,
                bend_radius=45.0,
            ),
            Goal(
                goal_id="coil",
                depends_on_goal_ids=["bend"],
                type="route",
                path_kind="spline",
                required_waypoints=coil,
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
            ),
            Goal(
                goal_id="s_curve",
                depends_on_goal_ids=["coil"],
                type="route",
                path_kind="spline",
                required_waypoints=s_curve,
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
            ),
            Goal(
                goal_id="escape",
                depends_on_goal_ids=["s_curve"],
                type="route",
                path_kind="spline",
                required_waypoints=escape,
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
                terminal_axis=(1.0, 0.0, 0.0),
            ),
            Goal(
                goal_id="taper",
                depends_on_goal_ids=["escape"],
                type="diameter_change",
                diameter_out=12.0,
                wall_thickness_out=1.5,
                transition_length=45.0,
            ),
        ],
        expected_open_ports=1,
        expected_open_ports_source="explicit",
    )

    pipeline._validate_intent_safety(prompt, intent, _settings())
