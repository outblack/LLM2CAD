from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

import cadgen.pipeline as pipeline
from cadgen.config import load_settings
from cadgen.freecad_script import anchored_inlet_count, geometry_payload
from cadgen.registry import validate_action, validate_draft
from cadgen.schemas import (
    ActionDraft,
    GlobalSpec,
    Goal,
    IntentResult,
    ProductionGlobalSpec,
    ProductionGoal,
    ProductionIntent,
)
from cadgen.state import StateEngine
from cadgen.static_validation import (
    _validate_final_graph,
    build_state_transition,
    validate_step_checkpoint,
)


def _engine() -> StateEngine:
    return StateEngine(load_settings(Path("missing.env")))


def _loop_intent() -> IntentResult:
    return IntentResult(
        global_spec=GlobalSpec(outer_diameter=2.0, wall_thickness=0.2),
        expected_open_ports=0,
        expected_open_ports_source="derived",
        target_behavior=[
            Goal(
                goal_id="G1",
                type="route",
                path_kind="spline",
                required_waypoints=[
                    (20.0, 0.0, 0.0),
                    (20.0, 30.0, 0.0),
                    (-10.0, 30.0, 0.0),
                    (-10.0, 10.0, 0.0),
                ],
                waypoint_frame="global",
                terminal_axis=(0.0, -1.0, 0.0),
            ),
            Goal(
                goal_id="G2",
                type="turn",
                angle=90.0,
                bend_radius=10.0,
                plane_normal=(0.0, 0.0, 1.0),
            ),
            Goal(
                goal_id="G3",
                type="connect",
                depends_on_goal_ids=["G2"],
                connection_target="start_anchor",
            ),
        ],
    )


@pytest.mark.parametrize("straight_count", [1, 4, 17])
def test_serial_closed_agenda_action_lower_bound_counts_shared_final_arc_once(
    straight_count,
):
    goals: list[Goal] = []
    previous_id: str | None = None
    for index in range(1, straight_count + 1):
        line_id = f"line_{index}"
        goals.append(
            Goal(
                goal_id=line_id,
                depends_on_goal_ids=[previous_id] if previous_id else [],
                type="route",
                path_kind="line",
                length=25.0,
            )
        )
        turn_id = f"turn_{index}"
        goals.append(
            Goal(
                goal_id=turn_id,
                depends_on_goal_ids=[line_id],
                type="turn",
                angle=-60.0,
                bend_radius=5.0,
                plane_normal=(0.0, 0.0, 1.0),
            )
        )
        previous_id = turn_id
    goals.append(
        Goal(
            goal_id="close",
            depends_on_goal_ids=[previous_id] if previous_id else [],
            type="connect",
            connection_target="start_anchor",
        )
    )
    state = _engine().initial_state(
        IntentResult(
            global_spec=GlobalSpec(outer_diameter=4.0, wall_thickness=0.4),
            expected_open_ports=0,
            expected_open_ports_source="derived",
            target_behavior=goals,
        )
    )

    assert pipeline._exclusive_goal_action_lower_bound(state) == 2 * straight_count


def _build_pending_loop():
    engine = _engine()
    intent = _loop_intent()
    before = engine.initial_state(intent)
    draft = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "path_kind": "spline",
            "section_source": "inherit_target",
            "waypoint_frame": "global",
            "waypoints": [
                (20.0, 0.0, 0.0),
                (20.0, 30.0, 0.0),
                (-10.0, 30.0, 0.0),
                (-10.0, 10.0, 0.0),
            ],
        },
    )
    assert validate_draft(draft, before).valid
    action = engine.resolve_action(draft, before)
    assert validate_action(action, before).valid
    after = engine.apply_action(action, before)
    transition = build_state_transition(before, action, after, 1)
    assert not [
        issue
        for issue in validate_step_checkpoint(
            before,
            action,
            after,
            intent,
            transition,
        )
        if issue.severity == "error"
    ]
    return engine, intent, before, after


def test_start_anchor_connect_changes_topology_by_one_open_front():
    intent = ProductionIntent(
        global_spec=ProductionGlobalSpec(
            outer_diameter=20.0,
            wall_thickness=2.0,
            is_hollow=True,
            units="mm",
        ),
        start_position=(0.0, 0.0, 0.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[
            ProductionGoal(
                goal_id="G1",
                depends_on_goal_ids=[],
                allow_parallel=False,
                type="route",
                path_kind="line",
                length=40.0,
            ),
            ProductionGoal(
                goal_id="G2",
                depends_on_goal_ids=["G1"],
                allow_parallel=False,
                type="connect",
                connection_target="start_anchor",
            ),
        ],
        expected_open_ports=0,
        expected_open_ports_source="derived",
        required_components=[],
        hard_constraints=[],
        geometric_constraints=[],
    )

    assert intent.expected_open_ports == 0
    assert intent.target_behavior[-1].connection_target == "start_anchor"


def test_start_anchor_contract_cannot_be_first_or_repeated():
    common = {
        "global_spec": ProductionGlobalSpec(
            outer_diameter=20.0,
            wall_thickness=2.0,
            is_hollow=True,
            units="mm",
        ),
        "start_position": (0.0, 0.0, 0.0),
        "start_axis": (1.0, 0.0, 0.0),
        "expected_open_ports": 0,
        "expected_open_ports_source": "derived",
        "required_components": [],
        "hard_constraints": [],
        "geometric_constraints": [],
    }
    with pytest.raises(ValidationError, match="requires prior hollow-run geometry"):
        ProductionIntent(
            **common,
            target_behavior=[
                ProductionGoal(
                    goal_id="G1",
                    depends_on_goal_ids=[],
                    allow_parallel=False,
                    type="connect",
                    connection_target="start_anchor",
                )
            ],
        )

    with pytest.raises(ValidationError, match="only once"):
        ProductionIntent(
            **common,
            target_behavior=[
                ProductionGoal(
                    goal_id="G1",
                    depends_on_goal_ids=[],
                    allow_parallel=False,
                    type="route",
                    path_kind="line",
                    length=10.0,
                ),
                ProductionGoal(
                    goal_id="G2",
                    depends_on_goal_ids=["G1"],
                    allow_parallel=False,
                    type="connect",
                    connection_target="start_anchor",
                ),
                ProductionGoal(
                    goal_id="G3",
                    depends_on_goal_ids=["G2"],
                    allow_parallel=False,
                    type="connect",
                    connection_target="start_anchor",
                ),
            ],
        )


def test_first_module_inlet_becomes_real_reserved_anchor_without_dummy_edge():
    _engine_value, _intent, before, after = _build_pending_loop()

    anchor = after.reserved_start_anchor
    assert anchor is not None
    assert anchor.id == "M1.in"
    assert "START" not in after.port_nodes
    assert after.placed_modules[0].input_bindings == {}
    assert after.connection_edges == []
    assert before.open_port_ids == ["START"]
    assert after.open_port_ids == ["M1.out"]


def test_final_arc_connect_consumes_anchor_and_preserves_turn_contract():
    engine, intent, _initial, before = _build_pending_loop()
    anchor = before.reserved_start_anchor
    assert anchor is not None
    midpoint = (
        -10.0 / math.sqrt(2.0),
        10.0 - 10.0 / math.sqrt(2.0),
        0.0,
    )
    draft = ActionDraft(
        target_port="M1.out",
        module="connect_ports",
        catalog_schema_version=2,
        affected_goal_ids=["G2", "G3"],
        completed_goal_ids=["G2", "G3"],
        params={
            "path_kind": "circular_arc",
            "section_source": "inherit_target",
            "other_port_id": anchor.id,
            "waypoints": [midpoint],
        },
    )
    assert validate_draft(draft, before).valid
    action = engine.resolve_action(draft, before)
    assert action.params["bend_radius"] == pytest.approx(10.0)
    assert action.params["sweep_angle"] == pytest.approx(90.0)
    assert action.params["plane_normal"] == pytest.approx((0.0, 0.0, 1.0))
    assert validate_action(action, before).valid

    after = engine.apply_action(action, before)
    transition = build_state_transition(before, action, after, 2)
    issues = validate_step_checkpoint(
        before,
        action,
        after,
        intent,
        transition,
    )

    assert not [issue for issue in issues if issue.severity == "error"]
    assert after.open_ports == []
    assert after.reserved_start_anchor is None
    assert transition.removed_port_ids == ["M1.out", "M1.in"]
    assert len(after.connection_edges) == 2
    assert set(after.placed_modules[-1].input_bindings.values()) == {
        "M1.out",
        "M1.in",
    }
    final_graph_issues = []
    _validate_final_graph(final_graph_issues, after)
    assert final_graph_issues == []


def test_start_anchor_is_unavailable_before_bootstrap_and_after_consumption():
    engine = _engine()
    intent = _loop_intent()
    initial = engine.initial_state(intent)
    premature = ActionDraft(
        target_port="START",
        module="connect_ports",
        catalog_schema_version=2,
        affected_goal_ids=["G3"],
        completed_goal_ids=["G3"],
        params={
            "path_kind": "line",
            "section_source": "inherit_target",
            "other_port_id": "M1.in",
        },
    )
    result = validate_draft(premature, initial)
    assert not result.valid
    assert any("unavailable until" in error for error in result.errors)


def test_same_action_dependency_exception_is_limited_to_final_turn_arc():
    _engine_value, _intent, _initial, pending = _build_pending_loop()
    anchor = pending.reserved_start_anchor
    assert anchor is not None
    non_arc = ActionDraft(
        target_port="M1.out",
        module="connect_ports",
        catalog_schema_version=2,
        affected_goal_ids=["G2", "G3"],
        completed_goal_ids=["G2", "G3"],
        params={
            "path_kind": "line",
            "section_source": "inherit_target",
            "other_port_id": anchor.id,
        },
    )

    result = validate_draft(non_arc, pending)

    assert not result.valid
    assert any("incomplete dependencies" in error for error in result.errors)


def test_freecad_root_tracks_pending_anchor_and_disappears_after_closure():
    engine, _intent, initial, pending = _build_pending_loop()
    assert anchored_inlet_count(initial) == 1
    assert geometry_payload(initial)["root_port"]["id"] == "START"
    assert anchored_inlet_count(pending) == 1
    assert geometry_payload(pending)["root_port"]["id"] == "M1.in"
    assert geometry_payload(pending)["root_port"]["axis"] == [1.0, 0.0, 0.0]

    anchor = pending.reserved_start_anchor
    assert anchor is not None
    midpoint = (
        -10.0 / math.sqrt(2.0),
        10.0 - 10.0 / math.sqrt(2.0),
        0.0,
    )
    action = engine.resolve_action(
        ActionDraft(
            target_port="M1.out",
            module="connect_ports",
            catalog_schema_version=2,
            affected_goal_ids=["G2", "G3"],
            completed_goal_ids=["G2", "G3"],
            params={
                "path_kind": "circular_arc",
                "section_source": "inherit_target",
                "other_port_id": anchor.id,
                "waypoints": [midpoint],
            },
        ),
        pending,
    )
    closed = engine.apply_action(action, pending)
    assert anchored_inlet_count(closed) == 0
    assert geometry_payload(closed)["root_port"] is None


def test_length_only_line_inherits_target_tangent():
    engine = _engine()
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        expected_open_ports=1,
        expected_open_ports_source="derived",
        start_axis=(0.0, 1.0, 0.0),
        target_behavior=[Goal(goal_id="G1", type="route", length=25.0)],
    )
    state = engine.initial_state(intent)
    draft = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "path_kind": "line",
            "section_source": "inherit_target",
            "length": 25.0,
        },
    )

    assert validate_draft(draft, state).valid
    action = engine.resolve_action(draft, state)
    assert action.params["direction"] == pytest.approx((0.0, 1.0, 0.0))
    assert action.params["axis"] == pytest.approx((0.0, 1.0, 0.0))
