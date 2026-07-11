from __future__ import annotations

import math
from pathlib import Path

import pytest

from cadgen.config import load_settings
from cadgen.freecad_script import build_freecad_script
from cadgen.geometry_policy import (
    CIRCULAR_SWEEP_EQUALITY_ULPS,
    classify_circular_sweep_radius,
    minimum_spline_curvature_radius,
)
from cadgen.registry import validate_action, validate_draft
from cadgen.schemas import (
    ActionDraft,
    GlobalSpec,
    Goal,
    IntentResult,
    Port,
)
from cadgen.state import StateEngine
from cadgen.vector import normalize, rotate


def _engine() -> StateEngine:
    return StateEngine(load_settings(Path("missing.env")))


def _intent(
    outer_radius: float,
    goal: Goal,
    *,
    start_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    start_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> IntentResult:
    return IntentResult(
        global_spec=GlobalSpec(
            outer_diameter=outer_radius * 2.0,
            wall_thickness=outer_radius * 0.2,
        ),
        start_position=start_position,
        start_axis=start_axis,
        target_behavior=[goal],
        expected_open_ports=1,
        expected_open_ports_source="explicit",
    )


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e6])
def test_circular_sweep_radius_classification_is_scale_aware_at_ulps(scale):
    profile_radius = 7.25 * scale
    below_one_ulp = math.nextafter(profile_radius, -math.inf)
    above_one_ulp = math.nextafter(profile_radius, math.inf)
    below_accumulated_roundoff = profile_radius - 32.0 * math.ulp(profile_radius)

    exact = classify_circular_sweep_radius(profile_radius, profile_radius)
    below = classify_circular_sweep_radius(below_one_ulp, profile_radius)
    above = classify_circular_sweep_radius(above_one_ulp, profile_radius)
    accumulated = classify_circular_sweep_radius(
        below_accumulated_roundoff,
        profile_radius,
    )

    assert exact.classification == "horn_boundary"
    assert below.classification == "horn_boundary"
    assert above.classification == "horn_boundary"
    assert accumulated.classification == "horn_boundary"
    assert below.raw_radial_clearance < 0.0
    assert below.radial_clearance == 0.0
    assert exact.equality_roundoff_band == pytest.approx(
        CIRCULAR_SWEEP_EQUALITY_ULPS * math.ulp(profile_radius)
    )

    outside_delta = exact.equality_roundoff_band * 4.0
    assert (
        classify_circular_sweep_radius(
            profile_radius - outside_delta,
            profile_radius,
        ).classification
        == "self_intersecting"
    )
    assert (
        classify_circular_sweep_radius(
            profile_radius + outside_delta,
            profile_radius,
        ).classification
        == "regular"
    )


@pytest.mark.parametrize("centerline_radius", [0.0, -1.0, math.inf, math.nan])
def test_circular_sweep_radius_classification_rejects_invalid_radius(
    centerline_radius,
):
    with pytest.raises(ValueError, match="centerline_radius"):
        classify_circular_sweep_radius(centerline_radius, 1.0)


@pytest.mark.parametrize(
    ("clearance_sign", "expected_valid"),
    [(-1.0, False), (0.0, True), (1.0, True)],
)
def test_route_registry_uses_ring_horn_spindle_policy(
    clearance_sign,
    expected_valid,
):
    outer_radius = 13.0
    ulp_delta = math.ulp(outer_radius) * (CIRCULAR_SWEEP_EQUALITY_ULPS + 4)
    bend_radius = outer_radius + clearance_sign * ulp_delta
    engine = _engine()
    before = engine.initial_state(
        _intent(
            outer_radius,
            Goal(goal_id="arc", type="route", path_kind="circular_arc"),
        )
    )
    draft = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["arc"],
        completed_goal_ids=["arc"],
        params={
            "path_kind": "circular_arc",
            "section_source": "inherit_target",
            "bend_radius": bend_radius,
            "sweep_angle": 72.0,
            "plane_normal": (0.0, 0.0, 1.0),
        },
    )

    assert validate_draft(draft, before).valid is expected_valid
    action = engine.resolve_action(draft, before)
    result = validate_action(action, before)
    assert result.valid is expected_valid
    if not expected_valid:
        assert any(
            "classification=self_intersecting" in error for error in result.errors
        )


@pytest.mark.parametrize(
    ("start_axis", "plane_normal", "sweep_angle"),
    [
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), -72.0),
        ((1.0, 1.0, 0.0), (0.0, 0.0, 1.0), 144.0),
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), -90.0),
    ],
)
def test_analytic_circular_arc_preserves_signed_sweep_and_arbitrary_frame(
    start_axis,
    plane_normal,
    sweep_angle,
):
    outer_radius = 9.0
    start_position = (3.0, -5.0, 7.0)
    engine = _engine()
    intent = _intent(
        outer_radius,
        Goal(goal_id="arc", type="route", path_kind="circular_arc"),
        start_position=start_position,
        start_axis=normalize(start_axis),
    )
    before = engine.initial_state(intent)
    draft = ActionDraft(
        target_port="START",
        module="route",
        catalog_schema_version=2,
        affected_goal_ids=["arc"],
        completed_goal_ids=["arc"],
        params={
            "path_kind": "circular_arc",
            "section_source": "inherit_target",
            "bend_radius": outer_radius,
            "sweep_angle": sweep_angle,
            "plane_normal": plane_normal,
        },
    )
    action = engine.resolve_action(draft, before)
    assert validate_action(action, before).valid
    after = engine.apply_action(action, before)
    module = after.placed_modules[-1]

    assert module.params["path_points"][0] == pytest.approx(start_position)
    assert module.ports["out"].axis == pytest.approx(
        rotate(
            normalize(start_axis),
            normalize(plane_normal),
            math.radians(sweep_angle),
        )
    )
    script = build_freecad_script(after)
    assert "analytic_torus_segment_outer_minus_bore" in script
    assert "make_analytic_circular_tube" in script
    assert "module_circular_sweep_evidence" in script
    compile(script, "generated_freecad.py", "exec")


@pytest.mark.parametrize(
    ("clearance_sign", "expected_valid"),
    [(-1.0, False), (0.0, True), (1.0, True)],
)
def test_legacy_bend_registry_uses_same_radius_policy(
    clearance_sign,
    expected_valid,
):
    outer_radius = 4.5
    delta = math.ulp(outer_radius) * (CIRCULAR_SWEEP_EQUALITY_ULPS + 4)
    engine = _engine()
    before = engine.initial_state(
        _intent(
            outer_radius,
            Goal(
                goal_id="turn",
                type="turn",
                direction="+Y",
                angle=90.0,
                bend_radius=outer_radius,
            ),
        )
    )
    draft = ActionDraft(
        target_port="START",
        module="bend_pipe",
        affected_goal_ids=["turn"],
        completed_goal_ids=["turn"],
        params={
            "angle": 90.0,
            "turn_direction": "+Y",
            "bend_radius": outer_radius + clearance_sign * delta,
            "segment_resolution": 24,
        },
    )
    action = engine.resolve_action(draft, before)
    assert validate_action(action, before).valid is expected_valid


@pytest.mark.parametrize(
    ("centerline_scale", "expected_valid"),
    [(-1.0, False), (0.0, True), (1.0, True)],
)
def test_circular_connect_uses_same_radius_policy(
    centerline_scale,
    expected_valid,
):
    outer_radius = 6.0
    delta = math.ulp(outer_radius) * (CIRCULAR_SWEEP_EQUALITY_ULPS + 4)
    centerline_radius = outer_radius + centerline_scale * delta
    engine = _engine()
    connect_goal = Goal(
        goal_id="close",
        type="connect",
        connection_target="start_anchor",
    )
    state = engine.initial_state(_intent(outer_radius, connect_goal))
    target = Port(
        id="front",
        position=(2.0 * centerline_radius, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
        outer_diameter=2.0 * outer_radius,
        wall_thickness=0.2 * outer_radius,
    )
    anchor = Port(
        id="anchor",
        position=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
        outer_diameter=2.0 * outer_radius,
        wall_thickness=0.2 * outer_radius,
    )
    state = state.model_copy(
        update={
            "open_ports": [target],
            "open_port_ids": [target.id],
            "port_nodes": {target.id: target, anchor.id: anchor},
            "reserved_start_anchor": anchor,
        }
    )
    draft = ActionDraft(
        target_port=target.id,
        module="connect_ports",
        catalog_schema_version=2,
        affected_goal_ids=["close"],
        completed_goal_ids=["close"],
        params={
            "path_kind": "circular_arc",
            "section_source": "inherit_target",
            "other_port_id": anchor.id,
            "waypoints": [(centerline_radius, centerline_radius, 0.0)],
        },
    )
    action = engine.resolve_action(draft, state)
    result = validate_action(action, state)
    assert result.valid is expected_valid
    if expected_valid:
        assert action.params["minimum_curvature_radius"] == pytest.approx(outer_radius)
    else:
        assert any(
            "classification=self_intersecting" in error for error in result.errors
        )


def test_spline_reserve_remains_stricter_than_horn_boundary():
    outer_radius = 11.0
    required = minimum_spline_curvature_radius(
        outer_radius * 2.0,
        1e-4,
    )
    assert required >= outer_radius * 2.0
    assert (
        classify_circular_sweep_radius(
            outer_radius,
            outer_radius,
        ).classification
        == "horn_boundary"
    )
