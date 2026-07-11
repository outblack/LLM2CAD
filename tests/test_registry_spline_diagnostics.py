from __future__ import annotations

from pathlib import Path

import pytest

from cadgen.config import load_settings
from cadgen.geometry_policy import predict_c1_spline
from cadgen.registry import validate_action
from cadgen.schemas import ActionDraft, GlobalSpec, Goal, IntentResult
from cadgen.state import StateEngine


def test_c1_prediction_localizes_the_maximum_curvature_sample() -> None:
    prediction = predict_c1_spline(
        [
            (60.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
            (120.0, 20.0, 20.0),
            (150.0, 0.0, 0.0),
        ],
        (1.0, 0.0, 0.0),
        (30.0, -20.0, -20.0),
        modeling_tolerance=1e-4,
    )

    assert prediction.minimum_radius == pytest.approx(14.24213473058492)
    assert prediction.critical_span_index == 1
    assert prediction.critical_t == pytest.approx(0.8359375)
    assert prediction.critical_position == pytest.approx(
        (112.40213513424118, 18.82587379972116, 18.82587379972116)
    )


def test_retry_shapes_expose_harmful_insertions_and_noncausal_tail_extension() -> None:
    start = (60.0, 0.0, 0.0)
    core = [
        (90.0, 0.0, 0.0),
        (120.0, 20.0, 20.0),
        (150.0, 0.0, 0.0),
    ]
    initial = (1.0, 0.0, 0.0)
    final = (0.7276068751089989, -0.48507125007266594, -0.48507125007266594)

    baseline = predict_c1_spline(
        [start, *core], initial, final, modeling_tolerance=1e-4
    )
    close_insertions = predict_c1_spline(
        [
            start,
            core[0],
            (105.0, 10.0, 10.0),
            core[1],
            (135.0, 10.0, 10.0),
            core[2],
        ],
        initial,
        final,
        modeling_tolerance=1e-4,
    )
    medium_tail = predict_c1_spline(
        [start, *core, (190.0, -26.7, -26.7)],
        initial,
        final,
        modeling_tolerance=1e-4,
    )
    long_tail = predict_c1_spline(
        [start, *core, (240.0, -60.0, -60.0)],
        initial,
        final,
        modeling_tolerance=1e-4,
    )

    assert baseline.minimum_radius == pytest.approx(14.24213473058492)
    assert close_insertions.minimum_radius == pytest.approx(8.12187395585)
    assert medium_tail.minimum_radius == pytest.approx(baseline.minimum_radius)
    assert long_tail.minimum_radius == pytest.approx(baseline.minimum_radius)
    assert medium_tail.critical_span_index == 1
    assert long_tail.critical_span_index == 1


def test_spline_curvature_failure_has_machine_readable_inverse_evidence() -> None:
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_position=(60.0, 0.0, 0.0),
        target_behavior=[
            Goal(
                goal_id="central_spline",
                type="route",
                path_kind="spline",
                required_waypoints=[
                    (30.0, 0.0, 0.0),
                    (60.0, 20.0, 20.0),
                    (90.0, 0.0, 0.0),
                ],
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
                minimum_curvature_radius=20.0,
            )
        ],
        expected_open_ports=1,
    )
    engine = StateEngine(load_settings(Path("missing.env")))
    state = engine.initial_state(intent)
    resolved = engine.resolve_action(
        ActionDraft(
            target_port="START",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["central_spline"],
            completed_goal_ids=["central_spline"],
            params={
                "section_source": "inherit_target",
                "path_kind": "spline",
                "waypoint_frame": "relative_to_target",
                "waypoints": [
                    (30.0, 0.0, 0.0),
                    (60.0, 20.0, 20.0),
                    (90.0, 0.0, 0.0),
                ],
            },
        ),
        state,
    )

    result = validate_action(resolved, state)

    assert not result.valid
    assert "spline curvature preflight failed" in " ".join(result.errors)
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "SPLINE_CURVATURE_PREFLIGHT"
    assert diagnostic.evaluator_id == "predict_c1_spline"
    assert diagnostic.evaluator_version == "c1-cubic-sampling-v1"
    assert "minimum_radius=1/max(kappa)" in diagnostic.calculation_method
    assert diagnostic.metric == "minimum_curvature_radius"
    assert diagnostic.comparator == ">="
    assert diagnostic.required == pytest.approx(20.0)
    assert diagnostic.actual == pytest.approx(14.24213473058492)
    assert diagnostic.gap == pytest.approx(diagnostic.required - diagnostic.actual)
    assert diagnostic.ratio == pytest.approx(diagnostic.required / diagnostic.actual)
    assert diagnostic.modeling_tolerance == pytest.approx(1e-4)
    assert diagnostic.critical_span_index == 1
    assert diagnostic.critical_t == pytest.approx(0.8359375)
    assert diagnostic.critical_span_endpoints == (
        (90.0, 0.0, 0.0),
        (120.0, 20.0, 20.0),
    )
    assert diagnostic.critical_location == pytest.approx(
        (112.40213513424118, 18.82587379972116, 18.82587379972116)
    )
    assert diagnostic.handle_factors == pytest.approx([0.4, 0.5, 0.45, 0.4])
    assert diagnostic.curve_length == pytest.approx(117.37710770667105)
    assert diagnostic.polyline_length == pytest.approx(112.4621125123532)
    assert diagnostic.minimum_chord == pytest.approx(30.0)
    assert diagnostic.implicated_parameter_paths == [
        "/resolved_action/params/minimum_curvature_radius",
        "/resolved_action/params/start_position",
        "/params/waypoints/0",
        "/params/waypoints/1",
        "/params/waypoints/2",
    ]


def test_passing_spline_keeps_the_structured_diagnostic_list_empty() -> None:
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=8.0, wall_thickness=1.0),
        target_behavior=[
            Goal(
                goal_id="broad_spline",
                type="route",
                path_kind="spline",
                required_waypoints=[(40.0, 0.0, 0.0), (80.0, 10.0, 0.0)],
                waypoint_frame="relative_to_target",
                waypoint_scale_policy="fixed",
                minimum_curvature_radius=4.1,
            )
        ],
        expected_open_ports=1,
    )
    engine = StateEngine(load_settings(Path("missing.env")))
    state = engine.initial_state(intent)
    resolved = engine.resolve_action(
        ActionDraft(
            target_port="START",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["broad_spline"],
            completed_goal_ids=["broad_spline"],
            params={
                "section_source": "inherit_target",
                "path_kind": "spline",
                "waypoint_frame": "relative_to_target",
                "waypoints": [(40.0, 0.0, 0.0), (80.0, 10.0, 0.0)],
            },
        ),
        state,
    )

    result = validate_action(resolved, state)

    assert result.valid
    assert result.errors == []
    assert result.diagnostics == []
