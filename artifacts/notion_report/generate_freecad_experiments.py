from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from cadgen.config import load_settings
from cadgen.freecad_script import build_freecad_script, candidate_document_name
from cadgen.registry import validate_action, validate_draft
from cadgen.schemas import ActionDraft, GlobalSpec, Goal, IntentResult, PipeState, Port
from cadgen.state import StateEngine
from cadgen.static_validation import build_final_critic_report, build_step_verification


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "notion_report"
SETTINGS = load_settings(ROOT / "missing.env").with_overrides(
    output_dir=OUT,
    skip_freecad=True,
)
ENGINE = StateEngine(SETTINGS)


@dataclass
class Scenario:
    name: str
    title: str
    state: PipeState
    target_ids: list[str]
    context_ids: list[str]
    note: str
    static_records: list[dict]
    keep_open: bool = False


def intent(
    *goals: Goal,
    expected_open_ports: int | None = 1,
    required_components: list[str] | None = None,
) -> IntentResult:
    return IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_position=(0.0, 0.0, 0.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=list(goals),
        expected_open_ports=expected_open_ports,
        expected_open_ports_source="explicit",
        required_components=required_components or [],
        contract_digest="notion-report-live-freecad-experiment",
    )


def apply_with_checks(
    before: PipeState,
    draft: ActionDraft,
    intent_value: IntentResult,
    step_index: int,
) -> tuple[PipeState, dict]:
    draft_result = validate_draft(draft, before)
    action = ENGINE.resolve_action(draft, before)
    action_result = validate_action(action, before)
    after = ENGINE.apply_action(action, before)
    step = build_step_verification(
        before,
        action,
        after,
        intent_value,
        step_index,
    )
    return after, {
        "step": step_index,
        "state_before": before.state_id,
        "state_after": after.state_id,
        "action": action.model_dump(mode="json"),
        "draft_valid": draft_result.valid,
        "draft_errors": draft_result.errors,
        "resolved_action_valid": action_result.valid,
        "resolved_action_errors": action_result.errors,
        "static_step_status": step.status,
        "static_issue_codes": [issue.issue_code for issue in step.issues],
        "connection_edges": [
            edge.model_dump(mode="json")
            for edge in after.connection_edges
            if edge.action_id == action.action_id
        ],
        "open_port_ids": list(after.open_port_ids),
    }


def one_action_scenario(
    *,
    name: str,
    title: str,
    goal: Goal,
    draft: ActionDraft,
    expected_open_ports: int,
    target_ids: list[str] | None = None,
    context_ids: list[str] | None = None,
    note: str,
    required_components: list[str] | None = None,
) -> Scenario:
    intent_value = intent(
        goal,
        expected_open_ports=expected_open_ports,
        required_components=required_components,
    )
    state0 = ENGINE.initial_state(intent_value)
    state1, record = apply_with_checks(state0, draft, intent_value, 1)
    return Scenario(
        name=name,
        title=title,
        state=state1,
        target_ids=target_ids or ["M1"],
        context_ids=context_ids or [],
        note=note,
        static_records=[record],
    )


def route_scenarios() -> list[Scenario]:
    line = one_action_scenario(
        name="primitive_route_default_line",
        title="Route primitive — line, L=80 mm",
        goal=Goal(goal_id="G1", type="move", direction="+X", length=80.0),
        draft=ActionDraft(
            target_port="START",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params={
                "path_kind": "line",
                "section_source": "inherit_target",
                "length": 80.0,
                "direction": (1.0, 0.0, 0.0),
            },
        ),
        expected_open_ports=1,
        note="Constant-section line route. Increasing length moves the outlet along the authored direction.",
    )
    arc = one_action_scenario(
        name="primitive_route_changed_arc",
        title="Route primitive — circular arc, R=40 mm, 90°",
        goal=Goal(
            goal_id="G1",
            type="turn",
            direction="+Z",
            angle=90.0,
            bend_radius=40.0,
            plane_normal=(0.0, -1.0, 0.0),
        ),
        draft=ActionDraft(
            target_port="START",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params={
                "path_kind": "circular_arc",
                "section_source": "inherit_target",
                "bend_radius": 40.0,
                "sweep_angle": 90.0,
                "plane_normal": (0.0, -1.0, 0.0),
                "terminal_axis": (0.0, 0.0, 1.0),
            },
        ),
        expected_open_ports=1,
        note="Changing the route variant and curvature parameters changes both terminal position and tangent.",
    )
    changed = one_action_scenario(
        name="primitive_route_changed_line_140",
        title="Route primitive — line, L=140 mm",
        goal=Goal(goal_id="G1", type="move", direction="+X", length=140.0),
        draft=ActionDraft(
            target_port="START",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params={
                "path_kind": "line",
                "section_source": "inherit_target",
                "length": 140.0,
                "direction": (1.0, 0.0, 0.0),
            },
        ),
        expected_open_ports=1,
        note="Increasing L from 80 to 140 mm translates the outlet by 60 mm without changing section or tangent.",
    )
    return [line, changed, arc]


def transition_scenarios() -> list[Scenario]:
    concentric = one_action_scenario(
        name="primitive_transition_default_concentric",
        title="Transition primitive — concentric 20→30 mm",
        goal=Goal(
            goal_id="G1",
            type="diameter_change",
            direction="+X",
            diameter_out=30.0,
            wall_thickness_out=2.5,
            transition_length=60.0,
            offset=(0.0, 0.0, 0.0),
        ),
        draft=ActionDraft(
            target_port="START",
            module="transition",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params={
                "section_source": "inherit_target",
                "diameter_out": 30.0,
                "wall_thickness_out": 2.5,
                "length": 60.0,
                "offset": (0.0, 0.0, 0.0),
            },
        ),
        expected_open_ports=1,
        note="Concentric loft changes outer and inner radii along the same axis.",
    )
    eccentric = one_action_scenario(
        name="primitive_transition_changed_eccentric",
        title="Transition primitive — eccentric 20→12 mm, offset 6 mm",
        goal=Goal(
            goal_id="G1",
            type="diameter_change",
            diameter_out=12.0,
            wall_thickness_out=1.5,
            transition_length=75.0,
            offset=(0.0, 6.0, 0.0),
        ),
        draft=ActionDraft(
            target_port="START",
            module="transition",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params={
                "section_source": "inherit_target",
                "diameter_out": 12.0,
                "wall_thickness_out": 1.5,
                "length": 75.0,
                "offset": (0.0, 6.0, 0.0),
            },
        ),
        expected_open_ports=1,
        note="A transverse offset produces an eccentric reducer while preserving the inlet tangent frame.",
    )
    return [concentric, eccentric]


def junction_scenarios() -> list[Scenario]:
    base_goal = Goal(
        goal_id="G1",
        type="branch",
        branch_count=1,
        include_primary_outlet=True,
        required_outlet_vectors=[(0.0, 1.0, 0.0)],
        junction_style="hard_fuse",
        max_hub_radius=14.0,
    )
    tee = one_action_scenario(
        name="primitive_junction_default_tee",
        title="Junction primitive — one primary + one branch",
        goal=base_goal,
        draft=ActionDraft(
            target_port="START",
            module="junction",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params={
                "section_source": "inherit_target",
                "outlets": [
                    {
                        "role": "primary",
                        "axis": (1.0, 0.0, 0.0),
                        "length": 65.0,
                        "outer_diameter": 20.0,
                        "wall_thickness": 2.0,
                    },
                    {
                        "role": "branch",
                        "axis": (0.0, 1.0, 0.0),
                        "length": 50.0,
                        "outer_diameter": 20.0,
                        "wall_thickness": 2.0,
                    },
                ],
                "blend_mode": "hard",
                "max_hub_radius": 14.0,
            },
        ),
        expected_open_ports=2,
        note="The explicit outlet list determines arity, axes, lengths, and sections; this is a T-junction.",
    )
    diagonal_axis = (0.0, 0.7071067811865476, 0.7071067811865476)
    diagonal = one_action_scenario(
        name="primitive_junction_changed_diagonal_branch",
        title="Junction primitive — branch axis 45° upward, L=70 mm",
        goal=Goal(
            goal_id="G1",
            type="branch",
            branch_count=1,
            include_primary_outlet=True,
            required_outlet_vectors=[diagonal_axis],
            junction_style="hard_fuse",
            max_hub_radius=14.0,
        ),
        draft=ActionDraft(
            target_port="START",
            module="junction",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params={
                "section_source": "inherit_target",
                "outlets": [
                    {
                        "role": "primary",
                        "axis": (1.0, 0.0, 0.0),
                        "length": 65.0,
                        "outer_diameter": 20.0,
                        "wall_thickness": 2.0,
                    },
                    {
                        "role": "branch",
                        "axis": diagonal_axis,
                        "length": 70.0,
                        "outer_diameter": 20.0,
                        "wall_thickness": 2.0,
                    },
                ],
                "blend_mode": "hard",
                "max_hub_radius": 14.0,
            },
        ),
        expected_open_ports=2,
        note="Changing only branch axis and length rotates and extends one outlet while preserving 1→2 topology.",
    )
    manifold_goal = Goal(
        goal_id="G1",
        type="branch",
        branch_count=2,
        include_primary_outlet=True,
        required_outlet_vectors=[(0.0, 1.0, 0.0), (0.0, -1.0, 0.0)],
        junction_style="hard_fuse",
        max_hub_radius=14.0,
    )
    manifold = one_action_scenario(
        name="primitive_junction_changed_three_outlets",
        title="Junction primitive — one primary + two branches",
        goal=manifold_goal,
        draft=ActionDraft(
            target_port="START",
            module="junction",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params={
                "section_source": "inherit_target",
                "outlets": [
                    {
                        "role": "primary",
                        "axis": (1.0, 0.0, 0.0),
                        "length": 65.0,
                        "outer_diameter": 20.0,
                        "wall_thickness": 2.0,
                    },
                    {
                        "role": "branch",
                        "axis": (0.0, 1.0, 0.0),
                        "length": 50.0,
                        "outer_diameter": 20.0,
                        "wall_thickness": 2.0,
                    },
                    {
                        "role": "branch",
                        "axis": (0.0, -1.0, 0.0),
                        "length": 50.0,
                        "outer_diameter": 20.0,
                        "wall_thickness": 2.0,
                    },
                ],
                "blend_mode": "hard",
                "max_hub_radius": 14.0,
            },
        ),
        expected_open_ports=3,
        note="Adding an outlet changes graph arity from 1→2 to 1→3; each outlet owns its own section.",
    )
    return [tee, diagonal, manifold]


def two_facing_port_state(goal: Goal, distance: float = 100.0) -> tuple[IntentResult, PipeState]:
    intent_value = intent(goal, expected_open_ports=0)
    state = ENGINE.initial_state(intent_value)
    other = Port(
        id="P2",
        position=(distance, 0.0, 0.0),
        axis=(-1.0, 0.0, 0.0),
        outer_diameter=20.0,
        wall_thickness=2.0,
    )
    return intent_value, state.model_copy(
        update={
            "open_ports": [state.open_ports[0], other],
            "open_port_ids": ["START", "P2"],
            "port_nodes": {"START": state.open_ports[0], "P2": other},
        }
    )


def connect_scenarios() -> list[Scenario]:
    goal = Goal(goal_id="G1", type="connect")
    intent_value, state0 = two_facing_port_state(goal)
    line_draft = ActionDraft(
        target_port="START",
        module="connect_ports",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "other_port_id": "P2",
            "path_kind": "line",
            "section_source": "inherit_target",
        },
    )
    line_state, line_record = apply_with_checks(state0, line_draft, intent_value, 1)
    line = Scenario(
        name="primitive_connect_ports_default_line",
        title="Connect-ports primitive — direct closure",
        state=line_state,
        target_ids=["M1"],
        context_ids=[],
        note="Two existing compatible ports are consumed and no new port is produced.",
        static_records=[line_record],
    )

    intent_value_changed, state_changed = two_facing_port_state(goal, distance=160.0)
    changed_draft = ActionDraft(
        target_port="START",
        module="connect_ports",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "other_port_id": "P2",
            "path_kind": "line",
            "section_source": "inherit_target",
        },
    )
    changed_state, changed_record = apply_with_checks(
        state_changed, changed_draft, intent_value_changed, 1
    )
    changed = Scenario(
        name="primitive_connect_ports_changed_line_160",
        title="Connect-ports primitive — direct closure, separation 160 mm",
        state=changed_state,
        target_ids=["M1"],
        context_ids=[],
        note="Increasing endpoint separation from 100 to 160 mm lengthens the closure while topology remains 2→0.",
        static_records=[changed_record],
    )

    intent_value2, state02 = two_facing_port_state(goal)
    spline_draft = ActionDraft(
        target_port="START",
        module="connect_ports",
        catalog_schema_version=2,
        affected_goal_ids=["G1"],
        completed_goal_ids=["G1"],
        params={
            "other_port_id": "P2",
            "path_kind": "spline",
            "section_source": "inherit_target",
            "waypoints": [(25.0, 28.0, 0.0), (75.0, 28.0, 0.0)],
            "initial_tangent": (1.0, 0.0, 0.0),
            "final_tangent": (1.0, 0.0, 0.0),
            "interpolation": "bspline",
            "frenet": False,
            "minimum_curvature_radius": 10.1,
        },
    )
    spline_state, spline_record = apply_with_checks(
        state02, spline_draft, intent_value2, 1
    )
    spline = Scenario(
        name="primitive_connect_ports_changed_spline",
        title="Connect-ports primitive — B-spline closure",
        state=spline_state,
        target_ids=["M1"],
        context_ids=[],
        note="Waypoints and endpoint tangents bend the closure while the topology remains 2→0.",
        static_records=[spline_record],
    )
    return [line, changed, spline]


def terminate_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    for termination_type, thickness, name, title in (
        ("cap", 4.0, "primitive_terminate_default_cap", "Terminate primitive — cap, t=4 mm"),
        ("plug", 8.0, "primitive_terminate_changed_plug", "Terminate primitive — plug, t=8 mm"),
    ):
        intent_value = intent(
            Goal(goal_id="G1", type="move", direction="+X", length=70.0),
            Goal(
                goal_id="G2",
                type="end",
                end_type=termination_type,
                termination_thickness=thickness,
            ),
            expected_open_ports=0,
        )
        state0 = ENGINE.initial_state(intent_value)
        route = ActionDraft(
            target_port="START",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params={
                "path_kind": "line",
                "section_source": "inherit_target",
                "length": 70.0,
                "direction": (1.0, 0.0, 0.0),
            },
        )
        state1, record1 = apply_with_checks(state0, route, intent_value, 1)
        terminate = ActionDraft(
            target_port="M1.out",
            module="terminate",
            catalog_schema_version=2,
            affected_goal_ids=["G2"],
            completed_goal_ids=["G2"],
            params={
                "section_source": "inherit_target",
                "termination_type": termination_type,
                "thickness": thickness,
            },
        )
        state2, record2 = apply_with_checks(state1, terminate, intent_value, 2)
        scenarios.append(
            Scenario(
                name=name,
                title=title,
                state=state2,
                target_ids=["M2"],
                context_ids=["M1"],
                note=(
                    "A cap grows downstream from the terminal plane; a plug occupies the upstream bore. "
                    "Both consume one open port and produce none."
                ),
                static_records=[record1, record2],
            )
        )
    return scenarios


def inline_scenario(
    *,
    name: str,
    title: str,
    component_type: str,
    params: dict,
    note: str,
) -> Scenario:
    length = float(params["length"])
    goal = Goal(
        goal_id="G1",
        type="connector",
        direction="+X",
        length=length,
        component=component_type,
    )
    draft_params = {
        "section_source": "inherit_target",
        "component_type": component_type,
        "connector_type_out": "plain",
        "connector_gender_out": "neutral",
        "connector_standard_out": None,
        **params,
    }
    return one_action_scenario(
        name=name,
        title=title,
        goal=goal,
        draft=ActionDraft(
            target_port="START",
            module="inline_component",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params=draft_params,
        ),
        expected_open_ports=1,
        required_components=[component_type],
        note=note,
    )


def inline_scenarios() -> list[Scenario]:
    return [
        inline_scenario(
            name="primitive_inline_default_flange_4bolt",
            title="Inline component — flange, 4 bolts",
            component_type="flange",
            params={
                "length": 24.0,
                "body_outer_diameter": 40.0,
                "body_start_offset": 0.0,
                "body_length": 5.0,
                "flange_bolt_count": 4,
                "flange_bolt_circle_diameter": 30.0,
                "flange_bolt_hole_diameter": 4.0,
                # Rotate the four-hole pattern by 22.5 degrees so deterministic
                # wall-section probes do not pass through the authored bolt cuts.
                "flange_reference_axis": (
                    0.0,
                    0.9238795325112867,
                    0.3826834323650898,
                ),
            },
            note="The flange subtype exposes collar and bolt-pattern parameters as real cut geometry.",
        ),
        inline_scenario(
            name="primitive_inline_changed_flange_8bolt",
            title="Inline component — flange, 8 bolts / larger body",
            component_type="flange",
            params={
                "length": 28.0,
                "body_outer_diameter": 50.0,
                "body_start_offset": 0.0,
                "body_length": 7.0,
                "flange_bolt_count": 8,
                "flange_bolt_circle_diameter": 38.0,
                "flange_bolt_hole_diameter": 4.0,
                "flange_reference_axis": (0.0, 1.0, 0.0),
            },
            note="Increasing bolt count and body dimensions changes the annulus and repeated hole geometry.",
        ),
        inline_scenario(
            name="primitive_inline_subtype_coupling",
            title="Inline component subtype — coupling",
            component_type="coupling",
            params={
                "length": 30.0,
                "body_outer_diameter": 28.0,
                "body_start_offset": 0.0,
                "body_length": 30.0,
            },
            note="A coupling is modeled as a sleeve spanning the full authored axial length.",
        ),
        inline_scenario(
            name="primitive_inline_subtype_union",
            title="Inline component subtype — union",
            component_type="union",
            params={
                "length": 46.0,
                "body_outer_diameter": 34.0,
                "body_start_offset": 9.0,
                "body_length": 28.0,
                "union_ring_outer_diameter": 40.0,
                "union_ring_length": 5.0,
            },
            note="A union has two necks and two explicit end rings around a central body.",
        ),
        inline_scenario(
            name="primitive_inline_subtype_valve",
            title="Inline component subtype — valve",
            component_type="valve",
            params={
                "length": 52.0,
                "body_outer_diameter": 38.0,
                "body_start_offset": 10.0,
                "body_length": 32.0,
                "actuator_diameter": 20.0,
                "actuator_height": 36.0,
                "actuator_axis": (0.0, 0.0, 1.0),
            },
            note="A valve adds a perpendicular actuator solid while retaining an axial flow bore.",
        ),
    ]


def sequential_assembly() -> list[Scenario]:
    intent_value = intent(
        Goal(goal_id="G1", type="move", direction="+X", length=80.0),
        Goal(
            goal_id="G2",
            type="connector",
            direction="+X",
            length=30.0,
            component="coupling",
        ),
        Goal(
            goal_id="G3",
            type="diameter_change",
            direction="+X",
            diameter_out=12.0,
            wall_thickness_out=1.5,
            transition_length=40.0,
            offset=(0.0, 0.0, 0.0),
        ),
        Goal(goal_id="G4", type="move", direction="+X", length=60.0),
        Goal(
            goal_id="G5",
            type="end",
            end_type="cap",
            termination_thickness=3.0,
        ),
        expected_open_ports=0,
        required_components=["coupling"],
    )
    drafts = [
        ActionDraft(
            target_port="START",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["G1"],
            completed_goal_ids=["G1"],
            params={
                "path_kind": "line",
                "section_source": "inherit_target",
                "length": 80.0,
                "direction": (1.0, 0.0, 0.0),
            },
        ),
        ActionDraft(
            target_port="M1.out",
            module="inline_component",
            catalog_schema_version=2,
            affected_goal_ids=["G2"],
            completed_goal_ids=["G2"],
            params={
                "section_source": "inherit_target",
                "component_type": "coupling",
                "length": 30.0,
                "body_outer_diameter": 28.0,
                "body_start_offset": 0.0,
                "body_length": 30.0,
                "connector_type_out": "plain",
                "connector_gender_out": "neutral",
                "connector_standard_out": None,
            },
        ),
        ActionDraft(
            target_port="M2.out",
            module="transition",
            catalog_schema_version=2,
            affected_goal_ids=["G3"],
            completed_goal_ids=["G3"],
            params={
                "section_source": "inherit_target",
                "diameter_out": 12.0,
                "wall_thickness_out": 1.5,
                "length": 40.0,
                "offset": (0.0, 0.0, 0.0),
            },
        ),
        ActionDraft(
            target_port="M3.out",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["G4"],
            completed_goal_ids=["G4"],
            params={
                "path_kind": "line",
                "section_source": "inherit_target",
                "length": 60.0,
                "direction": (1.0, 0.0, 0.0),
            },
        ),
        ActionDraft(
            target_port="M4.out",
            module="terminate",
            catalog_schema_version=2,
            affected_goal_ids=["G5"],
            completed_goal_ids=["G5"],
            params={
                "section_source": "inherit_target",
                "termination_type": "cap",
                "thickness": 3.0,
            },
        ),
    ]
    state = ENGINE.initial_state(intent_value)
    records: list[dict] = []
    scenarios: list[Scenario] = []
    steps = []
    for index, draft in enumerate(drafts, start=1):
        before = state
        state, record = apply_with_checks(before, draft, intent_value, index)
        records.append(record)
        action = state.action_history[-1]
        steps.append(
            build_step_verification(before, action, state, intent_value, index)
        )
        scenarios.append(
            Scenario(
                name=f"assembly_step_{index}",
                title=f"Sequential assembly — step {index}: {draft.module}",
                state=state,
                target_ids=[f"M{index}"],
                context_ids=[f"M{j}" for j in range(1, index)],
                note=(
                    "Natural-language example: straight 80 mm → 30 mm coupling → "
                    "20→12 mm reducer → straight 60 mm → cap."
                ),
                static_records=list(records),
                keep_open=index == len(drafts),
            )
        )
    connection_source = scenarios[1]
    scenarios.append(
        Scenario(
            name="connection_overview_route_to_coupling",
            title="Connection overview — route outlet to coupling inlet",
            state=connection_source.state,
            target_ids=["M2"],
            context_ids=["M1"],
            note=(
                "The shared interface has coincident position, anti-parallel mating axes, "
                "and equal OD/ID/wall values."
            ),
            static_records=list(connection_source.static_records),
        )
    )
    critic = build_final_critic_report(intent_value, state, steps)
    (OUT / "assembly_static_critic.json").write_text(
        json.dumps(critic.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return scenarios


def capture_suffix(scenario: Scenario, image_path: Path, validation_path: Path) -> str:
    shown = scenario.context_ids + scenario.target_ids
    shown_literal = repr(shown)
    targets_literal = repr(set(scenario.target_ids))
    image_literal = repr(str(image_path))
    validation_literal = repr(str(validation_path))
    keep_open = "True" if scenario.keep_open else "False"
    return f'''
import FreeCADGui as Gui
with open({validation_literal}, "w", encoding="utf-8") as handle:
    json.dump(validation, handle, ensure_ascii=False, indent=2, sort_keys=True)
shown_ids = {shown_literal}
target_ids = {targets_literal}
palette = [
    (0.76, 0.80, 0.84),
    (0.34, 0.68, 0.88),
    (0.35, 0.78, 0.58),
    (0.93, 0.66, 0.25),
    (0.76, 0.45, 0.82),
]
for obj in doc.Objects:
    if hasattr(obj, "ViewObject"):
        obj.ViewObject.Visibility = False
for index, module_id in enumerate(shown_ids):
    obj = doc.getObject("solid_" + module_id)
    if obj is None:
        continue
    obj.ViewObject.Visibility = True
    if module_id in target_ids:
        obj.ViewObject.ShapeColor = (0.10, 0.54, 0.86)
        obj.ViewObject.LineColor = (0.05, 0.16, 0.25)
        obj.ViewObject.Transparency = 0
    else:
        obj.ViewObject.ShapeColor = palette[index % len(palette)]
        obj.ViewObject.LineColor = (0.20, 0.24, 0.28)
        obj.ViewObject.Transparency = 18
    obj.ViewObject.DisplayMode = "Flat Lines"
view = Gui.activeDocument().activeView()
view.setAnimationEnabled(False)
view.viewAxonometric()
view.fitAll()
view.saveImage({image_literal}, 1200, 800, "White")
try:
    from PySide import QtCore, QtGui
    source = QtGui.QImage({image_literal})
    width = source.width()
    height = source.height()
    sample_x = list(range(0, width, max(1, width // 80)))
    def mostly_black_row(y):
        black = 0
        for x in sample_x:
            color = source.pixelColor(x, y)
            if color.red() < 12 and color.green() < 12 and color.blue() < 12:
                black += 1
        return black >= max(1, int(len(sample_x) * 0.95))
    top = 0
    while top < height and mostly_black_row(top):
        top += 1
    bottom = height - 1
    while bottom >= top and mostly_black_row(bottom):
        bottom -= 1
    cropped = source.copy(0, top, width, max(1, bottom - top + 1))
    keep_aspect = getattr(
        getattr(QtCore.Qt, "AspectRatioMode", QtCore.Qt),
        "KeepAspectRatio",
    )
    smooth = getattr(
        getattr(QtCore.Qt, "TransformationMode", QtCore.Qt),
        "SmoothTransformation",
    )
    scaled = cropped.scaled(1160, 760, keep_aspect, smooth)
    image_format = getattr(
        getattr(QtGui.QImage, "Format", QtGui.QImage),
        "Format_RGB32",
    )
    canvas = QtGui.QImage(1200, 800, image_format)
    canvas.fill(QtGui.QColor(255, 255, 255))
    painter = QtGui.QPainter(canvas)
    painter.drawImage((1200 - scaled.width()) // 2, (800 - scaled.height()) // 2, scaled)
    painter.end()
    canvas.save({image_literal}, "PNG")
except Exception as postprocess_error:
    print("CADGEN_SCREENSHOT_POSTPROCESS_WARNING=" + str(postprocess_error))
print("CADGEN_SCREENSHOT=" + {image_literal})
print("CADGEN_KEEP_OPEN=" + str({keep_open}))
'''


def write_scenarios(scenarios: Iterable[Scenario]) -> list[dict]:
    index: list[dict] = []
    for scenario in scenarios:
        image_path = OUT / f"{scenario.name}.png"
        validation_path = OUT / f"{scenario.name}.freecad_validation.json"
        state_path = OUT / f"{scenario.name}.state.json"
        static_path = OUT / f"{scenario.name}.static_validation.json"
        script_path = OUT / f"{scenario.name}.freecad.py"
        script = build_freecad_script(
            scenario.state,
            run_id=scenario.name,
            attempt_id=1,
        ) + capture_suffix(scenario, image_path, validation_path)
        script_path.write_text(script, encoding="utf-8")
        state_path.write_text(
            json.dumps(
                scenario.state.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        static_path.write_text(
            json.dumps(scenario.static_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        index.append(
            {
                "name": scenario.name,
                "title": scenario.title,
                "note": scenario.note,
                "candidate_document": candidate_document_name(
                    scenario.state,
                    run_id=scenario.name,
                    attempt_id=1,
                ),
                "state_id": scenario.state.state_id,
                "state_version": scenario.state.state_version,
                "module_ids": [module.id for module in scenario.state.placed_modules],
                "module_types": [module.type for module in scenario.state.placed_modules],
                "target_ids": scenario.target_ids,
                "context_ids": scenario.context_ids,
                "target_parameters": {
                    module.id: module.params
                    for module in scenario.state.placed_modules
                    if module.id in scenario.target_ids
                },
                "static_step_statuses": [
                    record["static_step_status"] for record in scenario.static_records
                ],
                "png": str(image_path),
                "capture_method": "freecad_mcp.execute_code + FreeCADGui.activeView().saveImage",
                "freecad_validation": str(validation_path),
                "static_validation": str(static_path),
                "state": str(state_path),
                "script": str(script_path),
                "keep_open": scenario.keep_open,
            }
        )
    return index


def main() -> None:
    scenarios = [
        *route_scenarios(),
        *transition_scenarios(),
        *junction_scenarios(),
        *connect_scenarios(),
        *terminate_scenarios(),
        *inline_scenarios(),
        *sequential_assembly(),
    ]
    index = write_scenarios(scenarios)
    (OUT / "freecad_experiment_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"scenario_count": len(index), "names": [x["name"] for x in index]}, indent=2))


if __name__ == "__main__":
    main()
