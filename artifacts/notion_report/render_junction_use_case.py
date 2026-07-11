from __future__ import annotations

import asyncio
import json
from pathlib import Path

from cadgen.config import load_settings
from cadgen.freecad_mcp import (
    FreeCADValidationError,
    assess_freecad_validation,
    capture_freecad_views,
    execute_freecad_code,
)
from cadgen.freecad_script import (
    GENERATOR_VERSION,
    build_freecad_candidate_cleanup_script,
    build_freecad_script,
    candidate_document_name,
    geometry_payload_digest,
)
from cadgen.registry import validate_action, validate_draft
from cadgen.schemas import ActionDraft, GlobalSpec, Goal, IntentResult
from cadgen.state import StateEngine
from cadgen.static_validation import build_final_critic_report, build_step_verification


OUT = Path(__file__).resolve().parent / "junction_connected_use_case"
RUN_ID = "junction_connected_use_case"
ATTEMPT_ID = 1


def build_state():
    settings = load_settings(Path("missing.env")).with_overrides(skip_freecad=True)
    engine = StateEngine(settings)
    intent = IntentResult(
        global_spec=GlobalSpec(outer_diameter=20.0, wall_thickness=2.0),
        start_position=(0.0, 0.0, 0.0),
        start_axis=(1.0, 0.0, 0.0),
        target_behavior=[
            Goal(goal_id="G1", type="move", direction="+X", length=70.0),
            Goal(
                goal_id="G2",
                type="branch",
                branch_count=1,
                include_primary_outlet=True,
                required_outlet_vectors=[(0.0, 1.0, 0.0)],
                junction_style="smooth_hub",
                blend_radius=2.0,
                inner_blend_radius=1.0,
                max_hub_radius=14.0,
                depends_on_goal_ids=["G1"],
            ),
            Goal(
                goal_id="G3",
                type="move",
                direction="+X",
                length=60.0,
                depends_on_goal_ids=["G2"],
            ),
            Goal(
                goal_id="G4",
                type="move",
                direction="+Y",
                length=55.0,
                depends_on_goal_ids=["G2"],
                allow_parallel=True,
            ),
        ],
        expected_open_ports=2,
        expected_open_ports_source="explicit",
        contract_digest="junction-connected-use-case",
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
                "length": 70.0,
                "direction": (1.0, 0.0, 0.0),
            },
        ),
        ActionDraft(
            target_port="M1.out",
            module="junction",
            catalog_schema_version=2,
            affected_goal_ids=["G2"],
            completed_goal_ids=["G2"],
            params={
                "section_source": "inherit_target",
                "outlets": [
                    {
                        "role": "primary",
                        "axis": (1.0, 0.0, 0.0),
                        "length": 35.0,
                        "outer_diameter": 20.0,
                        "wall_thickness": 2.0,
                    },
                    {
                        "role": "branch",
                        "axis": (0.0, 1.0, 0.0),
                        "length": 35.0,
                        "outer_diameter": 20.0,
                        "wall_thickness": 2.0,
                    },
                ],
                "blend_mode": "fillet",
                "blend_radius": 2.0,
                "inner_blend_radius": 1.0,
                "max_hub_radius": 14.0,
            },
        ),
        ActionDraft(
            target_port="M2.out",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["G3"],
            completed_goal_ids=["G3"],
            params={
                "path_kind": "line",
                "section_source": "inherit_target",
                "length": 60.0,
                "direction": (1.0, 0.0, 0.0),
            },
        ),
        ActionDraft(
            target_port="M2.out_1",
            module="route",
            catalog_schema_version=2,
            affected_goal_ids=["G4"],
            completed_goal_ids=["G4"],
            params={
                "path_kind": "line",
                "section_source": "inherit_target",
                "length": 55.0,
                "direction": (0.0, 1.0, 0.0),
            },
        ),
    ]

    state = engine.initial_state(intent)
    steps = []
    journal = []
    for index, draft in enumerate(drafts, start=1):
        draft_check = validate_draft(draft, state)
        if not draft_check.valid:
            raise ValueError(draft_check.errors)
        action = engine.resolve_action(draft, state)
        action_check = validate_action(action, state)
        if not action_check.valid:
            raise ValueError(action_check.errors)
        candidate = engine.apply_action(action, state)
        step = build_step_verification(state, action, candidate, intent, index)
        if step.status != "passed":
            raise ValueError([issue.issue_code for issue in step.issues])
        journal.append(step.model_dump(mode="json"))
        steps.append(step)
        state = candidate
    critic = build_final_critic_report(intent, state, steps)
    if not critic.passed:
        raise ValueError([issue.issue_code for issue in critic.issues])
    return state, journal, critic.model_dump(mode="json")


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state, journal, critic = build_state()
    live_settings = load_settings()
    script = build_freecad_script(
        state,
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        modeling_tolerance=live_settings.modeling_tolerance,
    )
    script_path = OUT / "junction_connected_use_case.freecad.py"
    script_path.write_text(script, encoding="utf-8")
    (OUT / "state.json").write_text(
        json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "static_validation.json").write_text(
        json.dumps({"steps": journal, "critic": critic}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    document = candidate_document_name(state, run_id=RUN_ID, attempt_id=ATTEMPT_ID)
    digest = geometry_payload_digest(state)
    cleanup = build_freecad_candidate_cleanup_script(
        state, run_id=RUN_ID, attempt_id=ATTEMPT_ID
    )
    try:
        raw = await execute_freecad_code(live_settings, script)
        try:
            evidence = assess_freecad_validation(
                raw,
                expected_digest=digest,
                expected_state_id=state.state_id,
                expected_module_ids=[module.id for module in state.placed_modules],
                expected_open_port_count=len(state.open_ports),
                expected_anchored_inlet_count=1,
                expected_generator_version=GENERATOR_VERSION,
                expected_run_id=RUN_ID,
                expected_state_version=state.state_version,
                expected_attempt_id=ATTEMPT_ID,
                expected_candidate_document=document,
            )
        except FreeCADValidationError as exc:
            (OUT / "freecad_validation.json").write_text(
                json.dumps(exc.evidence, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raise
        (OUT / "freecad_validation.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        colored_setup = f'''import FreeCAD as App
import FreeCADGui as Gui
doc = App.getDocument({document!r})
App.setActiveDocument(doc.Name)
assembly = doc.getObject("PipeAssembly")
assembly.ViewObject.Visibility = False
palette = {{
    "M1": (0.72, 0.76, 0.80),
    "M2": (0.10, 0.54, 0.86),
    "M3": (0.31, 0.72, 0.50),
    "M4": (0.93, 0.62, 0.20),
}}
for module_id, color in palette.items():
    obj = doc.getObject("solid_" + module_id)
    obj.ViewObject.Visibility = True
    obj.ViewObject.ShapeColor = color
    obj.ViewObject.LineColor = (0.10, 0.14, 0.18)
    obj.ViewObject.DisplayMode = "Flat Lines"
Gui.activeDocument().activeView().viewAxonometric()
Gui.activeDocument().activeView().fitAll()
'''
        await execute_freecad_code(live_settings, colored_setup)
        colored_paths = await capture_freecad_views(
            live_settings,
            OUT / "colored_modules",
            document_name=document,
            payload_digest=digest,
            views=("Isometric",),
            width=1200,
            height=800,
        )

        fused_setup = f'''import FreeCAD as App
import FreeCADGui as Gui
doc = App.getDocument({document!r})
App.setActiveDocument(doc.Name)
for obj in doc.Objects:
    if hasattr(obj, "Shape") and not obj.Shape.isNull():
        obj.ViewObject.Visibility = obj.Name == "PipeAssembly"
assembly = doc.getObject("PipeAssembly")
assembly.ViewObject.ShapeColor = (0.10, 0.54, 0.86)
assembly.ViewObject.LineColor = (0.06, 0.14, 0.20)
assembly.ViewObject.DisplayMode = "Flat Lines"
Gui.activeDocument().activeView().viewAxonometric()
Gui.activeDocument().activeView().fitAll()
'''
        await execute_freecad_code(live_settings, fused_setup)
        fused_paths = await capture_freecad_views(
            live_settings,
            OUT / "fused_assembly",
            document_name=document,
            payload_digest=digest,
            focus_object="PipeAssembly",
            views=("Isometric",),
            width=1200,
            height=800,
        )
        print(
            json.dumps(
                {
                    "passed": True,
                    "state_id": state.state_id,
                    "open_ports": state.open_port_ids,
                    "colored": colored_paths,
                    "fused": fused_paths,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        await execute_freecad_code(live_settings, cleanup)


if __name__ == "__main__":
    asyncio.run(main())
