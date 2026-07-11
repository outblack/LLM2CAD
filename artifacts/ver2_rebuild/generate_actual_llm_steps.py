from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cadgen.config import load_settings
from cadgen.freecad_script import build_freecad_script
from cadgen.registry import validate_action, validate_draft
from cadgen.schemas import ActionDraft, IntentResult
from cadgen.state import StateEngine
from cadgen.static_validation import build_step_verification


SOURCE_RUN = ROOT / "outputs" / "20260710T083734331241Z"
OUT = ROOT / "artifacts" / "ver2_rebuild"


def capture_suffix(step_index: int, image_path: Path, validation_path: Path) -> str:
    shown_ids = [f"M{i}" for i in range(1, step_index + 1)]
    target_id = f"M{step_index}"
    return f'''
import FreeCADGui as Gui
with open({str(validation_path)!r}, "w", encoding="utf-8") as handle:
    json.dump(validation, handle, ensure_ascii=False, indent=2, sort_keys=True)
for obj in doc.Objects:
    if hasattr(obj, "ViewObject"):
        obj.ViewObject.Visibility = False
palette = [
    (0.76, 0.80, 0.84),
    (0.34, 0.68, 0.88),
    (0.35, 0.78, 0.58),
    (0.93, 0.66, 0.25),
    (0.76, 0.45, 0.82),
]
for index, module_id in enumerate({shown_ids!r}):
    obj = doc.getObject("solid_" + module_id)
    if obj is None:
        continue
    obj.ViewObject.Visibility = True
    if module_id == {target_id!r}:
        obj.ViewObject.ShapeColor = (0.10, 0.54, 0.86)
        obj.ViewObject.LineColor = (0.05, 0.16, 0.25)
        obj.ViewObject.Transparency = 0
    else:
        obj.ViewObject.ShapeColor = palette[index % len(palette)]
        obj.ViewObject.LineColor = (0.20, 0.24, 0.28)
        obj.ViewObject.Transparency = 12
    obj.ViewObject.DisplayMode = "Flat Lines"
view = Gui.activeDocument().activeView()
view.setAnimationEnabled(False)
view.viewAxonometric()
view.fitAll()
view.saveImage({str(image_path)!r}, 1400, 900, "White")
print("CADGEN_SCREENSHOT=" + {str(image_path)!r})
'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    intent = IntentResult.model_validate_json((SOURCE_RUN / "intent.json").read_text())
    attempts = json.loads((SOURCE_RUN / "action_attempts.json").read_text())
    accepted = [item for item in attempts if item["status"] == "accepted"]

    settings = load_settings(ROOT / "missing.env").with_overrides(
        output_dir=OUT,
        skip_freecad=True,
    )
    engine = StateEngine(settings)
    state = engine.initial_state(intent)
    records: list[dict] = []

    for step_index, attempt in enumerate(accepted, start=1):
        draft = ActionDraft.model_validate(attempt["draft"])
        before = state
        draft_check = validate_draft(draft, before)
        action = engine.resolve_action(draft, before)
        action_check = validate_action(action, before)
        state = engine.apply_action(action, before)
        step_check = build_step_verification(before, action, state, intent, step_index)

        image_path = OUT / f"actual_llm_step_{step_index}.png"
        validation_path = OUT / f"actual_llm_step_{step_index}.freecad_validation.json"
        script_path = OUT / f"actual_llm_step_{step_index}.freecad.py"
        state_path = OUT / f"actual_llm_step_{step_index}.state.json"
        decision_path = OUT / f"actual_llm_step_{step_index}.decision.json"

        script = build_freecad_script(
            state,
            run_id=f"ver2_actual_llm_step_{step_index}",
            attempt_id=1,
        ) + capture_suffix(step_index, image_path, validation_path)
        script_path.write_text(script, encoding="utf-8")
        state_path.write_text(
            json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        record = {
            "source_run": SOURCE_RUN.name,
            "step_index": step_index,
            "state_before": before.state_id,
            "state_after": state.state_id,
            "llm_draft": draft.model_dump(mode="json"),
            "resolved_action": action.model_dump(mode="json"),
            "llm_rationale": attempt["draft"].get("rationale"),
            "draft_valid": draft_check.valid,
            "draft_errors": draft_check.errors,
            "resolved_action_valid": action_check.valid,
            "resolved_action_errors": action_check.errors,
            "step_status": step_check.status,
            "step_issues": [issue.model_dump(mode="json") for issue in step_check.issues],
            "open_ports_before": list(before.open_port_ids),
            "open_ports_after": list(state.open_port_ids),
        }
        decision_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        records.append(record)

    (OUT / "actual_llm_steps_index.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"steps": len(records), "source_run": SOURCE_RUN.name}))


if __name__ == "__main__":
    main()
