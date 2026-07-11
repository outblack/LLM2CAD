from __future__ import annotations

import hashlib
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent
INDEX_PATH = OUT / "freecad_experiment_index.json"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def failure_reasons(validation: dict) -> list[str]:
    checks = validation.get("checks") or {}
    reasons: list[str] = []
    assembly = checks.get("assembly") or {}
    reasons.extend(str(value) for value in assembly.get("bop_errors") or [])
    for module_id, module_check in (checks.get("modules") or {}).items():
        for value in module_check.get("bop_errors") or []:
            reasons.append(f"{module_id}: {value}")
    for key in (
        "module_errors",
        "assembly_errors",
        "non_adjacent_overlaps",
        "connection_failures",
        "terminal_bore_failures",
        "anchored_inlet_bore_failures",
        "termination_seal_failures",
        "wall_section_failures",
    ):
        values = checks.get(key) or []
        if values:
            reasons.append(f"{key}: {json.dumps(values, ensure_ascii=False)}")
    result: list[str] = []
    for reason in reasons:
        compact = " ".join(reason.split())
        if compact not in result:
            result.append(compact)
    return result


def gallery_role(name: str, passed: bool) -> str:
    if not passed:
        return "strict_rejection_case"
    if name.startswith("assembly_step_"):
        return "sequential_example"
    if name.startswith("connection_overview"):
        return "connection_overview"
    if "subtype" in name:
        return "inline_subtype_gallery"
    if "default" in name:
        return "default_gallery"
    if "changed" in name:
        return "changed_gallery"
    return "supporting_gallery"


def main() -> None:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    passed_count = 0
    failed_count = 0
    for item in index:
        validation_path = Path(item["freecad_validation"])
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        passed = bool(validation.get("passed"))
        passed_count += int(passed)
        failed_count += int(not passed)
        png_path = Path(item["png"])
        checks = validation.get("checks") or {}
        module_checks = checks.get("modules") or {}
        item["mcp_execution_status"] = "executed"
        item["freecad_result"] = {
            "passed": passed,
            "schema_version": validation.get("schema_version"),
            "generator_version": validation.get("generator_version"),
            "freecad_version": validation.get("freecad_version"),
            "payload_digest": validation.get("payload_digest"),
            "candidate_shape_fingerprints": validation.get(
                "candidate_shape_fingerprints"
            ),
            "assembly_passed": bool((checks.get("assembly") or {}).get("passed")),
            "assembly_solid_count": (checks.get("assembly") or {}).get("solid_count"),
            "all_modules_passed": bool(module_checks) and all(
                bool(value.get("passed")) for value in module_checks.values()
            ),
            "outer_network_passed": bool(
                (checks.get("outer_network") or {}).get("passed")
            ),
            "bore_network_passed": bool(
                (checks.get("bore_network") or {}).get("passed")
            ),
            "wall_section_failure_count": len(
                checks.get("wall_section_failures") or []
            ),
            "connection_failure_count": len(
                checks.get("connection_failures") or []
            ),
            "non_adjacent_overlap_count": len(
                checks.get("non_adjacent_overlaps") or []
            ),
            "failure_reasons": failure_reasons(validation),
        }
        item["gallery_role"] = gallery_role(item["name"], passed)
        item["recommended_for_report"] = passed
        item["image"] = {
            "width": 1200,
            "height": 800,
            "background": "white",
            "black_bar_postprocess": "FreeCAD PySide QImage crop + white pad",
            "bytes": png_path.stat().st_size,
            "sha256": sha256(png_path),
        }
        get_view_path = OUT / f"{item['name']}.get_view_raw.png"
        if get_view_path.exists():
            item["additional_mcp_capture"] = {
                "capture_method": "freecad_mcp.get_view",
                "view": "Isometric",
                "focus_object": "PipeAssembly",
                "path": str(get_view_path),
                "bytes": get_view_path.stat().st_size,
                "sha256": sha256(get_view_path),
            }
    INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "scenario_count": len(index),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "strict_rejection_cases": [
            {
                "name": item["name"],
                "failure_reasons": item["freecad_result"]["failure_reasons"],
            }
            for item in index
            if not item["freecad_result"]["passed"]
        ],
        "all_sequential_steps_passed": all(
            item["freecad_result"]["passed"]
            for item in index
            if item["name"].startswith("assembly_step_")
        ),
        "connection_overview_passed": next(
            item["freecad_result"]["passed"]
            for item in index
            if item["name"] == "connection_overview_route_to_coupling"
        ),
        "capture_methods": [
            "freecad_mcp.execute_code + FreeCADGui.activeView().saveImage",
            "freecad_mcp.get_view (final assembly and connection overview raw evidence)",
        ],
    }
    (OUT / "freecad_experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
