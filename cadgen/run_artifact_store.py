"""실행 중 생성되는 JSONㆍ스크립트ㆍ검증 파일을 원자적으로 저장한다.

파이프라인은 설계와 검증 순서만 결정하고, 이 모듈은 경로 규칙과 파일 교체,
진행 스냅샷, 아티팩트 가용성 판정을 담당한다. 임시 파일을 ``fsync``한 뒤
교체하므로 중단된 프로세스가 반쪽 JSON을 정상 체크포인트로 남기지 않는다.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from cadgen.typed_data_models import (
    ActionAttempt,
    ArtifactStatus,
    CriticReport,
    GenerationArtifacts,
    PipeState,
    StaticIssue,
    StepVerification,
)
from cadgen.static_geometry_validator import top_issue_ids


# 보고서의 아티팩트 이름과 ``GenerationArtifacts`` 필드 연결을 한곳에 둔다.
# 순서는 기존 보고서 출력 순서를 그대로 보존한다.
_ARTIFACT_FIELDS: tuple[tuple[str, str], ...] = (
    ("prompt", "prompt_path"),
    ("source_measurement_contract", "source_measurement_contract_path"),
    ("intent", "intent_path"),
    ("intent_attempts", "intent_attempts_path"),
    ("intent_diagnostics", "intent_diagnostics_path"),
    ("actions", "actions_path"),
    ("action_attempts", "action_attempts_path"),
    ("repair_advice", "repair_advice_path"),
    ("diagnostics_index", "diagnostics_index_path"),
    ("constraint_ledger", "constraint_ledger_path"),
    ("global_preflight", "global_preflight_path"),
    ("search_events", "search_events_path"),
    ("state", "state_path"),
    ("step_verification", "step_verification_path"),
    ("critic_report", "critic_report_path"),
    ("freecad_script", "freecad_script_path"),
    ("checkpoint", "checkpoint_path"),
    ("run_report", "report_path"),
    ("mcp_result", "mcp_result_path"),
    ("freecad_validation", "freecad_validation_path"),
    ("freecad_document", "freecad_document_path"),
)


def _artifact_paths(run_dir: Path) -> dict[str, Path]:
    """한 실행 디렉터리 안의 표준 아티팩트 경로를 반환한다."""

    return {
        "prompt": run_dir / "prompt.txt",
        "source_measurement_contract": run_dir / "source_measurement_contract.json",
        "intent": run_dir / "intent.json",
        "intent_attempts": run_dir / "intent_attempts.json",
        "intent_diagnostics": run_dir / "intent_diagnostics.json",
        "constraint_ledger": run_dir / "constraint_ledger.json",
        "global_preflight": run_dir / "global_preflight.json",
        "search_events": run_dir / "search_events.jsonl",
        "actions": run_dir / "actions.json",
        "attempts": run_dir / "action_attempts.json",
        "repair_advice": run_dir / "repair_advice.json",
        "diagnostics_dir": run_dir / "diagnostics",
        "diagnostics_index": run_dir / "diagnostics" / "index.json",
        "state": run_dir / "state.json",
        "steps": run_dir / "step_verification.json",
        "critic": run_dir / "critic_report.json",
        "script": run_dir / "freecad_script.py",
        "report": run_dir / "run_report.json",
        "checkpoint": run_dir / "checkpoint.json",
        "mcp_result": run_dir / "mcp_result.json",
        "freecad_validation": run_dir / "freecad_validation.json",
    }


def _write_progress(
    paths: dict[str, Path],
    actions: list[dict[str, Any]],
    attempts: list[ActionAttempt],
    state: PipeState | None,
    step_verifications: list[StepVerification],
    critic: CriticReport | None,
) -> None:
    """현재까지 확정된 행동ㆍ상태ㆍ검증 결과를 재시작 가능한 형태로 쓴다."""

    _atomic_write_json(paths["actions"], actions)
    _atomic_write_json(
        paths["attempts"],
        [item.model_dump(mode="json") for item in attempts],
    )
    if state is not None:
        _atomic_write_json(paths["state"], state.model_dump(mode="json"))
    _atomic_write_json(
        paths["steps"],
        [item.model_dump(mode="json") for item in step_verifications],
    )
    if critic is not None:
        _atomic_write_json(paths["critic"], critic.model_dump(mode="json"))


def _artifact_statuses(
    artifacts: GenerationArtifacts,
    *,
    failed_stage: str | None,
    issues: list[StaticIssue],
) -> list[ArtifactStatus]:
    """보고서에 기록할 파일별 생성 여부와 차단 원인을 계산한다."""

    blocking = top_issue_ids(issues)
    statuses: list[ArtifactStatus] = []
    for name, field_name in _ARTIFACT_FIELDS:
        path = getattr(artifacts, field_name)
        exists = bool(path) and (Path(path).exists() or name == "run_report")
        if name == "freecad_document" and exists:
            exists = _freecad_document_matches_manifest(artifacts, Path(path))
        statuses.append(
            ArtifactStatus(
                name=name,
                path=path,
                status="available" if exists else "unavailable",
                producer_stage=failed_stage or "complete",
                blocking_issue_ids=[] if exists else blocking,
                unavailable_reason=None if exists else "Artifact was not produced.",
            )
        )
    return statuses


def _freecad_document_matches_manifest(
    artifacts: GenerationArtifacts,
    document_path: Path,
) -> bool:
    """FreeCAD 문서가 현재 payload manifest에 결합된 최신 결과인지 확인한다."""

    manifest_path = Path(artifacts.output_dir) / "freecad_artifact.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = str(manifest.get("payload_digest", ""))
        return (
            manifest.get("fcstd_path") == str(document_path.resolve())
            and bool(digest)
            and document_path.name.endswith(digest[:12] + ".FCStd")
        )
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def _next_visual_review_path(run_dir: Path) -> Path:
    """resume 후에도 덮어쓰지 않는 다음 visual review 파일명을 계산한다."""

    review_dir = run_dir / "visual_review"
    latest_round = 0
    for path in review_dir.glob("round_*.json"):
        match = re.fullmatch(r"round_(\d+)\.json", path.name)
        if match:
            latest_round = max(latest_round, int(match.group(1)))
    return review_dir / f"round_{latest_round + 1}.json"


def _atomic_write_json(path: Path, payload: Any) -> None:
    """JSON을 UTF-8 임시 파일에 쓴 뒤 원자적으로 대상 경로와 교체한다."""

    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
    )


def _atomic_write_text(path: Path, payload: str) -> None:
    """텍스트를 flush/fsync한 임시 파일로 저장하고 최종 경로로 교체한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


__all__ = [
    "_artifact_paths",
    "_artifact_statuses",
    "_atomic_write_json",
    "_atomic_write_text",
    "_next_visual_review_path",
    "_write_progress",
]
