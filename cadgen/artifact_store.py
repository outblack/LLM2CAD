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

from cadgen.schemas import (
    ActionAttempt,
    ArtifactStatus,
    CriticReport,
    GenerationArtifacts,
    PipeState,
    StaticIssue,
    StepVerification,
)
from cadgen.static_validation import top_issue_ids


def _artifact_paths(run_dir: Path) -> dict[str, Path]:
    """한 실행 디렉터리 안의 표준 아티팩트 경로를 반환한다."""

    return {
        "prompt": run_dir / "prompt.txt",
        "intent": run_dir / "intent.json",
        "intent_attempts": run_dir / "intent_attempts.json",
        "actions": run_dir / "actions.json",
        "attempts": run_dir / "action_attempts.json",
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

    entries = {
        "prompt": artifacts.prompt_path,
        "intent": artifacts.intent_path,
        "intent_attempts": artifacts.intent_attempts_path,
        "actions": artifacts.actions_path,
        "action_attempts": artifacts.action_attempts_path,
        "state": artifacts.state_path,
        "step_verification": artifacts.step_verification_path,
        "critic_report": artifacts.critic_report_path,
        "freecad_script": artifacts.freecad_script_path,
        "checkpoint": artifacts.checkpoint_path,
        "run_report": artifacts.report_path,
        "mcp_result": artifacts.mcp_result_path,
        "freecad_validation": artifacts.freecad_validation_path,
        "freecad_document": artifacts.freecad_document_path,
    }
    blocking = top_issue_ids(issues)
    statuses: list[ArtifactStatus] = []
    for name, path in entries.items():
        exists = bool(path) and (Path(path).exists() or name == "run_report")
        if name == "freecad_document" and exists:
            manifest_path = Path(artifacts.output_dir) / "freecad_artifact.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                digest = str(manifest.get("payload_digest", ""))
                exists = (
                    manifest.get("fcstd_path") == str(Path(path).resolve())
                    and bool(digest)
                    and Path(path).name.endswith(digest[:12] + ".FCStd")
                )
            except (OSError, json.JSONDecodeError, AttributeError):
                exists = False
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


def _next_visual_review_path(run_dir: Path) -> Path:
    """resume 후에도 덮어쓰지 않는 다음 visual review 파일명을 계산한다."""

    review_dir = run_dir / "visual_review"
    round_numbers: list[int] = []
    for path in review_dir.glob("round_*.json"):
        match = re.fullmatch(r"round_(\d+)\.json", path.name)
        if match:
            round_numbers.append(int(match.group(1)))
    next_round = max(round_numbers, default=0) + 1
    return review_dir / f"round_{next_round}.json"


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
