"""pipeline run 디렉터리와 초기 journal을 준비한다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from cadgen.conflict_certificate import append_search_event
from cadgen.run_artifact_store import (
    _artifact_paths,
    _atomic_write_json,
    _atomic_write_text,
)
from cadgen.runtime_settings import Settings
from cadgen.thinking_progress_stream import ThinkingStream
from cadgen.typed_data_models import DiagnosticJournal, GenerationArtifacts


@dataclass(frozen=True)
class RunWorkspace:
    """한 pipeline 실행에 고정되는 경로와 시작 journal을 묶는다."""

    prompt: str
    run_id: str
    run_dir: Path
    paths: dict[str, Path]
    artifacts: GenerationArtifacts
    intent_attempts: list[dict[str, Any]]
    intent_diagnostics: list[dict[str, Any]]


def prepare_run_workspace(
    prompt: str,
    settings: Settings,
    *,
    resume_dir: Path | None,
    stream: ThinkingStream,
) -> RunWorkspace:
    """새 실행 또는 재개 실행의 경로와 journal을 준비한다."""

    if resume_dir is None and not prompt.strip():
        raise ValueError("Prompt must contain a non-whitespace pipe design request")

    if resume_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = settings.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = Path(resume_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise ValueError(f"Resume run directory does not exist: {run_dir}")
        prompt_path = run_dir / "prompt.txt"
        if not prompt_path.is_file():
            raise ValueError(f"Resume prompt artifact is missing: {prompt_path}")
        prompt = prompt_path.read_text(encoding="utf-8")
        run_id = run_dir.name

    paths = _artifact_paths(run_dir)
    append_search_event(
        paths["search_events"],
        {
            "event_type": "run_started" if resume_dir is None else "run_resumed",
            "run_id": run_id,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        },
    )
    stream.emit(
        f"Run directory: {run_dir}. If externally interrupted, resume with "
        f"--resume {run_dir}.",
        force=True,
    )
    initialize_run_journals(paths, prompt=prompt, is_resume=resume_dir is not None)

    return RunWorkspace(
        prompt=prompt,
        run_id=run_id,
        run_dir=run_dir,
        paths=paths,
        artifacts=new_generation_artifacts(run_id, run_dir, paths),
        intent_attempts=load_dict_journal(paths["intent_attempts"]),
        intent_diagnostics=load_dict_journal(paths["intent_diagnostics"]),
    )


def initialize_run_journals(
    paths: dict[str, Path],
    *,
    prompt: str,
    is_resume: bool,
) -> None:
    """신규 실행과 재개 실행에 필요한 최소 영속 파일을 보장한다."""

    if not is_resume:
        _atomic_write_text(paths["prompt"], prompt)
        _atomic_write_json(paths["intent_attempts"], [])
        _atomic_write_json(paths["intent_diagnostics"], [])
    elif not paths["intent_diagnostics"].is_file():
        _atomic_write_json(paths["intent_diagnostics"], [])

    if not paths["repair_advice"].is_file():
        _atomic_write_json(paths["repair_advice"], [])
    if not paths["diagnostics_index"].is_file():
        _atomic_write_json(
            paths["diagnostics_index"],
            DiagnosticJournal().model_dump(mode="json"),
        )


def new_generation_artifacts(
    run_id: str,
    run_dir: Path,
    paths: dict[str, Path],
) -> GenerationArtifacts:
    """표준 실행 경로를 보고서용 artifact manifest로 변환한다."""

    return GenerationArtifacts(
        run_id=run_id,
        output_dir=str(run_dir),
        prompt_path=str(paths["prompt"]),
        source_measurement_contract_path=str(
            paths["source_measurement_contract"]
        ),
        intent_path=str(paths["intent"]),
        intent_attempts_path=str(paths["intent_attempts"]),
        intent_diagnostics_path=str(paths["intent_diagnostics"]),
        actions_path=str(paths["actions"]),
        state_path=str(paths["state"]),
        freecad_script_path=str(paths["script"]),
        report_path=str(paths["report"]),
        step_verification_path=str(paths["steps"]),
        critic_report_path=str(paths["critic"]),
        mcp_result_path=None,
        action_attempts_path=str(paths["attempts"]),
        repair_advice_path=str(paths["repair_advice"]),
        diagnostics_index_path=str(paths["diagnostics_index"]),
        constraint_ledger_path=str(paths["constraint_ledger"]),
        global_preflight_path=str(paths["global_preflight"]),
        search_events_path=str(paths["search_events"]),
        checkpoint_path=str(paths["checkpoint"]),
        freecad_validation_path=None,
        freecad_document_path=None,
    )


def load_dict_journal(path: Path) -> list[dict[str, Any]]:
    """손상되거나 비목록인 JSON journal을 안전한 빈 목록으로 읽는다."""

    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


__all__ = [
    "RunWorkspace",
    "initialize_run_journals",
    "load_dict_journal",
    "new_generation_artifacts",
    "prepare_run_workspace",
]
