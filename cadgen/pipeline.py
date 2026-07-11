"""LLM 계획부터 검증ㆍFreeCAD 게시까지 전체 생성 트랜잭션을 조정한다.

사용자 prompt와 ``Settings``를 입력받아 checkpoint, CAD artifact와 ``RunReport``를 만든다.
어느 경계에서든 검증되지 않은 응답이나 형상을 기본값으로 commit하지 않는다.
"""

from __future__ import annotations

import asyncio
import base64
import copy
from collections import Counter
import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cadgen.artifact_store import (
    _artifact_paths,
    _artifact_statuses,
    _atomic_write_json,
    _atomic_write_text,
    _next_visual_review_path,
    _write_progress,
)
from cadgen.config import Settings
from cadgen.conflict_kernel import (
    append_search_event,
    candidate_digest,
    duplicate_candidate_certificate,
    issue_certificate,
    pipe_state_digest,
    rejected_candidate_match,
)
from cadgen.contract_core import (
    preflight_and_realize_intent,
    structural_intent_issues,
)
from cadgen.diagnostics import (
    DiagnosticValidationError,
    bind_advisor_response,
    bind_diagnosis,
    build_diagnostic_case,
    diagnostic_case_digest,
    diagnostic_case_id,
    planner_directive_from_diagnosis,
    should_call_advisor,
    validate_diagnosis,
)
from cadgen.freecad_app import FreeCADLaunchError, ensure_freecad_open
from cadgen.freecad_mcp import (
    FreeCADMCPError,
    FreeCADValidationError,
    assess_freecad_publish,
    assess_freecad_validation,
    capture_freecad_views,
    execute_freecad_code,
    probe_freecad_mcp,
    probe_freecad_visual,
)
from cadgen.freecad_script import (
    GENERATOR_VERSION,
    _build_freecad_candidate_cleanup_script,
    _build_freecad_publish_script,
    _candidate_document_name,
    anchored_inlet_count,
    build_freecad_script,
    candidate_document_name,
    geometry_payload_digest,
    published_document_name,
)
from cadgen.geometry_policy import (
    minimum_spline_curvature_radius,
    predict_c1_spline,
)
from cadgen.gemini_client import (
    GeminiBudgetError,
    GeminiClient,
    GeminiConfigError,
    GeminiInvalidRequestError,
    GeminiLineageError,
    GeminiRequestError,
    HostContractValidationError,
    MAX_STRUCTURED_NUMBER_LITERAL_BYTES,
    MAX_STRUCTURED_NUMBER_LITERALS,
    StructuredOutputError,
    StructuredOutputIncompleteError,
    _strict_json_loads,
)
from cadgen.local_heuristic import infer_intent, plan_next_action
from cadgen.prompts import (
    compact_planner_payload,
    compact_visual_module_map,
    final_repair_prompt,
    intent_repair_advisor_prompt,
    intent_repair_advisor_system_instruction,
    intent_repair_reviewer_prompt,
    intent_repair_reviewer_system_instruction,
    intent_prompt,
    intent_system_instruction,
    step_lineage_repair_prompt,
    step_repair_advisor_prompt,
    step_repair_advisor_system_instruction,
    step_planner_prompt,
    step_planner_system_instruction,
    realized_terminal_topology,
)
from cadgen.primitive_compiler import compile_next_action
from cadgen.registry import (
    SUPPORTED_INLINE_COMPONENTS,
    filter_draft_params,
    validate_action,
    validate_draft,
)
from cadgen.schemas import (
    ActionAttempt,
    ActionDraft,
    AgendaRepairDirective,
    AgendaRepairDirectiveWire,
    AssemblyBounds,
    CriticReport,
    DiagnosticJournal,
    DiagnosticRecordRef,
    CorePlannerDecision,
    CorePlannerDecisionWire,
    GenerationArtifacts,
    GeometryValidationAdvisorResponse,
    Goal,
    IntentRepairAdvice,
    IntentRepairAdviceWire,
    IntentResult,
    LLMIntentJSONEnvelope,
    LLMProductionIntent,
    LLMUsage,
    PipeState,
    PlannerDecision,
    PlannerDecisionWire,
    ProductionIntent,
    ResolvedAction,
    RunReport,
    StaticIssue,
    StepRepairAdvice,
    StepRepairAdviceWire,
    StepRepairDiagnosis,
    StepRepairDiagnosisBody,
    StepVerification,
    VisualCriticResult,
    VisualCriticResultWire,
)
from cadgen.state import StateEngine
from cadgen.static_validation import (
    CriticValidationError,
    StaticValidationError,
    build_final_critic_report,
    build_step_verification,
    error_count,
    has_errors,
    top_issue_ids,
    validate_step_mcp_evidence,
    warning_count,
)
from cadgen.stream import ThinkingStream
from cadgen.vector import (
    add,
    canonical_circular_arc_frame,
    choose_perpendicular_axis,
    direction_to_vector,
    dot,
    length,
    mul,
    normalize,
    rotate,
    sub,
    vec,
)


@dataclass
class _PreservedSuffix:
    """Accepted plan branch retained while a localized repair is replanned."""

    repair_start_step: int
    original_actions: list[ResolvedAction]
    original_drafts: list[ActionDraft]
    original_checkpoints: list[PipeState]
    repair_hint: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "repair_start_step": self.repair_start_step,
            "original_actions": [
                action.model_dump(mode="json") for action in self.original_actions
            ],
            "original_drafts": [
                draft.model_dump(mode="json") for draft in self.original_drafts
            ],
            "original_checkpoints": [
                state.model_dump(mode="json") for state in self.original_checkpoints
            ],
            "repair_hint": self.repair_hint,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "_PreservedSuffix | None":
        if payload in (None, {}):
            return None
        if not isinstance(payload, dict):
            raise ValueError("preserved_suffix must be an object")
        repair_start_step = payload.get("repair_start_step")
        if type(repair_start_step) is not int:
            raise ValueError("preserved_suffix repair_start_step must be an integer")
        actions = [
            ResolvedAction.model_validate(item)
            for item in payload.get("original_actions", [])
        ]
        drafts = [
            ActionDraft.model_validate(item)
            for item in payload.get("original_drafts", [])
        ]
        checkpoints = [
            PipeState.model_validate(item)
            for item in payload.get("original_checkpoints", [])
        ]
        if len(actions) != len(drafts) or len(checkpoints) != len(actions) + 1:
            raise ValueError("preserved_suffix journals are not aligned")
        if not (1 <= repair_start_step < len(actions)):
            raise ValueError("preserved_suffix has no reusable action after repair")
        for index, (action, checkpoint) in enumerate(
            zip(actions, checkpoints[1:]), start=1
        ):
            if (
                action.action_id != f"A{index}"
                or checkpoint.state_version != index
                or not checkpoint.action_history
                or checkpoint.action_history[-1] != action
            ):
                raise ValueError(
                    "preserved_suffix action/checkpoint history is invalid"
                )
        return cls(
            repair_start_step=repair_start_step,
            original_actions=actions,
            original_drafts=drafts,
            original_checkpoints=checkpoints,
            repair_hint=str(payload.get("repair_hint") or ""),
        )


@dataclass
class _SuffixReplayResult:
    state: PipeState
    actions: list[ResolvedAction]
    step_verifications: list[StepVerification]
    checkpoints: list[PipeState]
    attempts: list[ActionAttempt]
    rejoin_original_step: int
    reused_original_steps: list[int]


@dataclass
class _ResumeContext:
    intent: IntentResult
    state: PipeState
    actions: list[dict[str, Any]]
    attempts: list[ActionAttempt]
    step_verifications: list[StepVerification]
    checkpoints: list[PipeState]
    planner_lineage: dict[str, Any]
    planner_schema_profiles: dict[str, str]
    llm_usage: LLMUsage
    pending_repair_observations: list[dict[str, Any]]
    diagnostic_journal: DiagnosticJournal
    pending_draft: ActionDraft | None = None
    pending_draft_attempt_index: int | None = None
    preserved_suffix: _PreservedSuffix | None = None
    next_attempt_index: int = 1
    semantic_mcp_passed: bool = False
    mcp_used: bool = False
    mcp_error: str | None = None
    mcp_result_path: str | None = None
    freecad_validation_path: str | None = None
    freecad_document_path: str | None = None


class _FreeCADSemanticError(FreeCADMCPError):
    """Candidate geometry/evidence failed before the publish phase."""

    def __init__(self, message: str, evidence: dict[str, Any] | None = None):
        super().__init__(message)
        self.evidence = evidence or {}


class _PlannerSchemaCapacityError(ValueError):
    """The immutable state cannot fit the provider-safe planner schema."""


class _PlannerStagnationError(RuntimeError):
    """동일한 검증 실패가 전략 전환 뒤에도 반복되어 조기 중단되었음을 뜻한다."""


class PipelinePausedError(StaticValidationError):
    """A recoverable run was checkpointed instead of being declared failed."""

    paused = True

    def __init__(
        self,
        stage: str,
        artifact_path: str,
        issues: list[StaticIssue],
        resume_command: str,
    ) -> None:
        self.resume_command = resume_command
        super().__init__(stage, artifact_path, issues)
        self.args = (
            f"{stage} paused after bounded recovery. Report: {artifact_path}. "
            f"Resume: {resume_command}",
        )


class _IntentSafetyValidationError(ValueError):
    """A parsed intent failed one or more deterministic semantic checks."""

    def __init__(self, diagnostics: list[str]):
        self.diagnostics = list(diagnostics)
        super().__init__("; ".join(self.diagnostics))


class _IntentScopeValidationError(ValueError):
    """A parsed intent failed a pre-planning catalog/scope contract."""

    def __init__(self, issues: list[StaticIssue]):
        if not issues:
            raise ValueError("intent scope failure requires at least one issue")
        self.issues = [issue.model_copy(deep=True) for issue in issues]
        super().__init__(
            "; ".join(f"{issue.issue_code}: {issue.message}" for issue in issues)
        )


class _IntentAdvisorAuthorityError(ValueError):
    """An advisor response requested authority the current issue does not grant."""

    def __init__(
        self, code: str, message: str, *, details: dict[str, Any] | None = None
    ):
        self.code = code
        self.details = details or {}
        super().__init__(message)


class _IntentSemanticValidationExhausted(RuntimeError):
    """A bounded intent loop ended after a deterministic semantic rejection."""

    def __init__(
        self,
        validation_details: list[dict[str, Any]],
        cause: Exception,
        *,
        terminal_reason: str,
    ) -> None:
        self.validation_details = [dict(item) for item in validation_details]
        self.cause = cause
        self.terminal_reason = terminal_reason
        super().__init__(
            "intent semantic validation exhausted: "
            f"{terminal_reason}: {type(cause).__name__}: {cause}"
        )


@dataclass
class _IntentAdvisorOutcome:
    advice: IntentRepairAdvice | None
    call_count: int
    error: str | None
    attempts: list[dict[str, Any]]
    source: str
    fallback_used: bool


@dataclass(frozen=True)
class _RunWorkspace:
    """한 pipeline 실행에 고정되는 경로와 시작 journal을 묶는다."""

    prompt: str
    run_id: str
    run_dir: Path
    paths: dict[str, Path]
    artifacts: GenerationArtifacts
    intent_attempts: list[dict[str, Any]]
    intent_diagnostics: list[dict[str, Any]]


# 96개/512바이트 제한은 사용자가 작성한 필수 값을 지키는 강제 상한이다.
# 선택적인 구성 편의 값은 이보다 작게 유지해 provider grammar가 숫자 표기와
# 주변 schema를 컴파일할 여유를 남긴다.
PLANNER_PREFERRED_NUMBER_LITERALS = 64
PLANNER_PREFERRED_NUMBER_LITERAL_BYTES = 384
_PLANNER_SCHEMA_PROFILES = {"preferred", "mandatory", "encoded"}
_PLANNER_SCHEMA_PROFILE_ATTR = "_cadgen_step_planner_schema_profiles"
MAX_IDENTICAL_VALIDATOR_FAILURES = 6
MAX_IDENTICAL_INTENT_FAILURES = 3


def _prepare_run_workspace(
    prompt: str,
    settings: Settings,
    *,
    resume_dir: Path | None,
    stream: ThinkingStream,
) -> _RunWorkspace:
    """새 실행 또는 재개 실행의 디렉터리와 기본 journal을 준비한다."""

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
    _initialize_run_journals(paths, prompt=prompt, is_resume=resume_dir is not None)

    return _RunWorkspace(
        prompt=prompt,
        run_id=run_id,
        run_dir=run_dir,
        paths=paths,
        artifacts=_new_generation_artifacts(run_id, run_dir, paths),
        intent_attempts=_load_dict_journal(paths["intent_attempts"]),
        intent_diagnostics=_load_dict_journal(paths["intent_diagnostics"]),
    )


def _initialize_run_journals(
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


def _new_generation_artifacts(
    run_id: str,
    run_dir: Path,
    paths: dict[str, Path],
) -> GenerationArtifacts:
    """표준 실행 경로를 보고서용 아티팩트 manifest 모델로 변환한다."""

    return GenerationArtifacts(
        run_id=run_id,
        output_dir=str(run_dir),
        prompt_path=str(paths["prompt"]),
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


def _load_dict_journal(path: Path) -> list[dict[str, Any]]:
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


def run_pipeline(
    prompt: str,
    settings: Settings,
    *,
    dry_run: bool = False,
    stream: ThinkingStream | None = None,
    resume_dir: Path | None = None,
) -> RunReport:
    """새 실행 또는 checkpoint 재개를 끝까지 처리하고 감사 가능한 보고서를 반환한다."""

    stream = stream or ThinkingStream(settings.stream_thinking_summary)
    workspace = _prepare_run_workspace(
        prompt,
        settings,
        resume_dir=resume_dir,
        stream=stream,
    )
    prompt = workspace.prompt
    run_id = workspace.run_id
    run_dir = workspace.run_dir
    paths = workspace.paths
    artifacts = workspace.artifacts

    actions: list[dict[str, Any]] = []
    attempts: list[ActionAttempt] = []
    step_verifications: list[StepVerification] = []
    checkpoints: list[PipeState] = []
    critic: CriticReport | None = None
    state: PipeState | None = None
    freecad_opened = False
    mcp_used = False
    semantic_mcp_passed = False
    mcp_error: str | None = None
    visual_paths: list[str] = []
    visual_reviewed_digest: str | None = None
    pending_repair_observations: list[dict[str, Any]] = []
    diagnostic_journal = DiagnosticJournal()
    pending_draft: ActionDraft | None = None
    pending_draft_attempt_index: int | None = None
    preserved_suffix: _PreservedSuffix | None = None
    next_attempt_index = 1
    intent_attempts = workspace.intent_attempts
    intent_diagnostics = workspace.intent_diagnostics

    if (
        not dry_run
        and settings.visual_validation_mode == "final_required"
        and (not settings.freecad_mcp_enabled or not settings.freecad_capture_views)
    ):
        issue = _issue(
            0,
            "REQUIRED_VISUAL_VALIDATION_UNAVAILABLE",
            "Required visual validation needs FreeCAD MCP and screenshot capture enabled.",
            phase="mcp_preflight",
        )
        _fail_run(
            run_id,
            paths,
            artifacts,
            dry_run,
            freecad_opened,
            mcp_used,
            "Required visual validation is disabled by configuration.",
            actions,
            attempts,
            state,
            step_verifications,
            _critic_with_issue(None, issue),
            "mcp_preflight",
            "Stopped before any Gemini call.",
            None,
        )

    if not dry_run and settings.freecad_auto_open:
        try:
            freecad_opened = ensure_freecad_open(settings, stream)
        except FreeCADLaunchError as exc:
            mcp_error = str(exc)
            if settings.require_freecad_app:
                issue = _issue(
                    0,
                    "FREECAD_LAUNCH_FAILED",
                    "FreeCAD launch was required but failed.",
                    phase="freecad_launch",
                    actual={"error": str(exc)},
                )
                _fail_run(
                    run_id,
                    paths,
                    artifacts,
                    dry_run,
                    freecad_opened,
                    mcp_used,
                    mcp_error,
                    actions,
                    attempts,
                    state,
                    step_verifications,
                    _critic_with_issue(None, issue),
                    "freecad_launch",
                    "FreeCAD launch failed before planning.",
                    None,
                )
            stream.emit("FreeCAD launch failed; continuing as unverified.", force=True)

    if (
        not dry_run
        and settings.freecad_mcp_required
        and not settings.freecad_mcp_enabled
    ):
        issue = _issue(
            0,
            "REQUIRED_MCP_DISABLED",
            "FreeCAD MCP is required but disabled by configuration.",
            phase="mcp_preflight",
        )
        _fail_run(
            run_id,
            paths,
            artifacts,
            dry_run,
            freecad_opened,
            mcp_used,
            "FreeCAD MCP is disabled.",
            actions,
            attempts,
            state,
            step_verifications,
            _critic_with_issue(None, issue),
            "mcp_preflight",
            "Stopped before any Gemini call.",
            None,
        )

    if not dry_run and settings.freecad_mcp_enabled:
        stream.emit(
            "Checking FreeCAD MCP readiness before spending Gemini tokens.", force=True
        )
        try:
            asyncio.run(probe_freecad_mcp(settings, executor=execute_freecad_code))
            if settings.visual_validation_mode == "final_required":
                asyncio.run(
                    asyncio.wait_for(
                        probe_freecad_visual(settings),
                        timeout=settings.freecad_mcp_timeout_sec,
                    )
                )
            mcp_used = True
        except Exception as exc:
            mcp_error = str(exc)
            if (
                settings.freecad_mcp_required
                or settings.visual_validation_mode == "final_required"
            ):
                issue = _issue(
                    0,
                    "REQUIRED_MCP_PREFLIGHT_FAILED",
                    "Required FreeCAD MCP or visual readiness check failed.",
                    phase="mcp_preflight",
                    actual={"error": str(exc)},
                )
                _fail_run(
                    run_id,
                    paths,
                    artifacts,
                    dry_run,
                    freecad_opened,
                    mcp_used,
                    mcp_error,
                    actions,
                    attempts,
                    state,
                    step_verifications,
                    _critic_with_issue(None, issue),
                    "mcp_preflight",
                    "Stopped before any Gemini call.",
                    None,
                )
            settings = replace(
                settings,
                freecad_mcp_enabled=False,
                freecad_step_mcp_enabled=False,
                freecad_capture_views=False,
            )
            stream.emit(
                "FreeCAD MCP is unavailable; circuit breaker disabled repeated retries.",
                force=True,
            )

    gemini: GeminiClient | None = None
    if not dry_run:
        stream.emit("Creating Gemini client.", force=True)
        try:
            gemini = GeminiClient(settings)
        except Exception as exc:
            issue = _issue(
                0,
                "GEMINI_CLIENT_CONFIGURATION_FAILED",
                "Gemini client configuration is invalid.",
                phase="gemini_config",
                actual={"error": str(exc)},
            )
            _fail_run(
                run_id,
                paths,
                artifacts,
                dry_run,
                freecad_opened,
                mcp_used,
                mcp_error,
                actions,
                attempts,
                state,
                step_verifications,
                _critic_with_issue(None, issue),
                "gemini_config",
                "Stopped before any Gemini request.",
                None,
            )

    engine = StateEngine(settings)
    if resume_dir is not None:
        stream.emit("Loading and reconciling the durable checkpoint.", force=True)
        try:
            context = _load_resume_context(
                paths["checkpoint"],
                settings,
                engine,
                dry_run=dry_run,
                run_dir=run_dir,
                expected_run_id=run_id,
            )
        except Exception as exc:
            issue = _issue(
                0,
                "CHECKPOINT_RESUME_FAILED",
                "The durable checkpoint could not be safely reconciled.",
                phase="checkpoint_resume",
                actual={"error": str(exc)},
            )
            _fail_run(
                run_id,
                paths,
                artifacts,
                dry_run,
                freecad_opened,
                mcp_used,
                str(exc),
                actions,
                attempts,
                state,
                step_verifications,
                _critic_with_issue(None, issue),
                "checkpoint_resume",
                "Resume stopped before planning because checkpoint reconciliation failed.",
                gemini,
            )
            raise AssertionError("unreachable")
        intent = context.intent
        state = context.state
        actions = context.actions
        attempts = context.attempts
        step_verifications = context.step_verifications
        checkpoints = context.checkpoints
        semantic_mcp_passed = context.semantic_mcp_passed
        mcp_used = context.mcp_used
        mcp_error = context.mcp_error
        pending_repair_observations = context.pending_repair_observations
        diagnostic_journal = context.diagnostic_journal
        pending_draft = context.pending_draft
        pending_draft_attempt_index = context.pending_draft_attempt_index
        preserved_suffix = context.preserved_suffix
        next_attempt_index = context.next_attempt_index
        artifacts = artifacts.model_copy(
            update={
                "mcp_result_path": context.mcp_result_path,
                "freecad_validation_path": context.freecad_validation_path,
                "freecad_document_path": context.freecad_document_path,
            }
        )
        if gemini is not None and hasattr(gemini, "restore_lineage"):
            gemini.restore_lineage(context.planner_lineage)
        if gemini is not None:
            for state_id, profile in context.planner_schema_profiles.items():
                _remember_planner_schema_profile(gemini, state_id, profile)
        if gemini is not None and hasattr(gemini, "restore_usage"):
            gemini.restore_usage(context.llm_usage)
        _atomic_write_json(paths["intent"], intent.model_dump(mode="json"))
        _atomic_write_json(
            paths["diagnostics_index"],
            diagnostic_journal.model_dump(mode="json"),
        )
        _write_progress(paths, actions, attempts, state, step_verifications, critic)
        _write_checkpoint(
            paths["checkpoint"],
            phase="COMMITTED",
            run_id=run_id,
            intent=intent,
            state=state,
            previous_state=checkpoints[-2] if len(checkpoints) > 1 else None,
            actions=actions,
            step_verifications=step_verifications,
            attempts=attempts,
            gemini=gemini,
            committed_states=checkpoints,
            freecad_verified=semantic_mcp_passed,
            pending_repair_observations=pending_repair_observations,
            pending_draft=pending_draft,
            pending_draft_attempt_index=pending_draft_attempt_index,
            preserved_suffix=preserved_suffix,
            next_attempt_index=next_attempt_index,
            diagnostic_journal=diagnostic_journal,
        )
    else:
        stream.emit(
            "Extracting the immutable contract and execution agenda.", force=True
        )
        try:
            intent = _extract_intent(
                prompt,
                settings,
                dry_run=dry_run,
                gemini=gemini,
                attempt_journal=intent_attempts,
                attempt_journal_path=paths["intent_attempts"],
                diagnostic_journal=intent_diagnostics,
                diagnostic_journal_path=paths["intent_diagnostics"],
                stream=stream,
            )
        except _IntentSafetyValidationError as exc:
            issue = _issue(
                0,
                "INTENT_SAFETY_CONTRACT",
                (
                    "Intent candidates exhausted the bounded repair loop without "
                    "satisfying deterministic semantic validation."
                ),
                phase="intent_semantic_validation",
                expected={"semantic_validation": "passed"},
                actual={"diagnostics": exc.diagnostics},
                suggestion={"operation": "review_intent_attempts_and_advisor_chain"},
            )
            _fail_run(
                run_id,
                paths,
                artifacts,
                dry_run,
                freecad_opened,
                mcp_used,
                mcp_error,
                actions,
                attempts,
                state,
                step_verifications,
                _critic_with_issue(None, issue),
                "intent_semantic_validation",
                "Stopped only after the bounded intent repair loop was exhausted.",
                gemini,
            )
            raise AssertionError("unreachable")
        except _IntentSemanticValidationExhausted as exc:
            semantic_critic: CriticReport | None = None
            for detail in reversed(exc.validation_details):
                semantic_issue = _issue(
                    0,
                    str(
                        detail.get("issue_code") or "INTENT_STRUCTURED_OR_HOST_CONTRACT"
                    ),
                    str(
                        detail.get("message")
                        or "Intent failed deterministic semantic validation."
                    ),
                    phase=str(detail.get("check_name") or "intent_semantic_validation"),
                    expected=(
                        dict(detail.get("expected"))
                        if isinstance(detail.get("expected"), dict)
                        else {}
                    ),
                    actual=(
                        dict(detail.get("actual"))
                        if isinstance(detail.get("actual"), dict)
                        else {"value": detail.get("actual")}
                    ),
                    suggestion=(
                        dict(detail.get("suggestion"))
                        if isinstance(detail.get("suggestion"), dict)
                        else {}
                    ),
                )
                semantic_critic = _critic_with_issue(
                    semantic_critic,
                    semantic_issue,
                )
            if semantic_critic is None:  # pragma: no cover - constructor guards it.
                raise AssertionError("semantic exhaustion lost its validator issue")
            _fail_run(
                run_id,
                paths,
                artifacts,
                dry_run,
                freecad_opened,
                mcp_used,
                mcp_error,
                actions,
                attempts,
                state,
                step_verifications,
                semantic_critic,
                "intent_semantic_validation",
                (
                    "Stopped only after the bounded intent repair loop ended: "
                    f"{exc.terminal_reason}."
                ),
                gemini,
            )
            raise AssertionError("unreachable")
        except _IntentScopeValidationError as exc:
            scope_critic: CriticReport | None = None
            for scope_issue in reversed(exc.issues):
                scope_critic = _critic_with_issue(scope_critic, scope_issue)
            if scope_critic is None:  # pragma: no cover - constructor forbids it.
                raise AssertionError("intent scope failure lost its issues")
            _fail_run(
                run_id,
                paths,
                artifacts,
                dry_run,
                freecad_opened,
                mcp_used,
                mcp_error,
                actions,
                attempts,
                state,
                step_verifications,
                scope_critic,
                "intent_scope",
                (
                    "Stopped before planning after bounded intent diagnosis and "
                    "repair could not produce a scope-valid immutable contract."
                ),
                gemini,
            )
            raise AssertionError("unreachable")
        except Exception as exc:
            issue = _issue(
                0,
                "INTENT_EXTRACTION_FAILED",
                "Gemini did not return a usable production intent.",
                phase="intent",
                actual={"error": str(exc)},
            )
            _fail_run(
                run_id,
                paths,
                artifacts,
                dry_run,
                freecad_opened,
                mcp_used,
                mcp_error,
                actions,
                attempts,
                state,
                step_verifications,
                _critic_with_issue(None, issue),
                "intent",
                "Stopped before CAD state initialization.",
                gemini,
            )
            raise AssertionError("unreachable")

        intent, constraint_ledger, global_preflight = preflight_and_realize_intent(
            prompt,
            intent,
            modeling_tolerance=settings.modeling_tolerance,
            feasibility_mode=settings.feasibility_mode,  # type: ignore[arg-type]
            max_uniform_scale=settings.max_uniform_centerline_scale,
        )
        _atomic_write_json(
            paths["constraint_ledger"],
            constraint_ledger.model_dump(mode="json"),
        )
        _atomic_write_json(
            paths["global_preflight"],
            global_preflight.model_dump(mode="json"),
        )
        append_search_event(
            paths["search_events"],
            {
                "event_type": "global_preflight_completed",
                "run_id": run_id,
                "status": global_preflight.status,
                "ledger_digest": constraint_ledger.ledger_digest,
                "authored_program_digest": global_preflight.authored_program_digest,
                "realized_program_digest": global_preflight.realized_program_digest,
                "deviation_count": len(global_preflight.deviations),
                "conflict_ids": [
                    item.certificate_id for item in global_preflight.conflicts
                ],
            },
        )
        stream.emit(
            "Global centerline preflight status="
            f"{global_preflight.status}, deviations="
            f"{len(global_preflight.deviations)}, conflicts="
            f"{len(global_preflight.conflicts)}.",
            force=True,
        )
        intent = _bind_contract(prompt, intent)
        _atomic_write_json(paths["intent"], intent.model_dump(mode="json"))
        state = engine.initial_state(intent)
        checkpoints = [state.model_copy(deep=True)]
        _write_progress(paths, actions, attempts, state, step_verifications, critic)
        _write_checkpoint(
            paths["checkpoint"],
            phase="COMMITTED",
            run_id=run_id,
            intent=intent,
            state=state,
            previous_state=None,
            actions=actions,
            step_verifications=step_verifications,
            attempts=attempts,
            gemini=gemini,
            committed_states=checkpoints,
            freecad_verified=False,
            diagnostic_journal=diagnostic_journal,
        )

    configured_action_limit = settings.max_iter
    action_limit_ceiling = (
        settings.max_iter
        if settings.max_iter_is_hard_limit
        else settings.max_iter_hard_ceiling
    )
    initial_required_actions = len(actions) + _exclusive_goal_action_lower_bound(state)
    effective_action_limit = min(
        action_limit_ceiling,
        max(configured_action_limit, initial_required_actions),
    )
    action_budget_policy: dict[str, Any] = {
        "configured_soft_baseline": configured_action_limit,
        "deterministic_required_actions": initial_required_actions,
        "effective_action_limit": effective_action_limit,
        "hard_ceiling": action_limit_ceiling,
        "explicit_hard_limit": settings.max_iter_is_hard_limit,
        "expanded": effective_action_limit > configured_action_limit,
        "expansion_history": [],
    }
    if effective_action_limit > configured_action_limit:
        action_budget_policy["expansion_history"].append(
            {
                "state_id": state.state_id,
                "accepted_actions": len(actions),
                "from": configured_action_limit,
                "to": effective_action_limit,
                "reason": "validated_goal_agenda_lower_bound",
            }
        )
    if gemini is not None:
        setattr(gemini, "_cadgen_action_budget_policy", action_budget_policy)
    if effective_action_limit != settings.max_iter:
        settings = replace(settings, max_iter=effective_action_limit)
        stream.emit(
            "Accepted-action budget expanded from "
            f"{configured_action_limit} to {effective_action_limit} from the "
            "validated goal agenda "
            f"(deterministic minimum {initial_required_actions}, hard ceiling "
            f"{action_limit_ceiling}).",
            force=True,
        )

    final_repair_round = 0
    causal_backjump_fingerprints: set[str] = set()
    while True:
        while state.remaining_goals:
            atomic_lower_bound = _exclusive_goal_action_lower_bound(state)
            required_total = len(actions) + atomic_lower_bound
            if (
                not settings.max_iter_is_hard_limit
                and required_total > settings.max_iter
                and settings.max_iter < action_limit_ceiling
            ):
                previous_limit = settings.max_iter
                effective_action_limit = min(action_limit_ceiling, required_total)
                settings = replace(settings, max_iter=effective_action_limit)
                action_budget_policy["effective_action_limit"] = effective_action_limit
                action_budget_policy["expanded"] = True
                action_budget_policy["expansion_history"].append(
                    {
                        "state_id": state.state_id,
                        "accepted_actions": len(actions),
                        "from": previous_limit,
                        "to": effective_action_limit,
                        "reason": "runtime_remaining_action_lower_bound",
                    }
                )
                stream.emit(
                    "Accepted-action budget expanded from "
                    f"{previous_limit} to {effective_action_limit} after a committed "
                    "state exposed a larger deterministic remaining-action lower "
                    f"bound (hard ceiling {action_limit_ceiling}).",
                    force=True,
                )
            remaining_action_budget = settings.max_iter - len(actions)
            if (
                not dry_run
                and remaining_action_budget > 0
                and atomic_lower_bound > remaining_action_budget
            ):
                issue = _issue(
                    len(actions) + 1,
                    "EXCLUSIVE_GOAL_BUDGET_INFEASIBLE",
                    "The non-double-countable goals cannot fit in the accepted-action budget.",
                    phase="planning_feasibility",
                    expected={
                        "minimum_required_actions": atomic_lower_bound,
                        "remaining_action_budget": remaining_action_budget,
                    },
                    actual={
                        "remaining_goal_ids": [
                            goal.goal_id for goal in state.remaining_goals
                        ]
                    },
                )
                critic = _critic_with_issue(
                    build_final_critic_report(intent, state, step_verifications),
                    issue,
                )
                _fail_run(
                    run_id,
                    paths,
                    artifacts,
                    dry_run,
                    freecad_opened,
                    mcp_used,
                    mcp_error,
                    actions,
                    attempts,
                    state,
                    step_verifications,
                    critic,
                    "planning_feasibility",
                    "Stopped before a step-planner call that cannot fit the action budget.",
                    gemini,
                )
            if len(actions) >= settings.max_iter:
                issue = _issue(
                    len(actions) + 1,
                    "MAX_ITER_REACHED",
                    "The accepted action ceiling was reached before all goals completed.",
                    phase="max_iter",
                    actual={
                        "max_iter": settings.max_iter,
                        "remaining_goal_ids": [
                            goal.goal_id for goal in state.remaining_goals
                        ],
                    },
                )
                critic = _critic_with_issue(
                    build_final_critic_report(intent, state, step_verifications), issue
                )
                _fail_run(
                    run_id,
                    paths,
                    artifacts,
                    dry_run,
                    freecad_opened,
                    mcp_used,
                    mcp_error,
                    actions,
                    attempts,
                    state,
                    step_verifications,
                    critic,
                    "max_iter",
                    "Accepted action ceiling reached.",
                    gemini,
                )

            step_index = len(actions) + 1
            # Verification evidence is state-version specific. Evidence for S(n)
            # must never make an unverified S(n+1) appear fully verified.
            previous_freecad_verified = semantic_mcp_passed
            semantic_mcp_passed = False
            stream.emit(
                f"Planning and validating state transition {step_index}.",
                force=True,
            )
            accepted: tuple[ResolvedAction, PipeState, StepVerification] | None = None
            observations = [dict(item) for item in pending_repair_observations]
            pending_repair_observations = []
            last_issue: StaticIssue | None = None
            last_step: StepVerification | None = None
            last_phase = "planning"
            terminal_step_summary: str | None = None

            def persist_rejected_attempt(
                repair_observations: list[dict[str, Any]],
                following_attempt_index: int,
            ) -> None:
                nonlocal pending_draft
                nonlocal pending_draft_attempt_index
                nonlocal next_attempt_index
                pending_draft = None
                pending_draft_attempt_index = None
                next_attempt_index = following_attempt_index
                _write_checkpoint(
                    paths["checkpoint"],
                    phase="COMMITTED",
                    run_id=run_id,
                    intent=intent,
                    state=state,
                    previous_state=(checkpoints[-2] if len(checkpoints) > 1 else None),
                    actions=actions,
                    step_verifications=step_verifications,
                    attempts=attempts,
                    gemini=gemini,
                    committed_states=checkpoints,
                    freecad_verified=previous_freecad_verified,
                    pending_repair_observations=repair_observations,
                    preserved_suffix=preserved_suffix,
                    next_attempt_index=following_attempt_index,
                    diagnostic_journal=diagnostic_journal,
                )
                # Checkpoint is the recovery authority; user-facing projections
                # follow it so a crash can at worst leave a stale journal, never
                # a journal that claims an uncommitted recovery state.
                _write_progress(
                    paths,
                    actions,
                    attempts,
                    state,
                    step_verifications,
                    critic,
                )
                if attempts:
                    latest_attempt = attempts[-1]
                    append_search_event(
                        paths["search_events"],
                        {
                            "event_type": "candidate_rejected",
                            "run_id": run_id,
                            "state_id": state.state_id,
                            "step_index": step_index,
                            "attempt_index": latest_attempt.attempt_index,
                            "phase": latest_attempt.phase,
                            "issue_codes": latest_attempt.issue_codes,
                            "candidate_digest": (
                                candidate_digest(latest_attempt.resolved)
                                if isinstance(latest_attempt.resolved, dict)
                                else None
                            ),
                            "conflict_certificate": (
                                issue_certificate(
                                    last_issue,
                                    candidate=(
                                        ResolvedAction.model_validate(
                                            latest_attempt.resolved
                                        )
                                        if isinstance(latest_attempt.resolved, dict)
                                        else None
                                    ),
                                ).model_dump(mode="json")
                                if last_issue is not None
                                else None
                            ),
                        },
                    )
                    stream.emit(
                        f"Step {step_index} attempt {latest_attempt.attempt_index} "
                        f"rejected in {latest_attempt.phase}: "
                        + ", ".join(latest_attempt.issue_codes[:4]),
                        force=True,
                    )

            def persist_diagnostic_state(
                repair_observations: list[dict[str, Any]],
                journal: DiagnosticJournal,
                in_flight_operation: str | None,
            ) -> None:
                """Make the advisor journal authoritative before any paid call."""

                nonlocal diagnostic_journal
                diagnostic_journal = journal
                _write_checkpoint(
                    paths["checkpoint"],
                    phase="COMMITTED",
                    run_id=run_id,
                    intent=intent,
                    state=state,
                    previous_state=(checkpoints[-2] if len(checkpoints) > 1 else None),
                    actions=actions,
                    step_verifications=step_verifications,
                    attempts=attempts,
                    gemini=gemini,
                    committed_states=checkpoints,
                    freecad_verified=previous_freecad_verified,
                    pending_repair_observations=repair_observations,
                    preserved_suffix=preserved_suffix,
                    next_attempt_index=next_attempt_index,
                    in_flight_operation=in_flight_operation,
                    diagnostic_journal=diagnostic_journal,
                )
                _atomic_write_json(
                    paths["diagnostics_index"],
                    diagnostic_journal.model_dump(mode="json"),
                )

            max_attempt_index = settings.step_repair_attempts + 1
            for attempt_index in range(
                next_attempt_index,
                max_attempt_index + 1,
            ):
                stream.emit(
                    f"Step {step_index} attempt {attempt_index}/"
                    f"{max_attempt_index} started.",
                    force=True,
                )
                draft: ActionDraft | None = None
                resolved: ResolvedAction | None = None
                speculative: PipeState | None = None
                planner_repair_context: list[dict[str, Any]] = []
                if pending_draft is None:
                    observations, diagnostic_journal = _run_step_geometry_diagnostician(
                        run_id=run_id,
                        run_dir=run_dir,
                        paths=paths,
                        state=state,
                        step_index=step_index,
                        observations=observations,
                        attempts=attempts,
                        settings=settings,
                        gemini=gemini,
                        journal=diagnostic_journal,
                        stream=stream,
                        persist=persist_diagnostic_state,
                    )
                    terminal_diagnosis = _terminal_geometry_diagnosis(observations)
                    if terminal_diagnosis is not None:
                        diagnostic_journal = diagnostic_journal.model_copy(
                            update={
                                "futile_retry_avoided_count": (
                                    diagnostic_journal.futile_retry_avoided_count + 1
                                )
                            }
                        )
                        persist_diagnostic_state(
                            observations,
                            diagnostic_journal,
                            None,
                        )
                        terminal_step_summary = terminal_diagnosis
                        break
                    try:
                        planner_repair_context = _planner_repair_context(
                            observations,
                            attempts,
                            step_index,
                        )
                    except _PlannerStagnationError:
                        # 새 모델 호출 전에 멈추고, 직전 validator의 정확한
                        # phase/issue를 최종 보고서에 그대로 보존한다.
                        terminal_step_summary = (
                            "The same deterministic validation failure remained "
                            "after a full-context strategy reset; further identical "
                            "LLM calls were stopped to protect the token budget."
                        )
                        break
                if observations or pending_draft is None:
                    _write_checkpoint(
                        paths["checkpoint"],
                        phase="COMMITTED",
                        run_id=run_id,
                        intent=intent,
                        state=state,
                        previous_state=(
                            checkpoints[-2] if len(checkpoints) > 1 else None
                        ),
                        actions=actions,
                        step_verifications=step_verifications,
                        attempts=attempts,
                        gemini=gemini,
                        committed_states=checkpoints,
                        freecad_verified=previous_freecad_verified,
                        pending_repair_observations=observations,
                        pending_draft=pending_draft,
                        pending_draft_attempt_index=pending_draft_attempt_index,
                        preserved_suffix=preserved_suffix,
                        next_attempt_index=next_attempt_index,
                        in_flight_operation=(
                            "step_planner" if pending_draft is None else None
                        ),
                        diagnostic_journal=diagnostic_journal,
                    )
                try:
                    if pending_draft is not None:
                        if pending_draft_attempt_index != attempt_index:
                            raise ValueError(
                                "Pending planner draft attempt does not match the retry journal"
                            )
                        draft = pending_draft
                    else:
                        draft = _plan_action(
                            state,
                            dry_run=dry_run,
                            gemini=gemini,
                            host_compiler_enabled=(settings.primitive_compiler_enabled),
                            repair_observations=planner_repair_context,
                            reusable_suffix_context=_suffix_rejoin_context(
                                preserved_suffix,
                                state,
                            ),
                        )
                        # The paid response is journaled before any validation or
                        # state mutation. Resume consumes this exact draft first.
                        pending_draft = draft
                        pending_draft_attempt_index = attempt_index
                        next_attempt_index = attempt_index
                        _write_checkpoint(
                            paths["checkpoint"],
                            phase="COMMITTED",
                            run_id=run_id,
                            intent=intent,
                            state=state,
                            previous_state=(
                                checkpoints[-2] if len(checkpoints) > 1 else None
                            ),
                            actions=actions,
                            step_verifications=step_verifications,
                            attempts=attempts,
                            gemini=gemini,
                            committed_states=checkpoints,
                            freecad_verified=previous_freecad_verified,
                            pending_repair_observations=observations,
                            pending_draft=pending_draft,
                            pending_draft_attempt_index=(pending_draft_attempt_index),
                            preserved_suffix=preserved_suffix,
                            next_attempt_index=next_attempt_index,
                            diagnostic_journal=diagnostic_journal,
                        )
                except Exception as exc:
                    last_phase = "planning"
                    lineage_reset = False
                    if (
                        isinstance(exc, StructuredOutputError)
                        and exc.part == "step_planner"
                        and gemini is not None
                        and hasattr(gemini, "reset_lineage")
                    ):
                        # A schema-invalid response is not a useful conversational
                        # state.  Drop it before checkpointing so the next retry
                        # receives the complete state/catalog instead of a minimal
                        # continuation that can anchor the same invalid structure.
                        gemini.reset_lineage("step_planner")
                        lineage_reset = True
                    last_issue = _issue(
                        step_index,
                        "PLANNING_FAILED",
                        "Planner did not return a usable next action.",
                        phase=last_phase,
                        actual={
                            "error_type": type(exc).__name__,
                            "diagnostic": (
                                _intent_repair_diagnostic(exc)
                                if isinstance(
                                    exc,
                                    (
                                        StructuredOutputError,
                                        HostContractValidationError,
                                    ),
                                )
                                else str(exc)[:1200]
                            ),
                            "planner_lineage_reset": lineage_reset,
                        },
                    )
                    observations = [_repair_observation(last_issue)]
                    attempts.append(
                        _attempt(
                            step_index,
                            attempt_index,
                            state,
                            last_phase,
                            "rejected",
                            draft,
                            resolved,
                            [last_issue],
                        )
                    )
                    persist_rejected_attempt(observations, attempt_index + 1)
                    if isinstance(
                        exc,
                        (
                            GeminiBudgetError,
                            GeminiConfigError,
                            GeminiRequestError,
                            _PlannerSchemaCapacityError,
                            _PlannerStagnationError,
                        ),
                    ):
                        if isinstance(exc, GeminiBudgetError):
                            terminal_step_summary = (
                                "Gemini global call/token budget stopped this step; "
                                "the local repair budget was not exhausted."
                            )
                        elif isinstance(exc, _PlannerStagnationError):
                            terminal_step_summary = (
                                "The same deterministic validation failure remained "
                                "after a full-context strategy reset; further identical "
                                "LLM calls were stopped to protect the token budget."
                            )
                        else:
                            terminal_step_summary = (
                                "Planner configuration or provider infrastructure "
                                "failed; additional local LLM repairs were not applicable."
                            )
                        break
                    continue

                # 프로덕션은 반환 타입이나 테스트 더블 구현과 무관하게 항상
                # schema-v2 경계를 통과해야 한다. 이 검사를 버전 조건으로
                # 건너뛰면 legacy resolver의 길이/각도 기본값이 실제 CAD 수치로
                # 들어갈 수 있으므로, 로컬 휴리스틱은 dry-run에서만 허용한다.
                if not dry_run:
                    draft_check = validate_draft(draft, state)
                    if not draft_check.valid:
                        last_phase = "draft_validation"
                        last_issue = _issue(
                            step_index,
                            "DRAFT_VALIDATION_FAILED",
                            "LLM action violates the authored schema-v2 contract.",
                            phase=last_phase,
                            target_port=draft.target_port,
                            actual={"errors": draft_check.errors},
                            suggestion={
                                "operation": "revise_action_draft",
                                "parameter_errors": draft_check.errors[:12],
                                "instruction": (
                                    "Return a complete schema-v2 action and change "
                                    "only the named primitive/parameter fields."
                                ),
                            },
                        )
                        observations = [_repair_observation(last_issue)]
                        attempts.append(
                            _attempt(
                                step_index,
                                attempt_index,
                                state,
                                last_phase,
                                "rejected",
                                draft,
                                resolved,
                                [last_issue],
                            )
                        )
                        persist_rejected_attempt(observations, attempt_index + 1)
                        continue

                try:
                    resolved = engine.resolve_action(draft, state)
                except Exception as exc:
                    last_phase = "action_resolution"
                    last_issue = _issue(
                        step_index,
                        "ACTION_RESOLUTION_FAILED",
                        "Planner action could not be resolved against the current state.",
                        phase=last_phase,
                        target_port=draft.target_port,
                        actual={"module": draft.module, "error": str(exc)},
                        suggestion={
                            "operation": "revise_target_or_parameters",
                            "instruction": (
                                "Choose an existing open target port and revise the "
                                "parameter named by the resolver error; do not use defaults."
                            ),
                        },
                    )
                    observations = [_repair_observation(last_issue)]
                    attempts.append(
                        _attempt(
                            step_index,
                            attempt_index,
                            state,
                            last_phase,
                            "rejected",
                            draft,
                            resolved,
                            [last_issue],
                        )
                    )
                    persist_rejected_attempt(observations, attempt_index + 1)
                    continue

                action_check = validate_action(resolved, state)
                if not action_check.valid:
                    last_phase = "registry_validation"
                    quantitative_diagnostics = [
                        item.model_dump(mode="json")
                        for item in action_check.diagnostics
                    ]
                    primary_diagnostic = (
                        quantitative_diagnostics[0]
                        if quantitative_diagnostics
                        else None
                    )
                    last_issue = _issue(
                        step_index,
                        "REGISTRY_VALIDATION_FAILED",
                        "Resolved action failed registry validation.",
                        phase=last_phase,
                        target_port=resolved.target_port,
                        action_id=resolved.action_id,
                        expected=(
                            {
                                "metric": primary_diagnostic["metric"],
                                "comparator": primary_diagnostic["comparator"],
                                "required": primary_diagnostic["required"],
                                "units": primary_diagnostic.get("units"),
                            }
                            if primary_diagnostic is not None
                            else {}
                        ),
                        actual={
                            "errors": action_check.errors,
                            "validation_diagnostics": quantitative_diagnostics,
                        },
                        suggestion={
                            "operation": "revise_resolved_geometry_inputs",
                            "parameter_errors": action_check.errors[:12],
                            "quantitative_constraints": quantitative_diagnostics,
                            "instruction": (
                                "Use the structured metric gap, critical location, "
                                "implicated paths, and prior trial response to select "
                                "new LLM-authored values. Change a path only when it "
                                "can influence the critical span; the system will not "
                                "patch or commit a recommended value."
                            ),
                        },
                    )
                    observations = [_repair_observation(last_issue)]
                    attempts.append(
                        _attempt(
                            step_index,
                            attempt_index,
                            state,
                            last_phase,
                            "rejected",
                            draft,
                            resolved,
                            [last_issue],
                        )
                    )
                    persist_rejected_attempt(observations, attempt_index + 1)
                    continue

                prior_rejection = (
                    rejected_candidate_match(
                        resolved,
                        attempts,
                        state_id=state.state_id,
                        state_digest=pipe_state_digest(state),
                    )
                    if settings.conflict_search_enabled
                    else None
                )
                if prior_rejection is not None:
                    certificate = duplicate_candidate_certificate(
                        resolved,
                        prior_rejection,
                    )
                    last_phase = "conflict_routing"
                    last_issue = _issue(
                        step_index,
                        "DUPLICATE_REJECTED_CANDIDATE",
                        "An identical state-bound geometry candidate was already rejected.",
                        phase=last_phase,
                        target_port=resolved.target_port,
                        action_id=resolved.action_id,
                        expected={
                            "progress_witness": (
                                "new primitive, changed causal field, changed prefix, or new evidence"
                            )
                        },
                        actual={
                            "candidate_digest": certificate.candidate_digest,
                            "prior_attempt": prior_rejection.attempt_index,
                            "prior_phase": prior_rejection.phase,
                            "conflict_certificate": certificate.model_dump(mode="json"),
                        },
                        suggestion={
                            "operation": "change_primitive_or_backjump",
                            "forbidden_candidate_digest": certificate.candidate_digest,
                            "allowed_routes": certificate.allowed_routes,
                            "instruction": (
                                "Do not replay this geometry. Change a causally relevant "
                                "primitive/variant or request a checkpoint backjump."
                            ),
                        },
                    )
                    observations = [_repair_observation(last_issue)]
                    attempts.append(
                        _attempt(
                            step_index,
                            attempt_index,
                            state,
                            last_phase,
                            "rejected",
                            draft,
                            resolved,
                            [last_issue],
                        )
                    )
                    persist_rejected_attempt(observations, attempt_index + 1)
                    continue

                try:
                    speculative = engine.apply_action(resolved, state)
                except Exception as exc:
                    last_phase = "state_application"
                    last_issue = _issue(
                        step_index,
                        "STATE_APPLICATION_FAILED",
                        "Resolved action could not be speculatively applied.",
                        phase=last_phase,
                        action_id=resolved.action_id,
                        target_port=resolved.target_port,
                        actual={"error": str(exc)},
                        suggestion={
                            "operation": "revise_state_transition_inputs",
                            "instruction": (
                                "Correct the target/geometry identified by the state "
                                "error and return a complete replacement action."
                            ),
                        },
                    )
                    observations = [_repair_observation(last_issue)]
                    attempts.append(
                        _attempt(
                            step_index,
                            attempt_index,
                            state,
                            last_phase,
                            "rejected",
                            draft,
                            resolved,
                            [last_issue],
                        )
                    )
                    persist_rejected_attempt(observations, attempt_index + 1)
                    continue

                step = build_step_verification(
                    state,
                    resolved,
                    speculative,
                    intent,
                    step_index,
                    mcp_required=(
                        settings.freecad_mcp_required
                        and settings.freecad_step_mcp_enabled
                    ),
                    skipped_mcp_reason=_step_mcp_skip_reason(settings, dry_run),
                )
                last_step = step
                if has_errors(step.issues):
                    last_phase = "static_step_validation"
                    last_issue = next(
                        issue for issue in step.issues if issue.severity == "error"
                    )
                    observations = [
                        _repair_observation(issue)
                        for issue in step.issues
                        if issue.severity == "error"
                    ]
                    attempts.append(
                        _attempt(
                            step_index,
                            attempt_index,
                            state,
                            last_phase,
                            "rejected",
                            draft,
                            resolved,
                            step.issues,
                        )
                    )
                    persist_rejected_attempt(observations, attempt_index + 1)
                    continue

                step_skip_reason = _step_mcp_skip_reason(settings, dry_run)
                run_step_mcp = (
                    _should_run_step_mcp(settings, dry_run)
                    or _requires_risk_mcp(
                        settings,
                        dry_run,
                        resolved,
                        step,
                    )
                    or _requires_progress_mcp(
                        settings,
                        dry_run,
                        state,
                        resolved,
                    )
                )
                if (
                    not dry_run
                    and settings.freecad_mcp_required
                    and settings.freecad_step_mcp_enabled
                    and step_skip_reason is not None
                ):
                    last_phase = "step_mcp_required_skipped"
                    last_issue = _issue(
                        step_index,
                        "REQUIRED_STEP_MCP_SKIPPED",
                        "Required per-step FreeCAD MCP validation is unavailable.",
                        phase=last_phase,
                        action_id=resolved.action_id,
                        target_port=resolved.target_port,
                        actual={"reason": step_skip_reason},
                    )
                    attempts.append(
                        _attempt(
                            step_index,
                            attempt_index,
                            state,
                            last_phase,
                            "rejected",
                            draft,
                            resolved,
                            [last_issue],
                        )
                    )
                    persist_rejected_attempt(
                        [_repair_observation(last_issue)],
                        attempt_index + 1,
                    )
                    terminal_step_summary = (
                        "Required per-step FreeCAD validation was unavailable; "
                        "the local LLM repair budget was not exhausted."
                    )
                    break

                if run_step_mcp:
                    stream.emit(
                        f"Step {step_index} attempt {attempt_index} is entering "
                        f"FreeCAD validation (timeout "
                        f"{settings.freecad_mcp_timeout_sec:g}s).",
                        force=True,
                    )
                    prepared = _prepared_manifest(
                        run_id,
                        intent,
                        state,
                        speculative,
                        actions,
                        step_verifications,
                        attempts,
                        checkpoints,
                        draft,
                        resolved,
                        step,
                        attempt_index,
                        gemini,
                        previous_freecad_verified=previous_freecad_verified,
                        preserved_suffix=preserved_suffix,
                        diagnostic_journal=diagnostic_journal,
                    )
                    _atomic_write_json(paths["checkpoint"], prepared)
                    try:
                        evidence_holder: dict[str, Any] = {}

                        def validate_candidate_evidence(
                            candidate_evidence: dict[str, Any],
                        ) -> None:
                            measurements = _freecad_measurements(candidate_evidence)
                            merged_measurements = {
                                module_id: dict(values)
                                for module_id, values in state.module_measurements.items()
                            }
                            merged_measurements.update(measurements)
                            measured_state = speculative.model_copy(
                                update={"module_measurements": merged_measurements}
                            )
                            measured_step = step.model_copy(
                                update={
                                    "mcp_status": "passed",
                                    "mcp_measurements": measurements,
                                    "mcp_assembly_bounds": _freecad_assembly_bounds(
                                        candidate_evidence
                                    ),
                                }
                            )
                            evidence_issues = validate_step_mcp_evidence(
                                intent,
                                measured_state,
                                resolved,
                                measured_step.transition,
                                [*step_verifications, measured_step],
                            )
                            if has_errors(evidence_issues):
                                augmented = dict(candidate_evidence)
                                augmented_checks = dict(
                                    candidate_evidence.get("checks") or {}
                                )
                                augmented_checks[
                                    "deterministic_constraint_failures"
                                ] = [
                                    issue.model_dump(mode="json")
                                    for issue in evidence_issues
                                    if issue.severity == "error"
                                ][:8]
                                augmented["checks"] = augmented_checks
                                raise _FreeCADSemanticError(
                                    "Digest-bound FreeCAD measurements violate the step contract",
                                    augmented,
                                )
                            evidence_holder["state"] = measured_state
                            evidence_holder["step"] = measured_step

                        selected_raw_path = (
                            run_dir
                            / "step_mcp"
                            / f"step_{step_index}_attempt_{attempt_index}.json"
                        )
                        selected_validation_path = (
                            run_dir
                            / "step_mcp"
                            / f"step_{step_index}_validation_{attempt_index}.json"
                        )
                        for kernel_retry in range(2):
                            if kernel_retry:
                                selected_raw_path = (
                                    run_dir
                                    / "step_mcp"
                                    / (
                                        f"step_{step_index}_attempt_{attempt_index}"
                                        f"_kernel_retry_{kernel_retry}.json"
                                    )
                                )
                                selected_validation_path = (
                                    run_dir
                                    / "step_mcp"
                                    / (
                                        f"step_{step_index}_validation_{attempt_index}"
                                        f"_kernel_retry_{kernel_retry}.json"
                                    )
                                )
                            try:
                                raw, evidence, publish_raw = (
                                    _validate_and_publish_freecad(
                                        settings,
                                        speculative,
                                        run_id=run_id,
                                        attempt_id=attempt_index,
                                        raw_result_path=selected_raw_path,
                                        validation_path=selected_validation_path,
                                        evidence_validator=(
                                            validate_candidate_evidence
                                        ),
                                    )
                                )
                                break
                            except _FreeCADSemanticError as kernel_exc:
                                if kernel_retry == 0 and _is_transient_occ_failure(
                                    kernel_exc.evidence
                                ):
                                    stream.emit(
                                        f"Step {step_index} attempt {attempt_index} "
                                        "hit a potentially transient OCC kernel "
                                        "failure; retrying the identical candidate "
                                        "once before involving the LLM.",
                                        force=True,
                                    )
                                    continue
                                raise
                        del raw, publish_raw
                        speculative = evidence_holder["state"]
                        step = evidence_holder["step"]
                        mcp_used = True
                        semantic_mcp_passed = True
                        # Earlier speculative candidates may have failed. A later
                        # digest-bound success is the terminal MCP state; do not
                        # leak a stale transient error into a success report.
                        mcp_error = None
                        step = step.model_copy(
                            update={
                                "mcp_status": "passed",
                                "mcp_result_path": str(selected_raw_path),
                                "freecad_validation_path": str(
                                    selected_validation_path
                                ),
                                "skipped_mcp_reason": None,
                                "mcp_measurements": _freecad_measurements(evidence),
                                "mcp_assembly_bounds": _freecad_assembly_bounds(
                                    evidence
                                ),
                            }
                        )
                        artifacts = artifacts.model_copy(
                            update={
                                "mcp_result_path": step.mcp_result_path,
                                "freecad_validation_path": step.freecad_validation_path,
                                "freecad_document_path": str(
                                    _freecad_document_path(
                                        Path(step.mcp_result_path), speculative
                                    )
                                ),
                            }
                        )
                        _atomic_write_json(
                            paths["checkpoint"],
                            {
                                **prepared,
                                "phase": "PUBLISHED",
                                "evidence": evidence,
                                "candidate_state": speculative.model_dump(mode="json"),
                                "candidate_state_digest": _pipe_state_digest(
                                    speculative
                                ),
                                "step_verification": step.model_dump(mode="json"),
                            },
                        )
                    except FreeCADMCPError as exc:
                        mcp_error = str(exc)
                        if isinstance(exc, _FreeCADSemanticError):
                            last_phase = "freecad_semantic_validation"
                            evidence_summary = _compact_freecad_failure_evidence(
                                exc.evidence
                            )
                            (
                                repair_expected,
                                repair_actual,
                                recommended_changes,
                            ) = _freecad_repair_contract(
                                evidence_summary,
                                module_path_kinds={
                                    module.id: str(
                                        module.params.get("path_kind") or module.type
                                    )
                                    for module in speculative.placed_modules
                                },
                                module_params={
                                    module.id: dict(module.params)
                                    for module in speculative.placed_modules
                                },
                            )
                            candidate_module_id = (
                                speculative.placed_modules[-1].id
                                if speculative.placed_modules
                                else None
                            )
                            last_issue = _issue(
                                step_index,
                                "FREECAD_GEOMETRY_VALIDATION_FAILED",
                                "FreeCAD rejected the speculative geometry.",
                                phase=last_phase,
                                action_id=resolved.action_id,
                                target_port=resolved.target_port,
                                module_id=candidate_module_id,
                                expected=repair_expected,
                                actual={
                                    "error": str(exc),
                                    "evidence_artifact_path": str(
                                        selected_validation_path
                                    ),
                                    "evidence_digest": _canonical_json_digest(
                                        exc.evidence
                                    ),
                                    **repair_actual,
                                    "evidence": evidence_summary,
                                },
                                suggestion={
                                    "operation": "revise_freecad_geometry_inputs",
                                    "recommended_changes": recommended_changes,
                                    "instruction": (
                                        "Use failed_checks and centerline_context to "
                                        "change the implicated radius, waypoint, length, "
                                        "offset, or blend value; return a new full action."
                                    ),
                                },
                            )
                            last_step = step.model_copy(
                                update={
                                    "status": "failed",
                                    "issues": [*step.issues, last_issue],
                                    "mcp_status": "failed",
                                    "mcp_error": str(exc),
                                    "skipped_mcp_reason": None,
                                }
                            )
                            observations = [_repair_observation(last_issue)]
                            attempts.append(
                                _attempt(
                                    step_index,
                                    attempt_index,
                                    state,
                                    last_phase,
                                    "rejected",
                                    draft,
                                    resolved,
                                    [last_issue],
                                )
                            )
                            persist_rejected_attempt(
                                observations,
                                attempt_index + 1,
                            )
                            continue
                        if settings.freecad_mcp_required:
                            last_phase = "step_mcp"
                            last_issue = _issue(
                                step_index,
                                "REQUIRED_STEP_MCP_FAILED",
                                "Required FreeCAD MCP infrastructure failed.",
                                phase=last_phase,
                                action_id=resolved.action_id,
                                actual={"error": str(exc)},
                            )
                            last_step = step.model_copy(
                                update={
                                    "status": "failed",
                                    "issues": [*step.issues, last_issue],
                                    "mcp_status": "failed",
                                    "mcp_error": str(exc),
                                    "skipped_mcp_reason": None,
                                }
                            )
                            observations = [_repair_observation(last_issue)]
                            attempts.append(
                                _attempt(
                                    step_index,
                                    attempt_index,
                                    state,
                                    last_phase,
                                    "rejected",
                                    draft,
                                    resolved,
                                    [last_issue],
                                )
                            )
                            persist_rejected_attempt(
                                observations,
                                attempt_index + 1,
                            )
                            terminal_step_summary = (
                                "Required FreeCAD MCP infrastructure failed; the "
                                "local LLM repair budget was not exhausted."
                            )
                            break
                        step = step.model_copy(
                            update={
                                "mcp_status": "unavailable",
                                "mcp_error": str(exc),
                                "skipped_mcp_reason": "FreeCAD MCP evidence unavailable.",
                            }
                        )
                        settings = replace(
                            settings,
                            freecad_mcp_enabled=False,
                            freecad_step_mcp_enabled=False,
                            freecad_capture_views=False,
                        )

                attempts.append(
                    _attempt(
                        step_index,
                        attempt_index,
                        state,
                        "commit",
                        "accepted",
                        draft,
                        resolved,
                        step.issues,
                    )
                )
                accepted = (resolved, speculative, step)
                append_search_event(
                    paths["search_events"],
                    {
                        "event_type": "candidate_accepted",
                        "run_id": run_id,
                        "state_before_id": state.state_id,
                        "state_after_id": speculative.state_id,
                        "step_index": step_index,
                        "attempt_index": attempt_index,
                        "candidate_digest": candidate_digest(resolved),
                        "module": resolved.module,
                    },
                )
                break

            if accepted is None:
                backjumpable_phases = {
                    "draft_validation",
                    "action_resolution",
                    "registry_validation",
                    "state_application",
                    "static_step_validation",
                    "freecad_semantic_validation",
                    "conflict_routing",
                }
                backjump_payload = {
                    "state_id": state.state_id,
                    "step_index": step_index,
                    "phase": last_phase,
                    "issue_code": last_issue.issue_code if last_issue else None,
                    "module_id": last_issue.module_id if last_issue else None,
                    "observation_digest": _canonical_json_digest(observations),
                }
                backjump_fingerprint = _canonical_json_digest(backjump_payload)
                if (
                    settings.conflict_search_enabled
                    and step_index > 1
                    and last_phase in backjumpable_phases
                    and len(causal_backjump_fingerprints)
                    < settings.max_causal_backjumps
                    and backjump_fingerprint not in causal_backjump_fingerprints
                ):
                    causal_backjump_fingerprints.add(backjump_fingerprint)
                    module_steps = {
                        module.id: index
                        for index, module in enumerate(state.placed_modules, start=1)
                    }
                    implicated_step = (
                        module_steps.get(last_issue.module_id)
                        if last_issue is not None and last_issue.module_id is not None
                        else None
                    )
                    rollback_step = max(
                        1,
                        min(
                            step_index - 1,
                            int(implicated_step or (step_index - 1)),
                        ),
                    )
                    rollback_index = rollback_step - 1
                    restored = checkpoints[rollback_index].model_copy(deep=True)
                    certificate = (
                        issue_certificate(last_issue)
                        if last_issue is not None
                        else None
                    )
                    pending_repair_observations = [
                        *observations,
                        {
                            "context_type": "causal_backjump",
                            "rollback_step": rollback_step,
                            "failed_step": step_index,
                            "failed_state_id": state.state_id,
                            "instruction": (
                                "Select a different primitive/variant at the restored "
                                "causal decision. The rejected suffix is a nogood."
                            ),
                            "conflict_certificate": (
                                certificate.model_dump(mode="json")
                                if certificate is not None
                                else None
                            ),
                        },
                    ]
                    append_search_event(
                        paths["search_events"],
                        {
                            "event_type": "causal_backjump_scheduled",
                            "run_id": run_id,
                            "failed_step": step_index,
                            "rollback_step": rollback_step,
                            "state_before_id": state.state_id,
                            "state_after_id": restored.state_id,
                            "fingerprint": backjump_fingerprint,
                            "conflict_certificate": (
                                certificate.model_dump(mode="json")
                                if certificate is not None
                                else None
                            ),
                        },
                    )
                    state = restored
                    actions = actions[:rollback_index]
                    step_verifications = step_verifications[:rollback_index]
                    checkpoints = checkpoints[: rollback_index + 1]
                    preserved_suffix = None
                    pending_draft = None
                    pending_draft_attempt_index = None
                    next_attempt_index = 1
                    semantic_mcp_passed = False
                    artifacts = _clear_state_bound_evidence(artifacts)
                    if gemini is not None and hasattr(gemini, "reset_lineage"):
                        gemini.reset_lineage("step_planner")
                    _write_progress(
                        paths,
                        actions,
                        attempts,
                        state,
                        step_verifications,
                        critic,
                    )
                    _write_checkpoint(
                        paths["checkpoint"],
                        phase="COMMITTED",
                        run_id=run_id,
                        intent=intent,
                        state=state,
                        previous_state=(
                            checkpoints[-2] if len(checkpoints) > 1 else None
                        ),
                        actions=actions,
                        step_verifications=step_verifications,
                        attempts=attempts,
                        gemini=gemini,
                        committed_states=checkpoints,
                        freecad_verified=False,
                        pending_repair_observations=pending_repair_observations,
                        next_attempt_index=next_attempt_index,
                        diagnostic_journal=diagnostic_journal,
                    )
                    stream.emit(
                        f"Step {step_index} exhausted local candidates; conflict-directed "
                        f"backjump restored step {rollback_step} for a different primitive.",
                        force=True,
                    )
                    continue
                if last_issue is None:
                    last_issue = _issue(
                        step_index,
                        "STEP_REPAIR_EXHAUSTED",
                        "No valid action was accepted within the repair budget.",
                        phase=last_phase,
                    )
                if last_step is not None:
                    step_verifications.append(last_step)
                # This is an interrupted transition, not a candidate final
                # state.  Running the final critic here turns expected
                # consequences (remaining goals, no geometry, open START) into
                # misleading independent failures.  Preserve state metadata but
                # report only the transition's actual blocking issue.
                critic = _critic_with_issue(None, last_issue).model_copy(
                    update={
                        "expected_open_ports": intent.expected_open_ports,
                        "actual_open_ports": len(state.open_ports),
                        "expected_open_ports_source": (
                            intent.expected_open_ports_source
                        ),
                    }
                )
                _fail_run(
                    run_id,
                    paths,
                    artifacts,
                    dry_run,
                    freecad_opened,
                    mcp_used,
                    mcp_error,
                    actions,
                    attempts,
                    state,
                    step_verifications,
                    critic,
                    last_phase,
                    terminal_step_summary
                    or "Step-local LLM repair budget was exhausted.",
                    gemini,
                    pause=settings.conflict_search_enabled,
                )

            resolved, state, accepted_step = accepted
            actions.append(resolved.model_dump(mode="json"))
            step_verifications.append(accepted_step)
            checkpoints.append(state.model_copy(deep=True))
            replay = _try_replay_preserved_suffix(
                preserved=preserved_suffix,
                state=state,
                actions=actions,
                step_verifications=step_verifications,
                engine=engine,
                intent=intent,
                settings=settings,
            )
            if replay is not None:
                state = replay.state
                actions.extend(item.model_dump(mode="json") for item in replay.actions)
                step_verifications.extend(replay.step_verifications)
                checkpoints.extend(replay.checkpoints)
                attempts.extend(replay.attempts)
                stream.emit(
                    "Rejoined the preserved plan after original step "
                    f"{replay.rejoin_original_step}; replayed original steps "
                    + ", ".join(str(item) for item in replay.reused_original_steps)
                    + " without new planner calls.",
                    force=True,
                )
                # The accepted repair step may have live evidence, but replay
                # then advances to a different state/digest without executing
                # FreeCAD. Never let evidence for the rejoin state suppress the
                # required final validation of the replayed terminal state.
                semantic_mcp_passed = False
                preserved_suffix = None
            elif not state.remaining_goals:
                preserved_suffix = None
            pending_draft = None
            pending_draft_attempt_index = None
            next_attempt_index = 1
            visual_paths = []
            visual_reviewed_digest = None
            if semantic_mcp_passed:
                artifacts = artifacts.model_copy(update={"visual_evidence_paths": []})
            else:
                artifacts = _clear_state_bound_evidence(artifacts)
            if gemini is not None and hasattr(gemini, "reset_lineage"):
                # Markov state is fully re-sent on the next accepted step. Keep
                # lineage only inside localized repair retries for this step.
                gemini.reset_lineage("step_planner")
            _write_progress(
                paths,
                actions,
                attempts,
                state,
                step_verifications,
                critic,
            )
            _write_checkpoint(
                paths["checkpoint"],
                phase="COMMITTED",
                run_id=run_id,
                intent=intent,
                state=state,
                previous_state=checkpoints[-2],
                actions=actions,
                step_verifications=step_verifications,
                attempts=attempts,
                gemini=gemini,
                committed_states=checkpoints,
                freecad_verified=semantic_mcp_passed,
                preserved_suffix=preserved_suffix,
                next_attempt_index=next_attempt_index,
                diagnostic_journal=diagnostic_journal,
            )

        critic = build_final_critic_report(
            intent,
            state,
            step_verifications,
            skipped_mcp_reason=_final_mcp_skip_reason(settings, dry_run),
        )
        blocking_pre_mcp_errors = [
            issue
            for issue in critic.issues
            if issue.severity == "error"
            and issue.issue_code
            not in {
                "GEOMETRIC_CONSTRAINT_REQUIRES_FREECAD_LENGTH",
                "GEOMETRIC_CONSTRAINT_REQUIRES_FREECAD_BOUNDS",
                "GOAL_LENGTH_REQUIRES_FREECAD",
                "SPLINE_CURVATURE_REQUIRES_FREECAD",
            }
        ]
        if (
            not dry_run
            and settings.freecad_mcp_enabled
            and not semantic_mcp_passed
            and not blocking_pre_mcp_errors
        ):
            try:
                final_evidence_holder: dict[str, Any] = {}

                def validate_final_candidate_evidence(
                    candidate_evidence: dict[str, Any],
                ) -> None:
                    if not step_verifications or (
                        step_verifications[-1].transition.state_after_id
                        != state.state_id
                    ):
                        raise FreeCADMCPError(
                            "Final FreeCAD evidence cannot be bound to the final state transition"
                        )
                    measurements = _freecad_measurements(candidate_evidence)
                    merged_measurements = {
                        module_id: dict(values)
                        for module_id, values in state.module_measurements.items()
                    }
                    merged_measurements.update(measurements)
                    measured_state = state.model_copy(
                        update={"module_measurements": merged_measurements}
                    )
                    measured_steps = list(step_verifications)
                    measured_steps[-1] = measured_steps[-1].model_copy(
                        update={
                            "mcp_status": "passed",
                            "mcp_result_path": str(paths["mcp_result"]),
                            "freecad_validation_path": str(paths["freecad_validation"]),
                            "mcp_error": None,
                            "skipped_mcp_reason": None,
                            "mcp_measurements": measurements,
                            "mcp_assembly_bounds": _freecad_assembly_bounds(
                                candidate_evidence
                            ),
                        }
                    )
                    measured_critic = build_final_critic_report(
                        intent,
                        measured_state,
                        measured_steps,
                        skipped_mcp_reason=None,
                    )
                    if not measured_critic.passed:
                        augmented = dict(candidate_evidence)
                        augmented_checks = dict(candidate_evidence.get("checks") or {})
                        augmented_checks["deterministic_constraint_failures"] = [
                            issue.model_dump(mode="json")
                            for issue in measured_critic.issues
                            if issue.severity == "error"
                        ][:8]
                        augmented["checks"] = augmented_checks
                        raise _FreeCADSemanticError(
                            "Digest-bound final measurements violate the immutable CAD contract",
                            augmented,
                        )
                    final_evidence_holder["state"] = measured_state
                    final_evidence_holder["steps"] = measured_steps
                    final_evidence_holder["critic"] = measured_critic

                raw, evidence, publish_raw = _validate_and_publish_freecad(
                    settings,
                    state,
                    run_id=run_id,
                    attempt_id=1000 + final_repair_round,
                    raw_result_path=paths["mcp_result"],
                    validation_path=paths["freecad_validation"],
                    evidence_validator=validate_final_candidate_evidence,
                )
                del raw, publish_raw
                state = final_evidence_holder["state"]
                step_verifications = final_evidence_holder["steps"]
                critic = final_evidence_holder["critic"]
                checkpoints[-1] = state.model_copy(deep=True)
                mcp_used = True
                semantic_mcp_passed = True
                mcp_error = None
                artifacts = artifacts.model_copy(
                    update={
                        "mcp_result_path": str(paths["mcp_result"]),
                        "freecad_validation_path": str(paths["freecad_validation"]),
                    }
                )
            except _FreeCADSemanticError as exc:
                mcp_error = str(exc)
                semantic_mcp_passed = False
                artifacts = _clear_state_bound_evidence(artifacts)
                evidence_summary = _compact_freecad_failure_evidence(exc.evidence)
                module_steps = {
                    module.id: index
                    for index, module in enumerate(state.placed_modules, start=1)
                }
                causal_module_id = _freecad_causal_repair_module(
                    evidence_summary, module_steps
                )
                target_step = (
                    module_steps[causal_module_id]
                    if causal_module_id in module_steps
                    else len(actions)
                )
                (
                    repair_expected,
                    repair_actual,
                    recommended_changes,
                ) = _freecad_repair_contract(
                    evidence_summary,
                    module_path_kinds={
                        module.id: str(module.params.get("path_kind") or module.type)
                        for module in state.placed_modules
                    },
                    module_params={
                        module.id: dict(module.params)
                        for module in state.placed_modules
                    },
                )
                critic = _critic_with_issue(
                    critic,
                    _issue(
                        target_step,
                        "FREECAD_GEOMETRY_VALIDATION_FAILED",
                        "FreeCAD rejected the final B-Rep geometry.",
                        phase="final_mcp",
                        module_id=causal_module_id,
                        expected=repair_expected,
                        actual={
                            "error": str(exc),
                            "evidence_artifact_path": str(paths["freecad_validation"]),
                            "evidence_digest": _canonical_json_digest(exc.evidence),
                            **repair_actual,
                            "evidence": evidence_summary,
                        },
                        suggestion={
                            "operation": "revise_freecad_geometry_inputs",
                            "recommended_changes": recommended_changes,
                            "instruction": (
                                "Re-plan the earliest implicated module using the "
                                "measured failed_checks; do not reuse the rejected values."
                            ),
                        },
                    ),
                )
            except FreeCADMCPError as exc:
                mcp_error = str(exc)
                semantic_mcp_passed = False
                artifacts = _clear_state_bound_evidence(artifacts)
                if (
                    settings.freecad_mcp_required
                    or settings.visual_validation_mode == "final_required"
                ):
                    critic = _critic_with_issue(
                        critic,
                        _issue(
                            len(actions),
                            "REQUIRED_FINAL_MCP_FAILED",
                            "Required final FreeCAD validation infrastructure failed.",
                            phase="final_mcp",
                            actual={"error": str(exc)},
                        ),
                    )
                    _fail_run(
                        run_id,
                        paths,
                        artifacts,
                        dry_run,
                        freecad_opened,
                        mcp_used,
                        mcp_error,
                        actions,
                        attempts,
                        state,
                        step_verifications,
                        critic,
                        "final_mcp",
                        "Required final FreeCAD validation failed.",
                        gemini,
                    )
                settings = replace(
                    settings,
                    freecad_mcp_enabled=False,
                    freecad_step_mcp_enabled=False,
                    freecad_capture_views=False,
                )
                critic = build_final_critic_report(
                    intent,
                    state,
                    step_verifications,
                    skipped_mcp_reason="FreeCAD MCP evidence unavailable.",
                )

        # 같은 최종 상태의 visual 조건, capture 요청과 review 결합에 동일한
        # geometry digest를 사용한다. 큰 상태를 세 번 직렬화하지 않는다.
        visual_candidate_digest = (
            geometry_payload_digest(state)
            if (
                critic.passed
                and not dry_run
                and settings.visual_validation_mode == "final_required"
            )
            else None
        )
        if (
            visual_candidate_digest is not None
            and visual_reviewed_digest != visual_candidate_digest
        ):
            try:
                if not semantic_mcp_passed:
                    raise FreeCADMCPError(
                        "Required visual validation has no digest-bound published document"
                    )
                visual_paths = asyncio.run(
                    capture_freecad_views(
                        settings,
                        run_dir / "views",
                        document_name=published_document_name(state, run_id=run_id),
                        payload_digest=visual_candidate_digest,
                    )
                )
                critic = _attach_view_evidence(critic, visual_paths)
                artifacts = artifacts.model_copy(
                    update={"visual_evidence_paths": visual_paths}
                )
                if gemini is None:
                    raise RuntimeError("Gemini is required for final visual validation")
                visual_result = _visual_review(
                    gemini,
                    state,
                    visual_paths,
                    intent=intent,
                )
                _atomic_write_json(
                    _next_visual_review_path(run_dir),
                    visual_result.model_dump(mode="json"),
                )
                if visual_result.passed:
                    visual_reviewed_digest = visual_candidate_digest
                else:
                    module_steps = {
                        module.id: step_index
                        for step_index, module in enumerate(
                            state.placed_modules, start=1
                        )
                    }
                    for issue_index, visual_issue in enumerate(
                        visual_result.issues or [], start=1
                    ):
                        unknown_module_ids = sorted(
                            set(visual_issue.module_ids) - set(module_steps)
                        )
                        if unknown_module_ids:
                            raise ValueError(
                                "Visual critic referenced unknown module IDs: "
                                + ", ".join(unknown_module_ids)
                            )
                        target_step = min(
                            module_steps[module_id]
                            for module_id in visual_issue.module_ids
                        )
                        action_id = (
                            str(actions[target_step - 1].get("action_id"))
                            if actions and actions[target_step - 1].get("action_id")
                            else None
                        )
                        critic = _critic_with_issue(
                            critic,
                            _issue(
                                target_step,
                                f"VISUAL_GEOMETRY_REJECTED_{issue_index}",
                                visual_issue.observation,
                                phase="visual_validation",
                                action_id=action_id,
                                actual={
                                    "visual_issue_code": visual_issue.issue_code,
                                    "module_ids": visual_issue.module_ids,
                                    "target_step": target_step,
                                    "claimed_target_step": visual_issue.target_step,
                                },
                            ),
                        )
            except Exception as exc:
                mcp_error = (mcp_error + "; " if mcp_error else "") + str(exc)
                critic = _critic_with_issue(
                    critic,
                    _issue(
                        len(actions),
                        "VISUAL_VALIDATION_FAILED",
                        "Required visual validation failed.",
                        phase="visual_validation",
                        actual={"error": str(exc)},
                    ),
                )
                _fail_run(
                    run_id,
                    paths,
                    artifacts,
                    dry_run,
                    freecad_opened,
                    mcp_used,
                    mcp_error,
                    actions,
                    attempts,
                    state,
                    step_verifications,
                    critic,
                    "visual_validation",
                    "Required visual infrastructure or structured review failed.",
                    gemini,
                )
        if critic.passed:
            break
        evidence_only_codes = {
            "GEOMETRIC_CONSTRAINT_REQUIRES_FREECAD_LENGTH",
            "GEOMETRIC_CONSTRAINT_REQUIRES_FREECAD_BOUNDS",
            "GOAL_LENGTH_REQUIRES_FREECAD",
            "SPLINE_CURVATURE_REQUIRES_FREECAD",
        }
        blocking_errors = [
            issue for issue in critic.issues if issue.severity == "error"
        ]
        if (
            not dry_run
            and not settings.freecad_mcp_enabled
            and blocking_errors
            and all(
                issue.issue_code in evidence_only_codes for issue in blocking_errors
            )
        ):
            # Geometry cannot manufacture missing infrastructure evidence. End
            # as explicit partial without spending a futile agenda-patch call.
            break
        if (
            dry_run
            or gemini is None
            or final_repair_round >= settings.final_repair_rounds
        ):
            _fail_run(
                run_id,
                paths,
                artifacts,
                dry_run,
                freecad_opened,
                mcp_used,
                mcp_error,
                actions,
                attempts,
                state,
                step_verifications,
                critic,
                "final_critic",
                "Final critic failed and no agenda repair remained.",
                gemini,
            )
        final_repair_round += 1
        try:
            base_repair_prompt = final_repair_prompt(
                state,
                critic,
                contract=_immutable_contract(intent),
            )
            repair_request = base_repair_prompt
            validated_repair: (
                tuple[
                    AgendaRepairDirective,
                    int,
                    PipeState,
                    list[dict[str, Any]],
                ]
                | None
            ) = None
            repair_error: Exception | None = None
            for patch_attempt in range(2):
                try:
                    directive = _call_structured(
                        gemini,
                        repair_request,
                        AgendaRepairDirectiveWire,
                        part="patch",
                        thinking_level="high",
                    )
                    if isinstance(directive, AgendaRepairDirectiveWire):
                        directive = AgendaRepairDirective.model_validate(
                            directive.model_dump(mode="python")
                        )
                    validated_repair = _validate_agenda_repair_directive(
                        directive,
                        state=state,
                        critic=critic,
                        actions=actions,
                        checkpoints=checkpoints,
                    )
                    break
                except (
                    GeminiLineageError,
                    StructuredOutputError,
                    HostContractValidationError,
                    TypeError,
                    ValueError,
                ) as exc:
                    repair_error = exc
                    if patch_attempt == 1:
                        raise
                    has_lineage = bool(
                        hasattr(gemini, "has_previous") and gemini.has_previous("patch")
                    )
                    repair_request = (
                        "" if has_lineage else base_repair_prompt + "\n\n"
                    ) + (
                        "Repair the previous agenda-localization JSON. Keep goals immutable. "
                        "Return the complete corrected directive. Diagnostic: "
                        + str(exc)[:1000]
                    )
            if validated_repair is None:
                raise repair_error or ValueError("agenda repair validation failed")
            directive, rollback_index, restored, pending_repair_observations = (
                validated_repair
            )
            preserved_suffix = _build_preserved_suffix(
                repair_start_step=directive.rollback_step,
                actions=actions,
                attempts=attempts,
                checkpoints=checkpoints,
                repair_hint=directive.repair_hint,
            )
            source_attempt = next(
                (
                    attempt
                    for attempt in reversed(attempts)
                    if attempt.step_index == directive.rollback_step
                    and attempt.status == "accepted"
                ),
                None,
            )
            repair_issues = [
                issue
                for issue in critic.issues
                if issue.issue_id in set(directive.target_issue_ids)
            ]
            if source_attempt is not None and repair_issues:
                attempts.append(
                    _attempt(
                        directive.rollback_step,
                        0,
                        restored,
                        "final_repair_replan",
                        "rejected",
                        (
                            ActionDraft.model_validate(source_attempt.draft)
                            if source_attempt.draft is not None
                            else None
                        ),
                        (
                            ResolvedAction.model_validate(source_attempt.resolved)
                            if source_attempt.resolved is not None
                            else None
                        ),
                        repair_issues,
                    )
                )
            state = restored
            actions = actions[:rollback_index]
            step_verifications = step_verifications[:rollback_index]
            checkpoints = checkpoints[: rollback_index + 1]
            checkpoints[-1] = restored.model_copy(deep=True)
            if hasattr(gemini, "reset_lineage"):
                gemini.reset_lineage("step_planner")
                gemini.reset_lineage("visual_validator")
            semantic_mcp_passed = False
            visual_paths = []
            visual_reviewed_digest = None
            artifacts = _clear_state_bound_evidence(artifacts)
            pending_draft = None
            pending_draft_attempt_index = None
            next_attempt_index = 1
            if _should_run_step_mcp(settings, dry_run) and restored.placed_modules:
                rollback_raw_path = (
                    run_dir / "rollback_mcp" / f"state_{restored.state_version}.json"
                )
                rollback_validation_path = (
                    run_dir
                    / "rollback_mcp"
                    / f"state_{restored.state_version}_validation.json"
                )
                _validate_and_publish_freecad(
                    settings,
                    restored,
                    run_id=run_id,
                    attempt_id=final_repair_round,
                    raw_result_path=rollback_raw_path,
                    validation_path=rollback_validation_path,
                )
                mcp_used = True
                semantic_mcp_passed = True
                artifacts = artifacts.model_copy(
                    update={
                        "mcp_result_path": str(rollback_raw_path),
                        "freecad_validation_path": str(rollback_validation_path),
                        "freecad_document_path": str(
                            _freecad_document_path(rollback_raw_path, restored)
                        ),
                    }
                )
            elif (
                not dry_run
                and settings.freecad_mcp_required
                and not restored.placed_modules
            ):
                # No empty candidate can be B-Rep validated. The replacement
                # action must be planned immediately from S0.
                semantic_mcp_passed = False
            _write_checkpoint(
                paths["checkpoint"],
                phase="COMMITTED",
                run_id=run_id,
                intent=intent,
                state=state,
                previous_state=(checkpoints[-2] if len(checkpoints) > 1 else None),
                actions=actions,
                step_verifications=step_verifications,
                attempts=attempts,
                gemini=gemini,
                committed_states=checkpoints,
                freecad_verified=semantic_mcp_passed,
                pending_repair_observations=pending_repair_observations,
                preserved_suffix=preserved_suffix,
                next_attempt_index=next_attempt_index,
                diagnostic_journal=diagnostic_journal,
            )
            _write_progress(paths, actions, attempts, state, step_verifications, critic)
        except Exception as exc:
            issue = _issue(
                len(actions) + 1,
                "FINAL_REPAIR_FAILED",
                "LLM final repair directive was invalid.",
                phase="final_repair",
                actual={"error": str(exc)},
            )
            critic = _critic_with_issue(critic, issue)
            _fail_run(
                run_id,
                paths,
                artifacts,
                dry_run,
                freecad_opened,
                mcp_used,
                mcp_error,
                actions,
                attempts,
                state,
                step_verifications,
                critic,
                "final_repair",
                "Final agenda repair could not be applied.",
                gemini,
            )

    script = build_freecad_script(
        state,
        run_id=run_id,
        attempt_id=1,
        modeling_tolerance=settings.modeling_tolerance,
    )
    _atomic_write_text(paths["script"], script)

    if not dry_run and settings.freecad_mcp_enabled and not semantic_mcp_passed:
        try:
            raw, evidence, publish_raw = _validate_and_publish_freecad(
                settings,
                state,
                run_id=run_id,
                attempt_id=1,
                raw_result_path=paths["mcp_result"],
                validation_path=paths["freecad_validation"],
            )
            del raw, publish_raw
            mcp_used = True
            semantic_mcp_passed = True
            artifacts = artifacts.model_copy(
                update={
                    "mcp_result_path": str(paths["mcp_result"]),
                    "freecad_validation_path": str(paths["freecad_validation"]),
                }
            )
        except Exception as exc:
            mcp_error = str(exc)
            artifacts = _clear_state_bound_evidence(artifacts)
            if settings.freecad_mcp_required:
                issue = _issue(
                    len(actions),
                    "REQUIRED_FINAL_MCP_FAILED",
                    "Required final FreeCAD validation failed.",
                    phase="final_mcp",
                    actual={"error": str(exc)},
                )
                critic = _critic_with_issue(critic, issue)
                _fail_run(
                    run_id,
                    paths,
                    artifacts,
                    dry_run,
                    freecad_opened,
                    mcp_used,
                    mcp_error,
                    actions,
                    attempts,
                    state,
                    step_verifications,
                    critic,
                    "final_mcp",
                    "Required final FreeCAD validation failed.",
                    gemini,
                )

    # 게시가 확인된 최종 상태의 digest는 view, 문서 경로와 최종 상태 판정이
    # 함께 사용하므로 한 번만 계산한다.
    final_geometry_digest = (
        geometry_payload_digest(state) if semantic_mcp_passed else None
    )
    if (
        semantic_mcp_passed
        and settings.freecad_capture_views
        and settings.visual_validation_mode == "on_warning"
        and critic.warning_count > 0
    ):
        try:
            visual_paths = asyncio.run(
                capture_freecad_views(
                    settings,
                    run_dir / "views",
                    document_name=published_document_name(state, run_id=run_id),
                    payload_digest=final_geometry_digest,
                )
            )
            critic = _attach_view_evidence(critic, visual_paths)
            artifacts = artifacts.model_copy(
                update={"visual_evidence_paths": visual_paths}
            )
        except Exception as exc:
            mcp_error = (mcp_error + "; " if mcp_error else "") + f"view capture: {exc}"

    if semantic_mcp_passed:
        current_mcp_result_path = (
            Path(artifacts.mcp_result_path)
            if artifacts.mcp_result_path is not None
            else paths["mcp_result"]
        )
        artifacts = artifacts.model_copy(
            update={
                "freecad_document_path": str(
                    _freecad_document_path(
                        current_mcp_result_path,
                        state,
                        payload_digest=final_geometry_digest,
                    )
                )
            }
        )
    _atomic_write_json(paths["critic"], critic.model_dump(mode="json"))
    _write_checkpoint(
        paths["checkpoint"],
        phase="COMMITTED",
        run_id=run_id,
        intent=intent,
        state=state,
        previous_state=checkpoints[-2] if len(checkpoints) > 1 else None,
        actions=actions,
        step_verifications=step_verifications,
        attempts=attempts,
        gemini=gemini,
        committed_states=checkpoints,
        freecad_verified=semantic_mcp_passed,
        preserved_suffix=preserved_suffix,
        diagnostic_journal=diagnostic_journal,
    )
    verified = semantic_mcp_passed and (
        settings.visual_validation_mode != "final_required"
        or visual_reviewed_digest == final_geometry_digest
    )
    status = "success" if verified else "partial"
    verification_status = "passed" if verified else "partial"
    report = _make_report(
        run_id,
        dry_run,
        freecad_opened,
        mcp_used,
        mcp_error,
        artifacts,
        step_verifications,
        critic,
        status=status,
        verification_status=verification_status,
        skipped_mcp_reason=None
        if verified
        else _final_mcp_skip_reason(settings, dry_run),
        summary=(
            f"Generated {len(state.placed_modules)} modules with verified FreeCAD evidence."
            if verified
            else f"Generated {len(state.placed_modules)} modules; live FreeCAD verification is incomplete."
        ),
        gemini=gemini,
        repair_attempt_count=sum(1 for item in attempts if item.status == "rejected"),
    )
    _atomic_write_json(paths["report"], report.model_dump(mode="json"))
    _write_progress(paths, actions, attempts, state, step_verifications, critic)
    stream.emit(f"Done. Artifacts saved to {run_dir}.", force=True)
    return report


def _extract_intent(
    prompt: str,
    settings: Settings,
    *,
    dry_run: bool,
    gemini: GeminiClient | None,
    attempt_journal: list[dict[str, Any]] | None = None,
    attempt_journal_path: Path | None = None,
    diagnostic_journal: list[dict[str, Any]] | None = None,
    diagnostic_journal_path: Path | None = None,
    stream: ThinkingStream | None = None,
) -> IntentResult:
    """사용자 요청을 불변 설계 계약으로 변환하고 의미 검증까지 완료한다.

    프로덕션에서는 Gemini가 반환한 JSON만 허용한다. 전체 typed grammar가
    provider에서 거절되면 작은 JSON-string envelope로 한 번 재협상한 뒤에도
    동일한 strict JSON, Pydantic, safety, scope 검증을 모두 적용한다. 스키마
    협상, 불완전 출력, 의미 교정은 서로 독립된 제한 예산을 사용한다.
    """

    def record_attempt(payload: dict[str, Any]) -> None:
        if attempt_journal is None:
            return
        attempt_journal.append(
            {
                "attempt_index": len(attempt_journal) + 1,
                **payload,
            }
        )
        if attempt_journal_path is not None:
            _atomic_write_json(attempt_journal_path, attempt_journal)

    def update_last_attempt(payload: dict[str, Any]) -> None:
        if attempt_journal is None or not attempt_journal:
            return
        attempt_journal[-1].update(payload)
        if attempt_journal_path is not None:
            _atomic_write_json(attempt_journal_path, attempt_journal)

    def record_diagnostic(payload: dict[str, Any]) -> int | None:
        if diagnostic_journal is None:
            return None
        diagnostic_journal.append(
            {
                "diagnostic_index": len(diagnostic_journal) + 1,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
        )
        if diagnostic_journal_path is not None:
            _atomic_write_json(diagnostic_journal_path, diagnostic_journal)
        return len(diagnostic_journal) - 1

    def update_diagnostic(index: int | None, payload: dict[str, Any]) -> None:
        if diagnostic_journal is None or index is None:
            return
        diagnostic_journal[index].update(payload)
        if diagnostic_journal_path is not None:
            _atomic_write_json(diagnostic_journal_path, diagnostic_journal)

    if dry_run:
        result = infer_intent(prompt, settings)
        result = _canonicalize_dependent_intent_geometry(result, settings)
        _validate_intent_safety(prompt, result, settings)
        _validate_intent_scope(result, dry_run=True, prompt=prompt)
        record_attempt(
            {
                "status": "accepted",
                "phase": "local_dry_run",
                "consumes_semantic_budget": False,
                "parsed_intent": True,
                "candidate_digest": _intent_candidate_digest(result),
                "diagnostic": None,
                "lineage_reset": False,
            }
        )
        return result
    if gemini is None:
        raise RuntimeError("Gemini client is required outside dry-run mode.")
    explicit_mm_values = _explicit_mm_values(prompt)
    exact_mm_values = _exact_mm_contract_values(prompt)
    explicit_mm_ranges = _explicit_mm_ranges(prompt)
    # Gemini's Interactions API accepts the native-number intent schema but can
    # reject schemas which replace every number with a shared string/decimal
    # $ref.  Native JSON numbers are therefore the production default. Exact
    # authored measurements remain protected by deterministic intent safety and
    # the advisor-backed semantic repair loop after a candidate is returned.
    schema_profiles: list[tuple[str, str, list[str] | None, type[Any]]] = [
        ("preferred_plain_numeric", "plain", None, LLMProductionIntent),
        (
            "host_validated_json_envelope",
            "plain",
            None,
            LLMIntentJSONEnvelope,
        ),
    ]
    schema_profile_index = 0
    base_prompt = intent_prompt(
        prompt,
        defaults={
            "outer_diameter": settings.default_outer_diameter,
            "wall_thickness": settings.default_wall_thickness,
            "bend_radius": settings.default_bend_radius,
        },
    )
    if explicit_mm_values:
        base_prompt += (
            "\nDeterministic source measurement vocabulary: the user authored "
            "these millimeter values and range endpoints: "
            f"{json.dumps(explicit_mm_values, ensure_ascii=False)}. "
            "Preserve every exact or approximate nominal measurement in the "
            "appropriate typed metric field (or an unsupported: hard "
            "constraint); words such as about/approximately do not authorize "
            "inventing an arbitrary tolerance. For an explicit range, select "
            "a typed physical value inside the inclusive interval instead of "
            "copying both endpoints as unrelated dimensions. Encode authored "
            "floats exactly in the numeric representation required by the "
            "current response schema; never expand or perturb a decimal.\n"
        )
        if exact_mm_values:
            base_prompt += (
                "Exact/nominal millimeter contracts: "
                f"{json.dumps(exact_mm_values, ensure_ascii=False)}.\n"
            )
        if explicit_mm_ranges:
            base_prompt += (
                "Inclusive millimeter ranges: "
                f"{json.dumps([[item.minimum, item.maximum] for item in explicit_mm_ranges], ensure_ascii=False)}.\n"
            )
    (
        schema_profile,
        schema_numeric_mode,
        schema_numeric_literals,
        schema_response_model,
    ) = schema_profiles[schema_profile_index]
    request = (
        _intent_json_envelope_request(base_prompt)
        if schema_response_model is LLMIntentJSONEnvelope
        else base_prompt
    )
    last_error: Exception | None = None
    last_semantic_validation_details: list[dict[str, Any]] = []
    diagnostic_history: list[str] = []
    intent_thinking_level = "low"
    last_semantic_diagnostic: str | None = None
    semantic_diagnostic_repeat_count = 0
    # Provider schema negotiation, malformed output retries, and semantic
    # correction are distinct failure classes and must not consume one
    # another's bounded retry budgets.
    semantic_attempt = 0
    schema_retry_attempt = 0
    structured_retry_attempt = 0
    validation_signature_counts: Counter[str] = Counter()
    exact_failure_counts: Counter[str] = Counter()
    # Incomplete/malformed structured output is not a correction of a parsed
    # design. Give that transport/schema class its own small bounded retry pool
    # so one truncated response cannot consume the remaining semantic repairs.
    max_structured_retries = 2
    max_schema_retries = max(0, len(schema_profiles) - 1)
    while True:
        if stream is not None:
            if structured_retry_attempt:
                stream.emit(
                    f"Intent protocol retry {structured_retry_attempt}/"
                    f"{max_structured_retries} for semantic attempt "
                    f"{semantic_attempt + 1}, schema={schema_profile} started.",
                    force=True,
                )
            elif schema_retry_attempt:
                stream.emit(
                    f"Intent schema retry {schema_retry_attempt}/"
                    f"{max_schema_retries} for semantic attempt "
                    f"{semantic_attempt + 1}, schema={schema_profile} started.",
                    force=True,
                )
            else:
                stream.emit(
                    f"Intent attempt {semantic_attempt + 1}/"
                    f"{settings.intent_repair_attempts + 1} started "
                    f"with schema={schema_profile}.",
                    force=True,
                )
        parsed_intent: IntentResult | None = None
        provider_response_received = False
        try:
            result = _call_structured(
                gemini,
                request,
                schema_response_model,
                part="intent",
                thinking_level=intent_thinking_level,
                numeric_literals=schema_numeric_literals,
                numeric_schema_mode=schema_numeric_mode,
                system_instruction=intent_system_instruction(),
            )
            if isinstance(result, LLMIntentJSONEnvelope):
                try:
                    envelope_payload = _strict_json_loads(result.intent_json)
                except ValueError as exc:
                    raise StructuredOutputError(
                        "intent",
                        result.intent_json,
                        exc,
                    ) from exc
                try:
                    result = LLMProductionIntent.model_validate(envelope_payload)
                except (TypeError, ValueError) as exc:
                    provider_response_received = True
                    raise HostContractValidationError(
                        "intent",
                        envelope_payload,
                        exc,
                    ) from exc
            provider_response_received = True
            if isinstance(result, LLMProductionIntent):
                intent = result.to_intent_result()
            elif isinstance(result, ProductionIntent):
                intent = result.to_intent_result()
            elif isinstance(result, IntentResult):
                # Test-double/v1 compatibility only; real calls are schema-v2.
                intent = result
            else:
                raise TypeError(
                    f"Unexpected intent result type: {type(result).__name__}"
                )
            intent = _canonicalize_dependent_intent_geometry(intent, settings)
            parsed_intent = intent
            _validate_intent_safety(prompt, intent, settings)
            _validate_intent_scope(intent, dry_run=False, prompt=prompt)
            record_attempt(
                {
                    "status": "accepted",
                    "phase": "semantic_validation",
                    "scope_validated": True,
                    "semantic_attempt": semantic_attempt + 1,
                    "schema_retry_attempt": schema_retry_attempt,
                    "consumes_semantic_budget": True,
                    "parsed_intent": True,
                    "candidate_digest": _intent_candidate_digest(intent),
                    "schema_profile": schema_profile,
                    "diagnostic": None,
                    "lineage_reset": False,
                }
            )
            if stream is not None:
                stream.emit(
                    f"Intent attempt {semantic_attempt + 1} passed deterministic "
                    "semantic and scope validation.",
                    force=True,
                )
            return intent
        except GeminiInvalidRequestError as exc:
            # A provider-side grammar rejection happened before any model draft
            # existed.  Negotiate progressively smaller, value-independent
            # schemas without spending semantic repair turns.
            if not _is_invalid_planner_request(exc):
                raise
            next_profile_index = schema_profile_index + 1
            will_retry_schema = (
                next_profile_index < len(schema_profiles)
                and schema_retry_attempt < max_schema_retries
            )
            if will_retry_schema and hasattr(gemini, "reset_lineage"):
                gemini.reset_lineage("intent")
            record_attempt(
                {
                    "status": (
                        "schema_retry" if will_retry_schema else "schema_rejected"
                    ),
                    "phase": "provider_schema_negotiation",
                    "semantic_attempt": semantic_attempt + 1,
                    "schema_retry_attempt": schema_retry_attempt + 1,
                    "structured_retry_attempt": structured_retry_attempt,
                    "consumes_semantic_budget": False,
                    "parsed_intent": False,
                    "candidate_digest": None,
                    "diagnostic": str(exc)[:1200],
                    "schema_profile": schema_profile,
                    "next_schema_profile": (
                        schema_profiles[next_profile_index][0]
                        if will_retry_schema
                        else None
                    ),
                    "lineage_reset": will_retry_schema,
                    "will_retry": will_retry_schema,
                }
            )
            if not will_retry_schema:
                raise
            schema_profile_index = next_profile_index
            (
                schema_profile,
                schema_numeric_mode,
                schema_numeric_literals,
                schema_response_model,
            ) = schema_profiles[schema_profile_index]
            schema_retry_attempt += 1
            if schema_response_model is LLMIntentJSONEnvelope:
                request = _intent_json_envelope_request(request)
            continue
        except (GeminiBudgetError, GeminiConfigError) as exc:
            if last_semantic_validation_details:
                raise _IntentSemanticValidationExhausted(
                    last_semantic_validation_details,
                    exc,
                    terminal_reason="llm_budget_or_configuration_exhausted",
                ) from exc
            raise
        except (
            GeminiLineageError,
            StructuredOutputError,
            HostContractValidationError,
            TypeError,
            ValueError,
        ) as exc:
            last_error = exc
            diagnostic = _intent_repair_diagnostic(exc)
            semantic_repair = (
                parsed_intent is not None
                or provider_response_received
                or isinstance(exc, HostContractValidationError)
            )
            if semantic_repair:
                if diagnostic == last_semantic_diagnostic:
                    semantic_diagnostic_repeat_count += 1
                else:
                    last_semantic_diagnostic = diagnostic
                    semantic_diagnostic_repeat_count = 1
            validation_details = (
                _intent_validation_details(exc, diagnostic) if semantic_repair else []
            )
            if semantic_repair:
                last_semantic_validation_details = [
                    dict(item) for item in validation_details
                ]
            issue_codes = list(
                dict.fromkeys(
                    str(item.get("issue_code"))
                    for item in validation_details
                    if item.get("issue_code")
                )
            )
            candidate_digest = (
                _intent_candidate_digest(parsed_intent)
                if parsed_intent is not None
                else None
            )
            validation_signature = (
                _intent_validation_signature(validation_details, diagnostic)
                if semantic_repair
                else None
            )
            exact_failure_signature = (
                hashlib.sha256(
                    f"{candidate_digest}:{validation_signature}".encode("utf-8")
                ).hexdigest()
                if candidate_digest is not None and validation_signature is not None
                else None
            )
            signature_repeat_count = 0
            exact_repeat_count = 0
            if validation_signature is not None:
                validation_signature_counts[validation_signature] += 1
                signature_repeat_count = validation_signature_counts[
                    validation_signature
                ]
            if exact_failure_signature is not None:
                exact_failure_counts[exact_failure_signature] += 1
                exact_repeat_count = exact_failure_counts[exact_failure_signature]
            will_retry = (
                semantic_attempt < settings.intent_repair_attempts
                if semantic_repair
                else structured_retry_attempt < max_structured_retries
            )
            phase = (
                "intent_scope"
                if isinstance(exc, _IntentScopeValidationError)
                else ("semantic_validation" if semantic_repair else "structured_output")
            )
            record_attempt(
                {
                    "status": "rejected",
                    "phase": phase,
                    "semantic_attempt": semantic_attempt + 1,
                    "schema_retry_attempt": schema_retry_attempt,
                    "structured_retry_attempt": (
                        structured_retry_attempt
                        if semantic_repair
                        else structured_retry_attempt + 1
                    ),
                    "consumes_semantic_budget": semantic_repair,
                    "parsed_intent": parsed_intent is not None,
                    "provider_response_received": provider_response_received
                    or isinstance(exc, HostContractValidationError),
                    "candidate_digest": candidate_digest,
                    "validation_signature": validation_signature,
                    "exact_failure_signature": exact_failure_signature,
                    "validation_signature_repeat_count": signature_repeat_count,
                    "exact_failure_repeat_count": exact_repeat_count,
                    "issue_codes": issue_codes,
                    "diagnostic": diagnostic,
                    "lineage_reset": False,
                    "will_retry": will_retry,
                }
            )
            diagnostic_record_index = (
                record_diagnostic(
                    {
                        "intent_attempt": semantic_attempt + 1,
                        "phase": phase,
                        "candidate_digest": candidate_digest,
                        "validation_signature": validation_signature,
                        "exact_failure_signature": exact_failure_signature,
                        "validation_signature_repeat_count": signature_repeat_count,
                        "exact_failure_repeat_count": exact_repeat_count,
                        "issue_codes": issue_codes,
                        "deterministic_issues": validation_details,
                        "rejected_candidate": (
                            parsed_intent.model_dump(mode="json")
                            if parsed_intent is not None
                            else None
                        ),
                        "advisor_required": settings.intent_repair_advisor_required,
                        "advisor_status": "pending",
                        "advisor_protocol_attempts": 0,
                        "advisor": None,
                        "advisor_error": None,
                        "terminal_reason": None,
                        "will_retry": will_retry,
                    }
                )
                if semantic_repair
                else None
            )

            advisor: IntentRepairAdvice | None = None
            advisor_error: str | None = None
            advisor_protocol_attempts = 0
            advisor_attempts: list[dict[str, Any]] = []
            advisor_source = "not_applicable"
            advisor_fallback_used = False
            advisor_status = "not_applicable"
            terminal_reason: str | None = None
            if exact_repeat_count >= MAX_IDENTICAL_INTENT_FAILURES:
                will_retry = False
                terminal_reason = "identical_intent_failure_stagnation"
            elif semantic_repair and not will_retry:
                terminal_reason = "intent_repair_budget_exhausted"
            if semantic_repair and parsed_intent is not None:
                if not will_retry:
                    advisor_status = "skipped_no_author_repair_budget"
                    advisor_source = (
                        "host_stagnation"
                        if terminal_reason == "identical_intent_failure_stagnation"
                        else "host_repair_budget_exhausted"
                    )
                elif not settings.intent_repair_advisor_enabled:
                    advisor_status = "disabled"
                    advisor_source = "deterministic_evidence_only"
                    advisor_fallback_used = True
                elif not getattr(gemini, "supports_intent_repair_advisor", False):
                    advisor_status = "unsupported_by_client"
                    advisor_source = "deterministic_evidence_only"
                    advisor_fallback_used = True
                else:
                    if stream is not None:
                        stream.emit(
                            "Intent validation advisor is tracing the rejected "
                            "contract before the next authoring attempt.",
                            force=True,
                        )
                    advisor_outcome = _request_intent_repair_advice(
                        gemini,
                        settings=settings,
                        prompt=prompt,
                        candidate=parsed_intent,
                        validation_details=validation_details,
                        candidate_digest=candidate_digest or "",
                        validation_signature=validation_signature or "",
                        prior_diagnostics=(diagnostic_journal or [])[:-1][-4:],
                    )
                    advisor = advisor_outcome.advice
                    advisor_protocol_attempts = advisor_outcome.call_count
                    advisor_error = advisor_outcome.error
                    advisor_attempts = advisor_outcome.attempts
                    advisor_source = advisor_outcome.source
                    advisor_fallback_used = advisor_outcome.fallback_used
                    advisor_status = (
                        "complete" if advisor is not None else "degraded_fallback"
                    )
                    if advisor is not None:
                        host_terminal_reason = _host_authorized_intent_terminal_reason(
                            advisor,
                            validation_details,
                        )
                        if host_terminal_reason is not None:
                            will_retry = False
                            terminal_reason = host_terminal_reason
                        if stream is not None:
                            stream.emit(
                                "Intent advisor: "
                                f"{advisor.diagnosis_class}/{advisor.disposition} — "
                                f"{advisor.summary[:320]}",
                                force=True,
                            )
                    elif stream is not None:
                        stream.emit(
                            "Intent advisors did not produce host-valid advice; "
                            "continuing with deterministic validation evidence.",
                            force=True,
                        )

            will_reset_lineage = (
                will_retry
                and hasattr(gemini, "reset_lineage")
                and (
                    not semantic_repair
                    or semantic_diagnostic_repeat_count == 2
                    or exact_repeat_count == 2
                )
            )
            update_last_attempt(
                {
                    "advisor_status": advisor_status,
                    "advisor_disposition": (
                        advisor.disposition if advisor is not None else None
                    ),
                    "advisor_protocol_attempts": advisor_protocol_attempts,
                    "advisor_error": advisor_error,
                    "advisor_required": settings.intent_repair_advisor_required,
                    "advisor_source": advisor_source,
                    "advisor_fallback_used": advisor_fallback_used,
                    "terminal_reason": terminal_reason,
                    "lineage_reset": will_reset_lineage,
                    "will_retry": will_retry,
                }
            )
            update_diagnostic(
                diagnostic_record_index,
                {
                    "advisor_status": advisor_status,
                    "advisor_protocol_attempts": advisor_protocol_attempts,
                    "advisor": (
                        advisor.model_dump(mode="json") if advisor is not None else None
                    ),
                    "advisor_error": advisor_error,
                    "advisor_attempts": advisor_attempts,
                    "advisor_source": advisor_source,
                    "advisor_fallback_used": advisor_fallback_used,
                    "retry_decision_source": (
                        "host_stagnation"
                        if terminal_reason == "identical_intent_failure_stagnation"
                        else (
                            "host_repair_budget_exhausted"
                            if terminal_reason == "intent_repair_budget_exhausted"
                            else (
                                "host_authorized_advisor_terminal"
                                if terminal_reason is not None
                                else advisor_source
                            )
                        )
                    ),
                    "terminal_authority": (
                        "host" if terminal_reason is not None else None
                    ),
                    "terminal_reason": terminal_reason,
                    "lineage_reset": will_reset_lineage,
                    "will_retry": will_retry,
                },
            )
            if stream is not None:
                stream.emit(
                    f"Intent attempt {semantic_attempt + 1} rejected in {phase}: "
                    + (", ".join(issue_codes[:4]) or diagnostic[:320]),
                    force=True,
                )
            if not will_retry:
                break
            if semantic_repair:
                semantic_attempt += 1
            else:
                structured_retry_attempt += 1
            if diagnostic not in diagnostic_history:
                diagnostic_history.append(diagnostic)
            diagnostic_history = diagnostic_history[-4:]
            if _is_spline_intent_safety_diagnostic(diagnostic) or isinstance(
                exc, HostContractValidationError
            ):
                intent_thinking_level = "medium"
            if will_reset_lineage:
                # Malformed output has no trustworthy object to edit. A second
                # identical semantic miss is also de-anchored once so the next
                # call sees the complete request instead of repeating one draft.
                gemini.reset_lineage("intent")
            targeted_guidance = _intent_repair_guidance(diagnostic)
            repair_envelope = {
                "context_type": "intent_recursive_repair",
                "failed_attempt": semantic_attempt,
                "rejected_candidate": (
                    parsed_intent.model_dump(mode="json")
                    if parsed_intent is not None
                    else None
                ),
                "deterministic_issues": validation_details,
                "validation_history": diagnostic_history,
                "validation_signature": validation_signature,
                "signature_repeat_count": signature_repeat_count,
                "exact_failure_repeat_count": exact_repeat_count,
                "independent_advisor": (
                    advisor.model_dump(mode="json") if advisor is not None else None
                ),
                "advisor_source": advisor_source,
                "advisor_fallback_used": advisor_fallback_used,
                "advisor_attempts": advisor_attempts,
                "advisor_required": settings.intent_repair_advisor_required,
                "targeted_repair_rule": targeted_guidance or None,
                "authority": (
                    "The advisor only diagnoses. Return a new complete intent; "
                    "the deterministic semantic and scope validators must both "
                    "accept it. Never delete or weaken a user-authored requirement."
                ),
                "fallback_instruction": (
                    "No advisor response has acceptance authority. If independent "
                    "advice is absent, repair the exact deterministic_issues using "
                    "the targeted rule and rejected candidate. Preserve unrelated "
                    "fields byte-for-byte in meaning and return a complete object."
                    if advisor is None
                    else None
                ),
            }
            request = (
                base_prompt
                + "\n\n"
                + (
                    "The prior intent was schema-valid but failed deterministic "
                    "semantic validation. Return a complete corrected intent JSON "
                    "object, preserving every unaffected topology, measurement, "
                    "dependency, and already-valid goal. Change only the fields "
                    "needed to fix the diagnostics; do not regenerate or omit "
                    "unrelated requirements. "
                    if semantic_repair
                    else "The prior attempt was discarded. Generate a new complete "
                    "intent JSON object from the full user request without changing "
                    "it. Do not continue, salvage, or quote any prior partial output. "
                )
                + "Every validation diagnostic in this bounded history must be "
                + "fixed simultaneously; do not regress an earlier correction. "
                + "Validation diagnostic history and current typed evidence are "
                + "contained in the following envelope. "
                + "Use this complete typed repair envelope, including the rejected "
                + "candidate when one exists: "
                + json.dumps(
                    repair_envelope,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            if schema_response_model is LLMIntentJSONEnvelope:
                request = _intent_json_envelope_request(request)
    if last_error is not None:
        if last_semantic_validation_details and not isinstance(
            last_error,
            (_IntentSafetyValidationError, _IntentScopeValidationError),
        ):
            raise _IntentSemanticValidationExhausted(
                last_semantic_validation_details,
                last_error,
                terminal_reason="intent_repair_budget_exhausted",
            ) from last_error
        raise last_error
    raise RuntimeError("Intent extraction failed without a diagnostic")


def _intent_validation_details(
    exc: Exception,
    diagnostic: str,
) -> list[dict[str, Any]]:
    """Preserve every deterministic intent failure as structured advisor input."""

    if isinstance(exc, _IntentScopeValidationError):
        return [issue.model_dump(mode="json") for issue in exc.issues]
    if isinstance(exc, _IntentSafetyValidationError):
        return [
            {
                "issue_id": f"INTENT_SAFETY_{index:02d}",
                "issue_code": "INTENT_SAFETY_CONTRACT",
                "check_name": "intent_semantic_validation",
                "message": message,
                "expected": {},
                "actual": {},
                "suggestion": {},
            }
            for index, message in enumerate(exc.diagnostics, start=1)
        ]
    return [
        {
            "issue_code": "INTENT_STRUCTURED_OR_HOST_CONTRACT",
            "check_name": "intent_semantic_validation",
            "message": diagnostic,
            "expected": {},
            "actual": {"error_type": type(exc).__name__},
            "suggestion": {},
        }
    ]


def _intent_validation_signature(
    validation_details: list[dict[str, Any]],
    diagnostic: str,
) -> str:
    """Hash the validator facts independently from a particular candidate."""

    facts = [
        {
            "issue_code": item.get("issue_code"),
            "check_name": item.get("check_name"),
            "message": item.get("message"),
            "expected": item.get("expected") or {},
            "actual": item.get("actual") or {},
        }
        for item in validation_details
    ]
    payload = json.dumps(
        {"facts": facts, "fallback_diagnostic": diagnostic if not facts else None},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _request_intent_repair_advice(
    gemini: Any,
    *,
    settings: Settings,
    prompt: str,
    candidate: IntentResult,
    validation_details: list[dict[str, Any]],
    candidate_digest: str,
    validation_signature: str,
    prior_diagnostics: list[dict[str, Any]],
) -> _IntentAdvisorOutcome:
    """Run feedback-aware diagnosis, then a separate repair-only reviewer.

    The result is advice only.  The intent author must still return a complete
    replacement and every deterministic gate remains authoritative.
    """

    compact_history = [
        {
            "candidate_digest": item.get("candidate_digest"),
            "validation_signature": item.get("validation_signature"),
            "issue_codes": item.get("issue_codes") or [],
            "advisor_status": item.get("advisor_status"),
            "advisor_disposition": (
                (item.get("advisor") or {}).get("disposition")
                if isinstance(item.get("advisor"), dict)
                else None
            ),
            "terminal_reason": item.get("terminal_reason"),
        }
        for item in prior_diagnostics[-4:]
    ]
    allowed_dispositions = _allowed_intent_advisor_dispositions(validation_details)
    context: dict[str, Any] = {
        "protocol_version": 1,
        "source_request": prompt,
        "candidate_digest": candidate_digest,
        "validation_signature": validation_signature,
        "rejected_candidate": candidate.model_dump(mode="json"),
        "deterministic_issues": validation_details,
        "allowed_dispositions": allowed_dispositions,
        "current_blocker_policy": (
            "repair_only"
            if allowed_dispositions == ["retry_intent"]
            else "bounded_host_authority"
        ),
        "current_blocker_instruction": (
            "Diagnose and repair only deterministic_issues from the current "
            "validator phase. Do not stop for latent candidate fields that have "
            "not produced a supplied issue."
        ),
        "prior_attempts": compact_history,
        "prior_advisor_rejections": [],
        "catalog_contract": {
            "supported_inline_components": list(SUPPORTED_INLINE_COMPONENTS),
            "initial_open_construction_fronts": 1,
            "connect_goal_consumes_distinct_open_fronts": 2,
            "hard_constraints_require_deterministic_predicates": True,
            "advisor_may_author_or_accept_intent": False,
        },
    }
    last_error: Exception | None = None
    attempt_records: list[dict[str, Any]] = []
    call_count = 0
    for protocol_attempt in range(1, 3):
        response_payload: dict[str, Any] | None = None
        if not _intent_advisor_call_preserves_author_reserve(gemini):
            attempt_records.append(
                {
                    "role": "intent_repair_advisor",
                    "protocol_attempt": protocol_attempt,
                    "status": "skipped_author_call_reserve",
                    "response": None,
                    "host_rejection": {
                        "code": "AUTHOR_CALL_RESERVE",
                        "message": (
                            "advisor call skipped to reserve one remaining "
                            "Gemini call for the intent author"
                        ),
                    },
                }
            )
            return _IntentAdvisorOutcome(
                advice=None,
                call_count=call_count,
                error="advisor skipped to preserve the intent author call reserve",
                attempts=attempt_records,
                source="deterministic_evidence_only",
                fallback_used=True,
            )
        try:
            call_count += 1
            result = _call_structured(
                gemini,
                intent_repair_advisor_prompt(context),
                IntentRepairAdviceWire,
                part="intent_repair_advisor",
                thinking_level="high",
                system_instruction=intent_repair_advisor_system_instruction(),
            )
            response_payload = _intent_advisor_response_payload(result)
            advice = _coerce_intent_repair_advice(result)
            _validate_intent_repair_advice_authority(advice, validation_details)
            attempt_records.append(
                {
                    "role": "intent_repair_advisor",
                    "protocol_attempt": protocol_attempt,
                    "status": "accepted",
                    "response": response_payload,
                    "host_rejection": None,
                }
            )
            return _IntentAdvisorOutcome(
                advice=advice,
                call_count=call_count,
                error=None,
                attempts=attempt_records,
                source="primary_advisor",
                fallback_used=False,
            )
        except (GeminiBudgetError, GeminiConfigError) as exc:
            last_error = exc
            attempt_records.append(
                {
                    "role": "intent_repair_advisor",
                    "protocol_attempt": protocol_attempt,
                    "status": "provider_failed",
                    "response": response_payload,
                    "host_rejection": {
                        "code": type(exc).__name__,
                        "message": str(exc)[:2000],
                    },
                }
            )
            return _IntentAdvisorOutcome(
                advice=None,
                call_count=call_count,
                error=f"{type(exc).__name__}: {exc}"[:2000],
                attempts=attempt_records,
                source="deterministic_evidence_only",
                fallback_used=True,
            )
        except (
            GeminiRequestError,
            StructuredOutputError,
            HostContractValidationError,
            TypeError,
            ValueError,
        ) as exc:
            last_error = exc
            rejection = _intent_advisor_host_rejection(exc)
            record = {
                "role": "intent_repair_advisor",
                "protocol_attempt": protocol_attempt,
                "status": "host_rejected",
                "response": response_payload,
                "host_rejection": rejection,
            }
            attempt_records.append(record)
            context["prior_advisor_rejections"] = [
                {
                    "role": item["role"],
                    "rejected_response": item.get("response"),
                    "host_rejection": item.get("host_rejection"),
                }
                for item in attempt_records
                if item["status"] != "accepted"
            ]

    if settings.intent_repair_reviewer_enabled and getattr(
        gemini, "supports_intent_repair_reviewer", False
    ):
        reviewer_context = {
            **context,
            "reviewer_role": "repair_only_secondary_review",
            "allowed_dispositions": ["retry_intent"],
            "current_blocker_policy": "repair_only",
            "primary_attempts": attempt_records,
        }
        response_payload = None
        if not _intent_advisor_call_preserves_author_reserve(gemini):
            attempt_records.append(
                {
                    "role": "intent_repair_reviewer",
                    "protocol_attempt": 1,
                    "status": "skipped_author_call_reserve",
                    "response": None,
                    "host_rejection": {
                        "code": "AUTHOR_CALL_RESERVE",
                        "message": (
                            "reviewer call skipped to reserve one remaining "
                            "Gemini call for the intent author"
                        ),
                    },
                }
            )
            return _IntentAdvisorOutcome(
                advice=None,
                call_count=call_count,
                error="reviewer skipped to preserve the intent author call reserve",
                attempts=attempt_records,
                source="deterministic_evidence_only",
                fallback_used=True,
            )
        try:
            call_count += 1
            result = _call_structured(
                gemini,
                intent_repair_reviewer_prompt(reviewer_context),
                IntentRepairAdviceWire,
                part="intent_repair_reviewer",
                thinking_level="high",
                system_instruction=intent_repair_reviewer_system_instruction(),
            )
            response_payload = _intent_advisor_response_payload(result)
            advice = _coerce_intent_repair_advice(result)
            _validate_intent_repair_advice_authority(advice, validation_details)
            if advice.disposition != "retry_intent":
                raise _IntentAdvisorAuthorityError(
                    "REVIEWER_MUST_REPAIR",
                    "secondary intent reviewer may only return retry_intent",
                    details={"actual_disposition": advice.disposition},
                )
            attempt_records.append(
                {
                    "role": "intent_repair_reviewer",
                    "protocol_attempt": 1,
                    "status": "accepted",
                    "response": response_payload,
                    "host_rejection": None,
                }
            )
            return _IntentAdvisorOutcome(
                advice=advice,
                call_count=call_count,
                error=None,
                attempts=attempt_records,
                source="secondary_reviewer",
                fallback_used=True,
            )
        except (
            GeminiBudgetError,
            GeminiConfigError,
            GeminiRequestError,
            StructuredOutputError,
            HostContractValidationError,
            TypeError,
            ValueError,
        ) as exc:
            last_error = exc
            attempt_records.append(
                {
                    "role": "intent_repair_reviewer",
                    "protocol_attempt": 1,
                    "status": "failed",
                    "response": response_payload,
                    "host_rejection": _intent_advisor_host_rejection(exc),
                }
            )

    return _IntentAdvisorOutcome(
        advice=None,
        call_count=call_count,
        error=(
            f"{type(last_error).__name__}: {last_error}"[:2000]
            if last_error is not None
            else "intent validation advisors failed without an exception"
        ),
        attempts=attempt_records,
        source="deterministic_evidence_only",
        fallback_used=True,
    )


def _intent_advisor_response_payload(result: Any) -> dict[str, Any] | None:
    """Capture provider-wire JSON before stricter host semantic coercion."""

    if isinstance(result, (IntentRepairAdviceWire, IntentRepairAdvice)):
        return result.model_dump(mode="json")
    return None


def _intent_advisor_call_preserves_author_reserve(gemini: Any) -> bool:
    """Use an advisor only when one Gemini call remains for intent authoring."""

    remaining = getattr(gemini, "remaining_call_budget", None)
    if not callable(remaining):
        return True
    try:
        return int(remaining()) > 1
    except (TypeError, ValueError):
        return True


def _coerce_intent_repair_advice(result: Any) -> IntentRepairAdvice:
    if isinstance(result, IntentRepairAdviceWire):
        return IntentRepairAdvice.model_validate(result.model_dump(mode="python"))
    if isinstance(result, IntentRepairAdvice):
        return result
    raise TypeError(
        "Intent validation advisor returned an unexpected type: "
        f"{type(result).__name__}"
    )


def _intent_advisor_host_rejection(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, _IntentAdvisorAuthorityError):
        return {
            "code": exc.code,
            "type": type(exc).__name__,
            "message": str(exc)[:2000],
            "details": exc.details,
        }
    payload: dict[str, Any] = {
        "code": type(exc).__name__,
        "type": type(exc).__name__,
        "message": str(exc)[:2000],
    }
    if isinstance(exc, HostContractValidationError):
        payload["provider_payload"] = exc.payload
    return payload


def _allowed_intent_advisor_dispositions(
    validation_details: list[dict[str, Any]],
) -> list[str]:
    issue_codes = {
        str(item.get("issue_code"))
        for item in validation_details
        if item.get("issue_code")
    }
    capability_issue_codes = {
        "UNSUPPORTED_REQUIRED_COMPONENT",
        "UNSUPPORTED_HARD_CONSTRAINT",
    }
    capability_terminal_proven = bool(
        issue_codes
        and issue_codes.issubset(capability_issue_codes)
        and all(
            isinstance(item.get("actual"), dict)
            and item["actual"].get("source_provenance_complete") is True
            for item in validation_details
        )
    )
    if capability_terminal_proven:
        return ["retry_intent", "stop_contract_infeasible"]
    if issue_codes and all("VALIDATOR_POLICY" in code for code in issue_codes):
        return ["retry_intent", "escalate_validator_review"]
    return ["retry_intent"]


def _host_authorized_intent_terminal_reason(
    advice: IntentRepairAdvice,
    validation_details: list[dict[str, Any]],
) -> str | None:
    """Convert an advisor proposal into a terminal host decision, if proven."""

    if advice.disposition == "retry_intent":
        return None
    if advice.disposition in _allowed_intent_advisor_dispositions(validation_details):
        return advice.disposition
    return None


def _validate_intent_repair_advice_authority(
    advice: IntentRepairAdvice,
    validation_details: list[dict[str, Any]],
) -> None:
    """Reject advice that tries to turn an obviously repairable miss terminal."""

    issue_codes = {
        str(item.get("issue_code"))
        for item in validation_details
        if item.get("issue_code")
    }
    allowed_dispositions = _allowed_intent_advisor_dispositions(validation_details)
    if advice.disposition in allowed_dispositions:
        return
    capability_codes = {
        "UNSUPPORTED_REQUIRED_COMPONENT",
        "UNSUPPORTED_HARD_CONSTRAINT",
    }
    if advice.disposition == "stop_contract_infeasible":
        only_capability_issues = bool(
            issue_codes and issue_codes.issubset(capability_codes)
        )
        code = (
            "TERMINAL_SOURCE_PROVENANCE_REQUIRED"
            if only_capability_issues
            else "CURRENT_BLOCKER_REQUIRES_REPAIR"
        )
        raise _IntentAdvisorAuthorityError(
            code,
            (
                "stop_contract_infeasible requires only current unsupported "
                "capability issues with deterministic source provenance"
            ),
            details={
                "issue_codes": sorted(issue_codes),
                "actual_disposition": advice.disposition,
                "allowed_dispositions": allowed_dispositions,
            },
        )
    if advice.disposition == "stop_futile_retry":
        raise _IntentAdvisorAuthorityError(
            "STAGNATION_IS_HOST_OWNED",
            "stop_futile_retry is authorized only by the host repeat counter",
            details={"issue_codes": sorted(issue_codes)},
        )
    if advice.disposition == "escalate_validator_review" and not any(
        "VALIDATOR_POLICY" in code for code in issue_codes
    ):
        raise _IntentAdvisorAuthorityError(
            "VALIDATOR_ESCALATION_NOT_CORROBORATED",
            "validator escalation requires a current validator-policy issue",
            details={"issue_codes": sorted(issue_codes)},
        )
    raise _IntentAdvisorAuthorityError(
        "CURRENT_BLOCKER_REQUIRES_REPAIR",
        "the supplied deterministic issues authorize only retry_intent",
        details={
            "issue_codes": sorted(issue_codes),
            "actual_disposition": advice.disposition,
            "allowed_dispositions": allowed_dispositions,
        },
    )


def _intent_repair_diagnostic(exc: Exception) -> str:
    """Return actionable validation facts without re-anchoring malformed JSON."""

    if isinstance(exc, _IntentScopeValidationError):
        return json.dumps(
            [
                {
                    "issue_code": issue.issue_code,
                    "check_name": issue.check_name,
                    "message": issue.message,
                    "expected": issue.expected,
                    "actual": issue.actual,
                }
                for issue in exc.issues
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )[:8000]
    if isinstance(exc, _IntentSafetyValidationError):
        return json.dumps(
            exc.diagnostics,
            ensure_ascii=False,
            separators=(",", ":"),
        )[:8000]
    if isinstance(exc, StructuredOutputIncompleteError):
        return (
            "provider structured generation was incomplete: "
            f"status={exc.status}, output_limit={exc.output_limit}, "
            f"output_tokens={exc.output_tokens}, thought_tokens={exc.thought_tokens}"
        )[:1200]
    if isinstance(exc, (StructuredOutputError, HostContractValidationError)):
        cause = exc.cause
        if hasattr(cause, "errors"):
            try:
                details = cause.errors(include_url=False)[:8]
            except (TypeError, ValueError):
                details = []
            if details:
                compact = [
                    {
                        "location": list(item.get("loc") or []),
                        "message": str(item.get("msg") or "")[:300],
                        "type": item.get("type"),
                    }
                    for item in details
                ]
                return json.dumps(compact, ensure_ascii=False)[:1200]
        return f"structured output was invalid: {type(cause).__name__}"[:1200]
    return str(exc)[:1200]


def _intent_candidate_digest(intent: IntentResult) -> str:
    """원문 JSON을 저장하지 않고 동일 후보 여부를 감사할 digest를 만든다."""

    payload = json.dumps(
        intent.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _intent_repair_guidance(diagnostic: str) -> str:
    """반복하기 쉬운 Intent 오류를 해당 typed field의 수정 규칙으로 바꾼다."""

    normalized = diagnostic.lower()
    guidance: list[str] = []
    if "sequential heading contradiction" in normalized:
        guidance.append(
            "Re-simulate the serial tangent after every turn. A move has a "
            "global cardinal direction and may be used only when that direction "
            "already matches the incoming tangent. For a straight run following "
            "a non-cardinal signed-plane turn, use type=route, path_kind=line, "
            "with a length-only geometry_contract and omit direction and "
            "terminal_axis so it inherits the incoming tangent. Preserve every "
            "authored length and turn; do not reset later segments to start_axis."
        )
    if "max_extent requires axis and value" in normalized:
        guidance.append(
            "For each max_extent constraint return exactly constraint_id, "
            "type=max_extent, one cardinal axis, and one scalar value. Do not "
            "attach minimum/maximum vectors. If the requirement is an exact "
            "shape diameter rather than an upper bound, realize it in the route "
            "geometry instead of inventing max_extent."
        )
    if "bounding_box requires minimum and maximum" in normalized:
        guidance.append(
            "For each bounding_box constraint return exactly constraint_id, "
            "type=bounding_box, minimum XYZ, and maximum XYZ. Do not use axis "
            "or scalar value fields."
        )
    if "bounding_box minimum must be below maximum" in normalized:
        guidance.append(
            "For a user-authored bounding box, every minimum XYZ component must "
            "be strictly lower than its matching maximum component. A planar "
            "pipe solid still needs a non-zero normal-axis interval. If the user "
            "only gave an exact shape diameter/width and did not request an "
            "envelope or box, remove the invented bounding_box and realize that "
            "dimension solely in the route geometry."
        )
    if "expected_open_ports conflicts with target_behavior topology" in normalized:
        guidance.append(
            "Recompute expected_open_ports from the returned goal sequence. "
            "Ordinary serial route/turn/transition/connector goals preserve one "
            "open construction front, end consumes one, and connect consumes two "
            "distinct currently-open fronts. Never force zero merely because the "
            "requested shape is described as closed. If a single unbranched "
            "closed loop cannot be represented from the anchored START, keep the "
            "realizable derived count and preserve the explicit closed-loop demand "
            "as an unsupported: hard constraint so scope validation fails honestly."
        )
    if any(
        marker in normalized
        for marker in (
            "junction_blend_radius",
            "junction_inner_blend_radius",
            "junction_max_hub_radius",
        )
    ):
        guidance.append(
            "Copy every authored outer blend_radius, inner_blend_radius, and "
            "max_hub_radius directly onto its existing owning branch goal. Do "
            "not create a separate measurement-only branch goal."
        )
    if "consumes more open terminals than its prefix can produce" in normalized:
        guidance.append(
            "Remove the extra topology-consuming goal, merge its parameter fields "
            "into the intended existing branch goal, and recompute the open-port "
            "prefix after every goal before returning."
        )
    if "lost or altered explicit millimeter values" in normalized:
        guidance.append(
            "Place each missing nominal value in the typed field that owns the "
            "same physical requirement; never hide it in a new unrelated goal, "
            "coordinate, or rationale."
        )
    if "required_waypoints require path_kind=spline" in normalized:
        guidance.append(
            "Any route that keeps a waypoints geometry contract must set "
            "path_kind=spline so deterministic curvature preflight cannot be "
            "bypassed. If the user only requested a straight or diagonal manifold "
            "arm, remove the invented waypoints contract and use a length/direction "
            "contract instead."
        )
    if "direct required-anchor realization predicts" in normalized:
        guidance.append(
            "Repair the immutable intent geometry itself. Do not preserve a "
            "model-invented terminal_axis or append clustered lead-out points merely "
            "because they appeared in the rejected draft. Prefer the diagnostic's "
            "passing natural-final-chord option when the user did not explicitly "
            "author a terminal axis; otherwise replace the invented anchors with a "
            "few broadly separated anchors and recompute the whole chain."
        )
    if (
        "ordered required-waypoint polyline lower bound" in normalized
        or "deterministic spline centerline length" in normalized
    ):
        guidance.append(
            "A traveled route length cannot be shorter than its ordered waypoint "
            "polyline and must match the predicted spline length. For an ordinary "
            "manifold arm, remove model-invented waypoints and author a line "
            "length/direction contract. For a genuinely user-requested freeform "
            "path, replace the whole anchor chain and keep its computed length "
            "inside the user's allowed interval."
        )
    if "branch_angles conflict with its outlet axes" in normalized:
        guidance.append(
            "Branch angles are acute centerline-to-centerline angles. Recompute "
            "each outlet vector from the selected source-allowed angle (or select "
            "the matching angle for an already valid diagonal vector); do not keep "
            "an approximate midpoint angle beside an incompatible exact vector."
        )
    if "each-branch length range requires" in normalized:
        guidance.append(
            "Represent every terminal branch with a concrete source-allowed "
            "length. The START-side arm owns its route length; each junction's "
            "terminal branches must use outlet_contract mode=outlets with one "
            "length per outlet. Primary continuation stubs are not terminal arms."
        )
    if "different branch lengths" in normalized:
        guidance.append(
            "Choose at least two distinct terminal-arm lengths inside the authored "
            "inclusive range while keeping all four arms within that range."
        )
    if "junction_style='smooth_hub'" in normalized:
        guidance.append(
            "Set junction_style=smooth_hub on the existing branch goals; do not "
            "create extra goals or invent source blend dimensions."
        )
    if "expected_open_ports_source must be 'explicit'" in normalized:
        guidance.append(
            "Keep expected_open_ports=3 and mark its source explicit because the "
            "user directly named a four-port manifold."
        )
    if "terminal-arm axes violate the explicit acute branch-angle range" in normalized:
        guidance.append(
            "Treat 'from the main axis' as an acute global centerline relation. "
            "Choose a diagonal START/start_axis and terminal outlet vectors inside "
            "the authored angle interval. Do not force those values into the "
            "junction's inlet-relative branch_angles field unless they are also "
            "mathematically the same local angle."
        )
    if "junction width/pipe diameter is a full transverse size" in normalized:
        guidance.append(
            "Remove max_hub_radius values copied from a width or diameter. Keep "
            "the width statement in the branch notes/design contract and leave "
            "non-authored hub/blend radii for candidate-level action selection."
        )
    return " ".join(guidance)


def _is_spline_intent_safety_diagnostic(diagnostic: str) -> bool:
    """Detect semantic spline failures that need unconstrained repair geometry."""

    normalized = diagnostic.lower()
    return any(
        marker in normalized
        for marker in (
            "direct required-anchor realization",
            "required_waypoints require path_kind=spline",
            "relative first waypoint",
            "spline waypoint chain",
            "direction change is spread",
            "180-degree cusp",
            "ordered required-waypoint polyline lower bound",
            "deterministic spline centerline length",
        )
    )


_MM_NUMBER_TEXT = (
    r"[+\-−]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+\-]?\d+)?"
)

_EXPLICIT_MM_VALUE = re.compile(
    r"(?<![\d.,])"
    rf"({_MM_NUMBER_TEXT})"
    r"\s*(?:mm|㎜|밀리미터)(?![A-Za-z])",
    re.IGNORECASE,
)

# A range commonly writes the unit only once (``85–100 mm``).  The ordinary
# value expression consequently sees only the upper bound.  Keep the range as
# a first-class source contract so both bounds enter the model vocabulary while
# the intent validator can accept a typed value *inside* the interval instead
# of incorrectly demanding two independent exact dimensions.
_EXPLICIT_MM_RANGE = re.compile(
    rf"(?<![\d.,])(?P<start>{_MM_NUMBER_TEXT})\s*"
    r"(?:(?:mm|㎜|밀리미터)\s*)?"
    r"(?:[–—~〜]|-(?!\s*[+\-−])|\bto\b|부터)\s*"
    rf"(?P<end>{_MM_NUMBER_TEXT})\s*(?:mm|㎜|밀리미터)(?![A-Za-z])",
    re.IGNORECASE,
)

_EXPLICIT_DEGREE_RANGE = re.compile(
    rf"(?<![\d.,])(?P<start>{_MM_NUMBER_TEXT})\s*"
    r"(?:(?:degrees?|deg|°|도)\s*)?"
    r"(?:[–—~〜]|-(?!\s*[+\-−])|\bto\b|부터)\s*"
    rf"(?P<end>{_MM_NUMBER_TEXT})\s*(?:degrees?|deg|°|도)",
    re.IGNORECASE,
)

_ANY_NUMERIC_VALUE = re.compile(
    r"(?<![\d.,])"
    r"([+\-−]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+\-]?\d+)?)"
    r"(?![\d.,])"
)

_VECTOR_NUMBER = r"[+\-−]?(?:(?:\d+)(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?"
_EXPLICIT_VECTOR3 = re.compile(
    rf"[\[(]\s*(?P<x>{_VECTOR_NUMBER})\s*,\s*"
    rf"(?P<y>{_VECTOR_NUMBER})\s*,\s*"
    rf"(?P<z>{_VECTOR_NUMBER})\s*[\])]"
)

# A tuple used after one of these phrases is a direction ratio, not an exact
# coordinate.  Keep this deliberately narrower than generic words such as
# ``axis`` or ``heading``: those words can also describe an exact authored
# vector, while these phrases explicitly permit a positive scalar multiple.
_PROPORTIONAL_VECTOR_PREFIX = re.compile(
    r"(?:proportional\s+to|parallel\s+(?:to|with)|"
    r"positive\s+(?:scalar\s+)?multiple\s+of)"
    r"\s*(?:(?:the\s+)?(?:(?:global|local)\s+)?"
    r"(?:vector|axis|direction)\s*)?$",
    re.IGNORECASE,
)
_PROPORTIONAL_VECTOR_SUFFIX = re.compile(
    r"^\s*(?:(?:에|와|과)\s*)?(?:비례|평행)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _ProportionalDirectionContract:
    value: tuple[float, float, float]
    role: str


@dataclass(frozen=True)
class _IntentDirectionCandidate:
    semantic_id: str
    role: str
    value: tuple[float, float, float]


@dataclass(frozen=True)
class _ExplicitMMRange:
    """사용자가 직접 작성한 포괄적 millimeter 구간이다."""

    authored_start: float
    authored_end: float
    span: tuple[int, int]

    @property
    def minimum(self) -> float:
        return min(self.authored_start, self.authored_end)

    @property
    def maximum(self) -> float:
        return max(self.authored_start, self.authored_end)


@dataclass(frozen=True)
class _ExplicitAngleRange:
    """main centerline을 기준으로 사용자가 직접 쓴 acute 각도 범위다."""

    minimum: float
    maximum: float
    span: tuple[int, int]


_FOUR_CORNER_TERMINAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:upper|top)[\s-]*left|좌상|왼쪽\s*(?:위|상단)",
        r"(?:lower|bottom)[\s-]*left|좌하|왼쪽\s*(?:아래|하단)",
        r"(?:upper|top)[\s-]*right|우상|오른쪽\s*(?:위|상단)",
        r"(?:lower|bottom)[\s-]*right|우하|오른쪽\s*(?:아래|하단)",
    )
)

_EXPLICIT_FOUR_PORT = re.compile(
    r"(?:\bfour[\s-]*port\b|\b4[\s-]*port\b|"
    r"\b(?:all\s+)?four\s+(?:branch\s+)?ends?\b|"
    r"4\s*(?:개\s*)?(?:포트|개구|끝단))",
    re.IGNORECASE,
)
_SMOOTH_JUNCTION_REQUEST = re.compile(
    r"(?:smooth(?:ly)?[^.;\n]{0,45}(?:Y[\s-]*junction|junction|branches)|"
    r"(?:Y[\s-]*junction|junction)[^.;\n]{0,45}smooth|"
    r"no\s+sharp\s+Boolean\s+intersections?|"
    r"부드러운?[^.;\n]{0,30}(?:Y\s*분기|접합|분기))",
    re.IGNORECASE,
)
_DIFFERENT_BRANCH_LENGTHS = re.compile(
    r"(?:different|unequal|asymmetric|slightly\s+different)"
    r"[^.;\n]{0,40}branch\s+lengths?",
    re.IGNORECASE,
)
_JUNCTION_WIDTH_REFERENCE = re.compile(
    rf"\bjunction\s+width\b[^.;\n]{{0,60}}?"
    rf"(?P<value>{_MM_NUMBER_TEXT})\s*(?:mm|㎜|밀리미터)",
    re.IGNORECASE,
)

_MEASUREMENT_NUMBER = (
    r"(?P<value>[+\-−]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+\-]?\d+)?)"
)
_MEASUREMENT_UNIT = r"(?:mm|㎜|밀리미터)"

# These patterns are intentionally conservative.  They establish a semantic
# field binding only when the source text contains a high-confidence domain
# phrase.  Unclassified measurements are still protected by the general
# multiplicity-preserving check below.
_ANCHORED_MM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "diameter_in_reference",
        re.compile(
            rf"(?:outer\s+diameter|outside\s+diameter|\bOD\b)"
            rf"[^.;\n]{{0,50}}?\bfrom\s*(?:about\s+|roughly\s+|approximately\s+)?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "wall_thickness_in_reference",
        re.compile(
            rf"(?:wall\s+thickness|\bWT\b)"
            rf"[^.;\n]{{0,50}}?\bfrom\s*(?:about\s+|roughly\s+|approximately\s+)?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "route_rise",
        re.compile(
            rf"\b(?:rise|rises|rising)\b[^.;\n]{{0,35}}?\bby\s*"
            rf"(?:about\s+|roughly\s+|approximately\s+)?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "diameter_out",
        re.compile(
            rf"(?:외경|직경)(?:을|를)\s*{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}"
            rf"(?=(?:(?!(?:외경|직경)).){{0,60}}?(?:로\s*)?"
            rf"(?:줄|늘|변경|감소|증가))",
            re.IGNORECASE,
        ),
    ),
    (
        "wall_thickness_out",
        re.compile(
            rf"두께(?:를|을)\s*{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}"
            rf"(?=(?:(?!두께).){{0,50}}?(?:로\s*)?"
            rf"(?:줄|늘|변경|감소|증가))",
            re.IGNORECASE,
        ),
    ),
    (
        "diameter_out",
        re.compile(
            rf"(?:reduce|decrease|increase|change)[^.;\n]{{0,60}}?"
            rf"(?:outer\s+diameter|outside\s+diameter|\bOD\b)"
            rf"[^.;\n]{{0,40}}?\bto\s*{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "wall_thickness_out",
        re.compile(
            rf"(?:reduce|decrease|increase|change)[^.;\n]{{0,80}}?"
            rf"(?:wall\s+thickness|\bWT\b)"
            rf"[^.;\n]{{0,40}}?\bto\s*{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "transition_length",
        re.compile(
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}\s*(?:의\s*)?구간에서"
            rf"(?=[\s\S]{{0,100}}(?:줄|늘|변경|외경|직경|두께))",
            re.IGNORECASE,
        ),
    ),
    (
        "transition_length",
        re.compile(
            rf"(?:transition|taper|reducer)[^.;\n]{{0,50}}?"
            rf"(?:length|over)\s*(?:of\s*)?{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "connector_length",
        re.compile(
            rf"(?:길이|length(?:\s+of)?)\s*{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}"
            rf"\s*(?:의\s*)?(?:coupling|커플링|flange|플랜지|union|유니온|valve|밸브)",
            re.IGNORECASE,
        ),
    ),
    (
        "connector_length",
        re.compile(
            rf"(?:coupling|커플링|flange|플랜지|union|유니온|valve|밸브)"
            rf"[^.;\n]{{0,30}}?(?:길이|length)\s*{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "junction_blend_radius",
        re.compile(
            rf"(?:outer(?:[-\s]+surface)?[-\s]+blend\s+radius|"
            rf"외부\s*블렌드\s*(?:반경|반지름))\s*(?:of\s+|[:=]\s*)?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "junction_inner_blend_radius",
        re.compile(
            rf"(?:inner(?:[-\s]+bore)?[-\s]+blend\s+radius|"
            rf"내부(?:\s*보어)?\s*블렌드\s*(?:반경|반지름))\s*"
            rf"(?:of\s+|[:=]\s*)?{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "junction_max_hub_radius",
        re.compile(
            rf"(?:maximum|max)\s+hub\s+radius\s*(?:of\s+|[:=]\s*)?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "straight_length",
        re.compile(
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}\s*(?:만큼\s*)?"
            rf"(?:직진|straight(?:\s+(?:run|section))?)",
            re.IGNORECASE,
        ),
    ),
    (
        "global_inner_diameter",
        re.compile(
            rf"(?:내경|내부\s*직경|inner\s+diameter|inside\s+diameter|"
            rf"\bID\b|bore\s+diameter)"
            rf"\s*(?:(?:은|는|이|가|:|=)\s*|of\s+)?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "global_outer_diameter",
        re.compile(
            rf"(?:외경|외부\s*직경|outer\s+diameter|outside\s+diameter|\bOD\b)"
            rf"\s*(?:(?:은|는|이|가|:|=)\s*|of\s+)?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "global_outer_diameter",
        re.compile(
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}\s*"
            rf"(?:의\s*)?(?:pipe|tube)\s+(?:outer\s+|outside\s+)?"
            rf"(?:diameter|OD\b|직경|외경)",
            re.IGNORECASE,
        ),
    ),
    (
        "global_wall_thickness",
        re.compile(
            rf"(?:두께|wall\s+thickness|\bWT\b)"
            rf"\s*(?:(?:은|는|이|가|:|=)\s*|of\s+)?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}",
            re.IGNORECASE,
        ),
    ),
    (
        "global_wall_thickness",
        re.compile(
            rf"(?:all(?:\s+[A-Za-z0-9-]+){{0,4}}\s+ends?|"
            rf"every\s+(?:open\s+)?end|"
            rf"각\s*(?:개방\s*)?(?:끝단|단부)|모든\s*(?:개방\s*)?(?:끝|끝단))"
            rf"[^.;\n]{{0,60}}?"
            rf"{_MEASUREMENT_NUMBER}\s*{_MEASUREMENT_UNIT}\s*"
            rf"(?:의\s*)?(?:두께|wall\s+thickness|\bWT\b)",
            re.IGNORECASE,
        ),
    ),
)

# These roles describe one global section property. Repeating the same value in
# prose (for example, declaring 3.5 mm once and later asking every open end to
# show that 3.5 mm wall) reinforces one contract; it is not a second physical
# dimension. Different values are deliberately retained so contradictions fail.
_SINGLETON_MM_ROLES = {
    "global_outer_diameter",
    "global_inner_diameter",
    "global_wall_thickness",
}

_GOAL_LENGTH_FIELDS = (
    "length",
    "bend_radius",
    "blend_radius",
    "inner_blend_radius",
    "max_hub_radius",
    "diameter_out",
    "wall_thickness_out",
    "transition_length",
    "branch_outer_diameter",
    "branch_wall_thickness",
    "termination_thickness",
    "minimum_curvature_radius",
)

_COMPONENT_LENGTH_FIELDS = (
    "body_outer_diameter",
    "body_start_offset",
    "body_length",
    "flange_bolt_circle_diameter",
    "flange_bolt_hole_diameter",
    "union_ring_outer_diameter",
    "union_ring_length",
    "actuator_diameter",
    "actuator_height",
)


def _canonicalize_dependent_intent_geometry(
    intent: IntentResult,
    settings: Settings,
) -> IntentResult:
    """LLM 의도에서 수학적으로 종속된 방향 프레임만 정규화한다.

    signed bend의 plane normal처럼 독립 설계값이 아닌 프레임 표현은 같은
    의미를 유지한 채 정규화할 수 있다. 반면 spline waypoint는 LLM이 선택한
    독립 형상 파라미터이므로 이 함수가 확대하거나 교체하지 않는다. 곡률이
    부족하면 뒤의 ``_validate_intent_safety``가 필요한 반경과 현재 계산값을
    진단해 LLM에 새 waypoint를 요청한다.
    """

    try:
        current = normalize(vec(intent.start_axis))
    except ValueError:
        return intent

    goals = list(intent.target_behavior)

    for index, original_goal in enumerate(goals):
        if index > 0:
            previous_goal_id = goals[index - 1].goal_id
            if original_goal.allow_parallel or (
                original_goal.depends_on_goal_ids
                and previous_goal_id not in original_goal.depends_on_goal_ids
            ):
                break

        goal = original_goal
        if goal.type == "move" and goal.direction is not None:
            current = direction_to_vector(goal.direction)
            continue

        if goal.type == "turn" and goal.angle is not None:
            if goal.direction is not None:
                current = direction_to_vector(goal.direction)
                continue
            if goal.plane_normal is None:
                break
            try:
                plane_normal, _initial, terminal = canonical_circular_arc_frame(
                    current,
                    vec(goal.plane_normal),
                    float(goal.angle),
                )
            except ValueError:
                break
            goal = goal.model_copy(update={"plane_normal": plane_normal})
            goals[index] = goal
            current = terminal
            continue

        if goal.type == "route":
            if (
                goal.path_kind == "spline"
                and goal.waypoint_frame == "relative_to_target"
                and goal.required_waypoints
            ):
                points = [vec(point) for point in goal.required_waypoints]
                try:
                    final_tangent = (
                        normalize(vec(goal.terminal_axis))
                        if goal.terminal_axis is not None
                        else normalize(
                            points[0]
                            if len(points) == 1
                            else sub(points[-1], points[-2])
                        )
                    )
                except ValueError:
                    break

                # waypoint의 안전성은 아래 validator가 계산한다. 여기서 자동
                # 확대하면 원본 LLM 선택과 실제 CAD 입력을 구분할 수 없으므로
                # 어떤 scale policy가 표시돼 있더라도 값을 그대로 보존한다.
                current = final_tangent
                continue

            if goal.path_kind == "line":
                if goal.direction is not None:
                    current = direction_to_vector(goal.direction)
                # A length-only line inherits the incoming tangent.  It is the
                # general representation for a straight after a non-cardinal
                # signed turn; do not stop heading propagation here.
                continue
            if goal.terminal_axis is not None:
                try:
                    current = normalize(vec(goal.terminal_axis))
                except ValueError:
                    pass
                continue
            break

        if goal.type == "diameter_change":
            continue
        if goal.type == "connector":
            continue
        if goal.type in {"branch", "connect", "end"}:
            break

    return intent.model_copy(update={"target_behavior": goals})


def _validate_intent_safety(
    prompt: str,
    intent: IntentResult,
    settings: Settings,
) -> None:
    """Reject internally valid intents that cannot preserve/model the request.

    This is deliberately a validator, not a repairer.  The LLM must return a
    corrected immutable contract on the next intent attempt.
    """

    issues: list[str] = []

    # waypoint 계약은 이후 resolver에서 항상 spline으로 구현된다. Intent가
    # path_kind를 비워 두면 아래 곡률 preflight를 우회한 뒤 action 단계에서
    # 갑자기 spline이 선택되어, 고칠 수 없는 불변 좌표를 반복 재시도하게 된다.
    # 이 모순은 후보 상태를 만들기 전에 Intent 작성자에게 돌려보낸다.
    for goal_index, goal in enumerate(intent.target_behavior):
        if (
            goal.type == "route"
            and goal.required_waypoints
            and goal.path_kind != "spline"
        ):
            goal_label = goal.goal_id or f"target_behavior[{goal_index}]"
            issues.append(
                f"{goal_label} required_waypoints require path_kind=spline; "
                f"actual path_kind={goal.path_kind!r}. Do not leave the path kind "
                "implicit because waypoint curvature must be preflighted before "
                "the immutable contract is accepted"
            )
    undersized = [
        (path, value)
        for path, value in _positive_intent_dimensions(intent)
        if value <= settings.modeling_tolerance
    ]
    if undersized:
        rendered = ", ".join(path for path, _value in undersized[:12])
        issues.append(
            "these physical dimension fields must exceed modeling_tolerance "
            f"{settings.modeling_tolerance:.12g} mm: {rendered}. Do not reuse "
            "the rejected numeric spelling; restore the concise value from the "
            "user request"
        )

    anchored_values = _anchored_mm_values(prompt)
    candidate_values = _anchored_intent_values(intent)
    explicit_values = _exact_mm_contract_values(prompt)
    preserved_values = list(_intent_metric_values(intent))
    # A taper's authored `from` section is inherited graph state, not a second
    # independent field on diameter_change.  Add that derived evidence only
    # when the source text actually contains a typed input-section reference;
    # an unrelated repeated number therefore still cannot borrow it.
    for role in (
        "diameter_in_reference",
        "wall_thickness_in_reference",
        "global_inner_diameter",
    ):
        if role in anchored_values:
            preserved_values.extend(candidate_values.get(role, []))
    missing_values: list[float] = []
    for value in explicit_values:
        match_index = next(
            (
                index
                for index, candidate in enumerate(preserved_values)
                if _same_metric_value(value, candidate)
            ),
            None,
        )
        if match_index is None:
            missing_values.append(value)
        else:
            # Preserve source multiplicity: one typed value cannot account for
            # two independent measurements authored by the user.
            preserved_values.pop(match_index)
    if missing_values:
        issues.append(
            "intent lost or altered explicit millimeter values from the user "
            f"request: {missing_values}"
        )

    # A source range is one inclusive contract, not two exact dimensions.  The
    # LLM may select any typed physical measurement inside it; the resolver and
    # later geometry checks still operate on that explicit chosen value.
    range_candidates = _intent_range_candidate_values(intent)
    for source_range in _explicit_mm_ranges(prompt):
        if not any(
            source_range.minimum <= candidate <= source_range.maximum
            for candidate in range_candidates
        ):
            issues.append(
                "intent has no typed physical dimension inside explicit "
                "millimeter range "
                f"[{source_range.minimum}, {source_range.maximum}]"
            )

    # ``each branch ... 85–100 mm long``은 범위 안의 숫자 하나만 어디엔가
    # 존재하면 되는 계약이 아니다. START가 대표하는 첫 terminal arm과 각
    # binary junction의 최종 branch outlet 모두가 독립 length를 가져야 한다.
    for source_range in _explicit_branch_length_ranges(prompt):
        arm_lengths = _terminal_arm_length_contracts(intent)
        missing_arm_indexes = [
            index for index, value in enumerate(arm_lengths) if value is None
        ]
        outside = [
            {"arm_index": index, "length": value}
            for index, value in enumerate(arm_lengths)
            if value is not None
            and not (source_range.minimum <= value <= source_range.maximum)
        ]
        if missing_arm_indexes or outside:
            issues.append(
                "each-branch length range requires one typed terminal-arm length "
                f"inside [{source_range.minimum}, {source_range.maximum}] mm for "
                "START plus every final branch outlet; "
                f"actual={arm_lengths}, missing_arm_indexes={missing_arm_indexes}, "
                f"outside={outside}. Use branch outlet_contract mode=outlets when "
                "individual outlet lengths must be authored"
            )
        elif _DIFFERENT_BRANCH_LENGTHS.search(prompt):
            distinct = {
                round(float(value), 9) for value in arm_lengths if value is not None
            }
            if len(distinct) < 2:
                issues.append(
                    "the user requested different branch lengths, but every typed "
                    f"terminal-arm length is {arm_lengths[0]:.6g} mm; select at least "
                    "two distinct values inside the authored range"
                )

    for angle_range in _explicit_branch_angle_ranges(prompt):
        try:
            actual_angles = _main_axis_terminal_arm_angles(intent)
        except ValueError as exc:
            issues.append(
                "could not derive terminal-arm axes for the explicit branch-angle "
                f"range: {exc}"
            )
            continue
        outside_angles = [
            {"arm_index": index, "angle_degrees": angle}
            for index, angle in enumerate(actual_angles)
            if not (angle_range.minimum - 1e-6 <= angle <= angle_range.maximum + 1e-6)
        ]
        if outside_angles:
            issues.append(
                "terminal-arm axes violate the explicit acute branch-angle range "
                f"[{angle_range.minimum}, {angle_range.maximum}] degrees from the "
                f"main axis: actual={actual_angles}, outside={outside_angles}. "
                "Encode the main-axis relation in terminal outlet vectors/start "
                "arm heading; do not copy a main-axis angle into an incompatible "
                "inlet-relative branch_angles field"
            )

    for role, expected in anchored_values.items():
        missing_for_role = _ordered_missing_values(
            expected,
            candidate_values.get(role, []),
        )
        if missing_for_role:
            issues.append(
                f"intent moved or altered source measurements bound to {role}: "
                f"{missing_for_role}"
            )

    source_vector_contracts = _explicit_vector3_contracts(prompt)
    source_vectors = [
        value for value, is_direction in source_vector_contracts if not is_direction
    ]
    source_directions = _explicit_proportional_direction_contracts(prompt)
    candidate_vectors = _intent_vector_values(intent)
    missing_vectors: list[list[float]] = []
    for source_vector in source_vectors:
        match_index = next(
            (
                index
                for index, candidate in enumerate(candidate_vectors)
                if all(
                    _same_metric_value(source_vector[axis], candidate[axis])
                    for axis in range(3)
                )
            ),
            None,
        )
        if match_index is None:
            missing_vectors.append([float(value) for value in source_vector])
        else:
            candidate_vectors.pop(match_index)
    if missing_vectors:
        issues.append(
            "intent lost or altered explicit XYZ vectors from the user request: "
            f"{missing_vectors[:12]}"
        )

    # Direction ratios such as ``proportional to (+2,-1,+1)`` may be authored
    # at any positive magnitude.  Validate those against typed intent axes and
    # waypoint chords rather than incorrectly requiring the ratio tuple to be
    # copied as an absolute coordinate or terminal_axis field.
    candidate_directions = _intent_direction_candidates(intent)
    missing_directions: list[dict[str, Any]] = []
    for source_direction in source_directions:
        match_index = next(
            (
                index
                for index, candidate in enumerate(candidate_directions)
                if _direction_roles_compatible(
                    source_direction.role,
                    candidate.role,
                )
                and _positive_parallel_direction(
                    source_direction.value,
                    candidate.value,
                )
            ),
            None,
        )
        if match_index is None:
            missing_directions.append(
                {
                    "role": source_direction.role,
                    "vector": [float(value) for value in source_direction.value],
                }
            )
        else:
            candidate_directions.pop(match_index)
    if missing_directions:
        issues.append(
            "intent lost or altered explicit XYZ vectors from the user request "
            "that were authored as proportional directions: "
            f"{missing_directions[:12]}"
        )

    issues.extend(_sequential_heading_issues(intent, settings))
    issues.extend(_branch_successor_spline_issues(intent, settings))
    issues.extend(_sequential_position_issues(intent, settings))
    issues.extend(_branch_angle_vector_issues(intent))

    if _EXPLICIT_FOUR_PORT.search(prompt):
        if intent.expected_open_ports != 3:
            issues.append(
                "an explicit four-port manifold rooted at START requires exactly "
                "3 generated downstream open ports (START is the fourth physical port)"
            )
        if intent.expected_open_ports_source != "explicit":
            issues.append(
                "four-port is explicit source topology, so "
                "expected_open_ports_source must be 'explicit', not "
                f"{intent.expected_open_ports_source!r}"
            )

    if _SMOOTH_JUNCTION_REQUEST.search(prompt):
        wrong_styles = [
            goal.goal_id or f"target_behavior[{index}]"
            for index, goal in enumerate(intent.target_behavior)
            if goal.type == "branch" and goal.junction_style != "smooth_hub"
        ]
        if wrong_styles:
            issues.append(
                "the user explicitly requested smooth Y-junctions/no sharp Boolean "
                "intersections; every branch goal must set "
                f"junction_style='smooth_hub'. Missing goals: {wrong_styles}"
            )

    width_references = [
        _metric_number(match.group("value"))
        for match in _JUNCTION_WIDTH_REFERENCE.finditer(prompt)
    ]
    copied_widths = [
        {
            "goal_id": goal.goal_id or f"target_behavior[{index}]",
            "max_hub_radius": float(goal.max_hub_radius),
        }
        for index, goal in enumerate(intent.target_behavior)
        if goal.type == "branch"
        and goal.max_hub_radius is not None
        and any(
            _same_metric_value(float(goal.max_hub_radius), width)
            for width in width_references
        )
    ]
    if copied_widths:
        issues.append(
            "junction width/pipe diameter is a full transverse size, not a hub "
            "radius; do not copy the source width directly into max_hub_radius. "
            f"width_references={width_references}, invalid_mappings={copied_widths}. "
            "Leave non-authored hub/blend radii unset for action planning"
        )

    # A branch placed directly on START must not emit an outlet back over the
    # consumed inlet ray. Apart from being semantically duplicative, that exact
    # topology is unstable in OCC's junction Boolean. Force the intent model to
    # author a positive-length run from a named terminal into the junction.
    first_goal = intent.target_behavior[0] if intent.target_behavior else None
    if first_goal is not None and first_goal.type == "branch":
        if all(pattern.search(prompt) for pattern in _FOUR_CORNER_TERMINAL_PATTERNS):
            issues.append(
                "a four-corner terminal manifold must start with a positive-length "
                "route from the anchored remote START arm to the first junction; "
                "the first goal cannot place that junction directly on START"
            )
        try:
            start_axis = normalize(vec(intent.start_axis))
        except ValueError:
            start_axis = None
        if start_axis is not None:
            reverse_vectors: list[tuple[float, float, float]] = []
            for raw_vector in first_goal.required_outlet_vectors:
                reverse_vectors.append(normalize(vec(raw_vector)))
            for outlet in first_goal.required_outlets:
                reverse_vectors.append(normalize(vec(outlet.axis)))
            for direction in first_goal.required_outlet_directions:
                reverse_vectors.append(direction_to_vector(direction))
            if any(
                dot(start_axis, candidate) <= -0.999 for candidate in reverse_vectors
            ):
                issues.append(
                    "the first branch recreates the anchored START arm with an "
                    "outlet opposite start_axis; add a positive-length route from "
                    "START to the junction and keep the START terminal out of all "
                    "downstream outlet contracts"
                )

    # Whole-program structural mistakes are intent-authoring errors, not
    # step-local numeric repair problems.  Only conflicts that continuous host
    # relaxation cannot fix (wrong source plane or coincident non-adjacent
    # centerlines) are returned to the LLM here.  Curvature, ordinary clearance
    # deficits and closure dimensions are solved after intent acceptance by the
    # deterministic ContractCore.
    issues.extend(
        structural_intent_issues(
            prompt,
            intent,
            modeling_tolerance=settings.modeling_tolerance,
        )
    )

    if issues:
        raise _IntentSafetyValidationError(issues)


def _predicted_c1_spline_minimum_radius(
    offsets: list[tuple[float, float, float]],
    initial_tangent: tuple[float, float, float],
    final_tangent: tuple[float, float, float] | None,
    *,
    modeling_tolerance: float,
) -> tuple[float, list[float]]:
    """Mirror the FreeCAD spline handle optimizer for intent preflight.

    Relative qualitative anchors are LLM-authored, but their feasibility is a
    dependent geometric calculation. Running the same scale-aware cubic model
    here prevents an impossible immutable route contract from reaching the
    paid step-repair loop.
    """

    try:
        prediction = predict_c1_spline(
            [(0.0, 0.0, 0.0), *[vec(point) for point in offsets]],
            normalize(initial_tangent),
            normalize(final_tangent) if final_tangent is not None else None,
            modeling_tolerance=modeling_tolerance,
        )
    except ValueError:
        return 0.0, []
    return prediction.minimum_radius, list(prediction.handle_factors)


def _sequential_heading_issues(
    intent: IntentResult,
    settings: Settings,
) -> list[str]:
    """Find linear-prefix heading contracts that no connected pipe can realize.

    Intent extraction is LLM-authored, but a straight run cannot change heading
    and a circular turn's endpoint separation is fixed by its sweep magnitude.
    Catching contradictions here gives the intent model a chance to repair the
    immutable agenda instead of making the step planner retry an impossible goal.
    Once topology forks or a freeform route has no terminal tangent, there is no
    single heading to simulate and this deliberately stops rather than guessing.
    """

    try:
        current = normalize(vec(intent.start_axis))
    except ValueError:
        return ["start_axis must be a finite non-zero sequential heading"]

    issues: list[str] = []
    current_outer_diameter = float(intent.global_spec.outer_diameter)
    for index, goal in enumerate(intent.target_behavior):
        if index > 0:
            previous_goal_id = intent.target_behavior[index - 1].goal_id
            if goal.allow_parallel or (
                goal.depends_on_goal_ids
                and previous_goal_id not in goal.depends_on_goal_ids
            ):
                # This is no longer a unique serial centerline, so one heading
                # cannot be propagated without guessing which branch is meant.
                break
        goal_label = goal.goal_id or f"target_behavior[{index}]"
        if goal.type == "move" and goal.direction is not None:
            desired = direction_to_vector(goal.direction)
            alignment = dot(current, desired)
            if alignment < 0.9999:
                issues.append(
                    f"sequential heading contradiction at {goal_label}: move.direction="
                    f"{goal.direction} is a straight-run heading but the incoming "
                    f"heading is {[round(value, 6) for value in current]} "
                    f"(dot={alignment:.6g}); align start_axis/the preceding outlet "
                    "or represent the heading change as a turn"
                )
            current = desired
            continue

        if goal.type == "turn" and goal.angle is not None:
            desired = (
                direction_to_vector(goal.direction)
                if goal.direction is not None
                else None
            )
            plane_normal = None
            if goal.plane_normal is not None:
                try:
                    plane_normal = normalize(vec(goal.plane_normal))
                except ValueError:
                    issues.append(
                        f"{goal_label}.plane_normal must be a finite non-zero vector"
                    )

            if desired is not None:
                cosine = max(-1.0, min(1.0, dot(current, desired)))
                outlet_separation = math.degrees(math.acos(cosine))
                sweep = abs(float(goal.angle)) % 360.0
                required_separation = min(sweep, 360.0 - sweep)
                direction_tolerance = math.degrees(math.acos(0.9999))
                tolerance = max(
                    direction_tolerance,
                    0.1,
                    required_separation * 1e-3,
                )
                if abs(outlet_separation - required_separation) > tolerance:
                    issues.append(
                        f"sequential heading contradiction at {goal_label}: incoming "
                        f"heading {[round(value, 6) for value in current]}, requested "
                        f"turn.direction={goal.direction}, and turn.angle={goal.angle:g} "
                        f"cannot all hold; those headings are {outlet_separation:.6g} "
                        f"degrees apart but the sweep requires {required_separation:.6g} "
                        "degrees. A turn direction is the outlet tangent, not a prose "
                        "label for the bend's location"
                    )
                if plane_normal is not None:
                    inlet_plane_error = abs(dot(current, plane_normal))
                    outlet_plane_error = abs(dot(desired, plane_normal))
                    if max(inlet_plane_error, outlet_plane_error) > 1e-4:
                        issues.append(
                            f"sequential heading contradiction at {goal_label}: "
                            "turn.plane_normal must be perpendicular to both the "
                            "incoming and outlet tangents; absolute dot products are "
                            f"{inlet_plane_error:.6g} and {outlet_plane_error:.6g}"
                        )
                current = desired
                continue

            if plane_normal is None:
                issues.append(
                    f"{goal_label} must provide either an exact cardinal outlet "
                    "direction or a signed bend plane"
                )
                break
            inlet_plane_error = abs(dot(current, plane_normal))
            if inlet_plane_error > 1e-4:
                issues.append(
                    f"sequential heading contradiction at {goal_label}: "
                    "turn.plane_normal must be perpendicular to the incoming tangent; "
                    f"incoming heading is {[round(value, 6) for value in current]} "
                    f"and absolute dot product is {inlet_plane_error:.6g}"
                )
                break
            current = normalize(
                rotate(current, plane_normal, math.radians(float(goal.angle)))
            )
            continue

        if goal.type == "route":
            if goal.waypoint_frame == "relative_to_target" and goal.required_waypoints:
                first_offset = vec(goal.required_waypoints[0])
                axial = dot(first_offset, current)
                total = length(first_offset)
                lateral = math.sqrt(max(0.0, total * total - axial * axial))
                required_radius = minimum_spline_curvature_radius(
                    current_outer_diameter,
                    settings.modeling_tolerance,
                    goal.minimum_curvature_radius,
                )
                if axial <= settings.modeling_tolerance:
                    issues.append(
                        f"{goal_label} relative first waypoint must advance along "
                        "the incoming tangent before the lateral freeform turn; "
                        f"axial projection is {axial:.6g} mm for incoming heading "
                        f"{[round(value, 6) for value in current]} and first offset "
                        f"{[round(value, 6) for value in first_offset]}. Make that "
                        "offset a positive parallel multiple of the incoming heading"
                    )
                if lateral > settings.modeling_tolerance:
                    implied_entry_radius = (total * total) / (2.0 * lateral)
                    if (
                        implied_entry_radius + settings.modeling_tolerance
                        < required_radius
                    ):
                        issues.append(
                            f"{goal_label} relative first waypoint is too close/off-axis "
                            "for a regular tangent entry: its circular-entry radius is "
                            f"{implied_entry_radius:.6g} mm but at least "
                            f"{required_radius:.6g} mm is required. Move the first "
                            "anchor farther along the incoming tangent before turning"
                        )
                try:
                    if goal.terminal_axis is not None:
                        final_tangent = normalize(vec(goal.terminal_axis))
                    elif len(goal.required_waypoints) == 1:
                        final_tangent = normalize(first_offset)
                    else:
                        final_tangent = normalize(
                            sub(
                                vec(goal.required_waypoints[-1]),
                                vec(goal.required_waypoints[-2]),
                            )
                        )
                    predicted_radius = None
                    predicted_curve_length: float | None = None
                    polyline_lower_bound: float | None = None
                    handle_factors: list[float] = []
                    if len(goal.required_waypoints) >= 2:
                        prediction = predict_c1_spline(
                            [
                                (0.0, 0.0, 0.0),
                                *[vec(point) for point in goal.required_waypoints],
                            ],
                            current,
                            final_tangent,
                            modeling_tolerance=settings.modeling_tolerance,
                        )
                        predicted_radius = prediction.minimum_radius
                        predicted_curve_length = prediction.curve_length
                        polyline_lower_bound = prediction.polyline_length
                        handle_factors = list(prediction.handle_factors)
                except ValueError:
                    issues.append(
                        f"{goal_label} relative waypoint chain contains a zero-length "
                        "tangent or 180-degree cusp; redistribute the anchors"
                    )
                    break
                if goal.length is not None and polyline_lower_bound is not None:
                    expected_length = float(goal.length)
                    tolerance = max(
                        settings.modeling_tolerance * 10.0,
                        expected_length * 1e-3,
                    )
                    if polyline_lower_bound > expected_length + tolerance:
                        issues.append(
                            f"{goal_label} route.length={expected_length:.6g} mm is "
                            "mathematically shorter than its ordered required-waypoint "
                            f"polyline lower bound {polyline_lower_bound:.6g} mm; no "
                            "spline can satisfy both. Remove model-invented waypoints "
                            "for an ordinary line arm, or choose a source-allowed "
                            "length and anchor chain whose calculated curve length agrees"
                        )
                    elif (
                        predicted_curve_length is not None
                        and abs(predicted_curve_length - expected_length) > tolerance
                    ):
                        issues.append(
                            f"{goal_label} deterministic spline centerline length is "
                            f"{predicted_curve_length:.6g} mm but route.length requires "
                            f"{expected_length:.6g}±{tolerance:.6g} mm. Revise the "
                            "model-authored anchors/terminal tangent or choose a "
                            "source-allowed route length before freezing the Intent"
                        )
                if (
                    predicted_radius is not None
                    and predicted_radius + settings.modeling_tolerance < required_radius
                ):
                    natural_radius: float | None = None
                    if (
                        goal.terminal_axis is not None
                        and len(goal.required_waypoints) >= 2
                    ):
                        try:
                            natural_radius, _natural_factors = (
                                _predicted_c1_spline_minimum_radius(
                                    [vec(point) for point in goal.required_waypoints],
                                    current,
                                    None,
                                    modeling_tolerance=settings.modeling_tolerance,
                                )
                            )
                        except ValueError:
                            natural_radius = None
                    natural_hint = ""
                    if (
                        natural_radius is not None
                        and natural_radius + settings.modeling_tolerance
                        >= required_radius
                    ):
                        natural_hint = (
                            " The same anchors with no terminal_axis contract use "
                            "their natural final chord and predict "
                            f"{natural_radius:.6g} mm; if terminal_axis was not "
                            "explicitly authored by the user, omit it instead of "
                            "adding short lead-out waypoints."
                        )
                    issues.append(
                        f"{goal_label} direct required-anchor realization predicts "
                        "a minimum curvature radius of "
                        f"{predicted_radius:.6g} mm but "
                        f"at least {required_radius:.6g} mm is required after "
                        f"deterministic handle optimization (factors "
                        f"{[round(value, 3) for value in handle_factors]}). "
                        "Redistribute or add well-separated relative anchors so each "
                        "direction change is spread over more distance" + natural_hint
                    )
                # With no explicit terminal-axis contract, the supported spline
                # uses the final waypoint chord as its natural downstream heading.
                # Propagating it lets the next qualitative segment receive the
                # same entry-feasibility check instead of stopping at this route.
                current = final_tangent
                continue
            if goal.path_kind == "line":
                line_headings: list[tuple[str, tuple[float, float, float]]] = []
                if goal.direction is not None:
                    line_headings.append(
                        ("direction", direction_to_vector(goal.direction))
                    )
                if goal.terminal_axis is not None:
                    try:
                        line_headings.append(
                            ("terminal_axis", normalize(vec(goal.terminal_axis)))
                        )
                    except ValueError:
                        issues.append(
                            f"{goal_label}.terminal_axis must be a finite non-zero heading"
                        )
                for field_name, desired in line_headings:
                    alignment = dot(current, desired)
                    if alignment < 0.9999:
                        issues.append(
                            f"sequential heading contradiction at {goal_label}: line "
                            f"route {field_name} cannot mate to incoming heading "
                            f"{[round(value, 6) for value in current]} without a turn"
                        )
                continue
            if goal.terminal_axis is not None:
                try:
                    current = normalize(vec(goal.terminal_axis))
                except ValueError:
                    return [
                        *issues,
                        f"{goal_label}.terminal_axis must be a finite non-zero heading",
                    ]
                continue
            break

        if goal.type in {"diameter_change", "connector"}:
            if goal.direction is not None:
                desired = direction_to_vector(goal.direction)
                alignment = dot(current, desired)
                if alignment < 0.9999:
                    issues.append(
                        f"sequential heading contradiction at {goal_label}: "
                        f"{goal.type}.direction={goal.direction} cannot change the "
                        f"incoming axial heading {[round(value, 6) for value in current]}"
                    )
            if goal.type == "diameter_change" and goal.diameter_out is not None:
                current_outer_diameter = float(goal.diameter_out)
            continue

        if goal.type in {"branch", "connect", "end"}:
            break

    return issues


def _branch_outlet_heading_candidates(
    incoming: tuple[float, float, float],
    goal: Goal,
) -> list[tuple[float, float, float]]:
    """Return every heading a branch contract may expose to its successors.

    This mirrors the authored outlet surface without deciding which outlet a
    later planner action must consume.  A successor contract is rejected only
    when *all* of these candidates fail, so the preflight cannot silently make
    a topology choice on the LLM's behalf.
    """

    candidates: list[tuple[float, float, float]] = []

    def append(raw: tuple[float, float, float]) -> None:
        try:
            candidate = normalize(vec(raw))
        except ValueError:
            return
        if not any(dot(candidate, existing) >= 1.0 - 1e-9 for existing in candidates):
            candidates.append(candidate)

    if goal.include_primary_outlet is not False:
        append(incoming)
    for raw in goal.required_outlet_vectors:
        append(vec(raw))
    for outlet in goal.required_outlets:
        append(vec(outlet.axis))
    for direction in goal.required_outlet_directions:
        append(direction_to_vector(direction))

    # Some legacy branch contracts express their outlet fan with only signed
    # angles.  Reproduce that generic construction so those candidates receive
    # the same feasibility check as explicit outlet vectors.
    if not candidates and goal.branch_angles:
        base = (
            direction_to_vector(goal.direction)
            if goal.direction is not None
            else incoming
        )
        try:
            base = normalize(base)
            if goal.branch_plane_normal is not None:
                normal = normalize(vec(goal.branch_plane_normal))
                for angle in goal.branch_angles:
                    append(rotate(base, normal, math.radians(float(angle))))
            else:
                side = choose_perpendicular_axis(base)
                for angle in goal.branch_angles:
                    radians = math.radians(float(angle))
                    append(
                        add(
                            mul(base, math.cos(radians)),
                            mul(side, math.sin(radians)),
                        )
                    )
        except ValueError:
            pass
    return candidates


def _serial_heading_before_goal(
    intent: IntentResult,
    stop_index: int,
) -> tuple[float, float, float] | None:
    """Propagate only the unambiguous serial heading before ``stop_index``."""

    try:
        current = normalize(vec(intent.start_axis))
    except ValueError:
        return None
    for index, goal in enumerate(intent.target_behavior[:stop_index]):
        if index > 0:
            previous_id = intent.target_behavior[index - 1].goal_id
            if goal.allow_parallel or (
                goal.depends_on_goal_ids and previous_id not in goal.depends_on_goal_ids
            ):
                return None
        try:
            if goal.type == "move" and goal.direction is not None:
                current = direction_to_vector(goal.direction)
            elif goal.type == "turn" and goal.angle is not None:
                if goal.direction is not None:
                    current = direction_to_vector(goal.direction)
                elif goal.plane_normal is not None:
                    current = normalize(
                        rotate(
                            current,
                            normalize(vec(goal.plane_normal)),
                            math.radians(float(goal.angle)),
                        )
                    )
                else:
                    return None
            elif goal.type == "route":
                if goal.terminal_axis is not None:
                    current = normalize(vec(goal.terminal_axis))
                elif len(goal.required_waypoints) >= 2:
                    current = normalize(
                        sub(
                            vec(goal.required_waypoints[-1]),
                            vec(goal.required_waypoints[-2]),
                        )
                    )
                elif goal.direction is not None:
                    current = direction_to_vector(goal.direction)
                elif goal.path_kind != "line":
                    return None
            elif goal.type in {"diameter_change", "connector"}:
                if goal.direction is not None:
                    current = direction_to_vector(goal.direction)
            elif goal.type in {"branch", "connect", "end"}:
                return None
        except ValueError:
            return None
    return normalize(current)


def _branch_successor_spline_issues(
    intent: IntentResult,
    settings: Settings,
) -> list[str]:
    """Preflight fixed spline anchors immediately downstream of a branch.

    The ordinary sequential simulator intentionally stops at a fork because it
    must not guess an outlet.  That previously let an infeasible immutable
    spline contract reach the step retry loop.  Here every authored outlet is
    evaluated independently and the contract is returned to the Intent LLM
    only when no possible successor heading can satisfy it.
    """

    goals_by_id = {
        goal.goal_id: (index, goal)
        for index, goal in enumerate(intent.target_behavior)
        if goal.goal_id is not None
    }
    issues: list[str] = []
    current_outer_diameter = float(intent.global_spec.outer_diameter)
    diameter_by_index: list[float] = []
    for goal in intent.target_behavior:
        diameter_by_index.append(current_outer_diameter)
        if goal.type == "diameter_change" and goal.diameter_out is not None:
            current_outer_diameter = float(goal.diameter_out)

    for route_index, route in enumerate(intent.target_behavior):
        if not (
            route.type == "route"
            and route.path_kind == "spline"
            and route.waypoint_frame == "relative_to_target"
            and route.required_waypoints
        ):
            continue
        branch_refs = [
            goals_by_id[goal_id]
            for goal_id in route.depends_on_goal_ids
            if goal_id in goals_by_id and goals_by_id[goal_id][1].type == "branch"
        ]
        if len(branch_refs) != 1:
            continue
        branch_index, branch = branch_refs[0]
        incoming = _serial_heading_before_goal(intent, branch_index)
        if incoming is None:
            continue
        headings = _branch_outlet_heading_candidates(incoming, branch)
        if not headings:
            continue

        required_radius = minimum_spline_curvature_radius(
            diameter_by_index[route_index],
            settings.modeling_tolerance,
            route.minimum_curvature_radius,
        )
        points = [
            (0.0, 0.0, 0.0),
            *[vec(point) for point in route.required_waypoints],
        ]
        try:
            final_tangent = (
                normalize(vec(route.terminal_axis))
                if route.terminal_axis is not None
                else normalize(sub(points[-1], points[-2]))
            )
        except ValueError:
            continue

        evaluations: list[dict[str, Any]] = []
        any_feasible = False
        for heading in headings:
            first_offset = points[1]
            axial = dot(first_offset, heading)
            try:
                prediction = predict_c1_spline(
                    points,
                    heading,
                    final_tangent,
                    modeling_tolerance=settings.modeling_tolerance,
                )
                radius = float(prediction.minimum_radius)
            except ValueError:
                radius = 0.0
            feasible = (
                axial > settings.modeling_tolerance
                and radius + settings.modeling_tolerance >= required_radius
            )
            any_feasible = any_feasible or feasible
            evaluations.append(
                {
                    "incoming_heading": [round(value, 6) for value in heading],
                    "first_waypoint_axial_projection": round(axial, 6),
                    "predicted_minimum_radius": round(radius, 6),
                    "feasible": feasible,
                }
            )
        if any_feasible:
            continue
        route_label = route.goal_id or f"target_behavior[{route_index}]"
        branch_label = branch.goal_id or f"target_behavior[{branch_index}]"
        issues.append(
            f"{route_label} fixed required-anchor spline is infeasible after every "
            f"authored outlet of {branch_label}: required minimum curvature radius "
            f"is {required_radius:.6g} mm; outlet evaluations={evaluations}. The "
            "system will not choose an outlet or patch coordinates. Re-author the "
            "LLM-inferred relative waypoint contract so at least one permitted "
            "branch heading has positive entry advance and passes the calculated "
            "curvature bound; preserve any coordinates explicitly supplied by the user"
        )
    return issues


def _sequential_position_issues(
    intent: IntentResult,
    settings: Settings,
) -> list[str]:
    """Integrate the uniquely serial linear prefix and reject impossible poses.

    A line with a fixed inlet, heading and length has exactly one endpoint.  If
    Intent also freezes a different terminal_position, no amount of step-level
    replanning can satisfy it.  This preflight returns that contradiction to the
    Intent author before the contract becomes immutable.
    """

    try:
        position = vec(intent.start_position)
        heading = normalize(vec(intent.start_axis))
    except ValueError:
        return ["start_position/start_axis must define a finite serial pose"]

    issues: list[str] = []
    for index, goal in enumerate(intent.target_behavior):
        if index > 0:
            previous_goal_id = intent.target_behavior[index - 1].goal_id
            if goal.allow_parallel or (
                goal.depends_on_goal_ids
                and previous_goal_id not in goal.depends_on_goal_ids
            ):
                break
        label = goal.goal_id or f"target_behavior[{index}]"
        if goal.type == "move":
            if goal.direction is not None:
                heading = direction_to_vector(goal.direction)
            if goal.length is not None:
                position = add(position, mul(heading, float(goal.length)))
            continue
        if goal.type == "route" and goal.path_kind == "line":
            route_heading = (
                direction_to_vector(goal.direction)
                if goal.direction is not None
                else heading
            )
            if goal.terminal_axis is not None:
                try:
                    terminal_heading = normalize(vec(goal.terminal_axis))
                except ValueError:
                    terminal_heading = route_heading
            else:
                terminal_heading = route_heading
            predicted = (
                add(position, mul(route_heading, float(goal.length)))
                if goal.length is not None
                else None
            )
            if goal.terminal_position is not None:
                terminal = vec(goal.terminal_position)
                delta = sub(terminal, position)
                distance = length(delta)
                tolerance = max(
                    settings.modeling_tolerance * 10.0,
                    max(distance, float(goal.length or 0.0), 1.0) * 1e-6,
                )
                if predicted is not None:
                    endpoint_error = length(sub(predicted, terminal))
                    if endpoint_error > tolerance:
                        issues.append(
                            f"{label} line pose is over-constrained: start "
                            f"{[round(value, 6) for value in position]} + length "
                            f"{float(goal.length):.6g} * heading "
                            f"{[round(value, 6) for value in route_heading]} gives "
                            f"{[round(value, 6) for value in predicted]}, not "
                            f"terminal_position {[round(value, 6) for value in terminal]} "
                            f"(error {endpoint_error:.6g} mm). Revise the Intent's "
                            "length/heading/terminal position together"
                        )
                if distance > tolerance:
                    alignment = dot(normalize(delta), route_heading)
                    if alignment < 0.9999:
                        issues.append(
                            f"{label} terminal_position lies off its line heading: "
                            f"displacement {[round(value, 6) for value in delta]} has "
                            f"alignment {alignment:.6g} with heading "
                            f"{[round(value, 6) for value in route_heading]}"
                        )
                position = terminal
            elif predicted is not None:
                position = predicted
            heading = terminal_heading
            continue
        if (
            goal.type == "route"
            and goal.path_kind == "spline"
            and goal.required_waypoints
        ):
            endpoint = vec(goal.required_waypoints[-1])
            if goal.waypoint_frame == "relative_to_target":
                endpoint = add(position, endpoint)
            if goal.terminal_position is not None:
                terminal = vec(goal.terminal_position)
                tolerance = max(settings.modeling_tolerance * 10.0, 1e-6)
                endpoint_error = length(sub(endpoint, terminal))
                if endpoint_error > tolerance:
                    issues.append(
                        f"{label} final required waypoint and terminal_position "
                        f"differ by {endpoint_error:.6g} mm"
                    )
                position = terminal
            else:
                position = endpoint
            if goal.terminal_axis is not None:
                try:
                    heading = normalize(vec(goal.terminal_axis))
                except ValueError:
                    pass
            continue
        # Circular arcs, turns and branches need additional topology/frame
        # choices. Stop at that boundary instead of guessing a downstream pose.
        break
    return issues


def _branch_angle_vector_issues(intent: IntentResult) -> list[str]:
    """branch angle과 전역 outlet 축이 같은 작은 선각을 뜻하는지 검사한다.

    branch 전후의 축은 방향 화살표이지만 사용자가 말하는 Y 각도는 보통 두
    centerline 사이의 acute line angle이다. 직렬 primary 축이 Goal에 없으면
    다음 명시 move/line에서 다시 heading을 잡고, 모르는 값을 추측하지 않는다.
    """

    try:
        current: tuple[float, float, float] | None = normalize(vec(intent.start_axis))
    except ValueError:
        return []
    issues: list[str] = []
    for index, goal in enumerate(intent.target_behavior):
        label = goal.goal_id or f"target_behavior[{index}]"
        if goal.type == "move" and goal.direction is not None:
            current = direction_to_vector(goal.direction)
            continue
        if goal.type == "route":
            try:
                if goal.terminal_axis is not None:
                    current = normalize(vec(goal.terminal_axis))
                elif len(goal.required_waypoints) >= 2:
                    current = normalize(
                        sub(
                            vec(goal.required_waypoints[-1]),
                            vec(goal.required_waypoints[-2]),
                        )
                    )
                elif goal.direction is not None:
                    current = direction_to_vector(goal.direction)
                elif goal.path_kind == "line":
                    current = None
            except ValueError:
                current = None
            continue
        if goal.type == "turn" and current is not None and goal.angle is not None:
            try:
                if goal.direction is not None:
                    current = direction_to_vector(goal.direction)
                elif goal.plane_normal is not None:
                    current = normalize(
                        rotate(
                            current,
                            normalize(vec(goal.plane_normal)),
                            math.radians(float(goal.angle)),
                        )
                    )
            except ValueError:
                current = None
            continue
        if goal.type != "branch":
            continue

        vectors: list[tuple[float, float, float]] = []
        vectors.extend(vec(value) for value in goal.required_outlet_vectors)
        vectors.extend(vec(value.axis) for value in goal.required_outlets)
        vectors.extend(
            direction_to_vector(value) for value in goal.required_outlet_directions
        )
        if current is not None and goal.branch_angles and vectors:
            actual_angles = []
            for outlet_axis in vectors:
                cosine = max(-1.0, min(1.0, dot(current, normalize(outlet_axis))))
                directed = math.degrees(math.acos(cosine))
                actual_angles.append(min(directed, 180.0 - directed))
            expected_angles = [abs(float(value)) for value in goal.branch_angles]
            available = list(actual_angles)
            unmatched: list[float] = []
            for expected in expected_angles:
                if not available:
                    unmatched.append(expected)
                    continue
                best_index = min(
                    range(len(available)),
                    key=lambda item: abs(available[item] - expected),
                )
                if abs(available[best_index] - expected) > 0.5:
                    unmatched.append(expected)
                else:
                    available.pop(best_index)
            if unmatched or available:
                issues.append(
                    f"{label} branch_angles conflict with its outlet axes: "
                    f"expected acute angles {expected_angles}, actual "
                    f"{[round(value, 6) for value in actual_angles]}, tolerance "
                    "0.5 degree. Select one source-allowed angle per outlet and "
                    "make each vector mathematically consistent with it"
                )
        current = None
        if goal.include_primary_outlet is False:
            break
    return issues


def _positive_intent_dimensions(intent: IntentResult) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = [
        ("global_spec.outer_diameter", float(intent.global_spec.outer_diameter)),
        ("global_spec.wall_thickness", float(intent.global_spec.wall_thickness)),
    ]
    for goal_index, goal in enumerate(intent.target_behavior):
        prefix = f"target_behavior[{goal_index}]"
        for field_name in _GOAL_LENGTH_FIELDS:
            value = getattr(goal, field_name)
            if value is not None:
                values.append((f"{prefix}.{field_name}", float(value)))
        if goal.offset is not None:
            magnitude = math.sqrt(
                sum(float(component) ** 2 for component in goal.offset)
            )
            if magnitude > 0.0:
                values.append((f"{prefix}.offset_magnitude", magnitude))
        for outlet_index, outlet in enumerate(goal.required_outlets):
            for field_name in ("length", "outer_diameter", "wall_thickness"):
                value = getattr(outlet, field_name)
                if value is not None:
                    values.append(
                        (
                            f"{prefix}.required_outlets[{outlet_index}].{field_name}",
                            float(value),
                        )
                    )
        if goal.component_spec is not None:
            for field_name in _COMPONENT_LENGTH_FIELDS:
                value = getattr(goal.component_spec, field_name)
                if value is not None and float(value) > 0.0:
                    values.append(
                        (f"{prefix}.component_spec.{field_name}", float(value))
                    )
    for constraint_index, constraint in enumerate(intent.geometric_constraints):
        if constraint.type != "max_module_count" and constraint.value is not None:
            values.append(
                (
                    f"geometric_constraints[{constraint_index}].value",
                    float(constraint.value),
                )
            )
        if (
            constraint.type == "bounding_box"
            and constraint.minimum is not None
            and constraint.maximum is not None
        ):
            for axis_index, (minimum, maximum) in enumerate(
                zip(constraint.minimum, constraint.maximum)
            ):
                values.append(
                    (
                        f"geometric_constraints[{constraint_index}]"
                        f".bounding_box_extent[{axis_index}]",
                        float(maximum) - float(minimum),
                    )
                )
    return values


def _metric_number(raw: str) -> float:
    """쉼표와 유니코드 minus를 정규화해 유한한 실수로 변환한다."""

    return float(raw.replace(",", "").replace("−", "-"))


def _explicit_mm_ranges(text: str) -> list[_ExplicitMMRange]:
    """``85–100 mm``처럼 단위를 공유하는 명시적 범위를 추출한다."""

    result: list[_ExplicitMMRange] = []
    for match in _EXPLICIT_MM_RANGE.finditer(text):
        authored_start = _metric_number(match.group("start"))
        authored_end = _metric_number(match.group("end"))
        if not (math.isfinite(authored_start) and math.isfinite(authored_end)):
            continue
        result.append(
            _ExplicitMMRange(
                authored_start=authored_start,
                authored_end=authored_end,
                span=match.span(),
            )
        )
    return result


def _explicit_branch_length_ranges(text: str) -> list[_ExplicitMMRange]:
    """``each branch ... 85–100 mm long``에 귀속된 범위만 반환한다."""

    result: list[_ExplicitMMRange] = []
    for source_range in _explicit_mm_ranges(text):
        left = max(
            text.rfind(".", 0, source_range.span[0]),
            text.rfind("\n", 0, source_range.span[0]),
            text.rfind(";", 0, source_range.span[0]),
        )
        sentence_end_candidates = [
            position
            for marker in (".", "\n", ";")
            for position in [text.find(marker, source_range.span[1])]
            if position >= 0
        ]
        right = min(sentence_end_candidates) if sentence_end_candidates else len(text)
        context = text[left + 1 : right]
        if re.search(
            r"\b(?:each|every|all)\s+branch(?:es)?\b", context, re.IGNORECASE
        ) and re.search(
            r"\b(?:long|lengths?)\b",
            context,
            re.IGNORECASE,
        ):
            result.append(source_range)
    return result


def _explicit_branch_angle_ranges(text: str) -> list[_ExplicitAngleRange]:
    """branch angle과 main axis가 같은 clause에 있는 degree 범위를 찾는다."""

    result: list[_ExplicitAngleRange] = []
    for match in _EXPLICIT_DEGREE_RANGE.finditer(text):
        left = max(
            text.rfind(".", 0, match.start()),
            text.rfind("\n", 0, match.start()),
            text.rfind(";", 0, match.start()),
        )
        sentence_end_candidates = [
            position
            for marker in (".", "\n", ";")
            for position in [text.find(marker, match.end())]
            if position >= 0
        ]
        right = min(sentence_end_candidates) if sentence_end_candidates else len(text)
        context = text[left + 1 : right]
        if not re.search(r"\b(?:branch|arm)\s+angles?\b", context, re.IGNORECASE):
            continue
        if not re.search(r"\bmain\s+(?:axis|centerline)\b", context, re.IGNORECASE):
            continue
        start = _metric_number(match.group("start"))
        end = _metric_number(match.group("end"))
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        result.append(
            _ExplicitAngleRange(
                minimum=min(start, end),
                maximum=max(start, end),
                span=match.span(),
            )
        )
    return result


def _main_axis_terminal_arm_angles(intent: IntentResult) -> list[float]:
    """중앙 직선 축선과 모든 물리 terminal arm 축선의 acute 각도를 계산한다."""

    branch_indexes = [
        index
        for index, goal in enumerate(intent.target_behavior)
        if goal.type == "branch"
    ]
    main_axis: tuple[float, float, float] | None = None
    if len(branch_indexes) >= 2:
        for goal in intent.target_behavior[branch_indexes[0] + 1 : branch_indexes[-1]]:
            try:
                if goal.type == "move" and goal.direction is not None:
                    main_axis = direction_to_vector(goal.direction)
                    break
                if goal.type == "route" and goal.path_kind == "line":
                    if goal.direction is not None:
                        main_axis = direction_to_vector(goal.direction)
                        break
                    if goal.terminal_axis is not None:
                        main_axis = normalize(vec(goal.terminal_axis))
                        break
            except ValueError:
                continue
    if main_axis is None:
        main_axis = normalize(vec(intent.start_axis))

    arm_axes: list[tuple[float, float, float]] = []
    first_branch_index = (
        branch_indexes[0] if branch_indexes else len(intent.target_behavior)
    )
    prefix = intent.target_behavior[:first_branch_index]
    root_axis: tuple[float, float, float] | None = None
    for goal in reversed(prefix):
        try:
            if goal.type == "route":
                if goal.terminal_axis is not None:
                    root_axis = normalize(vec(goal.terminal_axis))
                elif len(goal.required_waypoints) >= 1:
                    root_axis = normalize(vec(goal.required_waypoints[-1]))
                elif goal.direction is not None:
                    root_axis = direction_to_vector(goal.direction)
            elif goal.type == "move" and goal.direction is not None:
                root_axis = direction_to_vector(goal.direction)
        except ValueError:
            root_axis = None
        if root_axis is not None:
            break
    arm_axes.append(root_axis or normalize(vec(intent.start_axis)))

    for goal in intent.target_behavior:
        if goal.type != "branch":
            continue
        arm_axes.extend(vec(value) for value in goal.required_outlet_vectors)
        arm_axes.extend(vec(outlet.axis) for outlet in goal.required_outlets)
        arm_axes.extend(
            direction_to_vector(value) for value in goal.required_outlet_directions
        )

    result: list[float] = []
    for arm_axis in arm_axes:
        cosine = max(
            -1.0,
            min(1.0, abs(dot(normalize(main_axis), normalize(arm_axis)))),
        )
        result.append(math.degrees(math.acos(cosine)))
    return result


def _terminal_arm_length_contracts(intent: IntentResult) -> list[float | None]:
    """START arm과 각 최종 branch outlet의 독립 중심선 길이를 순서대로 모은다."""

    first_branch_index = next(
        (
            index
            for index, goal in enumerate(intent.target_behavior)
            if goal.type == "branch"
        ),
        len(intent.target_behavior),
    )
    root_length = 0.0
    has_root_length = False
    for goal in intent.target_behavior[:first_branch_index]:
        if goal.type in {"move", "route"} and goal.length is not None:
            root_length += float(goal.length)
            has_root_length = True
        elif (
            goal.type == "turn"
            and goal.bend_radius is not None
            and goal.angle is not None
        ):
            root_length += math.radians(abs(float(goal.angle))) * float(
                goal.bend_radius
            )
            has_root_length = True
    result: list[float | None] = [root_length if has_root_length else None]

    for goal in intent.target_behavior:
        if goal.type != "branch":
            continue
        described_count = (
            goal.branch_count
            or len(goal.required_outlets)
            or len(goal.required_outlet_vectors)
            or len(goal.required_outlet_directions)
        )
        if goal.required_outlets:
            result.extend(
                float(outlet.length) if outlet.length is not None else None
                for outlet in goal.required_outlets
            )
        elif goal.length is not None:
            result.extend(float(goal.length) for _index in range(described_count))
        else:
            result.extend(None for _index in range(described_count))
    return result


def _explicit_mm_values(text: str) -> list[float]:
    """명시적 mm 값과 범위의 양 끝점을 원문 순서대로 반환한다."""

    ranges = _explicit_mm_ranges(text)
    events: list[tuple[int, int, list[float]]] = [
        (
            source_range.span[0],
            0,
            [source_range.authored_start, source_range.authored_end],
        )
        for source_range in ranges
    ]
    for match in _EXPLICIT_MM_VALUE.finditer(text):
        if any(
            match.start() < source_range.span[1] and source_range.span[0] < match.end()
            for source_range in ranges
        ):
            continue
        value = _metric_number(match.group(1))
        if math.isfinite(value):
            events.append((match.start(), 1, [value]))
    values: list[float] = []
    for _offset, _kind, event_values in sorted(events):
        values.extend(event_values)
    return values


def _exact_mm_contract_values(text: str) -> list[float]:
    """범위와 동일 전역 속성의 반복을 제외한 exact 수치 계약을 만든다.

    근사 표현(``about 68 mm`` 등)은 임의 허용오차를 만들어 값을 바꾸지
    않고 68 mm라는 nominal 계약으로 유지한다. 반면 명시적 범위만 그
    경계 안의 선택을 허용한다.
    """

    range_spans = [source_range.span for source_range in _explicit_mm_ranges(text)]
    duplicate_singleton_spans: set[tuple[int, int]] = set()
    seen_singletons: list[tuple[str, float]] = []
    for _offset, span, role, value in _anchored_mm_matches(text):
        if role not in _SINGLETON_MM_ROLES:
            continue
        if any(
            prior_role == role and _same_metric_value(prior_value, value)
            for prior_role, prior_value in seen_singletons
        ):
            duplicate_singleton_spans.add(span)
        else:
            seen_singletons.append((role, value))

    result: list[float] = []
    for match in _EXPLICIT_MM_VALUE.finditer(text):
        if any(
            match.start() < right and left < match.end() for left, right in range_spans
        ):
            continue
        if match.span(1) in duplicate_singleton_spans:
            continue
        value = _metric_number(match.group(1))
        if math.isfinite(value):
            result.append(value)
    return result


def _intent_range_candidate_values(intent: IntentResult) -> list[float]:
    """범위 계약과 비교 가능한 양의 typed 물리 치수만 수집한다."""

    return [
        value
        for _path, value in _positive_intent_dimensions(intent)
        if math.isfinite(value) and value > 0.0
    ]


def _explicit_vector3_values(
    text: str,
) -> list[tuple[float, float, float]]:
    return [value for value, _is_direction in _explicit_vector3_contracts(text)]


def _explicit_vector3_contracts(
    text: str,
) -> list[tuple[tuple[float, float, float], bool]]:
    """Return source tuples with exact-coordinate vs direction-ratio roles."""

    result: list[tuple[tuple[float, float, float], bool]] = []
    for match in _EXPLICIT_VECTOR3.finditer(text):
        vector_value = tuple(
            float(match.group(name).replace("−", "-")) for name in ("x", "y", "z")
        )
        if all(math.isfinite(value) for value in vector_value):
            prefix = text[max(0, match.start() - 96) : match.start()]
            suffix = text[match.end() : min(len(text), match.end() + 48)]
            is_direction = bool(
                _PROPORTIONAL_VECTOR_PREFIX.search(prefix)
                or _PROPORTIONAL_VECTOR_SUFFIX.search(suffix)
            )
            result.append((vector_value, is_direction))
    return result


def _explicit_proportional_direction_contracts(
    text: str,
) -> list[_ProportionalDirectionContract]:
    """Extract proportional vector ratios with their authored semantic role."""

    result: list[_ProportionalDirectionContract] = []
    for match in _EXPLICIT_VECTOR3.finditer(text):
        prefix = text[max(0, match.start() - 192) : match.start()]
        suffix = text[match.end() : min(len(text), match.end() + 64)]
        if not (
            _PROPORTIONAL_VECTOR_PREFIX.search(prefix)
            or _PROPORTIONAL_VECTOR_SUFFIX.search(suffix)
        ):
            continue
        value = tuple(
            float(match.group(name).replace("−", "-")) for name in ("x", "y", "z")
        )
        if not all(math.isfinite(component) for component in value):
            continue
        # Restrict role words to the current clause. A prior sentence may name
        # START or a branch and must not relabel a later spline heading.
        local_prefix = re.split(r"[.;\n]", prefix)[-1]
        local_suffix = re.split(r"[.;\n]", suffix)[0]
        context = f"{local_prefix[-128:]} {local_suffix[:48]}".lower()
        if re.search(
            r"\bplane[-\s]*normal\b|\bnormal\s+(?:of|to)\s+(?:the\s+)?plane\b|"
            r"평면[^.;\n]{0,24}법선|법선[^.;\n]{0,24}평면",
            context,
        ):
            role = "plane_normal"
        elif re.search(
            r"\b(?:branch|arm|terminal\s+branch|branch\s+outlet)\b|"
            r"분기|가지|암\b",
            context,
        ):
            role = "branch_outlet"
        elif re.search(
            r"\b(?:actuator|flange|component)\b|액추에이터|플랜지",
            context,
        ):
            role = "component_axis"
        elif re.search(
            r"(?:\bstart\b|\bstarting\b|\binlet\b)[^.;\n]{0,48}\baxis\b|"
            r"\bstart_axis\b|시작[^.;\n]{0,32}(?:축|방향)",
            context,
        ):
            role = "start_axis"
        elif re.search(
            r"\b(?:outlet|terminal|final)\s+(?:axis|heading|direction)\b|"
            r"\b(?:spline|route|bend|turn)\b|\b(?:last|final)\s+chord\b|"
            r"출구|말단|최종[^.;\n]{0,24}(?:축|방향)|마지막[^.;\n]{0,24}현",
            context,
        ):
            role = "goal_terminal"
        else:
            # Ambiguous proportional directions may bind to a typed generated
            # terminal, but never borrow START or a plane normal.
            role = "generated_terminal"
        result.append(_ProportionalDirectionContract(value=value, role=role))
    return result


def _intent_vector_values(
    intent: IntentResult,
) -> list[tuple[float, float, float]]:
    result = [vec(intent.start_position), vec(intent.start_axis)]
    for goal in intent.target_behavior:
        for value in (
            goal.plane_normal,
            goal.branch_plane_normal,
            goal.offset,
            goal.terminal_position,
            goal.terminal_axis,
        ):
            if value is not None:
                result.append(vec(value))
        result.extend(vec(value) for value in goal.required_waypoints)
        result.extend(vec(value) for value in goal.required_outlet_vectors)
        result.extend(vec(outlet.axis) for outlet in goal.required_outlets)
        if goal.component_spec is not None:
            for value in (
                goal.component_spec.flange_reference_axis,
                goal.component_spec.actuator_axis,
            ):
                if value is not None:
                    result.append(vec(value))
    return result


def _intent_direction_candidates(
    intent: IntentResult,
) -> list[_IntentDirectionCandidate]:
    """Collect one role-labelled candidate per physical authored direction."""

    result = [
        _IntentDirectionCandidate(
            semantic_id="START.axis",
            role="start_axis",
            value=vec(intent.start_axis),
        )
    ]
    for goal_index, goal in enumerate(intent.target_behavior):
        goal_key = goal.goal_id or f"goal_{goal_index}"
        for field_name, value in (
            ("plane_normal", goal.plane_normal),
            ("branch_plane_normal", goal.branch_plane_normal),
        ):
            if value is not None:
                result.append(
                    _IntentDirectionCandidate(
                        semantic_id=f"{goal_key}.{field_name}",
                        role="plane_normal",
                        value=vec(value),
                    )
                )

        outlet_values: list[tuple[float, float, float]] = []
        outlet_values.extend(vec(value) for value in goal.required_outlet_vectors)
        outlet_values.extend(vec(outlet.axis) for outlet in goal.required_outlets)
        outlet_values.extend(
            direction_to_vector(value) for value in goal.required_outlet_directions
        )
        result.extend(
            _IntentDirectionCandidate(
                semantic_id=f"{goal_key}.branch_outlet[{index}]",
                role="branch_outlet",
                value=value,
            )
            for index, value in enumerate(outlet_values)
        )

        terminal_value: tuple[float, float, float] | None = None
        if goal.type == "route":
            if goal.terminal_axis is not None:
                terminal_value = vec(goal.terminal_axis)
            elif len(goal.required_waypoints) >= 2:
                terminal_value = sub(
                    vec(goal.required_waypoints[-1]),
                    vec(goal.required_waypoints[-2]),
                )
            elif (
                len(goal.required_waypoints) == 1
                and goal.waypoint_frame == "relative_to_target"
            ):
                terminal_value = vec(goal.required_waypoints[0])
            elif goal.direction is not None:
                terminal_value = direction_to_vector(goal.direction)
        elif (
            goal.type
            in {
                "move",
                "turn",
                "diameter_change",
                "connector",
            }
            and goal.direction is not None
        ):
            terminal_value = direction_to_vector(goal.direction)
        if terminal_value is not None:
            result.append(
                _IntentDirectionCandidate(
                    semantic_id=f"{goal_key}.terminal_heading",
                    role="goal_terminal",
                    value=terminal_value,
                )
            )

        if goal.component_spec is not None:
            for field_name, value in (
                (
                    "flange_reference_axis",
                    goal.component_spec.flange_reference_axis,
                ),
                ("actuator_axis", goal.component_spec.actuator_axis),
            ):
                if value is not None:
                    result.append(
                        _IntentDirectionCandidate(
                            semantic_id=f"{goal_key}.component.{field_name}",
                            role="component_axis",
                            value=vec(value),
                        )
                    )
    return result


def _direction_roles_compatible(source_role: str, candidate_role: str) -> bool:
    if source_role == "generated_terminal":
        return candidate_role in {
            "goal_terminal",
            "branch_outlet",
            "component_axis",
        }
    return source_role == candidate_role


def _positive_parallel_direction(
    expected: tuple[float, float, float],
    candidate: tuple[float, float, float],
) -> bool:
    try:
        expected_unit = normalize(vec(expected))
        candidate_unit = normalize(vec(candidate))
    except ValueError:
        return False
    return dot(expected_unit, candidate_unit) >= 1.0 - 1e-6


def _anchored_mm_matches(
    text: str,
) -> list[tuple[int, tuple[int, int], str, float]]:
    """역할과 원문 span을 보존한 high-confidence 측정값을 추출한다."""

    matches: list[tuple[int, tuple[int, int], str, float]] = []
    claimed_spans: list[tuple[int, int]] = []
    for role, pattern in _ANCHORED_MM_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span("value")
            if any(span[0] < right and left < span[1] for left, right in claimed_spans):
                continue
            value = _metric_number(match.group("value"))
            if not math.isfinite(value):
                continue
            claimed_spans.append(span)
            matches.append((span[0], span, role, value))
    return sorted(matches)


def _anchored_mm_values(text: str) -> dict[str, list[float]]:
    """Extract only high-confidence source measurement roles.

    Specific change/transition patterns run before generic global-section
    patterns.  A numeric source span is claimed once, preventing a phrase such
    as ``reduce outer diameter`` from being interpreted as both an initial and
    an output diameter. Repeated mentions of the same global section property
    collapse to one semantic contract, while contradictory values remain.
    """

    result: dict[str, list[float]] = {}
    for _offset, _span, role, value in _anchored_mm_matches(text):
        role_values = result.setdefault(role, [])
        if role in _SINGLETON_MM_ROLES and any(
            _same_metric_value(value, previous) for previous in role_values
        ):
            continue
        role_values.append(value)
    return result


def _anchored_intent_values(intent: IntentResult) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {
        "global_outer_diameter": [float(intent.global_spec.outer_diameter)],
        "global_inner_diameter": [
            float(intent.global_spec.outer_diameter)
            - 2.0 * float(intent.global_spec.wall_thickness)
        ],
        "global_wall_thickness": [float(intent.global_spec.wall_thickness)],
        "straight_length": [],
        "connector_length": [],
        "transition_length": [],
        "diameter_out": [],
        "wall_thickness_out": [],
        "diameter_in_reference": [],
        "wall_thickness_in_reference": [],
        "route_rise": [],
        "junction_blend_radius": [],
        "junction_inner_blend_radius": [],
        "junction_max_hub_radius": [],
    }
    current_outer_diameter = float(intent.global_spec.outer_diameter)
    current_wall_thickness = float(intent.global_spec.wall_thickness)
    for goal in intent.target_behavior:
        if goal.type in {"move", "route"} and goal.length is not None:
            result["straight_length"].append(float(goal.length))
        if goal.type == "connector" and goal.length is not None:
            result["connector_length"].append(float(goal.length))
        if goal.type == "branch":
            for role, value in (
                ("junction_blend_radius", goal.blend_radius),
                ("junction_inner_blend_radius", goal.inner_blend_radius),
                ("junction_max_hub_radius", goal.max_hub_radius),
            ):
                if value is not None:
                    result[role].append(float(value))
        if goal.type == "diameter_change":
            result["diameter_in_reference"].append(current_outer_diameter)
            result["wall_thickness_in_reference"].append(current_wall_thickness)
            for role, value in (
                ("transition_length", goal.transition_length),
                ("diameter_out", goal.diameter_out),
                ("wall_thickness_out", goal.wall_thickness_out),
            ):
                if value is not None:
                    result[role].append(float(value))
            if goal.diameter_out is not None:
                current_outer_diameter = float(goal.diameter_out)
            if goal.wall_thickness_out is not None:
                current_wall_thickness = float(goal.wall_thickness_out)
        if (
            goal.type == "route"
            and goal.path_kind == "spline"
            and goal.waypoint_frame == "relative_to_target"
            and goal.required_waypoints
        ):
            result["route_rise"].append(float(goal.required_waypoints[-1][2]))
    return result


def _ordered_missing_values(
    expected: list[float],
    candidates: list[float],
) -> list[float]:
    """Return values that cannot map to distinct role-compatible fields.

    Candidate values may include inferred intermediate goals, so source values
    need only form an ordered subsequence.  Each candidate is consumed at most
    once, preserving multiplicity without forcing a brittle one-to-one count.
    """

    missing: list[float] = []
    start = 0
    for value in expected:
        match_index = next(
            (
                index
                for index in range(start, len(candidates))
                if _same_metric_value(value, candidates[index])
            ),
            None,
        )
        if match_index is None:
            missing.append(value)
        else:
            start = match_index + 1
    return missing


def _intent_numeric_literals(prompt: str, settings: Settings) -> list[str]:
    """Build a finite, source-grounded float vocabulary for intent extraction."""

    values: list[float] = [
        -1.0,
        0.0,
        1.0,
        settings.default_outer_diameter,
        settings.default_wall_thickness,
        settings.default_bend_radius,
        # Contextual inference pool used only when the user omitted a dimension.
        2.5,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        40.0,
        50.0,
        60.0,
        75.0,
        80.0,
        90.0,
        100.0,
    ]
    for match in _ANY_NUMERIC_VALUE.finditer(prompt):
        value = float(match.group(1).replace(",", "").replace("−", "-"))
        if math.isfinite(value):
            values.append(value)
    for vector_value in _explicit_vector3_values(prompt):
        values.extend(vector_value)
    literals: list[str] = []
    for value in values:
        signed_values = (value,) if value == 0.0 else (value, -value)
        for signed_value in signed_values:
            literal = (
                str(int(signed_value))
                if signed_value == math.trunc(signed_value)
                else format(signed_value, ".12g")
            )
            if literal not in literals:
                literals.append(literal)
    return literals


def _numeric_literal_schema_fits(literals: list[str]) -> bool:
    if len(literals) > MAX_STRUCTURED_NUMBER_LITERALS:
        return False
    payload_bytes = len(
        json.dumps(
            literals,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return payload_bytes <= MAX_STRUCTURED_NUMBER_LITERAL_BYTES


def _intent_json_envelope_request(request: str) -> str:
    """Wrap an intent request for the minimal provider-schema fallback."""

    return (
        request + "\n\nThe provider could not compile the full typed response grammar. "
        "Use the minimal envelope required by the active response schema. Set "
        "intent_json to a JSON-encoded string containing one complete intent "
        "root object. The inner object must still obey every field, topology, "
        "measurement, dependency, and safety requirement above; it will be "
        "strictly parsed and validated by the host. Do not put commentary or "
        "Markdown inside or outside intent_json."
    )


@dataclass(frozen=True)
class _PlannerNumericLiteralBundle:
    """한 번의 상태 순회로 만든 필수 및 선호 숫자 리터럴 묶음이다."""

    mandatory_literals: list[str]
    preferred_literals: list[str]


def _build_planner_numeric_literal_bundle(
    state: PipeState,
    *,
    include_optional: bool = True,
) -> _PlannerNumericLiteralBundle:
    """불변 계약과 ``S_t``를 한 번 순회해 숫자 어휘 묶음을 만든다."""

    exact: list[float] = []

    def collect(value: Any, destination: list[float] = exact) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            numeric = float(value)
            if math.isfinite(numeric):
                destination.append(numeric)
            return
        if isinstance(value, dict):
            for child in value.values():
                collect(child, destination)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                collect(child, destination)

    collect(compact_planner_payload(state, include_catalog=False))
    required_basis: list[float] = [
        -360.0,
        -180.0,
        -90.0,
        -45.0,
        -1.0,
        0.0,
        0.5,
        1.0,
        2.0,
        2.5,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        40.0,
        45.0,
        50.0,
        60.0,
        75.0,
        80.0,
        90.0,
        100.0,
        180.0,
        360.0,
    ]

    # 첫 미완료 목표뿐 아니라 의존성이 준비된 모든 목표를 보수적으로 보존한다.
    # 뒤쪽 순차 목표가 앞 목표와 한 행동을 공유하거나 병렬 목표가 직접 선택될 수
    # 있으므로, 여기서 범위를 더 줄이면 schema 경계에서 가능한 선택을 잃는다.
    completed_history = {
        goal_id
        for action in state.action_history
        for goal_id in action.completed_goal_ids
    }
    eligible_goal_values: list[float] = []
    for goal in state.remaining_goals:
        if not set(goal.depends_on_goal_ids).issubset(completed_history):
            continue
        collect(
            goal.model_dump(
                mode="json",
                exclude_none=True,
                exclude_defaults=True,
            ),
            eligible_goal_values,
        )
    active_context: list[float] = []
    collect(
        state.global_spec.model_dump(mode="json", exclude_none=True),
        active_context,
    )
    for constraint in state.geometric_constraints:
        collect(
            constraint.model_dump(mode="json", exclude_none=True),
            active_context,
        )
    for port in state.open_ports:
        collect(port.model_dump(mode="json", exclude_none=True), active_context)

    mandatory_active = [*eligible_goal_values, *active_context]
    section_seeds = [
        float(state.global_spec.outer_diameter),
        float(state.global_spec.wall_thickness),
    ]
    for port in state.open_ports:
        section_seeds.extend((float(port.outer_diameter), float(port.wall_thickness)))

    # 파생된 구성 편의 값은 부차적이므로 잘릴 수 있지만, 작성된 목표 값과
    # 선택 가능한 포트 값은 누락하면 안 된다.
    priority_seeds = list(dict.fromkeys([*section_seeds, *eligible_goal_values]))[:12]
    priority_derived: list[float] = []
    for value in priority_seeds:
        for factor in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0):
            priority_derived.append(value * factor)

    # 모든 선택 가능 목표, 불변 형상 제약, 대상 포트 및 기초 값은 필수다.
    # 이력 및 파생 후보는 provider-safe 어휘에서 남은 자리만 사용한다.
    mandatory_candidates = [*mandatory_active, *required_basis]

    def literal_for(value: float, *, preserve: bool) -> str | None:
        if not math.isfinite(value) or abs(value) > 1e9:
            return None
        normalized = value if preserve else round(value, 9)
        return (
            str(int(normalized))
            if normalized == math.trunc(normalized)
            else repr(normalized)
        )

    unencodable_mandatory = list(
        dict.fromkeys(
            value
            for value in mandatory_candidates
            if literal_for(value, preserve=True) is None
        )
    )
    if unencodable_mandatory:
        raise _PlannerSchemaCapacityError(
            "the LLM-authored planning contract contains mandatory numeric "
            "values outside the provider-safe ±1e9 range: "
            + ", ".join(format(value, ".9g") for value in unencodable_mandatory[:4])
        )

    mandatory_literals = list(
        dict.fromkeys(
            literal
            for value in mandatory_candidates
            if (literal := literal_for(value, preserve=True)) is not None
        )
    )
    if len(mandatory_literals) > MAX_STRUCTURED_NUMBER_LITERALS:
        raise _PlannerSchemaCapacityError(
            "the LLM-authored planning contract requires "
            f"{len(mandatory_literals)} mandatory numeric literals; the "
            "provider-safe maximum is "
            f"{MAX_STRUCTURED_NUMBER_LITERALS}, so the authored intent must "
            "be simplified before planning"
        )
    mandatory_literal_bytes = len(
        json.dumps(
            mandatory_literals,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if mandatory_literal_bytes > MAX_STRUCTURED_NUMBER_LITERAL_BYTES:
        raise _PlannerSchemaCapacityError(
            "the LLM-authored planning contract requires "
            f"{mandatory_literal_bytes} serialized numeric-literal bytes; the "
            "provider-safe maximum is "
            f"{MAX_STRUCTURED_NUMBER_LITERAL_BYTES}, so the authored intent "
            "must be simplified before planning"
        )
    if not include_optional:
        return _PlannerNumericLiteralBundle(
            mandatory_literals=mandatory_literals,
            preferred_literals=list(mandatory_literals),
        )

    literal_limit = max(
        len(mandatory_literals),
        PLANNER_PREFERRED_NUMBER_LITERALS,
    )
    literal_limit = min(literal_limit, MAX_STRUCTURED_NUMBER_LITERALS)
    optional_byte_limit = max(
        mandatory_literal_bytes,
        PLANNER_PREFERRED_NUMBER_LITERAL_BYTES,
    )

    candidates: list[float] = [*priority_derived, *exact]
    seeds = list(dict.fromkeys([*mandatory_active, *exact]))[:32]
    for value in seeds:
        for factor in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0):
            candidates.append(value * factor)
    for index, left in enumerate(seeds[:20]):
        for right in seeds[index + 1 : 20]:
            candidates.extend((left + right, left - right, right - left))

    mandatory_values = set(mandatory_candidates)
    literals: list[str] = list(mandatory_literals)
    for value in candidates:
        if len(literals) >= literal_limit:
            break
        if value in mandatory_values:
            continue
        literal = literal_for(value, preserve=False)
        if literal is None:
            continue
        if literal not in literals:
            expanded = [*literals, literal]
            payload_bytes = len(
                json.dumps(
                    expanded,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            if payload_bytes > optional_byte_limit:
                continue
            literals.append(literal)
    return _PlannerNumericLiteralBundle(
        mandatory_literals=mandatory_literals,
        preferred_literals=literals,
    )


def _planner_numeric_literals(
    state: PipeState,
    *,
    include_optional: bool = True,
) -> list[str]:
    """기존 호출 계약에 맞춰 유한한 planner 숫자 어휘를 반환한다."""

    bundle = _build_planner_numeric_literal_bundle(
        state,
        include_optional=include_optional,
    )
    return bundle.preferred_literals if include_optional else bundle.mandatory_literals


_DEFAULT_PLANNER_NUMERIC_LITERALS = _planner_numeric_literals


def _planner_numeric_literal_lists_for_action(
    state: PipeState,
) -> tuple[list[str], list[str]]:
    """일반 실행은 한 번 계산하고 monkeypatch된 기존 함수는 그대로 호출한다."""

    if _planner_numeric_literals is not _DEFAULT_PLANNER_NUMERIC_LITERALS:
        # 테스트가 기존 진입점을 대체한 경우에는 이전과 같은 호출 순서와
        # 인자를 유지해 테스트 더블의 동작을 바꾸지 않는다.
        return (
            _planner_numeric_literals(state, include_optional=False),
            _planner_numeric_literals(state),
        )
    bundle = _build_planner_numeric_literal_bundle(state)
    return bundle.mandatory_literals, bundle.preferred_literals


def _intent_metric_values(intent: IntentResult) -> list[float]:
    alias_paths: set[str] = set()
    for goal_index, goal in enumerate(intent.target_behavior):
        if (
            goal.type == "connector"
            and goal.length is not None
            and goal.component_spec is not None
            and goal.component_spec.body_length is not None
            and _same_metric_value(goal.length, goal.component_spec.body_length)
        ):
            # One physical authored length is represented in both the connector
            # goal and its same-span body detail.  It must count once when source
            # measurement multiplicity is checked.
            alias_paths.add(f"target_behavior[{goal_index}].component_spec.body_length")
    values = [
        value
        for path, value in _positive_intent_dimensions(intent)
        if path not in alias_paths
    ]
    values.extend(float(component) for component in intent.start_position)
    text_contracts = [
        contract
        for contract in intent.hard_constraints
        if contract.startswith("unsupported:")
    ]
    for goal in intent.target_behavior:
        if goal.offset is not None:
            values.extend(float(component) for component in goal.offset)
        if goal.terminal_position is not None:
            values.extend(float(component) for component in goal.terminal_position)
        for waypoint in goal.required_waypoints:
            values.extend(float(component) for component in waypoint)
    for constraint in intent.geometric_constraints:
        if constraint.minimum is not None:
            values.extend(float(component) for component in constraint.minimum)
        if constraint.maximum is not None:
            values.extend(float(component) for component in constraint.maximum)
    for text in text_contracts:
        values.extend(_explicit_mm_values(text))
    return [value for value in values if math.isfinite(value)]


def _same_metric_value(left: float, right: float) -> bool:
    tolerance = max(1e-9, abs(left) * 1e-9)
    return abs(left - right) <= tolerance


def _plan_action(
    state: PipeState,
    *,
    dry_run: bool,
    gemini: GeminiClient | None,
    host_compiler_enabled: bool = False,
    repair_observations: list[dict[str, Any]] | None = None,
    reusable_suffix_context: dict[str, Any] | None = None,
) -> ActionDraft:
    if dry_run:
        return plan_next_action(state)
    if host_compiler_enabled and not repair_observations:
        compiled = compile_next_action(state)
        if compiled is not None:
            return compiled
    if gemini is None:
        raise RuntimeError("Gemini client is required outside dry-run mode.")
    schema_profile_before = _planner_schema_profile(gemini, state.state_id)
    force_geometry_decimals = (
        any(
            goal.type == "route" and goal.path_kind == "spline"
            for goal in state.remaining_goals
        )
        or any(
            str(item.get("check_name", ""))
            in {
                "route_continuity",
                "route_curvature",
                "intra_module_clearance",
                "freecad_semantic_validation",
                "freecad_geometry",
            }
            or "centerline_context"
            in json.dumps(item.get("actual") or {}, ensure_ascii=False)
            for item in (repair_observations or [])
            if isinstance(item, dict)
        )
        or _contains_numeric_diagnostic(repair_observations or [])
    )
    stagnation = next(
        (
            item
            for item in (repair_observations or [])
            if item.get("context_type") == "planner_stagnation"
        ),
        None,
    )
    if stagnation is not None:
        # A continuation anchored to the same rejected idea is counterproductive.
        # Re-send the complete immutable state/catalog. Precision-related stalls
        # also widen the exact decimal vocabulary; unrelated semantic stalls keep
        # their current grammar.
        if hasattr(gemini, "reset_lineage"):
            gemini.reset_lineage("step_planner")
        if stagnation.get("schema_strategy") == "encoded":
            _remember_planner_schema_profile(gemini, state.state_id, "encoded")
    geometry_grammar_changed = (
        force_geometry_decimals and schema_profile_before != "encoded"
    )
    if force_geometry_decimals:
        # A single finite numeric enum is shared by every JSON number field.
        # Angle vocabulary such as -180 is therefore legal in spline XYZ fields
        # and has produced catastrophic waypoint spikes in real repairs. Curved
        # geometry uses the exact decimal-object schema from the first call; a
        # retry also resets lineage before changing its response grammar.
        if (
            geometry_grammar_changed
            and hasattr(gemini, "has_previous")
            and gemini.has_previous("step_planner")
            and hasattr(gemini, "reset_lineage")
        ):
            gemini.reset_lineage("step_planner")
        _remember_planner_schema_profile(gemini, state.state_id, "encoded")
    has_lineage = (
        False
        if stagnation is not None or geometry_grammar_changed
        else bool(
            hasattr(gemini, "has_previous") and gemini.has_previous("step_planner")
        )
    )
    include_catalog = not has_lineage
    schema_profile = _planner_schema_profile(gemini, state.state_id)
    numeric_literals: list[str] = []
    mandatory_numeric_literals: list[str] = []
    if schema_profile != "encoded":
        try:
            (
                mandatory_numeric_literals,
                numeric_literals,
            ) = _planner_numeric_literal_lists_for_action(state)
        except _PlannerSchemaCapacityError:
            # The finite enum is an optimization, not a limit on authored CAD
            # complexity. Preserve the immutable state/goal values in the
            # request and move directly to the exact decimal-object grammar.
            schema_profile = "encoded"
            _remember_planner_schema_profile(
                gemini,
                state.state_id,
                "encoded",
            )
    selected_literals = (
        mandatory_numeric_literals
        if schema_profile == "mandatory"
        else None
        if schema_profile == "encoded"
        else numeric_literals
    )
    planner_schema = (
        PlannerDecisionWire
        if _needs_inline_component_planner(state)
        else CorePlannerDecisionWire
    )
    planner_thinking_level = (
        "medium"
        if stagnation is not None
        or any(
            goal.type == "route" and goal.path_kind == "spline"
            for goal in state.remaining_goals
        )
        or any(
            str(item.get("check_name", ""))
            in {
                "route_continuity",
                "route_curvature",
                "intra_module_clearance",
                "freecad_geometry",
                "freecad_semantic_validation",
                "visual_validation",
            }
            for item in (repair_observations or [])
        )
        else "low"
    )
    full_request = step_planner_prompt(
        state,
        include_catalog=True,
        repair_observations=repair_observations,
        reusable_suffix_context=reusable_suffix_context,
    )
    request = (
        step_lineage_repair_prompt(
            state,
            repair_observations,
            reusable_suffix_context,
        )
        if repair_observations and has_lineage
        else step_planner_prompt(
            state,
            include_catalog=include_catalog,
            repair_observations=repair_observations,
            reusable_suffix_context=reusable_suffix_context,
        )
    )
    selected_request = (
        _encoded_planner_request(request, mandatory_numeric_literals)
        if schema_profile == "encoded"
        else request
    )
    selected_full_request = (
        _encoded_planner_request(full_request, mandatory_numeric_literals)
        if schema_profile == "encoded"
        else full_request
    )

    def call_planner(
        planner_request: str,
        literals: list[str] | None,
    ) -> Any:
        return _call_structured(
            gemini,
            planner_request,
            planner_schema,
            part="step_planner",
            thinking_level=planner_thinking_level,
            numeric_literals=literals,
            system_instruction=step_planner_system_instruction(),
        )

    try:
        result = call_planner(selected_request, selected_literals)
    except StructuredOutputError as exc:
        # A malformed/incomplete schema response did not create a candidate and
        # therefore must not consume one of the semantic validation-repair
        # slots. Retry the same logical attempt once with fresh lineage and the
        # full state/catalog; a second protocol failure is journaled by the
        # outer loop as the terminal planning failure for this attempt.
        if hasattr(gemini, "reset_lineage"):
            gemini.reset_lineage("step_planner")
        protocol_request = selected_full_request + (
            "\n\nPLANNING_FAILED protocol diagnostic: the previous response did "
            "not satisfy the structured action schema. Return one complete "
            "replacement object; do not repeat partial or malformed JSON. "
            "Diagnostic: " + _intent_repair_diagnostic(exc)[:1000]
        )
        result = call_planner(protocol_request, selected_literals)
    except GeminiLineageError:
        try:
            result = call_planner(selected_full_request, selected_literals)
        except GeminiRequestError as exc:
            result = _retry_planner_with_reduced_schemas(
                exc,
                gemini=gemini,
                state_id=state.state_id,
                call_planner=call_planner,
                full_request=full_request,
                preferred_literals=numeric_literals,
                mandatory_literals=mandatory_numeric_literals,
                attempted_profile=schema_profile,
            )
    except GeminiRequestError as exc:
        result = _retry_planner_with_reduced_schemas(
            exc,
            gemini=gemini,
            state_id=state.state_id,
            call_planner=call_planner,
            full_request=full_request,
            preferred_literals=numeric_literals,
            mandatory_literals=mandatory_numeric_literals,
            attempted_profile=schema_profile,
        )
    if isinstance(
        result,
        (
            PlannerDecision,
            CorePlannerDecision,
            PlannerDecisionWire,
            CorePlannerDecisionWire,
        ),
    ):
        return result.to_action_draft()
    if isinstance(result, ActionDraft):
        # 테스트 더블이 이미 변환된 draft를 반환하는 경우에도 production과
        # 동일한 schema-v2 계약만 허용한다. 누락 goal이나 legacy primitive를
        # 여기서 보충하면 resolver의 fixture 기본값이 설계값으로 승격된다.
        if result.consumes_goal_index != 0:
            raise ValueError("Legacy step planner must use consumes_goal_index=0.")
        if result.catalog_schema_version != 2:
            raise ValueError(
                "Production step planner must return catalog_schema_version=2; "
                "legacy/default-filled actions are dry-run only."
            )
        return result
    raise TypeError(f"Unexpected planner result type: {type(result).__name__}")


def _contains_numeric_diagnostic(value: Any) -> bool:
    """Whether repair evidence needs values beyond the finite numeric enum."""

    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        return any(_contains_numeric_diagnostic(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_numeric_diagnostic(item) for item in value)
    return False


def _retry_planner_with_reduced_schemas(
    initial_error: GeminiRequestError,
    *,
    gemini: Any,
    state_id: str,
    call_planner: Any,
    full_request: str,
    preferred_literals: list[str],
    mandatory_literals: list[str],
    attempted_profile: str,
) -> Any:
    """Negotiate a smaller planner schema after a provider HTTP 400.

    No model draft exists at this point, so this bounded protocol fallback is a
    single logical planning attempt rather than a semantic repair.  Each retry
    changes the response grammar materially; unrelated request failures remain
    fatal and are never repeated here.
    """

    if not _is_invalid_planner_request(initial_error):
        raise initial_error

    if hasattr(gemini, "reset_lineage"):
        gemini.reset_lineage("step_planner")

    if attempted_profile == "preferred" and mandatory_literals != preferred_literals:
        _remember_planner_schema_profile(gemini, state_id, "mandatory")
        try:
            return call_planner(full_request, mandatory_literals)
        except GeminiRequestError as exc:
            if not _is_invalid_planner_request(exc):
                raise
            if hasattr(gemini, "reset_lineage"):
                gemini.reset_lineage("step_planner")

    if attempted_profile == "encoded":
        raise initial_error

    _remember_planner_schema_profile(gemini, state_id, "encoded")
    encoded_request = _encoded_planner_request(full_request, mandatory_literals)
    try:
        return call_planner(encoded_request, None)
    except GeminiRequestError as exc:
        if not _is_invalid_planner_request(exc):
            raise
        raise GeminiInvalidRequestError(
            "Gemini rejected the preferred numeric enum, mandatory-only enum, "
            "and encoded-decimal planner schemas; last provider error: "
            f"{exc}",
            status_code=getattr(exc, "status_code", 400),
            provider_code=getattr(exc, "provider_code", "invalid_request"),
        ) from exc


def _encoded_planner_request(
    request: str,
    mandatory_literals: list[str],
) -> str:
    grounding = (
        " Ground authored geometry in these mandatory state values: "
        + json.dumps(
            mandatory_literals,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if mandatory_literals
        else " All exact authored geometry remains present in the state payload."
    )
    return (
        request + "\n\nThe provider could not compile the enum-constrained numeric "
        "response grammar. Use the bounded decimal-object representation in "
        "the current response schema. Preserve every value's decimal scale: "
        "increase p to retain fractional digits and never shift a decimal point "
        "to satisfy the coefficient field." + grounding
    )


def _planner_schema_profile(gemini: Any, state_id: str) -> str:
    profiles = getattr(gemini, _PLANNER_SCHEMA_PROFILE_ATTR, None)
    if not isinstance(profiles, dict):
        return "preferred"
    profile = profiles.get(state_id)
    return profile if profile in _PLANNER_SCHEMA_PROFILES else "preferred"


def _remember_planner_schema_profile(
    gemini: Any,
    state_id: str,
    profile: str,
) -> None:
    if profile not in _PLANNER_SCHEMA_PROFILES:
        raise ValueError(f"Unknown planner schema profile: {profile}")
    profiles = getattr(gemini, _PLANNER_SCHEMA_PROFILE_ATTR, None)
    if not isinstance(profiles, dict):
        profiles = {}
        setattr(gemini, _PLANNER_SCHEMA_PROFILE_ATTR, profiles)
    profiles[state_id] = profile


def _is_invalid_planner_request(exc: GeminiRequestError) -> bool:
    if isinstance(exc, GeminiInvalidRequestError):
        provider_code = (exc.provider_code or "").lower()
        message = str(exc).lower()
        return exc.status_code == 400 and (
            provider_code in {"invalid_request", "invalid_argument"}
            or any(
                marker in message
                for marker in (
                    "invalid_request",
                    "invalid_argument",
                    "invalid argument",
                )
            )
        )
    message = str(exc).lower()
    return "400" in message and any(
        marker in message
        for marker in ("invalid_request", "invalid_argument", "invalid argument")
    )


def _needs_inline_component_planner(state: PipeState) -> bool:
    required = Counter(state.required_components)
    placed = Counter(
        str(module.params.get("component_type"))
        for module in state.placed_modules
        if module.type == "inline_component"
        and module.params.get("component_type") is not None
    )
    return bool(required - placed)


def _call_structured(
    gemini: Any,
    prompt: Any,
    schema: type[Any],
    *,
    part: str,
    thinking_level: str,
    numeric_literals: list[str] | None = None,
    numeric_schema_mode: str | None = None,
    system_instruction: str | None = None,
) -> Any:
    provider_wire_parts = {
        "intent",
        "intent_repair_advisor",
        "intent_repair_reviewer",
        "step_planner",
        "patch",
        "visual_validator",
        "step_repair_advisor",
        "parameter",
    }
    if part in provider_wire_parts and not getattr(
        schema, "provider_wire_contract", False
    ):
        raise GeminiConfigError(
            f"Structured part {part} must use a ProviderWireModel; "
            f"received {schema.__name__}"
        )
    if getattr(gemini, "supports_interaction_controls", False):
        kwargs: dict[str, Any] = {
            "part": part,
            "thinking_level": thinking_level,
        }
        if numeric_literals is not None and getattr(
            gemini, "supports_numeric_literals", False
        ):
            kwargs["numeric_literals"] = numeric_literals
        if numeric_schema_mode is not None and getattr(
            gemini, "supports_numeric_schema_modes", False
        ):
            kwargs["numeric_schema_mode"] = numeric_schema_mode
        if system_instruction is not None and getattr(
            gemini, "supports_system_instruction", False
        ):
            kwargs["system_instruction"] = system_instruction
        return gemini.stream_structured(prompt, schema, **kwargs)
    return gemini.stream_structured(prompt, schema, part=part)


def _normalized_source_fragment(value: str) -> str:
    """Normalize a verbatim source fragment without translating or paraphrasing."""

    return " ".join(str(value).casefold().split())


def _intent_source_provenance(
    prompt: str | None,
    values: list[str],
    *,
    strip_unsupported_prefix: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    """Return deterministic verbatim provenance for terminal scope evidence."""

    normalized_prompt = _normalized_source_fragment(prompt or "")
    records: list[dict[str, Any]] = []
    for raw_value in values:
        source_fragment = str(raw_value)
        if strip_unsupported_prefix and source_fragment.startswith("unsupported:"):
            source_fragment = source_fragment.split(":", 1)[1]
        normalized_fragment = _normalized_source_fragment(source_fragment)
        source_authored = bool(
            normalized_fragment
            and normalized_prompt
            and re.search(
                rf"(?<!\w){re.escape(normalized_fragment)}(?!\w)",
                normalized_prompt,
            )
        )
        records.append(
            {
                "value": str(raw_value),
                "source_fragment": source_fragment,
                "source_authored": source_authored,
            }
        )
    return records, bool(records) and all(
        item["source_authored"] is True for item in records
    )


def _intent_scope_issues(
    intent: IntentResult,
    *,
    dry_run: bool,
    prompt: str | None = None,
) -> list[StaticIssue]:
    """Return every catalog/scope rejection before an intent is accepted."""

    issues: list[StaticIssue] = []
    nonbinary_branches = _nonbinary_branch_goal_ids(intent)
    if not dry_run and nonbinary_branches:
        issues.append(
            _issue(
                0,
                "NON_BINARY_BRANCH_CONTRACT",
                "Production branch goals must each describe one binary junction.",
                phase="intent_scope",
                expected={"total_outlets_per_branch_goal": 2},
                actual={"nonbinary_branch_goal_ids": nonbinary_branches},
            )
        )

    unsupported_components = _unsupported_required_components(intent)
    if unsupported_components:
        component_provenance, component_provenance_complete = _intent_source_provenance(
            prompt, unsupported_components
        )
        issues.append(
            _issue(
                0,
                "UNSUPPORTED_REQUIRED_COMPONENT",
                (
                    "The LLM intent contains an accessory outside the explicit "
                    "geometry catalog."
                ),
                phase="intent_scope",
                expected={"supported_components": list(SUPPORTED_INLINE_COMPONENTS)},
                actual={
                    "unsupported_components": unsupported_components,
                    "source_provenance": component_provenance,
                    "source_provenance_complete": (component_provenance_complete),
                },
            )
        )

    component_contract_error = _component_contract_error(intent)
    if component_contract_error is not None:
        issues.append(
            _issue(
                0,
                "INCONSISTENT_COMPONENT_CONTRACT",
                (
                    "Required accessory multiplicity must match distinct "
                    "connector goals."
                ),
                phase="intent_scope",
                expected=component_contract_error["expected"],
                actual=component_contract_error["actual"],
            )
        )

    if intent.hard_constraints:
        hard_constraint_provenance, hard_constraint_provenance_complete = (
            _intent_source_provenance(
                prompt,
                list(intent.hard_constraints),
                strip_unsupported_prefix=True,
            )
        )
        issues.append(
            _issue(
                0,
                "UNSUPPORTED_HARD_CONSTRAINT",
                (
                    "The LLM preserved a hard constraint that has no "
                    "deterministic predicate."
                ),
                phase="intent_scope",
                expected={"hard_constraints": []},
                actual={
                    "hard_constraints": list(intent.hard_constraints),
                    "source_provenance": hard_constraint_provenance,
                    "source_provenance_complete": (hard_constraint_provenance_complete),
                },
            )
        )
    return issues


def _validate_intent_scope(
    intent: IntentResult,
    *,
    dry_run: bool,
    prompt: str | None = None,
) -> None:
    """Raise a structured, repair-loop-aware failure for scope-invalid intent."""

    issues = _intent_scope_issues(intent, dry_run=dry_run, prompt=prompt)
    if issues:
        raise _IntentScopeValidationError(issues)


def _bind_contract(prompt: str, intent: IntentResult) -> IntentResult:
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    goals = [
        goal if goal.goal_id else goal.model_copy(update={"goal_id": f"G{index}"})
        for index, goal in enumerate(intent.target_behavior, start=1)
    ]
    contract = {
        "prompt_sha256": prompt_hash,
        "global_spec": intent.global_spec.model_dump(mode="json"),
        "start_position": list(intent.start_position),
        "start_axis": list(intent.start_axis),
        "expected_open_ports": intent.expected_open_ports,
        "expected_open_ports_source": intent.expected_open_ports_source,
        "required_components": intent.required_components,
        "hard_constraints": intent.hard_constraints,
        "geometric_constraints": [
            item.model_dump(mode="json") for item in intent.geometric_constraints
        ],
        "design_notes": intent.design_notes,
        "target_behavior": [goal.model_dump(mode="json") for goal in goals],
        "required_terminal_vectors": [
            list(vector) for goal in goals for vector in goal.required_outlet_vectors
        ],
    }
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return intent.model_copy(
        update={
            "prompt_sha256": prompt_hash,
            "contract_digest": digest,
            "target_behavior": goals,
        }
    )


def _unsupported_required_components(intent: IntentResult) -> list[str]:
    supported = set(SUPPORTED_INLINE_COMPONENTS)
    return sorted(
        {
            component
            for component in [
                *intent.required_components,
                *[
                    goal.component
                    for goal in intent.target_behavior
                    if goal.component is not None
                ],
            ]
            if component not in supported
        }
    )


def _nonbinary_branch_goal_ids(intent: IntentResult) -> list[str]:
    invalid: list[str] = []
    for index, goal in enumerate(intent.target_behavior):
        if goal.type != "branch":
            continue
        authored_outlets = (
            goal.branch_count
            or len(goal.required_outlet_directions)
            or len(goal.required_outlet_vectors)
            or len(goal.required_outlets)
        )
        include_primary = (
            goal.include_primary_outlet
            if goal.include_primary_outlet is not None
            else not bool(goal.required_outlet_vectors or goal.required_outlets)
        )
        if authored_outlets + int(include_primary) != 2:
            invalid.append(goal.goal_id or f"target_behavior[{index}]")
    return invalid


def _component_contract_error(intent: IntentResult) -> dict[str, Any] | None:
    required = Counter(intent.required_components)
    goals = Counter(
        goal.component
        for goal in intent.target_behavior
        if goal.type == "connector" and goal.component is not None
    )
    if required == goals:
        return None
    return {
        "expected": {"required_component_counts": dict(sorted(required.items()))},
        "actual": {"connector_goal_counts": dict(sorted(goals.items()))},
    }


def _immutable_contract(intent: IntentResult) -> dict[str, Any]:
    return {
        "prompt_sha256": intent.prompt_sha256,
        "contract_digest": intent.contract_digest,
        "global_spec": intent.global_spec.model_dump(mode="json"),
        "start_position": list(intent.start_position),
        "start_axis": list(intent.start_axis),
        "expected_open_ports": intent.expected_open_ports,
        "expected_open_ports_source": intent.expected_open_ports_source,
        "required_components": intent.required_components,
        "hard_constraints": intent.hard_constraints,
        "geometric_constraints": [
            item.model_dump(mode="json") for item in intent.geometric_constraints
        ],
        "design_notes": intent.design_notes,
        "target_behavior": [
            goal.model_dump(mode="json") for goal in intent.target_behavior
        ],
    }


def _validate_agenda_repair_directive(
    directive: Any,
    *,
    state: PipeState,
    critic: CriticReport,
    actions: list[dict[str, Any]],
    checkpoints: list[PipeState],
) -> tuple[AgendaRepairDirective, int, PipeState, list[dict[str, Any]]]:
    if not isinstance(directive, AgendaRepairDirective):
        raise TypeError("Final repair did not return AgendaRepairDirective")
    rollback_index = directive.rollback_step - 1
    if (
        rollback_index < 0
        or directive.rollback_step > len(actions)
        or rollback_index >= len(checkpoints)
    ):
        raise ValueError("rollback_step is outside the committed checkpoint range")
    errors_by_id = {
        issue.issue_id: issue for issue in critic.issues if issue.severity == "error"
    }
    unknown_issue_ids = sorted(set(directive.target_issue_ids) - set(errors_by_id))
    if unknown_issue_ids:
        raise ValueError(
            "Final repair referenced unknown issue IDs: " + ", ".join(unknown_issue_ids)
        )
    module_steps = {
        module.id: index for index, module in enumerate(state.placed_modules, start=1)
    }
    unknown_module_ids = sorted(set(directive.target_module_ids) - set(module_steps))
    if unknown_module_ids:
        raise ValueError(
            "Final repair referenced unknown module IDs: "
            + ", ".join(unknown_module_ids)
        )
    issue_localized_steps = [
        issue.step_index
        for issue_id in directive.target_issue_ids
        for issue in [errors_by_id[issue_id]]
        if issue.step_index is not None and issue.step_index > 0
    ]
    # Issue localization is host-derived from failure adapters.  Implicated
    # module IDs may include an already-valid parent in an overlap pair and are
    # evidence labels, not authority to force an earlier rollback.
    localized_steps = issue_localized_steps or [
        module_steps[module_id] for module_id in directive.target_module_ids
    ]
    if localized_steps and directive.rollback_step > min(localized_steps):
        raise ValueError(
            "rollback_step is later than the earliest localized defect step"
        )
    restored = checkpoints[rollback_index].model_copy(deep=True)
    observations = []
    for issue_id in directive.target_issue_ids:
        observation = _repair_observation(errors_by_id[issue_id])
        observation["agenda_repair"] = {
            "repair_hint": directive.repair_hint,
            "rationale": directive.rationale,
            "target_module_ids": directive.target_module_ids,
            "rollback_step": directive.rollback_step,
        }
        observations.append(observation)
    return directive, rollback_index, restored, observations


def _build_preserved_suffix(
    *,
    repair_start_step: int,
    actions: list[dict[str, Any]],
    attempts: list[ActionAttempt],
    checkpoints: list[PipeState],
    repair_hint: str,
) -> _PreservedSuffix | None:
    """Keep the accepted tail so a local repair can rejoin and replay it."""

    parsed_actions = [ResolvedAction.model_validate(item) for item in actions]
    if not (1 <= repair_start_step < len(parsed_actions)):
        return None
    if len(checkpoints) != len(parsed_actions) + 1:
        return None
    try:
        drafts = [
            _accepted_draft_for_action(action, attempts) for action in parsed_actions
        ]
    except (KeyError, TypeError, ValueError):
        # Suffix reuse is an optimization. An old/legacy action journal that
        # cannot be reconstructed must fall back to ordinary replanning.
        return None
    return _PreservedSuffix(
        repair_start_step=repair_start_step,
        original_actions=parsed_actions,
        original_drafts=drafts,
        original_checkpoints=[item.model_copy(deep=True) for item in checkpoints],
        repair_hint=repair_hint,
    )


def _accepted_draft_for_action(
    action: ResolvedAction,
    attempts: list[ActionAttempt],
) -> ActionDraft:
    expected = action.model_dump(mode="json")
    for attempt in reversed(attempts):
        if (
            attempt.status != "accepted"
            or attempt.draft is None
            or attempt.resolved is None
        ):
            continue
        try:
            resolved = ResolvedAction.model_validate(attempt.resolved)
            if resolved.model_dump(mode="json") == expected:
                stored = ActionDraft.model_validate(attempt.draft)
                return stored.model_copy(
                    deep=True,
                    update={
                        "params": filter_draft_params(
                            stored.module,
                            copy.deepcopy(stored.params),
                        )
                    },
                )
        except (TypeError, ValueError):
            continue
    return ActionDraft(
        target_port=action.target_port,
        module=action.module,  # type: ignore[arg-type]
        params=filter_draft_params(action.module, copy.deepcopy(action.params)),
        catalog_schema_version=2,
        affected_goal_ids=list(action.affected_goal_ids),
        completed_goal_ids=list(action.completed_goal_ids),
        satisfied_components=list(action.satisfied_components),
        rationale="Recovered from the accepted resolved-action journal.",
    )


def _suffix_rejoin_context(
    preserved: _PreservedSuffix | None,
    state: PipeState,
    *,
    limit: int = 3,
) -> dict[str, Any] | None:
    """Give the planner soft interface targets, not brittle absolute coordinates."""

    if preserved is None:
        return None
    current_goals = _remaining_goal_ids(state)
    candidates: list[dict[str, Any]] = []
    for boundary_step in range(
        preserved.repair_start_step,
        len(preserved.original_actions),
    ):
        boundary = preserved.original_checkpoints[boundary_step]
        boundary_goals = _remaining_goal_ids(boundary)
        if not _ordered_suffix(boundary_goals, current_goals):
            continue
        ports = list(boundary.open_ports)
        anchor = vec(ports[0].position) if ports else (0.0, 0.0, 0.0)
        candidates.append(
            {
                "after_original_step": boundary_step,
                "reuse_starts_at_original_step": boundary_step + 1,
                "remaining_goal_ids": boundary_goals,
                "translation_flexible": True,
                "open_port_interfaces": [
                    {
                        "role": port.id.split(".", 1)[-1],
                        "axis": _rounded_vector(port.axis),
                        "relative_position": _rounded_vector(
                            tuple(
                                float(port.position[index]) - anchor[index]
                                for index in range(3)
                            )
                        ),
                        "outer_diameter": port.outer_diameter,
                        "wall_thickness": port.wall_thickness,
                        "connector_type": port.connector_type,
                        "connector_gender": port.connector_gender,
                        "connector_standard": port.connector_standard,
                    }
                    for port in ports
                ],
                "reusable_action_count": len(preserved.original_actions)
                - boundary_step,
            }
        )
        if len(candidates) >= limit:
            break
    if not candidates:
        return None
    return {
        "objective": (
            "Absorb the repair deviation in the smallest number of actions, then "
            "recover one compatible interface so the preserved suffix can replay."
        ),
        "flexibility": (
            "These are soft reuse targets, not new user constraints. Absolute "
            "position may translate uniformly; topology, axis, section, connector, "
            "and relative multi-port layout must match. Immutable user-authored "
            "goals always take precedence."
        ),
        "repair_hint": preserved.repair_hint,
        "candidate_interfaces": candidates,
    }


def _try_replay_preserved_suffix(
    *,
    preserved: _PreservedSuffix | None,
    state: PipeState,
    actions: list[dict[str, Any]],
    step_verifications: list[StepVerification],
    engine: StateEngine,
    intent: IntentResult,
    settings: Settings,
) -> _SuffixReplayResult | None:
    """Replay the longest safe old tail after an early compatible rejoin."""

    if preserved is None or settings.freecad_step_mcp_enabled:
        # Per-step MCP transactions publish state as they go. Replaying those
        # atomically needs a separate transaction protocol; keep current behavior
        # when that optional mode is enabled.
        return None
    for boundary_step in range(
        preserved.repair_start_step,
        len(preserved.original_actions),
    ):
        boundary = preserved.original_checkpoints[boundary_step]
        if _remaining_goal_ids(state) != _remaining_goal_ids(boundary):
            continue
        matched = _match_rejoin_ports(boundary, state)
        if matched is None:
            continue
        port_mapping, translation = matched
        reuse_count = len(preserved.original_actions) - boundary_step
        if len(actions) + reuse_count > settings.max_iter:
            continue
        replayed = _replay_suffix_from_boundary(
            preserved=preserved,
            boundary_step=boundary_step,
            state=state,
            existing_steps=step_verifications,
            port_mapping=port_mapping,
            translation=translation,
            engine=engine,
            intent=intent,
        )
        if replayed is not None:
            return replayed
    return None


def _replay_suffix_from_boundary(
    *,
    preserved: _PreservedSuffix,
    boundary_step: int,
    state: PipeState,
    existing_steps: list[StepVerification],
    port_mapping: dict[str, str],
    translation: tuple[float, float, float],
    engine: StateEngine,
    intent: IntentResult,
) -> _SuffixReplayResult | None:
    working = state.model_copy(deep=True)
    mapping = dict(port_mapping)
    replayed_actions: list[ResolvedAction] = []
    replayed_steps: list[StepVerification] = []
    replayed_checkpoints: list[PipeState] = []
    replayed_attempts: list[ActionAttempt] = []
    reused_original_steps: list[int] = []

    for old_action_index in range(
        boundary_step,
        len(preserved.original_actions),
    ):
        original_step = old_action_index + 1
        old_draft = preserved.original_drafts[old_action_index]
        try:
            rebound = _rebind_replay_draft(old_draft, mapping, translation)
            draft_check = validate_draft(rebound, working)
            if not draft_check.valid:
                return None
            resolved = engine.resolve_action(rebound, working)
            action_check = validate_action(resolved, working)
            if not action_check.valid:
                return None
            before = working
            candidate = engine.apply_action(resolved, before)
            verification = build_step_verification(
                before,
                resolved,
                candidate,
                intent,
                candidate.state_version,
                mcp_required=False,
                skipped_mcp_reason=(
                    "Accepted suffix replay; final FreeCAD validation remains authoritative."
                ),
            )
            if has_errors(verification.issues):
                return None
        except (KeyError, TypeError, ValueError):
            return None

        old_after = preserved.original_checkpoints[original_step]
        old_module = old_after.placed_modules[old_action_index]
        new_module = candidate.placed_modules[-1]
        if old_module.type != new_module.type or set(old_module.ports) != set(
            new_module.ports
        ):
            return None
        for local_name, old_port in old_module.ports.items():
            mapping[old_port.id] = new_module.ports[local_name].id

        replayed_actions.append(resolved)
        replayed_steps.append(verification)
        replayed_checkpoints.append(candidate.model_copy(deep=True))
        replayed_attempts.append(
            _attempt(
                candidate.state_version,
                0,
                before,
                "suffix_replay",
                "accepted",
                rebound,
                resolved,
                verification.issues,
            )
        )
        reused_original_steps.append(original_step)
        working = candidate

    if working.remaining_goals:
        return None
    critic = build_final_critic_report(
        intent,
        working,
        [*existing_steps, *replayed_steps],
        skipped_mcp_reason="Suffix replay awaits final FreeCAD validation.",
    )
    evidence_only = {
        "GEOMETRIC_CONSTRAINT_REQUIRES_FREECAD_LENGTH",
        "GEOMETRIC_CONSTRAINT_REQUIRES_FREECAD_BOUNDS",
        "GOAL_LENGTH_REQUIRES_FREECAD",
        "SPLINE_CURVATURE_REQUIRES_FREECAD",
    }
    if any(
        issue.severity == "error" and issue.issue_code not in evidence_only
        for issue in critic.issues
    ):
        return None
    return _SuffixReplayResult(
        state=working,
        actions=replayed_actions,
        step_verifications=replayed_steps,
        checkpoints=replayed_checkpoints,
        attempts=replayed_attempts,
        rejoin_original_step=boundary_step,
        reused_original_steps=reused_original_steps,
    )


def _rebind_replay_draft(
    draft: ActionDraft,
    port_mapping: dict[str, str],
    translation: tuple[float, float, float],
) -> ActionDraft:
    params = copy.deepcopy(draft.params)
    if "other_port_id" in params:
        params["other_port_id"] = port_mapping[str(params["other_port_id"])]
    if "waypoints" in params and params.get("waypoint_frame", "global") == "global":
        params["waypoints"] = [
            [float(point[index]) + translation[index] for index in range(3)]
            for point in params["waypoints"]
        ]
    params = filter_draft_params(draft.module, params)
    return draft.model_copy(
        deep=True,
        update={
            "target_port": port_mapping[draft.target_port],
            "params": params,
            "rationale": (
                "System replay of an accepted suffix after a compatible local rejoin."
            ),
        },
    )


def _match_rejoin_ports(
    old_state: PipeState,
    new_state: PipeState,
) -> tuple[dict[str, str], tuple[float, float, float]] | None:
    old_ports = list(old_state.open_ports)
    new_ports = list(new_state.open_ports)
    if len(old_ports) != len(new_ports):
        return None
    if not old_ports:
        return ({}, (0.0, 0.0, 0.0))

    candidates: dict[str, list[Any]] = {
        old.id: [new for new in new_ports if _rejoin_port_shape_compatible(old, new)]
        for old in old_ports
    }
    if any(not values for values in candidates.values()):
        return None
    ordered = sorted(old_ports, key=lambda port: len(candidates[port.id]))
    position_tolerance = (
        max(
            old_state.modeling_tolerance,
            new_state.modeling_tolerance,
            1e-4,
        )
        * 10.0
    )

    def search(
        index: int,
        mapping: dict[str, str],
        used: set[str],
        translation: tuple[float, float, float] | None,
    ) -> tuple[dict[str, str], tuple[float, float, float]] | None:
        if index == len(ordered):
            return mapping, translation or (0.0, 0.0, 0.0)
        old = ordered[index]
        choices = sorted(
            candidates[old.id],
            key=lambda port: (port.id != old.id, port.id),
        )
        for new in choices:
            if new.id in used:
                continue
            delta = tuple(
                float(new.position[axis]) - float(old.position[axis])
                for axis in range(3)
            )
            if translation is not None and not _vectors_near(
                delta,
                translation,
                position_tolerance,
            ):
                continue
            result = search(
                index + 1,
                {**mapping, old.id: new.id},
                {*used, new.id},
                translation if translation is not None else delta,
            )
            if result is not None:
                return result
        return None

    return search(0, {}, set(), None)


def _rejoin_port_shape_compatible(old: Any, new: Any) -> bool:
    dimension_tolerance = 1e-6 * max(
        1.0,
        abs(float(old.outer_diameter)),
        abs(float(new.outer_diameter)),
    )
    return (
        dot(normalize(vec(old.axis)), normalize(vec(new.axis))) >= 0.9999
        and abs(float(old.outer_diameter) - float(new.outer_diameter))
        <= dimension_tolerance
        and abs(float(old.wall_thickness) - float(new.wall_thickness))
        <= dimension_tolerance
        and old.connector_type == new.connector_type
        and old.connector_gender == new.connector_gender
        and old.connector_standard == new.connector_standard
    )


def _remaining_goal_ids(state: PipeState) -> list[str]:
    return [
        goal.goal_id or f"remaining_{index}"
        for index, goal in enumerate(state.remaining_goals, start=1)
    ]


def _ordered_suffix(candidate: list[str], full: list[str]) -> bool:
    if len(candidate) > len(full):
        return False
    if not candidate:
        return True
    return full[-len(candidate) :] == candidate


def _vectors_near(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    tolerance: float,
) -> bool:
    return (
        math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))
        <= tolerance
    )


def _rounded_vector(value: Any) -> list[float]:
    return [round(float(item), 6) for item in value]


def _validate_and_publish_freecad(
    settings: Settings,
    state: PipeState,
    *,
    run_id: str,
    attempt_id: int,
    raw_result_path: Path,
    validation_path: Path,
    evidence_validator: Any = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """후보 B-Rep을 생성ㆍ검증한 뒤 digest가 같은 문서만 게시한다."""

    try:
        raw_result_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        script = build_freecad_script(
            state,
            run_id=run_id,
            attempt_id=attempt_id,
            modeling_tolerance=settings.modeling_tolerance,
        )
        raw = asyncio.run(execute_freecad_code(settings, script))
        _atomic_write_json(raw_result_path, raw)
        digest = geometry_payload_digest(state)
        try:
            geometry_modules = [
                module
                for module in state.placed_modules
                if not (
                    module.type == "connect_ports"
                    and module.params.get("path_kind") == "seam"
                )
            ]
            evidence = assess_freecad_validation(
                raw,
                expected_digest=digest,
                expected_state_id=state.state_id,
                expected_module_ids=[module.id for module in geometry_modules],
                expected_internal_section_module_count=sum(
                    module.type not in {"terminate", "cap_pipe"}
                    for module in geometry_modules
                ),
                expected_open_port_count=len(state.open_ports),
                expected_anchored_inlet_count=(anchored_inlet_count(state)),
                expected_generator_version=GENERATOR_VERSION,
                expected_run_id=run_id,
                expected_state_version=state.state_version,
                expected_attempt_id=attempt_id,
                expected_candidate_document=_candidate_document_name(
                    state,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    digest=digest,
                ),
            )
        except FreeCADMCPError as exc:
            if isinstance(
                exc, FreeCADValidationError
            ) or _is_semantic_freecad_validation_error(str(exc)):
                rejected_evidence = getattr(exc, "evidence", None)
                if isinstance(rejected_evidence, dict):
                    # 거절된 B-Rep도 유효한 진단 근거다. 중단 시 transport 수준
                    # MCP envelope만 남지 않도록 cleanup 전에 먼저 저장한다.
                    _atomic_write_json(validation_path, rejected_evidence)
                try:
                    cleanup_script = _build_freecad_candidate_cleanup_script(
                        state,
                        run_id=run_id,
                        attempt_id=attempt_id,
                        payload_digest=digest,
                    )
                    asyncio.run(execute_freecad_code(settings, cleanup_script))
                except Exception:
                    pass
                raise _FreeCADSemanticError(
                    str(exc),
                    rejected_evidence,
                ) from exc
            raise
        if evidence_validator is not None:
            try:
                evidence_validator(evidence)
            except Exception as exc:
                rejected_evidence = (
                    exc.evidence
                    if isinstance(exc, _FreeCADSemanticError) and exc.evidence
                    else evidence
                )
                # host 결정론 검사는 FreeCAD 근거에 추가 반례를 붙일 수 있다.
                # resume과 diagnostician이 축약된 ActionAttempt에서 손실된 case를
                # 재구성하지 않도록 cleanup 전에 보강 payload를 그대로 저장한다.
                _atomic_write_json(validation_path, rejected_evidence)
                try:
                    cleanup_script = _build_freecad_candidate_cleanup_script(
                        state,
                        run_id=run_id,
                        attempt_id=attempt_id,
                        payload_digest=digest,
                    )
                    asyncio.run(execute_freecad_code(settings, cleanup_script))
                except Exception:
                    pass
                raise
        _atomic_write_json(validation_path, evidence)
        document_path = _freecad_document_path(
            raw_result_path,
            state,
            payload_digest=digest,
        )
        publish_script = _build_freecad_publish_script(
            state,
            run_id=run_id,
            attempt_id=attempt_id,
            fcstd_path=str(document_path),
            candidate_shape_fingerprints=evidence["candidate_shape_fingerprints"],
            payload_digest=digest,
        )
        publish_raw = asyncio.run(execute_freecad_code(settings, publish_script))
        assess_freecad_publish(
            publish_raw,
            expected_digest=digest,
            expected_document=published_document_name(state, run_id=run_id),
            expected_fcstd_path=str(document_path),
        )
        _atomic_write_json(
            _freecad_artifact_manifest_path(raw_result_path),
            {
                "state_id": state.state_id,
                "state_version": state.state_version,
                "payload_digest": digest,
                "fcstd_path": str(document_path),
            },
        )
        return raw, evidence, publish_raw
    except FreeCADMCPError:
        raise
    except Exception as exc:
        raise FreeCADMCPError(f"FreeCAD MCP transaction failed: {exc}") from exc


def _freecad_measurements(evidence: dict[str, Any]) -> dict[str, dict[str, float]]:
    centerlines = (evidence.get("checks") or {}).get("centerlines") or {}
    measurements: dict[str, dict[str, float]] = {}
    if not isinstance(centerlines, dict):
        return measurements
    for module_id, values in centerlines.items():
        if not isinstance(values, dict):
            continue
        module_measurements: dict[str, float] = {}
        if values.get("curve_length") is not None:
            try:
                curve_length = float(values["curve_length"])
            except (TypeError, ValueError):
                curve_length = -1.0
            if math.isfinite(curve_length) and curve_length >= 0.0:
                module_measurements["centerline_length"] = curve_length
        if values.get("minimum_radius") is not None:
            try:
                minimum_radius = float(values["minimum_radius"])
            except (TypeError, ValueError):
                minimum_radius = -1.0
            if math.isfinite(minimum_radius) and minimum_radius > 0.0:
                module_measurements["minimum_curvature_radius"] = minimum_radius
        elif values.get("zero_curvature") is True:
            # A straight B-spline has an infinite curvature radius.  Keep the
            # wire evidence strict-JSON-safe (null + boolean) and bind the
            # finite authored threshold as the proven lower bound here.
            try:
                required_radius = float(values["required_radius"])
            except (KeyError, TypeError, ValueError):
                required_radius = -1.0
            if math.isfinite(required_radius) and required_radius > 0.0:
                module_measurements["minimum_curvature_radius"] = required_radius
        if module_measurements:
            measurements[str(module_id)] = module_measurements
    return measurements


def _compact_freecad_failure_evidence(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    checks = evidence.get("checks") if isinstance(evidence, dict) else None
    if not isinstance(checks, dict):
        return {"module_ids": [], "failed_checks": {}}
    module_ids: set[str] = set()
    failed_checks: dict[str, Any] = {}
    for check_name in ("assembly", "outer_network", "bore_network"):
        value = checks.get(check_name)
        if isinstance(value, dict) and value.get("passed") is not True:
            failed_checks[check_name] = value
            for module_id in value.get("intrusion_module_ids") or []:
                if isinstance(module_id, str):
                    module_ids.add(module_id)
    for check_name in ("modules", "centerlines"):
        values = checks.get(check_name)
        if not isinstance(values, dict):
            continue
        failed = {
            str(module_id): (
                _compact_centerline_repair_context(value)
                if check_name == "centerlines"
                else value
            )
            for module_id, value in values.items()
            if isinstance(value, dict) and value.get("passed") is not True
        }
        if failed:
            failed_checks[check_name] = dict(list(failed.items())[:8])
            module_ids.update(failed)
    for check_name in (
        "module_errors",
        "assembly_errors",
        "non_adjacent_overlaps",
        "connection_failures",
        "terminal_bore_failures",
        "anchored_inlet_bore_failures",
        "termination_seal_failures",
        "wall_section_failures",
        "deterministic_constraint_failures",
    ):
        values = checks.get(check_name)
        if not isinstance(values, list) or not values:
            continue
        bounded = values[:8]
        failed_checks[check_name] = bounded
        for value in bounded:
            if not isinstance(value, dict):
                continue
            module_id = value.get("module_id")
            if isinstance(module_id, str):
                module_ids.add(module_id)
            for pair_module_id in value.get("module_ids") or []:
                if isinstance(pair_module_id, str):
                    module_ids.add(pair_module_id)
            candidate_port_ids = [
                value.get("port_id"),
                value.get("port_a_id"),
                value.get("port_b_id"),
                *(value.get("port_ids") or []),
            ]
            for port_id in candidate_port_ids:
                if port_id == "START":
                    module_ids.add("M1")
                elif (
                    isinstance(port_id, str)
                    and port_id.startswith("M")
                    and "." in port_id
                ):
                    module_ids.add(port_id.split(".", 1)[0])
    # A solid-construction failure can coexist with a valid centerline.  Keep
    # the implicated centerline diagnostics even though they are not themselves
    # failed checks: curvature location, clearance, and the repair hint are the
    # independent geometry context the planner needs to produce a material
    # waypoint correction instead of repeating an opaque OCC exception.
    centerlines = checks.get("centerlines")
    centerline_context = {}
    if isinstance(centerlines, dict):
        centerline_context = {
            module_id: _compact_centerline_repair_context(centerlines[module_id])
            for module_id in sorted(module_ids)
            if isinstance(centerlines.get(module_id), dict)
        }
    summary = {
        "module_ids": sorted(module_ids),
        "failed_checks": _bounded_diagnostic(failed_checks),
    }
    validator_policy = evidence.get("validator_policy")
    if isinstance(validator_policy, dict):
        summary["validator_policy"] = _bounded_diagnostic(validator_policy)
    if evidence.get("generator_version") is not None:
        summary["generator_version"] = evidence.get("generator_version")
    if evidence.get("schema_version") is not None:
        summary["validation_schema_version"] = evidence.get("schema_version")
    if centerline_context:
        # Already whitelisted and bounded above; a second generic truncation
        # would again drop late priority keys such as the measured location.
        summary["centerline_context"] = centerline_context
    encoded = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 8000:
        summary = {
            "module_ids": sorted(module_ids),
            "failed_check_names": sorted(failed_checks),
            "diagnostic_prefix": encoded[:7000],
            "truncated": True,
        }
        if isinstance(validator_policy, dict):
            summary["validator_policy"] = _bounded_diagnostic(validator_policy)
    return summary


def _freecad_causal_repair_module(
    evidence_summary: dict[str, Any], module_steps: dict[str, int]
) -> str | None:
    """Localize repair to the candidate/child, not the innocent parent module."""

    failed_checks = evidence_summary.get("failed_checks")
    if isinstance(failed_checks, dict):
        overlaps = failed_checks.get("non_adjacent_overlaps")
        if isinstance(overlaps, list):
            for overlap in overlaps:
                if not isinstance(overlap, dict):
                    continue
                child = overlap.get("child_module_id")
                if isinstance(child, str) and child in module_steps:
                    return child
                pair = [
                    module_id
                    for module_id in overlap.get("module_ids") or []
                    if isinstance(module_id, str) and module_id in module_steps
                ]
                if pair:
                    return max(pair, key=module_steps.__getitem__)
        for check_name in ("module_errors", "centerlines", "modules"):
            values = failed_checks.get(check_name)
            candidates: list[str] = []
            if isinstance(values, dict):
                candidates.extend(
                    module_id for module_id in values if module_id in module_steps
                )
            elif isinstance(values, list):
                candidates.extend(
                    str(item.get("module_id"))
                    for item in values
                    if isinstance(item, dict)
                    and str(item.get("module_id")) in module_steps
                )
            if candidates:
                return max(candidates, key=module_steps.__getitem__)
    candidates = [
        module_id
        for module_id in evidence_summary.get("module_ids") or []
        if isinstance(module_id, str) and module_id in module_steps
    ]
    return max(candidates, key=module_steps.__getitem__) if candidates else None


def _is_transient_occ_failure(evidence: dict[str, Any]) -> bool:
    """Classify opaque OCC construction exceptions eligible for one replay.

    Measured contract failures (curvature, overlap, connection, wall probes,
    etc.) are deterministic candidate evidence and must go directly to repair.
    Only an opaque kernel exception gets one identical-digest retry, which does
    not consume a planner attempt.
    """

    checks = evidence.get("checks") if isinstance(evidence, dict) else None
    if not isinstance(checks, dict):
        return False
    for name in (
        "non_adjacent_overlaps",
        "connection_failures",
        "terminal_bore_failures",
        "anchored_inlet_bore_failures",
        "termination_seal_failures",
        "wall_section_failures",
        "deterministic_constraint_failures",
    ):
        if checks.get(name):
            return False
    messages: list[str] = []
    for name in ("module_errors", "assembly_errors"):
        values = checks.get(name)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict) and value.get("error") is not None:
                messages.append(str(value["error"]).lower())
    if not messages:
        return False
    deterministic_markers = (
        "no suitable edges",
        "exceeds max_hub_radius",
        "left unresolved",
        "made no topology progress",
        "must be",
    )
    if any(
        marker in message for marker in deterministic_markers for message in messages
    ):
        return False
    transient_markers = (
        "brep_api",
        "standard_failure",
        "command not done",
        "bnd_box is void",
        "makepipeshell",
        "ncollection",
    )
    return all(
        any(marker in message for marker in transient_markers) for message in messages
    )


def _compact_centerline_repair_context(values: dict[str, Any]) -> dict[str, Any]:
    """곡선 교정에 필요한 measured/required 값을 우선순위대로 보존한다.

    일반 진단 축약기는 JSON 삽입 순서에 따라 뒤쪽 키를 자른다. FreeCAD가
    ``minimum_radius`` 뒤에 기록하는 ``required_radius``가 그 과정에서 빠지면
    LLM은 얼마까지 곡률을 개선해야 하는지 알 수 없다. 이 경로는 수치 교정에
    필요한 필드만 명시적으로 선택해 expected/actual 쌍을 항상 함께 전달한다.
    """

    keys = (
        "passed",
        "required_radius",
        "minimum_radius",
        "zero_curvature",
        "minimum_radius_nearest_path_point_index",
        "curvature_repair_hint",
        "required_self_clearance",
        "minimum_nonlocal_distance",
        "self_clearance_passed",
        "endpoint_tangency_passed",
        "endpoint_tangent_dots",
        "minimum_join_tangent_dot",
        "curve_length",
        "optimized_handle_factors",
    )
    result = {key: _bounded_diagnostic(values[key]) for key in keys if key in values}
    location = values.get("minimum_radius_location")
    if isinstance(location, dict):
        result["minimum_radius_location"] = {
            key: _bounded_diagnostic(location[key])
            for key in ("sample_index", "edge_parameter", "position")
            if key in location
        }
    return result


def _freecad_repair_contract(
    evidence_summary: dict[str, Any],
    *,
    module_path_kinds: dict[str, str] | None = None,
    module_params: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """FreeCAD 증거에서 LLM이 바로 사용할 expected/actual/추천을 만든다."""

    expected: dict[str, Any] = {"freecad_checks": "all implicated checks pass"}
    actual: dict[str, Any] = {}
    recommendations: list[dict[str, Any]] = []
    centerlines = evidence_summary.get("centerline_context")
    if isinstance(centerlines, dict):
        required_by_module: dict[str, float] = {}
        measured_by_module: dict[str, float] = {}
        for module_id, values in centerlines.items():
            if not isinstance(values, dict):
                continue
            try:
                required = float(values["required_radius"])
                measured = float(values["minimum_radius"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(required) and math.isfinite(measured)):
                continue
            required_by_module[str(module_id)] = required
            measured_by_module[str(module_id)] = measured
            if measured + 1e-9 < required:
                path_kind = (module_path_kinds or {}).get(str(module_id))
                nearest_path_index = values.get(
                    "minimum_radius_nearest_path_point_index"
                )
                if path_kind == "spline":
                    parameters = ["waypoints"]
                elif path_kind == "circular_arc":
                    parameters = ["bend_radius", "sweep_angle", "plane_normal"]
                else:
                    parameters = ["waypoints", "bend_radius"]
                recommendations.append(
                    {
                        "module_id": str(module_id),
                        "path_kind": path_kind,
                        "parameters": parameters,
                        "current_minimum_radius": measured,
                        "required_minimum_radius": required,
                        "nearest_path_point_index": nearest_path_index,
                        "nearest_waypoint_index": (
                            int(nearest_path_index) - 1
                            if isinstance(nearest_path_index, int)
                            and nearest_path_index > 0
                            else None
                        ),
                        "instruction": values.get("curvature_repair_hint")
                        or (
                            "Replace or move the implicated candidate parameter until "
                            "the measured radius meets the requirement; do not change "
                            "an immutable goal length."
                        ),
                    }
                )
        if required_by_module:
            expected["minimum_curvature_radius_by_module"] = required_by_module
            actual["measured_minimum_curvature_radius_by_module"] = measured_by_module

    failed_checks = evidence_summary.get("failed_checks")
    if isinstance(failed_checks, dict):
        overlaps = failed_checks.get("non_adjacent_overlaps")
        if isinstance(overlaps, list):
            overlap_residuals: list[dict[str, Any]] = []
            for item in overlaps:
                if not isinstance(item, dict):
                    continue
                try:
                    common_volume = float(item["common_volume"])
                    allowed_volume = float(item["allowed_volume"])
                except (KeyError, TypeError, ValueError):
                    common_volume = allowed_volume = 0.0
                module_ids = [
                    str(value)
                    for value in item.get("module_ids") or []
                    if isinstance(value, str)
                ]
                junction_id = next(
                    (
                        module_id
                        for module_id in module_ids
                        if (module_path_kinds or {}).get(module_id) == "junction"
                    ),
                    None,
                )
                residual = max(0.0, common_volume - allowed_volume)
                overlap_residuals.append(
                    {
                        "module_ids": module_ids,
                        "adjacent": item.get("adjacent") is True,
                        "common_volume": common_volume,
                        "allowed_volume": allowed_volume,
                        "excess_volume": residual,
                    }
                )
                if item.get("adjacent") is True:
                    recommendations.append(
                        {
                            "module_id": junction_id,
                            "classification": "adjacent_interface_overlap",
                            "parameters": [
                                "outlets[*].axis",
                                "blend_mode",
                                "blend_radius",
                                "inner_blend_radius",
                            ],
                            "non_causal_parameters": [
                                "outlets[*].length",
                                "max_hub_radius",
                            ],
                            "excess_volume": residual,
                            "instruction": (
                                "Preserve the junction primitive, target and goal "
                                "claims. Outlet length and a hard junction's "
                                "max_hub_radius do not change its local mating "
                                "intersection. If evidence says the overlap extends "
                                "outside the permitted local interface, change the "
                                "offending outlet axis or supported blend geometry; "
                                "otherwise classify repeated invariant evidence as "
                                "a validator/kernel conflict rather than random-walk "
                                "unrelated values."
                            ),
                        }
                    )
                else:
                    recommendations.append(
                        {
                            "module_id": junction_id
                            or (module_ids[-1] if module_ids else None),
                            "classification": "nonlocal_module_overlap",
                            "parameters": [
                                "waypoints",
                                "direction",
                                "bend_radius",
                                "outlets[*].axis",
                            ],
                            "excess_volume": residual,
                            "instruction": (
                                "Move the implicated path or outlet away from the "
                                "non-adjacent module while preserving immutable "
                                "terminal contracts."
                            ),
                        }
                    )
            if overlap_residuals:
                expected["unexpected_overlap_excess_volume"] = 0.0
                actual["unexpected_overlap_residuals"] = overlap_residuals
        module_errors = failed_checks.get("module_errors")
        if isinstance(module_errors, list):
            hub_pattern = re.compile(
                r"junction blend exceeds max_hub_radius:\s*required\s*"
                rf"({_MM_NUMBER_TEXT}),\s*maximum\s*({_MM_NUMBER_TEXT})",
                re.IGNORECASE,
            )
            for item in module_errors:
                if not isinstance(item, dict):
                    continue
                message = str(item.get("error") or "")
                match = hub_pattern.search(message)
                if match is None:
                    lower_message = message.lower()
                    if "junction fillet" in lower_message or (
                        "chamfer or fillet" in lower_message
                    ):
                        recommendations.append(
                            {
                                "module_id": item.get("module_id"),
                                "classification": "junction_fillet_construction",
                                "parameters": [
                                    "blend_mode",
                                    "blend_radius",
                                    "inner_blend_radius",
                                ],
                                "current_params": (module_params or {}).get(
                                    str(item.get("module_id")), {}
                                ),
                                "instruction": (
                                    "Keep the junction/target/goals fixed. Select a "
                                    "different authored fillet radius only when the "
                                    "Intent permits it; if a smooth hub is not "
                                    "immutable, hard blend_mode is an admissible "
                                    "topology change. Repeating outlet length or "
                                    "max_hub_radius cannot repair an unsuitable seam."
                                ),
                            }
                        )
                    elif any(
                        marker in lower_message
                        for marker in (
                            "brep_api",
                            "standard_failure",
                            "command not done",
                            "bnd_box is void",
                            "makepipeshell",
                        )
                    ):
                        recommendations.append(
                            {
                                "module_id": item.get("module_id"),
                                "classification": "occ_kernel_construction",
                                "parameters": [],
                                "instruction": (
                                    "First retry the identical digest in a fresh "
                                    "candidate document. If the OCC exception is "
                                    "persistent, classify it as generator/kernel "
                                    "failure before asking the planner to perturb "
                                    "unrelated geometry."
                                ),
                            }
                        )
                    continue
                required = _metric_number(match.group(1))
                maximum = _metric_number(match.group(2))
                recommendations.append(
                    {
                        "module_id": item.get("module_id"),
                        "parameters": [
                            "max_hub_radius",
                            "blend_radius",
                            "inner_blend_radius",
                            "outlets",
                        ],
                        "required_minimum_hub_radius": required,
                        "current_maximum_hub_radius": maximum,
                        "instruction": (
                            "Choose a junction geometry whose exact required hub radius "
                            "does not exceed max_hub_radius while preserving authored limits."
                        ),
                    }
                )
    return expected, actual, recommendations


def _bounded_diagnostic(value: Any, depth: int = 0) -> Any:
    if depth >= 6:
        return "<depth-truncated>"
    if isinstance(value, dict):
        return {
            str(key): _bounded_diagnostic(child, depth + 1)
            for key, child in list(value.items())[:12]
        }
    if isinstance(value, list):
        return [_bounded_diagnostic(child, depth + 1) for child in value[:8]]
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:500]


def _freecad_assembly_bounds(evidence: dict[str, Any]) -> AssemblyBounds | None:
    bounds = ((evidence.get("checks") or {}).get("assembly") or {}).get("bounds")
    if not isinstance(bounds, dict):
        return None
    minimum = bounds.get("minimum")
    maximum = bounds.get("maximum")
    if (
        not isinstance(minimum, list)
        or not isinstance(maximum, list)
        or len(minimum) != 3
        or len(maximum) != 3
    ):
        return None
    try:
        numeric_minimum = tuple(float(value) for value in minimum)
        numeric_maximum = tuple(float(value) for value in maximum)
    except (TypeError, ValueError):
        return None
    if not all(
        math.isfinite(value) for value in [*numeric_minimum, *numeric_maximum]
    ) or any(low > high for low, high in zip(numeric_minimum, numeric_maximum)):
        return None
    return AssemblyBounds(minimum=numeric_minimum, maximum=numeric_maximum)


def _freecad_document_path(
    raw_result_path: Path,
    state: PipeState,
    *,
    payload_digest: str | None = None,
) -> Path:
    """MCP 결과 위치와 상태 identity로 게시된 FCStd 경로를 계산한다."""

    run_dir = _freecad_run_dir(raw_result_path)
    digest = payload_digest or geometry_payload_digest(state)
    return (
        (run_dir / f"pipe_v{state.state_version}_{digest[:12]}.FCStd")
        .expanduser()
        .resolve()
    )


def _freecad_artifact_manifest_path(raw_result_path: Path) -> Path:
    """MCP 결과 파일에서 실행 단위 FreeCAD manifest 경로를 찾는다."""

    return _freecad_run_dir(raw_result_path) / "freecad_artifact.json"


def _freecad_run_dir(raw_result_path: Path) -> Path:
    """일반ㆍ단계ㆍ복구 MCP 결과가 속한 최상위 실행 디렉터리를 반환한다."""

    parent = raw_result_path.parent
    return (
        parent.parent
        if parent.name in {"step_mcp", "recovery_mcp", "rollback_mcp"}
        else parent
    )


def _is_semantic_freecad_validation_error(message: str) -> bool:
    normalized = message.lower()
    return any(
        marker in normalized
        for marker in (
            "b-rep validation failed",
            "assembly check failed",
            "outer_network check failed",
            "bore_network check failed",
            "module checks failed",
            "centerline checks failed",
            "module_errors evidence is not empty",
            "assembly_errors evidence is not empty",
            "non_adjacent_overlaps evidence is not empty",
            "connection_failures evidence is not empty",
            "terminal_bore_failures evidence is not empty",
            "anchored_inlet_bore_failures evidence is not empty",
            "termination_seal_failures evidence is not empty",
            "wall_section_failures evidence is not empty",
            "minimum wall thickness is not positive",
            "declared downstream open-port count mismatch",
            "anchored-inlet count mismatch",
        )
    )


def _visual_review(
    gemini: GeminiClient,
    state: PipeState,
    paths: list[str],
    *,
    intent: IntentResult,
) -> VisualCriticResult:
    digest = geometry_payload_digest(state)
    evidence_hashes: list[str] = []
    image_inputs: list[dict[str, Any]] = []
    evidence_views: list[dict[str, str]] = []
    for view_index, path_string in enumerate(paths[:6], start=1):
        data = Path(path_string).read_bytes()
        evidence_hash = hashlib.sha256(data).hexdigest()
        evidence_hashes.append(evidence_hash)
        view_name = Path(path_string).stem
        evidence_views.append({"view": view_name, "sha256": evidence_hash})
        image_inputs.append(
            {
                "type": "text",
                "text": f"Evidence view {view_index}: {view_name} camera.",
            }
        )
        image_inputs.append(
            {
                "type": "image",
                "data": base64.b64encode(data).decode("ascii"),
                "mime_type": "image/png"
                if path_string.endswith(".png")
                else "image/jpeg",
            }
        )
    visual_contract = {
        "global_spec": intent.global_spec.model_dump(mode="json", exclude_none=True),
        "downstream_open_port_contract": {
            "expected_count": intent.expected_open_ports,
            "source": intent.expected_open_ports_source,
            "scope": "free_downstream_only_excludes_anchored_START",
        },
        "realized_terminal_topology": realized_terminal_topology(state),
        "target_behavior": [
            goal.model_dump(
                mode="json",
                exclude_none=True,
                exclude_defaults=True,
            )
            for goal in intent.target_behavior
        ],
        "geometric_constraints": [
            item.model_dump(mode="json", exclude_none=True)
            for item in intent.geometric_constraints
        ],
        "design_notes": intent.design_notes,
    }
    inputs: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Inspect this pipe CAD both for construction defects and for visible "
                "fidelity to the supplied immutable geometry contract. Check the "
                "ordered route/turn/transition structure, requested spatial behavior "
                "such as rising or coiling paths and curve reversals, diameter changes, "
                "outlet orientation, open ends, and the qualitative design notes when "
                "they are visually judgeable. Also check for disconnected parts, "
                "collisions, blocked bores, unnatural junction bulges, visible seams, "
                "faceting, pinching, self-intersection, or implausible curvature. Do not "
                "invent a defect for a numerical fact that the images cannot establish. "
                "The digest-bound FreeCAD gate already proved one connected valid pipe "
                "assembly, one continuous bore network, zero non-adjacent overlaps, "
                "zero blocked terminal/inlet bores, and zero sampled wall failures. "
                "Treat those topology and internal-passage facts as authoritative; use "
                "the images to judge visible route fidelity, smoothness, taper, outlet "
                "direction, and whether an apparent contradiction is genuinely clear. "
                "The downstream open-port contract excludes the anchored START inlet. "
                "When realized_terminal_topology says START is physically open, START "
                "and its graph-connected module inlet are one open physical end even "
                "though that interface is consumed/mated in the construction graph. "
                "Consumed or internal_mated_interface never means capped: only an "
                "explicit sealed_terminal_modules entry authorizes a closed end. Do not "
                "double-count the two local port records at one hollow mated interface. "
                "Localize every defect using the supplied step/module map and return "
                "the matching module_ids and target_step. Echo the ordered evidence_sha256 "
                "list exactly; it binds the following images. "
                f"Return state_id={state.state_id} and payload_digest={digest}. "
                "Ordered evidence_sha256: "
                + json.dumps(evidence_hashes, separators=(",", ":"))
                + ". Labeled evidence views: "
                + json.dumps(evidence_views, separators=(",", ":"))
                + ". Immutable geometry contract: "
                + json.dumps(
                    visual_contract,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + ". Step/module map: "
                + json.dumps(
                    compact_visual_module_map(state),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
        },
        *image_inputs,
    ]
    result = _call_structured(
        gemini,
        inputs,
        VisualCriticResultWire,
        part="visual_validator",
        thinking_level="medium",
    )
    if isinstance(result, VisualCriticResultWire):
        result = VisualCriticResult.model_validate(result.model_dump(mode="python"))
    if not isinstance(result, VisualCriticResult):
        raise TypeError("Visual validator returned the wrong schema")
    if result.state_id != state.state_id or result.payload_digest != digest:
        raise ValueError("Visual critic state/digest mismatch")
    if result.evidence_sha256 != evidence_hashes:
        raise ValueError("Visual critic evidence hash mismatch")
    return result


def _freecad_attempt_generator_version(attempt: ActionAttempt) -> str | None:
    """Recover the generator binding from a rejected attempt when available."""

    for observation in reversed(attempt.observations):
        actual = observation.get("actual")
        if not isinstance(actual, dict):
            continue
        candidates = [actual, actual.get("evidence")]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            version = candidate.get("generator_version")
            if isinstance(version, str) and version:
                return version
            policy = candidate.get("validator_policy")
            if isinstance(policy, dict):
                version = policy.get("generator_version")
                if isinstance(version, str) and version:
                    return version
    return None


def _load_resume_context(
    checkpoint_path: Path,
    settings: Settings,
    engine: StateEngine,
    *,
    dry_run: bool,
    run_dir: Path,
    expected_run_id: str,
) -> _ResumeContext:
    if not checkpoint_path.is_file():
        raise ValueError(f"Resume checkpoint is missing: {checkpoint_path}")
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Resume checkpoint is unreadable: {checkpoint_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Resume checkpoint must be a JSON object")
    phase = payload.get("phase")
    if phase not in {"COMMITTED", "PREPARED", "PUBLISHED"}:
        raise ValueError(f"Unsupported checkpoint phase: {phase!r}")
    if payload.get("run_id") != expected_run_id:
        raise ValueError("Checkpoint run_id does not match the run directory")

    intent = IntentResult.model_validate(payload.get("intent"))
    if not dry_run:
        nonbinary_branches = _nonbinary_branch_goal_ids(intent)
        if nonbinary_branches:
            raise ValueError(
                "Checkpoint contains non-binary branch goals and cannot be "
                "deterministically migrated: " + ", ".join(nonbinary_branches)
            )
    unsupported_components = _unsupported_required_components(intent)
    if unsupported_components:
        raise ValueError(
            "Checkpoint requires unsupported inline components: "
            + ", ".join(unsupported_components)
        )
    if _component_contract_error(intent) is not None:
        raise ValueError("Checkpoint component multiplicity contract is inconsistent")
    if intent.hard_constraints:
        raise ValueError("Checkpoint contains unsupported hard constraints")
    prompt_text = (run_dir / "prompt.txt").read_text(encoding="utf-8")
    rebound_intent = _bind_contract(prompt_text, intent)
    if (
        rebound_intent.prompt_sha256 != intent.prompt_sha256
        or rebound_intent.contract_digest != intent.contract_digest
    ):
        raise ValueError(
            "Checkpoint intent no longer matches its prompt/contract digest"
        )
    actions = [dict(item) for item in payload.get("actions", [])]
    attempts = [
        ActionAttempt.model_validate(item) for item in payload.get("attempts", [])
    ]
    steps = [
        StepVerification.model_validate(item)
        for item in payload.get("step_verifications", [])
    ]
    raw_lineage = payload.get("planner_lineage") or {}
    if not isinstance(raw_lineage, dict):
        raise ValueError("Checkpoint planner_lineage must be an object")
    lineage = {str(key): value for key, value in raw_lineage.items() if value}
    raw_profiles = payload.get("planner_schema_profiles") or {}
    if not isinstance(raw_profiles, dict) or any(
        not isinstance(state_id, str) or profile not in _PLANNER_SCHEMA_PROFILES
        for state_id, profile in raw_profiles.items()
    ):
        raise ValueError("Checkpoint planner_schema_profiles is invalid")
    planner_schema_profiles = {
        str(state_id): str(profile) for state_id, profile in raw_profiles.items()
    }
    llm_usage = LLMUsage.model_validate(payload.get("llm_usage") or {})
    diagnostic_journal = DiagnosticJournal.model_validate(
        payload.get("diagnostic_journal") or {}
    )
    raw_pending_observations = payload.get("pending_repair_observations") or []
    if not isinstance(raw_pending_observations, list) or any(
        not isinstance(item, dict) for item in raw_pending_observations
    ):
        raise ValueError(
            "Checkpoint pending_repair_observations must be a list of objects"
        )
    pending_repair_observations = [dict(item) for item in raw_pending_observations]
    preserved_suffix = _PreservedSuffix.from_payload(payload.get("preserved_suffix"))
    if preserved_suffix is not None and any(
        checkpoint.contract_digest != intent.contract_digest
        for checkpoint in preserved_suffix.original_checkpoints
    ):
        raise ValueError("preserved_suffix contract digest mismatch")

    if phase == "COMMITTED":
        state = PipeState.model_validate(payload.get("state"))
        if abs(state.modeling_tolerance - settings.modeling_tolerance) > 1e-15:
            raise ValueError(
                "Checkpoint modeling tolerance differs from the current generator setting"
            )
        _validate_checkpoint_state(
            state,
            intent,
            payload.get("state_digest"),
            actions,
        )
        checkpoints = _checkpoint_history(payload, state=state)
        _validate_checkpoint_history(
            intent,
            actions,
            steps,
            checkpoints,
        )
        pending_draft_payload = payload.get("pending_draft")
        pending_draft = (
            ActionDraft.model_validate(pending_draft_payload)
            if pending_draft_payload is not None
            else None
        )
        if pending_draft is not None:
            pending_draft = pending_draft.model_copy(
                deep=True,
                update={
                    "params": filter_draft_params(
                        pending_draft.module,
                        copy.deepcopy(pending_draft.params),
                    )
                },
            )
        pending_draft_attempt_index = payload.get("pending_draft_attempt_index")
        pending_draft_state_digest = payload.get("pending_draft_state_digest")
        current_step_index = len(actions) + 1
        prior_current_attempts = [
            item.attempt_index
            for item in attempts
            if item.step_index == current_step_index and item.status == "rejected"
        ]
        default_next_attempt = (
            max(prior_current_attempts) + 1 if prior_current_attempts else 1
        )
        next_attempt_index = payload.get(
            "next_attempt_index",
            default_next_attempt,
        )
        if type(next_attempt_index) is not int or not (
            1 <= next_attempt_index <= settings.step_repair_attempts + 2
        ):
            raise ValueError(
                "Checkpoint next_attempt_index is outside the retry journal"
            )
        if (
            pending_draft is None
            and next_attempt_index <= settings.step_repair_attempts + 1
        ):
            latest_rejected = next(
                (
                    item
                    for item in reversed(attempts)
                    if item.step_index == current_step_index
                    and item.status == "rejected"
                ),
                None,
            )
            if (
                latest_rejected is not None
                and latest_rejected.phase == "freecad_semantic_validation"
                and latest_rejected.draft is not None
                and _freecad_attempt_generator_version(latest_rejected)
                != GENERATOR_VERSION
            ):
                legacy_draft = ActionDraft.model_validate(latest_rejected.draft)
                pending_draft = legacy_draft.model_copy(
                    deep=True,
                    update={
                        "params": filter_draft_params(
                            legacy_draft.module,
                            copy.deepcopy(legacy_draft.params),
                        )
                    },
                )
                pending_draft_attempt_index = next_attempt_index
                pending_draft_state_digest = _pipe_state_digest(state)
                pending_repair_observations = [
                    *pending_repair_observations,
                    {
                        "context_type": "generator_migration_replay",
                        "from_generator_version": (
                            _freecad_attempt_generator_version(latest_rejected)
                            or "unversioned_legacy_evidence"
                        ),
                        "to_generator_version": GENERATOR_VERSION,
                        "instruction": (
                            "Revalidate the exact rejected draft with the current "
                            "generator and validator policy before another paid "
                            "planner or diagnostician call."
                        ),
                    },
                ]
        if pending_draft is None:
            if (
                pending_draft_attempt_index is not None
                or pending_draft_state_digest is not None
            ):
                raise ValueError("Checkpoint has orphaned pending-draft metadata")
        else:
            if not state.remaining_goals:
                raise ValueError("Checkpoint has a pending draft for a terminal state")
            if (
                type(pending_draft_attempt_index) is not int
                or pending_draft_attempt_index != next_attempt_index
                or pending_draft_state_digest != _pipe_state_digest(state)
                or pending_draft_attempt_index > settings.step_repair_attempts + 1
            ):
                raise ValueError(
                    "Checkpoint pending draft does not bind to state/retry"
                )
        semantic_mcp_passed = False
        mcp_used = False
        mcp_error: str | None = None
        mcp_result_path: str | None = None
        freecad_validation_path: str | None = None
        freecad_document_path: str | None = None
        if (
            bool(payload.get("freecad_verified", False))
            and state.placed_modules
            and _should_run_step_mcp(settings, dry_run)
        ):
            try:
                recovery_raw_path = run_dir / "recovery_mcp" / "committed.json"
                recovery_validation_path = (
                    run_dir / "recovery_mcp" / "committed_validation.json"
                )
                _raw, recovery_evidence, _publish_raw = _validate_and_publish_freecad(
                    settings,
                    state,
                    run_id=expected_run_id,
                    attempt_id=1,
                    raw_result_path=recovery_raw_path,
                    validation_path=recovery_validation_path,
                )
                del _raw, _publish_raw
                recovery_measurements = _freecad_measurements(recovery_evidence)
                merged_measurements = {
                    module_id: dict(values)
                    for module_id, values in state.module_measurements.items()
                }
                merged_measurements.update(recovery_measurements)
                state = state.model_copy(
                    update={"module_measurements": merged_measurements}
                )
                checkpoints[-1] = state.model_copy(deep=True)
                if steps and steps[-1].transition.state_after_id == state.state_id:
                    steps[-1] = steps[-1].model_copy(
                        update={
                            "mcp_status": "passed",
                            "mcp_result_path": str(recovery_raw_path),
                            "freecad_validation_path": str(recovery_validation_path),
                            "mcp_error": None,
                            "skipped_mcp_reason": None,
                            "mcp_measurements": recovery_measurements,
                            "mcp_assembly_bounds": _freecad_assembly_bounds(
                                recovery_evidence
                            ),
                        }
                    )
                semantic_mcp_passed = True
                mcp_used = True
                mcp_result_path = str(recovery_raw_path)
                freecad_validation_path = str(recovery_validation_path)
                freecad_document_path = str(
                    _freecad_document_path(recovery_raw_path, state)
                )
            except FreeCADMCPError as exc:
                mcp_used = True
                mcp_error = str(exc)
                if settings.freecad_mcp_required:
                    raise FreeCADMCPError(
                        f"Required committed-state reconciliation failed: {exc}"
                    ) from exc
        return _ResumeContext(
            intent=intent,
            state=state,
            actions=actions,
            attempts=attempts,
            step_verifications=steps,
            checkpoints=checkpoints,
            planner_lineage=lineage,
            planner_schema_profiles=planner_schema_profiles,
            llm_usage=llm_usage,
            pending_repair_observations=pending_repair_observations,
            diagnostic_journal=diagnostic_journal,
            pending_draft=pending_draft,
            pending_draft_attempt_index=pending_draft_attempt_index,
            preserved_suffix=preserved_suffix,
            next_attempt_index=next_attempt_index,
            semantic_mcp_passed=semantic_mcp_passed,
            mcp_used=mcp_used,
            mcp_error=mcp_error,
            mcp_result_path=mcp_result_path,
            freecad_validation_path=freecad_validation_path,
            freecad_document_path=freecad_document_path,
        )

    previous = PipeState.model_validate(payload.get("previous_state"))
    candidate = PipeState.model_validate(payload.get("candidate_state"))
    if (
        abs(previous.modeling_tolerance - settings.modeling_tolerance) > 1e-15
        or abs(candidate.modeling_tolerance - settings.modeling_tolerance) > 1e-15
    ):
        raise ValueError(
            "Prepared checkpoint modeling tolerance differs from the current generator setting"
        )
    _validate_checkpoint_state(previous, intent, None, actions)
    candidate_digest = geometry_payload_digest(candidate)
    stored_candidate_digest = payload.get("candidate_digest")
    generator_migration = stored_candidate_digest != candidate_digest
    if generator_migration and not (
        isinstance(stored_candidate_digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", stored_candidate_digest)
        and payload.get("candidate_state_digest") == _pipe_state_digest(candidate)
    ):
        raise ValueError("Prepared candidate digest does not match candidate_state")
    action = ResolvedAction.model_validate(payload.get("action"))
    _validate_checkpoint_state(
        candidate,
        intent,
        payload.get("candidate_state_digest"),
        [*actions, action.model_dump(mode="json")],
    )
    raw_attempt_id = payload.get("attempt_id", 1)
    if type(raw_attempt_id) is not int or not (
        1 <= raw_attempt_id <= settings.step_repair_attempts + 1
    ):
        raise ValueError("Prepared checkpoint attempt_id is outside the retry budget")
    attempt_id = raw_attempt_id
    if not generator_migration and payload.get(
        "candidate_document"
    ) != candidate_document_name(
        candidate,
        run_id=expected_run_id,
        attempt_id=attempt_id,
    ):
        raise ValueError("Prepared candidate document name mismatch")
    if not generator_migration and payload.get(
        "published_document"
    ) != published_document_name(
        candidate,
        run_id=expected_run_id,
    ):
        raise ValueError("Prepared published document name mismatch")
    draft_payload = payload.get("draft")
    draft = ActionDraft.model_validate(draft_payload) if draft_payload else None
    step_payload = payload.get("step_verification")
    step = (
        StepVerification.model_validate(step_payload)
        if step_payload
        else build_step_verification(
            previous,
            action,
            candidate,
            intent,
            len(actions) + 1,
            mcp_required=(
                settings.freecad_mcp_required and settings.freecad_step_mcp_enabled
            ),
        )
    )
    if has_errors(step.issues):
        raise ValueError("Prepared candidate no longer passes static validation")
    checkpoints = _checkpoint_history(payload, state=previous)
    _validate_checkpoint_history(intent, actions, steps, checkpoints)
    mcp_used = False
    mcp_error: str | None = None
    # A valid old digest with an unchanged candidate state means generator or
    # validator policy code changed. Never trust the old document/evidence;
    # execute the current generator and all gates before deciding roll-forward.
    roll_forward = phase == "PUBLISHED" and not generator_migration
    roll_forward_verified = False
    semantic_rejection = False
    semantic_rejection_evidence: dict[str, Any] = {}
    semantic_rejection_evidence_digest: str | None = None
    recovery_measurements: dict[str, dict[str, float]] = {}
    recovery_bounds: dict[str, Any] | None = None
    recovery_holder: dict[str, Any] = {}
    recovery_raw_path: Path | None = None
    recovery_validation_path: Path | None = None

    def validate_recovery_evidence(recovery_evidence: dict[str, Any]) -> None:
        measurements = _freecad_measurements(recovery_evidence)
        merged = {
            module_id: dict(values)
            for module_id, values in candidate.module_measurements.items()
        }
        merged.update(measurements)
        measured_candidate = candidate.model_copy(
            update={"module_measurements": merged}
        )
        measured_step = step.model_copy(
            update={
                "mcp_status": "passed",
                "mcp_measurements": measurements,
                "mcp_assembly_bounds": _freecad_assembly_bounds(recovery_evidence),
            }
        )
        evidence_issues = validate_step_mcp_evidence(
            intent,
            measured_candidate,
            action,
            measured_step.transition,
            [*steps, measured_step],
        )
        if has_errors(evidence_issues):
            augmented = dict(recovery_evidence)
            augmented_checks = dict(recovery_evidence.get("checks") or {})
            augmented_checks["deterministic_constraint_failures"] = [
                issue.model_dump(mode="json")
                for issue in evidence_issues
                if issue.severity == "error"
            ][:8]
            augmented["checks"] = augmented_checks
            raise _FreeCADSemanticError(
                "Recovered FreeCAD measurements violate the step contract",
                augmented,
            )
        recovery_holder["state"] = measured_candidate
        recovery_holder["step"] = measured_step
        recovery_holder["measurements"] = measurements
        recovery_holder["bounds"] = _freecad_assembly_bounds(recovery_evidence)

    if phase == "PUBLISHED" and not generator_migration:
        evidence = payload.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("Published checkpoint is missing validation evidence")
        try:
            geometry_modules = [
                module
                for module in candidate.placed_modules
                if not (
                    module.type == "connect_ports"
                    and module.params.get("path_kind") == "seam"
                )
            ]
            assess_freecad_validation(
                {
                    "content": [
                        {
                            "type": "text",
                            "text": "CADGEN_VALIDATION="
                            + json.dumps(evidence, separators=(",", ":")),
                        }
                    ]
                },
                expected_digest=candidate_digest,
                expected_state_id=candidate.state_id,
                expected_module_ids=[module.id for module in geometry_modules],
                expected_internal_section_module_count=sum(
                    module.type not in {"terminate", "cap_pipe"}
                    for module in geometry_modules
                ),
                expected_open_port_count=len(candidate.open_ports),
                expected_anchored_inlet_count=(anchored_inlet_count(candidate)),
                expected_generator_version=GENERATOR_VERSION,
                expected_run_id=expected_run_id,
                expected_state_version=candidate.state_version,
                expected_attempt_id=attempt_id,
                expected_candidate_document=candidate_document_name(
                    candidate,
                    run_id=expected_run_id,
                    attempt_id=attempt_id,
                ),
            )
        except FreeCADMCPError as exc:
            raise ValueError(
                "Published checkpoint evidence does not bind to the candidate"
            ) from exc
        validate_recovery_evidence(evidence)
        recovery_measurements = recovery_holder["measurements"]
        recovery_bounds = recovery_holder["bounds"]
        candidate = recovery_holder["state"]
        step = recovery_holder["step"]
        if not dry_run and settings.freecad_mcp_enabled:
            try:
                recovery_raw_path = (
                    run_dir
                    / "recovery_mcp"
                    / f"published_{candidate.state_version}_{attempt_id}.json"
                )
                recovery_validation_path = (
                    run_dir
                    / "recovery_mcp"
                    / f"published_{candidate.state_version}_{attempt_id}_validation.json"
                )
                _validate_and_publish_freecad(
                    settings,
                    candidate,
                    run_id=expected_run_id,
                    attempt_id=attempt_id,
                    raw_result_path=recovery_raw_path,
                    validation_path=recovery_validation_path,
                    evidence_validator=validate_recovery_evidence,
                )
                roll_forward_verified = True
                mcp_used = True
            except FreeCADMCPError as exc:
                raise FreeCADMCPError(
                    f"Published checkpoint live reconciliation failed: {exc}"
                ) from exc
    elif not dry_run and settings.freecad_mcp_enabled:
        try:
            recovery_raw_path = (
                run_dir
                / "recovery_mcp"
                / f"step_{candidate.state_version}_attempt_{attempt_id}.json"
            )
            recovery_validation_path = (
                run_dir
                / "recovery_mcp"
                / f"step_{candidate.state_version}_validation_{attempt_id}.json"
            )
            _, evidence, _ = _validate_and_publish_freecad(
                settings,
                candidate,
                run_id=expected_run_id,
                attempt_id=attempt_id,
                raw_result_path=recovery_raw_path,
                validation_path=recovery_validation_path,
                evidence_validator=validate_recovery_evidence,
            )
            recovery_measurements = recovery_holder["measurements"]
            recovery_bounds = recovery_holder["bounds"]
            candidate = recovery_holder["state"]
            step = recovery_holder["step"]
            roll_forward = True
            roll_forward_verified = True
            mcp_used = True
        except _FreeCADSemanticError as exc:
            mcp_used = True
            mcp_error = str(exc)
            semantic_rejection = True
            semantic_rejection_evidence_digest = _canonical_json_digest(exc.evidence)
            semantic_rejection_evidence = _compact_freecad_failure_evidence(
                exc.evidence
            )
        except FreeCADMCPError as exc:
            raise FreeCADMCPError(
                "Prepared checkpoint has an uncertain FreeCAD outcome and cannot "
                f"be rolled back safely: {exc}"
            ) from exc
    else:
        raise FreeCADMCPError(
            "PREPARED checkpoint requires live FreeCAD MCP reconciliation; "
            "the publish outcome is unknown"
        )

    # PREPARED/PUBLISHED lineage describes the action that was in flight at the
    # crash boundary. After either accepting or rejecting that action, the next
    # planner turn must start from the recovered canonical state and catalog.
    lineage = dict(lineage)
    lineage.pop("step_planner", None)

    if roll_forward:
        merged_measurements = {
            module_id: dict(values)
            for module_id, values in candidate.module_measurements.items()
        }
        merged_measurements.update(recovery_measurements)
        candidate = candidate.model_copy(
            update={"module_measurements": merged_measurements}
        )
        step = step.model_copy(
            update={
                "mcp_status": "passed" if roll_forward_verified else "skipped",
                "mcp_error": None,
                "skipped_mcp_reason": (
                    None
                    if roll_forward_verified
                    else "Durable publish evidence was accepted, but no live FreeCAD session was reconciled."
                ),
                "mcp_measurements": recovery_measurements,
                "mcp_assembly_bounds": recovery_bounds,
            }
        )
        actions.append(action.model_dump(mode="json"))
        steps.append(step)
        attempts.append(
            _attempt(
                len(actions),
                attempt_id,
                previous,
                "recovery_commit",
                "accepted",
                draft,
                action,
                step.issues,
            )
        )
        checkpoints.append(candidate.model_copy(deep=True))
        return _ResumeContext(
            intent=intent,
            state=candidate,
            actions=actions,
            attempts=attempts,
            step_verifications=steps,
            checkpoints=checkpoints,
            planner_lineage=lineage,
            planner_schema_profiles=planner_schema_profiles,
            llm_usage=llm_usage,
            pending_repair_observations=[],
            diagnostic_journal=diagnostic_journal,
            preserved_suffix=preserved_suffix,
            next_attempt_index=1,
            semantic_mcp_passed=roll_forward_verified,
            mcp_used=mcp_used,
            mcp_error=mcp_error,
            mcp_result_path=(
                str(recovery_raw_path)
                if roll_forward_verified and recovery_raw_path is not None
                else None
            ),
            freecad_validation_path=(
                str(recovery_validation_path)
                if roll_forward_verified and recovery_validation_path is not None
                else None
            ),
            freecad_document_path=(
                str(_freecad_document_path(recovery_raw_path, candidate))
                if roll_forward_verified and recovery_raw_path is not None
                else None
            ),
        )

    if not semantic_rejection:
        raise FreeCADMCPError(
            "Prepared checkpoint recovery reached an indeterminate rollback path"
        )

    rollback_issue = _issue(
        len(actions) + 1,
        "RECOVERY_ROLLED_BACK",
        "An unverified prepared candidate was discarded during resume.",
        phase="checkpoint_recovery",
        action_id=action.action_id,
        module_id=(
            semantic_rejection_evidence.get("module_ids", [None])[0]
            if semantic_rejection_evidence.get("module_ids")
            else None
        ),
        actual={
            "reason": mcp_error or _step_mcp_skip_reason(settings, dry_run),
            "evidence": semantic_rejection_evidence,
            "evidence_artifact_path": (
                str(recovery_validation_path)
                if recovery_validation_path is not None
                else None
            ),
            "evidence_digest": semantic_rejection_evidence_digest,
            "original_failure_phase": "freecad_semantic_validation",
            "generator_version": GENERATOR_VERSION,
        },
    )
    attempts.append(
        _attempt(
            len(actions) + 1,
            attempt_id,
            previous,
            "checkpoint_recovery",
            "rejected",
            draft,
            action,
            [rollback_issue],
        )
    )
    return _ResumeContext(
        intent=intent,
        state=previous,
        actions=actions,
        attempts=attempts,
        step_verifications=steps,
        checkpoints=checkpoints,
        planner_lineage=lineage,
        planner_schema_profiles=planner_schema_profiles,
        llm_usage=llm_usage,
        pending_repair_observations=[
            *pending_repair_observations,
            _repair_observation(rollback_issue),
        ],
        diagnostic_journal=diagnostic_journal,
        preserved_suffix=preserved_suffix,
        next_attempt_index=attempt_id + 1,
        semantic_mcp_passed=False,
        mcp_used=mcp_used,
        mcp_error=mcp_error,
    )


def _exclusive_goal_action_lower_bound(state: PipeState) -> int:
    """Necessary action count from claim exclusivity and the goal DAG.

    This remains a lower bound rather than a generated plan.  Claim groups catch
    parallel goals that cannot share one physical measurement.  The weighted DAG
    catches serial agendas whose different primitive families were previously
    hidden by ``max(group_counts)``.  Only the statically proved final-turn /
    START-anchor circular closure may occupy the same dependency transition.
    """

    groups: Counter[str] = Counter()
    goals_by_id = {
        str(goal.goal_id): goal
        for goal in state.remaining_goals
        if goal.goal_id is not None
    }
    longest_path: dict[str, int] = {}
    previous_goal: Goal | None = None
    for goal in state.remaining_goals:
        if goal.type in {"move", "route", "connector"} and goal.length is not None:
            groups["linear_length"] += 1
        if goal.type == "turn" and goal.angle is not None:
            groups["turn_angle"] += 1
        if goal.type == "branch":
            groups["branch_topology"] += 1
        if goal.type == "diameter_change":
            groups["diameter_change"] += 1
        if goal.type == "connect":
            groups["connect_topology"] += 1
        if goal.type == "end":
            groups["end_topology"] += 1
        if goal.type == "connector" and goal.component is not None:
            groups["component_instance"] += 1

        goal_id = str(goal.goal_id) if goal.goal_id is not None else ""
        dependencies = [
            goals_by_id[dependency_id]
            for dependency_id in goal.depends_on_goal_ids
            if dependency_id in goals_by_id
        ]
        # A non-parallel ordered agenda cannot bypass an earlier pending goal
        # even when the author omitted a redundant explicit dependency edge.
        if not goal.allow_parallel and not dependencies and previous_goal is not None:
            dependencies = [previous_goal]
        path_length = 1
        for dependency in dependencies:
            dependency_id = str(dependency.goal_id)
            dependency_length = longest_path.get(dependency_id, 1)
            shares_transition = bool(
                goal.type == "connect"
                and goal.connection_target == "start_anchor"
                and dependency.type == "turn"
                and dependency.angle is not None
            )
            path_length = max(
                path_length,
                dependency_length + (0 if shares_transition else 1),
            )
        if goal_id:
            longest_path[goal_id] = path_length
        previous_goal = goal

    return max(
        max(groups.values(), default=0),
        max(longest_path.values(), default=0),
    )


def _validate_checkpoint_state(
    state: PipeState,
    intent: IntentResult,
    expected_digest: Any,
    actions: list[dict[str, Any]],
) -> None:
    digest = _pipe_state_digest(state)
    if expected_digest is not None and expected_digest != digest:
        raise ValueError("Checkpoint state digest mismatch")
    if state.contract_digest != intent.contract_digest:
        raise ValueError("Checkpoint state contract digest mismatch")
    if (
        state.expected_open_ports != intent.expected_open_ports
        or state.expected_open_ports_source != intent.expected_open_ports_source
        or state.required_components != intent.required_components
        or state.hard_constraints != intent.hard_constraints
        or state.geometric_constraints != intent.geometric_constraints
        or state.design_notes != intent.design_notes
    ):
        raise ValueError("Checkpoint state immutable contract fields mismatch")
    if state.state_version != len(actions):
        raise ValueError(
            "Checkpoint state version does not match accepted action count"
        )
    if len(state.action_history) != len(actions):
        raise ValueError("Checkpoint action history does not match accepted actions")
    parsed_actions = [ResolvedAction.model_validate(item) for item in actions]
    if [item.model_dump(mode="json") for item in parsed_actions] != [
        item.model_dump(mode="json") for item in state.action_history
    ]:
        raise ValueError("Checkpoint accepted actions differ from state.action_history")
    if state.state_id != f"S{state.state_version}":
        raise ValueError("Checkpoint state_id is not derived from state_version")
    if len(state.placed_modules) != state.state_version:
        raise ValueError("Checkpoint module count does not match state_version")
    for index, (action, module) in enumerate(
        zip(parsed_actions, state.placed_modules), start=1
    ):
        if (
            action.action_id != f"A{index}"
            or module.id != f"M{index}"
            or module.type != action.module
        ):
            raise ValueError("Checkpoint action/module IDs or types are inconsistent")


def _checkpoint_history(
    payload: dict[str, Any], *, state: PipeState
) -> list[PipeState]:
    raw_history = payload.get("committed_states") or []
    history = [PipeState.model_validate(item) for item in raw_history]
    if not history or _pipe_state_digest(history[-1]) != _pipe_state_digest(state):
        history.append(state.model_copy(deep=True))
    return history


def _validate_checkpoint_history(
    intent: IntentResult,
    actions: list[dict[str, Any]],
    steps: list[StepVerification],
    checkpoints: list[PipeState],
) -> None:
    if len(checkpoints) != len(actions) + 1:
        raise ValueError("Checkpoint history length does not match accepted actions")
    if len(steps) != len(actions):
        raise ValueError(
            "Checkpoint step journal length does not match accepted actions"
        )
    for index, action_payload in enumerate(actions, start=1):
        action = ResolvedAction.model_validate(action_payload)
        before = checkpoints[index - 1]
        after = checkpoints[index]
        if (
            before.state_version != index - 1
            or after.state_version != index
            or after.action_history[-1].action_id != action.action_id
        ):
            raise ValueError("Checkpoint history state/action sequence is inconsistent")
        recomputed = build_step_verification(
            before,
            action,
            after,
            intent,
            index,
            mcp_required=steps[index - 1].mcp_required,
        )
        if has_errors(recomputed.issues):
            raise ValueError(
                "Checkpoint history no longer passes deterministic validation"
            )
        if recomputed.transition.model_dump(mode="json") != steps[
            index - 1
        ].transition.model_dump(mode="json"):
            raise ValueError("Checkpoint stored transition differs from recomputation")
        if [issue.issue_code for issue in recomputed.issues] != [
            issue.issue_code for issue in steps[index - 1].issues
        ]:
            raise ValueError("Checkpoint stored step issues differ from recomputation")


def _pipe_state_digest(state: PipeState) -> str:
    return _canonical_json_digest(state.model_dump(mode="json"))


def _canonical_json_digest(value: Any) -> str:
    """Return the stable SHA-256 used by diagnostic and checkpoint bindings."""

    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _prepared_manifest(
    run_id: str,
    intent: IntentResult,
    previous_state: PipeState,
    candidate_state: PipeState,
    actions: list[dict[str, Any]],
    steps: list[StepVerification],
    attempts: list[ActionAttempt],
    checkpoints: list[PipeState],
    draft: ActionDraft,
    action: ResolvedAction,
    step: StepVerification,
    attempt_id: int,
    gemini: Any,
    *,
    previous_freecad_verified: bool,
    preserved_suffix: _PreservedSuffix | None = None,
    diagnostic_journal: DiagnosticJournal | None = None,
) -> dict[str, Any]:
    return {
        "phase": "PREPARED",
        "run_id": run_id,
        "intent": intent.model_dump(mode="json"),
        "previous_state": previous_state.model_dump(mode="json"),
        "candidate_state": candidate_state.model_dump(mode="json"),
        "candidate_digest": geometry_payload_digest(candidate_state),
        "candidate_state_digest": _pipe_state_digest(candidate_state),
        "candidate_document": candidate_document_name(
            candidate_state, run_id=run_id, attempt_id=attempt_id
        ),
        "published_document": published_document_name(candidate_state, run_id=run_id),
        "previous_freecad_verified": previous_freecad_verified,
        "action": action.model_dump(mode="json"),
        "draft": draft.model_dump(mode="json"),
        "attempt_id": attempt_id,
        "step_verification": step.model_dump(mode="json"),
        "actions": actions,
        "step_verifications": [item.model_dump(mode="json") for item in steps],
        "attempts": [item.model_dump(mode="json") for item in attempts],
        "committed_states": [item.model_dump(mode="json") for item in checkpoints],
        "planner_lineage": _lineage_snapshot(gemini),
        "planner_schema_profiles": _planner_schema_profiles_snapshot(gemini),
        "llm_usage": _usage_snapshot(gemini).model_dump(mode="json"),
        "action_budget_policy": copy.deepcopy(
            getattr(gemini, "_cadgen_action_budget_policy", None)
        ),
        "diagnostic_journal": (diagnostic_journal or DiagnosticJournal()).model_dump(
            mode="json"
        ),
        "preserved_suffix": (
            preserved_suffix.to_payload() if preserved_suffix is not None else None
        ),
    }


def _write_checkpoint(
    path: Path,
    *,
    phase: str,
    run_id: str,
    intent: IntentResult,
    state: PipeState,
    previous_state: PipeState | None,
    actions: list[dict[str, Any]],
    step_verifications: list[StepVerification],
    attempts: list[ActionAttempt],
    gemini: Any,
    committed_states: list[PipeState],
    freecad_verified: bool,
    pending_repair_observations: list[dict[str, Any]] | None = None,
    pending_draft: ActionDraft | None = None,
    pending_draft_attempt_index: int | None = None,
    preserved_suffix: _PreservedSuffix | None = None,
    next_attempt_index: int = 1,
    in_flight_operation: str | None = None,
    diagnostic_journal: DiagnosticJournal | None = None,
) -> None:
    diagnostic_journal = diagnostic_journal or DiagnosticJournal()
    if next_attempt_index < 1:
        raise ValueError("next_attempt_index must be positive")
    if (pending_draft is None) != (pending_draft_attempt_index is None):
        raise ValueError("Pending draft and attempt index must be journaled together")
    if (
        pending_draft_attempt_index is not None
        and pending_draft_attempt_index != next_attempt_index
    ):
        raise ValueError("Pending draft attempt must equal next_attempt_index")
    if pending_draft is not None and any(
        record.status == "pending"
        and record.state_id == state.state_id
        and record.step_index == len(actions) + 1
        for record in diagnostic_journal.records
    ):
        raise ValueError(
            "A pending planner draft cannot coexist with a pending geometry diagnosis"
        )
    _atomic_write_json(
        path,
        {
            "phase": phase,
            "run_id": run_id,
            "intent": intent.model_dump(mode="json"),
            "previous_state": (
                previous_state.model_dump(mode="json") if previous_state else None
            ),
            "state": state.model_dump(mode="json"),
            "state_digest": _pipe_state_digest(state),
            "geometry_digest": geometry_payload_digest(state),
            "freecad_verified": freecad_verified,
            "committed_states": [
                item.model_dump(mode="json") for item in committed_states
            ],
            "actions": actions,
            "step_verifications": [
                item.model_dump(mode="json") for item in step_verifications
            ],
            "attempts": [item.model_dump(mode="json") for item in attempts],
            "planner_lineage": _lineage_snapshot(gemini),
            "planner_schema_profiles": _planner_schema_profiles_snapshot(gemini),
            "llm_usage": _usage_snapshot(gemini).model_dump(mode="json"),
            "action_budget_policy": copy.deepcopy(
                getattr(gemini, "_cadgen_action_budget_policy", None)
            ),
            "diagnostic_journal": (diagnostic_journal).model_dump(mode="json"),
            "pending_repair_observations": pending_repair_observations or [],
            "pending_draft": (
                pending_draft.model_dump(mode="json") if pending_draft else None
            ),
            "pending_draft_state_digest": (
                _pipe_state_digest(state) if pending_draft else None
            ),
            "pending_draft_attempt_index": pending_draft_attempt_index,
            "next_attempt_index": next_attempt_index,
            "in_flight_operation": in_flight_operation,
            "in_flight_started_at": (
                datetime.now(timezone.utc).isoformat()
                if in_flight_operation is not None
                else None
            ),
            "preserved_suffix": (
                preserved_suffix.to_payload() if preserved_suffix is not None else None
            ),
        },
    )


def _lineage_snapshot(gemini: Any) -> dict[str, Any]:
    if gemini is not None and hasattr(gemini, "lineage_snapshot"):
        return gemini.lineage_snapshot()
    return {}


def _planner_schema_profiles_snapshot(gemini: Any) -> dict[str, str]:
    profiles = getattr(gemini, _PLANNER_SCHEMA_PROFILE_ATTR, None)
    if not isinstance(profiles, dict):
        return {}
    return {
        str(state_id): str(profile)
        for state_id, profile in profiles.items()
        if profile in _PLANNER_SCHEMA_PROFILES
    }


def _usage_snapshot(gemini: Any) -> LLMUsage:
    if gemini is not None and hasattr(gemini, "usage_snapshot"):
        return gemini.usage_snapshot()
    return LLMUsage()


def _attempt(
    step_index: int,
    attempt_index: int,
    state: PipeState,
    phase: str,
    status: str,
    draft: ActionDraft | None,
    resolved: ResolvedAction | None,
    issues: list[StaticIssue],
) -> ActionAttempt:
    return ActionAttempt(
        step_index=step_index,
        attempt_index=attempt_index,
        state_id=state.state_id,
        state_digest=pipe_state_digest(state),
        phase=phase,
        status=status,  # type: ignore[arg-type]
        draft=draft.model_dump(mode="json") if draft else None,
        resolved=resolved.model_dump(mode="json") if resolved else None,
        issue_codes=[issue.issue_code for issue in issues],
        observations=[_repair_observation(issue) for issue in issues],
    )


def _record_repair_advice(
    path: Path,
    *,
    step_index: int,
    attempt_index: int,
    state_id: str,
    context: dict[str, Any],
) -> None:
    """Append one advisor decision to a durable, user-visible audit journal."""

    history: list[Any] = []
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                history = loaded
        except (OSError, json.JSONDecodeError):
            history = []
    history.append(
        {
            "step_index": step_index,
            "attempt_index": attempt_index,
            "state_id": state_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **context,
        }
    )
    _atomic_write_json(path, history)


_DIAGNOSABLE_REJECTION_PHASES = {
    "draft_validation",
    "action_resolution",
    "registry_validation",
    "state_application",
    "static_step_validation",
    "freecad_semantic_validation",
    "final_repair_replan",
    "checkpoint_recovery",
}


def _diagnostic_repair_epoch(attempts: list[ActionAttempt], step_index: int) -> int:
    """Accepted versions of one step delimit rollback repair epochs."""

    return sum(
        attempt.step_index == step_index and attempt.status == "accepted"
        for attempt in attempts
    )


def _diagnostic_attempt_history(
    attempts: list[ActionAttempt], step_index: int, repair_epoch: int
) -> list[ActionAttempt]:
    accepted_seen = 0
    result: list[ActionAttempt] = []
    for attempt in attempts:
        if attempt.step_index != step_index:
            continue
        if attempt.status == "accepted":
            accepted_seen += 1
            continue
        if accepted_seen == repair_epoch:
            result.append(attempt)
    return result


def _diagnostic_evidence_from_observations(
    observations: list[dict[str, Any]], run_dir: Path
) -> tuple[dict[str, Any], str | None]:
    """Prefer the full digest-bound evidence artifact over lossy observations."""

    run_root = run_dir.resolve()
    for observation in reversed(observations):
        actual = observation.get("actual")
        if not isinstance(actual, dict):
            continue
        path_value = actual.get("evidence_artifact_path")
        expected_digest = actual.get("evidence_digest")
        if not isinstance(path_value, str) or not path_value:
            continue
        try:
            evidence_path = Path(path_value).expanduser().resolve()
            if not evidence_path.is_relative_to(run_root):
                raise ValueError("evidence artifact escapes the run directory")
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("evidence artifact must contain a JSON object")
            if (
                isinstance(expected_digest, str)
                and _canonical_json_digest(payload) != expected_digest
            ):
                raise ValueError("evidence artifact digest mismatch")
            return payload, None
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return {"observations": observations}, str(exc)

    merged: dict[str, Any] = {"observations": observations}
    compact_checks: dict[str, Any] = {}
    for observation in observations:
        actual = observation.get("actual")
        if not isinstance(actual, dict):
            continue
        compact = actual.get("evidence")
        if not isinstance(compact, dict):
            continue
        if isinstance(compact.get("failed_checks"), dict):
            compact_checks.update(compact["failed_checks"])
        if isinstance(compact.get("validator_policy"), dict):
            merged["validator_policy"] = compact["validator_policy"]
        for key, value in compact.items():
            merged.setdefault(str(key), value)
    if compact_checks:
        merged["checks"] = compact_checks
    return merged, None


def _diagnostic_recommendations(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for observation in observations:
        suggestion = observation.get("suggestion")
        if not isinstance(suggestion, dict):
            continue
        changes = suggestion.get("recommended_changes")
        if isinstance(changes, list):
            recommendations.extend(
                dict(item) for item in changes if isinstance(item, dict)
            )
        elif any(
            suggestion.get(key)
            for key in ("parameter_errors", "operation", "instruction")
        ):
            recommendations.append(dict(suggestion))
    return recommendations[:24]


def _diagnostic_artifact_paths(paths: dict[str, Path], case: Any) -> dict[str, Path]:
    binding = case.binding
    stem = f"step_{binding.step_index}_attempt_{binding.attempt_index}"
    if binding.repair_epoch:
        stem += f"_epoch_{binding.repair_epoch}"
    directory = paths["diagnostics_dir"]
    primary_case_path = directory / f"{stem}_case.json"
    if primary_case_path.is_file():
        try:
            prior_case = json.loads(primary_case_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior_case = None
        if not isinstance(prior_case, dict) or _canonical_json_digest(
            prior_case
        ) != diagnostic_case_digest(case):
            stem += f"_{diagnostic_case_id(case)[:12]}"
    return {
        "case": directory / f"{stem}_case.json",
        "diagnosis": directory / f"{stem}_diagnosis.json",
        "failure": directory / f"{stem}_advisor_failure.json",
    }


def _upsert_diagnostic_record(
    journal: DiagnosticJournal,
    record: DiagnosticRecordRef,
    *,
    increment_call: bool = False,
    increment_cache_hit: bool = False,
    increment_futile_avoided: bool = False,
) -> DiagnosticJournal:
    records = [item for item in journal.records if item.case_id != record.case_id]
    records.append(record)
    calls = dict(journal.calls_by_step)
    if increment_call:
        key = f"{record.step_index}:{record.repair_epoch}"
        calls[key] = calls.get(key, 0) + 1
    return journal.model_copy(
        update={
            "records": records,
            "calls_by_step": calls,
            "cache_hit_count": journal.cache_hit_count + int(increment_cache_hit),
            "futile_retry_avoided_count": journal.futile_retry_avoided_count
            + int(increment_futile_avoided),
        }
    )


def _diagnostic_record(
    case: Any,
    *,
    status: str,
    artifact_path: Path | None = None,
    failure_reason: str | None = None,
) -> DiagnosticRecordRef:
    return DiagnosticRecordRef(
        case_id=diagnostic_case_id(case),
        diagnostic_context_digest=diagnostic_case_digest(case),
        failure_signature=case.binding.failure_signature,
        state_id=case.binding.state_id,
        step_index=case.binding.step_index,
        attempt_index=case.binding.attempt_index,
        repair_epoch=case.binding.repair_epoch,
        status=status,
        artifact_path=str(artifact_path) if artifact_path is not None else None,
        failure_reason=failure_reason,
    )


def _diagnostic_artifact_payload_digest(payload: dict[str, Any]) -> str:
    return _canonical_json_digest(
        {key: value for key, value in payload.items() if key != "artifact_digest"}
    )


def _load_diagnostic_artifact(
    path: Path,
    *,
    case: Any,
) -> tuple[StepRepairDiagnosis, dict[str, Any]] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("diagnosis artifact must be a JSON object")
    if payload.get("artifact_digest") != _diagnostic_artifact_payload_digest(payload):
        raise ValueError("diagnosis artifact digest mismatch")
    if payload.get("case_id") != diagnostic_case_id(case):
        raise ValueError("diagnosis artifact case binding mismatch")
    diagnosis = StepRepairDiagnosis.model_validate(payload.get("diagnosis"))
    usage_delta = payload.get("usage_delta")
    return validate_diagnosis(case, diagnosis), (
        dict(usage_delta) if isinstance(usage_delta, dict) else {}
    )


def _load_diagnostic_failure(path: Path, *, case: Any) -> tuple[str, bool] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("advisor failure artifact must be a JSON object")
    if payload.get("artifact_digest") != _diagnostic_artifact_payload_digest(payload):
        raise ValueError("advisor failure artifact digest mismatch")
    if payload.get("case_id") != diagnostic_case_id(case):
        raise ValueError("advisor failure artifact case binding mismatch")
    return (
        str(payload.get("failure_reason") or "advisor_failure"),
        payload.get("call_attempted") is True,
    )


def _llm_usage_delta(before: LLMUsage, after: LLMUsage) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, before_value in before.model_dump(mode="json").items():
        after_value = after.model_dump(mode="json").get(key)
        if isinstance(before_value, int) and isinstance(after_value, int):
            result[key] = max(0, after_value - before_value)
    result["accounting_complete"] = after.accounting_complete
    return result


def _roll_forward_diagnostic_usage(gemini: Any, delta: dict[str, Any]) -> None:
    """Recover usage lost between diagnosis artifact and checkpoint writes."""

    if gemini is None or not hasattr(gemini, "restore_usage"):
        return
    current = _usage_snapshot(gemini).model_dump(mode="json")
    for key, value in delta.items():
        if key == "accounting_complete":
            current[key] = bool(current.get(key, True)) and bool(value)
        elif isinstance(value, int) and isinstance(current.get(key), int):
            current[key] += max(0, value)
    gemini.restore_usage(LLMUsage.model_validate(current))


def _advisor_failure_reason(exc: Exception) -> str:
    if isinstance(exc, GeminiBudgetError):
        return "budget_exhausted"
    if isinstance(exc, StructuredOutputError):
        return "structured_output_error"
    if isinstance(exc, HostContractValidationError):
        return "host_contract_validation_error"
    if isinstance(exc, GeminiRequestError):
        return "provider_error"
    message = str(exc).lower()
    if "binding" in message:
        return "binding_mismatch"
    if "evidence" in message:
        return "unknown_evidence_reference"
    if "cannot change" in message or "field" in message:
        return "forbidden_field_recommendation"
    return "host_validation_error"


def _advisor_unavailable_observation(
    case: Any,
    *,
    reason: str,
    protocol_attempt_count: int,
) -> dict[str, Any]:
    """Record degraded diagnosis while preserving deterministic planner repair.

    This observation never supplies replacement parameters.  It exists solely
    to make the missing independent diagnosis auditable.  The ordinary planner
    still receives the exact validator evidence and may use its remaining
    bounded semantic attempts; an advisor provider/schema failure is not
    authority to terminate repair.
    """

    return {
        "context_type": "geometry_validation_advisor_unavailable",
        "terminal": False,
        "fallback": "deterministic_evidence_only",
        "state_id": case.binding.state_id,
        "diagnostic_context_digest": diagnostic_case_digest(case),
        "failure_signature": case.binding.failure_signature,
        "failure_reason": reason,
        "protocol_attempt_count": protocol_attempt_count,
        "instruction": (
            "The independent advisor is unavailable. Replan only from the last "
            "deterministic validator rejection, preserve unrelated fields, and "
            "run the replacement through every existing validation gate."
        ),
    }


def _required_advisor_capacity_failure(
    case: Any,
    journal: DiagnosticJournal,
    settings: Settings,
) -> str | None:
    """Explain a policy-cap skip that must terminate rather than replan."""

    key = f"{case.binding.step_index}:{case.binding.repair_epoch}"
    if (
        journal.calls_by_step.get(key, 0)
        >= settings.step_repair_advisor_max_calls_per_step
    ):
        return "advisor_call_cap_exhausted"

    records = [
        record
        for record in journal.records
        if record.step_index == case.binding.step_index
        and record.repair_epoch == case.binding.repair_epoch
        and record.status in {"pending", "complete"}
    ]
    signatures = {record.failure_signature for record in records}
    if (
        case.binding.failure_signature not in signatures
        and len(signatures) >= settings.step_repair_advisor_max_signatures_per_step
    ):
        return "advisor_failure_family_cap_exhausted"
    return None


def _run_step_geometry_diagnostician(
    *,
    run_id: str,
    run_dir: Path,
    paths: dict[str, Path],
    state: PipeState,
    step_index: int,
    observations: list[dict[str, Any]],
    attempts: list[ActionAttempt],
    settings: Settings,
    gemini: Any,
    journal: DiagnosticJournal,
    stream: ThinkingStream,
    persist: Any,
) -> tuple[list[dict[str, Any]], DiagnosticJournal]:
    """Run or recover one candidate-bound advisor episode before replanning."""

    step_attempts = [
        attempt for attempt in attempts if attempt.step_index == step_index
    ]
    if not observations or not step_attempts:
        return observations, journal
    latest = step_attempts[-1]
    if latest.status != "rejected" or latest.phase not in _DIAGNOSABLE_REJECTION_PHASES:
        return observations, journal
    advisor_runtime_enabled = bool(
        settings.step_repair_advisor_enabled
        and gemini is not None
        and getattr(gemini, "supports_step_repair_advisor", False)
    )
    advisor_required = bool(
        advisor_runtime_enabled and settings.step_repair_advisor_required
    )

    repair_epoch = _diagnostic_repair_epoch(attempts, step_index)
    evidence, evidence_error = _diagnostic_evidence_from_observations(
        observations, run_dir
    )
    case = build_diagnostic_case(
        run_id=run_id,
        state=state,
        step_index=step_index,
        attempt_index=latest.attempt_index,
        repair_epoch=repair_epoch,
        draft=latest.draft or {},
        resolved_action=latest.resolved,
        issues=latest.observations or observations,
        evidence=evidence,
        attempt_history=_diagnostic_attempt_history(attempts, step_index, repair_epoch),
        deterministic_recommendations=_diagnostic_recommendations(observations),
    )
    context_digest = diagnostic_case_digest(case)
    if any(
        item.get("context_type") == "step_geometry_diagnosis"
        and item.get("diagnostic_context_digest") == context_digest
        for item in observations
    ):
        return observations, journal
    # Candidate-specific advice must never bleed into a new digest, even when
    # the family signature is intentionally the same for call deduplication.
    observations = [
        item
        for item in observations
        if item.get("context_type") != "step_geometry_diagnosis"
    ]

    artifact_paths = _diagnostic_artifact_paths(paths, case)
    _atomic_write_json(artifact_paths["case"], case.model_dump(mode="json"))
    case_id = diagnostic_case_id(case)
    existing = next(
        (record for record in journal.records if record.case_id == case_id),
        None,
    )

    # Artifact-before-checkpoint crash: validate and roll forward without a
    # second paid call.  Never trust the record's arbitrary artifact_path.
    try:
        cached_artifact = _load_diagnostic_artifact(
            artifact_paths["diagnosis"], case=case
        )
        cached_failure_artifact = _load_diagnostic_failure(
            artifact_paths["failure"], case=case
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        cached_artifact = None
        cached_failure_artifact = (f"cached_artifact_invalid: {exc}", False)

    if cached_artifact is not None:
        cached, cached_usage_delta = cached_artifact
        rolled_forward = existing is None or existing.status == "pending"
        if rolled_forward:
            _roll_forward_diagnostic_usage(gemini, cached_usage_delta)
        directive = planner_directive_from_diagnosis(cached, case)
        observations = [*observations, directive]
        journal = _upsert_diagnostic_record(
            journal,
            _diagnostic_record(
                case,
                status="complete",
                artifact_path=artifact_paths["diagnosis"],
            ),
            increment_call=rolled_forward,
            increment_cache_hit=True,
        )
        persist(observations, journal, None)
        stream.emit(
            f"Step {step_index} reused a digest-bound inverse geometry diagnosis.",
            force=True,
        )
        return observations, journal

    if cached_failure_artifact is not None:
        cached_failure, cached_call_attempted = cached_failure_artifact
        if not artifact_paths["failure"].is_file():
            failure_payload = {
                "case_id": case_id,
                "diagnostic_context_digest": context_digest,
                "failure_signature": case.binding.failure_signature,
                "failure_reason": cached_failure[:1000],
                "call_attempted": False,
                "error_type": "CachedArtifactValidationError",
                "error": cached_failure[:2000],
                "model_part": "step_repair_advisor",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            failure_payload["artifact_digest"] = _diagnostic_artifact_payload_digest(
                failure_payload
            )
            _atomic_write_json(artifact_paths["failure"], failure_payload)
        journal = _upsert_diagnostic_record(
            journal,
            _diagnostic_record(
                case,
                status="failed",
                artifact_path=artifact_paths["failure"],
                failure_reason=cached_failure[:1000],
            ),
            increment_call=(
                cached_call_attempted
                and (existing is None or existing.status == "pending")
            ),
        )
        if advisor_required:
            observations = [
                *observations,
                _advisor_unavailable_observation(
                    case,
                    reason=cached_failure[:1000],
                    protocol_attempt_count=int(cached_call_attempted),
                ),
            ]
        persist(observations, journal, None)
        return observations, journal

    if existing is not None and existing.status in {"complete", "failed", "skipped"}:
        # Complete-without-artifact is never silently trusted.  It degrades to
        # generic deterministic feedback without another call.
        if existing.status == "complete":
            replacement = _diagnostic_record(
                case,
                status="failed",
                artifact_path=artifact_paths["case"],
                failure_reason="complete diagnosis artifact is missing",
            )
            journal = _upsert_diagnostic_record(journal, replacement)
        if advisor_required and existing.status in {"complete", "failed"}:
            observations = [
                *observations,
                _advisor_unavailable_observation(
                    case,
                    reason=(
                        existing.failure_reason or "validated_advisor_artifact_missing"
                    ),
                    protocol_attempt_count=0,
                ),
            ]
        if existing.status in {"complete", "failed"}:
            persist(observations, journal, None)
        return observations, journal

    if evidence_error is not None:
        journal = _upsert_diagnostic_record(
            journal,
            _diagnostic_record(
                case,
                status="skipped",
                artifact_path=artifact_paths["case"],
                failure_reason=f"evidence_artifact_invalid: {evidence_error}"[:1000],
            ),
        )
        if advisor_required:
            observations = [
                *observations,
                _advisor_unavailable_observation(
                    case,
                    reason="evidence_artifact_invalid",
                    protocol_attempt_count=0,
                ),
            ]
        persist(observations, journal, None)
        return observations, journal

    decision_journal = journal
    if existing is not None and existing.status == "pending":
        decision_journal = journal.model_copy(
            update={
                "records": [
                    record for record in journal.records if record.case_id != case_id
                ]
            }
        )
    call_advisor = should_call_advisor(
        case,
        decision_journal,
        enabled=advisor_runtime_enabled,
        dry_run=False,
        freecad_enabled=settings.freecad_mcp_enabled,
        trigger_attempt=settings.step_repair_advisor_trigger_attempt,
        max_calls_per_step=settings.step_repair_advisor_max_calls_per_step,
        max_signatures_per_step=(settings.step_repair_advisor_max_signatures_per_step),
    )
    if not call_advisor:
        reason = (
            "disabled"
            if not advisor_runtime_enabled
            else "deduplicated_or_deterministic_fast_path"
        )
        journal = _upsert_diagnostic_record(
            journal,
            _diagnostic_record(
                case,
                status="skipped",
                artifact_path=artifact_paths["case"],
                failure_reason=reason,
            ),
        )
        capacity_failure = (
            _required_advisor_capacity_failure(case, decision_journal, settings)
            if advisor_required
            else None
        )
        if capacity_failure is not None:
            observations = [
                *observations,
                _advisor_unavailable_observation(
                    case,
                    reason=capacity_failure,
                    protocol_attempt_count=0,
                ),
            ]
        persist(observations, journal, None)
        return observations, journal

    journal = _upsert_diagnostic_record(
        journal,
        _diagnostic_record(
            case,
            status="pending",
            artifact_path=artifact_paths["case"],
        ),
    )
    # This checkpoint is deliberately before the optional paid call.
    persist(observations, journal, "step_repair_advisor")
    before_usage = _usage_snapshot(gemini)
    advisor_response_payload: dict[str, Any] | None = None
    protocol_errors: list[dict[str, str]] = []
    protocol_attempt_count = 0
    try:
        diagnosis: StepRepairDiagnosis | None = None
        for protocol_attempt_count in range(1, 3):
            if hasattr(gemini, "reset_lineage"):
                gemini.reset_lineage("step_repair_advisor")
            retry_note = ""
            if protocol_errors:
                previous = protocol_errors[-1]
                retry_note = (
                    "\n\nThe previous independent-advisor response was rejected "
                    "at the protocol/host-validation boundary. Re-evaluate the "
                    "same typed case from scratch and return one complete object. "
                    f"error_type={previous['error_type']}; "
                    f"failure_reason={previous['failure_reason']}."
                )
            try:
                advisor_response = _call_structured(
                    gemini,
                    step_repair_advisor_prompt(case),
                    GeometryValidationAdvisorResponse,
                    part="step_repair_advisor",
                    thinking_level=(
                        "medium" if protocol_attempt_count == 1 else "high"
                    ),
                    system_instruction=(
                        step_repair_advisor_system_instruction() + retry_note
                    ),
                )
                if isinstance(advisor_response, GeometryValidationAdvisorResponse):
                    advisor_response_payload = advisor_response.model_dump(mode="json")
                    diagnosis = bind_advisor_response(case, advisor_response)
                elif isinstance(
                    advisor_response, StepRepairDiagnosisBody
                ) and not isinstance(advisor_response, StepRepairDiagnosis):
                    advisor_response_payload = advisor_response.model_dump(mode="json")
                    diagnosis = bind_diagnosis(case, advisor_response)
                elif isinstance(advisor_response, StepRepairDiagnosis):
                    # Compatibility for existing test doubles and resumable v1
                    # integrations. Real calls use the provider-safe wire DTO.
                    advisor_response_payload = advisor_response.model_dump(mode="json")
                    diagnosis = validate_diagnosis(case, advisor_response)
                else:
                    raise TypeError("step repair advisor returned the wrong schema")
                break
            except (
                GeminiRequestError,
                StructuredOutputError,
                HostContractValidationError,
                TypeError,
                ValueError,
            ) as protocol_exc:
                protocol_errors.append(
                    {
                        "error_type": type(protocol_exc).__name__,
                        "failure_reason": _advisor_failure_reason(protocol_exc),
                        "error": str(protocol_exc)[:1000],
                    }
                )
                if protocol_attempt_count >= 2:
                    raise
                stream.emit(
                    f"Step {step_index} geometry diagnostician protocol response "
                    "was rejected; retrying the same diagnostic episode once.",
                    force=True,
                )
        if diagnosis is None:
            raise DiagnosticValidationError(
                "advisor protocol completed without a validated diagnosis"
            )
        directive = planner_directive_from_diagnosis(diagnosis, case)
        after_usage = _usage_snapshot(gemini)
        artifact_payload = {
            "case_id": case_id,
            "binding": case.binding.model_dump(mode="json"),
            "advisor_response_body": advisor_response_payload,
            "diagnosis": diagnosis.model_dump(mode="json"),
            "validated": True,
            "model_part": "step_repair_advisor",
            "model": settings.model_for("step_repair_advisor"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "usage_delta": _llm_usage_delta(before_usage, after_usage),
            "protocol_attempt_count": protocol_attempt_count,
            "protocol_errors": protocol_errors,
            "planner_directive": directive,
        }
        artifact_payload["artifact_digest"] = _diagnostic_artifact_payload_digest(
            artifact_payload
        )
        _atomic_write_json(artifact_paths["diagnosis"], artifact_payload)
        observations = [*observations, directive]
        journal = _upsert_diagnostic_record(
            journal,
            _diagnostic_record(
                case,
                status="complete",
                artifact_path=artifact_paths["diagnosis"],
            ),
            increment_call=True,
        )
        # Journal completion and planner directive become durable together.
        persist(observations, journal, None)
        stream.emit(
            f"Step {step_index} received one evidence-bound inverse geometry "
            "parameter diagnosis before replanning.",
            force=True,
        )
    except Exception as exc:
        reason = _advisor_failure_reason(exc)
        failure_payload = {
            "case_id": case_id,
            "diagnostic_context_digest": context_digest,
            "failure_signature": case.binding.failure_signature,
            "failure_reason": reason,
            "call_attempted": True,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "advisor_response": advisor_response_payload,
            "protocol_attempt_count": protocol_attempt_count,
            "protocol_errors": protocol_errors,
            "model_part": "step_repair_advisor",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        failure_payload["artifact_digest"] = _diagnostic_artifact_payload_digest(
            failure_payload
        )
        _atomic_write_json(artifact_paths["failure"], failure_payload)
        journal = _upsert_diagnostic_record(
            journal,
            _diagnostic_record(
                case,
                status="failed",
                artifact_path=artifact_paths["failure"],
                failure_reason=reason,
            ),
            increment_call=True,
        )
        if advisor_required:
            observations = [
                *observations,
                _advisor_unavailable_observation(
                    case,
                    reason=reason,
                    protocol_attempt_count=protocol_attempt_count,
                ),
            ]
        persist(observations, journal, None)
        stream.emit(
            f"Step {step_index} geometry diagnostician failed safely "
            f"({reason}); deterministic evidence-only replanning remains active.",
            force=True,
        )
    finally:
        if hasattr(gemini, "reset_lineage"):
            gemini.reset_lineage("step_repair_advisor")
    return observations, journal


def _repair_observation(issue: StaticIssue) -> dict[str, Any]:
    return {
        "issue_id": issue.issue_id,
        "issue_code": issue.issue_code,
        "check_name": issue.check_name,
        "message": issue.message,
        "step_index": issue.step_index,
        "action_id": issue.action_id,
        "module_id": issue.module_id,
        "port_ids": issue.port_ids,
        "target_port_id": issue.target_port_id,
        "expected": issue.expected,
        "actual": issue.actual,
        "suggestion": issue.suggestion,
    }


def _planner_repair_context(
    observations: list[dict[str, Any]],
    attempts: list[ActionAttempt],
    step_index: int,
) -> list[dict[str, Any]]:
    """다음 planner 호출에 제한된 실패 이력과 교정 근거를 붙인다.

    Stateful lineage normally contains the previous draft, but it can expire or
    be reset after malformed structured output. The explicit history keeps the
    actionable failure context intact across retries and process resume without
    replaying the unbounded journal.
    """

    if not observations:
        return []
    # Final-critic rollback keeps the full audit journal but starts a new repair
    # epoch for the rolled-back step. Rejections from before that step's latest
    # accepted action must not make the first new failure look like the sixth.
    last_accepted_position = max(
        (
            index
            for index, attempt in enumerate(attempts)
            if attempt.step_index == step_index and attempt.status == "accepted"
        ),
        default=-1,
    )
    all_rejected = [
        attempt
        for attempt in attempts[last_accepted_position + 1 :]
        if attempt.step_index == step_index and attempt.status == "rejected"
    ]
    rejected = all_rejected[-3:]
    history: list[dict[str, Any]] = []
    for attempt in rejected:
        action = attempt.resolved or attempt.draft
        compact_action = None
        if isinstance(action, dict):
            compact_action = {
                key: action[key]
                for key in (
                    "target_port",
                    "module",
                    "params",
                    "affected_goal_ids",
                    "completed_goal_ids",
                )
                if key in action
            }
        history.append(
            {
                "attempt_index": attempt.attempt_index,
                "failure_phase": attempt.phase,
                "issue_codes": attempt.issue_codes,
                "rejected_action": compact_action,
                "validation_observations": _bounded_diagnostic(attempt.observations),
            }
        )
    result: list[dict[str, Any]] = [
        {
            "context_type": "rejected_attempt_history",
            "instruction": (
                "Do not repeat these rejected actions; use the validation "
                "observations below to make a material correction."
            ),
            "attempts": history,
        },
        *[dict(item) for item in observations],
    ]
    if rejected:
        latest = rejected[-1]
        latest_action = latest.resolved or latest.draft
        if latest.phase in {
            "static_step_validation",
            "freecad_semantic_validation",
            "final_repair_replan",
        } and isinstance(latest_action, dict):
            diagnosis_directive = next(
                (
                    item
                    for item in reversed(observations)
                    if item.get("context_type") == "step_geometry_diagnosis"
                ),
                None,
            )
            repair_scope = (
                str(diagnosis_directive.get("repair_scope") or "params")
                if diagnosis_directive is not None
                else "params"
            )
            preserve_keys = {
                "params": (
                    "target_port",
                    "module",
                    "affected_goal_ids",
                    "completed_goal_ids",
                ),
                "variant": (
                    "target_port",
                    "module",
                    "affected_goal_ids",
                    "completed_goal_ids",
                ),
                "primitive": (
                    "target_port",
                    "affected_goal_ids",
                    "completed_goal_ids",
                ),
                "topology": ("affected_goal_ids", "completed_goal_ids"),
                "rollback": ("affected_goal_ids", "completed_goal_ids"),
            }.get(
                repair_scope,
                (
                    "target_port",
                    "module",
                    "affected_goal_ids",
                    "completed_goal_ids",
                ),
            )
            result.insert(
                1,
                {
                    "context_type": "causal_repair_envelope",
                    "preserve_action_fields": {
                        key: latest_action.get(key) for key in preserve_keys
                    },
                    "repair_scope": repair_scope,
                    "editable_scope": (
                        "Only evidence-cited planner-authored fields in the "
                        f"validated {repair_scope} scope"
                    ),
                    "instruction": (
                        "Keep every field listed in preserve_action_fields exactly. "
                        "A validated diagnosis may release only the structural level "
                        "named by repair_scope; it cannot alter immutable goals or "
                        "accept geometry. Change only candidate-controlled fields "
                        "cited by expected/actual or diagnostic evidence, and never "
                        "repeat a parameter proven non-causal."
                    ),
                },
            )
    repeat_count, repeated_failure = _planner_stagnation_run(all_rejected)
    exact_repeat_count, exact_repeated_failure = _planner_exact_stagnation_run(
        all_rejected
    )
    if (
        exact_repeat_count >= MAX_IDENTICAL_VALIDATOR_FAILURES
        and exact_repeated_failure is not None
    ):
        # 같은 draft와 같은 expected/actual 증거가 반복될 때만 hard stop한다.
        # 서로 다른 waypoint가 요구 반경에 가까워지는 중이라면 같은 issue code
        # 라도 남은 repair 예산을 사용할 수 있어야 한다.
        raise _PlannerStagnationError(
            "identical deterministic validation failure repeated "
            f"{exact_repeat_count} times after planner strategy reset: "
            + json.dumps(exact_repeated_failure, ensure_ascii=False)[:900]
        )
    if repeat_count == 3 and repeated_failure is not None:
        schema_strategy = _stagnation_schema_strategy(repeated_failure)
        result.insert(
            0,
            {
                "context_type": "planner_stagnation",
                "repeat_count": repeat_count,
                "repeated_failure": repeated_failure,
                "schema_strategy": schema_strategy,
                "instruction": (
                    "The same validation failure persisted across materially "
                    "unsuccessful drafts. The planner lineage is being reset once. "
                    + (
                        "The exact-decimal response grammar is also enabled. "
                        if schema_strategy == "encoded"
                        else "The current numeric response grammar is retained. "
                    )
                    + "Re-plan from the full immutable state and causal repair "
                    "envelope. Preserve structural fields that passed earlier "
                    "validators; change only the independent geometry or sign "
                    "actually implicated by the evidence."
                ),
            },
        )
    return result


def _maybe_request_step_repair_advice(
    gemini: Any,
    state: PipeState,
    repair_context: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    attempts: list[ActionAttempt],
    step_index: int,
) -> dict[str, Any] | None:
    """Ask an independent model for bounded advice when ordinary repair stalls."""

    if not getattr(gemini, "supports_repair_advisor", False):
        return None
    if any(item.get("context_type") == "repair_advisor" for item in observations):
        return None
    rejected = [
        attempt
        for attempt in attempts
        if attempt.step_index == step_index and attempt.status == "rejected"
    ]
    repeat_count, _failure = _planner_stagnation_run(rejected)
    empty_freecad_recommendation = any(
        item.get("check_name") == "freecad_semantic_validation"
        and not (
            (item.get("suggestion") or {}).get("recommended_changes")
            if isinstance(item.get("suggestion"), dict)
            else None
        )
        for item in observations
        if isinstance(item, dict)
    )
    if repeat_count < 2 and not empty_freecad_recommendation:
        return None
    if hasattr(gemini, "reset_lineage"):
        gemini.reset_lineage("parameter")
    try:
        advice = _call_structured(
            gemini,
            (
                "Legacy bounded repair-advisor compatibility request. The typed "
                "Step Geometry Diagnostician is used by run_pipeline. Current "
                "state and deterministic repair context: "
                + json.dumps(
                    {
                        "state": compact_planner_payload(state, include_catalog=False),
                        "repair_context": repair_context,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
            StepRepairAdviceWire,
            part="parameter",
            thinking_level="high",
            system_instruction=step_repair_advisor_system_instruction(),
        )
    except (
        GeminiBudgetError,
        GeminiConfigError,
        GeminiRequestError,
        StructuredOutputError,
        HostContractValidationError,
        TypeError,
        ValueError,
    ):
        return None
    finally:
        if hasattr(gemini, "reset_lineage"):
            gemini.reset_lineage("parameter")
    if isinstance(advice, StepRepairAdviceWire):
        advice = StepRepairAdvice.model_validate(advice.model_dump(mode="python"))
    if not isinstance(advice, StepRepairAdvice):
        return None
    return {
        "context_type": "repair_advisor",
        "trigger_issue_codes": sorted(
            {
                str(item.get("issue_code"))
                for item in observations
                if isinstance(item, dict) and item.get("issue_code")
            }
        ),
        "advice": advice.model_dump(mode="json"),
        "instruction": (
            "Use this independent diagnosis as a bounded repair plan. It cannot "
            "change Intent and is not itself an executable action."
        ),
    }


def _terminal_geometry_diagnosis(
    observations: list[dict[str, Any]],
) -> str | None:
    """Honor an independent LLM decision that another blind retry is futile.

    The advisor still cannot mutate or approve geometry.  This only prevents
    another planner call when the advisor explicitly selected a terminal
    disposition from the typed response schema; the last validator rejection
    remains the run's blocking issue and evidence authority.
    """

    for observation in reversed(observations):
        if observation.get("context_type") == "geometry_validation_advisor_unavailable":
            # Includes legacy checkpoints that recorded terminal=true. Provider,
            # schema, host-coercion, or capacity failure by the optional advisor
            # cannot overrule a remaining bounded planner repair attempt.
            continue
        if observation.get("context_type") != "step_geometry_diagnosis":
            continue
        disposition = observation.get("disposition")
        if disposition not in {"stop_contract_infeasible", "stop_futile_retry"}:
            return None
        instruction = str(observation.get("planner_instruction") or "").strip()
        diagnosis_class = str(observation.get("diagnosis_class") or "unknown")
        return (
            "The independent inverse-geometry advisor classified the rejected "
            f"transition as {diagnosis_class} and selected {disposition}; "
            "additional blind LLM action retries were stopped."
            + (f" Advisor conclusion: {instruction}" if instruction else "")
        )
    return None


def _planner_stagnation_run(
    attempts: list[ActionAttempt],
) -> tuple[int, dict[str, Any] | None]:
    """Count consecutive rejections with the same validator-level failure.

    Parameters are intentionally excluded: repeatedly perturbing numbers while
    hitting the same phase/code on the same goal is exactly the stagnation this
    detector must notice.
    """

    if not attempts:
        return 0, None

    def signature(attempt: ActionAttempt) -> dict[str, Any]:
        action = attempt.resolved or attempt.draft or {}
        return {
            "phase": attempt.phase,
            "issue_codes": sorted(attempt.issue_codes),
            "module": action.get("module") if isinstance(action, dict) else None,
            "target_port": (
                action.get("target_port") if isinstance(action, dict) else None
            ),
            "affected_goal_ids": sorted(
                action.get("affected_goal_ids") or []
                if isinstance(action, dict)
                else []
            ),
            "detail_identity": _attempt_failure_detail_identity(attempt),
        }

    repeated_failure = signature(attempts[-1])
    repeat_count = 0
    for attempt in reversed(attempts):
        if signature(attempt) != repeated_failure:
            break
        repeat_count += 1
    return repeat_count, repeated_failure


def _planner_exact_stagnation_run(
    attempts: list[ActionAttempt],
) -> tuple[int, dict[str, Any] | None]:
    """동일 LLM draft와 동일 검증 증거가 연속된 횟수만 계산한다."""

    if not attempts:
        return 0, None

    def signature(attempt: ActionAttempt) -> dict[str, Any]:
        action = attempt.draft or attempt.resolved or {}
        material_action = {
            key: action.get(key)
            for key in (
                "target_port",
                "module",
                "params",
                "affected_goal_ids",
                "completed_goal_ids",
            )
            if isinstance(action, dict) and key in action
        }
        evidence_payload = [
            {
                "issue_code": observation.get("issue_code"),
                "check_name": observation.get("check_name"),
                "expected": observation.get("expected"),
                "actual": observation.get("actual"),
            }
            for observation in attempt.observations
            if isinstance(observation, dict)
        ]
        canonical = json.dumps(
            _canonical_stagnation_value(
                {"action": material_action, "evidence": evidence_payload}
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return {
            "phase": attempt.phase,
            "issue_codes": sorted(attempt.issue_codes),
            "action_evidence_sha256": hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest(),
        }

    repeated_failure = signature(attempts[-1])
    repeat_count = 0
    for attempt in reversed(attempts):
        if signature(attempt) != repeated_failure:
            break
        repeat_count += 1
    return repeat_count, repeated_failure


def _canonical_stagnation_value(value: Any) -> Any:
    """Remove sub-measurement float jitter without erasing real progress."""

    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or value == 0.0:
            return value
        digits = max(0, 9 - int(math.floor(math.log10(abs(value)))) - 1)
        return round(value, digits)
    if isinstance(value, dict):
        return {
            str(key): _canonical_stagnation_value(child) for key, child in value.items()
        }
    if isinstance(value, list):
        return [_canonical_stagnation_value(child) for child in value]
    return value


def _attempt_failure_detail_identity(attempt: ActionAttempt) -> list[str]:
    """Return stable validator details while ignoring incidental numeric values."""

    details: list[str] = []
    for observation in attempt.observations:
        actual = observation.get("actual")
        if not isinstance(actual, dict):
            continue
        candidates: list[Any] = []
        raw_errors = actual.get("errors")
        if isinstance(raw_errors, list):
            candidates.extend(raw_errors)
        for key in ("error", "diagnostic"):
            if actual.get(key) is not None:
                candidates.append(actual[key])
        for candidate in candidates:
            normalized = str(candidate).strip().lower()
            normalized = _ANY_NUMERIC_VALUE.sub("#", normalized)
            normalized = re.sub(r"\s+", " ", normalized)[:500]
            if normalized:
                details.append(normalized)
    return sorted(set(details))


def _stagnation_schema_strategy(repeated_failure: dict[str, Any]) -> str:
    """Escalate numeric grammar only for precision/analytic-value failures."""

    detail = " ".join(repeated_failure.get("detail_identity") or [])
    exact_value_markers = (
        "analytic",
        "precision",
        "representable",
        "rim_error",
        "tangent",
        "exact decimal",
        "numeric literal",
    )
    return (
        "encoded"
        if any(marker in detail for marker in exact_value_markers)
        else "unchanged"
    )


def _issue(
    step_index: int,
    code: str,
    message: str,
    *,
    phase: str,
    action_id: str | None = None,
    target_port: str | None = None,
    module_id: str | None = None,
    port_ids: list[str] | None = None,
    expected: dict[str, Any] | None = None,
    actual: dict[str, Any] | None = None,
    suggestion: dict[str, Any] | None = None,
) -> StaticIssue:
    prefix = f"STEP_{step_index:04d}" if step_index > 0 else "FINAL"
    return StaticIssue(
        issue_id=f"{prefix}_01_{code}",
        severity="error",
        issue_code=code,
        check_name=phase,
        message=message,
        step_index=step_index if step_index > 0 else None,
        action_id=action_id,
        module_id=module_id,
        port_ids=port_ids or [],
        target_port_id=target_port,
        expected=expected or {},
        actual=actual or {},
        suggestion=suggestion or {},
    )


def _critic_with_issue(
    critic: CriticReport | None,
    issue: StaticIssue,
) -> CriticReport:
    if critic is None:
        issues = [issue]
        return CriticReport(
            passed=False,
            verification_status="failed",
            error_count=1,
            actual_open_ports=0,
            issues=issues,
            next_actions=[f"Fix {issue.issue_code}: {issue.message}"],
        )
    issues = [
        issue,
        *[item for item in critic.issues if item.issue_id != issue.issue_id],
    ]
    return critic.model_copy(
        update={
            "passed": False,
            "verification_status": "failed",
            "issues": issues,
            "error_count": error_count(issues),
            "warning_count": warning_count(issues),
            "next_actions": [
                f"Fix {issue.issue_code}: {issue.message}",
                *critic.next_actions,
            ],
        }
    )


def _attach_view_evidence(
    critic: CriticReport,
    paths: list[str],
) -> CriticReport:
    path_by_view = {Path(path).stem: path for path in paths}
    views = []
    for request in critic.view_requests:
        path = path_by_view.get(request.view_id)
        if path:
            views.append(
                request.model_copy(
                    update={
                        "evidence_status": "available",
                        "evidence_path": path,
                        "unavailable_reason": None,
                    }
                )
            )
        else:
            views.append(
                request.model_copy(
                    update={
                        "evidence_status": "unavailable",
                        "unavailable_reason": "The requested MCP view was not captured.",
                    }
                )
            )
    return critic.model_copy(update={"view_requests": views})


def _fail_run(
    run_id: str,
    paths: dict[str, Path],
    artifacts: GenerationArtifacts,
    dry_run: bool,
    freecad_opened: bool,
    mcp_used: bool,
    mcp_error: str | None,
    actions: list[dict[str, Any]],
    attempts: list[ActionAttempt],
    state: PipeState | None,
    step_verifications: list[StepVerification],
    critic: CriticReport,
    failed_stage: str,
    summary: str,
    gemini: Any,
    *,
    pause: bool = False,
) -> None:
    _write_progress(paths, actions, attempts, state, step_verifications, critic)
    report = _make_report(
        run_id,
        dry_run,
        freecad_opened,
        mcp_used,
        mcp_error,
        artifacts,
        step_verifications,
        critic,
        status="paused" if pause else "failed",
        verification_status="failed",
        failed_stage=failed_stage,
        skipped_mcp_reason=critic.skipped_mcp_reason,
        summary=summary,
        gemini=gemini,
        repair_attempt_count=sum(1 for item in attempts if item.status == "rejected"),
    )
    report = report.model_copy(
        update={
            "pause_reason": failed_stage if pause else None,
            "resume_command": (
                f"./run.sh --resume {artifacts.output_dir}" if pause else None
            ),
            "recovery_state": (
                {
                    "checkpoint_path": artifacts.checkpoint_path,
                    "failed_stage": failed_stage,
                    "next_action": "resume bounded conflict search from checkpoint",
                }
                if pause
                else {}
            ),
            "artifact_statuses": _artifact_statuses(
                artifacts,
                failed_stage=failed_stage,
                issues=critic.issues,
            ),
        }
    )
    _atomic_write_json(paths["report"], report.model_dump(mode="json"))
    if pause:
        resume_command = f"./run.sh --resume {artifacts.output_dir}"
        append_search_event(
            paths["search_events"],
            {
                "event_type": "run_paused",
                "run_id": run_id,
                "reason": failed_stage,
                "checkpoint_path": artifacts.checkpoint_path,
                "resume_command": resume_command,
            },
        )
        raise PipelinePausedError(
            failed_stage,
            str(paths["report"]),
            critic.issues,
            resume_command,
        )
    error_type = (
        CriticValidationError
        if failed_stage in {"final_critic", "max_iter"}
        else StaticValidationError
    )
    raise error_type(failed_stage, str(paths["report"]), critic.issues)


def _make_report(
    run_id: str,
    dry_run: bool,
    freecad_opened: bool,
    mcp_used: bool,
    mcp_error: str | None,
    artifacts: GenerationArtifacts,
    step_verifications: list[StepVerification],
    critic: CriticReport | None,
    *,
    status: str,
    verification_status: str,
    summary: str,
    failed_stage: str | None = None,
    skipped_mcp_reason: str | None = None,
    gemini: Any = None,
    repair_attempt_count: int = 0,
) -> RunReport:
    issues = (
        critic.issues
        if critic
        else [issue for step in step_verifications for issue in step.issues]
    )
    usage = LLMUsage()
    if gemini is not None and hasattr(gemini, "usage_snapshot"):
        usage = gemini.usage_snapshot()
    llm_policy: dict[str, Any] = {}
    if gemini is not None and hasattr(gemini, "policy_snapshot"):
        llm_policy = dict(gemini.policy_snapshot())
    action_budget_policy = (
        getattr(gemini, "_cadgen_action_budget_policy", None)
        if gemini is not None
        else None
    )
    if isinstance(action_budget_policy, dict):
        llm_policy["action_budget"] = copy.deepcopy(action_budget_policy)
    (
        intent_attempt_count,
        intent_repair_count,
        intent_protocol_retry_count,
    ) = _intent_attempt_stats(artifacts)
    (
        intent_advisor_call_count,
        intent_advisor_success_count,
        intent_advisor_failure_count,
        intent_futile_retry_avoided_count,
    ) = _intent_diagnostic_stats(artifacts)
    diagnostic_journal, advisor_artifact_paths = _diagnostic_report_stats(artifacts)
    advisor_call_count = sum(diagnostic_journal.calls_by_step.values())
    advisor_success_count = sum(
        record.status == "complete" for record in diagnostic_journal.records
    )
    advisor_failure_count = sum(
        record.status == "failed" for record in diagnostic_journal.records
    )
    artifacts = artifacts.model_copy(
        update={"advisor_artifact_paths": advisor_artifact_paths}
    )
    realization_status = "not_run"
    deviation_count = 0
    if artifacts.global_preflight_path:
        try:
            preflight_payload = json.loads(
                Path(artifacts.global_preflight_path).read_text(encoding="utf-8")
            )
            if isinstance(preflight_payload, dict):
                realization_status = str(preflight_payload.get("status") or "not_run")
                deviations = preflight_payload.get("deviations")
                deviation_count = len(deviations) if isinstance(deviations, list) else 0
        except (OSError, json.JSONDecodeError):
            realization_status = "not_run"
    if status == "success" and realization_status == "adjusted":
        status = "success_with_deviations"
    return RunReport(
        run_id=run_id,
        status=status,  # type: ignore[arg-type]
        realization_status=realization_status,  # type: ignore[arg-type]
        deviation_count=deviation_count,
        failed_stage=failed_stage,
        dry_run=dry_run,
        freecad_opened=freecad_opened,
        freecad_mcp_used=mcp_used,
        freecad_mcp_error=mcp_error,
        verification_status=verification_status,  # type: ignore[arg-type]
        static_error_count=error_count(issues),
        static_warning_count=warning_count(issues),
        critic_passed=critic.passed if critic else None,
        skipped_mcp_reason=skipped_mcp_reason,
        top_issues=top_issue_ids(issues, limit=1),
        llm_usage=usage,
        llm_policy=llm_policy,
        intent_attempt_count=intent_attempt_count,
        intent_repair_count=intent_repair_count,
        intent_protocol_retry_count=intent_protocol_retry_count,
        intent_advisor_call_count=intent_advisor_call_count,
        intent_advisor_success_count=intent_advisor_success_count,
        intent_advisor_failure_count=intent_advisor_failure_count,
        intent_futile_retry_avoided_count=(intent_futile_retry_avoided_count),
        action_repair_count=repair_attempt_count,
        repair_attempt_count=repair_attempt_count,
        step_repair_advisor_count=(
            advisor_call_count or _repair_advice_count(artifacts)
        ),
        advisor_call_count=advisor_call_count,
        advisor_success_count=advisor_success_count,
        advisor_failure_count=advisor_failure_count,
        advisor_cache_hit_count=diagnostic_journal.cache_hit_count,
        advisor_artifact_paths=advisor_artifact_paths,
        futile_retry_avoided_count=(diagnostic_journal.futile_retry_avoided_count),
        artifacts=artifacts,
        artifact_statuses=_artifact_statuses(
            artifacts,
            failed_stage=failed_stage,
            issues=issues,
        ),
        summary=summary,
    )


def _intent_attempt_stats(artifacts: GenerationArtifacts) -> tuple[int, int, int]:
    """Intent journal에서 실제 호출 수와 의미 교정 횟수를 계산한다."""

    if not artifacts.intent_attempts_path:
        return 0, 0, 0
    try:
        payload = json.loads(
            Path(artifacts.intent_attempts_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return 0, 0, 0
    if not isinstance(payload, list):
        return 0, 0, 0
    attempts = [item for item in payload if isinstance(item, dict)]
    semantic_calls = sum(
        item.get("consumes_semantic_budget") is True for item in attempts
    )
    protocol_retries = sum(
        item.get("phase") in {"structured_output", "provider_schema_negotiation"}
        and item.get("will_retry", item.get("status") == "schema_retry") is True
        for item in attempts
    )
    return len(attempts), max(0, semantic_calls - 1), protocol_retries


def _intent_diagnostic_stats(
    artifacts: GenerationArtifacts,
) -> tuple[int, int, int, int]:
    """Summarize durable intent-advisor calls and avoided futile retries."""

    if not artifacts.intent_diagnostics_path:
        return 0, 0, 0, 0
    try:
        payload = json.loads(
            Path(artifacts.intent_diagnostics_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return 0, 0, 0, 0
    if not isinstance(payload, list):
        return 0, 0, 0, 0
    records = [item for item in payload if isinstance(item, dict)]
    calls = sum(
        max(0, int(item.get("advisor_protocol_attempts", 0) or 0)) for item in records
    )
    successes = sum(item.get("advisor_status") == "complete" for item in records)
    failures = sum(
        item.get("advisor_status") in {"failed", "degraded_fallback"}
        for item in records
    )
    avoided = sum(
        item.get("terminal_reason")
        in {"stop_futile_retry", "identical_intent_failure_stagnation"}
        for item in records
    )
    return calls, successes, failures, avoided


def _repair_advice_count(artifacts: GenerationArtifacts) -> int:
    if not artifacts.repair_advice_path:
        return 0
    try:
        payload = json.loads(
            Path(artifacts.repair_advice_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return 0
    return len(payload) if isinstance(payload, list) else 0


def _diagnostic_report_stats(
    artifacts: GenerationArtifacts,
) -> tuple[DiagnosticJournal, list[str]]:
    payload: Any = None
    candidates = [
        artifacts.diagnostics_index_path,
        artifacts.checkpoint_path,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            loaded = json.loads(Path(candidate).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if candidate == artifacts.checkpoint_path and isinstance(loaded, dict):
            loaded = loaded.get("diagnostic_journal")
        if isinstance(loaded, dict):
            payload = loaded
            break
    try:
        journal = DiagnosticJournal.model_validate(payload or {})
    except ValueError:
        journal = DiagnosticJournal()
    diagnostic_dir = Path(artifacts.output_dir) / "diagnostics"
    artifact_paths = sorted(
        str(path) for path in diagnostic_dir.glob("*.json") if path.name != "index.json"
    )
    return journal, artifact_paths


def _clear_state_bound_evidence(
    artifacts: GenerationArtifacts,
) -> GenerationArtifacts:
    """Drop evidence pointers whenever their exact CAD state is no longer current."""

    return artifacts.model_copy(
        update={
            "mcp_result_path": None,
            "freecad_validation_path": None,
            "freecad_document_path": None,
            "visual_evidence_paths": [],
        }
    )


def _should_run_step_mcp(settings: Settings, dry_run: bool) -> bool:
    return (
        not dry_run
        and settings.freecad_mcp_enabled
        and settings.freecad_step_mcp_enabled
    )


def _requires_progress_mcp(
    settings: Settings,
    dry_run: bool,
    state: PipeState,
    action: ResolvedAction,
) -> bool:
    if dry_run or not settings.freecad_mcp_enabled:
        return False
    if action.params.get("path_kind") != "spline":
        return False
    completed = set(action.completed_goal_ids)
    for goal in state.remaining_goals:
        if (
            goal.goal_id in set(action.affected_goal_ids)
            and goal.goal_id not in completed
            and goal.length is not None
        ):
            return True
    return False


def _requires_risk_mcp(
    settings: Settings,
    dry_run: bool,
    action: ResolvedAction,
    step: StepVerification,
) -> bool:
    """Use B-Rep evidence early for high-risk geometry even in adaptive mode."""

    if dry_run or not settings.freecad_mcp_enabled:
        return False
    if action.module in {
        "transition",
        "junction",
        "connect_ports",
        "terminate",
        "inline_component",
    }:
        return True
    if action.module == "route" and action.params.get("path_kind") == "spline":
        return True
    return any(
        issue.issue_code == "STATIC_COLLISION_REQUIRES_FREECAD" for issue in step.issues
    )


def _step_mcp_skip_reason(settings: Settings, dry_run: bool) -> str | None:
    if dry_run:
        return "Dry-run skips step FreeCAD MCP."
    if not settings.freecad_mcp_enabled:
        return "FreeCAD MCP is disabled."
    if not settings.freecad_step_mcp_enabled:
        return "Step FreeCAD MCP is disabled."
    return None


def _final_mcp_skip_reason(settings: Settings, dry_run: bool) -> str | None:
    if dry_run:
        return "Dry-run skips final FreeCAD MCP."
    if not settings.freecad_mcp_enabled:
        return "FreeCAD MCP is disabled."
    return None
