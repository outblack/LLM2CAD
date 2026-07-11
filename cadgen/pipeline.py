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
    build_freecad_candidate_cleanup_script,
    build_freecad_publish_script,
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
    MAX_STRUCTURED_NUMBER_LITERAL_BYTES,
    MAX_STRUCTURED_NUMBER_LITERALS,
    StructuredOutputError,
    StructuredOutputIncompleteError,
)
from cadgen.local_heuristic import infer_intent, plan_next_action
from cadgen.prompts import (
    compact_planner_payload,
    compact_visual_module_map,
    final_repair_prompt,
    intent_prompt,
    intent_system_instruction,
    step_lineage_repair_prompt,
    step_planner_prompt,
    step_planner_system_instruction,
    realized_terminal_topology,
)
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
    AssemblyBounds,
    CriticReport,
    CorePlannerDecision,
    GenerationArtifacts,
    IntentResult,
    LLMProductionIntent,
    LLMUsage,
    PipeState,
    PlannerDecision,
    ProductionIntent,
    ResolvedAction,
    RunReport,
    StaticIssue,
    StepVerification,
    VisualCriticResult,
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
    canonical_circular_arc_frame,
    direction_to_vector,
    dot,
    length,
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


# The 96-item/512-byte limits are hard guards for authored mandatory values.
# Optional construction conveniences stay well below them so provider grammar
# compilation has headroom for value spelling and surrounding schema structure.
PLANNER_PREFERRED_NUMBER_LITERALS = 64
PLANNER_PREFERRED_NUMBER_LITERAL_BYTES = 384
_PLANNER_SCHEMA_PROFILES = {"preferred", "mandatory", "encoded"}
_PLANNER_SCHEMA_PROFILE_ATTR = "_cadgen_step_planner_schema_profiles"
MAX_IDENTICAL_VALIDATOR_FAILURES = 6


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
    if resume_dir is None and not prompt.strip():
        raise ValueError("Prompt must contain a non-whitespace pipe design request")
    if resume_dir is not None:
        run_dir = Path(resume_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise ValueError(f"Resume run directory does not exist: {run_dir}")
        prompt_path = run_dir / "prompt.txt"
        if not prompt_path.is_file():
            raise ValueError(f"Resume prompt artifact is missing: {prompt_path}")
        prompt = prompt_path.read_text(encoding="utf-8")
        run_id = run_dir.name
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = settings.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(run_dir)
    if resume_dir is None:
        _atomic_write_text(paths["prompt"], prompt)
        _atomic_write_json(paths["intent_attempts"], [])

    artifacts = GenerationArtifacts(
        run_id=run_id,
        output_dir=str(run_dir),
        prompt_path=str(paths["prompt"]),
        intent_path=str(paths["intent"]),
        intent_attempts_path=str(paths["intent_attempts"]),
        actions_path=str(paths["actions"]),
        state_path=str(paths["state"]),
        freecad_script_path=str(paths["script"]),
        report_path=str(paths["report"]),
        step_verification_path=str(paths["steps"]),
        critic_report_path=str(paths["critic"]),
        mcp_result_path=None,
        action_attempts_path=str(paths["attempts"]),
        checkpoint_path=str(paths["checkpoint"]),
        freecad_validation_path=None,
        freecad_document_path=None,
    )

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
    pending_draft: ActionDraft | None = None
    pending_draft_attempt_index: int | None = None
    preserved_suffix: _PreservedSuffix | None = None
    next_attempt_index = 1
    intent_attempts: list[dict[str, Any]] = []
    if paths["intent_attempts"].is_file():
        try:
            loaded_intent_attempts = json.loads(
                paths["intent_attempts"].read_text(encoding="utf-8")
            )
            if isinstance(loaded_intent_attempts, list):
                intent_attempts = [
                    dict(item) for item in loaded_intent_attempts if isinstance(item, dict)
                ]
        except (OSError, json.JSONDecodeError):
            intent_attempts = []

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
            )
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

        nonbinary_branches = _nonbinary_branch_goal_ids(intent)
        if not dry_run and nonbinary_branches:
            issue = _issue(
                0,
                "NON_BINARY_BRANCH_CONTRACT",
                "Production branch goals must each describe one binary junction.",
                phase="intent_scope",
                expected={"total_outlets_per_branch_goal": 2},
                actual={"nonbinary_branch_goal_ids": nonbinary_branches},
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
                "intent_scope",
                "Stopped before planning because an atomic branch goal is not binary.",
                gemini,
            )
            raise AssertionError("unreachable")

        unsupported_components = _unsupported_required_components(intent)
        if unsupported_components:
            issue = _issue(
                0,
                "UNSUPPORTED_REQUIRED_COMPONENT",
                "The LLM intent contains an accessory outside the explicit geometry catalog.",
                phase="intent_scope",
                expected={"supported_components": list(SUPPORTED_INLINE_COMPONENTS)},
                actual={"unsupported_components": unsupported_components},
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
                "intent_scope",
                "Stopped before planning because a required accessory has no geometry primitive.",
                gemini,
            )
            raise AssertionError("unreachable")

        component_contract_error = _component_contract_error(intent)
        if component_contract_error is not None:
            issue = _issue(
                0,
                "INCONSISTENT_COMPONENT_CONTRACT",
                "Required accessory multiplicity must match distinct connector goals.",
                phase="intent_scope",
                expected=component_contract_error["expected"],
                actual=component_contract_error["actual"],
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
                "intent_scope",
                "Stopped before planning because the LLM component contract is inconsistent.",
                gemini,
            )
            raise AssertionError("unreachable")

        if intent.hard_constraints:
            issue = _issue(
                0,
                "UNSUPPORTED_HARD_CONSTRAINT",
                "The LLM preserved a hard constraint that has no deterministic predicate.",
                phase="intent_scope",
                expected={"hard_constraints": []},
                actual={"hard_constraints": intent.hard_constraints},
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
                "intent_scope",
                "Stopped before planning because a hard constraint is not measurable.",
                gemini,
            )
            raise AssertionError("unreachable")

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
        )

    final_repair_round = 0
    while True:
        while state.remaining_goals:
            atomic_lower_bound = _exclusive_goal_action_lower_bound(state)
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
                )

            max_attempt_index = settings.step_repair_attempts + 1
            for attempt_index in range(
                next_attempt_index,
                max_attempt_index + 1,
            ):
                draft: ActionDraft | None = None
                resolved: ResolvedAction | None = None
                speculative: PipeState | None = None
                planner_repair_context: list[dict[str, Any]] = []
                if pending_draft is None:
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
                if observations:
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
                                if isinstance(exc, StructuredOutputError)
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
                    last_issue = _issue(
                        step_index,
                        "REGISTRY_VALIDATION_FAILED",
                        "Resolved action failed registry validation.",
                        phase=last_phase,
                        target_port=resolved.target_port,
                        action_id=resolved.action_id,
                        actual={"errors": action_check.errors},
                        suggestion={
                            "operation": "revise_resolved_geometry_inputs",
                            "parameter_errors": action_check.errors[:12],
                            "instruction": (
                                "Use the reported bounds/invariants to select new "
                                "LLM-authored values; the system will not patch them."
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

                        raw, evidence, publish_raw = _validate_and_publish_freecad(
                            settings,
                            speculative,
                            run_id=run_id,
                            attempt_id=attempt_index,
                            raw_result_path=run_dir
                            / "step_mcp"
                            / f"step_{step_index}_attempt_{attempt_index}.json",
                            validation_path=run_dir
                            / "step_mcp"
                            / f"step_{step_index}_validation_{attempt_index}.json",
                            evidence_validator=validate_candidate_evidence,
                        )
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
                                "mcp_result_path": str(
                                    run_dir
                                    / "step_mcp"
                                    / f"step_{step_index}_attempt_{attempt_index}.json"
                                ),
                                "freecad_validation_path": str(
                                    run_dir
                                    / "step_mcp"
                                    / f"step_{step_index}_validation_{attempt_index}.json"
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
                            )
                            last_issue = _issue(
                                step_index,
                                "FREECAD_GEOMETRY_VALIDATION_FAILED",
                                "FreeCAD rejected the speculative geometry.",
                                phase=last_phase,
                                action_id=resolved.action_id,
                                target_port=resolved.target_port,
                                module_id=(
                                    evidence_summary["module_ids"][0]
                                    if evidence_summary["module_ids"]
                                    else None
                                ),
                                expected=repair_expected,
                                actual={
                                    "error": str(exc),
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
                break

            if accepted is None:
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
                failed_steps = [
                    module_steps[module_id]
                    for module_id in evidence_summary["module_ids"]
                    if module_id in module_steps
                ]
                target_step = min(failed_steps) if failed_steps else len(actions)
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
                        module_id=(
                            evidence_summary["module_ids"][0]
                            if evidence_summary["module_ids"]
                            else None
                        ),
                        expected=repair_expected,
                        actual={
                            "error": str(exc),
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

        if (
            critic.passed
            and not dry_run
            and settings.visual_validation_mode == "final_required"
            and visual_reviewed_digest != geometry_payload_digest(state)
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
                        payload_digest=geometry_payload_digest(state),
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
                    visual_reviewed_digest = geometry_payload_digest(state)
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
                        AgendaRepairDirective,
                        part="patch",
                        thinking_level="high",
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
                    payload_digest=geometry_payload_digest(state),
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
                    _freecad_document_path(current_mcp_result_path, state)
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
    )
    verified = semantic_mcp_passed and (
        settings.visual_validation_mode != "final_required"
        or visual_reviewed_digest == geometry_payload_digest(state)
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
) -> IntentResult:
    """사용자 요청을 불변 설계 계약으로 변환하고 의미 검증까지 완료한다.

    프로덕션에서는 Gemini가 반환한 구조화 객체만 허용한다. 공급자가 숫자
    enum 스키마 자체를 거절한 경우에는 모델 초안이 생성되지 않았으므로,
    정확한 decimal-object 스키마로 바꾸되 의미 교정 횟수는 소비하지 않는다.
    불완전/형식 오류는 별도의 작은 structured retry 예산을 쓰고, 완전한
    Intent가 의미 검증에 실패한 경우에만 의미 교정 예산을 쓴다.
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

    if dry_run:
        result = infer_intent(prompt, settings)
        result = _canonicalize_dependent_intent_geometry(result, settings)
        _validate_intent_safety(prompt, result, settings)
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
    numeric_literals = _intent_numeric_literals(prompt, settings)
    encoded_numeric_schema = not _numeric_literal_schema_fits(numeric_literals)
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
    if encoded_numeric_schema:
        base_prompt = _encoded_intent_request(base_prompt, numeric_literals)
    request = base_prompt
    last_error: Exception | None = None
    diagnostic_history: list[str] = []
    intent_thinking_level = "low"
    last_semantic_diagnostic: str | None = None
    semantic_diagnostic_repeat_count = 0
    # 스키마 프로토콜 재협상과 LLM 의미 교정은 서로 다른 실패 종류다.
    # 별도의 카운터를 사용해야 교정 예산이 0인 경우에도 enum 거절 뒤
    # encoded-decimal 요청을 정확히 한 번 실행할 수 있다.
    semantic_attempt = 0
    structured_retry_attempt = 0
    # Incomplete/malformed structured output is not a correction of a parsed
    # design. Give that transport/schema class its own small bounded retry pool
    # so one truncated response cannot consume the remaining semantic repairs.
    max_structured_retries = 2
    while True:
        parsed_intent: IntentResult | None = None
        try:
            result = _call_structured(
                gemini,
                request,
                LLMProductionIntent,
                part="intent",
                thinking_level=intent_thinking_level,
                numeric_literals=None if encoded_numeric_schema else numeric_literals,
                system_instruction=intent_system_instruction(),
            )
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
            parsed_intent = intent
            intent = _canonicalize_dependent_intent_geometry(intent, settings)
            _validate_intent_safety(prompt, intent, settings)
            record_attempt(
                {
                    "status": "accepted",
                    "phase": "semantic_validation",
                    "semantic_attempt": semantic_attempt + 1,
                    "consumes_semantic_budget": True,
                    "parsed_intent": True,
                    "candidate_digest": _intent_candidate_digest(intent),
                    "diagnostic": None,
                    "lineage_reset": False,
                }
            )
            return intent
        except GeminiInvalidRequestError as exc:
            # A provider-side grammar rejection happened before any model draft
            # existed. Switch once from the finite enum to the exact bounded
            # decimal representation instead of spending semantic repair turns
            # on the identical un-compilable schema.
            if encoded_numeric_schema or not _is_invalid_planner_request(exc):
                raise
            encoded_numeric_schema = True
            if hasattr(gemini, "reset_lineage"):
                gemini.reset_lineage("intent")
            record_attempt(
                {
                    "status": "schema_retry",
                    "phase": "provider_schema_negotiation",
                    "semantic_attempt": semantic_attempt + 1,
                    "structured_retry_attempt": structured_retry_attempt + 1,
                    "consumes_semantic_budget": False,
                    "parsed_intent": False,
                    "candidate_digest": None,
                    "diagnostic": str(exc)[:1200],
                    "lineage_reset": True,
                }
            )
            base_prompt = _encoded_intent_request(base_prompt, numeric_literals)
            request = base_prompt
            continue
        except (
            GeminiLineageError,
            StructuredOutputError,
            TypeError,
            ValueError,
        ) as exc:
            last_error = exc
            diagnostic = _intent_repair_diagnostic(exc)
            semantic_repair = parsed_intent is not None
            if semantic_repair:
                if diagnostic == last_semantic_diagnostic:
                    semantic_diagnostic_repeat_count += 1
                else:
                    last_semantic_diagnostic = diagnostic
                    semantic_diagnostic_repeat_count = 1
            will_retry = (
                semantic_attempt < settings.intent_repair_attempts
                if semantic_repair
                else structured_retry_attempt < max_structured_retries
            )
            will_reset_lineage = (
                will_retry
                and hasattr(gemini, "reset_lineage")
                and (
                    not semantic_repair
                    or semantic_diagnostic_repeat_count == 2
                )
            )
            record_attempt(
                {
                    "status": "rejected",
                    "phase": (
                        "semantic_validation"
                        if parsed_intent is not None
                        else "structured_output"
                    ),
                    "semantic_attempt": semantic_attempt + 1,
                    "structured_retry_attempt": structured_retry_attempt + 1,
                    "consumes_semantic_budget": semantic_repair,
                    "parsed_intent": parsed_intent is not None,
                    "candidate_digest": (
                        _intent_candidate_digest(parsed_intent)
                        if parsed_intent is not None
                        else None
                    ),
                    "diagnostic": diagnostic,
                    "lineage_reset": will_reset_lineage,
                    "will_retry": will_retry,
                }
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
            if _is_spline_intent_safety_diagnostic(diagnostic):
                intent_thinking_level = "medium"
            if will_reset_lineage:
                # Malformed output has no trustworthy object to edit. A second
                # identical semantic miss is also de-anchored once so the next
                # call sees the complete request instead of repeating one draft.
                gemini.reset_lineage("intent")
            targeted_guidance = _intent_repair_guidance(diagnostic)
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
                + "Validation diagnostic history: "
                + json.dumps(diagnostic_history, ensure_ascii=False)
                + (
                    " Targeted repair rule: " + targeted_guidance
                    if targeted_guidance
                    else ""
                )
            )
    if last_error is not None:
        raise last_error
    raise RuntimeError("Intent extraction failed without a diagnostic")


def _intent_repair_diagnostic(exc: Exception) -> str:
    """Return actionable validation facts without re-anchoring malformed JSON."""

    if isinstance(exc, StructuredOutputIncompleteError):
        return (
            "provider structured generation was incomplete: "
            f"status={exc.status}, output_limit={exc.output_limit}, "
            f"output_tokens={exc.output_tokens}, thought_tokens={exc.thought_tokens}"
        )[:1200]
    if isinstance(exc, StructuredOutputError):
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

            if goal.path_kind == "line" and goal.direction is not None:
                current = direction_to_vector(goal.direction)
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
        if goal.type == "route" and goal.required_waypoints and goal.path_kind != "spline":
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
            if not (
                angle_range.minimum - 1e-6
                <= angle
                <= angle_range.maximum + 1e-6
            )
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

    if issues:
        raise ValueError("; ".join(issues))


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
            if (
                goal.waypoint_frame == "relative_to_target"
                and goal.required_waypoints
            ):
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
                    if goal.terminal_axis is not None and len(goal.required_waypoints) >= 2:
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
                        and natural_radius + settings.modeling_tolerance >= required_radius
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
                        "direction change is spread over more distance"
                        + natural_hint
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
        if re.search(r"\b(?:each|every|all)\s+branch(?:es)?\b", context, re.IGNORECASE) and re.search(
            r"\b(?:long|lengths?)\b", context,
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
    first_branch_index = branch_indexes[0] if branch_indexes else len(intent.target_behavior)
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


def _encoded_intent_request(request: str, source_literals: list[str]) -> str:
    bounded_examples = source_literals[:32]
    return (
        request + "\n\nThe finite numeric-enum grammar is too large for this request. "
        "Use the bounded decimal-object representation required by the response "
        "schema: encode each value as c*10^(-p), for example 1.5 as "
        '{"k":"d","c":15,"p":1}. Preserve all user-authored values exactly. '
        "Representative source/default values: "
        + json.dumps(
            bounded_examples,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def _planner_numeric_literals(
    state: PipeState,
    *,
    include_optional: bool = True,
) -> list[str]:
    """Build a finite action vocabulary from the immutable contract and ``S_t``."""

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

    # Conservatively preserve every dependency-ready goal, not only the first
    # pending goal. A later sequential goal can sometimes share an action that
    # completes earlier compatible goals, while allow_parallel goals can be
    # selected directly. Restricting this set further would silently make that
    # LLM choice impossible at the schema boundary.
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

    # Derived construction conveniences are secondary.  They may be truncated,
    # but no authored eligible-goal or targetable-port value may be dropped.
    priority_seeds = list(dict.fromkeys([*section_seeds, *eligible_goal_values]))[:12]
    priority_derived: list[float] = []
    for value in priority_seeds:
        for factor in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0):
            priority_derived.append(value * factor)

    # Every value in every eligible goal, immutable geometric constraint, and
    # targetable port plus the basis is mandatory. Secondary historical/derived
    # candidates use only the remaining slots in the provider-safe vocabulary.
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
        return mandatory_literals

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
    return literals


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
    repair_observations: list[dict[str, Any]] | None = None,
    reusable_suffix_context: dict[str, Any] | None = None,
) -> ActionDraft:
    if dry_run:
        return plan_next_action(state)
    if gemini is None:
        raise RuntimeError("Gemini client is required outside dry-run mode.")
    schema_profile_before = _planner_schema_profile(gemini, state.state_id)
    force_geometry_decimals = any(
        goal.type == "route" and goal.path_kind == "spline"
        for goal in state.remaining_goals
    ) or any(
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
            and
            hasattr(gemini, "has_previous")
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
            mandatory_numeric_literals = _planner_numeric_literals(
                state,
                include_optional=False,
            )
            numeric_literals = _planner_numeric_literals(state)
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
        PlannerDecision
        if _needs_inline_component_planner(state)
        else CorePlannerDecision
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
    if isinstance(result, (PlannerDecision, CorePlannerDecision)):
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
        "the current response schema." + grounding
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
    system_instruction: str | None = None,
) -> Any:
    if getattr(gemini, "supports_interaction_controls", False):
        kwargs: dict[str, Any] = {
            "part": part,
            "thinking_level": thinking_level,
        }
        if numeric_literals is not None and getattr(
            gemini, "supports_numeric_literals", False
        ):
            kwargs["numeric_literals"] = numeric_literals
        if system_instruction is not None and getattr(
            gemini, "supports_system_instruction", False
        ):
            kwargs["system_instruction"] = system_instruction
        return gemini.stream_structured(prompt, schema, **kwargs)
    return gemini.stream_structured(prompt, schema, part=part)


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
    localized_steps = [
        issue.step_index
        for issue_id in directive.target_issue_ids
        for issue in [errors_by_id[issue_id]]
        if issue.step_index is not None and issue.step_index > 0
    ] + [module_steps[module_id] for module_id in directive.target_module_ids]
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
            evidence = assess_freecad_validation(
                raw,
                expected_digest=digest,
                expected_state_id=state.state_id,
                expected_module_ids=[module.id for module in state.placed_modules],
                expected_internal_section_module_count=sum(
                    module.type not in {"terminate", "cap_pipe"}
                    for module in state.placed_modules
                ),
                expected_open_port_count=len(state.open_ports),
                expected_anchored_inlet_count=(
                    1
                    if state.placed_modules
                    and state.placed_modules[0].type not in {"terminate", "cap_pipe"}
                    else 0
                ),
                expected_generator_version=GENERATOR_VERSION,
                expected_run_id=run_id,
                expected_state_version=state.state_version,
                expected_attempt_id=attempt_id,
                expected_candidate_document=candidate_document_name(
                    state,
                    run_id=run_id,
                    attempt_id=attempt_id,
                ),
            )
        except FreeCADMCPError as exc:
            if isinstance(
                exc, FreeCADValidationError
            ) or _is_semantic_freecad_validation_error(str(exc)):
                try:
                    cleanup_script = build_freecad_candidate_cleanup_script(
                        state,
                        run_id=run_id,
                        attempt_id=attempt_id,
                    )
                    asyncio.run(execute_freecad_code(settings, cleanup_script))
                except Exception:
                    pass
                raise _FreeCADSemanticError(
                    str(exc),
                    getattr(exc, "evidence", None),
                ) from exc
            raise
        if evidence_validator is not None:
            try:
                evidence_validator(evidence)
            except Exception:
                try:
                    cleanup_script = build_freecad_candidate_cleanup_script(
                        state,
                        run_id=run_id,
                        attempt_id=attempt_id,
                    )
                    asyncio.run(execute_freecad_code(settings, cleanup_script))
                except Exception:
                    pass
                raise
        _atomic_write_json(validation_path, evidence)
        publish_script = build_freecad_publish_script(
            state,
            run_id=run_id,
            attempt_id=attempt_id,
            fcstd_path=str(_freecad_document_path(raw_result_path, state)),
            candidate_shape_fingerprints=evidence["candidate_shape_fingerprints"],
        )
        publish_raw = asyncio.run(execute_freecad_code(settings, publish_script))
        assess_freecad_publish(
            publish_raw,
            expected_digest=digest,
            expected_document=published_document_name(state, run_id=run_id),
            expected_fcstd_path=str(_freecad_document_path(raw_result_path, state)),
        )
        _atomic_write_json(
            _freecad_artifact_manifest_path(raw_result_path),
            {
                "state_id": state.state_id,
                "state_version": state.state_version,
                "payload_digest": digest,
                "fcstd_path": str(_freecad_document_path(raw_result_path, state)),
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
    return summary


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
    if depth >= 4:
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


def _freecad_document_path(raw_result_path: Path, state: PipeState) -> Path:
    parent = raw_result_path.parent
    run_dir = (
        parent.parent
        if parent.name in {"step_mcp", "recovery_mcp", "rollback_mcp"}
        else parent
    )
    digest = geometry_payload_digest(state)
    return (
        (run_dir / f"pipe_v{state.state_version}_{digest[:12]}.FCStd")
        .expanduser()
        .resolve()
    )


def _freecad_artifact_manifest_path(raw_result_path: Path) -> Path:
    parent = raw_result_path.parent
    run_dir = (
        parent.parent
        if parent.name in {"step_mcp", "recovery_mcp", "rollback_mcp"}
        else parent
    )
    return run_dir / "freecad_artifact.json"


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
        VisualCriticResult,
        part="visual_validator",
        thinking_level="medium",
    )
    if not isinstance(result, VisualCriticResult):
        raise TypeError("Visual validator returned the wrong schema")
    if result.state_id != state.state_id or result.payload_digest != digest:
        raise ValueError("Visual critic state/digest mismatch")
    if result.evidence_sha256 != evidence_hashes:
        raise ValueError("Visual critic evidence hash mismatch")
    return result


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
    if payload.get("candidate_digest") != candidate_digest:
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
    if payload.get("candidate_document") != candidate_document_name(
        candidate,
        run_id=expected_run_id,
        attempt_id=attempt_id,
    ):
        raise ValueError("Prepared candidate document name mismatch")
    if payload.get("published_document") != published_document_name(
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
    roll_forward = phase == "PUBLISHED"
    roll_forward_verified = False
    semantic_rejection = False
    semantic_rejection_evidence: dict[str, Any] = {}
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

    if phase == "PUBLISHED":
        evidence = payload.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("Published checkpoint is missing validation evidence")
        try:
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
                expected_module_ids=[module.id for module in candidate.placed_modules],
                expected_internal_section_module_count=sum(
                    module.type not in {"terminate", "cap_pipe"}
                    for module in candidate.placed_modules
                ),
                expected_open_port_count=len(candidate.open_ports),
                expected_anchored_inlet_count=(
                    1
                    if candidate.placed_modules
                    and candidate.placed_modules[0].type
                    not in {"terminate", "cap_pipe"}
                    else 0
                ),
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
        preserved_suffix=preserved_suffix,
        next_attempt_index=attempt_id + 1,
        semantic_mcp_passed=False,
        mcp_used=mcp_used,
        mcp_error=mcp_error,
    )


def _exclusive_goal_action_lower_bound(state: PipeState) -> int:
    """Necessary action count from non-double-counting rules, never a plan."""

    groups: Counter[str] = Counter()
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
    return max(groups.values(), default=0)


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
    raw = json.dumps(
        state.model_dump(mode="json"),
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
) -> None:
    if next_attempt_index < 1:
        raise ValueError("next_attempt_index must be positive")
    if (pending_draft is None) != (pending_draft_attempt_index is None):
        raise ValueError("Pending draft and attempt index must be journaled together")
    if (
        pending_draft_attempt_index is not None
        and pending_draft_attempt_index != next_attempt_index
    ):
        raise ValueError("Pending draft attempt must equal next_attempt_index")
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
            "pending_repair_observations": pending_repair_observations or [],
            "pending_draft": (
                pending_draft.model_dump(mode="json") if pending_draft else None
            ),
            "pending_draft_state_digest": (
                _pipe_state_digest(state) if pending_draft else None
            ),
            "pending_draft_attempt_index": pending_draft_attempt_index,
            "next_attempt_index": next_attempt_index,
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
        phase=phase,
        status=status,  # type: ignore[arg-type]
        draft=draft.model_dump(mode="json") if draft else None,
        resolved=resolved.model_dump(mode="json") if resolved else None,
        issue_codes=[issue.issue_code for issue in issues],
        observations=[_repair_observation(issue) for issue in issues],
    )


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
                "validation_observations": _bounded_diagnostic(
                    attempt.observations
                ),
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
                    + "Re-plan from the full immutable state; change the independent "
                    "geometry, sign, module, or goal interpretation responsible for "
                    "the failure."
                ),
            },
        )
    return result


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
            {"action": material_action, "evidence": evidence_payload},
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
        status="failed",
        verification_status="failed",
        failed_stage=failed_stage,
        skipped_mcp_reason=critic.skipped_mcp_reason,
        summary=summary,
        gemini=gemini,
        repair_attempt_count=sum(1 for item in attempts if item.status == "rejected"),
    )
    report = report.model_copy(
        update={
            "artifact_statuses": _artifact_statuses(
                artifacts,
                failed_stage=failed_stage,
                issues=critic.issues,
            )
        }
    )
    _atomic_write_json(paths["report"], report.model_dump(mode="json"))
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
    (
        intent_attempt_count,
        intent_repair_count,
        intent_protocol_retry_count,
    ) = _intent_attempt_stats(artifacts)
    return RunReport(
        run_id=run_id,
        status=status,  # type: ignore[arg-type]
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
        action_repair_count=repair_attempt_count,
        repair_attempt_count=repair_attempt_count,
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
