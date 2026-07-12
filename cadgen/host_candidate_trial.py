"""PAPR host discrete trial queue: single path for post-reject host recovery.

Algorithm (finite discrete search):
1. Build legal_candidate_menu once from state + reject evidence.
2. Walk unbanned complete candidates (recommended first).
3. On FreeCAD rejects: prefer freecad_* geometric lattice only (exclusive phase).
4. Fall back to compile_next_action when no menu row works (non-FreeCAD only).
5. Never invent continuous parameters — only catalog-legal host rows.

Pipeline / thin planner / advisor gates all consume this module instead of
rebuilding menus with divergent rules.

FreeCAD exclusive exhaust (hard invariant):
  FREECAD reject → ban fingerprint → next freecad_* host trial (LLM 0)
  freecad lattice empty → FREECAD_HOST_MENU_EXHAUSTED (never fat planner / Gemini 400)
"""

from __future__ import annotations

from typing import Any, Literal

from cadgen.intent_action_compiler import _junction_style_params, compile_next_action
from cadgen.legal_candidate_menu import (
    action_draft_fingerprint,
    build_legal_candidate_menu,
    iter_menu_drafts,
    menu_has_complete_candidates,
)
from cadgen.typed_data_models import ActionDraft, Goal, PipeState

Authorship = Literal[
    "host_compile",
    "host_menu",
    "style_repair",
    "llm_menu",
    "llm_planner",
]

HOST_OWNED_AUTHORSHIP = frozenset(
    {
        "host_compile",
        "host_menu",
        "style_repair",
        "llm_menu",  # discrete pick; params still host-owned from menu
    }
)

_FREECAD_MENU_SOURCES = frozenset(
    {
        "freecad_junction_repair",
        "freecad_connect_repair",
        "freecad_route_repair",
        "freecad_inline_repair",
    }
)

FREECAD_HOST_MENU_EXHAUSTED = "FREECAD_HOST_MENU_EXHAUSTED"


class FreeCADHostMenuExhausted(RuntimeError):
    """Raised when FreeCAD host discrete lattice is empty; fat planner forbidden."""

    def __init__(self, detail: str = "") -> None:
        message = FREECAD_HOST_MENU_EXHAUSTED
        if detail:
            message = f"{FREECAD_HOST_MENU_EXHAUSTED}: {detail}"
        super().__init__(message)
        self.detail = detail


def draft_fingerprint(draft: ActionDraft) -> str:
    """Stable fingerprint for host thrash ban."""

    return action_draft_fingerprint(
        target_port=draft.target_port,
        module=draft.module,
        params=dict(draft.params or {}),
        affected_goal_ids=list(draft.affected_goal_ids),
        completed_goal_ids=list(draft.completed_goal_ids),
    )


def junction_l0_issue_tags(params: dict[str, Any] | None) -> list[str]:
    """L0 analytic junction feasibility tags (ranking only, never hard-block).

    Industrial-ish heuristics: arm length ≳ 0.75×OD, outer blend ≤ 0.35×min OD.
    """

    data = dict(params or {})
    tags: list[str] = []
    od = float(data.get("outer_diameter") or 20.0)
    outlets = [item for item in (data.get("outlets") or []) if isinstance(item, dict)]
    for index, outlet in enumerate(outlets):
        length_val = outlet.get("length")
        if length_val is None:
            continue
        try:
            length_f = float(length_val)
        except (TypeError, ValueError):
            continue
        out_od = float(outlet.get("outer_diameter") or od)
        if length_f + 1e-9 < 0.75 * out_od:
            tags.append(f"short_arm_{index}")
    blend = data.get("blend_radius")
    if blend is not None:
        try:
            blend_f = float(blend)
            min_od = min(
                [od]
                + [
                    float(item.get("outer_diameter") or od)
                    for item in outlets
                    if item.get("outer_diameter") is not None
                ]
            )
            if blend_f > 0.35 * min_od + 1e-9:
                tags.append("blend_cap")
        except (TypeError, ValueError):
            pass
    outer_b = data.get("blend_radius")
    inner_b = data.get("inner_blend_radius")
    try:
        if (
            outer_b is not None
            and inner_b is not None
            and float(inner_b) > float(outer_b) + 1e-9
        ):
            tags.append("inner_gt_outer")
    except (TypeError, ValueError):
        pass
    return tags


def is_host_owned_draft(draft: ActionDraft) -> bool:
    """Whether draft continuous geometry is host-owned (ban on reject)."""

    if draft.authorship in HOST_OWNED_AUTHORSHIP:
        return True
    if draft.authorship == "llm_planner":
        return False
    # Legacy drafts without authorship: fall back to rationale markers.
    rationale = draft.rationale or ""
    return (
        "Host compiler" in rationale
        or "legal_candidate_menu" in rationale
        or "Host repaired junction" in rationale
        or "LLM menu selection" in rationale
    )


def _with_authorship(draft: ActionDraft, authorship: Authorship) -> ActionDraft:
    if draft.authorship == authorship:
        return draft
    return draft.model_copy(update={"authorship": authorship})


def _source_to_authorship(source: str) -> Authorship:
    if source in _FREECAD_MENU_SOURCES or source in {
        "host_compiler",
        "style_contract",
        "goal_seed",
        "alternate_open_port",
    }:
        return "host_menu"
    return "host_menu"


def apply_style_repairs(
    draft: ActionDraft,
    state: PipeState,
    repair_observations: list[dict[str, Any]],
) -> ActionDraft:
    """BRANCH_STYLE_MISMATCH → intent junction_style remapped by host."""

    if draft.module != "junction":
        return draft
    goal = next(
        (
            item
            for item in state.remaining_goals
            if item.goal_id in set(draft.completed_goal_ids or draft.affected_goal_ids)
        ),
        state.remaining_goals[0] if state.remaining_goals else None,
    )
    if goal is None or goal.type != "branch":
        return draft

    style_reject = any(
        (obs.get("issue_code") == "BRANCH_STYLE_MISMATCH")
        or (
            isinstance(obs.get("actual"), dict)
            and obs.get("actual", {}).get("blend_mode") == "hard"
            and (obs.get("expected") or {}).get("junction_style") == "smooth_hub"
        )
        for obs in repair_observations
        if isinstance(obs, dict)
    )
    intent_wants_smooth = (goal.junction_style or "smooth_hub") == "smooth_hub"
    current_mode = (draft.params or {}).get("blend_mode")
    if not (
        style_reject
        or (intent_wants_smooth and current_mode == "hard")
        or (goal.junction_style == "hard_fuse" and current_mode == "fillet")
    ):
        return draft

    default_od = float(
        goal.branch_outer_diameter
        if goal.branch_outer_diameter is not None
        else (
            state.open_ports[0].outer_diameter
            if state.open_ports
            else state.global_spec.outer_diameter
        )
    )
    style_params = _junction_style_params(goal, default_od=default_od)
    merged = dict(draft.params or {})
    for key in ("blend_radius", "inner_blend_radius"):
        merged.pop(key, None)
    merged.update(style_params)
    return draft.model_copy(
        update={
            "params": merged,
            "authorship": "style_repair",
            "rationale": (
                (draft.rationale or "")
                + " Host repaired junction blend style to match intent contract."
            ).strip(),
        }
    )


def apply_freecad_junction_repairs(
    draft: ActionDraft,
    state: PipeState,
    repair_observations: list[dict[str, Any]],
    *,
    source_hint: str | None = None,
) -> ActionDraft:
    """FreeCAD junction raw-solid failures → compact hub + longer arms.

    Skipped when the draft is already a freecad_junction_repair menu row
    (avoids double morph / fingerprint collapse).
    """

    if draft.module != "junction":
        return draft
    source = source_hint or ""
    rationale = draft.rationale or ""
    if (
        source == "freecad_junction_repair"
        or "junction_fc_" in rationale
        or "freecad_junction_repair" in rationale
    ):
        return draft
    freecad_hit = any(
        str(obs.get("issue_code") or "").startswith("FREECAD")
        or "JUNCTION_RAW" in str(obs.get("actual") or {})
        or "junction_material" in str(obs.get("actual") or {}).lower()
        or "raw compact junction" in str(obs.get("actual") or {}).lower()
        or "raw compact junction" in str(obs.get("message") or "").lower()
        for obs in repair_observations
        if isinstance(obs, dict)
    )
    if not freecad_hit:
        return draft
    params = dict(draft.params or {})
    outlets = [
        dict(item) for item in (params.get("outlets") or []) if isinstance(item, dict)
    ]
    if not outlets:
        return draft
    od = float(params.get("outer_diameter") or outlets[0].get("outer_diameter") or 20.0)
    max_hub = float(params.get("max_hub_radius") or od)
    max_hub = max(od * 0.55, min(max_hub * 0.72, od * 1.02))
    params["max_hub_radius"] = max_hub
    if params.get("blend_mode") == "fillet":
        br = float(params.get("blend_radius") or max(od * 0.2, 1.5))
        ibr = float(params.get("inner_blend_radius") or br * 0.55)
        params["blend_radius"] = min(br * 0.8, max_hub * 0.45)
        params["inner_blend_radius"] = min(ibr * 0.8, float(params["blend_radius"]))
    min_len = max(max_hub * 2.8, od * 4.0, 28.0)
    for outlet in outlets:
        length_val = float(outlet.get("length") or min_len)
        outlet["length"] = max(length_val * 1.2, min_len)
    params["outlets"] = outlets
    return draft.model_copy(
        update={
            "params": params,
            "authorship": draft.authorship or "host_compile",
            "rationale": (
                (draft.rationale or "")
                + " Host repaired junction hub/arm sizing for FreeCAD OCC stability."
            ).strip(),
        }
    )


def finalize_host_draft(
    draft: ActionDraft,
    state: PipeState,
    observations: list[dict[str, Any]],
    *,
    source_hint: str | None,
) -> ActionDraft:
    """Style repair always; FreeCAD morph only when not already a freecad menu row."""

    draft = apply_style_repairs(draft, state, observations)
    draft = apply_freecad_junction_repairs(
        draft, state, observations, source_hint=source_hint
    )
    return draft


def extract_menu(
    observations: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    for item in observations or []:
        if (
            isinstance(item, dict)
            and item.get("context_type") == "legal_candidate_menu"
            and item.get("candidates")
        ):
            return item
    return None


def observations_are_freecad(
    observations: list[dict[str, Any]] | None,
) -> bool:
    """True when observations indicate FreeCAD semantic/geometry reject."""

    for item in observations or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("issue_code") or "")
        check = str(item.get("check_name") or "")
        if code.startswith("FREECAD") or code in {
            "FREECAD_GEOMETRY_VALIDATION_FAILED",
            "FREECAD_GEOMETRY",
        }:
            return True
        if check in {"freecad_semantic_validation", "freecad_geometry"}:
            return True
        actual = item.get("actual")
        if isinstance(actual, dict):
            blob = str(actual)
            if "JUNCTION_RAW" in blob or "junction_material" in blob.lower():
                return True
        if "raw compact junction" in str(item.get("message") or "").lower():
            return True
    return False


def build_trial_menu(
    state: PipeState,
    *,
    observations: list[dict[str, Any]] | None = None,
    rejected_draft: ActionDraft | dict[str, Any] | None = None,
    forbidden_fingerprints: set[str] | None = None,
    max_candidates: int = 7,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Single menu builder for host trial + planner observation.

    When ``rejected_draft`` is set or ``force_rebuild``, always rebuild so
    freecad_* lattice is derived from the failed geometry (not a stale menu).
    """

    if not force_rebuild and rejected_draft is None:
        menu = extract_menu(observations)
        if menu is not None and menu.get("candidates"):
            return menu
    return build_legal_candidate_menu(
        state,
        rejected_draft=rejected_draft,
        issues=list(observations or []),
        max_candidates=max_candidates,
        forbidden_fingerprints=forbidden_fingerprints,
    )


def inject_trial_menu(
    observations: list[dict[str, Any]],
    state: PipeState,
    *,
    rejected_draft: ActionDraft | dict[str, Any] | None = None,
    forbidden_fingerprints: set[str] | None = None,
    max_candidates: int = 7,
) -> list[dict[str, Any]]:
    """Replace any existing legal menu with a fresh host trial menu."""

    menu = build_trial_menu(
        state,
        observations=observations,
        rejected_draft=rejected_draft,
        forbidden_fingerprints=forbidden_fingerprints,
        max_candidates=max_candidates,
        force_rebuild=True,
    )
    stripped = [
        item
        for item in observations
        if not (
            isinstance(item, dict) and item.get("context_type") == "legal_candidate_menu"
        )
    ]
    return [*stripped, menu]


def _finalize_menu_row(
    draft: ActionDraft,
    candidate: dict[str, Any],
    state: PipeState,
    observations: list[dict[str, Any]],
    banned: set[str],
) -> ActionDraft | None:
    authorship = _source_to_authorship(str(candidate.get("source") or ""))
    draft = _with_authorship(draft, authorship)
    finalized = finalize_host_draft(
        draft,
        state,
        observations,
        source_hint=str(candidate.get("source") or ""),
    )
    if draft_fingerprint(finalized) in banned:
        return None
    return finalized


def next_host_trial(
    state: PipeState,
    *,
    host_compiler_enabled: bool = True,
    repair_observations: list[dict[str, Any]] | None = None,
    forbidden_digests: set[str] | None = None,
    rejected_draft: ActionDraft | dict[str, Any] | None = None,
    freecad_exclusive: bool | None = None,
) -> ActionDraft | None:
    """Return the next unbanned host-owned draft, or None.

    FreeCAD exclusive mode (default when observations are FreeCAD): walk only
    freecad_* menu rows — never style re-picks or bare host recompile that
    replay near-identical continuous envelopes, and never fall through to
    compile_next_action (that path re-emits the banned junction).
    """

    if not host_compiler_enabled:
        return None
    observations = list(repair_observations or [])
    banned = set(forbidden_digests or ())
    freecad = (
        freecad_exclusive
        if freecad_exclusive is not None
        else observations_are_freecad(observations)
    )
    menu = build_trial_menu(
        state,
        observations=observations,
        rejected_draft=rejected_draft,
        forbidden_fingerprints=banned or None,
        force_rebuild=rejected_draft is not None or freecad,
    )

    # FreeCAD: freecad_* lattice first (and only, in exclusive mode).
    if freecad:
        ranked: list[tuple[int, ActionDraft]] = []
        for draft, candidate in iter_menu_drafts(
            state,
            menu,
            forbidden_fingerprints=banned,
            preferred_sources=_FREECAD_MENU_SOURCES,
        ):
            finalized = _finalize_menu_row(
                draft, candidate, state, observations, banned
            )
            if finalized is None:
                continue
            # L0 ranking only — fewer analytic issues preferred.
            score = len(junction_l0_issue_tags(dict(finalized.params or {})))
            ranked.append((score, finalized))
        ranked.sort(key=lambda item: item[0])
        if ranked:
            return ranked[0][1]
        # Rebuild once from rejected draft if the embedded menu was stale.
        if rejected_draft is not None:
            recovery = build_legal_candidate_menu(
                state,
                rejected_draft=rejected_draft,
                issues=observations,
                max_candidates=7,
                forbidden_fingerprints=banned,
            )
            for draft, candidate in iter_menu_drafts(
                state,
                recovery,
                forbidden_fingerprints=banned,
                preferred_sources=_FREECAD_MENU_SOURCES,
            ):
                finalized = _finalize_menu_row(
                    draft, candidate, state, observations, banned
                )
                if finalized is not None:
                    return finalized
        return None

    for draft, candidate in iter_menu_drafts(
        state, menu, forbidden_fingerprints=banned
    ):
        finalized = _finalize_menu_row(draft, candidate, state, observations, banned)
        if finalized is not None:
            return finalized

    draft = compile_next_action(state)
    if draft is None:
        return None
    draft = _with_authorship(draft, "host_compile")
    draft = finalize_host_draft(draft, state, observations, source_hint=None)
    if banned and draft_fingerprint(draft) in banned:
        recovery = build_legal_candidate_menu(
            state,
            rejected_draft=draft,
            issues=observations,
            max_candidates=7,
            forbidden_fingerprints=banned,
        )
        for alt, candidate in iter_menu_drafts(
            state, recovery, forbidden_fingerprints=banned
        ):
            finalized = _finalize_menu_row(
                alt, candidate, state, observations, banned
            )
            if finalized is not None:
                return finalized
        return None
    return draft


def freecad_lattice_available(
    state: PipeState,
    *,
    observations: list[dict[str, Any]] | None = None,
    rejected_draft: ActionDraft | dict[str, Any] | None = None,
    forbidden_fingerprints: set[str] | None = None,
) -> bool:
    """True when this FreeCAD reject is a freecad_* lattice problem class.

    Bans are ignored for *presence* so exhausting every lattice fingerprint still
    counts as the exclusive FreeCAD channel (not a fall-through to fat planner).
    """

    del forbidden_fingerprints  # presence is ban-independent
    if not observations_are_freecad(observations):
        return False
    menu = build_trial_menu(
        state,
        observations=list(observations or []),
        rejected_draft=rejected_draft,
        forbidden_fingerprints=None,
        force_rebuild=True,
    )
    return freecad_repair_sources_present(menu)


def freecad_host_search_exhausted(
    state: PipeState,
    *,
    observations: list[dict[str, Any]],
    forbidden_fingerprints: set[str],
    rejected_draft: ActionDraft | dict[str, Any] | None = None,
    host_compiler_enabled: bool = True,
) -> bool:
    """True when freecad_* lattice class applies and every unbanned trial is gone.

    Returns False when FreeCAD fail has no freecad_* lattice (e.g. spline
    curvature) — those channels still use encoded thin/fat planner.
    """

    if not observations_are_freecad(observations):
        return False
    if not freecad_lattice_available(
        state,
        observations=observations,
        rejected_draft=rejected_draft,
    ):
        return False
    trial = next_host_trial(
        state,
        host_compiler_enabled=host_compiler_enabled,
        repair_observations=observations,
        forbidden_digests=set(forbidden_fingerprints or ()),
        rejected_draft=rejected_draft,
        freecad_exclusive=True,
    )
    return trial is None


def host_can_advance(
    state: PipeState,
    *,
    observations: list[dict[str, Any]],
    forbidden_fingerprints: set[str],
    host_compiler_enabled: bool = True,
    rejected_draft: ActionDraft | dict[str, Any] | None = None,
) -> bool:
    """True when a *material* host repair trial remains before LLM/advisor.

    Skip advisor when:
    - BRANCH_STYLE / DUPLICATE can be host-repaired (style map / menu), or
    - FreeCAD has freecad_* geometric variants still unbanned.
    Do not skip for arbitrary static rejects with only bare recompile.
    """

    if not observations:
        return False
    banned = set(forbidden_fingerprints or ())
    issue_codes = {
        str(item.get("issue_code"))
        for item in observations
        if isinstance(item, dict) and item.get("issue_code")
    }
    discrete = bool(
        issue_codes
        & {
            "BRANCH_STYLE_MISMATCH",
            "DUPLICATE_REJECTED_CANDIDATE",
        }
    )
    freecad = observations_are_freecad(observations)
    lattice = freecad and freecad_lattice_available(
        state,
        observations=observations,
        rejected_draft=rejected_draft,
        forbidden_fingerprints=banned,
    )
    if not discrete and not lattice:
        return False
    trial = next_host_trial(
        state,
        host_compiler_enabled=host_compiler_enabled,
        repair_observations=observations,
        forbidden_digests=banned,
        rejected_draft=rejected_draft,
        freecad_exclusive=True if lattice else None,
    )
    if trial is None:
        return False
    if discrete and not lattice:
        return True
    if lattice:
        return not freecad_host_search_exhausted(
            state,
            observations=observations,
            forbidden_fingerprints=banned,
            rejected_draft=rejected_draft,
            host_compiler_enabled=host_compiler_enabled,
        )
    return True


def host_goal_is_compilable(state: PipeState) -> bool:
    """Whether the next remaining goal has a host compile or complete menu."""

    if compile_next_action(state) is not None:
        return True
    menu = build_legal_candidate_menu(state, max_candidates=5)
    return menu_has_complete_candidates(menu)


def freecad_repair_sources_present(menu: dict[str, Any] | None) -> bool:
    if not menu:
        return False
    return any(
        str(c.get("source")) in _FREECAD_MENU_SOURCES
        for c in (menu.get("candidates") or [])
    )


__all__ = [
    "FREECAD_HOST_MENU_EXHAUSTED",
    "FreeCADHostMenuExhausted",
    "HOST_OWNED_AUTHORSHIP",
    "apply_freecad_junction_repairs",
    "apply_style_repairs",
    "build_trial_menu",
    "draft_fingerprint",
    "extract_menu",
    "finalize_host_draft",
    "freecad_host_search_exhausted",
    "freecad_lattice_available",
    "freecad_repair_sources_present",
    "host_can_advance",
    "host_goal_is_compilable",
    "inject_trial_menu",
    "is_host_owned_draft",
    "junction_l0_issue_tags",
    "next_host_trial",
    "observations_are_freecad",
]
