"""거절 이후 planner/host가 고를 수 있는 catalog-legal 추천 후보 메뉴.

검증기가 거절했을 때 LLM이 자유 숫자 walk를 하지 않도록, host가 합법 후보
목록을 만들고 observation에 실어 준다. 후보 0번은 보통 권장(recommended).
"""

from __future__ import annotations

from typing import Any

from cadgen.intent_action_compiler import _junction_style_params, compile_next_action
from cadgen.port_algebra import is_primary_port_id, select_construction_port
from cadgen.typed_data_models import ActionDraft, Goal, PipeState, StaticIssue


def build_legal_candidate_menu(
    state: PipeState,
    *,
    rejected_draft: ActionDraft | dict[str, Any] | None = None,
    issues: list[StaticIssue | dict[str, Any]] | None = None,
    max_candidates: int = 5,
    forbidden_fingerprints: set[str] | None = None,
) -> dict[str, Any]:
    """현재 상태와 거절 근거로 legal candidate menu를 만든다."""

    issue_codes = _issue_codes(issues or [])
    goal = state.remaining_goals[0] if state.remaining_goals else None
    draft_dict = _as_dict(rejected_draft)
    banned = set(forbidden_fingerprints or ())
    candidates: list[dict[str, Any]] = []

    freecad_like = any(
        code.startswith("FREECAD")
        or "JUNCTION" in code
        or "FILLET" in code
        or code in {"FREECAD_GEOMETRY_VALIDATION_FAILED"}
        for code in issue_codes
    )
    if freecad_like:
        # Host + style + freecad lattice (~5) must all fit for exclusive exhaust.
        max_candidates = max(int(max_candidates), 9)

    # 1) Host recompile — preferred only when FreeCAD is not asking for a
    # different geometric envelope (otherwise replaying the same hub loses).
    host = compile_next_action(state)
    if host is not None:
        candidates.append(
            _candidate(
                candidate_id="host_recompile_0",
                module=host.module,
                target_port=host.target_port,
                params=dict(host.params or {}),
                completed_goal_ids=list(host.completed_goal_ids),
                affected_goal_ids=list(host.affected_goal_ids),
                recommended=not freecad_like,
                source="host_compiler",
                rationale="Deterministic host recompile of the next intent goal.",
            )
        )

    # 2) Style repairs for junctions (BRANCH_STYLE_MISMATCH and smooth_hub goals).
    if goal is not None and goal.type == "branch":
        candidates.extend(_junction_style_candidates(state, goal, draft_dict, issue_codes))

    # 2b) FreeCAD junction OCC failures: host-owned geometric variants (no LLM numbers).
    if freecad_like and (
        (draft_dict or {}).get("module") == "junction"
        or (goal is not None and goal.type == "branch")
    ):
        candidates.extend(
            _junction_freecad_repair_candidates(state, goal, draft_dict, host)
        )

    # 2c) FreeCAD connect/route failures: midpoint / chord alternatives.
    if freecad_like and (draft_dict or {}).get("module") in {
        "connect_ports",
        "route",
    }:
        candidates.extend(_connect_route_freecad_repair_candidates(state, draft_dict, host))

    # 2d) FreeCAD flange/inline wall failures: re-clamp annulus plate sizes.
    if freecad_like and (draft_dict or {}).get("module") == "inline_component":
        candidates.extend(_inline_freecad_repair_candidates(state, draft_dict, host))

    # 3) Alternate open-port targets when multi-port ambiguity exists.
    if goal is not None:
        candidates.extend(_alternate_port_candidates(state, goal, draft_dict))

    # 4) Goal-type seeds when host cannot compile or as thin-wire backups.
    if goal is not None and (host is None or freecad_like):
        candidates.extend(_goal_type_seed_candidates(state, goal))

    # Deduplicate by (module, target, blend_mode, sorted outlet axes).
    # Drop candidates that match already-rejected host fingerprints so the
    # planner is not asked to re-pick a known nogood.
    uniq: list[dict[str, Any]] = []
    seen: set[str] = set()
    excluded_banned = 0
    for item in candidates:
        key = _candidate_key(item)
        if key in seen:
            continue
        if banned and _candidate_fingerprint(item) in banned:
            excluded_banned += 1
            continue
        seen.add(key)
        uniq.append(item)
        if len(uniq) >= max_candidates:
            break

    if uniq and not any(item.get("recommended") for item in uniq):
        uniq[0]["recommended"] = True

    return {
        "context_type": "legal_candidate_menu",
        "must_select_from_menu": bool(uniq),
        "selection_policy": (
            "Pick exactly one candidate_id. Copy module, target_port, and params "
            "from that candidate unless a listed optional_override is used. "
            "Do not invent blend_mode/style values outside the chosen candidate."
        ),
        "issue_codes": issue_codes,
        "candidates": uniq,
        "candidate_count": len(uniq),
        "excluded_banned_count": excluded_banned,
    }


def menu_recommended_draft(state: PipeState, menu: dict[str, Any]) -> ActionDraft | None:
    """메뉴의 recommended 후보를 ActionDraft로 변환한다 (host apply용)."""

    candidates = list(menu.get("candidates") or [])
    if not candidates:
        return None
    chosen = next((c for c in candidates if c.get("recommended")), candidates[0])
    return menu_draft_from_candidate(
        chosen,
        rationale_prefix="Host compiler applied legal_candidate_menu:",
    )


def menu_candidate_params_complete(candidate: dict[str, Any]) -> bool:
    """후보가 host 수화만으로 ActionDraft 가 될 만큼 params 를 갖는지."""

    if not candidate.get("module") or not candidate.get("target_port"):
        return False
    params = candidate.get("params")
    if not isinstance(params, dict) or not params:
        return False
    module = str(candidate["module"])
    if module == "route":
        kind = params.get("path_kind")
        if kind == "line":
            return params.get("length") is not None
        if kind == "circular_arc":
            return (
                params.get("bend_radius") is not None
                and params.get("sweep_angle") is not None
                and params.get("plane_normal") is not None
            )
        if kind == "spline":
            return bool(params.get("waypoints"))
        return kind is not None and (
            params.get("length") is not None or params.get("waypoints") is not None
        )
    if module == "junction":
        return bool(params.get("outlets")) and params.get("blend_mode") is not None
    if module == "inline_component":
        return params.get("component_type") is not None and params.get("length") is not None
    if module == "transition":
        return params.get("diameter_out") is not None and params.get("length") is not None
    if module == "terminate":
        return params.get("termination_type") is not None
    if module == "connect_ports":
        return params.get("path_kind") is not None
    return True


def menu_has_complete_candidates(menu: dict[str, Any]) -> bool:
    """메뉴 후보가 전부 params_complete 이면 thin menu wire 사용 가능."""

    candidates = list(menu.get("candidates") or [])
    if not candidates:
        return False
    return all(menu_candidate_params_complete(item) for item in candidates)


def iter_menu_drafts(
    state: PipeState,
    menu: dict[str, Any],
    *,
    forbidden_fingerprints: set[str] | None = None,
    preferred_sources: set[str] | None = None,
) -> list[tuple[ActionDraft, dict[str, Any]]]:
    """Unbanned menu rows as (draft, candidate), recommended first.

    ``state`` is accepted for API symmetry with host call sites.
    """

    del state
    banned = set(forbidden_fingerprints or ())
    rows = list(menu.get("candidates") or [])
    rows.sort(key=lambda item: (0 if item.get("recommended") else 1))
    out: list[tuple[ActionDraft, dict[str, Any]]] = []
    for candidate in rows:
        if preferred_sources is not None and str(candidate.get("source")) not in preferred_sources:
            continue
        if not menu_candidate_params_complete(candidate):
            continue
        if _candidate_fingerprint(candidate) in banned:
            continue
        source = str(candidate.get("source") or "")
        hostish = source.startswith(
            ("host", "freecad", "style", "goal", "alternate")
        )
        draft = menu_draft_from_candidate(
            candidate,
            rationale_prefix=(
                "Host compiler applied legal_candidate_menu:"
                if hostish
                else "LLM menu selection:"
            ),
        )
        fp = action_draft_fingerprint(
            target_port=draft.target_port,
            module=draft.module,
            params=dict(draft.params or {}),
            affected_goal_ids=list(draft.affected_goal_ids),
            completed_goal_ids=list(draft.completed_goal_ids),
        )
        if fp in banned:
            continue
        out.append((draft, candidate))
    return out


def menu_draft_from_candidate_id(
    menu: dict[str, Any],
    candidate_id: str,
    *,
    rationale_prefix: str = "LLM menu selection:",
) -> ActionDraft | None:
    """candidate_id 로 menu row 를 ActionDraft 로 수화한다."""

    for item in menu.get("candidates") or []:
        if str(item.get("candidate_id")) == str(candidate_id):
            if not menu_candidate_params_complete(item):
                return None
            return menu_draft_from_candidate(
                item, rationale_prefix=rationale_prefix
            )
    return None


def menu_draft_from_candidate(
    chosen: dict[str, Any],
    *,
    rationale_prefix: str,
) -> ActionDraft:
    """단일 menu row → ActionDraft."""

    source = str(chosen.get("source") or "")
    authorship = (
        "llm_menu"
        if rationale_prefix.startswith("LLM menu")
        else "host_menu"
    )
    return ActionDraft(
        target_port=str(chosen["target_port"]),
        module=str(chosen["module"]),
        params=dict(chosen.get("params") or {}),
        catalog_schema_version=2,
        affected_goal_ids=list(chosen.get("affected_goal_ids") or []),
        completed_goal_ids=list(chosen.get("completed_goal_ids") or []),
        rationale=f"{rationale_prefix}{chosen.get('candidate_id')}",
        authorship=authorship,
    )


def _junction_freecad_repair_candidates(
    state: PipeState,
    goal: Goal | None,
    draft_dict: dict[str, Any] | None,
    host: ActionDraft | None,
) -> list[dict[str, Any]]:
    """OCC-invalid junction raw solids → longer arms / smaller hub / compact fillet."""

    base: ActionDraft | None = host
    if base is None and draft_dict and draft_dict.get("module") == "junction":
        try:
            base = ActionDraft.model_validate(
                {
                    "target_port": draft_dict.get("target_port"),
                    "module": "junction",
                    "params": dict(draft_dict.get("params") or {}),
                    "catalog_schema_version": 2,
                    "affected_goal_ids": list(draft_dict.get("affected_goal_ids") or []),
                    "completed_goal_ids": list(draft_dict.get("completed_goal_ids") or []),
                    "rationale": "history",
                }
            )
        except Exception:
            base = None
    if base is None or base.module != "junction":
        return []
    params0 = dict(base.params or {})
    outlets0 = list(params0.get("outlets") or [])
    if not outlets0:
        return []
    od = float(params0.get("outer_diameter") or outlets0[0].get("outer_diameter") or 20.0)
    out: list[dict[str, Any]] = []

    def _variant(
        *,
        candidate_id: str,
        length_scale: float,
        hub_scale: float,
        blend_scale: float,
        recommended: bool,
        rationale: str,
    ) -> dict[str, Any]:
        params = dict(params0)
        max_hub = float(params.get("max_hub_radius") or od)
        max_hub = max(od * 0.55, min(max_hub * hub_scale, od * 1.05))
        params["max_hub_radius"] = max_hub
        if params.get("blend_mode") == "fillet":
            br = float(params.get("blend_radius") or max(od * 0.2, 1.5))
            ibr = float(params.get("inner_blend_radius") or br * 0.55)
            params["blend_radius"] = min(br * blend_scale, max_hub * 0.45)
            params["inner_blend_radius"] = min(ibr * blend_scale, params["blend_radius"])
        new_outlets = []
        min_len = max(max_hub * 2.8, od * 4.0, 28.0)
        for outlet in outlets0:
            item = dict(outlet)
            length_val = float(item.get("length") or min_len)
            item["length"] = max(length_val * length_scale, min_len)
            new_outlets.append(item)
        params["outlets"] = new_outlets
        goal_ids = list(base.completed_goal_ids or base.affected_goal_ids)
        return _candidate(
            candidate_id=candidate_id,
            module="junction",
            target_port=base.target_port,
            params=params,
            completed_goal_ids=goal_ids,
            affected_goal_ids=goal_ids,
            recommended=recommended,
            source="freecad_junction_repair",
            rationale=rationale,
        )

    # Finite discrete lattice for FreeCAD OCC junction failures (no LLM numbers).
    # Ordered by typical recovery strength for acute Y hubs.
    out.append(
        _variant(
            candidate_id="junction_fc_compact_hub",
            length_scale=1.15,
            hub_scale=0.7,
            blend_scale=0.75,
            recommended=True,
            rationale="Compact hub + slightly longer arms for OCC Y-junction stability.",
        )
    )
    out.append(
        _variant(
            candidate_id="junction_fc_long_arms",
            length_scale=1.5,
            hub_scale=0.85,
            blend_scale=0.85,
            recommended=False,
            rationale="Longer free arms to avoid short-outlet OCC fuse failures.",
        )
    )
    out.append(
        _variant(
            candidate_id="junction_fc_tiny_blend",
            length_scale=1.25,
            hub_scale=0.65,
            blend_scale=0.5,
            recommended=False,
            rationale="Minimal fillet radii with compact hub for acute Y hubs.",
        )
    )
    out.append(
        _variant(
            candidate_id="junction_fc_ultra_compact",
            length_scale=1.35,
            hub_scale=0.55,
            blend_scale=0.4,
            recommended=False,
            rationale="Ultra-compact hub lattice point for severe multiFuse failures.",
        )
    )
    out.append(
        _variant(
            candidate_id="junction_fc_arm_2x",
            length_scale=2.0,
            hub_scale=0.75,
            blend_scale=0.7,
            recommended=False,
            rationale="2× arm length with moderate hub for short-stem OCC fuse fails.",
        )
    )
    return out


def _junction_style_candidates(
    state: PipeState,
    goal: Goal,
    draft_dict: dict[str, Any] | None,
    issue_codes: list[str],
) -> list[dict[str, Any]]:
    target = select_construction_port(state, goal)
    if target is None:
        return []
    default_od = float(
        goal.branch_outer_diameter
        if goal.branch_outer_diameter is not None
        else target.outer_diameter
    )
    base_params: dict[str, Any] = {}
    if draft_dict and draft_dict.get("module") == "junction":
        base_params = dict(draft_dict.get("params") or {})
    style_params = _junction_style_params(goal, default_od=default_od)
    params = dict(base_params)
    for key in ("blend_radius", "inner_blend_radius"):
        params.pop(key, None)
    params.update(style_params)
    if "outlets" not in params and base_params.get("outlets"):
        params["outlets"] = base_params["outlets"]
    # If outlets missing, inherit from a fresh host recompile of the branch goal.
    if "outlets" not in params:
        host = compile_next_action(state)
        if host is not None and host.module == "junction":
            params["outlets"] = list((host.params or {}).get("outlets") or [])
    if "outlets" not in params or not params["outlets"]:
        return []
    goal_ids = [goal.goal_id] if goal.goal_id else []
    recommended = (
        "BRANCH_STYLE_MISMATCH" in issue_codes
        or (goal.junction_style or "smooth_hub") == "smooth_hub"
    )
    return [
        _candidate(
            candidate_id="junction_style_contract_0",
            module="junction",
            target_port=target.id,
            params=params,
            completed_goal_ids=goal_ids,
            affected_goal_ids=goal_ids,
            recommended=recommended,
            source="style_contract",
            rationale=(
                f"Apply blend_mode for junction_style="
                f"{goal.junction_style or 'smooth_hub'}."
            ),
        )
    ]


def _alternate_port_candidates(
    state: PipeState,
    goal: Goal,
    draft_dict: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if len(state.open_ports) < 2 or draft_dict is None:
        return []
    module = str(draft_dict.get("module") or "")
    if module not in {
        "route",
        "transition",
        "junction",
        "terminate",
        "inline_component",
        "connect_ports",
    }:
        return []
    current = str(draft_dict.get("target_port") or "")
    params = dict(draft_dict.get("params") or {})
    out: list[dict[str, Any]] = []
    # Prefer primary first.
    ports = sorted(
        state.open_ports,
        key=lambda port: (0 if is_primary_port_id(port.id) else 1, port.id),
    )
    for port in ports:
        if port.id == current:
            continue
        out.append(
            _candidate(
                candidate_id=f"retarget_{port.id}",
                module=module,
                target_port=port.id,
                params=params,
                completed_goal_ids=list(draft_dict.get("completed_goal_ids") or []),
                affected_goal_ids=list(draft_dict.get("affected_goal_ids") or []),
                recommended=False,
                source="alternate_open_port",
                rationale=f"Retry same primitive on open port {port.id}.",
            )
        )
        if len(out) >= 2:
            break
    return out


def _inline_freecad_repair_candidates(
    state: PipeState,
    draft_dict: dict[str, Any] | None,
    host: ActionDraft | None,
) -> list[dict[str, Any]]:
    """inline_component FreeCAD wall failures → safer flange/body envelope."""

    base = host
    if base is None and draft_dict and draft_dict.get("module") == "inline_component":
        try:
            base = ActionDraft.model_validate(
                {
                    "target_port": draft_dict.get("target_port"),
                    "module": "inline_component",
                    "params": dict(draft_dict.get("params") or {}),
                    "catalog_schema_version": 2,
                    "affected_goal_ids": list(draft_dict.get("affected_goal_ids") or []),
                    "completed_goal_ids": list(draft_dict.get("completed_goal_ids") or []),
                    "rationale": "history",
                }
            )
        except Exception:
            base = None
    if base is None or base.module != "inline_component":
        return []
    params0 = dict(base.params or {})
    if params0.get("component_type") != "flange":
        return []
    od = float(
        params0.get("outer_diameter")
        or (state.open_ports[0].outer_diameter if state.open_ports else 20.0)
    )
    wall = float(
        params0.get("wall_thickness")
        or (state.open_ports[0].wall_thickness if state.open_ports else 2.0)
    )
    from cadgen.intent_action_compiler import _flange_host_dimensions

    safe = _flange_host_dimensions(outer_diameter=od, wall_thickness=wall)
    goal_ids = list(base.completed_goal_ids or base.affected_goal_ids)
    out: list[dict[str, Any]] = []
    for idx, body_scale in enumerate((1.0, 1.2, 1.4), start=1):
        params = dict(params0)
        params.update(safe)
        body = float(safe["body_outer_diameter"]) * body_scale
        hole = float(safe["flange_bolt_hole_diameter"])
        # Keep PCD near outer rim of the (possibly enlarged) plate.
        circle = max(od + hole + 2.0, body - hole - 3.0)
        params["body_outer_diameter"] = body
        params["flange_bolt_circle_diameter"] = circle
        params["flange_bolt_hole_diameter"] = hole
        if params0.get("flange_reference_axis") is not None:
            params["flange_reference_axis"] = params0["flange_reference_axis"]
        out.append(
            _candidate(
                candidate_id=f"flange_fc_annulus_{idx}",
                module="inline_component",
                target_port=base.target_port,
                params=params,
                completed_goal_ids=goal_ids,
                affected_goal_ids=goal_ids,
                recommended=idx == 1,
                source="freecad_inline_repair",
                rationale="Recompute flange plate/PCD to clear FreeCAD mid-wall samples.",
            )
        )
    return out


def _connect_route_freecad_repair_candidates(
    state: PipeState,
    draft_dict: dict[str, Any] | None,
    host: ActionDraft | None,
) -> list[dict[str, Any]]:
    """connect_ports / curved route FreeCAD failures → midpoint scale variants."""

    base = host
    if base is None and draft_dict and draft_dict.get("module") in {
        "connect_ports",
        "route",
    }:
        try:
            base = ActionDraft.model_validate(
                {
                    "target_port": draft_dict.get("target_port"),
                    "module": draft_dict.get("module"),
                    "params": dict(draft_dict.get("params") or {}),
                    "catalog_schema_version": 2,
                    "affected_goal_ids": list(draft_dict.get("affected_goal_ids") or []),
                    "completed_goal_ids": list(draft_dict.get("completed_goal_ids") or []),
                    "rationale": "history",
                }
            )
        except Exception:
            base = None
    if base is None or base.module not in {"connect_ports", "route"}:
        return []
    params0 = dict(base.params or {})
    out: list[dict[str, Any]] = []
    goal_ids = list(base.completed_goal_ids or base.affected_goal_ids)

    if base.module == "connect_ports" and params0.get("path_kind") == "spline":
        waypoints = list(params0.get("waypoints") or [])
        # Scale lateral offset of midpoints (or invent one from host recompile).
        for idx, scale in enumerate((0.6, 1.4, 2.0), start=1):
            params = dict(params0)
            if waypoints:
                # Nudge each waypoint slightly along a crude normal (z-up bias).
                scaled = []
                for point in waypoints:
                    if not isinstance(point, (list, tuple)) or len(point) < 3:
                        continue
                    scaled.append(
                        [
                            float(point[0]),
                            float(point[1]) + 2.0 * scale,
                            float(point[2]),
                        ]
                    )
                params["waypoints"] = scaled or waypoints
            else:
                params["waypoints"] = [[0.0, 5.0 * scale, 0.0]]
            out.append(
                _candidate(
                    candidate_id=f"connect_fc_mid_{idx}",
                    module="connect_ports",
                    target_port=base.target_port,
                    params=params,
                    completed_goal_ids=goal_ids,
                    affected_goal_ids=goal_ids,
                    recommended=idx == 1,
                    source="freecad_connect_repair",
                    rationale="Adjust connect spline midpoints for FreeCAD curvature.",
                )
            )
        # Chord fallback when nearly feasible.
        line_params = {
            "path_kind": "line",
            "section_source": params0.get("section_source") or "inherit_target",
            "other_port_id": params0.get("other_port_id"),
        }
        if line_params.get("other_port_id"):
            out.append(
                _candidate(
                    candidate_id="connect_fc_line_chord",
                    module="connect_ports",
                    target_port=base.target_port,
                    params=line_params,
                    completed_goal_ids=goal_ids,
                    affected_goal_ids=goal_ids,
                    recommended=False,
                    source="freecad_connect_repair",
                    rationale="Try collinear chord connect if spline FreeCAD fails.",
                )
            )
    elif base.module == "route" and params0.get("path_kind") == "circular_arc":
        for idx, scale in enumerate((0.8, 1.25, 1.6), start=1):
            params = dict(params0)
            if params.get("bend_radius") is not None:
                params["bend_radius"] = float(params["bend_radius"]) * scale
            out.append(
                _candidate(
                    candidate_id=f"route_fc_arc_r_{idx}",
                    module="route",
                    target_port=base.target_port,
                    params=params,
                    completed_goal_ids=goal_ids,
                    affected_goal_ids=goal_ids,
                    recommended=idx == 1,
                    source="freecad_route_repair",
                    rationale="Scale bend radius after FreeCAD arc/curvature failure.",
                )
            )
    return out


def _goal_type_seed_candidates(state: PipeState, goal: Goal) -> list[dict[str, Any]]:
    target = select_construction_port(state, goal)
    if target is None and state.open_ports:
        target = state.open_ports[0]
    if target is None:
        return []
    goal_ids = [goal.goal_id] if goal.goal_id else []
    if goal.type in {"move", "route"} and goal.path_kind in {None, "line"} and goal.length:
        return [
            _candidate(
                candidate_id="seed_line_route",
                module="route",
                target_port=target.id,
                params={
                    "path_kind": "line",
                    "section_source": "inherit_target",
                    "length": float(goal.length),
                },
                completed_goal_ids=goal_ids,
                affected_goal_ids=goal_ids,
                recommended=True,
                source="goal_seed",
                rationale="Seed line route from remaining goal length.",
            )
        ]
    if (
        goal.type == "turn"
        and goal.angle is not None
        and goal.bend_radius is not None
        and goal.plane_normal is not None
    ):
        return [
            _candidate(
                candidate_id="seed_turn_arc",
                module="route",
                target_port=target.id,
                params={
                    "path_kind": "circular_arc",
                    "section_source": "inherit_target",
                    "bend_radius": float(goal.bend_radius),
                    "sweep_angle": float(goal.angle),
                    "plane_normal": list(goal.plane_normal),
                },
                completed_goal_ids=goal_ids,
                affected_goal_ids=goal_ids,
                recommended=True,
                source="goal_seed",
                rationale="Seed turn as host circular_arc route.",
            )
        ]
    if goal.type == "connector" and goal.component in {
        "flange",
        "coupling",
        "union",
        "valve",
    }:
        # Prefer host recompile path (usually already present). Seed is a
        # fallback when host compile is temporarily unavailable.
        host = compile_next_action(state)
        if host is not None and host.module == "inline_component":
            return [
                _candidate(
                    candidate_id="seed_connector_host",
                    module=host.module,
                    target_port=host.target_port,
                    params=dict(host.params or {}),
                    completed_goal_ids=list(host.completed_goal_ids),
                    affected_goal_ids=list(host.affected_goal_ids),
                    recommended=True,
                    source="goal_seed",
                    rationale="Seed host-compiled connector dimensions.",
                )
            ]
    return []


def _candidate(
    *,
    candidate_id: str,
    module: str,
    target_port: str,
    params: dict[str, Any],
    completed_goal_ids: list[str],
    affected_goal_ids: list[str],
    recommended: bool,
    source: str,
    rationale: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "module": module,
        "target_port": target_port,
        "params": params,
        "completed_goal_ids": completed_goal_ids,
        "affected_goal_ids": affected_goal_ids,
        "recommended": recommended,
        "source": source,
        "rationale": rationale,
    }


def _candidate_key(item: dict[str, Any]) -> str:
    params = item.get("params") or {}
    blend = params.get("blend_mode")
    outlets = params.get("outlets") or []
    axes = []
    lengths = []
    for outlet in outlets:
        if isinstance(outlet, dict):
            if "axis" in outlet:
                axes.append(tuple(outlet["axis"]))
            if "length" in outlet:
                lengths.append(round(float(outlet["length"]), 6))
    return "|".join(
        [
            str(item.get("module")),
            str(item.get("target_port")),
            str(blend),
            str(params.get("blend_radius")),
            str(params.get("max_hub_radius")),
            str(axes),
            str(lengths),
            str(params.get("path_kind")),
            str(params.get("length")),
            str(params.get("component_type")),
            str(params.get("body_outer_diameter")),
            str(params.get("flange_bolt_circle_diameter")),
            str(params.get("bend_radius")),
            str(params.get("waypoints")),
        ]
    )


def action_draft_fingerprint(
    *,
    target_port: str,
    module: str,
    params: dict[str, Any] | None,
    affected_goal_ids: list[str] | None = None,
    completed_goal_ids: list[str] | None = None,
) -> str:
    """Host draft / menu candidate 공용 fingerprint."""

    from cadgen.stable_content_hash import stable_digest

    payload = {
        "target_port": target_port,
        "module": module,
        "params": params or {},
        "affected_goal_ids": list(affected_goal_ids or []),
        "completed_goal_ids": list(completed_goal_ids or []),
    }
    return stable_digest(payload)


def _candidate_fingerprint(item: dict[str, Any]) -> str:
    """menu candidate → host draft fingerprint 스키마."""

    return action_draft_fingerprint(
        target_port=str(item.get("target_port") or ""),
        module=str(item.get("module") or ""),
        params=dict(item.get("params") or {}),
        affected_goal_ids=list(item.get("affected_goal_ids") or []),
        completed_goal_ids=list(item.get("completed_goal_ids") or []),
    )


def _issue_codes(issues: list[StaticIssue | dict[str, Any]]) -> list[str]:
    codes: list[str] = []
    for issue in issues:
        if isinstance(issue, StaticIssue):
            codes.append(issue.issue_code)
        elif isinstance(issue, dict) and issue.get("issue_code"):
            codes.append(str(issue["issue_code"]))
    return codes


def _as_dict(draft: ActionDraft | dict[str, Any] | None) -> dict[str, Any] | None:
    if draft is None:
        return None
    if isinstance(draft, ActionDraft):
        return draft.model_dump(mode="json")
    return dict(draft)


__all__ = [
    "action_draft_fingerprint",
    "build_legal_candidate_menu",
    "iter_menu_drafts",
    "menu_candidate_params_complete",
    "menu_draft_from_candidate",
    "menu_draft_from_candidate_id",
    "menu_has_complete_candidates",
    "menu_recommended_draft",
]
