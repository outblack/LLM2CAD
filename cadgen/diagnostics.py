"""Evidence-bound diagnosis for rejected CAD state transitions.

The module deliberately does not plan or mutate geometry.  It turns one durable
rejection into a compact, typed case, validates an optional LLM diagnosis
against that case, and emits a bounded directive for the existing step planner.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel

from cadgen.freecad_script import (
    GENERATOR_VERSION,
    VALIDATION_SCHEMA_VERSION,
    VALIDATOR_POLICY_DIGEST,
    VALIDATOR_POLICY_ID,
)
from cadgen.schemas import (
    DiagnosticBinding,
    DiagnosticEvidenceUse,
    DiagnosticFact,
    DiagnosticJournal,
    FieldOwnership,
    GeometryValidationAdvisorResponse,
    ParameterCausality,
    ParameterDirectionRecommendation,
    ParameterRangeRecommendation,
    PipeState,
    RepairStrategy,
    RepairStrategyKind,
    StepRepairDiagnosticContext,
    StepRepairDiagnosis,
    StepRepairDiagnosisBody,
)


_FAILURE_CHECKS = (
    "non_adjacent_overlaps",
    "centerlines",
    "wall_section_failures",
    "connection_failures",
    "terminal_bore_failures",
    "anchored_inlet_bore_failures",
    "termination_seal_failures",
    "module_errors",
    "assembly_errors",
    "deterministic_constraint_failures",
)
_FAILURE_IDENTITY_KEYS = {
    "issue_code",
    "check_name",
    "failure_code",
    "stage",
    "module_id",
    "module_ids",
    "parent_module_id",
    "child_module_id",
    "adjacent",
    "policy_id",
    "policy_digest",
    "generator_version",
    "path_kind",
    "variant",
}
_METRIC_KEYS = {
    "allowed_volume",
    "common_volume",
    "excess_volume",
    "outside_interface_volume",
    "outside_interface_allowance",
    "minimum_radius",
    "required_minimum_radius",
    "minimum_wall_thickness",
    "distance",
    "gap",
}
_INFRASTRUCTURE_MARKERS = {
    "required_step_mcp_failed",
    "required_step_mcp_skipped",
    "required_final_mcp_failed",
    "planning_failed",
    "provider_error",
    "transport_error",
    "timeout",
}
_RETRYABLE_ADVISOR_FAILURES = {
    "binding_mismatch",
    "structured_output_error",
    "provider_error",
}
_GEOMETRY_STRATEGIES = {
    "parameter_change",
    "mode_change",
    "primitive_change",
    "topology_change",
    "rollback_earlier_step",
}
_SCOPE_RANK = {
    "params": 0,
    "variant": 1,
    "primitive": 2,
    "topology": 3,
    "rollback": 4,
}


class DiagnosticValidationError(ValueError):
    """A typed advisor response violates the host-owned evidence contract."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    payload = _jsonable(value)
    return dict(payload) if isinstance(payload, dict) else {}


def _as_dicts(values: Iterable[Any]) -> list[dict[str, Any]]:
    return [payload for value in values if (payload := _as_dict(value))]


def _walk(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            child_path = f"{path}/{escaped}"
            yield child_path, child
            yield from _walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}/{index}"
            yield child_path, child
            yield from _walk(child, child_path)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _failure_identity(value: Any) -> Any:
    """Keep failure-family identity while dropping candidate-specific numbers."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if key_text in _FAILURE_IDENTITY_KEYS:
                result[key_text] = _jsonable(child)
            elif key_text in _FAILURE_CHECKS and child not in (None, [], {}):
                result[key_text] = _failure_identity(child)
            elif key_text in {"failed_checks", "checks", "observations"}:
                nested = _failure_identity(child)
                if nested not in (None, [], {}):
                    result[key_text] = nested
        return result
    if isinstance(value, list):
        identities = [_failure_identity(child) for child in value]
        identities = [item for item in identities if item not in (None, [], {})]
        return sorted(identities, key=lambda item: json.dumps(item, sort_keys=True))
    return None


def failure_signature(value: Any) -> str:
    """Return a family-level signature suitable for retry deduplication."""

    if isinstance(value, StepRepairDiagnosticContext):
        return value.binding.failure_signature
    identity = _failure_identity(_jsonable(value))
    if identity in (None, [], {}):
        identity = {"failure_type": type(value).__name__}
    return _digest(identity)


def diagnostic_case_digest(case: StepRepairDiagnosticContext) -> str:
    """Bind the complete case presented to the diagnostician."""

    return _digest(case.model_dump(mode="json"))


def diagnostic_case_id(case: StepRepairDiagnosticContext) -> str:
    """Return the candidate-specific journal and artifact identity."""

    binding = case.binding
    return _digest(
        {
            "protocol_version": binding.protocol_version,
            "run_id": binding.run_id,
            "state_id": binding.state_id,
            "state_digest": binding.state_digest,
            "contract_digest": binding.contract_digest,
            "step_index": binding.step_index,
            "attempt_index": binding.attempt_index,
            "repair_epoch": binding.repair_epoch,
            "action_digest": binding.action_digest,
            "failure_signature": binding.failure_signature,
            "evidence_digest": binding.evidence_digest,
            "generator_version": binding.generator_version,
            "validator_policy_digest": binding.validator_policy_digest,
            "diagnostic_context_digest": diagnostic_case_digest(case),
        }
    )


def _compact_state(state: PipeState) -> dict[str, Any]:
    return {
        "state_id": state.state_id,
        "state_version": state.state_version,
        "contract_digest": state.contract_digest,
        "remaining_goal_ids": [goal.goal_id for goal in state.remaining_goals],
        "open_ports": [
            {
                "id": port.id,
                "position": list(port.position),
                "axis": list(port.axis),
                "outer_diameter": port.outer_diameter,
                "wall_thickness": port.wall_thickness,
            }
            for port in state.open_ports
        ],
        "placed_module_ids": [module.id for module in state.placed_modules],
        "connection_edges": [
            edge.model_dump(mode="json") for edge in state.connection_edges
        ],
    }


def _goal_slice(
    state: PipeState,
    draft: dict[str, Any],
    supplied: Any,
) -> dict[str, Any]:
    if supplied is not None:
        return _as_dict(supplied)
    goal_ids = {
        str(goal_id)
        for key in ("affected_goal_ids", "completed_goal_ids")
        for goal_id in (draft.get(key) or [])
    }
    goals = [
        goal.model_dump(mode="json", exclude_none=True)
        for goal in state.remaining_goals
        if not goal_ids or goal.goal_id in goal_ids
    ]
    return {
        "contract_digest": state.contract_digest,
        "goal_ids": sorted(goal_ids),
        "goals": goals,
        "expected_open_ports": state.expected_open_ports,
        "expected_open_ports_source": state.expected_open_ports_source,
    }


def _extract_validator_policy(
    evidence: dict[str, Any],
    supplied: Any,
    *,
    generator_version: str,
    validator_schema_version: int,
) -> dict[str, Any]:
    policy = _as_dict(supplied)
    if not policy:
        policy = _as_dict(evidence.get("validator_policy"))
    if not policy:
        policy = {
            "policy_id": VALIDATOR_POLICY_ID,
            "policy_digest": VALIDATOR_POLICY_DIGEST,
            "generator_version": generator_version,
            "validation_schema_version": validator_schema_version,
        }
    policy.setdefault("policy_id", VALIDATOR_POLICY_ID)
    policy.setdefault("generator_version", generator_version)
    policy.setdefault("validation_schema_version", validator_schema_version)
    policy.setdefault(
        "policy_digest",
        _digest(
            {key: value for key, value in policy.items() if key != "policy_digest"}
        ),
    )
    return policy


def _failed_checks(
    issues: list[dict[str, Any]], evidence: dict[str, Any]
) -> list[dict[str, Any]]:
    checks = evidence.get("checks")
    result: list[dict[str, Any]] = []
    if isinstance(checks, Mapping):
        for check_name in _FAILURE_CHECKS:
            value = checks.get(check_name)
            if check_name == "centerlines" and isinstance(value, Mapping):
                value = {
                    str(module_id): item
                    for module_id, item in value.items()
                    if isinstance(item, Mapping) and item.get("passed") is not True
                }
            if value not in (None, [], {}):
                result.append({"check_name": check_name, "evidence": _jsonable(value)})
    for issue in issues:
        name = str(issue.get("check_name") or issue.get("phase") or "unknown")
        payload = {
            "check_name": name,
            "issue_id": issue.get("issue_id"),
            "issue_code": issue.get("issue_code"),
            "expected": issue.get("expected") or {},
            "actual": issue.get("actual") or {},
        }
        if not any(
            item.get("issue_id") == payload["issue_id"] and payload["issue_id"]
            for item in result
        ):
            result.append(payload)
    return result or [{"check_name": "unknown", "evidence": evidence}]


def _passed_check_summary(evidence: dict[str, Any]) -> list[str]:
    checks = evidence.get("checks")
    if not isinstance(checks, Mapping):
        return []
    passed: list[str] = []
    for name, value in checks.items():
        if name in _FAILURE_CHECKS:
            if value in (None, [], {}):
                passed.append(str(name))
            continue
        if isinstance(value, Mapping):
            status = value.get("valid", value.get("passed", value.get("ok")))
            if status is True:
                passed.append(str(name))
    return sorted(set(passed))[:24]


def _module_ids(value: Any) -> set[str]:
    result: set[str] = set()
    for path, child in _walk(value):
        leaf = path.rsplit("/", 1)[-1]
        if leaf == "module_id" and isinstance(child, str):
            result.add(child)
        elif leaf in {"module_ids", "implicated_module_ids"} and isinstance(
            child, list
        ):
            result.update(str(item) for item in child if isinstance(item, str))
    return result


def _implicated_modules(
    state: PipeState,
    evidence: dict[str, Any],
    resolved: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    implicated = _module_ids(evidence)
    if resolved and isinstance(resolved.get("module_id"), str):
        implicated.add(str(resolved["module_id"]))
    modules = []
    for module in state.placed_modules:
        if not implicated or module.id in implicated:
            modules.append(module.model_dump(mode="json"))
    return modules[-8:]


def _pointer_token(value: Any) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _goal_records(goal_slice: dict[str, Any]) -> list[dict[str, Any]]:
    goals = goal_slice.get("goals")
    if isinstance(goals, list):
        return [dict(goal) for goal in goals if isinstance(goal, Mapping)]
    if isinstance(goal_slice.get("required_waypoints"), list):
        return [goal_slice]
    return []


def _same_waypoint(left: Any, right: Any) -> bool:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    if len(left) != 3 or len(right) != 3:
        return False
    left_values = [_finite_number(value) for value in left]
    right_values = [_finite_number(value) for value in right]
    if any(value is None for value in [*left_values, *right_values]):
        return False
    return all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
        for a, b in zip(left_values, right_values)
    )


def _required_waypoint_matches(
    params: Mapping[str, Any], goal_slice: dict[str, Any]
) -> dict[int, tuple[str, int]]:
    """Align immutable anchors as an ordered subsequence of authored points."""

    waypoints = params.get("waypoints")
    if not isinstance(waypoints, list):
        return {}
    draft_frame = params.get("waypoint_frame")
    required: list[tuple[str, int, Any]] = []
    for goal_index, goal in enumerate(_goal_records(goal_slice)):
        goal_frame = goal.get("waypoint_frame")
        if (
            goal_frame is not None
            and draft_frame is not None
            and goal_frame != draft_frame
        ):
            continue
        goal_id = str(goal.get("goal_id") or f"goal_{goal_index}")
        for required_index, point in enumerate(goal.get("required_waypoints") or []):
            required.append((goal_id, required_index, point))
    matches: dict[int, tuple[str, int]] = {}
    cursor = 0
    for goal_id, required_index, required_point in required:
        for candidate_index in range(cursor, len(waypoints)):
            if _same_waypoint(waypoints[candidate_index], required_point):
                matches[candidate_index] = (goal_id, required_index)
                cursor = candidate_index + 1
                break
    return matches


def _field_ownership(
    draft: dict[str, Any], goal_slice: dict[str, Any]
) -> list[FieldOwnership]:
    cards = [
        FieldOwnership(
            path="/target_port",
            owner="downstream_state_sensitive",
            mutable_in_current_repair=False,
            reason="Changing the attachment port changes committed graph topology.",
        ),
        FieldOwnership(
            path="/module",
            owner="planner_authored",
            mutable_in_current_repair=True,
            reason="A validated primitive/topology diagnosis may release the module choice.",
        ),
        FieldOwnership(
            path="/affected_goal_ids",
            owner="goal_derived_immutable",
            mutable_in_current_repair=False,
            reason="Goal attribution is bound to the immutable intent.",
        ),
        FieldOwnership(
            path="/completed_goal_ids",
            owner="goal_derived_immutable",
            mutable_in_current_repair=False,
            reason="A failed candidate cannot rewrite completion claims.",
        ),
    ]
    params = draft.get("params")
    if isinstance(params, Mapping):
        waypoint_matches = _required_waypoint_matches(params, goal_slice)
        cards.append(
            FieldOwnership(
                path="/params",
                owner="planner_authored",
                mutable_in_current_repair=True,
                reason="The planner authors the primitive parameter object.",
            )
        )
        waypoints = params.get("waypoints")
        if isinstance(waypoints, list):
            for index in range(len(waypoints)):
                required = index in waypoint_matches
                cards.append(
                    FieldOwnership(
                        path=f"/params/waypoints/{index}",
                        owner=(
                            "goal_derived_immutable" if required else "planner_authored"
                        ),
                        mutable_in_current_repair=not required,
                        reason=(
                            "This exact waypoint is an ordered immutable goal anchor."
                            if required
                            else "This smoothing waypoint was inserted by the planner."
                        ),
                    )
                )
        for path, child in _walk(params, "/params"):
            if isinstance(child, (Mapping, list)):
                continue
            waypoint_match = re.fullmatch(r"/params/waypoints/(\d+)(?:/\d+)?", path)
            required = bool(
                waypoint_match and int(waypoint_match.group(1)) in waypoint_matches
            )
            cards.append(
                FieldOwnership(
                    path=path,
                    owner=(
                        "goal_derived_immutable" if required else "planner_authored"
                    ),
                    mutable_in_current_repair=not required,
                    reason=(
                        "This coordinate belongs to an immutable required waypoint."
                        if required
                        else "This value was authored in the rejected draft."
                    ),
                )
            )
    cards.extend(
        [
            FieldOwnership(
                path="/resolved_action",
                owner="resolver_owned",
                mutable_in_current_repair=False,
                reason="Resolved positions, axes and derived geometry are recomputed by the host.",
            ),
            FieldOwnership(
                path="/validator_policy",
                owner="validator_policy",
                mutable_in_current_repair=False,
                reason="Validation policy is executable host code, never a planner knob.",
            ),
        ]
    )
    # Preserve first occurrence if a malformed draft creates duplicate paths.
    unique: dict[str, FieldOwnership] = {}
    for card in cards:
        unique.setdefault(card.path, card)
    return list(unique.values())


def _flatten_trial_params(
    action: dict[str, Any], goal_slice: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    params = action.get("params")
    if not isinstance(params, Mapping):
        return {}, {}
    flattened = {
        path: value
        for path, value in _walk(params, "/params")
        if not isinstance(value, (Mapping, list))
        and not path.startswith("/params/waypoints/")
    }
    current_paths = {path: path for path in flattened}
    waypoints = params.get("waypoints")
    if not isinstance(waypoints, list):
        return flattened, current_paths
    required_matches = _required_waypoint_matches(params, goal_slice)
    optional_ordinals: dict[int, int] = {}
    required_seen = 0
    for waypoint_index, waypoint in enumerate(waypoints):
        required_identity = required_matches.get(waypoint_index)
        if required_identity is not None:
            goal_id, required_index = required_identity
            base_path = (
                "/params/waypoints/@required/"
                f"{_pointer_token(goal_id)}/{required_index}"
            )
            required_seen += 1
        else:
            segment = required_seen
            ordinal = optional_ordinals.get(segment, 0)
            optional_ordinals[segment] = ordinal + 1
            base_path = f"/params/waypoints/@optional/{segment}/{ordinal}"
        if isinstance(waypoint, (list, tuple)):
            for coordinate_index, value in enumerate(waypoint):
                semantic_path = f"{base_path}/{coordinate_index}"
                flattened[semantic_path] = value
                current_paths[semantic_path] = (
                    f"/params/waypoints/{waypoint_index}/{coordinate_index}"
                )
    return flattened, current_paths


def _flatten_metrics(value: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    payload = _jsonable(value)

    def collect_validation_diagnostics(node: Any) -> None:
        if isinstance(node, Mapping):
            metric_name = node.get("metric")
            actual = _finite_number(node.get("actual"))
            if isinstance(metric_name, str) and metric_name and actual is not None:
                base = f"/validation_metrics/{_pointer_token(metric_name)}"
                # The measured value is the causal response.  Index-based
                # `/.../validation_diagnostics/0/gap` paths are deliberately
                # avoided because diagnostic ordering may change by attempt.
                metrics[f"{base}/actual"] = actual
                for key in ("required", "gap", "ratio"):
                    number = _finite_number(node.get(key))
                    if number is not None:
                        metrics[f"{base}/{key}"] = number
            for child in node.values():
                collect_validation_diagnostics(child)
        elif isinstance(node, list):
            for child in node:
                collect_validation_diagnostics(child)

    collect_validation_diagnostics(payload)
    for path, child in _walk(payload):
        leaf = path.rsplit("/", 1)[-1]
        number = _finite_number(child)
        if leaf in _METRIC_KEYS and number is not None:
            metrics[path] = number
    return metrics


def _parameter_trials(
    history: Sequence[Any], goal_slice: dict[str, Any]
) -> list[dict[str, Any]]:
    attempts: list[tuple[dict[str, Any], dict[str, float], dict[str, str]]] = []
    for item in history:
        payload = _as_dict(item)
        if payload.get("status") != "rejected":
            continue
        # draft 좌표는 planner의 불변 waypoint frame을 유지하므로 의미 단위로
        # 정렬할 수 있다. resolved 전역 index는 선택 점의 삽입에 따라 바뀌어
        # 안정적인 trial ID로 사용할 수 없다.
        action = payload.get("draft") or payload.get("resolved") or {}
        if not isinstance(action, dict):
            continue
        flattened, current_paths = _flatten_trial_params(action, goal_slice)
        attempts.append(
            (
                flattened,
                _flatten_metrics(payload.get("observations") or []),
                current_paths,
            )
        )
    if len(attempts) < 2:
        return []
    all_paths = sorted({path for params, _, _ in attempts for path in params})
    changed_paths = {
        path
        for path in all_paths
        if len({_digest(params.get(path)) for params, _, _ in attempts}) > 1
    }

    # 비교 metric은 파라미터 path와 무관하며 모든 trial에 공통이다. 변경된
    # 파라미터마다 같은 set 교집합과 metric span을 다시 계산하지 않는다.
    common_metric_paths = set.intersection(
        *(set(metrics) for _, metrics, _ in attempts)
    )
    preferred_metric_paths = sorted(
        path
        for path in common_metric_paths
        if path.startswith("/validation_metrics/") and path.endswith("/actual")
    )
    metric_path = (
        preferred_metric_paths[0]
        if preferred_metric_paths
        else next(iter(sorted(common_metric_paths)), None)
    )
    metric_values = (
        [metrics[metric_path] for _, metrics, _ in attempts] if metric_path else []
    )
    metric_span = max(metric_values) - min(metric_values) if metric_values else None
    metric_scale = max([1.0, *(abs(value) for value in metric_values)])
    metric_is_invariant = metric_span is not None and metric_span <= 1e-9 * metric_scale

    trials: list[dict[str, Any]] = []
    for path in sorted(changed_paths):
        # 시도마다 한 slot을 유지한다. ``None``은 optional waypoint/field가
        # 삽입ㆍ삭제됐다는 근거이므로 제거하면 metric 값과 정렬이 깨진다.
        values = [params.get(path) for params, _, _ in attempts]
        current_path = attempts[-1][2].get(path)
        finding = "not_comparable"
        if metric_is_invariant:
            finding = (
                "invariant_over_tested_range"
                if len(changed_paths) == 1
                else "invariant_under_joint_trial"
            )
        trials.append(
            {
                "path": path,
                "current_path": current_path,
                "values": values,
                "metric_id": metric_path,
                # 각 trial payload는 기존처럼 독립적으로 수정 가능한 목록을 가진다.
                "metric_values": list(metric_values),
                "metric_span": metric_span,
                "aligned_samples": [
                    {"parameter_value": value, "metric_value": metric_value}
                    for value, metric_value in zip(values, metric_values)
                ],
                "jointly_changed_paths": sorted(changed_paths - {path}),
                "finding": finding,
            }
        )
    return trials[:24]


def _facts(
    *,
    state: PipeState,
    draft: dict[str, Any],
    issues: list[dict[str, Any]],
    evidence: dict[str, Any],
    policy: dict[str, Any],
    failed_checks: list[dict[str, Any]],
    passed_checks: list[str],
    trials: list[dict[str, Any]],
) -> list[DiagnosticFact]:
    facts: list[DiagnosticFact] = []

    def add(kind: str, statement: str, data: dict[str, Any]) -> None:
        facts.append(
            DiagnosticFact(
                evidence_id=f"E{len(facts) + 1:03d}",
                kind=kind,  # type: ignore[arg-type]
                statement=statement,
                data=data,
            )
        )

    add(
        "immutable_contract",
        "The rejected transition remains bound to this state and immutable contract.",
        {
            "state_id": state.state_id,
            "contract_digest": state.contract_digest,
            "goal_ids": sorted(
                {
                    str(item)
                    for key in ("affected_goal_ids", "completed_goal_ids")
                    for item in (draft.get(key) or [])
                }
            ),
        },
    )
    add(
        "validator_policy",
        "The host validator policy is immutable for this case.",
        policy,
    )
    for issue in issues[:8]:
        add(
            "relationship",
            "A deterministic validator rejected the candidate.",
            {
                "issue_id": issue.get("issue_id"),
                "issue_code": issue.get("issue_code"),
                "check_name": issue.get("check_name"),
                "expected": issue.get("expected") or {},
                "actual": issue.get("actual") or {},
            },
        )
    for check in failed_checks[:12]:
        name = str(check.get("check_name") or "unknown")
        kind = "kernel_error" if name == "module_errors" else "measurement"
        add(kind, f"FreeCAD or host check {name} supplied failure evidence.", check)
    if passed_checks:
        add(
            "passed_check",
            "These checks passed for the same candidate and must not be casually invalidated.",
            {"check_names": passed_checks},
        )
    for trial in trials[:12]:
        add(
            "attempt_delta",
            "Rejected attempts expose an observed parameter/metric response.",
            trial,
        )
    params = draft.get("params") or {}
    if draft.get("module") == "junction" and params.get("blend_mode") == "hard":
        add(
            "parameter_effect",
            "In the v21 hard-junction builder max_hub_radius does not construct the local material or define overlap allowance.",
            {
                "policy_id": "resolver-local-interface-band-v1",
                "generator_version": GENERATOR_VERSION,
                "parameter_path": "/params/max_hub_radius",
                "construction_effect": "none_in_hard_material_builder",
                "failed_metric_effect": "none_on_local_interface_allowance",
            },
        )
    # Preserve the top-level policy metadata even if evidence compaction omitted it.
    if evidence.get("validator_policy") and evidence.get("validator_policy") != policy:
        add(
            "validator_policy",
            "The evidence policy metadata differs from the active validator policy.",
            {
                "evidence_policy": evidence.get("validator_policy"),
                "active_policy": policy,
            },
        )
    return facts


def _allowed_strategies(
    issues: list[dict[str, Any]], failed_checks: list[dict[str, Any]]
) -> list[RepairStrategyKind]:
    issue_text = json.dumps(issues, ensure_ascii=False).lower()
    if any(marker in issue_text for marker in _INFRASTRUCTURE_MARKERS):
        return ["infrastructure_retry"]
    check_names = {str(item.get("check_name")) for item in failed_checks}
    result: list[RepairStrategyKind] = ["parameter_change"]
    if "module_errors" in check_names:
        result.extend(["mode_change", "primitive_change", "kernel_review"])
    if "non_adjacent_overlaps" in check_names:
        result.extend(
            ["mode_change", "primitive_change", "topology_change", "validator_review"]
        )
    if "deterministic_constraint_failures" in check_names:
        result.extend(["primitive_change", "rollback_earlier_step"])
    if len(result) == 1:
        result.extend(["mode_change", "primitive_change"])
    return list(dict.fromkeys(result))


def build_diagnostic_case(
    *,
    run_id: str,
    state: PipeState,
    step_index: int,
    attempt_index: int,
    repair_epoch: int,
    draft: Any,
    issues: Sequence[Any],
    evidence: Any,
    resolved_action: Any = None,
    immutable_goal_slice: Any = None,
    attempt_history: Sequence[Any] = (),
    deterministic_recommendations: Sequence[Any] = (),
    generator_version: str = GENERATOR_VERSION,
    validator_schema_version: int = VALIDATION_SCHEMA_VERSION,
    validator_policy: Any = None,
) -> StepRepairDiagnosticContext:
    """Build the exact typed input for one rejected candidate."""

    draft_payload = _as_dict(draft)
    issue_payloads = _as_dicts(issues)
    evidence_payload = _as_dict(evidence)
    resolved_payload = _as_dict(resolved_action) or None
    policy = _extract_validator_policy(
        evidence_payload,
        validator_policy,
        generator_version=generator_version,
        validator_schema_version=validator_schema_version,
    )
    checks = _failed_checks(issue_payloads, evidence_payload)
    passed = _passed_check_summary(evidence_payload)
    goal_slice = _goal_slice(state, draft_payload, immutable_goal_slice)
    trials = _parameter_trials(attempt_history, goal_slice)
    recommendations = _as_dicts(deterministic_recommendations)
    facts = _facts(
        state=state,
        draft=draft_payload,
        issues=issue_payloads,
        evidence=evidence_payload,
        policy=policy,
        failed_checks=checks,
        passed_checks=passed,
        trials=trials,
    )
    issue_ids = [
        str(item.get("issue_id") or item.get("issue_code") or f"ISSUE_{index}")
        for index, item in enumerate(issue_payloads, start=1)
    ] or [f"STEP_{step_index:04d}_ATTEMPT_{attempt_index:02d}_REJECTED"]
    failure_payload = {
        "issues": issue_payloads,
        "failed_checks": checks,
        "validator_policy": policy,
    }
    contract_digest = state.contract_digest or _digest(
        {
            "global_spec": state.global_spec.model_dump(mode="json"),
            "remaining_goals": [
                goal.model_dump(mode="json") for goal in state.remaining_goals
            ],
        }
    )
    binding = DiagnosticBinding(
        run_id=run_id,
        state_id=state.state_id,
        state_digest=_digest(state.model_dump(mode="json")),
        contract_digest=contract_digest,
        step_index=step_index,
        attempt_index=attempt_index,
        action_digest=_digest(
            {"draft": draft_payload, "resolved_action": resolved_payload}
        ),
        failure_signature=failure_signature(failure_payload),
        evidence_digest=_digest(evidence_payload),
        generator_version=generator_version,
        validator_schema_version=validator_schema_version,
        validator_policy_digest=str(policy.get("policy_digest")),
        repair_epoch=repair_epoch,
    )
    return StepRepairDiagnosticContext(
        binding=binding,
        issue_ids=list(dict.fromkeys(issue_ids))[:8],
        current_state=_compact_state(state),
        immutable_goal_slice=goal_slice,
        rejected_draft=draft_payload,
        resolved_action=resolved_payload,
        implicated_modules=_implicated_modules(
            state, evidence_payload, resolved_payload
        ),
        failed_checks=checks,
        passed_check_summary=passed,
        facts=facts,
        field_ownership=_field_ownership(draft_payload, goal_slice),
        parameter_trials=trials,
        deterministic_recommendations=recommendations,
        allowed_strategy_kinds=_allowed_strategies(issue_payloads, checks),
    )


def _journal_step_key(case: StepRepairDiagnosticContext) -> str:
    return f"{case.binding.step_index}:{case.binding.repair_epoch}"


def should_call_advisor(
    case: StepRepairDiagnosticContext,
    journal: DiagnosticJournal | None = None,
    *,
    enabled: bool = True,
    dry_run: bool = False,
    freecad_enabled: bool = True,
    trigger_attempt: int = 1,
    max_calls_per_step: int = 1,
    max_signatures_per_step: int = 2,
) -> bool:
    """Apply the first-reject trigger plus bounded per-epoch family caps."""

    # Registry/static constraint analysis is useful even when live FreeCAD is
    # disabled. ``freecad_enabled`` remains in the compatibility signature but
    # is no longer a blanket gate for the independent advisor.
    if not enabled or dry_run:
        return False
    journal = journal or DiagnosticJournal()
    key = _journal_step_key(case)
    if journal.calls_by_step.get(key, 0) >= max_calls_per_step:
        return False
    records = [
        record
        for record in journal.records
        if record.step_index == case.binding.step_index
        and record.repair_epoch == case.binding.repair_epoch
    ]
    terminal_records = [
        record
        for record in records
        if record.status in {"pending", "complete"}
        or (
            record.status == "failed"
            and record.failure_reason not in _RETRYABLE_ADVISOR_FAILURES
        )
    ]
    signatures = {record.failure_signature for record in terminal_records}
    if (
        case.binding.failure_signature not in signatures
        and len(signatures) >= max_signatures_per_step
    ):
        return False
    serialized = json.dumps(
        case.failed_checks, ensure_ascii=False, separators=(",", ":")
    ).lower()
    if any(marker in serialized for marker in _INFRASTRUCTURE_MARKERS):
        return False
    # A machine-readable inequality with actual/required values is exactly the
    # inverse-parameter advisor's fast path. Generic prose suggestions must not
    # delay this case until another planner attempt has already been spent.
    if (
        '"metric":' in serialized
        and '"actual":' in serialized
        and '"required":' in serialized
        and '"comparator":' in serialized
    ):
        return True
    if not case.deterministic_recommendations:
        return True
    return case.binding.attempt_index >= trigger_attempt


def _ownership_map(case: StepRepairDiagnosticContext) -> dict[str, FieldOwnership]:
    return {card.path: card for card in case.field_ownership}


def _closest_ownership(
    ownership: dict[str, FieldOwnership], path: str
) -> FieldOwnership | None:
    current = path
    while current:
        if current in ownership:
            return ownership[current]
        current = current.rsplit("/", 1)[0]
    return None


def _assert_mutable_path(
    path: str,
    ownership: dict[str, FieldOwnership],
    *,
    directive: str,
) -> None:
    card = _closest_ownership(ownership, path)
    if card is None:
        raise DiagnosticValidationError(f"unknown diagnostic field path: {path}")
    if directive in {"change", "change_mode_first"} and not (
        card.owner == "planner_authored" and card.mutable_in_current_repair
    ):
        raise DiagnosticValidationError(
            f"advisor cannot change {card.owner} field: {path}"
        )


def _structured_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            yield number
        return
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _structured_numbers(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _structured_numbers(child)


def _number_is_traced(number: float, evidence_values: Iterable[float]) -> bool:
    return any(
        math.isclose(
            number,
            candidate,
            rel_tol=1e-9,
            abs_tol=1e-9 * max(1.0, abs(number), abs(candidate)),
        )
        for candidate in evidence_values
    )


def bind_diagnosis(
    case: StepRepairDiagnosticContext,
    body: StepRepairDiagnosisBody,
) -> StepRepairDiagnosis:
    """Inject exact host identity into an LLM-authored diagnosis body."""

    binding = case.binding
    diagnosis = StepRepairDiagnosis(
        protocol_version=binding.protocol_version,
        state_id=binding.state_id,
        contract_digest=binding.contract_digest,
        diagnostic_context_digest=diagnostic_case_digest(case),
        failure_signature=binding.failure_signature,
        issue_ids=case.issue_ids,
        **body.model_dump(mode="python"),
    )
    return validate_diagnosis(case, diagnosis)


def _advisor_wire_number(value: str, label: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        number = float(Decimal(text))
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise DiagnosticValidationError(
            f"advisor {label} is not a finite decimal"
        ) from exc
    if not math.isfinite(number):
        raise DiagnosticValidationError(f"advisor {label} is not finite")
    return number


def _validate_advisor_wire_semantics(
    response: GeometryValidationAdvisorResponse,
) -> None:
    """Validate cross-field advisor meaning after the provider wire boundary."""

    def require_nonblank(value: str, label: str) -> None:
        if not value.strip():
            raise DiagnosticValidationError(f"advisor {label} must not be blank")

    def require_unique_nonblank(values: list[str], label: str) -> None:
        if any(not value.strip() for value in values):
            raise DiagnosticValidationError(
                f"advisor {label} must not contain blank values"
            )
        if len(values) != len(set(values)):
            raise DiagnosticValidationError(
                f"advisor {label} must contain unique values"
            )

    require_nonblank(response.summary, "summary")
    require_nonblank(response.planner_instruction, "planner instruction")
    require_unique_nonblank(response.causal_chain, "causal chain")
    require_unique_nonblank(response.evidence_ids, "evidence IDs")
    require_unique_nonblank(response.verification_checks, "verification checks")
    require_unique_nonblank(response.missing_evidence, "missing evidence")

    paths = [item.path for item in response.recommendations]
    if len(paths) != len(set(paths)):
        raise DiagnosticValidationError("advisor recommendations must use unique paths")
    declared_evidence = set(response.evidence_ids)
    for item in response.recommendations:
        require_nonblank(item.path, "recommendation path")
        require_nonblank(item.unit, "recommendation unit")
        require_nonblank(item.rationale, "recommendation rationale")
        require_nonblank(item.evidence_id, "recommendation evidence ID")
        if item.evidence_id not in declared_evidence:
            raise DiagnosticValidationError(
                "advisor recommendation cites undeclared evidence"
            )
        required_text = {
            "none": (),
            "lower": (item.lower_text,),
            "upper": (item.upper_text,),
            "closed": (item.lower_text, item.upper_text),
        }[item.bound_mode]
        if any(not value.strip() for value in required_text):
            raise DiagnosticValidationError(
                "advisor recommendation bound text is missing"
            )

    if response.disposition in {"retry_planner", "change_primitive_or_mode"}:
        if response.strategy_kind == "none":
            raise DiagnosticValidationError("advisor retry requires a primary strategy")
        if response.strategy_kind in {
            "parameter_change",
            "mode_change",
        } and not any(
            item.action in {"increase", "decrease", "replace"}
            for item in response.recommendations
        ):
            raise DiagnosticValidationError(
                "advisor parameter/mode retry requires actionable guidance"
            )
    if (
        response.diagnosis_class == "candidate_parameter"
        and response.disposition == "retry_planner"
        and not any(
            item.action in {"increase", "decrease", "replace"}
            for item in response.recommendations
        )
    ):
        raise DiagnosticValidationError(
            "candidate parameter retry requires a change direction"
        )
    if response.strategy_kind != "none":
        require_nonblank(response.strategy_instruction, "strategy instruction")
        if not response.verification_checks:
            raise DiagnosticValidationError(
                "advisor strategy requires a verification check"
            )


def bind_advisor_response(
    case: StepRepairDiagnosticContext,
    response: GeometryValidationAdvisorResponse,
) -> StepRepairDiagnosis:
    """Reconstruct the rich host diagnosis from the provider-safe wire DTO."""

    _validate_advisor_wire_semantics(response)
    evidence_uses = [
        DiagnosticEvidenceUse(
            evidence_id=evidence_id,
            supports=response.summary,
        )
        for evidence_id in response.evidence_ids
    ]
    parameter_causality: list[ParameterCausality] = []
    direction_guidance: list[ParameterDirectionRecommendation] = []
    parameter_ranges: list[ParameterRangeRecommendation] = []
    for item in response.recommendations:
        if item.action in {"increase", "decrease", "replace"}:
            influence = "direct"
            metric_response = "not_comparable"
            directive = "change"
        elif item.action == "avoid":
            influence = "non_causal"
            metric_response = "unchanged"
            directive = "avoid_repeating"
        elif item.action == "keep":
            influence = "conditional"
            metric_response = "not_comparable"
            directive = "keep"
        elif item.action == "collect_probe":
            influence = "unproven"
            metric_response = "not_comparable"
            directive = "collect_probe"
        else:
            influence = "unproven"
            metric_response = "not_comparable"
            directive = "unknown"
        parameter_causality.append(
            ParameterCausality(
                parameter_path=item.path,
                influence=influence,
                observed_metric_response=metric_response,
                directive=directive,
                explanation=item.rationale,
                evidence_ids=[item.evidence_id],
            )
        )
        if item.action in {"increase", "decrease", "replace", "keep", "avoid"}:
            direction_guidance.append(
                ParameterDirectionRecommendation(
                    path=item.path,
                    direction=item.action,
                    rationale=item.rationale,
                    evidence_ids=[item.evidence_id],
                )
            )
        if item.bound_mode == "none":
            continue
        lower = _advisor_wire_number(item.lower_text, "lower bound")
        upper = _advisor_wire_number(item.upper_text, "upper bound")
        preferred = _advisor_wire_number(item.preferred_text, "preferred value")
        if item.bound_mode == "lower":
            upper = None
        elif item.bound_mode == "upper":
            lower = None
        parameter_ranges.append(
            ParameterRangeRecommendation(
                path=item.path,
                lower=lower,
                upper=upper,
                preferred=preferred,
                unit=item.unit,
                classification=item.classification,
                rationale=item.rationale,
                evidence_ids=[item.evidence_id],
            )
        )

    strategies: list[RepairStrategy] = []
    if response.strategy_kind != "none":
        strategy_targets = [
            item.path
            for item in response.recommendations
            if item.action in {"increase", "decrease", "replace"}
        ]
        strategies.append(
            RepairStrategy(
                priority=1,
                kind=response.strategy_kind,
                target_fields=strategy_targets,
                instruction=response.strategy_instruction,
                expected_effect=response.summary,
                verification_checks=response.verification_checks,
                risks=response.missing_evidence[:6],
            )
        )
    body = StepRepairDiagnosisBody(
        diagnosis_class=response.diagnosis_class,
        disposition=response.disposition,
        confidence=response.confidence,
        summary=response.summary,
        causal_chain=response.causal_chain,
        evidence_uses=evidence_uses,
        parameter_causality=parameter_causality,
        direction_guidance=direction_guidance,
        parameter_ranges=parameter_ranges,
        strategies=strategies,
        missing_evidence=response.missing_evidence,
        planner_instruction=response.planner_instruction,
    )
    return bind_diagnosis(case, body)


def validate_diagnosis(
    case: StepRepairDiagnosticContext,
    diagnosis: StepRepairDiagnosis,
) -> StepRepairDiagnosis:
    """Apply host-owned binding, evidence, ownership and authority checks."""

    binding = case.binding
    expected = {
        "protocol_version": binding.protocol_version,
        "state_id": binding.state_id,
        "contract_digest": binding.contract_digest,
        "diagnostic_context_digest": diagnostic_case_digest(case),
        "failure_signature": binding.failure_signature,
    }
    actual = {
        "protocol_version": diagnosis.protocol_version,
        "state_id": diagnosis.state_id,
        "contract_digest": diagnosis.contract_digest,
        "diagnostic_context_digest": diagnosis.diagnostic_context_digest,
        "failure_signature": diagnosis.failure_signature,
    }
    if actual != expected:
        raise DiagnosticValidationError("diagnosis binding does not match its case")
    known_issue_ids = set(case.issue_ids)
    if not set(diagnosis.issue_ids) <= known_issue_ids:
        raise DiagnosticValidationError("diagnosis references an unknown issue ID")
    known_evidence_ids = {fact.evidence_id for fact in case.facts}
    cited = {use.evidence_id for use in diagnosis.evidence_uses}
    causal_citations = {
        evidence_id
        for item in diagnosis.parameter_causality
        for evidence_id in item.evidence_ids
    }
    direction_citations = {
        evidence_id
        for item in diagnosis.direction_guidance
        for evidence_id in item.evidence_ids
    }
    if (
        not cited <= known_evidence_ids
        or not causal_citations <= known_evidence_ids
        or not direction_citations <= known_evidence_ids
    ):
        raise DiagnosticValidationError("diagnosis references unknown evidence")
    fact_by_id = {fact.evidence_id: fact for fact in case.facts}
    ownership = _ownership_map(case)
    directives: dict[str, str] = {}
    for item in diagnosis.parameter_causality:
        _assert_mutable_path(
            item.parameter_path,
            ownership,
            directive=item.directive,
        )
        previous = directives.setdefault(item.parameter_path, item.directive)
        if previous != item.directive:
            raise DiagnosticValidationError(
                f"conflicting directives for {item.parameter_path}"
            )
    for item in diagnosis.direction_guidance:
        _assert_mutable_path(
            item.path,
            ownership,
            directive=(
                "change"
                if item.direction in {"increase", "decrease", "replace"}
                else "keep"
            ),
        )
    for recommendation in diagnosis.parameter_ranges:
        _assert_mutable_path(
            recommendation.path,
            ownership,
            directive="change",
        )
        if not set(recommendation.evidence_ids) <= known_evidence_ids:
            raise DiagnosticValidationError(
                "parameter range references unknown evidence"
            )
        cited_values = [
            number
            for evidence_id in recommendation.evidence_ids
            for number in _structured_numbers(fact_by_id[evidence_id].data)
        ]
        if not cited_values and any(
            value is not None
            for value in (
                recommendation.lower,
                recommendation.upper,
                recommendation.preferred,
            )
        ):
            raise DiagnosticValidationError(
                "parameter range has no cited numeric evidence"
            )
        inferred_limit = 10.0 * max([1.0, *(abs(value) for value in cited_values)])
        for value in (
            recommendation.lower,
            recommendation.upper,
            recommendation.preferred,
        ):
            if value is None:
                continue
            if recommendation.classification in {"feasible", "avoid"}:
                if not _number_is_traced(value, cited_values):
                    raise DiagnosticValidationError(
                        "tested feasible/avoid range value is not traceable to "
                        f"cited evidence: {recommendation.path}={value:g}"
                    )
            elif abs(value) > inferred_limit:
                raise DiagnosticValidationError(
                    "inferred parameter range exceeds the evidence-scaled safety "
                    f"envelope: {recommendation.path}={value:g}"
                )
    allowed_kinds = set(case.allowed_strategy_kinds)
    for strategy in diagnosis.strategies:
        if strategy.kind not in allowed_kinds:
            raise DiagnosticValidationError(
                f"strategy {strategy.kind} is not allowed for this case"
            )
        for path in strategy.target_fields:
            _assert_mutable_path(path, ownership, directive="change")
    if (
        diagnosis.disposition in {"retry_planner", "change_primitive_or_mode"}
        and not diagnosis.strategies
    ):
        raise DiagnosticValidationError("planner retry requires a bounded strategy")
    if diagnosis.diagnosis_class == "infrastructure_failure" and any(
        strategy.kind in _GEOMETRY_STRATEGIES for strategy in diagnosis.strategies
    ):
        raise DiagnosticValidationError(
            "infrastructure diagnosis cannot prescribe geometry changes"
        )
    text = " ".join(
        [
            diagnosis.summary,
            diagnosis.planner_instruction,
            *(strategy.instruction for strategy in diagnosis.strategies),
            *(item.rationale for item in diagnosis.direction_guidance),
        ]
    ).lower()
    forbidden = (
        "waive",
        "bypass",
        "mark passed",
        "accept candidate",
        "disable validator",
    )
    if any(marker in text for marker in forbidden):
        raise DiagnosticValidationError("diagnosis attempts to bypass validation")
    if (
        diagnosis.diagnosis_class == "validator_policy_mismatch"
        and diagnosis.disposition
        not in {
            "escalate_validator_review",
            "stop_futile_retry",
            "collect_more_evidence",
        }
    ):
        raise DiagnosticValidationError(
            "validator policy mismatch cannot directly retry or accept geometry"
        )
    # Exact numeric targets must be traceable to the supplied case.  Small list
    # ordinals are ignored; decimal and multi-digit values are evidence claims.
    case_text = json.dumps(case.model_dump(mode="json"), ensure_ascii=False)
    numeric_claims = re.findall(
        r"(?<![A-Za-z0-9_])-?(?:\d{2,}|\d+\.\d+)(?![A-Za-z0-9_])", text
    )
    for claim in numeric_claims:
        if claim not in case_text:
            raise DiagnosticValidationError(
                f"diagnosis invents an unsupplied numeric target: {claim}"
            )
    return diagnosis


def planner_directive_from_diagnosis(
    diagnosis: StepRepairDiagnosis,
    case: StepRepairDiagnosticContext | None = None,
) -> dict[str, Any]:
    """Compact a validated diagnosis without granting it action authority."""

    if case is not None:
        validate_diagnosis(case, diagnosis)
    scope = "params"
    for strategy in diagnosis.strategies:
        candidate_scope = {
            "parameter_change": "params",
            "mode_change": "variant",
            "primitive_change": "primitive",
            "topology_change": "topology",
            "rollback_earlier_step": "rollback",
        }.get(strategy.kind, "params")
        if _SCOPE_RANK[candidate_scope] > _SCOPE_RANK[scope]:
            scope = candidate_scope
    change_fields = sorted(
        {
            *(
                item.parameter_path
                for item in diagnosis.parameter_causality
                if item.directive in {"change", "change_mode_first"}
            ),
            *(
                field
                for strategy in diagnosis.strategies
                for field in strategy.target_fields
            ),
            *(
                item.path
                for item in diagnosis.parameter_ranges
                if item.classification in {"feasible", "promising"}
            ),
        }
    )
    keep_fields = sorted(
        item.parameter_path
        for item in diagnosis.parameter_causality
        if item.directive == "keep"
    )
    avoid_fields = sorted(
        {
            *(
                item.parameter_path
                for item in diagnosis.parameter_causality
                if item.directive in {"avoid_repeating", "change_mode_first"}
            ),
            *(
                item.path
                for item in diagnosis.parameter_ranges
                if item.classification == "avoid"
            ),
        }
    )
    return {
        "context_type": "step_geometry_diagnosis",
        "state_id": diagnosis.state_id,
        "diagnostic_context_digest": diagnosis.diagnostic_context_digest,
        "failure_signature": diagnosis.failure_signature,
        "diagnosis_class": diagnosis.diagnosis_class,
        "disposition": diagnosis.disposition,
        "confidence": diagnosis.confidence,
        "repair_scope": scope,
        "change_fields": change_fields,
        "keep_fields": keep_fields,
        "avoid_repeating_fields": avoid_fields,
        "direction_guidance": [
            item.model_dump(mode="json") for item in diagnosis.direction_guidance
        ],
        "parameter_ranges": [
            item.model_dump(mode="json") for item in diagnosis.parameter_ranges
        ],
        "strategies": [
            strategy.model_dump(mode="json")
            for strategy in sorted(diagnosis.strategies, key=lambda item: item.priority)
        ],
        # Keep the model prose as explanation only.  The next Action remains
        # planner-authored, and only the validated structured fields above are
        # allowed to steer its geometry choices.
        "advisor_explanation": diagnosis.planner_instruction,
        "planner_instruction": (
            "Select and author the next complete Action using the structured "
            "direction_guidance, parameter_ranges, and strategies in this "
            "diagnosis. Preserve keep_fields, do not repeat avoid fields, and "
            "revalidate the resulting candidate through every existing gate."
        ),
        "planner_selection_authority": (
            "Ranges are advisory evidence only. The ordinary step planner must "
            "choose every final exact parameter and author the replacement Action."
        ),
        "validation_authority": (
            "This diagnosis cannot accept the candidate or waive any check. "
            "The replacement action must pass every existing validator again."
        ),
    }


__all__ = [
    "DiagnosticValidationError",
    "bind_advisor_response",
    "bind_diagnosis",
    "build_diagnostic_case",
    "diagnostic_case_digest",
    "diagnostic_case_id",
    "failure_signature",
    "planner_directive_from_diagnosis",
    "should_call_advisor",
    "validate_diagnosis",
]
