"""durable checkpoint의 digest, history와 불변 계약을 검증한다."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from cadgen.typed_data_models import (
    IntentResult,
    PipeState,
    ResolvedAction,
    StepVerification,
)
from cadgen.static_geometry_validator import build_step_verification, has_errors


def _validate_checkpoint_state(
    state: PipeState,
    intent: IntentResult,
    expected_digest: Any,
    actions: list[dict[str, Any]],
) -> None:
    """입력·상태가 계약을 만족하는지 검증한다."""

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
    """checkpoint_history를 계산하거나 반환한다."""

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
    *,
    validation_enforcement: str = "strict",
) -> None:
    """입력·상태가 계약을 만족하는지 검증한다."""

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
            validation_enforcement=validation_enforcement,
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
    """pipe_state_digest를 계산하거나 반환한다."""

    return _canonical_json_digest(state.model_dump(mode="json"))

def _canonical_json_digest(value: Any) -> str:
    """canonical_json_digest를 계산하거나 반환한다."""

    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

