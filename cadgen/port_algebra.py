"""Port-Algebraic Progressive Realization (PAPR) 핵심 대수.

LLM이 고른 의미 primitive/파라미터를 입력으로, 호스트가 포트 프레임과
이산 결합을 결정론적으로 계산한다. 연속 좌표를 LLM이 다시 추측하지 않도록
하는 경계 계층이다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

from cadgen.typed_data_models import Goal, PipeState, Port
from cadgen.vector3_math import direction_to_vector, dot, length, normalize, vec

Vector3 = tuple[float, float, float]

_PRIMARY_OUT_RE = re.compile(r"\.out$")
_BRANCH_OUT_RE = re.compile(r"\.out_\d+$")

# Goals that continue a main construction path (prefer primary outlet).
_MAIN_PATH_GOAL_TYPES: Final[frozenset[str]] = frozenset(
    {
        "move",
        "turn",
        "route",
        "diameter_change",
        "branch",
        "connector",
        "end",
    }
)

_AXIS_ALIGN_MIN = 0.85
_AXIS_ALIGN_MARGIN = 0.05

_HOST_OWNED_CONTINUOUS_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "start_position",
        "end_position",
        "axis",
        "out_axis",
        "direction",
        "midpoint",
        "waypoints",
        "plane_normal",
        "bend_radius",
        "sweep_angle",
        "trunk_end",
    }
)


@dataclass(frozen=True)
class PortFrame:
    """접합 불변식을 담는 포트 프레임."""

    port_id: str
    origin: Vector3
    tangent: Vector3
    outer_diameter: float
    wall_thickness: float

    @classmethod
    def from_port(cls, port: Port) -> "PortFrame":
        """런타임 Port를 대수 프레임으로 변환한다."""

        return cls(
            port_id=port.id,
            origin=tuple(float(v) for v in port.position),
            tangent=tuple(float(v) for v in normalize(vec(port.axis))),
            outer_diameter=float(port.outer_diameter),
            wall_thickness=float(port.wall_thickness),
        )


@dataclass(frozen=True)
class ProgressCertificate:
    """유한 탐색을 강제하는 진행 증명."""

    remaining_goals: int
    open_ports: int
    completed_modules: int
    state_digest_prefix: str

    def strictly_progressed_from(self, previous: "ProgressCertificate") -> bool:
        """이전 수락 상태 대비 단조 진행했는지 판정한다."""

        if self.remaining_goals < previous.remaining_goals:
            return True
        return (
            self.remaining_goals == previous.remaining_goals
            and self.completed_modules > previous.completed_modules
        )


def progress_from_state(
    state: PipeState, *, state_digest: str = ""
) -> ProgressCertificate:
    """PipeState에서 진행 증명을 만든다."""

    return ProgressCertificate(
        remaining_goals=len(state.remaining_goals),
        open_ports=len(state.open_ports),
        completed_modules=len(state.placed_modules),
        state_digest_prefix=(state_digest or state.state_id)[:16],
    )


def discrete_search_budget(state: PipeState, *, variants_per_goal: int = 4) -> int:
    """이산 후보 상한을 계산한다. 이 값을 넘는 continuous walk는 금지한다."""

    n_goals = max(1, len(state.remaining_goals))
    n_ports = max(1, len(state.open_ports))
    return max(8, n_goals * n_ports * max(1, variants_per_goal))


def is_primary_port_id(port_id: str) -> bool:
    """junction primary outlet 식별자인지 판정한다."""

    return bool(_PRIMARY_OUT_RE.search(port_id)) and not bool(
        _BRANCH_OUT_RE.search(port_id)
    )


def is_branch_port_id(port_id: str) -> bool:
    """junction branch outlet 식별자인지 판정한다."""

    return bool(_BRANCH_OUT_RE.search(port_id))


def _unique_primary(opens: list[Port]) -> Port | None:
    primaries = [port for port in opens if is_primary_port_id(port.id)]
    return primaries[0] if len(primaries) == 1 else None


def _axis_aligned_port(opens: list[Port], desired: Vector3) -> Port | None:
    """desired 축에 뚜렷하게 정렬된 유일 포트를 고른다."""

    ranked = sorted(
        opens,
        key=lambda port: -dot(normalize(vec(port.axis)), desired),
    )
    best = ranked[0]
    best_score = dot(normalize(vec(best.axis)), desired)
    second_score = (
        dot(normalize(vec(ranked[1].axis)), desired) if len(ranked) > 1 else -2.0
    )
    if best_score >= _AXIS_ALIGN_MIN and (best_score - second_score) >= _AXIS_ALIGN_MARGIN:
        return best
    return None


def select_construction_port(state: PipeState, goal: Goal) -> Port | None:
    """다음 goal을 붙일 입력 포트를 결정론적으로 고른다.

    유일하지 않아 host가 안전하게 고를 수 없으면 None을 반환한다.
    None은 '이산 LLM 선택 필요' 신호이며, 연속 수치 LLM 위임 신호가 아니다.
    """

    opens = list(state.open_ports)
    if not opens:
        return None
    if len(opens) == 1:
        return opens[0]

    # Prefer an open port that already faces the goal's world direction before
    # defaulting to junction primary (primary may be diagonal after a Y hub).
    desired = _goal_desired_axis(goal)
    if desired is not None:
        aligned = _axis_aligned_port(opens, desired)
        if aligned is not None:
            return aligned

    primary = _unique_primary(opens)
    if goal.type in _MAIN_PATH_GOAL_TYPES and primary is not None:
        if desired is None:
            return primary
        # Primary only if it is not fighting the requested direction.
        if dot(normalize(vec(primary.axis)), desired) >= _AXIS_ALIGN_MIN:
            return primary

    if primary is not None and goal.type != "connect" and desired is None:
        return primary

    non_branch = [port for port in opens if not is_branch_port_id(port.id)]
    if len(non_branch) == 1:
        return non_branch[0]
    return None


def _goal_desired_axis(goal: Goal) -> Vector3 | None:
    """goal이 암시하는 진행 축이 있으면 정규화 벡터를 반환한다."""

    if goal.direction is not None:
        try:
            return normalize(vec(direction_to_vector(goal.direction)))
        except Exception:
            return None
    if goal.terminal_axis is not None:
        axis = vec(goal.terminal_axis)
        if length(axis) > 1e-12:
            return normalize(axis)
    # required_outlet_vectors usually describe terminal arms, not the main path.
    return None


def repair_allows_host_recompile(
    repair_observations: list[dict[str, Any]] | None,
) -> bool:
    """repair 중에도 host continuous recompile을 허용할지 판정한다.

    PAPR: 연속 수치 소유권은 host에 고정. observation은 이산 대안 탐색 힌트일 뿐
    host compiler를 끄지 않는다.
    """

    del repair_observations
    return True


def continuous_fields_owned_by_host() -> frozenset[str]:
    """LLM step repair가 직접 흔들면 안 되는 필드 집합."""

    return _HOST_OWNED_CONTINUOUS_FIELDS


__all__ = [
    "PortFrame",
    "ProgressCertificate",
    "continuous_fields_owned_by_host",
    "discrete_search_budget",
    "is_branch_port_id",
    "is_primary_port_id",
    "progress_from_state",
    "repair_allows_host_recompile",
    "select_construction_port",
]
