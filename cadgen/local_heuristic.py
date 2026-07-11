"""API 없이 실행하는 제한적 dry-run 의도/행동 fixture를 제공한다.

간단한 prompt 또는 ``PipeState``를 입력받아 legacy intent/action 초안을 반환한다.
프로덕션 실패의 대체 경로로 사용하지 않으며 표현할 수 없는 요청은 거부한다.
"""

from __future__ import annotations

import re

from cadgen.config import Settings
from cadgen.schemas import (
    ActionDraft,
    Direction,
    GlobalSpec,
    Goal,
    IntentResult,
    PipeState,
)


def _compile_patterns(*patterns: str) -> tuple[re.Pattern[str], ...]:
    """고정 정규식을 대소문자 무시 pattern tuple로 한 번만 컴파일한다."""

    return tuple(re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns)


_OUTER_DIAMETER_PATTERNS = _compile_patterns(
    r"(?:diameter|od|outer diameter)\D{0,20}(\d+(?:\.\d+)?)",
    r"(\d+(?:\.\d+)?)\s*mm\s*(?:hollow\s*)?(?:pipe|tube)",
    r"jireum\D{0,20}(\d+(?:\.\d+)?)",
    r"(?:지름|외경)\D{0,20}(\d+(?:\.\d+)?)",
    r"(\d+(?:\.\d+)?)\s*mm\s*(?:중공\s*)?파이프",
)
_WALL_THICKNESS_PATTERNS = _compile_patterns(
    r"(?:wall|thickness)\D{0,20}(\d+(?:\.\d+)?)",
    r"wall_thickness\D{0,20}(\d+(?:\.\d+)?)",
    r"(?:벽두께|두께)\D{0,20}(\d+(?:\.\d+)?)",
)
_STRAIGHT_PATTERNS = _compile_patterns(
    r"(?:straight|forward|jikjin|go|직진|앞으로)\D{0,20}(\d+(?:\.\d+)?)\s*mm",
    r"(\d+(?:\.\d+)?)\s*mm\D{0,20}(?:straight|forward|jikjin|직진|앞으로)",
)
_TURN_PATTERNS = _compile_patterns(
    r"(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|do|도)\D{0,20}(?:bend|turn|up|upward|꺾|위|상향)",
    r"(?:bend|turn|gg|kkeok|upward|up|꺾|굽|휘)\D{0,20}(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|do|도)",
    r"(?:bend|turn|upward)\D{0,20}(\d+(?:\.\d+)?)",
)
_IMPLICIT_RIGHT_ANGLE_PATTERN = re.compile(
    r"90\D{0,10}(?:up|upward|bend|turn)",
    flags=re.IGNORECASE,
)
_UPWARD_PATTERNS = _compile_patterns(
    r"(?:up|upward|vertical|olra|rise|올라|위로|상승)\D{0,20}(\d+(?:\.\d+)?)\s*mm",
    r"(\d+(?:\.\d+)?)\s*mm\D{0,20}(?:up|upward|vertical|olra|rise|올라|위로|상승)",
)
_BRANCH_PATTERN = re.compile(
    r"(?:branch|junction|tee|split|manifold|multi[- ]?port|분기|가지|갈래|매니폴드)",
    flags=re.IGNORECASE,
)
_MANIFOLD_PATTERN = re.compile(
    r"(?:manifold|multi[- ]?port|four[- ]?port|4[- ]?port|갈래|매니폴드)",
    flags=re.IGNORECASE,
)
_FOUR_BRANCH_PATTERN = re.compile(r"(?:four|4|네)", flags=re.IGNORECASE)
_THREE_BRANCH_PATTERN = re.compile(r"(?:three|3|세|셋)", flags=re.IGNORECASE)
_BRANCH_ANGLE_PATTERNS = _compile_patterns(
    r"(?:branch|junction|tee|split|분기|가지)\D{0,40}(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|도)",
    r"(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|도)\D{0,40}(?:branch|junction|tee|split|분기|가지)",
)
_CONNECTOR_PATTERN = re.compile(
    r"(?:connector|coupler|coupling|joint|커넥터|커플러|접합)",
    flags=re.IGNORECASE,
)
_CONNECTOR_LENGTH_PATTERNS = _compile_patterns(
    r"(?:connector|coupler|coupling|joint|커넥터|커플러|접합)\D{0,30}(\d+(?:\.\d+)?)\s*mm",
    r"(\d+(?:\.\d+)?)\s*mm\D{0,30}(?:connector|coupler|coupling|joint|커넥터|커플러|접합)",
)
_CAP_PATTERN = re.compile(r"(?:cap|capped|close|closed|마개|막아)", flags=re.IGNORECASE)
_OPEN_END_PATTERN = re.compile(
    r"(?:open\s*end|leave\s*open|열린\s*끝)", flags=re.IGNORECASE
)
_UNSUPPORTED_FREEFORM_PATTERN = re.compile(
    r"(?:spiral|coil|helix|helical|s[- ]?curve|spline|freeform|serpentine|"
    r"나선|코일|헬릭스|자유곡선)",
    flags=re.IGNORECASE,
)
_ANY_MM_PATTERNS = _compile_patterns(r"(\d+(?:\.\d+)?)\s*mm")

_BRANCH_DIRECTION_PATTERNS: tuple[tuple[re.Pattern[str], Direction], ...] = (
    (
        re.compile(r"(?:left|upper-left|lower-left|왼쪽|좌측)", flags=re.IGNORECASE),
        "-X",
    ),
    (
        re.compile(
            r"(?:right|upper-right|lower-right|오른쪽|우측)", flags=re.IGNORECASE
        ),
        "+X",
    ),
    (re.compile(r"(?:up|upper|위|상향)", flags=re.IGNORECASE), "+Z"),
    (re.compile(r"(?:down|lower|아래|하향)", flags=re.IGNORECASE), "-Z"),
)
_BRANCH_OUTLET_VECTOR_PATTERNS = (
    (
        re.compile(
            r"(?:upper[- ]?left|left[- ]?upper|왼쪽\s*위|좌상|상좌)",
            flags=re.IGNORECASE,
        ),
        (-1.0, 0.0, 1.0),
    ),
    (
        re.compile(
            r"(?:lower[- ]?left|left[- ]?lower|왼쪽\s*아래|좌하|하좌)",
            flags=re.IGNORECASE,
        ),
        (-1.0, 0.0, -1.0),
    ),
    (
        re.compile(
            r"(?:upper[- ]?right|right[- ]?upper|오른쪽\s*위|우상|상우)",
            flags=re.IGNORECASE,
        ),
        (1.0, 0.0, 1.0),
    ),
    (
        re.compile(
            r"(?:lower[- ]?right|right[- ]?lower|오른쪽\s*아래|우하|하우)",
            flags=re.IGNORECASE,
        ),
        (1.0, 0.0, -1.0),
    ),
)

_EXPLICIT_FOUR_PORT_PATTERN = re.compile(
    r"(?:four[- ]?port|4[- ]?port|four\s+open|4\s+open)",
    flags=re.IGNORECASE,
)
_EXPLICIT_OPEN_PORT_NUMBER_PATTERN = re.compile(
    r"(\d+)\s*(?:open\s*)?(?:port|ports|end|ends|outlet|outlets|개\s*포트)",
    flags=re.IGNORECASE,
)
_EXPLICIT_OPEN_PORT_WORD_COUNTS = (
    ("one", 1),
    ("two", 2),
    ("three", 3),
    ("four", 4),
    ("five", 5),
    ("six", 6),
    ("하나", 1),
    ("두", 2),
    ("둘", 2),
    ("세", 3),
    ("셋", 3),
    ("네", 4),
)
_EXPLICIT_OPEN_PORT_WORD_PATTERNS = tuple(
    (
        re.compile(
            rf"\b{re.escape(word)}\b\s*(?:open\s*)?"
            r"(?:port|ports|end|ends|outlet|outlets)",
            flags=re.IGNORECASE,
        ),
        count,
    )
    for word, count in _EXPLICIT_OPEN_PORT_WORD_COUNTS
)


def _first_number(patterns: tuple[re.Pattern[str], ...], text: str) -> float | None:
    """우선순위가 지정된 첫 정규식의 숫자 capture를 반환한다."""

    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def _infer_global_spec(text: str, settings: Settings) -> GlobalSpec:
    """prompt에서 파이프 단면을 추출하고 기존 truthy 기본값을 적용한다."""

    outer = _first_number(_OUTER_DIAMETER_PATTERNS, text)
    wall = _first_number(_WALL_THICKNESS_PATTERNS, text)
    return GlobalSpec(
        outer_diameter=outer or settings.default_outer_diameter,
        wall_thickness=wall or settings.default_wall_thickness,
        is_hollow=True,
    )


def _infer_straight_goal(text: str) -> Goal | None:
    """먼저 일치하는 직선 길이를 +X 이동 목표로 변환한다."""

    length = _first_number(_STRAIGHT_PATTERNS, text)
    if length is None:
        return None
    return Goal(type="move", direction="+X", length=length)


def _infer_turn_goal(text: str) -> Goal | None:
    """명시 각도 또는 기존 90도 관용 표현을 +Z 회전 목표로 변환한다."""

    angle = _first_number(_TURN_PATTERNS, text)
    if angle is not None:
        return Goal(type="turn", direction="+Z", angle=angle)
    if _IMPLICIT_RIGHT_ANGLE_PATTERN.search(text):
        return Goal(type="turn", direction="+Z", angle=90.0)
    return None


def _infer_upward_goal(text: str, prior_goals: list[Goal]) -> Goal | None:
    """상승 길이를 +Z 이동으로 만들되 기존 마지막 이동과의 중복은 피한다."""

    up_length = _first_number(_UPWARD_PATTERNS, text)
    if up_length is None:
        return None
    if (
        prior_goals
        and prior_goals[-1].type == "move"
        and prior_goals[-1].length == up_length
    ):
        return None
    return Goal(type="move", direction="+Z", length=up_length)


def _infer_branch_goal(text: str, explicit_open_count: int | None) -> Goal | None:
    """분기 수, 방향, 각도와 outlet 벡터를 legacy 분기 목표로 묶는다."""

    if not _BRANCH_PATTERN.search(text):
        return None

    outlet_vectors = _branch_outlet_vectors(text)
    is_manifold = _MANIFOLD_PATTERN.search(text)
    branch_count = 2
    include_primary_outlet: bool | None = None
    if outlet_vectors:
        branch_count = len(outlet_vectors)
        include_primary_outlet = False
    elif explicit_open_count is not None and is_manifold:
        branch_count = explicit_open_count
        include_primary_outlet = False
        if branch_count == 4:
            outlet_vectors = _default_four_port_vectors()
    elif _FOUR_BRANCH_PATTERN.search(text):
        branch_count = 4
    elif _THREE_BRANCH_PATTERN.search(text):
        branch_count = 3

    angle = _first_number(_BRANCH_ANGLE_PATTERNS, text)
    angle = angle or 45.0
    branch_angles = [angle, -angle]
    while len(branch_angles) < branch_count:
        branch_angles.append(90.0)
    return Goal(
        type="branch",
        direction=_branch_direction(text),
        branch_count=branch_count,
        branch_angles=branch_angles[:branch_count],
        required_outlet_vectors=outlet_vectors,
        include_primary_outlet=include_primary_outlet,
        junction_style="smooth_hub" if is_manifold or outlet_vectors else None,
    )


def _infer_connector_goal(text: str) -> Goal | None:
    """connector 표현과 선택적 길이를 legacy connector 목표로 변환한다."""

    if not _CONNECTOR_PATTERN.search(text):
        return None
    length = _first_number(_CONNECTOR_LENGTH_PATTERNS, text)
    return Goal(type="connector", length=length or 20.0)


def _infer_end_goal(text: str) -> Goal | None:
    """open 표현을 cap 표현보다 우선해 legacy end 목표를 만든다."""

    cap_match = _CAP_PATTERN.search(text)
    open_end_match = _OPEN_END_PATTERN.search(text)
    if not cap_match and not open_end_match:
        return None
    return Goal(type="end", end_type="open" if open_end_match else "cap")


def _fallback_move_goal(text: str) -> Goal:
    """목표가 없으면 단순 길이 이동을 만들고 freeform 요청은 그대로 거부한다."""

    if _UNSUPPORTED_FREEFORM_PATTERN.search(text):
        raise ValueError(
            "The dry-run fixture planner cannot represent a qualitative "
            "freeform route; use the production LLM planner."
        )
    length = _first_number(_ANY_MM_PATTERNS, text) or 100.0
    return Goal(type="move", direction="+X", length=length)


def _assign_legacy_goal_ids(goals: list[Goal]) -> list[Goal]:
    """생성 순서대로 기존 G1, G2 형식의 goal ID를 부여한다."""

    return [
        goal.model_copy(update={"goal_id": f"G{index}"})
        for index, goal in enumerate(goals, start=1)
    ]


def _start_axis_for_goals(goals: list[Goal]) -> tuple[float, float, float]:
    """첫 이동 방향이 있으면 시작 축으로 사용하고 아니면 +X를 유지한다."""

    if not goals or goals[0].type != "move" or not goals[0].direction:
        return (1.0, 0.0, 0.0)

    from cadgen.vector import direction_to_vector

    return direction_to_vector(goals[0].direction)


def infer_intent(prompt: str, settings: Settings) -> IntentResult:
    """간단한 dry-run 문장에서 제한된 legacy 목표 계약을 추출한다."""

    compact = prompt.replace(",", " ").replace("\n", " ")
    global_spec = _infer_global_spec(compact, settings)
    goals: list[Goal] = []
    straight_goal = _infer_straight_goal(compact)
    if straight_goal is not None:
        goals.append(straight_goal)
    turn_goal = _infer_turn_goal(compact)
    if turn_goal is not None:
        goals.append(turn_goal)
    upward_goal = _infer_upward_goal(compact, goals)
    if upward_goal is not None:
        goals.append(upward_goal)

    explicit_open_count = _explicit_open_port_count(compact)
    branch_goal = _infer_branch_goal(compact, explicit_open_count)
    if branch_goal is not None:
        goals.append(branch_goal)
    connector_goal = _infer_connector_goal(compact)
    if connector_goal is not None:
        goals.append(connector_goal)
    end_goal = _infer_end_goal(compact)
    if end_goal is not None:
        goals.append(end_goal)

    if not goals:
        goals.append(_fallback_move_goal(compact))

    expected_open_ports, source = _expected_open_ports(
        compact,
        goals,
        explicit_count=explicit_open_count,
    )
    goals = _assign_legacy_goal_ids(goals)
    return IntentResult(
        global_spec=global_spec,
        target_behavior=goals,
        start_axis=_start_axis_for_goals(goals),
        expected_open_ports=expected_open_ports,
        expected_open_ports_source=source,
    )


def plan_next_action(state: PipeState) -> ActionDraft:
    """dry-run 상태의 첫 목표를 지원되는 legacy 행동 하나로 변환한다."""

    if not state.remaining_goals:
        raise ValueError("No remaining goals to plan")
    goal = state.remaining_goals[0]
    target_port = state.open_ports[0].id
    affected_goal_ids, completed_goal_ids = _goal_id_envelope(goal)
    if goal.type == "move":
        return ActionDraft(
            target_port=target_port,
            module="straight_pipe",
            params={"length": goal.length, "direction": goal.direction},
            affected_goal_ids=affected_goal_ids,
            completed_goal_ids=completed_goal_ids,
        )
    if goal.type == "turn":
        return ActionDraft(
            target_port=target_port,
            module="bend_pipe",
            params={"angle": goal.angle, "turn_direction": goal.direction},
            affected_goal_ids=affected_goal_ids,
            completed_goal_ids=completed_goal_ids,
        )
    if goal.type == "branch":
        branch_count = goal.branch_count or len(goal.required_outlet_vectors) or 2
        return ActionDraft(
            target_port=target_port,
            module="junction_pipe",
            params={
                "branch_count": branch_count,
                "branch_angles": goal.branch_angles or [45.0, -45.0],
                "direction": goal.direction,
                "required_outlet_directions": goal.required_outlet_directions,
                "required_outlet_vectors": goal.required_outlet_vectors,
                "outlet_vectors": goal.required_outlet_vectors,
                "include_primary_outlet": goal.include_primary_outlet,
                "junction_style": goal.junction_style,
            },
            affected_goal_ids=affected_goal_ids,
            completed_goal_ids=completed_goal_ids,
        )
    if goal.type == "diameter_change":
        return ActionDraft(
            target_port=target_port,
            module="reducer_pipe",
            params={
                "length": goal.transition_length or 50.0,
                "diameter_out": goal.diameter_out,
                "wall_thickness_out": goal.wall_thickness_out,
                "offset": goal.offset,
            },
            affected_goal_ids=affected_goal_ids,
            completed_goal_ids=completed_goal_ids,
        )
    if goal.type == "connector":
        return ActionDraft(
            target_port=target_port,
            module="connector_pipe",
            params={"length": goal.length or 10.0},
            affected_goal_ids=affected_goal_ids,
            completed_goal_ids=completed_goal_ids,
        )
    if goal.type == "end":
        return ActionDraft(
            target_port=target_port,
            module="cap_pipe",
            params={"end_type": goal.end_type or "cap"},
            affected_goal_ids=affected_goal_ids,
            completed_goal_ids=completed_goal_ids,
        )
    raise ValueError(
        "The dry-run fixture planner does not support goal type "
        f"{goal.type!r}; it will not substitute unrelated cap geometry."
    )


def _goal_id_envelope(goal: Goal) -> tuple[list[str], list[str]]:
    """affected/completed에 넣을 서로 독립적인 legacy goal ID 목록을 만든다."""

    goal_ids = [goal.goal_id] if goal.goal_id else []
    return goal_ids, goal_ids.copy()


def _branch_direction(text: str) -> Direction | None:
    """기존 left/right/up/down 우선순서로 대표 분기 방향을 찾는다."""

    for pattern, direction in _BRANCH_DIRECTION_PATTERNS:
        if pattern.search(text):
            return direction
    return None


def _branch_outlet_vectors(text: str) -> list[tuple[float, float, float]]:
    """고정된 좌상ㆍ좌하ㆍ우상ㆍ우하 순서로 명시 outlet 벡터를 반환한다."""

    vectors: list[tuple[float, float, float]] = []
    for pattern, vector in _BRANCH_OUTLET_VECTOR_PATTERNS:
        if pattern.search(text):
            vectors.append(vector)
    return vectors


def _default_four_port_vectors() -> list[tuple[float, float, float]]:
    """legacy four-port fixture의 좌상ㆍ좌하ㆍ우상ㆍ우하 벡터를 반환한다."""

    return [
        (-1.0, 0.0, 1.0),
        (-1.0, 0.0, -1.0),
        (1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0),
    ]


def _expected_open_ports(
    text: str,
    goals: list[Goal],
    *,
    explicit_count: int | None = None,
) -> tuple[int | None, str]:
    """명시된 open-port 수를 우선하고 없으면 목표 topology에서 계산한다."""

    explicit_count = (
        explicit_count
        if explicit_count is not None
        else _explicit_open_port_count(text)
    )
    if explicit_count is not None:
        return explicit_count, "explicit"

    open_ports = 1
    for goal in goals:
        if goal.type == "branch":
            branch_count = goal.branch_count or len(goal.required_outlet_vectors) or 2
            include_primary = _goal_include_primary(goal)
            open_ports = open_ports - 1 + branch_count + (1 if include_primary else 0)
        elif goal.type == "end" and goal.end_type == "cap":
            open_ports = max(0, open_ports - 1)
    return open_ports, "derived"


def _goal_include_primary(goal: Goal) -> bool:
    """명시 정책 또는 outlet vector 유무로 primary outlet 포함 여부를 정한다."""

    if goal.include_primary_outlet is not None:
        return goal.include_primary_outlet
    return not bool(goal.required_outlet_vectors)


def _explicit_open_port_count(text: str) -> int | None:
    """four-port 관용구, 숫자, 단어 순으로 명시 open-port 수를 찾는다."""

    if _EXPLICIT_FOUR_PORT_PATTERN.search(text):
        return 4
    match = _EXPLICIT_OPEN_PORT_NUMBER_PATTERN.search(text)
    if match:
        return int(match.group(1))
    for pattern, count in _EXPLICIT_OPEN_PORT_WORD_PATTERNS:
        if pattern.search(text):
            return count
    return None
