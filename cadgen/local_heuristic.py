"""API 없이 실행하는 제한적 dry-run 의도/행동 fixture를 제공한다.

간단한 prompt 또는 ``PipeState``를 입력받아 legacy intent/action 초안을 반환한다.
프로덕션 실패의 대체 경로로 사용하지 않으며 표현할 수 없는 요청은 거부한다.
"""

from __future__ import annotations

import re

from cadgen.config import Settings
from cadgen.schemas import ActionDraft, Direction, GlobalSpec, Goal, IntentResult, PipeState


def _first_number(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def infer_intent(prompt: str, settings: Settings) -> IntentResult:
    """간단한 dry-run 문장에서 제한된 legacy 목표 계약을 추출한다."""

    compact = prompt.replace(",", " ").replace("\n", " ")
    outer = _first_number(
        [
            r"(?:diameter|od|outer diameter)\D{0,20}(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*mm\s*(?:hollow\s*)?(?:pipe|tube)",
            r"jireum\D{0,20}(\d+(?:\.\d+)?)",
            r"(?:지름|외경)\D{0,20}(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*mm\s*(?:중공\s*)?파이프",
        ],
        compact,
    )
    wall = _first_number(
        [
            r"(?:wall|thickness)\D{0,20}(\d+(?:\.\d+)?)",
            r"wall_thickness\D{0,20}(\d+(?:\.\d+)?)",
            r"(?:벽두께|두께)\D{0,20}(\d+(?:\.\d+)?)",
        ],
        compact,
    )

    global_spec = GlobalSpec(
        outer_diameter=outer or settings.default_outer_diameter,
        wall_thickness=wall or settings.default_wall_thickness,
        is_hollow=True,
    )

    goals: list[Goal] = []
    straight_match = re.search(
        r"(?:straight|forward|jikjin|go|직진|앞으로)\D{0,20}(\d+(?:\.\d+)?)\s*mm",
        compact,
        flags=re.IGNORECASE,
    )
    if straight_match is None:
        straight_match = re.search(
            r"(\d+(?:\.\d+)?)\s*mm\D{0,20}(?:straight|forward|jikjin|직진|앞으로)",
            compact,
            flags=re.IGNORECASE,
        )
    if straight_match:
        goals.append(Goal(type="move", direction="+X", length=float(straight_match.group(1))))

    bend_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|do|도)\D{0,20}(?:bend|turn|up|upward|꺾|위|상향)",
        compact,
        flags=re.IGNORECASE,
    )
    if bend_match is None:
        bend_match = re.search(
            r"(?:bend|turn|gg|kkeok|upward|up|꺾|굽|휘)\D{0,20}(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|do|도)",
            compact,
            flags=re.IGNORECASE,
        )
    if bend_match is None:
        bend_match = re.search(
            r"(?:bend|turn|upward)\D{0,20}(\d+(?:\.\d+)?)",
            compact,
            flags=re.IGNORECASE,
        )
    if bend_match:
        goals.append(Goal(type="turn", direction="+Z", angle=float(bend_match.group(1))))
    elif re.search(r"90\D{0,10}(?:up|upward|bend|turn)", compact, flags=re.IGNORECASE):
        goals.append(Goal(type="turn", direction="+Z", angle=90.0))

    up_match = re.search(
        r"(?:up|upward|vertical|olra|rise|올라|위로|상승)\D{0,20}(\d+(?:\.\d+)?)\s*mm",
        compact,
        flags=re.IGNORECASE,
    )
    if up_match is None:
        up_match = re.search(
            r"(\d+(?:\.\d+)?)\s*mm\D{0,20}(?:up|upward|vertical|olra|rise|올라|위로|상승)",
            compact,
            flags=re.IGNORECASE,
        )
    if up_match:
        up_length = float(up_match.group(1))
        if not goals or goals[-1].type != "move" or goals[-1].length != up_length:
            goals.append(Goal(type="move", direction="+Z", length=up_length))

    explicit_open_count = _explicit_open_port_count(compact)
    branch_match = re.search(
        r"(?:branch|junction|tee|split|manifold|multi[- ]?port|분기|가지|갈래|매니폴드)",
        compact,
        flags=re.IGNORECASE,
    )
    if branch_match:
        outlet_vectors = _branch_outlet_vectors(compact)
        is_manifold = re.search(
            r"(?:manifold|multi[- ]?port|four[- ]?port|4[- ]?port|갈래|매니폴드)",
            compact,
            flags=re.IGNORECASE,
        )
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
        elif re.search(r"(?:four|4|네)", compact, flags=re.IGNORECASE):
            branch_count = 4
        elif re.search(r"(?:three|3|세|셋)", compact, flags=re.IGNORECASE):
            branch_count = 3
        direction = _branch_direction(compact)
        angle = _first_number(
            [
                r"(?:branch|junction|tee|split|분기|가지)\D{0,40}(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|도)",
                r"(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|도)\D{0,40}(?:branch|junction|tee|split|분기|가지)",
            ],
            compact,
        )
        angle = angle or 45.0
        branch_angles = [angle, -angle]
        while len(branch_angles) < branch_count:
            branch_angles.append(90.0)
        goals.append(
            Goal(
                type="branch",
                direction=direction,
                branch_count=branch_count,
                branch_angles=branch_angles[:branch_count],
                required_outlet_vectors=outlet_vectors,
                include_primary_outlet=include_primary_outlet,
                junction_style="smooth_hub" if is_manifold or outlet_vectors else None,
            )
        )

    connector_match = re.search(
        r"(?:connector|coupler|coupling|joint|커넥터|커플러|접합)",
        compact,
        flags=re.IGNORECASE,
    )
    if connector_match:
        connector_length = _first_number(
            [
                r"(?:connector|coupler|coupling|joint|커넥터|커플러|접합)\D{0,30}(\d+(?:\.\d+)?)\s*mm",
                r"(\d+(?:\.\d+)?)\s*mm\D{0,30}(?:connector|coupler|coupling|joint|커넥터|커플러|접합)",
            ],
            compact,
        )
        goals.append(Goal(type="connector", length=connector_length or 20.0))

    cap_match = re.search(r"(?:cap|capped|close|closed|마개|막아)", compact, flags=re.IGNORECASE)
    open_end_match = re.search(r"(?:open\s*end|leave\s*open|열린\s*끝)", compact, flags=re.IGNORECASE)
    if cap_match or open_end_match:
        goals.append(Goal(type="end", end_type="open" if open_end_match else "cap"))

    if not goals:
        if re.search(
            r"(?:spiral|coil|helix|helical|s[- ]?curve|spline|freeform|serpentine|"
            r"나선|코일|헬릭스|자유곡선)",
            compact,
            flags=re.IGNORECASE,
        ):
            raise ValueError(
                "The dry-run fixture planner cannot represent a qualitative "
                "freeform route; use the production LLM planner."
            )
        length = _first_number([r"(\d+(?:\.\d+)?)\s*mm"], compact) or 100.0
        goals.append(Goal(type="move", direction="+X", length=length))

    expected_open_ports, source = _expected_open_ports(
        compact,
        goals,
        explicit_count=explicit_open_count,
    )
    goals = [
        goal.model_copy(update={"goal_id": f"G{index}"})
        for index, goal in enumerate(goals, start=1)
    ]
    start_axis = (1.0, 0.0, 0.0)
    if goals and goals[0].type == "move" and goals[0].direction:
        from cadgen.vector import direction_to_vector

        start_axis = direction_to_vector(goals[0].direction)
    return IntentResult(
        global_spec=global_spec,
        target_behavior=goals,
        start_axis=start_axis,
        expected_open_ports=expected_open_ports,
        expected_open_ports_source=source,
    )


def plan_next_action(state: PipeState) -> ActionDraft:
    """dry-run 상태의 첫 목표를 지원되는 legacy 행동 하나로 변환한다."""

    if not state.remaining_goals:
        raise ValueError("No remaining goals to plan")
    goal = state.remaining_goals[0]
    target_port = state.open_ports[0].id
    if goal.type == "move":
        return ActionDraft(
            target_port=target_port,
            module="straight_pipe",
            params={"length": goal.length, "direction": goal.direction},
            affected_goal_ids=[goal.goal_id] if goal.goal_id else [],
            completed_goal_ids=[goal.goal_id] if goal.goal_id else [],
        )
    if goal.type == "turn":
        return ActionDraft(
            target_port=target_port,
            module="bend_pipe",
            params={"angle": goal.angle, "turn_direction": goal.direction},
            affected_goal_ids=[goal.goal_id] if goal.goal_id else [],
            completed_goal_ids=[goal.goal_id] if goal.goal_id else [],
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
            affected_goal_ids=[goal.goal_id] if goal.goal_id else [],
            completed_goal_ids=[goal.goal_id] if goal.goal_id else [],
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
            affected_goal_ids=[goal.goal_id] if goal.goal_id else [],
            completed_goal_ids=[goal.goal_id] if goal.goal_id else [],
        )
    if goal.type == "connector":
        return ActionDraft(
            target_port=target_port,
            module="connector_pipe",
            params={"length": goal.length or 10.0},
            affected_goal_ids=[goal.goal_id] if goal.goal_id else [],
            completed_goal_ids=[goal.goal_id] if goal.goal_id else [],
        )
    if goal.type == "end":
        return ActionDraft(
            target_port=target_port,
            module="cap_pipe",
            params={"end_type": goal.end_type or "cap"},
            affected_goal_ids=[goal.goal_id] if goal.goal_id else [],
            completed_goal_ids=[goal.goal_id] if goal.goal_id else [],
        )
    raise ValueError(
        "The dry-run fixture planner does not support goal type "
        f"{goal.type!r}; it will not substitute unrelated cap geometry."
    )


def _branch_direction(text: str) -> Direction | None:
    if re.search(r"(?:left|upper-left|lower-left|왼쪽|좌측)", text, flags=re.IGNORECASE):
        return "-X"
    if re.search(r"(?:right|upper-right|lower-right|오른쪽|우측)", text, flags=re.IGNORECASE):
        return "+X"
    if re.search(r"(?:up|upper|위|상향)", text, flags=re.IGNORECASE):
        return "+Z"
    if re.search(r"(?:down|lower|아래|하향)", text, flags=re.IGNORECASE):
        return "-Z"
    return None


def _branch_outlet_vectors(text: str) -> list[tuple[float, float, float]]:
    patterns: list[tuple[str, tuple[float, float, float]]] = [
        (
            r"(?:upper[- ]?left|left[- ]?upper|왼쪽\s*위|좌상|상좌)",
            (-1.0, 0.0, 1.0),
        ),
        (
            r"(?:lower[- ]?left|left[- ]?lower|왼쪽\s*아래|좌하|하좌)",
            (-1.0, 0.0, -1.0),
        ),
        (
            r"(?:upper[- ]?right|right[- ]?upper|오른쪽\s*위|우상|상우)",
            (1.0, 0.0, 1.0),
        ),
        (
            r"(?:lower[- ]?right|right[- ]?lower|오른쪽\s*아래|우하|하우)",
            (1.0, 0.0, -1.0),
        ),
    ]
    vectors: list[tuple[float, float, float]] = []
    for pattern, vector in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            vectors.append(vector)
    return vectors


def _default_four_port_vectors() -> list[tuple[float, float, float]]:
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
    explicit_count = explicit_count if explicit_count is not None else _explicit_open_port_count(text)
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
    if goal.include_primary_outlet is not None:
        return goal.include_primary_outlet
    return not bool(goal.required_outlet_vectors)


def _explicit_open_port_count(text: str) -> int | None:
    word_counts = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "하나": 1,
        "두": 2,
        "둘": 2,
        "세": 3,
        "셋": 3,
        "네": 4,
    }
    if re.search(r"(?:four[- ]?port|4[- ]?port|four\s+open|4\s+open)", text, flags=re.IGNORECASE):
        return 4
    match = re.search(
        r"(\d+)\s*(?:open\s*)?(?:port|ports|end|ends|outlet|outlets|개\s*포트)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return int(match.group(1))
    for word, count in word_counts.items():
        if re.search(
            rf"\b{re.escape(word)}\b\s*(?:open\s*)?(?:port|ports|end|ends|outlet|outlets)",
            text,
            flags=re.IGNORECASE,
        ):
            return count
    return None
