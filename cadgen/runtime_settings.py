"""환경변수와 파일 설정을 검증된 불변 ``Settings``로 변환한다.

모델ㆍ예산ㆍFreeCAD 정책 값을 입력받아 전체 실행의 단일 설정 객체를 출력한다.
잘못된 형식이나 안전하지 않은 수치는 자동 보정하지 않고 즉시 거부한다.
"""

from __future__ import annotations

import math
import os
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping


MODEL_PART_ENV = {
    "text": "GEMINI_TEXT_MODEL",
    "intent": "GEMINI_INTENT_MODEL",
    "intent_repair_advisor": "GEMINI_INTENT_REPAIR_ADVISOR_MODEL",
    "intent_repair_reviewer": "GEMINI_INTENT_REPAIR_REVIEWER_MODEL",
    "step_planner": "GEMINI_STEP_PLANNER_MODEL",
    "step_repair_advisor": "GEMINI_STEP_REPAIR_ADVISOR_MODEL",
    # 이전 프로토타입과의 호환을 위해 독립 모델 선택을 유지한다.
    # 새 인과 진단은 step_repair_advisor를 사용한다.
    "parameter": "GEMINI_PARAMETER_MODEL",
    "visual_validator": "GEMINI_VISUAL_VALIDATOR_MODEL",
    "patch": "GEMINI_PATCH_MODEL",
    "mcp": "GEMINI_MCP_MODEL",
}

# 각 production agent는 GEMINI_MODEL의 별칭이 아닌 독립 기본값을 사용한다.
# 명시적인 환경변수는 언제나 이 기본값보다 우선한다.
AGENT_MODEL_DEFAULTS = {
    "intent": "gemini-3.1-flash-lite",
    "intent_repair_advisor": "gemini-3.1-pro-preview",
    "intent_repair_reviewer": "gemini-3.5-flash",
    "step_planner": "gemini-3.5-flash",
    "step_repair_advisor": "gemini-3.1-pro-preview",
}

_OUTPUT_TOKEN_LIMIT_FIELDS = {
    "intent": "gemini_intent_max_output_tokens",
    "intent_repair_advisor": "gemini_intent_repair_advisor_max_output_tokens",
    "intent_repair_reviewer": "gemini_intent_repair_advisor_max_output_tokens",
    "step_repair_advisor": "gemini_step_repair_advisor_max_output_tokens",
}


def load_env_file(path: Path) -> None:
    """간단한 dotenv 파일을 읽되 이미 설정된 환경변수는 덮어쓰지 않는다."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def env_bool(name: str, default: bool) -> bool:
    """환경변수를 허용된 명시적 boolean 표기로 읽는다."""

    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"{name} must be an explicit boolean (true/false, yes/no, on/off, or 1/0)"
    )


def env_int(name: str, default: int) -> int:
    """환경변수를 정수로 읽고 비어 있으면 기본값을 반환한다."""

    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    """환경변수를 실수로 읽고 비어 있으면 기본값을 반환한다."""

    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def env_args(name: str, default: str) -> tuple[str, ...]:
    """shell 표기 환경변수를 실행 인자 tuple로 분리한다."""

    raw = os.getenv(name, default)
    return tuple(shlex.split(raw))


def env_choice(name: str, default: str, choices: set[str]) -> str:
    """환경변수를 정규화하고 허용된 선택지인지 검증한다."""

    raw = os.getenv(name, default).strip().lower()
    if raw not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {allowed}")
    return raw


@dataclass(frozen=True)
class Settings:
    """한 실행에서 공유되는 모델, 예산, 검증과 FreeCAD 정책의 불변 모음이다."""

    gemini_api_key: str | None = field(repr=False)
    gemini_default_model: str
    gemini_models: Mapping[str, str]
    max_iter: int
    max_iter_hard_ceiling: int = field(default=64, kw_only=True)
    max_iter_is_hard_limit: bool = field(default=False, kw_only=True)
    output_dir: Path
    default_outer_diameter: float
    default_wall_thickness: float
    default_bend_radius: float
    intent_repair_attempts: int
    intent_repair_advisor_enabled: bool = field(default=True, kw_only=True)
    # 필수 advisor 실패는 evidence-only 복구로 기록하되, 작성자의 남은 의미
    # 복구 권한을 소모하지 않는 호환ㆍ감사 정책이다.
    intent_repair_advisor_required: bool = field(default=True, kw_only=True)
    intent_repair_reviewer_enabled: bool = field(default=True, kw_only=True)
    step_repair_attempts: int
    final_repair_rounds: int
    step_repair_advisor_enabled: bool
    step_repair_advisor_required: bool
    step_repair_advisor_trigger_attempt: int
    step_repair_advisor_max_calls_per_step: int
    step_repair_advisor_max_signatures_per_step: int
    step_repair_advisor_probe_limit: int
    gemini_stateful: bool
    gemini_max_calls: int
    gemini_max_total_tokens: int
    gemini_max_output_tokens: int
    gemini_intent_max_output_tokens: int
    gemini_intent_repair_advisor_max_output_tokens: int = field(
        default=8192,
        kw_only=True,
    )
    gemini_step_repair_advisor_max_output_tokens: int
    gemini_history_max_turns: int
    gemini_history_token_threshold: int
    gemini_request_timeout_sec: float
    stream_thinking_summary: bool
    freecad_auto_open: bool
    require_freecad_app: bool
    freecad_app_name: str
    freecad_process_name: str
    freecad_open_timeout_sec: float
    freecad_mcp_enabled: bool
    freecad_step_mcp_enabled: bool
    freecad_mcp_required: bool
    freecad_mcp_command: str
    freecad_mcp_args: tuple[str, ...]
    freecad_mcp_execute_tool: str
    freecad_mcp_execute_arg: str
    freecad_mcp_timeout_sec: float
    freecad_capture_views: bool
    visual_validation_mode: str
    modeling_tolerance: float
    feasibility_mode: str = field(default="best_effort", kw_only=True)
    max_uniform_centerline_scale: float = field(default=8.0, kw_only=True)
    primitive_compiler_enabled: bool = field(default=True, kw_only=True)
    provider_transport_retries: int = field(default=2, kw_only=True)
    max_causal_backjumps: int = field(default=4, kw_only=True)
    conflict_search_enabled: bool = field(default=True, kw_only=True)
    # A step-level advisor is probabilistic.  Do not let one rejected candidate
    # become a terminal contract verdict before the planner has tried at least
    # one materially different candidate in the same step.
    step_terminal_min_rejections: int = field(default=2, kw_only=True)
    # ``physical_only`` keeps the non-singular OCC sweep floor hard while
    # treating the former extra visual-quality radius as a recommendation.
    validation_enforcement: str = field(default="physical_only", kw_only=True)

    def __post_init__(self) -> None:
        """설정 그룹별 검증을 기존 오류 우선순서대로 수행한다."""

        _validate_iteration_limits(self)
        _validate_repair_limits(self)
        _validate_output_token_limits(self)
        _validate_positive_finite_values(self)
        _validate_geometry_policy(self)
        _validate_search_limits(self)

    def model_for(self, part: str) -> str:
        """agent part에 명시된 모델 또는 해당 part의 기본 모델을 반환한다."""

        value = self.gemini_models.get(part)
        return value or AGENT_MODEL_DEFAULTS.get(part) or self.gemini_default_model

    def output_token_limit_for(self, part: str) -> int:
        """agent part별 출력 토큰 한도를 반환한다."""

        field_name = _OUTPUT_TOKEN_LIMIT_FIELDS.get(part, "gemini_max_output_tokens")
        return getattr(self, field_name)

    def with_overrides(
        self,
        *,
        output_dir: Path | None = None,
        max_iter: int | None = None,
        skip_freecad: bool = False,
    ) -> "Settings":
        """명시된 실행 옵션만 반영한 새 설정 객체를 만든다."""

        updates: dict[str, object] = {}
        if output_dir is not None:
            updates["output_dir"] = output_dir
        if max_iter is not None:
            _add_max_iter_overrides(updates, max_iter)
        if skip_freecad:
            _add_skip_freecad_overrides(updates)
        return replace(self, **updates)


def _validate_iteration_limits(settings: Settings) -> None:
    """action 반복 기준과 hard ceiling의 관계를 검증한다."""

    if settings.max_iter <= 0:
        raise ValueError("CADGEN_MAX_ITER must be greater than zero")
    if settings.max_iter_hard_ceiling <= 0:
        raise ValueError("CADGEN_MAX_ITER_HARD_CEILING must be greater than zero")
    if settings.max_iter > settings.max_iter_hard_ceiling:
        raise ValueError("CADGEN_MAX_ITER must not exceed CADGEN_MAX_ITER_HARD_CEILING")


def _validate_repair_limits(settings: Settings) -> None:
    """의미 복구와 advisor 호출 횟수의 하한을 검증한다."""

    if (
        settings.intent_repair_attempts < 0
        or settings.step_repair_attempts < 0
        or settings.final_repair_rounds < 0
    ):
        raise ValueError("repair attempt counts must be non-negative")
    if settings.step_repair_advisor_trigger_attempt < 1:
        raise ValueError(
            "CADGEN_STEP_REPAIR_ADVISOR_TRIGGER_ATTEMPT must be at least one"
        )
    _require_non_negative(
        "CADGEN_STEP_REPAIR_ADVISOR_MAX_CALLS_PER_STEP",
        settings.step_repair_advisor_max_calls_per_step,
    )
    _require_non_negative(
        "CADGEN_STEP_REPAIR_ADVISOR_MAX_SIGNATURES_PER_STEP",
        settings.step_repair_advisor_max_signatures_per_step,
    )
    _require_non_negative(
        "CADGEN_STEP_REPAIR_ADVISOR_PROBE_LIMIT",
        settings.step_repair_advisor_probe_limit,
    )


def _validate_output_token_limits(settings: Settings) -> None:
    """구조화 응답을 만들 수 있는 최소 출력 토큰을 보장한다."""

    if settings.gemini_step_repair_advisor_max_output_tokens < 256:
        raise ValueError(
            "CADGEN_GEMINI_STEP_REPAIR_ADVISOR_MAX_OUTPUT_TOKENS must be at least 256"
        )
    if settings.gemini_intent_repair_advisor_max_output_tokens < 256:
        raise ValueError(
            "CADGEN_GEMINI_INTENT_REPAIR_ADVISOR_MAX_OUTPUT_TOKENS must be at least 256"
        )


def _validate_positive_finite_values(settings: Settings) -> None:
    """물리 치수와 timeout이 양의 유한수인지 검증한다."""

    _require_positive_finite(
        "CADGEN_DEFAULT_OUTER_DIAMETER", settings.default_outer_diameter
    )
    _require_positive_finite(
        "CADGEN_DEFAULT_WALL_THICKNESS", settings.default_wall_thickness
    )
    _require_positive_finite("CADGEN_DEFAULT_BEND_RADIUS", settings.default_bend_radius)
    _require_positive_finite(
        "CADGEN_FREECAD_OPEN_TIMEOUT_SEC", settings.freecad_open_timeout_sec
    )
    _require_positive_finite(
        "CADGEN_FREECAD_MCP_TIMEOUT_SEC", settings.freecad_mcp_timeout_sec
    )
    _require_positive_finite(
        "CADGEN_GEMINI_REQUEST_TIMEOUT_SEC", settings.gemini_request_timeout_sec
    )
    _require_positive_finite("CADGEN_MODELING_TOLERANCE", settings.modeling_tolerance)


def _validate_geometry_policy(settings: Settings) -> None:
    """단면 치수와 전역 geometry 완화 정책을 검증한다."""

    if settings.default_wall_thickness * 2.0 >= settings.default_outer_diameter:
        raise ValueError(
            "CADGEN_DEFAULT_WALL_THICKNESS must be less than half "
            "CADGEN_DEFAULT_OUTER_DIAMETER"
        )
    if settings.feasibility_mode not in {"best_effort", "strict", "off"}:
        raise ValueError("CADGEN_FEASIBILITY_MODE must be best_effort, strict, or off")
    if settings.validation_enforcement not in {"physical_only", "strict"}:
        raise ValueError(
            "CADGEN_VALIDATION_ENFORCEMENT must be physical_only or strict"
        )
    if (
        not math.isfinite(settings.max_uniform_centerline_scale)
        or settings.max_uniform_centerline_scale < 1.0
    ):
        raise ValueError(
            "CADGEN_MAX_UNIFORM_CENTERLINE_SCALE must be finite and at least one"
        )


def _validate_search_limits(settings: Settings) -> None:
    """provider 재시도와 causal backjump 횟수의 하한을 검증한다."""

    _require_non_negative(
        "CADGEN_PROVIDER_TRANSPORT_RETRIES", settings.provider_transport_retries
    )
    _require_non_negative("CADGEN_MAX_CAUSAL_BACKJUMPS", settings.max_causal_backjumps)
    if settings.step_terminal_min_rejections < 1:
        raise ValueError("CADGEN_STEP_TERMINAL_MIN_REJECTIONS must be at least one")


def _require_non_negative(name: str, value: int) -> None:
    """이름이 지정된 정수 설정의 음수 값을 거부한다."""

    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive_finite(name: str, value: float) -> None:
    """이름이 지정된 실수 설정이 양의 유한수인지 검증한다."""

    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")


def _add_max_iter_overrides(updates: dict[str, object], max_iter: int) -> None:
    """명시적 max_iter를 기존 의미인 hard action 한도로 추가한다."""

    updates["max_iter"] = max_iter
    updates["max_iter_hard_ceiling"] = max_iter
    updates["max_iter_is_hard_limit"] = True


def _add_skip_freecad_overrides(updates: dict[str, object]) -> None:
    """FreeCAD 실행과 그 증거가 필요한 기능만 비활성화한다."""

    # 정적 수치 제약용 inverse advisor는 유지한다. FreeCAD 근거가 필요한
    # 검사는 자연스럽게 건너뛰며, dry-run은 별도로 LLM advisor를 차단한다.
    updates["freecad_auto_open"] = False
    updates["freecad_mcp_enabled"] = False
    updates["freecad_step_mcp_enabled"] = False
    updates["require_freecad_app"] = False
    updates["freecad_mcp_required"] = False
    updates["freecad_capture_views"] = False
    updates["visual_validation_mode"] = "off"


def _step_advisor_call_reserve(
    *,
    enabled: bool,
    planning_rounds: int,
    max_iter: int,
    max_calls_per_step: int,
) -> int:
    """step advisor의 최초 호출과 protocol 재호출 예산을 예약한다."""

    if not enabled:
        return 0
    return planning_rounds * max_iter * max_calls_per_step * 2


def _intent_advisor_call_reserve(
    *,
    enabled: bool,
    reviewer_enabled: bool,
    intent_repair_attempts: int,
) -> int:
    """intent 진단 재호출과 reviewer fallback 예산을 예약한다."""

    if not enabled:
        return 0
    calls_per_candidate = 2 + int(reviewer_enabled)
    return (intent_repair_attempts + 1) * calls_per_candidate


def _default_gemini_call_budget(
    *,
    max_iter: int,
    intent_repair_attempts: int,
    step_repair_attempts: int,
    final_repair_rounds: int,
    intent_repair_advisor_enabled: bool,
    intent_repair_reviewer_enabled: bool,
    step_repair_advisor_enabled: bool,
    step_repair_advisor_max_calls_per_step: int,
) -> int:
    """모든 의미 시도와 제한된 protocol 재호출을 포함한 기본 예산을 계산한다."""

    planning_rounds = final_repair_rounds + 1
    planner_semantic_calls = planning_rounds * max_iter * (step_repair_attempts + 1)
    intent_advisor_calls = _intent_advisor_call_reserve(
        enabled=intent_repair_advisor_enabled,
        reviewer_enabled=intent_repair_reviewer_enabled,
        intent_repair_attempts=intent_repair_attempts,
    )
    step_advisor_calls = _step_advisor_call_reserve(
        enabled=step_repair_advisor_enabled,
        planning_rounds=planning_rounds,
        max_iter=max_iter,
        max_calls_per_step=step_repair_advisor_max_calls_per_step,
    )
    # planner 후보 하나가 생기기 전 schema/protocol 재호출 1회를 별도로 둔다.
    planner_calls_with_protocol_retries = planner_semantic_calls * 2
    return (
        intent_repair_attempts
        + 1
        + planner_calls_with_protocol_retries
        + intent_advisor_calls
        + step_advisor_calls
        + final_repair_rounds * 2
        + 4
    )


def load_settings(env_file: Path | None = None) -> Settings:
    """환경 파일과 프로세스 환경을 읽어 검증된 ``Settings``를 생성한다."""

    load_env_file(env_file if env_file is not None else Path(".env"))

    default_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    models = {
        part: os.getenv(env_name, "").strip()
        for part, env_name in MODEL_PART_ENV.items()
    }

    freecad_mcp_enabled = env_bool("CADGEN_FREECAD_MCP_ENABLED", True)
    freecad_step_mcp_enabled = env_bool(
        "CADGEN_FREECAD_STEP_MCP_ENABLED",
        False,
    )
    max_iter = env_int("CADGEN_MAX_ITER", 12)
    max_iter_hard_ceiling = env_int("CADGEN_MAX_ITER_HARD_CEILING", 64)
    # 의미 교정은 같은 상태에서만 수행한다. 기본값을 작게 유지해 한 설계가
    # 수십 번의 유사 호출로 토큰을 소모하지 않게 하고, 복잡한 실험만 환경
    # 변수로 명시적으로 확장한다.
    intent_repair_attempts = env_int("CADGEN_INTENT_REPAIR_ATTEMPTS", 3)
    intent_repair_advisor_enabled = env_bool(
        "CADGEN_INTENT_REPAIR_ADVISOR_ENABLED",
        True,
    )
    intent_repair_advisor_required = env_bool(
        "CADGEN_INTENT_REPAIR_ADVISOR_REQUIRED",
        True,
    )
    intent_repair_reviewer_enabled = env_bool(
        "CADGEN_INTENT_REPAIR_REVIEWER_ENABLED",
        True,
    )
    step_repair_attempts = env_int("CADGEN_STEP_REPAIR_ATTEMPTS", 6)
    final_repair_rounds = env_int("CADGEN_FINAL_REPAIR_ROUNDS", 1)
    step_repair_advisor_enabled = env_bool(
        "CADGEN_STEP_REPAIR_ADVISOR_ENABLED",
        True,
    )
    step_repair_advisor_required = env_bool(
        "CADGEN_STEP_REPAIR_ADVISOR_REQUIRED",
        True,
    )
    step_repair_advisor_trigger_attempt = env_int(
        "CADGEN_STEP_REPAIR_ADVISOR_TRIGGER_ATTEMPT",
        1,
    )
    step_repair_advisor_max_calls_per_step = env_int(
        "CADGEN_STEP_REPAIR_ADVISOR_MAX_CALLS_PER_STEP",
        3,
    )
    step_repair_advisor_max_signatures_per_step = env_int(
        "CADGEN_STEP_REPAIR_ADVISOR_MAX_SIGNATURES_PER_STEP",
        3,
    )
    step_repair_advisor_probe_limit = env_int(
        "CADGEN_STEP_REPAIR_ADVISOR_PROBE_LIMIT",
        0,
    )
    step_terminal_min_rejections = env_int(
        "CADGEN_STEP_TERMINAL_MIN_REJECTIONS",
        2,
    )
    retry_aware_call_default = _default_gemini_call_budget(
        max_iter=max_iter,
        intent_repair_attempts=intent_repair_attempts,
        step_repair_attempts=step_repair_attempts,
        final_repair_rounds=final_repair_rounds,
        intent_repair_advisor_enabled=intent_repair_advisor_enabled,
        intent_repair_reviewer_enabled=intent_repair_reviewer_enabled,
        step_repair_advisor_enabled=step_repair_advisor_enabled,
        step_repair_advisor_max_calls_per_step=(step_repair_advisor_max_calls_per_step),
    )

    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        gemini_default_model=default_model,
        gemini_models=models,
        max_iter=max_iter,
        max_iter_hard_ceiling=max_iter_hard_ceiling,
        max_iter_is_hard_limit=False,
        output_dir=Path(os.getenv("CADGEN_OUTPUT_DIR", "outputs")),
        default_outer_diameter=env_float("CADGEN_DEFAULT_OUTER_DIAMETER", 20.0),
        default_wall_thickness=env_float("CADGEN_DEFAULT_WALL_THICKNESS", 2.0),
        default_bend_radius=env_float("CADGEN_DEFAULT_BEND_RADIUS", 30.0),
        intent_repair_attempts=intent_repair_attempts,
        intent_repair_advisor_enabled=intent_repair_advisor_enabled,
        intent_repair_advisor_required=intent_repair_advisor_required,
        intent_repair_reviewer_enabled=intent_repair_reviewer_enabled,
        step_repair_attempts=step_repair_attempts,
        final_repair_rounds=final_repair_rounds,
        step_repair_advisor_enabled=step_repair_advisor_enabled,
        step_repair_advisor_required=step_repair_advisor_required,
        step_repair_advisor_trigger_attempt=step_repair_advisor_trigger_attempt,
        step_repair_advisor_max_calls_per_step=step_repair_advisor_max_calls_per_step,
        step_repair_advisor_max_signatures_per_step=(
            step_repair_advisor_max_signatures_per_step
        ),
        step_repair_advisor_probe_limit=step_repair_advisor_probe_limit,
        gemini_stateful=env_bool("CADGEN_GEMINI_STATEFUL", True),
        gemini_max_calls=max(
            1,
            env_int("CADGEN_GEMINI_MAX_CALLS", retry_aware_call_default),
        ),
        gemini_max_total_tokens=max(
            1,
            env_int("CADGEN_GEMINI_MAX_TOTAL_TOKENS", 1_000_000),
        ),
        gemini_max_output_tokens=max(
            256,
            env_int("CADGEN_GEMINI_MAX_OUTPUT_TOKENS", 16384),
        ),
        gemini_intent_max_output_tokens=max(
            256,
            env_int("CADGEN_GEMINI_INTENT_MAX_OUTPUT_TOKENS", 16384),
        ),
        gemini_intent_repair_advisor_max_output_tokens=max(
            256,
            env_int(
                "CADGEN_GEMINI_INTENT_REPAIR_ADVISOR_MAX_OUTPUT_TOKENS",
                8192,
            ),
        ),
        gemini_step_repair_advisor_max_output_tokens=max(
            256,
            env_int(
                "CADGEN_GEMINI_STEP_REPAIR_ADVISOR_MAX_OUTPUT_TOKENS",
                4096,
            ),
        ),
        gemini_history_max_turns=max(
            1,
            env_int("CADGEN_GEMINI_HISTORY_MAX_TURNS", 32),
        ),
        gemini_history_token_threshold=max(
            1024,
            env_int("CADGEN_GEMINI_HISTORY_TOKEN_THRESHOLD", 48000),
        ),
        gemini_request_timeout_sec=env_float(
            "CADGEN_GEMINI_REQUEST_TIMEOUT_SEC",
            180.0,
        ),
        stream_thinking_summary=env_bool("CADGEN_STREAM_THINKING_SUMMARY", True),
        freecad_auto_open=env_bool("CADGEN_FREECAD_AUTO_OPEN", True),
        require_freecad_app=env_bool("CADGEN_REQUIRE_FREECAD_APP", False),
        freecad_app_name=os.getenv("CADGEN_FREECAD_APP_NAME", "FreeCAD"),
        freecad_process_name=os.getenv("CADGEN_FREECAD_PROCESS_NAME", "FreeCAD"),
        freecad_open_timeout_sec=env_float("CADGEN_FREECAD_OPEN_TIMEOUT_SEC", 30.0),
        freecad_mcp_enabled=freecad_mcp_enabled,
        freecad_step_mcp_enabled=freecad_step_mcp_enabled,
        freecad_mcp_required=env_bool("CADGEN_FREECAD_MCP_REQUIRED", True),
        freecad_mcp_command=os.getenv("CADGEN_FREECAD_MCP_COMMAND", "uvx"),
        freecad_mcp_args=env_args(
            "CADGEN_FREECAD_MCP_ARGS",
            "--from freecad-mcp==0.1.19 freecad-mcp --only-text-feedback",
        ),
        freecad_mcp_execute_tool=os.getenv(
            "CADGEN_FREECAD_MCP_EXECUTE_TOOL", "execute_code"
        ),
        freecad_mcp_execute_arg=os.getenv("CADGEN_FREECAD_MCP_EXECUTE_ARG", "code"),
        freecad_mcp_timeout_sec=env_float("CADGEN_FREECAD_MCP_TIMEOUT_SEC", 360.0),
        freecad_capture_views=env_bool("CADGEN_FREECAD_CAPTURE_VIEWS", True),
        visual_validation_mode=env_choice(
            "CADGEN_VISUAL_VALIDATION_MODE",
            "off",
            {"off", "on_warning", "final_required"},
        ),
        modeling_tolerance=env_float("CADGEN_MODELING_TOLERANCE", 1e-4),
        feasibility_mode=env_choice(
            "CADGEN_FEASIBILITY_MODE",
            "best_effort",
            {"best_effort", "strict", "off"},
        ),
        max_uniform_centerline_scale=env_float(
            "CADGEN_MAX_UNIFORM_CENTERLINE_SCALE",
            8.0,
        ),
        primitive_compiler_enabled=env_bool(
            "CADGEN_PRIMITIVE_COMPILER_ENABLED",
            True,
        ),
        provider_transport_retries=max(
            0,
            env_int("CADGEN_PROVIDER_TRANSPORT_RETRIES", 2),
        ),
        max_causal_backjumps=max(
            0,
            env_int("CADGEN_MAX_CAUSAL_BACKJUMPS", 4),
        ),
        conflict_search_enabled=env_bool(
            "CADGEN_CONFLICT_SEARCH_ENABLED",
            True,
        ),
        step_terminal_min_rejections=step_terminal_min_rejections,
        validation_enforcement=env_choice(
            "CADGEN_VALIDATION_ENFORCEMENT",
            "physical_only",
            {"physical_only", "strict"},
        ),
    )
