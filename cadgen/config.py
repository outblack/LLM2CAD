"""환경변수와 파일 설정을 검증된 불변 ``Settings``로 변환한다.

모델ㆍ예산ㆍFreeCAD 정책 값을 입력받아 전체 실행의 단일 설정 객체를 출력한다.
잘못된 형식이나 안전하지 않은 수치는 자동 보정하지 않고 즉시 거부한다.
"""

from __future__ import annotations

import os
import shlex
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping


MODEL_PART_ENV = {
    "text": "GEMINI_TEXT_MODEL",
    "intent": "GEMINI_INTENT_MODEL",
    "step_planner": "GEMINI_STEP_PLANNER_MODEL",
    "parameter": "GEMINI_PARAMETER_MODEL",
    "visual_validator": "GEMINI_VISUAL_VALIDATOR_MODEL",
    "patch": "GEMINI_PATCH_MODEL",
    "mcp": "GEMINI_MCP_MODEL",
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
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"{name} must be an explicit boolean "
        "(true/false, yes/no, on/off, or 1/0)"
    )


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def env_args(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(shlex.split(raw))


def env_choice(name: str, default: str, choices: set[str]) -> str:
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
    output_dir: Path
    default_outer_diameter: float
    default_wall_thickness: float
    default_bend_radius: float
    intent_repair_attempts: int
    step_repair_attempts: int
    final_repair_rounds: int
    gemini_stateful: bool
    gemini_max_calls: int
    gemini_max_total_tokens: int
    gemini_max_output_tokens: int
    gemini_intent_max_output_tokens: int
    gemini_history_max_turns: int
    gemini_history_token_threshold: int
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

    def __post_init__(self) -> None:
        if self.max_iter <= 0:
            raise ValueError("CADGEN_MAX_ITER must be greater than zero")
        if (
            self.intent_repair_attempts < 0
            or self.step_repair_attempts < 0
            or self.final_repair_rounds < 0
        ):
            raise ValueError("repair attempt counts must be non-negative")
        positive_finite = {
            "CADGEN_DEFAULT_OUTER_DIAMETER": self.default_outer_diameter,
            "CADGEN_DEFAULT_WALL_THICKNESS": self.default_wall_thickness,
            "CADGEN_DEFAULT_BEND_RADIUS": self.default_bend_radius,
            "CADGEN_FREECAD_OPEN_TIMEOUT_SEC": self.freecad_open_timeout_sec,
            "CADGEN_FREECAD_MCP_TIMEOUT_SEC": self.freecad_mcp_timeout_sec,
            "CADGEN_MODELING_TOLERANCE": self.modeling_tolerance,
        }
        for name, value in positive_finite.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if self.default_wall_thickness * 2.0 >= self.default_outer_diameter:
            raise ValueError(
                "CADGEN_DEFAULT_WALL_THICKNESS must be less than half "
                "CADGEN_DEFAULT_OUTER_DIAMETER"
            )
    def model_for(self, part: str) -> str:
        value = self.gemini_models.get(part)
        return value or self.gemini_default_model

    def output_token_limit_for(self, part: str) -> int:
        if part == "intent":
            return self.gemini_intent_max_output_tokens
        return self.gemini_max_output_tokens

    def with_overrides(
        self,
        *,
        output_dir: Path | None = None,
        max_iter: int | None = None,
        skip_freecad: bool = False,
    ) -> "Settings":
        updates = {}
        if output_dir is not None:
            updates["output_dir"] = output_dir
        if max_iter is not None:
            updates["max_iter"] = max_iter
        if skip_freecad:
            updates["freecad_auto_open"] = False
            updates["freecad_mcp_enabled"] = False
            updates["freecad_step_mcp_enabled"] = False
            updates["require_freecad_app"] = False
            updates["freecad_mcp_required"] = False
            updates["freecad_capture_views"] = False
            updates["visual_validation_mode"] = "off"
        return replace(self, **updates)


def load_settings(env_file: Path | None = None) -> Settings:
    """환경 파일과 프로세스 환경을 읽어 검증된 ``Settings``를 생성한다."""

    if env_file is not None:
        load_env_file(env_file)
    else:
        load_env_file(Path(".env"))

    default_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    models = {
        part: os.getenv(env_name, "") or default_model
        for part, env_name in MODEL_PART_ENV.items()
    }

    freecad_mcp_enabled = env_bool("CADGEN_FREECAD_MCP_ENABLED", True)
    freecad_step_mcp_enabled = env_bool(
        "CADGEN_FREECAD_STEP_MCP_ENABLED",
        False,
    )
    max_iter = env_int("CADGEN_MAX_ITER", 12)
    # 의미 교정은 같은 상태에서만 수행한다. 기본값을 작게 유지해 한 설계가
    # 수십 번의 유사 호출로 토큰을 소모하지 않게 하고, 복잡한 실험만 환경
    # 변수로 명시적으로 확장한다.
    intent_repair_attempts = env_int("CADGEN_INTENT_REPAIR_ATTEMPTS", 3)
    step_repair_attempts = env_int("CADGEN_STEP_REPAIR_ATTEMPTS", 6)
    final_repair_rounds = env_int("CADGEN_FINAL_REPAIR_ROUNDS", 1)
    # One initial planning pass plus one full replay after each final rollback.
    # Explicit environment overrides may intentionally choose a lower safety cap.
    retry_aware_call_default = (
        intent_repair_attempts
        + 1
        + (final_repair_rounds + 1)
        * max_iter
        * (step_repair_attempts + 1)
        + final_repair_rounds * 2
        + 4
    )

    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        gemini_default_model=default_model,
        gemini_models=models,
        max_iter=max_iter,
        output_dir=Path(os.getenv("CADGEN_OUTPUT_DIR", "outputs")),
        default_outer_diameter=env_float("CADGEN_DEFAULT_OUTER_DIAMETER", 20.0),
        default_wall_thickness=env_float("CADGEN_DEFAULT_WALL_THICKNESS", 2.0),
        default_bend_radius=env_float("CADGEN_DEFAULT_BEND_RADIUS", 30.0),
        intent_repair_attempts=intent_repair_attempts,
        step_repair_attempts=step_repair_attempts,
        final_repair_rounds=final_repair_rounds,
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
        gemini_history_max_turns=max(
            1,
            env_int("CADGEN_GEMINI_HISTORY_MAX_TURNS", 32),
        ),
        gemini_history_token_threshold=max(
            1024,
            env_int("CADGEN_GEMINI_HISTORY_TOKEN_THRESHOLD", 48000),
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
    )
