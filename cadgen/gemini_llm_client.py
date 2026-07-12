"""Gemini 구조화 응답, 대화 lineage와 토큰 사용량을 관리한다.

prompt와 Pydantic schema를 입력받아 완전히 검증된 타입 객체를 반환한다.
불완전 JSON, 예산 초과와 provider 오류를 기본값으로 대체하지 않고 분류해 전달한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, TypeVar, get_args

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import BaseModel, ValidationError

from cadgen.runtime_settings import Settings
from cadgen.typed_data_models import LLMUsage

T = TypeVar("T", bound=BaseModel)


# Intent and action schemas can both exceed Gemini's practical finite-enum
# grammar capacity when a design contains many independent coordinates.  Both
# boundaries therefore support the same exact bounded decimal-object fallback.
_ENCODED_DECIMAL_PARTS = {"intent", "step_planner"}
# Diagnostic episodes carry their complete typed context in one request.  They
# must never inherit another CAD step's conversational state or persist a
# lineage that could later be restored from a checkpoint.
_STATELESS_PARTS = {
    "intent_repair_advisor",
    "intent_repair_reviewer",
    "step_repair_advisor",
}
_DECIMAL_TAG = "k"
_DECIMAL_COEFFICIENT = "c"
_DECIMAL_PLACES = "p"
_MAX_EXACT_INTEGER = (1 << 53) - 1
# Python/JSON float round-trips commonly need more than nine decimal places.
# Nine made values such as 33.67006979750809 impossible to spell and caused a
# real planner response to shift the decimal point by 10^13.  Fifteen keeps the
# coefficient inside the exactly representable integer range for ordinary CAD
# magnitudes while recovering sub-nanometre decimal detail when it is present in
# an immutable state value.
MAX_ENCODED_DECIMAL_PLACES = 15
MAX_STRUCTURED_NUMBER_LITERALS = 96
# Gemini 3 Flash accepted the same planner schema at a 686-byte enum and
# rejected a longer 821-byte enum before generation, while other literal sets
# have failed below that boundary. This is a hard local preflight guard, not a
# provider guarantee; the planner applies a lower soft ceiling and adaptive
# schema fallback because grammar complexity also depends on literal content.
MAX_STRUCTURED_NUMBER_LITERAL_BYTES = 512


class GeminiConfigError(RuntimeError):
    """API 키나 모델 설정 때문에 호출을 시작할 수 없음을 나타낸다."""

    pass


class GeminiBudgetError(RuntimeError):
    """감사 가능한 호출 또는 토큰 예산이 남지 않았음을 나타낸다."""

    pass


class GeminiRequestError(RuntimeError):
    """provider 요청이 정상적인 구조화 응답 없이 실패했음을 나타낸다."""

    pass


class GeminiInvalidRequestError(GeminiRequestError):
    """구조화 생성 전 Gemini가 요청 본문을 거절했음을 나타낸다."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        provider_code: str | None = None,
    ) -> None:
        """무효 요청 예외를 메타데이터와 함께 초기화한다."""

        super().__init__(message)
        self.status_code = status_code
        self.provider_code = provider_code


class GeminiLineageError(GeminiRequestError):
    """이전 interaction lineage를 재사용할 수 없어 전체 문맥 재전송이 필요하다."""

    pass


class StructuredOutputError(RuntimeError):
    """응답이 provider에 광고한 JSON 문법/schema를 통과하지 못했다."""

    def __init__(self, part: str, raw_text: str, cause: Exception):
        """구조화 출력 실패 예외를 초기화한다."""

        self.part = part
        self.raw_text = raw_text
        self.cause = cause
        compact = raw_text[:500].replace("\n", " ")
        super().__init__(
            f"Gemini structured output failed for {part}: {cause}; raw={compact!r}"
        )


class HostContractValidationError(RuntimeError):
    """provider 스키마는 통과했으나 host/domain 계약에 실패했다."""

    def __init__(self, part: str, payload: Any, cause: Exception):
        """host 계약 검증 실패 예외를 초기화한다."""

        self.part = part
        self.payload = payload
        self.cause = cause
        super().__init__(
            f"Gemini JSON passed the advertised schema for {part} but failed "
            f"the host contract: {cause}"
        )


class StructuredOutputIncompleteError(StructuredOutputError):
    """완전한 구조화 응답 전에 유료 상호작용이 종료됐다."""

    def __init__(
        self,
        part: str,
        *,
        status: str,
        output_limit: int,
        output_tokens: int,
        thought_tokens: int,
    ):
        """불완전 구조화 출력 예외를 초기화한다."""

        self.part = part
        self.raw_text = ""
        self.status = status
        self.output_limit = output_limit
        self.output_tokens = output_tokens
        self.thought_tokens = thought_tokens
        self.cause = RuntimeError(
            "provider interaction did not complete within the generation allowance"
        )
        RuntimeError.__init__(
            self,
            "Gemini structured output incomplete for "
            f"{part}: status={status}, output_limit={output_limit}, "
            f"output_tokens={output_tokens}, thought_tokens={thought_tokens}",
        )


@dataclass
class _Lineage:
    """Lineage 데이터 모델이다."""

    interaction_id: str
    turns: int
    last_input_tokens: int


class GeminiClient:
    """구조화 Gemini 호출을 실행하고 lineage와 누적 사용량을 함께 관리한다."""

    supports_interaction_controls = True
    supports_numeric_literals = True
    supports_numeric_schema_modes = True
    supports_system_instruction = True
    supports_intent_repair_advisor = True
    supports_intent_repair_reviewer = True
    supports_step_repair_advisor = True
    # Backward-compatible capability flag used by the parameter-part prototype.
    supports_repair_advisor = True

    def __init__(self, settings: Settings):
        """설정과 예산으로 Gemini 클라이언트를 초기화한다."""

        if not settings.gemini_api_key:
            raise GeminiConfigError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required unless --dry-run is used."
            )
        from google import genai
        from google.genai import types

        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=max(1, int(settings.gemini_request_timeout_sec * 1000.0)),
            ),
        )
        self._settings = settings
        self._lineages: dict[str, _Lineage] = {}
        self._usage = LLMUsage()
        self._transport_retry_count = 0

    def stream_structured(
        self,
        prompt: Any,
        schema: type[T],
        *,
        part: str,
        thinking_level: str = "low",
        numeric_literals: list[str] | None = None,
        numeric_schema_mode: str | None = None,
        system_instruction: str | None = None,
    ) -> T:
        """한 interaction을 실행하고 JSON Schema까지 통과한 모델 객체를 반환한다."""

        lineage = None if part in _STATELESS_PARTS else self._usable_lineage(part)
        if numeric_schema_mode is None:
            numeric_schema_mode = (
                "enum"
                if numeric_literals is not None
                else "encoded"
                if part in _ENCODED_DECIMAL_PARTS
                else "plain"
            )
        if numeric_schema_mode not in {"plain", "enum", "encoded"}:
            raise GeminiConfigError(
                f"Unknown numeric schema mode: {numeric_schema_mode}"
            )
        if numeric_schema_mode == "enum" and numeric_literals is None:
            raise GeminiConfigError("enum numeric schema requires numeric_literals")
        if numeric_schema_mode != "enum" and numeric_literals is not None:
            raise GeminiConfigError(
                f"{numeric_schema_mode} numeric schema forbids numeric_literals"
            )
        response_schema = gemini_json_schema(
            schema,
            encode_decimals=numeric_schema_mode == "encoded",
            number_literals=(
                numeric_literals if numeric_schema_mode == "enum" else None
            ),
        )
        request_output_limit = self._request_output_limit(
            prompt,
            response_schema,
            lineage,
            part=part,
            system_instruction=system_instruction,
        )
        body: dict[str, Any] = {
            "model": self._settings.model_for(part),
            "input": prompt,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": response_schema,
            },
            "generation_config": {
                "thinking_level": thinking_level,
                "max_output_tokens": request_output_limit,
            },
        }
        if system_instruction:
            # Interactions only carries input/output history through
            # previous_interaction_id. System instructions are interaction-scoped
            # and must be re-specified on every repair continuation.
            body["system_instruction"] = system_instruction
        if self._settings.gemini_stateful and part not in _STATELESS_PARTS:
            if lineage is not None:
                body["previous_interaction_id"] = lineage.interaction_id
        else:
            body["store"] = False

        interaction = None
        for transport_attempt in range(self._settings.provider_transport_retries + 1):
            try:
                interaction = self._client.interactions.create(**body)
                break
            except Exception as exc:
                if "previous_interaction_id" in body and _lineage_rejected(exc):
                    self.reset_lineage(part)
                    raise GeminiLineageError(
                        "Gemini rejected the previous interaction; retry with full context"
                    ) from exc
                invalid_request = _invalid_request_metadata(exc)
                if invalid_request is not None:
                    status_code, provider_code = invalid_request
                    raise GeminiInvalidRequestError(
                        f"Gemini interaction request was rejected: {exc}",
                        status_code=status_code,
                        provider_code=provider_code,
                    ) from exc
                if (
                    transport_attempt < self._settings.provider_transport_retries
                    and _retryable_transport_error(exc)
                ):
                    self._transport_retry_count = (
                        getattr(self, "_transport_retry_count", 0) + 1
                    )
                    continue
                raise GeminiRequestError(
                    f"Gemini interaction request failed after "
                    f"{transport_attempt + 1} transport attempt(s): {exc}"
                ) from exc
        if interaction is None:  # pragma: no cover - loop either returns or raises.
            raise GeminiRequestError("Gemini transport retry loop produced no result")
        raw_json = (getattr(interaction, "output_text", None) or "").strip()
        usage_values = _interaction_usage_values(interaction)
        self._record_usage(usage_values)
        if self._usage.total_tokens > self._settings.gemini_max_total_tokens:
            raise GeminiBudgetError(
                "Gemini reported usage above the conservative request ceiling; "
                "no further calls will be made"
            )
        status = _interaction_status(interaction)
        if status in {"incomplete", "budget_exceeded"}:
            raise StructuredOutputIncompleteError(
                part,
                status=status,
                output_limit=request_output_limit,
                output_tokens=usage_values["output_tokens"],
                thought_tokens=usage_values["thought_tokens"],
            )
        if status not in {None, "completed"}:
            raise GeminiRequestError(
                "Gemini structured interaction did not complete: "
                f"part={part}, status={status}"
            )
        self._record_lineage(part, interaction, usage_values)
        if not raw_json:
            raise StructuredOutputError(
                part,
                raw_json,
                RuntimeError("Gemini returned an empty structured response"),
            )
        # Every structured response crosses two intentionally explicit gates.
        # Gate 1 is exactly the grammar advertised to Gemini. Gate 2 contains
        # host/domain semantics that JSON Schema cannot always express. Never
        # collapse a Gate-2 miss into a Gate-1 protocol failure.
        try:
            json_payload = _strict_json_loads(raw_json)
            Draft202012Validator(response_schema).validate(json_payload)
        except (JSONSchemaValidationError, ValueError) as exc:
            raise StructuredOutputError(part, raw_json, exc) from exc
        try:
            payload = (
                _decode_decimal_numbers(json_payload)
                if numeric_schema_mode == "encoded"
                else json_payload
            )
            return schema.model_validate(payload)
        except (ValidationError, ValueError) as exc:
            raise HostContractValidationError(part, json_payload, exc) from exc

    def has_previous(self, part: str) -> bool:
        """대화 계보에 이전 응답이 있는지 반환한다."""

        if not self._settings.gemini_stateful:
            return False
        return self._usable_lineage(part) is not None

    def reset_lineage(self, part: str) -> None:
        """역할별 대화 계보를 초기화한다."""

        self._lineages.pop(part, None)

    def restore_lineage(self, snapshot: dict[str, Any]) -> None:
        """저장된 계보 스냅샷을 복원한다."""

        restored: dict[str, _Lineage] = {}
        for part, raw in snapshot.items():
            if str(part) in _STATELESS_PARTS:
                continue
            if isinstance(raw, dict):
                interaction_id = raw.get("interaction_id")
                turns = int(raw.get("turns", 0) or 0)
                last_input_tokens = int(raw.get("last_input_tokens", 0) or 0)
            else:
                # Backward-compatible checkpoints stored only the interaction ID.
                interaction_id = raw
                turns = 0
                last_input_tokens = 0
            if interaction_id:
                restored[str(part)] = _Lineage(
                    interaction_id=str(interaction_id),
                    turns=max(0, turns),
                    last_input_tokens=max(0, last_input_tokens),
                )
        self._lineages = restored

    def lineage_snapshot(self) -> dict[str, dict[str, Any]]:
        """현재 계보 상태를 직렬화 가능한 스냅샷으로 만든다."""

        return {
            part: {
                "interaction_id": lineage.interaction_id,
                "turns": lineage.turns,
                "last_input_tokens": lineage.last_input_tokens,
            }
            for part, lineage in self._lineages.items()
            if part not in _STATELESS_PARTS
        }

    def restore_usage(self, snapshot: LLMUsage | dict[str, Any]) -> None:
        """토큰·호출 사용량 스냅샷을 복원한다."""

        self._usage = (
            snapshot.model_copy(deep=True)
            if isinstance(snapshot, LLMUsage)
            else LLMUsage.model_validate(snapshot)
        )

    def usage_snapshot(self) -> LLMUsage:
        """현재 사용량 스냅샷을 반환한다."""

        return self._usage.model_copy(deep=True)

    def remaining_call_budget(self) -> int:
        """전역 호출 상한 이전의 남은 호출 예산을 반환한다."""

        if not self._usage.accounting_complete:
            return 0
        return max(0, self._settings.gemini_max_calls - self._usage.calls)

    def policy_snapshot(self) -> dict[str, Any]:
        """감사를 위해 실제 요청에 사용되는 모델 매핑을 반환한다."""

        agent_parts = ("intent", "step_planner", "step_repair_advisor")
        return {
            "models": {
                part: self._settings.model_for(part)
                for part in self._settings.gemini_models
            },
            "default_model": self._settings.gemini_default_model,
            "agents": {
                "intent_agent": {
                    "part": "intent",
                    "model": self._settings.model_for("intent"),
                    "stateful": True,
                },
                "intent_validation_advisor_agent": {
                    "part": "intent_repair_advisor",
                    "model": self._settings.model_for("intent_repair_advisor"),
                    "stateful": False,
                },
                "intent_repair_reviewer_agent": {
                    "part": "intent_repair_reviewer",
                    "model": self._settings.model_for("intent_repair_reviewer"),
                    "stateful": False,
                },
                "parameter_planner_agent": {
                    "part": "step_planner",
                    "model": self._settings.model_for("step_planner"),
                    "stateful": True,
                },
                "geometry_validation_advisor_agent": {
                    "part": "step_repair_advisor",
                    "model": self._settings.model_for("step_repair_advisor"),
                    "stateful": False,
                },
            },
            "agent_parts_are_distinct": len(
                {self._settings.model_for(part) for part in agent_parts}
            )
            == len(agent_parts),
            "provider_transport_retries": {
                "configured_per_call": self._settings.provider_transport_retries,
                "used": getattr(self, "_transport_retry_count", 0),
                "geometry_repair_budget_consumed": False,
            },
        }

    def _usable_lineage(self, part: str) -> _Lineage | None:
        """provider에 넘길 유효 계보 메시지를 고른다."""

        lineage = self._lineages.get(part)
        if lineage is None:
            return None
        if lineage.turns >= self._settings.gemini_history_max_turns:
            self._lineages.pop(part, None)
            return None
        if lineage.last_input_tokens >= self._settings.gemini_history_token_threshold:
            self._lineages.pop(part, None)
            return None
        return lineage

    def _check_budget(self) -> None:
        """호출·토큰 예산 초과 여부를 검사한다."""

        if not self._usage.accounting_complete:
            raise GeminiBudgetError(
                "Gemini usage metadata was missing from a prior call; "
                "further paid calls are disabled because the token budget cannot be audited"
            )
        if self._usage.calls >= self._settings.gemini_max_calls:
            raise GeminiBudgetError(
                f"Gemini call ceiling reached: {self._settings.gemini_max_calls}"
            )

    def _request_output_limit(
        self,
        prompt: Any,
        response_schema: dict[str, Any],
        lineage: _Lineage | None,
        *,
        part: str,
        system_instruction: str | None = None,
    ) -> int:
        """요청별 출력 토큰 상한을 계산한다."""

        self._check_budget()
        remaining = self._settings.gemini_max_total_tokens - self._usage.total_tokens
        reserve = _conservative_request_token_bound(
            prompt,
            response_schema,
            system_instruction=system_instruction,
        )
        if lineage is not None:
            # Stateful billing may include prior context. Reserve the configured
            # reset threshold instead of assuming the server cache is free.
            reserve += self._settings.gemini_history_token_threshold
        # Gemini's max_output_tokens allowance is shared by hidden thought and
        # visible response tokens.  Reserve the conservative input/schema bound;
        # the remainder is the largest auditable response allowance for this call.
        output_limit = min(
            self._settings.output_token_limit_for(part),
            remaining - reserve,
        )
        if output_limit < 256:
            raise GeminiBudgetError(
                "Gemini token ceiling cannot safely fit the next prompt/schema; "
                f"remaining={remaining}, conservative_input_reserve={reserve}, "
                "minimum_structured_output=256"
            )
        return output_limit

    def _record_usage(self, values: dict[str, int]) -> None:
        """응답 사용량을 누적 기록한다."""

        if values["total_tokens"] == 0:
            values = {
                **values,
                "total_tokens": (
                    values["input_tokens"]
                    + values["output_tokens"]
                    + values["thought_tokens"]
                    + values["tool_use_tokens"]
                ),
            }
        usage_is_missing = not any(values.values())
        update = self._usage.model_dump()
        update["calls"] += 1
        if usage_is_missing:
            update["accounting_complete"] = False
            update["unmetered_calls"] += 1
        for key, value in values.items():
            update[key] += value
        self._usage = LLMUsage.model_validate(update)

    def _record_lineage(
        self,
        part: str,
        interaction: Any,
        usage_values: dict[str, int],
    ) -> None:
        """성공 응답을 역할 계보에 추가한다."""

        if part in _STATELESS_PARTS:
            self._lineages.pop(part, None)
            return
        interaction_id = getattr(interaction, "id", None)
        if self._settings.gemini_stateful and interaction_id:
            previous_turns = (
                self._lineages.get(part).turns if part in self._lineages else 0
            )
            self._lineages[part] = _Lineage(
                interaction_id=str(interaction_id),
                turns=previous_turns + 1,
                last_input_tokens=usage_values["input_tokens"],
            )


def gemini_json_schema(
    schema: type[BaseModel],
    *,
    encode_decimals: bool = False,
    number_literals: list[str] | None = None,
) -> dict[str, Any]:
    """Gemini structured output이 지원하는 JSON Schema만 반환한다."""

    if encode_decimals and number_literals is not None:
        raise ValueError("choose either encoded decimals or numeric literals")
    raw_schema = schema.model_json_schema()
    if schema.__name__ in {"PlannerDecision", "CorePlannerDecision"}:
        raw_schema = _lock_planner_inlet_sections(raw_schema)
    if encode_decimals:
        raw_schema = _encode_decimal_number_schema(raw_schema)
        # The decimal-object profile is a provider-compatibility fallback.  It
        # must be materially smaller than the schema which the provider just
        # rejected; descriptive annotations do not affect the wire contract
        # and can push otherwise valid structured-output grammars over opaque
        # provider complexity/size limits.
        raw_schema = _strip_schema_annotations(raw_schema)
    elif number_literals is not None:
        raw_schema = _encode_number_literal_schema(raw_schema, number_literals)
    # 클래스 docstring은 provider 계약에 불필요하며 스키마 크기만 키운다.
    # Field(description=...)는 속성 노드에 남긴다.
    raw_schema = _strip_model_level_descriptions(raw_schema)
    sanitized = _sanitize_schema_node(raw_schema)
    if getattr(schema, "provider_wire_contract", False):
        _assert_provider_wire_parity(schema, raw_schema, sanitized)
    return sanitized


def _strip_schema_annotations(value: Any) -> Any:
    """fallback 문법에서 비계약 JSON Schema 주석을 제거한다."""

    if isinstance(value, list):
        return [_strip_schema_annotations(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _strip_schema_annotations(item)
        for key, item in value.items()
        if key not in {"description", "title", "examples", "$comment", "default"}
    }


def _strip_model_level_descriptions(value: Any) -> Any:
    """객체 모델 docstring description만 제거하고 필드 description은 유지한다."""

    if isinstance(value, list):
        return [_strip_model_level_descriptions(item) for item in value]
    if not isinstance(value, dict):
        return value
    stripped = {
        key: _strip_model_level_descriptions(item) for key, item in value.items()
    }
    if "properties" in stripped:
        stripped.pop("description", None)
    return stripped


def _assert_provider_wire_parity(
    schema: type[BaseModel],
    raw_schema: dict[str, Any],
    provider_schema: dict[str, Any],
) -> None:
    """wire가 host 전용 검증을 숨기면 API 호출 전에 실패한다."""

    seen: set[type[BaseModel]] = set()
    stack: list[type[BaseModel]] = [schema]
    hidden_validators: list[str] = []

    def enqueue_models(annotation: Any) -> None:
        """주석 타입에서 중첩 BaseModel을 검사 스택에 넣는다."""

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            stack.append(annotation)
            return
        for candidate in get_args(annotation):
            enqueue_models(candidate)

    while stack:
        model = stack.pop()
        if model in seen:
            continue
        seen.add(model)
        decorators = model.__pydantic_decorators__
        validator_names = sorted(
            {
                *decorators.model_validators,
                *decorators.field_validators,
            }
        )
        if validator_names:
            hidden_validators.append(f"{model.__name__}({','.join(validator_names)})")
        for field in model.model_fields.values():
            enqueue_models(field.annotation)

    # These keywords are intentionally unsupported by the Gemini sanitizer and
    # would make the local wire parser stricter than the advertised grammar.
    weakening_keywords = (
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
    )
    raw_text = json.dumps(raw_schema, separators=(",", ":"))
    provider_text = json.dumps(provider_schema, separators=(",", ":"))
    lost_keywords = [
        keyword
        for keyword in weakening_keywords
        if f'"{keyword}"' in raw_text and f'"{keyword}"' not in provider_text
    ]
    if hidden_validators or lost_keywords:
        details = []
        if hidden_validators:
            details.append("host-only validators=" + ";".join(hidden_validators))
        if lost_keywords:
            details.append("sanitized constraints=" + ",".join(lost_keywords))
        raise GeminiConfigError(
            f"Provider wire schema {schema.__name__} is not parity-safe: "
            + "; ".join(details)
        )


def _lock_planner_inlet_sections(schema: dict[str, Any]) -> dict[str, Any]:
    """입구 단면 소스를 target port 하나로 고정한다."""

    def visit(value: Any) -> Any:
        """visit를 계산하거나 반환한다."""

        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: visit(item) for key, item in value.items()}
        properties = result.get("properties")
        if isinstance(properties, dict) and {
            "section_source",
            "outer_diameter",
            "wall_thickness",
        } <= set(properties):
            properties = dict(properties)
            properties["section_source"] = {
                "type": "string",
                "enum": ["inherit_target"],
            }
            properties.pop("outer_diameter", None)
            properties.pop("wall_thickness", None)
            result["properties"] = properties
            if isinstance(result.get("required"), list):
                result["required"] = [
                    name
                    for name in result["required"]
                    if name not in {"outer_diameter", "wall_thickness"}
                ]
        return result

    return visit(schema)


def _encode_number_literal_schema(
    schema: dict[str, Any],
    literals: list[str],
) -> dict[str, Any]:
    """intent 실수를 소스 근거 소수 문자열로 제약한다."""

    normalized = list(dict.fromkeys(str(value) for value in literals))
    if not normalized:
        raise ValueError("number_literals must not be empty")
    if len(normalized) > MAX_STRUCTURED_NUMBER_LITERALS:
        raise ValueError(
            "number_literals exceeds the provider-safe maximum of "
            f"{MAX_STRUCTURED_NUMBER_LITERALS} distinct values"
        )
    payload_bytes = len(
        json.dumps(
            normalized,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if payload_bytes > MAX_STRUCTURED_NUMBER_LITERAL_BYTES:
        raise ValueError(
            "number_literals exceeds the provider-safe serialized size of "
            f"{MAX_STRUCTURED_NUMBER_LITERAL_BYTES} bytes"
        )
    existing_definitions = set((schema.get("$defs") or {}).keys())
    definition_name = "NL"
    while definition_name in existing_definitions:
        definition_name = "_" + definition_name

    used = False

    def visit(value: Any) -> Any:
        """visit를 계산하거나 반환한다."""

        nonlocal used
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        if value.get("type") == "number":
            used = True
            return {"$ref": f"#/$defs/{definition_name}"}
        return {key: visit(item) for key, item in value.items()}

    result = visit(schema)
    if used:
        definitions = dict(result.get("$defs") or {})
        definitions[definition_name] = {
            "type": "string",
            "enum": normalized,
        }
        result["$defs"] = definitions
    return result


def _encode_decimal_number_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """JSON float를 유계 정수 쌍 표현으로 바꾼다."""

    existing_definitions = set((schema.get("$defs") or {}).keys())
    definition_names: dict[str, str] = {}
    for kind, base_name in (
        ("any", "CD"),
        ("nonnegative", "CDN"),
        ("positive", "CDP"),
    ):
        name = base_name
        while name in existing_definitions or name in definition_names.values():
            name = "_" + name
        definition_names[kind] = name

    used_kinds: set[str] = set()

    def visit(value: Any) -> Any:
        """visit를 계산하거나 반환한다."""

        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        if value.get("type") == "number":
            minimum = value.get("minimum")
            exclusive_minimum = value.get("exclusiveMinimum")
            if exclusive_minimum is not None and float(exclusive_minimum) >= 0.0:
                kind = "positive"
            elif minimum is not None and float(minimum) >= 0.0:
                kind = "nonnegative"
            else:
                kind = "any"
            used_kinds.add(kind)
            return {"$ref": f"#/$defs/{definition_names[kind]}"}
        return {key: visit(item) for key, item in value.items()}

    result = visit(schema)
    definitions = dict(result.get("$defs") or {})
    for kind in used_kinds:
        coefficient_minimum = (
            1
            if kind == "positive"
            else 0
            if kind == "nonnegative"
            else -_MAX_EXACT_INTEGER
        )
        definitions[definition_names[kind]] = {
            "type": "object",
            "description": (
                "Decimal c*10^(-p): 1.5 => {k:d,c:15,p:1}. Preserve "
                "the source decimal point; p may be 0..15 and must never be "
                "reduced merely to fit c."
            ),
            "properties": {
                _DECIMAL_TAG: {"type": "string", "enum": ["d"]},
                _DECIMAL_COEFFICIENT: {
                    "type": "integer",
                    "minimum": coefficient_minimum,
                    "maximum": _MAX_EXACT_INTEGER,
                },
                _DECIMAL_PLACES: {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": MAX_ENCODED_DECIMAL_PLACES,
                },
            },
            "required": [_DECIMAL_TAG, _DECIMAL_COEFFICIENT, _DECIMAL_PLACES],
            "additionalProperties": False,
        }
    result["$defs"] = definitions
    return result


def _decode_decimal_numbers(value: Any) -> Any:
    """Pydantic 검사 전 LLM 경계 숫자 표현을 디코딩한다."""

    if isinstance(value, list):
        return [_decode_decimal_numbers(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {_DECIMAL_TAG, _DECIMAL_COEFFICIENT, _DECIMAL_PLACES}:
        if value[_DECIMAL_TAG] != "d":
            raise ValueError("invalid encoded decimal")
        coefficient = value[_DECIMAL_COEFFICIENT]
        places = value[_DECIMAL_PLACES]
        if (
            isinstance(coefficient, bool)
            or not isinstance(coefficient, int)
            or isinstance(places, bool)
            or not isinstance(places, int)
            or not 0 <= places <= MAX_ENCODED_DECIMAL_PLACES
            or abs(coefficient) > _MAX_EXACT_INTEGER
        ):
            raise ValueError("invalid encoded decimal")
        try:
            decoded = coefficient / (10**places)
        except OverflowError as exc:
            raise ValueError("encoded decimal is outside float range") from exc
        if not math.isfinite(decoded):
            raise ValueError("encoded decimal is not finite")
        return decoded
    return {key: _decode_decimal_numbers(item) for key, item in value.items()}


def _reject_nonstandard_json_constant(value: str) -> Any:
    """NaN/Infinity 등 비표준 JSON 상수를 거절한다."""

    raise ValueError(f"non-standard JSON numeric constant is forbidden: {value}")


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """JSON 객체의 중복 키를 last-key-wins로 조용히 덮어쓰지 않는다."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key!r}")
        result[key] = value
    return result


def _strict_json_loads(raw_json: str) -> Any:
    """표준 JSON만 읽고 모든 깊이의 중복 객체 키를 거부한다."""

    payload = json.loads(
        raw_json,
        parse_constant=_reject_nonstandard_json_constant,
        object_pairs_hook=_reject_duplicate_object_keys,
    )

    def reject_nonfinite(value: Any) -> None:
        """잘못된 입력을 거절한다."""

        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite JSON number is forbidden")
        if isinstance(value, dict):
            for item in value.values():
                reject_nonfinite(item)
        elif isinstance(value, list):
            for item in value:
                reject_nonfinite(item)

    # JSON permits large exponents such as 1e999 even though Python decodes
    # them to infinity. Reject them at the advertised-grammar gate so they can
    # never become a later host-only Pydantic failure.
    reject_nonfinite(payload)
    return payload


def _sanitize_schema_node(value: Any, *, property_map: bool = False) -> Any:
    """provider wire 스키마 노드를 정화한다."""

    if isinstance(value, list):
        return [_sanitize_schema_node(item) for item in value]
    if not isinstance(value, dict):
        return value
    if property_map:
        return {key: _sanitize_schema_node(item) for key, item in value.items()}

    result: dict[str, Any] = {}
    for key, item in value.items():
        if key == "const":
            result["enum"] = [item]
            continue
        if key == "oneOf":
            # Gemini documents anyOf but not oneOf for structured output.
            result["anyOf"] = _sanitize_schema_node(item)
            continue
        if key == "exclusiveMinimum":
            # Gemini supports inclusive bounds only.  For integers the next
            # integer exactly preserves a strict bound.  For floats, using
            # nextafter(0, +inf) advertises 5e-324 and can make constrained
            # generation emit pathological subnormal/scientific values.  Keep
            # the ordinary boundary remotely and enforce strictness in the
            # local Pydantic/intent safety validators.
            result["minimum"] = (
                int(item) + 1 if value.get("type") == "integer" else float(item)
            )
            continue
        if key == "exclusiveMaximum":
            result["maximum"] = (
                int(item) - 1 if value.get("type") == "integer" else float(item)
            )
            continue
        if key in {
            "default",
            "discriminator",
            "minLength",
            "maxLength",
            "examples",
            "title",
        }:
            continue
        if key == "properties":
            result[key] = _sanitize_schema_node(item, property_map=True)
            continue
        if key == "$defs":
            result[key] = _sanitize_schema_node(item, property_map=True)
            continue
        result[key] = _sanitize_schema_node(item)
    return result


def schema_keywords(schema: dict[str, Any]) -> set[str]:
    """허용된 JSON Schema 키워드 집합을 반환한다."""

    keywords: set[str] = set()

    def visit(value: Any, *, property_map: bool = False) -> None:
        """visit를 계산하거나 반환한다."""

        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            if not property_map:
                keywords.add(key)
            visit(item, property_map=key in {"properties", "$defs"})

    visit(schema)
    return keywords


def _usage_value(usage: Any, name: str) -> int:
    """usage 필드에서 정수 토큰 값을 안전하게 읽는다."""

    value = getattr(usage, name, 0) if usage is not None else 0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _interaction_usage_values(interaction: Any) -> dict[str, int]:
    """한 상호작용의 입출력 토큰을 추출한다."""

    usage = getattr(interaction, "usage", None)
    return {
        "input_tokens": _usage_value(usage, "total_input_tokens"),
        "cached_tokens": _usage_value(usage, "total_cached_tokens"),
        "output_tokens": _usage_value(usage, "total_output_tokens"),
        "thought_tokens": _usage_value(usage, "total_thought_tokens"),
        "tool_use_tokens": _usage_value(usage, "total_tool_use_tokens"),
        "total_tokens": _usage_value(usage, "total_tokens"),
    }


def _interaction_status(interaction: Any) -> str | None:
    """상호작용 성공/실패 상태를 정규화한다."""

    raw = getattr(interaction, "status", None)
    if raw is None:
        return None
    value = getattr(raw, "value", raw)
    normalized = str(value).strip().lower()
    return normalized or None


def _conservative_request_token_bound(
    prompt: Any,
    schema: dict[str, Any],
    *,
    system_instruction: str | None = None,
) -> int:
    """요청 토큰의 보수적 상한을 추정한다."""

    schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    # A tokenizer cannot emit more ordinary content tokens than UTF-8 bytes.
    # The fixed allowance covers request framing and special tokens.
    return (
        _prompt_token_bound(prompt)
        + (_prompt_token_bound(system_instruction) if system_instruction else 0)
        + len(schema_text.encode("utf-8"))
        + 512
    )


def _prompt_token_bound(value: Any) -> int:
    """prompt 텍스트 토큰 상한을 추정한다."""

    if isinstance(value, list):
        return sum(_prompt_token_bound(item) for item in value)
    if isinstance(value, dict):
        if value.get("type") == "image" and isinstance(value.get("data"), str):
            # 512px evidence views are media inputs, not base64 text tokens.
            # This is intentionally well above documented tiled-image costs.
            return 4096
        return sum(
            len(str(key).encode("utf-8")) + _prompt_token_bound(item)
            for key, item in value.items()
        )
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if value is None:
        return 1
    try:
        return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        return len(str(value).encode("utf-8"))


def _lineage_rejected(exc: Exception) -> bool:
    """계보 거절 오류인지 판정한다."""

    message = str(exc).lower()
    return "previous_interaction_id" in message and any(
        marker in message for marker in ("expired", "invalid", "not found", "unknown")
    )


def _retryable_transport_error(exc: Exception) -> bool:
    """일시 전송/서버 실패만 재시도 가능으로 분류한다."""

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        legacy_code = getattr(exc, "code", None)
        status_code = legacy_code if isinstance(legacy_code, int) else None
    if status_code in {408, 409, 425, 429}:
        return True
    if isinstance(status_code, int) and 500 <= status_code <= 599:
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "timed out",
            "timeout",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "service unavailable",
            "server disconnected",
        )
    )


def _invalid_request_metadata(exc: Exception) -> tuple[int, str | None] | None:
    """provider가 선언한 HTTP 400 무효 요청만 인식한다."""

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        legacy_code = getattr(exc, "code", None)
        if isinstance(legacy_code, int):
            status_code = legacy_code

    body = getattr(exc, "body", None)
    if body is None:
        body = getattr(exc, "details", None)
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except (TypeError, ValueError):
            pass

    provider_code: str | None = None
    provider_message: str | None = None
    if isinstance(body, dict):
        error = body.get("error")
        payload = error if isinstance(error, dict) else body
        raw_code = payload.get("code")
        raw_status = payload.get("status")
        raw_message = payload.get("message")
        if isinstance(raw_code, str):
            provider_code = raw_code.lower()
        elif isinstance(raw_status, str):
            provider_code = raw_status.lower()
        if isinstance(raw_message, str):
            provider_message = raw_message.lower()

    message = str(exc).lower()
    if status_code is None and "error code: 400" in message:
        status_code = 400
    if provider_code is None:
        if "invalid_request" in message:
            provider_code = "invalid_request"
        elif "invalid_argument" in message:
            provider_code = "invalid_argument"
    invalid_marker = provider_code in {"invalid_request", "invalid_argument"} or any(
        marker in (provider_message or message)
        for marker in ("invalid argument", "invalid request")
    )
    if status_code == 400 and invalid_marker:
        return 400, provider_code
    return None
