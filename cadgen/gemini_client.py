"""Gemini 구조화 응답, 대화 lineage와 토큰 사용량을 관리한다.

prompt와 Pydantic schema를 입력받아 완전히 검증된 타입 객체를 반환한다.
불완전 JSON, 예산 초과와 provider 오류를 기본값으로 대체하지 않고 분류해 전달한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, TypeVar

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import BaseModel, ValidationError

from cadgen.config import Settings
from cadgen.schemas import LLMUsage

T = TypeVar("T", bound=BaseModel)


# Intent and action schemas can both exceed Gemini's practical finite-enum
# grammar capacity when a design contains many independent coordinates.  Both
# boundaries therefore support the same exact bounded decimal-object fallback.
_ENCODED_DECIMAL_PARTS = {"intent", "step_planner"}
_DECIMAL_TAG = "k"
_DECIMAL_COEFFICIENT = "c"
_DECIMAL_PLACES = "p"
_MAX_EXACT_INTEGER = (1 << 53) - 1
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
    """Gemini rejected the request body before structured generation began."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        provider_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_code = provider_code


class GeminiLineageError(GeminiRequestError):
    """이전 interaction lineage를 재사용할 수 없어 전체 문맥 재전송이 필요하다."""

    pass


class StructuredOutputError(RuntimeError):
    """응답 JSON이 schema 또는 Pydantic 계약을 통과하지 못했음을 나타낸다."""

    def __init__(self, part: str, raw_text: str, cause: Exception):
        self.part = part
        self.raw_text = raw_text
        self.cause = cause
        compact = raw_text[:500].replace("\n", " ")
        super().__init__(f"Gemini structured output failed for {part}: {cause}; raw={compact!r}")


class StructuredOutputIncompleteError(StructuredOutputError):
    """A paid interaction ended before a complete structured response existed."""

    def __init__(
        self,
        part: str,
        *,
        status: str,
        output_limit: int,
        output_tokens: int,
        thought_tokens: int,
    ):
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
    interaction_id: str
    turns: int
    last_input_tokens: int


class GeminiClient:
    """구조화 Gemini 호출을 실행하고 lineage와 누적 사용량을 함께 관리한다."""

    supports_interaction_controls = True
    supports_numeric_literals = True
    supports_system_instruction = True

    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise GeminiConfigError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required unless --dry-run is used."
            )
        from google import genai

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._settings = settings
        self._lineages: dict[str, _Lineage] = {}
        self._usage = LLMUsage()

    def stream_structured(
        self,
        prompt: Any,
        schema: type[T],
        *,
        part: str,
        thinking_level: str = "low",
        numeric_literals: list[str] | None = None,
        system_instruction: str | None = None,
    ) -> T:
        """한 interaction을 실행하고 JSON Schema까지 통과한 모델 객체를 반환한다."""

        lineage = self._usable_lineage(part)
        response_schema = gemini_json_schema(
            schema,
            encode_decimals=(
                part in _ENCODED_DECIMAL_PARTS and numeric_literals is None
            ),
            number_literals=numeric_literals,
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
        if self._settings.gemini_stateful:
            if lineage is not None:
                body["previous_interaction_id"] = lineage.interaction_id
        else:
            body["store"] = False

        try:
            interaction = self._client.interactions.create(**body)
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
            raise GeminiRequestError(f"Gemini interaction request failed: {exc}") from exc
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
        try:
            # 모든 구조화 응답은 동일한 경계를 통과한다. 먼저 엄격한 JSON
            # 문법(비표준 숫자 및 중복 객체 키 금지)을 확인하고, 공급자에게
            # 전달한 JSON Schema를 검증한 뒤에만 Pydantic 모델로 변환한다.
            json_payload = _strict_json_loads(raw_json)
            Draft202012Validator(response_schema).validate(json_payload)
            payload = (
                _decode_decimal_numbers(json_payload)
                if part in _ENCODED_DECIMAL_PARTS and numeric_literals is None
                else json_payload
            )
            return schema.model_validate(payload)
        except (JSONSchemaValidationError, ValidationError, ValueError) as exc:
            raise StructuredOutputError(part, raw_json, exc) from exc

    def has_previous(self, part: str) -> bool:
        if not self._settings.gemini_stateful:
            return False
        return self._usable_lineage(part) is not None

    def reset_lineage(self, part: str) -> None:
        self._lineages.pop(part, None)

    def restore_lineage(self, snapshot: dict[str, Any]) -> None:
        restored: dict[str, _Lineage] = {}
        for part, raw in snapshot.items():
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
        return {
            part: {
                "interaction_id": lineage.interaction_id,
                "turns": lineage.turns,
                "last_input_tokens": lineage.last_input_tokens,
            }
            for part, lineage in self._lineages.items()
        }

    def restore_usage(self, snapshot: LLMUsage | dict[str, Any]) -> None:
        self._usage = (
            snapshot.model_copy(deep=True)
            if isinstance(snapshot, LLMUsage)
            else LLMUsage.model_validate(snapshot)
        )

    def usage_snapshot(self) -> LLMUsage:
        return self._usage.model_copy(deep=True)

    def policy_snapshot(self) -> dict[str, Any]:
        """감사를 위해 실제 요청에 사용되는 모델 매핑을 반환한다."""

        return {
            "models": dict(self._settings.gemini_models),
            "default_model": self._settings.gemini_default_model,
        }

    def _usable_lineage(self, part: str) -> _Lineage | None:
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
        interaction_id = getattr(interaction, "id", None)
        if self._settings.gemini_stateful and interaction_id:
            previous_turns = self._lineages.get(part).turns if part in self._lineages else 0
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
    """Return only JSON Schema keywords supported by Gemini structured output."""
    if encode_decimals and number_literals is not None:
        raise ValueError("choose either encoded decimals or numeric literals")
    raw_schema = schema.model_json_schema()
    if schema.__name__ in {"PlannerDecision", "CorePlannerDecision"}:
        raw_schema = _lock_planner_inlet_sections(raw_schema)
    if encode_decimals:
        raw_schema = _encode_decimal_number_schema(raw_schema)
    elif number_literals is not None:
        raw_schema = _encode_number_literal_schema(raw_schema, number_literals)
    return _sanitize_schema_node(raw_schema)


def _lock_planner_inlet_sections(schema: dict[str, Any]) -> dict[str, Any]:
    """Make the target port the only planner-authored inlet-section source.

    A primitive attached to an open port must mate with that port's section.
    Letting the model repeat an explicit inlet OD/wall value is redundant and
    creates a conditional schema that Gemini can violate.  Diameter changes
    remain explicit through the transition primitive's output fields.
    """

    def visit(value: Any) -> Any:
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
    """Constrain intent floats to concise source-grounded decimal strings."""

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
    """Replace every JSON float with a bounded integer-pair representation."""

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
                "Decimal c*10^(-p): 1.5 => {k:d,c:15,p:1}."
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
                    "maximum": 9,
                },
            },
            "required": [_DECIMAL_TAG, _DECIMAL_COEFFICIENT, _DECIMAL_PLACES],
            "additionalProperties": False,
        }
    result["$defs"] = definitions
    return result


def _decode_decimal_numbers(value: Any) -> Any:
    """Decode the exact LLM boundary representation before Pydantic checks."""

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
            or not 0 <= places <= 9
            or abs(coefficient) > _MAX_EXACT_INTEGER
        ):
            raise ValueError("invalid encoded decimal")
        try:
            decoded = coefficient / (10 ** places)
        except OverflowError as exc:
            raise ValueError("encoded decimal is outside float range") from exc
        if not math.isfinite(decoded):
            raise ValueError("encoded decimal is not finite")
        return decoded
    return {key: _decode_decimal_numbers(item) for key, item in value.items()}


def _reject_nonstandard_json_constant(value: str) -> Any:
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

    return json.loads(
        raw_json,
        parse_constant=_reject_nonstandard_json_constant,
        object_pairs_hook=_reject_duplicate_object_keys,
    )


def _sanitize_schema_node(value: Any, *, property_map: bool = False) -> Any:
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
                int(item) + 1
                if value.get("type") == "integer"
                else float(item)
            )
            continue
        if key == "exclusiveMaximum":
            result["maximum"] = (
                int(item) - 1
                if value.get("type") == "integer"
                else float(item)
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
    keywords: set[str] = set()

    def visit(value: Any, *, property_map: bool = False) -> None:
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
    value = getattr(usage, name, 0) if usage is not None else 0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _interaction_usage_values(interaction: Any) -> dict[str, int]:
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
    message = str(exc).lower()
    return "previous_interaction_id" in message and any(
        marker in message for marker in ("expired", "invalid", "not found", "unknown")
    )


def _invalid_request_metadata(exc: Exception) -> tuple[int, str | None] | None:
    """Recognize only provider-declared HTTP 400 invalid-request failures.

    The experimental Interactions client and the stable SDK expose different
    exception shapes.  Prefer structured attributes and retain a narrow string
    fallback for older SDKs so auth, rate-limit, server, and connection errors
    are never mistaken for a schema negotiation failure.
    """

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
