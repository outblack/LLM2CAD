"""FreeCAD MCP 전송과 digest 결합 검증 증거의 해석을 담당한다.

설정ㆍ스크립트ㆍ예상 상태 식별자를 입력받아 측정 증거와 화면 경로를 반환한다.
도구 오류, 시간 초과, 누락되거나 다른 상태의 증거는 성공으로 간주하지 않는다.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import threading
from pathlib import Path
from typing import Any

from cadgen.config import Settings


class FreeCADMCPError(RuntimeError):
    """MCP 전송이나 증거 결합을 신뢰할 수 없을 때 발생한다."""

    pass


class FreeCADValidationError(FreeCADMCPError):
    """Digest-bound evidence proves that candidate geometry is invalid."""

    def __init__(self, message: str, evidence: dict[str, Any]):
        super().__init__(message)
        self.evidence = evidence


_MUTATION_LOCK = threading.Lock()
_VALIDATION_PREFIX = "CADGEN_VALIDATION="
_PUBLISH_PREFIX = "CADGEN_PUBLISH="
_DOCUMENT_PREFIX = "CADGEN_DOCUMENT="
_READY_PREFIX = "CADGEN_READY="
_VISUAL_READY_PREFIX = "CADGEN_VISUAL_READY="


async def probe_freecad_mcp(
    settings: Settings,
    *,
    executor: Any = None,
) -> dict[str, Any]:
    """전체 제한 시간 안에 실행 도구와 readiness sentinel을 확인한다."""

    execute = executor or execute_freecad_code
    code = (
        'import json\nprint("CADGEN_READY=" + json.dumps('
        '{"passed": True, "protocol": 1}, sort_keys=True, separators=(",", ":")))\n'
    )
    # Treat the configured timeout as one overall readiness deadline. A cold
    # uvx launch can use the full window, while quick connection-refused races
    # keep retrying until the Addon RPC server is actually ready.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + settings.freecad_mcp_timeout_sec
    last_error: Exception | None = None
    attempt = 0
    backoff = 0.25
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0.0:
            break
        attempt += 1
        try:
            result = await asyncio.wait_for(
                execute(settings, code),
                timeout=remaining,
            )
            _raise_for_mcp_failure(result)
            payloads = _sentinel_payloads(result, _READY_PREFIX)
            if len(payloads) != 1 or payloads[0].get("passed") is not True:
                raise FreeCADMCPError(
                    "FreeCAD MCP readiness sentinel is missing or invalid"
                )
            return payloads[0]
        except Exception as exc:
            last_error = exc
            remaining = deadline - loop.time()
            if remaining <= 0.0:
                break
            # Read-only preflight may safely absorb the normal race between the
            # process appearing and the Addon RPC server becoming ready.
            await asyncio.sleep(min(backoff, remaining))
            backoff = min(backoff * 2.0, 1.5)
    if isinstance(last_error, asyncio.TimeoutError):
        raise FreeCADMCPError("FreeCAD MCP readiness probe timed out") from last_error
    raise FreeCADMCPError(
        f"FreeCAD MCP readiness failed after {attempt} attempt(s): {last_error}"
    ) from last_error


async def probe_freecad_visual(settings: Settings) -> None:
    """Prove get_view can return an image before any paid planning call."""

    document_name = "CadGenVisualProbe"
    create_code = f'''import json
import FreeCAD as App
import FreeCADGui as Gui
import Part
name = {document_name!r}
if name in App.listDocuments():
    App.closeDocument(name)
doc = App.newDocument(name)
obj = doc.addObject("Part::Feature", "VisualProbe")
obj.Shape = Part.makeBox(1.0, 1.0, 1.0)
doc.recompute()
App.setActiveDocument(name)
Gui.activeDocument().activeView().fitAll()
print("{_VISUAL_READY_PREFIX}" + json.dumps({{"passed": True}}, sort_keys=True, separators=(",", ":")))
'''
    cleanup_code = f'''import FreeCAD as App
name = {document_name!r}
if name in App.listDocuments():
    App.closeDocument(name)
'''
    try:
        create_result = (
            await _call_freecad_tools(
                settings,
                [
                    (
                        settings.freecad_mcp_execute_tool,
                        {settings.freecad_mcp_execute_arg: create_code},
                    )
                ],
            )
        )[0]
        _raise_for_mcp_failure(create_result)
        sentinels = _sentinel_payloads(create_result, _VISUAL_READY_PREFIX)
        if len(sentinels) != 1 or sentinels[0].get("passed") is not True:
            raise FreeCADMCPError("FreeCAD visual readiness sentinel is invalid")
        view_result = (
            await _call_freecad_tools(
                settings,
                [("get_view", {"view_name": "Isometric", "width": 32, "height": 32})],
            )
        )[0]
        _raise_for_mcp_failure(view_result)
        if len(_image_payloads(view_result)) != 1:
            raise FreeCADMCPError("FreeCAD get_view readiness probe returned no image")
    finally:
        try:
            await _call_freecad_tools(
                settings,
                [
                    (
                        settings.freecad_mcp_execute_tool,
                        {settings.freecad_mcp_execute_arg: cleanup_code},
                    )
                ],
            )
        except Exception:
            pass


async def execute_freecad_code(settings: Settings, code: str) -> dict[str, Any]:
    """변경 작업을 직렬화하여 FreeCAD 코드를 실행하고 원시 MCP 결과를 반환한다."""

    try:
        return await asyncio.wait_for(
            _execute_freecad_code(settings, code),
            timeout=settings.freecad_mcp_timeout_sec,
        )
    except asyncio.TimeoutError as exc:
        raise FreeCADMCPError(
            f"FreeCAD MCP execution timed out after "
            f"{settings.freecad_mcp_timeout_sec:g} seconds. Outcome is uncertain."
        ) from exc


async def _execute_freecad_code(settings: Settings, code: str) -> dict[str, Any]:
    acquired = False
    try:
        while not acquired:
            acquired = _MUTATION_LOCK.acquire(blocking=False)
            if not acquired:
                await asyncio.sleep(0.01)
        results = await _call_freecad_tools(
            settings,
            [(settings.freecad_mcp_execute_tool, {settings.freecad_mcp_execute_arg: code})],
        )
    finally:
        if acquired:
            _MUTATION_LOCK.release()
    return results[0]


async def call_freecad_tool(
    settings: Settings,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """지정 MCP 도구를 제한 시간과 함께 한 번 호출한다."""

    try:
        results = await asyncio.wait_for(
            _call_freecad_tools(settings, [(tool_name, arguments)]),
            timeout=settings.freecad_mcp_timeout_sec,
        )
    except asyncio.TimeoutError as exc:
        raise FreeCADMCPError(
            f"FreeCAD MCP tool {tool_name} timed out after "
            f"{settings.freecad_mcp_timeout_sec:g} seconds."
        ) from exc
    return results[0]


async def _call_freecad_tools(
    settings: Settings,
    calls: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=settings.freecad_mcp_command,
        args=list(settings.freecad_mcp_args),
        env=None,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            tool_names = {tool.name for tool in getattr(tools_result, "tools", [])}
            missing = sorted({name for name, _ in calls} - tool_names)
            if missing:
                raise FreeCADMCPError(f"FreeCAD MCP tool(s) not found: {missing}")
            results = []
            for name, arguments in calls:
                result = await session.call_tool(name, arguments)
                results.append(_jsonable(result))
            return results


def assess_freecad_validation(
    result: dict[str, Any],
    *,
    expected_digest: str,
    expected_state_id: str,
    expected_module_ids: list[str],
    expected_internal_section_module_count: int | None = None,
    expected_open_port_count: int | None = None,
    expected_anchored_inlet_count: int | None = None,
    expected_generator_version: str | None = None,
    expected_run_id: str | None = None,
    expected_state_version: int | None = None,
    expected_attempt_id: int | None = None,
    expected_candidate_document: str | None = None,
) -> dict[str, Any]:
    """후보 증거가 예상 상태ㆍ모듈ㆍdigest와 완전히 일치하는지 검증한다."""

    _raise_for_mcp_failure(result)
    payloads = _sentinel_payloads(result, _VALIDATION_PREFIX)
    if len(payloads) != 1:
        raise FreeCADMCPError(
            f"Expected exactly one CADGEN_VALIDATION sentinel, found {len(payloads)}"
        )
    evidence = payloads[0]
    if evidence.get("schema_version") != 3:
        raise FreeCADMCPError("Unsupported FreeCAD validation schema version")
    if evidence.get("payload_digest") != expected_digest:
        raise FreeCADMCPError("FreeCAD validation payload digest mismatch")
    if evidence.get("state_id") != expected_state_id:
        raise FreeCADMCPError("FreeCAD validation state_id mismatch")
    identity_expectations = {
        "generator_version": expected_generator_version,
        "run_id": expected_run_id,
        "state_version": expected_state_version,
        "attempt_id": expected_attempt_id,
        "candidate_document": expected_candidate_document,
    }
    for field, expected in identity_expectations.items():
        if expected is not None and evidence.get(field) != expected:
            raise FreeCADMCPError(
                f"FreeCAD validation {field} mismatch"
            )
    if evidence.get("module_ids") != expected_module_ids:
        raise FreeCADMCPError("FreeCAD validation module list mismatch")
    checks = evidence.get("checks")
    if not isinstance(checks, dict):
        raise FreeCADMCPError("FreeCAD validation checks are missing")
    required_checks = {
        "assembly",
        "outer_network",
        "bore_network",
        "modules",
        "centerlines",
        "module_errors",
        "assembly_errors",
        "non_adjacent_overlaps",
        "connection_failures",
        "terminal_bore_failures",
        "anchored_inlet_bore_failures",
        "termination_seal_failures",
        "wall_section_failures",
        "sampled_internal_section_count",
        "minimum_authored_wall_thickness",
        "declared_downstream_open_port_count",
        "anchored_inlet_count",
    }
    if not required_checks.issubset(checks):
        missing = sorted(required_checks - set(checks))
        raise FreeCADMCPError(f"FreeCAD validation checks are incomplete: {missing}")
    for name in ("module_errors", "assembly_errors"):
        if checks.get(name) != []:
            raise FreeCADValidationError(
                f"FreeCAD {name} evidence is not empty",
                evidence,
            )
    for name in ("assembly", "outer_network", "bore_network"):
        value = checks.get(name)
        if not isinstance(value, dict) or value.get("passed") is not True:
            raise FreeCADValidationError(
                f"FreeCAD {name} check failed",
                evidence,
            )
    assembly_bounds = checks["assembly"].get("bounds")
    if not isinstance(assembly_bounds, dict):
        raise FreeCADMCPError("FreeCAD assembly bounds evidence is missing")
    minimum = assembly_bounds.get("minimum")
    maximum = assembly_bounds.get("maximum")
    if (
        not isinstance(minimum, list)
        or not isinstance(maximum, list)
        or len(minimum) != 3
        or len(maximum) != 3
    ):
        raise FreeCADMCPError("FreeCAD assembly bounds evidence is invalid")
    try:
        numeric_minimum = [float(value) for value in minimum]
        numeric_maximum = [float(value) for value in maximum]
    except (TypeError, ValueError) as exc:
        raise FreeCADMCPError("FreeCAD assembly bounds evidence is invalid") from exc
    if (
        not all(math.isfinite(value) for value in [*numeric_minimum, *numeric_maximum])
        or any(low > high for low, high in zip(numeric_minimum, numeric_maximum))
    ):
        raise FreeCADMCPError("FreeCAD assembly bounds evidence is invalid")
    modules = checks.get("modules")
    if (
        not isinstance(modules, dict)
        or len(modules) != len(expected_module_ids)
        or set(modules) != set(expected_module_ids)
    ):
        raise FreeCADMCPError("FreeCAD per-module evidence does not match the state")
    if any(
        not isinstance(value, dict) or value.get("passed") is not True
        for value in modules.values()
    ):
        raise FreeCADValidationError(
            "One or more FreeCAD module checks failed",
            evidence,
        )
    centerlines = checks.get("centerlines")
    if (
        not isinstance(centerlines, dict)
        or len(centerlines) != len(expected_module_ids)
        or set(centerlines) != set(expected_module_ids)
    ):
        raise FreeCADMCPError("FreeCAD centerline evidence does not match the state")
    if any(
        not isinstance(value, dict) or value.get("passed") is not True
        for value in centerlines.values()
    ):
        raise FreeCADValidationError(
            "One or more FreeCAD centerline checks failed",
            evidence,
        )
    for name in (
        "non_adjacent_overlaps",
        "connection_failures",
        "terminal_bore_failures",
        "anchored_inlet_bore_failures",
        "termination_seal_failures",
        "wall_section_failures",
    ):
        if checks.get(name) != []:
            raise FreeCADValidationError(
                f"FreeCAD {name} evidence is not empty",
                evidence,
            )
    try:
        minimum_wall = float(checks["minimum_authored_wall_thickness"])
    except (TypeError, ValueError) as exc:
        raise FreeCADMCPError("FreeCAD minimum wall evidence is invalid") from exc
    if not math.isfinite(minimum_wall) or minimum_wall <= 0.0:
        raise FreeCADValidationError(
            "FreeCAD minimum wall thickness is not finite and positive",
            evidence,
        )
    sampled_sections = checks.get("sampled_internal_section_count")
    if not isinstance(sampled_sections, int) or sampled_sections < 0:
        raise FreeCADMCPError("FreeCAD internal wall sample count is invalid")
    if expected_internal_section_module_count is not None:
        declared_required = checks.get("required_internal_section_module_count")
        sampled_by_module = checks.get("sampled_internal_sections_by_module")
        if declared_required != expected_internal_section_module_count:
            raise FreeCADValidationError(
                "FreeCAD required internal-section module count mismatch",
                evidence,
            )
        if not isinstance(sampled_by_module, dict) or any(
            not isinstance(count, int) or count <= 0
            for count in sampled_by_module.values()
        ):
            raise FreeCADValidationError(
                "FreeCAD per-module internal wall sample evidence is invalid",
                evidence,
            )
        if (
            len(sampled_by_module) != expected_internal_section_module_count
            or sampled_sections < expected_internal_section_module_count
        ):
            raise FreeCADValidationError(
                "FreeCAD geometry-bearing modules lack internal wall samples",
                evidence,
            )
    if (
        expected_open_port_count is not None
        and checks.get("declared_downstream_open_port_count")
        != expected_open_port_count
    ):
        raise FreeCADValidationError(
            "FreeCAD declared downstream open-port count mismatch",
            evidence,
        )
    anchored_inlet_count = checks.get("anchored_inlet_count")
    if not isinstance(anchored_inlet_count, int) or anchored_inlet_count not in {0, 1}:
        raise FreeCADMCPError("FreeCAD anchored-inlet count is invalid")
    if (
        expected_anchored_inlet_count is not None
        and anchored_inlet_count != expected_anchored_inlet_count
    ):
        raise FreeCADValidationError(
            "FreeCAD anchored-inlet count mismatch",
            evidence,
        )
    if evidence.get("passed") is not True:
        raise FreeCADValidationError(
            "FreeCAD B-Rep validation failed: "
            + json.dumps(evidence.get("checks", {}), ensure_ascii=False)[:2000],
            evidence,
        )
    # Failed B-Reps naturally have no candidate shapes to fingerprint. Classify
    # their structured semantic evidence above so an LLM repair can proceed.
    # A passing candidate still cannot be published unless every expected shape
    # is digest-bound by an exact fingerprint.
    if expected_candidate_document is not None:
        fingerprints = evidence.get("candidate_shape_fingerprints")
        required_objects = {
            "PipeAssembly",
            *{f"solid_{item}" for item in expected_module_ids},
        }
        if (
            not isinstance(fingerprints, dict)
            or set(fingerprints) != required_objects
            or any(
                not isinstance(name, str)
                or not isinstance(value, str)
                or not re.fullmatch(r"[0-9a-f]{64}", value)
                for name, value in fingerprints.items()
            )
        ):
            raise FreeCADMCPError(
                "FreeCAD validation candidate shape fingerprints are invalid"
            )
    return evidence


def assess_freecad_publish(
    result: dict[str, Any],
    *,
    expected_digest: str,
    expected_document: str,
    expected_fcstd_path: str | None = None,
) -> dict[str, Any]:
    """게시 문서와 저장 파일이 검증된 후보 digest를 유지하는지 확인한다."""

    _raise_for_mcp_failure(result)
    payloads = _sentinel_payloads(result, _PUBLISH_PREFIX)
    if len(payloads) != 1:
        raise FreeCADMCPError(
            f"Expected exactly one CADGEN_PUBLISH sentinel, found {len(payloads)}"
        )
    evidence = payloads[0]
    if evidence.get("payload_digest") != expected_digest:
        raise FreeCADMCPError("Published FreeCAD digest mismatch")
    if evidence.get("published_document") != expected_document:
        raise FreeCADMCPError("Published FreeCAD document mismatch")
    if expected_fcstd_path is not None and (
        evidence.get("fcstd_path") != expected_fcstd_path
        or evidence.get("saved") is not True
    ):
        raise FreeCADMCPError("Published FreeCAD FCStd artifact was not saved")
    if expected_fcstd_path is not None:
        artifact = Path(expected_fcstd_path)
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise FreeCADMCPError("Published FreeCAD FCStd artifact is missing on the host")
    if evidence.get("passed") is not True:
        raise FreeCADMCPError("FreeCAD publish verification failed")
    return evidence


async def capture_freecad_views(
    settings: Settings,
    output_dir: Path,
    *,
    document_name: str,
    payload_digest: str,
    focus_object: str | None = None,
    views: tuple[str, ...] = (
        "Isometric",
        "Front",
        "Top",
        "Right",
        "Left",
        "Back",
    ),
    width: int = 640,
    height: int = 640,
) -> list[str]:
    """활성 문서와 digest를 전후 확인하며 요청한 검증 화면을 저장한다."""

    activate_probe = _document_probe_code(
        document_name, payload_digest, set_active=True
    )
    verify_probe = _document_probe_code(
        document_name, payload_digest, set_active=False
    )
    calls: list[tuple[str, dict[str, Any]]] = []
    for view_name in views:
        calls.append(
            (
                settings.freecad_mcp_execute_tool,
                {settings.freecad_mcp_execute_arg: activate_probe},
            )
        )
        args: dict[str, Any] = {
            "view_name": view_name,
            "width": width,
            "height": height,
        }
        if focus_object:
            args["focus_object"] = focus_object
        calls.append(("get_view", args))
        calls.append(
            (
                settings.freecad_mcp_execute_tool,
                {settings.freecad_mcp_execute_arg: verify_probe},
            )
        )
    results = await asyncio.wait_for(
        _call_freecad_tools(settings, calls),
        timeout=settings.freecad_mcp_timeout_sec,
    )
    expected = {
        "document": document_name,
        "payload_digest": payload_digest,
        "active_document": document_name,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, view_name in enumerate(views):
        before_result, result, after_result = results[index * 3 : index * 3 + 3]
        before = _document_evidence(before_result)
        after = _document_evidence(after_result)
        if before != expected or after != expected:
            raise FreeCADMCPError(
                f"Screenshot {view_name} document/digest binding changed during capture"
            )
        _raise_for_mcp_failure(result)
        images = _image_payloads(result)
        if len(images) != 1:
            raise FreeCADMCPError(
                f"Expected one image for {view_name}, found {len(images)}"
            )
        data, mime = images[0]
        suffix = ".png" if mime == "image/png" else ".jpg"
        path = output_dir / f"{view_name.lower()}{suffix}"
        _atomic_write_bytes(path, base64.b64decode(data))
        paths.append(str(path))
    return paths


def _document_probe_code(
    document_name: str,
    payload_digest: str,
    *,
    set_active: bool,
) -> str:
    return f'''import json\nimport FreeCAD as App\nimport FreeCADGui as Gui\ndoc = App.getDocument({document_name!r})\nif {set_active!r}:\n    App.setActiveDocument(doc.Name)\n    Gui.Selection.clearSelection()\n    Gui.activeDocument().activeView().fitAll()\nobj = doc.getObject("PipeAssembly")\nactual = getattr(obj, "CadGenPayloadDigest", "") if obj else ""\nactive = App.ActiveDocument.Name if App.ActiveDocument else ""\nprint("{_DOCUMENT_PREFIX}" + json.dumps({{"document": doc.Name, "payload_digest": actual, "active_document": active}}, sort_keys=True, separators=(",", ":")))\n'''


def _document_evidence(result: dict[str, Any]) -> dict[str, Any]:
    _raise_for_mcp_failure(result)
    payloads = _sentinel_payloads(result, _DOCUMENT_PREFIX)
    if len(payloads) != 1:
        raise FreeCADMCPError("Document digest probe did not return exactly one sentinel")
    return payloads[0]


def _raise_for_mcp_failure(result: dict[str, Any]) -> None:
    if result.get("isError") is True or result.get("is_error") is True:
        raise FreeCADMCPError("FreeCAD MCP returned isError=true")
    if result.get("error") not in (None, False, "", {}):
        raise FreeCADMCPError(f"FreeCAD MCP returned an error: {result['error']!s}")
    # Structured sentinel JSON may legitimately contain words such as
    # "execution error" in a module-level semantic failure.  Only scan the
    # transport text before a sentinel; the assessor below owns its payload.
    transport_lines: list[str] = []
    prefixes = (
        _VALIDATION_PREFIX,
        _PUBLISH_PREFIX,
        _DOCUMENT_PREFIX,
        _READY_PREFIX,
        _VISUAL_READY_PREFIX,
    )
    for payload in _text_payloads(result):
        for line in payload.splitlines():
            marker_positions = [
                line.find(prefix) for prefix in prefixes if prefix in line
            ]
            if marker_positions:
                line = line[: min(marker_positions)]
            if line.strip():
                transport_lines.append(line)
    text = "\n".join(transport_lines)
    if re.search(
        r"(?:failed to execute code|error executing code|traceback \(most recent call last\)|execution error|uncaught exception)",
        text,
        flags=re.IGNORECASE,
    ):
        raise FreeCADMCPError(f"FreeCAD MCP reported execution failure: {text[:2000]}")


def _sentinel_payloads(result: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for text in dict.fromkeys(_text_payloads(result)):
        for line in text.splitlines():
            marker_index = line.find(prefix)
            if marker_index < 0:
                continue
            raw = line[marker_index + len(prefix):].strip()
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise FreeCADMCPError(f"Malformed {prefix[:-1]} sentinel") from exc
            if not isinstance(value, dict):
                raise FreeCADMCPError(f"{prefix[:-1]} sentinel must be an object")
            payloads.append(value)
    return payloads


def _text_payloads(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            texts.append(value["text"])
        for child in value.values():
            texts.extend(_text_payloads(child))
    elif isinstance(value, list):
        for child in value:
            texts.extend(_text_payloads(child))
    return texts


def _image_payloads(value: Any) -> list[tuple[str, str]]:
    images: list[tuple[str, str]] = []
    if isinstance(value, dict):
        if value.get("type") == "image" and isinstance(value.get("data"), str):
            mime = value.get("mimeType") or value.get("mime_type") or "image/png"
            images.append((value["data"], str(mime)))
        for child in value.values():
            images.extend(_image_payloads(child))
    elif isinstance(value, list):
        for child in value:
            images.extend(_image_payloads(child))
    return list(dict.fromkeys(images))


def _jsonable(value: Any) -> dict[str, Any]:
    try:
        raw = value.model_dump()
    except AttributeError:
        try:
            raw = vars(value)
        except TypeError:
            raw = {"result": repr(value)}
    return json.loads(json.dumps(raw, default=repr))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
