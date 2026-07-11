from __future__ import annotations

import pytest

import cadgen.pipeline as pipeline
import cadgen.freecad_mcp as freecad_mcp


@pytest.fixture(autouse=True)
def forbid_pipeline_external_processes(monkeypatch):
    """Keep the static suite hermetic: no app launch, uvx, MCP, or paid API."""

    async def ready_probe(unused_settings, *, executor=None):
        del unused_settings, executor
        return {"passed": True, "protocol": 1}

    monkeypatch.setattr(pipeline, "probe_freecad_mcp", ready_probe)

    async def ready_visual_probe(unused_settings):
        del unused_settings

    monkeypatch.setattr(pipeline, "probe_freecad_visual", ready_visual_probe)
    monkeypatch.setattr(pipeline, "ensure_freecad_open", lambda settings, stream: False)

    async def forbid_live_mcp(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "Live FreeCAD MCP is forbidden in the hermetic static test suite"
        )

    monkeypatch.setattr(freecad_mcp, "_call_freecad_tools", forbid_live_mcp)

    class ForbidGeminiClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError(
                "Real Gemini is forbidden in the hermetic static test suite"
            )

    monkeypatch.setattr(pipeline, "GeminiClient", ForbidGeminiClient)
