"""호환용 별칭 모듈 — 실제 구현은 ``cadgen.freecad_mcp_client`` 에 있다.

역할: FreeCAD MCP 클라이언트
새 코드에서는 ``from cadgen.freecad_mcp_client import ...`` 를 사용하세요.
이 파일은 기존 import 경로를 깨지 않기 위한 re-export 전용이다.
"""

from __future__ import annotations

from cadgen.freecad_mcp_client import *  # noqa: F403
