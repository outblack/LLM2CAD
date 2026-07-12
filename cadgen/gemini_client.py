"""호환용 별칭 모듈 — 실제 구현은 ``cadgen.gemini_llm_client`` 에 있다.

역할: Gemini LLM 클라이언트
새 코드에서는 ``from cadgen.gemini_llm_client import ...`` 를 사용하세요.
이 파일은 기존 import 경로를 깨지 않기 위한 re-export 전용이다.
"""

from __future__ import annotations

from cadgen.gemini_llm_client import *  # noqa: F403
