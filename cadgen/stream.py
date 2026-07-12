"""호환용 별칭 모듈 — 실제 구현은 ``cadgen.thinking_progress_stream`` 에 있다.

역할: thinking 진행 스트림
새 코드에서는 ``from cadgen.thinking_progress_stream import ...`` 를 사용하세요.
이 파일은 기존 import 경로를 깨지 않기 위한 re-export 전용이다.
"""

from __future__ import annotations

from cadgen.thinking_progress_stream import *  # noqa: F403
