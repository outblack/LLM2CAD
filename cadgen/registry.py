"""호환용 별칭 모듈 — 실제 구현은 ``cadgen.primitive_action_catalog`` 에 있다.

역할: primitive 카탈로그·검증
새 코드에서는 ``from cadgen.primitive_action_catalog import ...`` 를 사용하세요.
이 파일은 기존 import 경로를 깨지 않기 위한 re-export 전용이다.
"""

from __future__ import annotations

from cadgen.primitive_action_catalog import *  # noqa: F403
