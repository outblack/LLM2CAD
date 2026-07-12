"""호환용 별칭 모듈 — 실제 구현은 ``cadgen.geometry_safety_policy`` 에 있다.

역할: 기하 안전 정책
새 코드에서는 ``from cadgen.geometry_safety_policy import ...`` 를 사용하세요.
이 파일은 기존 import 경로를 깨지 않기 위한 re-export 전용이다.
"""

from __future__ import annotations

from cadgen.geometry_safety_policy import *  # noqa: F403
