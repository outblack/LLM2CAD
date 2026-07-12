"""호환용 별칭 모듈 — 실제 구현은 ``cadgen.vector3_math`` 에 있다.

역할: 3D 벡터 연산
새 코드에서는 ``from cadgen.vector3_math import ...`` 를 사용하세요.
이 파일은 기존 import 경로를 깨지 않기 위한 re-export 전용이다.
"""

from __future__ import annotations

from cadgen.vector3_math import *  # noqa: F403
