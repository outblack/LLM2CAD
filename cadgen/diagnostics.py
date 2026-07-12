"""호환용 별칭 모듈 — 실제 구현은 ``cadgen.step_geometry_diagnostics`` 에 있다.

역할: step 기하 진단
새 코드에서는 ``from cadgen.step_geometry_diagnostics import ...`` 를 사용하세요.
이 파일은 기존 import 경로를 깨지 않기 위한 re-export 전용이다.
"""

from __future__ import annotations

from cadgen.step_geometry_diagnostics import *  # noqa: F403

from cadgen.step_geometry_diagnostics import __all__ as __all__  # noqa: E402,F401
