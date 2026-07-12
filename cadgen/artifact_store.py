"""호환용 별칭 모듈 — 실제 구현은 ``cadgen.run_artifact_store`` 에 있다.

역할: 실행 산출물 원자적 저장
새 코드에서는 ``from cadgen.run_artifact_store import ...`` 를 사용하세요.
이 파일은 기존 import 경로를 깨지 않기 위한 re-export 전용이다.
"""

from __future__ import annotations

from cadgen.run_artifact_store import *  # noqa: F403

from cadgen.run_artifact_store import __all__ as __all__  # noqa: E402,F401
