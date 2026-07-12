"""호환용 별칭 모듈 — 실제 구현은 ``cadgen.generation_pipeline`` 에 있다.

역할: CAD 생성 파이프라인
새 코드에서는 ``from cadgen.generation_pipeline import ...`` 를 사용하세요.
이 파일은 기존 import 경로를 깨지 않기 위한 re-export 전용이다.
"""

from __future__ import annotations

from cadgen.generation_pipeline import *  # noqa: F403
