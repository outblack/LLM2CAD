"""호환용 별칭 모듈 — 실제 구현은 ``cadgen.typed_data_models`` 에 있다.

역할: typed 데이터 모델
새 코드에서는 ``from cadgen.typed_data_models import ...`` 를 사용하세요.
이 파일은 기존 import 경로를 깨지 않기 위한 re-export 전용이다.
"""

from __future__ import annotations

from cadgen.typed_data_models import *  # noqa: F403
