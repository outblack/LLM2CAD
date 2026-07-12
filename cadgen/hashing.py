"""호환용 별칭 모듈 — 실제 구현은 ``cadgen.stable_content_hash`` 에 있다.

역할: 결정론 콘텐츠 해시
새 코드에서는 ``from cadgen.stable_content_hash import ...`` 를 사용하세요.
이 파일은 기존 import 경로를 깨지 않기 위한 re-export 전용이다.
"""

from __future__ import annotations

from cadgen.stable_content_hash import *  # noqa: F403

from cadgen.stable_content_hash import __all__ as __all__  # noqa: E402,F401
