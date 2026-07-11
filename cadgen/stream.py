"""파이프라인 진행 상황을 짧은 stderr 메시지로 전달한다.

활성화 설정과 메시지를 입력받아 과도한 출력을 제한한 진행 로그를 남긴다.
출력 실패나 비활성 상태가 CAD 상태와 결과를 변경하지 않는다.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field


@dataclass
class ThinkingStream:
    """짧은 간격 제한을 적용해 진행 메시지를 stderr로 출력한다."""

    enabled: bool = True
    _last_emit: float = field(default=0.0, init=False)

    def emit(self, message: str, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_emit < 0.05:
            return
        self._last_emit = now
        print(f"thinking: {message}", file=sys.stderr, flush=True)
