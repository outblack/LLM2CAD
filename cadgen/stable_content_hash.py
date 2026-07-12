"""결정론적 JSON 직렬화 기반 콘텐츠 해시를 제공한다.

충돌 인증서, 계약 ledger, 진단 케이스 등 동일 페이로드를 동일 digest로
묶어야 하는 모듈이 공유한다. 해시 알고리즘 변경은 재현성 계약을 깨뜨리므로
이 모듈 한곳에서만 수행한다.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_digest(payload: Any, *, allow_nan: bool = True) -> str:
    """정렬된 JSON 직렬화로 SHA-256 hex digest를 계산한다."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
        allow_nan=allow_nan,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["stable_digest"]
