"""Stable request IDs.

The same record with the same settings always gets the same ID. That is
what makes retries safe: if a line comes back twice, or you resubmit work
after a crash, you can tell it is the same request and not count it twice.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import Request

# Anthropic allows [a-zA-Z0-9_-]{1,64}; OpenAI is looser. Stay within both.
_PREFIX = "sb-"
_HASH_CHARS = 32


def custom_id_for(request: Request, model: str, response_schema: dict[str, Any] | None = None) -> str:
    payload = {
        "key": request.key,
        "model": model,
        "messages": request.messages,
        "system": request.system,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "schema": response_schema,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _PREFIX + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:_HASH_CHARS]
