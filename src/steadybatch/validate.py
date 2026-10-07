"""Check that a model's answer is the JSON you asked for."""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class InvalidOutput(ValueError):
    pass


def parse_json(text: str | None) -> Any:
    if text is None or not text.strip():
        raise InvalidOutput("empty response")
    match = _FENCE.match(text)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidOutput(f"not valid JSON: {exc.msg} at char {exc.pos}") from exc


def check(text: str | None, schema: dict[str, Any] | None) -> Any:
    """Return the parsed object, or raise InvalidOutput with a short reason."""
    data = parse_json(text)
    if schema is not None:
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as exc:
            where = "/".join(str(p) for p in exc.absolute_path) or "(root)"
            raise InvalidOutput(f"schema: {where}: {exc.message}") from exc
    return data
