import json

import pytest

from steadybatch.chunking import chunk
from steadybatch.ids import custom_id_for
from steadybatch.models import Request
from steadybatch.validate import InvalidOutput, check

SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["a", "b"]}},
    "required": ["label"],
    "additionalProperties": False,
}


def req(key="r1", text="hello"):
    return Request(key=key, messages=[{"role": "user", "content": text}])


def test_custom_id_is_stable_and_provider_safe():
    a = custom_id_for(req(), "m1")
    assert a == custom_id_for(req(), "m1")
    assert len(a) <= 64 and a.replace("-", "").replace("_", "").isalnum()


def test_custom_id_changes_when_anything_that_matters_changes():
    base = custom_id_for(req(), "m1")
    assert custom_id_for(req(text="other"), "m1") != base
    assert custom_id_for(req(), "m2") != base
    assert custom_id_for(req(), "m1", SCHEMA) != base


def test_chunk_respects_count_and_size():
    items = list(range(10))
    batches = list(chunk(items, max_count=3, max_bytes=1000, size_of=lambda _: 10))
    assert [len(b) for b in batches] == [3, 3, 3, 1]
    batches = list(chunk(items, max_count=100, max_bytes=50, size_of=lambda _: 10, margin=1.0))
    assert [len(b) for b in batches] == [5, 5]


def test_chunk_rejects_a_single_oversized_item():
    with pytest.raises(ValueError):
        list(chunk([1], max_count=10, max_bytes=100, size_of=lambda _: 200))


def test_check_accepts_valid_json_and_code_fences():
    assert check('{"label": "a"}', SCHEMA) == {"label": "a"}
    assert check('```json\n{"label": "b"}\n```', SCHEMA) == {"label": "b"}


@pytest.mark.parametrize("text, reason", [
    ("", "empty"),
    ("Sure! {not json", "not valid JSON"),
    ('{"label": "c"}', "schema"),
    ('{"label": "a", "extra": 1}', "schema"),
])
def test_check_rejects_bad_output_with_a_reason(text, reason):
    with pytest.raises(InvalidOutput, match=reason):
        check(text, SCHEMA)
