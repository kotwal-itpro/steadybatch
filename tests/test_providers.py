"""Check the OpenAI and Anthropic adapters build and parse the right shapes.

These use small stand-in clients, so they run without network or API keys.
"""

import json
from types import SimpleNamespace as NS

from steadybatch.models import BatchState, PreparedRequest, Request
from steadybatch.providers.anthropic_batch import AnthropicBatch
from steadybatch.providers.openai_batch import OpenAIBatch

SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"],
          "additionalProperties": False}


def prepared(cid="sb-1"):
    return PreparedRequest(cid, Request(key="k", system="be brief",
                                        messages=[{"role": "user", "content": "hi"}]), "m", SCHEMA)


def test_openai_line_shape():
    line = OpenAIBatch(client=object()).to_line(prepared())
    assert line["url"] == "/v1/chat/completions" and line["method"] == "POST"
    assert line["body"]["messages"][0] == {"role": "system", "content": "be brief"}
    assert line["body"]["response_format"]["type"] == "json_schema"


def test_openai_parses_success_error_refusal_and_truncation():
    ok = OpenAIBatch._parse({"custom_id": "a", "response": {"status_code": 200, "body": {
        "choices": [{"message": {"content": '{"x":"y"}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3}}}})
    assert ok.ok and ok.text == '{"x":"y"}' and ok.input_tokens == 10
    err = OpenAIBatch._parse({"custom_id": "b", "error": {"code": "rate_limit"}})
    assert not err.ok and "rate_limit" in err.error
    refused = OpenAIBatch._parse({"custom_id": "c", "response": {"status_code": 200, "body": {
        "choices": [{"message": {"content": None, "refusal": "no"}}]}}})
    assert not refused.ok and refused.error.startswith("refusal")
    cut = OpenAIBatch._parse({"custom_id": "d", "response": {"status_code": 200, "body": {
        "choices": [{"message": {"content": '{"x":'}, "finish_reason": "length"}]}}})
    assert cut.ok and "truncated" in cut.error


def test_openai_submit_status_results_round_trip():
    uploads = {}

    class Files:
        def create(self, file, purpose):
            uploads["body"] = file[1]
            return NS(id="file-in")

        def content(self, file_id):
            out = {"custom_id": "sb-1", "response": {"status_code": 200, "body": {
                "choices": [{"message": {"content": '{"x":"1"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}}}
            return NS(text=json.dumps(out) if file_id == "file-out" else "")

    class Batches:
        def create(self, **kw):
            return NS(id="batch-1")

        def retrieve(self, batch_id):
            return NS(status="completed", output_file_id="file-out", error_file_id=None,
                      request_counts=NS(total=1, completed=1, failed=0))

    p = OpenAIBatch(client=NS(files=Files(), batches=Batches()))
    assert p.submit([prepared()]) == "batch-1"
    assert json.loads(uploads["body"].decode())["custom_id"] == "sb-1"
    assert p.status("batch-1").state is BatchState.DONE
    assert [r.custom_id for r in p.results("batch-1")] == ["sb-1"]


def test_anthropic_line_puts_schema_in_system_prompt():
    line = AnthropicBatch(client=object()).to_line(prepared())
    assert line["custom_id"] == "sb-1"
    assert "JSON Schema" in line["params"]["system"]
    assert line["params"]["system"].startswith("be brief")


def test_anthropic_results_cover_every_result_type():
    msg = NS(content=[NS(text='{"x":"1"}')], usage=NS(input_tokens=5, output_tokens=2), stop_reason="end_turn")
    entries = [
        NS(custom_id="a", result=NS(type="succeeded", message=msg)),
        NS(custom_id="b", result=NS(type="errored", error="overloaded")),
        NS(custom_id="c", result=NS(type="expired")),
    ]
    client = NS(messages=NS(batches=NS(results=lambda bid: iter(entries))))
    out = list(AnthropicBatch(client=client).results("x"))
    assert [(r.custom_id, r.ok) for r in out] == [("a", True), ("b", False), ("c", False)]
    assert out[0].input_tokens == 5 and out[2].error == "expired"
