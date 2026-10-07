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


def test_anthropic_line_uses_structured_output_format():
    line = AnthropicBatch(client=object()).to_line(prepared())
    assert line["custom_id"] == "sb-1"
    params = line["params"]
    assert params["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert params["system"] == "be brief" and "temperature" not in params


def test_anthropic_sends_thinking_only_when_set():
    assert "thinking" not in AnthropicBatch(client=object()).to_line(prepared())["params"]
    params = AnthropicBatch(client=object(), thinking="disabled").to_line(prepared())["params"]
    assert params["thinking"] == {"type": "disabled"}


def test_anthropic_can_fall_back_to_schema_in_system_prompt():
    params = AnthropicBatch(client=object(), native_schema=False).to_line(prepared())["params"]
    assert "output_config" not in params
    assert params["system"].startswith("be brief") and "JSON Schema" in params["system"]


def test_anthropic_results_cover_every_result_type():
    usage = NS(input_tokens=5, output_tokens=2)
    msg = NS(content=[NS(type="thinking", thinking=""), NS(type="text", text='{"x":"1"}')],
             usage=usage, stop_reason="end_turn")
    refused = NS(content=[], usage=usage, stop_reason="refusal")
    entries = [
        NS(custom_id="a", result=NS(type="succeeded", message=msg)),
        NS(custom_id="b", result=NS(type="errored", error="overloaded")),
        NS(custom_id="c", result=NS(type="expired")),
        NS(custom_id="d", result=NS(type="succeeded", message=refused)),
    ]
    client = NS(messages=NS(batches=NS(results=lambda bid: iter(entries))))
    out = list(AnthropicBatch(client=client).results("x"))
    assert [(r.custom_id, r.ok) for r in out] == [("a", True), ("b", False), ("c", False), ("d", False)]
    assert out[0].text == '{"x":"1"}' and out[0].input_tokens == 5
    assert out[2].error == "expired" and out[3].error == "refusal"


# --- Gemini -------------------------------------------------------------

from steadybatch.providers.gemini_batch import GeminiBatch


def test_gemini_line_uses_key_and_native_json_schema():
    line = GeminiBatch(client=object()).to_line(prepared())
    assert line["key"] == "sb-1"
    req = line["request"]
    assert req["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert req["systemInstruction"] == {"parts": [{"text": "be brief"}]}
    assert req["generationConfig"]["responseMimeType"] == "application/json"
    assert req["generationConfig"]["responseJsonSchema"] == SCHEMA


def test_gemini_can_fall_back_to_schema_in_the_prompt():
    line = GeminiBatch(client=object(), native_schema=False).to_line(prepared())
    assert "responseJsonSchema" not in line["request"]["generationConfig"]
    assert "JSON Schema" in line["request"]["systemInstruction"]["parts"][0]["text"]


def test_gemini_parses_success_error_block_and_truncation():
    ok = GeminiBatch._parse({"key": "a", "response": {
        "candidates": [{"content": {"parts": [{"text": '{"x":'}, {"text": '"1"}'}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}}})
    assert ok.ok and ok.text == '{"x":"1"}' and ok.input_tokens == 7 and ok.output_tokens == 3
    err = GeminiBatch._parse({"key": "b", "error": {"code": 429, "message": "quota"}})
    assert not err.ok and "quota" in err.error
    blocked = GeminiBatch._parse({"key": "c", "response": {"promptFeedback": {"blockReason": "SAFETY"}}})
    assert not blocked.ok and "SAFETY" in blocked.error
    cut = GeminiBatch._parse({"key": "d", "response": {
        "candidates": [{"content": {"parts": [{"text": '{"x":'}]}, "finishReason": "MAX_TOKENS"}]}})
    assert cut.ok and "truncated" in cut.error


def test_gemini_submit_status_results_round_trip():
    seen = {}
    out_line = {"key": "sb-1", "response": {"candidates": [
        {"content": {"parts": [{"text": '{"x":"1"}'}]}, "finishReason": "STOP"}]}}

    class Files:
        def upload(self, file, config):
            seen["upload"] = file.read().decode()
            seen["mime"] = config["mime_type"]
            return NS(name="files/in")

        def download(self, file):
            assert file == "files/out"
            return (json.dumps(out_line) + "\n").encode()

    class Batches:
        def create(self, model, src, config):
            seen["model"], seen["src"] = model, src
            return NS(name="batches/1")

        def get(self, name):
            return NS(state=NS(name="JOB_STATE_SUCCEEDED"), dest=NS(file_name="files/out"),
                      completion_stats=NS(successful_count=1, failed_count=0))

    p = GeminiBatch(client=NS(files=Files(), batches=Batches()))
    assert p.submit([prepared()]) == "batches/1"
    assert json.loads(seen["upload"])["key"] == "sb-1" and seen["mime"] == "jsonl"
    assert seen["model"] == "m" and seen["src"] == "files/in"
    status = p.status("batches/1")
    assert status.state is BatchState.DONE and status.succeeded == 1
    assert [r.custom_id for r in p.results("batches/1")] == ["sb-1"]


def test_gemini_partial_success_counts_as_done_so_missing_lines_get_retried():
    job = NS(state=NS(name="JOB_STATE_PARTIALLY_SUCCEEDED"), dest=None,
             completion_stats=NS(successful_count=8, failed_count=2))
    p = GeminiBatch(client=NS(batches=NS(get=lambda name: job)))
    assert p.status("b").state is BatchState.DONE
    assert list(p.results("b")) == []


def test_openai_reasoning_models_use_max_completion_tokens_and_no_temperature():
    gpt5 = PreparedRequest("sb-2", Request(key="k", messages=[{"role": "user", "content": "hi"}]), "gpt-5-mini")
    body = OpenAIBatch(client=object()).to_line(gpt5)["body"]
    assert body["max_completion_tokens"] == 1024
    assert "max_tokens" not in body and "temperature" not in body
    body = OpenAIBatch(client=object()).to_line(prepared())["body"]   # model "m": a regular chat model
    assert body["max_tokens"] == 1024 and body["temperature"] == 0.0


def test_rejected_gemini_job_deletes_its_uploaded_file():
    import pytest
    deleted = []

    class Files:
        def upload(self, file, config):
            return NS(name="files/in")

        def delete(self, name):
            deleted.append(name)

    class Batches:
        def create(self, **kw):
            raise RuntimeError("404 model no longer available to new users")

    p = GeminiBatch(client=NS(files=Files(), batches=Batches()))
    with pytest.raises(RuntimeError):
        p.submit([prepared()])
    assert deleted == ["files/in"]


# --- vLLM ----------------------------------------------------------------

from steadybatch.providers.vllm_offline import VLLMOffline, build_messages


def test_vllm_messages_carry_system_prompt_and_schema():
    msgs = build_messages(prepared())
    assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith("be brief")
    assert "JSON Schema" in msgs[0]["content"] and msgs[1] == {"role": "user", "content": "hi"}


def test_vllm_submit_maps_outputs_in_order_and_flags_truncation():
    class FakeLLM:
        def chat(self, conversations, params, use_tqdm=False):
            self.seen = conversations
            return [NS(prompt_token_ids=[1, 2, 3], outputs=[NS(text='{"x":"1"}', token_ids=[9, 9], finish_reason="stop")]),
                    NS(prompt_token_ids=[1], outputs=[NS(text='{"x":', token_ids=[9], finish_reason="length")])]

    class NoParams(VLLMOffline):
        def sampling_params(self, req):   # vllm isn't installed on the test machine
            return None

    p = NoParams("m", llm=FakeLLM())
    bid = p.submit([prepared("a"), prepared("b")])
    out = list(p.results(bid))
    assert [(r.custom_id, r.input_tokens, r.output_tokens) for r in out] == [("a", 3, 2), ("b", 1, 1)]
    assert out[0].error is None and "truncated" in out[1].error
    assert p.status(bid).state is BatchState.DONE
