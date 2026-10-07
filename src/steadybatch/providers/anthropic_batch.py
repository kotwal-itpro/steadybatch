"""Anthropic Message Batches API.

Flow: create a batch with up to 100,000 requests in one call, poll until
processing_status is "ended", then stream results. Each result is
succeeded, errored, canceled or expired.

When a JSON Schema is given, it is sent as a structured output format
(`output_config.format`), so the reply is constrained to the schema, the
same as the OpenAI and Gemini adapters. Pass `native_schema=False` to
describe the schema in the system prompt instead. `thinking` ("disabled" or
"adaptive") is sent when given; left out, the model uses its default, and on
Claude Haiku 5.5 that default is adaptive thinking, which counts against
max_tokens. `temperature` is not sent:
current Claude models reject non-default sampling settings, and anthropic
SDK 1.x no longer takes the parameter.
"""

from __future__ import annotations

import json
from typing import Iterator

from ..models import BatchState, BatchStatus, PreparedRequest, RawResult
from .base import BatchProvider


class AnthropicBatch(BatchProvider):
    name = "anthropic"
    max_requests_per_batch = 100_000
    max_bytes_per_batch = 256 * 1024 * 1024

    def __init__(self, client=None, *, native_schema: bool = True, thinking: str | None = None):
        if client is None:
            import anthropic  # imported here so the package works without it
            client = anthropic.Anthropic()
        self.client = client
        self.native_schema = native_schema
        self.thinking = thinking

    def to_line(self, req: PreparedRequest) -> dict:
        params = {
            "model": req.model,
            "max_tokens": req.request.max_tokens,
            "messages": req.request.messages,
        }
        if self.thinking:
            params["thinking"] = {"type": self.thinking}
        system = req.request.system or ""
        if req.response_schema is not None and self.native_schema:
            params["output_config"] = {"format": {"type": "json_schema", "schema": req.response_schema}}
        elif req.response_schema is not None:
            # Ask for JSON in the prompt; the runner checks every answer against the schema.
            system = (system + "\n\nReply with only a JSON object that matches this JSON Schema:\n"
                      + json.dumps(req.response_schema)).strip()
        if system:
            params["system"] = system
        return {"custom_id": req.custom_id, "params": params}

    def submit(self, batch: list[PreparedRequest]) -> str:
        created = self.client.messages.batches.create(requests=[self.to_line(r) for r in batch])
        return created.id

    def is_over_capacity(self, exc: Exception) -> bool:
        return getattr(exc, "status_code", None) == 429

    def status(self, batch_id: str) -> BatchStatus:
        b = self.client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        if b.processing_status == "ended":
            state = BatchState.DONE
        elif b.processing_status == "canceling":
            state = BatchState.FAILED
        else:
            state = BatchState.RUNNING
        total = c.processing + c.succeeded + c.errored + c.canceled + c.expired
        return BatchStatus(state, total=total, succeeded=c.succeeded,
                           failed=c.errored + c.canceled + c.expired, detail=b.processing_status)

    def results(self, batch_id: str) -> Iterator[RawResult]:
        for entry in self.client.messages.batches.results(batch_id):
            result = entry.result
            if result.type == "succeeded":
                msg = result.message
                if getattr(msg, "stop_reason", None) == "refusal":
                    yield RawResult(entry.custom_id, ok=False, error="refusal")
                    continue
                text = "".join(block.text for block in msg.content if block.type == "text")
                truncated = getattr(msg, "stop_reason", None) == "max_tokens"
                yield RawResult(entry.custom_id, ok=True, text=text,
                                error="truncated (max_tokens)" if truncated else None,
                                input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens)
            elif result.type == "errored":
                yield RawResult(entry.custom_id, ok=False, error=str(getattr(result, "error", "errored")))
            else:
                yield RawResult(entry.custom_id, ok=False, error=result.type)
