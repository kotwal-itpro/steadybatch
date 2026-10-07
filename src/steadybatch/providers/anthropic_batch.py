"""Anthropic Message Batches API.

Flow: create a batch with up to 100,000 requests in one call, poll until
processing_status is "ended", then stream results. Each result is
succeeded, errored, canceled or expired.
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

    def __init__(self, client=None):
        if client is None:
            import anthropic  # imported here so the package works without it
            client = anthropic.Anthropic()
        self.client = client

    def to_line(self, req: PreparedRequest) -> dict:
        params = {
            "model": req.model,
            "max_tokens": req.request.max_tokens,
            "temperature": req.request.temperature,
            "messages": req.request.messages,
        }
        system = req.request.system or ""
        if req.response_schema is not None:
            # Ask for JSON in the prompt; the runner checks every answer against the schema.
            system = (system + "\n\nReply with only a JSON object that matches this JSON Schema:\n"
                      + json.dumps(req.response_schema)).strip()
        if system:
            params["system"] = system
        return {"custom_id": req.custom_id, "params": params}

    def submit(self, batch: list[PreparedRequest]) -> str:
        created = self.client.messages.batches.create(requests=[self.to_line(r) for r in batch])
        return created.id

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
                text = "".join(getattr(block, "text", "") for block in msg.content)
                truncated = getattr(msg, "stop_reason", None) == "max_tokens"
                yield RawResult(entry.custom_id, ok=True, text=text,
                                error="truncated (max_tokens)" if truncated else None,
                                input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens)
            elif result.type == "errored":
                yield RawResult(entry.custom_id, ok=False, error=str(getattr(result, "error", "errored")))
            else:
                yield RawResult(entry.custom_id, ok=False, error=result.type)
