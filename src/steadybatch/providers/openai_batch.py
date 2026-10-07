"""OpenAI Batch API.

Flow: write a JSONL file, upload it with purpose "batch", create a batch
against /v1/chat/completions, poll, then read the output file and the
error file. Lines can appear in either file, or in neither.
"""

from __future__ import annotations

import json
from typing import Iterator

from ..models import BatchState, BatchStatus, PreparedRequest, RawResult
from .base import BatchProvider

_DONE = {"completed"}
_FAILED = {"failed", "expired", "cancelled", "cancelling"}


class OpenAIBatch(BatchProvider):
    name = "openai"
    max_requests_per_batch = 50_000
    max_bytes_per_batch = 200 * 1024 * 1024

    def __init__(self, client=None, *, strict_schema: bool = True):
        if client is None:
            from openai import OpenAI  # imported here so the package works without it
            client = OpenAI()
        self.client = client
        self.strict_schema = strict_schema

    def to_line(self, req: PreparedRequest) -> dict:
        messages = list(req.request.messages)
        if req.request.system:
            messages = [{"role": "system", "content": req.request.system}] + messages
        body = {
            "model": req.model,
            "messages": messages,
            "max_tokens": req.request.max_tokens,
            "temperature": req.request.temperature,
        }
        if req.response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": req.response_schema,
                                "strict": self.strict_schema},
            }
        return {"custom_id": req.custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}

    def submit(self, batch: list[PreparedRequest]) -> str:
        payload = "\n".join(json.dumps(self.to_line(r)) for r in batch).encode("utf-8")
        upload = self.client.files.create(file=("batch.jsonl", payload), purpose="batch")
        created = self.client.batches.create(
            input_file_id=upload.id, endpoint="/v1/chat/completions", completion_window="24h"
        )
        return created.id

    def status(self, batch_id: str) -> BatchStatus:
        b = self.client.batches.retrieve(batch_id)
        counts = b.request_counts
        if b.status in _DONE:
            state = BatchState.DONE
        elif b.status in _FAILED:
            state = BatchState.FAILED
        elif b.status == "validating":
            state = BatchState.PENDING
        else:
            state = BatchState.RUNNING
        return BatchStatus(state, total=getattr(counts, "total", 0) or 0,
                           succeeded=getattr(counts, "completed", 0) or 0,
                           failed=getattr(counts, "failed", 0) or 0, detail=b.status)

    def results(self, batch_id: str) -> Iterator[RawResult]:
        b = self.client.batches.retrieve(batch_id)
        for file_id in (b.output_file_id, b.error_file_id):
            if not file_id:
                continue
            for line in self.client.files.content(file_id).text.splitlines():
                if line.strip():
                    yield self._parse(json.loads(line))

    @staticmethod
    def _parse(line: dict) -> RawResult:
        cid = line.get("custom_id", "")
        if line.get("error"):
            return RawResult(cid, ok=False, error=json.dumps(line["error"]))
        response = line.get("response") or {}
        body = response.get("body") or {}
        if response.get("status_code", 200) != 200:
            return RawResult(cid, ok=False, error=f"HTTP {response.get('status_code')}: {json.dumps(body)[:500]}")
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}
        text = message.get("content")
        if message.get("refusal"):
            return RawResult(cid, ok=False, error=f"refusal: {message['refusal']}")
        if choice.get("finish_reason") == "length":
            return RawResult(cid, ok=True, text=text, error="truncated (max_tokens)",
                             input_tokens=usage.get("prompt_tokens", 0),
                             output_tokens=usage.get("completion_tokens", 0))
        return RawResult(cid, ok=True, text=text, input_tokens=usage.get("prompt_tokens", 0),
                         output_tokens=usage.get("completion_tokens", 0))
