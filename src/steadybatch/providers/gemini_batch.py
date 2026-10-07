"""Gemini API batch mode (google-genai SDK).

Flow: write a JSONL file where each line is {"key": ..., "request": {...}},
upload it, create a batch job from the uploaded file, poll the job, then
download the output file. Each output line carries the same "key" and
either a "response" or an "error". We use the file form rather than inline
requests because the key is what lets us check off every line.
"""

from __future__ import annotations

import io
import json
from typing import Iterator

from ..models import BatchState, BatchStatus, PreparedRequest, RawResult
from .base import BatchProvider

_DONE = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
_FAILED = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_CANCELLING", "JOB_STATE_EXPIRED"}
_PENDING = {"JOB_STATE_QUEUED", "JOB_STATE_PENDING", "JOB_STATE_UNSPECIFIED"}


class GeminiBatch(BatchProvider):
    name = "gemini"
    max_requests_per_batch = 50_000          # conservative; the input file itself may be up to 2 GB
    max_bytes_per_batch = 2 * 1024 * 1024 * 1024

    def __init__(self, client=None, *, native_schema: bool = True):
        if client is None:
            from google import genai  # imported here so the package works without it
            client = genai.Client()
        self.client = client
        self.native_schema = native_schema

    def to_line(self, req: PreparedRequest) -> dict:
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in req.request.messages
        ]
        config: dict = {"temperature": req.request.temperature, "maxOutputTokens": req.request.max_tokens}
        system = req.request.system or ""
        if req.response_schema is not None:
            config["responseMimeType"] = "application/json"
            if self.native_schema:
                config["responseJsonSchema"] = req.response_schema
            else:
                system = (system + "\n\nReply with only a JSON object that matches this JSON Schema:\n"
                          + json.dumps(req.response_schema)).strip()
        request: dict = {"contents": contents, "generationConfig": config}
        if system:
            request["systemInstruction"] = {"parts": [{"text": system}]}
        return {"key": req.custom_id, "request": request}

    def submit(self, batch: list[PreparedRequest]) -> str:
        models = {r.model for r in batch}
        if len(models) != 1:
            raise ValueError("a Gemini batch job runs one model; split the work by model")
        payload = "\n".join(json.dumps(self.to_line(r)) for r in batch).encode("utf-8")
        uploaded = self.client.files.upload(
            file=io.BytesIO(payload), config={"mime_type": "jsonl", "display_name": "steadybatch-input"}
        )
        job = self.client.batches.create(model=models.pop(), src=uploaded.name,
                                         config={"display_name": "steadybatch"})
        return job.name

    def status(self, batch_id: str) -> BatchStatus:
        job = self.client.batches.get(name=batch_id)
        state_name = getattr(job.state, "name", str(job.state))
        if state_name in _DONE:
            state = BatchState.DONE
        elif state_name in _FAILED:
            state = BatchState.FAILED
        elif state_name in _PENDING:
            state = BatchState.PENDING
        else:
            state = BatchState.RUNNING
        stats = job.completion_stats
        ok = int(getattr(stats, "successful_count", 0) or 0) if stats else 0
        bad = int(getattr(stats, "failed_count", 0) or 0) if stats else 0
        return BatchStatus(state, total=ok + bad, succeeded=ok, failed=bad, detail=state_name)

    def results(self, batch_id: str) -> Iterator[RawResult]:
        job = self.client.batches.get(name=batch_id)
        dest = job.dest
        if dest is None or not dest.file_name:
            return
        data = self.client.files.download(file=dest.file_name)
        for line in data.decode("utf-8").splitlines():
            if line.strip():
                yield self._parse(json.loads(line))

    @staticmethod
    def _parse(line: dict) -> RawResult:
        key = line.get("key", "")
        if line.get("error"):
            return RawResult(key, ok=False, error=json.dumps(line["error"])[:500])
        response = line.get("response") or {}
        usage = response.get("usageMetadata") or {}
        tokens = dict(input_tokens=usage.get("promptTokenCount", 0) or 0,
                      output_tokens=usage.get("candidatesTokenCount", 0) or 0)
        candidates = response.get("candidates") or []
        if not candidates:
            reason = (response.get("promptFeedback") or {}).get("blockReason", "no candidates")
            return RawResult(key, ok=False, error=f"blocked: {reason}", **tokens)
        cand = candidates[0]
        text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []))
        finish = cand.get("finishReason")
        if finish == "MAX_TOKENS":
            return RawResult(key, ok=True, text=text, error="truncated (max_tokens)", **tokens)
        if finish not in (None, "STOP"):
            return RawResult(key, ok=False, error=f"finish reason: {finish}", **tokens)
        return RawResult(key, ok=True, text=text, **tokens)
