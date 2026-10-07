"""Self-hosted baseline: vLLM offline inference on your own GPUs.

There is no queue here: submit() runs the whole batch right away and keeps
the results in memory. That is enough for a cost and quality comparison
against the hosted batch APIs.
"""

from __future__ import annotations

import itertools
from typing import Iterator

from ..models import BatchState, BatchStatus, PreparedRequest, RawResult
from .base import BatchProvider


class VLLMOffline(BatchProvider):
    name = "vllm"
    max_requests_per_batch = 10_000
    max_bytes_per_batch = 1024 * 1024 * 1024

    def __init__(self, model: str, **llm_kwargs):
        from vllm import LLM  # imported here so the package works without it
        self.llm = LLM(model=model, **llm_kwargs)
        self._results: dict[str, list[RawResult]] = {}
        self._ids = itertools.count(1)

    def submit(self, batch: list[PreparedRequest]) -> str:
        from vllm import SamplingParams

        conversations = []
        for r in batch:
            msgs = list(r.request.messages)
            if r.request.system:
                msgs = [{"role": "system", "content": r.request.system}] + msgs
            conversations.append(msgs)
        first = batch[0].request
        params = SamplingParams(temperature=first.temperature, max_tokens=first.max_tokens)
        outputs = self.llm.chat(conversations, params)
        batch_id = f"vllm-{next(self._ids)}"
        self._results[batch_id] = [
            RawResult(r.custom_id, ok=True, text=o.outputs[0].text,
                      input_tokens=len(o.prompt_token_ids or []),
                      output_tokens=len(o.outputs[0].token_ids or []))
            for r, o in zip(batch, outputs)
        ]
        return batch_id

    def status(self, batch_id: str) -> BatchStatus:
        n = len(self._results[batch_id])
        return BatchStatus(BatchState.DONE, total=n, succeeded=n)

    def results(self, batch_id: str) -> Iterator[RawResult]:
        yield from self._results[batch_id]
