"""Self-hosted baseline: vLLM offline inference on your own GPUs.

There is no queue here: submit() runs the whole batch right away and keeps
the results in memory. That is enough for a cost and quality comparison
against the hosted batch APIs.

When a JSON Schema is given, generation is constrained to it with vLLM's
structured outputs, which is the closest match to the hosted services'
schema modes. The schema is also described in the system prompt so the
model knows what the fields mean.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Iterator

from ..models import BatchState, BatchStatus, PreparedRequest, RawResult
from .base import BatchProvider


def schema_params(schema: dict[str, Any] | None) -> dict[str, Any]:
    """SamplingParams keyword arguments that constrain output to `schema`.

    vLLM >= 0.10 uses `structured_outputs=StructuredOutputsParams(json=...)`;
    older releases used `guided_decoding=GuidedDecodingParams(json=...)`.
    """
    if schema is None:
        return {}
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return {"structured_outputs": StructuredOutputsParams(json=schema)}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return {"guided_decoding": GuidedDecodingParams(json=schema)}


def build_messages(req: PreparedRequest) -> list[dict[str, str]]:
    system = req.request.system or ""
    if req.response_schema is not None:
        system = (system + "\n\nReply with only a JSON object that matches this JSON Schema:\n"
                  + json.dumps(req.response_schema)).strip()
    msgs = list(req.request.messages)
    return ([{"role": "system", "content": system}] if system else []) + msgs


class VLLMOffline(BatchProvider):
    name = "vllm"
    max_requests_per_batch = 10_000
    max_bytes_per_batch = 1024 * 1024 * 1024

    def __init__(self, model: str, llm=None, **llm_kwargs):
        if llm is None:
            from vllm import LLM  # imported here so the package works without it
            llm = LLM(model=model, **llm_kwargs)
        self.llm = llm
        self._results: dict[str, list[RawResult]] = {}
        self._ids = itertools.count(1)

    def sampling_params(self, req: PreparedRequest):
        from vllm import SamplingParams
        return SamplingParams(temperature=req.request.temperature, max_tokens=req.request.max_tokens,
                              **schema_params(req.response_schema))

    def submit(self, batch: list[PreparedRequest]) -> str:
        conversations = [build_messages(r) for r in batch]
        outputs = self.llm.chat(conversations, self.sampling_params(batch[0]), use_tqdm=False)
        batch_id = f"vllm-{next(self._ids)}"
        results = []
        for r, o in zip(batch, outputs):
            out = o.outputs[0]
            truncated = getattr(out, "finish_reason", None) == "length"
            results.append(RawResult(r.custom_id, ok=True, text=out.text,
                                     error="truncated (max_tokens)" if truncated else None,
                                     input_tokens=len(o.prompt_token_ids or []),
                                     output_tokens=len(out.token_ids or [])))
        self._results[batch_id] = results
        return batch_id

    def status(self, batch_id: str) -> BatchStatus:
        n = len(self._results[batch_id])
        return BatchStatus(BatchState.DONE, total=n, succeeded=n)

    def results(self, batch_id: str) -> Iterator[RawResult]:
        yield from self._results[batch_id]
