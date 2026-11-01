"""Self-hosted baseline: SGLang's offline engine on your own GPUs.

Like the vLLM adapter, there is no queue: submit() runs the whole batch right
away in-process and keeps the results in memory.

SGLang's Engine.generate() takes plain prompts, so the chat messages are
rendered with the model's own chat template first. When a JSON Schema is
given, generation is constrained to it with SGLang's `json_schema` sampling
parameter, and the schema is also described in the system prompt so the
model knows what the fields mean.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Iterator

from ..models import BatchState, BatchStatus, PreparedRequest, RawResult
from .base import BatchProvider
from .vllm_offline import build_messages


def sampling_params(req: PreparedRequest) -> dict[str, Any]:
    """The SGLang sampling_params dict for one request."""
    params: dict[str, Any] = {"temperature": req.request.temperature,
                              "max_new_tokens": req.request.max_tokens}
    if req.response_schema is not None:
        params["json_schema"] = json.dumps(req.response_schema)
    return params


def finish_type(meta_info: dict[str, Any]) -> str | None:
    reason = meta_info.get("finish_reason")
    return reason.get("type") if isinstance(reason, dict) else reason


class SGLangOffline(BatchProvider):
    name = "sglang"
    max_requests_per_batch = 10_000
    max_bytes_per_batch = 1024 * 1024 * 1024

    def __init__(self, model: str, engine=None, tokenizer=None, **engine_kwargs):
        if engine is None:
            import sglang  # imported here so the package works without it
            engine = sglang.Engine(model_path=model, **engine_kwargs)
        self.engine = engine
        self.tokenizer = tokenizer if tokenizer is not None else engine.tokenizer_manager.tokenizer
        self._results: dict[str, list[RawResult]] = {}
        self._ids = itertools.count(1)

    def render(self, req: PreparedRequest) -> str:
        return self.tokenizer.apply_chat_template(build_messages(req), tokenize=False,
                                                  add_generation_prompt=True)

    def submit(self, batch: list[PreparedRequest]) -> str:
        prompts = [self.render(r) for r in batch]
        outputs = self.engine.generate(prompt=prompts, sampling_params=[sampling_params(r) for r in batch])
        batch_id = f"sglang-{next(self._ids)}"
        results = []
        for r, o in zip(batch, outputs):
            meta = o.get("meta_info", {})
            truncated = finish_type(meta) == "length"
            results.append(RawResult(r.custom_id, ok=True, text=o.get("text", ""),
                                     error="truncated (max_tokens)" if truncated else None,
                                     input_tokens=meta.get("prompt_tokens", 0),
                                     output_tokens=meta.get("completion_tokens", 0)))
        self._results[batch_id] = results
        return batch_id

    def status(self, batch_id: str) -> BatchStatus:
        n = len(self._results[batch_id])
        return BatchStatus(BatchState.DONE, total=n, succeeded=n)

    def results(self, batch_id: str) -> Iterator[RawResult]:
        yield from self._results[batch_id]
