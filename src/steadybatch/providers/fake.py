"""A provider that runs in memory and misbehaves on purpose.

It lets you test the runner against the failures real batch APIs produce,
without an API key or a bill: lines that silently go missing, per-line
errors, output that is not valid JSON, and answers that change between runs.
"""

from __future__ import annotations

import itertools
import json
import random
from typing import Callable, Iterator

from ..models import BatchState, BatchStatus, PreparedRequest, RawResult
from .base import BatchProvider


def default_answer(req: PreparedRequest) -> str:
    return json.dumps({"key": req.request.key, "summary": req.request.messages[-1]["content"][:40]})


_PRODUCTS = ["smart thermostat", "wireless earbuds", "router", "laptop", "phone", "printer"]
_ISSUE_WORDS = {"billing": ["charged", "invoice"], "shipping": ["arrived", "delivered"],
                "defect": ["stopped working", "restarting"], "how_to": ["how do i", "how to"]}
_MOOD_WORDS = {"negative": ["frustrating", "unhappy"], "positive": ["love", "helpful"]}


def demo_answer(req: PreparedRequest) -> str:
    """A keyword 'model' for the synthetic support tickets in examples/.

    It is right most of the time and wrong some of the time, like a real model,
    so the benchmark's accuracy numbers have something to measure.
    """
    text = req.request.messages[-1]["content"].lower()
    product = next((p for p in _PRODUCTS if p in text), "phone")
    issue = next((i for i, words in _ISSUE_WORDS.items() if any(w in text for w in words)), "how_to")
    mood = next((m for m, words in _MOOD_WORDS.items() if any(w in text for w in words)), "neutral")
    follow = "call me back" in text  # misses the "email me" phrasing on purpose
    return json.dumps({"product": product, "issue_type": issue, "sentiment": mood, "needs_follow_up": follow})


class FakeProvider(BatchProvider):
    name = "fake"

    def __init__(
        self,
        *,
        drop_rate: float = 0.0,
        error_rate: float = 0.0,
        invalid_rate: float = 0.0,
        nondeterministic: bool = False,
        max_requests_per_batch: int = 1000,
        max_bytes_per_batch: int = 10 * 1024 * 1024,
        answer: Callable[[PreparedRequest], str] = default_answer,
        seed: int = 7,
        fail_first_attempt_only: bool = False,
    ):
        self.drop_rate = drop_rate
        self.error_rate = error_rate
        self.invalid_rate = invalid_rate
        self.nondeterministic = nondeterministic
        self.max_requests_per_batch = max_requests_per_batch
        self.max_bytes_per_batch = max_bytes_per_batch
        self.answer = answer
        self.rng = random.Random(seed)
        self.fail_first_attempt_only = fail_first_attempt_only
        self._seen: set[str] = set()
        self._batches: dict[str, list[RawResult]] = {}
        self._ids = itertools.count(1)
        self.submitted_batches: list[list[str]] = []

    def submit(self, batch: list[PreparedRequest]) -> str:
        batch_id = f"fake-batch-{next(self._ids)}"
        self.submitted_batches.append([r.custom_id for r in batch])
        out: list[RawResult] = []
        for req in batch:
            misbehave = not (self.fail_first_attempt_only and req.custom_id in self._seen)
            self._seen.add(req.custom_id)
            roll = self.rng.random()
            if misbehave and roll < self.drop_rate:
                continue  # silently lost: no result, no error
            roll -= self.drop_rate
            if misbehave and roll < self.error_rate:
                out.append(RawResult(req.custom_id, ok=False, error="simulated provider error"))
                continue
            roll -= self.error_rate
            if misbehave and roll < self.invalid_rate:
                text = "Sure! Here is the JSON you asked for: {not json"
            else:
                text = self.answer(req)
                if self.nondeterministic and self.rng.random() < 0.5:
                    text = text.replace("}", ', "note": "varies"}', 1)
            out.append(RawResult(req.custom_id, ok=True, text=text, input_tokens=100, output_tokens=20))
        self._batches[batch_id] = out
        return batch_id

    def status(self, batch_id: str) -> BatchStatus:
        lines = self._batches[batch_id]
        return BatchStatus(BatchState.DONE, total=len(lines),
                           succeeded=sum(r.ok for r in lines), failed=sum(not r.ok for r in lines))

    def results(self, batch_id: str) -> Iterator[RawResult]:
        yield from self._batches[batch_id]
