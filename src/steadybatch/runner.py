"""The runner: submit, wait, check every line, retry what failed, and keep going.

The rules it follows:

* Every request has a stable ID, so a retry is the same request, never a copy.
* A batch that says "done" is not trusted. Every ID we sent is checked off;
  anything that did not come back is marked missing and retried.
* Every answer is checked against the schema before it counts as a success.
* One bad line never fails the batch. Only that line is retried.
* State is saved after every step, so a restart picks up where it stopped.
* A provider that refuses a batch because the queue is full gets smaller
  batches, without using up any request's retries.
* A dropped connection or server error while submitting is retried with a
  growing pause, instead of stopping a job that may have run for hours.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .chunking import chunk
from .ids import custom_id_for
from .models import BatchState, BatchStatus, Outcome, PreparedRequest, Request, Result
from .providers.base import BatchProvider
from .store import Store
from .validate import InvalidOutput, check

log = logging.getLogger("steadybatch")


@dataclass
class RunReport:
    results: list[Result]
    batches_submitted: int = 0
    lines_missing: int = 0
    lines_errored: int = 0
    lines_invalid: int = 0
    capacity_refusals: int = 0
    submit_retries: int = 0
    seconds: float = 0.0
    batch_seconds: list[float] = field(default_factory=list)

    @property
    def ok(self) -> int:
        return sum(r.outcome is Outcome.OK for r in self.results)

    @property
    def failed(self) -> int:
        return len(self.results) - self.ok


class Runner:
    def __init__(
        self,
        provider: BatchProvider,
        model: str,
        *,
        response_schema: dict[str, Any] | None = None,
        checkpoint: str | Path = ":memory:",
        max_attempts: int = 3,
        poll_every: float = 30.0,
        timeout: float = 26 * 3600,
        batch_size: int | None = None,
        max_open_batches: int | None = None,
        max_submit_retries: int = 8,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,  # wall clock: monotonic stops while a laptop sleeps
    ):
        self.provider = provider
        self.model = model
        self.schema = response_schema
        self.store = Store(checkpoint)
        # Providers cap how much work an account can have queued (OpenAI counts queued
        # input tokens). Smaller batches, fewer at a time, keep a big job under that cap.
        self.batch_size = batch_size
        self.max_open_batches = max_open_batches
        self.max_submit_retries = max_submit_retries
        self._size_limit: int | None = None  # lowered when the provider says the queue is full
        self.max_attempts = max_attempts
        self.poll_every = poll_every
        self.timeout = timeout
        self.sleep = sleep
        self.clock = clock

    def prepare(self, requests: Iterable[Request]) -> list[PreparedRequest]:
        return [
            PreparedRequest(custom_id_for(r, self.model, self.schema), r, self.model, self.schema)
            for r in requests
        ]

    def run(self, requests: Iterable[Request]) -> RunReport:
        started = self.clock()
        added = self.store.add(self.prepare(requests))
        log.info("registered %d new requests", added)
        report = RunReport(results=[])

        # Batches left open by an earlier run are collected first, not resubmitted.
        self._collect_open(report, started)

        while True:
            todo = self.store.ready_to_submit(self.max_attempts)
            if not todo:
                break
            failures = 0
            while todo:
                batch = next(chunk(todo, self._max_count(), self.provider.max_bytes_per_batch,
                                   self.provider.size_of))
                if self.max_open_batches and len(self.store.open_batches()) >= self.max_open_batches:
                    self._collect_open(report, started)
                try:
                    batch_id = self.provider.submit(batch)
                except Exception as exc:
                    if self.provider.is_over_capacity(exc):
                        report.capacity_refusals += 1
                        self._on_capacity_refusal(len(batch), report, started, reason=str(exc)[:200])
                        continue
                    if self.provider.is_transient(exc) and failures < self.max_submit_retries:
                        failures += 1
                        report.submit_retries += 1
                        pause = min(30.0 * 2 ** (failures - 1), 600.0)
                        log.warning("submit failed (%s: %s); retry %d of %d in %.0f s", type(exc).__name__,
                                    str(exc)[:200], failures, self.max_submit_retries, pause)
                        self.sleep(pause)
                        continue
                    raise
                failures = 0
                self.store.mark_submitted(batch_id, self.provider.name, [r.custom_id for r in batch])
                report.batches_submitted += 1
                log.info("submitted %s with %d requests", batch_id, len(batch))
                todo = todo[len(batch):]
            self._collect_open(report, started)

        self.store.give_up_on_exhausted(self.max_attempts)
        report.results = list(self.store.results())
        report.seconds = self.clock() - started
        for _, submitted, finished in self.store.batch_timings():
            if finished is not None:
                report.batch_seconds.append(finished - submitted)
        return report

    # ------------------------------------------------------------------

    def _max_count(self) -> int:
        limit = min(self.provider.max_requests_per_batch, self.batch_size or self.provider.max_requests_per_batch)
        return min(limit, self._size_limit) if self._size_limit else limit

    def _on_capacity_refusal(self, size: int, report: RunReport, started: float, reason: str) -> None:
        """The queue is full. If our own batches are in it, wait for them; otherwise
        halve the batch size. Give up only when a single request is refused."""
        if self.store.open_batches():
            log.warning("queue full (%s); waiting for open batches before sending more", reason)
            self._collect_open(report, started)
            return
        if size <= 1:
            raise RuntimeError(f"provider refused even a one-request batch as over capacity: {reason}")
        self._size_limit = max(1, size // 2)
        log.warning("queue full (%s); retrying with batches of at most %d", reason, self._size_limit)
        self.sleep(min(self.poll_every, 60.0))

    def _collect_open(self, report: RunReport, started: float) -> None:
        for batch_id, sent_ids in self.store.open_batches():
            status = self._wait(batch_id, started)
            if status.state is BatchState.FAILED and status.over_capacity:
                # Refused before any work was done: not the requests' fault.
                report.capacity_refusals += 1
                self.store.release_batch(batch_id)
                if not self.store.open_batches():
                    self._size_limit = max(1, len(sent_ids) // 2)
                    log.warning("batch %s refused as over capacity; retrying with batches of at most %d",
                                batch_id, self._size_limit)
                continue
            seen: set[str] = set()
            if status.state is not BatchState.FAILED:
                for raw in self.provider.results(batch_id):
                    if raw.custom_id in seen:
                        continue  # a duplicate line: count it once
                    seen.add(raw.custom_id)
                    self._record(raw, report)
            for cid in sent_ids:
                if cid not in seen:
                    report.lines_missing += 1
                    self.store.record(cid, Outcome.MISSING, retryable=True,
                                      error=f"no result returned by batch {batch_id}")
            self.store.close_batch(batch_id)

    def _wait(self, batch_id: str, started: float) -> BatchStatus:
        while True:
            status = self.provider.status(batch_id)
            if status.state in (BatchState.DONE, BatchState.FAILED):
                if status.state is BatchState.FAILED:
                    log.warning("batch %s failed: %s", batch_id, status.detail)
                return status
            if self.clock() - started > self.timeout:
                raise TimeoutError(f"batch {batch_id} still {status.state.value} after timeout")
            self.sleep(self.poll_every)

    def _record(self, raw, report: RunReport) -> None:
        tokens = dict(input_tokens=raw.input_tokens, output_tokens=raw.output_tokens)
        if not raw.ok:
            report.lines_errored += 1
            self.store.record(raw.custom_id, Outcome.ERROR, retryable=True, error=raw.error, **tokens)
            return
        try:
            data = check(raw.text, self.schema)
        except InvalidOutput as exc:
            report.lines_invalid += 1
            # Keep the provider's note (e.g. "truncated (max_tokens)"): it says why the text is bad.
            error = f"{raw.error}: {exc}" if raw.error else str(exc)
            self.store.record(raw.custom_id, Outcome.INVALID, retryable=True,
                              text=raw.text, error=error, **tokens)
            return
        self.store.record(raw.custom_id, Outcome.OK, retryable=False, data=data, text=raw.text, **tokens)
