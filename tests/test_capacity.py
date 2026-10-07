"""A provider whose queue is full refuses work. The runner should send smaller
batches (or wait for its own) without spending any request's retries."""

import pytest

from steadybatch.models import BatchState, BatchStatus
from steadybatch.providers.fake import FakeProvider

from test_runner import requests, runner


class QueueFull(Exception):
    pass


class LimitedQueue(FakeProvider):
    """Holds at most `capacity` requests across batches that are still open.

    mode="submit": an oversized batch is refused at submit (Gemini, Anthropic).
    mode="async":  it is accepted, then fails as over capacity (OpenAI).
    """

    def __init__(self, capacity, mode="submit", **kw):
        super().__init__(**kw)
        self.capacity, self.mode = capacity, mode
        self.queued: dict[str, int] = {}
        self.refused: set[str] = set()

    def is_over_capacity(self, exc):
        return isinstance(exc, QueueFull)

    def submit(self, batch):
        full = sum(self.queued.values()) + len(batch) > self.capacity
        if full and self.mode == "submit":
            raise QueueFull("429 RESOURCE_EXHAUSTED")
        batch_id = super().submit(batch)
        if full:
            self.refused.add(batch_id)
        else:
            self.queued[batch_id] = len(batch)
        return batch_id

    def status(self, batch_id):
        if batch_id in self.refused:
            return BatchStatus(BatchState.FAILED, detail="failed", over_capacity=True)
        return super().status(batch_id)

    def results(self, batch_id):
        self.queued.pop(batch_id, None)
        yield from super().results(batch_id)


@pytest.mark.parametrize("mode", ["submit", "async"])
def test_oversized_batch_is_split_without_spending_retries(mode):
    provider = LimitedQueue(capacity=6, mode=mode)
    report = runner(provider, max_attempts=1).run(requests(20))
    assert report.ok == 20, "every record should get through even with max_attempts=1"
    assert report.capacity_refusals >= 1
    assert all(r.attempts == 1 for r in report.results)
    accepted = [b for i, b in enumerate(provider.submitted_batches)
                if f"fake-batch-{i + 1}" not in provider.refused]
    assert max(len(b) for b in accepted) <= 6


def test_waits_for_its_own_open_batches_before_shrinking():
    provider = LimitedQueue(capacity=5)
    report = runner(provider, batch_size=5, max_attempts=1).run(requests(10))
    assert report.ok == 10
    # The second batch was refused while the first was queued; after collecting
    # the first, the same size fits, so the batch size never had to shrink.
    assert [len(b) for b in provider.submitted_batches] == [5, 5]
    assert report.capacity_refusals == 1


def test_other_errors_still_stop_the_run():
    class Broken(FakeProvider):
        def submit(self, batch):
            raise ValueError("bad request")

    with pytest.raises(ValueError):
        runner(Broken()).run(requests(3))


def test_gives_up_when_even_one_request_is_refused():
    with pytest.raises(RuntimeError, match="one-request batch"):
        runner(LimitedQueue(capacity=0)).run(requests(3))
