import json

from steadybatch import Outcome, Request, Runner
from steadybatch.providers.fake import FakeProvider

SCHEMA = {
    "type": "object",
    "properties": {"key": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["key", "summary"],
}


def requests(n):
    return [Request(key=f"rec-{i:04d}", messages=[{"role": "user", "content": f"record number {i}"}])
            for i in range(n)]


def runner(provider, **kw):
    return Runner(provider, "test-model", response_schema=SCHEMA, sleep=lambda s: None, **kw)


def test_happy_path_returns_every_record_in_order():
    report = runner(FakeProvider(max_requests_per_batch=7)).run(requests(20))
    assert report.ok == 20
    assert [r.key for r in report.results] == [f"rec-{i:04d}" for i in range(20)]
    assert report.batches_submitted == 3


def test_silently_dropped_lines_are_detected_and_retried():
    provider = FakeProvider(drop_rate=0.3, fail_first_attempt_only=True)
    report = runner(provider).run(requests(50))
    assert report.lines_missing > 0, "the fake should have dropped some lines"
    assert report.ok == 50, "every dropped line should succeed on retry"
    retried = [r for r in report.results if r.attempts == 2]
    assert len(retried) == report.lines_missing
    assert all(r.history[0].startswith("missing") for r in retried)


def test_errors_and_bad_json_are_retried_line_by_line():
    provider = FakeProvider(error_rate=0.2, invalid_rate=0.2, fail_first_attempt_only=True)
    report = runner(provider).run(requests(50))
    assert report.lines_errored > 0 and report.lines_invalid > 0
    assert report.ok == 50
    # Only the bad lines went back, never the whole batch.
    second_round = provider.submitted_batches[1]
    assert len(second_round) == report.lines_errored + report.lines_invalid


def test_lines_that_keep_failing_stop_after_max_attempts():
    provider = FakeProvider(error_rate=1.0)
    report = runner(provider, max_attempts=3).run(requests(5))
    assert report.ok == 0
    assert all(r.outcome is Outcome.ERROR and r.attempts == 3 for r in report.results)


def test_running_again_does_not_redo_finished_work(tmp_path):
    ckpt = tmp_path / "job.sqlite"
    first = FakeProvider()
    runner(first, checkpoint=ckpt).run(requests(10))
    second = FakeProvider()
    report = runner(second, checkpoint=ckpt).run(requests(10))
    assert second.submitted_batches == [], "nothing should be resubmitted"
    assert report.ok == 10


def test_restart_after_crash_collects_open_batch_instead_of_paying_twice(tmp_path):
    ckpt = tmp_path / "job.sqlite"
    provider = FakeProvider()
    r1 = runner(provider, checkpoint=ckpt)
    prepared = r1.prepare(requests(5))
    r1.store.add(prepared)
    batch_id = provider.submit(prepared)          # submitted...
    r1.store.mark_submitted(batch_id, "fake", [p.custom_id for p in prepared])
    r1.store.close()                              # ...then the process "crashes"

    report = runner(provider, checkpoint=ckpt).run(requests(5))
    assert len(provider.submitted_batches) == 1, "the open batch is collected, not resubmitted"
    assert report.ok == 5


def test_duplicate_lines_from_the_provider_count_once():
    class Doubling(FakeProvider):
        def results(self, batch_id):
            for r in super().results(batch_id):
                yield r
                yield r

    report = runner(Doubling()).run(requests(5))
    assert report.ok == 5
    assert all(r.input_tokens == 100 for r in report.results)


def test_truncated_output_keeps_the_reason_in_history():
    from steadybatch.models import RawResult

    class Truncating(FakeProvider):
        def results(self, batch_id):
            for raw in super().results(batch_id):
                yield RawResult(raw.custom_id, ok=True, text='{"key": "re', error="truncated (max_tokens)")

    report = runner(Truncating(), max_attempts=1).run(requests(2))
    assert all(r.outcome is Outcome.INVALID for r in report.results)
    assert all(r.history[0].startswith("invalid: truncated (max_tokens): not valid JSON") for r in report.results)


def test_batch_size_and_max_open_batches_limit_queued_work():
    provider = FakeProvider()
    open_counts = []
    real_submit = provider.submit

    def submit(batch):
        open_counts.append(len(r.store.open_batches()))
        return real_submit(batch)

    provider.submit = submit
    r = runner(provider, batch_size=4, max_open_batches=1)
    report = r.run(requests(10))
    assert report.ok == 10 and report.batches_submitted == 3
    assert [len(b) for b in provider.submitted_batches] == [4, 4, 2]
    assert open_counts == [0, 0, 0], "each batch should be collected before the next is sent"
