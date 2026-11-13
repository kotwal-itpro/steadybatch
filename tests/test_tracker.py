import json
from pathlib import Path

import pytest

from steadybatch.models import BatchState, BatchStatus, RawResult
from steadybatch.providers.fake import FakeProvider, demo_answer
from steadybatch.tracker import HARD_MAX_SUBMITS_PER_DAY, Config, collect, submit

ROOT = Path(__file__).resolve().parents[1]
DAY0 = 1_794_000_000.0  # a fixed moment, so day boundaries are predictable


class Clock:
    def __init__(self, t: float = DAY0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def make_cfg(tmp_path: Path, n: int = 20, **extra) -> Config:
    records = [json.loads(l) for l in (ROOT / "tracker/inputs.jsonl").read_text().splitlines()[:n]]
    (tmp_path / "inputs.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    raw = {
        "inputs": "inputs.jsonl",
        "schema": str(ROOT / "examples/support_ticket.schema.json"),
        "instructions": str(ROOT / "examples/support_ticket.instructions.txt"),
        "prices": str(ROOT / "tracker/prices.json"),
        "open": "open.json", "runs": "data/runs.jsonl", "latest": "data/latest",
        "max_records": n, "max_submits_per_day": 2, "give_up_after_hours": 50,
        "targets": [{"provider": "openai", "model": "m1", "key_env": "K1"},
                    {"provider": "anthropic", "model": "m2", "key_env": "K2"}],
        **extra,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return Config.load(tmp_path / "config.json")


def rows(cfg: Config) -> list[dict]:
    return [json.loads(l) for l in cfg.path("runs").read_text().splitlines()]


ENV = {"K1": "x", "K2": "x"}


def test_submit_then_collect_writes_one_row_per_batch(tmp_path):
    cfg, clock = make_cfg(tmp_path), Clock()
    providers = {}

    def factory(t):
        return providers.setdefault(t["model"], FakeProvider(answer=demo_answer, clock=clock, turnaround=600))

    log = submit(cfg, make_provider=factory, clock=clock, env=ENV)
    assert len(json.loads(cfg.path("open").read_text())) == 2, log

    clock.t += 60
    collect(cfg, make_provider=factory, clock=clock)
    assert not cfg.path("runs").exists()  # still running
    assert len(json.loads(cfg.path("open").read_text())) == 2

    clock.t += 600
    collect(cfg, make_provider=factory, clock=clock)
    out = rows(cfg)
    assert [r["target"] for r in out] == ["openai-m1", "anthropic-m2"]
    r = out[0]
    assert r["state"] == "done" and r["ok"] == 20 and r["missing"] == 0
    assert r["turnaround_s"] == 600 and r["turnaround_source"] == "provider"
    assert r["model_versions"] == ["fake-1"]
    assert r["changed_vs_previous"] is None  # nothing to compare with yet
    assert 0 < r["field_accuracy"]["product"] <= 1
    assert r["cost_usd"] is not None
    assert json.loads(cfg.path("open").read_text()) == []


def test_second_run_measures_drift(tmp_path):
    cfg, clock = make_cfg(tmp_path), Clock()
    first = FakeProvider(answer=demo_answer, clock=clock)
    submit(cfg, make_provider=lambda t: first, clock=clock, env={"K1": "x"})
    collect(cfg, make_provider=lambda t: first, clock=clock)

    clock.t += 12 * 3600
    second = FakeProvider(answer=lambda req: demo_answer(req).replace('"neutral"', '"negative"'), clock=clock)
    submit(cfg, make_provider=lambda t: second, clock=clock, env={"K1": "x"})
    collect(cfg, make_provider=lambda t: second, clock=clock)

    a, b = rows(cfg)
    assert b["compared_with_previous"] == 20
    assert b["changed_vs_previous"] > 0
    assert a["changed_vs_previous"] is None


def test_missing_errors_and_invalid_lines_are_counted_not_retried(tmp_path):
    cfg, clock = make_cfg(tmp_path), Clock()
    p = FakeProvider(answer=demo_answer, clock=clock, drop_rate=0.2, error_rate=0.2, invalid_rate=0.2, seed=3)
    submit(cfg, make_provider=lambda t: p, clock=clock, env={"K1": "x"})
    collect(cfg, make_provider=lambda t: p, clock=clock)
    (r,) = rows(cfg)
    assert r["ok"] + r["missing"] + r["error"] + r["invalid"] == 20
    assert r["missing"] > 0 and r["error"] > 0 and r["invalid"] > 0
    assert len(p.submitted_batches) == 1  # measured, never resubmitted


def test_daily_cap_and_open_batch_block_extra_submits(tmp_path):
    cfg, clock = make_cfg(tmp_path), Clock()
    p = FakeProvider(answer=demo_answer, clock=clock, turnaround=10)
    f = lambda t: p
    submit(cfg, make_provider=f, clock=clock, env={"K1": "x"})
    log = submit(cfg, make_provider=f, clock=clock, env={"K1": "x"})
    assert "still open" in log[0]
    for _ in range(3):
        clock.t += 60
        collect(cfg, make_provider=f, clock=clock)
        submit(cfg, make_provider=f, clock=clock, env={"K1": "x"})
    collect(cfg, make_provider=f, clock=Clock(clock.t + 60))
    assert len(p.submitted_batches) == 2  # max_submits_per_day


def test_hard_cap_overrides_config(tmp_path):
    cfg = make_cfg(tmp_path, max_submits_per_day=50)
    assert cfg.max_submits_per_day == HARD_MAX_SUBMITS_PER_DAY


def test_missing_key_skips_target(tmp_path):
    cfg = make_cfg(tmp_path)
    log = submit(cfg, make_provider=lambda t: FakeProvider(), env={})
    assert all("not set" in line for line in log)
    assert json.loads(cfg.path("open").read_text()) == []


def test_refused_submit_is_recorded(tmp_path):
    cfg = make_cfg(tmp_path)

    class Refuses(FakeProvider):
        def submit(self, batch):
            raise RuntimeError("429 RESOURCE_EXHAUSTED: queue full\nmore detail")

    submit(cfg, make_provider=lambda t: Refuses(), env={"K1": "x"})
    (r,) = rows(cfg)
    assert r["state"] == "submit_failed" and "RESOURCE_EXHAUSTED" in r["error_message"] and "\n" not in r["error_message"]


def test_gives_up_after_the_limit(tmp_path):
    cfg, clock = make_cfg(tmp_path), Clock()
    p = FakeProvider(answer=demo_answer, clock=clock, turnaround=10**9)
    submit(cfg, make_provider=lambda t: p, clock=clock, env={"K1": "x"})
    clock.t += 51 * 3600
    collect(cfg, make_provider=lambda t: p, clock=clock)
    (r,) = rows(cfg)
    assert r["state"] == "gave_up" and r["missing"] == 20 and r["turnaround_s"] is None


def test_status_errors_keep_the_batch_open(tmp_path):
    cfg, clock = make_cfg(tmp_path), Clock()
    p = FakeProvider(answer=demo_answer, clock=clock)
    submit(cfg, make_provider=lambda t: p, clock=clock, env={"K1": "x"})

    class Flaky(FakeProvider):
        def status(self, batch_id):
            raise ConnectionError("reset")

    collect(cfg, make_provider=lambda t: Flaky(), clock=clock)
    assert len(json.loads(cfg.path("open").read_text())) == 1
    assert not cfg.path("runs").exists()


@pytest.mark.parametrize("value,expected", [(None, None), (0, None), (1700000000, 1700000000.0),
                                            ("2026-11-12T10:00:00Z", 1794477600.0)])
def test_to_epoch(value, expected):
    from steadybatch.providers.base import to_epoch
    assert to_epoch(value) == expected


def test_provider_timestamps_and_model_versions():
    from types import SimpleNamespace as NS
    from steadybatch.providers.openai_batch import OpenAIBatch
    b = NS(status="completed", request_counts=NS(total=1, completed=1, failed=0), errors=None,
           created_at=100, in_progress_at=130, completed_at=700, failed_at=None, expired_at=None, cancelled_at=None)
    client = NS(batches=NS(retrieve=lambda _id: b))
    s = OpenAIBatch(client=client).status("b1")
    assert (s.created_at, s.started_at, s.ended_at) == (100.0, 130.0, 700.0)
    line = {"custom_id": "c", "response": {"status_code": 200, "body": {
        "model": "gpt-4.1-mini-2025-04-14", "choices": [{"message": {"content": "{}"}}], "usage": {}}}}
    assert OpenAIBatch._parse(line).model_version == "gpt-4.1-mini-2025-04-14"

    from steadybatch.providers.gemini_batch import GeminiBatch
    r = GeminiBatch._parse({"key": "k", "response": {"modelVersion": "gemini-x-001",
                                                    "candidates": [{"content": {"parts": [{"text": "{}"}]}}]}})
    assert r.model_version == "gemini-x-001"


def test_only_fingerprints_of_answers_are_kept(tmp_path):
    cfg, clock = make_cfg(tmp_path), Clock()
    p = FakeProvider(answer=demo_answer, clock=clock)
    submit(cfg, make_provider=lambda t: p, clock=clock, env={"K1": "x"})
    collect(cfg, make_provider=lambda t: p, clock=clock)
    saved = json.loads((cfg.path("latest") / "openai-m1.json").read_text())
    assert len(saved) == 20 and all(isinstance(v, str) and len(v) == 16 for v in saved.values())
    assert "sentiment" not in json.dumps(saved)


def test_monthly_cost_cap_stops_submits(tmp_path):
    cfg, clock = make_cfg(tmp_path, max_monthly_cost_usd_per_target=0.0001), Clock()
    p = FakeProvider(answer=demo_answer, clock=clock)
    submit(cfg, make_provider=lambda t: p, clock=clock, env={"K1": "x"})
    collect(cfg, make_provider=lambda t: p, clock=clock)
    clock.t += 60
    log = submit(cfg, make_provider=lambda t: p, clock=clock, env={"K1": "x"})
    assert "spent this month" in log[0] and len(p.submitted_batches) == 1


def weekly_cfg(tmp_path: Path, **extra) -> Config:
    records = [json.loads(l) for l in (ROOT / "tracker/inputs-5k.jsonl").read_text().splitlines()[:600]]
    (tmp_path / "big.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return make_cfg(tmp_path, targets=[
        {"provider": "openai", "model": "m1", "key_env": "K1", "id": "small"},
        {"provider": "openai", "model": "m1", "key_env": "K1", "id": "big", "inputs": "big.jsonl",
         "records": 600, "every_days": 7, "slot_hours": [13]}], **extra)


def at_hour(h: int, day: int = 0) -> Clock:
    base = DAY0 - DAY0 % 86400  # midnight UTC
    return Clock(base + day * 86400 + h * 3600)


def test_weekly_target_goes_out_in_its_slot_once_a_week(tmp_path):
    cfg = weekly_cfg(tmp_path)
    p = FakeProvider(answer=demo_answer)
    f = lambda t: p
    submit(cfg, make_provider=f, clock=at_hour(7), env={"K1": "x"})
    assert [b["target"] for b in json.loads(cfg.path("open").read_text())] == ["small"]
    collect(cfg, make_provider=f, clock=at_hour(8))
    submit(cfg, make_provider=f, clock=at_hour(13), env={"K1": "x"})
    sizes = {b["target"]: b["records"] for b in json.loads(cfg.path("open").read_text())}
    assert sizes == {"small": 20, "big": 600}  # 600 > the 500 cap for 4x-daily targets
    collect(cfg, make_provider=f, clock=at_hour(14))
    big = [r for r in rows(cfg) if r["target"] == "big"]
    assert len(big) == 1 and big[0]["ok"] == 600 and big[0]["cadence"] == "weekly"
    for day in (1, 6):  # too soon
        log = submit(cfg, make_provider=f, clock=at_hour(13, day), env={"K1": "x"})
        collect(cfg, make_provider=f, clock=at_hour(14, day))
        assert any("every 7 days" in line for line in log)
    submit(cfg, make_provider=f, clock=at_hour(13, 7), env={"K1": "x"})
    assert "big" in [b["target"] for b in json.loads(cfg.path("open").read_text())]


def test_large_batch_cap_only_for_weekly_targets(tmp_path):
    from steadybatch.tracker import HARD_MAX_LARGE_RECORDS
    cfg = make_cfg(tmp_path)
    assert cfg.records_for({"records": 10_000}) == 500
    assert cfg.records_for({"records": 10_000, "every_days": 7}) == HARD_MAX_LARGE_RECORDS


def test_batches_opened_with_stored_keys_still_collect(tmp_path):
    cfg, clock = make_cfg(tmp_path), Clock()
    p = FakeProvider(answer=demo_answer, clock=clock)
    submit(cfg, make_provider=lambda t: p, clock=clock, env={"K1": "x"})
    (b,) = json.loads(cfg.path("open").read_text())
    _, prepared = cfg.prepared(cfg.targets[0])
    b["keys"] = {x.custom_id: x.request.key for x in prepared}
    del b["records"]
    cfg.path("open").write_text(json.dumps([b]))
    collect(cfg, make_provider=lambda t: p, clock=clock)
    (r,) = rows(cfg)
    assert r["ok"] == 20 and r["exact_accuracy"] is not None


def test_selfhost_row_counts_gpu_time(tmp_path):
    from steadybatch.tracker import selfhost
    cfg = make_cfg(tmp_path)
    ticks = iter([0.0, 120.0, 180.0])  # 2 min to load, 1 min to run
    row = selfhost(cfg, engine="sglang", model="m", gpu="1x test GPU", gpu_usd_per_hour=2.0, records=20,
                   out=tmp_path / "sh.jsonl", make_provider=lambda e, m: FakeProvider(answer=demo_answer),
                   clock=lambda: next(ticks))
    assert row["ok"] == 20 and row["load_s"] == 120 and row["run_s"] == 60
    assert row["cost_usd"] == round(180 / 3600 * 2.0, 5)
    assert row["cost_per_1000_usd_run_only"] == round(60 / 3600 * 2.0 / 20 * 1000, 4)
    assert json.loads((tmp_path / "sh.jsonl").read_text())["engine"] == "sglang"


def test_issues_skip_history_then_report_new_serious_ones(tmp_path):
    from steadybatch.tracker import new_issues
    cfg, clock = make_cfg(tmp_path), Clock()
    lossy = FakeProvider(answer=demo_answer, clock=clock, drop_rate=0.3, seed=1)
    submit(cfg, make_provider=lambda t: lossy, clock=clock, env={"K1": "x"})
    collect(cfg, make_provider=lambda t: lossy, clock=clock)
    state = tmp_path / "reported.json"
    assert new_issues(cfg, state) == []  # first call: history is marked as seen
    clock.t += 6 * 3600
    submit(cfg, make_provider=lambda t: lossy, clock=clock, env={"K1": "x"})
    collect(cfg, make_provider=lambda t: lossy, clock=clock)
    (issue,) = new_issues(cfg, state)
    assert "lines lost" in issue["title"] and "Batch ID" in issue["body"]
    assert new_issues(cfg, state) == []  # not filed twice


def test_site_writes_status_badges_and_feed(tmp_path):
    import xml.dom.minidom
    from datetime import datetime, timezone
    from steadybatch.tracker_site import build
    cfg, clock = make_cfg(tmp_path), Clock()
    p = FakeProvider(answer=demo_answer, clock=clock, turnaround=300, drop_rate=0.3, seed=1)
    submit(cfg, make_provider=lambda t: p, clock=clock, env=ENV)
    clock.t += 400
    collect(cfg, make_provider=lambda t: p, clock=clock)
    sh = tmp_path / "sh.jsonl"
    sh.write_text(json.dumps({"engine": "sglang", "model": "m", "gpu": "g", "gpu_usd_per_hour": 1,
                              "records": 20, "turnaround_s": 90, "load_s": 60}) + "\n")
    out = tmp_path / "site"
    build(cfg.path("runs"), out, open_path=cfg.path("open"), selfhost_path=sh,
          now=datetime.fromtimestamp(clock.t + 60, timezone.utc))
    st = json.loads((out / "status.json").read_text())
    t = {x["target"]: x for x in st["targets"]}["openai-m1"]
    assert t["p50_turnaround_s"] == 300 and t["runs"] == 1 and t["lines_lost"] > 0
    b = json.loads((out / "badges/openai-m1.json").read_text())
    assert b["schemaVersion"] == 1 and b["message"] == "5 min"
    doc = xml.dom.minidom.parse(str(out / "feed.xml"))
    assert len(doc.getElementsByTagName("entry")) == 2
    html = (out / "index.html").read_text()
    assert '"selfhost":[{' in html and (out / "selfhost.jsonl").exists()
