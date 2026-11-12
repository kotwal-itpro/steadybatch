"""Batch API tracker: measure the batch APIs themselves, every day, in public.

    steadybatch-track submit    # send one small batch per provider, note the batch IDs
    steadybatch-track collect   # pick up finished batches and write one row each
    steadybatch-track site      # build the static page from the rows

``submit`` and ``collect`` are separate commands because a batch can take up
to 24 hours (48 on Gemini), far longer than one CI job may run. Between runs
the only state is ``open.json`` (batches in flight) and the data folder, both
small files that the scheduled workflow commits back to the repo.

Every run sends the same fixed records, so day-to-day answer changes are a
clean signal. Nothing is ever resubmitted: the point is to measure what the
provider did, not to repair it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .bench import load_records, to_requests
from .ids import custom_id_for
from .models import BatchState, PreparedRequest, Request
from .providers import BatchProvider, get_provider
from .validate import InvalidOutput, check

ProviderFactory = Callable[[dict[str, Any]], BatchProvider]

# Hard limits in code, whatever the config says, so a bad edit can't run up a bill.
HARD_MAX_RECORDS = 500
HARD_MAX_SUBMITS_PER_DAY = 4


@dataclass
class Config:
    root: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        return cls(root=path.parent, raw=json.loads(path.read_text()))

    def path(self, key: str) -> Path:
        p = Path(self.raw[key])
        return p if p.is_absolute() else (self.root / p)

    @property
    def max_records(self) -> int:
        return min(int(self.raw.get("max_records", 200)), HARD_MAX_RECORDS)

    @property
    def max_submits_per_day(self) -> int:
        return min(int(self.raw.get("max_submits_per_day", 2)), HARD_MAX_SUBMITS_PER_DAY)

    @property
    def give_up_after_s(self) -> float:
        return float(self.raw.get("give_up_after_hours", 50)) * 3600

    @property
    def targets(self) -> list[dict[str, Any]]:
        return self.raw["targets"]

    def schema(self) -> dict[str, Any] | None:
        return json.loads(self.path("schema").read_text()) if self.raw.get("schema") else None

    def requests(self) -> tuple[list[dict[str, Any]], list[Request]]:
        records = load_records(self.path("inputs"), self.max_records)
        instructions = self.path("instructions").read_text() if self.raw.get("instructions") else ""
        return records, to_requests(records, int(self.raw.get("max_tokens", 300)), instructions)


def target_id(t: dict[str, Any]) -> str:
    return t.get("id") or f"{t['provider']}-{t['model']}"


def default_factory(t: dict[str, Any]) -> BatchProvider:
    return get_provider(t["provider"], t["model"], thinking=t.get("thinking"))


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds") if ts else None


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _short_error(exc: BaseException) -> str:
    # Provider errors can be long; keep the first line. They never contain the key.
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"


def submit(cfg: Config, *, make_provider: ProviderFactory = default_factory,
           clock: Callable[[], float] = time.time, env: dict[str, str] | None = None) -> list[str]:
    """Send one batch per target, unless a cap or an open batch says not to."""
    env = os.environ if env is None else env
    open_path, runs_path = cfg.path("open"), cfg.path("runs")
    open_batches: list[dict[str, Any]] = _read_json(open_path, [])
    rows = _rows(runs_path)
    now = clock()
    today = _day(now)
    _records, requests = cfg.requests()
    log: list[str] = []
    for t in cfg.targets:
        tid = target_id(t)
        if t.get("key_env") and not env.get(t["key_env"]):
            log.append(f"{tid}: skipped, {t['key_env']} is not set")
            continue
        if any(b["target"] == tid for b in open_batches):
            log.append(f"{tid}: skipped, the previous batch is still open")
            continue
        sent_today = sum(r["target"] == tid and _day(r["submitted_ts"]) == today for r in rows)
        sent_today += sum(b["target"] == tid and _day(b["submitted_ts"]) == today for b in open_batches)
        if sent_today >= cfg.max_submits_per_day:
            log.append(f"{tid}: skipped, already {sent_today} today")
            continue
        provider = make_provider(t)
        schema = cfg.schema()
        prepared = [PreparedRequest(custom_id_for(r, t["model"], schema), r, t["model"], schema)
                    for r in requests]
        try:
            batch_id = provider.submit(prepared)
        except Exception as exc:  # a refused submit is itself a measurement
            _append_row(runs_path, {"target": tid, "label": t.get("label"),
                                    "provider": t["provider"], "model": t["model"],
                                    "state": "submit_failed", "error_message": _short_error(exc),
                                    "submitted_ts": now, "submitted_at": _iso(now),
                                    "submit_hour_utc": datetime.fromtimestamp(now, timezone.utc).hour,
                                    "records": len(prepared), "steadybatch_version": __version__})
            log.append(f"{tid}: submit failed: {_short_error(exc)}")
            continue
        open_batches.append({"target": tid, "label": t.get("label"),
                             "provider": t["provider"], "model": t["model"],
                             "batch_id": batch_id, "submitted_ts": now,
                             "keys": {p.custom_id: p.request.key for p in prepared}})
        log.append(f"{tid}: submitted {len(prepared)} records as {batch_id}")
    _write_json(open_path, open_batches)
    return log


def _accuracy(records: list[dict[str, Any]], answers: dict[str, Any]) -> tuple[dict[str, float], float | None]:
    fields: dict[str, list[int]] = {}
    exact: list[int] = []
    for r in records:
        labels, data = r.get("labels"), answers.get(str(r["key"]))
        if not labels or not isinstance(data, dict):
            continue
        hits = [int(data.get(f) == v) for f, v in labels.items()]
        for f, h in zip(labels, hits):
            fields.setdefault(f, []).append(h)
        exact.append(int(all(hits)))
    acc = {f: round(sum(v) / len(v), 4) for f, v in fields.items()}
    return acc, (round(sum(exact) / len(exact), 4) if exact else None)


def _cost(prices: dict[str, Any], provider: str, in_tok: int, out_tok: int) -> float | None:
    p = prices.get(provider)
    if not p:
        return None
    return (in_tok * p["input_per_mtok"] + out_tok * p["output_per_mtok"]) / 1e6 * p.get("batch_multiplier", 1.0)


def collect(cfg: Config, *, make_provider: ProviderFactory = default_factory,
            clock: Callable[[], float] = time.time) -> list[str]:
    """Write one row for every open batch that has finished, failed or run out of time."""
    open_path, runs_path, latest_dir = cfg.path("open"), cfg.path("runs"), cfg.path("latest")
    open_batches: list[dict[str, Any]] = _read_json(open_path, [])
    prices = _read_json(cfg.path("prices"), {}) if cfg.raw.get("prices") else {}
    schema = cfg.schema()
    records, _ = cfg.requests()
    targets = {target_id(t): t for t in cfg.targets}
    still_open, log = [], []
    for b in open_batches:
        tid = b["target"]
        provider = make_provider(targets.get(tid, b))
        now = clock()
        try:
            status = provider.status(b["batch_id"])
        except Exception as exc:
            log.append(f"{tid}: status check failed, will retry: {_short_error(exc)}")
            still_open.append(b)
            continue
        finished = status.state in (BatchState.DONE, BatchState.FAILED)
        gave_up = not finished and now - b["submitted_ts"] > cfg.give_up_after_s
        if not finished and not gave_up:
            still_open.append(b)
            log.append(f"{tid}: {status.state.value} ({status.detail}) after "
                       f"{(now - b['submitted_ts']) / 60:.0f} min")
            continue

        keys: dict[str, str] = b["keys"]
        answers: dict[str, Any] = {}
        counts = {"ok": 0, "error": 0, "invalid": 0, "truncated": 0}
        in_tok = out_tok = 0
        versions: set[str] = set()
        seen: set[str] = set()
        if finished:
            try:
                raws = list(provider.results(b["batch_id"]))
            except Exception as exc:
                log.append(f"{tid}: fetching results failed, will retry: {_short_error(exc)}")
                still_open.append(b)
                continue
            for raw in raws:
                if raw.custom_id not in keys or raw.custom_id in seen:
                    continue
                seen.add(raw.custom_id)
                in_tok += raw.input_tokens
                out_tok += raw.output_tokens
                if raw.model_version:
                    versions.add(raw.model_version)
                if not raw.ok:
                    counts["error"] += 1
                    continue
                if raw.error and "truncated" in raw.error:
                    counts["truncated"] += 1
                try:
                    answers[keys[raw.custom_id]] = check(raw.text, schema)
                    counts["ok"] += 1
                except InvalidOutput:
                    counts["invalid"] += 1

        latest_path = latest_dir / f"{tid}.json"
        previous = _read_json(latest_path, None)
        compared = changed = None
        if previous is not None:
            both = set(previous) & set(answers)
            compared = len(both)
            changed = sum(previous[k] != answers[k] for k in both)
        if answers:
            _write_json(latest_path, answers)

        field_acc, exact_acc = _accuracy(records, answers)
        cost = _cost(prices, b["provider"], in_tok, out_tok)
        created, started, ended = status.created_at, status.started_at, status.ended_at
        if finished and created and ended:
            turnaround, source = ended - created, "provider"
        elif finished:
            turnaround, source = now - b["submitted_ts"], "poll"
        else:
            turnaround, source = None, None
        row = {
            "target": tid, "label": targets.get(tid, b).get("label") or b.get("label"),
            "provider": b["provider"], "model": b["model"], "batch_id": b["batch_id"],
            "state": "gave_up" if gave_up else status.state.value, "provider_detail": status.detail,
            "submitted_ts": b["submitted_ts"], "submitted_at": _iso(b["submitted_ts"]),
            "submit_hour_utc": datetime.fromtimestamp(b["submitted_ts"], timezone.utc).hour,
            "created_at": _iso(created), "started_at": _iso(started), "ended_at": _iso(ended),
            "turnaround_s": round(turnaround, 1) if turnaround is not None else None,
            "turnaround_source": source,
            "queue_s": round(started - created, 1) if created and started else None,
            "collected_at": _iso(now),
            "records": len(keys), **counts, "missing": len(keys) - len(seen),
            "compared_with_previous": compared, "changed_vs_previous": changed,
            "field_accuracy": field_acc, "exact_accuracy": exact_acc,
            "input_tokens": in_tok, "output_tokens": out_tok,
            "cost_usd": round(cost, 5) if cost is not None else None,
            "cost_per_1000_usd": round(cost / len(keys) * 1000, 4) if cost is not None and keys else None,
            "model_versions": sorted(versions), "steadybatch_version": __version__,
        }
        _append_row(runs_path, row)
        log.append(f"{tid}: {row['state']}, {row['ok']}/{row['records']} ok, {row['missing']} missing, "
                   f"turnaround {row['turnaround_s']} s, changed {changed}/{compared}")
    _write_json(open_path, still_open)
    return log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="steadybatch-track", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["submit", "collect", "site"])
    parser.add_argument("--config", default="tracker/config.json")
    parser.add_argument("--out", default="tracker/site", help="site: output folder")
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    if args.command == "submit":
        lines = submit(cfg)
    elif args.command == "collect":
        lines = collect(cfg)
    else:
        from .tracker_site import build
        lines = [f"wrote {build(cfg.path('runs'), Path(args.out), open_path=cfg.path('open'))}"]
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
