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
import hashlib
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
# A weekly target (every_days >= 7) may send a larger batch, to check whether big jobs queue
# differently from small ones. Still capped in code.
HARD_MAX_LARGE_RECORDS = 5000


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
    def max_monthly_cost_usd(self) -> float:
        return float(self.raw.get("max_monthly_cost_usd_per_target", 10))

    @property
    def give_up_after_s(self) -> float:
        return float(self.raw.get("give_up_after_hours", 50)) * 3600

    @property
    def targets(self) -> list[dict[str, Any]]:
        return self.raw["targets"]

    def schema(self) -> dict[str, Any] | None:
        return json.loads(self.path("schema").read_text()) if self.raw.get("schema") else None

    def records_for(self, t: dict[str, Any] | None) -> int:
        """How many records a target sends: its own "records", else max_records, capped in code."""
        if not t or not t.get("records"):
            return self.max_records
        cap = HARD_MAX_LARGE_RECORDS if every_days(t) >= 7 else HARD_MAX_RECORDS
        return min(int(t["records"]), cap)

    def requests(self, t: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[Request]]:
        inputs = self.root / t["inputs"] if t and t.get("inputs") else self.path("inputs")
        records = load_records(inputs, self.records_for(t))
        instructions = self.path("instructions").read_text() if self.raw.get("instructions") else ""
        return records, to_requests(records, int(self.raw.get("max_tokens", 300)), instructions)

    def prepared(self, t: dict[str, Any]) -> tuple[list[dict[str, Any]], list[PreparedRequest]]:
        records, requests = self.requests(t)
        schema = self.schema()
        return records, [PreparedRequest(custom_id_for(r, t["model"], schema), r, t["model"], schema)
                         for r in requests]


def target_id(t: dict[str, Any]) -> str:
    return t.get("id") or f"{t['provider']}-{t['model']}"


def every_days(t: dict[str, Any]) -> float:
    return float(t.get("every_days") or 0)


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
    hour = datetime.fromtimestamp(now, timezone.utc).hour
    log: list[str] = []
    for t in cfg.targets:
        tid = target_id(t)
        if t.get("slot_hours") is not None and hour not in t["slot_hours"]:
            continue  # this target only goes out in certain slots
        if every_days(t):
            last = max([r["submitted_ts"] for r in rows if r["target"] == tid]
                       + [b["submitted_ts"] for b in open_batches if b["target"] == tid], default=None)
            # Half a day of slack, so a cron that starts late doesn't push it to the next week.
            if last is not None and now - last < (every_days(t) - 0.5) * 86400:
                log.append(f"{tid}: skipped, runs every {every_days(t):g} days")
                continue
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
        month = today[:7]
        spent = sum(r.get("cost_usd") or 0 for r in rows
                    if r["target"] == tid and _day(r["submitted_ts"])[:7] == month)
        if spent >= cfg.max_monthly_cost_usd:
            log.append(f"{tid}: skipped, ${spent:.2f} spent this month (cap ${cfg.max_monthly_cost_usd:.2f})")
            continue
        provider = make_provider(t)
        _records, prepared = cfg.prepared(t)
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
        # The custom_id -> record mapping is rebuilt from the frozen inputs at collect time,
        # so open.json stays small even for a 5,000-record batch.
        open_batches.append({"target": tid, "label": t.get("label"),
                             "provider": t["provider"], "model": t["model"],
                             "batch_id": batch_id, "submitted_ts": now, "records": len(prepared)})
        log.append(f"{tid}: submitted {len(prepared)} records as {batch_id}")
    _write_json(open_path, open_batches)
    return log


def fingerprint(answer: Any) -> str:
    """A short, stable hash of a parsed answer (key order doesn't matter)."""
    blob = json.dumps(answer, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


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


def _batch_keys(cfg: Config, b: dict[str, Any], target: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The records a batch sent and its custom_id -> record key mapping."""
    if "keys" in b:  # batches submitted before the mapping was rebuilt at collect time
        records, _ = cfg.requests(target)
        wanted = set(b["keys"].values())
        return [r for r in records if str(r["key"]) in wanted], b["keys"]
    records, prepared = cfg.prepared({**target, "records": b.get("records") or target.get("records")})
    return records, {p.custom_id: p.request.key for p in prepared}


def measure(raws: list[Any], keys: dict[str, str], records: list[dict[str, Any]], schema: Any,
            previous: dict[str, str] | None) -> tuple[dict[str, Any], dict[str, str]]:
    """Count, validate, score and fingerprint one batch's results. Returns the row fields and the
    new fingerprints. Shared by the batch APIs and the self-hosted run."""
    answers: dict[str, Any] = {}
    counts = {"ok": 0, "error": 0, "invalid": 0, "truncated": 0}
    in_tok = out_tok = 0
    versions: set[str] = set()
    seen: set[str] = set()
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
    # Keep only a fingerprint of each answer, not the answer itself: enough to tell
    # whether it changed next time, and no model output is republished.
    prints = {k: fingerprint(v) for k, v in answers.items()}
    compared = changed = None
    if previous is not None:
        both = set(previous) & set(prints)
        compared = len(both)
        changed = sum(previous[k] != prints[k] for k in both)
    field_acc, exact_acc = _accuracy(records, answers)
    return {"records": len(keys), **counts, "missing": len(keys) - len(seen),
            "compared_with_previous": compared, "changed_vs_previous": changed,
            "field_accuracy": field_acc, "exact_accuracy": exact_acc,
            "input_tokens": in_tok, "output_tokens": out_tok,
            "model_versions": sorted(versions)}, prints


def collect(cfg: Config, *, make_provider: ProviderFactory = default_factory,
            clock: Callable[[], float] = time.time) -> list[str]:
    """Write one row for every open batch that has finished, failed or run out of time."""
    open_path, runs_path, latest_dir = cfg.path("open"), cfg.path("runs"), cfg.path("latest")
    open_batches: list[dict[str, Any]] = _read_json(open_path, [])
    prices = _read_json(cfg.path("prices"), {}) if cfg.raw.get("prices") else {}
    schema = cfg.schema()
    targets = {target_id(t): t for t in cfg.targets}
    still_open, log = [], []
    for b in open_batches:
        tid = b["target"]
        target = targets.get(tid, b)
        provider = make_provider(target)
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

        records, keys = _batch_keys(cfg, b, target)
        raws: list[Any] = []
        if finished:
            try:
                raws = list(provider.results(b["batch_id"]))
            except Exception as exc:
                log.append(f"{tid}: fetching results failed, will retry: {_short_error(exc)}")
                still_open.append(b)
                continue

        latest_path = latest_dir / f"{tid}.json"
        fields, prints = measure(raws, keys, records, schema, _read_json(latest_path, None))
        if prints:
            _write_json(latest_path, prints)
        cost = _cost(prices, b["provider"], fields["input_tokens"], fields["output_tokens"])
        created, started, ended = status.created_at, status.started_at, status.ended_at
        if finished and created and ended:
            turnaround, source = ended - created, "provider"
        elif finished:
            turnaround, source = now - b["submitted_ts"], "poll"
        else:
            turnaround, source = None, None
        row = {
            "target": tid, "label": target.get("label") or b.get("label"),
            "provider": b["provider"], "model": b["model"], "batch_id": b["batch_id"],
            "state": "gave_up" if gave_up else status.state.value, "provider_detail": status.detail,
            "submitted_ts": b["submitted_ts"], "submitted_at": _iso(b["submitted_ts"]),
            "submit_hour_utc": datetime.fromtimestamp(b["submitted_ts"], timezone.utc).hour,
            "created_at": _iso(created), "started_at": _iso(started), "ended_at": _iso(ended),
            "turnaround_s": round(turnaround, 1) if turnaround is not None else None,
            "turnaround_source": source,
            "queue_s": round(started - created, 1) if created and started else None,
            "collected_at": _iso(now), **fields,
            "cost_usd": round(cost, 5) if cost is not None else None,
            "cost_per_1000_usd": round(cost / len(keys) * 1000, 4) if cost is not None and keys else None,
            "steadybatch_version": __version__,
        }
        if every_days(target):
            row["cadence"] = "weekly"
        _append_row(runs_path, row)
        log.append(f"{tid}: {row['state']}, {row['ok']}/{row['records']} ok, {row['missing']} missing, "
                   f"turnaround {row['turnaround_s']} s, changed {row['changed_vs_previous']}"
                   f"/{row['compared_with_previous']}")
    _write_json(open_path, still_open)
    return log


def selfhost(cfg: Config, *, engine: str, model: str, gpu: str, gpu_usd_per_hour: float,
             records: int, out: Path, make_provider: Callable[[str, str], BatchProvider] | None = None,
             clock: Callable[[], float] = time.perf_counter, wall: Callable[[], float] = time.time,
             note: str = "") -> dict[str, Any]:
    """Run the tracker's inputs through a self-hosted engine (SGLang or vLLM offline) once and
    write a row comparable to the batch API rows. Cost is GPU time at the given hourly price.

    Two times are recorded: ``load_s`` (starting the engine and loading the weights, paid once
    per pod) and ``run_s`` (generating the batch). Cost uses both, so a single small batch is
    not flattered; ``cost_per_1000_usd_run_only`` is the steady-state number for a warm engine.
    """
    if make_provider is None:
        make_provider = lambda e, m: get_provider(e, m)
    size = min(records, HARD_MAX_LARGE_RECORDS)
    t = {"provider": engine, "model": model, "records": size, "every_days": 7}
    if size > cfg.max_records:
        t["inputs"] = cfg.raw.get("inputs_large", cfg.raw["inputs"])
    recs, prepared = cfg.prepared(t)
    keys = {p.custom_id: p.request.key for p in prepared}
    started_at = wall()
    t0 = clock()
    provider = make_provider(engine, model)
    t1 = clock()
    batch_id = provider.submit(prepared)
    raws = list(provider.results(batch_id))
    t2 = clock()
    fields, _ = measure(raws, keys, recs, cfg.schema(), None)
    load_s, run_s = t1 - t0, t2 - t1
    cost = (load_s + run_s) / 3600 * gpu_usd_per_hour
    run_cost = run_s / 3600 * gpu_usd_per_hour
    row = {"engine": engine, "model": model, "gpu": gpu, "gpu_usd_per_hour": gpu_usd_per_hour,
           "ran_at": _iso(started_at), "load_s": round(load_s, 1), "run_s": round(run_s, 1),
           "turnaround_s": round(load_s + run_s, 1), **fields,
           "cost_usd": round(cost, 5), "cost_per_1000_usd": round(cost / size * 1000, 4),
           "cost_per_1000_usd_run_only": round(run_cost / size * 1000, 4),
           "note": note, "steadybatch_version": __version__}
    _append_row(out, row)
    return row


def new_issues(cfg: Config, state_path: Path, *, limit: int = 3,
               page_url: str = "https://kotwal-itpro.github.io/steadybatch/") -> list[dict[str, str]]:
    """Incidents worth a GitHub issue that haven't had one yet.

    Only serious ones (a refused or failed batch, no result after the time limit, lost lines,
    or a batch over 6 hours); smaller ones go to the page and the feed only. On the very first
    call, everything already recorded is marked as seen, so turning this on doesn't flood the
    repo with old incidents. At most ``limit`` per call.
    """
    from .tracker_site import incident_id, problems
    rows = _rows(cfg.path("runs"))
    first_time = not state_path.exists()
    seen: list[str] = _read_json(state_path, [])
    out: list[dict[str, str]] = []
    for r in rows:
        serious = [text for level, text in problems(r) if level == "bad" or text.startswith("took ")]
        iid = incident_id(r)
        if not serious or iid in seen:
            continue
        seen.append(iid)
        if first_time or len(out) >= limit:
            continue
        label = r.get("label") or f"{r['provider']} {r['model']}"
        when = (r.get("submitted_at") or "")[:16].replace("T", " ")
        body = "\n".join([
            f"The tracker recorded a problem with the **{label}** batch sent at {when} UTC.", "",
            *[f"- {t}" for t in serious], "",
            f"- Batch ID: `{r.get('batch_id', '–')}`",
            f"- Records: {r.get('records')}, OK: {r.get('ok', '–')}, lost: {r.get('missing', '–')}",
            f"- Provider state: {r.get('provider_detail') or r.get('state')}", "",
            f"Opened automatically by the tracker workflow. Full data: {page_url}",
        ])
        out.append({"id": iid, "title": f"[tracker] {label}: {serious[0]}"[:200], "body": body})
    _write_json(state_path, seen)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="steadybatch-track", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["submit", "collect", "site", "issues", "selfhost"])
    parser.add_argument("--config", default="tracker/config.json")
    parser.add_argument("--out", default="tracker/site", help="site: output folder; issues: JSON file")
    parser.add_argument("--state", default="tracker/data/reported.json", help="issues: incidents already filed")
    parser.add_argument("--engine", choices=["sglang", "vllm"], help="selfhost: the engine")
    parser.add_argument("--model", help="selfhost: model path or Hugging Face name")
    parser.add_argument("--gpu", help='selfhost: what it ran on, e.g. "1x H100 80GB (RunPod)"')
    parser.add_argument("--gpu-usd-per-hour", type=float, help="selfhost: price of that GPU per hour")
    parser.add_argument("--records", type=int, default=200, help="selfhost: 200 or 5000")
    parser.add_argument("--note", default="", help="selfhost: free text kept with the row")
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    if args.command == "submit":
        lines = submit(cfg)
    elif args.command == "collect":
        lines = collect(cfg)
    elif args.command == "issues":
        issues = new_issues(cfg, Path(args.state))
        _write_json(Path(args.out), issues)
        lines = [f"{len(issues)} new issue(s) written to {args.out}"]
    elif args.command == "selfhost":
        if not (args.engine and args.model and args.gpu and args.gpu_usd_per_hour):
            parser.error("selfhost needs --engine, --model, --gpu and --gpu-usd-per-hour")
        row = selfhost(cfg, engine=args.engine, model=args.model, gpu=args.gpu,
                       gpu_usd_per_hour=args.gpu_usd_per_hour, records=args.records,
                       out=cfg.path("selfhost"), note=args.note)
        lines = [json.dumps(row)]
    else:
        from .tracker_site import build
        selfhost_path = cfg.path("selfhost") if cfg.raw.get("selfhost") else None
        lines = [f"wrote {build(cfg.path('runs'), Path(args.out), open_path=cfg.path('open'), selfhost_path=selfhost_path)}"]
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
