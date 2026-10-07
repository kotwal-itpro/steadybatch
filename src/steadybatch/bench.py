"""Benchmark harness: run one workload on one provider and write down what happened.

    steadybatch-bench run --provider openai --model gpt-4o-mini \\
        --data examples/support_tickets.jsonl --schema examples/support_ticket.schema.json \\
        --out runs/openai-1 --prices examples/prices.example.json

    steadybatch-bench compare runs/openai-1 runs/openai-2

``run`` writes results.jsonl (one line per record, in input order) and
summary.json. ``compare`` checks two runs of the same workload against each
other, which is how we measure reproducibility.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

from .models import Outcome, Request
from .providers import get_provider
from .runner import Runner

SYSTEM_PROMPT = (
    "You read customer support conversations and extract structured facts. "
    "Answer with a single JSON object and nothing else."
)


def load_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    break
    return records


def to_requests(records: list[dict[str, Any]], max_tokens: int) -> list[Request]:
    return [
        Request(key=str(r["key"]), system=SYSTEM_PROMPT, max_tokens=max_tokens, temperature=0.0,
                messages=[{"role": "user", "content": r["text"]}])
        for r in records
    ]


def accuracy(records: list[dict[str, Any]], results_by_key: dict[str, Any]) -> dict[str, float]:
    """Field-level accuracy against the 'labels' in the dataset, when present."""
    scores: dict[str, list[int]] = {}
    for r in records:
        labels = r.get("labels")
        data = results_by_key.get(str(r["key"]))
        if not labels or not isinstance(data, dict):
            continue
        for field, expected in labels.items():
            scores.setdefault(field, []).append(int(data.get(field) == expected))
    return {f: round(sum(v) / len(v), 4) for f, v in scores.items() if v}


def cmd_run(args: argparse.Namespace) -> int:
    records = load_records(Path(args.data), args.limit)
    schema = json.loads(Path(args.schema).read_text()) if args.schema else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runner = Runner(get_provider(args.provider, args.model), args.model, response_schema=schema,
                    checkpoint=out / "checkpoint.sqlite", max_attempts=args.max_attempts,
                    poll_every=args.poll_every)
    report = runner.run(to_requests(records, args.max_tokens))

    with (out / "results.jsonl").open("w") as f:
        for r in report.results:
            f.write(json.dumps({"key": r.key, "outcome": r.outcome.value, "data": r.data,
                                "error": r.error, "attempts": r.attempts, "history": r.history}) + "\n")

    in_tok = sum(r.input_tokens for r in report.results)
    out_tok = sum(r.output_tokens for r in report.results)
    summary: dict[str, Any] = {
        "provider": args.provider,
        "model": args.model,
        "records": len(records),
        "ok": report.ok,
        "failed_after_retries": report.failed,
        "outcomes": {o.value: sum(r.outcome is o for r in report.results) for o in Outcome},
        "lines_missing_seen": report.lines_missing,
        "lines_errored_seen": report.lines_errored,
        "lines_invalid_seen": report.lines_invalid,
        "batches_submitted": report.batches_submitted,
        "retried_records": sum(r.attempts > 1 for r in report.results),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "wall_seconds": round(report.seconds, 1),
    }
    if report.batch_seconds:
        summary["batch_seconds_median"] = round(statistics.median(report.batch_seconds), 1)
        summary["batch_seconds_max"] = round(max(report.batch_seconds), 1)
    if args.prices:
        p = json.loads(Path(args.prices).read_text())[args.provider]
        cost = (in_tok * p["input_per_mtok"] + out_tok * p["output_per_mtok"]) / 1e6 * p.get("batch_multiplier", 1.0)
        summary["cost_usd"] = round(cost, 4)
        summary["cost_per_1000_records_usd"] = round(cost / max(len(records), 1) * 1000, 4)
    acc = accuracy(records, {r.key: r.data for r in report.results})
    if acc:
        summary["field_accuracy"] = acc
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    def load(d: str) -> dict[str, Any]:
        rows = [json.loads(l) for l in (Path(d) / "results.jsonl").read_text().splitlines() if l]
        return {r["key"]: r["data"] for r in rows if r["outcome"] == "ok"}

    a, b = load(args.run_a), load(args.run_b)
    both = sorted(set(a) & set(b))
    exact = sum(a[k] == b[k] for k in both)
    fields: dict[str, list[int]] = {}
    for k in both:
        if isinstance(a[k], dict) and isinstance(b[k], dict):
            for f in set(a[k]) | set(b[k]):
                fields.setdefault(f, []).append(int(a[k].get(f) == b[k].get(f)))
    result = {
        "records_in_both": len(both),
        "exact_match_rate": round(exact / len(both), 4) if both else None,
        "field_match_rate": {f: round(sum(v) / len(v), 4) for f, v in sorted(fields.items())},
    }
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="steadybatch-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run one workload on one provider")
    run.add_argument("--provider", required=True, choices=["fake", "openai", "anthropic", "vllm", "gemini", "bedrock"])
    run.add_argument("--model", required=True)
    run.add_argument("--data", required=True)
    run.add_argument("--schema")
    run.add_argument("--out", required=True)
    run.add_argument("--limit", type=int)
    run.add_argument("--max-tokens", type=int, default=512)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--poll-every", type=float, default=60.0)
    run.add_argument("--prices", help="JSON file with per-provider token prices")
    run.set_defaults(func=cmd_run)

    cmp_ = sub.add_parser("compare", help="compare two runs of the same workload")
    cmp_.add_argument("run_a")
    cmp_.add_argument("run_b")
    cmp_.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
