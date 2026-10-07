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


def to_requests(records: list[dict[str, Any]], max_tokens: int, instructions: str = "") -> list[Request]:
    system = SYSTEM_PROMPT + ("\n\n" + instructions.strip() if instructions.strip() else "")
    return [
        Request(key=str(r["key"]), system=system, max_tokens=max_tokens, temperature=0.0,
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

    runner = Runner(get_provider(args.provider, args.model, thinking=args.thinking), args.model, response_schema=schema,
                    checkpoint=out / "checkpoint.sqlite", max_attempts=args.max_attempts,
                    poll_every=args.poll_every)
    instructions = Path(args.instructions).read_text() if args.instructions else ""
    report = runner.run(to_requests(records, args.max_tokens, instructions))

    with (out / "results.jsonl").open("w") as f:
        for r in report.results:
            f.write(json.dumps({"key": r.key, "outcome": r.outcome.value, "data": r.data,
                                "error": r.error, "attempts": r.attempts, "history": r.history}) + "\n")

    in_tok = sum(r.input_tokens for r in report.results)
    out_tok = sum(r.output_tokens for r in report.results)
    summary: dict[str, Any] = {
        "provider": args.provider,
        "model": args.model,
        "instructions_file": args.instructions,
        "thinking": args.thinking or "model default",
        "max_tokens": args.max_tokens,
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


REPORT_COLUMNS = [
    ("run", "Run"),
    ("provider", "Provider"),
    ("model", "Model"),
    ("records", "Records"),
    ("ok", "OK"),
    ("failed_after_retries", "Failed"),
    ("lines_missing_seen", "Missing"),
    ("lines_errored_seen", "Errored"),
    ("lines_invalid_seen", "Invalid"),
    ("retried_records", "Retried"),
    ("batch_seconds_median", "Median batch (s)"),
    ("batch_seconds_max", "Slowest batch (s)"),
    ("cost_per_1000_records_usd", "Cost per 1k ($)"),
    ("mean_field_accuracy", "Accuracy"),
]


def load_summaries(run_dirs: list[str]) -> list[dict[str, Any]]:
    rows = []
    for d in run_dirs:
        path = Path(d) / "summary.json"
        if not path.exists():
            print(f"skipping {d}: no summary.json", file=sys.stderr)
            continue
        s = json.loads(path.read_text())
        s["run"] = Path(d).name
        acc = s.get("field_accuracy") or {}
        if acc:
            s["mean_field_accuracy"] = round(sum(acc.values()) / len(acc), 4)
        rows.append(s)
    return rows


def write_charts(rows: list[dict[str, Any]], out: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping charts (pip install matplotlib)", file=sys.stderr)
        return []
    labels = [r["run"] for r in rows]
    made = []

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.2), 4))
    bottom = [0] * len(rows)
    for key, name in [("lines_missing_seen", "Missing"), ("lines_errored_seen", "Errored"),
                      ("lines_invalid_seen", "Invalid JSON")]:
        vals = [100 * r.get(key, 0) / max(r.get("records", 1), 1) for r in rows]
        ax.bar(labels, vals, bottom=bottom, label=name)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("% of records (before retries)")
    ax.set_title("Problems caught per run")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    fig.savefig(out / "problems.png", dpi=150)
    plt.close(fig)
    made.append("problems.png")

    costs = [(r["run"], r["cost_per_1000_records_usd"]) for r in rows if "cost_per_1000_records_usd" in r]
    if costs:
        fig, ax = plt.subplots(figsize=(max(6, len(costs) * 1.2), 4))
        ax.bar([c[0] for c in costs], [c[1] for c in costs])
        ax.set_ylabel("USD per 1,000 records (incl. retries)")
        ax.set_title("Cost per 1,000 records")
        fig.tight_layout()
        fig.savefig(out / "cost.png", dpi=150)
        plt.close(fig)
        made.append("cost.png")
    return made


def add_costs(rows: list[dict[str, Any]], prices_path: str) -> None:
    """Fill in cost for runs whose summary has token counts but no cost yet."""
    prices = json.loads(Path(prices_path).read_text())
    for r in rows:
        p = prices.get(r.get("provider", ""))
        if not p or "cost_usd" in r:
            continue
        if p.get("model") and p["model"] != r.get("model"):
            continue  # prices were recorded for a different model
        cost = (r.get("input_tokens", 0) * p["input_per_mtok"] + r.get("output_tokens", 0) * p["output_per_mtok"]) / 1e6 * p.get("batch_multiplier", 1.0)
        r["cost_usd"] = round(cost, 6)
        r["cost_per_1000_records_usd"] = round(cost / max(r.get("records", 1), 1) * 1000, 4)


def cmd_report(args: argparse.Namespace) -> int:
    rows = load_summaries(args.runs)
    if not rows:
        print("no runs with a summary.json found", file=sys.stderr)
        return 1
    if args.prices:
        add_costs(rows, args.prices)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def cell(row: dict[str, Any], key: str) -> str:
        v = row.get(key)
        return "" if v is None else str(v)

    header = [name for _, name in REPORT_COLUMNS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(cell(r, k) for k, _ in REPORT_COLUMNS) + " |")

    import csv
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([cell(r, k) for k, _ in REPORT_COLUMNS])

    charts = [] if args.no_charts else write_charts(rows, out)
    md = ["# Benchmark results", "",
          "Counts of missing, errored and invalid lines are before retries; "
          "OK and Failed are after retries.", "", *lines, ""]
    md += [f"![{c}]({c})" for c in charts]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out / 'summary.md'} and {out / 'summary.csv'}" + (f" and {len(charts)} charts" if charts else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="steadybatch-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run one workload on one provider")
    run.add_argument("--provider", required=True, choices=["fake", "openai", "anthropic", "vllm", "gemini", "bedrock"])
    run.add_argument("--model", required=True)
    run.add_argument("--data", required=True)
    run.add_argument("--schema")
    run.add_argument("--instructions", help="text file with field definitions, added to the system prompt")
    run.add_argument("--out", required=True)
    run.add_argument("--limit", type=int)
    run.add_argument("--max-tokens", type=int, default=512)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--thinking", choices=["disabled", "adaptive"],
                     help="Anthropic only: set thinking explicitly instead of using the model default")
    run.add_argument("--poll-every", type=float, default=60.0)
    run.add_argument("--prices", help="JSON file with per-provider token prices")
    run.set_defaults(func=cmd_run)

    cmp_ = sub.add_parser("compare", help="compare two runs of the same workload")
    cmp_.add_argument("run_a")
    cmp_.add_argument("run_b")
    cmp_.set_defaults(func=cmd_compare)

    rep = sub.add_parser("report", help="turn run summaries into a table, CSV and charts")
    rep.add_argument("runs", nargs="+", help="run folders that contain summary.json")
    rep.add_argument("--out", default="results/latest")
    rep.add_argument("--no-charts", action="store_true")
    rep.add_argument("--prices", help="JSON price file; fills in cost for runs that lack it")
    rep.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
