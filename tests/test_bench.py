import json

from steadybatch import bench


def make_tickets(path, n=60):
    import random
    import sys
    sys.path.insert(0, "examples")
    from make_synthetic import make
    rng = random.Random(1)
    path.write_text("\n".join(json.dumps(make(i, rng)) for i in range(n)) + "\n")


def run_fake(tmp_path, name):
    data = tmp_path / "tickets.jsonl"
    if not data.exists():
        make_tickets(data)
    out = tmp_path / name
    rc = bench.main(["run", "--provider", "fake", "--model", "fake", "--data", str(data),
                     "--schema", "examples/support_ticket.schema.json", "--out", str(out),
                     "--prices", "examples/prices.example.json", "--poll-every", "0"])
    assert rc == 0
    return out


def test_run_writes_results_and_summary(tmp_path):
    out = run_fake(tmp_path, "run1")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["records"] == 60 and summary["ok"] == 60
    assert "cost_per_1000_records_usd" in summary and "field_accuracy" in summary
    lines = (out / "results.jsonl").read_text().splitlines()
    assert len(lines) == 60 and json.loads(lines[0])["key"] == "ticket-0000000"


def test_report_builds_table_csv_and_skips_bad_dirs(tmp_path, capsys):
    a, b = run_fake(tmp_path, "run1"), run_fake(tmp_path, "run2")
    out = tmp_path / "report"
    rc = bench.main(["report", str(a), str(b), str(tmp_path / "missing"), "--out", str(out), "--no-charts"])
    assert rc == 0
    md = (out / "summary.md").read_text()
    assert "| run1 |" in md and "| run2 |" in md
    csv_rows = (out / "summary.csv").read_text().splitlines()
    assert len(csv_rows) == 3 and csv_rows[0].startswith("Run,Provider")
    assert "skipping" in capsys.readouterr().err


def test_compare_reports_match_rates(tmp_path, capsys):
    a, b = run_fake(tmp_path, "run1"), run_fake(tmp_path, "run2")
    capsys.readouterr()
    assert bench.main(["compare", str(a), str(b)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["records_in_both"] == 60 and result["exact_match_rate"] == 1.0


def test_report_adds_cost_from_a_price_file(tmp_path):
    a = run_fake(tmp_path, "run1")
    summary = json.loads((a / "summary.json").read_text())
    for k in ("cost_usd", "cost_per_1000_records_usd"):
        summary.pop(k, None)
    (a / "summary.json").write_text(json.dumps(summary))
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"fake": {"model": "fake", "input_per_mtok": 1.0, "output_per_mtok": 2.0, "batch_multiplier": 0.5}}))
    out = tmp_path / "report"
    assert bench.main(["report", str(a), "--out", str(out), "--no-charts", "--prices", str(prices)]) == 0
    expected = round((summary["input_tokens"] * 1.0 + summary["output_tokens"] * 2.0) / 1e6 * 0.5 / 60 * 1000, 4)
    assert str(expected) in (out / "summary.md").read_text()
