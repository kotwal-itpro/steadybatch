# Results

Published benchmark results live here, one folder per run, named
`<date>-<provider>-<model>-<records>` (for example `2026-10-20-openai-gpt-4o-mini-1000`).

Each folder holds:

- `summary.json`: counts, retries, tokens, cost, turnaround and accuracy for the run
- `results.jsonl`: one line per record, in input order
- `prices.json`: the prices used for the cost figures, with the date they were taken

The local `checkpoint.sqlite` is not published.

To build the comparison table and charts from any set of runs:

```bash
pip install matplotlib   # optional, for charts
steadybatch-bench report results/2026-10-* --out results/latest
```

`report` writes `summary.md`, `summary.csv`, and (with matplotlib) `problems.png` and `cost.png`.
