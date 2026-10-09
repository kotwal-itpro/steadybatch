# steadybatch

[![tests](https://github.com/kotwal-itpro/steadybatch/actions/workflows/tests.yml/badge.svg)](https://github.com/kotwal-itpro/steadybatch/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23221956.svg)](https://doi.org/10.5281/zenodo.23221956)

Run millions of LLM requests through batch APIs, and know that every one of them came back right.

Batch APIs are the cheapest way to run a model over a lot of data. You upload a file, wait a few hours, and pay about half the normal price. They also fail in quiet ways:

- A batch says "completed", but some of your lines never came back. No error, they're just gone.
- A line comes back, but it isn't the JSON you asked for, or it was cut off halfway.
- You re-run the same batch and get different answers.
- Your job crashes halfway through, and you can't tell what you already paid for.

If you're turning a million documents into structured data, each of those turns into a wrong number somewhere downstream. steadybatch handles them for you.

## What it does

- **Checks off every request.** Each request gets a stable ID. When a batch finishes, steadybatch checks that every ID came back. Anything missing is marked and retried.
- **Checks every answer.** Each response is parsed and validated against your JSON Schema before it counts as done.
- **Retries one line, not the whole batch.** A bad line goes back on its own. Good lines are never paid for twice.
- **Survives restarts.** Progress is saved to a small SQLite file after every step. If the process dies, run it again: open batches are picked up, not resubmitted, and finished work is skipped.
- **Gives results back in your order.** Whatever order the provider returns things in, you get them back in the order you sent them.
- **Stays under provider limits.** Work is split so each batch stays under the provider's count and size caps.

It works with the OpenAI Batch API, Gemini API batch mode, the Anthropic Message Batches API, and self-hosted vLLM. AWS Bedrock is next (see [docs/ROADMAP.md](https://github.com/kotwal-itpro/steadybatch/blob/main/docs/ROADMAP.md)).

## Quick start

Install from PyPI. Pick the providers you need in the brackets:

```bash
pip install "steadybatch[openai,gemini,anthropic]"
# other extras: vllm (self-hosted), report (charts)
```

```python
from steadybatch import Request, Runner
from steadybatch.providers.openai_batch import OpenAIBatch

schema = {
    "type": "object",
    "properties": {"sentiment": {"type": "string", "enum": ["negative", "neutral", "positive"]}},
    "required": ["sentiment"],
    "additionalProperties": False,
}

requests = [
    Request(key=row_id, messages=[{"role": "user", "content": f"Sentiment of: {text}"}])
    for row_id, text in my_rows
]

runner = Runner(OpenAIBatch(), "gpt-4.1-mini", response_schema=schema, checkpoint="job.sqlite")
report = runner.run(requests)

print(report.ok, "ok,", report.failed, "failed")
for r in report.results:          # same order as `requests`
    print(r.key, r.outcome.value, r.data)
```

Run the same script again after a crash and it carries on from `job.sqlite`.

Before a big job, read [docs/GUIDE.md](https://github.com/kotwal-itpro/steadybatch/blob/main/docs/GUIDE.md): ten steps for running a batch job you can trust, each one learned from a real run.

## The benchmark

This repo also holds the harness for an open study comparing batch APIs on cost, turnaround, failure modes and reproducibility. It uses one structured-extraction workload across providers.

```bash
# make a synthetic dataset with known answers (no private data)
python examples/make_synthetic.py --n 1000 > examples/support_tickets.jsonl

# try the whole pipeline for free with the built-in fake provider
steadybatch-bench run --provider fake --model fake \
  --data examples/support_tickets.jsonl --schema examples/support_ticket.schema.json \
  --out runs/fake-1 --prices examples/prices.example.json --poll-every 0

# run it for real, twice, then compare the two runs
steadybatch-bench run --provider openai --model gpt-4.1-mini ... --out runs/openai-1
steadybatch-bench run --provider openai --model gpt-4.1-mini ... --out runs/openai-2
steadybatch-bench compare runs/openai-1 runs/openai-2
```

Each run writes `results.jsonl` (one line per record, in input order) and `summary.json` with:
- success and failure counts, including lines that went missing, errored or failed the schema
- how many records needed a retry
- token use and cost
- batch turnaround times
- field-by-field accuracy against the known answers

`compare` reports how often two runs of the same workload agree, both exactly and field by field.

To turn any set of runs into one table, a CSV and charts:

```bash
pip install matplotlib   # optional, for the charts
steadybatch-bench report runs/openai-1 runs/gemini-1 --out results/latest
```

Published results go in [results/](https://github.com/kotwal-itpro/steadybatch/blob/main/results/), one folder per run.

Useful `run` options for real jobs:
- `--instructions FILE` adds plain-language field definitions to the prompt (see finding 3 below for why this matters)
- `--batch-size N` and `--max-open-batches N` keep a big job under a provider's queue limit. If a provider still says its queue is full, the runner waits for its own batches or halves the batch size and retries, without using up any request's retries (finding 7)
- `--thinking disabled|adaptive` (Anthropic) sets thinking explicitly instead of using the model default

### Results so far

1,000 synthetic support tickets with known answers, the same schema and field definitions on every provider, batch prices taken on 2026-10-07. Full tables: [results/latest/summary.md](https://github.com/kotwal-itpro/steadybatch/blob/main/results/latest/summary.md).

| Model | Returned | Retries | Sentiment accuracy | Median batch time | Cost per 1,000 records |
|---|---|---|---|---|---|
| gpt-4.1-mini (OpenAI) | 1,000 / 1,000 | 0 | 92.9% | 4 min | $0.085 |
| gemini-3.5-flash-lite (Gemini) | 1,000 / 1,000 | 0 | 82.1% | 3 min | $0.088 |
| claude-haiku-5-5, default settings (Anthropic) | 999 / 1,000 | 10 | 89.1% | 10 min | $0.053 |
| claude-haiku-5-5, thinking disabled (Anthropic) | 1,000 / 1,000 | 0 | 91.0% | 5 min | $0.048 |
| Qwen2.5-7B-Instruct, self-hosted (vLLM, 1x L4) | 1,000 / 1,000 | 0 | 85.4% | 29 s | $0.004* |

\* GPU time for the run at $0.50/hour on demand. The whole rented-GPU session, including installing vLLM, downloading the model and one-time compilation, cost about $0.05. For vLLM the time is the whole run, since there is no queue.

All five got product and issue type 100% right. At 10,000 records, OpenAI and Gemini again returned every line with no retries. **At 100,000 records, all three hosted providers returned 100,000 of 100,000 with no retries**, at the same accuracy; what changed was turnaround (Gemini 45 minutes in 10 batches, Claude 2 h 44 min as one batch, OpenAI's batches slowing from 14 to 84 minutes through the day). Sending the same records again changed 0.7% of OpenAI's answers, 3% of Claude's and 6% of Gemini's (finding 10). What we learned along the way, including our own mistakes, is in [docs/FINDINGS.md](https://github.com/kotwal-itpro/steadybatch/blob/main/docs/FINDINGS.md). The short version:
- A model listed as available can still be refused when the batch is created.
- Undefined fields look like model errors: one plain sentence per field took follow-up accuracy from 31% to 98-100%.
- Models fail in different directions under the same instructions, and one setting (Claude's thinking) flipped the direction on the same model.
- A model that decides for itself when to think can use up the whole `max_tokens` budget and return nothing.
- A big job can be refused with a misleading "quota exceeded" error, and a dropped connection can stop an eight-hour run; the checkpoint resumed it without resending anything.
- Temperature 0 doesn't mean the same answer twice.

Write-up: [I Gave Two AI Models the Same Instructions. They Got It Wrong in Opposite Ways.](https://kotwal-itpro.github.io/2026/10/07/same-instructions-opposite-mistakes/)

Fill in `examples/prices.example.json` from each provider's pricing page on the day you run, and record the date with your results. Prices change.

## Running on Kubernetes

A batch job fits well as a Kubernetes Job. Keep the checkpoint on a persistent volume so a rescheduled pod picks up where the last one stopped. See [k8s/job.yaml](https://github.com/kotwal-itpro/steadybatch/blob/main/k8s/job.yaml).

## Testing

```bash
git clone https://github.com/kotwal-itpro/steadybatch.git
cd steadybatch
pip install -e ".[dev]"
pytest
```

The tests use a fake provider that drops lines, returns errors, sends malformed JSON and changes its answers between runs. You can check all the retry and recovery logic without an API key or a bill.

## Status

Early, and moving. The core is tested against a fake provider that misbehaves on purpose. The OpenAI, Gemini and Anthropic adapters have run real jobs of 1,000 to 10,000 records against the live services (100,000 is in progress). The vLLM adapter has run on a rented NVIDIA L4. Issues and pull requests are welcome.

## Citing

If you use steadybatch or the benchmark results in your work, please cite it: Kotwal, A. P. *steadybatch: reliable batch LLM inference*. Zenodo. https://doi.org/10.5281/zenodo.23221956 (see [CITATION.cff](https://github.com/kotwal-itpro/steadybatch/blob/main/CITATION.cff)).

## License

Apache-2.0.
