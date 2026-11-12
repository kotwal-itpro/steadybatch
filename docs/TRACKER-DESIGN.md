# Batch API tracker: design draft

2026-11-12. Ankur chose two runs a day. Built on the `tracker` branch and tested on the fake provider; not pushed, nothing runs and nothing is spent until the steps under Plan are done.

## The gap

Every big provider sells a batch API at about half price and promises results "within 24 hours".
Nobody publishes how long batches actually take, day after day, per provider, or whether every
line comes back. What exists today:

- Provider docs: a 24-hour ceiling (Gemini expires at 48 hours), "often sooner", no numbers.
- OpenRouter's batch launch post (Sep 22, 2026): one chart from a two-week beta, all providers
  pooled. Median 7 minutes, p90 1 hour, p99 10.3 hours, and a strong time-of-day effect (p90 up to
  4.5 hours for 9am Pacific submissions). It is a one-time chart, not per provider, and says nothing
  about missing lines or answer drift.
- DailyBench (github.com/jacobphillips99/daily-bench): a public dashboard of benchmark scores over
  time, but on normal synchronous calls, not batch APIs.

So the question an infra engineer asks before moving a pipeline to batch ("if I submit at 9am,
when will it be done, and will all of it come back?") has no public answer. This tracker gives one.

## What it measures

The same fixed 200 synthetic support tickets (from `examples/make_synthetic.py`, with known
answers) go to each provider's batch API twice a day.

| Metric | How |
|---|---|
| Turnaround | From the provider's own timestamps (OpenAI `created_at` to `completed_at`, Anthropic `created_at` to `ended_at`, Gemini's equivalent, to be checked). Polling delay doesn't affect it. |
| Completeness | Lines missing, errored, or not valid JSON for the schema. steadybatch already counts these. |
| Answer drift | Share of the 200 records whose answer changed since the previous run on the same provider. The same input set is used every time, so this is a clean daily signal (FINDINGS #10 showed 0.7% to 6%). |
| Accuracy | Field accuracy against the known answers. |
| Model version | The model ID or fingerprint the provider reports on each line, so drift can be tied to a silent model update. |
| Cost | Per 1,000 records, from token counts and the published price file. |

Models at the start: `gpt-4.1-mini`, `gemini-3.5-flash-lite` and `claude-haiku-5-5` (thinking off),
the same three as the published study, so the tracker continues that data. A larger model per
provider, and AWS Bedrock, can be added later.

Batches go out at 01:00 UTC (evening in the US) and 13:00 UTC (morning in the US, the busy time in
OpenRouter's data), so the data shows the time-of-day effect per provider.

## How it runs

GitHub Actions, in the public steadybatch repo. A job can run for at most 6 hours, but a batch can
take 24, so submitting and collecting are separate:

1. **Submit** (cron, 01:00 and 13:00 UTC): submits one batch per provider and records the batch IDs in
   `tracker/open.json`, which the bot commits.
2. **Collect** (cron, every 30 minutes): checks the open batches. For each finished one, it fetches
   the lines, validates them, compares them with the previous run, appends one row to
   `tracker/data/runs.jsonl` and removes the batch from `open.json`. Lines still missing after 25
   hours are recorded as missing; nothing is resubmitted, because the point is to measure.
3. **Publish**: after each collect that adds data, a script rebuilds a static page on GitHub Pages
   with these sections:
   - the latest status per provider;
   - p50 and p90 turnaround over the last 30 days;
   - turnaround by submission hour;
   - a log of incidents (batches over 6 hours, any missing or broken lines, drift above 5%);
   - drift and accuracy over time;
   - cost.

   The raw `runs.jsonl` and a CSV stay downloadable. Each month a GitHub release snapshots the
   data, so Zenodo gives it a DOI.

The provider interface already splits `submit`, `status` and `results`, which is what this needs.
New code:
- provider timestamps in `status`;
- a `steadybatch-track` command with `submit`, `collect` and `site`;
- the page;
- two workflow files;
- tests on the fake provider.

## Cost and safety

- 1,200 records a day. At the measured ~$0.05 per 1,000 records that is about $0.06 a day,
  roughly **$2 a month**.
- API keys live only in GitHub repository secrets, added by Ankur. The workflows run only on
  schedule and manual dispatch, never on pull requests, so a fork can't reach the secrets.
- A hard cap in code: at most 500 records per batch and 4 batches per provider per day, whatever
  the config says (the config sets 200 and 2). Ankur also sets a monthly spend limit on
  each provider account.
- Synthetic data only, public APIs only.

## Risks and limits

- 200-record batches may queue differently from million-record jobs. Say so on the page; consider
  one weekly 5,000-record batch per provider later.
- GitHub cron can start late; turnaround uses provider timestamps, and the page records actual
  submit times.
- GitHub turns off scheduled workflows after 60 days without repo activity. The bot's data
  commits should count, but check this in the first weeks.
- Provider terms: check that publishing measured performance is allowed for each one before going
  public.

## Plan

1. Build and test locally on the fake provider (Claude, no cost).
2. One real round per provider from this Mac, cents (needs Ankur's OK).
3. Ankur adds the three secrets and spend limits; enable the workflows and Pages.
4. Run quietly for 2 weeks and check the data.
5. Then announce. Any post under Ankur's name needs Ankur's OK.
6. A 30-day write-up becomes new material for the IEEE Software piece or a workshop paper.

## Decisions

- Two slots a day (Ankur, 2026-11-12).
- Same repo as the library, so the page sends traffic to it and builds on the existing DOI.

## What was built (branch `tracker`)

- `BatchStatus` now carries the provider's created, started and ended times, and `RawResult` the
  model version that answered (OpenAI, Anthropic, Gemini, fake).
- `src/steadybatch/tracker.py`: `steadybatch-track submit | collect | site`.
- `src/steadybatch/tracker_site.py`: the static page (no outside libraries, light and dark) plus
  runs.csv and runs.jsonl.
- `tracker/`: config, the frozen 200 inputs, prices, `open.json`, `data/`.
- `.github/workflows/tracker.yml`: schedule and manual runs only; commits data, deploys Pages.
- `tests/test_tracker.py`: 14 tests on the fake provider (drift, lost and broken lines, caps,
  missing keys, refused submits, giving up, flaky status checks, timestamps, model versions).

## Before it goes live

1. Check each provider's terms on publishing measured performance.
2. One real round from this Mac (a few cents).
3. Ankur adds the three repository secrets and a monthly spend limit per provider account.
4. Turn on GitHub Pages (source: GitHub Actions), merge the branch, run the workflow once by hand.
