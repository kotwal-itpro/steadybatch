# Findings log

Things we ran into while running real batch jobs, in the order we found them.
Each entry says what happened, how we noticed, and what steadybatch does about it.

## 1. A listed model can be closed to new accounts (Gemini, 2026-10-07)

**What happened.** `gemini-2.5-flash-lite` appears in the Gemini API's model list, with
`generateContent` among its supported actions. Creating a batch job with it on a new
account failed with `404 NOT_FOUND`: "This model models/gemini-2.5-flash-lite is no
longer available to new users. Please update your code to use
models/gemini-3.5-flash-lite."

**Why it matters.** The model list said yes; the batch endpoint said no. A pipeline that
picks a model from the list, or that worked on an older account, fails only at submit time.

**What we changed.** The input file is uploaded before the job is created, so the rejected
job left an orphaned upload behind. Both the Gemini and OpenAI adapters now delete the
uploaded input when job creation fails. Benchmark runs use `gemini-3.5-flash-lite`.

## 2. Reasoning models take different parameters (OpenAI, 2026-10-07)

**What happened.** GPT-5 and the o-series models reject `max_tokens` and any
non-default `temperature`; they take `max_completion_tokens` instead.

**What we changed.** The OpenAI adapter now sends `max_completion_tokens` and leaves out
`temperature` for those models. Note for the study: a fixed `temperature` of 0 is not
available on these models, which matters for the reproducibility measurements.

## 3. Undefined fields look like model errors (our test design, 2026-10-07)

**What happened.** The first OpenAI run (gpt-4.1-mini, 100 records) returned every line,
all valid JSON, but field accuracy was 64% for `sentiment` and 31% for `needs_follow_up`,
against 100% for `product` and `issue_type`. Looking at the errors, the model was not wrong
so much as answering a different question: with no definitions, it marked every ticket as
needing follow-up (an open issue needs one) and read "I was charged twice. Thanks." as
negative, while our labels meant "the customer explicitly asked to be contacted" and
"the customer's attitude toward the product".

**What we changed.** Field definitions now go in `examples/support_ticket.instructions.txt`
and are passed with `--instructions`. Runs without them are kept but marked as v1 and are
not used for accuracy comparisons.
