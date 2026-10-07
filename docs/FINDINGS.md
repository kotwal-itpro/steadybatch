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

## 4. Same definition, opposite mistakes (OpenAI and Gemini, 2026-10-07)

**What happened.** At 1,000 records with field definitions, both services returned every
line with no retries, and both got `product` and `issue_type` 100% right. On `sentiment`
they missed in opposite directions. gpt-4.1-mini read polite tickets about a problem
("I was charged twice. Thanks.") as negative: 71 misses, all neutral → negative.
gemini-3.5-flash-lite read clear frustration ("This is really frustrating.") as neutral:
158 of its 179 misses were negative → neutral. Our definition says sentiment is the
attitude toward the product and that "the problem they report does not count on its own";
one model under-applied that rule and the other over-applied it.

**Why it matters.** Accuracy alone (93% vs 82%) hides that the two models fail differently.
Switching providers on the same prompt changes which records are wrong, not only how many.

**Measurement note.** Batch turnaround is measured at the polling interval (60 s for these
runs), so a reported 182 s means "finished between the 3rd and 4th poll".

## 5. The Anthropic SDK no longer takes `temperature` (Anthropic, 2026-10-07)

**What happened.** Our Anthropic adapter sent `temperature`, like the other adapters. With
anthropic SDK 1.12, `messages.create()` raised `TypeError: unexpected keyword argument
'temperature'` before any request was sent. The current Claude models reject non-default
sampling settings on the API side too.

**Why it matters.** The reproducibility runs compare repeated runs of the same job. On
gpt-4.1-mini and gemini-3.5-flash-lite we fix `temperature` at 0. For Claude, as for the
OpenAI reasoning models (finding 2), that option no longer exists, so run-to-run agreement
has to be measured as the model ships rather than pinned down. We note this next to every
Claude run.

**What we changed.** The Anthropic adapter no longer sends `temperature`, and it now
constrains replies with structured outputs (`output_config.format`), matching the schema
modes we use on OpenAI and Gemini. A refusal is now counted as a failed line rather than
an empty answer. The package requires `anthropic>=1.0`.

## 6. A small model that decides when to think can run out of room (Anthropic, 2026-10-07)

**What happened.** The first 1,000-record run on `claude-haiku-5-5` used the model's
default settings, with the same `max_tokens` of 300 as every other run. Haiku 5.5 uses
adaptive thinking by default: it decides per request whether to think first. It thought on
91 of 1,000 requests, and on 10 of those the thinking used the whole 300-token budget
before the JSON was finished (`stop_reason: max_tokens`). Six came back with no text at all,
four with JSON cut off mid-string. Retries recovered nine; one failed all three attempts,
so the run finished with 999 of 1,000. It was the first record any provider in this study
failed to deliver.

**Why it matters.** Nothing in the request asked for reasoning, and 300 tokens is about
eight times what an answer needs (the median reply was 40 tokens). The same budget that
is plenty for gpt-4.1-mini is not a safe budget for a model whose default is to think
sometimes. The failures were also uneven: they landed on a few records the model found
harder, so a retry with the same budget did not always help.

**How we noticed, and a bug it exposed.** steadybatch caught every one and retried it,
but labelled them "empty response" or "not valid JSON". The adapter knew the reason
(`max_tokens`); the runner dropped it when validation failed. The runner now keeps the
provider's note, so these show up as `truncated (max_tokens): not valid JSON ...`.

**What we changed.** The Anthropic adapter and `steadybatch-bench run` take
`--thinking disabled|adaptive`, and each run's summary records the thinking setting and
`max_tokens`. We keep the default-settings run as published and add a thinking-disabled
run for the like-for-like comparison with gpt-4.1-mini.

**The thinking-disabled run.** Same 1,000 records, same 300-token budget, with
`--thinking disabled`: 1,000 of 1,000 on the first attempt, no retries, median batch
time 304 s instead of 606 s, and $0.048 per 1,000 records instead of $0.053. Sentiment
accuracy was about the same (91.0% vs 89.1%), but the mistakes changed direction. With
default settings the most common miss was negative → neutral (50), the way Gemini errs;
with thinking off it was neutral → negative (54), the way gpt-4.1-mini errs. The two
Claude runs agreed on sentiment for 88.7% of records. One setting on the same model moved
which records were wrong, not only how many.

## 7. A big job needs to respect the queue limit, and the error doesn't say so (Gemini, 2026-10-07)

**What happened.** For the 100,000-record runs we expected a queue limit on OpenAI, which
counts the input tokens an account has waiting in batches, so we split that job into
10,000-record batches sent one at a time. Gemini got a single 100,000-record job (about
34 million input tokens). The input file took about five minutes to upload, and then
creating the batch failed with `429 RESOURCE_EXHAUSTED: You exceeded your current quota,
please check your plan and billing details`. The same job split into 10,000-record
batches, one at a time, was accepted straight away. Anthropic accepted all 100,000
requests as one batch.

**Why it matters.** The error reads like a billing or rate-limit problem, not "this batch
is too big for your queue", and it only arrives after the whole upload. A pipeline that
retries the same submission on a 429 will keep failing. The fix is to send smaller
batches, a few at a time.

**What steadybatch did, and what we changed.** The adapter deleted the rejected upload
(finding 1), nothing was billed, and the checkpoint still had all 100,000 records as
pending, so the rerun started cleanly. The runner now has `--batch-size` and
`--max-open-batches` to keep a job under these limits. Open question for a later
version: catching a submit-time quota error and shrinking the batch automatically
instead of stopping.
