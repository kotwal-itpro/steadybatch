# Running a big batch job you can trust

This guide is the checklist we wish we'd had before the benchmark. Each step comes from something that went wrong, or nearly did, in a real run. The finding numbers point to [FINDINGS.md](FINDINGS.md).

## 1. Write down what every field means

Before you send a single record, write one plain sentence for each field you want back. Say what counts and what doesn't.

```text
- needs_follow_up: true only if the customer explicitly asks to be contacted again
  (for example asks for a call back or an email). Otherwise false.
- sentiment: the customer's attitude toward the product itself. The problem they
  report does not count on its own.
```

Without these, both models in our first run (gpt-4.1-mini and gemini-3.5-flash-lite) scored 31% on `needs_follow_up`, because they all read it as "does this ticket need more work?" With one sentence each, they scored 98-100% (finding 3). If a field scores badly, check the definition before you blame the model.

Pass the definitions with `--instructions` (benchmark) or as `Request.system` (library).

## 2. Ask for a schema, and check it anyway

Give a JSON Schema with `enum` values wherever the answer is a fixed set. steadybatch sends it as each provider's structured-output format and also validates every reply against it, because a reply can still be cut off or refused (findings 5 and 6).

## 3. Try 100 records first

Run 100 records, then read the wrong answers, not just the score. A small run is cheap and catches:

- a model that's listed but refused for batch on your account (finding 1)
- fields the model reads differently than you meant (finding 3)
- replies that run out of tokens (finding 6)

```bash
steadybatch-bench run --provider openai --model gpt-4.1-mini \
  --data my_records.jsonl --schema my.schema.json --instructions my_fields.txt \
  --limit 100 --out runs/try-100
```

## 4. Give the reply room, and know whether the model thinks

Set `max_tokens` well above what a correct answer needs. Then find out whether your model "thinks" before answering, because thinking counts against the same budget.

- Claude Haiku 5.5 decides for itself when to think. With a 300-token limit it ran out of room on 1% of records and returned nothing (finding 6). `--thinking disabled` fixed that, halved the batch time and cut the cost.
- Reasoning models on OpenAI take `max_completion_tokens` and no `temperature` (finding 2). The adapter handles this for you.
- Newer Claude models don't take `temperature` at all (finding 5), so you can't pin their answers down for repeat runs.

## 5. Size the job to the provider's queue

Providers limit how much work one account can have waiting. On OpenAI it's queued input tokens. Gemini returned a generic "429 quota exceeded" for a 100,000-record batch that went through fine in 10,000-record pieces (finding 7).

```bash
--batch-size 10000 --max-open-batches 1
```

If a provider still says the queue is full, the runner waits for its own batches or halves the batch size, and the refused records don't use up their retries.

## 6. Always use a checkpoint

```python
runner = Runner(provider, model, response_schema=schema, checkpoint="job.sqlite")
```

Batches can take hours. If your process dies, run the same script again: it collects the batches that are already open instead of paying for them twice, and only sends what's left. The benchmark CLI puts a checkpoint in every `--out` folder automatically.

## 7. Check every line off, then retry only the failures

Don't trust a batch that says "completed". steadybatch gives every record a stable ID, compares what you sent with what came back, and sorts each record into ok, missing, error or invalid. Only the bad records go back, never the whole batch, up to `max_attempts` times. The run summary tells you how many needed a retry and why.

## 8. Before switching models, compare the mistakes

Two models with similar scores can be wrong about different records. In our runs, OpenAI and the self-hosted Qwen model read polite complaints as negative, Gemini read real frustration as neutral, and one Claude setting moved it from balanced mistakes to mostly harsh ones. If something downstream counts the answers (unhappy customers per week, say), a model switch moves the count even when accuracy looks the same.

```bash
steadybatch-bench compare runs/model-a runs/model-b   # agreement, field by field
```

## 9. Run the same job twice

Run the same records again on another day and compare. It tells you how much your numbers move when nothing in your data changed.

## 10. Self-hosting: what it takes

`scripts/run_vllm_baseline.sh` runs the same benchmark on one rented GPU with vLLM. On an NVIDIA L4 ($0.50/hour on demand), Qwen2.5-7B-Instruct did 1,000 records in 29 seconds, about $0.004 per 1,000 records of GPU time. The first run also spent about 8 minutes installing, downloading the model and compiling, which cost more than the run itself. Copy the `runs/` folder off the machine before you shut it down: the disk is erased.

## Record what you ran

Prices and models change. Keep the date, model name, thinking setting, `max_tokens`, batch size and the prices you used with every result. The run summary saves most of these for you, and `--prices` takes a dated price file.
