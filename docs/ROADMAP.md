# Roadmap

## Next
- **AWS Bedrock batch inference adapter.** Input and output go through S3, so this one also needs a small S3 helper.
- **Live checks.** Run the OpenAI, Gemini and Anthropic adapters against the real services at 1,000 records and note anything the docs don't mention.

## Then
- Benchmark runs at 1,000, 100,000 and 1 million records, three times each on different days.
- Publish the raw results and a write-up.
- A command-line runner for production jobs (not just benchmarks).
- Optional async polling for many open batches at once.

## Ideas
- Per-provider notes on undocumented limits, as we find them.
- Cost guardrails: stop submitting once a spending cap is reached.
