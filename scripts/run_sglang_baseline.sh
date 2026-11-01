#!/usr/bin/env bash
# Run the self-hosted baseline with SGLang's offline engine on a rented GPU machine
# (Linux + NVIDIA, CUDA drivers installed). Same data, schema and settings as
# run_vllm_baseline.sh, so the two engines can be compared record by record.
# Usage:  bash run_sglang_baseline.sh [model] [records]
#   model    default Qwen/Qwen2.5-7B-Instruct (Apache-2.0, no license gate)
#   records  default 1000
# Prints the run summary; copy runs/<name>/ back to your machine to publish it.
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
N="${2:-1000}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -d steadybatch ]; then git clone -q https://github.com/kotwal-itpro/steadybatch.git; fi
cd steadybatch
python3 -m venv .venv && . .venv/bin/activate
pip install -q --upgrade pip && pip install -q -e ".[sglang]"
ROWS=$(( N > 10000 ? N : 10000 ))  # same seed as the hosted runs, so the files match theirs
python examples/make_synthetic.py --n "$ROWS" > "examples/support_tickets_$ROWS.jsonl"

NAME="$(date +%Y-%m-%d)-sglang-$(basename "$MODEL" | tr 'A-Z' 'a-z')-$N-r1"
steadybatch-bench run --provider sglang --model "$MODEL" \
  --data "examples/support_tickets_$ROWS.jsonl" --schema examples/support_ticket.schema.json \
  --instructions examples/support_ticket.instructions.txt \
  --limit "$N" --max-tokens 300 --poll-every 0 --out "runs/$NAME"
echo "Done: runs/$NAME  (record the GPU type and its hourly price with the result)"
