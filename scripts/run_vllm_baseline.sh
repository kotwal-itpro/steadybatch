#!/usr/bin/env bash
# Run the self-hosted vLLM baseline on a rented GPU machine (Linux + NVIDIA, CUDA drivers installed).
# Usage:  bash run_vllm_baseline.sh [model] [records]
#   model    default Qwen/Qwen2.5-7B-Instruct (Apache-2.0, no license gate)
#   records  default 1000 (the 100,000-record run used 100000)
# Prints the run summary; copy runs/<name>/ back to your machine to publish it.
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
N="${2:-1000}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -d steadybatch ]; then git clone -q https://github.com/kotwal-itpro/steadybatch.git; fi
cd steadybatch
python3 -m venv .venv && . .venv/bin/activate
pip install -q --upgrade pip && pip install -q -e ".[vllm]"
ROWS=$(( N > 10000 ? N : 10000 ))  # same seed as the hosted runs, so the 100k file matches theirs
python examples/make_synthetic.py --n "$ROWS" > "examples/support_tickets_$ROWS.jsonl"

# FlashInfer's sampler is compiled on first use and the build failed on a RunPod L4 in
# Oct 2026 (FINDINGS #11); vLLM's built-in sampler avoids it.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
NAME="$(date +%Y-%m-%d)-vllm-$(basename "$MODEL" | tr 'A-Z' 'a-z')-$N-r1"
steadybatch-bench run --provider vllm --model "$MODEL" \
  --data "examples/support_tickets_$ROWS.jsonl" --schema examples/support_ticket.schema.json \
  --instructions examples/support_ticket.instructions.txt \
  --limit "$N" --max-tokens 300 --poll-every 0 --out "runs/$NAME"
echo "Done: runs/$NAME  (record the GPU type and its hourly price with the result)"
