#!/usr/bin/env bash
# Self-hosted comparison for the Batch API Tracker: send the tracker's own inputs (200, then the
# weekly 5,000) through SGLang's offline engine on a rented GPU, and record time and GPU cost.
# Run on a Linux + NVIDIA machine (e.g. a RunPod pod). Each run appends one row to
# tracker/data/selfhost.jsonl; copy that file back (or commit it) and the page shows it.
# Usage:  bash run_tracker_selfhost.sh "<gpu name>" <usd per hour> [model]
#   e.g.  bash run_tracker_selfhost.sh "1x H100 80GB (RunPod)" 2.69 Qwen/Qwen2.5-7B-Instruct
set -euo pipefail
GPU="${1:?say which GPU, e.g. \"1x H100 80GB (RunPod)\"}"
PRICE="${2:?give the GPU price per hour in USD}"
MODEL="${3:-Qwen/Qwen2.5-7B-Instruct}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -d steadybatch ]; then git clone -q https://github.com/kotwal-itpro/steadybatch.git; fi
cd steadybatch
python3 -m venv .venv && . .venv/bin/activate
pip install -q --upgrade pip && pip install -q -e ".[sglang]"

# A fresh process per run, so each one pays (and records) the engine start-up honestly.
for N in 200 5000; do
  steadybatch-track selfhost --engine sglang --model "$MODEL" --gpu "$GPU" \
    --gpu-usd-per-hour "$PRICE" --records "$N" --note "temperature 0, max_tokens 300, JSON schema"
done
echo "Rows added to $(pwd)/tracker/data/selfhost.jsonl"
