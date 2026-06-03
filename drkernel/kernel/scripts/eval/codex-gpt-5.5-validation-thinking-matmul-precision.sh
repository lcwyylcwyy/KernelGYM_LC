#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/chen/NVS/KernelGYM_LC}"
cd "$REPO_ROOT"

RUN_DATE="${RUN_DATE:-$(date +%F)}"
RUN_SUFFIX="${RUN_SUFFIX:-notes-full}"

DATASET="${DATASET:-data/drkernel-validation-data/validation_data_thinking_matmul_precision.parquet}"
SERVER_URL="${KERNELGYM_SERVER_URL:-${SERVER_URL:-http://172.17.2.155:8002}}"
OUTPUT_DIR="${OUTPUT_DIR:-drkernel/kernel/scripts/eval/codex-gpt-5.5-validation-thinking-matmul-precision-${RUN_DATE}-${RUN_SUFFIX}}"

MODEL="${MODEL:-gpt-5.5}"
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
MAX_CODEX_WORKERS="${MAX_CODEX_WORKERS:-1}"
MAX_EVAL_WORKERS="${MAX_EVAL_WORKERS:-1}"
NUM_TURNS="${NUM_TURNS:-3}"
NUM_CORRECT_TRIALS="${NUM_CORRECT_TRIALS:-5}"
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-10}"
CODEX_TIMEOUT="${CODEX_TIMEOUT:-1800}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"

args=(
  --dataset "$DATASET"
  --server-url "$SERVER_URL"
  --output-dir "$OUTPUT_DIR"
  --model "$MODEL"
  --reasoning-effort "$REASONING_EFFORT"
  --max-codex-workers "$MAX_CODEX_WORKERS"
  --max-eval-workers "$MAX_EVAL_WORKERS"
  --num-turns "$NUM_TURNS"
  --num-correct-trials "$NUM_CORRECT_TRIALS"
  --num-perf-trials "$NUM_PERF_TRIALS"
  --codex-timeout "$CODEX_TIMEOUT"
  --eval-timeout "$EVAL_TIMEOUT"
)

if [[ "${ENABLE_NCU_PROFILING:-0}" == "1" ]]; then
  args+=(--enable-ncu-profiling)
fi

if [[ "${RESUME:-0}" == "1" ]]; then
  args+=(--resume)
fi

exec python valid_codex_gpt55/run_codex_kernelbench.py "${args[@]}" "$@"
