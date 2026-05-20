#!/bin/bash
# =============================================================================
# GPT-5.5 via weelinking.com proxy — KernelGYM Evaluation
# =============================================================================
# Uses https://api.weelinking.com as the OpenAI-compatible proxy to access
# GPT-5.5.
#
# Prerequisites:
#   export ANTHROPIC_AUTH_TOKEN="<your-key>"
#
# The proxy exposes both Anthropic (/v1/messages) and OpenAI-compatible
# (/v1/chat/completions) endpoints. This script uses the OpenAI-compatible
# path (BACKEND=openai) since the KernelGYM framework supports it natively,
# including multi-turn rollout and thinking-mode token handling.
#
# Output is saved to /mnt/hstorage/GKG/datasets/distill/ for distillation.
# =============================================================================

# --- Proxy fix: OpenAI SDK crashes with SOCKS proxy ---
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/grading_common.sh"

PROJECT_NAME="kernel-grading"
RUN_NAME="gpt-5.5-weelinking"
EXPERIMENT_NAME=${RUN_NAME}

REFERENCE_BACKEND="torch_compile"

HDFS_RUNS_PATH="/mnt/hstorage/GKG/datasets/distill"
# Use 10-row subset for small-scale testing (switch to full dataset for production)
# EVAL_DATASET="/mnt/hstorage/GKG/datasets/structured_datasets/drkernel/drkernel-validation-data/validation_data_thinking.parquet"
EVAL_DATASET="/mnt/hstorage/GKG/datasets/structured_datasets/drkernel/drkernel-validation-data/validation_data_thinking_matmul_precision_mini10.parquet"

MULTI_TURN=True
MAX_USER_TURNS=3

GRADIO_VISUALIZATION=False
GRADIO_SHARE=False
VISUALIZE_ONLY=False

MAX_PROMPT_LENGTH=20480
MAX_RESPONSE_LENGTH=8192

OUTPUT_DIR="${HDFS_RUNS_PATH}/${RUN_NAME}"
OUTPUT_PATH="${OUTPUT_DIR}/graded_results.parquet"
METRICS_OUTPUT_PATH="${OUTPUT_DIR}/metrics.json"
RAW_RESPONSE_PATH="${OUTPUT_DIR}/raw_responses.jsonl"

# --- Model path (only the tokenizer is loaded — not used for inference) ---
ORIGINAL_MODEL="/mnt/hstorage/GKG/pretrained_models/drkernel-8b"
ACTOR_PATH="${ORIGINAL_MODEL}"
HF_MODEL_PATH="${ORIGINAL_MODEL}"
MODEL_NAME="${HF_MODEL_PATH}"
MODEL_PATH="${MODEL_NAME}"

# --- Generation Parameters ---
# For small-scale test: N_SAMPLES=1; for production: N_SAMPLES=8
N_SAMPLES=8
BATCH_SIZE=128
TEMPERATURE=1.0
TOP_P=0.95
DO_SAMPLE=True

# --- Rollout Mode ---
ROLLOUT_MODE="standalone_vllm"
ROLLOUT_GPU_MEMORY_UTIL=0.7
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
FSDP_SIZE=-1
ROLLOUT_ENFORCE_EAGER=True

# --- Evaluation Metrics ---
SOLVE_THRESHOLD=0.99
PASS_AT_K=1

# =============================================================================
# weelinking API Configuration
# =============================================================================
BACKEND="openai"
OPENAI_MODEL="gpt-5.5"

# weelinking uses the proxy token carried in ANTHROPIC_AUTH_TOKEN.
if [[ -z "${ANTHROPIC_AUTH_TOKEN}" ]]; then
  echo "ERROR: ANTHROPIC_AUTH_TOKEN is not set."
  echo "Please run: export ANTHROPIC_AUTH_TOKEN=\"<your-key>\""
  exit 1
fi
OPENAI_API_KEY="${ANTHROPIC_AUTH_TOKEN}"

OPENAI_BASE_URL="https://api.weelinking.com/v1"
OPENAI_TIMEOUT=400
OPENAI_MAX_RETRIES=5
OPENAI_MAX_CONCURRENCY=2
OPENAI_THINKING_MODE=False

# =============================================================================
# Sandbox / Reward Configuration
# =============================================================================
REWARD_SERVER_URL="${REWARD_SERVER_URL:-${KERNELGYM_SERVER_URL:-"http://192.168.31.68:8002"}}"

REWARD_MANAGER="kernel_async"
REWARD_FUNC_NAME="calculate_reward_speedup"

# Reward Weights (compilation, correctness, performance)
REWARD_WEIGHTS="0.3_0.4_0.3"

REWARD_ENHANCED=True
REWARD_USE_SANDBOX_RATE_LIMIT=True
REWARD_RATE_LIMIT=643
REWARD_ACQUIRE_TIMEOUT=2400
REWARD_MAX_CONCURRENT=64
REWARD_TIMEOUT=1800
REWARD_MAX_RETRIES=3
REWARD_TASK_TIMEOUT=1800
REWARD_PRINT_STATUS=True
NUM_PERF_TRIALS=10
NUM_CORRECT_TRIALS=5
SPEEDUP_REWARD_UPPER_BOUND=3.0

# Custom Reward Function
CUSTOM_REWARD_PATH="/mnt/hstorage/GKG/framework/KernelGYM/drkernel/kernel/rewards/kernel_reward.py"
CUSTOM_REWARD_NAME="compute_kernel_reward_batch"

NNODES=1
# OpenAI backend does not need local GPUs — all inference is via API
N_GPUS_PER_NODE=1

FIX_QWEN3_CHAT_TEMPLATE=False

# =============================================================================
# Exports
# =============================================================================
export PROJECT_NAME
export RUN_NAME
export EVAL_DATASET
export OUTPUT_PATH
export METRICS_OUTPUT_PATH
export RAW_RESPONSE_PATH

export MODEL_NAME
export MODEL_PATH

export N_SAMPLES
export BATCH_SIZE
export TEMPERATURE
export TOP_P
export DO_SAMPLE

export ROLLOUT_MODE
export ROLLOUT_GPU_MEMORY_UTIL
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE

export OPENAI_API_KEY
export OPENAI_THINKING_MODE

export SOLVE_THRESHOLD
export PASS_AT_K

export REWARD_SERVER_URL
export REWARD_MANAGER
export REWARD_FUNC_NAME
export REWARD_WEIGHTS

export REWARD_ENHANCED
export REWARD_USE_SANDBOX_RATE_LIMIT
export REWARD_RATE_LIMIT
export REWARD_ACQUIRE_TIMEOUT
export REWARD_MAX_CONCURRENT
export REWARD_TIMEOUT
export REWARD_MAX_RETRIES
export REWARD_TASK_TIMEOUT
export REWARD_PRINT_STATUS
export NUM_PERF_TRIALS
export NUM_CORRECT_TRIALS
export SPEEDUP_REWARD_UPPER_BOUND

export CUSTOM_REWARD_PATH
export CUSTOM_REWARD_NAME

export NNODES
export N_GPUS_PER_NODE
export FIX_QWEN3_CHAT_TEMPLATE

# Create output directory
mkdir -p "$OUTPUT_DIR"

main "$@"
