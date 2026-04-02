#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/autodl_env.sh"

CKPT_DIR="${CKPT_DIR:-$DIFF_MOE_CKPT_DIR}"
LIB_PATH="${LIB_PATH:-$DIFF_MOE_LIB_PATH}"

if [[ ! -f "$LIB_PATH" ]]; then
  echo "Missing $LIB_PATH"
  exit 1
fi

if [[ ! -f "$CKPT_DIR/config.ini" ]]; then
  echo "Missing $CKPT_DIR/config.ini"
  exit 1
fi

BATCH_SIZE="${BATCH_SIZE:-1}"
BEAM_WIDTH="${BEAM_WIDTH:-1}"
SEQ_LEN="${SEQ_LEN:-32}"
DATA_TYPE="${DATA_TYPE:-fp32}"
TEST_TIME="${TEST_TIME:-3}"
SAMPLING_TOPK="${SAMPLING_TOPK:-1}"
MODEL_TYPE="${MODEL_TYPE:-Megatron-DeepSpeed}"
MODEL_NAME_FOR_BENCH="${MODEL_NAME_FOR_BENCH:-t5-base}"
DATASET_NAME="${DATASET_NAME:-EdinburghNLP/xsum}"
ITERATIONS="${ITERATIONS:-1}"
DURATION="${DURATION:-0}"
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
LAYER_NUM="${LAYER_NUM:-6}"
CACHE_SIZE="${CACHE_SIZE:-3}"
USE_MOE_CACHE="${USE_MOE_CACHE:-True}"
FIX_CACHE_SIZE="${FIX_CACHE_SIZE:-1}"
TOP_K_EXPERTS="${TOP_K_EXPERTS:-35,43,38,0,84,51,52,0,33,93,0,72,0,61,38,0,35,105}"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_FILE:-$DIFF_MOE_LOG_DIR/run_diffmoe_${timestamp}.log}"

diff_moe_write_cpp_config "$DIFF_MOE_CPP_CONFIG" "$CKPT_DIR" "$CACHE_SIZE" "$USE_MOE_CACHE" "$FIX_CACHE_SIZE" "$TOP_K_EXPERTS"

cd "$DIFF_MOE_BUILD_DIR"
diff_moe_run_perf "$log_file"
