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
BEAM_WIDTH="${BEAM_WIDTH:-4}"
SEQ_LEN="${SEQ_LEN:-32}"
DATA_TYPE="${DATA_TYPE:-fp32}"
TEST_TIME="${TEST_TIME:-1}"
SAMPLING_TOPK="${SAMPLING_TOPK:-1}"
MODEL_TYPE="${MODEL_TYPE:-Megatron-DeepSpeed}"
MODEL_NAME_FOR_BENCH="${MODEL_NAME_FOR_BENCH:-t5-base}"
DATASET_NAME="${DATASET_NAME:-EdinburghNLP/xsum}"
ITERATIONS="${ITERATIONS:-1}"
DURATION="${DURATION:-0}"
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"
LAYER_NUM="${LAYER_NUM:-6}"
CACHE_SIZE="${CACHE_SIZE:-2}"
USE_MOE_CACHE="${USE_MOE_CACHE:-True}"
FIX_CACHE_SIZE="${FIX_CACHE_SIZE:-0}"
TOP_K_EXPERTS="${TOP_K_EXPERTS:-35,43,0,84,52,0,93,0,0,61,0,35}"

RUN_NAME="${RUN_NAME:-ft_beamsearch_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$DIFF_MOE_LOG_DIR/$RUN_NAME"
SUMMARY_FILE="$RUN_DIR/summary.tsv"
LOG_FILE="$RUN_DIR/ft_beamsearch.log"

mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/README.txt" <<EOF
Run name: $RUN_NAME
Goal: run true FT beamsearch with the current best cache config
Checkpoint: $CKPT_DIR
Library: $LIB_PATH
Dataset: $DATASET_NAME
Cache config: cache_size=$CACHE_SIZE, use_moe_cache=$USE_MOE_CACHE, fix_cache_size=$FIX_CACHE_SIZE
Top-k experts: $TOP_K_EXPERTS
EOF

printf "case_name\tbatch_size\tbeam_width\tseq_len\tmax_samples\tcache_size\tfix_cache_size\tlog_file\n" > "$SUMMARY_FILE"

cat > "$RUN_DIR/notes.txt" <<EOF
case_name=ft_beamsearch
batch_size=$BATCH_SIZE
beam_width=$BEAM_WIDTH
seq_len=$SEQ_LEN
max_samples=$MAX_SAMPLES
cache_size=$CACHE_SIZE
use_moe_cache=$USE_MOE_CACHE
fix_cache_size=$FIX_CACHE_SIZE
warmup_iterations=$WARMUP_ITERATIONS
iterations=$ITERATIONS
test_time=$TEST_TIME
top_k_experts=$TOP_K_EXPERTS
EOF

diff_moe_write_cpp_config "$DIFF_MOE_CPP_CONFIG" "$CKPT_DIR" "$CACHE_SIZE" "$USE_MOE_CACHE" "$FIX_CACHE_SIZE" "$TOP_K_EXPERTS"

export BATCH_SIZE
export BEAM_WIDTH
export SEQ_LEN
export DATA_TYPE
export TEST_TIME
export SAMPLING_TOPK
export MODEL_TYPE
export MODEL_NAME_FOR_BENCH
export DATASET_NAME
export ITERATIONS
export DURATION
export WARMUP_ITERATIONS
export MAX_SAMPLES
export LAYER_NUM
export CACHE_SIZE
export USE_MOE_CACHE
export FIX_CACHE_SIZE
export TOP_K_EXPERTS

(
  cd "$DIFF_MOE_BUILD_DIR"
  diff_moe_run_perf "$LOG_FILE"
)

printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "ft_beamsearch" "$BATCH_SIZE" "$BEAM_WIDTH" "$SEQ_LEN" "$MAX_SAMPLES" "$CACHE_SIZE" "$FIX_CACHE_SIZE" "$LOG_FILE" \
  >> "$SUMMARY_FILE"

echo "Completed FT beamsearch run. Summary written to $SUMMARY_FILE"
