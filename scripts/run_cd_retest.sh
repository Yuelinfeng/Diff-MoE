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
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"
LAYER_NUM="${LAYER_NUM:-6}"

REAL_TOP_K_EXPERTS="${REAL_TOP_K_EXPERTS:-35,43,0,84,52,0,93,0,0,61,0,35}"
RUN_NAME="${RUN_NAME:-cd_retest_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$DIFF_MOE_LOG_DIR/$RUN_NAME"
SUMMARY_FILE="$RUN_DIR/summary.tsv"

mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/README.txt" <<EOF
Run name: $RUN_NAME
Goal: retest C (cache_size=2, fix_cache_size=0) vs D (cache_size=2, fix_cache_size=1)
Checkpoint: $CKPT_DIR
Library: $LIB_PATH
Dataset: $DATASET_NAME
Shared top_k_experts: $REAL_TOP_K_EXPERTS
EOF

printf "run_id\tvariant\tcache_size\tfix_cache_size\tmax_samples\tseq_len\tlog_file\n" > "$SUMMARY_FILE"

run_one() {
  local run_id="$1"
  local variant="$2"
  local cache_size="$3"
  local fix_cache_size="$4"

  local stage_dir="$RUN_DIR/$run_id"
  local log_file="$stage_dir/${run_id}.log"

  mkdir -p "$stage_dir"

  export CACHE_SIZE="$cache_size"
  export USE_MOE_CACHE="True"
  export FIX_CACHE_SIZE="$fix_cache_size"
  export TOP_K_EXPERTS="$REAL_TOP_K_EXPERTS"
  export MAX_SAMPLES="$MAX_SAMPLES"
  export SEQ_LEN="$SEQ_LEN"
  export WARMUP_ITERATIONS="$WARMUP_ITERATIONS"
  export ITERATIONS="$ITERATIONS"
  export BATCH_SIZE="$BATCH_SIZE"
  export BEAM_WIDTH="$BEAM_WIDTH"

  cat > "$stage_dir/notes.txt" <<EOF
run_id=$run_id
variant=$variant
cache_size=$cache_size
fix_cache_size=$fix_cache_size
max_samples=$MAX_SAMPLES
seq_len=$SEQ_LEN
warmup_iterations=$WARMUP_ITERATIONS
iterations=$ITERATIONS
batch_size=$BATCH_SIZE
beam_width=$BEAM_WIDTH
top_k_experts=$REAL_TOP_K_EXPERTS
EOF

  diff_moe_write_cpp_config "$DIFF_MOE_CPP_CONFIG" "$CKPT_DIR" "$CACHE_SIZE" "$USE_MOE_CACHE" "$FIX_CACHE_SIZE" "$TOP_K_EXPERTS"

  (
    cd "$DIFF_MOE_BUILD_DIR"
    diff_moe_run_perf "$log_file"
  )

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$run_id" "$variant" "$cache_size" "$fix_cache_size" "$MAX_SAMPLES" "$SEQ_LEN" "$log_file" \
    >> "$SUMMARY_FILE"
}

run_one "C1" "C" "2" "0"
run_one "D1" "D" "2" "1"
run_one "C2" "C" "2" "0"
run_one "D2" "D" "2" "1"
run_one "C3" "C" "2" "0"
run_one "D3" "D" "2" "1"

echo "Completed C vs D retest. Summary written to $SUMMARY_FILE"
