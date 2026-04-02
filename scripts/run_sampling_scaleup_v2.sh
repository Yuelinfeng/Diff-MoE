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

BEST_TOP_K_EXPERTS="${BEST_TOP_K_EXPERTS:-35,43,0,84,52,0,93,0,0,61,0,35}"
RUN_NAME="${RUN_NAME:-sampling_scaleup_v2_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$DIFF_MOE_LOG_DIR/$RUN_NAME"
SUMMARY_FILE="$RUN_DIR/summary.tsv"

mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/README.txt" <<EOF
Run name: $RUN_NAME
Goal: continue scaling the current best sampling config on 4090D
Checkpoint: $CKPT_DIR
Library: $LIB_PATH
Dataset: $DATASET_NAME
Best top_k_experts: $BEST_TOP_K_EXPERTS
EOF

printf "run_id\tcase_name\tbatch_size\tbeam_width\tseq_len\tmax_samples\tcache_size\tfix_cache_size\tlog_file\n" > "$SUMMARY_FILE"

run_one() {
  local run_id="$1"
  local case_name="$2"
  local batch_size="$3"
  local beam_width="$4"
  local seq_len="$5"

  local stage_dir="$RUN_DIR/$run_id"
  local log_file="$stage_dir/${run_id}.log"

  mkdir -p "$stage_dir"

  export BATCH_SIZE="$batch_size"
  export BEAM_WIDTH="$beam_width"
  export SEQ_LEN="$seq_len"
  export MAX_SAMPLES="$MAX_SAMPLES"
  export CACHE_SIZE="2"
  export USE_MOE_CACHE="True"
  export FIX_CACHE_SIZE="0"
  export TOP_K_EXPERTS="$BEST_TOP_K_EXPERTS"
  export WARMUP_ITERATIONS="$WARMUP_ITERATIONS"
  export ITERATIONS="$ITERATIONS"

  cat > "$stage_dir/notes.txt" <<EOF
run_id=$run_id
case_name=$case_name
batch_size=$batch_size
beam_width=$beam_width
seq_len=$seq_len
max_samples=$MAX_SAMPLES
cache_size=2
fix_cache_size=0
warmup_iterations=$WARMUP_ITERATIONS
iterations=$ITERATIONS
top_k_experts=$BEST_TOP_K_EXPERTS
EOF

  diff_moe_write_cpp_config "$DIFF_MOE_CPP_CONFIG" "$CKPT_DIR" "$CACHE_SIZE" "$USE_MOE_CACHE" "$FIX_CACHE_SIZE" "$TOP_K_EXPERTS"

  (
    cd "$DIFF_MOE_BUILD_DIR"
    diff_moe_run_perf "$log_file"
  )

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$run_id" "$case_name" "$batch_size" "$beam_width" "$seq_len" "$MAX_SAMPLES" "2" "0" "$log_file" \
    >> "$SUMMARY_FILE"
}

run_one "T1" "batch8_seq32" "8" "1" "32"
run_one "T2" "batch4_seq64" "4" "1" "64"

echo "Completed sampling scale-up v2 validation. Summary written to $SUMMARY_FILE"
