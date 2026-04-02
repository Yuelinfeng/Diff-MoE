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
DATA_TYPE="${DATA_TYPE:-fp32}"
TEST_TIME="${TEST_TIME:-3}"
SAMPLING_TOPK="${SAMPLING_TOPK:-1}"
MODEL_TYPE="${MODEL_TYPE:-Megatron-DeepSpeed}"
MODEL_NAME_FOR_BENCH="${MODEL_NAME_FOR_BENCH:-t5-base}"
DATASET_NAME="${DATASET_NAME:-EdinburghNLP/xsum}"
ITERATIONS="${ITERATIONS:-1}"
DURATION="${DURATION:-0}"
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-1}"
LAYER_NUM="${LAYER_NUM:-6}"

REAL_TOP_K_EXPERTS="${REAL_TOP_K_EXPERTS:-35,43,38,0,84,51,52,0,33,93,0,72,0,61,38,0,35,105}"
ZERO_TOP_K_EXPERTS_1="${ZERO_TOP_K_EXPERTS_1:-0,0,0,0,0,0}"
ZERO_TOP_K_EXPERTS_3="${ZERO_TOP_K_EXPERTS_3:-0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}"

RUN_NAME="${RUN_NAME:-staged_sweep_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$DIFF_MOE_LOG_DIR/$RUN_NAME"
SUMMARY_FILE="$RUN_DIR/summary.tsv"

mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/README.txt" <<EOF
Run name: $RUN_NAME
Log root: $RUN_DIR
Checkpoint: $CKPT_DIR
Library: $LIB_PATH
Goal: staged enlargement from smoke baseline to diffmoe config
EOF

printf "stage\tseq_len\tmax_samples\tcache_size\tuse_moe_cache\tfix_cache_size\ttop_k_experts\tlog_file\n" > "$SUMMARY_FILE"

run_stage() {
  local stage_name="$1"
  local stage_desc="$2"
  local seq_len="$3"
  local max_samples="$4"
  local cache_size="$5"
  local use_moe_cache="$6"
  local fix_cache_size="$7"
  local top_k_experts="$8"

  local stage_dir="$RUN_DIR/$stage_name"
  local log_file="$stage_dir/${stage_name}.log"

  mkdir -p "$stage_dir"
  cat > "$stage_dir/notes.txt" <<EOF
$stage_desc
seq_len=$seq_len
max_samples=$max_samples
cache_size=$cache_size
use_moe_cache=$use_moe_cache
fix_cache_size=$fix_cache_size
top_k_experts=$top_k_experts
EOF

  export SEQ_LEN="$seq_len"
  export MAX_SAMPLES="$max_samples"
  export CACHE_SIZE="$cache_size"
  export USE_MOE_CACHE="$use_moe_cache"
  export FIX_CACHE_SIZE="$fix_cache_size"
  export TOP_K_EXPERTS="$top_k_experts"

  diff_moe_write_cpp_config "$DIFF_MOE_CPP_CONFIG" "$CKPT_DIR" "$CACHE_SIZE" "$USE_MOE_CACHE" "$FIX_CACHE_SIZE" "$TOP_K_EXPERTS"

  (
    cd "$DIFF_MOE_BUILD_DIR"
    diff_moe_run_perf "$log_file"
  )

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$stage_name" "$seq_len" "$max_samples" "$cache_size" "$use_moe_cache" "$fix_cache_size" "$top_k_experts" "$log_file" \
    >> "$SUMMARY_FILE"
}

run_stage \
  "stage1_seq8_samples1_cacheoff" \
  "Smoke baseline. Keep cache disabled and verify the minimal path still runs." \
  "8" "1" "1" "False" "0" "$ZERO_TOP_K_EXPERTS_1"

run_stage \
  "stage2_seq32_samples1_cacheoff" \
  "Increase sequence length only from 8 to 32 while keeping the rest identical." \
  "32" "1" "1" "False" "0" "$ZERO_TOP_K_EXPERTS_1"

run_stage \
  "stage3_seq32_samples8_cacheoff" \
  "Increase sample count from 1 to 8 to validate the longer steady-state baseline." \
  "32" "8" "1" "False" "0" "$ZERO_TOP_K_EXPERTS_1"

run_stage \
  "stage4_seq32_samples8_cacheon_cache1" \
  "Enable moe cache with the smallest cache size first, still using dummy experts." \
  "32" "8" "1" "True" "0" "$ZERO_TOP_K_EXPERTS_1"

run_stage \
  "stage5_seq32_samples8_cacheon_cache3_dummy" \
  "Increase cache_size from 1 to 3 while keeping dummy experts to isolate cache-capacity effects." \
  "32" "8" "3" "True" "1" "$ZERO_TOP_K_EXPERTS_3"

run_stage \
  "stage6_seq32_samples8_cacheon_cache3_realexperts" \
  "Restore the real top_k_experts configuration for the full diffmoe run." \
  "32" "8" "3" "True" "1" "$REAL_TOP_K_EXPERTS"

echo "Completed staged sweep. Summary written to $SUMMARY_FILE"
