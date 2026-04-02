#!/usr/bin/env bash
set -euo pipefail

export DIFF_MOE_ROOT="${DIFF_MOE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DIFF_MOE_DATA_DIR="${DIFF_MOE_DATA_DIR:-$HOME/autodl-tmp/diff-moe-data}"
export DIFF_MOE_BUILD_DIR="${DIFF_MOE_BUILD_DIR:-$DIFF_MOE_ROOT/build}"
export DIFF_MOE_LOG_DIR="${DIFF_MOE_LOG_DIR:-$HOME/autodl-tmp/Diff-MoE/logs}"
export DIFF_MOE_CPP_CONFIG="${DIFF_MOE_CPP_CONFIG:-/workspace/FasterTransformer/cpp_config.ini}"
export DIFF_MOE_MODEL_NAME="${DIFF_MOE_MODEL_NAME:-switch-base-128}"
export DIFF_MOE_MODEL_ID="${DIFF_MOE_MODEL_ID:-google/$DIFF_MOE_MODEL_NAME}"
export DIFF_MOE_CKPT_DIR="${DIFF_MOE_CKPT_DIR:-$DIFF_MOE_DATA_DIR/ft/$DIFF_MOE_MODEL_NAME}"
export DIFF_MOE_LIB_PATH="${DIFF_MOE_LIB_PATH:-$DIFF_MOE_BUILD_DIR/lib/libth_transformer.so}"

export HF_HOME="${HF_HOME:-$HOME/autodl-tmp/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$DIFF_MOE_DATA_DIR"
mkdir -p "$DIFF_MOE_DATA_DIR/ft"
mkdir -p "$DIFF_MOE_LOG_DIR"
mkdir -p "$DIFF_MOE_BUILD_DIR"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
mkdir -p "$(dirname "$DIFF_MOE_CPP_CONFIG")"

echo "DIFF_MOE_ROOT=$DIFF_MOE_ROOT"
echo "DIFF_MOE_DATA_DIR=$DIFF_MOE_DATA_DIR"
echo "HF_HOME=$HF_HOME"
echo "HF_ENDPOINT=$HF_ENDPOINT"

diff_moe_normalize_bool() {
  local value="${1:-False}"
  case "${value,,}" in
    1|true|t|yes|y|on) echo "True" ;;
    0|false|f|no|n|off) echo "False" ;;
    *)
      echo "Invalid boolean value: $value" >&2
      return 1
      ;;
  esac
}

diff_moe_flat_experts_to_cpp() {
  local flat="${1:-}"
  local cache_size="${2:-0}"

  if [[ -z "${flat// }" || "$cache_size" -le 0 ]]; then
    echo "[]"
    return 0
  fi

  local sanitized="${flat// /}"
  local IFS=','
  read -r -a values <<< "$sanitized"
  local total="${#values[@]}"

  if (( total % cache_size != 0 )); then
    echo "top_k_experts length ($total) must be a multiple of cache_size ($cache_size)" >&2
    return 1
  fi

  local rows=$(( total / cache_size ))
  local out="["
  local idx=0

  for ((r = 0; r < rows; r++)); do
    out+="["
    for ((c = 0; c < cache_size; c++)); do
      out+="${values[idx]}"
      idx=$((idx + 1))
      if (( c + 1 < cache_size )); then
        out+=","
      fi
    done
    out+="]"
    if (( r + 1 < rows )); then
      out+=","
    fi
  done

  out+="]"
  echo "$out"
}

diff_moe_write_cpp_config() {
  local cpp_config_path="${1:-$DIFF_MOE_CPP_CONFIG}"
  local offload_path="${2:-$DIFF_MOE_CKPT_DIR}"
  local cache_size="${3:?cache_size is required}"
  local use_moe_cache="${4:?use_moe_cache is required}"
  local fix_cache_size="${5:?fix_cache_size is required}"
  local top_k_experts_flat="${6:-}"

  local normalized_use_moe_cache
  normalized_use_moe_cache="$(diff_moe_normalize_bool "$use_moe_cache")"

  local top_k_experts_cpp
  top_k_experts_cpp="$(diff_moe_flat_experts_to_cpp "$top_k_experts_flat" "$cache_size")"

  mkdir -p "$(dirname "$cpp_config_path")"

  cat > "$cpp_config_path" <<EOF
[default]
arena_size = ${CPP_ARENA_SIZE:-7247757312}
encoder_fetcher_mode = ${CPP_ENCODER_FETCHER_MODE:-1}
decoder_fetcher_mode = ${CPP_DECODER_FETCHER_MODE:-2}
profiling = ${CPP_PROFILING:-1}
detailed_timing = ${CPP_DETAILED_TIMING:-0}
offload_path = ${offload_path%/}/
load_from_cpp = ${CPP_LOAD_FROM_CPP:-1}
quant_mode = ${CPP_QUANT_MODE:-0}
vocab_size = ${CPP_VOCAB_SIZE:-32128}
cache_policy = ${CPP_CACHE_POLICY:-LFU}
cache_size = ${cache_size}
use_moe_cache = ${normalized_use_moe_cache}
fix_cache_size = ${fix_cache_size}
max_val = ${CPP_MAX_VAL:-2}
threshold = ${CPP_THRESHOLD:-1.0}
dec_in_cache = ${CPP_DEC_IN_CACHE:-0.4}
dec_out_cache = ${CPP_DEC_OUT_CACHE:-0.2}
top_k_experts = ${top_k_experts_cpp}
EOF

  echo "Wrote cpp_config.ini to $cpp_config_path"
}

diff_moe_run_perf() {
  local log_file="${1:-}"

  local -a cmd=(
    python "$DIFF_MOE_ROOT/examples/pytorch/t5/perf_benchmark.py"
    --batch_size "${BATCH_SIZE:-1}"
    --beam_width "${BEAM_WIDTH:-1}"
    --seq_len "${SEQ_LEN:-8}"
    --data_type "${DATA_TYPE:-fp32}"
    --test_time "${TEST_TIME:-3}"
    --sampling_topk "${SAMPLING_TOPK:-1}"
    --model_type "${MODEL_TYPE:-Megatron-DeepSpeed}"
    --ckpt_path "${CKPT_DIR:-$DIFF_MOE_CKPT_DIR}"
    --model "${MODEL_NAME_FOR_BENCH:-t5-base}"
    --dataset_name "${DATASET_NAME:-EdinburghNLP/xsum}"
    --iterations "${ITERATIONS:-1}"
    --duration "${DURATION:-0}"
    --warmup_iterations "${WARMUP_ITERATIONS:-1}"
    --skip_gemm
    --lib_path "${LIB_PATH:-$DIFF_MOE_LIB_PATH}"
    --cache_size "${CACHE_SIZE:?CACHE_SIZE is required}"
    --use_moe_cache "${USE_MOE_CACHE:?USE_MOE_CACHE is required}"
    --fix_cache_size "${FIX_CACHE_SIZE:?FIX_CACHE_SIZE is required}"
    --top_k_experts "${TOP_K_EXPERTS:-}"
    --layer_num "${LAYER_NUM:-6}"
    --max_samples "${MAX_SAMPLES:-1}"
  )

  printf 'Running command:\n'
  printf '  %q' "${cmd[@]}"
  printf '\n'

  if [[ -n "$log_file" ]]; then
    mkdir -p "$(dirname "$log_file")"
    {
      printf '%q' "${cmd[0]}"
      for ((i = 1; i < ${#cmd[@]}; i++)); do
        printf ' %q' "${cmd[i]}"
      done
      printf '\n'
    } > "${log_file%.log}.cmd"
    cp "$DIFF_MOE_CPP_CONFIG" "${log_file%.log}.ini"
    cat > "${log_file%.log}.env" <<EOF
BATCH_SIZE=${BATCH_SIZE:-1}
BEAM_WIDTH=${BEAM_WIDTH:-1}
SEQ_LEN=${SEQ_LEN:-8}
DATA_TYPE=${DATA_TYPE:-fp32}
TEST_TIME=${TEST_TIME:-3}
SAMPLING_TOPK=${SAMPLING_TOPK:-1}
MODEL_TYPE=${MODEL_TYPE:-Megatron-DeepSpeed}
CKPT_DIR=${CKPT_DIR:-$DIFF_MOE_CKPT_DIR}
MODEL_NAME_FOR_BENCH=${MODEL_NAME_FOR_BENCH:-t5-base}
DATASET_NAME=${DATASET_NAME:-EdinburghNLP/xsum}
ITERATIONS=${ITERATIONS:-1}
DURATION=${DURATION:-0}
WARMUP_ITERATIONS=${WARMUP_ITERATIONS:-1}
LIB_PATH=${LIB_PATH:-$DIFF_MOE_LIB_PATH}
CACHE_SIZE=${CACHE_SIZE:?CACHE_SIZE is required}
USE_MOE_CACHE=${USE_MOE_CACHE:?USE_MOE_CACHE is required}
FIX_CACHE_SIZE=${FIX_CACHE_SIZE:?FIX_CACHE_SIZE is required}
TOP_K_EXPERTS=${TOP_K_EXPERTS:-}
LAYER_NUM=${LAYER_NUM:-6}
MAX_SAMPLES=${MAX_SAMPLES:-1}
EOF
    "${cmd[@]}" 2>&1 | tee "$log_file"
  else
    "${cmd[@]}"
  fi
}
