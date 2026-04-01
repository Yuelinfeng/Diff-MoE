#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/autodl_env.sh"

MODEL_NAME="${MODEL_NAME:-switch-base-128}"
MODEL_ID="${MODEL_ID:-google/$MODEL_NAME}"
CKPT_DIR="$DIFF_MOE_DATA_DIR/ft/$MODEL_NAME"
LIB_PATH="${LIB_PATH:-$DIFF_MOE_BUILD_DIR/lib/libth_transformer.so}"

if [[ ! -f "$LIB_PATH" ]]; then
  echo "Missing $LIB_PATH. Run scripts/autodl_setup_switch_base.sh first."
  exit 1
fi

if [[ ! -f "$CKPT_DIR/config.ini" ]]; then
  rm -rf "$CKPT_DIR"
  python "$ROOT_DIR/examples/pytorch/t5/utils/huggingface_switch_transformer_ckpt_convert.py" \
    -saved_dir "$CKPT_DIR" \
    -in_file "$MODEL_ID" \
    -inference_tensor_para_size 1 \
    -weight_data_type fp32 \
    --cache_dir "$HF_HOME"

  if [[ -d "$CKPT_DIR/1-gpu" ]]; then
    mv "$CKPT_DIR/1-gpu"/* "$CKPT_DIR/"
    rmdir "$CKPT_DIR/1-gpu"
  fi
fi

cd "$DIFF_MOE_BUILD_DIR"
python "$ROOT_DIR/examples/pytorch/t5/perf_benchmark.py" \
  --batch_size 1 \
  --beam_width 1 \
  --seq_len 32 \
  --data_type fp16 \
  --test_time 3 \
  --sampling_topk 1 \
  --model_type Megatron-DeepSpeed \
  --ckpt_path "$CKPT_DIR" \
  --model t5-base \
  --dataset_name EdinburghNLP/xsum \
  --iterations 1 \
  --duration 0 \
  --warmup_iterations 1 \
  --skip_gemm \
  --lib_path "$LIB_PATH" \
  --cache_size 3 \
  --use_moe_cache True \
  --fix_cache_size 1 \
  --top_k_experts "35,43,38,0,84,51,52,0,33,93,0,72,0,61,38,0,35,105" \
  --layer_num 6 \
  --max_samples 8
