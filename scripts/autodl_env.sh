#!/usr/bin/env bash
set -euo pipefail

export DIFF_MOE_ROOT="${DIFF_MOE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DIFF_MOE_DATA_DIR="${DIFF_MOE_DATA_DIR:-$HOME/autodl-tmp/diff-moe-data}"
export DIFF_MOE_BUILD_DIR="${DIFF_MOE_BUILD_DIR:-$DIFF_MOE_ROOT/build}"
export DIFF_MOE_LOG_DIR="${DIFF_MOE_LOG_DIR:-$DIFF_MOE_ROOT/logs}"
export DIFF_MOE_CPP_CONFIG="${DIFF_MOE_CPP_CONFIG:-$DIFF_MOE_ROOT/cpp_config.ini}"

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

echo "DIFF_MOE_ROOT=$DIFF_MOE_ROOT"
echo "DIFF_MOE_DATA_DIR=$DIFF_MOE_DATA_DIR"
echo "HF_HOME=$HF_HOME"
echo "HF_ENDPOINT=$HF_ENDPOINT"
