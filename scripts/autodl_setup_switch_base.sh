#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/autodl_env.sh"

ENV_NAME="${ENV_NAME:-diff-moe-py38}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu117}"
DIFF_MOE_SM="${DIFF_MOE_SM:-80}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Please start from an AutoDL image with conda installed."
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "$TORCH_INDEX_URL" torch==1.13.1+cu117 torchvision==0.14.1+cu117
python -m pip install -i "$PYPI_INDEX_URL" "cmake>=3.22" ninja
python -m pip install -i "$PYPI_INDEX_URL" -r "$ROOT_DIR/requirements.autodl.txt"
python -m pip check

mkdir -p "$DIFF_MOE_BUILD_DIR"
cd "$DIFF_MOE_BUILD_DIR"
cmake -DSM="$DIFF_MOE_SM" -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
make -j"$(nproc)"

echo
echo "Setup completed in conda env: $ENV_NAME"
echo "Source $ROOT_DIR/scripts/autodl_env.sh before converting or running the model."
