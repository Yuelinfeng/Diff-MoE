# AutoDL 跑通 Diff-MoE（仅 `switch-base-128`）

本仓库原始脚本默认使用 `/workspace/FasterTransformer`、`/data` 和整套旧版 NGC 环境。为了便于在 AutoDL 上直接跑通，仓库里额外提供了以下脚本：

- `scripts/autodl_env.sh`
- `scripts/autodl_setup_switch_base.sh`
- `scripts/autodl_run_switch_base_smoke.sh`
- `requirements.autodl.txt`

## 1. 推荐环境

- Python: `3.8`
- PyTorch: `1.13.1+cu117`
- 推理模型: `google/switch-base-128`
- HuggingFace 镜像: `https://hf-mirror.com`
- HuggingFace 缓存目录: `~/autodl-tmp/hf-cache`

如果你的 AutoDL 镜像已经自带合适的 CUDA 驱动，直接使用仓库脚本即可。

## 2. 一键准备

在仓库根目录执行：

```bash
bash scripts/autodl_setup_switch_base.sh
```

这个脚本会完成：

- 创建 `conda` 环境 `diff-moe-py38`
- 安装 PyTorch 和最小 Python 依赖
- 设置 HuggingFace 国内镜像相关环境变量
- 创建以下目录：
  - `~/autodl-tmp/hf-cache`
  - `~/autodl-tmp/diff-moe-data`
  - `~/autodl-tmp/diff-moe-data/ft`
  - `<repo>/logs`
  - `<repo>/build`
- 编译项目

默认按 `A100` 的算力编译，即 `SM=80`。如果你的显卡不是 A100，可以在执行前覆盖：

```bash
export DIFF_MOE_SM=86   # 例如 A10
bash scripts/autodl_setup_switch_base.sh
```

## 3. 启用运行环境

```bash
eval "$(conda shell.bash hook)"
conda activate diff-moe-py38
source scripts/autodl_env.sh
```

## 4. 跑通 Switch-Base

```bash
bash scripts/autodl_run_switch_base_smoke.sh
```

这个脚本会：

- 从 `google/switch-base-128` 下载模型到 `~/autodl-tmp/hf-cache`
- 转换成 Diff-MoE / FasterTransformer 所需格式，输出到 `~/autodl-tmp/diff-moe-data/ft/switch-base-128`
- 使用 `EdinburghNLP/xsum` 做一个小样本 smoke test

## 5. 关键目录

- HuggingFace 缓存: `~/autodl-tmp/hf-cache`
- 转换后权重: `~/autodl-tmp/diff-moe-data/ft/switch-base-128`
- 构建目录: `<repo>/build`
- 日志目录: `<repo>/logs`

## 6. 与原仓库相比的调整

- `huggingface_switch_transformer_ckpt_convert.py` 不再把基础模型写死成 `switch-xxl-128`
- 基础模型跑通时不再强制依赖 `peft`
- `scripts/eval_cache.py` 改为优先读取以下环境变量：
  - `DIFF_MOE_DATA_DIR`
  - `DIFF_MOE_BUILD_DIR`
  - `DIFF_MOE_LOG_DIR`
  - `DIFF_MOE_CPP_CONFIG`

## 7. 常见问题

如果编译报 `SM` 不匹配：

```bash
export DIFF_MOE_SM=<你的显卡算力>
rm -rf build
bash scripts/autodl_setup_switch_base.sh
```

如果模型下载慢或失败，先确认以下变量已生效：

```bash
echo $HF_ENDPOINT
echo $HF_HOME
```

期望输出分别类似：

```bash
https://hf-mirror.com
/root/autodl-tmp/hf-cache
```
