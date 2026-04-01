# AutoDL 迁移与 4090D 部署文档

本文档用于两件事：

1. 在当前实例上保存已修改的工程代码
2. 在新的 `4090D + 60G DRAM` 实例上快速恢复并部署 Diff-MoE

## 一、当前实例如何保存工程

### 1. 推荐方式

优先级如下：

1. `git commit` 并推送到你自己的仓库
2. 如果暂时不能推送，至少导出补丁文件

### 2. 一键保存脚本

仓库已提供：

```bash
bash scripts/save_autodl_state.sh
```

这个脚本会：

- stage 主要代码改动
- 生成补丁文件：`~/autodl-tmp/diff-moe-autodl.patch`
- 生成改动文件清单：`~/autodl-tmp/diff-moe-changed-files.txt`
- 如果有 staged 改动，则自动提交一次

### 3. 手动推送

如果远程仓库是你自己的：

```bash
git remote -v
git push origin HEAD
```

### 4. 释放实例前建议保留

需要保留：

- Git 提交或补丁文件
- 这份部署文档

不建议保留：

- `build/`
- `logs/`
- `~/autodl-tmp/hf-cache`
- `~/autodl-tmp/diff-moe-data`
- `swapfile`

这些都可以在新实例重新生成。

## 二、新建 4090D 60G DRAM 实例后的完整步骤

假设：

- GPU: `RTX 4090D`
- 显存足够
- DRAM: `60G`
- 系统镜像仍是 `CUDA 12.4 + torch2.5.1` 这类旧环境

由于 4090D 的算力为 `SM=89`，这个项目可以直接编译，不需要像 5090 那样绕开 `sm_120`。

## 三、拉取代码

### 方式 A：从你自己的仓库恢复

```bash
cd ~/autodl-tmp
git clone https://github.com/Yuelinfeng/Diff-MoE.git Diff-MoE
cd Diff-MoE
git submodule update --init --recursive
```

### 方式 B：从原仓库 + patch 恢复

```bash
cd ~/autodl-tmp
git clone https://hub.njuu.cf/Yuelinfeng/Diff-MoE.git || \
git clone https://ghproxy.com/https://github.com/Yuelinfeng/Diff-MoE.git || \
git clone https://mirror.ghproxy.com/https://github.com/Yuelinfeng/Diff-MoE.git

cd Diff-MoE
git submodule update --init --recursive
git apply ~/autodl-tmp/diff-moe-autodl.patch
```

## 四、创建环境

```bash
eval "$(conda shell.bash hook)"
conda create -n diff-moe python=3.10 -y
conda activate diff-moe
```

## 五、安装依赖

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "cmake>=3.22" ninja
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "numpy<2" \
  sentencepiece==0.1.99 \
  datasets==2.16.1 \
  omegaconf==2.3.0 \
  rouge_score==0.1.2 \
  sacrebleu==2.4.2 \
  transformers==4.31.0 \
  configparser==6.0.1 \
  py-cpuinfo==9.0.0 \
  "protobuf<4"
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.is_available())"
```

期望类似：

```bash
2.7.0+cu128 12.8
NVIDIA GeForce RTX 4090D
(8, 9)
True
```

## 六、设置 HuggingFace 国内镜像和缓存目录

```bash
export HF_HOME=~/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
mkdir -p ~/autodl-tmp/diff-moe-data/ft
```

## 七、编译项目

仓库已把 C++ 标准提升到 `C++17`，这是为了兼容新版本 PyTorch。

4090D 使用 `SM=89`：

```bash
cd ~/autodl-tmp/Diff-MoE
rm -rf build
mkdir build
cd build

cmake -DSM=89 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCXX_STD=17 \
  -DBUILD_PYT=ON \
  -DBUILD_MULTI_GPU=OFF \
  -DBUILD_TRT=OFF \
  -DBUILD_TF=OFF \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  ..

make -j"$(nproc)"
```

## 八、兼容原项目硬编码路径

项目里仍有一部分代码写死了 `/workspace/FasterTransformer`，先用软链接兼容：

```bash
mkdir -p /workspace
ln -sfn ~/autodl-tmp/Diff-MoE /workspace/FasterTransformer
```

## 九、转换模型

### 1. FP32 转换

```bash
cd ~/autodl-tmp/Diff-MoE

python examples/pytorch/t5/utils/huggingface_switch_transformer_ckpt_convert.py \
  -saved_dir ~/autodl-tmp/diff-moe-data/ft/switch-base-128 \
  -in_file google/switch-base-128 \
  -inference_tensor_para_size 1 \
  -weight_data_type fp32 \
  -processes 1 \
  --low_cpu_mem_usage \
  --cache_dir "$HF_HOME"
```

整理目录：

```bash
if [ -d ~/autodl-tmp/diff-moe-data/ft/switch-base-128/1-gpu ]; then
  mv ~/autodl-tmp/diff-moe-data/ft/switch-base-128/1-gpu/* ~/autodl-tmp/diff-moe-data/ft/switch-base-128/
  rmdir ~/autodl-tmp/diff-moe-data/ft/switch-base-128/1-gpu
fi
```

### 2. 说明

当前仓库的 C++ offload 读取逻辑默认按 `float` 权重大小读取，所以：

- `FP32` 转换权重和当前推理链路兼容
- `FP16` 转换权重会出现文件大小不匹配 warning

如果你要走当前仓库这条 FT offload 路径，建议优先保留 `FP32` 转换产物。

## 十、准备 `cpp_config.ini`

```bash
cat > ~/autodl-tmp/Diff-MoE/cpp_config.ini <<'EOF'
[default]
arena_size = 7247757312
encoder_fetcher_mode = 1
decoder_fetcher_mode = 2
profiling = 1
detailed_timing = 0
offload_path = /root/autodl-tmp/diff-moe-data/ft/switch-base-128/
load_from_cpp = 1
quant_mode = 0
vocab_size = 32128
cache_policy = LFU
cache_size = 3
use_moe_cache = True
fix_cache_size = 1
max_val = 2
threshold = 1.0
dec_in_cache = 0.4
dec_out_cache = 0.2
top_k_experts = [[35,43,38],[0,84,51],[52,0,33],[93,0,72],[0,61,38],[0,35,105]]
EOF
```

## 十一、先跑最小 smoke test

```bash
cd ~/autodl-tmp/Diff-MoE/build

python ../examples/pytorch/t5/perf_benchmark.py \
  --batch_size 1 \
  --beam_width 1 \
  --seq_len 8 \
  --data_type fp32 \
  --test_time 3 \
  --sampling_topk 1 \
  --model_type Megatron-DeepSpeed \
  --ckpt_path ~/autodl-tmp/diff-moe-data/ft/switch-base-128 \
  --model t5-base \
  --dataset_name EdinburghNLP/xsum \
  --iterations 1 \
  --duration 0 \
  --warmup_iterations 1 \
  --skip_gemm \
  --lib_path ./lib/libth_transformer.so \
  --cache_size 1 \
  --use_moe_cache False \
  --fix_cache_size 0 \
  --top_k_experts "0,0,0,0,0,0" \
  --layer_num 6 \
  --max_samples 1
```

如果最小用例通过，再加大：

- `seq_len`
- `max_samples`
- `cache_size`
- `use_moe_cache`

## 十二、常见问题

### 1. `Killed`

一般表示：

- DRAM 不够
- 被 OOM killer 直接杀掉

先确认：

```bash
dmesg -T | tail -n 50
free -h
nvidia-smi
```

### 2. `Unsupported gpu architecture`

说明 `nvcc` 太老，不认识目标 SM。4090D 不需要 `sm_120`，用 `SM=89` 即可。

### 3. `C++17 or later compatible compiler is required to use PyTorch`

说明 CMake 仍按旧标准编译，确保：

```bash
-DCXX_STD=17
```

## 十三、建议

对 4090D + 60G DRAM：

- 比 5090 + 31G DRAM 更适合跑 `switch-base-128`
- 仍建议先从最小 smoke test 开始
- 确认跑通后再逐步放大参数
