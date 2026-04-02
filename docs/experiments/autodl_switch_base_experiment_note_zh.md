# AutoDL `switch-base-128` 实验记录

本文档记录在 AutoDL `RTX 4090D + 60G DRAM` 环境下，对 `Yuelinfeng/Diff-MoE` 项目的 `switch-base-128` 推理链路验证、阶段性结论和当前推荐配置。

## 1. 实验背景

- 目标模型：`google/switch-base-128`
- 推理链路：`FP32 + FasterTransformer offload`
- 数据集：`EdinburghNLP/xsum`
- 机器：AutoDL `RTX 4090D`
- PyTorch：`2.7.0+cu128`
- Python：`3.10`

已确认最小 smoke test 跑通，后续实验均基于该成功基线逐步放大。

## 2. 关键概念

### 2.1 `model_configs.json` 与热门专家

仓库中的 [model_configs.json](/D:/moe_offloading/Diff-MoE/scripts/model_configs.json) 保存了作者离线整理好的 `cache_init_experts`。这组专家可以理解为作者在 warmup / profiling 语境下得到的热门专家初始化结果。

例如 `switch-base-128-xsum` 中保存了每层若干个热门专家，当前实验常用的是其中前 2 个或前 3 个。

### 2.2 `cpp_config.ini` 的作用

C++ 侧运行时并不直接读取 `model_configs.json`，而是读取 `cpp_config.ini` 中的：

- `cache_size`
- `use_moe_cache`
- `fix_cache_size`
- `top_k_experts`

也就是说，热门专家的调用链是：

`model_configs.json` -> Python 脚本整理 -> 写入 `cpp_config.ini` -> C++ 侧读取 `cpp_config.ini`

### 2.3 `fix_cache_size` 的含义

若：

- `cache_size = 3`
- `fix_cache_size = 1`

则表示每层 cache 共 3 个槽位，其中：

- 前 1 个槽位固定保留
- 后 2 个槽位运行时可动态替换

因此，`top_k_experts` 不是“全部锁死在 GPU 上”，而是“提供初始 cache 种子，其中前 `fix_cache_size` 个固定，其余动态”。

## 3. 阶段性实验结果

### 3.1 staged sweep 结果

目录：

- `~/autodl-tmp/Diff-MoE/logs/staged_sweep_20260401_223857`

主要结果：

| 阶段 | 配置摘要 | 结果 |
| --- | --- | --- |
| Stage 1 | `seq_len=8, max_samples=1, cache off` | `25 tok/s` |
| Stage 2 | `seq_len=32, max_samples=1, cache off` | `62 tok/s` |
| Stage 3 | `seq_len=32, max_samples=8, cache off` | `72 tok/s` |
| Stage 4 | `seq_len=32, max_samples=8, cache on, cache_size=1` | `67 tok/s` |
| Stage 5 | `seq_len=32, max_samples=8, cache on, cache_size=3, dummy experts` | `65 tok/s` |
| Stage 6 | `seq_len=32, max_samples=8, cache on, cache_size=3, real experts` | `67 tok/s` |

结论：

- `switch-base-128 + FP32 + FT offload` 已经可以在 4090D 上稳定跑完整实验
- `cache_size=3` 在当前 workload 上并不是好配置
- 作者的热门专家初始化可以稳定运行，但在 `cache_size=3` 下没有打赢 baseline

### 3.2 C / D 对照实验

为确认 `cache_size=2` 和 `fix_cache_size` 的影响，进行了 A/B/C/D 四组测试：

| 组别 | 配置 | 结果 |
| --- | --- | --- |
| A | baseline | `65 tok/s` |
| B | `cache_size=3, fix_cache_size=0, real experts` | `67 tok/s` |
| C | `cache_size=2, fix_cache_size=0, real experts` | `72 tok/s` |
| D | `cache_size=2, fix_cache_size=1, real experts` | `74 tok/s` |

初看 D 最优，但短窗口波动较大，因此继续做长窗口复测。

### 3.3 C / D 长窗口复测

脚本：

- [run_cd_retest.sh](/D:/moe_offloading/Diff-MoE/scripts/run_cd_retest.sh)

日志目录：

- `~/autodl-tmp/Diff-MoE/logs/cd_retest_20260402_100140`

结果：

| 轮次 | C: `cache_size=2, fix_cache_size=0` | D: `cache_size=2, fix_cache_size=1` |
| --- | --- | --- |
| 1 | `72 tok/s` | `72 tok/s` |
| 2 | `72 tok/s` | `72 tok/s` |
| 3 | `72 tok/s` | `72 tok/s` |

更细的 batch time：

- C 约 `440.95 ~ 442.63 ms / batch`
- D 约 `444.70 ~ 446.06 ms / batch`

结论：

- 在更长测量窗口下，C 和 D 吞吐等价
- `fix_cache_size=1` 没有带来稳定额外收益
- 当前最稳的配置是 `cache_size=2, fix_cache_size=0`

### 3.4 更重 workload 验证

脚本：

- [run_next_workloads.sh](/D:/moe_offloading/Diff-MoE/scripts/run_next_workloads.sh)

日志目录：

- `~/autodl-tmp/Diff-MoE/logs/next_workloads_20260402_103355`

结果：

| 组别 | 配置 | 结果 |
| --- | --- | --- |
| W1 | 最佳配置，`batch_size=2` | `98 tok/s` |
| W2 | 最佳配置，`beam_width=4` | `72 tok/s` |
| W3 | 最佳配置，`seq_len=64` | `103 tok/s` |
| W4 | 最佳配置，dummy experts 对照 | `72 tok/s` |

结论：

- `batch_size=2` 和 `seq_len=64` 都能进一步提升吞吐
- 当前最佳配置在更重 workload 下仍然有效
- `dummy experts` 对照与 real experts 基本等价，说明在当前全动态 `cache_size=2` 配置下，主要收益来自 cache 容量本身，而不是热门专家初始化
- `W2` 仍然走的是 sampling 路径，因此不能当作真正的 beam-search 结论

## 4. 当前最佳配置卡片

当前推荐继续后续实验的配置为：

```text
cache_size = 2
use_moe_cache = True
fix_cache_size = 0
top_k_experts = 35,43,0,84,52,0,93,0,0,61,0,35
seq_len = 32
max_samples = 32
batch_size = 1
beam_width = 1
warmup_iterations = 2
iterations = 1
dataset = EdinburghNLP/xsum
data_type = fp32
```

推荐理由：

- 相比 baseline 稳定更快
- 相比 `cache_size=3` 更适合当前 workload
- 不依赖固定热门专家，逻辑更简单
- 在长窗口复测中表现最稳

若需要更高吞吐的 sampling 配置，可进一步尝试：

```text
cache_size = 2
use_moe_cache = True
fix_cache_size = 0
top_k_experts = 35,43,0,84,52,0,93,0,0,61,0,35
seq_len = 64
max_samples = 32
batch_size = 1
beam_width = 1
warmup_iterations = 2
iterations = 1
dataset = EdinburghNLP/xsum
data_type = fp32
```

该配置在当前实验中达到约 `103 tokens/sec`。

## 5. 当前可以确认的结论

1. `switch-base-128 + FP32 + FT offload` 已经不只是 smoke test，而是可以稳定进行批量实验。
2. 作者保存的热门专家初始化是可用的，但在当前 workload 下，固定其中一部分并没有带来稳定收益。
3. 在当前全动态 `cache_size=2` 配置下，热门专家初始化本身没有体现出稳定额外收益。
4. 真正带来收益的关键因素，是把 cache 容量从不合适的配置调整到更合适的 `cache_size=2`。
5. 当前 `cache_size=2` 已被稳定证明优于此前的较差配置，并且在 `batch_size=2`、`seq_len=64` 下仍能继续放大吞吐。

## 6. 下一轮推进顺序

当前不建议立刻扩大参数空间做大规模乱扫，建议按下面顺序推进。

### 6.1 第一优先级：继续做 sampling 路径放大实验

保持：

- `cache_size=2`
- `fix_cache_size=0`
- `use_moe_cache=True`

优先顺序建议：

1. `batch_size: 1 -> 2`
2. `batch_size: 2 -> 4`
3. `seq_len: 32 -> 64`
4. `seq_len: 64 -> 96`

目的：

- 判断当前最优配置在更重 workload 下是否仍然成立
- 看 Diff-MoE 的收益是否会随计算规模增加而更明显

### 6.2 第二优先级：单独做真正的 beam-search 实验

当前 `perf_benchmark.py` 的默认路径仍然是 `ft-sampling`，因此需要单独整理 beam-search 设置，避免把 `beam_width=4` 误当成已经完成的 beam-search 结论。

### 6.3 第三优先级：再回头微调策略参数

在上述主配置稳定后，再尝试小范围 sweep：

- `threshold`
- `dec_in_cache`
- `dec_out_cache`

不建议现在优先做它们，因为当前更大的收益已经来自 `cache_size` 调整。

## 7. 建议的下一批实验

建议优先跑下面 4 组：

1. 当前最佳配置，`batch_size=2`
2. 当前最佳配置，`batch_size=4`
3. 当前最佳配置，`seq_len=64`
4. 当前最佳配置，`seq_len=96`

如果这 4 组仍然稳定，那么就可以把当前结论从“局部最优配置”升级为“适合 4090D 的默认 sampling 实验起点”。
