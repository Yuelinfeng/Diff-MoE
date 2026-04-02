# AutoDL `switch-base-128` 阶段收口总结

本文档用于对当前 `switch-base-128` 在 AutoDL `RTX 4090D` 上的推进结果做阶段收口，总结已经确认的结论、当前推荐配置、关键日志位置和后续可选工作。

## 1. 当前阶段已经完成的事

已完成并确认：

1. `switch-base-128` 的 `FP32 + FasterTransformer offload` 推理链路已在 `RTX 4090D` 上稳定跑通。
2. 最小 smoke test 已通过。
3. 基于 smoke test 完成了 staged sweep，确认 `cache_size=3` 不是当前 workload 的最佳配置。
4. 完成了 `cache_size=2` 与 `fix_cache_size` 的对照实验和长窗口复测。
5. 完成了更重 sampling workload 的扩展验证。
6. 已真正跑通 `FT beamsearch` 路径。

## 2. 当前最可靠的结论

### 2.1 sampling 主线

当前在 4090D 上，最稳的 sampling 主配置是：

```text
cache_size = 2
use_moe_cache = True
fix_cache_size = 0
top_k_experts = 35,43,0,84,52,0,93,0,0,61,0,35
data_type = fp32
dataset = EdinburghNLP/xsum
```

已验证结论：

- `cache_size=2` 明显优于此前较差配置
- `fix_cache_size=1` 没有带来稳定额外收益
- 在当前全动态 `cache_size=2` 配置下，热门专家初始化本身没有体现出稳定额外收益
- 当前收益主要来自合理的 cache 容量选择和更高的 workload，而不是固定热门专家策略

### 2.2 beamsearch 主线

`FT beamsearch` 已真正跑通。  
之前 `beam_width=4` 但仍显示 `ft-sampling`，原因不是 beam-width 无效，而是 `perf_benchmark.py` 通过 `--test_time` 选择路径：

- `--test_time 1` -> `ft-beamsearch`
- `--test_time 3` -> `ft-sampling`

本轮已用 `--test_time 1` 成功跑通 beamsearch。

## 3. 当前阶段关键结果

### 3.1 staged sweep

日志目录：

- `~/autodl-tmp/Diff-MoE/logs/staged_sweep_20260401_223857`

代表性结果：

| 配置 | 结果 |
| --- | --- |
| `seq_len=32, max_samples=8, cache off` | `72 tok/s` |
| `seq_len=32, max_samples=8, cache on, cache_size=3, real experts` | `67 tok/s` |

结论：

- `cache_size=3` 在当前轻量 workload 下不优

### 3.2 `cache_size=2` 对照与复测

日志目录：

- `~/autodl-tmp/Diff-MoE/logs/cd_retest_20260402_100140`

长窗口复测结果：

| 配置 | 结果 |
| --- | --- |
| `cache_size=2, fix_cache_size=0` | `72 tok/s` |
| `cache_size=2, fix_cache_size=1` | `72 tok/s` |

结论：

- `fix_cache_size=1` 没有稳定优势
- 当前更推荐 `fix_cache_size=0`

### 3.3 sampling 放大结果

日志目录：

- `~/autodl-tmp/Diff-MoE/logs/next_workloads_20260402_103355`
- `~/autodl-tmp/Diff-MoE/logs/sampling_scaleup_20260402_110355`
- `~/autodl-tmp/Diff-MoE/logs/sampling_scaleup_v2_20260402_111033`

代表性结果：

| 配置 | 结果 |
| --- | --- |
| `batch_size=2, seq_len=32` | `98 tok/s` |
| `batch_size=1, seq_len=64` | `103 tok/s` |
| `batch_size=4, seq_len=32` | `130 tok/s` |
| `batch_size=1, seq_len=96` | `123 tok/s` |
| `batch_size=8, seq_len=32` | `175 tok/s` |
| `batch_size=4, seq_len=64` | `179 tok/s` |

当前最佳 sampling 结果：

```text
batch_size = 4
beam_width = 1
seq_len = 64
cache_size = 2
fix_cache_size = 0
=> 179 tokens/sec
```

### 3.4 FT beamsearch

日志目录：

- `~/autodl-tmp/Diff-MoE/logs/ft_beamsearch_20260402_112355`

结果：

```text
batch_size = 1
beam_width = 4
seq_len = 32
cache_size = 2
fix_cache_size = 0
=> 42 tokens/sec
```

## 4. 当前推荐配置

### 4.1 默认 sampling 推荐配置

若当前目标是稳定复现实验并获得较好吞吐，推荐先使用：

```text
cache_size = 2
use_moe_cache = True
fix_cache_size = 0
top_k_experts = 35,43,0,84,52,0,93,0,0,61,0,35
batch_size = 4
beam_width = 1
seq_len = 64
max_samples = 32
warmup_iterations = 2
iterations = 1
dataset = EdinburghNLP/xsum
data_type = fp32
```

用途：

- 作为当前 4090D 上的默认 sampling 实验起点

### 4.2 默认 beamsearch 参考配置

若当前目标是验证 beamsearch 链路，推荐先使用：

```text
cache_size = 2
use_moe_cache = True
fix_cache_size = 0
top_k_experts = 35,43,0,84,52,0,93,0,0,61,0,35
batch_size = 1
beam_width = 4
seq_len = 32
max_samples = 32
warmup_iterations = 2
iterations = 1
dataset = EdinburghNLP/xsum
data_type = fp32
test_time = 1
```

## 5. 本阶段新增脚本

当前工作区内新增了以下实验脚本：

- [run_cd_retest.sh](/D:/moe_offloading/Diff-MoE/scripts/run_cd_retest.sh)
- [run_next_workloads.sh](/D:/moe_offloading/Diff-MoE/scripts/run_next_workloads.sh)
- [run_sampling_scaleup.sh](/D:/moe_offloading/Diff-MoE/scripts/run_sampling_scaleup.sh)
- [run_sampling_scaleup_v2.sh](/D:/moe_offloading/Diff-MoE/scripts/run_sampling_scaleup_v2.sh)
- [run_ft_beamsearch.sh](/D:/moe_offloading/Diff-MoE/scripts/run_ft_beamsearch.sh)

## 6. 本阶段代码修复

本阶段确认并修复了一个真正影响 FT beamsearch 的 bug：

- 文件：
  [perf_benchmark.py](/D:/moe_offloading/Diff-MoE/examples/pytorch/t5/perf_benchmark.py)
- 问题：
  `to_word_list_format()` 只在 FT sampling 分支定义，但 FT beamsearch 分支也会调用，导致首次进入 `ft-beamsearch` 时触发 `UnboundLocalError`
- 处理：
  将 `to_word_list_format()` 提到 FT 公共分支，beamsearch 和 sampling 共用

## 7. 当前阶段建议

如果你准备暂时收口，当前就可以认为这一阶段目标已经完成得比较扎实：

- `switch-base-128` 已跑通
- sampling 最优区间已摸清
- beamsearch 已打通

后续建议按优先级选择：

1. 整理提交当前实验脚本、文档和必要代码修复
2. 如需继续推进，优先补一组更重的 beamsearch，例如：
   - `beam_width=4, seq_len=64`
3. 如需更接近作者完整实验，再单独整理 `eval_cache.py` 的 4090D 友好版本，而不是直接裸跑作者原始 sweep
