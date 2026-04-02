# DeepSpeed 基线复现说明

## 目标

本文档不讨论“这个仓库是否真正集成了 DeepSpeed”，而是只回答一个复现问题：

- **作者在项目里用什么实现思路来构造 `DeepSpeed` baseline**
- **如果不依赖 DeepSpeed runtime，如何按这个思路自己复现**

先给结论：

- 这个项目里的 `DeepSpeed`，本质上是一个 **MoE expert weight 的按需加载基线**
- 它对应的核心思想是 **FETCH_ON_DEMAND**
- 即：**当前 step 当前层需要哪个 expert，就在这一层临时把哪个 expert 的权重搬到工作区，再立刻计算**
- 它不是完整的 DeepSpeed ZeRO-Offload 实现

## 一句话理解

作者把 `DeepSpeed` baseline 近似成：

> 不做专家常驻，不做热点缓存，不做下一层预取；每次路由出当前层活跃 expert 后，再把这些 expert 的权重按需搬到 GPU 工作区执行。

如果你要复现，重点不是复现 DeepSpeed 框架，而是复现这条执行链。

---

## 1. 代码里 `DeepSpeed` 是怎么被定义的

入口在 [scripts/eval_cache.py](/D:/moe_offloading/Diff-MoE/scripts/eval_cache.py:108)：

```python
elif method == "DeepSpeed":
    encoder_fetcher_mode = "1"
    decoder_fetcher_mode = "1"
```

而 fetch mode 的定义在 [src/fastertransformer/utils/config.h](/D:/moe_offloading/Diff-MoE/src/fastertransformer/utils/config.h:10)：

```cpp
enum class FetchType {
    GPU_ONLY,
    FETCH_ON_DEMAND,
    PREFETCH
};
```

所以：

- `DeepSpeed` 对应 `FETCH_ON_DEMAND`

这就是它的核心语义。

---

## 2. 作者想复现的到底是什么思想

你可以把它拆成 4 个动作：

1. 当前 step 执行到某个 MoE layer
2. router 给出该层当前真正会用到的 experts
3. 不提前保留所有 experts，也不预取下一层 experts
4. 只把当前层当前需要的 experts 权重搬到工作区，然后立即计算

这个思路的目标是模拟一种“用到才加载”的 offload baseline。

它和 Diff-MoE 的区别在于：

- 没有局部性驱动的 expert cache
- 没有固定 HPC / 动态 MPC 这种分层缓存
- 没有 priority-driven replacement
- 不依赖下一层的预取机制

所以如果你自己复现，**只保留按需加载，不要把 Diff-MoE 的 cache 逻辑带进去**。

---

## 3. 最小执行链

按代码路径，`DeepSpeed` baseline 的最小执行链是：

1. `eval_cache.py` 把 method 设为 `DeepSpeed`
2. 配置 `decoder_fetcher_mode = FETCH_ON_DEMAND`
3. `T5Decoder` 初始化 `FetcherContext`
4. `FfnLayer` 把当前层专家路由结果交给 `fetcher`
5. `fetcher` 只提取当前活跃 experts
6. 将这些 experts 的权重搬到 working buffer
7. `sync()` 后交换 working/dst buffer
8. 用搬好的权重完成当前层 MoE FFN 计算

这条链已经构成了可复现的核心。

---

## 4. 关键代码是如何串起来的

### 4.1 初始化 fetcher

在 [src/fastertransformer/models/t5/T5Decoder.cc](/D:/moe_offloading/Diff-MoE/src/fastertransformer/models/t5/T5Decoder.cc:118)：

```cpp
ffn_layer_->initFetcherContext(
    GlobalConfig::instance().decoder_fetcher_mode,
    moe_k_,
    GlobalConfig::instance().arena_size);
```

在 [src/fastertransformer/layers/FfnLayer.cc](/D:/moe_offloading/Diff-MoE/src/fastertransformer/layers/FfnLayer.cc:556)：

```cpp
if (mode != FetchType::GPU_ONLY) {
    fetcher_context_ = std::make_shared<FetcherContext<T>>(...);
}
```

含义很简单：

- 如果不是 `GPU_ONLY`
- 就创建一个 fetcher
- `DeepSpeed` 和 `Diff-MoE` 都走这套 fetcher
- 差别不在“有没有 fetcher”，而在“fetcher 被怎样配置和使用”

### 4.2 当前层执行 MoE 前，交给 fetcher

在 [src/fastertransformer/layers/FfnLayer.cc](/D:/moe_offloading/Diff-MoE/src/fastertransformer/layers/FfnLayer.cc:161)：

```cpp
moe_fc_runner_->run_moe_fc(...)
```

这里会把：

- 当前层路由后的 experts
- 当前层 expert weights
- cache / priority 相关参数

一起传入底层 MoE 路径。

对于 `DeepSpeed` baseline，你要抓住的关键不是 cache 参数，而是：

- 当前层路由结果 `permuted_experts`
- 当前层权重源地址
- fetcher working buffer

### 4.3 fetcher 只取当前活跃 experts

在 [src/fastertransformer/utils/fetcher.cc](/D:/moe_offloading/Diff-MoE/src/fastertransformer/utils/fetcher.cc:100)：

```cpp
check_cuda_error(cudaMemcpy(permuted_experts_, permuted_experts, sizeof(int) * num_rows_, cudaMemcpyDeviceToHost));
auto new_end        = std::unique(permuted_experts_, permuted_experts_ + num_rows_);
num_active_experts_ = new_end - permuted_experts_;
```

这段逻辑的意义是：

- 先拿到当前层当前 step 路由出来的 expert id
- 再去重，得到这一轮真正活跃的 expert 集合

这就是“按需加载”的需求集合。

### 4.4 对每个活跃 expert，搬运权重到 working buffer

在 [src/fastertransformer/utils/fetcher.cc](/D:/moe_offloading/Diff-MoE/src/fastertransformer/utils/fetcher.cc:173) 左右，逻辑会对当前活跃 expert：

- 计算 expert 在源权重中的偏移
- 调用 arena 分配/搬运逻辑
- 把该 expert 的 FC1 / FC2 权重复制到 working buffer

你复现时，不一定要照搬这个 arena 设计，但机制必须保留：

- **输入：当前层活跃 expert id 列表**
- **输出：一个只包含这些 expert 权重的连续工作区**

### 4.5 `sync()` 后交换 working/dst

在 [src/fastertransformer/utils/fetcher.cc](/D:/moe_offloading/Diff-MoE/src/fastertransformer/utils/fetcher.cc:393)：

```cpp
for (auto& future : futures_) {
    future.wait();
}
check_cuda_error(cudaStreamSynchronize(stream));
std::swap(intermediate_dst_, intermediate_working_);
std::swap(output_dst_, output_working_);
```

这一步的含义是：

- 等当前需要的 expert 权重都搬完
- 再把 working buffer 切换成当前计算实际要读的 dst buffer

所以你复现时也需要这个双缓冲思想：

- `working buffer`：本轮正在加载
- `dst buffer`：本轮真正参与计算

不过对于最小复现，你也可以简化成：

- 同步加载完成后，直接在同一个 buffer 上计算

只要保持“先装载，再计算”的顺序即可。

---

## 5. `DeepSpeed` baseline 复现时必须保留的机制

如果你的目标是“复现作者这里的 DeepSpeed 思想”，下面这几条必须保留：

### 5.1 按层执行

它不是整模型统一搬权重，而是：

- 当前 step 执行到第 `i` 层
- 只处理第 `i` 层的 routed experts
- 只搬第 `i` 层对应 expert weights

### 5.2 按需专家集合

不是把整层 128 个 experts 都搬进来，而是：

- 先拿路由结果
- 再统计本层当前 step 真正活跃的 experts
- 只搬这些 experts

### 5.3 当前层即时使用

搬运不是为了长期缓存，而是为了当前层立即计算。

也就是：

- 本轮需要
- 本轮加载
- 本轮计算

### 5.4 不做 Diff-MoE 风格缓存优化

如果你要复现 `DeepSpeed` baseline，就不要加入这些内容：

- 全局热门专家初始化
- 局部热门专家 priority 更新
- HPC / MPC / LPC 分区
- `fix_cache_size`
- priority-driven replacement

因为这些是 Diff-MoE 的额外优化，不属于 `DeepSpeed` baseline 的核心。

---

## 6. 复现时建议抽象出的最小模块

如果你不直接复用这个仓库代码，可以按下面 5 个模块重写：

### 模块 A：Router 输出

输入：

- 当前层 token hidden states

输出：

- 每个 token 的 top-k experts

之后再做一次去重，得到：

- `active_experts_this_layer_this_step`

### 模块 B：Expert Weight Source

你需要有一个专家权重源，可能在：

- CPU memory
- pinned memory
- mmap 文件
- host 侧大 buffer

关键不是存放位置，而是：

- 能按 `expert_id` 定位该 expert 的 FC1/FC2 权重块

### 模块 C：Working Buffer

为当前 step 当前层创建工作区：

- `fc1_working`
- `fc2_working`

大小只要能容纳：

- 当前活跃 expert 数量

### 模块 D：On-Demand Loader

对 `active_experts_this_layer_this_step` 中每个 expert：

- 从 host/source 中取出该 expert 权重
- 拷到 GPU working buffer

这一步就是核心复现对象。

### 模块 E：MoE Compute

加载完成后：

- 用 working buffer 中的 expert weights 做当前层 FFN 计算

---

## 7. 最小伪代码

你可以按下面这个伪代码复现：

```python
for step in decoding_steps:
    for layer in moe_layers:
        routed = router(layer_hidden_states[layer])          # per-token top-k
        active_experts = unique(routed.expert_ids)           # layer-level active set

        working_fc1 = alloc_gpu_buffer(len(active_experts))
        working_fc2 = alloc_gpu_buffer(len(active_experts))

        for local_idx, expert_id in enumerate(active_experts):
            copy_to_gpu(
                dst=working_fc1[local_idx],
                src=host_fc1_weights[layer][expert_id]
            )
            copy_to_gpu(
                dst=working_fc2[local_idx],
                src=host_fc2_weights[layer][expert_id]
            )

        synchronize()

        output = moe_ffn_compute(
            hidden=layer_hidden_states[layer],
            routed_experts=routed,
            fc1=working_fc1,
            fc2=working_fc2
        )
```

如果你的实现满足这段逻辑，它在思想上就已经非常接近作者这里的 `DeepSpeed` baseline。

---

## 8. 它和真正 DeepSpeed / ZeRO-Offload 的关系

为了防止复现时跑偏，这里要明确区分：

作者复现的是：

- **按需加载的 offload 思想**

作者没有在仓库里真正复现的是：

- DeepSpeed engine
- ZeRO state partition
- optimizer / parameter / gradient 的完整 ZeRO-Offload 体系

因此，你在自己的复现里：

- 不需要先把 DeepSpeed runtime 搭起来
- 只要把 **MoE expert weights 的 fetch-on-demand 路径** 实现出来即可

---

## 9. 复现检查清单

你复现完成后，可以用下面几条检查自己是否对齐作者思路：

- 当前 step 是否先做路由，再决定搬哪些 experts
- 是否只搬当前层活跃 experts，而不是整层全部 experts
- 是否没有为 `DeepSpeed` baseline 加入热点缓存机制
- 是否每一层独立按需加载自己的 experts
- 是否加载完成后立即用于当前层计算

如果这 5 条都满足，你的实现思路就和该项目里的 `DeepSpeed` baseline 基本一致。

---

## 10. 一句话总结

在该项目中，`DeepSpeed` baseline 的实现思路可以概括为：

> **将每个 step、每个 MoE layer 的专家权重访问，简化为“先路由、再提取活跃 experts、再按需搬运这些 experts 的权重到 GPU 工作区、最后立刻计算”的 fetch-on-demand 流程。**

这就是你复现时真正应该抓住的核心。
