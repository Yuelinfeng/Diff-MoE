# Diff-MoE 论文阅读与代码对应笔记

## 文档目的

本文档整理对论文《Diff-MoE: Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching》的中文解读，并将论文中的关键设计与当前仓库实现进行对照，便于后续继续做实现分析、复现实验和代码修改。

论文 PDF：

- `C:\Users\Leave\Zotero\storage\8MIA4VHL\Li 等 - 2025 - Diff-MoE Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching.pdf`

论文元数据：

- 标题：Diff-MoE: Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching
- 作者：Kexin Li, Wenkan Huang, Qinggang Wang, Long Zheng, Xiaofei Liao, Hai Jin, Jingling Xue
- 会议：SC 2025
- DOI：`10.1145/3712285.3759903`

---

## 1. 论文拆解

### 1.1 论文要解决什么问题

这篇论文解决的是 MoE 模型在单卡 GPU、host-GPU 异构环境下进行大 batch 推理时的通信瓶颈问题。

MoE 的特点是：

- 计算时只激活少量 experts，因此理论上计算量较低。
- 但全部 expert 参数非常大，单卡 GPU 往往放不下。

所以常见做法是：

- 把 non-MoE 参数常驻 GPU。
- 把 expert weights 放在 host memory。
- 在推理时仅把当前激活的 experts 传到 GPU。

问题在于，随着 batch size 增大：

- 每一轮激活的 experts 数量明显上升。
- `host -> GPU` 的通信量随之增大。
- GPU 计算由于并行度高，增长反而没那么快。

最终，通信会压过计算，吞吐下降，系统性能被 PCIe 带宽卡住。

### 1.2 现有方法为什么不够

论文把已有方案大致分成两类。

#### 1.2.1 预取类方法

代表是 `Pre-gated MoE`。

这类方法会提前把“下一层可能要用到的 experts”预取到 GPU，希望把通信隐藏在当前层计算后面。

问题是：

- batch 变大时，激活 expert 的数量增长很快。
- 当前层计算能掩盖掉的传输量是有限的。
- 超过这部分的传输仍然暴露在关键路径上。

所以在大 batch 下，预取依然无法解决通信主导延迟的问题。

#### 1.2.2 缓存类方法

代表是 `MoE-Infinity`。

这类方法会把一部分 recently used experts 保留在 GPU 中，减少重复搬运。

问题是：

- batch 变大时，同时活跃的 experts 更多。
- 缓存 miss rate 上升。
- 某些 expert 还没来得及再次复用，就已经被更大 batch 中的新 experts 挤掉。

结果是缓存收益明显下降。

### 1.3 论文的关键观察

论文认为 expert activation 里有两种 locality，被现有工作利用得不够充分。

#### 1.3.1 Global locality

在某个固定的 MoE layer 里，总有少数 experts 在整个推理过程中频繁出现。

这意味着：

- 这些 experts 值得长期留在 GPU。
- 它们是“常驻热点”。

#### 1.3.2 Temporal locality

另一些 experts 不一定在整个推理过程中都很热，但会在若干连续 decoding iterations 里反复被激活。

这意味着：

- 这些 experts 适合被暂时保留。
- 如果过早淘汰，就会产生重复传输。

### 1.4 Diff-MoE 的核心方法

论文提出 `Diff-MoE`，核心是“差分缓存 + 优先级管理 + 轻量预测”。

#### 1.4.1 Differential cache hierarchy

Diff-MoE 不把所有缓存位置一视同仁，而是分成三层。

##### HPC

`HPC`，high-priority cache。

特点：

- 每个 MoE layer 私有。
- 存放 offline 阶段识别出的 globally hot experts。
- 这些 experts 基本常驻 GPU。

##### MPC

`MPC`，medium-priority cache。

特点：

- 每个 MoE layer 私有。
- 存放 runtime 过程中变热的 locally hot experts。
- 由 priority-driven replacement policy 动态维护。

##### LPC

`LPC`，low-priority cache。

特点：

- 是一个临时缓冲区域。
- 当前层刚从 host 拉进来的 experts 会先进入这里。
- 预测出的 next-layer experts 也会先被预取到这里。
- 用完后清空或被覆盖。

#### 1.4.2 Priority-driven replacement

论文没有直接使用 LRU，而是给每个 expert 一个 priority score。

其更新规则大致是：

- 当前 step 被激活：加分。
- 没被激活但仍在 GPU cache 中：减分，而且减得更快。
- 没被激活且不在 cache 中：也减分，但减得更慢。

这样做的目的：

- 已经不热却占着宝贵 GPU 空间的 expert，要尽快降温。
- 刚开始变热的 expert，要尽快进入缓存。
- 短期内会重复出现的 expert，不要像 LRU 那样仅凭“最近访问顺序”被误淘汰。

#### 1.4.3 Lightweight predictor

Diff-MoE 还加了一个 lightweight predictor，用来预测“下一个 MoE layer”最可能激活的 experts。

论文中的设计要点：

- 只预测下一层，不做远期多层预测。
- batch 较小时预取 1 个 expert。
- batch 较大时预取 2 个 expert。
- predictor 使用 GRU 来学习跨层 activation pattern。

这样做的目标是：

- 尽量让 expert migration 和当前层计算重叠。
- 减少通信暴露在关键路径上的时间。

### 1.5 论文实验结论

论文在 Switch-Base 和 Switch-Large 上进行了实验，任务包含摘要和问答等。

主要结论如下：

- 相比 `DeepSpeed-Offload`，平均吞吐提升 `2.74x`。
- 相比 `Pre-gated MoE`，平均吞吐提升 `2.22x`。
- 相比 `MoE-Infinity`，平均吞吐提升 `1.55x`。

在显存方面：

- Diff-MoE 的显存占用高于最朴素的 offload 方案，因为它确实会在 GPU 上保留更多有价值的 experts。
- 但整体仍只有 `No-Offload` 的约 `16%`。

在效率方面：

- 它的 memory efficiency 更高。
- 也就是说，在有限 GPU memory 下，它能换来更高吞吐。

预测器方面：

- batch size 为 1 时，top-1 预测准确率约为 `56.3%`。
- batch size 大于等于 `16` 后，top-1 和 top-2 准确率都能稳定在 `90%+`。

### 1.6 这篇论文的优点

#### 1.6.1 抓住了真正的系统瓶颈

它不是只从“MoE 很 sparse”这个角度讲故事，而是非常明确地把问题落在：

- PCIe 通信
- cache hit rate
- communication-computation overlap

这些都是真正影响 serving 吞吐的关键点。

#### 1.6.2 方法工程可实现性强

三级缓存和 priority policy 都比较直观：

- 容易实现
- 易于调参
- 也便于和现有 offloading 系统集成

#### 1.6.3 实验目标明确

论文没有只报 throughput，还同时看了：

- cache hit rate
- memory consumption
- memory efficiency
- predictor accuracy

这使得方法收益来源比较清晰。

### 1.7 这篇论文的局限

#### 1.7.1 依赖 locality

如果某类模型或任务下的 expert activation 足够分散：

- global locality 不明显
- temporal locality 也弱

那么 Diff-MoE 的收益会打折。

#### 1.7.2 需要离线信息和超参数

它依赖：

- offline 统计 globally hot experts
- priority 相关超参数
- cache size / threshold 等配置

因此系统调优成本会高于纯 on-demand baseline。

#### 1.7.3 predictor 增加了系统复杂度

尽管 predictor 很轻量，但它仍然是一个额外模块：

- 需要训练或准备
- 需要和推理流程协同

#### 1.7.4 模型覆盖面有限

论文主要验证的是 Switch 系列，不代表对所有现代 MoE 架构都能原样成立。

---

## 2. 论文设计与仓库代码对应

这一节的目标是把论文中的术语和仓库中的实现一一对应起来。

需要先强调一个重要事实：

- 仓库里已经较完整地体现了“差分缓存 + priority-driven replacement + offline hot experts 初始化”。
- 但论文中提到的 GRU predictor 训练和在线预测链路，在当前仓库中没有看到完整公开实现。

所以阅读代码时，应该区分：

1. 论文中提出的完整系统设计。
2. 仓库中当前已经落地的那一部分。

### 2.1 总入口：decoder 中初始化 fetcher context

论文里，Diff-MoE 的缓存与取数逻辑是围绕 decoder MoE layer 运作的。

仓库中的总入口在：

- `src/fastertransformer/models/t5/T5Decoder.cc`

关键位置：

- `ffn_layer_->initFetcherContext(...)`
- `ffn_layer_->set_layer("decoder::layer", l, moe_layer_index_)`

这些代码说明：

- 系统会为 FFN/MoE 层初始化 fetcher context。
- 系统知道当前 decoder layer 是不是 MoE layer。
- 系统也知道“下一个 MoE layer”是谁。

这与论文中“当前层计算 + 下一层 expert 预取”的思路一致。

### 2.2 FfnLayer 是论文机制进入算子的主要接口

真正把缓存、priority 和 expert fetch 送进 MoE 计算核心的是：

- `src/fastertransformer/layers/FfnLayer.cc`

在这个文件里，`run_moe_fc(...)` 的调用携带了很多和 Diff-MoE 论文直接相关的参数：

- `expert_priority`
- `expert_in_cache`
- `cache_size`
- `fix_cache_size`
- `max_val`
- `threshold`
- `dec_in_cache`
- `dec_out_cache`
- `use_cache`

这说明论文里的关键运行时状态和超参数，已经直接进入了 MoE FC runner。

### 2.3 论文中的 HPC / MPC / LPC 在代码里的对应

代码中没有直接把缓存命名为 `HPC`、`MPC`、`LPC`，但语义上是可以对应起来的。

#### 2.3.1 HPC 对应什么

最接近 `HPC` 的是：

- `top_k_experts`
- `fix_cache_size`

其中：

- `top_k_experts` 表示离线挑出的每层热点 expert 列表。
- `fix_cache_size` 表示总 cache 中固定保留的一部分。

脚本里还有注释直接说明：

- `(cache_size, fix_cache_size) i.e., (HPC+MPC, HPC)`

这与论文里“每层固定放 globally hot experts 的高优先级缓存”高度一致。

#### 2.3.2 MPC 对应什么

最接近 `MPC` 的是：

- `cache_size - fix_cache_size`

也就是每层 cache 中那部分可动态替换的区域。

这部分：

- 会随着 priority 动态变化
- 会引入新变热的 experts
- 会把冷掉的 experts 淘汰出去

#### 2.3.3 LPC 对应什么

最接近 `LPC` 的是 fetcher 内部的 working / dst buffer 以及临时拉取逻辑。

在 `fetcher.cc` 里可以看到：

- 当前 expert 会先被拉到 working buffer
- 同步后会和 dst buffer 交换

它体现的是一个临时 staging 区，而不是长期 per-layer 常驻 cache。

这与论文里“共享低优先级缓存，临时容纳当前层/预取 experts”的语义一致。

### 2.4 Offline globally hot experts 的落地方式

论文中说：

- globally hot experts 在 offline 阶段识别出来
- 然后固定放进 HPC

仓库里的对应实现方式是：

- `scripts/model_configs.json`

这里保存了每层的 `cache_init_experts`。

在：

- `scripts/eval_cache.py`

中，这些离线列表会被读取并处理为 `top_k_experts`，再写入配置，供 C++ 侧使用。

这意味着：

- 仓库当前不是“在线动态发现全局热点”
- 而是“先把离线得到的热点 expert 列表塞进运行配置”

这正符合论文的 offline initialization 思路。

### 2.5 Priority-driven replacement 在代码里的直接对应

论文最核心的算法之一是 priority-driven replacement。

这一点在：

- `src/fastertransformer/utils/fetcher.cc`

里有非常直接的实现。

关键步骤如下。

#### 2.5.1 先收集当前活跃 experts

代码里先把当前 step 的 `permuted_experts` 拷到 host，并做去重，得到 active expert set。

这对应论文里：

- gating network 选出当前 layer 激活的 experts

#### 2.5.2 更新 priority

代码中的更新规则和论文高度一致：

- 当前被激活的 expert：`+1.0`
- 未激活但仍在 cache 中：`-dec_in_cache`
- 未激活且不在 cache 中：`-dec_out_cache`

这对应论文里的：

- `Δinc`
- `Δdec_in`
- `Δdec_out`

#### 2.5.3 用 threshold 判断是否值得保留

代码中会根据 `threshold` 判断：

- 哪些 expert 应该保留
- 哪些 expert 应该被替换

这对应论文里：

- priority 超过阈值则视为 locally hot

#### 2.5.4 通过 cache_move 做晋升

当某个新的高优先级 expert 应该进入 cache 时，代码会调用 `cache_move(...)`。

这可以理解为论文里：

- 从临时区域向 MPC 的 promotion

### 2.6 Cache hit 路径的代码体现

论文里一个重要收益来源是：

- 减少重复 host-to-GPU expert migration

仓库中，当 expert 已经在 cache 中时：

- 不再从 host 拉取
- 而是直接做 GPU 内部复制

同时：

- `cache_hit_num` 会增加

这与论文中 cache hit rate 的实验指标直接对应。

### 2.7 Fetch mode 与 baseline 的对应

论文对比了多个 baseline，仓库在实验脚本里也做了明确区分。

文件：

- `scripts/eval_cache.py`

对应关系大致如下：

- `GPU-only`：不做 offload
- `DeepSpeed`：decoder 使用 `FETCH_ON_DEMAND`
- `Pre-gated`：decoder 使用 `PREFETCH`
- `Diff-MoE`：decoder 使用 `PREFETCH`，并额外启用 MoE cache

这说明：

- Diff-MoE 不是简单等于 prefetch
- 它是在 prefetch 基础上再叠加 differential cache 与 priority management

### 2.8 配置是如何进入 C++ 运行时的

实验脚本会把以下参数写入 `cpp_config.ini`：

- `cache_size`
- `use_moe_cache`
- `top_k_experts`
- `fix_cache_size`
- `max_val`
- `threshold`
- `dec_in_cache`
- `dec_out_cache`

然后在：

- `src/fastertransformer/utils/config.h`

中读取，并存入全局配置：

- `cache_size`
- `use_moe_cache`
- `fix_cache_size`
- `max_val`
- `threshold`
- `dec_in_cache`
- `dec_out_cache`
- `top_k_experts`

这意味着论文中的高层策略并不是“写死在某个算子里”，而是：

- 先经由 Python benchmark 脚本参数化
- 再通过 ini 配置进入 C++ runtime

这对做复现实验和调参很有帮助。

### 2.9 Predictor：论文里有，仓库里未完整公开

这是当前阅读代码时最需要小心的一点。

论文第 6 节明确说：

- 使用 GRU 预测下一层 experts
- 根据 batch 大小只预取 1 到 2 个 experts

但在当前仓库里，我没有找到：

- GRU 模型定义
- predictor 训练脚本
- predictor 在线推理入口

这意味着更准确的判断是：

- 论文中的完整 predictor 设计，在当前仓库里没有完整公开。
- 仓库重点公开和落地的是差分缓存、优先级更新、离线热点 expert 初始化以及相关 benchmark。

因此，后续如果我们要继续往下分析，需要把两件事明确分开：

1. 论文原始设计里 predictor 应该如何工作。
2. 当前仓库里实际已经支持了哪些部分。

---

## 3. 当前阶段可得出的结论

基于论文阅读与代码检查，可以先得到以下结论。

### 3.1 论文主线已经在仓库中有较强对应

仓库已经较明确地实现了：

- per-layer offline hot expert 初始化
- 差分式缓存容量划分
- priority-driven replacement
- cache hit / miss 统计
- 基于 fetch mode 的 baseline 对比

### 3.2 代码中的 HPC / MPC 是显式参数化的

尤其是：

- `cache_size`
- `fix_cache_size`

这两个参数直接体现了论文中：

- 总 cache 容量
- 固定高优先级 cache 容量

### 3.3 predictor 需要单独对待

如果后续要继续做深入解读或代码改造，不能默认：

- “论文里有 predictor”
- “仓库里就已经完整实现 predictor”

目前更像是：

- predictor 在论文中完整存在
- 仓库开源版本主要公开了缓存系统主干

---

## 4. 下一步建议

基于本文档，后续最值得继续做的工作有三条路线。

### 4.1 路线 A：继续做更细的代码对照

目标：

- 按照一次 decoder step 的真实调用链
- 从 `T5Decoder` -> `FfnLayer` -> `FetcherContext` -> cache update
- 画出运行时数据流

这会帮助我们真正看懂“每一步 expert 是怎么被加载、命中、提升、淘汰的”。

### 4.2 路线 B：补齐 predictor 现状

目标：

- 继续检查仓库是否存在未显式命名的 predictor 逻辑
- 或者确认 predictor 确实未开源

这有助于判断论文与开源仓库之间的差距。

### 4.3 路线 C：结合实验配置理解论文图表

目标：

- 把 `eval_cache.py` 中的参数 sweep
- 与论文图 7 到图 14 一一对应

这样可以帮助后续直接复现实验结果。

---

## 5. 简短总结

一句话概括：

Diff-MoE 的本质不是“再做一个 cache”，而是把 MoE expert 的冷热分层、生命周期和迁移时机都系统化管理起来，从而在大 batch host-GPU 推理里尽量减少无效搬运，并把真正必要的搬运藏到计算后面。

在当前仓库中，这套思想已经有较强代码落地，尤其是差分缓存和 priority policy；但论文中 predictor 的完整实现，在开源代码里暂时没有看到完整公开版本。

---

## 6. 单次 Decoder Step 的运行时调用链

这一节把论文中的概念按实际执行顺序串起来，回答一个更具体的问题：

- 在一次 decoder 的某个 MoE layer 执行时，expert 到底是如何被选出、加载、命中、晋升和使用的。

### 6.1 入口：T5Decoder 进入某一层

对应文件：

- `src/fastertransformer/models/t5/T5Decoder.cc`

当 decoder 跑到第 `l` 层时，会先判断当前层是不是 MoE layer。

如果是 MoE layer，则会：

1. 构造 `ffn_input_tensors`
2. 构造 MoE 专用输出张量：
   - `ffn_output`
   - `expert_scales`
   - `expanded_source_row_to_expanded_dest_row`
   - `expert_for_source_row`
3. 查找“下一个 MoE layer”的权重
4. 调用 `ffn_layer_->set_layer("decoder::layer", l, moe_layer_index_)`
5. 调用 `ffn_layer_->forward(...)`

这一步非常关键，因为它把论文中的两个上下文同时准备好了：

- 当前层的 MoE 计算上下文
- 下一层的 MoE 预取上下文

### 6.2 给 FfnLayer 注入“下一层权重”

在进入 `ffn_layer_->forward(...)` 前，代码会先在后续层中寻找下一个 MoE layer：

- 如果找到，就把其 `ffn_weights` 存到 `ffn_layer_->ffn_weights_of_the_next_moe_layer_`

这意味着：

- 当前层不是只知道“我现在要算什么”
- 还知道“下一个 MoE layer 的 expert 权重在哪”

这正是论文中“next-layer prefetch”能够成立的前提。

### 6.3 FfnLayer：把运行时状态送进 MoE FC runner

对应文件：

- `src/fastertransformer/layers/FfnLayer.cc`

在 `FfnLayer::forward(...)` 中，如果启用了 MoE：

1. 先计算 gating logits
2. 准备 `fetcher_context_`
3. 调用 `moe_fc_runner_->setFetcherContext(fetcher_context_.get())`
4. 调用 `run_moe_fc(...)`

`run_moe_fc(...)` 会收到以下与 Diff-MoE 强相关的参数：

- `expert_priority`
- `expert_in_cache`
- `cache_size`
- `fix_cache_size`
- `max_val`
- `threshold`
- `dec_in_cache`
- `dec_out_cache`
- `use_cache`

所以从这一层开始，论文里的缓存与优先级逻辑就已经正式进入执行路径。

### 6.4 FetcherContext 的双缓冲机制

对应文件：

- `src/fastertransformer/utils/fetcher.h`
- `src/fastertransformer/utils/fetcher.cc`

`FetcherContext` 内部维护了两套 GPU 缓冲：

- `*_working`
- `*_dst`

它们的作用大致是：

- `working`：当前正在异步搬运 expert 权重时使用的工作区
- `dst`：当前真正提供给 MoE kernel 消费的稳定区

执行顺序大致是：

1. `fetch()` 把本次需要的 experts 异步加载到 `working`
2. `sync()` 等待异步任务完成
3. `sync()` 中把 `working` 和 `dst` 交换
4. 后续 kernel 读取 `dst`

这就是典型的双缓冲设计，它让“搬运下一批数据”和“使用上一批已准备好的数据”可以解耦。

### 6.5 fetch() 的第一步：确定本轮 active experts

在 `FetcherContext::fetch(...)` 中，首先会：

1. 把 `permuted_experts` 从 GPU 拷回 CPU
2. 用 `std::unique` 去重
3. 得到当前 layer 当前 step 的 active expert set

这一步对应论文中的：

- gating network 决定当前被激活的 experts

注意这里的 active experts 是“本轮真正要参与计算”的 experts 集合，不是离线热点列表。

### 6.6 fetch() 的第二步：决定每个 expert 从哪里来

对每个 active expert，代码会做两件事：

1. 判断当前是在为“本层”取数，还是在为“下一层”预取
2. 判断这个 expert 是否已经在 per-layer cache 中

具体体现为：

- `prefetch ? next_weight_src_ : current_weight_src_`
- 在 `expert_in_cache[...]` 中查找 expert 是否已存在

如果命中 cache：

- 直接做 `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice, ...)`
- 这是一条 GPU 内部复制路径

如果没有命中：

- 调用 `GroupedMemoryArena::allocate(...)`
- 从 host/source 区搬运到当前 `working` buffer

这正对应论文中的：

- 命中时避免 host-to-GPU 重传
- miss 时才真正触发 expert migration

### 6.7 fetch() 的第三步：更新 priority

在启用 cache 的 decoder 场景下，代码会进入：

- `Priority-driven cache policy`

这里会发生三类更新：

1. 当前激活 expert：
   - `expert_priority += 1.0`
2. 未激活但在 cache 里：
   - `expert_priority -= dec_in_cache`
3. 未激活且不在 cache 里：
   - `expert_priority -= dec_out_cache`

这和论文的设计完全一致。

直觉上可以理解为：

- 现在被用到的 expert 升温
- 还占着 cache 但没用到的 expert 快速降温
- 不在 cache 中的冷 expert 缓慢降温

### 6.8 fetch() 的第四步：决定谁进入 MPC

更新完 priority 后，代码会：

1. 按 priority 对本轮 active experts 排序
2. 对 cache 中动态部分排序
3. 从低优先级 cache 项中选择可被替换者
4. 如果新 expert 的 priority 超过阈值且不在 cache 中，就写入 cache

这个过程里：

- `fix_cache_size` 之前的部分不参与普通替换
- `fix_cache_size` 之后的部分是动态可替换区域

因此可以直接理解为：

- 前半部分约等于论文中的 `HPC`
- 后半部分约等于论文中的 `MPC`

### 6.9 cache_move() 的意义

当某个新 expert 应该进入 cache 时，代码不会简单只改一个索引。

它还会调用：

- `GroupedMemoryArena::cache_move(...)`

把与该 expert 对应的权重块移动到 cache 对应位置。

这一步很重要，因为：

- cache 的“晋升”不只是逻辑标签变化
- 还伴随着实际 GPU 内存布局的更新

也就是说，论文里的 “Promote(LPC -> MPC)” 在代码里并不是抽象操作，而是实实在在的数据拷贝与位置调整。

### 6.10 sync()：完成本轮搬运并切换缓冲区

在 `FetcherContext::sync()` 中，代码会：

1. 等待所有 future 完成
2. `cudaStreamSynchronize(stream)`
3. 交换 `working` 和 `dst`

交换完成后：

- 当前轮准备好的 experts 会从 `working` 变成新的 `dst`
- 后续 MoE kernel 使用的是 `dst` 区中的数据

这是整个异步加载流程的收口点。

### 6.11 get_weights()：MoE kernel 消费准备好的 experts

在 `get_weights(...)` 中，`FetcherContext` 会把以下内容暴露给实际计算：

- `num_active_experts`
- `fc1_expert_weights = intermediate_dst_`
- `fc2_expert_weights = output_dst_`

这说明：

- 真正被 `run_moe_fc(...)` 消费的是 `dst` 区
- 而不是正在搬运中的 `working` 区

所以运行时形成了一个明确的数据流：

1. gating 选出 active experts
2. fetcher 把对应权重装入 `working`
3. `sync()` 之后 `working -> dst`
4. kernel 从 `dst` 中读取权重执行本轮 MoE FFN

### 6.12 当前仓库里“next-layer prefetch”到了什么程度

从调用链上看，当前仓库明确已经具备以下基础设施：

- 知道下一个 MoE layer 的权重地址
- 知道当前层和下一层的 layer name
- fetcher 区分 `current_weight_src_` 和 `next_weight_src_`
- 支持 `prefetch` 模式

所以“为下一层准备权重”这件事在底层取数框架里是有接口支撑的。

但需要继续谨慎区分的是：

- “支持 next-layer prefetch” 不等于 “完整公开了论文里的 GRU predictor”

当前仓库更像是：

- 预取通路是有的
- 但 predictor 的完整训练与在线决策逻辑没有完整公开出来

### 6.13 一次 MoE decoder layer 的简化执行序列

把上面所有步骤压缩成一条主线，可以写成：

1. `T5Decoder` 判断当前层是 MoE layer
2. 查找下一个 MoE layer 的 `ffn_weights`
3. 调用 `ffn_layer_->set_layer(...)`
4. 调用 `ffn_layer_->forward(...)`
5. `FfnLayer` 计算 gating，并调用 `run_moe_fc(...)`
6. `FetcherContext::fetch(...)` 识别当前 active experts
7. 对每个 active expert：
   - 命中 cache 则走 GPU 内复制
   - miss 则从 host/source 加载到 `working`
8. 更新 `expert_priority`
9. 根据 priority 和阈值决定是否把 expert 提升进动态 cache
10. `sync()` 等待搬运完成并交换 `working/dst`
11. kernel 从 `dst` 中读取 expert weights 完成 MoE FC
12. `finalize_moe_routing_kernelLauncher(...)` 汇总各 expert 输出

这条链路基本就是论文里：

- 激活 expert
- differential cache
- priority-driven replacement
- overlap 通信与计算

在仓库中的运行时落地形态。

---

## 7. `expert_priority` 与 `expert_in_cache` 的内存布局

这一节专门回答两个问题：

1. `expert_priority` 在内存里到底怎么排布。
2. `expert_in_cache` 在内存里到底怎么排布。

这两者决定了论文中的 `HPC/MPC` 在代码里究竟以什么形式存在。

### 7.1 初始化位置

这两个数组在 decoder 主循环开始前初始化，位置在：

- `src/fastertransformer/models/t5/T5Decoding.cc`

核心代码形态是：

- `float expert_priority[(num_layer_/2)*expert_num_]`
- `int expert_in_cache[(num_layer_/2)*moe_cache_size]`

这里直接暴露了两个事实：

1. 它们只按 `num_layer_/2` 分配，而不是按 `num_layer_` 分配。
2. 它们只为 decoder 侧的 MoE layers 建状态。

这与当前项目基于 T5 decoder 的 MoE 实现方式一致：

- decoder 中实际参与 Diff-MoE 缓存管理的 MoE layer 数量是 `num_layer_/2`

### 7.2 `expert_priority` 的逻辑形状

`expert_priority` 的逻辑形状可以理解成：

- `[decoder_moe_layer_count][expert_num]`

其中：

- `decoder_moe_layer_count = num_layer_ / 2`
- `expert_num` 是每个 MoE layer 的专家数，比如常见是 `128`

所以线性索引规则是：

- `expert_priority[layer * expert_num_ + expert_id]`

这意味着每个 decoder MoE layer 都有一整段连续的 priority 区域。

例如当：

- `expert_num = 128`
- `decoder_moe_layer_count = 6`

那么：

- 第 0 个 decoder MoE layer 的 priority 在 `[0, 127]`
- 第 1 个 decoder MoE layer 的 priority 在 `[128, 255]`
- 第 2 个 decoder MoE layer 的 priority 在 `[256, 383]`

以此类推。

### 7.3 `expert_in_cache` 的逻辑形状

`expert_in_cache` 的逻辑形状可以理解成：

- `[decoder_moe_layer_count][moe_cache_size]`

线性索引规则是：

- `expert_in_cache[layer * moe_cache_size + slot]`

这表示：

- 每个 decoder MoE layer 拥有一段固定长度的 cache slot 区间。
- 每个 slot 里存的是一个 expert ID。

注意这里存储的不是权重地址，也不是全局索引，而是：

- 当前 layer 内的 expert 编号

例如当：

- `moe_cache_size = 3`
- `decoder_moe_layer_count = 6`

那么：

- 第 0 层 cache 段是 `expert_in_cache[0..2]`
- 第 1 层 cache 段是 `expert_in_cache[3..5]`
- 第 2 层 cache 段是 `expert_in_cache[6..8]`

### 7.4 初始化时的含义

在 `T5Decoding.cc` 中，初始化逻辑是：

1. 先从 `GlobalConfig::instance().top_k_experts` 读出每层离线热点 experts。
2. 对每层的这批 experts：
   - 把 `expert_priority[layer * expert_num + expert_id]` 设为 `max_val`
   - 把 `expert_in_cache[layer * moe_cache_size + j]` 设为对应的 expert ID

这说明初始化语义是：

- `top_k_experts` 不只是“推荐列表”
- 它们在程序启动时就同时成为：
  - 初始 cache 的内容
  - 初始最高优先级 expert

这和论文中的“offline globally hot experts 放入 HPC”是基本一致的。

### 7.5 `HPC` 和 `MPC` 在这两个数组中的体现

这两个数组本身不直接区分 `HPC` 和 `MPC`，区分是靠 slot 范围实现的。

具体来说：

- `expert_in_cache[layer][0 : fix_cache_size]`
  - 逻辑上相当于 `HPC`
  - 这些 slot 不参与普通替换

- `expert_in_cache[layer][fix_cache_size : moe_cache_size]`
  - 逻辑上相当于 `MPC`
  - 这些 slot 会参与动态排序和替换

而 `expert_priority` 则是这两个区域共享的评分表。

可以把它理解为：

- `expert_in_cache` 决定“哪些 experts 当前在 cache 里”
- `expert_priority` 决定“这些 experts 热不热、该不该留”

### 7.6 当前实现中的一个重要现实：初始化并不只填满 HPC

从代码看，初始化时会把 `top_k_experts[layer][0..moe_cache_size-1]` 全部塞进 `expert_in_cache`。

这意味着：

- 初始化不是只填 `fix_cache_size` 个 HPC slot
- 而是直接把整段 cache 都用 `top_k_experts` 填满

换句话说，当前实现更接近：

- `top_k_experts` 提供的是整个 cache 的初始种子
- 其中前 `fix_cache_size` 个 slot 是固定区
- 后面的 slot 则是“初始被填充的动态区”

这比论文里的抽象描述更具体，也更工程化。

### 7.7 `fetcher.cc` 中如何索引这两个数组

在 `fetcher.cc` 中，当前层索引会被映射成：

- `layer = prefetch ? (layer_num + 2) / 2 : layer_num / 2`

这说明：

- 原始 decoder layer 编号是完整层号
- 只有奇数层或者特定 MoE 层才映射到 decoder MoE layer index
- 实际访问 `expert_priority` 和 `expert_in_cache` 时，用的是压缩后的 decoder-MoE-layer 下标

之后对数组的访问主要有两种：

#### 7.7.1 priority 访问

- `expert_priority[layer * 128 + expert_id]`

这里代码里把 `128` 写死了，而不是统一写成 `expert_num_`。

这说明当前实现默认目标模型的 expert 数就是 `128`，至少 priority policy 这部分是这样写的。

#### 7.7.2 cache 访问

- `expert_in_cache[layer * cache_size + slot]`

这里 `cache_size` 是动态配置值，因此 cache slot 数量是可调的。

### 7.8 为什么说 `expert_in_cache` 是“每层独立 cache”

判断 expert 是否命中 cache 的代码是：

- 在当前层对应的 `expert_in_cache[layer * cache_size ... (layer+1)*cache_size)` 范围里查找

这说明：

- 一个层的 cache 不会和另一个层共用 slot
- 不同层之间没有统一全局 cache 索引

因此当前实现更接近论文中的：

- per-layer `HPC`
- per-layer `MPC`

而不是像某些 baseline 那样使用单一全局 cache。

### 7.9 当前实现中值得注意的两个细节

#### 7.9.1 一个无效循环

初始化 `expert_in_cache` 时有一段：

- `for (int j = moe_cache_size; j < moe_cache_size; j++)`

这个循环永远不会执行。

因此：

- `expert_in_cache` 不会在这一步被额外补 `-1`
- 它实际上完全由 `top_k_experts` 填满

这再次说明当前版本假设：

- 初始化阶段 cache 的所有 slot 都已经有合法 expert ID

#### 7.9.2 `top_k_experts` 的长度要求

因为初始化直接访问：

- `top_k_experts[layer][rank]`

所以对于每个 layer，`top_k_experts[layer]` 至少要有 `moe_cache_size` 个元素。

这也是为什么 Python 侧会在 `eval_cache.py` 中：

- 按 `cache_size` 截断或补零 `cache_init_experts`

否则 C++ 初始化时会越界。

### 7.10 `ft_decoding.py` 对这个布局的支撑

在 Python 侧：

- `top_k_experts` 会先被整理成二维数组
- 然后在 `load_from_model(..., use_moe_cache=True, top_k_experts=...)` 中按 layer 迭代

它会把每层 `top_k_experts` 对应的 expert 权重直接加载到 GPU：

- `expert_{value}.wi.weight`
- `expert_{value}.wo.weight`

这进一步说明：

- `top_k_experts` 不只是 runtime 的逻辑配置
- 还是初始 cache resident weights 的装载清单

### 7.11 用一个具体例子理解

假设：

- decoder MoE layer 数量是 `6`
- 每层 expert 数量是 `128`
- `cache_size = 3`
- `fix_cache_size = 1`

那么：

- `expert_priority` 的形状逻辑上是 `[6][128]`
- `expert_in_cache` 的形状逻辑上是 `[6][3]`

对第 2 个 decoder MoE layer 来说：

- 它的 priority 区间是 `expert_priority[2*128 : 3*128)`
- 它的 cache 区间是 `expert_in_cache[2*3 : 3*3)`

如果这一层初始化的 `top_k_experts[2] = [52, 0, 33]`

那么：

- slot 0：`52`
- slot 1：`0`
- slot 2：`33`

其中：

- slot 0 对应固定区 `HPC`
- slot 1 和 slot 2 对应动态区 `MPC`

同时：

- `priority[2][52] = max_val`
- `priority[2][0] = max_val`
- `priority[2][33] = max_val`

这就是当前代码里“缓存内容”和“优先级状态”之间的同步初始化关系。

### 7.12 当前可以得出的更精确结论

综合这些实现细节，当前仓库中的缓存布局可以更精确地描述为：

1. 每个 decoder MoE layer 都有独立的一段 cache slot。
2. 每个 decoder MoE layer 都有独立的一段 expert priority 表。
3. `fix_cache_size` 之前的 slot 逻辑上是 `HPC`。
4. `fix_cache_size` 之后的 slot 逻辑上是 `MPC`。
5. 初始化时整个 cache 都由 `top_k_experts` 填满，而不是只填 `HPC`。
6. `top_k_experts` 同时决定：
   - 初始 resident experts
   - 初始最大优先级 experts

所以如果用一句最贴近代码的话来描述：

当前实现并不是“先只建一个很小的 HPC，再慢慢长出 MPC”，而是“先用 offline hot experts 给整段 per-layer cache 打种子，其中前 `fix_cache_size` 个 slot 固定保留，剩余 slot 再在运行时动态替换”。
