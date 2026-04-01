# DeepSpeed Benchmark Guide

## Purpose

This document explains what the `DeepSpeed` benchmark means in this project, how it is wired into the codebase, and how to reproduce its behavior.

The key point is:

- In this repository, `DeepSpeed` is a **benchmark label**, not a real DeepSpeed runtime integration.
- The implementation is closer to a **fetch-on-demand expert loading baseline**.
- It is **not** a direct implementation of DeepSpeed ZeRO-Offload.

## Bottom Line

The benchmark named `DeepSpeed` is introduced in [scripts/eval_cache.py](/D:/moe_offloading/Diff-MoE/scripts/eval_cache.py) by choosing:

- `encoder_fetcher_mode = 1`
- `decoder_fetcher_mode = 1`

According to [src/fastertransformer/utils/config.h](/D:/moe_offloading/Diff-MoE/src/fastertransformer/utils/config.h:10), `1` means `FETCH_ON_DEMAND`.

So the project uses `DeepSpeed` to represent the following idea:

- experts are not permanently resident on GPU
- when an expert is needed, its weights are fetched on demand
- no Diff-MoE priority-driven cache is used in this baseline
- no real DeepSpeed engine is invoked

## Where `DeepSpeed` Is Selected

The benchmark method is defined in [scripts/eval_cache.py](/D:/moe_offloading/Diff-MoE/scripts/eval_cache.py:108):

```python
if method == "GPU-only":
    encoder_fetcher_mode = "0"
    decoder_fetcher_mode = "0"
elif method == "Pre-gated":
    encoder_fetcher_mode = "1"
    decoder_fetcher_mode = "2"
elif method == "DeepSpeed":
    encoder_fetcher_mode = "1"
    decoder_fetcher_mode = "1"
elif method == "Diff-MoE":
    encoder_fetcher_mode = "1"
    decoder_fetcher_mode = "2"
```

This gives the benchmark semantics:

- `GPU-only`: all experts stay on GPU
- `DeepSpeed`: fetch on demand
- `Pre-gated`: prefetch
- `Diff-MoE`: prefetch plus expert cache management

## Meaning of Fetch Modes

The fetch mode enum is defined in [src/fastertransformer/utils/config.h](/D:/moe_offloading/Diff-MoE/src/fastertransformer/utils/config.h:10):

```cpp
enum class FetchType {
    GPU_ONLY,
    FETCH_ON_DEMAND,
    PREFETCH
};
```

The fetcher context also documents the intended meaning in [src/fastertransformer/utils/fetcher.h](/D:/moe_offloading/Diff-MoE/src/fastertransformer/utils/fetcher.h:68):

```cpp
FetchType mode;  // 1: FETCH_ON_DEMAND
                 // 2: PREFETCH
                 // it doesn't affect the functionality, just a signal.
```

That last comment matters: the benchmark distinction is driven by how the project configures and uses its own fetcher, not by external DeepSpeed code.

## How the Fetcher Is Created

The decoder FFN layer creates a fetcher from the configured mode in [src/fastertransformer/models/t5/T5Decoder.cc](/D:/moe_offloading/Diff-MoE/src/fastertransformer/models/t5/T5Decoder.cc:118):

```cpp
ffn_layer_->initFetcherContext(
    GlobalConfig::instance().decoder_fetcher_mode,
    moe_k_,
    GlobalConfig::instance().arena_size);
```

The fetcher context itself is created in [src/fastertransformer/layers/FfnLayer.cc](/D:/moe_offloading/Diff-MoE/src/fastertransformer/layers/FfnLayer.cc:556):

```cpp
void FfnLayer<T>::initFetcherContext(FetchType mode, int moe_k, size_t arena_size) {
    if (mode != FetchType::GPU_ONLY) {
        ...
        fetcher_context_ = std::make_shared<FetcherContext<T>>(...);
    }
}
```

So:

- `GPU_ONLY` does not create the offload fetcher
- `FETCH_ON_DEMAND` and `PREFETCH` both use the same fetcher framework
- the benchmark difference comes from the selected mode and cache settings

## What the `DeepSpeed` Baseline Actually Does

At runtime, expert loading happens inside [src/fastertransformer/utils/fetcher.cc](/D:/moe_offloading/Diff-MoE/src/fastertransformer/utils/fetcher.cc:94).

The fetch path:

1. copies routed expert ids from GPU to CPU
2. computes the active experts of the current layer
3. loads those experts into the working buffer

Relevant code in [src/fastertransformer/utils/fetcher.cc](/D:/moe_offloading/Diff-MoE/src/fastertransformer/utils/fetcher.cc:100):

```cpp
check_cuda_error(cudaMemcpy(permuted_experts_, permuted_experts, sizeof(int) * num_rows_, cudaMemcpyDeviceToHost));

auto new_end        = std::unique(permuted_experts_, permuted_experts_ + num_rows_);
num_active_experts_ = new_end - permuted_experts_;
```

When cache is disabled or not hit, the fetcher allocates and loads expert weights into the working buffer from the source weights. This is the core behavior used to approximate a DeepSpeed-like on-demand offload baseline.

## Why This Is Not ZeRO-Offload

This repository does not contain a real DeepSpeed runtime path:

- no `import deepspeed`
- no `deepspeed.initialize(...)`
- no ZeRO engine construction
- no ZeRO-Offload state management

Instead, the benchmark still runs the repository's own script:

- [scripts/eval_cache.py](/D:/moe_offloading/Diff-MoE/scripts/eval_cache.py:205) calls
  [examples/pytorch/t5/perf_benchmark.py](/D:/moe_offloading/Diff-MoE/examples/pytorch/t5/perf_benchmark.py)

The command is built as:

```python
python /workspace/FasterTransformer/examples/pytorch/t5/perf_benchmark.py ...
```

So the project is benchmarking **its own MoE runtime under different fetch policies**, not switching to the actual DeepSpeed framework.

## Why `Megatron-DeepSpeed` Appears in the Command

The benchmark command passes:

- `--model_type Megatron-DeepSpeed`

This appears in [scripts/eval_cache.py](/D:/moe_offloading/Diff-MoE/scripts/eval_cache.py:212).

That does **not** mean the code is running the DeepSpeed engine.

In [examples/pytorch/t5/perf_benchmark.py](/D:/moe_offloading/Diff-MoE/examples/pytorch/t5/perf_benchmark.py:352), `Megatron-DeepSpeed` is used to parse checkpoint structure:

- `num_experts`
- `moe_layer_index`
- encoder and decoder config values

So `Megatron-DeepSpeed` here means:

- the checkpoint/model layout follows Megatron-DeepSpeed conventions

It does **not** mean:

- inference is being executed by DeepSpeed itself

## Interaction with MoE Cache

Another important detail is in [scripts/eval_cache.py](/D:/moe_offloading/Diff-MoE/scripts/eval_cache.py:364):

```python
use_moecache = True if method == "Diff-MoE" else False
```

This means:

- `Diff-MoE` enables the priority-driven expert cache
- `DeepSpeed` does not

So in this codebase, the `DeepSpeed` baseline is conceptually:

- fetch on demand
- no priority-based expert cache
- no fixed high-priority cache
- no temporal locality update used for cache replacement

That is why it serves as a simpler offloading baseline against the full Diff-MoE method.

## Reproduction Path

To reproduce the `DeepSpeed` benchmark idea in this repository:

1. Use [scripts/eval_cache.py](/D:/moe_offloading/Diff-MoE/scripts/eval_cache.py).
2. Set `method = "DeepSpeed"`.
3. Ensure `use_moecache = False`.
4. Run the generated command, which calls
   [examples/pytorch/t5/perf_benchmark.py](/D:/moe_offloading/Diff-MoE/examples/pytorch/t5/perf_benchmark.py).

The effective configuration is:

- `encoder_fetcher_mode = FETCH_ON_DEMAND`
- `decoder_fetcher_mode = FETCH_ON_DEMAND`
- `use_moe_cache = False`

In code terms, this reproduces the benchmark's intended idea:

- experts are loaded only when they are routed to
- there is no Diff-MoE expert caching policy on top

## Suggested Mental Model

For reading the paper and this code together, it is best to interpret the benchmark as:

- `DeepSpeed` in the paper table:
  an on-demand expert offloading baseline
- `DeepSpeed` in this repository:
  the repository's own `FETCH_ON_DEMAND` implementation used to emulate that baseline

So the correct statement is:

- the project **benchmarks against a DeepSpeed-like offloading policy**

The incorrect statement is:

- the project **implements DeepSpeed ZeRO-Offload internally**

## One-Sentence Summary

In this repository, `DeepSpeed` is a **fetch-on-demand MoE weight loading baseline implemented with the project's own fetcher**, not a true integration of DeepSpeed ZeRO-Offload.
