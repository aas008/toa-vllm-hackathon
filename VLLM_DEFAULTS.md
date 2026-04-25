# vLLM v0.19.1 Default Configuration

**Important**: vLLM V1 (default since v0.18) enables many features implicitly.
Passing these flags via CLI does nothing — they are already active.

## Features ALREADY ENABLED by Default (do NOT test these)

| Feature | CLI Flag | Status in V1 | Notes |
|---------|----------|-------------|-------|
| Chunked Prefill | `--enable-chunked-prefill` | **ENABLED** (implicit) | `max_num_batched_tokens=2048` auto-set |
| Prefix Caching (APC) | `--enable-prefix-caching` | **ENABLED** (always on) | Automatic Prefix Caching in V1 |
| CUDA Graphs | (no flag) | **ENABLED** | `enforce_eager=False` by default |
| Continuous Batching | (no flag) | **ENABLED** (always on) | Iteration-level scheduling |
| Async Scheduling | (no flag) | **ENABLED** | Log: "Asynchronous scheduling is enabled" |

**Consequence**: Creating experiment pods with `--enable-chunked-prefill` or
`--enable-prefix-caching` produces identical behavior to baseline. Any
performance difference is noise from cold prefix cache on new pods.

## Actual Defaults (what to tune FROM)

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `--gpu-memory-utilization` | 0.9 | 0.80-0.95 | Fraction of GPU memory for KV cache |
| `--max-num-seqs` | auto | 1-1024 | Max concurrent sequences |
| `--max-num-batched-tokens` | 2048 (V1) | 256-32768 | Token budget per iteration |
| `--max-model-len` | auto (131072 for Llama-3.2) | model-dependent | Max context length |
| `--scheduling-policy` | fcfs | fcfs/priority | Request scheduling order |
| `--optimization-level` | 2 | 0-2 | Compilation optimization level |
| `--dtype` | auto | auto/float16/bfloat16 | Model weight dtype |
| `--enforce-eager` | False | True/False | True = disable CUDA graphs |
| `--tensor-parallel-size` | 1 | 1-8 | Multi-GPU parallelism |
| `--pipeline-parallel-size` | 1 | 1-N | Multi-node parallelism |
| `--quantization` | None (auto-detected) | fp8/awq/gptq/None | Weight quantization |
| `--kv-cache-dtype` | auto | auto/fp8/fp8_e4m3/fp8_e5m2 | KV cache precision |

## What ACTUALLY Changes Performance

These are the flags worth testing in experiments:

1. **`--max-num-batched-tokens`** — Controls token budget. Higher = more throughput, higher latency
2. **`--max-num-seqs`** — Controls max concurrent requests. Higher = more batching
3. **`--gpu-memory-utilization`** — More memory = larger KV cache = more concurrent requests
4. **`--max-model-len`** — Reducing frees KV cache memory for more concurrent requests
5. **`--enforce-eager`** — Disabling CUDA graphs (True) for debugging, never for perf
6. **`--kv-cache-dtype fp8_e4m3`** — Quantize KV cache to reduce memory per token
7. **`--quantization`** — Weight quantization (if model supports it)

## Cold Cache Warning

Every new experiment pod starts with an **empty prefix cache**. The baseline
pod (which has been running and served requests) has a **warm cache**. This
means experiment pods will always show worse prefix cache hit rates until
they warm up. Account for this in comparisons.

Baseline warm cache hit rate observed: **87.5%** (with synthetic benchmark data).
Fresh pod cache hit rate: **0%** (until patterns repeat during benchmark).
