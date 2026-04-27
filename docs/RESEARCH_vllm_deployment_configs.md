# vLLM Deployment & Configuration — Deep Research Report

> Generated: 2026-04-26 | Sources: 15+ web, 20+ source files | vLLM version: latest main branch

## TL;DR

- vLLM has **268+ CLI arguments** (via `EngineArgs` + `FrontendArgs`) and **238 environment variables** — an enormous configuration surface
- The V1 engine is now default (V0 removed); prefix caching enabled by default with <1% overhead at 0% hit rate
- Key performance levers: `--gpu-memory-utilization`, `--max-num-batched-tokens`, `--kv-cache-dtype fp8`, optimization levels (`-O0` to `-O3`), and `--performance-mode`
- For H100: use FP8 quantization, DeepGEMM (`VLLM_USE_DEEP_GEMM=1`), and O2 optimization (default)
- All env vars are **cached after engine init** — must be set BEFORE importing vLLM
- YAML config files supported via `--config config.yaml` (CLI args take precedence)

---

## Table of Contents

1. [Overview & Architecture](#overview--architecture)
2. [CLI Arguments — Complete Catalog](#cli-arguments--complete-catalog)
3. [Environment Variables — Practical Guide](#environment-variables--practical-guide)
4. [Hardware-Specific Optimization](#hardware-specific-optimization)
5. [Performance Tuning Guide](#performance-tuning-guide)
6. [Deployment Recipes](#deployment-recipes)
7. [Tool Calling & Structured Output](#tool-calling--structured-output)
8. [Vision Model (VLM) Deployment](#vision-model-vlm-deployment)
9. [Data Parallelism & Scaling](#data-parallelism--scaling)
10. [Config Validation & Interactions](#config-validation--interactions)
11. [Gotchas & Pitfalls](#gotchas--pitfalls)
12. [Sources](#sources)

---

## Overview & Architecture

### Entry Points

| File | Purpose |
|------|---------|
| `vllm/entrypoints/cli/main.py` | CLI dispatcher (`vllm serve`, `vllm bench`, etc.) |
| `vllm/entrypoints/cli/serve.py` | `vllm serve` subcommand handler |
| `vllm/entrypoints/openai/api_server.py` | FastAPI/Uvicorn OpenAI-compatible server |
| `vllm/entrypoints/openai/cli_args.py` | Frontend args (HTTP, CORS, SSL, tool calling) |
| `vllm/engine/arg_utils.py` | `EngineArgs` dataclass — THE central config definition (268 fields) |

### Config Hierarchy

`VllmConfig` (root, defined in `vllm/config/vllm.py`) aggregates:

| Sub-Config | File | Purpose |
|------------|------|---------|
| `ModelConfig` | `vllm/config/model.py` | Model, tokenizer, dtype, max_model_len |
| `CacheConfig` | `vllm/config/cache.py` | KV cache, prefix caching, block size |
| `ParallelConfig` | `vllm/config/parallel.py` | TP, PP, DP, EP, distributed |
| `SchedulerConfig` | `vllm/config/scheduler.py` | Batching, chunked prefill, scheduling policy |
| `LoadConfig` | `vllm/config/load.py` | Model loading format, download dir |
| `OffloadConfig` | `vllm/config/offload.py` | CPU offloading, weight prefetch |
| `AttentionConfig` | `vllm/config/attention.py` | Attention backend selection |
| `KernelConfig` | `vllm/config/kernel.py` | MoE backend, FlashInfer autotune |
| `LoRAConfig` | `vllm/config/lora.py` | LoRA adapter serving |
| `SpeculativeConfig` | `vllm/config/speculative.py` | Speculative decoding |
| `CompilationConfig` | `vllm/config/compilation.py` | CUDA graphs, torch.compile |
| `ObservabilityConfig` | `vllm/config/observability.py` | Metrics, tracing, profiling |
| `MultiModalConfig` | `vllm/config/multimodal.py` | Vision/audio/video processing |
| `StructuredOutputsConfig` | `vllm/config/structured_outputs.py` | Guided decoding backend |
| `ReasoningConfig` | Config for reasoning parsers |
| `KVTransferConfig` | Disaggregated prefill/decode |
| `WeightTransferConfig` | Weight transfer for RL training |

### YAML Config Support

vLLM's `FlexibleArgumentParser` (`vllm/utils/argparse_utils.py:372-519`) supports YAML config files:

```bash
vllm serve --config config.yaml
# or with model in config:
vllm serve meta-llama/Llama-3.1-8B-Instruct --config config.yaml
```

**Precedence**: CLI args > config file > defaults

**YAML format**:
```yaml
# config.yaml
model: meta-llama/Llama-3.1-70B-Instruct
port: 8000
host: "0.0.0.0"
tensor-parallel-size: 4
dtype: bfloat16
gpu-memory-utilization: 0.9
max-model-len: 8192
enable-prefix-caching: true
compilation-config:
  pass_config:
    fuse_allreduce_rms: true
```

Booleans: `true` adds `--flag`, `false` omits it. Nested dicts serialized as JSON.

---

## CLI Arguments — Complete Catalog

### Model Configuration

| CLI Flag | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `Qwen/Qwen3-0.6B` | HuggingFace model name or local path |
| `--tokenizer` | str | None | Override tokenizer |
| `--tokenizer-mode` | str | `auto` | `auto`, `hf`, `slow`, `mistral`, `deepseek_v32` |
| `--dtype` | str | `auto` | `auto`, `half`, `float16`, `bfloat16`, `float`, `float32` |
| `--max-model-len` | int | None | Max context length (accepts `1k`, `2M`, `auto`) |
| `--revision` | str | None | Model git revision |
| `--code-revision` | str | None | Code revision on HF Hub |
| `--tokenizer-revision` | str | None | Tokenizer revision |
| `--trust-remote-code` | bool | False | Trust remote code from HuggingFace |
| `--hf-token` | str/bool | None | HuggingFace API token |
| `--hf-overrides` | dict | None | HuggingFace config overrides (JSON) |
| `--config-format` | str | `auto` | `auto`, `hf`, `mistral` |
| `--seed` | int | 0 | Random seed |
| `--max-logprobs` | int | 20 | Max log probabilities to return |
| `--logprobs-mode` | str | `raw_logprobs` | `raw_logprobs`, `processed_logprobs`, `raw_logits`, `processed_logits` |
| `--quantization`, `-q` | str | None | `awq`, `gptq`, `fp8`, `bitsandbytes`, `gguf`, `compressed-tensors`, etc. |
| `--enforce-eager` | bool | False | Disable CUDA graphs and torch.compile |
| `--model-impl` | str | `auto` | `auto`, `vllm`, `transformers`, `terratorch` |
| `--served-model-name` | str/list | None | Model name(s) in API responses |
| `--generation-config` | str | `auto` | Path to generation config |
| `--override-generation-config` | dict | {} | Override generation params (JSON) |
| `--enable-sleep-mode` | bool | False | Enable sleep mode (CUDA/HIP only) |
| `--disable-sliding-window` | bool | False | Disable sliding window attention |
| `--disable-cascade-attn` | bool | True | Disable cascade attention |
| `--skip-tokenizer-init` | bool | False | Skip tokenizer init |
| `--enable-prompt-embeds` | bool | False | Enable prompt embeddings input |
| `--override-attention-dtype` | str | None | Override attention dtype |
| `--allowed-local-media-path` | str | "" | Allow local media from directories |
| `--allowed-media-domains` | list | None | Allowed media URL domains |
| `--hf-config-path` | str | None | HuggingFace config path |
| `--pooler-config` | dict | None | Pooler config for embedding models |

### Model Loading

| CLI Flag | Type | Default | Description |
|----------|------|---------|-------------|
| `--load-format` | str | `auto` | `auto`, `pt`, `safetensors`, `gguf`, `bitsandbytes`, `tensorizer` |
| `--download-dir` | str | None | Model download directory |
| `--safetensors-load-strategy` | str | None | Safetensors loading strategy |
| `--model-loader-extra-config` | dict | {} | Extra model loader config (JSON) |
| `--ignore-patterns` | str/list | [] | File patterns to ignore |
| `--use-tqdm-on-load` | bool | False | Show progress bar during loading |
| `--pt-load-map-location` | str | `cpu` | Map location for PyTorch loading |

### Parallelism & Distribution

| CLI Flag | Type | Default | Description |
|----------|------|---------|-------------|
| `--tensor-parallel-size`, `-tp` | int | 1 | Tensor parallel degree |
| `--pipeline-parallel-size`, `-pp` | int | 1 | Pipeline parallel degree |
| `--data-parallel-size`, `-dp` | int | 1 | Data parallel degree |
| `--data-parallel-rank`, `-dpn` | int | None | DP rank (enables external LB) |
| `--data-parallel-start-rank`, `-dpr` | int | None | Starting DP rank for secondary nodes |
| `--data-parallel-size-local`, `-dpl` | int | None | Local DP replicas on this node |
| `--data-parallel-address`, `-dpa` | str | None | DP cluster head-node address |
| `--data-parallel-rpc-port`, `-dpp` | int | None | DP RPC port |
| `--data-parallel-backend`, `-dpb` | str | `mp` | `mp`, `ray` |
| `--data-parallel-hybrid-lb`, `-dph` | bool | False | Enable hybrid load balancing |
| `--data-parallel-external-lb`, `-dpe` | bool | False | Enable external load balancing |
| `--prefill-context-parallel-size`, `-pcp` | int | 1 | Prefill context parallelism |
| `--decode-context-parallel-size`, `-dcp` | int | 1 | Decode context parallelism |
| `--distributed-executor-backend` | str | `ray` | `ray`, `mp`, `uni`, `external_launcher` |
| `--master-addr` | str | `127.0.0.1` | Master node address |
| `--master-port` | int | 29500 | Master node port |
| `--nnodes`, `-n` | int | 1 | Number of nodes |
| `--node-rank`, `-r` | int | 0 | This node's rank |
| `--distributed-timeout-seconds` | int | None | Timeout for distributed ops |
| `--enable-expert-parallel`, `-ep` | bool | False | Expert parallelism for MoE |
| `--enable-ep-weight-filter` | bool | False | Skip non-local expert weights |
| `--all2all-backend` | str | `pplx` | `naive`, `pplx`, `deepep_*`, `mori`, etc. |
| `--moe-backend` | str | `punica` | MoE compute backend |
| `--enable-elastic-ep` | bool | False | Elastic expert parallelism |
| `--enable-eplb` | bool | False | Expert parallel load balancing |
| `--eplb-config` | dict | {} | EPLB configuration (JSON) |
| `--expert-placement-strategy` | str | `linear` | `linear`, `round_robin` |
| `--enable-dbo` | bool | False | Disaggregated batch operations |
| `--ubatch-size` | int | 0 | Micro-batch size for DBO |
| `--dbo-decode-token-threshold` | int | 256 | DBO decode threshold |
| `--dbo-prefill-token-threshold` | int | 2048 | DBO prefill threshold |
| `--disable-custom-all-reduce` | bool | False | Disable custom all-reduce |
| `--disable-nccl-for-dp-synchronization` | bool | None | Disable NCCL for DP sync |
| `--ray-workers-use-nsight` | bool | False | Enable Nsight for Ray workers |
| `--max-parallel-loading-workers` | int | None | Max parallel model loading |
| `--worker-cls` | str | `auto` | Custom worker class |
| `--worker-extension-cls` | str | `auto` | Custom worker extension |

### Memory & KV Cache

| CLI Flag | Type | Default | Description |
|----------|------|---------|-------------|
| `--gpu-memory-utilization` | float | 0.9 | GPU memory fraction (0-1) |
| `--block-size` | int | 16 | KV cache block size in tokens |
| `--kv-cache-dtype` | str | `auto` | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` |
| `--kv-cache-memory-bytes` | int | None | Manual KV cache budget (accepts `1k`, `2M`) |
| `--enable-prefix-caching` | bool | True | Automatic prefix caching |
| `--prefix-caching-hash-algo` | str | `sha256` | `sha256`, `xxhash`, etc. |
| `--num-gpu-blocks-override` | int | None | Override GPU cache blocks |
| `--calculate-kv-scales` | bool | False | Dynamic KV scale calculation |
| `--kv-cache-dtype-skip-layers` | list | [] | Layers to skip KV quantization |
| `--kv-sharing-fast-prefill` | bool | False | KV sharing optimization |
| `--kv-offloading-size` | float | None | KV offload buffer (GiB) |
| `--kv-offloading-backend` | str | `native` | `native`, `lmcache` |
| `--mamba-cache-dtype` | str | `auto` | Mamba cache dtype |
| `--mamba-ssm-cache-dtype` | str | `auto` | Mamba SSM state dtype |
| `--mamba-block-size` | int | None | Mamba block size |
| `--mamba-cache-mode` | str | `none` | `none`, `all`, `align` |

### Offloading

| CLI Flag | Type | Default | Description |
|----------|------|---------|-------------|
| `--cpu-offload-gb` | float | 0.0 | CPU offload (GiB) — UVA zero-copy |
| `--offload-backend` | str | `default` | Weight offload backend |
| `--cpu-offload-params` | set | {} | Parameters to offload |
| `--offload-group-size` | int | 8 | Layer group size for prefetch offload |
| `--offload-num-in-group` | int | 4 | Layers to offload per group |
| `--offload-prefetch-step` | int | 0 | Prefetch step |
| `--offload-params` | set | {} | Params for prefetch offload |

### Scheduling & Batching

| CLI Flag | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-num-batched-tokens` | int | dynamic | Max tokens per iteration (H100: 8192, A100: 2048) |
| `--max-num-seqs` | int | dynamic | Max concurrent sequences (H100: 1024, A100: 256) |
| `--enable-chunked-prefill` | bool | True | Split long prefills into chunks |
| `--max-num-partial-prefills` | int | 1 | Max concurrent partial prefills |
| `--max-long-partial-prefills` | int | 1 | Max long partial prefills |
| `--long-prefill-token-threshold` | int | 0 | Threshold for "long" prefill |
| `--scheduling-policy` | str | `fcfs` | `fcfs`, `priority` |
| `--scheduler-cls` | str | None | Custom scheduler class |
| `--disable-chunked-mm-input` | bool | False | Disable chunked multimodal input |
| `--scheduler-reserve-full-isl` | bool | True | Check full seq fits before admit |
| `--disable-hybrid-kv-cache-manager` | bool | None | Disable hybrid KV cache manager |
| `--async-scheduling` | bool | None | Enable async scheduling |
| `--stream-interval` | int | 1 | Streaming buffer in tokens |

### LoRA Adapters

| CLI Flag | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable-lora` | bool | False | Enable LoRA support |
| `--max-loras` | int | 1 | Max concurrent LoRA adapters |
| `--max-lora-rank` | int | 16 | Max LoRA rank (1-512) |
| `--max-cpu-loras` | int | =max_loras | LoRA adapters in CPU memory |
| `--lora-dtype` | str | `auto` | LoRA computation dtype |
| `--fully-sharded-loras` | bool | False | Fully shard LoRA layers |
| `--lora-target-modules` | list | None | Target modules for LoRA |
| `--specialize-active-lora` | bool | False | Specialize kernels per LoRA |
| `--lora-modules` | list | None | Pre-registered LoRA modules (frontend) |

### Speculative Decoding

| CLI Flag | Type | Default | Description |
|----------|------|---------|-------------|
| `--speculative-config`, `-sc` | dict | None | Full spec decode config (JSON) |

Methods: `ngram`, `medusa`, `mlp_speculator`, `draft_model`, `eagle`, `eagle3`, `suffix`, various MTP variants.

Example:
```bash
--speculative-config '{"model": "nvidia/gpt-oss-120b-Eagle3-v2", "num_speculative_tokens": 3, "method": "eagle3"}'
```

### Compilation & Optimization

| CLI Flag | Type | Default | Description |
|----------|------|---------|-------------|
| `--optimization-level` | str | `o2` | `o0` (none), `o1` (dynamo), `o2` (full), `o3` (=o2) |
| `--performance-mode` | str | `balanced` | `balanced`, `interactivity`, `throughput` |
| `--compilation-config`, `-cc` | dict | auto | Compilation config (JSON) |
| `--cudagraph-capture-sizes` | list | auto | CUDA graph batch sizes |
| `--max-cudagraph-capture-size` | int | None | Max batch for CUDA graphs |

### Attention & Kernels

| CLI Flag | Type | Default | Description |
|----------|------|---------|-------------|
| `--attention-backend` | str | None | `flash_attn`, `xformers`, `flashinfer`, etc. |
| `--attention-config`, `-ac` | dict | auto | Attention config (JSON) |
| `--enable-flashinfer-autotune` | bool | False | FlashInfer autotuning |
| `--moe-backend` | str | `punica` | MoE compute backend |

### Guided Decoding & Structured Output

| CLI Flag | Type | Default | Description |
|----------|------|---------|-------------|
| `--structured-outputs-config` | dict | auto | Backend: `xgrammar`, `guidance`, `outlines`, `lm-format-enforcer` |
| `--reasoning-config` | dict | auto | Reasoning config |
| `--reasoning-parser` | str | `default` | Reasoning parser type |

### Observability

| CLI Flag | Type | Default | Description |
|----------|------|---------|-------------|
| `--otlp-traces-endpoint` | str | None | OpenTelemetry traces endpoint |
| `--collect-detailed-traces` | list | None | `model`, `worker`, `all` |
| `--kv-cache-metrics` | bool | False | KV cache metrics |
| `--cudagraph-metrics` | bool | False | CUDA graph metrics |
| `--enable-mfu-metrics` | bool | False | Model FLOPs Utilization |
| `--enable-layerwise-nvtx-tracing` | bool | False | NVTX tracing |
| `--enable-logging-iteration-details` | bool | False | Log iteration details |

### Server / Frontend

| CLI Flag | Type | Default | Description |
|----------|------|---------|-------------|
| `--host` | str | None | Server bind address |
| `--port` | int | 8000 | Server port |
| `--uds` | str | None | Unix domain socket |
| `--api-key` | list | None | API keys (multiple allowed) |
| `--uvicorn-log-level` | str | `info` | Uvicorn log level |
| `--root-path` | str | None | FastAPI root for reverse proxy |
| `--middleware` | list | [] | Custom ASGI middleware |
| `--ssl-keyfile` | str | None | SSL key file |
| `--ssl-certfile` | str | None | SSL cert file |
| `--ssl-ca-certs` | str | None | CA certificates |
| `--enable-ssl-refresh` | bool | False | Auto-reload on cert change |
| `--allowed-origins` | list | `["*"]` | CORS origins |
| `--allowed-methods` | list | `["*"]` | CORS methods |
| `--allowed-headers` | list | `["*"]` | CORS headers |
| `--allow-credentials` | bool | False | CORS credentials |
| `--chat-template` | str | None | Jinja2 chat template |
| `--chat-template-content-format` | str | `auto` | `string`, `openai`, `auto` |
| `--enable-auto-tool-choice` | bool | False | Auto tool calling |
| `--tool-call-parser` | str | None | Tool parser (per-model) |
| `--tool-parser-plugin` | str | "" | Custom tool parser |
| `--enable-request-id-headers` | bool | False | X-Request-Id header |
| `--disable-fastapi-docs` | bool | False | Disable Swagger UI |
| `--enable-log-requests` | bool | False | Log requests |
| `--disable-log-stats` | bool | False | Disable stats logging |
| `--max-log-len` | int | None | Max prompt chars logged |
| `--enable-prompt-tokens-details` | bool | False | Prompt token details |
| `--enable-server-load-tracking` | bool | False | Server load metrics |
| `--return-tokens-as-token-ids` | bool | False | Token IDs in response |
| `--response-role` | str | `assistant` | Response role name |
| `--h11-max-incomplete-event-size` | int | 4MB | Max HTTP event size |

### Serve-Specific

| CLI Flag | Type | Default | Description |
|----------|------|---------|-------------|
| (positional) `model_tag` | str | None | Model to serve |
| `--config` | str | None | YAML config file |
| `--headless` | bool | False | No API server (worker-only) |
| `--api-server-count`, `-asc` | int | auto | API server processes |
| `--grpc` | bool | False | gRPC instead of HTTP |
| `--shutdown-timeout` | int | 0 | Graceful shutdown (0=abort) |

### Advanced / Transfer

| CLI Flag | Type | Default | Description |
|----------|------|---------|-------------|
| `--kv-transfer-config` | dict | None | Disaggregated prefill/decode |
| `--kv-events-config` | dict | None | KV events config |
| `--ec-transfer-config` | dict | None | EC transfer config |
| `--weight-transfer-config` | dict | None | Weight transfer (RLHF) |
| `--additional-config` | dict | {} | Additional custom config |
| `--fail-on-environ-validation` | bool | False | Hard-fail on unknown env vars |

---

## Environment Variables — Practical Guide

### How Env Vars Work in vLLM

All env vars defined in `vllm/envs.py` (238 total). Key mechanism:

1. **Registry**: `environment_variables` dict maps var names to lambda getters with defaults
2. **Access**: `import vllm.envs as envs; envs.VLLM_USE_V1`
3. **Caching**: After `enable_envs_cache()`, all lookups are memoized — **env var changes after import are ignored**
4. **Validation**: `validate_environ()` warns/errors on unknown `VLLM_*` vars (catches typos)

**Critical**: Set ALL env vars BEFORE importing vLLM.

### Must-Know for Production

| Variable | Default | When to Change |
|----------|---------|----------------|
| `VLLM_API_KEY` | None | Always set for exposed endpoints |
| `VLLM_LOGGING_LEVEL` | `INFO` | `WARNING` in prod, `DEBUG` for debugging |
| `VLLM_CONFIGURE_LOGGING` | `1` | `0` when app manages its own logging |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `600` | Increase for >200B models (15+ min load time) |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | `60` | Increase for 100K+ context |
| `VLLM_PORT` | None | Always set explicitly (K8s collision risk) |
| `VLLM_HOST_IP` | `""` | Set for multi-NIC nodes |
| `VLLM_NO_USAGE_STATS` | `0` | `1` to disable telemetry |
| `VLLM_DO_NOT_TRACK` | `0` | Alternative telemetry disable |
| `VLLM_MAX_N_SEQUENCES` | `16384` | Lower for public APIs (DoS prevention) |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | `5` | Increase for slow clients |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | `0` | Only `1` in trusted air-gapped environments |
| `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` | `0` | `1` to keep API alive on engine crash |

### Performance Tuning

| Variable | Default | When to Change |
|----------|---------|----------------|
| `VLLM_FLOAT32_MATMUL_PRECISION` | `highest` | `high` for 5-10% speedup |
| `VLLM_BATCH_INVARIANT` | `0` | `1` on H100+ for deterministic output (10-15% cost) |
| `VLLM_LOG_STATS_INTERVAL` | `10.0` | `60.0` in prod to reduce overhead |
| `VLLM_DISABLE_COMPILE_CACHE` | `0` | `1` for debugging custom ops |
| `VLLM_FLASHINFER_MOE_BACKEND` | `latency` | `throughput` for batch serving (15-20% speedup) |
| `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` | `1` | Already enabled |
| `VLLM_DEEP_GEMM_WARMUP` | `relax` | `skip` for faster startup, `full` for consistent latency |

### Memory Management

| Variable | Default | When to Change |
|----------|---------|----------------|
| `VLLM_CPU_KVCACHE_SPACE` | ~4GB | Set for CPU backend (e.g., `32` for 64GB RAM) |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `0` | `1` to extend context beyond model default |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | `512` | Bounds MLA logits to prevent OOM |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `0` | `1` for more accurate memory profiling |

### Distributed & Multi-GPU

| Variable | Default | When to Change |
|----------|---------|----------------|
| `VLLM_DP_RANK` | `0` | Set per replica in DP |
| `VLLM_DP_SIZE` | `1` | Set to replica count |
| `VLLM_DP_MASTER_IP` | `127.0.0.1` | Set for multi-node DP |
| `VLLM_DP_MASTER_PORT` | `0` | Set for multi-node DP |
| `VLLM_SKIP_P2P_CHECK` | `1` | `0` to diagnose multi-GPU deadlocks |
| `VLLM_DISABLE_PYNCCL` | `0` | `1` if NCCL init fails |
| `VLLM_NCCL_SO_PATH` | auto | Set to specific NCCL version |
| `VLLM_WORKER_MULTIPROC_METHOD` | `fork` | `spawn` if CUDA context issues |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | `1` | Symmetric memory for allreduce |
| `VLLM_RAY_PER_WORKER_GPUS` | `1.0` | GPUs per Ray worker |
| `VLLM_RAY_DP_PACK_STRATEGY` | `strict` | `fill`/`span` for Ray DP |
| `VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY` | `""` | Extra env var prefixes for Ray workers |

### Quantization & Kernels

| Variable | Default | When to Change |
|----------|---------|----------------|
| `VLLM_USE_DEEP_GEMM` | `1` | DeepGEMM FP8 (Hopper/Blackwell) |
| `VLLM_MOE_USE_DEEP_GEMM` | `1` | DeepGEMM for MoE |
| `VLLM_MLA_DISABLE` | `0` | `1` to force standard attention (DeepSeek) |
| `VLLM_USE_TRITON_AWQ` | `0` | `1` for Triton AWQ fallback |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | `0` | `1` for Marlin correctness issues |
| `VLLM_DISABLED_KERNELS` | `[]` | Comma-separated list to disable specific kernels |
| `VLLM_USE_FLASHINFER_MOE_FP8` | `0` | `1` for FlashInfer FP8 MoE (H100) |
| `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER` | `1` | FP8 block-scale GEMM (SM90+) |
| `VLLM_KV_CACHE_LAYOUT` | None | `NHD` or `HND` for specific backends |
| `Q_SCALE_CONSTANT` / `K_SCALE_CONSTANT` / `V_SCALE_CONSTANT` | 200/200/100 | FP8 KV cache scale tuning |

### Debugging & Profiling

| Variable | Default | When to Change |
|----------|---------|----------------|
| `VLLM_TRACE_FUNCTION` | `0` | `1` for function tracing (10-30% overhead) |
| `VLLM_CUSTOM_SCOPES_FOR_PROFILING` | `0` | `1` for Nsys profiling |
| `VLLM_NVTX_SCOPES_FOR_PROFILING` | `0` | `1` for NVTX markers |
| `VLLM_DEBUG_WORKSPACE` | `0` | `1` for memory fragmentation diagnosis |
| `VLLM_GC_DEBUG` | `""` | `1` or JSON for GC debugging |
| `VLLM_COMPUTE_NANS_IN_LOGITS` | `0` | `1` to detect NaN outputs |
| `VLLM_LOG_MODEL_INSPECTION` | `0` | `1` to log model structure after loading |
| `VLLM_DEBUG_LOG_API_SERVER_RESPONSE` | `0` | `1` to log full API responses |
| `VLLM_PATTERN_MATCH_DEBUG` | None | Path to debug custom compile passes |
| `VLLM_DEBUG_DUMP_PATH` | None | Dump FX graphs to directory |

### Multimodal / Media

| Variable | Default | When to Change |
|----------|---------|----------------|
| `VLLM_IMAGE_FETCH_TIMEOUT` | `5` | Increase for slow CDNs |
| `VLLM_VIDEO_FETCH_TIMEOUT` | `30` | Increase for large videos |
| `VLLM_AUDIO_FETCH_TIMEOUT` | `10` | Increase for large audio |
| `VLLM_MEDIA_FETCH_MAX_RETRIES` | `3` | Increase for unreliable networks |
| `VLLM_MEDIA_URL_ALLOW_REDIRECTS` | `1` | `0` to block redirects (security) |
| `VLLM_MEDIA_LOADING_THREAD_COUNT` | `8` | Increase for high multimodal throughput |
| `VLLM_VIDEO_LOADER_BACKEND` | `opencv` | `identity` for pre-processed video |
| `VLLM_ALLOWED_URL_SCHEMES` | `file,http,https,data` | Restrict for security |
| `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` | `25` | Increase for long audio |

### ROCm / AMD Specific

| Variable | Default | When to Change |
|----------|---------|----------------|
| `VLLM_ROCM_USE_AITER` | `0` | `1` on MI300X for 20-30% speedup |
| `VLLM_ROCM_USE_AITER_MOE` | `1` | AITER MoE ops |
| `VLLM_ROCM_USE_AITER_MLA` | `1` | AITER MLA ops |
| `VLLM_ROCM_FP8_PADDING` | `1` | FP8 weight alignment |
| `VLLM_ROCM_MOE_PADDING` | `1` | MoE padding |
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` | `NONE` | `FP`, `INT8`, `INT4` for allreduce |
| `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` | `0` | Shuffled KV layout |

### Caching & Storage

| Variable | Default | When to Change |
|----------|---------|----------------|
| `VLLM_CACHE_ROOT` | `~/.cache/vllm` | Set to fast NVMe for compile cache |
| `VLLM_CONFIG_ROOT` | `~/.config/vllm` | Config directory |
| `VLLM_ASSETS_CACHE` | `~/.cache/vllm/assets` | Multimodal media cache |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache |
| `HF_TOKEN` | env | HuggingFace auth token |
| `VLLM_USE_MODELSCOPE` | `0` | `1` to use ModelScope (China) |

---

## Hardware-Specific Optimization

### H100 (Hopper, SM90) — Best Practices

```bash
# Optimal H100 config
VLLM_USE_DEEP_GEMM=1 \
VLLM_FLOAT32_MATMUL_PRECISION=high \
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  -O2
```

Key H100 features:
- **Native FP8**: `--quantization fp8` — 2x memory reduction, up to 1.6x throughput. No calibration needed for dynamic FP8
- **DeepGEMM**: Enabled by default. Provides optimized FP8 GEMM kernels
- **FlashInfer allreduce+RMSNorm fusion**: Enabled at O2+ with TP>1
- **Batch invariant mode**: `VLLM_BATCH_INVARIANT=1` for deterministic output (SM>=9.0 required)
- **FP8 KV cache**: Independent of model quantization — doubles KV capacity
- **`gpu_memory_utilization=0.95`**: Safe on dedicated servers, leaves ~4GB for system

### A100 (Ampere, SM80) — Best Practices

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  -O2
```

A100 differences from H100:
- **No native FP8 compute** — FP8 models run as W8A16 via Marlin kernels
- **FP8 KV cache still works** (software quantize/dequantize)
- **DeepGEMM not supported** (requires SM90+)
- **FlashInfer allreduce fusion not available** (requires SM90+)
- **bfloat16 preferred** over float16 (wider dynamic range)
- **Batch invariant**: Uses cuBLAS workspace config instead of Triton kernel replacement

### Optimization Levels

| Level | CUDA Graphs | Compilation | Best For |
|-------|-------------|-------------|----------|
| **O0** | None | None | Debugging, rapid iteration |
| **O1** | Piecewise | Dynamo+Inductor | Development |
| **O2** | Full+Piecewise | Full | **Production (default)** |
| **O3** | Full+Piecewise | Full | Same as O2 currently |

### Performance Modes

| Mode | Behavior | Best For |
|------|----------|----------|
| `balanced` | Standard CUDA graph sizes | General serving |
| `interactivity` | Fine-grained graphs (1 to 32) | Low-latency chat |
| `throughput` | Doubled `max_num_batched_tokens` and `max_num_seqs` | Batch inference |

---

## Performance Tuning Guide

### Key Levers (in order of impact)

1. **`--gpu-memory-utilization`** (default: 0.9)
   - Higher = more KV cache = more concurrent sequences = higher throughput
   - 0.95 is safe on dedicated GPU servers (saves ~4GB on H100 80GB)
   - Lower to 0.7-0.85 for multi-tenant or stability concerns

2. **`--quantization fp8`** (H100+)
   - 2x memory reduction, ~1.6x throughput
   - Dynamic FP8 requires zero calibration
   - Pre-quantized via llm-compressor gives best accuracy

3. **`--kv-cache-dtype fp8`** (independent of model quant)
   - ~50% KV cache memory reduction
   - Works on A100 too (software quantize)
   - Three calibration modes: none, random token, dataset (recommended)

4. **`--max-num-batched-tokens`** (default: H100=8192, A100=2048)
   - Higher = more throughput, worse per-request latency
   - Lower (2048) for better ITL (inter-token latency)
   - Higher (>8192) for better TTFT and throughput on big GPUs

5. **`--max-num-seqs`** (default: H100=1024, A100=256)
   - Upper limit on concurrent sequences
   - True concurrency determined by available KV cache
   - Lower for latency-sensitive workloads

6. **`--enable-prefix-caching`** (default: True)
   - <1% overhead at 0% hit rate — essentially free
   - 2-10x throughput for repeated prefixes (system prompts, RAG)
   - `cache_salt` parameter for multi-tenant isolation

7. **`--enable-chunked-prefill`** (default: True)
   - Splits long prefills, allows decode during prefill
   - Critical for latency in mixed workloads
   - `--max-num-partial-prefills` controls concurrency

8. **Optimization level** (`-O2` default, best for production)

### Memory Optimization Strategies

| Strategy | Memory Savings | Latency Impact | When to Use |
|----------|---------------|----------------|-------------|
| `--quantization fp8` | ~50% model | Minimal | H100+, always |
| `--quantization awq` | ~75% model | 5-10% | Memory-constrained |
| `--kv-cache-dtype fp8` | ~50% KV cache | Minimal | Always beneficial |
| `--cpu-offload-gb N` | +N GiB effective | PCIe bound | Fit larger models |
| `--gpu-memory-utilization 0.95` | +4GB on H100 | None | Dedicated servers |
| `--enforce-eager` | Saves CUDA graph mem | 10-20% slower | Debug / BitsAndBytes |
| Lower `--max-model-len` | Less KV allocation | Limits context | When full context unused |

### Quantization Performance Matrix

| Method | Memory | Throughput | Quality | Hardware |
|--------|--------|------------|---------|----------|
| FP8 (dynamic) | 2x reduction | 1.6x | Near-lossless | H100+ |
| FP8 (static, llm-compressor) | 2x | 1.6x | Best FP8 | H100+ |
| AWQ + Marlin | 4x | **10.9x vs raw AWQ** | Good | All NVIDIA |
| GPTQ + Marlin | 4x | Similar to AWQ-Marlin | Good | All NVIDIA |
| AWQ (without Marlin) | 4x | **Slower than baseline** | Good | All |
| BitsAndBytes 4-bit | 4x | Moderate | Good | All (forces eager) |
| GGUF | Varies | Moderate | Varies | All |

**Critical**: AWQ/GPTQ without Marlin kernels is **slower than unquantized** — the kernel matters enormously.

### CPU Requirements (often overlooked)

Minimum physical cores = `2 + N` (N = GPU count). With DP: `A + DP + N + (1 if DP>1)`.

On hyperthreaded systems, **double** the count. CPU starvation is the most common undiagnosed performance issue.

Source: `docs/configuration/optimization.md`

---

## Deployment Recipes

### Single GPU (<7B models)

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192
```

### Multi-GPU Single Node (13B-70B)

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9
```

### Multi-Node (405B+)

```bash
# Node 0 (head):
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend mp \
  --master-addr <HEAD_IP> --master-port 29501 \
  --nnodes 2 --node-rank 0

# Node 1:
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend mp \
  --master-addr <HEAD_IP> --master-port 29501 \
  --nnodes 2 --node-rank 1
```

### DeepSeek V3 with Expert Parallelism

```bash
VLLM_USE_DEEP_GEMM=1 \
vllm serve deepseek-ai/DeepSeek-V3 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --enable-eplb \
  --all2all-backend deepep_low_latency \
  --quantization fp8 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 40 \
  --max-model-len 16384
```

### Memory-Constrained Single GPU

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --cpu-offload-gb 4
```

### Low-Latency Interactive

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --performance-mode interactivity \
  --gpu-memory-utilization 0.9 \
  -O2
```

### Docker

```bash
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_API_KEY=sk-your-key \
  vllm/vllm-openai:latest \
  vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000
```

### Kubernetes Health Probes

```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5

livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 15
  periodSeconds: 10
```

### Prometheus Monitoring

Metrics auto-mounted at `/metrics`. Scrape config:
```yaml
scrape_configs:
  - job_name: vllm
    static_configs:
      - targets: ['vllm-host:8000']
    scrape_interval: 5s
```

Autoscaling should use `vllm:num_requests_waiting` (not CPU/GPU utilization).

### Production YAML Config

```yaml
# production.yaml
model: meta-llama/Llama-3.1-70B-Instruct
host: "0.0.0.0"
port: 8000
tensor-parallel-size: 4
dtype: bfloat16
gpu-memory-utilization: 0.95
quantization: fp8
kv-cache-dtype: fp8
enable-prefix-caching: true
api-key:
  - "sk-production-key-1"
  - "sk-production-key-2"
disable-log-stats: false
disable-fastapi-docs: true
```

```bash
VLLM_LOGGING_LEVEL=WARNING \
VLLM_LOG_STATS_INTERVAL=60 \
vllm serve --config production.yaml
```

---

## Tool Calling & Structured Output

### Tool Calling Setup

Both flags required together:
```bash
vllm serve <model> \
  --enable-auto-tool-choice \
  --tool-call-parser <parser_name>
```

### Parser Matrix (partial — see `vllm/tool_parsers/__init__.py` for full list)

| Model Family | Parser |
|-------------|--------|
| Hermes 2 Pro, NousResearch | `hermes` |
| Mistral | `mistral` |
| Llama 3 | `llama3_json` |
| Llama 4 | `llama4_json` or `llama4_pythonic` |
| DeepSeek V3 | `deepseek_v3` |
| Qwen 3 | `qwen3_xml` or `qwen3_coder` |
| Gemma 4 | `gemma4` |
| IBM Granite | `granite` or `granite4` |
| Phi-4 Mini | `phi4_mini_json` |
| InternLM 2 | `internlm` |

Custom parsers: `--tool-parser-plugin <import_path>`

### Structured Output Backends

```bash
vllm serve <model> \
  --structured-outputs-config '{"backend": "xgrammar"}'
```

| Backend | Speed | Flexibility | Notes |
|---------|-------|-------------|-------|
| `xgrammar` | Fastest | JSON schema, regex, EBNF | **Default** |
| `guidance` | Fast | JSON, regex, choices | Supports `disable_additional_properties` |
| `outlines` | Moderate | Most flexible | Higher cold-start latency |
| `lm-format-enforcer` | Moderate | JSON schema | |

Structured output adds 10-30% throughput overhead.

---

## Vision Model (VLM) Deployment

```bash
vllm serve llava-hf/llava-v1.6-mistral-7b-hf \
  --limit-mm-per-prompt '{"image": 16, "video": 2}'
```

With resolution control:
```bash
--limit-mm-per-prompt '{"video": {"count": 1, "num_frames": 32, "width": 512, "height": 512}}'
```

Key configs:
- `--mm-processor-kwargs '{"num_crops": 4}'` — per-model processor overrides
- `--mm-processor-cache-gb 4` — preprocessed data cache (GiB)
- `--mm-encoder-tp-mode data` — ~10% throughput improvement at TP=8
- `--language-model-only` — disable multimodal (sets all modality limits to 0)
- `--skip-mm-profiling` — faster startup (user estimates memory)
- `--enable-mm-embeds` — accept precomputed embeddings (trusted users only)

Media timeouts via env vars: `VLLM_IMAGE_FETCH_TIMEOUT=5`, `VLLM_VIDEO_FETCH_TIMEOUT=30`, etc.

Security: `--allowed-media-domains "cdn.example.com"` for domain allowlist.

---

## Data Parallelism & Scaling

### Three Load Balancing Modes

**Internal LB** (default, single-node):
```bash
vllm serve <model> --data-parallel-size 4 --tensor-parallel-size 2
# Launches 4 engine cores + 4 API servers internally
```

**External LB** (one-pod-per-rank, Kubernetes):
```bash
# Pod 0:
vllm serve <model> --data-parallel-size 4 --data-parallel-rank 0 -tp 2
# Pod 1:
vllm serve <model> --data-parallel-size 4 --data-parallel-rank 1 -tp 2
# External LB routes between pods
```

**Hybrid LB** (multi-node, local internal LB):
```bash
# Node 0 (ranks 0-1):
vllm serve <model> -dp 4 -dpl 2 --data-parallel-start-rank 0 -tp 2
# Node 1 (ranks 2-3):
vllm serve <model> -dp 4 -dpl 2 --data-parallel-start-rank 2 -tp 2
# Each node runs internal LB; external LB across nodes
```

### Headless Mode (Worker-Only)

```bash
# Head node:
vllm serve <model> -dp 4 -tp 4 --nnodes 2 --node-rank 0
# Worker node:
vllm serve <model> -dp 4 -tp 4 --nnodes 2 --node-rank 1 --headless
```

### TP Super-Linear Scaling

More GPUs = proportionally more KV cache = larger batches = better utilization. TP scaling can yield **more than linear** throughput improvement.

---

## Config Validation & Interactions

### Dynamic Defaults

These values change based on hardware and other configs:

| Config | H100/MI300x | A100/Others | Throughput Mode |
|--------|-------------|-------------|-----------------|
| `max_num_batched_tokens` (server) | 8192 | 2048 | Doubled |
| `max_num_batched_tokens` (batch) | 16384 | 8192 | Doubled |
| `max_num_seqs` | 1024 | 256 | Doubled |

Source: `vllm/engine/arg_utils.py` `_set_default_max_num_seqs_and_batched_tokens_args()`

### Mutually Exclusive Configs

- `attention_backend` vs `attention_config.backend` — raises ValueError
- `enable_flashinfer_autotune` vs `kernel_config.enable_flashinfer_autotune` — raises ValueError
- `cudagraph_capture_sizes` vs `compilation_config.cudagraph_capture_sizes` — raises ValueError
- `data_parallel_hybrid_lb` vs `data_parallel_external_lb` — cannot both be True
- `enforce_eager=True` forces `compilation_config.mode=NONE` and `cudagraph_mode=NONE`

### Cascading Effects

- `enable_chunked_prefill` defaults based on `model_config.is_chunked_prefill_supported`
- `enable_prefix_caching` defaults based on `model_config.is_prefix_caching_supported`
- `async_scheduling` auto-enabled unless incompatible with speculative decoding
- Pipeline parallelism is **not composable** with speculative decoding
- `BitsAndBytes 8-bit` forces `enforce_eager=True`

---

## Gotchas & Pitfalls

1. **K8s VLLM_PORT collision**: If your K8s service is named `vllm`, K8s auto-sets `VLLM_PORT` to a URI like `http://vllm:8000`, which crashes vLLM. **Fix**: Never name your service `vllm`, or explicitly set `VLLM_PORT`.

2. **Env vars are cached after import**: `enable_envs_cache()` wraps lookups with `functools.cache`. Changing env vars at runtime has no effect. Set everything before `import vllm`.

3. **AWQ/GPTQ without Marlin is slower than unquantized**: AWQ without Marlin: 67 tok/s vs 461 tok/s baseline. Always ensure Marlin kernels are active.

4. **CPU starvation is the #1 undiagnosed issue**: V1 multi-process architecture needs minimum `2 + N` physical cores (N = GPU count). In K8s with HT, that's `2*(2+N)` vCPUs.

5. **`max_num_seqs` is an upper limit, not actual concurrency**: True concurrency determined dynamically from available KV cache.

6. **`gpu_memory_utilization=0.9` leaves ~8GB unused on H100**: Can often safely increase to 0.95 for more KV cache capacity.

7. **Prefix caching should always be enabled**: <1% overhead at 0% hit rate, pure upside. But standard load balancers destroy cache locality in multi-replica setups — use prefix-aware routing.

8. **`VLLM_ENABLE_RESPONSES_API_STORE`**: Explicitly warned in source as causing a **memory leak** (messages never removed).

9. **`VLLM_SKIP_P2P_CHECK=1` by default**: Skips P2P validation because the check itself hangs on buggy NVIDIA 535 drivers. Set to `0` only to diagnose multi-GPU issues.

10. **Sleep mode for RLHF**: Level 1 offloads weights to CPU (fast resume). Level 2 discards weights. Use `wake_up(tags=["weights"])` pattern for weight sync.

11. **`VLLM_WORKER_MULTIPROC_METHOD` defaults to `fork`**: Can cause CUDA context issues in child processes. Switch to `spawn` if seeing random crashes.

12. **TGI is in maintenance mode**: HuggingFace defaults Inference Endpoints to vLLM now. The competitive landscape is vLLM vs SGLang.

13. **SGLang edges vLLM on small models (7-8B)**: ~29% higher throughput via RadixAttention. Gap narrows to 3-5% on 70B+.

14. **Multi-node DP**: Only `data_parallel_backend=mp` supports multi-node. Ray DP is single-node only.

---

## Sources

### Source Code (authoritative)
1. `vllm/engine/arg_utils.py` — EngineArgs dataclass, all 268 CLI fields
2. `vllm/envs.py` — All 238 environment variables
3. `vllm/config/vllm.py` — VllmConfig root, optimization levels, performance modes
4. `vllm/config/model.py` — ModelConfig
5. `vllm/config/cache.py` — CacheConfig
6. `vllm/config/parallel.py` — ParallelConfig, EPLBConfig
7. `vllm/config/scheduler.py` — SchedulerConfig
8. `vllm/config/speculative.py` — SpeculativeConfig
9. `vllm/config/lora.py` — LoRAConfig
10. `vllm/config/structured_outputs.py` — Structured output backends
11. `vllm/entrypoints/openai/cli_args.py` — Frontend args
12. `vllm/entrypoints/cli/serve.py` — Serve subcommand
13. `vllm/utils/argparse_utils.py` — YAML config support
14. `vllm/env_override.py` — Early env var handling
15. `vllm/model_executor/layers/batch_invariant.py` — Batch invariant NCCL settings
16. `vllm/ray/ray_env.py` — Ray env var propagation
17. `docs/configuration/optimization.md` — Official optimization guide
18. `docs/features/tool_calling.md` — Tool calling parser matrix
19. `docs/features/quantization/fp8.md` — FP8 quantization
20. `docs/features/quantization/quantized_kvcache.md` — FP8 KV cache
21. `docs/serving/data_parallel_deployment.md` — DP deployment guide
22. `docs/configuration/conserving_memory.md` — Memory reduction techniques
23. `docs/configuration/env_vars.md` — Env var documentation

### Web Sources
24. vLLM Official Documentation — https://docs.vllm.ai/en/latest/
25. vLLM Blog: Large-scale serving with Wide-EP — https://blog.vllm.ai/2025/12/17/large-scale-serving.html
26. Red Hat: vLLM performance tuning — https://developers.redhat.com/articles/2026/03/03/practical-strategies-vllm-performance-tuning
27. Google Cloud: vLLM xPU inference guide — https://cloud.google.com/blog/topics/developers-practitioners/vllm-performance-tuning-the-ultimate-guide-to-xpu-inference-configuration
28. Red Hat: Optimizing DeepSeek-R1 — https://developers.redhat.com/articles/2025/03/19/how-we-optimized-vllm-deepseek-r1
29. Spheron: vLLM vs TensorRT-LLM vs SGLang benchmarks — https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/
30. GPUStack: Impact of quantization on vLLM — https://docs.gpustack.ai/2.0/performance-lab/references/the-impact-of-quantization-on-vllm-inference-performance/
31. Jarvis Labs: vLLM quantization guide — https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks
32. vLLM production-stack — https://github.com/vllm-project/production-stack
33. DeepSeek V3 vLLM recipes — https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3.html
