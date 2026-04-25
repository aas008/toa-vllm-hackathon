# toa-vllm-hackathon

**Track 6: Performance Tuning & Evaluation Agent**

A CLI-based agentic loop that automatically benchmarks, profiles, analyzes, and tunes a vLLM inference server — then outputs a markdown report.

## Prerequisites

- Python 3.10+
- `oc` CLI (OpenShift) or SSH access to the vLLM host
- GuideLLM (`pip install guidellm`)
- Claude API access (direct Anthropic API key or Google Cloud Vertex AI)
- A running vLLM pod/server with GPU

## Setup (Step-by-Step for Team Members)

### Step 1: Clone and install dependencies

```bash
git clone https://github.com/aas008/toa-vllm-hackathon.git
cd toa-vllm-hackathon
pip install -r requirements.txt
```

### Step 2: Authenticate to OpenShift and Google Cloud

```bash
# OpenShift — log in and verify you can see the vLLM pod
export KUBECONFIG=/path/to/your/kubeconfig
oc login  # or: oc whoami to verify
oc get pods -n <namespace>

# Google Cloud — authenticate for Vertex AI (Claude API access)
gcloud auth application-default login
```

### Step 3: Find the vLLM pod and set up port-forward

The benchmark tool (GuideLLM) runs locally and sends HTTP requests to vLLM.
You need a port-forward so `localhost:8000` reaches the pod:

```bash
# Find available vLLM pods
oc get pods -A | grep vllm

# Check what port vLLM is listening on inside the pod
oc exec -n <namespace> <pod-name> -- ss -tlnp | grep python

# Set up port-forward (local 8000 -> pod's serving port)
oc port-forward -n <namespace> <pod-name> 8000:<pod-port> &
```

### Step 4: Identify the served model name

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
# Use the "id" field from the response as the --model argument
```

### Step 5: Set Vertex AI environment variables

```bash
export CLAUDE_CODE_USE_VERTEX=1
export ANTHROPIC_VERTEX_PROJECT_ID=itpc-gcp-pnd-pe-eng-claude
export CLOUD_ML_REGION=us-east5
```

### Step 6: Run the agent

```bash
python3 -m agent \
    --vllm-endpoint http://localhost:8000 \
    --model <model-name-from-step-4> \
    --oc-mode \
    --oc-pod <pod-name> \
    --oc-namespace <namespace> \
    --vertex \
    --vertex-project-id itpc-gcp-pnd-pe-eng-claude \
    --vertex-region us-east5 \
    --max-iterations 15 \
    --profiles balanced \
    --output reports/
```

Reports are saved to `reports/report_<timestamp>.md` and `reports/report_<timestamp>.json`.

## Running the Agent

### OpenShift mode (oc exec)

```bash
# Using Vertex AI for Claude API
python -m agent \
    --vllm-endpoint http://localhost:8000 \
    --model /models/facebook/opt-125m \
    --oc-mode \
    --oc-pod aanya-vllm-test-pod \
    --oc-namespace toa-hack \
    --kubeconfig /path/to/kubeconfig \
    --vertex \
    --vertex-project-id $ANTHROPIC_VERTEX_PROJECT_ID \
    --vertex-region us-east5 \
    --max-iterations 15 \
    --profiles balanced \
    --output reports/

# Using direct Anthropic API key
python -m agent \
    --vllm-endpoint http://localhost:8000 \
    --model /models/facebook/opt-125m \
    --oc-mode \
    --oc-pod aanya-vllm-test-pod \
    --oc-namespace toa-hack \
    --kubeconfig /path/to/kubeconfig \
    --api-key $ANTHROPIC_API_KEY \
    --max-iterations 15 \
    --profiles balanced \
    --output reports/
```

### SSH mode

```bash
python -m agent \
    --vllm-endpoint http://gpu-host:8000 \
    --vllm-host gpu-host \
    --ssh-user root \
    --model meta-llama/Llama-3.1-8B \
    --api-key $ANTHROPIC_API_KEY \
    --max-iterations 30 \
    --output reports/
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--vllm-endpoint` | (required) | URL of the vLLM server |
| `--model` | (required) | Model served by vLLM |
| `--oc-mode` | false | Use `oc exec` instead of SSH |
| `--oc-pod` | — | Pod name (required with `--oc-mode`) |
| `--oc-namespace` | toa-hack | OpenShift namespace |
| `--kubeconfig` | `$KUBECONFIG` | Path to kubeconfig file |
| `--vllm-host` | — | SSH hostname (required without `--oc-mode`) |
| `--ssh-user` | root | SSH user |
| `--api-key` | `$ANTHROPIC_API_KEY` | Anthropic API key |
| `--vertex` | false | Use Vertex AI for Claude |
| `--vertex-project-id` | `$ANTHROPIC_VERTEX_PROJECT_ID` | Vertex AI project |
| `--vertex-region` | us-east5 | Vertex AI region |
| `--claude-model` | sonnet | Claude model (sonnet/opus/haiku) |
| `--max-iterations` | 30 | Max agent loop iterations |
| `--profiles` | all 4 | Benchmark profiles to run |
| `--output` | reports/ | Output directory |

### Timeouts

| Where | Default | Description |
|-------|---------|-------------|
| `--max-iterations` | 30 | Max agent loop turns |
| `run_benchmark` max_seconds | 120s | GuideLLM benchmark duration per level |
| `run_command` timeout | 60s | Remote shell command timeout |

## Architecture

```
CLI Entry Point (main.py)
    │
    ▼
Claude Agent Loop (agentic.py)
    │  Uses: llm.py (Anthropic API + token tracking)
    │
    ├── Core Tools (tools.py)
    │   ├── run_command    — SSH commands on GPU host
    │   ├── read_file      — Read remote files
    │   ├── write_file     — Write remote files
    │   └── done           — Signal completion
    │
    ├── Benchmark Tool (tools.py)
    │   └── run_benchmark  — GuideLLM with 4 profiles × 7 concurrencies
    │
    ├── Analysis Tools (analysis/)
    │   ├── trace_analyzer — PyTorch profiler trace parsing
    │   ├── kernel_mapper  — CUDA kernel → source code mapping
    │   ├── regression     — Before/after comparison
    │   └── cost           — CPMT calculation
    │
    ├── Profiler (profiler/)
    │   ├── sitecustomize.py    — PyTorch profiler hook for vLLM
    │   └── profiler_config.yaml
    │
    └── Reporter (reporter.py)
        └── Markdown + JSON report generation
```

## Agent Workflow

1. **Baseline** — Benchmark all 4 profiles at selected concurrencies
2. **Collect** — GPU metrics (nvidia-smi) + vLLM config
3. **Profile** — Deploy PyTorch profiler, capture kernel traces
4. **Analyze** — Extract hot kernels, map to source, categorize bottlenecks
5. **Tune** — Modify vLLM serving parameters based on analysis
6. **Re-benchmark** — Verify improvements, iterate until convergence
7. **Report** — Generate markdown report with before/after comparison

## Benchmark Profiles

| Profile | ISL | OSL | Description |
|---------|-----|-----|-------------|
| Balanced | 1000 | 1000 | Even input/output |
| Decode-Heavy | 512 | 2048 | Long generation |
| Prefill-Heavy | 2048 | 128 | Long context input |
| Long-Context | 8000 | 1000 | Extended context |

## Source Repos

| Component | Source |
|-----------|--------|
| Agent framework | `ai-perf-hackathon/agent/` |
| Analysis tools | `AI-Analysis-Agent/psap-mcp-server/` |
| Profiler | `vllm-profiler/` |
| Benchmark guide | `LLM-inference-benchmark-guide/` |

## Project Structure

```
toa-vllm-hackathon/
├── agent/
│   ├── __init__.py
│   ├── __main__.py           # python -m agent entry point
│   ├── main.py               # CLI args + orchestration
│   ├── llm.py                # Claude API client + token tracking
│   ├── agentic.py            # Agent loop + vLLM system prompt
│   ├── tools.py              # Tool definitions + dispatch
│   ├── reporter.py           # Report generation
│   ├── ssh_client.py         # SSH wrapper for GPU host
│   ├── analysis/
│   │   ├── trace_analyzer.py # PyTorch trace analysis
│   │   ├── kernel_mapper.py  # Kernel → source mapping
│   │   ├── regression.py     # Regression detection
│   │   └── cost.py           # CPMT cost efficiency
│   └── profiler/
│       ├── sitecustomize.py  # Profiler hook
│       └── profiler_config.yaml
├── config/
│   └── settings.yaml         # Default profiles + tunable params
├── claude-plans/
│   └── hackathon-inventory.md
├── requirements.txt
└── README.md
```

## Verification: Prometheus Metrics Integration

Run executed on 2026-04-25 against `RedHatAI/Llama-3.2-1B-Instruct-FP8` on a single NVIDIA H100 80GB, vLLM v0.19.1. The agent used Prometheus metrics (`/metrics` endpoint) scraped automatically before and after each benchmark to drive tuning decisions.

### Setup

```bash
python3 -m agent \
    --vllm-endpoint http://localhost:8000 \
    --model RedHatAI/Llama-3.2-1B-Instruct-FP8 \
    --oc-mode --oc-pod vllm-llama-1b \
    --oc-namespace toa-hack \
    --kubeconfig /path/to/kubeconfig \
    --pod-template aanya-pod.yaml \
    --vertex --claude-model haiku \
    --max-iterations 50 --profiles balanced \
    --output reports/
```

### Baseline Results (balanced profile: ISL=128, OSL=128)

| Metric | Concurrency 1 | Concurrency 50 |
|--------|---------------|----------------|
| Output Throughput (p50) | 692 tok/sec | 13,531 tok/sec |
| TTFT p50 | 10.42 ms | 44.47 ms |
| TTFT p99 | — | 72.83 ms |
| ITL p50 | 1.44 ms | 1.99 ms |
| TPOT p50 | 1.51 ms | 2.33 ms |
| Success Rate | 100% | 98.8% |

### Prometheus Metrics (server-side, from `/metrics` auto-scrape)

| Metric | Value |
|--------|-------|
| KV Cache Usage | 0% (significant headroom) |
| Prefix Cache Hit Rate | 87.5% (477,232 / 545,408 tokens) |
| Preemptions | 0 |
| Server-side Generation Throughput | 6,392 tok/sec |
| Requests Waiting (at snapshot) | 0 |

### Experiments and Prometheus-Driven Analysis

| # | Parameter | Result | Prometheus Signal |
|---|-----------|--------|-------------------|
| 1 | `--enable-chunked-prefill` | -89.9% throughput | Prefix cache hit rate dropped to 0% — chunked prefills broke cache detection |
| 2 | `--gpu-memory-utilization 0.95` | -12.3% throughput | Prefix cache hit rate 0%, TTFT p99 +2652% |
| 3 | `--max-num-seqs 512` | -19.8% throughput | Prefix cache hit rate 0%, TTFT mean +137% |
| 4 | `--max-num-batched-tokens 8192` | -8.6% throughput | Prefix cache hit rate 0%, server throughput -7.3% |

### Key Finding

The agent identified through Prometheus counter deltas that **prefix caching was the dominant performance driver** (87.5% hit rate at baseline). Every experiment pod started with a cold cache, causing all tuning changes to appear as regressions. The agent correctly concluded the baseline was already optimally configured and called `done(success=True)` after 30 iterations.

**Prometheus metrics that drove the analysis:**
- `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` — cache hit rate per run
- `vllm:kv_cache_usage_perc` — confirmed KV cache headroom (0%), ruling out memory pressure
- `vllm:num_preemptions_total` — confirmed zero preemptions, ruling out scheduler eviction
- `vllm:generation_tokens_total` delta / duration — server-side throughput cross-checked against GuideLLM client-side measurements

Full report: [`reports/report_20260425_203355.md`](reports/report_20260425_203355.md)
