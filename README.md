# toa-vllm-hackathon

**Track 6: Performance Tuning & Evaluation Agent**

A CLI-based agentic loop that automatically benchmarks, profiles, analyzes, and tunes a vLLM inference server — then outputs a markdown report.

## Prerequisites

- Python 3.10+
- `oc` CLI (OpenShift) or SSH access to the vLLM host
- GuideLLM (`pip install guidellm`)
- Claude API access (direct Anthropic API key or Google Cloud Vertex AI)
- A running vLLM pod/server with GPU

## Setup

```bash
# 1. Clone and install dependencies
git clone https://github.com/aas008/toa-vllm-hackathon.git
cd toa-vllm-hackathon
pip install -r requirements.txt

# 2. Set up cluster access (OpenShift mode)
export KUBECONFIG=/path/to/your/kubeconfig
oc login  # or verify: oc whoami

# 3. Deploy vLLM pod (if not already running)
oc apply -f k8s/model-pvc.yaml
oc apply -f k8s/model-download-job.yaml   # wait for completion
oc apply -f aanya-pod.yaml

# 4. Wait for pod to be ready
oc get pods -n toa-hack -w

# 5. Set up port-forward to the vLLM pod
oc port-forward -n toa-hack aanya-vllm-test-pod 8000:8000 &

# 6. Verify endpoint
curl http://localhost:8000/v1/models
```

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
