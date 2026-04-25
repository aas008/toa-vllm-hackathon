# toa-vllm-hackathon

**Track 6: Performance Tuning & Evaluation Agent**

A CLI-based agentic loop that automatically benchmarks, profiles, analyzes, and tunes a vLLM inference server — then outputs a markdown report.

## Quick Start

```bash
pip install -r requirements.txt

python -m agent \
    --vllm-endpoint http://gpu:8000 \
    --vllm-host aansharm-0-yxg5 \
    --model meta-llama/Llama-3.1-8B \
    --api-key $ANTHROPIC_API_KEY \
    --max-iterations 30 \
    --output reports/
```

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
