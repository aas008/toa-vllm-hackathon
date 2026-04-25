# Autonomous vLLM Performance Tuning Agent

**Track 6: Performance Tuning & Evaluation Agent | Open Accelerator Hackathon**

---

## Abstract

We built an autonomous agent that benchmarks, profiles, and tunes vLLM inference servers without human intervention. Powered by Claude, the agent explores vLLM's configuration space — running controlled experiments, measuring throughput and latency, detecting regressions, and producing structured reports with before/after comparisons. It replaces a manual, expert-driven tuning loop with a fully automated pipeline that runs on GPU clusters via SSH or OpenShift.

## Problem

vLLM ships with default serving parameters that leave performance on the table. Tuning requires iteratively adjusting 10+ hyperparameters (batch sizes, GPU memory allocation, chunked prefill, CUDA graphs, tensor parallelism), restarting the server, benchmarking under realistic workloads, and interpreting the results — a time-consuming process that demands deep systems expertise. Each model and hardware combination requires its own tuning pass.

## Goal

Build a CLI tool that takes a vLLM endpoint and model name, then autonomously: (1) establishes baseline performance, (2) runs tuning experiments with modified server configurations, (3) detects improvements and regressions, and (4) outputs a comprehensive markdown report — all within a bounded iteration budget.

## Methodology

```
                         ┌─────────────────┐
                         │   CLI Entry      │
                         │   (main.py)      │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  Claude Agent     │
                         │  Loop (agentic.py)│◄──── System Prompt
                         │  max N iterations │      (tuning playbook)
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
     ┌────────▼────────┐ ┌───────▼───────┐ ┌────────▼────────┐
     │  Benchmark       │ │  Remote Exec   │ │  Analysis        │
     │                  │ │                │ │                  │
     │  • run_eval      │ │  • run_command │ │  • compare       │
     │    (eval pipeline│ │  • read_file   │ │    _benchmarks   │
     │    full lifecycle)│ │  • write_file  │ │  • analyze_trace │
     │  • run_benchmark │ │  • fetch_vllm  │ │  • map_kernel    │
     │    (GuideLLM)    │ │    _logs       │ │  • analyze_eval  │
     └────────┬─────────┘ └───────┬───────┘ │    _results      │
              │                   │          └────────┬─────────┘
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  │
                         ┌────────▼─────────┐
                         │  Reporter         │
                         │  (reporter.py)    │
                         │  MD + JSON output │
                         └──────────────────┘
```

**Agent Workflow:**

1. **Discover** — GPU hardware (nvidia-smi), running vLLM config, available resources
2. **Baseline** — Run benchmarks across workload profiles (throughput, latency, mixed, long-context) at multiple concurrency levels
3. **Experiment** — Modify one vLLM parameter at a time (e.g. enable chunked prefill), start a fresh server, re-benchmark
4. **Compare** — Detect regressions/improvements per metric with directionality awareness (higher throughput = better, lower latency = better)
5. **Report** — Generate structured markdown with before/after tables, cost analysis (CPMT), token usage

**Key Metrics:** Output throughput (tok/s), TTFT, ITL, TPOT at P50/P95/P99, request success rate, GPU cache usage

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| Agent Brain | Claude Sonnet 4 (Anthropic API / Vertex AI), prompt caching |
| Benchmarking | GuideLLM, vLLM Eval Pipeline (server lifecycle + Prometheus scraping) |
| Inference Server | vLLM v0.19+ on NVIDIA H100 GPUs |
| Analysis | Regression detection, kernel-to-source mapping (120+ regex patterns), CPMT cost modeling |
| Output | Markdown + JSON reports with decision logs |
