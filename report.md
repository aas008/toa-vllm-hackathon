# vLLM Optimization Report — Run 2: Qwen3-0.6B on 4x H100

## Setup

- **Model:** Qwen/Qwen3-0.6B
- **Hardware:** 4x H100 GPUs (indices 4,5,6,7) on rh-h100-02
- **Benchmark:** GuideLLM, concurrent rate 20, 10s duration
- **Workload:** prefill-heavy (prompt_tokens=2048, output_tokens=256)
- **SLO targets:** TTFT < 200ms, ITL < 5ms
- **Scoring metric:** Goodput (tok/s) = SLO-passing output tokens / duration
- **Starting point:** Best config from Run 1 = tp=4, gpu=0.95, -O3 @ 5888.0 tok/s

---

## Results Summary

| Exp | Config Changes | Goodput (tok/s) | TTFT p95 (ms) | ITL p95 (ms) | SLO % | Completed Reqs |
|-----|---------------|-----------------|---------------|--------------|-------|----------------|
| 100 | Baseline: tp=4, gpu=0.95, -O3, model-len=4096 | 5657.6 | 282.5 | 2.9 | 91.7% | 241 |

---

## Experiment-by-Experiment Analysis

### Exp 100 (Baseline)
**Config:** tp=4, max-model-len=4096, gpu=0.95, -O3, disable-log-stats
**Result:** 5657.6 tok/s, 91.7% SLO pass, TTFT p95=282.5ms, ITL p95=2.9ms
**Analysis:** TTFT p95 at 282.5ms is the primary SLO bottleneck — 82.5ms over the 200ms target. ITL is well within bounds at 2.9ms. The 20 concurrent requests create prefill waves, and requests in later waves exceed the TTFT SLO. 241 completed, 20 incomplete (timed out).

---
