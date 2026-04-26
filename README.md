# toa-vllm-hackathon

**Track 6: Performance Tuning & Evaluation Agent**

A CLI-based agentic loop that automatically benchmarks, profiles, analyzes, and tunes a vLLM inference server — then outputs a markdown report.

Demo:


https://github.com/user-attachments/assets/eefde2d6-9755-42c2-ace4-53d7aeb5b98c



## Quick Start

```bash
uv sync

claude --dangerously-skip-permissions \
  --effort max \
  --model "opus-4-6[1m]" \
  -p "@program.md I want you to use subagents, and make sure that all the shell commands they run regarding deployment, guidellm and tear down they run in pane:2.1 of the current tmux session."
```

## AutoResearch Agent

- Agent updates `deploy.sh` to optimize vllm deployment
- It uses `score.py` to get metric: `goodput = (completed_requests where TTFT < T1 AND ITL < T2) / total_time`
- Its prompted using `program.md`

