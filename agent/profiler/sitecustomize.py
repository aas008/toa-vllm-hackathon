"""
PyTorch Profiler Hook for vLLM

Monkey-patches vLLM to inject PyTorch profiler around inference calls.
Deployed to the vLLM host and activated via PYTHONPATH.

SOURCE: vllm-profiler/sitecustomize.py (verbatim)

How it works:
    1. Agent copies this file + profiler_config.yaml to vLLM host
    2. Agent sets PYTHONPATH to include the directory containing this file
    3. When vLLM starts, Python auto-imports sitecustomize.py
    4. This module patches vLLM's inference path to wrap with torch.profiler
    5. Profiler outputs Chrome trace JSON files
    6. Agent retrieves traces for analysis

Output: Chrome trace JSON files (consumed by analysis/trace_analyzer.py)
"""

# TODO: Copy verbatim from vllm-profiler/sitecustomize.py
