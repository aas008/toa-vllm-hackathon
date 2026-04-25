"""
Agent Loop & vLLM System Prompt

The core agentic loop: assembles messages, calls Claude, dispatches tool calls,
and iterates until the agent signals completion or hits max iterations.

SOURCE: ai-perf-hackathon/agent/agentic.py (loop reused, prompt replaced)

Key components to copy:
    - run_agent_loop()    — Main loop: message assembly → Claude call → tool dispatch → repeat
    - Message assembly    — System prompt + conversation history + tool results
    - Tool dispatch       — Route tool_use blocks to handler functions
    - Stop condition      — Handle stop_reason == "end_turn" or "done" tool call
    - Prompt caching      — cache_control on system prompt for cost optimization

Changes from source:
    - SYSTEM_PROMPT: Replace Nginx tuning content with vLLM tuning knowledge
    - Tool definitions: Replace 5 Nginx tools with 7 vLLM tools
    - Add vLLM-specific tunable parameters list
    - Add benchmark methodology from LLM-inference-benchmark-guide

vLLM System Prompt should cover:
    - Tuning order: baseline → GPU metrics → profile → analyze → tune → re-benchmark
    - Tunable params: max-num-seqs, max-num-batched-tokens, gpu-memory-utilization,
      enable-chunked-prefill, enable-prefix-caching, max-model-len, enforce-eager,
      tensor-parallel-size, quantization, speculative decoding, CUDA graph batch sizes,
      scheduling policy
    - Benchmark profiles: Balanced, Decode-Heavy, Prefill-Heavy, Long-Context
    - Concurrency sweep: 1, 50, 100, 200, 300, 500, 650
    - Output metrics: throughput (tok/sec), TTFT, ITL, TPOT at P50/P95/P99
"""

# TODO: Copy run_agent_loop() from source
# TODO: Write VLLM_SYSTEM_PROMPT with tuning knowledge
# TODO: Define TOOL_DEFINITIONS list (7 tools)
# TODO: Implement tool dispatch table
# TODO: Add prompt caching (cache_control) on system prompt
