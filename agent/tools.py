"""
Tool Definitions & Dispatch

Defines all tools available to the Claude agent and their handler functions.

SOURCE: ai-perf-hackathon/agent/tools.py (core tools reused, new tools added)

Core Tools (from ai-perf-hackathon, near-verbatim):
    - run_command(command, timeout)    — Execute shell command on vLLM host via SSH
    - read_file(path)                  — Read file contents from vLLM host
    - write_file(path, content)        — Write file to vLLM host
    - done(summary)                    — Signal agent completion

New Benchmark Tool:
    - run_benchmark(profile, concurrency, endpoint, model)
        Wraps GuideLLM to run benchmarks against vLLM endpoint.
        Profiles: Balanced (ISL=1000,OSL=1000), Decode-Heavy (ISL=512,OSL=2048),
                  Prefill-Heavy (ISL=2048,OSL=128), Long-Context (ISL=8000,OSL=1000)
        Concurrency sweep: 1, 50, 100, 200, 300, 500, 650
        Output: throughput (tok/sec), TTFT, ITL, TPOT at P50/P95/P99

New Analysis Tools:
    - analyze_trace(trace_json_path)
        Calls analysis/trace_analyzer.py to extract kernel stats, category breakdown.
    - map_kernel(kernel_name)
        Calls analysis/kernel_mapper.py to identify vLLM source for hot kernels.

Tool Definition Format:
    Each tool is a dict with: name, description, input_schema (JSON Schema)
    Compatible with Claude's tool_use API format.

Dispatch:
    TOOL_HANDLERS = {tool_name: handler_function}
    dispatch_tool(name, input) → result string
"""

# TODO: Copy run_command, read_file, write_file, done from source
# TODO: Implement run_benchmark wrapping GuideLLM CLI
# TODO: Implement analyze_trace calling trace_analyzer
# TODO: Implement map_kernel calling kernel_mapper
# TODO: Define TOOL_DEFINITIONS list for Claude API
# TODO: Implement dispatch_tool() router
