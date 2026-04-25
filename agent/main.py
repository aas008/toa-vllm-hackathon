"""
CLI Entry Point & Orchestration

Parses CLI arguments and wires together the LLM client, agent loop,
tools, and reporter to run the full tuning pipeline.

SOURCE: ai-perf-hackathon/agent/main.py (adapted)

CLI Arguments:
    --vllm-endpoint   URL of the vLLM server (e.g. http://gpu:8000)
    --vllm-host       SSH host for remote commands on the GPU node
    --model           Model name (e.g. meta-llama/Llama-3.1-8B)
    --api-key         Anthropic API key (or $ANTHROPIC_API_KEY)
    --max-iterations  Max agent loop iterations (default: 30)
    --output          Output directory for reports (default: reports/)
    --profiles        Benchmark profiles to run (default: all 4)

Changes from source:
    - Replace --sut/--benchmark with --vllm-endpoint, --vllm-host
    - Add --profiles flag for workload selection
    - Wire vLLM-specific tools and system prompt
"""

# TODO: Implement CLI argument parsing (argparse)
# TODO: Initialize ClaudeClient from llm.py
# TODO: Initialize SSHClient from ssh_client.py
# TODO: Build tool definitions and dispatch table
# TODO: Call run_agent_loop() from agentic.py
# TODO: Generate report via reporter.py
