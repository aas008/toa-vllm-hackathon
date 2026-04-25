"""
Agent Loop & vLLM System Prompt

The core agentic loop: assembles messages, calls Claude, dispatches tool calls,
and iterates until the agent signals completion or hits max iterations.

SOURCE: ai-perf-hackathon/agent/agentic.py (loop reused, prompt replaced)
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class AgentState:
    """Current state of the agent."""
    iteration: int = 0
    baseline_results: dict = field(default_factory=dict)
    current_results: dict = field(default_factory=dict)
    actions_taken: list = field(default_factory=list)
    kernel_analysis: dict = field(default_factory=dict)
    done: bool = False
    success: bool = False
    summary: str = ""


SYSTEM_PROMPT = """You are an autonomous vLLM performance tuning agent.

GOAL: Benchmark, profile, analyze, and tune a vLLM inference server to maximize
throughput while maintaining acceptable latency SLOs.

ENVIRONMENT:
- Use `uv` for all Python project and dependency management (install packages,
  run scripts, manage virtualenvs). For example: `uv pip install guidellm`,
  `uv run python script.py`. Do NOT use raw pip or conda for installing packages.

TOOLS AVAILABLE:
- run_command: Execute shell commands on the vLLM pod/host (runs REMOTELY on pod)
- read_file: Read files from the vLLM pod/host (runs REMOTELY on pod)
- write_file: Write files to the vLLM pod/host (runs REMOTELY on pod)
- run_benchmark: Run GuideLLM benchmark (runs LOCALLY, hits the port-forwarded endpoint)
- fetch_vllm_logs: Fetch + parse vLLM logs from pod with 120+ regex patterns (runs REMOTELY)
- read_benchmark_results: Read a GuideLLM JSON file and extract structured metrics (runs LOCALLY)
- compare_benchmarks: Compare two benchmark JSON files to detect regressions (runs LOCALLY)
- analyze_trace: Analyze a PyTorch profiler Chrome trace JSON (runs LOCALLY)
- map_kernel: Map a CUDA kernel name to its source and category (runs LOCALLY)
- done: Signal completion with summary

ARCHITECTURE:
- run_command/read_file/write_file/fetch_vllm_logs execute INSIDE the vLLM pod.
- run_benchmark/read_benchmark_results/compare_benchmarks run LOCALLY.
- run_benchmark endpoint and model are AUTO-FILLED — just specify the profile name.

MANDATORY WORKFLOW FOR EACH BENCHMARK CYCLE:
After EVERY run_benchmark call, you MUST do BOTH of these before making any decisions:

1. CALL fetch_vllm_logs: This parses the vLLM server logs and returns structured data:
   server config (model, non-default args), engine config (dtype, quantization, TP,
   chunked prefill, CUDA graphs), memory (KV cache size, model memory), compilation
   (attention backend, torch.compile time), and warnings/errors.

2. CALL read_benchmark_results with the JSON path from run_benchmark output:
   This returns structured per-concurrency metrics: success rate, throughput,
   TTFT, ITL, TPOT with P50/P95/P99 percentiles.

AFTER TUNING, use compare_benchmarks:
   Pass the baseline JSON path and the post-tuning JSON path to get a side-by-side
   comparison with regression detection (2% threshold, metric directionality aware).

IF BENCHMARK FAILS (errored requests > 0):
- Do NOT call done. Do NOT give up.
- Call fetch_vllm_logs to understand WHY requests are failing
- Common causes: model not loaded, wrong model name, OOM, CUDA error, timeout
- Try: run_command "curl -s http://localhost:8000/health" (inside pod)
- Try: run_command "curl -s http://localhost:8000/v1/models" (inside pod)
- Fix the issue, then re-run the benchmark

KEY METRICS (from GuideLLM output):
- Output Token Throughput (tokens/sec) — higher = better
- TTFT - Time to First Token (ms) at P50, P95, P99 — lower = better
- ITL - Inter-Token Latency (ms) at P50, P95, P99 — lower = better
- TPOT - Time Per Output Token (ms) at P50, P95, P99 — lower = better
- Request Success Rate (%) — must be > 0 to be useful

VLLM TUNABLE PARAMETERS:
1. max-num-seqs (1-1024, default 256): Max concurrent sequences per iteration
2. max-num-batched-tokens (256-32768, default auto): Max tokens per batch
3. gpu-memory-utilization (0.80-0.95, default 0.90): GPU memory for KV cache
4. enable-chunked-prefill (bool, default false): Chunk long prefills
5. enable-prefix-caching (bool, default false): Cache common prefixes
6. max-model-len (int, default auto): Max context length
7. enforce-eager (bool, default false): Disable CUDA graphs
8. tensor-parallel-size (1-8, default 1): Multi-GPU parallelism
9. quantization (null/fp8/awq/gptq): Quantization method
10. scheduling-policy (fcfs/priority): Request scheduling

ANALYSIS GUIDELINES:
- If TTFT is high: prefill is slow → try chunked-prefill or prefix-caching
- If ITL is high: decode is slow → check batch size, GPU utilization
- If throughput plateaus: may need more GPU memory for KV cache
- If OOM errors: reduce gpu-memory-utilization or max-num-seqs
- If all requests error: check vLLM health, model loading, port-forwarding

RESTARTING VLLM:
If you need to restart vLLM with new parameters:
1. First read the current launch command: cat /proc/1/cmdline | tr '\\0' ' '
2. Kill the process: kill 1
3. Start it again with your changes, using the SAME base command plus new args
4. Wait for health: retry "curl -s http://localhost:8000/health" every 5s, up to 60s
5. If it doesn't come back within 60s, call done and report the failure
Do NOT spend more than 3 iterations trying to recover a stuck server.

RULES:
- ALWAYS call fetch_vllm_logs AND read_benchmark_results after each benchmark
- ONE parameter change at a time
- Compare metrics before vs after each change using compare_benchmarks
- Only call done when you have actual performance numbers to report
- If you kill vLLM, you MUST restart it immediately with the correct command
- Do NOT spend more than 3 iterations debugging a stuck/crashed server"""


class AgenticRunner:
    """Runs the autonomous vLLM tuning agent loop."""

    def __init__(
        self,
        llm_client,
        tools,
        max_iterations: int = 30,
        vllm_endpoint: str = "http://localhost:8000",
        model_name: str = "",
        profiles: list = None,
        enable_cost_optimization: bool = True,
    ):
        self.tools = tools
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.vllm_endpoint = vllm_endpoint
        self.model_name = model_name
        self.profiles = profiles or ["balanced", "decode_heavy", "prefill_heavy", "long_context"]
        self.state = AgentState()
        self.messages: list = []
        self.decision_log: list = []
        self.enable_cost_optimization = enable_cost_optimization
        self._benchmark_called = False
        self._nudge_sent = False

    def run(self) -> AgentState:
        """Run the autonomous agent loop."""
        print(">> Starting vLLM performance tuning agent...", flush=True)

        # Initialize conversation
        self.messages = [
            {
                "role": "user",
                "content": f"""You are connected to a vLLM inference server.

Endpoint (port-forwarded, for benchmarks): {self.vllm_endpoint}
Model: {self.model_name}
Profiles to benchmark: {', '.join(self.profiles)}

CRITICAL RULES:
- The run_benchmark tool runs LOCALLY and the endpoint/model are AUTO-FILLED. Just specify profile.
- Do NOT pass endpoint or model to run_benchmark. Only pass profile (e.g. profile="balanced").
- You MUST call run_benchmark within the first 3 tool calls. No exceptions.
- Do NOT call curl, cat logs, journalctl, or any other exploration commands before benchmarking.
- NEVER call done after a benchmark failure. Diagnose from vLLM logs instead.

EXACT STEPS (follow this order strictly):
1. Call run_command with command="nvidia-smi" (1 tool call)
2. Call run_command with command="cat /proc/1/cmdline | tr '\\0' ' '" (see vLLM launch args)
3. Call run_benchmark with profile="balanced" (THIS IS MANDATORY ON STEP 3)
4. AFTER benchmark completes, ALWAYS call BOTH of these tools:
   a. fetch_vllm_logs (parses vLLM server config, memory, errors from pod logs)
   b. read_benchmark_results with the JSON path from step 3 output
5. If benchmark had errors: diagnose from fetch_vllm_logs output, fix issues, re-benchmark
6. If benchmark succeeded: analyze metrics, tune parameters, re-benchmark
7. After tuning + re-benchmark, call compare_benchmarks with baseline and new JSON paths
8. Call done ONLY when you have actual performance numbers to report

Steps 4a and 4b are MANDATORY after every benchmark."""
            }
        ]

        # Agentic loop
        while not self.state.done and self.state.iteration < self.max_iterations:
            self.state.iteration += 1
            print(f"\n>> Iteration {self.state.iteration}/{self.max_iterations}", flush=True)

            # Nudge: if we've done 3+ iterations without benchmarking, inject a reminder
            if (self.state.iteration >= 4
                    and not self._benchmark_called
                    and not self._nudge_sent):
                self._nudge_sent = True
                self.messages.append({
                    "role": "user",
                    "content": (
                        "STOP EXPLORING. You have spent enough iterations on system discovery. "
                        "Call the run_benchmark tool NOW with profile=\"balanced\". "
                        "Do NOT call run_command again until you have benchmark results. "
                        "The endpoint and model are auto-filled — just specify the profile."
                    ),
                })
                print("   [Nudge injected: forcing benchmark]", flush=True)

            # Call LLM with tools
            response = self._call_llm_with_tools()

            # Process response
            if response.stop_reason == "tool_use":
                self._handle_tool_calls(response)
            elif response.stop_reason == "end_turn":
                # Agent is thinking, add response and continue
                self._add_assistant_message(response)
                self.messages.append({
                    "role": "user",
                    "content": "Continue. Use tools to explore, benchmark, or apply changes."
                })

        if not self.state.done:
            print(">> Max iterations reached.", flush=True)
            self.state.summary = "Max iterations reached without completion."

        return self.state

    def _call_llm_with_tools(self):
        """Call LLM with tool definitions and cost optimizations."""
        # Build system prompt with caching
        if self.enable_cost_optimization:
            system_content = [{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}
            }]
        else:
            system_content = SYSTEM_PROMPT

        model = self.llm.model

        response = self.llm.client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_content,
            tools=self.tools.get_tool_definitions(),
            messages=self.messages,
        )

        # Track tokens
        self.llm._get_usage(model).add(
            response.usage.input_tokens,
            response.usage.output_tokens
        )

        return response

    def _handle_tool_calls(self, response):
        """Handle tool calls from LLM response."""
        self._add_assistant_message(response)

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = self._execute_tool(block.name, block.input, block.id)
                tool_results.append(result)

        self.messages.append({
            "role": "user",
            "content": tool_results
        })

    def _execute_tool(self, name: str, inputs: dict, tool_use_id: str) -> dict:
        """Execute a single tool and return result."""
        print(f"   Tool: {name}", flush=True)

        # Log decision (output filled in after execution)
        log_entry = {
            "iteration": self.state.iteration,
            "tool": name,
            "inputs": inputs,
            "timestamp": datetime.utcnow().isoformat(),
            "output": "",
        }
        self.decision_log.append(log_entry)

        if name == "run_benchmark":
            self._benchmark_called = True

        if name == "done":
            self.state.done = True
            self.state.success = inputs.get("success", False)
            self.state.summary = inputs.get("summary", "")
            output = "Agent signaled completion."

        else:
            # Dispatch to tools module (returns ToolResult dataclass)
            result = self.tools.dispatch(name, inputs)
            output = result.output or ""
            if result.error:
                output = f"Error: {result.error}"

            # Track state changes
            if name == "write_file":
                self.state.actions_taken.append({
                    "type": "write_file",
                    "path": inputs.get("path", ""),
                    "timestamp": datetime.utcnow().isoformat(),
                })
            elif name == "run_benchmark":
                # Store benchmark results
                profile = inputs.get("profile", "unknown")
                self.state.current_results[profile] = output
            elif name == "analyze_trace":
                try:
                    self.state.kernel_analysis = json.loads(output)
                except (json.JSONDecodeError, TypeError):
                    self.state.kernel_analysis = {"raw": output}

            if name in ("run_command", "read_file"):
                cmd_preview = inputs.get("command", inputs.get("path", ""))[:60]
                print(f"      {cmd_preview}", flush=True)

        log_entry["output"] = output[:4000]

        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": output[:8000],  # Truncate long outputs
        }

    def _add_assistant_message(self, response):
        """Add assistant response to messages."""
        content = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
                print(f"   Agent: {block.text[:120]}...", flush=True)
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        self.messages.append({"role": "assistant", "content": content})

    def get_decision_log(self) -> list:
        """Get the decision log."""
        return self.decision_log
