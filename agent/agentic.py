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
- run_eval: Run a complete benchmark cycle using the eval pipeline (runs REMOTELY).
  Handles full vLLM server lifecycle: start → warmup → benchmark → scrape metrics → kill.
  Workloads: throughput, latency, mixed, long-context.
  Returns structured JSON with throughput, latency percentiles, TTFT, GPU cache usage.
- analyze_eval_results: Analyze all eval results to find optimal configs (runs REMOTELY).
  Shows top configs by objective (max_throughput, min_latency, balanced) and Pareto frontier.
- done: Signal completion with summary

TUNING WORKFLOW (follow this order strictly):
1. Run baseline eval:
   a. Call run_eval with workload="throughput" (uses default server config)
   b. Note the baseline metrics (throughput, latency, TTFT)

2. For EACH tuning experiment:
   a. Call run_eval with workload="throughput" AND modified server params
      (e.g. enable_chunked_prefill=true, max_num_seqs=256, etc.)
   b. Compare metrics against baseline
   c. Record whether it improved or regressed

3. After all experiments:
   a. Call analyze_eval_results to get overall analysis and Pareto frontier
   b. Call done with all findings

Each run_eval call handles the FULL server lifecycle automatically.
You do NOT need to manage vLLM processes, health checks, or port-forwarding.

KEY METRICS (from GuideLLM output):
- Output Token Throughput (tokens/sec) — higher = better
- TTFT - Time to First Token (ms) at P50, P95, P99 — lower = better
- ITL - Inter-Token Latency (ms) at P50, P95, P99 — lower = better
- TPOT - Time Per Output Token (ms) at P50, P95, P99 — lower = better
- Request Success Rate (%) — must be > 0 to be useful

VLLM TUNABLE PARAMETERS (pass these to run_eval as server params):
1. --max-num-seqs (1-1024, default 256): Max concurrent sequences per iteration
2. --max-num-batched-tokens (256-32768, default auto): Max tokens per batch
3. --gpu-memory-utilization (0.80-0.95, default 0.90): GPU memory for KV cache
4. --enable-chunked-prefill (bool, default false): Chunk long prefills
5. --enable-prefix-caching (bool, default false): Cache common prefixes
6. --max-model-len (int, default auto): Max context length
7. --enforce-eager (bool, default false): Disable CUDA graphs
8. --tensor-parallel-size (1-8, default 1): Multi-GPU parallelism
9. --quantization (null/fp8/awq/gptq): Quantization method
10. --scheduling-policy (fcfs/priority): Request scheduling

ANALYSIS GUIDELINES:
- If TTFT is high: prefill is slow → try chunked-prefill or prefix-caching
- If ITL is high: decode is slow → check batch size, GPU utilization
- If throughput plateaus: may need more GPU memory for KV cache
- If OOM errors: reduce gpu-memory-utilization or max-num-seqs
- If all requests error: check vLLM health, model loading, port-forwarding

INCREMENTAL REPORTING:
- After EVERY benchmark+read_benchmark_results cycle, call done with your findings SO FAR.
  Include: baseline metrics, experiment metrics, comparison results, and next steps.
- You can call done multiple times. Each call OVERWRITES the previous summary.
  This ensures the report always reflects your latest progress.
- Format your done summary as structured text with clear sections:
  BASELINE: <metrics from baseline benchmark>
  EXPERIMENT <name>: <metrics and comparison vs baseline>
  FINDINGS: <what you learned>
  NEXT STEPS: <what you'd try next if you had more iterations>
- If you are on iteration 25+ out of 30, call done immediately with everything you have.
  Do NOT start new experiments after iteration 25.

RULES:
- ONE parameter change at a time (one experiment per tuning attempt)
- Compare metrics before vs after each change
- Call done after EVERY completed benchmark cycle (partial results are fine)"""


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

        # Extract port from vllm_endpoint URL
        from urllib.parse import urlparse
        _parsed = urlparse(self.vllm_endpoint)
        _port = _parsed.port or 8000

        # Initialize conversation
        self.messages = [
            {
                "role": "user",
                "content": f"""You are connected to a vLLM inference server.

Model: {self.model_name}
Profiles to benchmark: {', '.join(self.profiles)}

IMPORTANT: Use port={_port} and cuda_devices="2" for ALL run_eval calls.
Port 8000 and GPUs 0-1 are occupied by another process.

EXACT STEPS (follow this order strictly):

Phase 1 — Baseline:
1. Call run_command with command="nvidia-smi" (1 tool call)
2. Call run_eval with workload="throughput", port={_port}, cuda_devices="2" (baseline with default config)
3. Note baseline metrics

Phase 2 — Experiments:
4. Call run_eval with modified params (e.g. enable_chunked_prefill=true), always with port={_port}, cuda_devices="2"
5. Compare against baseline
6. Repeat with different params

Phase 3 — Analysis:
7. Call analyze_eval_results to get overall analysis
8. Call done with all findings"""
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

            # Budget warning: when approaching iteration cap, force report save
            remaining = self.max_iterations - self.state.iteration
            if remaining == 5:
                self.messages.append({
                    "role": "user",
                    "content": (
                        "WARNING: Only 5 iterations remaining. "
                        "Call done NOW with all findings so far. "
                        "Include baseline metrics, experiment results, and comparisons."
                    ),
                })
                print("   [Budget warning: 5 iterations left]", flush=True)

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
            print(">> Max iterations reached. Extracting results from decision log...", flush=True)
            self._extract_results_from_log()

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

    def _extract_results_from_log(self):
        """Extract benchmark results from decision_log when agent hits max iterations."""
        benchmark_outputs = []
        comparison_outputs = []
        last_agent_text = ""

        for entry in self.decision_log:
            if entry["tool"] == "run_benchmark" and entry.get("output", ""):
                benchmark_outputs.append(entry)
            elif entry["tool"] == "compare_benchmarks" and entry.get("output", ""):
                comparison_outputs.append(entry)
            elif entry["tool"] == "read_benchmark_results" and entry.get("output", ""):
                benchmark_outputs.append(entry)

        for msg in reversed(self.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            last_agent_text = block["text"]
                            break
                if last_agent_text:
                    break

        summary_parts = [f"Max iterations reached ({self.state.iteration}/{self.max_iterations})."]
        summary_parts.append(f"Benchmarks run: {len(benchmark_outputs)}")
        summary_parts.append(f"Comparisons run: {len(comparison_outputs)}")

        if comparison_outputs:
            last_comparison = comparison_outputs[-1]
            summary_parts.append(f"\nLatest comparison (iteration {last_comparison['iteration']}):")
            summary_parts.append(last_comparison["output"][:2000])

        if last_agent_text:
            summary_parts.append(f"\nAgent's last analysis:\n{last_agent_text[:1000]}")

        self.state.summary = "\n".join(summary_parts)
        self.state.success = len(benchmark_outputs) > 0

    def get_decision_log(self) -> list:
        """Get the decision log."""
        return self.decision_log
