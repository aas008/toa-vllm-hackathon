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

TUNING ORDER:
1. BASELINE: Collect system info and run benchmark across all 4 profiles
2. PROFILE: Deploy PyTorch profiler, run inference, collect trace
3. ANALYZE: Examine kernel bottlenecks from profiler trace
4. TUNE: Adjust vLLM serving parameters based on analysis
5. VERIFY: Re-benchmark to measure improvement
6. ITERATE: Repeat steps 2-5 until convergence or max iterations

TOOLS AVAILABLE:
- run_command: Execute shell commands on the vLLM pod/host
- read_file: Read files from the vLLM pod/host
- write_file: Write files to the vLLM pod/host
- run_benchmark: Run GuideLLM benchmark with a specific profile and concurrency
- analyze_trace: Analyze a PyTorch profiler Chrome trace JSON
- map_kernel: Map a CUDA kernel name to its source and category
- done: Signal completion with summary

BENCHMARK PROFILES:
- balanced: ISL=1000, OSL=1000 (conversational AI, Q&A)
- decode_heavy: ISL=512, OSL=2048 (code gen, creative writing)
- prefill_heavy: ISL=2048, OSL=128 (classification, short answers)
- long_context: ISL=8000, OSL=1000 (RAG, document analysis)

CONCURRENCY SWEEP: 1, 50, 100, 200, 300, 500, 650

KEY METRICS (higher throughput = better, lower latency = better):
- Output Token Throughput (tokens/sec)
- TTFT - Time to First Token (ms) at P50, P95, P99
- ITL - Inter-Token Latency (ms) at P50, P95, P99
- TPOT - Time Per Output Token (ms) at P50, P95, P99
- Request Success Rate (%)

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
11. cuda-graph-max-bs (int, default 512): Max batch for CUDA graphs
12. max-running-requests (int, default 512): Max in-flight requests

SYSTEMATIC APPROACH:
1. DISCOVER:
   - nvidia-smi (GPU model, memory, utilization)
   - Check vLLM launch args and current config
   - Check GPU memory usage and KV cache allocation
   - Identify model size vs available GPU memory

2. ANALYZE bottlenecks:
   - If TTFT is high: prefill is slow, try chunked-prefill or prefix-caching
   - If ITL is high: decode is slow, check batch size, GPU utilization
   - If throughput plateaus: may need more GPU memory for KV cache
   - If OOM errors: reduce gpu-memory-utilization or max-num-seqs

3. TUNE (one change at a time):
   - Apply parameter change
   - Restart vLLM server
   - Wait for health check to pass
   - Re-benchmark the relevant profile

4. VERIFY improvement:
   - Compare throughput and latency against baseline
   - If regression > 2%, rollback the change
   - If improvement, keep and move to next parameter

PROFILING WORKFLOW:
1. Copy sitecustomize.py and profiler_config.yaml to vLLM pod
2. Set PYTHONPATH to include profiler directory
3. Restart vLLM with profiling enabled
4. Run a short benchmark (triggers profiled inference calls)
5. Retrieve Chrome trace JSON from /tmp/trace_*.json
6. Use analyze_trace tool to extract kernel stats
7. Use map_kernel tool to identify hot kernel sources

RULES:
- ALWAYS benchmark before AND after changes
- ONE parameter change at a time
- If something breaks, rollback immediately
- Log your reasoning for each decision
- Use the done tool when finished with a detailed summary including:
  - Bottlenecks found
  - Changes applied
  - Before/after performance numbers
  - Kernel analysis highlights (if profiling was done)"""


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

    def run(self) -> AgentState:
        """Run the autonomous agent loop."""
        print(">> Starting vLLM performance tuning agent...", flush=True)

        # Initialize conversation
        self.messages = [
            {
                "role": "user",
                "content": f"""You are connected to a vLLM inference server.

Endpoint: {self.vllm_endpoint}
Model: {self.model_name}
Profiles to benchmark: {', '.join(self.profiles)}

Start by exploring the system to understand the current configuration:
1. Check GPU info (nvidia-smi)
2. Check vLLM process and launch arguments
3. Check current model configuration
4. Run a baseline benchmark

Then proceed to profile, analyze, and tune."""
            }
        ]

        # Agentic loop
        while not self.state.done and self.state.iteration < self.max_iterations:
            self.state.iteration += 1
            print(f"\n>> Iteration {self.state.iteration}/{self.max_iterations}", flush=True)

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

        # Log decision
        self.decision_log.append({
            "iteration": self.state.iteration,
            "tool": name,
            "inputs": inputs,
            "timestamp": datetime.utcnow().isoformat(),
        })

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
