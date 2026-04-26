"""
Profiling Agent — GPU Kernel Analysis for vLLM

Separate specialized agent that deploys profiled vLLM pods, collects
PyTorch profiler traces, and analyzes GPU kernel performance.

Runs independently from the tuning agent. Invoked via:
    python -m agent --profile ...
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ProfilingState:
    """State for the profiling agent."""
    iteration: int = 0
    baseline_results: dict = field(default_factory=dict)
    current_results: dict = field(default_factory=dict)
    actions_taken: list = field(default_factory=list)
    kernel_analysis: dict = field(default_factory=dict)
    baseline_profile: dict = field(default_factory=dict)
    experiment_profiles: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    done: bool = False
    success: bool = False
    summary: str = ""


PROFILING_SYSTEM_PROMPT = """You are an autonomous vLLM GPU profiling agent.

GOAL: Profile a vLLM inference server at the GPU kernel level to identify
performance bottlenecks, characterize workload behavior, and produce
actionable findings for tuning.

TOOLS AVAILABLE:
- run_command(command, timeout=60, pod_name=None): Execute shell on vLLM pod
- deploy_profiled_pod(vllm_args, profiler_ranges="100-150", profiler_activities="CPU,CUDA"):
    Create experiment pod with PyTorch profiler injection. The webhook auto-injects
    instrumentation that wraps Worker.execute_model with torch.profiler.
    Returns pod_name + endpoint.
- collect_profile(pod_name, num_requests=160, max_tokens=50):
    Send inference requests to trigger profiling, wait for completion, extract
    key_averages table + Chrome trace. Returns: GPU utilization %, top kernels
    by time, category breakdown, top CPU ops.
- scrape_vllm_metrics(endpoint=None, label=None, compare_with=None):
    Scrape Prometheus /metrics from vLLM endpoint for server-side metrics.
- fetch_vllm_logs(log_source="process", tail_lines=200, pod_name=None):
    Fetch + parse vLLM logs with 120+ regex patterns.
- delete_vllm_pod(pod_name): Delete experiment pod + cleanup.
- query_knowledge_base(topic, max_results=3): Search vLLM knowledge base for
    kernel details, optimization techniques, architecture internals. Use to
    understand what kernels do and how to optimize them. Available only if
    --knowledge-base is configured.
- done(summary, success): Signal completion with findings.

PROFILING WORKFLOW:

Phase 1 — Baseline Profile:
1. Check GPU: run_command("nvidia-smi")
2. Deploy profiled baseline pod (no extra vLLM args):
   deploy_profiled_pod(vllm_args=[], profiler_ranges="10-30")
   → This profiles WARMUP (calls 10-30, includes JIT/CUDA graph compilation)
3. collect_profile(pod_name, num_requests=40)
4. Analyze warmup kernel breakdown
5. Delete pod, deploy again with steady-state range:
   deploy_profiled_pod(vllm_args=[], profiler_ranges="100-150")
6. collect_profile(pod_name, num_requests=160)
7. Compare warmup vs steady-state:
   - JIT overhead (torch.compile time in warmup vs none in steady-state)
   - CUDA graph capture overhead
   - Kernel distribution differences
8. Delete pod

Phase 2 — Experiment Profiles (optional, if time permits):
For each experiment config worth profiling:
9. deploy_profiled_pod(vllm_args=[config], profiler_ranges="100-150")
10. collect_profile(pod_name, num_requests=160)
11. Compare kernel breakdown vs baseline steady-state
12. Delete pod

Phase 3 — Report:
13. Call done with structured findings:
    - GPU utilization (warmup vs steady-state)
    - Top kernel categories and their time share
    - Warmup overhead characterization
    - Bottleneck identification (attention-bound vs compute-bound vs memory-bound)
    - Tuning recommendations based on kernel analysis

KERNEL ANALYSIS GUIDE:
- Attention kernels (flash_fwd, paged_attention) dominant → attention-bound
  → Try: chunked-prefill, prefix-caching, different attention backend
- GEMM/linear kernels (cutlass, cublas) dominant → compute-bound
  → Try: quantization (fp8), tensor-parallel to distribute compute
- Communication (NCCL allreduce/allgather) dominant → TP overhead
  → Try: reduce tensor-parallel-size, pipeline-parallel instead
- Memory ops (memcpy, memset) dominant → memory-bandwidth-bound
  → Try: different dtype, reduce model size, quantization
- Normalization/activation (rms_norm, silu) significant → fused kernels help
  → Check if fused implementations are being used
- High CPU ops → Python overhead or CPU-GPU sync stalls
  → Try: enforce-eager=false (enable CUDA graphs), torch.compile

CONFIDENCE LEVELS:
- Profile 1 finding → low confidence, need more data
- Profile 2+ consistent findings → medium confidence
- Warmup + steady-state + experiment all agree → high confidence

RULES:
- ALWAYS profile warmup (10-30) AND steady-state (100-150) for baseline
- ALWAYS delete profiled pods after collecting results
- Compare kernel breakdowns between configs to identify what changed
- Focus on the top 5-10 kernels — they dominate GPU time
- Report findings with specific kernel names and time percentages"""


class ProfilingRunner:
    """Runs the autonomous GPU profiling agent loop."""

    def __init__(
        self,
        llm_client,
        tools,
        max_iterations: int = 15,
        vllm_endpoint: str = "http://localhost:8000",
        model_name: str = "",
        debug_recorder=None,
    ):
        self.tools = tools
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.vllm_endpoint = vllm_endpoint
        self.model_name = model_name
        self.state = ProfilingState()
        self.messages: list = []
        self.decision_log: list = []
        self.debug = debug_recorder

    def run(self) -> ProfilingState:
        """Run the profiling agent loop."""
        print(">> Starting vLLM GPU profiling agent...", flush=True)

        self.messages = [
            {
                "role": "user",
                "content": f"""You are connected to a vLLM inference server.

Model: {self.model_name}
Endpoint: {self.vllm_endpoint}

Your job is to PROFILE this vLLM instance at the GPU kernel level.
This is NOT a tuning run — focus on collecting and analyzing profiler traces.

STEPS:
1. run_command("nvidia-smi") to check GPU
2. Profile WARMUP: deploy_profiled_pod(vllm_args=[], profiler_ranges="10-30")
   then collect_profile(pod_name, num_requests=40)
3. Delete pod, profile STEADY-STATE: deploy_profiled_pod(vllm_args=[], profiler_ranges="100-150")
   then collect_profile(pod_name, num_requests=160)
4. Analyze both profiles, compare warmup vs steady-state
5. Call done with structured findings

Start now with step 1."""
            }
        ]

        while not self.state.done and self.state.iteration < self.max_iterations:
            self.state.iteration += 1
            print(f"\n>> Profiling iteration {self.state.iteration}/{self.max_iterations}", flush=True)

            response = self._call_llm_with_tools()

            if response.stop_reason == "tool_use":
                self._handle_tool_calls(response)
            elif response.stop_reason == "end_turn":
                self._add_assistant_message(response)
                self.messages.append({
                    "role": "user",
                    "content": "Continue profiling. Use tools to deploy, collect, or analyze."
                })

        if not self.state.done:
            print(">> Max profiling iterations reached.", flush=True)
            self.state.summary = "Max profiling iterations reached."

        return self.state

    def _call_llm_with_tools(self):
        system_content = [{
            "type": "text",
            "text": PROFILING_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}
        }]

        model = self.llm.model

        if self.debug:
            if self.state.iteration == 1:
                self.debug.record_system_prompt(PROFILING_SYSTEM_PROMPT)
                self.debug.record_tools(self.tools.get_tool_definitions())
            self.debug.record_llm_request(model, self.messages, self.state.iteration)

        response = self.llm.client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_content,
            tools=self.tools.get_tool_definitions(),
            messages=self.messages,
        )

        self.llm._get_usage(model).add(
            response.usage.input_tokens,
            response.usage.output_tokens
        )

        if self.debug:
            self.debug.record_llm_response(
                model=model,
                stop_reason=response.stop_reason,
                content=response.content,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                iteration=self.state.iteration,
            )

        return response

    def _handle_tool_calls(self, response):
        self._add_assistant_message(response)

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = self._execute_tool(block.name, block.input, block.id)
                tool_results.append(result)

        self.messages.append({"role": "user", "content": tool_results})

    def _execute_tool(self, name: str, inputs: dict, tool_use_id: str) -> dict:
        print(f"   Tool: {name}", flush=True)

        log_entry = {
            "iteration": self.state.iteration,
            "tool": name,
            "inputs": inputs,
            "timestamp": datetime.utcnow().isoformat(),
            "output": "",
        }
        self.decision_log.append(log_entry)

        if name == "done":
            self.state.done = True
            self.state.success = inputs.get("success", False)
            self.state.summary = inputs.get("summary", "")
            output = "Profiling agent signaled completion."
        else:
            result = self.tools.dispatch(name, inputs)
            output = result.output or ""
            if result.error:
                output = f"Error: {result.error}"

            if name == "collect_profile":
                try:
                    self.state.experiment_profiles.append({
                        "iteration": self.state.iteration,
                        "output": output[:4000],
                    })
                except Exception:
                    pass

        log_entry["output"] = output[:6000]

        if self.debug:
            self.debug.record_tool_call(name, inputs, tool_use_id, self.state.iteration)
            self.debug.record_tool_result(
                name, name != "done" and not output.startswith("Error:"),
                output, iteration=self.state.iteration,
            )
            if name in ("collect_profile", "query_knowledge_base"):
                self.debug.record_full_tool_output(name, output, self.state.iteration)

        max_content = 16000 if name == "collect_profile" else 8000
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": output[:max_content],
        }

    def _add_assistant_message(self, response):
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
        return self.decision_log
