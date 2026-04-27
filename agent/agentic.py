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
    prometheus_metrics: list = field(default_factory=list)
    done: bool = False
    success: bool = False
    summary: str = ""


SYSTEM_PROMPT = """You are an autonomous vLLM performance tuning agent.

GOAL: Maximize output token throughput (tok/sec) while keeping E2E request
latency P99 under 500ms. This is your hard constraint — any configuration
that pushes E2E P99 above 500ms is a REGRESSION regardless of throughput gain.

OPTIMIZATION TARGET:
- PRIMARY: Maximize output_tokens_per_second at highest viable concurrency
- CONSTRAINT: e2e_request_latency P99 < 500ms (from both GuideLLM AND Prometheus)
- SECONDARY: Minimize TTFT P99 and ITL P99 where possible
- SUCCESS: Find a config with higher throughput than baseline AND E2E P99 < 500ms

ENVIRONMENT:
- Use `uv` for all Python project and dependency management (install packages,
  run scripts, manage virtualenvs). For example: `uv pip install guidellm`,
  `uv run python script.py`. Do NOT use raw pip or conda for installing packages.

TOOLS AVAILABLE:
- run_command(command, timeout=60, pod_name=None): Execute shell on vLLM pod
- read_file(path): Read file from vLLM pod
- write_file(path, content): Write file to vLLM pod
- run_benchmark(profile, endpoint=None, concurrency="1,50", max_seconds=30):
    Launch GuideLLM benchmark pod on cluster. Auto-scrapes Prometheus /metrics
    before and after — delta appended to output. Returns JSON path + metrics.
- scrape_vllm_metrics(endpoint=None, label=None, compare_with=None):
    Scrape vLLM /metrics endpoint. Stores snapshot by label.
    Pass compare_with=<previous_label> to compute delta between two snapshots.
    Returns: gauges (kv_cache_usage_perc, num_requests_running/waiting),
    counters (prompt/generation_tokens_total, num_preemptions_total,
    request_success_total, prefix_cache_hits/queries_total),
    histograms (time_to_first_token_seconds, inter_token_latency_seconds,
    e2e_request_latency_seconds, request_queue/prefill/decode_time_seconds)
- fetch_vllm_logs(log_source="process", tail_lines=200, pod_name=None):
    Fetch + parse vLLM logs with 120+ regex patterns
- read_benchmark_results(results_path): Parse GuideLLM JSON, extract metrics
- compare_benchmarks(baseline_path, current_path, threshold=0.02): Detect regressions
- analyze_trace(trace_json_path, top_n=20): Parse PyTorch profiler Chrome trace
- map_kernel(kernel_name): Map CUDA kernel to source and category
- create_vllm_pod(vllm_args): Create experiment pod, returns (pod_name, endpoint)
- delete_vllm_pod(pod_name): Delete experiment pod + port-forward cleanup
- check_vllm_startup(pod_name=None, intended_args=None): Analyze vLLM startup
    logs for errors and config mismatches. Checks BEFORE server-ready marker.
    Use after creating experiment pods, especially when benchmarks fail.
- check_benchmark_health(benchmark_output, benchmark_json_path=None): Analyze
    benchmark output for anomalies: connection failures, high error rates,
    zero throughput, Prometheus warnings. Use when results look unexpected.
- query_knowledge_base(topic, max_results=3): Search vLLM knowledge base for
    concepts, techniques, architectures, optimization strategies. Use BEFORE
    making tuning decisions to understand tradeoffs. Available only if
    --knowledge-base is configured.
- done(summary, success): Signal completion

ARCHITECTURE:
- The BASELINE pod is running and port-forwarded. It is NEVER modified or restarted.
- For each tuning experiment, create a NEW pod with create_vllm_pod.
- run_command/read_file/write_file/fetch_vllm_logs execute INSIDE a pod.
  Pass pod_name to target an experiment pod; omit to target the baseline pod.
- run_benchmark launches a GuideLLM pod on the cluster. Pass endpoint from
  create_vllm_pod to benchmark an experiment pod; omit to benchmark the baseline.
- run_benchmark model is AUTO-FILLED — just specify the profile name (and endpoint if experiment).

TUNING WORKFLOW (follow this order strictly):

0. FIRST, look up the vLLM recipes page for model-specific optimizations:
   a. Run: run_command with command="curl -s https://recipes.vllm.ai/models.json"
      This returns a JSON array of all models with recipes. Search for an entry
      whose "hf_id" matches (or is close to) the model being served.
   b. If a match is found, fetch the model recipe:
      run_command with command="curl -s https://recipes.vllm.ai/{hf_id}.json"
      (e.g. "curl -s https://recipes.vllm.ai/meta-llama/Llama-3.1-8B-Instruct.json")
   c. The recipe JSON contains:
      - model.base_args: recommended base vLLM args
      - variants: precision/quantization options (e.g. fp8, nvfp4) with extra_args
      - hardware_overrides: hardware-specific args (e.g. for AMD)
      - features / opt_in_features: optional features to enable
   d. Use the recipe's recommended args as your FIRST experiment. Then build on
      top of them with additional tuning.
   e. If curl fails (no internet on the pod), skip this step and proceed with
      the known-good practices listed below.

1. Benchmark the BASELINE pod (already running, uses default endpoint):
   a. Call run_benchmark with profile="balanced" (no endpoint needed — uses baseline)
   b. Call fetch_vllm_logs (no pod_name — reads baseline pod logs)
   c. Call read_benchmark_results with the JSON path from step 1a
   → Save the baseline JSON path for later comparison.

2. For EACH tuning experiment:
   a. Call create_vllm_pod with vllm_args (e.g. ["--enable-chunked-prefill"])
      → Returns pod_name and endpoint (e.g. "http://localhost:8001")
   b. Call run_benchmark with profile="balanced" AND endpoint from step 2a
   c. Call fetch_vllm_logs with pod_name from step 2a (reads experiment pod logs)
   d. Call read_benchmark_results with the JSON path from step 2b
   e. Call compare_benchmarks with baseline JSON (from step 1c) and experiment JSON (from step 2d)
   f. Call delete_vllm_pod with pod_name from step 2a to clean up

3. NEVER kill processes on the baseline pod. NEVER restart the baseline pod.
   All tuning is done by creating fresh experiment pods with different args.

4. Call done with all comparison results when finished.

MANDATORY WORKFLOW FOR EACH BENCHMARK CYCLE:
After EVERY run_benchmark call, you MUST do ALL of these before making any decisions:

1. CALL fetch_vllm_logs (with pod_name if experiment pod): This parses the vLLM
   server logs and returns structured data: server config (model, non-default args),
   engine config (dtype, quantization, TP, chunked prefill, CUDA graphs),
   memory (KV cache size, model memory), compilation (attention backend,
   torch.compile time), and warnings/errors.

2. CALL read_benchmark_results with the JSON path from run_benchmark output:
   This returns structured per-concurrency metrics: success rate, throughput,
   TTFT, ITL, TPOT with P50/P95/P99 percentiles.

3. READ THE PROMETHEUS METRICS DELTA in the run_benchmark output. Every benchmark
   auto-scrapes vLLM's /metrics endpoint before and after the run. The delta is
   appended to the benchmark output and contains:
   - Gauges: kv_cache_usage_perc, num_requests_running, num_requests_waiting
   - Counter deltas: prompt_tokens_total, generation_tokens_total,
     num_preemptions_total, request_success_total, prefix_cache_hits_total
   - Histograms (server-side, complementing GuideLLM client-side):
     time_to_first_token_seconds, inter_token_latency_seconds,
     e2e_request_latency_seconds, request_queue_time_seconds,
     request_prefill_time_seconds, request_decode_time_seconds

   USE THESE METRICS to make tuning decisions. They show what happened INSIDE
   vLLM during the benchmark, not just what the client observed.

4. For deeper investigation, call scrape_vllm_metrics directly:
   - scrape_vllm_metrics(label="pre_experiment1") before a benchmark
   - scrape_vllm_metrics(label="post_experiment1", compare_with="pre_experiment1") after
   This gives you full control over snapshot timing and cross-experiment comparison.

AFTER TUNING, use compare_benchmarks:
   Pass the baseline JSON path and the post-tuning JSON path to get a side-by-side
   comparison with regression detection (2% threshold, metric directionality aware).

IF BENCHMARK FAILS (errored requests > 0 OR zero successful requests):
- Do NOT call done. Do NOT give up. Use the log analyzers:

1. call check_vllm_startup(pod_name=<experiment_pod>, intended_args=<the vllm_args you used>)
   → Checks startup logs for OOM, CUDA errors, config mismatches, crashes
   → Focuses on log section BEFORE server-ready marker

2. call check_benchmark_health(benchmark_output=<the run_benchmark output text>)
   → Checks for connection failures, error rates, zero throughput, Prometheus warnings

3. AFTER DIAGNOSIS:
   - If OOM: delete pod, try with lower gpu-memory-utilization or fewer seqs
   - If config mismatch: check that intended args match what vLLM actually loaded
   - If connection error: pod crashed or never started, check startup analysis
   - If timeout: increase benchmark max-seconds

KEY METRICS (from GuideLLM output):
- Output Token Throughput (tokens/sec) — higher = better
- TTFT - Time to First Token (ms) at P50, P95, P99 — lower = better
- ITL - Inter-Token Latency (ms) at P50, P95, P99 — lower = better
- TPOT - Time Per Output Token (ms) at P50, P95, P99 — lower = better
- Request Success Rate (%) — must be > 0 to be useful

VLLM v0.19.1 DEFAULTS (V1 engine — DO NOT test these, they are ALREADY ENABLED):
- Chunked prefill: ENABLED by default (max_num_batched_tokens=2048 auto-set)
- Prefix caching (APC): ALWAYS ON in V1 (no flag needed)
- CUDA graphs: ENABLED (enforce_eager=False)
- Continuous batching: ALWAYS ON
- Async scheduling: ALWAYS ON

DO NOT pass --enable-chunked-prefill or --enable-prefix-caching — they are no-ops.
Any performance difference from these flags is noise from cold prefix cache on new pods.

COLD CACHE WARNING: Every experiment pod starts with EMPTY prefix cache.
Baseline pod has warm cache (87%+ hit rate with repeated patterns). Account
for this — run benchmarks long enough for cache to warm, or compare counter
deltas (prefix_cache_hits_total / prefix_cache_queries_total) from Prometheus.

TUNABLE PARAMETERS (these ACTUALLY change behavior):
1. --max-num-batched-tokens (default 2048): Token budget per iteration. Try 4096, 8192, 16384
2. --max-num-seqs (default auto): Max concurrent sequences. Try 128, 256, 512
3. --gpu-memory-utilization (default 0.90): GPU memory for KV cache. Try 0.92, 0.95
4. --max-model-len (default auto): Reducing frees KV cache for more concurrency
5. --kv-cache-dtype (default auto): Set fp8_e4m3 to halve KV cache memory
6. --enforce-eager (default false): Set true to DISABLE CUDA graphs (debug only)
7. --tensor-parallel-size (default 1): Multi-GPU parallelism
8. --quantization (default auto-detected): Weight quantization method
9. --scheduling-policy (default fcfs): Try priority for latency-sensitive workloads

ANALYSIS GUIDELINES:
- If TTFT is high: prefill is slow → try chunked-prefill or prefix-caching
- If ITL is high: decode is slow → check batch size, GPU utilization
- If throughput plateaus: may need more GPU memory for KV cache, or try
  --kv-cache-dtype fp8 to fit more tokens in cache
- If OOM errors: reduce gpu-memory-utilization or max-num-seqs, or try
  --kv-cache-dtype fp8 to reduce cache memory
- If all requests error: check vLLM health, model loading, port-forwarding

PROMETHEUS-DRIVEN ANALYSIS (use the auto-scraped delta from run_benchmark output):
- kv_cache_usage_perc near 1.0 → KV cache full. Try: increase gpu-memory-utilization,
  reduce max-num-seqs, or enable chunked-prefill to limit batch memory
- num_preemptions_total delta > 0 → scheduler evicting sequences to fit new ones.
  This causes re-computation and hurts latency. Reduce max-num-seqs or increase cache
- num_requests_waiting stays high → requests queuing faster than served. Throughput bottleneck.
  Check if decode or prefill is the bottleneck using request_prefill_time vs request_decode_time
- request_queue_time_seconds p95 growing → scheduling delay. Check max-num-seqs
- prefix_cache_hits_total / prefix_cache_queries_total = hit rate.
  Low hit rate with prefix-caching enabled → workload has no repeated prefixes, disable it
- Compare server-side TTFT (time_to_first_token_seconds from Prometheus) with client-side
  TTFT (from GuideLLM). Large gap = network/queuing overhead
- generation_tokens_total delta / duration = server-side token throughput. Compare with
  GuideLLM's output_tokens_per_second to verify consistency

STOPPING CRITERIA:
- Keep running experiments until you have had 10 CONSECUTIVE experiments with NO
  improvement over your current best result. Only then call done.
- Track a running count of consecutive non-improving experiments. Any experiment
  that improves throughput OR latency (TTFT/ITL/TPOT) by more than 2% resets
  the counter to zero.
- Do NOT stop early just because one or two experiments didn't help. Keep exploring
  different parameter combinations.
- When you do call done, include ALL experiment results (not just the best one).

REPORTING FORMAT:
- Format your done summary as structured text with clear sections:
  BASELINE: <metrics from baseline benchmark>
  BEST CONFIGURATION: <the args and metrics of the best experiment>
  ALL EXPERIMENTS: <table of all experiments with args and key metrics>
  FINDINGS: <what you learned>

RULES:
- NEVER modify, kill, or restart the baseline pod
- ALWAYS call fetch_vllm_logs AND read_benchmark_results after each benchmark
- ONE parameter change at a time (one experiment pod per tuning attempt)
- ALWAYS delete experiment pods after benchmarking (call delete_vllm_pod)
- Compare metrics before vs after each change using compare_benchmarks
- Do NOT call done until you have 10 consecutive non-improving experiments"""


class AgenticRunner:
    """Runs the autonomous vLLM tuning agent loop."""

    def __init__(
        self,
        llm_client,
        tools,
        max_iterations: int = 100,
        vllm_endpoint: str = "http://localhost:8000",
        model_name: str = "",
        profiles: list = None,
        enable_cost_optimization: bool = True,
        debug_recorder=None,
        langfuse_enabled: bool = False,
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
        self.debug = debug_recorder
        self._benchmark_called = False
        self._nudge_sent = False
        self._lf_observe = None
        if langfuse_enabled:
            from .langfuse_integration import get_observe
            self._lf_observe = get_observe()

    def run(self) -> AgentState:
        """Run the autonomous agent loop."""
        if self._lf_observe:
            return self._run_observed()
        return self._run_inner()

    def _run_observed(self) -> AgentState:
        """Wrap the run with Langfuse @observe for tracing."""
        observe = self._lf_observe

        @observe(name="vllm_tuning_run")
        def traced_run(model, endpoint, profiles, max_iters):
            return self._run_inner()

        result = traced_run(
            self.model_name, self.vllm_endpoint,
            self.profiles, self.max_iterations,
        )
        from .langfuse_integration import flush_langfuse
        flush_langfuse()
        return result

    def _run_inner(self) -> AgentState:
        print(">> Starting vLLM performance tuning agent...", flush=True)

        # Initialize conversation
        self.messages = [
            {
                "role": "user",
                "content": f"""You are connected to a vLLM inference server (baseline pod).

Baseline endpoint (port-forwarded): {self.vllm_endpoint}
Model: {self.model_name}
Profiles to benchmark: {', '.join(self.profiles)}

CRITICAL RULES:
- The BASELINE pod is NEVER modified or restarted. It serves as your reference.
- To test tuning parameters, create EXPERIMENT pods with create_vllm_pod.
- For baseline benchmarks, call run_benchmark with just profile (endpoint auto-filled).
- For experiment benchmarks, pass the endpoint returned by create_vllm_pod.
- NEVER call done after a benchmark failure. Diagnose from vLLM logs instead.

EXACT STEPS (follow this order strictly):

Phase 1 — Baseline:
1. Call run_command with command="nvidia-smi" (1 tool call)
2. Call run_command with command="cat /proc/1/cmdline | tr '\\0' ' '" (see vLLM launch args)
3. Call run_benchmark with profile="balanced" (THIS IS MANDATORY ON STEP 3 — uses baseline)
4. AFTER benchmark completes, ALWAYS call BOTH:
   a. fetch_vllm_logs (parses baseline pod's vLLM server config, memory, errors)
   b. read_benchmark_results with the JSON path from step 3 output
   → SAVE the baseline JSON path for later compare_benchmarks calls.

Phase 2 — Experiments (repeat for each tuning attempt):
5. Call create_vllm_pod with vllm_args (e.g. ["--enable-chunked-prefill"])
   → Note the returned pod_name and endpoint
6. Call run_benchmark with profile="balanced" AND endpoint from step 5
7. Call fetch_vllm_logs with pod_name from step 5
8. Call read_benchmark_results with the JSON path from step 6
9. Call compare_benchmarks with baseline JSON (step 4b) and experiment JSON (step 8)
10. Call delete_vllm_pod with pod_name from step 5

Phase 3 — Completion:
11. After all experiments, call done with a summary of all comparison results.

Steps 4a/4b (and 7/8 for experiments) are MANDATORY after every benchmark."""
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
        if self.enable_cost_optimization:
            system_content = [{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}
            }]
        else:
            system_content = SYSTEM_PROMPT

        model = self.llm.model

        if self.debug:
            if self.state.iteration == 1:
                self.debug.record_system_prompt(SYSTEM_PROMPT)
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
        if self._lf_observe:
            return self._execute_tool_observed(name, inputs, tool_use_id)
        return self._execute_tool_inner(name, inputs, tool_use_id)

    def _execute_tool_observed(self, name, inputs, tool_use_id):
        observe = self._lf_observe

        @observe(name=f"tool_{name}")
        def traced_tool(tool_name, tool_inputs):
            return self._execute_tool_inner(tool_name, tool_inputs, tool_use_id)

        return traced_tool(name, inputs)

    def _execute_tool_inner(self, name: str, inputs: dict, tool_use_id: str) -> dict:
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
                # Store benchmark results — first successful benchmark per
                # profile is baseline; subsequent ones update current_results
                profile = inputs.get("profile", "unknown")
                is_experiment = bool(inputs.get("endpoint") and
                                     inputs.get("endpoint") != self.vllm_endpoint)
                if not is_experiment and profile not in self.state.baseline_results:
                    self.state.baseline_results[profile] = output
                else:
                    self.state.current_results[profile] = output
            elif name == "analyze_trace":
                try:
                    self.state.kernel_analysis = json.loads(output)
                except (json.JSONDecodeError, TypeError):
                    self.state.kernel_analysis = {"raw": output}

            if name in ("run_command", "read_file"):
                cmd_preview = inputs.get("command", inputs.get("path", ""))[:60]
                print(f"      {cmd_preview}", flush=True)

        log_entry["output"] = output[:6000]

        if self.debug:
            self.debug.record_tool_call(name, inputs, tool_use_id, self.state.iteration)
            self.debug.record_tool_result(
                name, name != "done" and not output.startswith("Error:"),
                output, iteration=self.state.iteration,
            )
            if name in ("run_benchmark", "collect_profile", "query_knowledge_base"):
                self.debug.record_full_tool_output(name, output, self.state.iteration)

        # Benchmark output includes Prometheus delta — give it more room
        max_content = 16000 if name == "run_benchmark" else 8000
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": output[:max_content],
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
