#!/usr/bin/env python3
"""
vLLM Performance Tuning Agent - CLI Entry Point
"""
import argparse
import atexit
import json
import os
import sys

from .llm import ClaudeClient, get_model_id
from .ssh_client import SSHClient
from .agentic import AgenticRunner
from .tools import AgentTools, OcExecutor, SSHExecutor
from .pod_manager import PodManager
from .reporter import Reporter, TuningReport


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="vLLM Performance Tuning Agent - autonomous benchmarking, profiling, and optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full tuning loop
  python -m agent \\
      --vllm-endpoint http://gpu:8000 \\
      --vllm-host aansharm-0-yxg5 \\
      --model meta-llama/Llama-3.1-8B \\
      --api-key $ANTHROPIC_API_KEY

  # Use a specific Claude model
  python -m agent --vllm-endpoint http://gpu:8000 --vllm-host gpu \\
      --model meta-llama/Llama-3.1-8B --claude-model opus

  # Limit iterations and select profiles
  python -m agent --vllm-endpoint http://gpu:8000 --vllm-host gpu \\
      --model meta-llama/Llama-3.1-8B \\
      --max-iterations 10 --profiles balanced decode_heavy

Available Claude models: sonnet (default), opus, haiku
        """
    )

    parser.add_argument(
        "--vllm-endpoint",
        required=True,
        help="URL of the vLLM server (e.g. http://gpu:8000)"
    )
    parser.add_argument(
        "--vllm-host",
        default=None,
        help="SSH hostname for the GPU node running vLLM (required for SSH mode)"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model served by vLLM (e.g. meta-llama/Llama-3.1-8B)"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key (default: $ANTHROPIC_API_KEY)"
    )
    parser.add_argument(
        "--claude-model",
        default="sonnet",
        help="Claude model for the agent (sonnet, opus, haiku, or full ID). Default: sonnet"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Max agent loop iterations (default: 100)"
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["balanced", "decode_heavy", "prefill_heavy", "long_context"],
        choices=["balanced", "decode_heavy", "prefill_heavy", "long_context"],
        help="Benchmark profiles to run (default: all 4)"
    )
    parser.add_argument(
        "--ssh-user",
        default="root",
        help="SSH user for vLLM host (default: root)"
    )
    parser.add_argument(
        "--output", "-o",
        default="reports",
        help="Output directory for reports (default: reports/)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    # Execution mode: SSH or oc exec
    parser.add_argument(
        "--oc-mode",
        action="store_true",
        default=False,
        help="Use 'oc exec' instead of SSH to reach the vLLM pod"
    )
    parser.add_argument(
        "--oc-namespace",
        default="toa-hack",
        help="OpenShift namespace for oc exec (default: toa-hack)"
    )
    parser.add_argument(
        "--oc-pod",
        default=None,
        help="Pod name for oc exec (required if --oc-mode)"
    )
    parser.add_argument(
        "--kubeconfig",
        default=os.environ.get("KUBECONFIG"),
        help="Path to kubeconfig file (default: $KUBECONFIG)"
    )
    parser.add_argument(
        "--pod-template",
        default=None,
        help=(
            "Path to a pod YAML template for creating experiment pods "
            "(e.g. aanya-pod.yaml). Enables the create_vllm_pod/delete_vllm_pod tools. "
            "Requires --oc-mode."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help=(
            "Run the GPU profiling agent instead of the tuning agent. "
            "Deploys profiled vLLM pods, collects PyTorch profiler traces, "
            "and analyzes GPU kernels. Produces a profiling report."
        ),
    )

    # Vertex AI options
    parser.add_argument(
        "--vertex",
        action="store_true",
        default=os.environ.get("CLAUDE_CODE_USE_VERTEX", "0") == "1",
        help="Use Google Cloud Vertex AI for Claude API access (default: true if CLAUDE_CODE_USE_VERTEX=1)"
    )
    parser.add_argument(
        "--vertex-project-id",
        default=os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"),
        help="Vertex AI project ID (default: $ANTHROPIC_VERTEX_PROJECT_ID)"
    )
    parser.add_argument(
        "--vertex-region",
        default=os.environ.get("CLOUD_ML_REGION", "us-east5"),
        help="Vertex AI region (default: $CLOUD_ML_REGION or us-east5)"
    )

    return parser


def print_header(msg: str):
    """Print a section header."""
    print(f"\n{'='*60}", flush=True)
    print(f"  {msg}", flush=True)
    print(f"{'='*60}\n", flush=True)


def print_step(msg: str):
    """Print a step message."""
    print(f">> {msg}", flush=True)


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate API key (only required when NOT using Vertex AI)
    if not args.vertex and not args.api_key:
        print("Error: ANTHROPIC_API_KEY not set. Use --api-key or set the environment variable.")
        print("       Alternatively, use --vertex for Google Cloud Vertex AI access.")
        sys.exit(1)

    if args.vertex and not args.vertex_project_id:
        print("Error: Vertex AI project ID not set. Use --vertex-project-id or set ANTHROPIC_VERTEX_PROJECT_ID.")
        sys.exit(1)

    # Determine backend label
    backend = "Vertex AI" if args.vertex else "Direct API"

    # Print banner
    print_header("vLLM Performance Tuning Agent v1.0")
    print(f"vLLM Endpoint:  {args.vllm_endpoint}")
    print(f"vLLM Host:      {args.vllm_host}")
    print(f"Served Model:   {args.model}")
    print(f"Claude Model:   {get_model_id(args.claude_model)}")
    print(f"Claude Backend: {backend}")
    if args.vertex:
        print(f"Vertex Project: {args.vertex_project_id}")
        print(f"Vertex Region:  {args.vertex_region}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Profiles:       {', '.join(args.profiles)}")
    print(f"Mode:           {'PROFILING' if args.profile else 'TUNING'}")
    print(f"Output Dir:     {args.output}")

    # Validate mode-specific args
    if not args.oc_mode and not args.vllm_host:
        print("Error: --vllm-host is required for SSH mode. Use --oc-mode for OpenShift.")
        sys.exit(1)

    # Set up remote executor (SSH or oc exec)
    if args.oc_mode:
        if not args.oc_pod:
            print("Error: --oc-pod is required when using --oc-mode")
            sys.exit(1)
        print_step(f"Using oc exec mode: {args.oc_namespace}/{args.oc_pod}")
        executor = OcExecutor(
            namespace=args.oc_namespace,
            pod_name=args.oc_pod,
            kubeconfig=args.kubeconfig,
        )
        # Test connectivity
        test_result = executor.run("echo OK")
        if not test_result.success:
            print(f"Error: Cannot connect to pod: {test_result.stderr}")
            sys.exit(1)
        print(f"  oc exec to {args.oc_pod}: OK")
    else:
        print_step("Testing SSH connectivity to vLLM host...")
        ssh = SSHClient(args.vllm_host, user=args.ssh_user)
        if not ssh.test_connection():
            print(f"Error: Cannot connect to vLLM host ({args.vllm_host})")
            sys.exit(1)
        print(f"  SSH to {args.vllm_host}: OK")
        executor = SSHExecutor(ssh)

    # Test Claude API connectivity
    print_step(f"Testing Claude API connectivity ({backend})...")
    llm = ClaudeClient(
        api_key=args.api_key,
        model=get_model_id(args.claude_model),
        use_vertex=args.vertex,
        vertex_project_id=args.vertex_project_id,
        vertex_region=args.vertex_region,
    )
    try:
        test_response = llm.analyze(
            system_prompt="You are a test assistant.",
            user_prompt="Reply with exactly: API_OK",
            max_tokens=16,
        )
        if "API_OK" in test_response.content:
            print(f"  Claude API: OK (model: {test_response.model})")
        else:
            print(f"  Claude API: Connected (model: {test_response.model})")
    except Exception as e:
        print(f"Error: Claude API connection failed: {e}")
        sys.exit(1)

    # Print token usage from connectivity test
    print_step("Connectivity verified. Ready to run agent loop.")
    print(f"\n  Token usage so far:")
    print(f"  {llm.get_usage_report()}")

    # Create PodManager if --pod-template is provided
    pod_manager = None
    if args.pod_template:
        if not args.oc_mode:
            print("Error: --pod-template requires --oc-mode")
            sys.exit(1)
        print_step(f"Initializing PodManager with template: {args.pod_template}")
        pod_manager = PodManager(
            namespace=args.oc_namespace,
            kubeconfig=args.kubeconfig,
            base_pod_yaml_path=args.pod_template,
        )
        # Register cleanup to delete leftover experiment pods on exit
        atexit.register(pod_manager.cleanup_all)
        print(f"  PodManager ready (namespace={args.oc_namespace})")

    # Create tools and agent
    tools = AgentTools(
        executor=executor,
        vllm_endpoint=args.vllm_endpoint,
        model_name=args.model,
        pod_manager=pod_manager,
        namespace=args.oc_namespace if args.oc_mode else None,
        kubeconfig=args.kubeconfig,
        baseline_pod_name=args.oc_pod if args.oc_mode else None,
    )

    if args.profile:
        from .profiling_agent import ProfilingRunner
        agent = ProfilingRunner(
            llm_client=llm,
            tools=tools,
            max_iterations=args.max_iterations,
            vllm_endpoint=args.vllm_endpoint,
            model_name=args.model,
        )
    else:
        agent = AgenticRunner(
            llm_client=llm,
            tools=tools,
            max_iterations=args.max_iterations,
            vllm_endpoint=args.vllm_endpoint,
            model_name=args.model,
            profiles=args.profiles,
        )

    # Run the agent loop
    mode_label = "profiling" if args.profile else "tuning"
    print_step(f"Starting {mode_label} agent loop...")
    try:
        state = agent.run()
    finally:
        if pod_manager:
            pod_manager.cleanup_all()

    # Generate report
    print_step("Generating report...")
    from datetime import datetime

    # Extract GPU info and actions from command history
    gpu_info = ""
    actions = list(state.actions_taken)
    for entry in tools.command_history:
        out = entry.get("output", "")
        cmd = entry.get("command", "")
        if entry.get("tool") == "run_command" and "nvidia-smi" in cmd and out:
            # Pull GPU name from nvidia-smi output
            for line in out.split("\n"):
                if "NVIDIA" in line and ("H100" in line or "H200" in line or "A100" in line or "GPU" in line):
                    gpu_info = line.strip()
                    break
        if entry.get("tool") == "run_command" and entry.get("success"):
            actions.append({
                "type": "run_command",
                "command": cmd[:80],
                "timestamp": "",
            })

    # Build summary from agent messages if the agent didn't call done
    summary = state.summary
    if summary == "Max iterations reached without completion.":
        # Pull the last agent message as a summary
        for msg in reversed(agent.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            summary = block["text"][:500]
                            break
                        elif hasattr(block, "type") and block.type == "text":
                            summary = block.text[:500]
                            break
                elif isinstance(content, str) and content:
                    summary = content[:500]
                if summary != "Max iterations reached without completion.":
                    break

    # Build structured baseline/final results for the report.
    def _parse_benchmark_output(results_dict: dict) -> list[dict]:
        """Extract structured metrics from stored benchmark output per profile."""
        import re
        parsed = []
        for profile, raw_output in results_dict.items():
            entry = {"profile": profile}
            if isinstance(raw_output, str):
                blocks = re.split(r'--- Concurrency: \d+', raw_output)
                text_block = blocks[-1] if len(blocks) > 1 else raw_output

                def _extract_kv(line):
                    return dict(re.findall(r'(\w+)=([\d.]+)', line))

                for line in text_block.split('\n'):
                    line_stripped = line.strip()
                    if line_stripped.startswith("Output Tokens/sec:"):
                        kv = _extract_kv(line_stripped)
                        if "mean" in kv and "throughput_tok_per_sec" not in entry:
                            entry["throughput_tok_per_sec"] = float(kv["mean"])
                    elif line_stripped.startswith("TTFT (ms):"):
                        kv = _extract_kv(line_stripped)
                        for p in ("p50", "p95", "p99"):
                            if p in kv and f"ttft_{p}" not in entry:
                                entry[f"ttft_{p}"] = float(kv[p])
                    elif line_stripped.startswith("ITL (ms):"):
                        kv = _extract_kv(line_stripped)
                        for p in ("p50", "p95", "p99"):
                            if p in kv and f"itl_{p}" not in entry:
                                entry[f"itl_{p}"] = float(kv[p])
                    elif line_stripped.startswith("TPOT (ms):"):
                        kv = _extract_kv(line_stripped)
                        for p in ("p50", "p95", "p99"):
                            if p in kv and f"tpot_{p}" not in entry:
                                entry[f"tpot_{p}"] = float(kv[p])

                if "throughput_tok_per_sec" not in entry:
                    best_metrics = {}
                    for m in re.finditer(r'\{[^{}]+\}', raw_output):
                        try:
                            obj = json.loads(m.group())
                            if obj.get("concurrency", 0) >= best_metrics.get("concurrency", 0):
                                best_metrics = obj
                        except (json.JSONDecodeError, ValueError):
                            continue
                    if best_metrics:
                        key_map = {
                            "output_tok/sec_mean": "throughput_tok_per_sec",
                            "ttft_p50": "ttft_p50", "ttft_p95": "ttft_p95", "ttft_p99": "ttft_p99",
                            "itl_p50": "itl_p50", "itl_p95": "itl_p95", "itl_p99": "itl_p99",
                            "tpot_p50": "tpot_p50", "tpot_p95": "tpot_p95", "tpot_p99": "tpot_p99",
                        }
                        for src, dst in key_map.items():
                            v = best_metrics.get(src)
                            if v is not None and dst not in entry:
                                entry[dst] = float(v)

            parsed.append(entry)
        return parsed

    baseline_results = _parse_benchmark_output(state.baseline_results)
    final_results = _parse_benchmark_output(state.current_results)

    # Collect Prometheus metric deltas from stored snapshots
    prometheus_metrics = []
    if tools.metrics_snapshots:
        try:
            from .analysis.prometheus_metrics import compute_delta
            labels = sorted(tools.metrics_snapshots.keys())
            pre_labels = [l for l in labels if l.startswith("pre_")]
            for pre_label in pre_labels:
                post_label = pre_label.replace("pre_", "post_", 1)
                if post_label in tools.metrics_snapshots:
                    delta = compute_delta(
                        tools.metrics_snapshots[pre_label],
                        tools.metrics_snapshots[post_label],
                    )
                    prometheus_metrics.append(delta.to_dict())
        except Exception:
            pass

    report = TuningReport(
        timestamp=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        model_name=args.model,
        vllm_endpoint=args.vllm_endpoint,
        gpu_info=gpu_info,
        agent_summary=summary,
        baseline_results=baseline_results,
        final_results=final_results,
        actions_taken=actions,
        kernel_analysis=state.kernel_analysis,
        token_usage=llm.get_usage_data(),
        decision_log=agent.get_decision_log(),
        prometheus_metrics=prometheus_metrics,
    )

    reporter = Reporter(output_dir=args.output)
    md_path, json_path = reporter.generate(report)
    print(f"  Markdown report: {md_path}")
    print(f"  JSON report:     {json_path}")

    # Final summary
    print_header("Agent Complete")
    print(f"  Success: {state.success}")
    print(f"  Iterations: {state.iteration}")
    print(f"  Summary: {state.summary[:200]}")
    print(f"\n  Token usage:")
    print(f"  {llm.get_usage_report()}")


if __name__ == "__main__":
    main()
