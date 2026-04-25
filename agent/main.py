#!/usr/bin/env python3
"""
vLLM Performance Tuning Agent - CLI Entry Point
"""
import argparse
import os
import sys

from .llm import ClaudeClient, get_model_id
from .ssh_client import SSHClient


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
        required=True,
        help="SSH hostname for the GPU node running vLLM"
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
        default=30,
        help="Max agent loop iterations (default: 30)"
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
    print(f"Output Dir:     {args.output}")

    # Test SSH connectivity
    print_step("Testing SSH connectivity to vLLM host...")
    ssh = SSHClient(args.vllm_host, user=args.ssh_user)
    if not ssh.test_connection():
        print(f"Error: Cannot connect to vLLM host ({args.vllm_host})")
        sys.exit(1)
    print(f"  SSH to {args.vllm_host}: OK")

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

    # TODO: Step 4 — Wire up AgenticRunner with vLLM tools and system prompt
    print_step("Agent loop not yet implemented (Step 4). Exiting.")


if __name__ == "__main__":
    main()
