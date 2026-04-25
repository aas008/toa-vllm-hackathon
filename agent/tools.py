"""
Tool Definitions & Dispatch

Defines all tools available to the Claude agent and their handler functions.

SOURCE: ai-perf-hackathon/agent/tools.py (core tools reused, new tools added)

Core Tools (from ai-perf-hackathon, adapted for RemoteExecutor):
    - run_command(command, timeout)    -- Execute shell command on vLLM host/pod
    - read_file(path)                 -- Read file contents from vLLM host/pod
    - write_file(path, content)       -- Write file to vLLM host/pod
    - done(summary, success)          -- Signal agent completion

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

Architecture:
    RemoteExecutor abstracts over SSH and OpenShift (`oc exec`) execution modes.
    - OcExecutor(namespace, pod_name, kubeconfig) -- runs `oc exec` via subprocess
    - SSHExecutor(ssh_client) -- wraps the existing SSHClient

Tool Definition Format:
    Each tool is a dict with: name, description, input_schema (JSON Schema)
    Compatible with Claude's tool_use API format.

Dispatch:
    dispatch_tool(name, args, executor) -> ToolResult
"""

from __future__ import annotations

import json
import sys
import subprocess
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .ssh_client import SSHClient, SSHResult


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool: str
    success: bool
    output: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"tool": self.tool, "success": self.success, "output": self.output}
        if self.error:
            d["error"] = self.error
        return d


# ---------------------------------------------------------------------------
# RemoteExecutor abstraction
# ---------------------------------------------------------------------------

@dataclass
class CommandResult:
    """Unified result from a remote command execution."""
    stdout: str
    stderr: str
    returncode: int
    success: bool

    @property
    def output(self) -> str:
        return self.stdout if self.success else self.stderr


class RemoteExecutor(ABC):
    """Abstract base for executing commands on a remote vLLM host/pod."""

    @abstractmethod
    def run(self, command: str, timeout: int = 60) -> CommandResult:
        """Execute a command on the remote target."""
        ...

    @abstractmethod
    def read_file(self, path: str) -> CommandResult:
        """Read a file from the remote target."""
        ...

    @abstractmethod
    def write_file(self, path: str, content: str) -> CommandResult:
        """Write content to a file on the remote target."""
        ...

    @abstractmethod
    def test_connection(self) -> bool:
        """Test connectivity to the remote target."""
        ...


class SSHExecutor(RemoteExecutor):
    """Execute commands on the vLLM host via SSH.

    Wraps the existing SSHClient for backward compatibility with the
    ai-perf-hackathon SSH-based workflow.
    """

    def __init__(self, ssh_client: SSHClient):
        self._client = ssh_client

    def run(self, command: str, timeout: int = 60) -> CommandResult:
        result = self._client.run(command, timeout=timeout)
        return CommandResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            success=result.success,
        )

    def read_file(self, path: str) -> CommandResult:
        result = self._client.read_file(path)
        return CommandResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            success=result.success,
        )

    def write_file(self, path: str, content: str) -> CommandResult:
        result = self._client.write_file(path, content)
        return CommandResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            success=result.success,
        )

    def test_connection(self) -> bool:
        return self._client.test_connection()


class OcExecutor(RemoteExecutor):
    """Execute commands on a vLLM pod via ``oc exec`` (OpenShift).

    Parameters
    ----------
    namespace : str
        Kubernetes namespace where the vLLM pod runs.
    pod_name : str
        Name (or label selector) of the vLLM pod.
    kubeconfig : str or None
        Path to a kubeconfig file. If *None*, the default kubeconfig
        (``~/.kube/config`` or ``$KUBECONFIG``) is used.
    container : str or None
        Container name inside the pod. If *None*, the default container
        is used (Kubernetes picks the first one).
    """

    def __init__(
        self,
        namespace: str,
        pod_name: str,
        kubeconfig: Optional[str] = None,
        container: Optional[str] = None,
    ):
        self.namespace = namespace
        self.pod_name = pod_name
        self.kubeconfig = kubeconfig
        self.container = container

    def _build_oc_cmd(self, command: str) -> list[str]:
        """Build the full ``oc exec`` command list."""
        cmd = ["oc"]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        cmd += ["exec", "-n", self.namespace]
        if self.container:
            cmd += ["-c", self.container]
        cmd += [self.pod_name, "--"]
        # Use /bin/sh -c to support shell features (pipes, redirects, etc.)
        cmd += ["/bin/sh", "-c", command]
        return cmd

    def run(self, command: str, timeout: int = 60) -> CommandResult:
        oc_cmd = self._build_oc_cmd(command)
        try:
            result = subprocess.run(
                oc_cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 10,
            )
            stdout = result.stdout.replace("\r\n", "\n").replace("\r", "\n")
            stderr = result.stderr.replace("\r\n", "\n").replace("\r", "\n")
            return CommandResult(
                stdout=stdout,
                stderr=stderr,
                returncode=result.returncode,
                success=result.returncode == 0,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                stdout="",
                stderr=f"oc exec timed out after {timeout}s",
                returncode=-1,
                success=False,
            )
        except Exception as e:
            return CommandResult(
                stdout="",
                stderr=str(e),
                returncode=-1,
                success=False,
            )

    def read_file(self, path: str) -> CommandResult:
        return self.run(f"cat {shlex.quote(path)}")

    def write_file(self, path: str, content: str) -> CommandResult:
        return self.run(f"cat > {shlex.quote(path)} << 'EOFAGENT'\n{content}\nEOFAGENT")

    def test_connection(self) -> bool:
        result = self.run("echo ok", timeout=15)
        return result.success and "ok" in result.stdout


# ---------------------------------------------------------------------------
# Benchmark profiles (from config/settings.yaml)
# ---------------------------------------------------------------------------

BENCHMARK_PROFILES = {
    "balanced": {
        "isl": 128,
        "osl": 128,
        "description": "Balanced workload (ISL=128, OSL=128)",
        "data_flag": '{"prompt_tokens": 128, "output_tokens": 128, "samples": 100}',
    },
    "decode_heavy": {
        "isl": 128,
        "osl": 512,
        "description": "Decode-heavy workload (ISL=128, OSL=512)",
        "data_flag": '{"prompt_tokens": 128, "output_tokens": 512, "samples": 100}',
    },
    "prefill_heavy": {
        "isl": 512,
        "osl": 64,
        "description": "Prefill-heavy workload (ISL=512, OSL=64)",
        "data_flag": '{"prompt_tokens": 512, "output_tokens": 64, "samples": 100}',
    },
    "long_context": {
        "isl": 1024,
        "osl": 128,
        "description": "Long-context workload (ISL=1024, OSL=128)",
        "data_flag": '{"prompt_tokens": 1024, "output_tokens": 128, "samples": 100}',
    },
}

DEFAULT_CONCURRENCY_LEVELS = [1, 50]


# ---------------------------------------------------------------------------
# Tool Definitions (for Claude's tool_use API)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "run_command",
        "description": (
            "Run a shell command on the vLLM host/pod to diagnose or fix issues. "
            "Use this to explore the system, check GPU status, inspect vLLM configs, "
            "restart services, apply tunings, etc. The command runs on the remote "
            "target (SSH host or OpenShift pod) via the configured executor."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute on the vLLM host/pod",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Command timeout in seconds. Default is 60. "
                        "Increase for long-running operations (e.g. model reload)."
                    ),
                    "default": 60,
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file on the vLLM host/pod. "
            "Use this to inspect configuration files, logs, profiler output, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file on the vLLM host/pod",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file on the vLLM host/pod. "
            "Use this to modify vLLM configuration, create scripts, "
            "or write profiler setup files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file on the vLLM host/pod",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_benchmark",
        "description": (
            "Run a GuideLLM benchmark against the vLLM endpoint from the LOCAL machine. "
            "This measures throughput (tokens/sec), TTFT, ITL, and TPOT at P50/P95/P99. "
            "Choose a profile to set ISL/OSL and concurrency for the load pattern. "
            "Profiles: balanced (ISL=1000,OSL=1000), decode_heavy (ISL=512,OSL=2048), "
            "prefill_heavy (ISL=2048,OSL=128), long_context (ISL=8000,OSL=1000). "
            "The benchmark takes 2-10 minutes depending on max-seconds. "
            "Results are saved as JSON and a summary is returned."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "profile": {
                    "type": "string",
                    "enum": ["balanced", "decode_heavy", "prefill_heavy", "long_context"],
                    "description": (
                        "Benchmark workload profile. Controls ISL/OSL ratios. "
                        "balanced: equal read/write, decode_heavy: long generation, "
                        "prefill_heavy: long context input, long_context: very large input."
                    ),
                },
                "concurrency": {
                    "type": "string",
                    "description": (
                        "Comma-separated concurrency levels for the sweep "
                        "(e.g. '1,50'). Default: '1,50'"
                    ),
                },
                "endpoint": {
                    "type": "string",
                    "description": (
                        "vLLM endpoint URL. Auto-filled from CLI args if omitted."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": "Model name served by vLLM. Auto-filled from CLI args if omitted.",
                },
                "max_seconds": {
                    "type": "integer",
                    "description": "Maximum seconds per concurrency level. Default: 30",
                    "default": 30,
                },
                "output_path": {
                    "type": "string",
                    "description": (
                        "Path to save the GuideLLM JSON results. "
                        "Default: ./benchmark_results/<profile>_<timestamp>.json"
                    ),
                },
            },
            "required": ["profile"],
        },
    },
    {
        "name": "analyze_trace",
        "description": (
            "Analyze a PyTorch profiler trace JSON file to extract kernel-level "
            "performance statistics. Returns top kernels by GPU time, category "
            "breakdown (attention, GEMM, normalization, etc.), and total GPU time. "
            "Use this after collecting a trace to identify performance bottlenecks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "trace_json_path": {
                    "type": "string",
                    "description": (
                        "Path to the Chrome trace JSON file from PyTorch profiler. "
                        "This is a local file path (on the machine running the agent)."
                    ),
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top kernels to return. Default: 20",
                    "default": 20,
                },
            },
            "required": ["trace_json_path"],
        },
    },
    {
        "name": "map_kernel",
        "description": (
            "Map a CUDA kernel name to its source code location in vLLM or PyTorch. "
            "Returns the source file, function name, description, category, and whether "
            "it is a PyTorch standard library kernel. Use this to understand what a hot "
            "kernel does and where to look for optimization opportunities."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kernel_name": {
                    "type": "string",
                    "description": (
                        "CUDA kernel name from the profiler trace "
                        "(e.g. 'flash_fwd_kernel', 'rms_norm_kernel')"
                    ),
                },
            },
            "required": ["kernel_name"],
        },
    },
    {
        "name": "done",
        "description": (
            "Signal that the tuning session is complete. Call this when you have "
            "achieved the performance target, exhausted viable tuning options, "
            "or want to report final results. Provide a summary of actions taken "
            "and results achieved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "Summary of what was done: tunings applied, benchmarks run, "
                        "performance changes observed, and final recommendations."
                    ),
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the performance target was achieved",
                },
            },
            "required": ["summary", "success"],
        },
    },
]


# ---------------------------------------------------------------------------
# Handler implementations
# ---------------------------------------------------------------------------

def _handle_run_command(
    args: dict,
    executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Execute a shell command on the vLLM host/pod."""
    command = args["command"]
    timeout = args.get("timeout", 60)

    result = executor.run(command, timeout=timeout)

    command_history.append({
        "tool": "run_command",
        "command": command,
        "success": result.success,
        "output": result.output[:2000],
    })

    return ToolResult(
        tool="run_command",
        success=result.success,
        output=result.output,
        error=result.stderr if not result.success else None,
    )


def _handle_read_file(
    args: dict,
    executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Read a file from the vLLM host/pod."""
    path = args["path"]

    result = executor.read_file(path)

    command_history.append({
        "tool": "read_file",
        "path": path,
        "success": result.success,
    })

    return ToolResult(
        tool="read_file",
        success=result.success,
        output=result.output,
        error=result.stderr if not result.success else None,
    )


def _handle_write_file(
    args: dict,
    executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Write content to a file on the vLLM host/pod."""
    path = args["path"]
    content = args["content"]

    result = executor.write_file(path, content)

    command_history.append({
        "tool": "write_file",
        "path": path,
        "success": result.success,
    })

    return ToolResult(
        tool="write_file",
        success=result.success,
        output=result.output,
        error=result.stderr if not result.success else None,
    )


def _handle_run_benchmark(
    args: dict,
    _executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Run a GuideLLM benchmark from the LOCAL machine against the vLLM endpoint.

    This does NOT run on the remote pod/host -- it runs locally using subprocess
    because GuideLLM sends HTTP requests to the vLLM endpoint.
    """
    profile_name = args["profile"]
    endpoint = args["endpoint"]
    model = args["model"]
    max_seconds = args.get("max_seconds", 30)
    concurrency = args.get("concurrency", ",".join(str(c) for c in DEFAULT_CONCURRENCY_LEVELS))

    if profile_name not in BENCHMARK_PROFILES:
        return ToolResult(
            tool="run_benchmark",
            success=False,
            output="",
            error=f"Unknown profile '{profile_name}'. Choose from: {list(BENCHMARK_PROFILES.keys())}",
        )

    profile = BENCHMARK_PROFILES[profile_name]

    # Build output path
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.get("output_path") or f"./benchmark_results/{profile_name}_{timestamp}.json"

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Build GuideLLM command (runs locally, targets the vLLM endpoint)
    # Target URL must end with /v1 for OpenAI-compatible API
    target_url = endpoint.rstrip("/")
    if not target_url.endswith("/v1"):
        target_url += "/v1"

    # Derive HuggingFace processor name from model path
    # e.g. "/models/facebook/opt-125m" -> "facebook/opt-125m"
    processor = args.get("processor", model)
    if processor.startswith("/models/"):
        processor = processor[len("/models/"):]

    guidellm_cmd = [
        sys.executable, "-m", "guidellm", "benchmark", "run",
        f"--target={target_url}",
        f"--model={model}",
        f"--processor={processor}",
        f"--rate-type=concurrent",
        f"--max-seconds={max_seconds}",
        f"--rate={concurrency}",
        f"--output-path={output_path}",
        "--processor-args", '{"trust-remote-code":"true"}',
        "--data", profile["data_flag"],
    ]

    command_str = " ".join(guidellm_cmd)

    command_history.append({
        "tool": "run_benchmark",
        "profile": profile_name,
        "concurrency": concurrency,
        "endpoint": endpoint,
        "model": model,
        "command": command_str,
    })

    try:
        # GuideLLM can take a long time; allow up to 30 minutes
        bench_timeout = max(max_seconds * len(concurrency.split(",")) + 120, 600)
        result = subprocess.run(
            guidellm_cmd,
            capture_output=True,
            text=True,
            timeout=bench_timeout,
        )
        stdout = result.stdout.replace("\r\n", "\n").replace("\r", "\n")
        stderr = result.stderr.replace("\r\n", "\n").replace("\r", "\n")

        if result.returncode == 0:
            # Try to read and summarize the JSON output
            summary_parts = [
                f"Profile: {profile_name} ({profile['description']})",
                f"Concurrency levels: {concurrency}",
                f"Max seconds per level: {max_seconds}",
                f"Results saved to: {output_path}",
                "",
                "--- GuideLLM stdout ---",
                stdout[-4000:] if len(stdout) > 4000 else stdout,
            ]

            # Attempt to parse the output JSON for a quick summary
            if os.path.exists(output_path):
                try:
                    with open(output_path, "r") as f:
                        bench_data = json.load(f)
                    summary_parts.append("")
                    summary_parts.append("--- Results JSON loaded successfully ---")
                    summary_parts.append(f"Keys: {list(bench_data.keys()) if isinstance(bench_data, dict) else 'array'}")
                except (json.JSONDecodeError, OSError):
                    summary_parts.append("(Could not parse results JSON)")

            output_text = "\n".join(summary_parts)
        else:
            output_text = (
                f"GuideLLM exited with code {result.returncode}\n"
                f"stdout:\n{stdout[-2000:]}\n"
                f"stderr:\n{stderr[-2000:]}"
            )

        command_history[-1]["success"] = result.returncode == 0

        return ToolResult(
            tool="run_benchmark",
            success=result.returncode == 0,
            output=output_text,
            error=stderr if result.returncode != 0 else None,
        )

    except subprocess.TimeoutExpired:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="run_benchmark",
            success=False,
            output="",
            error=f"GuideLLM timed out after {bench_timeout}s",
        )
    except FileNotFoundError:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="run_benchmark",
            success=False,
            output="",
            error=(
                "GuideLLM is not installed or not on PATH. "
                "Install with: pip install guidellm"
            ),
        )
    except Exception as e:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="run_benchmark",
            success=False,
            output="",
            error=f"Failed to run GuideLLM: {e}",
        )


def _handle_analyze_trace(
    args: dict,
    _executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Analyze a PyTorch profiler trace JSON file.

    Delegates to agent.analysis.trace_analyzer.  If the module is not yet
    populated (still contains TODOs), we fall back to a basic JSON parse
    so the tool remains functional during incremental development.
    """
    trace_path = args["trace_json_path"]
    top_n = args.get("top_n", 20)

    command_history.append({
        "tool": "analyze_trace",
        "trace_json_path": trace_path,
    })

    try:
        # Try importing the full analysis module first
        from .analysis import trace_analyzer
        if hasattr(trace_analyzer, "analyze_trace"):
            result = trace_analyzer.analyze_trace(trace_path, top_n=top_n)
            command_history[-1]["success"] = True
            return ToolResult(
                tool="analyze_trace",
                success=True,
                output=json.dumps(result, indent=2, default=str),
            )
    except (ImportError, AttributeError):
        pass

    # Fallback: basic trace parsing
    try:
        import os
        if not os.path.exists(trace_path):
            command_history[-1]["success"] = False
            return ToolResult(
                tool="analyze_trace",
                success=False,
                output="",
                error=f"Trace file not found: {trace_path}",
            )

        with open(trace_path, "r") as f:
            trace_data = json.load(f)

        # Basic Chrome trace event parsing
        events = trace_data if isinstance(trace_data, list) else trace_data.get("traceEvents", [])
        gpu_events = [
            e for e in events
            if isinstance(e, dict) and e.get("cat") in ("kernel", "gpu_memcpy", "cuda_runtime")
            and e.get("dur", 0) > 0
        ]

        # Aggregate by kernel name
        kernel_stats: dict[str, dict] = {}
        for e in gpu_events:
            name = e.get("name", "unknown")
            dur_us = e.get("dur", 0)
            if name not in kernel_stats:
                kernel_stats[name] = {"count": 0, "total_dur_us": 0, "min_dur_us": dur_us, "max_dur_us": dur_us}
            stats = kernel_stats[name]
            stats["count"] += 1
            stats["total_dur_us"] += dur_us
            stats["min_dur_us"] = min(stats["min_dur_us"], dur_us)
            stats["max_dur_us"] = max(stats["max_dur_us"], dur_us)

        # Sort by total duration and take top N
        sorted_kernels = sorted(kernel_stats.items(), key=lambda x: x[1]["total_dur_us"], reverse=True)
        top_kernels = [
            {"kernel": name, **stats, "avg_dur_us": round(stats["total_dur_us"] / stats["count"], 2)}
            for name, stats in sorted_kernels[:top_n]
        ]

        total_gpu_time_us = sum(s["total_dur_us"] for s in kernel_stats.values())

        result = {
            "trace_file": trace_path,
            "total_events": len(events),
            "gpu_events": len(gpu_events),
            "unique_kernels": len(kernel_stats),
            "total_gpu_time_us": total_gpu_time_us,
            "total_gpu_time_ms": round(total_gpu_time_us / 1000, 2),
            "top_kernels": top_kernels,
            "note": "Basic fallback parser (trace_analyzer module not yet populated)",
        }

        command_history[-1]["success"] = True
        return ToolResult(
            tool="analyze_trace",
            success=True,
            output=json.dumps(result, indent=2),
        )

    except json.JSONDecodeError as e:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="analyze_trace",
            success=False,
            output="",
            error=f"Failed to parse trace JSON: {e}",
        )
    except Exception as e:
        command_history[-1]["success"] = False
        return ToolResult(
            tool="analyze_trace",
            success=False,
            output="",
            error=f"Failed to analyze trace: {e}",
        )


def _handle_map_kernel(
    args: dict,
    _executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Map a CUDA kernel name to its source code location.

    Delegates to agent.analysis.kernel_mapper.  Falls back to a stub
    response if the module is not yet populated.
    """
    kernel_name = args["kernel_name"]

    command_history.append({
        "tool": "map_kernel",
        "kernel_name": kernel_name,
    })

    try:
        from .analysis import kernel_mapper
        if hasattr(kernel_mapper, "find_kernel_mapping"):
            mapping = kernel_mapper.find_kernel_mapping(kernel_name)
            is_stdlib = False
            if hasattr(kernel_mapper, "is_pytorch_stdlib"):
                is_stdlib = kernel_mapper.is_pytorch_stdlib(kernel_name)
            result = {
                "kernel_name": kernel_name,
                "mapping": mapping,
                "is_pytorch_stdlib": is_stdlib,
            }
            command_history[-1]["success"] = True
            return ToolResult(
                tool="map_kernel",
                success=True,
                output=json.dumps(result, indent=2, default=str),
            )
    except (ImportError, AttributeError):
        pass

    # Fallback: pattern-based heuristic when the full mapper is not available
    result = _fallback_kernel_mapping(kernel_name)
    command_history[-1]["success"] = True
    return ToolResult(
        tool="map_kernel",
        success=True,
        output=json.dumps(result, indent=2),
    )


def _fallback_kernel_mapping(kernel_name: str) -> dict:
    """Simple heuristic kernel mapping when kernel_mapper is not populated."""
    name_lower = kernel_name.lower()

    # Pattern-based classification
    patterns = [
        (["flash_fwd", "flash_bwd", "flash_attn"], "attention", "Flash Attention kernel", "vllm/attention/"),
        (["paged_attention", "paged_attn"], "attention", "PagedAttention kernel", "vllm/attention/"),
        (["fmha"], "attention", "Fused multi-head attention", "vllm/attention/"),
        (["cutlass", "gemm", "cublas", "matmul", "sgemm", "hgemm"], "linear/gemm", "Matrix multiplication kernel", "torch or cutlass"),
        (["rms_norm", "rmsnorm"], "normalization", "RMS normalization kernel", "vllm/model_executor/layers/"),
        (["layer_norm", "layernorm"], "normalization", "Layer normalization kernel", "torch/nn/"),
        (["silu", "gelu", "relu", "swiglu"], "activation", "Activation function kernel", "vllm/model_executor/layers/"),
        (["rotary", "rope"], "positional_encoding", "Rotary positional embedding kernel", "vllm/model_executor/layers/"),
        (["memcpy", "memset"], "memory", "Memory operation", "CUDA runtime"),
        (["nccl", "allreduce", "allgather"], "communication", "Collective communication kernel", "NCCL"),
        (["softmax"], "attention", "Softmax kernel", "torch or custom"),
        (["elementwise", "binary", "unary"], "elementwise", "Elementwise operation", "torch"),
        (["topk", "sampling", "argmax"], "sampling", "Sampling/selection kernel", "vllm/model_executor/layers/sampler"),
        (["quantize", "dequantize", "fp8", "awq", "gptq"], "quantization", "Quantization kernel", "vllm/model_executor/layers/quantization/"),
    ]

    for keywords, category, description, source_hint in patterns:
        if any(kw in name_lower for kw in keywords):
            return {
                "kernel_name": kernel_name,
                "category": category,
                "description": description,
                "source_hint": source_hint,
                "is_pytorch_stdlib": any(kw in name_lower for kw in ["cublas", "memcpy", "memset", "nccl"]),
                "note": "Heuristic mapping (kernel_mapper module not yet populated)",
            }

    return {
        "kernel_name": kernel_name,
        "category": "unknown",
        "description": "Kernel not recognized by heuristic mapper",
        "source_hint": "unknown",
        "is_pytorch_stdlib": False,
        "note": "Heuristic mapping (kernel_mapper module not yet populated)",
    }


def _handle_done(
    args: dict,
    _executor: RemoteExecutor,
    command_history: list[dict],
) -> ToolResult:
    """Signal that the agent has completed its work."""
    summary = args["summary"]
    success = args.get("success", False)

    command_history.append({
        "tool": "done",
        "summary": summary,
        "success": success,
    })

    return ToolResult(
        tool="done",
        success=success,
        output=summary,
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_TOOL_HANDLERS = {
    "run_command": _handle_run_command,
    "read_file": _handle_read_file,
    "write_file": _handle_write_file,
    "run_benchmark": _handle_run_benchmark,
    "analyze_trace": _handle_analyze_trace,
    "map_kernel": _handle_map_kernel,
    "done": _handle_done,
}


def dispatch_tool(
    name: str,
    args: dict,
    executor: RemoteExecutor,
    command_history: Optional[list[dict]] = None,
) -> ToolResult:
    """Route a tool call to the appropriate handler.

    Parameters
    ----------
    name : str
        Tool name (must match one of the keys in TOOL_DEFINITIONS).
    args : dict
        Tool input arguments from the Claude API response.
    executor : RemoteExecutor
        The configured executor (SSHExecutor or OcExecutor) for remote commands.
    command_history : list[dict] or None
        Mutable list to track command history across the agent session.
        If None, a temporary list is used (history is discarded).

    Returns
    -------
    ToolResult
        Result of the tool execution.
    """
    if command_history is None:
        command_history = []

    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return ToolResult(
            tool=name,
            success=False,
            output="",
            error=f"Unknown tool: '{name}'. Available tools: {list(_TOOL_HANDLERS.keys())}",
        )

    return handler(args, executor, command_history)


# ---------------------------------------------------------------------------
# Convenience: AgentTools class (higher-level wrapper, optional)
# ---------------------------------------------------------------------------

class AgentTools:
    """Higher-level wrapper that bundles an executor with tool dispatch.

    Provides a class-based interface similar to the original ai-perf-hackathon
    AgentTools, but backed by the RemoteExecutor abstraction.
    """

    def __init__(
        self,
        executor: RemoteExecutor,
        vllm_endpoint: str = "http://localhost:8000",
        model_name: str = "",
    ):
        self.executor = executor
        self.vllm_endpoint = vllm_endpoint
        self.model_name = model_name
        self.command_history: list[dict] = []

    def get_tool_definitions(self) -> list[dict]:
        """Return the tool definitions list for Claude's API."""
        return TOOL_DEFINITIONS

    def dispatch(self, name: str, args: dict) -> ToolResult:
        """Dispatch a tool call, tracking history on this instance.

        For run_benchmark, always uses the CLI-provided endpoint and model
        (the agent may see different ports/models inside the pod, but the
        benchmark runs locally against the port-forwarded endpoint).
        """
        if name == "run_benchmark":
            args["endpoint"] = self.vllm_endpoint
            args["model"] = self.model_name
        return dispatch_tool(name, args, self.executor, self.command_history)

    # Convenience methods for direct (non-agent) use

    def run_command(self, command: str, timeout: int = 60) -> ToolResult:
        return self.dispatch("run_command", {"command": command, "timeout": timeout})

    def read_file(self, path: str) -> ToolResult:
        return self.dispatch("read_file", {"path": path})

    def write_file(self, path: str, content: str) -> ToolResult:
        return self.dispatch("write_file", {"path": path, "content": content})

    def run_benchmark(
        self,
        profile: str,
        endpoint: str,
        model: str,
        concurrency: Optional[str] = None,
        max_seconds: int = 120,
        output_path: Optional[str] = None,
    ) -> ToolResult:
        args: dict = {
            "profile": profile,
            "endpoint": endpoint,
            "model": model,
            "max_seconds": max_seconds,
        }
        if concurrency:
            args["concurrency"] = concurrency
        if output_path:
            args["output_path"] = output_path
        return self.dispatch("run_benchmark", args)

    def analyze_trace(self, trace_json_path: str, top_n: int = 20) -> ToolResult:
        return self.dispatch("analyze_trace", {"trace_json_path": trace_json_path, "top_n": top_n})

    def map_kernel(self, kernel_name: str) -> ToolResult:
        return self.dispatch("map_kernel", {"kernel_name": kernel_name})

    def done(self, summary: str, success: bool = False) -> ToolResult:
        return self.dispatch("done", {"summary": summary, "success": success})
