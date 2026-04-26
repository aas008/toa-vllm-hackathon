"""
Log Anomaly Detectors for vLLM and GuideLLM benchmark logs.

Two analyzers:
1. VllmStartupAnalyzer — checks startup logs (before server ready) for
   config mismatches against intended args and actual errors.
2. BenchmarkLogAnalyzer — checks GuideLLM benchmark output for anomalies
   like connection failures, timeouts, zero throughput, error rates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Anomaly:
    severity: str  # "error", "warning", "info"
    category: str  # "config_mismatch", "startup_error", "connection", "performance"
    message: str
    detail: str = ""


# ── vLLM Startup Log Analyzer ───────────────────────────────────────────

# Markers that indicate server is ready (stop analyzing after these)
_READY_MARKERS = [
    "Application startup complete",
    "Uvicorn running on",
    "Started server process",
    "vLLM API server started",
    "serving",
]

# Known error patterns in startup logs
_ERROR_PATTERNS = [
    (r"(?i)(OOM|out of memory|CUDA out of memory)", "oom", "GPU out of memory during startup"),
    (r"(?i)(CUDA error|CUDA driver|CUDA runtime)", "cuda_error", "CUDA error during startup"),
    (r"(?i)(RuntimeError|AssertionError|ValueError|TypeError):\s*(.+)", "runtime_error", "Python error"),
    (r"(?i)(failed to load|cannot load|model.*not found)", "model_load", "Model loading failure"),
    (r"(?i)(killed|segfault|signal \d+|core dumped)", "crash", "Process crash"),
    (r"(?i)(connection refused|cannot connect|unreachable)", "network", "Network connectivity issue"),
    (r"(?i)(permission denied|access denied)", "permission", "Permission error"),
    (r"(?i)(torch\.compile.*error|compilation failed|inductor error)", "compile_error", "Compilation failure"),
    (r"(?i)(WARNING.*deprecated|WARNING.*removed)", "deprecation", "Deprecated feature warning"),
]

# Config keys we can extract from startup logs and compare
_CONFIG_EXTRACTORS = {
    "model": re.compile(r"(?:--model|model=)\s*['\"]?([^\s'\"]+)"),
    "dtype": re.compile(r"(?:dtype[=:]\s*|--dtype\s+)(\S+)"),
    "tensor_parallel_size": re.compile(r"tensor.parallel.size[=:\s]+(\d+)", re.IGNORECASE),
    "pipeline_parallel_size": re.compile(r"pipeline.parallel.size[=:\s]+(\d+)", re.IGNORECASE),
    "max_num_seqs": re.compile(r"max.num.seqs[=:\s]+(\d+)", re.IGNORECASE),
    "max_num_batched_tokens": re.compile(r"max.num.batched.tokens[=:\s]+(\d+)", re.IGNORECASE),
    "gpu_memory_utilization": re.compile(r"gpu.memory.utilization[=:\s]+([\d.]+)", re.IGNORECASE),
    "max_model_len": re.compile(r"max.model.len[=:\s]+(\d+)", re.IGNORECASE),
    "quantization": re.compile(r"quantization[=:\s]+(\S+)", re.IGNORECASE),
    "kv_cache_dtype": re.compile(r"kv.cache.dtype[=:\s]+(\S+)", re.IGNORECASE),
    "enforce_eager": re.compile(r"enforce.eager[=:\s]+(True|False)", re.IGNORECASE),
    "enable_chunked_prefill": re.compile(r"chunked.prefill[=:\s]+(True|enabled)", re.IGNORECASE),
    "enable_prefix_caching": re.compile(r"prefix.cach(?:ing|e)[=:\s]+(True|enabled)", re.IGNORECASE),
    "attention_backend": re.compile(r"(?:attention|attn).backend[=:\s]+(\S+)", re.IGNORECASE),
}


def analyze_vllm_startup(
    raw_logs: str,
    intended_args: Optional[list[str]] = None,
) -> list[Anomaly]:
    """Analyze vLLM startup logs for errors and config mismatches.

    Args:
        raw_logs: Full vLLM pod logs (will only analyze up to server-ready marker).
        intended_args: CLI args the pod was launched with (e.g. ["--max-num-seqs", "512"]).

    Returns:
        List of Anomaly objects sorted by severity.
    """
    anomalies: list[Anomaly] = []

    # Extract startup section (before server ready)
    startup_lines = []
    for line in raw_logs.splitlines():
        startup_lines.append(line)
        if any(marker.lower() in line.lower() for marker in _READY_MARKERS):
            break

    startup_text = "\n".join(startup_lines)

    # Check if server ever became ready
    server_ready = any(
        marker.lower() in raw_logs.lower() for marker in _READY_MARKERS
    )
    if not server_ready:
        anomalies.append(Anomaly(
            severity="error",
            category="startup_error",
            message="Server never reached ready state",
            detail="No server-ready marker found in logs. Server may have crashed during startup.",
        ))

    # Check for error patterns
    for pattern, category, description in _ERROR_PATTERNS:
        for match in re.finditer(pattern, startup_text):
            anomalies.append(Anomaly(
                severity="error" if category not in ("deprecation",) else "warning",
                category=category,
                message=description,
                detail=match.group(0).strip()[:200],
            ))

    # Extract actual config from logs and compare with intended
    if intended_args:
        actual_config = _extract_config_from_logs(startup_text)
        intended_config = _parse_intended_args(intended_args)
        mismatches = _compare_configs(intended_config, actual_config)
        anomalies.extend(mismatches)

    # Check for specific vLLM startup warnings
    if "Chunked prefill is enabled" in startup_text:
        m = re.search(r"max_num_batched_tokens=(\d+)", startup_text)
        if m:
            tokens = int(m.group(1))
            if tokens < 2048:
                anomalies.append(Anomaly(
                    severity="info",
                    category="config_mismatch",
                    message=f"Low max_num_batched_tokens ({tokens})",
                    detail="Chunked prefill enabled with small token budget. May limit throughput.",
                ))

    if re.search(r"(?i)falling back|fallback|using.*instead", startup_text):
        for match in re.finditer(r"(?i)(falling back|fallback|using.*instead).*", startup_text):
            anomalies.append(Anomaly(
                severity="warning",
                category="config_mismatch",
                message="Feature fallback detected",
                detail=match.group(0).strip()[:200],
            ))

    # Sort: errors first, then warnings, then info
    severity_order = {"error": 0, "warning": 1, "info": 2}
    anomalies.sort(key=lambda a: severity_order.get(a.severity, 3))

    return anomalies


def _extract_config_from_logs(text: str) -> dict[str, str]:
    """Extract configuration values from startup log text."""
    config = {}
    for key, pattern in _CONFIG_EXTRACTORS.items():
        m = pattern.search(text)
        if m:
            config[key] = m.group(1)
    return config


def _parse_intended_args(args: list[str]) -> dict[str, str]:
    """Parse CLI args like ["--max-num-seqs", "512"] into dict."""
    config = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-").replace("-", "_")
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                config[key] = args[i + 1]
                i += 2
            else:
                config[key] = "true"
                i += 1
        else:
            i += 1
    return config


def _compare_configs(intended: dict, actual: dict) -> list[Anomaly]:
    """Compare intended vs actual config, return mismatches."""
    anomalies = []
    for key, intended_val in intended.items():
        actual_val = actual.get(key)
        if actual_val is None:
            continue
        # Normalize for comparison
        iv = str(intended_val).lower().strip()
        av = str(actual_val).lower().strip()
        if iv != av:
            anomalies.append(Anomaly(
                severity="warning",
                category="config_mismatch",
                message=f"Config mismatch: {key}",
                detail=f"Intended: {intended_val}, Actual in logs: {actual_val}",
            ))
    return anomalies


# ── Benchmark Log Analyzer ───────────────────────────────────────────────

def analyze_benchmark_logs(
    benchmark_output: str,
    benchmark_json: Optional[dict] = None,
) -> list[Anomaly]:
    """Analyze GuideLLM benchmark output for anomalies.

    Args:
        benchmark_output: Raw text output from run_benchmark tool.
        benchmark_json: Parsed GuideLLM JSON result (optional, for deeper analysis).

    Returns:
        List of Anomaly objects.
    """
    anomalies: list[Anomaly] = []

    # Check for zero successful requests
    zero_match = re.search(r"(\d+)\s+successful,\s+(\d+)\s+errored,\s+(\d+)\s+total", benchmark_output)
    if zero_match:
        successful = int(zero_match.group(1))
        errored = int(zero_match.group(2))
        total = int(zero_match.group(3))

        if successful == 0 and total > 0:
            anomalies.append(Anomaly(
                severity="error",
                category="connection",
                message=f"All {total} requests failed (0 successful)",
                detail="vLLM pod was likely unreachable, crashed, or model not loaded.",
            ))
        elif errored > 0:
            error_rate = errored / total * 100 if total > 0 else 0
            if error_rate > 10:
                anomalies.append(Anomaly(
                    severity="error" if error_rate > 50 else "warning",
                    category="connection",
                    message=f"High error rate: {error_rate:.1f}% ({errored}/{total} requests failed)",
                    detail="Check vLLM pod health, memory pressure, or timeout settings.",
                ))

    # Check for connection errors in output
    conn_errors = re.findall(
        r"(?i)(ConnectError|ConnectionRefused|ConnectionReset|timeout|ECONNREFUSED|All connection attempts failed)",
        benchmark_output,
    )
    if conn_errors:
        anomalies.append(Anomaly(
            severity="error",
            category="connection",
            message=f"Connection errors detected ({len(conn_errors)} occurrences)",
            detail=f"Errors: {', '.join(set(conn_errors))}",
        ))

    # Check for GuideLLM-specific errors
    if "GuideLLM exited with code" in benchmark_output:
        m = re.search(r"GuideLLM exited with code (\d+)", benchmark_output)
        code = m.group(1) if m else "unknown"
        anomalies.append(Anomaly(
            severity="error",
            category="benchmark_error",
            message=f"GuideLLM exited with non-zero code: {code}",
            detail="Benchmark tool crashed. Check stderr for details.",
        ))

    if "timed out" in benchmark_output.lower():
        anomalies.append(Anomaly(
            severity="error",
            category="benchmark_error",
            message="Benchmark timed out",
            detail="GuideLLM exceeded timeout. Increase --max-seconds or check vLLM responsiveness.",
        ))

    # Analyze from JSON if available
    if benchmark_json:
        anomalies.extend(_analyze_benchmark_json(benchmark_json))

    # Check Prometheus delta in output for server-side issues
    if "PROMETHEUS METRICS DELTA" in benchmark_output:
        anomalies.extend(_analyze_prometheus_in_benchmark(benchmark_output))

    severity_order = {"error": 0, "warning": 1, "info": 2}
    anomalies.sort(key=lambda a: severity_order.get(a.severity, 3))
    return anomalies


def _analyze_benchmark_json(data: dict) -> list[Anomaly]:
    """Analyze parsed GuideLLM JSON for performance anomalies."""
    anomalies = []

    for bench in data.get("benchmarks", []):
        metrics = bench.get("metrics", {})
        config = bench.get("config", {})
        strategy = config.get("strategy", {})
        conc = strategy.get("max_concurrency", strategy.get("worker_count", "?"))

        totals = metrics.get("request_totals", {})
        successful = totals.get("successful", 0)
        errored = totals.get("errored", 0)
        total = totals.get("total", 0)

        if total > 0 and successful == 0:
            anomalies.append(Anomaly(
                severity="error",
                category="connection",
                message=f"Concurrency {conc}: 0/{total} successful requests",
                detail="All requests failed at this concurrency level.",
            ))

        # Check for very high TTFT
        ttft = metrics.get("time_to_first_token_ms", {})
        ttft_data = ttft.get("successful", ttft)
        if isinstance(ttft_data, dict):
            p99 = ttft_data.get("percentiles", {}).get("p99")
            if p99 and p99 > 5000:
                anomalies.append(Anomaly(
                    severity="warning",
                    category="performance",
                    message=f"Concurrency {conc}: very high TTFT p99 ({p99:.0f}ms)",
                    detail="TTFT > 5s indicates severe prefill congestion or scheduling delay.",
                ))

        # Check for zero throughput
        output_tps = metrics.get("output_tokens_per_second", {})
        tps_data = output_tps.get("successful", output_tps)
        if isinstance(tps_data, dict):
            mean_tps = tps_data.get("mean", 0)
            if successful > 0 and mean_tps == 0:
                anomalies.append(Anomaly(
                    severity="warning",
                    category="performance",
                    message=f"Concurrency {conc}: zero output throughput despite {successful} successful requests",
                    detail="Throughput measured as 0 tok/sec. May indicate measurement issue.",
                ))

    return anomalies


def _analyze_prometheus_in_benchmark(output: str) -> list[Anomaly]:
    """Extract anomalies from Prometheus delta embedded in benchmark output."""
    anomalies = []

    # Check for preemptions
    m = re.search(r"PREEMPTIONS:\s+(\d+)", output)
    if m and int(m.group(1)) > 0:
        count = int(m.group(1))
        anomalies.append(Anomaly(
            severity="warning",
            category="performance",
            message=f"{count} preemptions during benchmark",
            detail="Scheduler evicted sequences. Reduce concurrency or increase KV cache.",
        ))

    # Check KV cache pressure
    m = re.search(r"KV CACHE PRESSURE:\s+([\d.]+%)", output)
    if m:
        anomalies.append(Anomaly(
            severity="warning",
            category="performance",
            message=f"KV cache pressure: {m.group(1)}",
            detail="KV cache nearly full. Risk of preemptions under higher load.",
        ))

    # Check queuing
    m = re.search(r"QUEUING:\s+(\d+)\s+requests", output)
    if m and int(m.group(1)) > 10:
        anomalies.append(Anomaly(
            severity="warning",
            category="performance",
            message=f"{m.group(1)} requests queuing at end of benchmark",
            detail="Requests accumulating faster than served. Throughput bottleneck.",
        ))

    return anomalies


# ── Formatting ───────────────────────────────────────────────────────────

def format_anomalies(anomalies: list[Anomaly], title: str = "Log Analysis") -> str:
    """Format anomaly list for agent consumption."""
    if not anomalies:
        return f"=== {title} ===\nNo anomalies detected."

    lines = [f"=== {title}: {len(anomalies)} anomalies ==="]

    errors = [a for a in anomalies if a.severity == "error"]
    warnings = [a for a in anomalies if a.severity == "warning"]
    infos = [a for a in anomalies if a.severity == "info"]

    if errors:
        lines.append(f"\nERRORS ({len(errors)}):")
        for a in errors:
            lines.append(f"  [{a.category}] {a.message}")
            if a.detail:
                lines.append(f"    → {a.detail}")

    if warnings:
        lines.append(f"\nWARNINGS ({len(warnings)}):")
        for a in warnings:
            lines.append(f"  [{a.category}] {a.message}")
            if a.detail:
                lines.append(f"    → {a.detail}")

    if infos:
        lines.append(f"\nINFO ({len(infos)}):")
        for a in infos:
            lines.append(f"  [{a.category}] {a.message}")
            if a.detail:
                lines.append(f"    → {a.detail}")

    return "\n".join(lines)
