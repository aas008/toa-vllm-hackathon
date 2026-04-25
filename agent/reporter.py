"""
Markdown & JSON Report Generation for vLLM Performance Tuning

Generates the final performance tuning report with before/after comparisons,
kernel analysis, cost efficiency, and token usage.

SOURCE: ai-perf-hackathon/agent/reporter.py (adapted for vLLM metrics)
"""
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class BenchmarkSnapshot:
    """Benchmark results for a single profile at a single concurrency."""
    profile: str
    concurrency: int
    throughput_tok_per_sec: float = 0.0
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    itl_p50: float = 0.0
    itl_p95: float = 0.0
    itl_p99: float = 0.0
    tpot_p50: float = 0.0
    tpot_p95: float = 0.0
    tpot_p99: float = 0.0
    request_success_rate: float = 100.0

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "concurrency": self.concurrency,
            "throughput_tok_per_sec": self.throughput_tok_per_sec,
            "ttft_p50": self.ttft_p50,
            "ttft_p95": self.ttft_p95,
            "ttft_p99": self.ttft_p99,
            "itl_p50": self.itl_p50,
            "itl_p95": self.itl_p95,
            "itl_p99": self.itl_p99,
            "tpot_p50": self.tpot_p50,
            "tpot_p95": self.tpot_p95,
            "tpot_p99": self.tpot_p99,
            "request_success_rate": self.request_success_rate,
        }


@dataclass
class TuningReport:
    """Complete vLLM performance tuning report."""
    timestamp: str = ""
    model_name: str = ""
    vllm_endpoint: str = ""
    gpu_info: str = ""
    vllm_config: dict = field(default_factory=dict)
    baseline_results: list = field(default_factory=list)
    final_results: list = field(default_factory=list)
    kernel_analysis: dict = field(default_factory=dict)
    actions_taken: list = field(default_factory=list)
    bottlenecks: list = field(default_factory=list)
    agent_summary: str = ""
    token_usage: list = field(default_factory=list)
    decision_log: list = field(default_factory=list)
    prometheus_metrics: list = field(default_factory=list)


class Reporter:
    """Generates vLLM performance tuning reports."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir

    def generate(self, report: TuningReport) -> tuple:
        """Generate both markdown and JSON reports. Returns (md_path, json_path)."""
        os.makedirs(self.output_dir, exist_ok=True)
        ts = report.timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        md_path = os.path.join(self.output_dir, f"report_{ts}.md")
        json_path = os.path.join(self.output_dir, f"report_{ts}.json")

        md_content = self._generate_markdown(report)
        json_content = self._generate_json(report)

        with open(md_path, "w") as f:
            f.write(md_content)
        with open(json_path, "w") as f:
            f.write(json_content)

        return md_path, json_path

    def _generate_markdown(self, report: TuningReport) -> str:
        """Generate a markdown report."""
        sections = [
            self._header_section(report),
            self._executive_summary_section(report),
            self._performance_results_section(report),
            self._bottlenecks_section(report),
            self._kernel_analysis_section(report),
            self._prometheus_metrics_section(report),
            self._tuning_applied_section(report),
            self._cost_analysis_section(report),
            self._token_usage_section(report),
            self._system_config_section(report),
            self._footer_section(),
        ]
        return "\n".join(sections)

    def _header_section(self, report: TuningReport) -> str:
        return "\n".join([
            "# vLLM Performance Tuning Report",
            "",
            f"**Generated**: {report.timestamp}",
            f"**Model**: {report.model_name}",
            f"**Endpoint**: {report.vllm_endpoint}",
            f"**GPU**: {report.gpu_info}",
            "",
            "---",
            "",
        ])

    def _executive_summary_section(self, report: TuningReport) -> str:
        return "\n".join([
            "## Executive Summary",
            "",
            report.agent_summary or "*No summary provided by agent.*",
            "",
            "---",
            "",
        ])

    def _performance_results_section(self, report: TuningReport) -> str:
        lines = [
            "## Performance Results",
            "",
            "### Before vs After Comparison",
            "",
            "| Profile | Metric | Before | After | Change |",
            "|---------|--------|--------|-------|--------|",
        ]

        # Group by profile
        baseline_map = {}
        for r in report.baseline_results:
            key = r.get("profile", "") if isinstance(r, dict) else r.profile
            baseline_map[key] = r

        final_map = {}
        for r in report.final_results:
            key = r.get("profile", "") if isinstance(r, dict) else r.profile
            final_map[key] = r

        for profile in ["balanced", "decode_heavy", "prefill_heavy", "long_context"]:
            base = baseline_map.get(profile)
            final = final_map.get(profile)
            if not base and not final:
                continue

            def _val(obj, key):
                if obj is None:
                    return 0.0
                if isinstance(obj, dict):
                    return obj.get(key, 0.0)
                return getattr(obj, key, 0.0)

            metrics = [
                ("Throughput (tok/s)", "throughput_tok_per_sec", True),
                ("TTFT P50 (ms)", "ttft_p50", False),
                ("TTFT P99 (ms)", "ttft_p99", False),
                ("ITL P50 (ms)", "itl_p50", False),
                ("ITL P99 (ms)", "itl_p99", False),
            ]

            for metric_name, key, higher_is_better in metrics:
                before = _val(base, key)
                after = _val(final, key)
                if before > 0:
                    pct = ((after - before) / before) * 100
                    sign = "+" if pct >= 0 else ""
                    # For latency metrics, negative change is good
                    if not higher_is_better:
                        good = pct <= 0
                    else:
                        good = pct >= 0
                    indicator = "**" if good else ""
                    change = f"{indicator}{sign}{pct:.1f}%{indicator}"
                else:
                    change = "N/A"
                lines.append(
                    f"| {profile} | {metric_name} | {before:.1f} | {after:.1f} | {change} |"
                )

        lines.extend(["", "---", ""])
        return "\n".join(lines)

    def _bottlenecks_section(self, report: TuningReport) -> str:
        lines = ["## Bottlenecks Identified", ""]
        if report.bottlenecks:
            for i, b in enumerate(report.bottlenecks, 1):
                lines.append(f"{i}. {b}")
        else:
            lines.append("*No specific bottlenecks identified.*")
        lines.extend(["", "---", ""])
        return "\n".join(lines)

    def _kernel_analysis_section(self, report: TuningReport) -> str:
        lines = ["## Kernel Analysis (PyTorch Profiler)", ""]
        ka = report.kernel_analysis
        if not ka:
            lines.append("*Profiling was not performed in this run.*")
            lines.extend(["", "---", ""])
            return "\n".join(lines)

        # Top kernels
        top_kernels = ka.get("top_kernels", [])
        if top_kernels:
            lines.extend([
                "### Top CUDA Kernels by Time",
                "",
                "| Rank | Kernel | Category | GPU Time (ms) | % Total |",
                "|------|--------|----------|---------------|---------|",
            ])
            for i, k in enumerate(top_kernels[:15], 1):
                name = k.get("name", "unknown")[:50]
                cat = k.get("category", "")
                time_ms = k.get("gpu_time_ms", 0)
                pct = k.get("pct_total", 0)
                lines.append(f"| {i} | `{name}` | {cat} | {time_ms:.2f} | {pct:.1f}% |")
            lines.append("")

        # Category breakdown
        categories = ka.get("category_breakdown", {})
        if categories:
            lines.extend([
                "### GPU Time by Category",
                "",
                "| Category | GPU Time (ms) | % Total |",
                "|----------|---------------|---------|",
            ])
            for cat, data in categories.items():
                time_ms = data.get("gpu_time_ms", 0)
                pct = data.get("pct_total", 0)
                lines.append(f"| {cat} | {time_ms:.2f} | {pct:.1f}% |")
            lines.append("")

        lines.extend(["---", ""])
        return "\n".join(lines)

    def _prometheus_metrics_section(self, report: TuningReport) -> str:
        lines = ["## Prometheus Metrics (vLLM /metrics)", ""]
        if not report.prometheus_metrics:
            lines.append("*No Prometheus metrics were collected in this run.*")
            lines.extend(["", "---", ""])
            return "\n".join(lines)

        for i, delta in enumerate(report.prometheus_metrics, 1):
            endpoint = delta.get("endpoint", "unknown")
            duration = delta.get("duration_seconds", 0)
            lines.append(f"### Scrape {i} ({endpoint}, {duration:.1f}s interval)")
            lines.append("")

            gauges = delta.get("gauge_snapshots", {})
            if gauges:
                lines.extend([
                    "#### Server State (Gauges)",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                ])
                for name, val in sorted(gauges.items()):
                    short = name.replace("vllm:", "").replace("vllm_", "")
                    if "perc" in name:
                        lines.append(f"| {short} | {val:.1%} |")
                    else:
                        lines.append(f"| {short} | {val:.1f} |")
                lines.append("")

            counter_deltas = delta.get("counter_deltas", {})
            counter_rates = delta.get("counter_rates", {})
            if counter_deltas:
                lines.extend([
                    "#### Throughput Counters (Delta During Benchmark)",
                    "",
                    "| Metric | Delta | Rate (/sec) |",
                    "|--------|-------|-------------|",
                ])
                for name in sorted(counter_deltas.keys()):
                    short = name.replace("vllm:", "").replace("vllm_", "")
                    d = counter_deltas[name]
                    r = counter_rates.get(name, 0)
                    lines.append(f"| {short} | {d:,.0f} | {r:,.2f} |")
                lines.append("")

            hist_summaries = delta.get("histogram_summaries", {})
            if hist_summaries:
                lines.extend([
                    "#### Latency Distributions (Histograms)",
                    "",
                    "| Metric | P50 | P95 | P99 | Mean |",
                    "|--------|-----|-----|-----|------|",
                ])
                for name, summary in sorted(hist_summaries.items()):
                    short = name.replace("vllm:", "").replace("vllm_", "")
                    p50 = summary.get("p50")
                    p95 = summary.get("p95")
                    p99 = summary.get("p99")
                    mean = summary.get("mean")
                    if "seconds" in name or "time" in name:
                        def fmt(v):
                            return f"{v * 1000:.2f}ms" if isinstance(v, (int, float)) else "N/A"
                    elif "tokens" in name:
                        def fmt(v):
                            return f"{v:.0f}" if isinstance(v, (int, float)) else "N/A"
                    else:
                        def fmt(v):
                            return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"
                    lines.append(f"| {short} | {fmt(p50)} | {fmt(p95)} | {fmt(p99)} | {fmt(mean)} |")
                lines.append("")

            lines.extend(["---", ""])

        return "\n".join(lines)

    def _tuning_applied_section(self, report: TuningReport) -> str:
        lines = ["## Tuning Applied", ""]
        if report.actions_taken:
            lines.extend([
                "| # | Action | Details | Timestamp |",
                "|---|--------|---------|-----------|",
            ])
            for i, action in enumerate(report.actions_taken, 1):
                atype = action.get("type", "unknown")
                path = action.get("path", action.get("command", ""))[:50]
                ts = action.get("timestamp", "")
                lines.append(f"| {i} | {atype} | `{path}` | {ts} |")
        else:
            lines.append("*No tuning actions were applied.*")
        lines.extend(["", "---", ""])
        return "\n".join(lines)

    def _cost_analysis_section(self, report: TuningReport) -> str:
        lines = ["## Cost Analysis", ""]

        # Calculate CPMT if we have throughput data
        final_map = {}
        for r in report.final_results:
            key = r.get("profile", "") if isinstance(r, dict) else r.profile
            final_map[key] = r

        if final_map:
            lines.extend([
                "### Cost Per Million Tokens (CPMT)",
                "",
                "| Profile | Throughput (tok/s) | CPMT ($/M tokens) |",
                "|---------|--------------------|--------------------|",
            ])
            # H100 pricing ~$2.50/hr (approximate cloud pricing)
            hourly_cost = 2.50
            for profile, result in final_map.items():
                if isinstance(result, dict):
                    throughput = result.get("throughput_tok_per_sec", 0)
                else:
                    throughput = getattr(result, "throughput_tok_per_sec", 0)
                if throughput > 0:
                    tokens_per_hour = throughput * 3600
                    cpmt = (hourly_cost / tokens_per_hour) * 1_000_000
                    lines.append(f"| {profile} | {throughput:.0f} | ${cpmt:.4f} |")
        else:
            lines.append("*No cost data available.*")

        lines.extend(["", "---", ""])
        return "\n".join(lines)

    def _token_usage_section(self, report: TuningReport) -> str:
        lines = [
            "## Claude API Token Usage",
            "",
            "| Model | Input Tokens | Output Tokens | Total | API Calls | Est. Cost |",
            "|-------|--------------|---------------|-------|-----------|-----------|",
        ]

        total_input = 0
        total_output = 0
        total_calls = 0
        total_cost = 0

        for usage in report.token_usage:
            inp = usage.get("input_tokens", 0)
            out = usage.get("output_tokens", 0)
            total = inp + out
            calls = usage.get("api_calls", 0)
            cost = usage.get("cost_usd", 0)
            model = usage.get("model", "unknown")
            lines.append(
                f"| {model} | {inp:,} | {out:,} | {total:,} | {calls} | ${cost:.4f} |"
            )
            total_input += inp
            total_output += out
            total_calls += calls
            total_cost += cost

        lines.append(
            f"| **Total** | **{total_input:,}** | **{total_output:,}** | "
            f"**{total_input + total_output:,}** | **{total_calls}** | **${total_cost:.4f}** |"
        )

        lines.extend(["", "---", ""])
        return "\n".join(lines)

    def _system_config_section(self, report: TuningReport) -> str:
        lines = [
            "## System Configuration",
            "",
            f"- **Model**: {report.model_name}",
            f"- **GPU**: {report.gpu_info}",
            f"- **Endpoint**: {report.vllm_endpoint}",
            "",
        ]

        if report.vllm_config:
            lines.append("### vLLM Parameters")
            lines.append("")
            for key, value in report.vllm_config.items():
                lines.append(f"- `{key}`: {value}")
            lines.append("")

        lines.extend(["---", ""])
        return "\n".join(lines)

    def _footer_section(self) -> str:
        return "\n".join([
            "",
            "*Report generated by vLLM Performance Tuning Agent v1.0*",
        ])

    def _generate_json(self, report: TuningReport) -> str:
        """Generate a JSON report."""
        data = {
            "timestamp": report.timestamp,
            "model_name": report.model_name,
            "vllm_endpoint": report.vllm_endpoint,
            "gpu_info": report.gpu_info,
            "vllm_config": report.vllm_config,
            "agent_summary": report.agent_summary,
            "bottlenecks": report.bottlenecks,
            "baseline_results": [
                r.to_dict() if hasattr(r, "to_dict") else r
                for r in report.baseline_results
            ],
            "final_results": [
                r.to_dict() if hasattr(r, "to_dict") else r
                for r in report.final_results
            ],
            "kernel_analysis": report.kernel_analysis,
            "actions_taken": report.actions_taken,
            "token_usage": report.token_usage,
            "decision_log": report.decision_log,
            "prometheus_metrics": report.prometheus_metrics,
        }
        return json.dumps(data, indent=2)
