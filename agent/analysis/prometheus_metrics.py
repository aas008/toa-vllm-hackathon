"""
Prometheus Metrics Scraper & Parser for vLLM

Scrapes the /metrics endpoint exposed by vLLM, parses Prometheus text
exposition format, computes deltas between snapshots for per-run analysis.

Zero external dependencies — uses urllib for HTTP and regex for parsing.
"""
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ── Known vLLM metric classifications ────────────────────────────────────
# Both vllm: and vllm_ prefixes handled (varies by vLLM version)

VLLM_GAUGES = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:cache_config_info",
    "vllm:engine_sleep_state",
}

VLLM_COUNTERS = {
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:prompt_tokens_recomputed_total",
}

VLLM_HISTOGRAMS = {
    "vllm:time_to_first_token_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_prompt_tokens",
    "vllm:request_generation_tokens",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:iteration_tokens_total",
}

# Regex for parsing metric lines
_METRIC_LINE_RE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(?:\{(?P<labels>[^}]*)\})?\s+'
    r'(?P<value>[^\s]+)'
)
_TYPE_RE = re.compile(r'^#\s+TYPE\s+(\S+)\s+(\S+)')
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class HistogramData:
    """Parsed Prometheus histogram with bucket boundaries and counts."""
    bucket_counts: Dict[float, float] = field(default_factory=dict)
    sum_value: float = 0.0
    count: float = 0.0

    def percentile(self, p: float) -> Optional[float]:
        """Estimate percentile from histogram buckets via linear interpolation."""
        if self.count == 0:
            return None
        target = p * self.count
        boundaries = sorted(self.bucket_counts.keys())
        if not boundaries:
            return None

        prev_count = 0.0
        prev_bound = 0.0
        for bound in boundaries:
            curr_count = self.bucket_counts[bound]
            if curr_count >= target:
                if curr_count == prev_count:
                    return bound
                fraction = (target - prev_count) / (curr_count - prev_count)
                return prev_bound + fraction * (bound - prev_bound)
            prev_count = curr_count
            prev_bound = bound

        return boundaries[-1] if boundaries[-1] != float('inf') else prev_bound

    @property
    def mean(self) -> Optional[float]:
        if self.count == 0:
            return None
        return self.sum_value / self.count

    def summary_dict(self) -> dict:
        return {
            "count": self.count,
            "sum": self.sum_value,
            "mean": self.mean,
            "p50": self.percentile(0.50),
            "p95": self.percentile(0.95),
            "p99": self.percentile(0.99),
        }

    def to_dict(self) -> dict:
        return {
            "bucket_counts": {str(k): v for k, v in sorted(self.bucket_counts.items())},
            "sum": self.sum_value,
            "count": self.count,
        }


@dataclass
class MetricsSnapshot:
    """Point-in-time snapshot of all vLLM Prometheus metrics."""
    timestamp: str
    endpoint: str
    gauges: Dict[str, float] = field(default_factory=dict)
    counters: Dict[str, float] = field(default_factory=dict)
    histograms: Dict[str, HistogramData] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "endpoint": self.endpoint,
            "gauges": dict(self.gauges),
            "counters": dict(self.counters),
            "histograms": {k: v.to_dict() for k, v in self.histograms.items()},
        }


@dataclass
class MetricsDelta:
    """Computed delta between two MetricsSnapshots."""
    start_time: str
    end_time: str
    endpoint: str
    duration_seconds: float
    gauge_snapshots: Dict[str, float] = field(default_factory=dict)
    counter_deltas: Dict[str, float] = field(default_factory=dict)
    counter_rates: Dict[str, float] = field(default_factory=dict)
    histogram_summaries: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        vllm_only = lambda d: {k: v for k, v in d.items() if k.startswith("vllm:")}
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "endpoint": self.endpoint,
            "duration_seconds": self.duration_seconds,
            "gauge_snapshots": vllm_only(self.gauge_snapshots),
            "counter_deltas": vllm_only(self.counter_deltas),
            "counter_rates": vllm_only(self.counter_rates),
            "histogram_summaries": vllm_only(self.histogram_summaries),
        }

    def format_summary(self) -> str:
        """Verbose human-readable summary for agent consumption (vLLM metrics only)."""
        lines = [
            f"{'='*60}",
            f"  PROMETHEUS METRICS DELTA (server-side, during benchmark)",
            f"{'='*60}",
            f"Endpoint: {self.endpoint}",
            f"Duration: {self.duration_seconds:.1f}s",
        ]

        # ── Gauges ──
        vllm_gauges = {k: v for k, v in self.gauge_snapshots.items() if k.startswith("vllm:")}
        if vllm_gauges:
            lines.append("\n--- Server State (end-of-benchmark snapshot) ---")
            for name, val in sorted(vllm_gauges.items()):
                display = _format_metric_name(name)
                if "perc" in name:
                    lines.append(f"  {display}: {val:.1%}")
                else:
                    lines.append(f"  {display}: {val:.1f}")
            # Flag KV cache pressure
            kv = vllm_gauges.get("vllm:kv_cache_usage_perc", 0)
            if kv > 0.9:
                lines.append(f"  ⚠ KV CACHE PRESSURE: {kv:.1%} used — risk of preemptions")
            waiting = vllm_gauges.get("vllm:num_requests_waiting", 0)
            if waiting > 0:
                lines.append(f"  ⚠ QUEUING: {waiting:.0f} requests waiting at snapshot time")

        # ── Counters ──
        vllm_counters = {k: v for k, v in self.counter_deltas.items() if k.startswith("vllm:")}
        if vllm_counters:
            lines.append("\n--- Throughput Counters (delta during this benchmark run) ---")
            for name in sorted(vllm_counters.keys()):
                display = _format_metric_name(name)
                delta = self.counter_deltas[name]
                rate = self.counter_rates.get(name, 0)
                if delta < 0:
                    lines.append(f"  {display}: COUNTER RESET (server restarted?)")
                else:
                    lines.append(f"  {display}: +{delta:,.0f} ({rate:,.2f}/sec)")

            # Derived: preemption warning
            preemptions = self.counter_deltas.get("vllm:num_preemptions_total", 0)
            if preemptions > 0:
                lines.append(f"  ⚠ PREEMPTIONS: {preemptions:.0f} sequences evicted during run")

            # Derived: token throughput
            gen_delta = self.counter_deltas.get("vllm:generation_tokens_total", 0)
            if gen_delta > 0 and self.duration_seconds > 0:
                lines.append(f"  → Server-side generation throughput: {gen_delta / self.duration_seconds:,.1f} tok/sec")

            # Derived: prefix cache hit rate
            cache_hits = self.counter_deltas.get("vllm:prefix_cache_hits_total", 0)
            cache_queries = self.counter_deltas.get("vllm:prefix_cache_queries_total", 0)
            if cache_queries > 0:
                hit_rate = cache_hits / cache_queries
                lines.append(f"  → Prefix cache hit rate: {hit_rate:.1%} ({cache_hits:.0f}/{cache_queries:.0f})")

            # Derived: success rate
            success = self.counter_deltas.get("vllm:request_success_total", 0)
            if success > 0:
                lines.append(f"  → Completed requests (server-side): {success:.0f}")

        # ── Histograms ──
        vllm_hists = {k: v for k, v in self.histogram_summaries.items() if k.startswith("vllm:")}
        if vllm_hists:
            lines.append("\n--- Latency Distributions (server-side histograms) ---")
            # Show key latency histograms first
            key_order = [
                "vllm:time_to_first_token_seconds",
                "vllm:inter_token_latency_seconds",
                "vllm:e2e_request_latency_seconds",
                "vllm:request_queue_time_seconds",
                "vllm:request_prefill_time_seconds",
                "vllm:request_decode_time_seconds",
                "vllm:request_inference_time_seconds",
            ]
            shown = set()
            for name in key_order:
                if name in vllm_hists:
                    shown.add(name)
                    self._append_histogram_line(lines, name, vllm_hists[name])
            for name, summary in sorted(vllm_hists.items()):
                if name not in shown:
                    self._append_histogram_line(lines, name, summary)

        lines.append(f"{'='*60}")
        return "\n".join(lines)

    @staticmethod
    def _append_histogram_line(lines, name, summary):
        display = _format_metric_name(name)
        count = summary.get("count", 0)
        if count == 0:
            lines.append(f"  {display}: no samples")
            return
        mean = summary.get("mean")
        p50 = summary.get("p50")
        p95 = summary.get("p95")
        p99 = summary.get("p99")
        if "seconds" in name or "time" in name:
            fmt = lambda v: f"{v * 1000:.2f}ms" if v is not None else "N/A"
        else:
            fmt = lambda v: f"{v:.1f}" if v is not None else "N/A"
        lines.append(
            f"  {display}: count={count:.0f}, "
            f"mean={fmt(mean)}, p50={fmt(p50)}, p95={fmt(p95)}, p99={fmt(p99)}"
        )


# ── Parsing ──────────────────────────────────────────────────────────────

def _normalize_metric_name(name: str) -> str:
    """Normalize vllm_ prefix to vllm: for consistency."""
    if name.startswith("vllm_"):
        return "vllm:" + name[5:]
    return name


def _format_metric_name(name: str) -> str:
    """Human-friendly metric name."""
    short = name.replace("vllm:", "").replace("vllm_", "")
    return short


def parse_prometheus_text(
    text: str,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, HistogramData]]:
    """Parse Prometheus text exposition format.

    Returns (gauges, counters, histograms) dicts keyed by normalized metric name.
    """
    type_map: Dict[str, str] = {}
    gauges: Dict[str, float] = {}
    counters: Dict[str, float] = {}
    histograms: Dict[str, HistogramData] = {}

    # Pass 1: collect TYPE declarations
    for line in text.splitlines():
        m = _TYPE_RE.match(line)
        if m:
            raw_name, mtype = m.group(1), m.group(2).lower()
            type_map[_normalize_metric_name(raw_name)] = mtype

    # Pass 2: parse value lines
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue

        raw_name = m.group("name")
        labels_str = m.group("labels") or ""
        try:
            value = float(m.group("value"))
        except ValueError:
            continue

        norm_name = _normalize_metric_name(raw_name)
        labels = dict(_LABEL_RE.findall(labels_str))

        # Determine base metric name and suffix
        base_name = norm_name
        suffix = ""
        for s in ("_bucket", "_count", "_sum", "_total", "_created"):
            if norm_name.endswith(s):
                base_name = norm_name[: -len(s)]
                suffix = s
                break

        # Skip _created metrics (timestamps from prometheus_client, not useful)
        if suffix == "_created":
            continue

        # Look up type from base name
        mtype = type_map.get(base_name, "")

        # Route to appropriate container
        if mtype == "histogram":
            hist = histograms.setdefault(base_name, HistogramData())
            if suffix == "_bucket":
                le = labels.get("le")
                if le is not None:
                    try:
                        le_val = float(le)
                    except ValueError:
                        le_val = float("inf")
                    hist.bucket_counts[le_val] = value
            elif suffix == "_sum":
                hist.sum_value += value
            elif suffix == "_count":
                hist.count += value
        elif mtype == "counter" or suffix == "_total":
            counter_key = base_name + "_total" if not base_name.endswith("_total") else base_name
            # Sum across label variants (e.g. request_success by finished_reason)
            counters[counter_key] = counters.get(counter_key, 0) + value
        elif mtype == "gauge":
            gauges[norm_name] = value
        elif mtype == "summary":
            pass  # skip summaries for now
        else:
            # Unknown type — classify by known sets or default to gauge
            if base_name in VLLM_COUNTERS or any(base_name.startswith(c.rstrip("_total")) for c in VLLM_COUNTERS):
                counters[base_name] = counters.get(base_name, 0) + value
            elif base_name in VLLM_HISTOGRAMS:
                hist = histograms.setdefault(base_name, HistogramData())
                if suffix == "_bucket":
                    le = labels.get("le")
                    if le is not None:
                        try:
                            le_val = float(le)
                        except ValueError:
                            le_val = float("inf")
                        hist.bucket_counts[le_val] = value
                elif suffix == "_sum":
                    hist.sum_value = value
                elif suffix == "_count":
                    hist.count = value
            else:
                gauges[norm_name] = value

    return gauges, counters, histograms


# ── Scraping ─────────────────────────────────────────────────────────────

def scrape_metrics(endpoint: str, timeout: int = 10) -> MetricsSnapshot:
    """Scrape /metrics from a vLLM endpoint.

    Args:
        endpoint: Base URL like "http://localhost:8000"
        timeout: HTTP timeout in seconds

    Returns:
        MetricsSnapshot with parsed metrics
    """
    url = endpoint.rstrip("/") + "/metrics"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    gauges, counters, histograms = parse_prometheus_text(text)

    return MetricsSnapshot(
        timestamp=datetime.utcnow().isoformat(),
        endpoint=endpoint,
        gauges=gauges,
        counters=counters,
        histograms=histograms,
    )


# ── Delta Computation ────────────────────────────────────────────────────

def compute_delta(before: MetricsSnapshot, after: MetricsSnapshot) -> MetricsDelta:
    """Compute difference between two snapshots.

    - Gauges: take "after" value
    - Counters: after - before (negative = counter reset)
    - Counter rates: delta / duration
    - Histograms: summarize "after" snapshot
    """
    try:
        t_before = datetime.fromisoformat(before.timestamp)
        t_after = datetime.fromisoformat(after.timestamp)
        duration = (t_after - t_before).total_seconds()
    except (ValueError, TypeError):
        duration = 0.0

    if duration <= 0:
        duration = 1.0  # avoid division by zero

    # Gauges: latest value
    gauge_snapshots = dict(after.gauges)

    # Counters: delta
    counter_deltas: Dict[str, float] = {}
    counter_rates: Dict[str, float] = {}
    all_counter_keys = set(before.counters.keys()) | set(after.counters.keys())
    for key in all_counter_keys:
        after_val = after.counters.get(key, 0)
        before_val = before.counters.get(key, 0)
        delta = after_val - before_val
        counter_deltas[key] = delta
        counter_rates[key] = delta / duration if delta >= 0 else 0

    # Histograms: summarize after snapshot
    histogram_summaries: Dict[str, dict] = {}
    for key, hist in after.histograms.items():
        histogram_summaries[key] = hist.summary_dict()

    return MetricsDelta(
        start_time=before.timestamp,
        end_time=after.timestamp,
        endpoint=after.endpoint,
        duration_seconds=duration,
        gauge_snapshots=gauge_snapshots,
        counter_deltas=counter_deltas,
        counter_rates=counter_rates,
        histogram_summaries=histogram_summaries,
    )
