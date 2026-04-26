import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoodputResult:
    label: str
    successful: int
    errored: int
    incomplete: int
    duration: float
    slo_passing: int
    slo_pct: float
    goodput_rps: float
    goodput_tok_per_sec: float
    ttft_p95_ms: float
    itl_p95_ms: float


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("inf")
    values_sorted = sorted(values)
    idx = min(int(len(values_sorted) * p), len(values_sorted) - 1)
    return values_sorted[idx]


def compute_goodput(benchmark: dict, ttft_target_ms: float, itl_target_ms: float) -> GoodputResult:
    config = benchmark["config"]
    strategy = config["strategy"]["type_"]
    rate = config["strategy"].get("rate")
    label = strategy if rate is None else f"{strategy}@{rate:.1f}"

    successful = benchmark["requests"]["successful"]
    errored = benchmark["requests"]["errored"]
    incomplete = benchmark["requests"]["incomplete"]
    duration = benchmark["duration"]

    slo_passing = 0
    slo_output_tokens = 0
    ttft_values: list[float] = []
    itl_values: list[float] = []

    for r in successful:
        ttft = r["time_to_first_token_ms"]
        itl = r["inter_token_latency_ms"]
        ttft_values.append(ttft)
        itl_values.append(itl)
        if ttft <= ttft_target_ms and itl <= itl_target_ms:
            slo_passing += 1
            slo_output_tokens += r["output_tokens"]

    slo_pct = (slo_passing / len(successful) * 100) if successful else 0.0
    goodput_rps = slo_passing / duration if duration > 0 else 0.0
    goodput_tok_per_sec = slo_output_tokens / duration if duration > 0 else 0.0

    return GoodputResult(
        label=label,
        successful=len(successful),
        errored=len(errored),
        incomplete=len(incomplete),
        duration=duration,
        slo_passing=slo_passing,
        slo_pct=slo_pct,
        goodput_rps=goodput_rps,
        goodput_tok_per_sec=goodput_tok_per_sec,
        ttft_p95_ms=percentile(ttft_values, 0.95),
        itl_p95_ms=percentile(itl_values, 0.95),
    )


def main():
    parser = argparse.ArgumentParser(description="Compute goodput from GuideLLM benchmark results")
    parser.add_argument("benchmark_file", type=Path, help="Path to benchmarks.json")
    parser.add_argument("--ttft-target", type=float, default=200.0, help="TTFT SLO target in ms (default: 200)")
    parser.add_argument("--itl-target", type=float, default=5.0, help="ITL SLO target in ms (default: 5)")
    args = parser.parse_args()

    with open(args.benchmark_file) as f:
        data = json.load(f)

    results = [compute_goodput(b, args.ttft_target, args.itl_target) for b in data["benchmarks"]]

    print(f"SLO targets: TTFT < {args.ttft_target}ms, ITL < {args.itl_target}ms\n")

    header = (
        f"{'Strategy':<20} {'OK':>5} {'Err':>5} {'Inc':>5}"
        f" {'TTFT_p95':>9} {'ITL_p95':>8}"
        f" {'SLO%':>6} {'Goodput':>10} {'GoodTok/s':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r.label:<20} {r.successful:>5} {r.errored:>5} {r.incomplete:>5}"
            f" {r.ttft_p95_ms:>8.1f}ms {r.itl_p95_ms:>7.1f}ms"
            f" {r.slo_pct:>5.1f}% {r.goodput_rps:>8.2f}/s {r.goodput_tok_per_sec:>10.1f}"
        )

if __name__ == "__main__":
    main()
