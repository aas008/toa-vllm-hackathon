"""ProfilerConfig for webhook-based PyTorch profiler injection into vLLM pods."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProfilerConfig:
    """Configuration for PyTorch profiler injection via webhook annotations."""
    ranges: str = "100-150"
    activities: str = "CPU,CUDA"
    record_shapes: str = "true"
    with_stack: str = "true"
    memory: str = "false"
    export_trace: str = "true"
    debug: str = "false"
    output: str = "/tmp/trace_pid{pid}_range{start}-{end}.json"

    def to_annotations(self) -> dict[str, str]:
        return {
            "vllm.profiler/ranges": self.ranges,
            "vllm.profiler/activities": self.activities,
            "vllm.profiler/record-shapes": self.record_shapes,
            "vllm.profiler/with-stack": self.with_stack,
            "vllm.profiler/memory": self.memory,
            "vllm.profiler/export-trace": self.export_trace,
            "vllm.profiler/debug": self.debug,
            "vllm.profiler/output": self.output,
        }

    def max_call_count(self) -> int:
        """Highest call number needed to complete all profiling ranges."""
        max_end = 0
        for r in self.ranges.split(","):
            r = r.strip()
            if "-" in r:
                _, end = r.split("-", 1)
                max_end = max(max_end, int(end))
        return max_end
