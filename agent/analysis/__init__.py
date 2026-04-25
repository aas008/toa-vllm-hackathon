"""
Analysis Modules

Extracted from AI-Analysis-Agent MCP tools, converted to sync plain functions.
All MCP registration, S3 dependencies, async patterns, and loggers removed.

Modules:
    - trace_analyzer  -- PyTorch profiler trace analysis
    - kernel_mapper   -- Kernel-to-source-code mapping
    - regression      -- Performance regression detection
    - cost            -- Cost efficiency (CPMT) calculation

Quick-start::

    from agent.analysis import (
        analyze_trace,
        map_kernel,
        detect_regression,
        calculate_cost,
    )
"""

# -- Trace Analyzer --
from agent.analysis.trace_analyzer import (
    analyze_trace,
    extract_kernel_stats,
    merge_stats,
    get_top_kernels,
    get_category_breakdown,
    get_pipeline_breakdown,
    classify_kernel,
    FUNCTIONAL_PIPELINES,
    CATEGORY_GROUPS,
)

# -- Kernel Mapper --
from agent.analysis.kernel_mapper import (
    map_kernel,
    find_kernel_mapping,
    is_pytorch_stdlib,
    KERNEL_MAPPINGS,
    PYTORCH_STDLIB_OPS,
)

# -- Regression Detection --
from agent.analysis.regression import (
    detect_regression,
)

# -- Cost Efficiency --
from agent.analysis.cost import (
    calculate_cost,
    calculate_cpmt,
    filter_by_slo,
    compare_cost_efficiency,
    ACCELERATOR_PRICING,
)

__all__ = [
    # trace_analyzer
    "analyze_trace",
    "extract_kernel_stats",
    "merge_stats",
    "get_top_kernels",
    "get_category_breakdown",
    "get_pipeline_breakdown",
    "classify_kernel",
    "FUNCTIONAL_PIPELINES",
    "CATEGORY_GROUPS",
    # kernel_mapper
    "map_kernel",
    "find_kernel_mapping",
    "is_pytorch_stdlib",
    "KERNEL_MAPPINGS",
    "PYTORCH_STDLIB_OPS",
    # regression
    "detect_regression",
    # cost
    "calculate_cost",
    "calculate_cpmt",
    "filter_by_slo",
    "compare_cost_efficiency",
    "ACCELERATOR_PRICING",
]
