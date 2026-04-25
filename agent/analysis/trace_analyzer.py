"""
PyTorch Profiler Trace Analyzer

Parses Chrome trace JSON files from PyTorch profiler and extracts
kernel-level performance statistics.

SOURCE: AI-Analysis-Agent/psap-mcp-server/psap_mcp_server/src/tools/pytorch_profile_tool.py

Functions to extract (converted to sync, no MCP/S3/async):
    - extract_kernel_stats(trace_data)       — Parse trace events into kernel stats
    - merge_stats(stats_list)                — Merge stats from multiple traces
    - get_top_kernels(stats, n=20)           — Top N kernels by total duration
    - get_category_breakdown(stats)          — Breakdown by functional category
    - classify_kernel(kernel_name)           — Map kernel to functional pipeline

Constants to extract:
    - FUNCTIONAL_PIPELINES  — Dict mapping kernel patterns to pipeline categories
      (e.g., attention, linear/gemm, normalization, activation, memory, communication)

Input:  Chrome trace JSON (from PyTorch profiler via vllm-profiler)
Output: Dict with top_kernels, category_breakdown, total_gpu_time, etc.
"""

# TODO: Copy FUNCTIONAL_PIPELINES constant
# TODO: Copy _extract_kernel_stats() → extract_kernel_stats()
# TODO: Copy _merge_stats() → merge_stats()
# TODO: Copy _get_top_kernels() → get_top_kernels()
# TODO: Copy _get_category_breakdown() → get_category_breakdown()
# TODO: Copy classify_kernel() helper
# TODO: Remove all MCP registration, S3 loading, async, logging
