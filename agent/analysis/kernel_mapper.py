"""
Kernel-to-Source-Code Mapper

Maps CUDA kernel names to their vLLM/PyTorch source code locations
and provides human-readable descriptions of kernel functionality.

SOURCE: AI-Analysis-Agent/psap-mcp-server/psap_mcp_server/src/tools/kernel_code_mapper_tool.py

Functions to extract (converted to sync, no MCP):
    - find_kernel_mapping(kernel_name)   — Look up kernel in KERNEL_MAPPINGS
    - is_pytorch_stdlib(kernel_name)     — Check if kernel is PyTorch standard library

Constants to extract:
    - KERNEL_MAPPINGS     — Dict mapping kernel name patterns to:
                            {source_file, function, description, category}
    - PYTORCH_STDLIB_OPS  — Set of kernel names that are PyTorch built-ins
                            (not vLLM custom kernels)

Input:  Kernel name string (e.g., "flash_fwd_kernel", "rms_norm_kernel")
Output: Dict with source location, description, category, is_stdlib flag
"""

# TODO: Copy KERNEL_MAPPINGS constant
# TODO: Copy PYTORCH_STDLIB_OPS constant
# TODO: Copy _find_kernel_mapping() → find_kernel_mapping()
# TODO: Copy _is_pytorch_stdlib() → is_pytorch_stdlib()
# TODO: Remove MCP registration, logging
