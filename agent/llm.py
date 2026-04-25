"""
Claude LLM Client & Token Tracking

Wraps the Anthropic API client with token usage tracking and cost reporting.

SOURCE: ai-perf-hackathon/agent/llm.py (near-verbatim)

Key classes/functions to copy:
    - ClaudeClient        — Anthropic API wrapper with prompt caching
    - TokenUsage          — Dataclass for tracking input/output/cache tokens
    - LLMResponse         — Dataclass for structured API responses
    - MODEL_PRICING       — Dict of per-model token costs
    - get_usage_report()  — Formatted string of total tokens + cost

Changes from source:
    - Remove Vertex AI path (use direct Anthropic API only)
    - Keep prompt caching (cache_control) support
"""

# TODO: Copy ClaudeClient class
# TODO: Copy TokenUsage dataclass
# TODO: Copy LLMResponse dataclass
# TODO: Copy MODEL_PRICING dict
# TODO: Copy get_usage_report() function
