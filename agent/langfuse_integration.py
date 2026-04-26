"""
Langfuse Integration for vLLM Tuning/Profiling Agents

Provides observability via Langfuse traces, spans, and generations.
Enabled via --langfuse flag + LANGFUSE_* env vars.

For self-hosted instances with self-signed certs, SSL verification
is automatically disabled.
"""
from __future__ import annotations

import os
import warnings


def init_langfuse():
    """Initialize Langfuse with SSL bypass for self-signed certs.

    Must be called BEFORE any langfuse imports that trigger OTEL setup.
    Returns True if Langfuse is configured, False otherwise.
    """
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return False

    warnings.filterwarnings("ignore", message=".*InsecureRequestWarning.*")
    warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

    # Patch OTEL exporter to disable SSL verification (self-signed certs)
    try:
        import opentelemetry.exporter.otlp.proto.http.trace_exporter as otel_exp
        _orig_init = otel_exp.OTLPSpanExporter.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self._certificate_file = False

        otel_exp.OTLPSpanExporter.__init__ = _patched_init
    except ImportError:
        pass

    return True


def get_observe():
    """Get the @observe decorator, or a no-op if Langfuse is not available."""
    try:
        from langfuse import observe
        return observe
    except ImportError:
        def noop_observe(*args, **kwargs):
            def decorator(fn):
                return fn
            if args and callable(args[0]):
                return args[0]
            return decorator
        return noop_observe


def get_langfuse_client():
    """Get a Langfuse client instance, or None."""
    try:
        from langfuse import Langfuse
        return Langfuse()
    except Exception:
        return None


def flush_langfuse():
    """Flush any pending Langfuse events."""
    try:
        from langfuse import Langfuse
        lf = Langfuse()
        lf.flush()
    except Exception:
        pass
