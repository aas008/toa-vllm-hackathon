"""
Debug Recorder — captures full LLM conversation transcripts for post-run analysis.

Records every prompt, response, tool call, and tool result as JSONL.
Enabled via --debug flag.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional


class DebugRecorder:
    """Records full agent conversation to a JSONL file."""

    def __init__(self, output_dir: str, agent_type: str = "tuning"):
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(output_dir, f"debug_{agent_type}_{ts}.jsonl")
        self._file = open(self.path, "w")
        self._seq = 0
        self._write("session_start", {
            "agent_type": agent_type,
            "timestamp": ts,
        })

    def _write(self, event_type: str, data: dict):
        self._seq += 1
        record = {
            "seq": self._seq,
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            **data,
        }
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def record_system_prompt(self, prompt: str):
        self._write("system_prompt", {"content": prompt})

    def record_tools(self, tools: list[dict]):
        self._write("tool_definitions", {
            "count": len(tools),
            "names": [t["name"] for t in tools],
        })

    def record_user_message(self, content, iteration: int = 0):
        self._write("user_message", {
            "iteration": iteration,
            "content": _serialize_content(content),
        })

    def record_llm_request(self, model: str, messages: list, iteration: int = 0):
        self._write("llm_request", {
            "iteration": iteration,
            "model": model,
            "message_count": len(messages),
            "last_message_role": messages[-1]["role"] if messages else "",
        })

    def record_llm_response(
        self,
        model: str,
        stop_reason: str,
        content: list,
        usage: dict,
        iteration: int = 0,
    ):
        serialized = []
        for block in content:
            if hasattr(block, "type"):
                if block.type == "text":
                    serialized.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    serialized.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            elif isinstance(block, dict):
                serialized.append(block)

        self._write("llm_response", {
            "iteration": iteration,
            "model": model,
            "stop_reason": stop_reason,
            "content": serialized,
            "usage": usage,
        })

    def record_tool_call(
        self,
        name: str,
        inputs: dict,
        tool_use_id: str,
        iteration: int = 0,
    ):
        self._write("tool_call", {
            "iteration": iteration,
            "tool": name,
            "inputs": inputs,
            "tool_use_id": tool_use_id,
        })

    def record_tool_result(
        self,
        name: str,
        success: bool,
        output: str,
        error: Optional[str] = None,
        iteration: int = 0,
    ):
        self._write("tool_result", {
            "iteration": iteration,
            "tool": name,
            "success": success,
            "output_length": len(output),
            "output_preview": output[:2000],
            "error": error,
        })

    def record_full_tool_output(
        self,
        name: str,
        output: str,
        iteration: int = 0,
    ):
        """Record full untruncated tool output (can be large)."""
        self._write("tool_output_full", {
            "iteration": iteration,
            "tool": name,
            "output": output,
        })

    def close(self):
        self._write("session_end", {})
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _serialize_content(content) -> str | list:
    """Serialize message content to JSON-safe format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        result = []
        for item in content:
            if isinstance(item, dict):
                result.append(item)
            elif hasattr(item, "type"):
                if item.type == "text":
                    result.append({"type": "text", "text": item.text})
                elif item.type == "tool_use":
                    result.append({"type": "tool_use", "name": item.name, "input": item.input})
                elif item.type == "tool_result":
                    result.append({"type": "tool_result", "content": str(item.content)[:500]})
            else:
                result.append(str(item))
        return result
    return str(content)
