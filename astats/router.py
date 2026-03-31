"""
AStats Lightweight LLM Router
==============================
Replaces litellm with a zero-bloat, httpx-based client that speaks the
OpenAI-compatible chat-completions API.

Supported backends natively:
  * Ollama     — local open-weight models (default)
  * OpenAI     — gpt-4o, gpt-4-turbo, etc.
  * Gemini     — gemini-2.5-flash, gemini-1.5-pro, etc.
  * Groq       — llama3-70b-8192, mixtral-8x7b-32768, etc.
  * Anthropic  — claude-3-5-sonnet-20240620, claude-3-opus-20240229, etc.

Memory footprint: ~4 MB (only imports httpx and json).
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------

_OPENAI_API_BASE = "https://api.openai.com/v1"
_OLLAMA_DEFAULT_BASE = "http://localhost:11434"
_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_GROQ_API_BASE = "https://api.groq.com/openai/v1"
_ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"


def _resolve_provider(model: str, api_base: str | None) -> tuple[str, str, str | None]:
    """
    Returns (provider_name, resolved_model_name, resolved_api_base).
    """
    if model.startswith("ollama/"):
        resolved = model[len("ollama/"):]
        return "ollama", resolved, api_base or os.getenv("OLLAMA_API_BASE", _OLLAMA_DEFAULT_BASE)

    if model.startswith("gemini/"):
        resolved = model[len("gemini/"):]
        return "gemini", resolved, api_base or _GEMINI_API_BASE

    if model.startswith("groq/"):
        resolved = model[len("groq/"):]
        return "groq", resolved, api_base or _GROQ_API_BASE

    if model.startswith("anthropic/"):
        resolved = model[len("anthropic/"):]
        return "anthropic", resolved, api_base or _ANTHROPIC_API_BASE

    if model.startswith("openai/"):
        resolved = model[len("openai/"):]
        return "openai", resolved, api_base or _OPENAI_API_BASE

    # Default fallback to OpenAI if no prefix provided
    return "openai", model, api_base or _OPENAI_API_BASE


def _get_api_key(provider: str) -> str | None:
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    elif provider == "gemini":
        return os.getenv("GEMINI_API_KEY")
    elif provider == "groq":
        return os.getenv("GROQ_API_KEY")
    elif provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    return os.getenv("ASTATS_CUSTOM_API_KEY")


# ---------------------------------------------------------------------------
# Response wrapper — mirrors litellm/openai format for core.py compatibility
# ---------------------------------------------------------------------------

class _ToolCall:
    def __init__(self, raw: dict):
        self.id = raw.get("id", "call_default")
        self.type = raw.get("type", "function")

        class _Fn:
            def __init__(self, fn: dict):
                self.name = fn["name"]
                self.arguments = fn.get("arguments", "{}")

        self.function = _Fn(raw.get("function", {}))


class _Message:
    def __init__(self, raw: dict):
        self.role = raw.get("role", "assistant")
        self.content = raw.get("content")
        raw_tools = raw.get("tool_calls")
        self.tool_calls = [_ToolCall(tc) for tc in raw_tools] if raw_tools else None

    def __iter__(self):
        yield "role", self.role
        yield "content", self.content
        if self.tool_calls is not None:
            yield "tool_calls", [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]


class _Choice:
    def __init__(self, raw: dict):
        self.message = _Message(raw.get("message", {}))


class CompletionResponse:
    def __init__(self, raw: dict):
        self.choices = [_Choice(c) for c in raw.get("choices", [])]


# ---------------------------------------------------------------------------
# Provider Implementation Engines
# ---------------------------------------------------------------------------

def _ollama_chat(base: str, model: str, messages: list[dict], tools: list[dict] | None, timeout: float) -> CompletionResponse:
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if tools: payload["tools"] = tools
    response = httpx.post(f"{base.rstrip('/')}/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    raw = response.json()
    msg = raw.get("message", {})
    return CompletionResponse({"choices": [{"message": {"role": msg.get("role", "assistant"), "content": msg.get("content"), "tool_calls": msg.get("tool_calls")}}]})


def _openai_chat(base: str, model: str, messages: list[dict], tools: list[dict] | None, api_key: str | None, timeout: float) -> CompletionResponse:
    """Handles OpenAI, Groq, and Gemini (via beta OpenAI compatibility endpoint)."""
    headers = {"Content-Type": "application/json"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    response = httpx.post(f"{base.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return CompletionResponse(response.json())


def _anthropic_chat(base: str, model: str, messages: list[dict], tools: list[dict] | None, api_key: str | None, timeout: float) -> CompletionResponse:
    """Translates OpenAI messages/tools payload into Anthropic Messages API format."""
    headers = {
        "x-api-key": api_key or "",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Extract system prompt if present
    system_prompt = ""
    anthropic_msgs = []
    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        elif m["role"] == "tool":
            # Anthropic handles tool results differently
            anthropic_msgs.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""), "content": m.get("content")}]
            })
        else:
            # Reformat content if previous turn was an assistant tool call
            if m["role"] == "assistant" and "tool_calls" in m:
                content = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                    })
                anthropic_msgs.append({"role": "assistant", "content": content})
            else:
                anthropic_msgs.append({"role": m["role"], "content": m["content"]})

    payload: dict[str, Any] = {"model": model, "messages": anthropic_msgs, "max_tokens": 4096}
    if system_prompt: payload["system"] = system_prompt
    
    if tools:
        anthropic_tools = []
        for t in tools:
            fn = t["function"]
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}})
            })
        payload["tools"] = anthropic_tools

    response = httpx.post(f"{base.rstrip('/')}/messages", headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    raw = response.json()
    
    # Translate Anthropic response back to OpenAI CompletionResponse format
    content_text = ""
    tool_calls = []
    
    for block in raw.get("content", []):
        if block["type"] == "text":
            content_text += block["text"]
        elif block["type"] == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block["input"])
                }
            })
            
    norm_msg = {"role": "assistant", "content": content_text if content_text else None}
    if tool_calls: norm_msg["tool_calls"] = tool_calls
    return CompletionResponse({"choices": [{"message": norm_msg}]})


# ---------------------------------------------------------------------------
# Public API — drop-in replacement for litellm.completion()
# ---------------------------------------------------------------------------

def completion(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_choice: str = "auto",
    api_base: str | None = None,
    timeout: float = 120.0,
) -> CompletionResponse:
    # Sanitise messages: remove None keys
    clean_messages = [{k: v for k, v in m.items() if v is not None} for m in messages]
    provider, resolved_model, resolved_base = _resolve_provider(model, api_base)

    if provider == "ollama":
        return _ollama_chat(resolved_base, resolved_model, clean_messages, tools, timeout)
    elif provider == "anthropic":
        return _anthropic_chat(resolved_base, resolved_model, clean_messages, tools, _get_api_key(provider), timeout)
    else:
        # OpenAI, Gemini, Groq all use OpenAI-compatible format
        return _openai_chat(resolved_base, resolved_model, clean_messages, tools, _get_api_key(provider), timeout)
