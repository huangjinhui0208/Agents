from __future__ import annotations

import json
import re
import socket
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI

from .config import AgentConfig
from .contracts import ModelConnectionError, ModelReply, ModelToolCall, new_id


def strip_hidden_reasoning(text: str) -> str:
    """Do not surface model scratchpad tags in the user-facing answer."""
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in cleaned.lower():
        cleaned = re.split(r"</think>", cleaned, flags=re.IGNORECASE)[-1]
    return cleaned.strip()


class LocalLLMClient:
    def __init__(self, config: AgentConfig):
        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout_s,
        )

    def health_status(self) -> str:
        parsed = urlparse(self.config.base_url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((parsed.hostname or "127.0.0.1", port), timeout=0.25):
                return "ready"
        except OSError:
            return "provider_offline"

    @staticmethod
    def _content(message: Any) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return strip_hidden_reasoning(content)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
                elif getattr(item, "text", None):
                    parts.append(str(item.text))
            return strip_hidden_reasoning("".join(parts))
        return strip_hidden_reasoning(str(content or ""))

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
    ) -> ModelReply:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.0,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            print(f"[llm] {type(exc).__name__}: {exc}")
            raise ModelConnectionError(
                f"无法调用模型提供方 {self.config.provider}。请确认模型 "
                f"{self.config.model} 可用，并检查 {self.config.base_url}。"
            ) from exc

        choice = response.choices[0]
        message = choice.message
        calls: list[ModelToolCall] = []
        for raw_call in list(getattr(message, "tool_calls", None) or []):
            raw_arguments = getattr(raw_call.function, "arguments", "{}") or "{}"
            try:
                arguments = json.loads(raw_arguments)
                if not isinstance(arguments, dict):
                    arguments = {}
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                ModelToolCall(
                    id=str(raw_call.id or new_id("call")),
                    name=str(raw_call.function.name),
                    arguments=arguments,
                )
            )
        return ModelReply(
            content=self._content(message),
            tool_calls=calls,
            finish_reason=getattr(choice, "finish_reason", None),
        )
