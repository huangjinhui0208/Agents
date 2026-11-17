from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

try:
    import tiktoken
except ImportError:  # Token counts are diagnostics only.
    tiktoken = None

from .config import AgentConfig
from .contracts import RuntimeEvent, TaskRequest
from .runtime import RuntimeMemory


@dataclass(frozen=True)
class ContextBuildResult:
    messages: list[dict[str, Any]]
    token_count: int
    compacted: bool


class ContextBuilder:
    """Build the complete native-tool conversation for every model request.

    The application does not impose input or output token budgets. Repository
    tools remain bounded and paginated, while the selected model provider owns its
    context-window policy.
    """

    def __init__(self, config: AgentConfig, system_prompt: str):
        self.config = config
        self.system_prompt = system_prompt
        self._encoding = None
        if tiktoken is not None:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoding = None

    def count_tokens(self, value: Any) -> int:
        """Return a diagnostic estimate; never reject or truncate a request."""

        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return max(1, (len(text) + 1) // 2)

    @staticmethod
    def _event_messages(events: list[RuntimeEvent]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for event in events:
            message = event.payload.get("message")
            if isinstance(message, dict) and message.get("role"):
                messages.append(message)
        return messages

    @staticmethod
    def _complete_protocol_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Exclude only unfinished tool-call groups that would violate the API."""

        complete: list[dict[str, Any]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            tool_calls = message.get("tool_calls") if message.get("role") == "assistant" else None
            if not tool_calls:
                if message.get("role") != "tool":
                    complete.append(message)
                index += 1
                continue
            expected = {call.get("id") for call in tool_calls if call.get("id")}
            block = [message]
            cursor = index + 1
            seen: set[str] = set()
            while cursor < len(messages) and messages[cursor].get("role") == "tool":
                result = messages[cursor]
                if result.get("tool_call_id") not in expected:
                    break
                seen.add(result.get("tool_call_id"))
                block.append(result)
                cursor += 1
            if expected and seen == expected:
                complete.extend(block)
            index = cursor
        return complete

    @staticmethod
    def _note_context(notes: list[dict[str, Any]]) -> str:
        if not notes:
            return ""
        lines = [
            "以下是当前仓库通过向量检索找到的历史知识摘要。"
            "它们可能过期，不能替代当前源码核查："
        ]
        for note in notes:
            similarity = note.get("similarity")
            score = f", similarity={similarity:.3f}" if isinstance(similarity, float) else ""
            lines.append(
                f"- [{note.get('status')}] {note.get('title')} "
                f"(id={note.get('note_id')}, v{note.get('version')}{score}): "
                f"{note.get('summary', '')}"
            )
        return "\n".join(lines)

    def build(
        self,
        *,
        request: TaskRequest,
        runtime: RuntimeMemory,
        relevant_notes: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
    ) -> ContextBuildResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        note_context = self._note_context(relevant_notes)
        if note_context:
            messages.append({"role": "system", "content": note_context})
        history = self._complete_protocol_messages(
            self._event_messages(runtime.events(request.session_id))
        )
        messages.extend(history)
        token_count = self.count_tokens(messages) + self.count_tokens(tool_schemas)
        return ContextBuildResult(
            messages=messages,
            token_count=token_count,
            compacted=False,
        )
