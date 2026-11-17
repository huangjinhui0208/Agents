from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


ToolHandler = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"重复工具名：{definition.name}")
        self._tools[definition.name] = definition

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def execute(
        self, name: str, arguments: dict[str, Any], tool_call_id: str
    ) -> dict[str, Any]:
        definition = self._tools.get(name)
        if definition is None:
            return {
                "schema_version": "1.0",
                "tool_call_id": tool_call_id,
                "tool": name,
                "status": "error",
                "data": {},
                "source": {},
                "truncated": False,
                "next_cursor": None,
                "error": {
                    "code": "UNKNOWN_TOOL",
                    "message": f"未注册工具：{name}",
                    "retryable": False,
                },
            }
        try:
            return definition.handler(tool_call_id=tool_call_id, **arguments)
        except TypeError as exc:
            return {
                "schema_version": "1.0",
                "tool_call_id": tool_call_id,
                "tool": name,
                "status": "error",
                "data": {},
                "source": {},
                "truncated": False,
                "next_cursor": None,
                "error": {
                    "code": "INVALID_ARGUMENTS",
                    "message": str(exc),
                    "retryable": True,
                },
            }
        except Exception as exc:  # Keep tool failures inside the auditable loop.
            return {
                "schema_version": "1.0",
                "tool_call_id": tool_call_id,
                "tool": name,
                "status": "error",
                "data": {},
                "source": {},
                "truncated": False,
                "next_cursor": None,
                "error": {
                    "code": "TOOL_ERROR",
                    "message": str(exc),
                    "retryable": True,
                },
            }
