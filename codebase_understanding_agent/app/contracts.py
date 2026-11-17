from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Literal
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


@dataclass(frozen=True)
class TaskRequest:
    request_id: str
    session_id: str
    repo_id: str
    user_text: str
    repository_root: str
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class RuntimeEvent:
    event_id: str
    session_id: str
    request_id: str
    timestamp: str
    role: Literal["user", "assistant", "tool", "system"]
    payload: dict[str, Any]
    token_estimate: int
    artifact_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def as_openai_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass(frozen=True)
class ModelReply:
    content: str
    tool_calls: list[ModelToolCall]
    finish_reason: str | None = None


class AgentError(RuntimeError):
    """Base application error with a stable public code."""

    code = "AGENT_ERROR"

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class ConfigurationError(AgentError):
    code = "CONFIGURATION_ERROR"


class ModelConnectionError(AgentError):
    code = "MODEL_CONNECTION_ERROR"


class SessionNotFoundError(AgentError):
    code = "SESSION_NOT_FOUND"


def public_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, AgentError):
        return {
            "code": exc.code,
            "message": str(exc),
            "retryable": exc.retryable,
        }
    return {"code": "INTERNAL_ERROR", "message": str(exc), "retryable": False}
