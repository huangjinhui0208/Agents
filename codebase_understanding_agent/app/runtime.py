from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

from .contracts import RuntimeEvent, new_id, utc_now


def rough_token_estimate(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    # Chinese often approaches one token per character; using 2 chars/token is a
    # deliberately conservative language-agnostic estimate for persisted events.
    return max(1, (len(text) + 1) // 2)


class RuntimeMemory:
    """Append-only session history; context projection never mutates this log."""

    def __init__(self, data_dir: Path):
        self._sessions_dir = data_dir / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._events: dict[str, list[RuntimeEvent]] = {}
        self._summaries: dict[str, str] = {}
        self._lock = RLock()

    def append(
        self,
        session_id: str,
        request_id: str,
        role: str,
        payload: dict[str, Any],
        *,
        artifact_ref: str | None = None,
    ) -> RuntimeEvent:
        event = RuntimeEvent(
            event_id=new_id("event"),
            session_id=session_id,
            request_id=request_id,
            timestamp=utc_now(),
            role=role,  # type: ignore[arg-type]
            payload=payload,
            token_estimate=rough_token_estimate(payload),
            artifact_ref=artifact_ref,
        )
        with self._lock:
            self._events.setdefault(session_id, []).append(event)
            target = self._sessions_dir / f"{session_id}.jsonl"
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return event

    def events(self, session_id: str) -> list[RuntimeEvent]:
        with self._lock:
            return list(self._events.get(session_id, ()))

    def set_summary(self, session_id: str, summary: str) -> None:
        with self._lock:
            self._summaries[session_id] = summary

    def summary(self, session_id: str) -> str:
        with self._lock:
            return self._summaries.get(session_id, "")
