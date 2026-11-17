from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from threading import Lock, RLock
from typing import Any

from .config import AgentConfig
from .context_builder import ContextBuilder
from .contracts import (
    SessionNotFoundError,
    TaskRequest,
    new_id,
)
from .embeddings import EmbeddingClient
from .llm_client import LocalLLMClient
from .lsp import LspTools
from .notes import NoteStore
from .registry import ToolDefinition, ToolRegistry
from .runtime import RuntimeMemory
from .search_tools import SearchTools
from .terminal import TerminalBackend
from .workspace import WorkspaceGuard


@dataclass
class RepositorySession:
    session_id: str
    workspace: WorkspaceGuard
    search: SearchTools
    lsp: LspTools
    registry: ToolRegistry
    lock: Lock = field(default_factory=Lock)


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        result["required"] = required
    return result


def _stable_repo_id(root: Path) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", root.name).strip("-").lower() or "repo"
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}"


class CodebaseAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        config.data_dir.mkdir(parents=True, exist_ok=True)
        self.runtime = RuntimeMemory(config.data_dir)
        self.notes = NoteStore(
            config.data_dir / "notes.sqlite",
            EmbeddingClient(
                provider=config.embedding_provider,
                model=config.embedding_model,
                base_url=config.embedding_base_url,
                api_key=config.embedding_api_key,
                dimensions=config.embedding_dimensions,
                timeout_s=config.request_timeout_s,
            ),
        )
        self.llm = LocalLLMClient(config)
        system_prompt = config.prompt_path.read_text(encoding="utf-8")
        self.context = ContextBuilder(config, system_prompt)
        self._sessions: dict[str, RepositorySession] = {}
        self._sessions_lock = RLock()

    def effective_tool_mode(self) -> str:
        return "native"

    def select_repository(self, path: str) -> dict[str, Any]:
        selected = Path(path).expanduser().resolve(strict=True)
        workspace = WorkspaceGuard(selected, repo_id=_stable_repo_id(selected))
        terminal = (
            TerminalBackend(
                workspace.root,
                allowed_programs=self.config.terminal_allowed_programs,
                timeout_s=self.config.terminal_timeout_s,
                max_stdout_bytes=self.config.terminal_max_stdout_bytes,
            )
            if self.config.terminal_enabled
            else None
        )
        search = SearchTools(
            workspace,
            max_page_size=max(self.config.max_files, self.config.max_matches),
            max_read_lines=self.config.max_file_lines,
            max_file_bytes=self.config.max_file_bytes,
            terminal_backend=terminal,
        )
        session_id = new_id("session")
        session = RepositorySession(
            session_id=session_id,
            workspace=workspace,
            search=search,
            lsp=LspTools(workspace, timeout_s=self.config.search_timeout_s),
            registry=ToolRegistry(),
        )
        session.registry = self._make_registry(session)
        with self._sessions_lock:
            self._sessions[session_id] = session

        count = 0
        truncated = False
        for count, _ in enumerate(workspace.iter_files(), start=1):
            if count >= 20_000:
                truncated = True
                break
        return {
            "session_id": session_id,
            "repository": {
                "path": str(workspace.root),
                "name": workspace.root.name,
                "repo_id": workspace.repo_id,
            },
            "revision": workspace.revision,
            "summary": {
                "file_count": count,
                "file_count_truncated": truncated,
                "skipped_count": None,
            },
        }

    def reset_session(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        return self.select_repository(str(session.workspace.root))

    def _get_session(self, session_id: str) -> RepositorySession:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError("会话不存在或服务已重启，请重新选择代码仓库。")
        return session

    @staticmethod
    def _tool_result(
        session: RepositorySession,
        tool_call_id: str,
        tool: str,
        *,
        data: dict[str, Any],
        status: str = "ok",
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "tool_call_id": tool_call_id,
            "tool": tool,
            "status": status,
            "data": data,
            "source": {
                "repo_id": session.workspace.repo_id,
                "revision": session.workspace.revision,
            },
            "truncated": False,
            "next_cursor": None,
            "error": error,
        }

    def _make_registry(self, session: RepositorySession) -> ToolRegistry:
        registry = ToolRegistry()
        string = {"type": "string"}
        integer = {"type": "integer"}
        boolean = {"type": "boolean"}

        registry.register(
            ToolDefinition(
                "glob_files",
                "按仓库相对 glob 模式寻找文件。只返回允许读取的相对路径。",
                _object_schema(
                    {
                        "pattern": {**string, "description": "例如 **/*.py 或 src/**/*service*"},
                        "root": {**string, "description": "仓库相对目录，默认 ."},
                        "limit": {**integer, "minimum": 1, "maximum": self.config.max_files},
                        "cursor": {**string, "description": "上一页返回的游标"},
                    },
                    ["pattern"],
                ),
                session.search.glob_files,
            )
        )
        registry.register(
            ToolDefinition(
                "find_files",
                "按文件名、扩展名或类型寻找文件和目录。",
                _object_schema(
                    {
                        "root": string,
                        "name_pattern": string,
                        "extensions": {"type": "array", "items": string},
                        "kind": {"type": "string", "enum": ["file", "directory", "any"]},
                        "limit": {**integer, "minimum": 1, "maximum": self.config.max_files},
                        "cursor": string,
                    }
                ),
                session.search.find_files,
            )
        )
        registry.register(
            ToolDefinition(
                "grep_code",
                "在仓库源码文本中搜索字面量或正则，返回真实路径、行号和文件哈希。",
                _object_schema(
                    {
                        "pattern": {**string, "description": "要搜索的源码文本或正则"},
                        "path": {**string, "description": "仓库相对目录，默认 ."},
                        "match_mode": {"type": "string", "enum": ["literal", "regex"]},
                        "file_glob": string,
                        "case_sensitive": boolean,
                        "limit": {**integer, "minimum": 1, "maximum": self.config.max_matches},
                        "cursor": string,
                    },
                    ["pattern"],
                ),
                session.search.grep_code,
            )
        )
        registry.register(
            ToolDefinition(
                "read_file",
                "读取单个仓库文件的有限行范围；保留原始 1-based 行号。",
                _object_schema(
                    {
                        "path": {**string, "description": "仓库相对文件路径"},
                        "start_line": {**integer, "minimum": 1},
                        "end_line": {**integer, "minimum": 1},
                        "max_lines": {
                            **integer,
                            "minimum": 1,
                            "maximum": self.config.max_file_lines,
                        },
                        "cursor": string,
                    },
                    ["path"],
                ),
                session.search.read_file,
            )
        )

        lsp_position = {
            "path": {**string, "description": "仓库相对源文件"},
            "line": {**integer, "description": "1-based 行号", "minimum": 1},
            "character": {**integer, "description": "0-based UTF-16 字符偏移", "minimum": 0},
        }
        registry.register(
            ToolDefinition(
                "lsp_definition",
                "通过可用语言服务器查找符号定义；不可用时返回错误，再退回 grep/read。",
                _object_schema(lsp_position, ["path", "line", "character"]),
                session.lsp.definition,
            )
        )
        registry.register(
            ToolDefinition(
                "lsp_references",
                "通过可用语言服务器查找语义引用。",
                _object_schema(
                    {**lsp_position, "include_declaration": boolean},
                    ["path", "line", "character"],
                ),
                session.lsp.references,
            )
        )
        registry.register(
            ToolDefinition(
                "lsp_document_symbols",
                "列出文件中的类、函数、方法等语义符号。",
                _object_schema({"path": string}, ["path"]),
                session.lsp.document_symbols,
            )
        )
        for name, description, handler in (
            (
                "lsp_call_hierarchy_incoming",
                "查找调用当前符号的上游符号。",
                session.lsp.call_hierarchy_incoming,
            ),
            (
                "lsp_call_hierarchy_outgoing",
                "查找当前符号调用的下游符号。",
                session.lsp.call_hierarchy_outgoing,
            ),
        ):
            registry.register(
                ToolDefinition(
                    name,
                    description,
                    _object_schema(lsp_position, ["path", "line", "character"]),
                    handler,
                )
            )

        def note_search(tool_call_id: str, query: str, limit: int = 5) -> dict[str, Any]:
            notes = self.notes.search(session.workspace.repo_id, query, limit)
            return self._tool_result(
                session, tool_call_id, "note_search", data={"notes": notes, "count": len(notes)}
            )

        def note_get(tool_call_id: str, note_id: str) -> dict[str, Any]:
            note = self.notes.get(session.workspace.repo_id, note_id, session.workspace)
            if note is None:
                return self._tool_result(
                    session,
                    tool_call_id,
                    "note_get",
                    data={},
                    status="error",
                    error={
                        "code": "NOTE_NOT_FOUND",
                        "message": f"笔记不存在：{note_id}",
                        "retryable": False,
                    },
                )
            return self._tool_result(session, tool_call_id, "note_get", data={"note": note})

        def note_upsert(tool_call_id: str, **arguments: Any) -> dict[str, Any]:
            saved = self.notes.upsert(
                repo_id=session.workspace.repo_id,
                workspace=session.workspace,
                **arguments,
            )
            return self._tool_result(session, tool_call_id, "note_upsert", data=saved)

        evidence_schema = _object_schema(
            {
                "path": string,
                "start_line": {**integer, "minimum": 1},
                "end_line": {**integer, "minimum": 1},
                "observation_event_id": string,
            },
            ["path", "start_line", "end_line"],
        )
        registry.register(
            ToolDefinition(
                "note_search",
                "检索当前仓库的跨会话知识笔记；笔记可能陈旧，仍需按问题核查源码。",
                _object_schema(
                    {"query": string, "limit": {**integer, "minimum": 1, "maximum": 20}},
                    ["query"],
                ),
                note_search,
            )
        )
        registry.register(
            ToolDefinition(
                "note_get",
                "读取当前仓库的一条完整知识笔记和证据状态。",
                _object_schema({"note_id": string}, ["note_id"]),
                note_get,
            )
        )
        registry.register(
            ToolDefinition(
                "note_upsert",
                "保存或更新一项可复用的重要代码知识；普通搜索结果不要写。",
                _object_schema(
                    {
                        "subject_key": string,
                        "title": string,
                        "kind": {
                            "type": "string",
                            "enum": ["fact", "architecture", "open_question"],
                        },
                        "content": string,
                        "evidence": {"type": "array", "items": evidence_schema},
                        "status": {"type": "string", "enum": ["confirmed", "hypothesis"]},
                        "tags": {"type": "array", "items": string},
                        "open_questions": {"type": "array", "items": string},
                        "related_note_ids": {"type": "array", "items": string},
                        "note_id": string,
                        "expected_version": integer,
                    },
                    ["subject_key", "title", "kind", "content", "evidence"],
                ),
                note_upsert,
            )
        )
        return registry

    @staticmethod
    def _tool_message(call_id: str, name: str, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": json.dumps(result, ensure_ascii=False),
        }

    @staticmethod
    def _trace_item(result: dict[str, Any], mode: str = "native") -> dict[str, Any]:
        data = result.get("data", {})
        count = data.get("count")
        if count is None:
            if isinstance(data.get("matches"), list):
                count = len(data["matches"])
            elif isinstance(data.get("files"), list):
                count = len(data["files"])
            elif isinstance(data.get("lines"), list):
                count = len(data["lines"])
        summary = f"{count} 项结果" if count is not None else "已完成"
        if result.get("truncated"):
            summary += "（结果已截断，可继续分页）"
        if result.get("error"):
            summary = result["error"].get("message", "工具失败")
        return {
            "tool": result.get("tool", "unknown"),
            "status": result.get("status", "error"),
            "summary": summary,
            "mode": mode,
        }

    @staticmethod
    def _collect_citations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        citations: dict[tuple[str, int, int, str], dict[str, Any]] = {}
        for result in results:
            data = result.get("data", {})
            source_revision = result.get("source", {}).get("revision", "unknown")
            for match in data.get("matches", []) if isinstance(data.get("matches"), list) else []:
                path = match.get("path")
                line = match.get("line")
                if path and isinstance(line, int):
                    revision = match.get("file_hash") or match.get("revision") or source_revision
                    key = (path, line, line, revision)
                    citations[key] = {
                        "path": path,
                        "start_line": line,
                        "end_line": line,
                        "revision": revision,
                    }
            path = data.get("path")
            lines = data.get("lines")
            if path and isinstance(lines, list) and lines:
                numbers = [item.get("number") for item in lines if isinstance(item.get("number"), int)]
                if numbers:
                    revision = data.get("file_hash") or source_revision
                    key = (path, min(numbers), max(numbers), revision)
                    citations[key] = {
                        "path": path,
                        "start_line": min(numbers),
                        "end_line": max(numbers),
                        "revision": revision,
                    }
            note = data.get("note")
            if isinstance(note, dict):
                for evidence in note.get("evidence", []):
                    path = evidence.get("path")
                    start = evidence.get("start_line")
                    end = evidence.get("end_line")
                    revision = evidence.get("revision", source_revision)
                    if path and isinstance(start, int) and isinstance(end, int):
                        key = (path, start, end, revision)
                        citations[key] = {
                            "path": path,
                            "start_line": start,
                            "end_line": end,
                            "revision": revision,
                        }
        return list(citations.values())[:80]

    def ask(self, session_id: str, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空。")
        session = self._get_session(session_id)
        request = TaskRequest(
            request_id=new_id("request"),
            session_id=session_id,
            repo_id=session.workspace.repo_id,
            user_text=question,
            repository_root=str(session.workspace.root),
        )
        trace: list[dict[str, Any]] = []
        observed_results: list[dict[str, Any]] = []

        with session.lock:
            self.runtime.append(
                session_id,
                request.request_id,
                "user",
                {"message": {"role": "user", "content": question}},
            )
            relevant_notes = self.notes.search(
                session.workspace.repo_id, question, self.config.max_relevant_notes
            )

            for step in range(self.config.max_steps):
                context = self.context.build(
                    request=request,
                    runtime=self.runtime,
                    relevant_notes=relevant_notes,
                    tool_schemas=session.registry.schemas(),
                )
                reply = self.llm.complete(
                    context.messages,
                    tools=session.registry.schemas(),
                    tool_choice="auto",
                )
                calls = reply.tool_calls

                if calls:
                    assistant_message = {
                        "role": "assistant",
                        "content": reply.content,
                        "tool_calls": [call.as_openai_dict() for call in calls],
                    }
                    self.runtime.append(
                        session_id,
                        request.request_id,
                        "assistant",
                        {"message": assistant_message, "decision_mode": "native"},
                    )
                    for call in calls:
                        result = session.registry.execute(call.name, call.arguments, call.id)
                        observed_results.append(result)
                        trace.append(self._trace_item(result, "native"))
                        message = self._tool_message(call.id, call.name, result)
                        self.runtime.append(
                            session_id,
                            request.request_id,
                            "tool",
                            {"message": message, "result": result},
                        )
                        if call.name.startswith("note_"):
                            relevant_notes = self.notes.search(
                                session.workspace.repo_id,
                                question,
                                self.config.max_relevant_notes,
                            )
                    continue

                if not observed_results:
                    answer = (
                        "当前模型没有返回原生 Tool Call，因此本轮未读取仓库源码。"
                        "请使用支持 OpenAI-compatible Function Calling 的模型，"
                        "或检查该模型的工具调用模板。"
                    )
                    self.runtime.append(
                        session_id,
                        request.request_id,
                        "assistant",
                        {
                            "message": {"role": "assistant", "content": answer},
                            "stop_reason": "NATIVE_TOOL_CALL_REQUIRED",
                        },
                    )
                    return {
                        "answer": answer,
                        "status": "partial",
                        "request_id": request.request_id,
                        "session_id": session_id,
                        "trace": trace,
                        "citations": [],
                    }

                answer = reply.content.strip()
                if not answer:
                    answer = "模型没有返回可显示的答案。请缩小问题范围后重试。"
                self.runtime.append(
                    session_id,
                    request.request_id,
                    "assistant",
                    {"message": {"role": "assistant", "content": answer}},
                )
                return {
                    "answer": answer,
                    "status": "completed",
                    "request_id": request.request_id,
                    "session_id": session_id,
                    "trace": trace,
                    "citations": self._collect_citations(observed_results),
                }

            # Technical stop: ask for a bounded synthesis with tools disabled.
            context = self.context.build(
                request=request,
                runtime=self.runtime,
                relevant_notes=relevant_notes,
                tool_schemas=[],
            )
            final_messages = context.messages + [
                {
                    "role": "system",
                    "content": (
                        "已达到本次工具步数上限。请只根据已有 Observation 输出部分结论，"
                        "明确列出已确认内容、源码引用和仍未解决的问题。"
                    ),
                }
            ]
            final_reply = self.llm.complete(final_messages, tools=None)
            answer = final_reply.content.strip() or "已达到工具步数上限，但模型未生成部分总结。"
            self.runtime.append(
                session_id,
                request.request_id,
                "assistant",
                {"message": {"role": "assistant", "content": answer}, "stop_reason": "STEP_LIMIT"},
            )
            return {
                "answer": answer,
                "status": "partial",
                "request_id": request.request_id,
                "session_id": session_id,
                "trace": trace,
                "citations": self._collect_citations(observed_results),
            }

