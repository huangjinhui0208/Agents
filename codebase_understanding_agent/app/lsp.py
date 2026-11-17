"""Optional, read-only Language Server Protocol tools.

The bridge intentionally keeps the LSP process behind a narrow interface.  It
never executes a shell command, validates every input and output path against
the selected workspace, and returns the same ToolResult envelope as the other
code-search tools.  A missing server or unsupported capability is a normal
tool error so the agent can fall back to grep/read instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import shlex
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import unquote, urlparse


_MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
_MAX_MESSAGE_BYTES = 16 * 1024 * 1024


class _LspBridgeError(RuntimeError):
    """Base class for errors that must become a ToolResult."""


class _LspTimeout(_LspBridgeError):
    pass


class _LspStreamClosed(_LspBridgeError):
    pass


class _LspResponseError(_LspBridgeError):
    def __init__(self, code: int | str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _PreparedDocument:
    absolute_path: Path
    relative_path: str
    uri: str
    language_id: str
    text: str


class _StdioLspClient:
    """Small synchronous JSON-RPC client for one LSP request lifecycle."""

    def __init__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> None:
        self.argv = list(argv)
        self.cwd = cwd
        self.timeout_s = timeout_s
        self.process: subprocess.Popen[bytes] | None = None
        self._messages: queue.Queue[dict[str, Any] | BaseException] = queue.Queue()
        self._write_lock = threading.Lock()
        self._next_id = 1
        self._initialized = False
        self._stderr_tail = bytearray()

    def __enter__(self) -> "_StdioLspClient":
        try:
            self.process = subprocess.Popen(
                self.argv,
                cwd=str(self.cwd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                bufsize=0,
            )
        except (OSError, ValueError) as exc:
            raise _LspStreamClosed(f"无法启动 LSP Server：{exc}") from exc

        threading.Thread(target=self._reader_loop, daemon=True).start()
        threading.Thread(target=self._stderr_loop, daemon=True).start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def initialize(self) -> Mapping[str, Any]:
        root_uri = self.cwd.as_uri()
        response = self.request(
            "initialize",
            {
                "processId": os.getpid(),
                "clientInfo": {
                    "name": "codebase-understanding-agent",
                    "version": "1.0",
                },
                "rootUri": root_uri,
                "workspaceFolders": [{"uri": root_uri, "name": self.cwd.name}],
                "capabilities": {
                    "workspace": {
                        "configuration": True,
                        "workspaceFolders": True,
                    },
                    "textDocument": {
                        "definition": {"linkSupport": True},
                        "references": {},
                        "documentSymbol": {
                            "hierarchicalDocumentSymbolSupport": True,
                        },
                        "callHierarchy": {},
                    },
                },
                "trace": "off",
            },
        )
        if not isinstance(response, Mapping):
            raise _LspStreamClosed("LSP initialize 返回了无效响应")
        capabilities = response.get("capabilities", {})
        if not isinstance(capabilities, Mapping):
            raise _LspStreamClosed("LSP initialize 缺少 capabilities")
        self.notify("initialized", {})
        self._initialized = True
        return capabilities

    def open_document(self, document: _PreparedDocument) -> None:
        self.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": document.uri,
                    "languageId": document.language_id,
                    "version": 1,
                    "text": document.text,
                }
            },
        )

    def request(
        self,
        method: str,
        params: Mapping[str, Any] | None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        request_id = self._next_id
        self._next_id += 1
        self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": dict(params or {}),
            }
        )
        deadline = time.monotonic() + (timeout_s or self.timeout_s)

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _LspTimeout(f"LSP 请求超时：{method}")
            try:
                message = self._messages.get(timeout=remaining)
            except queue.Empty as exc:
                raise _LspTimeout(f"LSP 请求超时：{method}") from exc

            if isinstance(message, BaseException):
                raise message
            if "method" in message:
                self._handle_server_message(message)
                continue
            if message.get("id") != request_id:
                # Requests are serialized, so an unrelated response cannot be
                # useful to this lifecycle. Ignore it rather than mispairing it.
                continue
            error = message.get("error")
            if isinstance(error, Mapping):
                raise _LspResponseError(
                    error.get("code", "LSP_ERROR"),
                    str(error.get("message", "LSP request failed")),
                )
            return message.get("result")

    def notify(self, method: str, params: Mapping[str, Any] | None) -> None:
        self._send(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": dict(params or {}),
            }
        )

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None and self._initialized:
            try:
                self.request("shutdown", {}, timeout_s=min(self.timeout_s, 0.5))
                self.notify("exit", {})
                process.wait(timeout=0.5)
            except Exception:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self.process = None

    def _send(self, message: Mapping[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None or process.poll() is not None:
            detail = self._stderr_tail.decode("utf-8", errors="replace").strip()
            suffix = f"；stderr: {detail}" if detail else ""
            raise _LspStreamClosed(f"LSP Server 未运行{suffix}")
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
        try:
            with self._write_lock:
                process.stdin.write(header)
                process.stdin.write(payload)
                process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise _LspStreamClosed(f"无法写入 LSP Server：{exc}") from exc

    def _handle_server_message(self, message: Mapping[str, Any]) -> None:
        # LSP servers may ask for configuration while an ordinary request is
        # pending. Return conservative values so initialization cannot deadlock.
        if "id" not in message:
            return
        method = message.get("method")
        params = message.get("params")
        if method == "workspace/configuration":
            items = params.get("items", []) if isinstance(params, Mapping) else []
            result: Any = [{} for _ in items]
        elif method == "workspace/workspaceFolders":
            result = [{"uri": self.cwd.as_uri(), "name": self.cwd.name}]
        elif method == "workspace/applyEdit":
            result = {"applied": False, "failureReason": "read-only client"}
        else:
            result = None
        self._send({"jsonrpc": "2.0", "id": message["id"], "result": result})

    def _reader_loop(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        try:
            while True:
                message = self._read_message(process.stdout)
                if message is None:
                    raise _LspStreamClosed("LSP Server 已关闭输出流")
                self._messages.put(message)
        except BaseException as exc:
            self._messages.put(exc)

    def _stderr_loop(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        try:
            while True:
                chunk = process.stderr.read(4096)
                if not chunk:
                    return
                self._stderr_tail.extend(chunk)
                if len(self._stderr_tail) > 65536:
                    del self._stderr_tail[:-65536]
        except OSError:
            return

    @staticmethod
    def _read_message(stream: Any) -> dict[str, Any] | None:
        content_length: int | None = None
        while True:
            line = stream.readline()
            if line == b"":
                return None
            if line in (b"\r\n", b"\n"):
                break
            try:
                name, value = line.decode("ascii").split(":", 1)
            except (UnicodeDecodeError, ValueError) as exc:
                raise _LspStreamClosed("LSP 返回了无效消息头") from exc
            if name.lower().strip() == "content-length":
                try:
                    content_length = int(value.strip())
                except ValueError as exc:
                    raise _LspStreamClosed("LSP Content-Length 无效") from exc

        if content_length is None or not 0 <= content_length <= _MAX_MESSAGE_BYTES:
            raise _LspStreamClosed("LSP 消息长度缺失或超出限制")
        body = stream.read(content_length)
        if len(body) != content_length:
            raise _LspStreamClosed("LSP 消息体不完整")
        try:
            message = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _LspStreamClosed("LSP 返回了无效 JSON") from exc
        if not isinstance(message, dict):
            raise _LspStreamClosed("LSP JSON-RPC 消息不是对象")
        return message


class LspTools:
    """Read-only LSP operations scoped to one :class:`WorkspaceGuard`."""

    _CAPABILITIES = {
        "lsp_definition": "definitionProvider",
        "lsp_references": "referencesProvider",
        "lsp_document_symbols": "documentSymbolProvider",
        "lsp_call_hierarchy_incoming": "callHierarchyProvider",
        "lsp_call_hierarchy_outgoing": "callHierarchyProvider",
    }

    def __init__(self, workspace: Any, timeout_s: float = 10.0) -> None:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ValueError("timeout_s 必须是数字")
        if timeout_s <= 0:
            raise ValueError("timeout_s 必须大于 0")
        self.workspace = workspace
        self.timeout_s = float(timeout_s)

    def definition(
        self, tool_call_id: str, path: str, line: int, character: int
    ) -> dict[str, Any]:
        return self._run_location_request(
            "lsp_definition",
            "textDocument/definition",
            "locations",
            tool_call_id,
            path,
            line,
            character,
        )

    def references(
        self,
        tool_call_id: str,
        path: str,
        line: int,
        character: int,
        include_declaration: bool = False,
    ) -> dict[str, Any]:
        extra = {"context": {"includeDeclaration": bool(include_declaration)}}
        return self._run_location_request(
            "lsp_references",
            "textDocument/references",
            "locations",
            tool_call_id,
            path,
            line,
            character,
            extra,
        )

    def document_symbols(
        self,
        tool_call_id: str,
        path: str,
        line: int = 1,
        character: int = 0,
    ) -> dict[str, Any]:
        tool = "lsp_document_symbols"

        def operation(
            client: _StdioLspClient,
            document: _PreparedDocument,
        ) -> dict[str, Any]:
            raw = client.request(
                "textDocument/documentSymbol",
                {"textDocument": {"uri": document.uri}},
            )
            symbols, omitted = self._normalise_symbols(raw or [], document)
            return {"symbols": symbols, "omitted_out_of_scope": omitted}

        return self._invoke(tool, tool_call_id, path, line, character, operation)

    def call_hierarchy_incoming(
        self, tool_call_id: str, path: str, line: int, character: int
    ) -> dict[str, Any]:
        return self._run_call_hierarchy(
            "lsp_call_hierarchy_incoming",
            "callHierarchy/incomingCalls",
            "incoming_calls",
            "from",
            tool_call_id,
            path,
            line,
            character,
        )

    def call_hierarchy_outgoing(
        self, tool_call_id: str, path: str, line: int, character: int
    ) -> dict[str, Any]:
        return self._run_call_hierarchy(
            "lsp_call_hierarchy_outgoing",
            "callHierarchy/outgoingCalls",
            "outgoing_calls",
            "to",
            tool_call_id,
            path,
            line,
            character,
        )

    def _run_location_request(
        self,
        tool: str,
        method: str,
        data_key: str,
        tool_call_id: str,
        path: str,
        line: int,
        character: int,
        extra_params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        def operation(
            client: _StdioLspClient,
            document: _PreparedDocument,
        ) -> dict[str, Any]:
            params: dict[str, Any] = {
                "textDocument": {"uri": document.uri},
                "position": {"line": line - 1, "character": character},
            }
            if extra_params:
                params.update(extra_params)
            raw = client.request(method, params)
            if raw is None:
                candidates: list[Any] = []
            elif isinstance(raw, list):
                candidates = raw
            else:
                candidates = [raw]
            locations, omitted = self._normalise_locations(candidates)
            return {data_key: locations, "omitted_out_of_scope": omitted}

        return self._invoke(tool, tool_call_id, path, line, character, operation)

    def _run_call_hierarchy(
        self,
        tool: str,
        method: str,
        data_key: str,
        item_key: str,
        tool_call_id: str,
        path: str,
        line: int,
        character: int,
    ) -> dict[str, Any]:
        def operation(
            client: _StdioLspClient,
            document: _PreparedDocument,
        ) -> dict[str, Any]:
            prepared = client.request(
                "textDocument/prepareCallHierarchy",
                {
                    "textDocument": {"uri": document.uri},
                    "position": {"line": line - 1, "character": character},
                },
            )
            if not prepared:
                return {data_key: [], "omitted_out_of_scope": 0}
            items = prepared if isinstance(prepared, list) else [prepared]
            calls: list[dict[str, Any]] = []
            omitted = 0
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                if self._normalise_call_item(item) is None:
                    omitted += 1
                    continue
                raw_calls = client.request(method, {"item": item}) or []
                if not isinstance(raw_calls, list):
                    continue
                for raw_call in raw_calls:
                    if not isinstance(raw_call, Mapping):
                        continue
                    normalised_item = self._normalise_call_item(raw_call.get(item_key))
                    if normalised_item is None:
                        omitted += 1
                        continue
                    ranges = self._normalise_ranges(raw_call.get("fromRanges", []))
                    calls.append({item_key: normalised_item, "from_ranges": ranges})
            return {data_key: calls, "omitted_out_of_scope": omitted}

        return self._invoke(tool, tool_call_id, path, line, character, operation)

    def _invoke(
        self,
        tool: str,
        tool_call_id: str,
        path: str,
        line: int,
        character: int,
        operation: Callable[[_StdioLspClient, _PreparedDocument], dict[str, Any]],
    ) -> dict[str, Any]:
        document, error = self._prepare_document(path, line, character)
        if error is not None:
            return self._error(tool, tool_call_id, error[0], error[1], False)
        assert document is not None

        command = self._detect_command(document.absolute_path)
        if command is None:
            return self._error(
                tool,
                tool_call_id,
                "NOT_INITIALIZED",
                f"未找到适用于 {document.absolute_path.suffix or '该文件'} 的 LSP Server；"
                "可安装对应语言服务器或设置 CODE_AGENT_LSP_COMMAND",
                False,
                document,
            )

        try:
            with _StdioLspClient(command, Path(self.workspace.root), self.timeout_s) as client:
                try:
                    capabilities = client.initialize()
                except (_LspBridgeError, OSError) as exc:
                    return self._error(
                        tool,
                        tool_call_id,
                        "NOT_INITIALIZED",
                        str(exc),
                        True,
                        document,
                    )
                capability = self._CAPABILITIES[tool]
                if not capabilities.get(capability):
                    return self._error(
                        tool,
                        tool_call_id,
                        "UNSUPPORTED_CAPABILITY",
                        f"LSP Server 未声明 {capability} 能力",
                        False,
                        document,
                    )
                client.open_document(document)
                data = operation(client, document)
                status = "partial" if data.get("omitted_out_of_scope") else "ok"
                return self._result(tool, tool_call_id, status, data, document=document)
        except _LspResponseError as exc:
            code = (
                "UNSUPPORTED_CAPABILITY"
                if exc.code in (-32601, "-32601")
                else "LSP_REQUEST_FAILED"
            )
            return self._error(
                tool, tool_call_id, code, str(exc), False, document
            )
        except _LspTimeout as exc:
            return self._error(
                tool, tool_call_id, "TIMEOUT", str(exc), True, document
            )
        except (_LspStreamClosed, OSError) as exc:
            return self._error(
                tool, tool_call_id, "NOT_INITIALIZED", str(exc), True, document
            )
        except Exception as exc:
            # A tool boundary must never bring down the agent loop. Keep the
            # message actionable but do not claim that an LSP result exists.
            return self._error(
                tool,
                tool_call_id,
                "LSP_REQUEST_FAILED",
                f"LSP 请求失败：{exc}",
                False,
                document,
            )

    def _prepare_document(
        self, path: str, line: int, character: int
    ) -> tuple[_PreparedDocument | None, tuple[str, str] | None]:
        if not isinstance(path, str) or not path.strip():
            return None, ("INVALID_ARGUMENT", "path 必须是非空字符串")
        if isinstance(line, bool) or not isinstance(line, int) or line < 1:
            return None, ("INVALID_ARGUMENT", "line 必须是从 1 开始的整数")
        if (
            isinstance(character, bool)
            or not isinstance(character, int)
            or character < 0
        ):
            return None, ("INVALID_ARGUMENT", "character 必须是从 0 开始的整数")
        try:
            absolute = self.workspace.resolve_relative(path, require_file=True)
            relative = self.workspace.relative(absolute)
        except Exception as exc:
            code = getattr(exc, "code", "OUT_OF_SCOPE")
            return None, (str(code), str(exc))
        try:
            size = absolute.stat().st_size
            if size > _MAX_DOCUMENT_BYTES:
                return None, (
                    "TOO_LARGE",
                    f"文件大小 {size} bytes，超过 LSP 读取限制 {_MAX_DOCUMENT_BYTES}",
                )
            raw = absolute.read_bytes()
            if b"\x00" in raw:
                return None, ("BINARY_FILE", "LSP 不读取二进制文件")
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None, ("UNSUPPORTED_ENCODING", "LSP 当前仅接受 UTF-8 文本")
        except OSError as exc:
            return None, ("NOT_FOUND", f"无法读取文件：{exc}")

        line_count = max(1, text.count("\n") + 1)
        if line > line_count:
            return None, (
                "INVALID_ARGUMENT",
                f"line={line} 超出文件总行数 {line_count}",
            )
        return (
            _PreparedDocument(
                absolute_path=absolute,
                relative_path=relative,
                uri=absolute.as_uri(),
                language_id=self._language_id(absolute),
                text=text,
            ),
            None,
        )

    def _detect_command(self, path: Path) -> list[str] | None:
        configured = os.environ.get("CODE_AGENT_LSP_COMMAND", "").strip()
        if configured:
            try:
                argv = shlex.split(configured, posix=os.name != "nt")
            except ValueError:
                return None
            if not argv or not self._command_exists(argv[0]):
                return None
            argv[0] = os.path.expanduser(argv[0])
            return argv

        suffix = path.suffix.lower()
        if suffix in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}:
            candidates = [("clangd", [])]
        elif suffix in {".py", ".pyi"}:
            candidates = [
                ("basedpyright-langserver", ["--stdio"]),
                ("pyright-langserver", ["--stdio"]),
                ("basedpyright", ["--stdio"]),
                ("pyright", ["--stdio"]),
            ]
        elif suffix in {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}:
            candidates = [("typescript-language-server", ["--stdio"])]
        elif suffix == ".rs":
            candidates = [("rust-analyzer", [])]
        else:
            candidates = []
        for executable, args in candidates:
            resolved = shutil.which(executable)
            if resolved:
                return [resolved, *args]
        return None

    @staticmethod
    def _command_exists(executable: str) -> bool:
        expanded = os.path.expanduser(executable)
        if os.path.isabs(expanded) or os.sep in expanded:
            candidate = Path(expanded)
            return candidate.is_file() and os.access(candidate, os.X_OK)
        return shutil.which(expanded) is not None

    def _normalise_locations(
        self, locations: Sequence[Any]
    ) -> tuple[list[dict[str, Any]], int]:
        normalised: list[dict[str, Any]] = []
        omitted = 0
        for location in locations:
            if not isinstance(location, Mapping):
                continue
            uri = location.get("uri") or location.get("targetUri")
            range_value = location.get("range") or location.get("targetSelectionRange")
            item = self._normalise_location(uri, range_value)
            if item is None:
                omitted += 1
            else:
                normalised.append(item)
        return normalised, omitted

    def _normalise_symbols(
        self,
        symbols: Any,
        document: _PreparedDocument | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        if not isinstance(symbols, list):
            return [], 0
        output: list[dict[str, Any]] = []
        omitted = 0
        for symbol in symbols:
            if not isinstance(symbol, Mapping):
                continue
            if isinstance(symbol.get("location"), Mapping):
                location = symbol["location"]
                normalised_location = self._normalise_location(
                    location.get("uri"), location.get("range")
                )
            else:
                # DocumentSymbol ranges belong to the requested document. The
                # path is injected by the server only for SymbolInformation.
                range_result = self._normalise_range(symbol.get("selectionRange"))
                normalised_location = (
                    {"path": document.relative_path, **range_result}
                    if document is not None and range_result is not None
                    else range_result
                )
            if normalised_location is None:
                omitted += 1
                continue
            item: dict[str, Any] = {
                "name": str(symbol.get("name", "")),
                "kind": symbol.get("kind"),
                "detail": symbol.get("detail"),
                "location": normalised_location,
            }
            children, child_omitted = self._normalise_symbols(
                symbol.get("children", []), document
            )
            if children:
                item["children"] = children
            omitted += child_omitted
            output.append(item)
        return output, omitted

    def _normalise_call_item(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        location = self._normalise_location(value.get("uri"), value.get("selectionRange"))
        if location is None:
            return None
        result: dict[str, Any] = {
            "name": str(value.get("name", "")),
            "kind": value.get("kind"),
            "detail": value.get("detail"),
            "location": location,
        }
        if "data" in value:
            result["data"] = value["data"]
        return result

    def _normalise_location(self, uri: Any, range_value: Any) -> dict[str, Any] | None:
        path = self._path_from_uri(uri)
        range_result = self._normalise_range(range_value)
        if path is None or range_result is None:
            return None
        return {"path": path, **range_result}

    def _path_from_uri(self, uri: Any) -> str | None:
        if not isinstance(uri, str):
            return None
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            return None
        candidate = Path(unquote(parsed.path))
        try:
            resolved = candidate.resolve(strict=True)
            # resolve_relative provides the canonical symlink and root check.
            relative_hint = resolved.relative_to(Path(self.workspace.root).resolve()).as_posix()
            checked = self.workspace.resolve_relative(relative_hint, require_file=True)
            return self.workspace.relative(checked)
        except (OSError, ValueError, RuntimeError):
            return None

    @staticmethod
    def _normalise_range(value: Any) -> dict[str, int] | None:
        if not isinstance(value, Mapping):
            return None
        start = value.get("start")
        end = value.get("end")
        if not isinstance(start, Mapping) or not isinstance(end, Mapping):
            return None
        try:
            start_line = int(start["line"]) + 1
            start_character = int(start["character"])
            end_line = int(end["line"]) + 1
            end_character = int(end["character"])
        except (KeyError, TypeError, ValueError):
            return None
        if min(start_line, end_line) < 1 or min(start_character, end_character) < 0:
            return None
        return {
            "line": start_line,
            "character": start_character,
            "end_line": end_line,
            "end_character": end_character,
        }

    def _normalise_ranges(self, values: Any) -> list[dict[str, int]]:
        if not isinstance(values, list):
            return []
        return [item for value in values if (item := self._normalise_range(value))]

    @staticmethod
    def _language_id(path: Path) -> str:
        suffix = path.suffix.lower()
        return {
            ".c": "c",
            ".h": "cpp",
            ".cc": "cpp",
            ".cpp": "cpp",
            ".cxx": "cpp",
            ".hh": "cpp",
            ".hpp": "cpp",
            ".hxx": "cpp",
            ".py": "python",
            ".pyi": "python",
            ".js": "javascript",
            ".jsx": "javascriptreact",
            ".mjs": "javascript",
            ".cjs": "javascript",
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".rs": "rust",
        }.get(suffix, "plaintext")

    def _source(self, document: _PreparedDocument | None) -> dict[str, Any]:
        source: dict[str, Any] = {
            "repo_id": str(getattr(self.workspace, "repo_id", "unknown")),
            "revision": str(getattr(self.workspace, "revision", "unknown")),
        }
        if document is not None and hasattr(self.workspace, "file_revision"):
            try:
                details = self.workspace.file_revision(document.relative_path)
                if isinstance(details, Mapping):
                    for key in ("revision", "file_hash"):
                        if details.get(key):
                            source[key] = str(details[key])
            except Exception:
                pass
        return source

    def _result(
        self,
        tool: str,
        tool_call_id: str,
        status: str,
        data: Mapping[str, Any] | None,
        *,
        document: _PreparedDocument | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "tool_call_id": tool_call_id,
            "tool": tool,
            "status": status,
            "data": dict(data or {}),
            "source": self._source(document),
            "truncated": False,
            "next_cursor": None,
            "error": dict(error) if error else None,
        }

    def _error(
        self,
        tool: str,
        tool_call_id: str,
        code: str,
        message: str,
        retryable: bool,
        document: _PreparedDocument | None = None,
    ) -> dict[str, Any]:
        return self._result(
            tool,
            tool_call_id,
            "error",
            {},
            document=document,
            error={"code": code, "message": message, "retryable": retryable},
        )
