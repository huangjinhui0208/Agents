from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
import webbrowser

from .config import AgentConfig
from .contracts import AgentError, ModelConnectionError, SessionNotFoundError, public_error
from .orchestrator import CodebaseAgent


MAX_REQUEST_BYTES = 1_000_000


class LocalAgentHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], config: AgentConfig):
        self.config = config
        self.agent = CodebaseAgent(config)
        super().__init__(address, LocalAgentHandler)


class LocalAgentHandler(BaseHTTPRequestHandler):
    server: LocalAgentHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {fmt % args}")

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
        )
        self.send_header("Cache-Control", "no-store")

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        static_root = self.server.config.static_dir.resolve()
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(static_root)
        except (OSError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not resolved.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = resolved.read_bytes()
        mime, _ = mimetypes.guess_type(resolved.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", (mime or "application/octet-stream") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _request_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Content-Length 无效。") from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("请求正文为空或过大。")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求必须是有效的 UTF-8 JSON。") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求 JSON 必须是对象。")
        return payload

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            config = self.server.config
            self._json(
                HTTPStatus.OK,
                {
                    "provider": config.provider,
                    "model": config.model,
                    "base_url": config.base_url,
                    "status": self.server.agent.llm.health_status(),
                    "tool_mode": self.server.agent.effective_tool_mode(),
                    "terminal_backend": (
                        "rg" if config.terminal_enabled else "disabled"
                    ),
                    "embedding_provider": config.embedding_provider,
                    "default_browse_path": str(Path.home()),
                    "limits": {
                        "max_steps": config.max_steps,
                        "max_file_lines": config.max_file_lines,
                    },
                },
            )
            return
        if parsed.path == "/api/fs":
            self._list_directories(parsed.query)
            return
        if parsed.path == "/api/health":
            self._json(HTTPStatus.OK, {"ok": True, "status": self.server.agent.llm.health_status()})
            return

        relative = unquote(parsed.path.lstrip("/")) or "index.html"
        if relative.startswith("static/"):
            relative = relative.removeprefix("static/")
        if relative in {"", "."}:
            relative = "index.html"
        self._file(self.server.config.static_dir / relative)

    def _list_directories(self, query: str) -> None:
        params = parse_qs(query)
        raw = params.get("path", [str(Path.home())])[0]
        try:
            directory = Path(raw).expanduser().resolve(strict=True)
            if not directory.is_dir():
                raise ValueError("所选路径不是目录。")
            entries: list[dict[str, Any]] = []
            for child in directory.iterdir():
                if child.name.startswith("."):
                    continue
                try:
                    if child.is_dir() and not child.is_symlink():
                        entries.append(
                            {
                                "name": child.name,
                                "path": str(child.resolve(strict=True)),
                                "is_dir": True,
                            }
                        )
                except OSError:
                    continue
            entries.sort(key=lambda item: item["name"].casefold())
            parent = directory.parent if directory.parent != directory else None
            self._json(
                HTTPStatus.OK,
                {
                    "path": str(directory),
                    "parent": str(parent) if parent else None,
                    "entries": entries[:400],
                    "truncated": len(entries) > 400,
                },
            )
        except (OSError, ValueError) as exc:
            self._json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"code": "INVALID_DIRECTORY", "message": str(exc)}},
            )

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        try:
            payload = self._request_json()
            if parsed.path == "/api/repository":
                path = payload.get("path")
                if not isinstance(path, str) or not path.strip():
                    raise ValueError("path 必须是非空目录路径。")
                self._json(HTTPStatus.OK, self.server.agent.select_repository(path))
                return
            if parsed.path == "/api/chat":
                session_id = payload.get("session_id")
                question = payload.get("question")
                if not isinstance(session_id, str) or not isinstance(question, str):
                    raise ValueError("session_id 和 question 必须是字符串。")
                self._json(HTTPStatus.OK, self.server.agent.ask(session_id, question))
                return
            if parsed.path == "/api/session/reset":
                session_id = payload.get("session_id")
                if not isinstance(session_id, str):
                    raise ValueError("session_id 必须是字符串。")
                result = self.server.agent.reset_session(session_id)
                self._json(HTTPStatus.OK, {"ok": True, **result})
                return
            self._json(
                HTTPStatus.NOT_FOUND,
                {"error": {"code": "NOT_FOUND", "message": "接口不存在。"}},
            )
        except SessionNotFoundError as exc:
            self._json(HTTPStatus.NOT_FOUND, {"error": public_error(exc)})
        except ModelConnectionError as exc:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": public_error(exc)})
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            self._json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"code": "INVALID_REQUEST", "message": str(exc)}},
            )
        except AgentError as exc:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": public_error(exc)})
        except Exception as exc:
            # Keep the browser response safe and concise. The local console receives
            # the exception type for diagnosis without exposing request contents.
            print(f"[web] internal error: {type(exc).__name__}: {exc}")
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"code": "INTERNAL_ERROR", "message": "服务处理请求失败，请查看终端日志。"}},
            )


def serve(config: AgentConfig, *, open_browser: bool = True) -> None:
    server = LocalAgentHTTPServer((config.host, config.port), config)
    url = f"http://{config.host}:{config.port}"
    print("代码库理解 Agent 已启动")
    print(f"界面：{url}")
    print(f"模型：{config.model} @ {config.base_url}")
    print("读取边界：只读；不会执行仓库代码或测试。按 Ctrl+C 停止。")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n正在停止服务…")
    finally:
        server.server_close()
