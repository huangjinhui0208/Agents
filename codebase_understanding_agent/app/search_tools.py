"""Safe, read-only repository discovery and source inspection tools."""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import re
import signal
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

from .terminal import TerminalBackend, TerminalError
from .workspace import WorkspaceError, WorkspaceGuard


SCHEMA_VERSION = "1.0"
ToolResult = dict[str, Any]


def _fingerprint(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _encode_cursor(
    tool: str,
    offset: int,
    signature: str,
    *,
    state: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "v": 1,
        "tool": tool,
        "offset": offset,
        "signature": signature,
    }
    if state:
        payload["state"] = state
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(tool: str, cursor: str | None, signature: str) -> dict[str, Any]:
    if cursor is None:
        return {"offset": 0, "state": {}}
    if not isinstance(cursor, str) or not cursor or len(cursor) > 4096:
        raise WorkspaceError("INVALID_CURSOR", "分页 cursor 无效")
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.b64decode(padded, altchars=b"-_", validate=True)
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkspaceError("INVALID_CURSOR", "分页 cursor 无法解析") from exc
    if not isinstance(payload, dict):
        raise WorkspaceError("INVALID_CURSOR", "分页 cursor 内容无效")
    offset = payload.get("offset")
    if (
        payload.get("v") != 1
        or payload.get("tool") != tool
        or payload.get("signature") != signature
        or isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
    ):
        raise WorkspaceError("INVALID_CURSOR", "分页 cursor 与本次查询不匹配")
    state = payload.get("state", {})
    if not isinstance(state, dict):
        raise WorkspaceError("INVALID_CURSOR", "分页 cursor 状态无效")
    return {"offset": offset, "state": state}


class SearchTools:
    """Repository-scoped, read-only Glob/Find/Grep/Read operations."""

    def __init__(
        self,
        workspace: WorkspaceGuard | str | Path,
        *,
        max_page_size: int = 500,
        max_read_lines: int = 500,
        max_file_bytes: int = 2 * 1024 * 1024,
        max_line_chars: int = 20_000,
        max_scan_files: int = 100_000,
        terminal_backend: TerminalBackend | None = None,
    ) -> None:
        self.workspace = (
            workspace if isinstance(workspace, WorkspaceGuard) else WorkspaceGuard(workspace)
        )
        self.max_page_size = max_page_size
        self.max_read_lines = max_read_lines
        self.max_file_bytes = max_file_bytes
        self.max_line_chars = max_line_chars
        self.max_scan_files = max_scan_files
        self.terminal_backend = terminal_backend
        for name, value in {
            "max_page_size": max_page_size,
            "max_read_lines": max_read_lines,
            "max_file_bytes": max_file_bytes,
            "max_line_chars": max_line_chars,
            "max_scan_files": max_scan_files,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须是正整数")

    def _source(self, **extra: Any) -> dict[str, Any]:
        source: dict[str, Any] = {
            "repo_id": self.workspace.repo_id,
            "revision": self.workspace.revision,
        }
        source.update(extra)
        return source

    @staticmethod
    def _call_id(tool: str, tool_call_id: str | None) -> str:
        if tool_call_id is None:
            return f"{tool}_{uuid.uuid4().hex}"
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            return f"{tool}_invalid_call_id"
        return tool_call_id

    def _result(
        self,
        *,
        tool: str,
        tool_call_id: str | None,
        data: dict[str, Any],
        truncated: bool = False,
        next_cursor: str | None = None,
        source: dict[str, Any] | None = None,
    ) -> ToolResult:
        return {
            "schema_version": SCHEMA_VERSION,
            "tool_call_id": self._call_id(tool, tool_call_id),
            "tool": tool,
            "status": "partial" if truncated else "ok",
            "data": data,
            "source": source or self._source(),
            "truncated": truncated,
            "next_cursor": next_cursor,
            "error": None,
        }

    def _error(
        self,
        *,
        tool: str,
        tool_call_id: str | None,
        error: WorkspaceError,
    ) -> ToolResult:
        return {
            "schema_version": SCHEMA_VERSION,
            "tool_call_id": self._call_id(tool, tool_call_id),
            "tool": tool,
            "status": "error",
            "data": {},
            "source": self._source(),
            "truncated": False,
            "next_cursor": None,
            "error": {
                "code": error.code,
                "message": error.message,
                "retryable": error.retryable,
            },
        }

    def _limit(self, limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise WorkspaceError("INVALID_ARGUMENT", "limit 必须是整数")
        if limit < 1 or limit > self.max_page_size:
            raise WorkspaceError(
                "INVALID_ARGUMENT",
                f"limit 必须在 1 到 {self.max_page_size} 之间",
            )
        return limit

    @staticmethod
    def _validate_pattern(pattern: str, *, allow_empty: bool = False) -> str:
        if not isinstance(pattern, str) or (not allow_empty and not pattern):
            raise WorkspaceError("INVALID_PATTERN", "搜索模式必须是非空字符串")
        # Reuse path validation for absolute paths and traversal, while retaining
        # glob metacharacters as plain pattern text.
        return WorkspaceGuard._validate_tool_path(pattern).replace("\\", "/")

    @staticmethod
    def _glob_matches(relative_path: str, pattern: str) -> bool:
        normalized = relative_path.replace("\\", "/")
        if fnmatch.fnmatchcase(normalized, pattern):
            return True
        if pattern.startswith("**/") and fnmatch.fnmatchcase(normalized, pattern[3:]):
            return True
        if "/" not in pattern and fnmatch.fnmatchcase(Path(normalized).name, pattern):
            return True
        return False

    @staticmethod
    def _page(
        items: Sequence[Any],
        *,
        offset: int,
        limit: int,
        tool: str,
        signature: str,
    ) -> tuple[list[Any], bool, str | None]:
        page = list(items[offset : offset + limit])
        truncated = offset + len(page) < len(items)
        next_cursor = (
            _encode_cursor(tool, offset + len(page), signature) if truncated else None
        )
        return page, truncated, next_cursor

    def _safe_entries(
        self,
        root: str,
        *,
        include_files: bool,
        include_directories: bool,
    ) -> tuple[list[tuple[str, Path, str]], bool]:
        records: list[tuple[str, Path, str]] = []
        scan_limited = False
        if include_files:
            for file_path in self.workspace.iter_files(root):
                records.append((self.workspace.relative(file_path), file_path, "file"))
                if len(records) >= self.max_scan_files:
                    scan_limited = True
                    break
        if include_directories and not scan_limited:
            for directory in self.workspace.iter_directories(root):
                records.append(
                    (self.workspace.relative(directory), directory, "directory")
                )
                if len(records) >= self.max_scan_files:
                    scan_limited = True
                    break
        records.sort(key=lambda item: (item[0], item[2]))
        return records, scan_limited

    def glob_files(
        self,
        pattern: str,
        root: str = ".",
        limit: int = 100,
        cursor: str | None = None,
        *,
        tool_call_id: str | None = None,
    ) -> ToolResult:
        """Discover allowed files using a repository-relative glob pattern."""

        tool = "glob_files"
        try:
            page_limit = self._limit(limit)
            normalized_pattern = self._validate_pattern(pattern)
            search_root = self.workspace.resolve_relative(root, require_dir=True)
            normalized_root = self.workspace.relative(search_root)
            signature = _fingerprint(
                {"pattern": normalized_pattern, "root": normalized_root}
            )
            cursor_data = _decode_cursor(tool, cursor, signature)
            entries, scan_limited = self._safe_entries(
                normalized_root, include_files=True, include_directories=False
            )
            matches: list[str] = []
            for relative, path, _ in entries:
                path_under_root = path.relative_to(search_root).as_posix()
                if self._glob_matches(path_under_root, normalized_pattern):
                    matches.append(relative)
            page, truncated, next_cursor = self._page(
                matches,
                offset=cursor_data["offset"],
                limit=page_limit,
                tool=tool,
                signature=signature,
            )
            truncated = truncated or scan_limited
            return self._result(
                tool=tool,
                tool_call_id=tool_call_id,
                data={
                    "files": page,
                    "count": len(page),
                    "total_matches": None if scan_limited else len(matches),
                    "scan_limited": scan_limited,
                },
                truncated=truncated,
                next_cursor=next_cursor,
            )
        except WorkspaceError as exc:
            return self._error(tool=tool, tool_call_id=tool_call_id, error=exc)
        except (OSError, ValueError) as exc:
            return self._error(
                tool=tool,
                tool_call_id=tool_call_id,
                error=WorkspaceError("SEARCH_ERROR", f"文件匹配失败: {exc}"),
            )

    def find_files(
        self,
        root: str = ".",
        name_pattern: str | None = None,
        extensions: Sequence[str] | None = None,
        kind: Literal["file", "directory", "any"] = "file",
        limit: int = 100,
        cursor: str | None = None,
        *,
        tool_call_id: str | None = None,
    ) -> ToolResult:
        """Find entries by basename, extension and kind."""

        tool = "find_files"
        try:
            page_limit = self._limit(limit)
            if kind not in {"file", "directory", "any"}:
                raise WorkspaceError(
                    "INVALID_ARGUMENT", "kind 必须是 file、directory 或 any"
                )
            normalized_name_pattern = None
            if name_pattern is not None:
                normalized_name_pattern = self._validate_pattern(name_pattern)
                if "/" in normalized_name_pattern:
                    raise WorkspaceError(
                        "INVALID_PATTERN", "name_pattern 只能匹配名称，不能包含目录"
                    )
            normalized_extensions: tuple[str, ...] | None = None
            if extensions is not None:
                if isinstance(extensions, (str, bytes)):
                    raise WorkspaceError(
                        "INVALID_ARGUMENT", "extensions 必须是字符串列表"
                    )
                clean_extensions: list[str] = []
                for extension in extensions:
                    if not isinstance(extension, str) or not extension:
                        raise WorkspaceError(
                            "INVALID_ARGUMENT", "扩展名必须是非空字符串"
                        )
                    if "/" in extension or "\\" in extension:
                        raise WorkspaceError(
                            "INVALID_ARGUMENT", "扩展名不能包含路径分隔符"
                        )
                    clean_extensions.append(
                        (extension if extension.startswith(".") else f".{extension}").lower()
                    )
                normalized_extensions = tuple(sorted(set(clean_extensions)))

            search_root = self.workspace.resolve_relative(root, require_dir=True)
            normalized_root = self.workspace.relative(search_root)
            signature = _fingerprint(
                {
                    "root": normalized_root,
                    "name_pattern": normalized_name_pattern,
                    "extensions": normalized_extensions,
                    "kind": kind,
                }
            )
            cursor_data = _decode_cursor(tool, cursor, signature)
            entries, scan_limited = self._safe_entries(
                normalized_root,
                include_files=kind in {"file", "any"},
                include_directories=kind in {"directory", "any"},
            )

            matches: list[dict[str, Any]] = []
            for relative, path, entry_kind in entries:
                if normalized_name_pattern and not fnmatch.fnmatchcase(
                    path.name, normalized_name_pattern
                ):
                    continue
                if (
                    normalized_extensions is not None
                    and entry_kind == "file"
                    and path.suffix.lower() not in normalized_extensions
                ):
                    continue
                if normalized_extensions is not None and entry_kind == "directory":
                    continue
                record: dict[str, Any] = {"path": relative, "kind": entry_kind}
                if entry_kind == "file":
                    try:
                        record["size"] = path.stat().st_size
                    except OSError:
                        continue
                matches.append(record)

            page, truncated, next_cursor = self._page(
                matches,
                offset=cursor_data["offset"],
                limit=page_limit,
                tool=tool,
                signature=signature,
            )
            truncated = truncated or scan_limited
            return self._result(
                tool=tool,
                tool_call_id=tool_call_id,
                data={
                    "files": page,
                    "count": len(page),
                    "total_matches": None if scan_limited else len(matches),
                    "scan_limited": scan_limited,
                },
                truncated=truncated,
                next_cursor=next_cursor,
            )
        except WorkspaceError as exc:
            return self._error(tool=tool, tool_call_id=tool_call_id, error=exc)
        except (OSError, ValueError) as exc:
            return self._error(
                tool=tool,
                tool_call_id=tool_call_id,
                error=WorkspaceError("SEARCH_ERROR", f"文件查找失败: {exc}"),
            )

    def _read_text(self, path: Path) -> tuple[str, str, str]:
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise WorkspaceError("READ_ERROR", f"无法读取文件信息: {path.name}") from exc
        if size > self.max_file_bytes:
            raise WorkspaceError(
                "TOO_LARGE",
                f"文件超过 {self.max_file_bytes} 字节读取上限: {path.name}",
            )
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise WorkspaceError("READ_ERROR", f"无法读取文件: {path.name}") from exc
        if b"\x00" in raw[:8192]:
            raise WorkspaceError("BINARY_FILE", f"拒绝读取二进制文件: {path.name}")
        if re.search(
            br"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", raw[:65536]
        ):
            raise WorkspaceError("SENSITIVE_FILE", f"拒绝读取私钥内容: {path.name}")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise WorkspaceError(
                "UNSUPPORTED_ENCODING", f"文件不是可安全解析的 UTF-8 文本: {path.name}"
            ) from exc
        return text, "utf-8", hashlib.sha256(raw).hexdigest()

    def _terminal_grep_matches(
        self,
        *,
        pattern: str,
        path: str,
        match_mode: Literal["literal", "regex"],
        file_glob: str | None,
        case_sensitive: bool,
    ) -> tuple[list[dict[str, Any]], bool]:
        backend = self.terminal_backend
        if backend is None or not backend.available("rg"):
            raise TerminalError("rg 后端不可用")
        argv = ["rg", "--json", "--no-config", "--no-messages", "--line-number"]
        if match_mode == "literal":
            argv.append("--fixed-strings")
        if not case_sensitive:
            argv.append("--ignore-case")
        if file_glob:
            argv.extend(["--glob", file_glob])
        for directory in sorted(self.workspace.excluded_directories):
            argv.extend(["--glob", f"!{directory}/**"])
            argv.extend(["--glob", f"!**/{directory}/**"])
        for name in sorted(self.workspace.sensitive_names):
            argv.extend(["--glob", f"!**/{name}"])
        for suffix in sorted(self.workspace.sensitive_suffixes):
            argv.extend(["--glob", f"!**/*{suffix}"])
        argv.extend(["--", pattern, path])
        result = backend.run_readonly(argv, cwd=self.workspace.root)
        if result.timed_out:
            raise WorkspaceError("SEARCH_TIMEOUT", "rg 搜索超时，请缩小目录或文件类型")
        if result.returncode not in {0, 1, -signal.SIGTERM, -signal.SIGKILL} and not result.truncated:
            error = result.stderr.decode("utf-8", errors="replace")[:500]
            raise WorkspaceError("SEARCH_ERROR", f"rg 搜索失败: {error or result.returncode}")

        matches: list[dict[str, Any]] = []
        file_hashes: dict[str, str] = {}
        for raw_line in result.stdout.splitlines():
            try:
                payload = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if payload.get("type") != "match":
                continue
            data = payload.get("data", {})
            raw_path = data.get("path", {}).get("text")
            line_number = data.get("line_number")
            line_text = data.get("lines", {}).get("text")
            if not isinstance(raw_path, str) or not isinstance(line_number, int):
                continue
            try:
                file_path = self.workspace.resolve_relative(raw_path, require_file=True)
                relative = self.workspace.relative(file_path)
            except WorkspaceError:
                continue
            if relative not in file_hashes:
                file_hashes[relative] = self.workspace.file_hash(relative)
            visible_text = str(line_text or "").rstrip("\r\n")[: self.max_line_chars]
            matches.append(
                {
                    "path": relative,
                    "line": line_number,
                    "text": visible_text,
                    "text_truncated": len(visible_text) < len(str(line_text or "").rstrip("\r\n")),
                    "file_hash": file_hashes[relative],
                    "revision": self.workspace.revision,
                }
            )
        return matches, result.truncated

    def _iter_grep_matches(
        self,
        *,
        pattern: str,
        path: str,
        match_mode: Literal["literal", "regex"],
        file_glob: str | None,
        case_sensitive: bool,
        stats: dict[str, int],
    ) -> Iterator[dict[str, Any]]:
        if match_mode == "regex":
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags)
            except re.error as exc:
                raise WorkspaceError("INVALID_REGEX", f"正则表达式无效: {exc}") from exc
            matches_line = lambda value: regex.search(value) is not None
        else:
            needle = pattern if case_sensitive else pattern.casefold()

            def matches_line(value: str) -> bool:
                candidate = value if case_sensitive else value.casefold()
                return needle in candidate

        search_target = self.workspace.resolve_relative(path)
        if search_target.is_file():
            search_root = search_target.parent
            files: Iterable[Path] = (search_target,)
        elif search_target.is_dir():
            search_root = search_target
            files = self.workspace.iter_files(self.workspace.relative(search_target))
        else:
            raise WorkspaceError("NOT_FOUND", f"搜索路径不存在: {path}")
        file_count = 0
        for file_path in files:
            file_count += 1
            if file_count > self.max_scan_files:
                stats["scan_limited"] = 1
                return
            relative = self.workspace.relative(file_path)
            relative_under_root = file_path.relative_to(search_root).as_posix()
            if file_glob and not self._glob_matches(relative_under_root, file_glob):
                continue
            try:
                text, _, file_hash = self._read_text(file_path)
            except WorkspaceError:
                stats["skipped_files"] += 1
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not matches_line(line):
                    continue
                visible_text = line[: self.max_line_chars]
                text_truncated = len(visible_text) < len(line)
                if text_truncated:
                    stats["truncated_lines"] += 1
                yield {
                    "path": relative,
                    "line": line_number,
                    "text": visible_text,
                    "text_truncated": text_truncated,
                    "file_hash": file_hash,
                    "revision": self.workspace.revision,
                }

    def grep_code(
        self,
        pattern: str,
        path: str = ".",
        match_mode: Literal["literal", "regex"] = "literal",
        file_glob: str | None = None,
        case_sensitive: bool = True,
        limit: int = 50,
        cursor: str | None = None,
        *,
        tool_call_id: str | None = None,
    ) -> ToolResult:
        """Search source text using a safe Python fallback (no shell execution)."""

        tool = "grep_code"
        try:
            if not isinstance(pattern, str) or not pattern:
                raise WorkspaceError("INVALID_PATTERN", "pattern 必须是非空字符串")
            if match_mode not in {"literal", "regex"}:
                raise WorkspaceError(
                    "INVALID_ARGUMENT", "match_mode 必须是 literal 或 regex"
                )
            if not isinstance(case_sensitive, bool):
                raise WorkspaceError(
                    "INVALID_ARGUMENT", "case_sensitive 必须是布尔值"
                )
            normalized_glob = (
                self._validate_pattern(file_glob) if file_glob is not None else None
            )
            page_limit = self._limit(limit)
            search_target = self.workspace.resolve_relative(path)
            if not (search_target.is_file() or search_target.is_dir()):
                raise WorkspaceError("NOT_FOUND", f"搜索路径不存在: {path}")
            normalized_path = self.workspace.relative(search_target)
            signature = _fingerprint(
                {
                    "pattern": pattern,
                    "path": normalized_path,
                    "match_mode": match_mode,
                    "file_glob": normalized_glob,
                    "case_sensitive": case_sensitive,
                }
            )
            cursor_data = _decode_cursor(tool, cursor, signature)
            offset = cursor_data["offset"]
            if self.terminal_backend is not None and self.terminal_backend.available("rg"):
                matches, process_truncated = self._terminal_grep_matches(
                    pattern=pattern,
                    path=normalized_path,
                    match_mode=match_mode,
                    file_glob=normalized_glob,
                    case_sensitive=case_sensitive,
                )
                page = matches[offset : offset + page_limit]
                has_more = offset + len(page) < len(matches) or process_truncated
                next_cursor = (
                    _encode_cursor(tool, offset + len(page), signature)
                    if has_more and page
                    else None
                )
                return self._result(
                    tool=tool,
                    tool_call_id=tool_call_id,
                    data={
                        "matches": page,
                        "count": len(page),
                        "total_matches": None if process_truncated else len(matches),
                        "skipped_files": 0,
                        "scan_limited": process_truncated,
                        "backend": "terminal:rg",
                    },
                    truncated=has_more,
                    next_cursor=next_cursor,
                )
            stats = {"skipped_files": 0, "truncated_lines": 0, "scan_limited": 0}
            iterator = self._iter_grep_matches(
                pattern=pattern,
                path=normalized_path,
                match_mode=match_mode,
                file_glob=normalized_glob,
                case_sensitive=case_sensitive,
                stats=stats,
            )
            seen = 0
            page: list[dict[str, Any]] = []
            has_more = False
            for match in iterator:
                if seen < offset:
                    seen += 1
                    continue
                if len(page) < page_limit:
                    page.append(match)
                    seen += 1
                    continue
                has_more = True
                break
            total_matches = None if has_more or stats["scan_limited"] else seen
            next_cursor = (
                _encode_cursor(tool, offset + len(page), signature)
                if has_more
                else None
            )
            truncated = bool(
                has_more or stats["scan_limited"] or stats["truncated_lines"]
            )
            return self._result(
                tool=tool,
                tool_call_id=tool_call_id,
                data={
                    "matches": page,
                    "count": len(page),
                    "total_matches": total_matches,
                    "skipped_files": stats["skipped_files"],
                    "scan_limited": bool(stats["scan_limited"]),
                    "backend": "python",
                },
                truncated=truncated,
                next_cursor=next_cursor,
            )
        except WorkspaceError as exc:
            return self._error(tool=tool, tool_call_id=tool_call_id, error=exc)
        except (OSError, ValueError) as exc:
            return self._error(
                tool=tool,
                tool_call_id=tool_call_id,
                error=WorkspaceError("SEARCH_ERROR", f"代码搜索失败: {exc}"),
            )

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        max_lines: int = 120,
        cursor: str | None = None,
        *,
        tool_call_id: str | None = None,
    ) -> ToolResult:
        """Read a bounded, line-numbered slice of one allowed text file."""

        tool = "read_file"
        try:
            if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
                raise WorkspaceError(
                    "INVALID_ARGUMENT", "start_line 必须是大于等于 1 的整数"
                )
            if end_line is not None and (
                isinstance(end_line, bool)
                or not isinstance(end_line, int)
                or end_line < start_line
            ):
                raise WorkspaceError(
                    "INVALID_ARGUMENT", "end_line 必须是不小于 start_line 的整数"
                )
            if (
                isinstance(max_lines, bool)
                or not isinstance(max_lines, int)
                or max_lines < 1
                or max_lines > self.max_read_lines
            ):
                raise WorkspaceError(
                    "INVALID_ARGUMENT",
                    f"max_lines 必须在 1 到 {self.max_read_lines} 之间",
                )

            file_path = self.workspace.resolve_relative(path, require_file=True)
            normalized_path = self.workspace.relative(file_path)
            signature = _fingerprint({"path": normalized_path, "end_line": end_line})
            cursor_data = _decode_cursor(tool, cursor, signature)
            text, encoding, file_hash = self._read_text(file_path)
            cursor_hash = cursor_data["state"].get("file_hash")
            if cursor_hash is not None and cursor_hash != file_hash:
                raise WorkspaceError(
                    "FILE_CHANGED", "文件内容在分页读取期间发生变化，请重新读取"
                )

            lines = text.splitlines()
            total_lines = len(lines)
            effective_start = cursor_data["offset"] + 1 if cursor else start_line
            if total_lines and effective_start > total_lines:
                raise WorkspaceError(
                    "OUT_OF_RANGE", f"start_line 超过文件总行数 {total_lines}"
                )
            requested_end = total_lines if end_line is None else min(end_line, total_lines)
            actual_end = min(requested_end, effective_start + max_lines - 1)
            selected_lines = lines[effective_start - 1 : actual_end]
            line_items: list[dict[str, Any]] = []
            content_truncated = False
            for number, line in enumerate(selected_lines, start=effective_start):
                visible_text = line[: self.max_line_chars]
                line_truncated = len(visible_text) < len(line)
                content_truncated = content_truncated or line_truncated
                line_items.append(
                    {
                        "number": number,
                        "text": visible_text,
                        "truncated": line_truncated,
                    }
                )
            has_more = actual_end < requested_end
            next_cursor = (
                _encode_cursor(
                    tool,
                    actual_end,
                    signature,
                    state={"file_hash": file_hash},
                )
                if has_more
                else None
            )
            truncated = has_more or content_truncated
            source = self._source(path=normalized_path, file_hash=file_hash)
            return self._result(
                tool=tool,
                tool_call_id=tool_call_id,
                data={
                    "path": normalized_path,
                    "lines": line_items,
                    "start_line": effective_start,
                    "end_line": actual_end if line_items else None,
                    "total_lines": total_lines,
                    "has_more": has_more,
                    "encoding": encoding,
                    "file_hash": file_hash,
                    "revision": self.workspace.revision,
                },
                source=source,
                truncated=truncated,
                next_cursor=next_cursor,
            )
        except WorkspaceError as exc:
            return self._error(tool=tool, tool_call_id=tool_call_id, error=exc)
        except (OSError, ValueError) as exc:
            return self._error(
                tool=tool,
                tool_call_id=tool_call_id,
                error=WorkspaceError("READ_ERROR", f"读取文件失败: {exc}"),
            )


__all__ = ["SCHEMA_VERSION", "SearchTools", "ToolResult"]
