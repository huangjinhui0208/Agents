"""Read-only workspace boundary for repository inspection tools.

The guard accepts an absolute repository root from the host application, but every
path supplied by a tool caller must be relative to that root.  Resolving a path is
also the point where excluded directories, sensitive files, ``..`` segments and
escaping symlinks are rejected.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path, PureWindowsPath
from typing import Iterator, Sequence


DEFAULT_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "node_modules",
        "build",
        "dist",
        "__pycache__",
        ".agent_data",
    }
)

DEFAULT_SENSITIVE_NAMES = frozenset(
    {
        ".env",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "credentials.json",
        "service-account.json",
    }
)

DEFAULT_SENSITIVE_SUFFIXES = frozenset(
    {
        ".pem",
        ".key",
        ".crt",
        ".cer",
        ".der",
        ".p12",
        ".pfx",
        ".jks",
        ".keystore",
        ".pk8",
    }
)


class WorkspaceError(ValueError):
    """A caller-visible workspace validation error."""

    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class WorkspaceGuard:
    """Confine all read operations to one repository root.

    ``root`` is selected by the host UI and may be absolute.  In contrast,
    ``resolve_relative`` only accepts repository-relative paths because tool input
    is untrusted.  This class intentionally exposes no write operation.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        repo_id: str | None = None,
        *,
        excluded_directories: Sequence[str] | None = None,
        sensitive_names: Sequence[str] | None = None,
        sensitive_suffixes: Sequence[str] | None = None,
    ) -> None:
        selected_root = Path(root).expanduser()
        try:
            selected_root = selected_root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorkspaceError("NOT_FOUND", f"工作区不存在或无法访问: {root}") from exc
        if not selected_root.is_dir():
            raise WorkspaceError("NOT_A_DIRECTORY", f"工作区不是目录: {root}")

        self.root = selected_root
        self.repo_id = repo_id or selected_root.name or "repository"
        self.excluded_directories = frozenset(
            DEFAULT_EXCLUDED_DIRECTORIES | set(excluded_directories or ())
        )
        self.sensitive_names = frozenset(
            name.lower()
            for name in (DEFAULT_SENSITIVE_NAMES | set(sensitive_names or ()))
        )
        self.sensitive_suffixes = frozenset(
            suffix.lower()
            for suffix in (
                DEFAULT_SENSITIVE_SUFFIXES | set(sensitive_suffixes or ())
            )
        )
        self.revision = self._detect_revision()

    @staticmethod
    def _validate_tool_path(path: str | os.PathLike[str]) -> str:
        if not isinstance(path, (str, os.PathLike)):
            raise WorkspaceError("INVALID_PATH", "路径必须是字符串")
        text = os.fspath(path)
        if not isinstance(text, str):
            raise WorkspaceError("INVALID_PATH", "路径必须是文本字符串")
        if "\x00" in text:
            raise WorkspaceError("INVALID_PATH", "路径不能包含 NUL 字符")
        if text == "":
            text = "."

        # Path.is_absolute() follows the host platform.  PureWindowsPath also
        # rejects drive-letter and UNC paths when the app runs on macOS/Linux.
        if Path(text).is_absolute() or PureWindowsPath(text).is_absolute():
            raise WorkspaceError("OUT_OF_SCOPE", "工具路径必须是仓库内的相对路径")

        normalized = text.replace("\\", "/")
        if any(part == ".." for part in normalized.split("/")):
            raise WorkspaceError("OUT_OF_SCOPE", "工具路径不能包含 '..'")
        return text

    def _is_sensitive_name(self, name: str) -> bool:
        lowered = name.lower()
        if lowered in self.sensitive_names:
            return True
        if lowered.startswith(".env."):
            return True
        if any(lowered.endswith(suffix) for suffix in self.sensitive_suffixes):
            return True
        return bool(
            re.search(
                r"(?:^|[-_.])(private[-_.]?key|client[-_.]?secret)(?:$|[-_.])",
                lowered,
            )
        )

    def _check_allowed_relative(self, relative: Path) -> None:
        parts = relative.parts
        if any(part in self.excluded_directories for part in parts):
            raise WorkspaceError(
                "EXCLUDED_PATH", f"路径位于默认排除目录中: {relative.as_posix()}"
            )
        if parts and self._is_sensitive_name(parts[-1]):
            raise WorkspaceError(
                "SENSITIVE_FILE", f"拒绝读取敏感文件: {relative.as_posix()}"
            )

    def resolve_relative(
        self,
        path: str | os.PathLike[str] = ".",
        *,
        require_file: bool = False,
        require_dir: bool = False,
        allow_missing: bool = False,
    ) -> Path:
        """Resolve and validate an untrusted repository-relative path."""

        if require_file and require_dir:
            raise ValueError("require_file 与 require_dir 不能同时为 True")
        text = self._validate_tool_path(path)
        candidate = self.root / text
        try:
            resolved = candidate.resolve(strict=not allow_missing)
        except FileNotFoundError as exc:
            raise WorkspaceError("NOT_FOUND", f"路径不存在: {text}") from exc
        except (OSError, RuntimeError) as exc:
            raise WorkspaceError("INVALID_PATH", f"路径无法解析: {text}") from exc

        try:
            relative = resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError("OUT_OF_SCOPE", f"路径越过工作区边界: {text}") from exc

        self._check_allowed_relative(relative)
        if not allow_missing and not resolved.exists():
            raise WorkspaceError("NOT_FOUND", f"路径不存在: {text}")
        if require_file and not resolved.is_file():
            raise WorkspaceError("NOT_A_FILE", f"路径不是普通文件: {text}")
        if require_dir and not resolved.is_dir():
            raise WorkspaceError("NOT_A_DIRECTORY", f"路径不是目录: {text}")
        return resolved

    def relative(self, path: str | os.PathLike[str]) -> str:
        """Return a normalized POSIX path after re-checking workspace scope."""

        candidate = Path(path)
        if not candidate.is_absolute():
            return self.relative(self.resolve_relative(path))
        try:
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(self.root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise WorkspaceError("OUT_OF_SCOPE", f"路径不属于当前工作区: {path}") from exc
        self._check_allowed_relative(relative)
        value = relative.as_posix()
        return value if value else "."

    def iter_files(self, root: str | os.PathLike[str] = ".") -> Iterator[Path]:
        """Yield allowed files under ``root`` without following directory links."""

        start = self.resolve_relative(root, require_dir=True)
        for current, directory_names, file_names in os.walk(
            start, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            kept_directories: list[str] = []
            for name in sorted(directory_names):
                if name in self.excluded_directories:
                    continue
                directory = current_path / name
                # Never traverse a directory symlink.  This avoids duplicate trees
                # and makes escaping symlinks harmless during broad scans.
                if directory.is_symlink():
                    continue
                try:
                    relative = directory.resolve(strict=True).relative_to(self.root)
                    self._check_allowed_relative(relative)
                except (OSError, RuntimeError, ValueError, WorkspaceError):
                    continue
                kept_directories.append(name)
            directory_names[:] = kept_directories

            for name in sorted(file_names):
                if self._is_sensitive_name(name):
                    continue
                lexical_path = current_path / name
                # Direct reads may use an in-scope file symlink, but broad scans
                # skip file links to avoid duplicate evidence and lexical/real-path
                # ambiguity.  Escaping links are rejected by resolve_relative.
                if lexical_path.is_symlink():
                    continue
                try:
                    relative_input = lexical_path.relative_to(self.root).as_posix()
                    resolved = self.resolve_relative(relative_input, require_file=True)
                except (OSError, ValueError, WorkspaceError):
                    continue
                yield resolved

    def iter_directories(
        self, root: str | os.PathLike[str] = "."
    ) -> Iterator[Path]:
        """Yield allowed real directories without following directory links."""

        start = self.resolve_relative(root, require_dir=True)
        for current, directory_names, _ in os.walk(
            start, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            kept_directories: list[str] = []
            for name in sorted(directory_names):
                if name in self.excluded_directories:
                    continue
                directory = current_path / name
                if directory.is_symlink():
                    continue
                try:
                    relative = directory.resolve(strict=True).relative_to(self.root)
                    self._check_allowed_relative(relative)
                except (OSError, RuntimeError, ValueError, WorkspaceError):
                    continue
                kept_directories.append(name)
                yield directory.resolve(strict=True)
            directory_names[:] = kept_directories

    def file_hash(self, path: str | os.PathLike[str]) -> str:
        """Return the SHA-256 hash of an allowed file without modifying it."""

        file_path = self.resolve_relative(path, require_file=True)

        digest = hashlib.sha256()
        try:
            with file_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise WorkspaceError("READ_ERROR", f"无法读取文件: {path}") from exc
        return digest.hexdigest()

    def file_revision(self, path: str | os.PathLike[str]) -> dict[str, str]:
        """Return both repository revision and content hash for exact evidence."""

        return {"revision": self.revision, "file_hash": self.file_hash(path)}

    def summary(self) -> dict[str, object]:
        """Return non-secret workspace metadata for tool result sources."""

        return {
            "root": str(self.root),
            "repo_id": self.repo_id,
            "revision": self.revision,
            "excluded_directories": sorted(self.excluded_directories),
        }

    def _detect_revision(self) -> str:
        """Read Git HEAD when available; file hashes cover uncommitted content."""

        environment = os.environ.copy()
        environment["GIT_OPTIONAL_LOCKS"] = "0"
        environment["LC_ALL"] = "C"
        try:
            completed = subprocess.run(
                ["git", "-C", str(self.root), "rev-parse", "--verify", "HEAD"],
                shell=False,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError):
            return "worktree"
        revision = completed.stdout.strip()
        if completed.returncode == 0 and re.fullmatch(r"[0-9a-fA-F]{7,64}", revision):
            return revision.lower()
        return "worktree"


__all__ = [
    "DEFAULT_EXCLUDED_DIRECTORIES",
    "DEFAULT_SENSITIVE_NAMES",
    "DEFAULT_SENSITIVE_SUFFIXES",
    "WorkspaceError",
    "WorkspaceGuard",
]
