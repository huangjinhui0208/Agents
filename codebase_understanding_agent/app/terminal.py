"""Allowlisted, read-only subprocess backend for repository search tools."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import time
from typing import Sequence


class TerminalError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    returncode: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool
    truncated: bool


class TerminalBackend:
    """Execute only explicitly allowlisted programs without a shell.

    This backend is not registered as a model tool. Grep/Glob/Find may use it
    internally, so the model never receives arbitrary command execution rights.
    """

    def __init__(
        self,
        repository_root: Path,
        *,
        allowed_programs: Sequence[str] = ("rg",),
        timeout_s: float = 15.0,
        max_stdout_bytes: int = 8_000_000,
        max_stderr_bytes: int = 64_000,
    ) -> None:
        self.repository_root = repository_root.resolve(strict=True)
        self.allowed_programs = frozenset(allowed_programs)
        self.timeout_s = float(timeout_s)
        self.max_stdout_bytes = int(max_stdout_bytes)
        self.max_stderr_bytes = int(max_stderr_bytes)
        if not self.allowed_programs:
            raise ValueError("allowed_programs 不能为空")

    def available(self, program: str) -> bool:
        return program in self.allowed_programs and shutil.which(program) is not None

    def _resolve_cwd(self, cwd: Path | None) -> Path:
        selected = (cwd or self.repository_root).resolve(strict=True)
        try:
            selected.relative_to(self.repository_root)
        except ValueError as exc:
            raise TerminalError("命令工作目录不能越过仓库根目录。") from exc
        if not selected.is_dir():
            raise TerminalError("命令工作目录不是目录。")
        return selected

    def _validate_argv(self, argv: Sequence[str]) -> tuple[str, ...]:
        if not argv or any(not isinstance(item, str) or "\x00" in item for item in argv):
            raise TerminalError("argv 必须是非空字符串列表。")
        program = argv[0]
        if Path(program).name != program or program not in self.allowed_programs:
            raise TerminalError(f"程序不在只读白名单中：{program}")
        executable = shutil.which(program)
        if executable is None:
            raise TerminalError(f"找不到只读搜索程序：{program}")
        return (executable, *argv[1:])

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            process.terminate()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()

    def run_readonly(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout_s: float | None = None,
        max_stdout_bytes: int | None = None,
    ) -> ProcessResult:
        validated = self._validate_argv(argv)
        selected_cwd = self._resolve_cwd(cwd)
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        stdout_limit = (
            self.max_stdout_bytes
            if max_stdout_bytes is None
            else int(max_stdout_bytes)
        )
        if timeout <= 0 or stdout_limit < 1:
            raise TerminalError("timeout_s 和 max_stdout_bytes 必须大于 0。")

        environment = os.environ.copy()
        environment.update({"LC_ALL": "C", "NO_COLOR": "1", "RG_CONFIG_PATH": ""})
        try:
            process = subprocess.Popen(
                validated,
                cwd=str(selected_cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=True,
                env=environment,
            )
        except (OSError, ValueError) as exc:
            raise TerminalError(f"无法启动只读程序 {argv[0]}：{exc}") from exc

        assert process.stdout is not None and process.stderr is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        stdout = bytearray()
        stderr = bytearray()
        deadline = time.monotonic() + timeout
        timed_out = False
        truncated = False

        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    self._terminate(process)
                    break
                events = selector.select(timeout=min(remaining, 0.2))
                if not events and process.poll() is not None:
                    events = [(key, selectors.EVENT_READ) for key in selector.get_map().values()]
                for key, _ in events:
                    try:
                        chunk = os.read(key.fileobj.fileno(), 65_536)
                    except OSError:
                        chunk = b""
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    target = stdout if key.data == "stdout" else stderr
                    limit = stdout_limit if key.data == "stdout" else self.max_stderr_bytes
                    available = max(0, limit - len(target))
                    if available:
                        target.extend(chunk[:available])
                    if len(chunk) > available:
                        truncated = True
                        self._terminate(process)
                        break
                if truncated:
                    break
        finally:
            selector.close()
            if process.poll() is None:
                self._terminate(process)
            process.stdout.close()
            process.stderr.close()
        return ProcessResult(
            argv=tuple(argv),
            returncode=process.poll(),
            stdout=bytes(stdout),
            stderr=bytes(stderr),
            timed_out=timed_out,
            truncated=truncated,
        )


__all__ = ["ProcessResult", "TerminalBackend", "TerminalError"]
