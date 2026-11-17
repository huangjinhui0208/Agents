from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from codebase_understanding_agent.app.search_tools import SearchTools
from codebase_understanding_agent.app.terminal import TerminalBackend, TerminalError
from codebase_understanding_agent.app.workspace import WorkspaceGuard


class TerminalBackendTests(unittest.TestCase):
    def test_allowlist_blocks_arbitrary_shell_and_rg_powers_grep(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            repo.mkdir()
            (repo / "service.py").write_text(
                "def greet():\n    return 'hello'\n", encoding="utf-8"
            )
            backend = TerminalBackend(
                repo,
                allowed_programs=("rg",),
                timeout_s=2,
                max_stdout_bytes=100_000,
            )
            with self.assertRaises(TerminalError):
                backend.run_readonly(["sh", "-c", "echo unsafe"])

            tools = SearchTools(
                WorkspaceGuard(repo, repo_id="repo-test"),
                terminal_backend=backend,
            )
            result = tools.grep_code("greet", path=".", limit=10)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["data"]["backend"], "terminal:rg")
            self.assertEqual(result["data"]["matches"][0]["path"], "service.py")
            self.assertEqual(result["data"]["matches"][0]["line"], 1)


if __name__ == "__main__":
    unittest.main()
