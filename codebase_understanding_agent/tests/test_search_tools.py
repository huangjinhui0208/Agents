from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from codebase_understanding_agent.app.search_tools import SearchTools
from codebase_understanding_agent.app.workspace import WorkspaceError, WorkspaceGuard


RESULT_FIELDS = {
    "schema_version",
    "tool_call_id",
    "tool",
    "status",
    "data",
    "source",
    "truncated",
    "next_cursor",
    "error",
}


class SearchToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        temp_root = Path(self.temporary_directory.name)
        self.repo = temp_root / "sample-repo"
        self.repo.mkdir()
        self.outside = temp_root / "outside.py"
        self.outside.write_text("outside = True\n", encoding="utf-8")

        source_directory = self.repo / "src"
        source_directory.mkdir()
        self.source = source_directory / "sample.py"
        self.source.write_text(
            "first = 1\n"
            "needle = 'alpha'\n"
            "third = 3\n"
            "other_needle = 'beta'\n"
            "fifth = 5\n",
            encoding="utf-8",
        )
        (source_directory / "another.py").write_text(
            "def another():\n    return 'ok'\n", encoding="utf-8"
        )
        (self.repo / "README.md").write_text("# Sample\n", encoding="utf-8")

        for directory_name in (
            ".git",
            ".venv",
            "node_modules",
            "build",
            "dist",
            "__pycache__",
            ".agent_data",
        ):
            excluded = self.repo / directory_name
            excluded.mkdir()
            (excluded / "hidden.py").write_text("needle = 'hidden'\n", encoding="utf-8")
        (self.repo / ".env").write_text("SECRET=value\n", encoding="utf-8")
        (self.repo / "server.pem").write_text("certificate\n", encoding="utf-8")
        (self.repo / "disguised.txt").write_text(
            "-----BEGIN PRIVATE KEY-----\nsecret\n", encoding="utf-8"
        )

        self.guard = WorkspaceGuard(self.repo, repo_id="sample")
        self.tools = SearchTools(self.guard)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def assert_result_contract(self, result: dict, tool: str, call_id: str) -> None:
        self.assertEqual(set(result), RESULT_FIELDS)
        self.assertEqual(result["schema_version"], "1.0")
        self.assertEqual(result["tool"], tool)
        self.assertEqual(result["tool_call_id"], call_id)
        self.assertEqual(result["source"]["repo_id"], "sample")
        self.assertIn(result["status"], {"ok", "partial", "error"})

    def test_workspace_rejects_absolute_parent_and_escaping_symlink(self) -> None:
        with self.assertRaises(WorkspaceError) as absolute_error:
            self.guard.resolve_relative(str(self.source), require_file=True)
        self.assertEqual(absolute_error.exception.code, "OUT_OF_SCOPE")

        with self.assertRaises(WorkspaceError) as parent_error:
            self.guard.resolve_relative("../outside.py", require_file=True)
        self.assertEqual(parent_error.exception.code, "OUT_OF_SCOPE")

        link = self.repo / "escape.py"
        try:
            link.symlink_to(self.outside)
        except (OSError, NotImplementedError):
            self.skipTest("当前平台不支持测试符号链接")
        result = self.tools.read_file("escape.py", tool_call_id="call_escape")
        self.assert_result_contract(result, "read_file", "call_escape")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "OUT_OF_SCOPE")

    def test_default_exclusions_and_sensitive_files(self) -> None:
        result = self.tools.glob_files(
            "**/*", limit=20, tool_call_id="call_glob_exclusions"
        )
        self.assert_result_contract(result, "glob_files", "call_glob_exclusions")
        self.assertEqual(result["status"], "ok")
        paths = set(result["data"]["files"])
        self.assertIn("README.md", paths)
        self.assertIn("src/sample.py", paths)
        self.assertNotIn(".env", paths)
        self.assertNotIn("server.pem", paths)
        for path in paths:
            self.assertFalse(
                any(
                    component in path.split("/")
                    for component in (
                        ".git",
                        ".venv",
                        "node_modules",
                        "build",
                        "dist",
                        "__pycache__",
                        ".agent_data",
                    )
                )
            )

        private_key_result = self.tools.read_file(
            "disguised.txt", tool_call_id="call_private_key"
        )
        self.assertEqual(private_key_result["status"], "error")
        self.assertEqual(private_key_result["error"]["code"], "SENSITIVE_FILE")

    def test_grep_and_read_preserve_lines_hash_and_revision(self) -> None:
        expected_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        grep_result = self.tools.grep_code(
            "needle",
            path="src",
            file_glob="*.py",
            tool_call_id="call_grep",
        )
        self.assert_result_contract(grep_result, "grep_code", "call_grep")
        self.assertEqual(grep_result["status"], "ok")
        matches = grep_result["data"]["matches"]
        self.assertEqual([match["line"] for match in matches], [2, 4])
        self.assertTrue(all(match["path"] == "src/sample.py" for match in matches))
        self.assertTrue(all(match["file_hash"] == expected_hash for match in matches))
        self.assertTrue(
            all(match["revision"] == self.guard.revision for match in matches)
        )

        read_result = self.tools.read_file(
            "src/sample.py",
            start_line=2,
            end_line=3,
            tool_call_id="call_read",
        )
        self.assert_result_contract(read_result, "read_file", "call_read")
        self.assertEqual(
            [item["number"] for item in read_result["data"]["lines"]], [2, 3]
        )
        self.assertEqual(read_result["data"]["lines"][0]["text"], "needle = 'alpha'")
        self.assertEqual(read_result["data"]["file_hash"], expected_hash)
        self.assertEqual(read_result["source"]["file_hash"], expected_hash)
        self.assertEqual(read_result["data"]["revision"], self.guard.revision)

    def test_pagination_and_truncation_are_explicit(self) -> None:
        first_glob = self.tools.glob_files(
            "**/*.py", limit=1, tool_call_id="call_glob_page_1"
        )
        self.assertEqual(first_glob["status"], "partial")
        self.assertTrue(first_glob["truncated"])
        self.assertIsNotNone(first_glob["next_cursor"])
        second_glob = self.tools.glob_files(
            "**/*.py",
            limit=1,
            cursor=first_glob["next_cursor"],
            tool_call_id="call_glob_page_2",
        )
        self.assertNotEqual(
            first_glob["data"]["files"], second_glob["data"]["files"]
        )

        first_grep = self.tools.grep_code(
            "needle", path="src", limit=1, tool_call_id="call_grep_page_1"
        )
        self.assertTrue(first_grep["truncated"])
        self.assertIsNotNone(first_grep["next_cursor"])
        second_grep = self.tools.grep_code(
            "needle",
            path="src",
            limit=1,
            cursor=first_grep["next_cursor"],
            tool_call_id="call_grep_page_2",
        )
        self.assertEqual(first_grep["data"]["matches"][0]["line"], 2)
        self.assertEqual(second_grep["data"]["matches"][0]["line"], 4)

        first_read = self.tools.read_file(
            "src/sample.py", max_lines=2, tool_call_id="call_read_page_1"
        )
        self.assertTrue(first_read["truncated"])
        self.assertTrue(first_read["data"]["has_more"])
        self.assertIsNotNone(first_read["next_cursor"])
        second_read = self.tools.read_file(
            "src/sample.py",
            max_lines=2,
            cursor=first_read["next_cursor"],
            tool_call_id="call_read_page_2",
        )
        self.assertEqual(
            [item["number"] for item in second_read["data"]["lines"]], [3, 4]
        )

    def test_invalid_cursor_and_absolute_tool_path_return_errors(self) -> None:
        bad_cursor = self.tools.find_files(
            cursor="not-a-cursor", tool_call_id="call_bad_cursor"
        )
        self.assert_result_contract(bad_cursor, "find_files", "call_bad_cursor")
        self.assertEqual(bad_cursor["status"], "error")
        self.assertEqual(bad_cursor["error"]["code"], "INVALID_CURSOR")

        absolute = self.tools.read_file(
            str(self.source), tool_call_id="call_absolute"
        )
        self.assertEqual(absolute["status"], "error")
        self.assertEqual(absolute["error"]["code"], "OUT_OF_SCOPE")

    def test_tools_are_read_only(self) -> None:
        def snapshot() -> dict[str, tuple[str, int]]:
            result: dict[str, tuple[str, int]] = {}
            for path in sorted(self.repo.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    relative = path.relative_to(self.repo).as_posix()
                    result[relative] = (
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                        path.stat().st_mtime_ns,
                    )
            return result

        before = snapshot()
        self.tools.glob_files("**/*", tool_call_id="readonly_glob")
        self.tools.find_files(extensions=["py"], tool_call_id="readonly_find")
        self.tools.grep_code("needle", tool_call_id="readonly_grep")
        self.tools.read_file("src/sample.py", tool_call_id="readonly_read")
        after = snapshot()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
