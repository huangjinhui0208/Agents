from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from codebase_understanding_agent.app.context_builder import ContextBuilder
from codebase_understanding_agent.app.contracts import TaskRequest
from codebase_understanding_agent.app.notes import NoteStore
from codebase_understanding_agent.app.runtime import RuntimeMemory
from codebase_understanding_agent.app.workspace import WorkspaceGuard
from codebase_understanding_agent.tests.helpers import make_config


class ContextBuilderTests(unittest.TestCase):
    def test_build_keeps_complete_native_history_without_budget_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            config = make_config(temp)
            memory = RuntimeMemory(config.data_dir)
            session_id = "session_test"
            for index in range(12):
                request_id = f"request_{index}"
                memory.append(
                    session_id,
                    request_id,
                    "user",
                    {"message": {"role": "user", "content": f"问题 {index}"}},
                )
                call_id = f"call_{index}"
                memory.append(
                    session_id,
                    request_id,
                    "assistant",
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": "grep_code", "arguments": "{}"},
                                }
                            ],
                        }
                    },
                )
                memory.append(
                    session_id,
                    request_id,
                    "tool",
                    {
                        "message": {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": "grep_code",
                            "content": "源码证据" * 200,
                        }
                    },
                )

            request = TaskRequest(
                request_id="request_current",
                session_id=session_id,
                repo_id="repo",
                user_text="继续",
                repository_root=str(temp),
            )
            builder = ContextBuilder(config, "只读系统规则")
            result = builder.build(
                request=request,
                runtime=memory,
                relevant_notes=[],
                tool_schemas=[],
            )

            self.assertFalse(result.compacted)
            assistant_ids = {
                call["id"]
                for message in result.messages
                for call in message.get("tool_calls", [])
            }
            tool_ids = {
                message["tool_call_id"]
                for message in result.messages
                if message.get("role") == "tool"
            }
            self.assertEqual(assistant_ids, tool_ids)
            self.assertEqual(len(assistant_ids), 12)
            self.assertGreater(result.token_count, 0)
            self.assertEqual(memory.summary(session_id), "")


class NoteStoreTests(unittest.TestCase):
    def test_vector_search_evidence_versions_and_staleness(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            repo = temp / "repo"
            repo.mkdir()
            source = repo / "service.py"
            source.write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
            other = repo / "database.py"
            other.write_text("def connect():\n    return 'db'\n", encoding="utf-8")
            workspace = WorkspaceGuard(repo, repo_id="repo-test")
            store = NoteStore(temp / "notes.sqlite")

            with self.assertRaises(ValueError):
                store.upsert(
                    repo_id="repo-test",
                    workspace=workspace,
                    subject_key="service/greet",
                    title="greet 行为",
                    kind="fact",
                    content="返回 hello",
                    evidence=[],
                    status="confirmed",
                )

            created = store.upsert(
                repo_id="repo-test",
                workspace=workspace,
                subject_key="service/greeting",
                title="欢迎语生成逻辑",
                kind="fact",
                content="greet 生成 hello 欢迎语。",
                evidence=[{"path": "service.py", "start_line": 1, "end_line": 2}],
                status="confirmed",
                tags=["greeting", "welcome"],
            )
            store.upsert(
                repo_id="repo-test",
                workspace=workspace,
                subject_key="database/connection",
                title="数据库连接",
                kind="fact",
                content="connect 返回数据库连接标识。",
                evidence=[{"path": "database.py", "start_line": 1, "end_line": 2}],
            )
            results = store.search("repo-test", "欢迎 greeting 输出", limit=2)
            self.assertEqual(results[0]["note_id"], created["note_id"])
            self.assertIn("similarity", results[0])

            with closing(sqlite3.connect(temp / "notes.sqlite")) as connection:
                embedding_count = connection.execute(
                    "SELECT COUNT(*) FROM note_embeddings"
                ).fetchone()[0]
            self.assertEqual(embedding_count, 2)

            updated = store.upsert(
                repo_id="repo-test",
                workspace=workspace,
                subject_key="service/greeting",
                title="欢迎语生成逻辑",
                kind="fact",
                content="greet 返回 hello 字符串。",
                evidence=[{"path": "service.py", "start_line": 1, "end_line": 2}],
                expected_version=1,
            )
            self.assertEqual(updated["version"], 2)

            source.write_text("def greet():\n    return 'changed'\n", encoding="utf-8")
            note = store.get("repo-test", created["note_id"], workspace)
            self.assertIsNotNone(note)
            self.assertEqual(note["status"], "stale")


if __name__ == "__main__":
    unittest.main()
