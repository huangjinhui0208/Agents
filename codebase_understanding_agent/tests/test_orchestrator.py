from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from codebase_understanding_agent.app.contracts import ModelReply, ModelToolCall
from codebase_understanding_agent.app.orchestrator import CodebaseAgent
from codebase_understanding_agent.tests.helpers import make_config


class NativeToolFakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, tools=None, tool_choice="auto"):
        self.calls += 1
        if self.calls == 1:
            return ModelReply(
                content="",
                tool_calls=[
                    ModelToolCall(
                        "call_grep",
                        "grep_code",
                        {
                            "pattern": "greet",
                            "path": ".",
                            "match_mode": "literal",
                            "case_sensitive": True,
                            "limit": 10,
                        },
                    )
                ],
            )
        return ModelReply(
            content="`greet` 定义在 `service.py:L1-L2`，并返回 hello。",
            tool_calls=[],
        )


class NoToolFakeLLM:
    def complete(self, messages, *, tools=None, tool_choice="auto"):
        return ModelReply(content="我猜入口在 main.py。", tool_calls=[])


class OrchestratorTests(unittest.TestCase):
    def _repo(self, root: Path) -> Path:
        repo = root / "repo"
        repo.mkdir()
        (repo / "service.py").write_text(
            "def greet():\n    return 'hello'\n", encoding="utf-8"
        )
        (repo / "main.py").write_text(
            "from service import greet\nprint(greet())\n", encoding="utf-8"
        )
        return repo

    def test_native_tool_call_is_executed_and_cited(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            agent = CodebaseAgent(make_config(temp))
            agent.llm = NativeToolFakeLLM()  # type: ignore[assignment]
            selected = agent.select_repository(str(self._repo(temp)))
            result = agent.ask(selected["session_id"], "greet 做什么？")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["trace"][0]["tool"], "grep_code")
            self.assertEqual(result["trace"][0]["mode"], "native")
            self.assertTrue(any(item["path"] == "service.py" for item in result["citations"]))

    def test_missing_native_tool_call_is_not_replaced_by_program_search(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            agent = CodebaseAgent(make_config(temp))
            agent.llm = NoToolFakeLLM()  # type: ignore[assignment]
            selected = agent.select_repository(str(self._repo(temp)))
            result = agent.ask(selected["session_id"], "入口在哪里？")
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["trace"], [])
            self.assertIn("没有返回原生 Tool Call", result["answer"])


if __name__ == "__main__":
    unittest.main()
