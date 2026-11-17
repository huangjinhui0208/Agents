from __future__ import annotations

import os
from pathlib import Path
import shlex
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from codebase_understanding_agent.app.lsp import LspTools
from codebase_understanding_agent.app.workspace import WorkspaceGuard


_FAKE_LSP_SERVER = r'''
import json
import os
import sys


def read_message():
    length = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        name, value = line.decode("ascii").split(":", 1)
        if name.lower().strip() == "content-length":
            length = int(value.strip())
    if length is None:
        return None
    return json.loads(sys.stdin.buffer.read(length).decode("utf-8"))


def send(message):
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(
        ("Content-Length: %d\r\n\r\n" % len(payload)).encode("ascii")
    )
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def rng(line, start=1, end=7):
    return {
        "start": {"line": line, "character": start},
        "end": {"line": line, "character": end},
    }


while True:
    message = read_message()
    if message is None:
        break
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        unsupported = os.environ.get("FAKE_LSP_UNSUPPORTED") == "1"
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "capabilities": {
                    "definitionProvider": True,
                    "referencesProvider": True,
                    "documentSymbolProvider": True,
                    "callHierarchyProvider": not unsupported,
                }
            },
        })
    elif method in ("initialized", "textDocument/didOpen"):
        continue
    elif method == "textDocument/definition":
        uri = os.environ.get("FAKE_LSP_RESULT_URI") or params["textDocument"]["uri"]
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": [{"uri": uri, "range": rng(2, 3, 8)}],
        })
    elif method == "textDocument/references":
        uri = params["textDocument"]["uri"]
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": [
                {"uri": uri, "range": rng(0)},
                {"uri": uri, "range": rng(3)},
            ],
        })
    elif method == "textDocument/documentSymbol":
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": [{
                "name": "answer",
                "kind": 12,
                "range": rng(0, 0, 12),
                "selectionRange": rng(0, 4, 10),
                "children": [{
                    "name": "value",
                    "kind": 13,
                    "range": rng(1),
                    "selectionRange": rng(1),
                }],
            }],
        })
    elif method == "textDocument/prepareCallHierarchy":
        uri = params["textDocument"]["uri"]
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": [{
                "name": "answer",
                "kind": 12,
                "uri": uri,
                "range": rng(0),
                "selectionRange": rng(0),
                "data": {"symbol": "answer"},
            }],
        })
    elif method == "callHierarchy/incomingCalls":
        item = dict(params["item"])
        item["name"] = "caller"
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": [{"from": item, "fromRanges": [rng(3)]}],
        })
    elif method == "callHierarchy/outgoingCalls":
        item = dict(params["item"])
        item["name"] = "callee"
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": [{"to": item, "fromRanges": [rng(1)]}],
        })
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
    elif method == "exit":
        break
    elif request_id is not None:
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": "method not found"},
        })
'''


class LspToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "repo with spaces"
        self.root.mkdir()
        self.source = self.root / "sample.py"
        self.source.write_text(
            "def answer():\n    value = 42\n    return value\n\nanswer()\n",
            encoding="utf-8",
        )
        self.server = self.root / "fake_lsp_server.py"
        self.server.write_text(textwrap.dedent(_FAKE_LSP_SERVER), encoding="utf-8")
        self.workspace = WorkspaceGuard(self.root, repo_id="test-repo")
        self.tools = LspTools(self.workspace, timeout_s=2.0)
        self.command = shlex.join([sys.executable, str(self.server)])

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _environment(self, **extra: str):
        values = {"CODE_AGENT_LSP_COMMAND": self.command}
        values.update(extra)
        return patch.dict(os.environ, values, clear=False)

    def test_definition_and_references_are_normalised(self) -> None:
        with self._environment():
            definition = self.tools.definition("call-def", "sample.py", 1, 4)
            references = self.tools.references("call-ref", "sample.py", 1, 4)

        self.assertEqual("ok", definition["status"])
        self.assertIsNone(definition["error"])
        self.assertEqual("sample.py", definition["data"]["locations"][0]["path"])
        self.assertEqual(3, definition["data"]["locations"][0]["line"])
        self.assertEqual(64, len(definition["source"]["file_hash"]))
        self.assertEqual([1, 4], [item["line"] for item in references["data"]["locations"]])

    def test_document_symbols_and_call_hierarchy(self) -> None:
        with self._environment():
            symbols = self.tools.document_symbols("call-symbols", "sample.py")
            incoming = self.tools.call_hierarchy_incoming(
                "call-in", "sample.py", 1, 4
            )
            outgoing = self.tools.call_hierarchy_outgoing(
                "call-out", "sample.py", 1, 4
            )

        self.assertEqual("answer", symbols["data"]["symbols"][0]["name"])
        self.assertEqual("value", symbols["data"]["symbols"][0]["children"][0]["name"])
        self.assertEqual(
            "caller", incoming["data"]["incoming_calls"][0]["from"]["name"]
        )
        self.assertEqual(
            "callee", outgoing["data"]["outgoing_calls"][0]["to"]["name"]
        )
        self.assertEqual(4, incoming["data"]["incoming_calls"][0]["from_ranges"][0]["line"])

    def test_missing_server_returns_not_initialized(self) -> None:
        with patch.dict(
            os.environ,
            {"CODE_AGENT_LSP_COMMAND": "/definitely/missing/lsp-server"},
            clear=False,
        ):
            result = self.tools.definition("call-missing", "sample.py", 1, 0)

        self.assertEqual("error", result["status"])
        self.assertEqual("NOT_INITIALIZED", result["error"]["code"])
        self.assertFalse(result["error"]["retryable"])

    def test_unsupported_capability_returns_explicit_error(self) -> None:
        with self._environment(FAKE_LSP_UNSUPPORTED="1"):
            result = self.tools.call_hierarchy_incoming(
                "call-unsupported", "sample.py", 1, 0
            )

        self.assertEqual("error", result["status"])
        self.assertEqual("UNSUPPORTED_CAPABILITY", result["error"]["code"])

    def test_input_and_output_paths_cannot_escape_workspace(self) -> None:
        with self._environment():
            invalid_input = self.tools.definition(
                "call-input-escape", "../outside.py", 1, 0
            )

        outside = Path(self.temp_dir.name) / "outside.py"
        outside.write_text("outside = True\n", encoding="utf-8")
        with self._environment(FAKE_LSP_RESULT_URI=outside.resolve().as_uri()):
            invalid_output = self.tools.definition(
                "call-output-escape", "sample.py", 1, 0
            )

        self.assertEqual("OUT_OF_SCOPE", invalid_input["error"]["code"])
        self.assertEqual("partial", invalid_output["status"])
        self.assertEqual([], invalid_output["data"]["locations"])
        self.assertEqual(1, invalid_output["data"]["omitted_out_of_scope"])


if __name__ == "__main__":
    unittest.main()
