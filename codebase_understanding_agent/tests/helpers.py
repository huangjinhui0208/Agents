from __future__ import annotations

from pathlib import Path

from codebase_understanding_agent.app.config import AgentConfig


def make_config(tmp_path: Path, **overrides: object) -> AgentConfig:
    values = {
        "provider": "ollama",
        "model": "deepseek-r1:7b",
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "request_timeout_s": 5.0,
        "host": "127.0.0.1",
        "port": 8765,
        "data_dir": tmp_path / "agent-data",
        "max_steps": 4,
        "max_relevant_notes": 5,
        "max_file_lines": 120,
        "max_matches": 30,
        "max_files": 80,
        "max_file_bytes": 500_000,
        "search_timeout_s": 2.0,
        "embedding_provider": "local_hash",
        "embedding_model": "feature-hash-v1",
        "embedding_base_url": "",
        "embedding_api_key": "",
        "embedding_dimensions": 128,
        "terminal_enabled": False,
        "terminal_allowed_programs": ("rg",),
        "terminal_timeout_s": 2.0,
        "terminal_max_stdout_bytes": 1_000_000,
    }
    values.update(overrides)
    return AgentConfig(**values)  # type: ignore[arg-type]
