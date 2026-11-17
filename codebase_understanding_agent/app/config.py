from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from .contracts import ConfigurationError


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} 必须是整数。") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} 必须在 {minimum}–{maximum} 之间。")
    return value


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} 必须是数字。") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} 必须在 {minimum}–{maximum} 之间。")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} 必须是 1/0、true/false、yes/no 或 on/off。")


def _provider_value(prefix: str, name: str, legacy: str | None, default: str) -> str:
    value = os.getenv(f"{prefix}_{name}")
    if value is not None and value.strip():
        return value.strip()
    if legacy:
        legacy_value = os.getenv(legacy)
        if legacy_value is not None and legacy_value.strip():
            return legacy_value.strip()
    return default


@dataclass(frozen=True)
class AgentConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    request_timeout_s: float
    host: str
    port: int
    data_dir: Path
    max_steps: int
    max_relevant_notes: int
    max_file_lines: int
    max_matches: int
    max_files: int
    max_file_bytes: int
    search_timeout_s: float
    embedding_provider: str
    embedding_model: str
    embedding_base_url: str
    embedding_api_key: str
    embedding_dimensions: int
    terminal_enabled: bool
    terminal_allowed_programs: tuple[str, ...]
    terminal_timeout_s: float
    terminal_max_stdout_bytes: int

    @classmethod
    def load(cls) -> "AgentConfig":
        # A standalone checkout keeps .env at its repository root. The parent
        # fallback preserves compatibility when this app lives inside hello-agents.
        env_path = APP_DIR / ".env"
        if not env_path.exists():
            env_path = PROJECT_ROOT / ".env"
        load_dotenv(env_path, override=False)

        provider = os.getenv("CODE_AGENT_PROVIDER", "ollama").strip().lower()
        if provider not in {"ollama", "deepseek", "custom"}:
            raise ConfigurationError(
                "CODE_AGENT_PROVIDER 必须是 ollama、deepseek 或 custom。"
            )
        prefix = provider.upper()
        defaults = {
            "ollama": ("deepseek-r1:7b", "http://127.0.0.1:11434/v1", "ollama"),
            "deepseek": ("deepseek-flash", "", ""),
            "custom": ("", "", ""),
        }
        default_model, default_base_url, default_api_key = defaults[provider]
        legacy_model = "LLM_MODEL_ID" if provider == "ollama" else None
        legacy_base_url = "LLM_BASE_URL" if provider == "ollama" else None
        legacy_api_key = "LLM_API_KEY" if provider == "ollama" else None
        model = _provider_value(prefix, "MODEL_ID", legacy_model, default_model)
        base_url = _provider_value(
            prefix, "BASE_URL", legacy_base_url, default_base_url
        ).rstrip("/")
        api_key = _provider_value(prefix, "API_KEY", legacy_api_key, default_api_key)
        if not model:
            raise ConfigurationError(f"{prefix}_MODEL_ID 不能为空。")
        if not base_url:
            raise ConfigurationError(f"{prefix}_BASE_URL 不能为空。")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ConfigurationError(f"{prefix}_BASE_URL 不是有效的 HTTP 地址。")
        if provider == "ollama" and parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ConfigurationError(
                "OLLAMA_BASE_URL 默认只允许本机地址；远程 API 请使用 "
                "CODE_AGENT_PROVIDER=deepseek 或 custom。"
            )

        data_dir = Path(
            os.getenv("CODE_AGENT_DATA_DIR", str(Path.home() / ".codebase_understanding_agent"))
        ).expanduser()

        timeout_name = f"{prefix}_TIMEOUT"
        timeout_raw = os.getenv(timeout_name)
        if timeout_raw is None and provider == "ollama":
            timeout_raw = os.getenv("LLM_TIMEOUT")
        try:
            request_timeout = 180.0 if timeout_raw is None else float(timeout_raw)
        except ValueError as exc:
            raise ConfigurationError(f"{timeout_name} 必须是数字。") from exc
        if not 5.0 <= request_timeout <= 1800.0:
            raise ConfigurationError(f"{timeout_name} 必须在 5–1800 秒之间。")

        embedding_provider = os.getenv(
            "CODE_AGENT_EMBEDDING_PROVIDER", "local_hash"
        ).strip().lower()
        if embedding_provider not in {"local_hash", "openai_compatible"}:
            raise ConfigurationError(
                "CODE_AGENT_EMBEDDING_PROVIDER 必须是 local_hash 或 openai_compatible。"
            )
        embedding_model = os.getenv(
            "CODE_AGENT_EMBEDDING_MODEL", "feature-hash-v1"
        ).strip()
        embedding_base_url = os.getenv("CODE_AGENT_EMBEDDING_BASE_URL", "").strip().rstrip("/")
        embedding_api_key = os.getenv("CODE_AGENT_EMBEDDING_API_KEY", "").strip()
        if embedding_provider == "openai_compatible" and (
            not embedding_model or not embedding_base_url
        ):
            raise ConfigurationError(
                "使用 openai_compatible 向量模型时必须配置 "
                "CODE_AGENT_EMBEDDING_MODEL 和 CODE_AGENT_EMBEDDING_BASE_URL。"
            )

        allowed_programs = tuple(
            item.strip()
            for item in os.getenv("CODE_AGENT_TERMINAL_ALLOWED_PROGRAMS", "rg").split(",")
            if item.strip()
        )
        if not allowed_programs:
            raise ConfigurationError("CODE_AGENT_TERMINAL_ALLOWED_PROGRAMS 不能为空。")

        return cls(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key or "local-placeholder",
            request_timeout_s=request_timeout,
            host="127.0.0.1",
            port=_env_int("CODE_AGENT_PORT", 8765, 1024, 65535),
            data_dir=data_dir,
            max_steps=_env_int("CODE_AGENT_MAX_STEPS", 10, 1, 25),
            max_relevant_notes=_env_int("CODE_AGENT_RELEVANT_NOTES", 5, 0, 20),
            max_file_lines=_env_int("CODE_AGENT_MAX_FILE_LINES", 160, 20, 1000),
            max_matches=_env_int("CODE_AGENT_MAX_MATCHES", 50, 5, 500),
            max_files=_env_int("CODE_AGENT_MAX_FILES", 120, 10, 2000),
            max_file_bytes=_env_int("CODE_AGENT_MAX_FILE_BYTES", 1_500_000, 10_000, 20_000_000),
            search_timeout_s=_env_float("CODE_AGENT_SEARCH_TIMEOUT", 15.0, 1.0, 120.0),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_dimensions=_env_int(
                "CODE_AGENT_EMBEDDING_DIMENSIONS", 384, 64, 8192
            ),
            terminal_enabled=_env_bool("CODE_AGENT_TERMINAL_ENABLED", True),
            terminal_allowed_programs=allowed_programs,
            terminal_timeout_s=_env_float(
                "CODE_AGENT_TERMINAL_TIMEOUT", 15.0, 0.5, 300.0
            ),
            terminal_max_stdout_bytes=_env_int(
                "CODE_AGENT_TERMINAL_MAX_STDOUT_BYTES",
                8_000_000,
                65_536,
                100_000_000,
            ),
        )

    @property
    def static_dir(self) -> Path:
        return APP_DIR / "app" / "static"

    @property
    def prompt_path(self) -> Path:
        return APP_DIR / "app" / "prompts" / "system.md"
