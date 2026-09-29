"""Application settings, read from environment variables (and .env if present).

Database credentials are deliberately NOT read here: they are entered on the
connection screen and live only in the current session.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    ollama_host: str
    ollama_model: str
    ollama_num_ctx: int
    ollama_timeout_s: int
    ollama_keep_alive: str
    hindsight_api_url: str | None
    hindsight_api_key: str | None
    hindsight_retain_async: bool
    max_repair_attempts: int
    max_result_rows: int
    query_timeout_s: int
    schema_sample_rows: int
    schema_distinct_values: int
    schema_scan_rows: int
    schema_full_threshold: int
    schema_max_tables: int
    schema_hidden_columns: frozenset[str]
    history_turns: int


def load_settings() -> Settings:
    return Settings(
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv(
            "OLLAMA_MODEL", "hf.co/mradermacher/XiYanSQL-QwenCoder-7B-2504-GGUF:Q4_K_M"
        ),
        ollama_num_ctx=_int("OLLAMA_NUM_CTX", 8192),
        ollama_timeout_s=_int("OLLAMA_TIMEOUT_S", 600),
        ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
        hindsight_api_url=os.getenv("HINDSIGHT_API_URL") or None,
        hindsight_api_key=os.getenv("HINDSIGHT_API_KEY") or None,
        hindsight_retain_async=_bool("HINDSIGHT_RETAIN_ASYNC", False),
        max_repair_attempts=_int("MAX_REPAIR_ATTEMPTS", 2),
        max_result_rows=_int("MAX_RESULT_ROWS", 500),
        query_timeout_s=_int("QUERY_TIMEOUT_S", 30),
        schema_sample_rows=_int("SCHEMA_SAMPLE_ROWS", 3),
        schema_distinct_values=_int("SCHEMA_DISTINCT_VALUES", 6),
        schema_scan_rows=_int("SCHEMA_SCAN_ROWS", 50000),
        schema_full_threshold=_int("SCHEMA_FULL_THRESHOLD", 8),
        schema_max_tables=_int("SCHEMA_MAX_TABLES", 6),
        schema_hidden_columns=frozenset(
            c.strip().lower()
            for c in os.getenv(
                "SCHEMA_HIDDEN_COLUMNS", "LastEditedBy,LastEditedWhen,ValidFrom,ValidTo"
            ).split(",")
            if c.strip()
        ),
        history_turns=_int("HISTORY_TURNS", 3),
    )
