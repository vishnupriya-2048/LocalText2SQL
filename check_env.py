"""Smoke test: verify imports and that Ollama (with the model) and Hindsight are reachable.

Database connectivity is checked by the app's connection screen, where credentials
are entered; this script never handles database credentials.
"""

import sys

from app.config import load_settings
from app.llm.ollama import LLMError, OllamaLLM
from app.memory.hindsight import create_memory


def check_imports() -> bool:
    import hindsight_client  # noqa: F401
    import langgraph  # noqa: F401
    import psycopg  # noqa: F401
    import sqlglot  # noqa: F401
    import streamlit  # noqa: F401

    print(f"imports ok on Python {sys.version.split()[0]}")
    return True


def check_ollama() -> bool:
    settings = load_settings()
    try:
        OllamaLLM(settings).check()
    except LLMError as exc:
        print(f"ollama NOT ready: {exc}")
        return False
    print(f"ollama ok at {settings.ollama_host}, model {settings.ollama_model}")
    return True


def check_hindsight() -> bool:
    memory, warning = create_memory(load_settings())
    print(f"hindsight: {warning}" if warning else "hindsight ok")
    return memory.enabled


if __name__ == "__main__":
    required = [check_imports(), check_ollama()]
    check_hindsight()  # optional: the app runs without memory
    sys.exit(0 if all(required) else 1)
