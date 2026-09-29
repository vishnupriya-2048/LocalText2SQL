"""Terminal chat, handy for verifying the pipeline without the UI.

    python -m app.cli --host localhost:5432/mydb --user postgres

The password is prompted for (never passed as an argument, so it stays out of
shell history).
"""

import argparse
import getpass
import sys

from app.config import load_settings
from app.database import DATABASE_TYPES, ConnectionConfig, DatabaseError
from app.llm.ollama import LLMError, OllamaLLM
from app.session import open_session
from app.sql.generator import Turn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--type", default="postgresql", choices=list(DATABASE_TYPES))
    parser.add_argument("--host", required=True, help="host:port/database or a URL")
    parser.add_argument("--database")
    parser.add_argument("--user", required=True)
    args = parser.parse_args()
    # Windows consoles default to cp1252; don't crash on non-Latin data in results.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    config = ConnectionConfig(
        dialect=args.type,
        host=args.host,
        username=args.user,
        password=getpass.getpass("Password: "),
        database=args.database,
    )
    settings = load_settings()
    try:
        session = open_session(config, settings, OllamaLLM(settings))
    except (DatabaseError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Connected to {session.db.display_target} ({len(session.schema.tables)} tables)")
    for warning in session.warnings:
        print(f"Warning: {warning}")

    history: list[Turn] = []
    try:
        while True:
            question = input("\nQuestion> ").strip()
            if question.lower() in {"", "exit", "quit"}:
                break
            try:
                state = session.agent.ask(question, history=history)
            except LLMError as exc:
                print(f"Error: {exc}")
                continue
            print(f"\nSQL:\n{state.get('generated_sql', '')}")
            for attempt in state.get("failed_attempts", []):
                print(f"(repaired after error: {attempt['error']})")
            print(f"\nResult: {state['final_response']}")
            result = state.get("query_result")
            if result and result.rows:
                print(" | ".join(result.columns))
                for row in result.rows[:20]:
                    print(" | ".join(str(v) for v in row))
            if state["status"] in ("success", "empty"):
                history.append(Turn(question, state["generated_sql"]))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
