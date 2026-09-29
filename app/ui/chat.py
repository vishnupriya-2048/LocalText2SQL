"""Chat screen: question → SQL → result, with follow-up context."""

import pandas as pd
import streamlit as st

from app.agent.state import AgentState
from app.llm.ollama import LLMError
from app.session import Session
from app.sql.generator import Turn

_STATUS_ICON = {"success": "✅", "empty": "⚪", "rejected": "⛔", "failed": "❌", "error": "❌"}


def _frame(columns: list[str], rows: list[tuple]) -> pd.DataFrame:
    seen: dict[str, int] = {}
    unique = []
    for name in columns:  # dataframes need unique column names
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return pd.DataFrame(rows, columns=unique)


def _turn_from_state(question: str, state: AgentState) -> dict:
    result = state.get("query_result")
    return {
        "question": question,
        "sql": state.get("generated_sql", ""),
        "status": state.get("status", "failed"),
        "response": state.get("final_response", ""),
        "columns": result.columns if result else [],
        "rows": result.rows if result else [],
        "elapsed_ms": result.elapsed_ms if result else None,
        "attempts": state.get("failed_attempts", []),
        "memories": state.get("recalled_memories", []),
        "tables": state.get("relevant_tables", []),
        "memory_error": state.get("memory_error"),
        "memory_retained": state.get("memory_retained", False),
    }


def _render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        if turn["sql"]:
            st.markdown("**SQL**")
            st.code(turn["sql"], language="sql")
        timing = f" · {turn['elapsed_ms']} ms" if turn["elapsed_ms"] is not None else ""
        st.markdown(f"**Result** {_STATUS_ICON.get(turn['status'], '')} {turn['response']}{timing}")
        if turn["rows"]:
            st.dataframe(_frame(turn["columns"], turn["rows"]), hide_index=True)
        details = []
        if turn["attempts"]:
            details.append(("Repair attempts", turn["attempts"]))
        if turn["memories"]:
            details.append(("Memories used", turn["memories"]))
        if details or turn["tables"]:
            with st.expander("Details"):
                st.caption("Tables given to the model: " + ", ".join(turn["tables"]))
                for title, items in details:
                    st.markdown(f"**{title}**")
                    for item in items:
                        if isinstance(item, dict):
                            st.code(item["sql"], language="sql")
                            st.caption(f"Error: {item['error']}")
                        else:
                            st.caption(f"- {item}")
                if turn["memory_retained"]:
                    st.caption("Saved to long-term memory.")
        if turn["memory_error"]:
            st.caption(f"⚠️ {turn['memory_error']}")


def _sidebar(session: Session) -> None:
    with st.sidebar:
        st.subheader("Connected")
        st.write(session.db.display_target)
        st.caption(f"{len(session.schema.tables)} tables/views")
        for warning in session.warnings:
            st.warning(warning)
        with st.expander("Schema"):
            for table in session.schema.tables:
                st.markdown(f"**{table.qualified_name}**")
                st.caption(", ".join(c.name for c in table.columns))
        memory = session.agent.deps.memory
        if memory.enabled:
            with st.form("teach", clear_on_submit=True):
                note = st.text_area(
                    "Teach a fact about this database",
                    placeholder="e.g. 'active customer' means ordered in the last 90 days",
                )
                if st.form_submit_button("Remember") and note.strip():
                    try:
                        memory.retain(
                            session.db.identifier,
                            note.strip(),
                            context="Fact stated by the user about this database",
                        )
                        st.success("Saved to memory.")
                    except Exception as exc:  # noqa: BLE001 - show any memory failure to the user
                        st.error(f"Could not save: {exc}")
        if st.button("Disconnect"):
            session.close()
            st.session_state.clear()
            st.rerun()


def render() -> None:
    session: Session = st.session_state["session"]
    turns: list[dict] = st.session_state["turns"]
    _sidebar(session)
    st.title("Ask your database")

    for turn in turns:
        _render_turn(turn)

    question = st.chat_input("Ask a question in plain English")
    if not question:
        return
    history = [
        Turn(t["question"], t["sql"]) for t in turns if t["status"] in ("success", "empty")
    ]
    with st.spinner("Thinking (local model on CPU, this can take a minute)..."):
        try:
            state = session.agent.ask(question, history=history)
            turn = _turn_from_state(question, state)
        except LLMError as exc:
            turn = _turn_from_state(question, {"status": "error", "final_response": str(exc)})
    turns.append(turn)
    st.rerun()
