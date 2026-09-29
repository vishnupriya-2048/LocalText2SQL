"""Connection screen. The password lives only in this form and the DB engine."""

import os

import streamlit as st

from app.config import load_settings
from app.database import DATABASE_TYPES, ConnectionConfig, DatabaseError
from app.llm.ollama import LLMError, OllamaLLM
from app.session import open_session


@st.cache_resource
def get_llm() -> OllamaLLM:
    # One shared client (and lock) per process, so generations never run concurrently.
    return OllamaLLM(load_settings())


def _label(dialect: str) -> str:
    cls = DATABASE_TYPES[dialect]
    return cls.display_name if cls.supported else f"{cls.display_name} (coming soon)"


def render() -> None:
    st.title("Local Text-to-SQL")
    st.caption("Connect to your database. Credentials stay in this browser session only.")

    dialects = list(DATABASE_TYPES)
    default_type = os.getenv("DB_TYPE", "postgresql")
    with st.form("connect"):
        dialect = st.selectbox(
            "Database type",
            dialects,
            index=dialects.index(default_type) if default_type in dialects else 0,
            format_func=_label,
        )
        host = st.text_input(
            "Connection string / host",
            value=os.getenv("DB_HOST", "localhost:5432"),
            help="host, host:port, host:port/database, or postgresql://host:port/database",
        )
        database = st.text_input(
            "Database", value=os.getenv("DB_NAME", ""), help="Optional if included above."
        )
        username = st.text_input("Username", value=os.getenv("DB_USER", ""))
        password = st.text_input("Password", type="password", key="conn_password")
        submitted = st.form_submit_button("Connect", type="primary")

    if not submitted:
        return
    config = ConnectionConfig(
        dialect=dialect,
        host=host,
        username=username,
        password=password,
        database=database.strip() or None,
    )
    settings = load_settings()
    with st.spinner("Testing connection and reading schema..."):
        try:
            session = open_session(config, settings, get_llm())
        except (DatabaseError, LLMError) as exc:
            st.error(str(exc))
            return
    st.session_state.pop("conn_password", None)
    st.session_state["session"] = session
    st.session_state["turns"] = []
    st.rerun()
