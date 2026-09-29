from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.agent.graph import TextToSQLAgent
from app.agent.nodes import AgentDeps, AgentOptions
from app.session import Session
from tests.fakes import SECRET, FakeDB, FakeLLM, FakeMemory, sample_schema, settings

MAIN = str(Path(__file__).resolve().parent.parent / "app" / "main.py")


class _FakeLLM:
    def check(self):
        return None


def _fill(at, host):
    at.selectbox[0].set_value("postgresql")
    at.text_input[0].set_value(host)
    at.text_input[2].set_value("someone")
    at.text_input[3].set_value(SECRET)
    at.button[0].click().run(timeout=30)


def test_connection_screen_fields():
    at = AppTest.from_file(MAIN).run(timeout=30)
    labels = [w.label for w in at.text_input]
    assert labels == ["Connection string / host", "Database", "Username", "Password"]
    assert at.selectbox[0].label == "Database type"


def test_failed_connection_shows_error_without_password():
    at = AppTest.from_file(MAIN).run(timeout=30)
    with patch("app.ui.connection.get_llm", return_value=_FakeLLM()):
        _fill(at, "127.0.0.1:1/nodb")
    assert at.error, "expected an error message"
    assert SECRET not in at.error[0].value
    assert "session" not in at.session_state


def test_chat_shows_question_sql_and_result():
    llm = FakeLLM(["SELECT count(*) FROM customers"])
    deps = AgentDeps(FakeDB(), sample_schema(), llm, FakeMemory(), settings(), AgentOptions())
    session = Session(db=deps.db, schema=deps.schema, agent=TextToSQLAgent(deps))
    session.db.display_target = "PostgreSQL localhost:5432/shop"
    at = AppTest.from_file(MAIN)
    at.session_state["session"] = session
    at.session_state["turns"] = []
    at.run(timeout=30)
    at.chat_input[0].set_value("How many customers?").run(timeout=30)
    assert at.chat_message[0].markdown[0].value == "How many customers?"
    assert at.code[0].value == "SELECT count(*) FROM customers"
    assert at.dataframe[0].value.iloc[0, 0] == 3
