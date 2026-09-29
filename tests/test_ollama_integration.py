"""Runs the real XiYanSQL model through Ollama. Slow on CPU; opt in with RUN_OLLAMA_TESTS=1."""

import os

import pytest

from app.agent.graph import TextToSQLAgent
from app.agent.nodes import AgentDeps, AgentOptions
from app.config import load_settings
from app.llm.ollama import OllamaLLM
from app.memory.hindsight import NullMemory
from app.sql.validator import validate_sql
from tests.fakes import FakeDB, sample_schema

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1", reason="set RUN_OLLAMA_TESTS=1 to run"
)


@pytest.fixture(scope="module")
def llm():
    model = OllamaLLM(load_settings())
    model.check()
    return model


def test_xiyan_generates_valid_join_sql(llm):
    deps = AgentDeps(
        FakeDB(), sample_schema(), llm, NullMemory(), load_settings(), AgentOptions()
    )
    state = TextToSQLAgent(deps).ask("What is the total order amount for each customer name?")
    sql = state["generated_sql"]
    print("\nGenerated:", sql)
    assert validate_sql(sql, "postgres").ok
    lowered = sql.lower()
    assert "join" in lowered and "sum(" in lowered and "customers" in lowered
    assert "```" not in sql
