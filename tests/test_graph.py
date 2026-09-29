import pytest

from app.agent.graph import TextToSQLAgent
from app.agent.nodes import AgentDeps, AgentOptions
from app.memory.hindsight import NullMemory
from app.sql.generator import Turn
from tests.fakes import SECRET, FakeDB, FakeLLM, FakeMemory, sample_schema, settings


def make_agent(replies, fail_sql=None, memory=None, **option_overrides):
    db, llm = FakeDB(fail_sql), FakeLLM(replies)
    memory = memory if memory is not None else FakeMemory()
    options = AgentOptions(**{"max_repair_attempts": 2, **option_overrides})
    deps = AgentDeps(db, sample_schema(), llm, memory, settings(), options)
    return TextToSQLAgent(deps), db, llm, memory


def test_happy_path_generates_validates_executes_and_retains():
    agent, db, _, memory = make_agent(["SELECT count(*) FROM customers\n```"])
    state = agent.ask("How many customers are there?")
    assert state["status"] == "success"
    assert state["generated_sql"] == "SELECT count(*) FROM customers"
    assert db.executed == ["SELECT count(*) FROM customers"]
    assert state["repair_attempts"] == 0
    assert state["memory_retained"] is True
    (db_id, note), = memory.retained
    assert db_id == FakeDB.identifier
    assert "SELECT count(*) FROM customers" in note and "Tables used: customers" in note
    assert "(3,)" not in note  # result rows are never stored in memory


def test_execution_error_triggers_repair_then_succeeds():
    bad = "SELECT bad FROM customers"
    agent, db, llm, memory = make_agent([bad, "SELECT name FROM customers"], fail_sql={bad})
    state = agent.ask("customer names")
    assert state["status"] == "success"
    assert state["repair_attempts"] == 1
    assert db.executed == [bad, "SELECT name FROM customers"]
    # the repair prompt carries the failed SQL and the database error
    assert bad in llm.prompts[1] and 'column "bad" does not exist' in llm.prompts[1]
    assert "Correction:" in memory.retained[0][1]


def test_repair_loop_stops_at_limit():
    agent, db, llm, memory = make_agent(
        ["SELECT bad FROM t1", "SELECT bad FROM t2", "SELECT bad FROM t3", "never used"],
        fail_sql={"SELECT bad FROM t1", "SELECT bad FROM t2", "SELECT bad FROM t3"},
        max_repair_attempts=2,
    )
    state = agent.ask("anything")
    assert state["status"] == "failed"
    assert state["repair_attempts"] == 2
    assert len(db.executed) == 3 and len(llm.prompts) == 3
    assert memory.retained == []  # failures are not retained


def test_invalid_syntax_is_repaired_before_execution():
    agent, db, _, _ = make_agent(["SELEC * FRM customers", "SELECT * FROM customers"])
    state = agent.ask("all customers")
    assert state["status"] == "success"
    assert db.executed == ["SELECT * FROM customers"]  # garbage never reached the database


@pytest.mark.parametrize("sql", ["DELETE FROM customers", "SELECT 1; DROP TABLE customers"])
def test_unsafe_sql_is_refused_not_executed_or_repaired(sql):
    agent, db, llm, memory = make_agent([sql, "SELECT 1"])
    state = agent.ask("remove all customers")
    assert state["status"] == "rejected"
    assert "read-only" in state["final_response"]
    assert db.executed == [] and len(llm.prompts) == 1 and memory.retained == []


def test_memories_are_recalled_for_this_database_only_and_reach_prompt():
    memory = FakeMemory(
        {
            FakeDB.identifier: ["'big spender' means orders.total > 100"],
            "postgresql-other-db": ["unrelated memory from another database"],
        }
    )
    agent, _, llm, _ = make_agent(["SELECT * FROM orders WHERE total > 100"], memory=memory)
    agent.ask("who are the big spenders?")
    assert memory.recall_queries == [(FakeDB.identifier, "who are the big spenders?")]
    assert "'big spender' means orders.total > 100" in llm.prompts[0]
    assert "unrelated memory" not in llm.prompts[0]


def test_follow_up_questions_see_conversation_history():
    agent, _, llm, _ = make_agent(["SELECT name FROM customers WHERE id = 1"])
    history = [Turn("how many customers?", "SELECT count(*) FROM customers")]
    agent.ask("what is the first one's name?", history=history)
    assert "SELECT count(*) FROM customers" in llm.prompts[0]


def test_memory_disabled_variant_skips_recall_and_retain():
    memory = FakeMemory({FakeDB.identifier: ["some memory"]})
    agent, _, llm, _ = make_agent(
        ["SELECT 1"], memory=memory, use_memory=False, retain_memory=False
    )
    state = agent.ask("q")
    assert memory.recall_queries == [] and memory.retained == []
    assert "some memory" not in llm.prompts[0] and state["memory_retained"] is False


def test_works_without_hindsight():
    agent, _, _, _ = make_agent(["SELECT 1"], memory=NullMemory())
    assert agent.ask("q")["status"] == "success"


def test_no_credentials_in_state_or_prompts():
    agent, _, llm, _ = make_agent(["SELECT 1"])
    state = agent.ask("q")
    assert SECRET not in repr(state)
    assert not any(k in state for k in ("password", "connection_string", "username"))
    assert all(SECRET not in p for p in llm.prompts)


def test_recall_failure_does_not_break_question():
    class BrokenMemory(FakeMemory):
        def recall(self, database_id, query):
            raise RuntimeError("hindsight down")

    agent, _, _, _ = make_agent(["SELECT 1"], memory=BrokenMemory())
    state = agent.ask("q")
    assert state["status"] == "success"
    assert "hindsight down" in state["memory_error"]


def test_schema_sent_to_model_includes_samples_and_foreign_keys():
    agent, _, llm, _ = make_agent(["SELECT 1"])
    agent.ask("orders per customer")
    prompt = llm.prompts[0]
    assert "# Table: orders" in prompt and "orders.customer_id=customers.id" in prompt
    assert "(name:TEXT, Examples: [Alice, Bob, Carol])" in prompt  # distinct text values
    assert "(total:NUMERIC, Examples: [9.5])" in prompt  # non-text: sample rows
