import json

from app.agent.nodes import AgentOptions
from app.database.base import QueryResult
from app.memory.hindsight import NullMemory
from benchmark.dataset import Example, load_examples
from benchmark.metrics import Outcome, results_match, summarize
from benchmark.run import evaluate, variant_options
from tests.fakes import FakeDB, FakeLLM, sample_schema, settings


def test_results_match_is_order_insensitive_set_equality():
    assert results_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a")])
    assert not results_match([(1, "a")], [(1, "b")])


def test_load_examples_json_and_jsonl(tmp_path):
    item = {"question_id": 7, "db_id": "shop", "question": "q", "evidence": "e",
            "SQL": "SELECT 1", "difficulty": "simple"}  # fmt: skip
    (tmp_path / "a.json").write_text(json.dumps([item, {**item, "db_id": "other"}]))
    (tmp_path / "b.jsonl").write_text(json.dumps(item) + "\n")
    assert load_examples(tmp_path / "a.json", db_ids={"shop"}) == [
        Example("7", "shop", "q", "e", "SELECT 1", "simple")
    ]
    assert len(load_examples(tmp_path / "b.jsonl")) == 1


def test_summarize():
    o = lambda v, ok, d: Outcome(v, "1", d, ok, "success", "", None, 0, 1.0)
    summary = summarize([o("a", True, "simple"), o("a", False, "hard"), o("b", True, "simple")])
    assert summary["a"]["overall"] == "1/2 (50.0%)" and summary["b"]["simple"] == "1/1 (100.0%)"


def test_variants_isolate_each_component():
    v = variant_options(settings(max_repair_attempts=3), retain=False)
    assert v["baseline"] == AgentOptions(False, False, False, 0)
    assert v["schema_retrieval"].use_schema_retrieval and not v["schema_retrieval"].use_memory
    assert v["hindsight"].use_memory and v["hindsight"].max_repair_attempts == 0
    assert v["full_agent"].max_repair_attempts == 3


class GoldAwareDB(FakeDB):
    def execute_readonly(self, sql, max_rows, timeout_s):
        self.executed.append(sql)
        rows = [(3,)] if "count" in sql.lower() else [(99,)]
        return QueryResult(["n"], rows, False, 1)


class OneDB:
    def __init__(self, db):
        self.db = db

    def get(self, db_id):
        return self.db, sample_schema()


def test_evaluate_scores_by_execution_and_passes_evidence():
    db = GoldAwareDB()
    llm = FakeLLM(["SELECT count(*) FROM customers", "SELECT id FROM customers"])
    ex = Example("1", "shop", "how many customers", "customers are people", "SELECT COUNT(*) FROM customers", "simple")
    kwargs = {"dbs": OneDB(db), "llm": llm, "memory": NullMemory(), "settings": settings(),
              "use_evidence": True, "gold_cache": {}}  # fmt: skip
    right = evaluate(ex, "baseline", AgentOptions(False, False, False, 0), **kwargs)
    wrong = evaluate(ex, "baseline", AgentOptions(False, False, False, 0), **kwargs)
    assert right.correct and not wrong.correct
    assert "customers are people" in llm.prompts[0]
    assert db.executed.count("SELECT COUNT(*) FROM customers") == 1  # gold result cached
