"""In-memory stand-ins for the database, LLM and memory, for graph tests."""

from app.config import load_settings
from app.database.base import QueryExecutionError, QueryResult
from app.schema.extractor import Column, DatabaseSchema, ForeignKey, Table

SECRET = "s3cr3t-pa55"


def sample_schema() -> DatabaseSchema:
    return DatabaseSchema(
        database_name="shop",
        dialect="postgresql",
        tables=[
            Table(
                "customers",
                None,
                [Column("id", "INTEGER", False, True), Column("name", "TEXT")],
            ),
            Table(
                "orders",
                None,
                [
                    Column("id", "INTEGER", False, True),
                    Column("customer_id", "INTEGER"),
                    Column("total", "NUMERIC"),
                ],
                foreign_keys=[ForeignKey(("customer_id",), "customers", ("id",))],
            ),
        ],
    )


class FakeDB:
    display_name = "PostgreSQL"
    sqlglot_dialect = "postgres"
    prompt_hints = ""
    identifier = "postgresql-shop-abc123"

    def __init__(self, fail_sql: set[str] | None = None):
        self.fail_sql = fail_sql or set()
        self.executed: list[str] = []

    def sample_rows(self, table, limit):
        return [(1, "Alice")] if table.name == "customers" else [(1, 1, 9.5)]

    def distinct_values(self, table, column_name, limit, scan_rows):
        return ["Alice", "Bob", "Carol"][:limit]

    def execute_readonly(self, sql, max_rows, timeout_s):
        self.executed.append(sql)
        if sql in self.fail_sql:
            raise QueryExecutionError(f'column "bad" does not exist ({sql})')
        return QueryResult(columns=["n"], rows=[(3,)], truncated=False, elapsed_ms=1)


class FakeLLM:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt, max_tokens=512):
        self.prompts.append(prompt)
        return self.replies.pop(0)


class FakeMemory:
    enabled = True

    def __init__(self, memories: dict[str, list[str]] | None = None):
        self.memories = memories or {}
        self.retained: list[tuple[str, str]] = []
        self.recall_queries: list[tuple[str, str]] = []

    def recall(self, database_id, query):
        self.recall_queries.append((database_id, query))
        return self.memories.get(database_id, [])

    def retain(self, database_id, content, context):
        self.retained.append((database_id, content))


def settings(**overrides):
    from dataclasses import replace

    return replace(load_settings(), **overrides)
