"""Typed LangGraph state. Contains no credentials or connection details."""

from typing import Literal, TypedDict

from app.database.base import QueryResult
from app.sql.generator import Turn

Status = Literal["success", "empty", "rejected", "failed"]


class Attempt(TypedDict):
    sql: str
    error: str


class AgentState(TypedDict, total=False):
    # inputs
    user_question: str
    database_type: str  # dialect name shown to the model, e.g. "PostgreSQL"
    database_id: str  # non-secret identifier used to scope memory
    chat_history: list[Turn]
    hint: str | None  # optional extra evidence (e.g. BIRD "evidence")

    # working values
    recalled_memories: list[str]
    memory_error: str | None
    relevant_tables: list[str]
    relevant_schema: str
    base_evidence: str
    generated_sql: str
    validation_result: dict  # {"ok": bool, "reason": str, "repairable": bool}
    query_result: QueryResult | None
    execution_error: str | None
    repair_attempts: int
    failed_attempts: list[Attempt]

    # outputs
    status: Status
    final_response: str
    memory_retained: bool
