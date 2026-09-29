"""LangGraph node implementations and routing functions.

Each node takes the state and returns a partial update. Dependencies (database,
LLM, memory) are held on `AgentNodes`, never in state.
"""

import logging
from dataclasses import dataclass

from app.agent.state import AgentState
from app.config import Settings
from app.database.base import Database, DatabaseError, QueryExecutionError
from app.llm.ollama import OllamaLLM
from app.memory.hindsight import Memory
from app.schema.extractor import DatabaseSchema, Table
from app.schema.retriever import format_mschema, select_tables
from app.sql.generator import build_evidence, generate_sql
from app.sql.repair import repair_sql
from app.sql.validator import tables_in_sql, validate_sql

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentOptions:
    """Switches that let the benchmark compare pipeline variants."""

    use_schema_retrieval: bool = True
    use_memory: bool = True
    retain_memory: bool = True
    max_repair_attempts: int = 2


@dataclass
class AgentDeps:
    db: Database
    schema: DatabaseSchema
    llm: OllamaLLM
    memory: Memory
    settings: Settings
    options: AgentOptions


class AgentNodes:
    def __init__(self, deps: AgentDeps):
        self.deps = deps

    # -- nodes ------------------------------------------------------------------

    def recall_memory(self, state: AgentState) -> dict:
        d = self.deps
        if not (d.options.use_memory and d.memory.enabled):
            return {"recalled_memories": [], "memory_error": None}
        try:
            memories = d.memory.recall(state["database_id"], state["user_question"])
        except Exception as exc:  # noqa: BLE001 - memory is optional; never fail the question
            log.warning("Memory recall failed: %s", exc)
            return {"recalled_memories": [], "memory_error": f"Memory recall failed: {exc}"}
        return {"recalled_memories": memories, "memory_error": None}

    def identify_relevant_schema(self, state: AgentState) -> dict:
        d = self.deps
        if d.options.use_schema_retrieval:
            hints = list(state.get("recalled_memories", []))
            hints += [t.question for t in state.get("chat_history", [])]
            tables = select_tables(
                d.schema,
                state["user_question"],
                hints,
                max_tables=d.settings.schema_max_tables,
                full_threshold=d.settings.schema_full_threshold,
            )
        else:
            tables = list(d.schema.tables)
        if d.settings.schema_sample_rows > 0:
            for table in tables:
                if table.sample_rows is None:  # cached on the session's schema object
                    self._load_examples(table)
        return {
            "relevant_tables": [t.qualified_name for t in tables],
            "relevant_schema": format_mschema(
                d.schema.database_name, tables, d.settings.schema_hidden_columns
            ),
        }

    def _load_examples(self, table: Table) -> None:
        s = self.deps.settings
        try:
            table.sample_rows = self.deps.db.sample_rows(table, s.schema_sample_rows)
            for col in table.columns:
                if col.is_text:
                    table.column_values[col.name] = self.deps.db.distinct_values(
                        table, col.name, s.schema_distinct_values, s.schema_scan_rows
                    )
        except DatabaseError as exc:
            log.warning("Could not sample %s: %s", table.qualified_name, exc)
            table.sample_rows = table.sample_rows or []

    def generate_sql(self, state: AgentState) -> dict:
        history = state.get("chat_history", [])[-self.deps.settings.history_turns :]
        evidence = build_evidence(
            memories=state.get("recalled_memories", []),
            history=history,
            hint=state.get("hint"),
            extra=self.deps.db.prompt_hints or None,
        )
        sql = generate_sql(
            self.deps.llm,
            state["database_type"],
            state["user_question"],
            state["relevant_schema"],
            evidence,
        )
        return {"generated_sql": sql, "base_evidence": evidence}

    def validate_sql(self, state: AgentState) -> dict:
        result = validate_sql(state["generated_sql"], self.deps.db.sqlglot_dialect)
        return {
            "validation_result": {
                "ok": result.ok,
                "reason": result.reason,
                "repairable": result.repairable,
            }
        }

    def execute_sql(self, state: AgentState) -> dict:
        s = self.deps.settings
        try:
            result = self.deps.db.execute_readonly(
                state["generated_sql"], s.max_result_rows, s.query_timeout_s
            )
        except QueryExecutionError as exc:
            return {"query_result": None, "execution_error": str(exc)}
        return {"query_result": result, "execution_error": None}

    def repair_sql(self, state: AgentState) -> dict:
        error = _current_error(state)
        failed = state["generated_sql"]
        sql = repair_sql(
            self.deps.llm,
            state["database_type"],
            state["user_question"],
            state["relevant_schema"],
            state["base_evidence"],
            failed,
            error,
        )
        return {
            "generated_sql": sql,
            "repair_attempts": state.get("repair_attempts", 0) + 1,
            "failed_attempts": [*state.get("failed_attempts", []), {"sql": failed, "error": error}],
            "execution_error": None,
        }

    def evaluate_result(self, state: AgentState) -> dict:
        validation = state.get("validation_result") or {}
        result = state.get("query_result")
        if not validation.get("ok") and not validation.get("repairable", True):
            return {
                "status": "rejected",
                "final_response": f"Query refused: {validation.get('reason')}",
            }
        if validation.get("ok") and result is not None:
            if not result.rows:
                return {"status": "empty", "final_response": "The query ran but returned no rows."}
            more = f" (showing the first {len(result.rows)})" if result.truncated else ""
            return {
                "status": "success",
                "final_response": f"Returned {len(result.rows)} row(s){more}.",
            }
        attempts = state.get("repair_attempts", 0)
        return {
            "status": "failed",
            "final_response": (
                f"Could not produce a working query after {attempts} repair attempt(s). "
                f"Last error: {_current_error(state)}"
            ),
        }

    def retain_memory(self, state: AgentState) -> dict:
        d = self.deps
        if not (d.options.retain_memory and d.memory.enabled and state.get("status") == "success"):
            return {"memory_retained": False}
        sql = state["generated_sql"]
        if any(sql in m for m in state.get("recalled_memories", [])):
            return {"memory_retained": False}  # already known
        try:
            d.memory.retain(
                state["database_id"],
                memory_note(state, d.db.sqlglot_dialect),
                context="Verified text-to-SQL interaction (query executed successfully)",
            )
        except Exception as exc:  # noqa: BLE001 - memory is optional; never fail the question
            log.warning("Memory retain failed: %s", exc)
            return {"memory_retained": False, "memory_error": f"Memory retain failed: {exc}"}
        return {"memory_retained": True}

    # -- routing ----------------------------------------------------------------

    def route_after_validation(self, state: AgentState) -> str:
        v = state["validation_result"]
        if v["ok"]:
            return "execute_sql"
        if v["repairable"] and self._can_repair(state):
            return "repair_sql"
        return "evaluate_result"

    def route_after_execution(self, state: AgentState) -> str:
        if state.get("execution_error") is None:
            return "evaluate_result"
        return "repair_sql" if self._can_repair(state) else "evaluate_result"

    def _can_repair(self, state: AgentState) -> bool:
        return state.get("repair_attempts", 0) < self.deps.options.max_repair_attempts


def _current_error(state: AgentState) -> str:
    validation = state.get("validation_result") or {}
    if validation and not validation.get("ok"):
        return validation.get("reason", "")
    return state.get("execution_error") or ""


def memory_note(state: AgentState, dialect: str) -> str:
    """A compact, data-free summary of what worked. Result rows are never stored."""
    lines = [
        f'Question: "{state["user_question"]}"',
        f"Working SQL: {state['generated_sql']}",
    ]
    tables = tables_in_sql(state["generated_sql"], dialect)
    if tables:
        lines.append(f"Tables used: {', '.join(tables)}")
    for attempt in state.get("failed_attempts", []):
        lines.append(
            f"Correction: an earlier attempt `{attempt['sql']}` failed with "
            f"\"{attempt['error'][:300]}\"; the working SQL above fixed it."
        )
    return "\n".join(lines)
