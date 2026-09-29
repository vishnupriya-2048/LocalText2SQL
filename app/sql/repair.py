"""Ask the model to fix SQL that failed validation or execution."""

from app.llm.ollama import OllamaLLM
from app.sql.generator import generate_sql

_MAX_ERROR_CHARS = 1000


def repair_evidence(base_evidence: str, failed_sql: str, error: str) -> str:
    return (
        f"{base_evidence}\n\n"
        "A previous attempt to answer this question failed.\n"
        f"Failed SQL:\n{failed_sql}\n"
        f"Database error:\n{error[:_MAX_ERROR_CHARS]}\n"
        "Write a corrected query that avoids this error, using only tables and "
        "columns that exist in the schema above."
    )


def repair_sql(
    llm: OllamaLLM,
    dialect: str,
    question: str,
    schema: str,
    base_evidence: str,
    failed_sql: str,
    error: str,
) -> str:
    return generate_sql(
        llm, dialect, question, schema, repair_evidence(base_evidence, failed_sql, error)
    )
