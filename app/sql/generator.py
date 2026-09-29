"""Text-to-SQL prompt construction and SQL extraction for XiYanSQL.

The prompt follows XiYanSQL-QwenCoder's own template (Chinese section markers
and M-Schema), which is the format the model was fine-tuned on. Nothing here
ever sees connection credentials: inputs are the question, dialect, schema text,
memories and conversation turns.
"""

import re
from dataclasses import dataclass

from app.llm.ollama import OllamaLLM

_TEMPLATE = """你是一名{dialect}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用{dialect}知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{schema}

【参考信息】
{evidence}

【用户问题】
{question}

```sql"""

_FENCE = re.compile(r"```(?:sql|SQL|postgresql|postgres)?\s*\n?(.*?)(?:```|$)", re.DOTALL)


@dataclass(frozen=True)
class Turn:
    """A previous question in this conversation and the SQL that answered it."""

    question: str
    sql: str


def build_evidence(
    memories: list[str],
    history: list[Turn],
    hint: str | None = None,
    extra: str | None = None,
) -> str:
    parts = ["Return exactly one read-only SELECT statement and nothing else."]
    if hint:
        parts.append(f"Hint: {hint}")
    if memories:
        parts.append(
            "Notes learned in earlier sessions (may be outdated; the schema above is "
            "authoritative, ignore any note that conflicts with it):\n"
            + "\n".join(f"- {m}" for m in memories)
        )
    if history:
        parts.append(
            "Earlier questions in this conversation (the new question may refer to them):\n"
            + "\n".join(f"Q: {t.question}\nSQL: {t.sql}" for t in history)
        )
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


def build_prompt(dialect: str, question: str, schema: str, evidence: str) -> str:
    return _TEMPLATE.format(dialect=dialect, question=question, schema=schema, evidence=evidence)


def extract_sql(output: str) -> str:
    """Pull the SQL out of a model reply: drop code fences, labels and trailing semicolons."""
    text = output.strip()
    if text.startswith("```"):
        text = _FENCE.match(text).group(1)
    elif "```" in text:
        # The prompt ends with an open ```sql fence, so the reply is SQL then a closing fence.
        text = text.split("```", 1)[0]
    text = re.sub(r"^\s*(?:SQL|Answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().rstrip(";").strip()


def generate_sql(
    llm: OllamaLLM,
    dialect: str,
    question: str,
    schema: str,
    evidence: str,
) -> str:
    return extract_sql(llm.complete(build_prompt(dialect, question, schema, evidence)))
