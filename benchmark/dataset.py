"""Load BIRD Mini-Dev style question files (JSON array or JSONL).

Expected fields per item (BIRD names): question_id, db_id, question, evidence,
SQL, difficulty. Nothing BIRD-specific leaks into the application itself.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    question_id: str
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str


def load_examples(path: Path, limit: int | None = None, db_ids: set[str] | None = None) -> list[Example]:
    text = path.read_text(encoding="utf-8")
    items = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    examples = [
        Example(
            question_id=str(item.get("question_id", i)),
            db_id=item["db_id"],
            question=item["question"],
            evidence=item.get("evidence") or "",
            gold_sql=item.get("SQL") or item["sql"],
            difficulty=item.get("difficulty", "unknown"),
        )
        for i, item in enumerate(items)
    ]
    if db_ids:
        examples = [e for e in examples if e.db_id in db_ids]
    return examples[:limit] if limit else examples
