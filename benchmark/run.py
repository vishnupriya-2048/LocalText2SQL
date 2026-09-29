"""Measure execution accuracy of pipeline variants on BIRD Mini-Dev (or any compatible file).

    $env:BENCHMARK_DATABASE_URL = "postgresql://user:pass@localhost:5432/{db_id}"
    python -m benchmark.run --data mini_dev_postgresql.json --limit 20

`{db_id}` in the URL is replaced per question; without it every question uses
the same database. The URL comes from an environment variable so the password
never appears on the command line.

Variants (all use the application's own TextToSQLAgent):
  baseline          XiYanSQL with the full schema, no memory, no repair
  schema_retrieval  + relevant-schema selection
  hindsight         + Hindsight recall
  full_agent        + SQL error-repair loop
Runs are sequential (one LLM call at a time). By default nothing is written to
memory; --retain lets variants store their own executed queries (never gold SQL).
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from sqlalchemy.engine import make_url

from app.agent.graph import TextToSQLAgent
from app.agent.nodes import AgentDeps, AgentOptions
from app.config import Settings, load_settings
from app.database import (
    ConnectionConfig,
    Database,
    DatabaseError,
    QueryExecutionError,
    connect,
)
from app.llm.ollama import LLMError, OllamaLLM
from app.memory.hindsight import Memory, create_memory
from app.schema.extractor import DatabaseSchema
from benchmark.dataset import Example, load_examples
from benchmark.metrics import Outcome, results_match, summarize

log = logging.getLogger("benchmark")

GOLD_MAX_ROWS = 100_000


def variant_options(settings: Settings, retain: bool) -> dict[str, AgentOptions]:
    return {
        "baseline": AgentOptions(False, False, False, 0),
        "schema_retrieval": AgentOptions(True, False, False, 0),
        "hindsight": AgentOptions(True, True, retain, 0),
        "full_agent": AgentOptions(True, True, retain, settings.max_repair_attempts),
    }


class Databases:
    """One open connection + schema per db_id, reused across variants."""

    def __init__(self, url_template: str):
        self._template = url_template
        self._open: dict[str, tuple[Database, DatabaseSchema]] = {}

    def get(self, db_id: str) -> tuple[Database, DatabaseSchema]:
        if db_id not in self._open:
            url = make_url(self._template.replace("{db_id}", db_id))
            db = connect(
                ConnectionConfig(
                    dialect=url.get_backend_name(),
                    host=f"{url.host or 'localhost'}:{url.port or 5432}/{url.database}",
                    username=url.username or "",
                    password=url.password or "",
                )
            )
            self._open[db_id] = (db, db.get_schema())
        return self._open[db_id]

    def close(self) -> None:
        for db, _ in self._open.values():
            db.close()


def evaluate(
    example: Example,
    variant: str,
    options: AgentOptions,
    dbs: Databases,
    llm: OllamaLLM,
    memory: Memory,
    settings: Settings,
    use_evidence: bool,
    gold_cache: dict[str, list[tuple]],
) -> Outcome:
    db, schema = dbs.get(example.db_id)
    if example.question_id not in gold_cache:
        gold = db.execute_readonly(example.gold_sql, GOLD_MAX_ROWS, settings.query_timeout_s * 4)
        gold_cache[example.question_id] = gold.rows
    agent = TextToSQLAgent(AgentDeps(db, schema, llm, memory, settings, options))
    start = time.perf_counter()
    error = None
    try:
        state = agent.ask(example.question, hint=example.evidence if use_evidence else None)
    except LLMError as exc:
        state, error = {"status": "failed", "generated_sql": ""}, str(exc)
    result = state.get("query_result")
    return Outcome(
        variant=variant,
        question_id=example.question_id,
        difficulty=example.difficulty,
        correct=bool(result) and results_match(result.rows, gold_cache[example.question_id]),
        status=state.get("status", "failed"),
        predicted_sql=state.get("generated_sql", ""),
        error=error or state.get("execution_error") or (state.get("validation_result") or {}).get("reason") or None,
        repair_attempts=state.get("repair_attempts", 0),
        seconds=round(time.perf_counter() - start, 1),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Text-to-SQL execution-accuracy benchmark")
    parser.add_argument("--data", type=Path, required=True, help="BIRD Mini-Dev JSON/JSONL file")
    parser.add_argument("--limit", type=int, help="evaluate only the first N questions")
    parser.add_argument("--db-ids", help="comma-separated db_id filter")
    parser.add_argument("--variants", default="baseline,schema_retrieval,hindsight,full_agent")
    parser.add_argument("--no-evidence", action="store_true", help="do not pass BIRD evidence")
    parser.add_argument("--retain", action="store_true", help="let variants write to memory")
    parser.add_argument("--out", type=Path, default=Path("benchmark_results.jsonl"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    url_template = os.getenv("BENCHMARK_DATABASE_URL")
    if not url_template:
        print("Set BENCHMARK_DATABASE_URL (see module docstring).", file=sys.stderr)
        return 2

    settings = load_settings()
    llm = OllamaLLM(settings)
    llm.check()
    memory, warning = create_memory(settings)
    if warning:
        log.warning("%s The 'hindsight' variant will behave like 'schema_retrieval'.", warning)

    all_variants = variant_options(settings, args.retain)
    chosen = [v.strip() for v in args.variants.split(",")]
    unknown = set(chosen) - set(all_variants)
    if unknown:
        print(f"Unknown variants: {sorted(unknown)}", file=sys.stderr)
        return 2
    examples = load_examples(
        args.data, args.limit, set(args.db_ids.split(",")) if args.db_ids else None
    )
    log.info("%d questions x %d variants", len(examples), len(chosen))

    dbs = Databases(url_template)
    gold_cache: dict[str, list[tuple]] = {}
    outcomes: list[Outcome] = []
    try:
        with args.out.open("w", encoding="utf-8") as out:
            for variant in chosen:
                for i, ex in enumerate(examples, 1):
                    try:
                        outcome = evaluate(
                            ex, variant, all_variants[variant], dbs, llm, memory,
                            settings, not args.no_evidence, gold_cache,
                        )  # fmt: skip
                    except (DatabaseError, QueryExecutionError) as exc:
                        log.error("[%s] %s skipped: %s", variant, ex.question_id, exc)
                        continue
                    outcomes.append(outcome)
                    out.write(json.dumps(asdict(outcome)) + "\n")
                    out.flush()
                    log.info(
                        "[%s] %d/%d q=%s %s (%.0fs)", variant, i, len(examples),
                        ex.question_id, "OK" if outcome.correct else "wrong", outcome.seconds,
                    )  # fmt: skip
    finally:
        dbs.close()

    print(json.dumps(summarize(outcomes), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
