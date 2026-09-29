"""Execution accuracy, as defined by BIRD: predicted and gold result sets are equal."""

from collections import defaultdict
from dataclasses import dataclass


def results_match(predicted_rows: list[tuple], gold_rows: list[tuple]) -> bool:
    return set(map(tuple, predicted_rows)) == set(map(tuple, gold_rows))


@dataclass
class Outcome:
    variant: str
    question_id: str
    difficulty: str
    correct: bool
    status: str
    predicted_sql: str
    error: str | None
    repair_attempts: int
    seconds: float


def summarize(outcomes: list[Outcome]) -> dict[str, dict[str, str]]:
    """{variant: {"overall": "12/20 (60.0%)", "simple": ..., ...}}"""
    buckets: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for o in outcomes:
        buckets[o.variant]["overall"].append(o.correct)
        buckets[o.variant][o.difficulty].append(o.correct)
    return {
        variant: {
            key: f"{sum(vals)}/{len(vals)} ({100 * sum(vals) / len(vals):.1f}%)"
            for key, vals in groups.items()
        }
        for variant, groups in buckets.items()
    }
