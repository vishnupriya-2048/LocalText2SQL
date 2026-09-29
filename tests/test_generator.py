import pytest

from app.sql.generator import Turn, build_evidence, build_prompt, extract_sql
from app.sql.repair import repair_evidence


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("SELECT 1;", "SELECT 1"),
        ("```sql\nSELECT a\nFROM t;\n```", "SELECT a\nFROM t"),
        ("```\nSELECT 1\n```\nThis query returns one.", "SELECT 1"),
        ("SQL: SELECT 2", "SELECT 2"),
        # Prompt ends with an open ```sql fence, so the model often only closes it.
        ("SELECT name FROM t\n```", "SELECT name FROM t"),
        ("  \n SELECT 1 ;; ", "SELECT 1"),
    ],
)
def test_extract_sql(raw, expected):
    assert extract_sql(raw) == expected


def test_prompt_contains_all_parts():
    evidence = build_evidence(
        memories=["'VIP' means lifetime spend over 10000"],
        history=[Turn("how many customers?", "SELECT count(*) FROM customers")],
        hint="amount is in cents",
    )
    prompt = build_prompt("PostgreSQL", "list VIP customers", "【DB_ID】 shop", evidence)
    for piece in [
        "PostgreSQL",
        "list VIP customers",
        "【DB_ID】 shop",
        "VIP' means lifetime spend",
        "schema above is authoritative",
        "SELECT count(*) FROM customers",
        "amount is in cents",
    ]:
        assert piece in prompt


def test_repair_evidence_includes_error_and_failed_sql():
    ev = repair_evidence("base", "SELECT nme FROM t", 'column "nme" does not exist')
    assert "SELECT nme FROM t" in ev and 'column "nme" does not exist' in ev and ev.startswith("base")
