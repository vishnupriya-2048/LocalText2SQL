import pytest

from app.sql.validator import validate_sql

PG = "postgres"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select id, name from customers where name = 'delete me' order by id limit 5",
        "WITH t AS (SELECT id FROM orders) SELECT count(*) FROM t",
        "SELECT a FROM x UNION SELECT a FROM y",
        'SELECT "update" FROM "Delete"',  # quoted identifiers, not keywords
        (
            "SELECT c.name, sum(o.total) FROM customers c JOIN orders o ON o.customer_id = c.id "
            "GROUP BY c.name HAVING sum(o.total) > 100"
        ),
        "SELECT created_at, updated_by FROM audit",  # keyword substrings in identifiers
    ],
)
def test_accepts_read_only_queries(sql):
    result = validate_sql(sql, PG)
    assert result.ok, result.reason


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET a = 1",
        "DELETE FROM t",
        "DROP TABLE t",
        "ALTER TABLE t ADD COLUMN b int",
        "CREATE TABLE t (a int)",
        "TRUNCATE t",
        "GRANT SELECT ON t TO bob",
        "REVOKE SELECT ON t FROM bob",
        "SELECT 1; DROP TABLE t",
        "WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d",
        "SELECT * INTO new_table FROM t",
        "SELECT pg_sleep(100)",
        "SELECT pg_read_file('/etc/passwd')",
        "COPY t TO '/tmp/x'",
    ],
)
def test_rejects_unsafe_statements_as_not_repairable(sql):
    result = validate_sql(sql, PG)
    assert not result.ok
    assert not result.repairable, result.reason


def test_multiple_selects_rejected_but_repairable():
    result = validate_sql("SELECT 1; SELECT 2", PG)
    assert not result.ok and result.repairable
    assert "Multiple" in result.reason


@pytest.mark.parametrize("sql", ["", "   ", "SELEC * FROM", "SELECT FROM WHERE ("])
def test_garbage_is_repairable(sql):
    result = validate_sql(sql, PG)
    assert not result.ok and result.repairable


def test_row_locks_rejected():
    assert not validate_sql("SELECT * FROM t FOR UPDATE", PG).ok


@pytest.mark.parametrize("sql", ["SET ROLE admin", "SHOW search_path", "DO $$ BEGIN END $$"])
def test_non_query_statements_rejected(sql):
    assert not validate_sql(sql, PG).ok
