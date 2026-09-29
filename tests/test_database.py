"""Database layer tests.

Unit tests run anywhere. Integration tests need a PostgreSQL database the test
may create a scratch schema in:

    $env:TEST_DATABASE_URL = "postgresql://user:pass@localhost:5432/dbname"
"""

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.database import ConnectionConfig, DatabaseError, QueryExecutionError, connect
from app.database.postgres import PostgresDatabase
from app.database.sqlserver import SqlServerDatabase
from app.schema.retriever import format_mschema

SECRET = "s3cr3t-pa55"


def cfg(host="db.example.com:6543/sales", **kw):
    return ConnectionConfig(
        **{
            "dialect": "postgresql",
            "host": host,
            "username": "u",
            "password": SECRET,
            **kw,
        }
    )


def test_password_not_in_repr_identifier_or_display():
    config = cfg()
    db = PostgresDatabase(config)
    assert SECRET not in repr(config)
    assert SECRET not in db.identifier and SECRET not in db.display_target
    assert (db.host, db.port, db.database_name) == ("db.example.com", 6543, "sales")


def test_identifier_is_stable_and_differs_per_database():
    a, b = PostgresDatabase(cfg()), PostgresDatabase(cfg())
    c = PostgresDatabase(cfg(host="db.example.com:6543/other"))
    assert a.identifier == b.identifier != c.identifier


def test_accepts_url_and_separate_database_field():
    db = PostgresDatabase(
        cfg(host="postgresql://db.example.com/ignored", database="real")
    )
    assert (db.port, db.database_name) == (5432, "real")


def test_missing_database_name_is_a_clear_error():
    with pytest.raises(DatabaseError, match="database name is required"):
        PostgresDatabase(cfg(host="localhost"))


def test_unknown_dialect_is_refused():
    with pytest.raises(DatabaseError, match="Unknown database type"):
        connect(cfg(dialect="oracle"))


def test_sqlserver_url_uses_windows_auth_without_username():
    db = SqlServerDatabase(
        cfg(
            dialect="mssql", host=r"localhost\SQLEXPRESS/shop", username="", password=""
        )
    )
    url = db._engine.url
    assert (url.host, url.database, url.username) == (
        r"localhost\SQLEXPRESS",
        "shop",
        None,
    )
    assert url.query["TrustServerCertificate"] == "yes"


def test_connection_failure_message_has_no_password():
    with pytest.raises(DatabaseError) as info:
        connect(cfg(host="127.0.0.1:1/nodb"))
    assert SECRET not in str(info.value)


# -- integration ---------------------------------------------------------------

TEST_URL = os.getenv("TEST_DATABASE_URL")
needs_pg = pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set")
SCHEMA = "t2s_it"


@pytest.fixture(scope="module")
def pg():
    url = make_url(TEST_URL).set(drivername="postgresql+psycopg")
    admin = create_engine(url)
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {SCHEMA}"))
        conn.execute(
            text(f"CREATE TABLE {SCHEMA}.customers (id int PRIMARY KEY, name text)")
        )
        conn.execute(
            text(
                f"CREATE TABLE {SCHEMA}.orders (id int PRIMARY KEY, "
                f"customer_id int REFERENCES {SCHEMA}.customers(id), total numeric)"
            )
        )
        conn.execute(
            text(f"COMMENT ON TABLE {SCHEMA}.orders IS 'one row per purchase'")
        )
        conn.execute(
            text(f"INSERT INTO {SCHEMA}.customers VALUES (1,'Alice'),(2,'Bob')")
        )
        conn.execute(
            text(f"INSERT INTO {SCHEMA}.orders VALUES (10,1,50),(11,1,70),(12,2,5)")
        )
    db = connect(
        ConnectionConfig(
            dialect="postgresql",
            host=f"{url.host or 'localhost'}:{url.port or 5432}/{url.database}",
            username=url.username or "",
            password=url.password or "",
        )
    )
    yield db
    db.close()
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    admin.dispose()


@needs_pg
def test_schema_extraction(pg):
    schema = pg.get_schema()
    orders = schema.table(f"{SCHEMA}.orders")
    assert orders is not None and orders.comment == "one row per purchase"
    assert [c.name for c in orders.columns] == ["id", "customer_id", "total"]
    assert orders.columns[0].primary_key
    assert orders.foreign_keys[0].ref_table == f"{SCHEMA}.customers"
    orders.sample_rows = pg.sample_rows(orders, 2)
    rendered = format_mschema(schema.database_name, [orders])
    assert f"{SCHEMA}.orders.customer_id={SCHEMA}.customers.id" in rendered


@needs_pg
def test_execute_readonly_returns_rows(pg):
    result = pg.execute_readonly(
        f"SELECT c.name, sum(o.total) AS spent FROM {SCHEMA}.customers c "
        f"JOIN {SCHEMA}.orders o ON o.customer_id = c.id GROUP BY c.name ORDER BY c.name",
        max_rows=10,
        timeout_s=5,
    )
    assert result.columns == ["name", "spent"]
    assert [(n, int(s)) for n, s in result.rows] == [("Alice", 120), ("Bob", 5)]
    assert not result.truncated


@needs_pg
def test_row_cap_truncates(pg):
    result = pg.execute_readonly(
        f"SELECT * FROM {SCHEMA}.orders", max_rows=2, timeout_s=5
    )
    assert len(result.rows) == 2 and result.truncated


@needs_pg
def test_writes_fail_even_if_validation_were_bypassed(pg):
    with pytest.raises(QueryExecutionError, match="read-only"):
        pg.execute_readonly(f"DELETE FROM {SCHEMA}.orders", max_rows=1, timeout_s=5)
    assert len(pg.execute_readonly(f"SELECT * FROM {SCHEMA}.orders", 10, 5).rows) == 3


@needs_pg
def test_statement_timeout(pg):
    with pytest.raises(QueryExecutionError, match="timeout"):
        pg.execute_readonly("SELECT pg_sleep(3)", max_rows=1, timeout_s=1)


@needs_pg
def test_database_errors_are_captured_for_repair(pg):
    with pytest.raises(QueryExecutionError, match="does not exist"):
        pg.execute_readonly(
            f"SELECT nme FROM {SCHEMA}.customers", max_rows=1, timeout_s=5
        )
