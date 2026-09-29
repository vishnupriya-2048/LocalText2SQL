"""Database abstraction layer.

Credentials enter through `ConnectionConfig`, go into the SQLAlchemy engine,
and nowhere else: they are never part of agent state, prompts, memory, logs or
error messages (the password is excluded from repr and redacted from errors).
"""

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from sqlalchemy import (
    Connection,
    Engine,
    column,
    create_engine,
    null,
    select,
    table,
    text,
)
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError, DBAPIError, SQLAlchemyError

from app.schema.extractor import DatabaseSchema, Table, extract_schema


class DatabaseError(Exception):
    """A connection or query failure, with any secret removed from the message."""


class QueryExecutionError(DatabaseError):
    """The database rejected or failed a query (message is the driver's own error)."""


@dataclass(frozen=True)
class ConnectionConfig:
    dialect: str
    host: str  # "host", "host:port", "host:port/database", or a URL without password
    username: str
    password: str = field(repr=False)
    database: str | None = None


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool
    elapsed_ms: int


def redact(message: str, secret: str | None) -> str:
    return message.replace(secret, "***") if secret else message


class Database(ABC):
    """One live connection target. Subclasses supply dialect-specific pieces."""

    dialect: ClassVar[str]  # registry key
    display_name: ClassVar[str]
    driver: ClassVar[str]  # SQLAlchemy drivername
    sqlglot_dialect: ClassVar[str]
    default_port: ClassVar[int]
    supported: ClassVar[bool] = True
    # Dialect-specific guidance appended to the model's reference notes.
    prompt_hints: ClassVar[str] = ""

    def __init__(self, config: ConnectionConfig):
        self._password = config.password
        url = self._build_url(config)
        if not url.database:
            raise DatabaseError("A database name is required (e.g. host:5432/mydb).")
        self.host = url.host or "localhost"
        self.port = url.port or self.default_port
        self.database_name = url.database
        self._engine: Engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_size=1,
            max_overflow=1,
            hide_parameters=True,
            connect_args=self._connect_args(),
        )

    def _build_url(self, config: ConnectionConfig) -> URL:
        raw = config.host.strip()
        try:
            if "://" in raw:
                parsed = make_url(raw)
            else:
                parsed = make_url(f"{self.driver}://{raw}")
        except ArgumentError:
            # The parser's message echoes the input, which may contain a password.
            raise DatabaseError("Could not parse the connection string / host.") from None
        return URL.create(
            self.driver,
            username=config.username or parsed.username,
            password=config.password or parsed.password,
            host=parsed.host,
            port=parsed.port,
            database=config.database or parsed.database,
        )

    def _connect_args(self) -> dict:
        return {}

    # -- identity (never includes credentials) ---------------------------------

    @property
    def display_target(self) -> str:
        return f"{self.display_name} {self.host}:{self.port}/{self.database_name}"

    @property
    def identifier(self) -> str:
        """Stable id for this database, used to scope long-term memories."""
        key = f"{self.dialect}|{self.host.lower()}|{self.port}|{self.database_name}"
        return f"{self.dialect}-{self.database_name}-{hashlib.sha256(key.encode()).hexdigest()[:12]}"

    # -- operations -------------------------------------------------------------

    def test_connection(self) -> None:
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            raise DatabaseError(self._clean(exc)) from None

    def get_schema(self) -> DatabaseSchema:
        try:
            with self._engine.connect() as conn:
                schemas = self._user_schemas(conn)
                default_schema = self._engine.dialect.default_schema_name
            return extract_schema(self._engine, schemas, default_schema, self.database_name)
        except SQLAlchemyError as exc:
            raise DatabaseError(self._clean(exc)) from None

    def sample_rows(self, tbl: Table, limit: int) -> list[tuple]:
        # Columns of a type SQLAlchemy could not identify (e.g. geography) may be unreadable
        # by the driver, so they are sampled as NULL to keep row positions aligned.
        readable = [c for c in tbl.columns if c.type.upper() != "NULL"]
        source = table(tbl.name, *[column(c.name) for c in readable], schema=tbl.schema)
        cols = [
            source.c[c.name] if c.type.upper() != "NULL" else null().label(c.name)
            for c in tbl.columns
        ]
        return self._read(select(*cols).select_from(source).limit(limit))

    def distinct_values(self, tbl: Table, column_name: str, limit: int, scan_rows: int) -> list:
        """Up to `limit` distinct non-null values of a column, looking at `scan_rows` rows at most."""
        col = column(column_name)
        scanned = (
            select(col)
            .select_from(table(tbl.name, col, schema=tbl.schema))
            .where(col.is_not(None))
            .limit(scan_rows)
            .subquery()
        )
        return [r[0] for r in self._read(select(scanned.c[column_name]).distinct().limit(limit))]

    def _read(self, stmt) -> list[tuple]:
        """Run an app-built (not model-generated) read in a read-only transaction."""
        try:
            with self._engine.connect() as conn:
                conn.begin()
                try:
                    self._begin_readonly(conn, timeout_s=10)
                    return [tuple(r) for r in conn.execute(stmt).fetchall()]
                finally:
                    conn.rollback()
        except SQLAlchemyError as exc:
            raise DatabaseError(self._clean(exc)) from None

    def execute_readonly(self, sql: str, max_rows: int, timeout_s: int) -> QueryResult:
        """Run already-validated SQL inside a read-only transaction that is always rolled back."""
        start = time.perf_counter()
        try:
            with self._engine.connect() as conn:
                conn.begin()
                try:
                    self._begin_readonly(conn, timeout_s)
                    cursor = conn.exec_driver_sql(sql)
                    columns = list(cursor.keys()) if cursor.returns_rows else []
                    rows = [tuple(r) for r in cursor.fetchmany(max_rows + 1)] if columns else []
                finally:
                    conn.rollback()
        except DBAPIError as exc:
            raise QueryExecutionError(self._clean_driver_error(exc)) from None
        except SQLAlchemyError as exc:
            raise QueryExecutionError(self._clean(exc)) from None
        return QueryResult(
            columns=columns,
            rows=rows[:max_rows],
            truncated=len(rows) > max_rows,
            elapsed_ms=int((time.perf_counter() - start) * 1000),
        )

    def close(self) -> None:
        self._engine.dispose()

    # -- dialect hooks ----------------------------------------------------------

    @abstractmethod
    def _begin_readonly(self, conn: Connection, timeout_s: int) -> None:
        """Make the open transaction read-only and apply a statement timeout."""

    @abstractmethod
    def _user_schemas(self, conn: Connection) -> list[str]:
        """Schemas holding user tables (excluding system schemas)."""

    # -- error hygiene ----------------------------------------------------------

    def _clean(self, exc: Exception) -> str:
        return redact(str(exc).split("\n(Background on this error")[0], self._password)

    def _clean_driver_error(self, exc: DBAPIError) -> str:
        return redact(str(exc.orig).strip(), self._password)
