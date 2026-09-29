from app.database.base import (
    ConnectionConfig,
    Database,
    DatabaseError,
    QueryExecutionError,
    QueryResult,
)
from app.database.mysql import MySQLDatabase
from app.database.postgres import PostgresDatabase
from app.database.sqlserver import SqlServerDatabase

DATABASE_TYPES: dict[str, type[Database]] = {
    cls.dialect: cls for cls in (PostgresDatabase, MySQLDatabase, SqlServerDatabase)
}


def connect(config: ConnectionConfig) -> Database:
    """Create a database for `config` and verify it is reachable."""
    cls = DATABASE_TYPES.get(config.dialect)
    if cls is None:
        raise DatabaseError(f"Unknown database type: {config.dialect!r}")
    if not cls.supported:
        raise DatabaseError(f"{cls.display_name} is not supported yet; use PostgreSQL.")
    db = cls(config)
    try:
        db.test_connection()
    except DatabaseError:
        db.close()
        raise
    return db


__all__ = [
    "DATABASE_TYPES",
    "ConnectionConfig",
    "Database",
    "DatabaseError",
    "QueryExecutionError",
    "QueryResult",
    "connect",
]
