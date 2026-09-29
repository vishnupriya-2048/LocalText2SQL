from sqlalchemy import Connection, text

from app.database.base import Database

_SYSTEM_SCHEMAS = {"information_schema", "pg_catalog"}


class PostgresDatabase(Database):
    dialect = "postgresql"
    display_name = "PostgreSQL"
    driver = "postgresql+psycopg"
    sqlglot_dialect = "postgres"
    default_port = 5432

    def _connect_args(self) -> dict:
        return {"connect_timeout": 10, "application_name": "localtext2sql"}

    def _begin_readonly(self, conn: Connection, timeout_s: int) -> None:
        conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        # SET LOCAL does not accept bind parameters; timeout_s is an int from config.
        conn.exec_driver_sql(f"SET LOCAL statement_timeout = {int(timeout_s) * 1000}")

    def _user_schemas(self, conn: Connection) -> list[str]:
        rows = conn.execute(
            text(
                "SELECT nspname FROM pg_namespace "
                "WHERE nspname NOT LIKE 'pg\\_%' AND has_schema_privilege(nspname, 'USAGE') "
                "ORDER BY nspname"
            )
        ).scalars()
        return [s for s in rows if s not in _SYSTEM_SCHEMAS]
