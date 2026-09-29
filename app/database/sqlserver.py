"""SQL Server support (pyodbc + Microsoft ODBC driver).

SQL Server has no read-only transaction mode, so read-only safety rests on the SQL
validator, a rollback after every query, and a query timeout. Connect with a login that
only has db_datareader. Leave Username empty to use Windows authentication (as SSMS does).
Named instances work as `localhost\\SQLEXPRESS/mydb`.
"""

from sqlalchemy import Connection
from sqlalchemy.engine import URL

from app.database.base import ConnectionConfig, Database, DatabaseError

_DRIVERS = ("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server")


def _odbc_driver() -> str:
    import pyodbc

    installed = pyodbc.drivers()
    for name in _DRIVERS:
        if name in installed:
            return name
    raise DatabaseError(
        "No Microsoft ODBC Driver for SQL Server found. "
        "Install 'ODBC Driver 18 for SQL Server' from Microsoft."
    )


class SqlServerDatabase(Database):
    dialect = "mssql"
    display_name = "SQL Server"
    driver = "mssql+pyodbc"
    sqlglot_dialect = "tsql"
    default_port = 1433
    prompt_hints = (
        "Write T-SQL for SQL Server: use SELECT TOP n (never LIMIT), keep the schema prefix "
        "on table names exactly as shown (e.g. Sales.Invoices), give every computed column "
        "an alias, and write CTEs without ORDER BY. When a query joins a large line-item "
        "table to a header table, first total the line items per header id in a CTE, then "
        "join that CTE to the header table. This is much faster than joining then grouping. "
        "Never use tables ending in _Archive."
    )

    def _build_url(self, config: ConnectionConfig) -> URL:
        # Certificate trust is on so local/dev servers with self-signed certs connect;
        # with no username, SQLAlchemy's pyodbc dialect uses Windows authentication.
        return super()._build_url(config).update_query_dict(
            {"driver": _odbc_driver(), "TrustServerCertificate": "yes"}
        )

    def _connect_args(self) -> dict:
        return {"timeout": 10}  # login timeout, seconds

    def _begin_readonly(self, conn: Connection, timeout_s: int) -> None:
        conn.connection.driver_connection.timeout = int(timeout_s)  # pyodbc query timeout
        conn.exec_driver_sql(f"SET LOCK_TIMEOUT {int(timeout_s) * 1000}")

    def _user_schemas(self, conn: Connection) -> list[str]:
        rows = conn.exec_driver_sql(
            "SELECT DISTINCT TABLE_SCHEMA FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_SCHEMA NOT IN ('sys', 'INFORMATION_SCHEMA') ORDER BY TABLE_SCHEMA"
        ).scalars()
        return list(rows)
