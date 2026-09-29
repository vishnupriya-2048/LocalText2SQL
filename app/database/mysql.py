from sqlalchemy import Connection

from app.database.base import Database


class MySQLDatabase(Database):
    dialect = "mysql"
    display_name = "MySQL"
    driver = "mysql+pymysql"
    sqlglot_dialect = "mysql"
    default_port = 3306

    def _connect_args(self) -> dict:
        return {"connect_timeout": 10}

    def _begin_readonly(self, conn: Connection, timeout_s: int) -> None:
        # max_execution_time (ms) only limits SELECTs; timeout_s is an int from config.
        conn.exec_driver_sql(f"SET SESSION max_execution_time = {int(timeout_s) * 1000}")
        conn.exec_driver_sql("START TRANSACTION READ ONLY")

    def _user_schemas(self, conn: Connection) -> list[str]:
        # In MySQL a schema is a database; the app works within the connected one.
        db = conn.exec_driver_sql("SELECT DATABASE()").scalar()
        return [db] if db else []
