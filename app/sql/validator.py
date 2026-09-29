"""Programmatic read-only SQL validation (independent of anything the LLM says).

Two layers:
1. Token scan for write/DDL/DCL keywords (string literals and quoted
   identifiers are ignored, so `WHERE note = 'delete'` is fine).
2. Parse with sqlglot: exactly one statement, a query at the root, no
   data-modifying node anywhere (e.g. inside a CTE), no SELECT INTO, no row
   locks, no dangerous server functions.

Execution additionally runs inside a read-only transaction, so this is not the
only line of defence.
"""

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError
from sqlglot.tokens import TokenType

FORBIDDEN_KEYWORDS = frozenset(
    {
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
        "GRANT", "REVOKE", "MERGE", "COPY", "CALL", "EXEC", "EXECUTE", "VACUUM",
        "REINDEX", "CLUSTER", "LOCK", "UPSERT", "RENAME",
    }
)  # fmt: skip

_WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Merge, exp.Grant, exp.Revoke, exp.Copy,
    exp.Set, exp.Use, exp.Transaction, exp.Commit, exp.Rollback,
)  # fmt: skip

DANGEROUS_FUNCTIONS = frozenset(
    {
        "pg_sleep", "pg_sleep_for", "pg_sleep_until", "pg_read_file",
        "pg_read_binary_file", "pg_ls_dir", "pg_stat_file", "lo_import", "lo_export",
        "dblink", "dblink_exec", "pg_terminate_backend", "pg_cancel_backend",
        "pg_reload_conf", "set_config", "nextval", "setval", "pg_advisory_lock",
        "pg_advisory_xact_lock", "query_to_xml", "sleep", "benchmark", "load_file",
        "xp_cmdshell", "openrowset", "opendatasource",
    }
)  # fmt: skip

_SKIP_TOKENS = {TokenType.STRING, TokenType.IDENTIFIER, TokenType.NATIONAL_STRING}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str = ""
    # False for safety rejections: those stop the agent instead of being "repaired".
    repairable: bool = True


def _fail(reason: str, repairable: bool) -> ValidationResult:
    return ValidationResult(ok=False, reason=reason, repairable=repairable)


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def validate_sql(sql: str, dialect: str) -> ValidationResult:
    """Check that `sql` is a single read-only query in the given sqlglot dialect."""
    if not sql or not sql.strip():
        return _fail("The model returned no SQL.", repairable=True)

    try:
        tokens = sqlglot.tokenize(sql, read=dialect)
    except TokenError as exc:
        return _fail(f"SQL could not be tokenized: {exc}", repairable=True)
    for tok in tokens:
        if tok.token_type not in _SKIP_TOKENS and tok.text.upper() in FORBIDDEN_KEYWORDS:
            return _fail(
                f"Only read-only queries are allowed; found forbidden operation "
                f"{tok.text.upper()}.",
                repairable=False,
            )

    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except (ParseError, TokenError) as exc:
        return _fail(f"SQL syntax error: {str(exc).splitlines()[0]}", repairable=True)

    if not statements:
        return _fail("The model returned no SQL.", repairable=True)
    if len(statements) > 1:
        return _fail("Multiple SQL statements are not allowed; return one query.", repairable=True)

    tree = statements[0]
    for node in tree.walk():
        if isinstance(node, _WRITE_NODES):
            return _fail(
                f"Only read-only queries are allowed; found {node.key.upper()} statement.",
                repairable=False,
            )
    if not isinstance(tree, exp.Query):
        # Includes exp.Command, sqlglot's node for statements it does not model (and
        # for typos like "SELEC"). Explicit write keywords were already refused
        # above; anything else is never executed, so let the model try again.
        return _fail(
            f"Only SELECT queries are allowed; got {tree.key.upper()}.", repairable=True
        )
    for select in tree.find_all(exp.Select):
        if select.args.get("into"):
            return _fail("SELECT INTO creates a table and is not allowed.", repairable=False)
        if select.args.get("locks"):
            return _fail("Row-locking clauses (FOR UPDATE/SHARE) are not allowed.", repairable=True)
    for func in tree.find_all(exp.Func):
        name = _function_name(func)
        if name in DANGEROUS_FUNCTIONS:
            return _fail(f"Function {name}() is not allowed.", repairable=False)
    return ValidationResult(ok=True)


def tables_in_sql(sql: str, dialect: str) -> list[str]:
    """Names of the tables a (valid) query reads from, excluding CTE names."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except (ParseError, TokenError):
        return []
    ctes = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    names = {
        ".".join(p for p in (t.db, t.name) if p) for t in tree.find_all(exp.Table)
    }
    return sorted(n for n in names if n and n not in ctes)
