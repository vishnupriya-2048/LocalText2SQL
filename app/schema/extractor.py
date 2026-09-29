"""Database schema model and extraction via SQLAlchemy reflection.

Reflection goes through SQLAlchemy's Inspector, so the same code serves every
dialect SQLAlchemy supports; dialect classes only choose which schemas to read.
"""

from dataclasses import dataclass, field

from sqlalchemy import Engine, inspect
from sqlalchemy.engine.reflection import ObjectKind


@dataclass
class Column:
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False
    comment: str | None = None

    @property
    def is_text(self) -> bool:
        upper = self.type.upper()
        return any(k in upper for k in ("CHAR", "TEXT", "ENUM", "STRING"))


@dataclass(frozen=True)
class ForeignKey:
    columns: tuple[str, ...]
    ref_table: str  # qualified name, same format as Table.qualified_name
    ref_columns: tuple[str, ...]


@dataclass
class Table:
    name: str
    schema: str | None  # None for the default schema
    columns: list[Column]
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    comment: str | None = None
    is_view: bool = False
    # Filled lazily by the agent (only for tables actually sent to the model).
    sample_rows: list[tuple] | None = None
    # Distinct values of text columns, so the model sees real category values
    # (e.g. 'Withdraw', not a guessed 'Withdrawal') even if the first rows lack them.
    column_values: dict[str, list] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}" if self.schema else self.name


@dataclass
class DatabaseSchema:
    database_name: str
    dialect: str
    tables: list[Table]

    def table(self, qualified_name: str) -> Table | None:
        lowered = qualified_name.lower()
        for t in self.tables:
            if t.qualified_name.lower() == lowered:
                return t
        return None


def _qualify(schema: str | None, name: str) -> str:
    return f"{schema}.{name}" if schema else name


def extract_schema(
    engine: Engine, schemas: list[str], default_schema: str | None, database_name: str
) -> DatabaseSchema:
    """Reflect tables, views, columns, primary keys, foreign keys and comments."""
    insp = inspect(engine)
    views: set[tuple[str | None, str]] = set()
    tables: list[Table] = []
    for schema in schemas:
        # Reflect relative to None for the default schema so names stay unqualified.
        reflect_schema = None if schema == default_schema else schema
        columns = insp.get_multi_columns(schema=reflect_schema, kind=ObjectKind.ANY)
        pks = insp.get_multi_pk_constraint(schema=reflect_schema, kind=ObjectKind.ANY)
        fks = insp.get_multi_foreign_keys(schema=reflect_schema, kind=ObjectKind.ANY)
        comments = insp.get_multi_table_comment(schema=reflect_schema, kind=ObjectKind.ANY)
        views.update((reflect_schema, v) for v in insp.get_view_names(schema=reflect_schema))
        if engine.dialect.name == "postgresql":
            views.update(
                (reflect_schema, v)
                for v in insp.get_materialized_view_names(schema=reflect_schema)
            )

        for key in sorted(columns, key=lambda k: k[1]):
            _, name = key
            pk_cols = set((pks.get(key) or {}).get("constrained_columns") or [])
            cols = [
                Column(
                    name=c["name"],
                    type=str(c["type"]),
                    nullable=bool(c.get("nullable", True)),
                    primary_key=c["name"] in pk_cols,
                    comment=c.get("comment"),
                )
                for c in columns[key]
            ]
            foreign_keys = [
                ForeignKey(
                    columns=tuple(fk["constrained_columns"]),
                    ref_table=_qualify(
                        fk.get("referred_schema") or reflect_schema, fk["referred_table"]
                    ),
                    ref_columns=tuple(fk["referred_columns"]),
                )
                for fk in fks.get(key) or []
            ]
            tables.append(
                Table(
                    name=name,
                    schema=reflect_schema,
                    columns=cols,
                    foreign_keys=foreign_keys,
                    comment=(comments.get(key) or {}).get("text"),
                    is_view=(reflect_schema, name) in views,
                )
            )
    return DatabaseSchema(database_name=database_name, dialect=engine.dialect.name, tables=tables)
