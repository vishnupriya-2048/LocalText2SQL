"""Pick the tables relevant to a question and render them in M-Schema format.

Retrieval is lexical (question/memory words vs. table, column and comment
names) plus foreign-key expansion. It needs no embedding model, which keeps
RAM free for the LLM on a CPU-only machine.
"""

import re

from app.schema.extractor import DatabaseSchema, Table

_WORD = re.compile(r"[a-z0-9]+")
_MAX_EXAMPLE_CHARS = 40


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: str) -> set[str]:
    # Split snake_case / camelCase identifiers into words too.
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower().replace("_", " ")
    return {_stem(w) for w in _WORD.findall(text) if len(w) > 1}


def _score(table: Table, words: set[str], text: str) -> float:
    score = 0.0
    if re.search(rf"\b{re.escape(table.name.lower())}\b", text):
        score += 5
    score += 3 * len(_tokens(table.name) & words)
    for col in table.columns:
        score += len(_tokens(col.name) & words)
    if table.comment:
        score += 0.5 * len(_tokens(table.comment) & words)
    return score


def _without_history_tables(tables: list[Table]) -> list[Table]:
    """Drop `<name>_Archive` copies (SQL Server temporal history) when `<name>` exists.

    They repeat the base table's columns, so they crowd real tables out of the prompt
    and the model may query stale history instead of current data.
    """
    names = {t.qualified_name.lower() for t in tables}
    return [
        t
        for t in tables
        if not (
            t.qualified_name.lower().endswith("_archive")
            and t.qualified_name.lower()[: -len("_archive")] in names
        )
    ]


def select_tables(
    schema: DatabaseSchema,
    question: str,
    hints: list[str],
    max_tables: int,
    full_threshold: int,
) -> list[Table]:
    """Return the tables to show the model, most relevant first.

    Small schemas are returned whole. `hints` (e.g. recalled memories and recent
    conversation) count at half weight so they steer but do not dominate.
    """
    tables = _without_history_tables(schema.tables)
    if len(tables) <= full_threshold:
        return list(tables)

    q_text = question.lower()
    q_words = _tokens(question)
    h_text = " ".join(hints).lower()
    h_words = _tokens(h_text)
    scored = sorted(
        (
            (
                (_score(t, q_words, q_text) + 0.5 * _score(t, h_words, h_text))
                * (0.5 if t.is_view else 1.0),  # views repeat base-table data
                t,
            )
            for t in tables
        ),
        key=lambda pair: -pair[0],
    )
    chosen = [t for s, t in scored if s > 0][:max_tables]
    if not chosen:
        # Nothing matched: show the first tables rather than nothing.
        return list(tables[:max_tables])

    # Add tables directly linked by foreign keys to the top matches, so joins are possible.
    names = {t.qualified_name for t in chosen}
    for tbl in list(chosen[:3]):
        linked = {fk.ref_table for fk in tbl.foreign_keys} | {
            t.qualified_name
            for t in tables
            if any(fk.ref_table == tbl.qualified_name for fk in t.foreign_keys)
        }
        for name in sorted(linked - names):
            if len(chosen) >= max_tables + 4:
                break
            linked_table = schema.table(name)
            if linked_table is not None:
                chosen.append(linked_table)
                names.add(name)
    return chosen


def _example_values(table: Table, col_index: int) -> list[str]:
    col_name = table.columns[col_index].name
    if col_name in table.column_values:
        raw = table.column_values[col_name]
    else:
        raw = [row[col_index] for row in table.sample_rows or []]
    values: list[str] = []
    for value in raw:
        if value is None:
            continue
        shown = str(value)
        if len(shown) > _MAX_EXAMPLE_CHARS:
            shown = shown[:_MAX_EXAMPLE_CHARS] + "..."
        if shown not in values:
            values.append(shown)
    return values


def format_mschema(
    database_name: str, tables: list[Table], hidden_columns: frozenset[str] = frozenset()
) -> str:
    """Render tables in XiYan's M-Schema format.

    Columns named in `hidden_columns` (audit noise) and columns of a type the reflection
    could not identify (shown as NULL, e.g. geography) are left out to save prompt space.
    """
    lines = [f"【DB_ID】 {database_name}", "【Schema】"]
    fk_lines: list[str] = []
    for table in tables:
        header = f"# Table: {table.qualified_name}"
        if table.comment:
            header += f", {table.comment}"
        lines += [header, "["]
        col_lines = []
        for i, col in enumerate(table.columns):
            if col.name.lower() in hidden_columns or col.type.upper() == "NULL":
                continue
            parts = [f"{col.name}:{col.type}"]
            if col.primary_key:
                parts.append("Primary Key")
            if col.comment:
                parts.append(col.comment)
            examples = _example_values(table, i)
            if examples:
                parts.append(f"Examples: [{', '.join(examples)}]")
            col_lines.append(f"({', '.join(parts)})")
        lines.append(",\n".join(col_lines))
        lines.append("]")
        for fk in table.foreign_keys:
            for src, dst in zip(fk.columns, fk.ref_columns):
                fk_lines.append(f"{table.qualified_name}.{src}={fk.ref_table}.{dst}")
    if fk_lines:
        lines += ["【Foreign keys】", *fk_lines]
    return "\n".join(lines)
