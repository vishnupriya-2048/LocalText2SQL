from app.schema.extractor import Column, DatabaseSchema, Table
from app.schema.retriever import select_tables


def _table(name, schema="Sales", cols=("Id", "Name")):
    return Table(name=name, schema=schema, columns=[Column(c, "INTEGER") for c in cols])


def test_archive_copies_are_never_selected_when_base_table_exists():
    schema = DatabaseSchema(
        "db",
        "mssql",
        [
            _table("Customers"),
            _table("Customers_Archive"),
            _table("Orders", cols=("OrderId", "CustomerId")),
        ],
    )
    chosen = select_tables(schema, "customers", [], max_tables=6, full_threshold=1)
    assert "Customers_Archive" not in [t.name for t in chosen]
    assert "Customers" in [t.name for t in chosen]


def test_small_schema_still_returned_whole_without_archives():
    schema = DatabaseSchema("db", "mssql", [_table("A"), _table("A_Archive"), _table("B")])
    chosen = select_tables(schema, "anything", [], max_tables=6, full_threshold=8)
    assert sorted(t.name for t in chosen) == ["A", "B"]


def test_hidden_and_unknown_type_columns_are_left_out_of_the_prompt():
    from app.schema.retriever import format_mschema

    table = Table(
        name="Cities",
        schema="Application",
        columns=[
            Column("CityID", "INTEGER"),
            Column("Location", "NULL"),
            Column("LastEditedBy", "INTEGER"),
        ],
    )
    text = format_mschema("db", [table], frozenset({"lasteditedby"}))
    assert "CityID" in text
    assert "Location" not in text and "LastEditedBy" not in text


def test_archive_table_without_base_table_is_kept():
    schema = DatabaseSchema("db", "mssql", [_table("Old_Archive"), _table("B")])
    chosen = select_tables(schema, "anything", [], max_tables=6, full_threshold=8)
    assert sorted(t.name for t in chosen) == ["B", "Old_Archive"]
