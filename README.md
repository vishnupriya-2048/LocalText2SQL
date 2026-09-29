# LocalText2SQL

A **fully local Text-to-SQL assistant for developers and data teams.** Point it at your own
PostgreSQL, MySQL or SQL Server database, ask questions in plain English, and get back the generated
SQL and its results. No cloud service is involved at any point.

- **Nothing leaves the machine.** The language model runs locally through Ollama, the agent is a local
  process, long-term memory is a local Hindsight server, and the UI is bound to `localhost`.
- **Built for data governance.** Schemas, questions, generated SQL and results never reach a
  third-party API. That makes it usable with client, customer or regulated data where sending prompts
  to a hosted LLM would not be acceptable.
- **Safe by construction.** Every generated statement is validated as a single read-only query before
  it runs, then executed inside a read-only transaction that is always rolled back.
- **Runs on a laptop.** Designed for a 16 GB Windows machine with CPU-only inference: one model, one
  LLM call at a time.

| Layer | Choice |
| --- | --- |
| Orchestration | **LangGraph**: the agent is an explicit state graph, not a hidden loop |
| SQL model | **XiYanSQL-QwenCoder-7B-2504** (Q4_K_M) served by **Ollama** |
| Long-term memory | **Hindsight**, one memory bank per database |
| Databases | PostgreSQL, MySQL, SQL Server (via SQLAlchemy) |
| SQL safety | **sqlglot** parsing plus a token scan, then a read-only transaction |
| UI | **Streamlit** web app, plus a terminal client |

---

## Contents

1. [Architecture](#architecture)
2. [Why Hindsight, and how it is used](#why-hindsight-and-how-it-is-used)
3. [Security model](#security-model)
4. [User guide](#user-guide)
5. [Configuration reference](#configuration-reference)
6. [Troubleshooting](#troubleshooting)
7. [Benchmark](#benchmark)
8. [Tests](#tests)
9. [Project layout and extending](#project-layout-and-extending)

---

## Architecture

```
                    ┌────────────────────────── your machine ──────────────────────────┐
  Browser / CLI ──► │  Streamlit UI  ─►  LangGraph agent  ─►  Ollama (XiYanSQL 7B)     │
                    │                        │    │                                    │
                    │                        │    └──►  Hindsight server (memory)      │
                    │                        ▼                                         │
                    │                  Database layer  ──►  your database             │
                    └───────────────────────────────────────────────────────────────────┘
```

### How a question flows

```
START → recall_memory → identify_relevant_schema → generate_sql → validate_sql
validate_sql    → ok → execute_sql
                → fixable (syntax, multiple statements) and repairs left → repair_sql
                → unsafe (write/DDL) or no repairs left → evaluate_result
execute_sql     → success → evaluate_result
                → DB error and repairs left → repair_sql → validate_sql
evaluate_result → retain_memory → END
```

| Node | What it does |
| --- | --- |
| `recall_memory` | Asks Hindsight for memories relevant to this question, from this database's own memory bank. |
| `identify_relevant_schema` | Chooses which tables to show the model (see below), adds a few example values per column, and renders them in XiYan's M-Schema format. |
| `generate_sql` | Builds XiYanSQL's official prompt (dialect, question, schema, notes, recent turns) and strips code fences from the reply. |
| `validate_sql` | Programmatic read-only check. Never skipped. |
| `execute_sql` | Runs the SQL in a read-only transaction with a timeout and a row cap. |
| `repair_sql` | Sends the question, schema, failed SQL and the database error back to the model. Capped by `MAX_REPAIR_ATTEMPTS`. |
| `evaluate_result` | Sets the status: success, empty, rejected or failed. |
| `retain_memory` | After a successful, non-empty result, stores what worked in Hindsight. Result rows are never stored. |

The graph state (`app/agent/state.py`) contains no credentials or connection details. Follow-up
questions receive the last `HISTORY_TURNS` question and SQL pairs.

### Schema selection

The model only sees the tables it needs, because a 7B model on CPU has a small effective context.

- Schemas with at most `SCHEMA_FULL_THRESHOLD` tables are sent whole.
- Larger schemas are filtered by matching the words of the question (and of recalled memories and recent
  turns) against table, column and comment names, then adding foreign-key neighbours so joins are
  possible. No embedding model is used, which keeps RAM free for the LLM.
- SQL Server `<name>_Archive` temporal history tables are excluded whenever `<name>` exists, and views
  score lower than base tables.
- Audit columns (`SCHEMA_HIDDEN_COLUMNS`) and columns whose type cannot be read (such as `geography`)
  are hidden from the prompt to save space.
- Example values (sampled rows and distinct values of text columns) are read once per table per session,
  so the model sees real values such as `'Withdraw'` rather than guessing `'Withdrawal'`.

### Database layer

`app/database/base.py` defines the abstraction. Each dialect supplies only what differs:

| | PostgreSQL | MySQL | SQL Server |
| --- | --- | --- | --- |
| Driver | `psycopg` | `pymysql` | `pyodbc` with Microsoft ODBC Driver 17/18 |
| Read-only execution | `SET TRANSACTION READ ONLY` | `START TRANSACTION READ ONLY` | none available; validator, rollback and timeout only |
| Timeout | `statement_timeout` | `max_execution_time` (SELECT only) | ODBC query timeout and `LOCK_TIMEOUT` |
| Schemas read | all visible, non-system | the connected database | all schemas containing tables |
| Authentication | user and password | user and password | user and password, or Windows authentication |

Schema reflection, validation and the agent itself are dialect-agnostic. The dialect also contributes
`prompt_hints`; for SQL Server these tell the model to write T-SQL (`TOP`, not `LIMIT`) and to total
line items per header row in a CTE before joining, which is far faster than join-then-group.

---

## Why Hindsight, and how it is used

A text-to-SQL model starts every question knowing nothing about your business. It does not know that
"active customer" means "ordered in the last 90 days", that "revenue" lives in one particular column, or
that the obvious join is the wrong one. Fixing this by fine-tuning is heavy; stuffing documentation into
every prompt wastes the small context window. A memory layer that **learns from use** fits better, and
Hindsight is that layer. It runs locally, so it does not compromise the data-governance goal.

### What the app uses Hindsight for

The app uses two Hindsight operations, **recall** and **retain**, plus a per-database bank.

| Component | Where | Behaviour |
| --- | --- | --- |
| **Bank per database** | `app/memory/hindsight.py` | Bank id is `text2sql-<dialect>-<database>-<hash>`, where the hash covers dialect, host, port and database name. It never includes username or password. Memories from one database are never recalled for another. |
| **Bank mission** | `HindsightMemory.retain` | The first retain for a bank sets a short mission telling Hindsight what to remember: table relationships, business terminology, ambiguous-question interpretations, corrections and approaches that worked. |
| **Recall** | `recall_memory` node | Fetches up to 8 memories for the question (`budget="low"`, 1024 tokens). They go into the prompt as notes labelled *possibly outdated*, and they also steer schema selection. The live schema always wins over a note. |
| **Retain (automatic)** | `retain_memory` node | After a successful, non-empty result, stores the question, the working SQL, the tables used, and any correction learned from a failed first attempt. |
| **Retain (manual)** | sidebar "Teach a fact" | Stores a fact you type, such as `'active customer' means ordered in the last 90 days`. |

### What is deliberately not stored

Failures, empty results, unsafe SQL, result rows, credentials and connection strings. Only interactions
that executed successfully and returned rows are retained, so memory does not accumulate wrong answers.

### Graceful degradation

Memory is optional. If `HINDSIGHT_API_URL` is unset or the server is unreachable, the app says
"Memory is off for this session" and works without it. A recall or retain failure never fails a
question.

### Running the Hindsight server

The server is a separate process; see the Hindsight documentation for installing and starting it. Its
server uses an LLM of its own to extract facts. To stay within 16 GB, point it at the same Ollama
instance and set `OLLAMA_MAX_LOADED_MODELS=1` so only one model is resident. Either reuse the XiYan
model or pick a small general model and accept the model swaps. Retains are synchronous by default
(`HINDSIGHT_RETAIN_ASYNC=false`), so memory extraction never competes with SQL generation for the CPU.

---

## Security model

- **Credentials:** the password exists only in the connection form and the SQLAlchemy engine for the
  current session. It is excluded from `repr`, redacted from error messages, and never placed in agent
  state, prompts, memory or logs. Nothing is written to disk.
- **Validation:** SQL generated by the model is always validated before execution
  (`app/sql/validator.py`). The validator:
  - scans tokens for INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, GRANT, REVOKE, MERGE, COPY
    and similar (string literals and quoted identifiers are ignored);
  - parses with sqlglot and requires exactly one statement with a query at the root;
  - rejects write nodes anywhere, including data-modifying CTEs;
  - rejects `SELECT INTO`, `FOR UPDATE`, and dangerous functions (`pg_sleep`, `pg_read_file`, `dblink`, ...).
- **Unsafe SQL is refused** with a clear message and is not "repaired" into something else.
- **Execution:** runs in a read-only transaction with a timeout and is always rolled back. SQL Server has
  no read-only transaction mode, so on that platform the validator and rollback are the guards.
- **Network:** the Streamlit server is bound to `localhost`, and usage telemetry is off
  (`.streamlit/config.toml`).
- **Use a read-only database login** regardless (`db_datareader` on SQL Server, a `SELECT`-only role
  elsewhere). Defence in depth means the app's checks are never the only line.

---

## User guide

### Prerequisites

- Windows, macOS or Linux with Python 3.12+ and [uv](https://docs.astral.sh/uv/) (or plain `pip`)
- [Ollama](https://ollama.com) installed and running
- About 8 GB of free RAM for the model, and roughly 5 GB of disk for it
- A database you can log in to. For SQL Server, Microsoft ODBC Driver 17 or 18
- Optional: a running Hindsight server for long-term memory

### Setup (one time)

```powershell
uv sync                                   # or: python -m venv .venv; .venv\Scripts\activate; pip install -r requirements-dev.txt
copy .env.example .env                    # optional settings; no database password goes here
ollama pull hf.co/mradermacher/XiYanSQL-QwenCoder-7B-2504-GGUF:Q4_K_M
.venv\Scripts\python check_env.py         # checks imports, Ollama + model, Hindsight
```

### Start the app

Make sure Ollama is running, then:

```powershell
.venv\Scripts\streamlit run app/main.py   # UI at http://localhost:8501
```

or use the terminal client:

```powershell
.venv\Scripts\python -m app.cli --type postgresql --host localhost:5432/mydb --user me   # prompts for the password
```

### Connect to a database

The connection screen asks for the database type, host, database, username and password. The connection
is tested before the chat opens, and the schema is read once per session.

| Type | Host field | Database | Username / password |
| --- | --- | --- | --- |
| PostgreSQL | `localhost:5432` or `localhost:5432/mydb` | your database | your login |
| MySQL | `localhost:3306` | your database | your login |
| SQL Server (default instance) | `localhost` or `localhost:1433` | your database | leave **both blank** for Windows authentication, or a SQL login |
| SQL Server (named instance, as shown in SSMS) | `localhost\SQLEXPRESS` | your database | as above |

Host also accepts a URL such as `postgresql://host:5432/mydb`. The **Database** field may be left empty
if the host field already includes it. The password is never saved.

### Ask questions

Type in plain English in the chat box. Each answer shows the generated SQL, the result table, timing,
and a **Details** panel listing the tables given to the model, any repair attempts and any memories used.
Expect 20 to 60 seconds per simple question on CPU, and several minutes for very large queries.

Examples of what works well:

- "How many rows are in each table?"
- "Which ten customers spent the most in 2023, and what share of the total is each?"
- "Show monthly revenue for the last 12 months and the change from the previous month."

Tips for good results:

- **Use the names in your schema.** The model matches your words to table and column names. Say
  "invoices" if the table is `Invoices`.
- **Be explicit about time ranges and thresholds**, for example "in 2015" or "more than 1 million".
- **Correct rather than restart.** Follow-ups see your recent questions and their SQL, so "use invoice
  lines, not orders" or "also add the country" works.
- **Break very large questions into steps.** Ask for the first part, then say "now add the ranking".
- **Teach business terms.** With Hindsight running, use the sidebar's *Teach a fact* box for definitions
  such as "revenue means the sum of ExtendedPrice on invoice lines".

Statuses you may see:

| Status | Meaning |
| --- | --- |
| success | The query ran and returned rows. |
| empty | The query ran and returned nothing. Check filters and spelling of values. |
| rejected | The SQL was not read-only and was refused. Nothing ran. |
| failed | The query still failed after the allowed repair attempts. The last error is shown. |

### Sidebar

Shows the connection, a warning if memory is off, the schema browser, the *Teach a fact* box (when
Hindsight is connected), and **Disconnect**, which closes the connection and clears the session.

---

## Configuration reference

Set these in `.env` (copy from `.env.example`). Database credentials are never read from here.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DB_TYPE`, `DB_HOST`, `DB_NAME`, `DB_USER` | postgresql, localhost:5432 | Pre-fill the connection screen. Never the password. |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server. |
| `OLLAMA_MODEL` | XiYanSQL Q4_K_M | Model used for generation and repair. |
| `OLLAMA_NUM_CTX` | 8192 | Context window. Bigger costs more RAM and time. |
| `OLLAMA_TIMEOUT_S` | 600 | Per-call timeout. |
| `OLLAMA_KEEP_ALIVE` | 10m | How long the model stays loaded. |
| `HINDSIGHT_API_URL` | `http://localhost:8888` | Hindsight server. Empty disables memory. |
| `HINDSIGHT_API_KEY` | empty | Only if your server needs one. |
| `HINDSIGHT_RETAIN_ASYNC` | false | `true` returns before extraction finishes but competes for CPU. |
| `MAX_REPAIR_ATTEMPTS` | 2 | Repair loops after a failed or invalid query. |
| `MAX_RESULT_ROWS` | 500 | Rows returned per query. |
| `QUERY_TIMEOUT_S` | 30 | Per-query database timeout. Raise for large databases. |
| `SCHEMA_SAMPLE_ROWS` | 3 | Sample rows per table (0 = none). |
| `SCHEMA_DISTINCT_VALUES` | 6 | Distinct values shown per text column. |
| `SCHEMA_SCAN_ROWS` | 50000 | Rows scanned when finding distinct values. Lower is faster. |
| `SCHEMA_FULL_THRESHOLD` | 8 | Schemas up to this size are sent whole. |
| `SCHEMA_MAX_TABLES` | 6 | Tables sent when the schema is filtered. |
| `SCHEMA_HIDDEN_COLUMNS` | LastEditedBy, LastEditedWhen, ValidFrom, ValidTo | Column names left out of the prompt. |
| `HISTORY_TURNS` | 3 | Previous turns given to follow-ups. |

For a database with a few dozen tables, raise `SCHEMA_FULL_THRESHOLD` so the whole schema is sent; it
avoids the model missing a table it needs. Lower `SCHEMA_SCAN_ROWS` if first-question sampling is slow.

---

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| "Hindsight is not reachable ... Memory is off" | The Hindsight server is not running. This is a warning only; everything else works. |
| Ollama or model error on connect | Start Ollama and run the `ollama pull` command above. `check_env.py` reports which piece is missing. |
| SQL Server: "No Microsoft ODBC Driver found" | Install *ODBC Driver 18 for SQL Server* from Microsoft. |
| SQL Server login fails | Leave username and password blank for Windows authentication; use the instance name shown in SSMS. |
| "Query timeout expired" | The query is heavy. Raise `QUERY_TIMEOUT_S`, or ask for it in steps. On SQL Server, totalling line items per header before joining is much faster. |
| "Invalid object name" / invented tables | The model was not shown the right tables. Raise `SCHEMA_FULL_THRESHOLD`, or name the table in your question. |
| Slow first question | Example values are sampled once per session. Lower `SCHEMA_SCAN_ROWS` or `SCHEMA_SAMPLE_ROWS`. |
| Answers take minutes | Expected on CPU for large queries. Keep `OLLAMA_KEEP_ALIVE` high so the model stays loaded. |

---

## Benchmark

The benchmark lives in a separate `benchmark/` package; the app has no benchmark-specific code. It
sends each question through the same `TextToSQLAgent` and measures execution accuracy (whether the
predicted and gold result sets are equal) on BIRD Mini-Dev for four variants:

| Variant | Schema retrieval | Hindsight | Repair |
| --- | --- | --- | --- |
| `baseline` | (full schema) | no | no |
| `schema_retrieval` | yes | no | no |
| `hindsight` | yes | yes | no |
| `full_agent` | yes | yes | yes |

```powershell
$env:BENCHMARK_DATABASE_URL = "postgresql://user:pass@localhost:5432/{db_id}"   # or a single DB without {db_id}
.venv\Scripts\python -m benchmark.run --data mini_dev_postgresql.json --limit 20
```

- BIRD `evidence` is passed to the model as a hint; `--no-evidence` turns this off.
- It writes nothing to memory unless you pass `--retain`, and even then only the pipeline's own
  executed SQL, never the gold SQL.
- Per-question results go to `benchmark_results.jsonl`.
- On CPU, expect roughly 20 to 60 seconds per question per variant, so use `--limit`.

---

## Tests

```powershell
.venv\Scripts\ruff check .
.venv\Scripts\pytest                              # unit, graph and UI tests; no external services needed
$env:TEST_DATABASE_URL = "postgresql://user:pass@localhost:5432/scratch"; .venv\Scripts\pytest tests/test_database.py
$env:RUN_OLLAMA_TESTS = "1"; .venv\Scripts\pytest tests/test_ollama_integration.py -s
```

The PostgreSQL integration tests create and then drop a `t2s_it` schema, so they need a login that can
create schemas in a scratch database. MySQL and SQL Server currently have unit tests only.

---

## Project layout and extending

```
app/
  main.py            Streamlit entry point        cli.py      terminal chat
  config.py          settings from env            session.py  connect + schema + model check + memory
  agent/             state.py, nodes.py, graph.py (LangGraph StateGraph)
  database/          base.py (abstraction, read-only execution), postgres.py, mysql.py, sqlserver.py
  llm/ollama.py      Ollama client (serialised calls)
  memory/hindsight.py
  schema/            extractor.py (reflection), retriever.py (relevance + M-Schema)
  sql/               generator.py, validator.py, repair.py
  ui/                connection.py, chat.py
benchmark/           dataset.py, metrics.py, run.py
tests/
```

**Adding another database type:** subclass `Database` in `app/database/`, set `dialect`, `driver`,
`sqlglot_dialect` and `default_port`, implement `_begin_readonly` and `_user_schemas`, add optional
`prompt_hints`, and register the class in `app/database/__init__.py`. Reflection, validation and the
agent are dialect-agnostic.

Dependencies are declared in both `requirements.txt` and `pyproject.toml`; keep them in sync.
