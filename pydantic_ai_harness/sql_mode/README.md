# SQL Mode

Let the model orchestrate tool calls by writing **SQL** instead of issuing one tool call per round-trip.

## The idea

SQL is already a contained, well-defined language for expressing a wide range of logic — filtering, joining, aggregating, transforming — and DuckDB is a mature execution engine that [can lock itself down](https://duckdb.org/docs/stable/operations_manual/securing_duckdb/overview). SQL Mode turns that into an orchestration layer: the tools you register become DuckDB functions, and the model writes one SQL query that calls them, pipes data between them, and shapes the result — in a single round-trip.

It is the SQL counterpart to [Code Mode](../code_mode/README.md): where Code Mode sandboxes generated *Python*, SQL Mode runs generated *SQL* against a sandboxed, in-memory DuckDB.

## Usage

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai_harness import SQLModeBuilder


class Location(BaseModel):
    lat: float
    lon: float


def geocode(city: str) -> Location:
    """Look up a city's coordinates."""
    ...


async def fetch_weather(place: Location) -> dict[str, float]:
    """Get the current weather at a location."""
    ...


sql_mode = (
    SQLModeBuilder()
    .register_tool(geocode)
    .register_tool(fetch_weather)
    .build()
)

agent = Agent('anthropic:claude-sonnet-4-6', toolsets=[sql_mode])
result = agent.run_sync('Compare the temperature in Paris, Tokyo, and Lima.')
```

The model gets a single `run_sql` tool. It writes one query that fans the tools over a list and joins the results:

```sql
WITH cities AS (SELECT unnest(['Paris', 'Tokyo', 'Lima']) AS city),
     located AS (SELECT city, geocode(city) AS loc FROM cities)
SELECT city,
       fetch_weather(loc)->>'temp_c' AS temp_c
FROM located
ORDER BY temp_c DESC;
```

## How it works

`SQLModeBuilder().register_tool(...).build()` returns a `SQLModeToolset` that exposes one tool, `run_sql`. The registered tools are *not* exposed as native tool calls — they exist only as DuckDB functions inside the query.

Each `run_sql` call:

1. Opens a fresh, in-memory DuckDB database.
2. Registers every tool as a DuckDB user-defined function.
3. Locks the database down (see below).
4. Runs the model's query off the event loop, in a worker thread.
5. Returns the result set as `columns` (name + type) and `rows`.

Nothing persists between calls — every `run_sql` gets a brand-new database.

### Sync and async tools

Register either. The query runs in a worker thread; **async** tools are bridged back to the running event loop through an [`anyio` blocking portal](https://anyio.readthedocs.io/en/stable/threads.html), so an `async def` tool is awaited on the loop while the worker thread blocks for its result. Calls are made one row at a time — use `unnest`, `array_agg`, and `struct_pack` to shape data into and out of the tools.

### JSON and pydantic typing

Scalar parameters and return values (`str`, `int`, `float`, `bool`, `bytes`) map to native DuckDB column types. Everything else — pydantic models, `TypedDict`s, `dict`s, `list`s — is carried as DuckDB's `JSON` type:

- **Inputs** are validated against the tool's pydantic types with `TypeAdapter.validate_json`. Malformed or mistyped JSON surfaces as a retryable error.
- **Outputs** are serialized with `TypeAdapter.dump_json`.
- The **JSON Schema** of every JSON parameter and return value is rendered into the `run_sql` description, so the model knows the exact shape of what it is piping between tools.

Because a JSON return value and a JSON parameter share the same DuckDB type, one tool's output flows straight into another's input: `fetch_weather(geocode(city))`. Read fields with DuckDB's [JSON functions](https://duckdb.org/docs/stable/data/json/json_functions) — `->`, `->>`, `json_extract`, and friends.

## Security

The query is sandboxed. Before the model's SQL runs, the connection applies DuckDB's [hardening settings](https://duckdb.org/docs/stable/operations_manual/securing_duckdb/overview):

```sql
SET autoload_known_extensions = false;
SET autoinstall_known_extensions = false;
SET allow_community_extensions = false;
SET enable_external_access = false;
SET lock_configuration = true;          -- last: freezes the settings above
```

This blocks all filesystem and network access (`read_csv`, `read_parquet`, `ATTACH`, `COPY ... TO`, `httpfs`, …) and extension loading, and prevents the model's SQL from turning any of it back on. The model can only run pure SQL and call the tools you registered — the tools themselves are trusted code and reach the outside world on your terms.

DuckDB's own documentation notes that configuration hardening is defense-in-depth, not a substitute for OS-level sandboxing when running fully untrusted input.

## Installation

```bash
uv add "pydantic-ai-harness[sql-mode]"
```

This pulls in `duckdb`, `numpy` (DuckDB needs it to register Python functions), and `anyio`. Importing the feature without these installed raises `ImportError`.

## API

```python
SQLModeBuilder(
    max_rows: int = 1000,      # result rows returned before truncation
    max_retries: int = 3,      # retries for run_sql on a query/tool error
)
.register_tool(
    fn,                        # a sync or async function
    name: str | None = None,   # SQL function name; defaults to fn.__name__
    description: str | None = None,
)
.build() -> SQLModeToolset
```

## Limitations

- **One row at a time.** A tool used in `SELECT my_tool(x) FROM big_table` is called once per row, sequentially. Shape data with `unnest`/`array_agg` deliberately; this is orchestration, not bulk compute.
- **No state between calls.** Each `run_sql` gets a fresh database; tables created in one call are gone in the next.
- **Tool names must not collide with DuckDB built-in functions** (e.g. `add`, `length`, `concat`). Pass `name=` to rename.
- **Plain functions only.** Tools cannot use `*args`/`**kwargs`, and `RunContext`/dependency injection is not yet supported.
