"""PostgreSQL-backed memory store over a caller-owned connection pool.

Deliberately driver-agnostic: the store talks to a minimal `PostgresPool`
protocol (`execute` / `fetchval` / `fetch`, asyncpg-compatible), so
`pydantic-ai-harness` gains no database dependency -- the same policy that
keeps the S3 media store free of `boto3`. An `asyncpg.Pool` satisfies the
protocol out of the box; any other driver needs only a thin adapter.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

_TABLE_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,62}')


@runtime_checkable
class PostgresPool(Protocol):
    """The pool surface `PostgresMemoryStore` needs (asyncpg-compatible).

    Queries use `$1`-style positional parameters. `fetch` rows only need
    positional access (`row[0]`), which `asyncpg.Record` provides.
    """

    async def execute(self, query: str, *args: object) -> object:
        """Run a statement that returns no rows."""
        ...  # pragma: no cover

    async def fetchval(self, query: str, *args: object) -> object:
        """Run a query and return the first column of the first row, or `None`."""
        ...  # pragma: no cover

    async def fetch(self, query: str, *args: object) -> Sequence[Sequence[object]]:
        """Run a query and return all rows."""
        ...  # pragma: no cover


class PostgresMemoryStore:
    """Memory store for production multi-user apps: one row per memory file.

    The pool is caller-owned (create it at app startup, close it at
    shutdown); the store never manages connection lifecycle. Schema --
    `{table}(path TEXT PRIMARY KEY, content TEXT NOT NULL)` -- is created
    lazily on first use. Upserts are single statements, so cross-process
    writers get atomicity from PostgreSQL itself. Paths are opaque keys,
    always bound as parameters; the table name is the only interpolated
    identifier and is validated at construction.

    ```python
    import asyncpg

    from pydantic_ai_harness.memory import Memory, PostgresMemoryStore

    pool = await asyncpg.create_pool('postgres://...')
    memory = Memory(
        store=PostgresMemoryStore(pool),
        namespace=lambda ctx: ctx.deps.user_id,
    )
    ```
    """

    def __init__(self, pool: PostgresPool, *, table: str = 'agent_memory') -> None:
        if not _TABLE_RE.fullmatch(table):
            raise ValueError(f'invalid table name: {table!r}')
        self._pool = pool
        self._table = table
        self._schema_ready = False

    async def _ensure_schema(self) -> None:
        if not self._schema_ready:
            await self._pool.execute(
                f'CREATE TABLE IF NOT EXISTS {self._table} (path TEXT PRIMARY KEY, content TEXT NOT NULL)'
            )
            self._schema_ready = True

    async def read(self, path: str) -> str | None:
        """Return the content at `path`, or `None` if it does not exist."""
        await self._ensure_schema()
        value = await self._pool.fetchval(f'SELECT content FROM {self._table} WHERE path = $1', path)
        return value if isinstance(value, str) else None

    async def write(self, path: str, content: str) -> None:
        """Upsert `content` at `path` (atomic per statement)."""
        await self._ensure_schema()
        await self._pool.execute(
            f'INSERT INTO {self._table} (path, content) VALUES ($1, $2) '
            'ON CONFLICT (path) DO UPDATE SET content = EXCLUDED.content',
            path,
            content,
        )

    async def delete(self, path: str) -> None:
        """Delete `path` if it exists (idempotent)."""
        await self._ensure_schema()
        await self._pool.execute(f'DELETE FROM {self._table} WHERE path = $1', path)

    async def list_paths(self, prefix: str = '') -> list[str]:
        """Return all stored paths starting with `prefix`, sorted."""
        await self._ensure_schema()
        rows = await self._pool.fetch(
            f'SELECT path FROM {self._table} WHERE starts_with(path, $1) ORDER BY path', prefix
        )
        return [str(row[0]) for row in rows]
