"""SQLite access layer for the collector.

Design:
  - A single dedicated writer thread owns the connection (SQLite connections are
    not safe to share across threads). All DB work runs on that one-thread
    executor via run_in_executor, so the asyncio event loop never blocks on disk.
  - WAL + synchronous=NORMAL; lets analytical reads (separate
    connections, later phases) run concurrently with ingestion.
  - Column lists are introspected from the live schema at bootstrap, so callers
    pass plain dicts (partial dicts are fine — missing columns become NULL).
    This keeps the big column lists in schema.py only, never duplicated here.

Type contract: callers serialize complex values (JSON blobs, wallet lists,
raw payloads) to str before handing them over. db.py is type-agnostic — it
maps dict keys to columns and binds values as-is. bool -> 0/1, None -> NULL.
"""

from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import schema


class Database:
    def __init__(self, path: str | Path):
        self._path = str(path)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="db")
        self._conn: sqlite3.Connection | None = None
        self._columns: dict[str, list[str]] = {}

    # -- lifecycle -----------------------------------------------------------
    async def connect(self) -> None:
        await self._run(self._connect_sync)

    def _connect_sync(self) -> None:
        # check_same_thread=False is safe here: the connection is only ever
        # touched from this single executor thread, never concurrently.
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, check_same_thread=False)
        for pragma in schema.PRAGMAS:
            conn.execute(pragma)
        for stmt in schema.all_statements():
            conn.execute(stmt)
        # Additive migrations for pre-existing tables (ADD COLUMN if missing).
        for table, col, coltype in schema.MIGRATIONS:
            existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
        conn.commit()
        # Cache column order per table for dict-based inserts.
        for name, _ in schema.TABLES:
            rows = conn.execute(f"PRAGMA table_info({name})").fetchall()
            self._columns[name] = [r[1] for r in rows]
        self._conn = conn

    async def close(self) -> None:
        await self._run(self._close_sync)
        self._executor.shutdown(wait=True)

    def _close_sync(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    # -- writes --------------------------------------------------------------
    async def insert(self, table: str, row: Mapping[str, Any]) -> None:
        """Insert one row (OR IGNORE on PK/UNIQUE conflict). Partial dicts OK."""
        await self.insert_many(table, [row])

    async def insert_many(self, table: str, rows: Sequence[Mapping[str, Any]],
                          or_replace: bool = False) -> int:
        if not rows:
            return 0
        return await self._run(self._insert_many_sync, table, list(rows), or_replace)

    def _insert_many_sync(self, table: str, rows: list[Mapping[str, Any]],
                          or_replace: bool = False) -> int:
        cols = self._columns[table]
        placeholders = ", ".join("?" for _ in cols)
        verb = "INSERT OR REPLACE" if or_replace else "INSERT OR IGNORE"
        sql = f"{verb} INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        params = [[r.get(c) for c in cols] for r in rows]
        cur = self._conn.executemany(sql, params)
        self._conn.commit()
        return cur.rowcount

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Run an arbitrary write (UPDATE / upsert). Returns affected rowcount."""
        return await self._run(self._execute_sync, sql, tuple(params))

    def _execute_sync(self, sql: str, params: tuple) -> int:
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return cur.rowcount

    async def upsert(self, table: str, row: Mapping[str, Any], conflict_key: str = "token_address") -> None:
        """Insert `row`; on conflict, update exactly the provided columns.

        Unlike insert()'s OR IGNORE, the update is never silently dropped — used
        where two writers each own a disjoint subset of a row's columns."""
        cols = list(row.keys())
        set_cols = [c for c in cols if c != conflict_key]
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        if set_cols:
            sql += (f" ON CONFLICT({conflict_key}) DO UPDATE SET "
                    + ", ".join(f"{c}=excluded.{c}" for c in set_cols))
        else:
            sql += f" ON CONFLICT({conflict_key}) DO NOTHING"
        await self.execute(sql, [row[c] for c in cols])

    # -- reads ---------------------------------------------------------------
    async def fetchone(self, sql: str, params: Sequence[Any] = ()) -> tuple | None:
        return await self._run(lambda: self._conn.execute(sql, tuple(params)).fetchone())

    async def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[tuple]:
        return await self._run(lambda: self._conn.execute(sql, tuple(params)).fetchall())

    def columns(self, table: str) -> list[str]:
        return list(self._columns[table])

    # -- executor plumbing ---------------------------------------------------
    async def _run(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)
