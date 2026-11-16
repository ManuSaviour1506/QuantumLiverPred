"""Portable relational data layer (SQLite for local development / tests, PostgreSQL for deployment).

SQLAlchemy is not available in the build environment, so the application talks to the database through DB-API 2.0
with portable SQL:

* identifiers are TEXT (uuid4 hex), timestamps TEXT (ISO-8601 UTC, sortable), booleans INTEGER 0/1, JSON TEXT;
* placeholders are written as `?` and translated for the driver (`%s` for psycopg, `$n` for the test wire client);
* upserts use `INSERT ... ON CONFLICT (...) DO UPDATE`, supported by SQLite >= 3.24 and PostgreSQL >= 9.5.

URLs: `sqlite:///absolute/path.db`, `sqlite:///:memory:` (tests), `postgresql://user:pass@host:5432/db`.
PostgreSQL uses psycopg 3 (installed in the Docker image). The pure-Python wire client in `tests/support/pgwire.py`
is used only by the test suite to check PostgreSQL compatibility where psycopg cannot be installed.
"""
from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote, urlparse

_PLACEHOLDER = re.compile(r"\?")


def _to_format(sql: str) -> str:
    return _PLACEHOLDER.sub("%s", sql.replace("%", "%%"))


def _to_numbered(sql: str) -> str:
    n = 0

    def repl(_m):
        nonlocal n
        n += 1
        return f"${n}"
    return _PLACEHOLDER.sub(repl, sql)


class Connection:
    """Thin wrapper: `one`, `all`, `scalar`, `run` and a `transaction()` context manager. Rows are dicts."""

    def __init__(self, raw, dialect: str, paramstyle: str):
        self.raw, self.dialect, self._style = raw, dialect, paramstyle
        self._depth = 0

    def _sql(self, sql: str) -> str:
        if self._style == "format":
            return _to_format(sql)
        if self._style == "numbered":
            return _to_numbered(sql)
        return sql

    @staticmethod
    def _params(params) -> tuple:
        out = []
        for p in params or ():
            if isinstance(p, bool):
                out.append(int(p))
            elif isinstance(p, (dict, list)):
                out.append(json.dumps(p, default=str))
            elif isinstance(p, Path):
                out.append(str(p))
            else:
                out.append(p)
        return tuple(out)

    def execute(self, sql: str, params=()):
        cur = self.raw.cursor()
        cur.execute(self._sql(sql), self._params(params))
        return cur

    @staticmethod
    def _rows(cur) -> list[dict]:
        if cur.description is None:
            return []
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def all(self, sql: str, params=()) -> list[dict]:
        return self._rows(self.execute(sql, params))

    def one(self, sql: str, params=()) -> dict | None:
        rows = self.all(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params=()) -> Any:
        row = self.one(sql, params)
        return None if row is None else next(iter(row.values()))

    def run(self, sql: str, params=()) -> int:
        cur = self.execute(sql, params)
        return cur.rowcount if cur.rowcount is not None else -1

    def script(self, sql: str) -> None:
        for stmt in split_statements(sql):
            self.execute(stmt)

    @contextlib.contextmanager
    def transaction(self) -> Iterator["Connection"]:
        if self._depth:
            self._depth += 1
            try:
                yield self
            finally:
                self._depth -= 1
            return
        self._depth = 1
        try:
            yield self
            self.raw.commit()
        except BaseException:
            self.raw.rollback()
            raise
        finally:
            self._depth = 0

    def commit(self) -> None:
        self.raw.commit()

    def close(self) -> None:
        try:
            self.raw.close()
        except Exception:
            pass


def split_statements(sql: str) -> list[str]:
    """Split on semicolons at line ends; strips `--` comment lines. Statements must not embed `;\\n` in literals."""
    out, buf = [], []
    for line in sql.splitlines():
        s = line.strip()
        if not s or s.startswith("--"):
            continue
        buf.append(line)
        if s.endswith(";"):
            stmt = "\n".join(buf).strip().rstrip(";").strip()
            if stmt:
                out.append(stmt)
            buf = []
    tail = "\n".join(buf).strip()
    if tail:
        out.append(tail)
    return out


class Database:
    def __init__(self, url: str):
        self.url = url
        p = urlparse(url)
        self._memory_conn = None
        self._lock = threading.Lock()
        if p.scheme == "sqlite":
            self.dialect = "sqlite"
            path = url[len("sqlite:///"):]
            self.path = path
            if path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
        elif p.scheme in ("postgresql", "postgres", "postgresql+psycopg"):
            self.dialect = "postgresql"
            from urllib.parse import parse_qs
            q = parse_qs(p.query)
            host = q.get("host", [unquote(p.hostname or "") or "localhost"])[0]
            self._pg = {"host": host, "port": int(q.get("port", [p.port or 5432])[0]),
                        "user": unquote(p.username or ""), "password": unquote(p.password or ""),
                        "dbname": (p.path or "/").lstrip("/") or "postgres", "query": p.query}
            self._driver = None
        else:
            raise ValueError("DATABASE_URL must start with sqlite:/// or postgresql://")

    @property
    def safe_url(self) -> str:
        """URL with the password removed (for logs and the status page)."""
        p = urlparse(self.url)
        if p.password:
            return self.url.replace(f":{p.password}@", ":***@")
        return self.url

    def _connect_sqlite(self) -> Connection:
        if self.path == ":memory:":
            with self._lock:
                if self._memory_conn is None:
                    raw = sqlite3.connect(":memory:", check_same_thread=False)
                    raw.execute("PRAGMA foreign_keys=ON")
                    self._memory_conn = raw
            return _SharedConnection(self._memory_conn, "sqlite", "qmark", self._lock)
        raw = sqlite3.connect(self.path, timeout=15)
        raw.execute("PRAGMA foreign_keys=ON")
        raw.execute("PRAGMA busy_timeout=15000")
        raw.execute("PRAGMA journal_mode=WAL")
        return Connection(raw, "sqlite", "qmark")

    def _connect_pg(self) -> Connection:
        c = self._pg
        try:
            import psycopg  # type: ignore
            raw = psycopg.connect(host=c["host"], port=c["port"], user=c["user"], password=c["password"],
                                  dbname=c["dbname"], connect_timeout=10)
            return Connection(raw, "postgresql", "format")
        except ImportError:
            pass
        import os
        if os.environ.get("LTI_TEST_PGWIRE") == "1":
            from tests.support.pgwire import connect  # test-only fallback, never used in deployment
            raw = connect(host=c["host"], port=c["port"], user=c["user"], password=c["password"], dbname=c["dbname"])
            return Connection(raw, "postgresql", "numbered")
        raise RuntimeError("PostgreSQL URL configured but psycopg is not installed (pip install 'psycopg[binary]')")

    def connect(self) -> Connection:
        return self._connect_sqlite() if self.dialect == "sqlite" else self._connect_pg()

    @contextlib.contextmanager
    def session(self) -> Iterator[Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except BaseException:
            try:
                conn.raw.rollback()
            except Exception:
                pass
            raise
        finally:
            conn.close()

    def ping(self) -> bool:
        with self.session() as c:
            return c.scalar("SELECT 1") == 1


class _SharedConnection(Connection):
    """In-memory SQLite (tests only): one shared connection serialised by a lock."""

    def __init__(self, raw, dialect, style, lock):
        super().__init__(raw, dialect, style)
        self._shared_lock = lock

    def execute(self, sql, params=()):
        with self._shared_lock:
            return super().execute(sql, params)

    def close(self) -> None:  # keep the shared connection open
        pass


def loads(value, default=None):
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default
