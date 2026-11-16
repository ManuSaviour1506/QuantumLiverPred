"""Versioned SQL migrations (the role Alembic plays in the requested stack; Alembic is not installable here).

Each file in `migrations/` is `NNNN_name.sql` with two sections:

    -- migrate:up
    ...statements...
    -- migrate:down
    ...statements that undo them...

Applied versions are recorded in `schema_migrations` with a SHA-256 of the file. If an applied file is later edited the
runner refuses to continue, so the history cannot silently diverge between environments.

    python -m app.db.migrate upgrade            # apply all pending migrations (equivalent of `alembic upgrade head`)
    python -m app.db.migrate downgrade [--to N] # undo the latest migration, or down to version N
    python -m app.db.migrate status
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

from app.db.database import Connection, Database, split_statements

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


class MigrationError(RuntimeError):
    pass


@dataclass
class Migration:
    version: str
    name: str
    up: str
    down: str
    checksum: str


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    out = []
    for p in sorted(directory.glob("[0-9][0-9][0-9][0-9]_*.sql")):
        text = p.read_text(encoding="utf-8")
        if "-- migrate:up" not in text or "-- migrate:down" not in text:
            raise MigrationError(f"{p.name}: missing '-- migrate:up' or '-- migrate:down' section")
        up = text.split("-- migrate:up", 1)[1].split("-- migrate:down", 1)[0]
        down = text.split("-- migrate:down", 1)[1]
        out.append(Migration(p.name[:4], p.stem[5:], up, down, hashlib.sha256(text.encode()).hexdigest()))
    versions = [m.version for m in out]
    if len(set(versions)) != len(versions):
        raise MigrationError("duplicate migration version numbers")
    return out


def _ensure_table(c: Connection) -> None:
    c.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, name TEXT NOT NULL, "
              "checksum TEXT NOT NULL, applied_at TEXT NOT NULL)")
    c.commit()


def applied(c: Connection) -> dict[str, dict]:
    _ensure_table(c)
    return {r["version"]: r for r in c.all("SELECT version, name, checksum, applied_at FROM schema_migrations")}


def _now() -> str:
    from app.core.security import iso, utcnow
    return iso(utcnow())


def upgrade(db: Database, target: str | None = None) -> list[str]:
    done = []
    with db.session() as c:
        have = applied(c)
        for m in discover():
            if m.version in have:
                if have[m.version]["checksum"] != m.checksum:
                    raise MigrationError(f"migration {m.version}_{m.name} was edited after being applied "
                                         "(checksum mismatch); add a new migration instead")
                continue
            if target and m.version > target:
                break
            with c.transaction():
                for stmt in split_statements(m.up):
                    c.execute(stmt)
                c.execute("INSERT INTO schema_migrations (version, name, checksum, applied_at) VALUES (?, ?, ?, ?)",
                          (m.version, m.name, m.checksum, _now()))
            done.append(f"{m.version}_{m.name}")
    return done


def downgrade(db: Database, to: str | None = None) -> list[str]:
    undone = []
    with db.session() as c:
        have = applied(c)
        for m in reversed(discover()):
            if m.version not in have:
                continue
            if to is not None and m.version <= to:
                break
            with c.transaction():
                for stmt in split_statements(m.down):
                    c.execute(stmt)
                c.execute("DELETE FROM schema_migrations WHERE version = ?", (m.version,))
            undone.append(f"{m.version}_{m.name}")
            if to is None:
                break
    return undone


def status(db: Database) -> list[dict]:
    with db.session() as c:
        have = applied(c)
    rows = []
    for m in discover():
        a = have.get(m.version)
        rows.append({"version": m.version, "name": m.name, "applied": bool(a),
                     "applied_at": a["applied_at"] if a else None,
                     "checksum_ok": (a["checksum"] == m.checksum) if a else None})
    return rows


def main(argv=None) -> int:
    from app.core.config import Settings
    ap = argparse.ArgumentParser(prog="python -m app.db.migrate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upgrade")
    up.add_argument("target", nargs="?", default=None, help="version to stop at (default: all, i.e. 'head')")
    dn = sub.add_parser("downgrade")
    dn.add_argument("--to", default=None, help="version to keep (default: undo only the latest migration)")
    sub.add_parser("status")
    args = ap.parse_args(argv)
    db = Database(Settings.from_env().database_url)
    if args.cmd == "upgrade":
        target = None if args.target in (None, "head") else args.target
        done = upgrade(db, target)
        print("applied: " + (", ".join(done) if done else "nothing to apply (database is up to date)"))
    elif args.cmd == "downgrade":
        undone = downgrade(db, args.to)
        print("reverted: " + (", ".join(undone) if undone else "nothing to revert"))
    else:
        for r in status(db):
            flag = "applied " if r["applied"] else "pending "
            print(f"{r['version']}  {flag} {r['name']}  {r['applied_at'] or ''}"
                  f"{'' if r['checksum_ok'] in (None, True) else '  CHECKSUM MISMATCH'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
