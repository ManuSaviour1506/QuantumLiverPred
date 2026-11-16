"""Audit log: who did what, when. IPs are stored only as keyed hashes; details never contain file names or images."""
from __future__ import annotations

from app.core.security import hash_ip, iso, new_id, utcnow
from app.db.database import Connection, loads


def log(conn: Connection, action: str, user_id: str | None, entity_type: str | None = None,
        entity_id: str | None = None, ip: str | None = None, details: dict | None = None, secret: str = "") -> None:
    conn.run("INSERT INTO audit_logs (id, user_id, action, entity_type, entity_id, ip_hash, details, created_at) "
             "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
             (new_id("al_"), user_id, action, entity_type, entity_id, hash_ip(ip, secret), details or {},
              iso(utcnow())))


def recent(conn: Connection, user_id: str | None, limit: int = 15) -> list[dict]:
    if user_id is None:
        rows = conn.all("SELECT action, entity_type, entity_id, user_id, details, created_at FROM audit_logs "
                        "ORDER BY created_at DESC LIMIT ?", (limit,))
    else:
        rows = conn.all("SELECT action, entity_type, entity_id, user_id, details, created_at FROM audit_logs "
                        "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit))
    for r in rows:
        r["details"] = loads(r["details"], {})
    return rows
