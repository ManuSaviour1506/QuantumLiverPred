"""TEST-ONLY minimal PostgreSQL client (wire protocol v3, extended query protocol, text format).

Purpose: run the backend test suite against a real PostgreSQL server in an environment where psycopg cannot be
installed. Deployments use psycopg 3 (see `app/db/database.py`); this module is never imported unless
LTI_TEST_PGWIRE=1. It supports trust, clear-text and MD5 authentication, unix sockets and TCP, implicit transactions
(DB-API style) and the column types the application uses (text, integers, floating point, numeric, boolean).
"""
from __future__ import annotations

import hashlib
import socket
import struct

_INT = {20, 21, 23, 26}
_FLOAT = {700, 701, 1700}


class PGError(Exception):
    def __init__(self, fields: dict):
        self.fields = fields
        super().__init__(f"{fields.get('S', 'ERROR')} {fields.get('C', '')}: {fields.get('M', '')}")


class _Cursor:
    def __init__(self, conn):
        self.conn, self.description, self.rowcount, self._rows = conn, None, -1, []

    def execute(self, sql: str, params=()):
        self.description, self._rows, self.rowcount = self.conn._extended(sql, params)
        return self

    def fetchall(self):
        rows, self._rows = self._rows, []
        return rows

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None


class Connection:
    def __init__(self, host, port, user, password, dbname):
        if str(host).startswith("/"):
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(f"{host}/.s.PGSQL.{port}")
        else:
            self.sock = socket.create_connection((host, port), timeout=30)
        self._buf = b""
        self.user, self.password = user, password
        self._in_tx = False
        body = b"".join(k.encode() + b"\0" + v.encode() + b"\0" for k, v in (("user", user), ("database", dbname)))
        body += b"\0"
        self.sock.sendall(struct.pack("!ii", len(body) + 8, 196608) + body)
        self._auth()

    # ------------------------------------------------------------------ framing
    def _recv(self, n):
        while len(self._buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("server closed the connection")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _msg(self):
        t = self._recv(1)
        (ln,) = struct.unpack("!i", self._recv(4))
        return t, self._recv(ln - 4)

    def _send(self, t: bytes, payload: bytes = b""):
        self.sock.sendall(t + struct.pack("!i", len(payload) + 4) + payload)

    @staticmethod
    def _fields(payload: bytes) -> dict:
        out = {}
        for part in payload.split(b"\0"):
            if part:
                out[chr(part[0])] = part[1:].decode("utf-8", "replace")
        return out

    def _auth(self):
        while True:
            t, p = self._msg()
            if t == b"R":
                (code,) = struct.unpack("!i", p[:4])
                if code == 0:
                    continue
                if code == 3:
                    self._send(b"p", self.password.encode() + b"\0")
                elif code == 5:
                    salt = p[4:8]
                    inner = hashlib.md5((self.password + self.user).encode()).hexdigest().encode()
                    self._send(b"p", b"md5" + hashlib.md5(inner + salt).hexdigest().encode() + b"\0")
                else:
                    raise PGError({"M": f"unsupported authentication method {code} (test client)"})
            elif t == b"E":
                raise PGError(self._fields(p))
            elif t == b"Z":
                return

    # ------------------------------------------------------------------ queries
    def _simple(self, sql: str):
        self._send(b"Q", sql.encode() + b"\0")
        err = None
        while True:
            t, p = self._msg()
            if t == b"E":
                err = PGError(self._fields(p))
            elif t == b"Z":
                if err:
                    raise err
                return

    @staticmethod
    def _encode(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return b"true" if v else b"false"
        if isinstance(v, bytes):
            return b"\\x" + v.hex().encode()
        return str(v).encode("utf-8")

    @staticmethod
    def _decode(oid, raw):
        if raw is None:
            return None
        s = raw.decode("utf-8")
        if oid in _INT:
            return int(s)
        if oid in _FLOAT:
            return float(s)
        if oid == 16:
            return s == "t"
        return s

    def _extended(self, sql: str, params):
        if not self._in_tx:
            self._simple("BEGIN")
            self._in_tx = True
        vals = [self._encode(v) for v in params]
        self._send(b"P", b"\0" + sql.encode() + b"\0" + struct.pack("!h", 0))
        bind = b"\0\0" + struct.pack("!h", 0) + struct.pack("!h", len(vals))
        for v in vals:
            bind += struct.pack("!i", -1) if v is None else struct.pack("!i", len(v)) + v
        bind += struct.pack("!h", 0)
        self._send(b"B", bind)
        self._send(b"D", b"P\0")
        self._send(b"E", b"\0" + struct.pack("!i", 0))
        self._send(b"S")
        desc, oids, rows, count, err = None, [], [], -1, None
        while True:
            t, p = self._msg()
            if t == b"T":
                (n,) = struct.unpack("!h", p[:2])
                pos, desc = 2, []
                for _ in range(n):
                    end = p.index(b"\0", pos)
                    name = p[pos:end].decode()
                    pos = end + 1
                    _tbl, _col, oid, _sz, _mod, _fmt = struct.unpack("!ihihih", p[pos:pos + 18])
                    pos += 18
                    desc.append((name, oid, None, None, None, None, None))
                    oids.append(oid)
            elif t == b"D":
                (n,) = struct.unpack("!h", p[:2])
                pos, row = 2, []
                for i in range(n):
                    (ln,) = struct.unpack("!i", p[pos:pos + 4])
                    pos += 4
                    raw = None if ln == -1 else p[pos:pos + ln]
                    pos += max(ln, 0)
                    row.append(self._decode(oids[i], raw))
                rows.append(tuple(row))
            elif t == b"C":
                tag = p.rstrip(b"\0").decode().split()
                if tag and tag[-1].isdigit():
                    count = int(tag[-1])
            elif t == b"E":
                err = PGError(self._fields(p))
            elif t == b"Z":
                if err:
                    raise err
                return desc, rows, count

    # ------------------------------------------------------------------ DB-API surface
    def cursor(self):
        return _Cursor(self)

    def commit(self):
        if self._in_tx:
            self._simple("COMMIT")
            self._in_tx = False

    def rollback(self):
        if self._in_tx:
            self._simple("ROLLBACK")
            self._in_tx = False

    def close(self):
        try:
            self.rollback()
            self._send(b"X")
            self.sock.close()
        except Exception:
            pass


def connect(host, port, user, password, dbname):
    return Connection(host, port, user, password, dbname)
