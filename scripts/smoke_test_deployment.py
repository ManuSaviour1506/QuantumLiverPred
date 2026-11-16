#!/usr/bin/env python3
"""Post-deployment smoke test (standard library only), for a local Docker stack or a hosted deployment.

    python scripts/smoke_test_deployment.py --base-url https://your-app.example.org \
        --image dataset/sample_data/lits_sample/images/<slice>.png [--register]

Checks: liveness and readiness, the SPA and its security headers, the OpenAPI document, registration or login
(credentials come from LTI_SMOKE_EMAIL / LTI_SMOKE_PASSWORD or are generated for --register), the model list
(synthetic test models must be absent in production), one real prediction on a CT slice with the first servable
model, the PDF report, and deletion of the test prediction. Exits non-zero on the first failure.
Nothing is printed that contains a password or token.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


class Client:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.token: str | None = None

    def call(self, method: str, path: str, body: bytes | None = None, ctype: str | None = None,
             raw: bool = False, expect: tuple[int, ...] = (200,)):
        req = urllib.request.Request(self.base + path, data=body, method=method)
        if ctype:
            req.add_header("Content-Type", ctype)
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                status, headers, data = r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            status, headers, data = e.code, dict(e.headers), e.read()
        if status not in expect:
            raise SystemExit(f"FAIL {method} {path}: HTTP {status} {data[:300]!r}")
        return (status, headers, data) if raw else json.loads(data or b"null")

    def json(self, method: str, path: str, payload: dict, expect=(200,)):
        return self.call(method, path, json.dumps(payload).encode(), "application/json", expect=expect)

    def upload(self, path: str, fields: dict, file_field: str, file_path: Path, expect=(200, 201, 202)):
        boundary = uuid.uuid4().hex
        parts = []
        for k, v in fields.items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
                     f'filename="{file_path.name}"\r\nContent-Type: {mime}\r\n\r\n'.encode() + file_path.read_bytes() + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        return self.call("POST", path, b"".join(parts), f"multipart/form-data; boundary={boundary}", expect=expect)


def ok(msg: str) -> None:
    print(f"  ok   {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--image", type=Path, help="CT slice (PNG/JPEG/DICOM/NIfTI) for the prediction check")
    ap.add_argument("--register", action="store_true", help="register a throw-away account (registration must be open)")
    ap.add_argument("--expect-production", action="store_true", help="fail if synthetic test models are listed")
    a = ap.parse_args()
    c = Client(a.base_url)
    print(f"Smoke test against {c.base}")

    assert c.call("GET", "/health")["status"] == "ok"; ok("GET /health")
    ready = c.call("GET", "/health/ready", expect=(200, 503))
    if ready.get("status") != "ready":
        raise SystemExit(f"FAIL readiness: {ready}")
    ok(f"GET /health/ready {ready['checks']}")
    _, h, html = c.call("GET", "/", raw=True)
    hl = {k.lower(): v for k, v in h.items()}
    assert b"<div id=\"root\"" in html or b"id=root" in html, "SPA index not served"
    for hdr in ("content-security-policy", "x-content-type-options", "x-frame-options"):
        assert hdr in hl, f"missing security header {hdr}"
    ok("SPA index served with CSP / nosniff / frame protections")
    spec = c.call("GET", "/openapi.json"); ok(f"OpenAPI {spec.get('openapi')} with {len(spec.get('paths', {}))} paths")

    if a.register:
        email = f"smoke-{secrets.token_hex(4)}@example.org"
        password = secrets.token_urlsafe(18) + "Aa1!"
        c.json("POST", "/api/v1/auth/register", {"email": email, "password": password, "full_name": "Smoke Test"},
               expect=(200, 201))
        ok("register (throw-away account)")
    else:
        email, password = os.environ.get("LTI_SMOKE_EMAIL"), os.environ.get("LTI_SMOKE_PASSWORD")
        if not email or not password:
            raise SystemExit("set LTI_SMOKE_EMAIL and LTI_SMOKE_PASSWORD, or pass --register")
    tok = c.json("POST", "/api/v1/auth/login", {"email": email, "password": password})
    c.token = tok["access_token"]; ok("login (access token received)")
    me = c.call("GET", "/api/v1/auth/me"); ok(f"GET /auth/me role={me.get('role')}")

    models = c.call("GET", "/api/v1/models")
    items = models.get("items", models) if isinstance(models, dict) else models
    servable = [m for m in items if m.get("servable", m.get("available", True))]
    synthetic = [m["id"] for m in items if "synthetic" in str(m.get("id", ""))]
    if a.expect_production and synthetic:
        raise SystemExit(f"FAIL synthetic test models are listed in production: {synthetic}")
    ok(f"{len(items)} models listed, {len(servable)} servable, synthetic listed: {bool(synthetic)}")

    if a.image:
        if not servable:
            raise SystemExit("FAIL no servable model for the prediction check")
        model_id = servable[0]["id"]
        job = c.upload("/api/v1/predictions", {"model_id": model_id}, "file", a.image)
        pid = job.get("id") or job.get("prediction_id")
        t0 = time.time()
        while True:
            p = c.call("GET", f"/api/v1/predictions/{pid}")
            if p.get("status") in ("succeeded", "failed"):
                break
            if time.time() - t0 > 300:
                raise SystemExit("FAIL prediction did not finish within 300 s")
            time.sleep(1.5)
        if p["status"] != "succeeded":
            raise SystemExit(f"FAIL prediction {p['status']}: {p.get('error')}")
        ok(f"prediction with {model_id}: label={p.get('label')} probability={p.get('probability')} "
           f"threshold={p.get('threshold')} mode={p.get('execution_mode')} ({p.get('processing_ms')} ms server-side)")
        _, rh, pdf = c.call("GET", f"/api/v1/predictions/{pid}/report", raw=True)
        assert pdf[:4] == b"%PDF", "report is not a PDF"
        ok(f"PDF report {len(pdf)} bytes")
        c.call("DELETE", f"/api/v1/predictions/{pid}", expect=(200, 204))
        ok("test prediction deleted")
    c.call("POST", "/api/v1/auth/logout", b"{}", "application/json", expect=(200, 204))
    ok("logout")
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
