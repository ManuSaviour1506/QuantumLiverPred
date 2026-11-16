"""ASGI middleware: request ids and access logs, request-body size limits, security headers, rate limiting."""
from __future__ import annotations

import logging
import threading
import time
import uuid

from app.core.config import Settings, parse_rate
from app.core.errors import ApiError, error_body
from app.core.logging import log_event, request_id_var

access_log = logging.getLogger("lti.access")


class RequestContextMiddleware:
    """Assigns/propagates X-Request-ID and writes one structured access-log line per request (path only, no query
    string, so tokens or search terms never reach the logs)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        rid = incoming if incoming and len(incoming) <= 64 and incoming.replace("-", "").isalnum() else uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        start = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                headers = list(message.get("headers") or [])
                headers.append((b"x-request-id", rid.encode()))
                message = {**message, "headers": headers}
            await send(message)
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if not scope["path"].startswith("/assets/"):
                log_event(access_log, "request", method=scope["method"], path=scope["path"], status=status["code"],
                          duration_ms=round((time.perf_counter() - start) * 1000, 1))
            request_id_var.reset(token)


class BodySizeLimitMiddleware:
    """Rejects request bodies above a limit (413) even when Content-Length is absent (chunked uploads)."""

    def __init__(self, app, default_limit: int, overrides: dict[str, int]):
        self.app, self.default, self.overrides = app, default_limit, overrides

    def _limit(self, path: str) -> int:
        for prefix, lim in self.overrides.items():
            if path.startswith(prefix):
                return lim
        return self.default

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = self._limit(scope["path"]) + 1024 * 1024   # allowance for multipart framing
        headers = dict(scope.get("headers") or [])
        cl = headers.get(b"content-length")
        if cl is not None and cl.isdigit() and int(cl) > limit:
            return await _send_error(send, 413, "payload_too_large", "Request body too large")
        received = {"n": 0}

        async def limited_receive():
            message = await receive()
            if message["type"] == "http.request":
                received["n"] += len(message.get("body", b""))
                if received["n"] > limit:
                    raise ApiError(413, "payload_too_large", "Request body too large")
            return message
        try:
            await self.app(scope, limited_receive, send)
        except ApiError as e:
            await _send_error(send, e.status, e.code, e.message)


async def _send_error(send, status: int, code: str, message: str):
    from app.core.apikit import to_json_bytes
    body = to_json_bytes(error_body(code, message))
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


class SecurityHeadersMiddleware:
    CSP = ("default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
           "connect-src 'self' {extra}; font-src 'self' data:; frame-ancestors 'none'; base-uri 'self'; "
           "form-action 'self'; object-src 'none'")
    DOCS_CSP = ("default-src 'self'; img-src 'self' data: https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; connect-src 'self'; frame-ancestors 'none'")

    def __init__(self, app, hsts: bool, connect_extra: str = ""):
        self.app, self.hsts = app, hsts
        self.csp = self.CSP.format(extra=connect_extra).replace("  ", " ")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                h = list(message.get("headers") or [])
                h += [(b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"),
                      (b"referrer-policy", b"no-referrer"),
                      (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                      (b"cross-origin-opener-policy", b"same-origin"),
                      (b"content-security-policy", (self.DOCS_CSP if path == "/docs" else self.csp).encode())]
                if path.startswith("/api/") and not any(k.lower() == b"cache-control" for k, _ in h):
                    h.append((b"cache-control", b"no-store"))   # never cached by browsers, proxies or CDNs
                if self.hsts:
                    h.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                message = {**message, "headers": h}
            await send(message)
        await self.app(scope, receive, send_wrapper)


class RateLimiter:
    """Fixed-window in-process limiter. Per process only: behind several replicas put limits in the reverse proxy too
    (see DEPLOYMENT.md)."""

    def __init__(self):
        self._hits: dict[tuple, list] = {}
        self._lock = threading.Lock()

    def check(self, bucket: str, key: str, limit: int, window: int) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            start, count = self._hits.get((bucket, key), (now, 0))
            if now - start >= window:
                start, count = now, 0
            count += 1
            self._hits[(bucket, key)] = (start, count)
            if len(self._hits) > 50000:
                self._hits = {k: v for k, v in self._hits.items() if now - v[0] < 3600}
            return count <= limit, max(1, int(window - (now - start)))

    def enforce(self, bucket: str, key: str, settings: Settings) -> None:
        spec = {"auth": settings.rate_limit_auth, "refresh": settings.rate_limit_refresh,
                "predict_user": settings.rate_limit_predict}.get(
            bucket, settings.rate_limit_default)
        limit, window = parse_rate(spec)
        ok, retry = self.check(bucket, key, limit, window)
        if not ok:
            raise ApiError(429, "rate_limited", "Too many requests; slow down and retry later",
                           {"retry_after_seconds": retry}, headers={"Retry-After": str(retry)})

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
