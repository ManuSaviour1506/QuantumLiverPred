"""Structured error responses. Clients always receive

    {"error": {"code": "...", "message": "...", "details": ..., "request_id": "..."}}

and never a stack trace, a server path or a secret. Unexpected exceptions are logged server-side with the request id
so that a user-visible id can be matched to the log line.
"""
from __future__ import annotations

import logging

from pydantic import ValidationError

from app.core.logging import request_id_var

log = logging.getLogger("lti.errors")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details=None, headers: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details, self.headers = status, code, message, details, headers


def bad_request(message: str, details=None) -> ApiError:
    return ApiError(400, "bad_request", message, details)


def unauthorized(message: str = "Authentication required") -> ApiError:
    return ApiError(401, "unauthorized", message, headers={"WWW-Authenticate": "Bearer"})


def forbidden(message: str = "You do not have permission to perform this action") -> ApiError:
    return ApiError(403, "forbidden", message)


def not_found(what: str = "Resource") -> ApiError:
    return ApiError(404, "not_found", f"{what} not found")


def conflict(message: str) -> ApiError:
    return ApiError(409, "conflict", message)


def unprocessable(message: str, details=None) -> ApiError:
    return ApiError(422, "unprocessable", message, details)


def error_body(code: str, message: str, details=None) -> dict:
    return {"error": {"code": code, "message": message, "details": details, "request_id": request_id_var.get()}}


def validation_details(exc: ValidationError) -> list[dict]:
    out = []
    for e in exc.errors(include_url=False, include_context=False, include_input=False):
        out.append({"field": ".".join(str(p) for p in e.get("loc", ())), "message": e.get("msg", "invalid value"),
                    "type": e.get("type")})
    return out
