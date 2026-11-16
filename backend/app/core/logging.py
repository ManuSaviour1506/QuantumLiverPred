"""Structured (JSON) logging with a per-request correlation id.

Privacy rule: log records carry ids (user id, prediction id, request id, SHA-256 prefixes) but never e-mail addresses,
passwords, tokens, uploaded file names or image content, because file names of medical images can contain personal
information.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_REDACT = {"password", "new_password", "current_password", "token", "refresh_token", "access_token", "authorization",
           "email", "filename", "jwt_secret", "smtp_password"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
               "level": record.levelname, "logger": record.name, "msg": record.getMessage(),
               "request_id": request_id_var.get()}
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            out.update({k: ("[redacted]" if k.lower() in _REDACT else v) for k, v in extra.items()})
        if record.exc_info:
            out["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{time.strftime('%H:%M:%S', time.gmtime(record.created))} {record.levelname:<7} {record.name}: " \
               f"{record.getMessage()} [{request_id_var.get()}]"
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict) and extra:
            base += " " + " ".join(f"{k}={'[redacted]' if k.lower() in _REDACT else v}" for k, v in extra.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_lti", False):
            root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler._lti = True  # type: ignore[attr-defined]
    handler.setFormatter(JsonFormatter() if json_logs else TextFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    for noisy in ("uvicorn.access", "multipart", "python_multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, msg, extra={"fields": fields})
