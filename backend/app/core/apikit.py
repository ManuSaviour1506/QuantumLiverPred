"""Typed routing on Starlette: Pydantic validation, authentication, role checks, rate limits, upload limits, thread-pool
execution and OpenAPI 3.1 generation.

This plays the role FastAPI plays in the requested stack (FastAPI itself is built on Starlette + Pydantic but is not
installable in the build environment). Handlers are plain synchronous functions `handler(ctx) -> result`; they run in
Starlette's thread pool so database access and file I/O never block the event loop. Long work (inference,
experiments, dataset validation) is not done in handlers at all but queued as background jobs.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import tempfile
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError
from pydantic.json_schema import models_json_schema
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from app.core.errors import ApiError, error_body, unauthorized, validation_details
from app.core.logging import request_id_var


# --------------------------------------------------------------------------------------------- JSON response
def clean_json(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    return o


def _default(o):
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    if isinstance(o, (dt.datetime, dt.date)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    if hasattr(o, "item"):          # numpy scalars
        return o.item()
    if hasattr(o, "tolist"):        # numpy arrays
        return o.tolist()
    return str(o)


def to_json_bytes(content) -> bytes:
    if isinstance(content, BaseModel):
        content = content.model_dump(mode="json")
    return json.dumps(clean_json(json.loads(json.dumps(content, default=_default))), ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


class JSONResponse(Response):
    media_type = "application/json"

    def render(self, content) -> bytes:
        return to_json_bytes(content)


def error_response(err: ApiError) -> JSONResponse:
    return JSONResponse(error_body(err.code, err.message, err.details), status_code=err.status, headers=err.headers)


# --------------------------------------------------------------------------------------------- request context
@dataclass
class UploadedFile:
    field: str
    filename: str          # sanitised display name (never used to build a path)
    content_type: str
    path: Path             # temporary file, deleted after the handler returns
    size: int
    sha256: str

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()


@dataclass
class FileField:
    name: str
    required: bool = True
    description: str = ""
    max_bytes_setting: str = "max_upload_bytes"


@dataclass
class TextField:
    name: str
    required: bool = False
    description: str = ""
    max_length: int = 2000


@dataclass
class Ctx:
    request: Request
    state: Any
    user: dict | None = None
    path: dict = field(default_factory=dict)
    query: Any = None
    body: Any = None
    form: dict = field(default_factory=dict)

    @property
    def settings(self):
        return self.state.settings

    @property
    def db(self):
        return self.state.db

    @property
    def request_id(self) -> str:
        return request_id_var.get()

    @property
    def client_ip(self) -> str | None:
        return self.request.client.host if self.request.client else None

    @property
    def is_admin(self) -> bool:
        return bool(self.user and self.user.get("role") == "admin")


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def safe_display_name(name: str | None, limit: int = 100) -> str:
    base = (name or "upload").replace("\\", "/").split("/")[-1]
    base = _SAFE_NAME.sub("_", base).strip(" .") or "upload"
    return base[-limit:]


# --------------------------------------------------------------------------------------------- endpoints
@dataclass
class Endpoint:
    method: str
    path: str
    handler: Callable[[Ctx], Any]
    summary: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    auth: bool | tuple = True          # False = public; True = any active user; tuple = allowed roles
    body: type[BaseModel] | None = None
    query: type[BaseModel] | None = None
    form: list = field(default_factory=list)
    response: Any = None
    status: int = 200
    rate: str | None = None            # rate-limit bucket ("auth", "predict"); None = default bucket
    media_type: str | None = None      # non-JSON success responses (application/pdf, text/csv, image/png)


class APIRouter:
    def __init__(self, prefix: str = "", tags: list[str] | None = None):
        self.prefix, self.tags, self.endpoints = prefix.rstrip("/"), tags or [], []

    def _add(self, method: str, path: str, **kw):
        def deco(fn):
            ep = Endpoint(method=method, path=self.prefix + path, handler=fn, tags=kw.pop("tags", self.tags),
                          summary=kw.pop("summary", (fn.__doc__ or fn.__name__).strip().splitlines()[0]),
                          description=kw.pop("description", (fn.__doc__ or "").strip()), **kw)
            self.endpoints.append(ep)
            return fn
        return deco

    def get(self, path: str, **kw):
        return self._add("GET", path, **kw)

    def post(self, path: str, **kw):
        return self._add("POST", path, **kw)

    def patch(self, path: str, **kw):
        return self._add("PATCH", path, **kw)

    def put(self, path: str, **kw):
        return self._add("PUT", path, **kw)

    def delete(self, path: str, **kw):
        return self._add("DELETE", path, **kw)


def _bearer(request: Request) -> str | None:
    h = request.headers.get("authorization", "")
    if h.lower().startswith("bearer "):
        return h[7:].strip() or None
    return None


async def _read_form(request: Request, ep: Endpoint, settings) -> tuple[dict, list[Path]]:
    """Copy uploaded parts to temporary files, enforcing per-field size limits; return text fields as strings."""
    allowed = {f.name: f for f in ep.form}
    form = await request.form(max_files=8, max_fields=32, max_part_size=64 * 1024)
    out, temps = {}, []
    tmp_dir = Path(settings.storage_dir) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for key, value in form.multi_items():
            spec = allowed.get(key)
            if spec is None:
                continue
            if isinstance(spec, FileField):
                if not isinstance(value, UploadFile):
                    raise ApiError(422, "validation_error", f"'{key}' must be a file")
                limit = getattr(settings, spec.max_bytes_setting)
                h, size = hashlib.sha256(), 0
                fd = tempfile.NamedTemporaryFile(dir=tmp_dir, prefix="up_", suffix=".bin", delete=False)
                path = Path(fd.name)
                temps.append(path)
                with fd:
                    while True:
                        chunk = await value.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > limit:
                            raise ApiError(413, "payload_too_large",
                                           f"'{key}' exceeds the {limit // (1024 * 1024)} MB upload limit")
                        h.update(chunk)
                        fd.write(chunk)
                if size == 0:
                    raise ApiError(422, "validation_error", f"'{key}' is empty")
                out[key] = UploadedFile(key, safe_display_name(value.filename), value.content_type or "", path, size,
                                        h.hexdigest())
            else:
                if isinstance(value, UploadFile):
                    raise ApiError(422, "validation_error", f"'{key}' must be a text field")
                if len(value) > spec.max_length:
                    raise ApiError(422, "validation_error", f"'{key}' is too long")
                out[key] = value
        missing = [f.name for f in ep.form if f.required and f.name not in out]
        if missing:
            raise ApiError(422, "validation_error", "Missing form field(s): " + ", ".join(missing),
                           [{"field": m, "message": "field required", "type": "missing"} for m in missing])
    except BaseException:
        for t in temps:
            t.unlink(missing_ok=True)
        raise
    finally:
        await form.close()
    return out, temps


def make_route(ep: Endpoint) -> Route:
    async def endpoint(request: Request) -> Response:
        state = request.app.state
        settings = state.settings
        temps: list[Path] = []
        try:
            # Rate limit before any parsing.
            bucket = ep.rate or "default"
            key_part = request.client.host if request.client else "unknown"
            state.rate_limiter.enforce(bucket, key_part, settings)
            claims = None
            if ep.auth:
                token = _bearer(request)
                if not token:
                    raise unauthorized()
                claims = state.auth_service.decode(token)      # signature/expiry check, no DB, before reading body
            ctx = Ctx(request=request, state=state, path=dict(request.path_params))
            raw_body = None
            if ep.body is not None:
                raw_body = await request.body()
            elif ep.form:
                ctype = request.headers.get("content-type", "")
                if not ctype.startswith("multipart/form-data"):
                    raise ApiError(415, "unsupported_media_type", "Expected multipart/form-data")
                ctx.form, temps = await _read_form(request, ep, settings)
            query = dict(request.query_params)

            def run():
                if claims is not None:
                    ctx.user = state.auth_service.user_from_claims(claims)
                    if isinstance(ep.auth, tuple) and ctx.user["role"] not in ep.auth:
                        raise ApiError(403, "forbidden", "Your role does not allow this action")
                    if ep.rate == "predict":
                        state.rate_limiter.enforce("predict_user", ctx.user["id"], settings)
                if ep.query is not None:
                    ctx.query = ep.query.model_validate(query)
                if ep.body is not None:
                    try:
                        data = json.loads(raw_body or b"null")
                    except ValueError:
                        raise ApiError(400, "invalid_json", "Request body is not valid JSON")
                    ctx.body = ep.body.model_validate(data)
                return ep.handler(ctx)

            result = await run_in_threadpool(run)
        except ApiError as e:
            return error_response(e)
        except ValidationError as e:
            return error_response(ApiError(422, "validation_error", "Request validation failed", validation_details(e)))
        finally:
            for t in temps:
                t.unlink(missing_ok=True)
        if isinstance(result, Response):
            return result
        if result is None:
            return Response(status_code=204)
        return JSONResponse(result, status_code=ep.status)

    endpoint.__name__ = ep.handler.__name__
    return Route(ep.path, endpoint, methods=[ep.method], name=f"{ep.method.lower()}_{ep.handler.__name__}")


# --------------------------------------------------------------------------------------------- OpenAPI
def _unwrap(tp):
    origin = typing.get_origin(tp)
    if origin in (list, tuple):
        return "array", typing.get_args(tp)[0]
    return "object", tp


def openapi_schema(endpoints: list[Endpoint], title: str, version: str, description: str) -> dict:
    models: dict[tuple, None] = {}
    for ep in endpoints:
        if ep.body is not None:
            models[(ep.body, "validation")] = None
        if ep.response is not None:
            _, inner = _unwrap(ep.response)
            if isinstance(inner, type) and issubclass(inner, BaseModel):
                models[(inner, "serialization")] = None
    from app.schemas.common import ErrorResponse
    models[(ErrorResponse, "serialization")] = None
    keys = list(models)
    refs, top = models_json_schema(keys, ref_template="#/components/schemas/{model}")
    schemas = top.get("$defs", {})

    def ref(model, mode):
        return refs[(model, mode)]

    paths: dict = {}
    for ep in endpoints:
        op: dict = {"operationId": f"{ep.method.lower()}_{ep.handler.__name__}", "summary": ep.summary,
                    "tags": ep.tags, "responses": {}}
        if ep.description and ep.description != ep.summary:
            op["description"] = ep.description
        params = [{"name": n, "in": "path", "required": True, "schema": {"type": "string"}}
                  for n in re.findall(r"{(\w+)(?::\w+)?}", ep.path)]
        if ep.query is not None:
            qs = ep.query.model_json_schema()
            for name, prop in qs.get("properties", {}).items():
                params.append({"name": name, "in": "query", "required": name in qs.get("required", []),
                               "schema": {k: v for k, v in prop.items() if k != "title"},
                               "description": prop.get("description", "")})
        if params:
            op["parameters"] = params
        if ep.body is not None:
            op["requestBody"] = {"required": True,
                                 "content": {"application/json": {"schema": ref(ep.body, "validation")}}}
        elif ep.form:
            props, req = {}, []
            for f in ep.form:
                props[f.name] = ({"type": "string", "format": "binary", "description": f.description}
                                 if isinstance(f, FileField) else {"type": "string", "description": f.description})
                if f.required:
                    req.append(f.name)
            op["requestBody"] = {"required": True, "content": {"multipart/form-data": {
                "schema": {"type": "object", "properties": props, "required": req}}}}
        ok: dict = {"description": "Success"}
        if ep.media_type:
            ok["content"] = {ep.media_type: {"schema": {"type": "string", "format": "binary"}}}
        elif ep.response is not None:
            kind, inner = _unwrap(ep.response)
            if isinstance(inner, type) and issubclass(inner, BaseModel):
                s = ref(inner, "serialization")
                ok["content"] = {"application/json": {"schema": {"type": "array", "items": s} if kind == "array" else s}}
        op["responses"][str(ep.status)] = ok if ep.status != 204 else {"description": "No content"}
        err = {"content": {"application/json": {"schema": ref(ErrorResponse, "serialization")}}}
        if ep.auth:
            op["security"] = [{"bearerAuth": []}]
            op["responses"]["401"] = {"description": "Not authenticated", **err}
            if isinstance(ep.auth, tuple):
                op["x-required-role"] = " / ".join(ep.auth)
                op["responses"]["403"] = {"description": "Role not allowed", **err}
        if ep.body is not None or ep.query is not None or ep.form:
            op["responses"]["422"] = {"description": "Validation error", **err}
        op["responses"]["429"] = {"description": "Rate limit exceeded", **err}
        paths.setdefault(re.sub(r"{(\w+):\w+}", r"{\1}", ep.path), {})[ep.method.lower()] = op
    return {"openapi": "3.1.0", "info": {"title": title, "version": version, "description": description},
            "paths": paths,
            "components": {"schemas": schemas,
                           "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer",
                                                              "bearerFormat": "JWT"}}}}


SWAGGER_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title} – API docs</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"></head>
<body><div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>window.ui = SwaggerUIBundle({{url: '{spec}', dom_id: '#swagger-ui', persistAuthorization: true}});</script>
<noscript>Swagger UI needs JavaScript and access to cdn.jsdelivr.net. The raw specification is at {spec}.</noscript>
</body></html>"""
