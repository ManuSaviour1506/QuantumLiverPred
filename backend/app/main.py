"""Liver Tumour Intelligence Platform V10 — API application factory.

    uvicorn app.main:app --host 0.0.0.0 --port 8000        (run from backend/, project root on PYTHONPATH)

Serves the REST API under /api/v1, health checks under /health, OpenAPI at /openapi.json, Swagger UI at /docs and,
when `frontend/dist` exists, the built single-page application.
"""
from __future__ import annotations

import contextlib
import importlib
import logging
import sys
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from app.core.apikit import SWAGGER_HTML, JSONResponse, make_route, openapi_schema
from app.core.config import DISCLAIMER, PROJECT_ROOT, Settings
from app.core.errors import error_body
from app.core.logging import configure_logging, log_event
from app.core.middleware import BodySizeLimitMiddleware, RateLimiter, RequestContextMiddleware, SecurityHeadersMiddleware
from app.db import migrate
from app.db.database import Database
from app.ml.engine import EngineHolder
from app.services.auth import AuthService
from app.services.jobs import JobQueue
from app.services.storage import make_storage

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger("lti.app")

ROUTER_MODULES = ["system", "auth", "users", "models", "predictions", "datasets", "experiments", "analytics",
                  "research", "benchmarks", "compare"]


def _routers():
    out = []
    for name in ROUTER_MODULES:
        mod = importlib.import_module(f"app.api.v1.{name}")
        out.append(mod.router)
    return out


def create_app(settings: Settings | None = None, start_background: bool = True) -> Starlette:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level, settings.log_json)
    if settings.jwt_secret_generated:
        log.warning("JWT_SECRET not set: using a random per-process secret (sessions end when the server restarts)")
    db = Database(settings.database_url)
    if settings.auto_migrate:
        applied = migrate.upgrade(db)
        if applied:
            log_event(log, "migrations applied", migrations=applied)

    routers = _routers()
    endpoints = [ep for r in routers for ep in r.endpoints]
    spec = openapi_schema(endpoints, settings.app_name + " API", settings.version,
                          "REST API of the research platform. " + DISCLAIMER)

    async def openapi_json(request: Request):
        return JSONResponse(spec)

    async def docs(request: Request):
        return HTMLResponse(SWAGGER_HTML.format(title=settings.app_name, spec="/openapi.json"))

    routes = [Route("/openapi.json", openapi_json, methods=["GET"]), Route("/docs", docs, methods=["GET"])]
    routes += [make_route(ep) for ep in endpoints]

    dist = Path(settings.frontend_dist)
    if (dist / "index.html").is_file():
        index_html = (dist / "index.html").read_bytes()
        if (dist / "assets").is_dir():
            routes.append(Mount("/assets", app=StaticFiles(directory=dist / "assets"), name="assets"))

        async def spa(request: Request):
            path = request.path_params.get("path", "")
            if path.startswith("api/") or path in ("health", "openapi.json", "docs"):
                return JSONResponse(error_body("not_found", "Not found"), status_code=404)
            candidate = (dist / path).resolve()
            if path and dist.resolve() in candidate.parents and candidate.is_file() and candidate.name != "index.html":
                return FileResponse(candidate)
            return Response(index_html, media_type="text/html", headers={"cache-control": "no-cache"})
        routes.append(Route("/{path:path}", spa, methods=["GET"]))

    async def not_found(request: Request, exc):
        return JSONResponse(error_body("not_found", "Not found"), status_code=404)

    async def method_not_allowed(request: Request, exc):
        return JSONResponse(error_body("method_not_allowed", "Method not allowed"), status_code=405)

    async def server_error(request: Request, exc):
        log.error("unhandled error", exc_info=exc)
        return JSONResponse(error_body("internal_error", "An unexpected error occurred; quote the request id when "
                                                         "reporting it"), status_code=500)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        if start_background:
            if settings.worker_mode == "inline":
                app.state.jobs.recover()
                app.state.jobs.start(settings.worker_threads)
            if settings.preload_models:
                app.state.engine.preload_async()
            app.state.process_monitor.start()
        yield
        app.state.process_monitor.stop()
        app.state.jobs.stop()

    app = Starlette(routes=routes, lifespan=lifespan,
                    exception_handlers={404: not_found, 405: method_not_allowed, 500: server_error,
                                        Exception: server_error})
    st = app.state
    st.settings, st.db = settings, db
    st.storage = make_storage(settings)
    st.rate_limiter = RateLimiter()
    st.auth_service = AuthService(settings, db)
    st.engine = EngineHolder(settings)
    st.jobs = JobQueue(db, settings)
    st.openapi = spec
    _wire_services(app)

    app.add_middleware(BodySizeLimitMiddleware, default_limit=settings.max_upload_bytes * 4,
                       overrides={"/api/v1/datasets/upload": settings.max_dataset_upload_bytes})
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                       allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
                       allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
                       expose_headers=["X-Request-ID", "Content-Disposition", "Retry-After"], max_age=600)
    app.add_middleware(SecurityHeadersMiddleware, hsts=settings.hsts)
    app.add_middleware(RequestContextMiddleware)
    return app


def _wire_services(app) -> None:
    """Create domain services and register their background-job handlers."""
    st = app.state
    from app.services.datasets import DatasetService
    from app.services.experiments import ExperimentService
    from app.services.models import ModelService
    from app.services.predictions import PredictionService
    st.model_service = ModelService(st.settings, st.db, st.engine)
    st.prediction_service = PredictionService(st.settings, st.db, st.storage, st.engine, st.jobs, st.model_service)
    st.dataset_service = DatasetService(st.settings, st.db, st.storage, st.jobs)
    st.experiment_service = ExperimentService(st.settings, st.db, st.jobs, st.dataset_service)
    from app.execution.manager import ComparisonManager
    from app.execution.service import ComparisonService
    st.comparison_manager = ComparisonManager(st.settings, st.db, st.storage, st.engine, st.model_service)
    st.comparison_service = ComparisonService(st.settings, st.db, st.storage, st.model_service,
                                              st.comparison_manager)
    st.model_service.comparison_service = st.comparison_service
    from app.benchmarking.service import BenchmarkService, ProcessMonitor
    st.benchmark_service = BenchmarkService(st.settings, st.db, st.model_service)
    st.process_monitor = ProcessMonitor()
    st.jobs.register("prediction", st.prediction_service.execute)
    st.jobs.register("dataset_validation", st.dataset_service.execute_validation)
    st.jobs.register("experiment", st.experiment_service.execute)
    try:
        st.experiment_service.import_recorded()
    except Exception:
        log.error("could not import recorded experiments", exc_info=True)


def _lazy_app():
    return create_app()


class _LazyASGI:
    """`app.main:app` for uvicorn without building the app at import time (tests import create_app directly)."""

    def __init__(self):
        self._app = None

    async def __call__(self, scope, receive, send):
        if self._app is None:
            self._app = _lazy_app()
        await self._app(scope, receive, send)


app = _LazyASGI()
