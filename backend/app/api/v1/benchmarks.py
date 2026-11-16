"""V11 new-evaluation benchmark runs and the live system monitor."""
from __future__ import annotations

from pydantic import BaseModel, Field
from starlette.responses import Response

from app.core.apikit import APIRouter, Ctx

router = APIRouter("/api/v1", tags=["benchmarks"])


class BenchmarkRunIn(BaseModel):
    dataset: str = Field("lits_sample", max_length=100)
    split: str | None = Field(None, max_length=30)
    max_slices: int = Field(0, ge=0, le=5000)
    model_ids: list[str] | None = Field(None, max_length=100)


@router.get("/benchmarks/datasets")
def benchmark_datasets(ctx: Ctx):
    """Ground-truth-labelled slice sets available for new evaluations (with splits and scan groups)."""
    return ctx.state.benchmark_service.datasets()


@router.post("/benchmarks/runs", status=202, body=BenchmarkRunIn)
def start_benchmark(ctx: Ctx):
    """Start a new evaluation: every (or the selected) model on a labelled dataset; metrics per evaluation role."""
    b = ctx.body
    return ctx.state.benchmark_service.start(ctx.user, b.dataset, b.split, b.max_slices, b.model_ids)


@router.get("/benchmarks/runs")
def list_benchmarks(ctx: Ctx):
    """New-evaluation runs (newest first)."""
    return ctx.state.benchmark_service.list(ctx.user)


@router.get("/benchmarks/runs/{bench_id}")
def get_benchmark(ctx: Ctx):
    """Status, progress and results of a new-evaluation run."""
    return ctx.state.benchmark_service.get(ctx.user, ctx.path["bench_id"])


@router.post("/benchmarks/runs/{bench_id}/cancel")
def cancel_benchmark(ctx: Ctx):
    """Stop a running evaluation after the current slice."""
    return ctx.state.benchmark_service.cancel(ctx.user, ctx.path["bench_id"])


@router.get("/benchmarks/runs/{bench_id}/export.csv", media_type="text/csv")
def export_benchmark(ctx: Ctx):
    """Per-model, per-role metrics of a new-evaluation run as CSV."""
    bid = ctx.path["bench_id"]
    return Response(ctx.state.benchmark_service.export_csv(ctx.user, bid), media_type="text/csv; charset=utf-8",
                    headers={"content-disposition": f'attachment; filename="benchmark_{bid}.csv"'})


@router.get("/system/metrics")
def system_metrics(ctx: Ctx):
    """Live process metrics of the API server (CPU %, memory, threads, GPU detection), engine load times and the
    number of executing comparison runs."""
    snap = ctx.state.process_monitor.snapshot()
    mgr = ctx.state.comparison_manager
    snap["active_runs"] = sum(1 for s in list(mgr.runs.values()) if s.status in ("queued", "running"))
    eng = ctx.state.engine.engine
    if eng is not None:
        lm = eng.model_load_ms
        snap["engine"] = {"load_seconds": eng.load_seconds, "models_loaded": len(lm),
                          "slowest_loads_ms": sorted(lm.items(), key=lambda kv: -kv[1])[:8],
                          "total_model_load_ms": round(sum(lm.values()), 1)}
    snap["hardware"] = mgr.hardware
    return snap
