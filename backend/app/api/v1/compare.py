"""Four-paradigm comparison API: upload + validation, dataset samples, run-all / run-selected, live stream (SSE),
cancel, history, run-to-run comparison, CSV / PDF export, benchmarks and system status."""
from __future__ import annotations

from pydantic import BaseModel, Field
from starlette.responses import Response, StreamingResponse

from app.core.apikit import APIRouter, Ctx, FileField, TextField

router = APIRouter("/api/v1", tags=["comparison"])


class SampleUploadIn(BaseModel):
    sample_id: str = Field(..., max_length=200)
    modality: str = Field("CT", max_length=10)


class RunAllIn(BaseModel):
    upload_id: str = Field(..., max_length=64)


class RunSelectedIn(BaseModel):
    upload_id: str = Field(..., max_length=64)
    model_ids: list[str] = Field(..., min_length=1, max_length=100)


class HistoryQuery(BaseModel):
    limit: int = Field(50, ge=1, le=200)
    offset: int = Field(0, ge=0)


class CompareRunsQuery(BaseModel):
    a: str = Field(..., max_length=64)
    b: str = Field(..., max_length=64)


class StreamQuery(BaseModel):
    last_event_id: int = Field(-1, ge=-1)


@router.post("/upload", status=201, rate="predict",
             form=[FileField("file", True, "CT slice: PNG, JPEG, uncompressed DICOM or NIfTI"),
                   FileField("prev_file", False, "previous slice (2.5-D models)"),
                   FileField("next_file", False, "next slice (2.5-D models)"),
                   FileField("prev2_file", False, "second slice above (3-D / 5-slice models)"),
                   FileField("next2_file", False, "second slice below (3-D / 5-slice models)"),
                   TextField("modality", False, "CT (default) or MRI")])
def upload(ctx: Ctx):
    """Upload and validate an image: format, quality control, out-of-distribution check, liver gate + U-Net liver
    validation, and the compatibility of every registered model with this input."""
    f = ctx.form
    context = {"prev": f.get("prev_file"), "next": f.get("next_file"), "prev2": f.get("prev2_file"),
               "next2": f.get("next2_file")}
    return ctx.state.comparison_service.create_upload(ctx.user, f["file"], context, f.get("modality") or "CT")


@router.post("/upload/sample", status=201, rate="predict", body=SampleUploadIn)
def upload_sample(ctx: Ctx):
    """Use a slice from the bundled dataset; its real neighbouring slices from the same scan are attached."""
    return ctx.state.comparison_service.create_upload(ctx.user, None, {}, ctx.body.modality, ctx.body.sample_id)


@router.get("/uploads/{upload_id}")
def get_upload(ctx: Ctx):
    """A validated upload with its analysis."""
    return ctx.state.comparison_service.get_upload(ctx.user, ctx.path["upload_id"])


@router.get("/compare/samples")
def samples(ctx: Ctx):
    """Real CT sample slices (with ground truth and available neighbours) and synthetic OOD test phantoms."""
    return ctx.state.comparison_service.samples()


@router.post("/compare/run-all", status=202, rate="predict", body=RunAllIn)
def run_all(ctx: Ctx):
    """Run every registered model: the four family pipelines start concurrently. Returns the run id; follow
    GET /api/v1/compare/{run_id}/stream for live events."""
    return ctx.state.comparison_service.start(ctx.user, ctx.body.upload_id, "run_all", None)


@router.post("/compare/run-selected", status=202, rate="predict", body=RunSelectedIn)
def run_selected(ctx: Ctx):
    """Run only the selected models (still grouped into concurrent family pipelines)."""
    return ctx.state.comparison_service.start(ctx.user, ctx.body.upload_id, "run_selected", ctx.body.model_ids)


@router.get("/compare/{run_id}")
def get_run(ctx: Ctx):
    """Run status and, when finished, the full result."""
    return ctx.state.comparison_service.get_run(ctx.user, ctx.path["run_id"])


@router.get("/compare/{run_id}/stream", query=StreamQuery, media_type="text/event-stream")
def stream(ctx: Ctx):
    """Server-Sent Events: run_started, preprocessing_completed, plan, family_started, model_started,
    model_completed, model_failed, model_skipped, family_completed, run_completed / run_cancelled / run_failed."""
    last = ctx.query.last_event_id
    hdr = ctx.request.headers.get("last-event-id")
    if hdr and hdr.isdigit():
        last = int(hdr)
    gen = ctx.state.comparison_manager.stream(ctx.user, ctx.path["run_id"], last)
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"cache-control": "no-cache", "x-accel-buffering": "no"})


@router.post("/compare/{run_id}/cancel")
def cancel(ctx: Ctx):
    """Cancel a running comparison (models not yet started are skipped)."""
    return ctx.state.comparison_manager.cancel(ctx.user, ctx.path["run_id"])


@router.get("/compare/{run_id}/export.csv", media_type="text/csv")
def export_csv(ctx: Ctx):
    """Per-model and per-family results of a run as CSV."""
    rid = ctx.path["run_id"]
    return Response(ctx.state.comparison_service.export_csv(ctx.user, rid), media_type="text/csv; charset=utf-8",
                    headers={"content-disposition": f'attachment; filename="comparison_{rid}.csv"'})


@router.get("/compare/{run_id}/report", media_type="application/pdf")
def report(ctx: Ctx):
    """PDF research report of a run (all four families, measured timings)."""
    rid = ctx.path["run_id"]
    return Response(ctx.state.comparison_service.report_pdf(ctx.user, rid), media_type="application/pdf",
                    headers={"content-disposition": f'attachment; filename="comparison_{rid}.pdf"'})


@router.get("/history", query=HistoryQuery)
def history(ctx: Ctx):
    """Saved comparison runs (newest first)."""
    return ctx.state.comparison_service.history(ctx.user, ctx.query.limit, ctx.query.offset)


@router.get("/history/compare", query=CompareRunsQuery)
def compare_two(ctx: Ctx):
    """Side-by-side difference of two runs (labels, probabilities, inference times)."""
    return ctx.state.comparison_service.compare_runs(ctx.user, ctx.query.a, ctx.query.b)


@router.get("/benchmarks")
def benchmarks(ctx: Ctx):
    """Recorded evaluation metrics per model and family (labelled by protocol) plus latencies measured in runs."""
    return ctx.state.comparison_service.benchmarks(ctx.user)


@router.get("/benchmarks/summary.csv", media_type="text/csv")
def benchmarks_csv(ctx: Ctx):
    """Benchmark summary as CSV."""
    return Response(ctx.state.comparison_service.benchmarks_csv(ctx.user), media_type="text/csv; charset=utf-8",
                    headers={"content-disposition": 'attachment; filename="benchmark_summary.csv"'})


@router.get("/system/status")
def system_status(ctx: Ctx):
    """Engine state, model availability counts, weights, liver validation, comparison settings and run statistics."""
    eng_status = ctx.state.engine.status()
    out = {"engine": eng_status, "backend": "ok", "version": ctx.settings.version,
           "comparison": {"max_concurrent_runs": ctx.settings.compare_max_concurrent_runs,
                          "model_workers_per_family": ctx.settings.compare_model_workers_per_family,
                          "run_timeout_s": ctx.settings.compare_run_timeout_s,
                          "scheduling": "family_parallel_model_sequential" if
                          ctx.settings.compare_model_workers_per_family == 1 else "family_parallel_model_parallel"},
           "hardware": ctx.state.comparison_manager.hardware}
    out["stats"] = ctx.state.comparison_service.stats(ctx.user)
    eng = ctx.state.engine.engine
    if eng is not None:
        ds = eng.list(include_legacy=True)
        from collections import Counter
        out["availability"] = dict(Counter(d.get("availability") for d in ds))
        out["families"] = {f: {"registered": sum(1 for d in ds if d.get("compare_family") == f and d.get("status") != "legacy"),
                               "servable": sum(1 for d in ds if d.get("compare_family") == f and d.get("servable"))}
                           for f in ("classical", "deep_learning", "quantum", "hybrid")}
    return out
