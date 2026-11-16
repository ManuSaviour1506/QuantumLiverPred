"""Prediction endpoints: create (multipart upload, queued), list / search / export, detail, artifacts, report, delete."""
from __future__ import annotations

from starlette.responses import Response

from app.core.apikit import APIRouter, Ctx, FileField, TextField
from app.core.errors import ApiError
from app.schemas.common import page_meta
from app.schemas.predictions import PredictionCreated, PredictionListQuery, PredictionOut, PredictionPage

router = APIRouter("/api/v1/predictions", tags=["predictions"])


@router.post("", status=202, rate="predict", response=PredictionCreated,
             form=[FileField("file", True, "CT slice: PNG, JPEG, uncompressed DICOM or NIfTI"),
                   TextField("model_id", True, "id of a servable model (GET /api/v1/models?servable_only=true)"),
                   FileField("prev_file", False, "previous slice (required by 2.5-D models)"),
                   FileField("next_file", False, "next slice (required by 2.5-D models)"),
                   TextField("explain", False, "'true' to compute explanation maps when supported")])
def create_prediction(ctx: Ctx):
    """Queue a prediction. Returns 202 with the id; poll GET /api/v1/predictions/{id} until status is final."""
    f = ctx.form
    return ctx.state.prediction_service.create(ctx.user, f["model_id"].strip(), f["file"], f.get("prev_file"),
                                               f.get("next_file"), str(f.get("explain", "")).lower() in ("1", "true"),
                                               ctx.client_ip)


@router.get("", query=PredictionListQuery, response=PredictionPage)
def list_predictions(ctx: Ctx):
    """Prediction history with search, filters, sorting and pagination (own records; admins may request all)."""
    items, total = ctx.state.prediction_service.list(ctx.user, ctx.query)
    return {"items": items, "meta": page_meta(ctx.query.page, ctx.query.page_size, total)}


@router.get("/export.csv", query=PredictionListQuery, media_type="text/csv")
def export_predictions(ctx: Ctx):
    """CSV export of the filtered history (no images)."""
    csv_text = ctx.state.prediction_service.export_csv(ctx.user, ctx.query)
    return Response(csv_text, media_type="text/csv; charset=utf-8",
                    headers={"content-disposition": 'attachment; filename="prediction_history.csv"'})


@router.get("/{prediction_id}", response=PredictionOut)
def get_prediction(ctx: Ctx):
    """Prediction detail: status, result, explanation metadata and artifact links."""
    return ctx.state.prediction_service.get(ctx.user, ctx.path["prediction_id"])


@router.get("/{prediction_id}/artifacts/{kind}", media_type="image/png")
def prediction_artifact(ctx: Ctx):
    """Preview, patch-probability map or Grad-CAM image of a prediction (owner or admin only)."""
    kind = ctx.path["kind"]
    if kind not in ("preview", "patch_map", "gradcam"):
        raise ApiError(404, "not_found", "Artifact not found")
    data, mime = ctx.state.prediction_service.artifact(ctx.user, ctx.path["prediction_id"], kind)
    return Response(data, media_type=mime, headers={"cache-control": "private, max-age=300"})


@router.get("/{prediction_id}/report", media_type="application/pdf")
def prediction_report(ctx: Ctx):
    """PDF research report of a completed prediction (not a clinical report)."""
    pdf = ctx.state.prediction_service.report_pdf(ctx.user, ctx.path["prediction_id"])
    return Response(pdf, media_type="application/pdf",
                    headers={"content-disposition": f'attachment; filename="prediction_{ctx.path["prediction_id"]}.pdf"'})


@router.delete("/{prediction_id}", status=204)
def delete_prediction(ctx: Ctx):
    """Delete a prediction and its stored images."""
    ctx.state.prediction_service.delete(ctx.user, ctx.path["prediction_id"], ctx.client_ip)
    return None
