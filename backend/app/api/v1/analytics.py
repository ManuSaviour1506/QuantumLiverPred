"""Analytics endpoints for the dashboard and model comparison."""
from __future__ import annotations

import csv
import io

from pydantic import BaseModel
from starlette.responses import Response

from app.core.apikit import APIRouter, Ctx
from app.schemas.models import OverviewOut
from app.services import analytics as A

router = APIRouter("/api/v1/analytics", tags=["analytics"])


class DistQuery(BaseModel):
    all_users: bool = False


@router.get("/overview", response=OverviewOut)
def overview(ctx: Ctx):
    """Counts from the database and the model registry (user totals for administrators only)."""
    return A.overview(ctx.state, ctx.user)


@router.get("/model-comparison")
def model_comparison(ctx: Ctx):
    """Saved evaluation results of every trained and evaluated model, with split and evaluation unit."""
    return A.model_comparison(ctx.state)


@router.get("/model-comparison.csv", media_type="text/csv")
def model_comparison_csv(ctx: Ctx):
    """Model comparison as CSV."""
    rows = A.model_comparison(ctx.state)["rows"]
    buf = io.StringIO()
    w = csv.writer(buf)
    keys = list(A.METRIC_KEYS)
    w.writerow(["model_id", "name", "family", "source", "evaluation", "evaluation_unit", "primary_metric", "roc_auc",
                *keys, "train_seconds", "inference_ms_per_patch", "size_bytes", "parameters", "execution_mode"])
    for r in rows:
        w.writerow([r["model_id"], r["name"], r["family"], r["source"], r["evaluation"], r["evaluation_unit"],
                    r["primary_metric"], r["roc_auc"], *[r["metrics"].get(k) for k in keys], r["train_seconds"],
                    r["inference_ms_per_patch"], r["size_bytes"], r["parameters"], r["execution_mode"]])
    return Response(buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"content-disposition": 'attachment; filename="model_comparison.csv"'})


@router.get("/model-comparison.pdf", media_type="application/pdf")
def model_comparison_pdf(ctx: Ctx):
    """Model comparison report (PDF)."""
    from app.services.reports import comparison_report
    data = A.model_comparison(ctx.state)
    return Response(comparison_report(data["rows"], data["generated_at"]), media_type="application/pdf",
                    headers={"content-disposition": 'attachment; filename="model_comparison.pdf"'})


@router.get("/prediction-distribution", query=DistQuery)
def prediction_distribution(ctx: Ctx):
    """Predictions by label, model, status, day and probability bin."""
    return A.prediction_distribution(ctx.state, ctx.user, ctx.query.all_users)
