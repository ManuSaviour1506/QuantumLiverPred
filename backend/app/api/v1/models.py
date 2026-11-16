"""Model registry endpoints."""
from __future__ import annotations

from pydantic import BaseModel, Field
from starlette.responses import Response

from app.core.apikit import APIRouter, Ctx, to_json_bytes
from app.schemas.models import ModelMetricsOut, ModelOut, ModelStatusOut

router = APIRouter("/api/v1/models", tags=["models"])


class ModelListQuery(BaseModel):
    family: str | None = Field(None, max_length=30)
    servable_only: bool = False


@router.get("", query=ModelListQuery, response=list[ModelOut])
def list_models(ctx: Ctx):
    """Every registered model (V9 and V10) with status, deployment state and recorded metrics."""
    return ctx.state.model_service.list(ctx.query.family, ctx.query.servable_only)


@router.get("/availability")
def availability(ctx: Ctx):
    """Runtime availability of every registered model: category, dependencies (weights, upstream models), input
    requirements (neighbouring slices), last successful inference, and the reason when it cannot run."""
    return ctx.state.model_service.availability()


@router.get("/{model_id}", response=ModelOut)
def get_model(ctx: Ctx):
    """One model: architecture, version, training data, artifact, input contract, metrics."""
    return ctx.state.model_service.get(ctx.path["model_id"])


@router.get("/{model_id}/metrics", response=ModelMetricsOut)
def model_metrics(ctx: Ctx):
    """Saved evaluation results with their split and evaluation unit."""
    return ctx.state.model_service.metrics(ctx.path["model_id"])


@router.get("/{model_id}/status", response=ModelStatusOut)
def model_status(ctx: Ctx):
    """Load / integrity status and whether the model can serve predictions."""
    return ctx.state.model_service.status(ctx.path["model_id"])


@router.get("/{model_id}/metadata.json", media_type="application/json")
def model_metadata_export(ctx: Ctx):
    """Download the model's metadata as a JSON file."""
    m = ctx.state.model_service.get(ctx.path["model_id"])
    return Response(to_json_bytes(m), media_type="application/json",
                    headers={"content-disposition": f'attachment; filename="{m["id"]}_metadata.json"'})


@router.post("/sync", auth=("admin",))
def sync_registry(ctx: Ctx):
    """Reload the model engine from disk and mirror the registry into the database (administrators)."""
    ctx.state.engine.reload()
    ctx.state.model_service._synced = False
    return ctx.state.model_service.sync()
