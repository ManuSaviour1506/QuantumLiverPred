"""Dataset endpoints: upload / register, list, detail, validation (queued), statistics, manifest viewer, delete."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.apikit import APIRouter, Ctx, FileField, TextField
from app.core.errors import ApiError
from app.schemas.common import page_meta
from app.schemas.datasets import DatasetFilesPage, DatasetFilesQuery, DatasetOut, JobAccepted, ValidateIn

router = APIRouter("/api/v1/datasets", tags=["datasets"])


@router.post("/upload", status=201, response=DatasetOut,
             form=[FileField("file", True, ".zip with images and manifest.csv", "max_dataset_upload_bytes"),
                   TextField("name", True, "dataset name", 120), TextField("description", False, "", 2000),
                   TextField("license", False, "licence of the images", 200),
                   TextField("data_source", False, "where the images come from", 500)])
def upload_dataset(ctx: Ctx):
    """Upload a dataset archive; it is stored privately and validated in the background."""
    f = ctx.form
    name = (f.get("name") or "").strip()
    if not name:
        raise ApiError(422, "validation_error", "name is required")
    return ctx.state.dataset_service.upload(ctx.user, f["file"], name, f.get("description", ""), f.get("license", ""),
                                            f.get("data_source", ""), ctx.client_ip)


class DatasetRegisterIn(BaseModel):
    path: str = Field(..., min_length=1, max_length=500)
    name: str = Field(..., min_length=1, max_length=120)
    layout: str = Field("lits_verified", pattern="^(lits_verified|manifest)$")


@router.post("/register", auth=("admin",), body=DatasetRegisterIn, response=DatasetOut, status=201)
def register_dataset(ctx: Ctx):
    """Register a folder already on the server (must be inside DATASET_ROOTS); administrators only."""
    from pathlib import Path
    b = ctx.body
    return ctx.state.dataset_service.register_local(ctx.user, Path(b.path), b.name, b.layout)


@router.get("", response=list[DatasetOut])
def list_datasets(ctx: Ctx):
    """Datasets visible to the user (own and shared)."""
    return ctx.state.dataset_service.list(ctx.user)


@router.get("/{dataset_id}", response=DatasetOut)
def get_dataset(ctx: Ctx):
    """Dataset with all versions and their validation reports."""
    return ctx.state.dataset_service.get(ctx.user, ctx.path["dataset_id"])


@router.post("/{dataset_id}/validate", body=ValidateIn, response=JobAccepted, status=202)
def validate_dataset(ctx: Ctx):
    """Queue validation of the latest (or a given) version."""
    return ctx.state.dataset_service.request_validation(ctx.user, ctx.path["dataset_id"], ctx.body.version,
                                                        ctx.client_ip)


@router.get("/{dataset_id}/statistics")
def dataset_statistics(ctx: Ctx):
    """Label, split and group distributions, image sizes, intensity summary, identifier availability."""
    return ctx.state.dataset_service.statistics(ctx.user, ctx.path["dataset_id"])


@router.get("/{dataset_id}/files", query=DatasetFilesQuery, response=DatasetFilesPage)
def dataset_files(ctx: Ctx):
    """Manifest viewer (paginated, filterable)."""
    items, total = ctx.state.dataset_service.files(ctx.user, ctx.path["dataset_id"], ctx.query)
    return {"items": items, "meta": page_meta(ctx.query.page, ctx.query.page_size, total)}


@router.delete("/{dataset_id}", status=204)
def delete_dataset(ctx: Ctx):
    """Delete a dataset you own (uploaded files are removed; registered folders are never touched)."""
    ctx.state.dataset_service.delete(ctx.user, ctx.path["dataset_id"], ctx.client_ip)
    return None
