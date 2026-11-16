"""Dataset schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.common import PageMeta, PageQuery


class DatasetVersionOut(BaseModel):
    id: str
    version: int
    status: str
    n_files: int
    content_sha256: str | None = None
    manifest_columns: list[str] = Field(default_factory=list)
    validation: dict[str, Any] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    created_at: str
    validated_at: str | None = None


class DatasetOut(BaseModel):
    id: str
    name: str
    description: str
    owner_id: str | None = None
    visibility: str
    source: str
    license: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    latest_version: DatasetVersionOut | None = None
    versions: list[DatasetVersionOut] = Field(default_factory=list)


class DatasetFilesQuery(PageQuery):
    status: str | None = Field(None, max_length=30)
    label: str | None = Field(None, max_length=60)
    q: str | None = Field(None, max_length=100)
    version: int | None = Field(None, ge=1)


class DatasetFileOut(BaseModel):
    row_index: int
    image_path: str
    label: str | None = None
    patient_id: str | None = None
    scan_id: str | None = None
    volume_id: str | None = None
    slice_id: str | None = None
    modality: str | None = None
    mask_path: str | None = None
    split: str | None = None
    data_source: str | None = None
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    file_status: str
    issues: list[Any] = Field(default_factory=list)


class DatasetFilesPage(BaseModel):
    items: list[DatasetFileOut]
    meta: PageMeta


class ValidateIn(BaseModel):
    version: int | None = Field(None, ge=1, description="default: latest version")


class JobAccepted(BaseModel):
    job_id: str
    status: str
    message: str
