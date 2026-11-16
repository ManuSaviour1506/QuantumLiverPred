"""Prediction schemas."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.common import PageMeta, PageQuery

PredictionStatus = Literal["queued", "running", "succeeded", "failed"]


class PredictionListQuery(PageQuery):
    q: str | None = Field(None, max_length=100, description="search in file name, model id and label")
    model_id: str | None = Field(None, max_length=100)
    status: PredictionStatus | None = None
    label: str | None = Field(None, max_length=60)
    date_from: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    date_to: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    sort: Literal["created_at", "-created_at", "probability", "-probability", "model_id", "-model_id"] = "-created_at"
    all_users: bool = Field(False, description="admins only: include every user's predictions")


class ArtifactOut(BaseModel):
    kind: str
    mime: str
    size_bytes: int
    url: str


class PredictionSummary(BaseModel):
    id: str
    user_id: str
    model_id: str
    model_name: str | None = None
    model_version: str | None = None
    status: PredictionStatus
    label: str | None = None
    probability: float | None = None
    threshold: float | None = None
    input_name: str | None = None
    input_format: str | None = None
    processing_ms: float | None = None
    execution_mode: str | None = None
    created_at: str
    finished_at: str | None = None


class PredictionPage(BaseModel):
    items: list[PredictionSummary]
    meta: PageMeta


class PredictionOut(PredictionSummary):
    dataset_version: str | None = None
    input_sha256_prefix: str | None = None
    input_bytes: int | None = None
    started_at: str | None = None
    expires_at: str | None = None
    error: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactOut] = Field(default_factory=list)
    disclaimer: str


class PredictionCreated(BaseModel):
    id: str
    status: PredictionStatus
    poll_url: str
