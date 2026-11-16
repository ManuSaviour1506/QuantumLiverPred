"""Model registry, experiment, analytics and system schemas."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelOut(BaseModel):
    id: str
    name: str
    family: str
    architecture: str
    task: str
    description: str = ""
    version: str
    source: str
    status: str
    deployment_status: str
    servable: bool
    reason: str | None = None
    training_status: str | None = None
    training_date: str | None = None
    dataset_version: str | None = None
    execution_mode: str | None = None
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    size_bytes: int | None = None
    role: str | None = None
    input_contract: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    availability: str | None = None
    compare_family: str | None = None
    dependencies: list[dict[str, Any]] = Field(default_factory=list)


class ModelStatusOut(BaseModel):
    id: str
    status: str
    deployment_status: str
    servable: bool
    reason: str | None = None
    engine_loaded: bool
    checks: dict[str, Any] = Field(default_factory=dict)


class ModelMetricsOut(BaseModel):
    id: str
    version: str
    evaluations: list[dict[str, Any]]
    notes: list[str] = Field(default_factory=list)


ExperimentKind = Literal["nested_evaluation", "noise_study", "quantum_diagnostics"]


class ExperimentCreateIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    kind: ExperimentKind
    dataset_id: str | None = Field(None, max_length=64)
    config: dict[str, Any] = Field(default_factory=dict)


class ExperimentOut(BaseModel):
    id: str
    name: str
    kind: str
    source: str
    status: str
    owner_id: str | None = None
    dataset_version_id: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    has_results: bool = False


class ExperimentResultsOut(BaseModel):
    id: str
    status: str
    results: dict[str, Any] = Field(default_factory=dict)


class OverviewOut(BaseModel):
    users_total: int | None = None
    predictions_total: int
    predictions_mine: int
    predictions_by_status: dict[str, int]
    models_registered: int
    models_servable: int
    models_by_family: dict[str, int]
    datasets_total: int
    dataset_files_total: int
    experiments_total: int
    jobs_pending: int
    recent_activity: list[dict[str, Any]]
    engine: dict[str, Any]
    generated_at: str


class SystemInfoOut(BaseModel):
    app: str
    version: str
    environment: str
    disclaimer: str
    database: dict[str, Any]
    storage: dict[str, Any]
    worker_mode: str
    engine: dict[str, Any]
    optional_packages: dict[str, Any]
    execution_modes: list[dict[str, Any]]
    registration_open: bool
    password_reset_available: bool
    limits: dict[str, Any]
