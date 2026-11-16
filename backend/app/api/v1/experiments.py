"""Experiment endpoints."""
from __future__ import annotations

from starlette.responses import Response

from app.core.apikit import APIRouter, Ctx
from app.schemas.models import ExperimentCreateIn, ExperimentOut, ExperimentResultsOut

router = APIRouter("/api/v1/experiments", tags=["experiments"])


@router.get("", response=list[ExperimentOut])
def list_experiments(ctx: Ctx):
    """Recorded experiments (saved result files) and the user's own runs."""
    return ctx.state.experiment_service.list(ctx.user)


@router.post("", body=ExperimentCreateIn, response=ExperimentOut, status=202)
def create_experiment(ctx: Ctx):
    """Queue an experiment: quantum_diagnostics, noise_study or nested_evaluation."""
    b = ctx.body
    return ctx.state.experiment_service.create(ctx.user, b.name, b.kind, b.dataset_id, b.config, ctx.client_ip)


@router.get("/{experiment_id}", response=ExperimentOut)
def get_experiment(ctx: Ctx):
    """Experiment status and configuration."""
    return ctx.state.experiment_service.get(ctx.user, ctx.path["experiment_id"])


@router.get("/{experiment_id}/results", response=ExperimentResultsOut)
def experiment_results(ctx: Ctx):
    """Experiment results (metrics, per-unit tables, logs)."""
    return ctx.state.experiment_service.results(ctx.user, ctx.path["experiment_id"])


@router.get("/{experiment_id}/report", media_type="application/pdf")
def experiment_report(ctx: Ctx):
    """PDF experiment report."""
    pdf = ctx.state.experiment_service.report_pdf(ctx.user, ctx.path["experiment_id"])
    return Response(pdf, media_type="application/pdf",
                    headers={"content-disposition": f'attachment; filename="experiment_{ctx.path["experiment_id"]}.pdf"'})
