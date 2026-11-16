"""Research dashboard endpoints: quantum circuits, kernel and noise diagnostics, nested comparisons, ablations, runs."""
from __future__ import annotations

from app.core.apikit import APIRouter, Ctx
from app.services import analytics as A

router = APIRouter("/api/v1/research", tags=["research"])


@router.get("/quantum")
def quantum(ctx: Ctx):
    """Quantum and hybrid models: qubits, depth, gate counts, feature map, readout, execution mode."""
    return {"models": A.quantum_models(ctx.state),
            "execution_note": "custom NumPy simulator only; no quantum hardware"}


@router.get("/summary")
def summary(ctx: Ctx):
    """Nested-evaluation comparisons, hybrid ablations, noise study, Stage 2 and mathematical validation, run logs."""
    return A.research_summary(ctx.state)
