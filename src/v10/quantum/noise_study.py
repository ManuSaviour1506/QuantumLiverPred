"""Noise study of a deployed V10 VQC: exact density-matrix simulation with depolarising noise (p1 = p2 / 10 per
one-qubit gate, p2 per two-qubit gate), symmetric readout error and optional finite shots.

Noise parameters are hypothetical (not calibrated to any device). Inputs are real patch features when the dataset is
available; the deployed model was fitted on all units, so absolute AUCs are optimistic — the study compares noise
levels with the noiseless model on the same inputs (logit change, correlation, AUC change), not absolute performance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.v10.engine import artifacts as A
from src.v10.engine.v10_models import pp_from_arrays
from src.v10.evaluation.metrics import roc_auc
from src.v10.quantum.noise import DensityMatrixSimulator, NoiseModel, sample_expectations
from src.v10.quantum.vqc import VQCClassifier


def run(model_dir: Path, X: np.ndarray | None, y: np.ndarray | None, p2_levels, readout_error: float = 0.0,
        shots: int = 0, n_samples: int = 300, seed: int = 0) -> dict:
    card = A.load_card(model_dir)
    A.verify(model_dir, card)
    if card.get("kind") != "v10_patch_vqc":
        raise ValueError("noise study supports V10 VQC artifacts (kind v10_patch_vqc)")
    arr = A.load_arrays(model_dir)
    vqc, pp = VQCClassifier.from_arrays(arr), pp_from_arrays(arr)
    rng = np.random.default_rng(seed)
    if X is not None and y is not None:
        idx = rng.choice(len(X), min(n_samples, len(X)), replace=False)
        A_in, yy = pp.transform(X[idx]), np.asarray(y)[idx]
        source = (f"{len(idx)} real patches (random sample). The deployed model was fitted on all units: compare "
                  "noise levels with the noiseless model, not absolute AUCs.")
    else:
        A_in, yy = rng.uniform(0, np.pi, (n_samples, vqc.n)), None
        source = "random angle inputs (circuit-level study; no dataset features available)"
    ideal = vqc._logits(vqc.circ.forward(A_in, vqc.w, vqc.masks), vqc.r)
    rows = []
    for p2 in p2_levels:
        nm = NoiseModel(p1=p2 / 10, p2=p2, readout_e0=readout_error, readout_e1=readout_error)
        sim = DensityMatrixSimulator(vqc.circ, nm)
        if shots:
            obs = sample_expectations(sim.probabilities(A_in, vqc.w), vqc.n, vqc.masks, shots, rng)
        else:
            obs = sim.expect(A_in, vqc.w, vqc.masks)
        z = vqc._logits(obs, vqc.r)
        rec = {"p2": float(p2), "p1": float(p2) / 10, "readout_error": readout_error, "shots": shots or "exact",
               "mean_abs_logit_change": float(np.mean(np.abs(z - ideal))),
               "logit_correlation_with_ideal": float(np.corrcoef(z, ideal)[0, 1]) if np.std(z) > 0 else None,
               "mean_abs_expectation": float(np.mean(np.abs(obs)))}
        if yy is not None and len(np.unique(yy)) == 2:
            rec["auc"] = roc_auc(yy, z)
            rec["auc_noiseless"] = roc_auc(yy, ideal)
        rows.append(rec)
    return {"model_id": card["model_id"], "inputs": source, "execution_mode": "custom NumPy density-matrix simulator",
            "noise_model": "depolarising (p1 = p2/10) + symmetric readout error; hypothetical, not device-calibrated",
            "circuit": vqc.describe(), "summary": rows, "hardware": False}
