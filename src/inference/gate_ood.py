"""Liver-tissue gate and out-of-distribution (OOD) detector used at inference.

Liver gate  logistic regression on frozen ImageNet EfficientNet-B0 features of 96x96 patches: liver tissue (tumour or
            tumour-free, from the expert masks) vs body tissue without liver. Only patches the gate accepts are passed
            to the tumour models, which were trained on liver tissue only.
OOD         V7 detector (standardise -> PCA(32) Mahalanobis distance + reconstruction residual) on the intermediate-stage
            EfficientNet-B0 features of the whole slice (resized to 224). Fitted on the training-split slices;
            thresholds = 99th percentile of leave-one-scan-group-out scores (src/quality_control/ood.py).
"""
from __future__ import annotations

import numpy as np

from src.common.io import load_arrays, read_json
from src.common.paths import ARTIFACTS
from src.quality_control.ood import OODDetector

GATE_DIR = ARTIFACTS / "classical" / "liver_gate"
OOD_DIR = ARTIFACTS / "ood"


class Gate:
    def __init__(self, a, meta):
        self.mu, self.sd, self.coef, self.b = a["mu"], a["sd"], a["coef"], float(a["intercept"][0])
        self.threshold = float(meta["threshold"])
        self.meta = meta

    def prob(self, gap):
        z = ((np.asarray(gap, float) - self.mu) / self.sd) @ self.coef + self.b
        return 1 / (1 + np.exp(-z))


def load_gate():
    if not (GATE_DIR / "model.json").exists():
        return None
    meta = read_json(GATE_DIR / "model.json")
    return Gate(load_arrays(GATE_DIR / "model.npz"), meta)


def load_ood():
    if not (OOD_DIR / "ood.npz").exists():
        return None
    det = OODDetector.from_arrays(load_arrays(OOD_DIR / "ood.npz"))
    det.meta = read_json(OOD_DIR / "ood.json")
    return det
