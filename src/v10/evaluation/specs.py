"""Model specifications for the V10 nested evaluation (grids are fixed here, before any outer result exists)."""
from __future__ import annotations

import numpy as np

from src.v10.classical import models as CM
from src.v10.evaluation.nested import ModelSpec, SklearnAdapter
from src.v10.hybrid.models import HybridClassifier, LateFusionClassifier
from src.v10.quantum.qksvm import QuantumKernelSVM
from src.v10.quantum.vqc import VQCClassifier

N_QUBITS = 4
EPOCHS = 30
ANGLES = {"scaler": "standard", "pca": N_QUBITS, "angle_max": float(np.pi)}
PCA_STD = {"scaler": "standard", "pca": N_QUBITS}
HYBRID_IN = {"scaler": "standard", "pca": 16}


def classical_specs(kinds=None) -> list[ModelSpec]:
    out = []
    for k in kinds or CM.available():
        out.append(ModelSpec(k, CM.NAMES[k], "classical", CM.GRIDS[k],
                             (lambda kind: lambda cfg, seed: SklearnAdapter(CM.make(kind, cfg, seed)))(k),
                             {"scaler": "standard"}, k in CM.STOCHASTIC))
    return out


def control_specs() -> list[ModelSpec]:
    """Classical models on exactly the inputs the quantum models see (PCA to N_QUBITS components)."""
    return [
        ModelSpec("logreg_pca4", f"Logistic Regression on PCA({N_QUBITS}) [VQC control]", "control",
                  CM.GRIDS["logreg"], lambda cfg, seed: SklearnAdapter(CM.make("logreg", cfg, seed)), PCA_STD),
        ModelSpec("svm_rbf_pca4", f"RBF-SVM on PCA({N_QUBITS}) [quantum-kernel control]", "control",
                  [{"C": c, "gamma": g} for c in (0.3, 3.0) for g in ("scale", 0.1, 1.0)],
                  lambda cfg, seed: SklearnAdapter(CM.make("svm_rbf", cfg, seed)), PCA_STD),
    ]


def quantum_specs() -> list[ModelSpec]:
    vqc_grid = [{"layers": 2, "readout": "linear_z"}, {"layers": 2, "readout": "z0"},
                {"layers": 1, "readout": "linear_z"}]
    qk_grid = [{"pair": p, "bandwidth": b, "C": c} for p in ("v9", "product") for b in (0.25, 0.5, 1.0)
               for c in (0.3, 3.0)]
    return [
        ModelSpec("vqc", f"VQC ({N_QUBITS} qubits, brickwork)", "quantum", vqc_grid,
                  lambda cfg, seed: VQCClassifier(N_QUBITS, cfg["layers"], "brick", cfg["readout"], epochs=EPOCHS,
                                                  seed=seed), ANGLES, True),
        ModelSpec("qksvm", f"Quantum kernel SVM ({N_QUBITS} qubits, fidelity kernel)", "quantum", qk_grid,
                  lambda cfg, seed: QuantumKernelSVM(N_QUBITS, 2, cfg["pair"], cfg["bandwidth"], cfg["C"]), ANGLES),
    ]


def hybrid_specs() -> list[ModelSpec]:
    grid = [{"layers": 2, "lr": 0.03}, {"layers": 1, "lr": 0.03}]

    def hyb(mode):
        return lambda cfg, seed: HybridClassifier(mode, N_QUBITS, cfg["layers"], "brick", epochs=EPOCHS,
                                                  lr=cfg["lr"], seed=seed)
    return [
        ModelSpec("hybrid_no_fusion", "Hybrid (encoder -> VQC -> head)", "hybrid", grid, hyb("no_fusion"),
                  HYBRID_IN, True),
        ModelSpec("hybrid_twin", "Classical twin (encoder -> matched classical layer -> head)", "control", grid,
                  hyb("classical_only"), HYBRID_IN, True),
        ModelSpec("hybrid_feature_fusion", "Hybrid feature fusion (VQC + classical branch)", "hybrid", grid,
                  hyb("feature_fusion"), HYBRID_IN, True, fixed_from="hybrid_no_fusion"),
        ModelSpec("hybrid_random_quantum", "Ablation: hybrid with untrained random circuit", "hybrid", grid,
                  hyb("random_quantum"), HYBRID_IN, True, fixed_from="hybrid_no_fusion"),
        ModelSpec("hybrid_quantum_only", "Ablation: VQC + head without classical encoder", "hybrid", grid,
                  hyb("quantum_only"), ANGLES, True, fixed_from="hybrid_no_fusion"),
        ModelSpec("late_fusion", "Late fusion (LR + VQC)", "hybrid", [{"C": 1.0}],
                  lambda cfg, seed: LateFusionClassifier(N_QUBITS, 2, "brick", "linear_z", EPOCHS, C=cfg["C"],
                                                         seed=seed), {"scaler": "standard"}, True),
    ]


CONFIRMATORY = [("vqc", "logreg_pca4"), ("qksvm", "svm_rbf_pca4"), ("hybrid_no_fusion", "hybrid_twin")]


def all_specs() -> list[ModelSpec]:
    return classical_specs() + control_specs() + quantum_specs() + hybrid_specs()
