"""Quantum-kernel SVM: fidelity kernel K(x, x') = |<phi(x)|phi(x')>|^2 of a diagonal Z feature map (ZZ family or
product map), or the projected (Bloch-vector RBF) kernel, with scikit-learn's SVC on the precomputed Gram matrix.

The model stores only the support vectors' input angles, dual coefficients and intercept; states are recomputed at load
time, so the artifact is small and contains no pickled objects. Execution mode: custom NumPy state-vector simulator
(exact kernel values, no shot noise).
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.svm import SVC

from src.v10.quantum.feature_maps import ZFeatureMap
from src.v10.quantum.kernels import bloch_vectors, fidelity_kernel, full_report, projected_kernel

EXECUTION_MODE = "custom_numpy_simulator"
PAIRS = ("v9", "product", "none", "v9_bilinear")


class QuantumKernelSVM:
    def __init__(self, n_qubits: int = 4, reps: int = 2, pair: str = "v9", bandwidth: float = 1.0, C: float = 1.0,
                 kernel: str = "fidelity", gamma_projected: float = 1.0, entanglement: str = "linear"):
        self.n, self.reps, self.pair, self.bandwidth, self.C = n_qubits, reps, pair, bandwidth, C
        self.kernel, self.gamma_projected, self.entanglement = kernel, gamma_projected, entanglement
        self.fmap = ZFeatureMap(n_qubits, reps, pair, alpha=1.0, entanglement=entanglement, bandwidth=bandwidth)

    def _rep(self, A):
        S = self.fmap.states(np.asarray(A, float)[:, : self.n])
        return bloch_vectors(S, self.n).reshape(len(S), -1) if self.kernel == "projected" else S

    def _gram(self, RA, RB):
        if self.kernel == "projected":
            return projected_kernel(RA, RB, self.gamma_projected)
        return fidelity_kernel(RA, RB)

    def fit(self, A: np.ndarray, y: np.ndarray):
        A = np.asarray(A, float)[:, : self.n]
        t0 = time.perf_counter()
        R = self._rep(A)
        K = self._gram(R, R)
        self.svc = SVC(kernel="precomputed", C=self.C, class_weight="balanced").fit(K, np.asarray(y).astype(int))
        sv = self.svc.support_
        self.A_sv, self.R_sv = A[sv], R[sv]
        self.dual, self.b = self.svc.dual_coef_.ravel().astype(float), float(self.svc.intercept_[0])
        self.train_kernel_report = full_report(K, y) if len(A) <= 3000 else None
        self.fit_seconds = time.perf_counter() - t0
        return self

    def decision_function(self, A: np.ndarray) -> np.ndarray:
        R = self._rep(np.asarray(A, float))
        return self._gram(R, self.R_sv) @ self.dual + self.b

    def predict_proba(self, A):
        p = 1 / (1 + np.exp(-self.decision_function(A)))
        return np.stack([1 - p, p], 1)

    def describe(self) -> dict:
        return {"model": "Quantum kernel SVM", "qubits": self.n, "feature_map": self.fmap.describe(),
                "kernel": self.kernel, "C": self.C, "n_support_vectors": int(len(getattr(self, "dual", []))),
                "execution_mode": EXECUTION_MODE,
                "circuit": {"reps": self.reps, "two_qubit_gates_per_state": 2 * len(self.fmap.pairs) * self.reps,
                            "compute_uncompute_depth_note": "fidelity estimated exactly from state vectors"}}

    def to_arrays(self) -> dict:
        cfg = np.array([self.n, self.reps, PAIRS.index(self.pair) if self.pair in PAIRS else -1, self.bandwidth,
                        self.C, 1.0 if self.kernel == "projected" else 0.0, self.gamma_projected], float)
        return {"qk_cfg": cfg, "qk_entanglement": np.array(list(self.entanglement.encode()), np.uint8),
                "qk_A_sv": self.A_sv, "qk_dual": self.dual, "qk_b": np.array([self.b])}

    @classmethod
    def from_arrays(cls, a: dict) -> "QuantumKernelSVM":
        n, reps, pi, bw, C, proj, gp = a["qk_cfg"]
        m = cls(int(n), int(reps), PAIRS[int(pi)], float(bw), float(C), "projected" if proj else "fidelity", float(gp),
                bytes(a["qk_entanglement"].tolist()).decode())
        m.A_sv, m.dual, m.b = np.asarray(a["qk_A_sv"], float), np.asarray(a["qk_dual"], float), float(a["qk_b"][0])
        m.R_sv = m._rep(m.A_sv)
        return m
