"""Variational quantum classifier (V10 ansatz) trained with exact adjoint gradients on the custom NumPy simulator.

Circuit: per layer RY(x_q) data re-upload, trainable RY+RZ on every qubit, an entangling block (brick / ring / linear),
then a final trainable RY layer (so no parameter is dead; see the V10 mathematical audit). Inputs are angles produced by
a fold-fitted preprocessor (standardise -> PCA(n_qubits) -> bounded map to [0, pi]).

Readouts (logit = classical affine map of measured expectation values):
  z0        a * <Z_0> + b
  mean_z    a * mean_q <Z_q> + b
  linear_z  sum_q v_q <Z_q> + b
  parity    a * <Z_0 Z_1 ... Z_{n-1}> + b
Loss: class-balanced binary cross-entropy; Adam with cosine learning-rate decay; fixed number of epochs (chosen in the
inner loop, so no extra validation split is needed for early stopping). Execution mode: custom NumPy state-vector
simulator (exact expectation values, no shot noise) — never hardware.
"""
from __future__ import annotations

import time

import numpy as np

from src.v10.quantum.circuits import build_v10_vqc, gate_counts
from src.v10.quantum.optim import Adam, balanced_weights, cosine_lr, sigmoid

EXECUTION_MODE = "custom_numpy_simulator"


class VQCClassifier:
    READOUTS = ("z0", "mean_z", "linear_z", "parity")

    def __init__(self, n_qubits: int = 4, layers: int = 2, entangler: str = "brick", readout: str = "linear_z",
                 epochs: int = 40, batch: int = 64, lr: float = 0.05, seed: int = 0, init_scale: float = 0.1,
                 trainable_scale: bool = False):
        if readout not in self.READOUTS:
            raise ValueError(f"readout must be one of {self.READOUTS}")
        self.n, self.layers, self.entangler, self.readout = n_qubits, layers, entangler, readout
        self.epochs, self.batch, self.lr, self.seed, self.init_scale = epochs, batch, lr, seed, init_scale
        self.trainable_scale = trainable_scale
        self.circ = build_v10_vqc(n_qubits, layers, entangler, trainable_scale)
        self.w = None
        self.r = None
        self.history: list[dict] = []

    # ------------------------------------------------------------------ structure
    @property
    def masks(self):
        if self.readout == "z0":
            return [(0,)]
        if self.readout == "parity":
            return [tuple(range(self.n))]
        return [(q,) for q in range(self.n)]

    def _n_readout(self) -> int:
        return self.n + 1 if self.readout == "linear_z" else 2

    def _logits(self, obs, r):
        if self.readout == "linear_z":
            return obs @ r[:-1] + r[-1]
        if self.readout == "mean_z":
            return r[0] * obs.mean(1) + r[1]
        return r[0] * obs[:, 0] + r[1]

    def _readout_grads(self, obs, dz, r):
        if self.readout == "linear_z":
            return np.concatenate([obs.T @ dz, [dz.sum()]]), dz[:, None] * r[None, :-1]
        if self.readout == "mean_z":
            return np.array([np.sum(dz * obs.mean(1)), dz.sum()]), np.repeat(dz[:, None] * r[0] / self.n, self.n, 1)
        return np.array([np.sum(dz * obs[:, 0]), dz.sum()]), dz[:, None] * r[0]

    # ------------------------------------------------------------------ training
    def fit(self, A: np.ndarray, y: np.ndarray):
        A, y = np.asarray(A, float)[:, : self.n], np.asarray(y).astype(int)
        rng = np.random.default_rng(self.seed)
        nw = self.circ.n_weights
        w = rng.normal(0.0, self.init_scale, nw)
        if self.trainable_scale:
            w[: self.n] = 1.0
        r = np.zeros(self._n_readout())
        r[: (self.n if self.readout == "linear_z" else 1)] = 1.0
        theta = np.concatenate([w, r])
        opt = Adam(theta.size, self.lr)
        sw = balanced_weights(y)
        t0 = time.perf_counter()
        for ep in range(self.epochs):
            lr = cosine_lr(self.lr, ep, self.epochs)
            perm = rng.permutation(len(A))
            loss_sum = 0.0
            for s in range(0, len(A), self.batch):
                idx = perm[s:s + self.batch]
                Ab, yb, wb = A[idx], y[idx], sw[idx]
                w, r = theta[:nw], theta[nw:]
                obs = self.circ.forward(Ab, w, self.masks)
                z = self._logits(obs, r)
                p = sigmoid(z)
                dz = wb * (p - yb) / wb.sum()
                gr, dobs = self._readout_grads(obs, dz, r)
                gw, _ = self.circ.adjoint_grad(Ab, w, dobs, self.masks)
                theta = opt.step(theta, np.concatenate([gw, gr]), lr)
                loss_sum += float(-np.sum(wb * (yb * np.log(p + 1e-12) + (1 - yb) * np.log(1 - p + 1e-12))))
            self.history.append({"epoch": ep + 1, "train_loss": loss_sum / sw.sum()})
        self.fit_seconds = time.perf_counter() - t0
        self.w, self.r = theta[:nw].copy(), theta[nw:].copy()
        return self

    def decision_function(self, A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, float)[:, : self.n]
        out = []
        for s in range(0, len(A), 512):
            out.append(self._logits(self.circ.forward(A[s:s + 512], self.w, self.masks), self.r))
        return np.concatenate(out) if out else np.zeros(0)

    def predict_proba(self, A):
        p = sigmoid(self.decision_function(A))
        return np.stack([1 - p, p], 1)

    def feature_sensitivity(self, A: np.ndarray) -> np.ndarray:
        """d logit / d input angle for each row (exact, adjoint method)."""
        A = np.asarray(A, float)[:, : self.n]
        obs = self.circ.forward(A, self.w, self.masks)
        _, dobs = self._readout_grads(obs, np.ones(len(A)), self.r)
        _, gx = self.circ.adjoint_grad(A, self.w, dobs, self.masks)
        return gx

    # ------------------------------------------------------------------ description and persistence
    def describe(self) -> dict:
        gc = gate_counts(self.circ)
        return {"model": "VQC", "qubits": self.n, "layers": self.layers, "entangler": self.entangler,
                "readout": self.readout, "observables": [list(m) for m in self.masks],
                "trainable_circuit_parameters": int(self.circ.n_weights), "readout_parameters": self._n_readout(),
                "gate_counts": gc, "execution_mode": EXECUTION_MODE, "epochs": self.epochs, "lr": self.lr,
                "batch": self.batch, "seed": self.seed}

    def n_params(self) -> int:
        return int(self.circ.n_weights + self._n_readout())

    def to_arrays(self) -> dict:
        cfg = np.array([self.n, self.layers, self.READOUTS.index(self.readout), int(self.trainable_scale)], float)
        return {"vqc_cfg": cfg, "vqc_entangler": np.array(list(self.entangler.encode()), np.uint8),
                "vqc_w": self.w, "vqc_r": self.r}

    @classmethod
    def from_arrays(cls, a: dict) -> "VQCClassifier":
        n, layers, ro, ts = (int(v) for v in a["vqc_cfg"])
        m = cls(n, layers, bytes(a["vqc_entangler"].tolist()).decode(), cls.READOUTS[ro], trainable_scale=bool(ts))
        m.w, m.r = np.asarray(a["vqc_w"], float), np.asarray(a["vqc_r"], float)
        m.epochs = m.lr = m.batch = m.seed = None   # training settings are not stored in the arrays (see the model card)
        return m
