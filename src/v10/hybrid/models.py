"""Hybrid quantum-classical classifiers and their ablations, trained end to end with Adam.

All variants share a trainable classical encoder (the "classical feature extractor" head on top of fixed radiomics or
ImageNet features): h = W x + c, angles a = (pi / 2) (tanh(h) + 1) in (0, pi), one angle per qubit.

  no_fusion        encoder -> VQC (V10 ansatz) -> <Z_q> for every qubit -> linear head.  The quantum circuit is the
                   only path from input to output (primary hybrid).
  feature_fusion   [<Z_q> from the VQC, tanh(h_c) from a parallel classical branch] -> linear head.
  classical_only   ablation / classical twin: the VQC is replaced by a classical layer g = tanh(U tanh(h) + e) whose
                   parameter count matches the VQC's trainable circuit parameters.
  random_quantum   ablation: as no_fusion but the circuit weights stay at their random initial values (only the encoder
                   and head train) — tests whether trained quantum weights matter.
  quantum_only     ablation: no trainable encoder (angles come straight from the fold-fitted preprocessor), VQC + head.

Late fusion (averaging the logits of separately trained classical and quantum models with a weight learned on inner
out-of-fold scores) is implemented in `src/v10/evaluation/nested.py` because it combines fitted models.

Gradients: the encoder receives d loss / d angle from the simulator's adjoint method (exact), so the whole hybrid is
trained jointly. Execution mode: custom NumPy state-vector simulator.
"""
from __future__ import annotations

import time

import numpy as np

from src.v10.quantum.circuits import build_v10_vqc, gate_counts
from src.v10.quantum.optim import Adam, balanced_weights, cosine_lr, sigmoid

MODES = ("no_fusion", "feature_fusion", "classical_only", "random_quantum", "quantum_only")
HALF_PI = np.pi / 2


class HybridClassifier:
    def __init__(self, mode: str = "no_fusion", n_qubits: int = 4, layers: int = 2, entangler: str = "brick",
                 epochs: int = 40, batch: int = 64, lr: float = 0.03, l2: float = 1e-3, seed: int = 0,
                 init_scale: float = 0.1):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self.n, self.layers, self.entangler = mode, n_qubits, layers, entangler
        self.epochs, self.batch, self.lr, self.l2, self.seed, self.init_scale = epochs, batch, lr, l2, seed, init_scale
        self.circ = build_v10_vqc(n_qubits, layers, entangler)
        self.masks = [(q,) for q in range(n_qubits)]
        self.m_twin = max(1, int(round(self.circ.n_weights / (n_qubits + 1))))
        self.history: list[dict] = []

    # ------------------------------------------------------------------ parameter layout
    def _shapes(self, d: int) -> dict:
        n, s = self.n, {}
        if self.mode != "quantum_only":
            s["W"], s["c"] = (n, d), (n,)
        if self.mode == "feature_fusion":
            s["Wc"], s["cc"] = (n, d), (n,)
        if self.mode == "classical_only":
            s["U"], s["e"] = (self.m_twin, n), (self.m_twin,)
        else:
            s["w"] = (self.circ.n_weights,)
        head_in = {"feature_fusion": 2 * n, "classical_only": self.m_twin}.get(self.mode, n)
        s["v"], s["b"] = (head_in,), (1,)
        return s

    def _unpack(self, theta):
        out, i = {}, 0
        for k, shp in self.shapes.items():
            size = int(np.prod(shp))
            out[k] = theta[i:i + size].reshape(shp)
            i += size
        return out

    def _init(self, d, rng):
        parts = []
        for k, shp in self.shapes.items():
            if k in ("W", "Wc"):
                parts.append(rng.normal(0, 1 / np.sqrt(d), shp).ravel())
            elif k == "U":
                parts.append(rng.normal(0, 1 / np.sqrt(self.n), shp).ravel())
            elif k == "w":
                parts.append(rng.normal(0, self.init_scale, shp).ravel())
            elif k == "v":
                parts.append(np.full(shp, 0.5).ravel())
            else:
                parts.append(np.zeros(shp).ravel())
        return np.concatenate(parts)

    # ------------------------------------------------------------------ forward / backward
    def _forward(self, X, P, cache=False):
        c = {}
        if self.mode == "quantum_only":
            A = X[:, : self.n]
        else:
            t = np.tanh(X @ P["W"].T + P["c"])
            A = HALF_PI * (t + 1)
            c["t"] = t
        c["A"] = A
        if self.mode == "classical_only":
            g = np.tanh(c["t"] @ P["U"].T + P["e"])
            feats = g
            c["g"] = g
        else:
            obs = self.circ.forward(A, P["w"], self.masks)
            feats = obs
            c["obs"] = obs
            if self.mode == "feature_fusion":
                tc = np.tanh(X @ P["Wc"].T + P["cc"])
                c["tc"] = tc
                feats = np.concatenate([obs, tc], 1)
        c["feats"] = feats
        z = feats @ P["v"] + P["b"][0]
        return (z, c) if cache else z

    def _grads(self, X, P, c, dz):
        G = {k: np.zeros_like(v) for k, v in P.items()}
        G["v"] = c["feats"].T @ dz
        G["b"] = np.array([dz.sum()])
        dfeat = dz[:, None] * P["v"][None, :]
        n = self.n
        if self.mode == "classical_only":
            dg = dfeat * (1 - c["g"] ** 2)
            G["U"] = dg.T @ c["t"]
            G["e"] = dg.sum(0)
            dt = dg @ P["U"]
        else:
            dobs = dfeat[:, :n]
            gw, gA = self.circ.adjoint_grad(c["A"], P["w"], dobs, self.masks)
            if self.mode != "random_quantum":
                G["w"] = gw
            dt = gA[:, :n] * HALF_PI if self.mode != "quantum_only" else None
            if self.mode == "feature_fusion":
                dtc = dfeat[:, n:] * (1 - c["tc"] ** 2)
                G["Wc"] = dtc.T @ X
                G["cc"] = dtc.sum(0)
        if dt is not None and "W" in P:
            dh = dt * (1 - c["t"] ** 2)
            G["W"] = dh.T @ X
            G["c"] = dh.sum(0)
        for k in ("W", "Wc", "U"):
            if k in P:
                G[k] = G[k] + self.l2 * P[k]
        return np.concatenate([G[k].ravel() for k in self.shapes])

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = np.asarray(X, float), np.asarray(y).astype(int)
        rng = np.random.default_rng(self.seed)
        self.d = X.shape[1]
        self.shapes = self._shapes(self.d)
        theta = self._init(self.d, rng)
        opt = Adam(theta.size, self.lr)
        sw = balanced_weights(y)
        t0 = time.perf_counter()
        for ep in range(self.epochs):
            lr = cosine_lr(self.lr, ep, self.epochs)
            perm = rng.permutation(len(X))
            tot = 0.0
            for s in range(0, len(X), self.batch):
                idx = perm[s:s + self.batch]
                P = self._unpack(theta)
                z, c = self._forward(X[idx], P, cache=True)
                p = sigmoid(z)
                wb, yb = sw[idx], y[idx]
                dz = wb * (p - yb) / wb.sum()
                theta = opt.step(theta, self._grads(X[idx], P, c, dz), lr)
                tot += float(-np.sum(wb * (yb * np.log(p + 1e-12) + (1 - yb) * np.log(1 - p + 1e-12))))
            self.history.append({"epoch": ep + 1, "train_loss": tot / sw.sum()})
        self.fit_seconds = time.perf_counter() - t0
        self.theta = theta
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        P = self._unpack(self.theta)
        X = np.asarray(X, float)
        return np.concatenate([self._forward(X[s:s + 512], P) for s in range(0, len(X), 512)]) if len(X) else np.zeros(0)

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.stack([1 - p, p], 1)

    def n_params(self) -> int:
        return int(sum(np.prod(s) for s in self.shapes.values()))

    def describe(self) -> dict:
        out = {"model": "Hybrid", "mode": self.mode, "qubits": self.n, "layers": self.layers,
               "entangler": self.entangler, "total_parameters": self.n_params(), "epochs": self.epochs,
               "lr": self.lr, "seed": self.seed}
        if self.mode != "classical_only":
            out.update(trainable_circuit_parameters=0 if self.mode == "random_quantum" else int(self.circ.n_weights),
                       gate_counts=gate_counts(self.circ), execution_mode="custom_numpy_simulator")
        else:
            out.update(classical_replacement_units=self.m_twin, execution_mode="classical")
        return out

    def to_arrays(self) -> dict:
        cfg = np.array([MODES.index(self.mode), self.n, self.layers, self.d, self.m_twin], float)
        return {"hy_cfg": cfg, "hy_entangler": np.array(list(self.entangler.encode()), np.uint8), "hy_theta": self.theta}

    @classmethod
    def from_arrays(cls, a: dict) -> "HybridClassifier":
        mode, n, layers, d, _m = (int(v) for v in a["hy_cfg"])
        m = cls(MODES[mode], n, layers, bytes(a["hy_entangler"].tolist()).decode())
        m.d = d
        m.shapes = m._shapes(d)
        m.theta = np.asarray(a["hy_theta"], float)
        m.epochs = m.lr = m.seed = None   # training settings are not stored in the arrays (see the model card)
        return m


class LateFusionClassifier:
    """Late fusion: a classical logistic regression on all standardised features and a VQC on fold-fitted PCA angles
    are trained separately; their logits are standardised on the training data and averaged with equal weight
    (a fixed rule, so nothing is tuned on held-out data)."""

    def __init__(self, n_qubits: int = 4, layers: int = 2, entangler: str = "brick", readout: str = "linear_z",
                 epochs: int = 30, lr: float = 0.05, C: float = 1.0, seed: int = 0):
        from src.v10.quantum.vqc import VQCClassifier
        self.vqc = VQCClassifier(n_qubits, layers, entangler, readout, epochs=epochs, lr=lr, seed=seed)
        self.C, self.n = C, n_qubits

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from src.v10.evaluation.grouped_cv import FoldPreprocessor
        X = np.asarray(X, float)
        self.lr_model = LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000).fit(X, y)
        self.pp = FoldPreprocessor("standard", pca=self.n, angle_max=np.pi).fit(X)
        self.vqc.fit(self.pp.transform(X), y)
        zc, zq = self.lr_model.decision_function(X), self.vqc.decision_function(self.pp.transform(X))
        self.sc, self.sq = float(np.std(zc) + 1e-9), float(np.std(zq) + 1e-9)
        return self

    def decision_function(self, X):
        X = np.asarray(X, float)
        return 0.5 * self.lr_model.decision_function(X) / self.sc + \
            0.5 * self.vqc.decision_function(self.pp.transform(X)) / self.sq

    def n_params(self):
        return int(self.lr_model.coef_.size + 1 + self.vqc.n_params())

    def describe(self):
        return {"model": "Late fusion (LR + VQC, equal-weight standardised logits)", "vqc": self.vqc.describe()}
