"""Q3 hybrid classical-quantum model and its classical twin (same training procedure, quantum layer replaced).

CT patch -> frozen ImageNet EfficientNet-B0 (classical CNN) -> 1280-d features -> standardise ->
trainable linear projection to 8 values -> tanh -> angles theta = pi/2 (1 + tanh) in [0, pi] ->
8-qubit variational circuit (RY data re-uploading, 3 layers of RY/RZ + CNOT ring; SIMULATED state vector) ->
<Z_q> for every qubit -> trainable linear readout -> logit.
End-to-end training: gradients flow through the quantum layer by the adjoint method (to the circuit weights and
to the encoding angles), then through tanh into the projection. Class-balanced BCE + L2 on the projection,
input dropout 0.1, Adam, early stopping on validation loss.
ClassicalTwin: identical except the circuit is replaced by a dense 8 -> 8 tanh layer (72 parameters vs 48 circuit
parameters), so any difference is attributable to the quantum layer, not to the rest of the pipeline.
"""
from __future__ import annotations

import time

import numpy as np

from src.quantum.models import Adam, _bce
from src.quantum.simulator import vqc_circuit

N_Q = 8


class _Base:
    def __init__(self, d_in, seed=42):
        rng = np.random.default_rng(seed)
        self.seed = seed
        self.W = rng.normal(0, 1 / np.sqrt(d_in), (d_in, N_Q))
        self.bW = np.zeros(N_Q)
        self.v = rng.normal(0, 0.3, N_Q)
        self.c = 0.0
        self.mu = self.sd = None

    def _pre(self, X, train=False, rng=None):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        if train and rng is not None:
            Z = Z * (rng.random(Z.shape) >= 0.1) / 0.9
        return Z

    def logits(self, X):
        Z = self._pre(X)
        h = np.tanh(Z @ self.W + self.bW)
        return self._layer(h) @ self.v + self.c

    def fit(self, X, y, Xv, yv, epochs=60, batch=64, lr=0.01, l2=1e-3, patience=8, log=print, name="hybrid"):
        X, Xv = np.asarray(X, float), np.asarray(Xv, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        rng = np.random.default_rng(self.seed)
        cw = {1: 0.5 / y.mean(), 0: 0.5 / (1 - y.mean())}
        params = self.params()
        opt = Adam([p.shape if hasattr(p, "shape") else () for p in params], lr)
        best, best_state, wait, hist, t0 = np.inf, None, 0, [], time.time()
        for ep in range(epochs):
            order = rng.permutation(len(y))
            for s in range(0, len(y), batch):
                i = order[s:s + batch]
                Z = self._pre(X[i], True, rng)
                pre = Z @ self.W + self.bW
                h = np.tanh(pre)
                q = self._layer(h)
                z = q @ self.v + self.c
                wt = np.where(y[i] == 1, cw[1], cw[0])
                dz = wt * (1 / (1 + np.exp(-z)) - y[i]) / wt.sum()
                gv, gc = q.T @ dz, float(dz.sum())
                dq = dz[:, None] * self.v[None, :]
                g_layer, dh = self._layer_grad(h, dq)
                dpre = dh * (1 - h ** 2)
                gW = Z.T @ dpre + l2 * self.W
                gbW = dpre.sum(0)
                new = opt.step(self.params(), [gW, gbW, gv, gc] + g_layer)
                self.set_params(new)
            vl = _bce(self.logits(Xv), yv, np.where(yv == 1, cw[1], cw[0]))
            trl = _bce(self.logits(X), y, np.where(y == 1, cw[1], cw[0]))
            hist.append({"epoch": ep + 1, "train_loss": trl, "val_loss": vl, "seconds": round(time.time() - t0, 1)})
            log(f"[{name}] epoch {ep + 1}: train {trl:.4f} val {vl:.4f}")
            if vl < best - 1e-4:
                best, best_state, wait = vl, [np.copy(p) for p in self.params()], 0
            else:
                wait += 1
                if wait >= patience:
                    break
        self.set_params(best_state)
        return {"history": hist, "best_epoch": int(np.argmin([h["val_loss"] for h in hist]) + 1),
                "train_seconds": round(time.time() - t0, 1), "epochs_max": epochs, "batch": batch, "lr": lr,
                "l2": l2, "patience": patience}


class HybridQuantum(_Base):
    def __init__(self, d_in, n_layers=3, seed=42):
        super().__init__(d_in, seed)
        self.circ = vqc_circuit(N_Q, n_layers, reupload=True)
        self.n_layers = n_layers
        self.w = np.random.default_rng(seed + 1).uniform(-0.1, 0.1, self.circ.n_weights) * np.pi

    def _angles(self, h):
        return 0.5 * np.pi * (1 + h)

    def _layer(self, h):
        return self.circ.forward(self._angles(h), self.w)

    def _layer_grad(self, h, dq):
        gw, gx = self.circ.adjoint_grad(self._angles(h), self.w, dq)
        return [gw], gx * 0.5 * np.pi

    def params(self):
        return [self.W, self.bW, self.v, self.c, self.w]

    def set_params(self, p):
        self.W, self.bW, self.v, self.c, self.w = p[0], p[1], p[2], float(p[3]), p[4]

    def to_arrays(self):
        return {"W": self.W, "bW": self.bW, "v": self.v, "c": np.array(self.c), "w": self.w, "mu": self.mu, "sd": self.sd,
                "n_layers": np.array(self.n_layers)}

    @classmethod
    def from_arrays(cls, a):
        m = cls(a["W"].shape[0], int(a["n_layers"]))
        m.W, m.bW, m.v, m.c, m.w, m.mu, m.sd = a["W"], a["bW"], a["v"], float(a["c"]), a["w"], a["mu"], a["sd"]
        return m


class ClassicalTwin(_Base):
    def __init__(self, d_in, seed=42):
        super().__init__(d_in, seed)
        rng = np.random.default_rng(seed + 1)
        self.U = rng.normal(0, 1 / np.sqrt(N_Q), (N_Q, N_Q))
        self.bU = np.zeros(N_Q)

    def _layer(self, h):
        return np.tanh(h @ self.U + self.bU)

    def _layer_grad(self, h, dq):
        q = np.tanh(h @ self.U + self.bU)
        dpre = dq * (1 - q ** 2)
        return [h.T @ dpre, dpre.sum(0)], dpre @ self.U.T

    def params(self):
        return [self.W, self.bW, self.v, self.c, self.U, self.bU]

    def set_params(self, p):
        self.W, self.bW, self.v, self.c, self.U, self.bU = p[0], p[1], p[2], float(p[3]), p[4], p[5]

    def to_arrays(self):
        return {"W": self.W, "bW": self.bW, "v": self.v, "c": np.array(self.c), "U": self.U, "bU": self.bU,
                "mu": self.mu, "sd": self.sd}

    @classmethod
    def from_arrays(cls, a):
        m = cls(a["W"].shape[0])
        m.W, m.bW, m.v, m.c, m.U, m.bU, m.mu, m.sd = (a["W"], a["bW"], a["v"], float(a["c"]), a["U"], a["bU"], a["mu"],
                                                     a["sd"])
        return m
