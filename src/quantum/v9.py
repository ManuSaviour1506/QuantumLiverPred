"""V9 quantum and hybrid models (exact NumPy state-vector SIMULATION; PennyLane / Qiskit are not installable offline).

VQCv9         variational classifier with n qubits and L layers (RY data re-uploading, RY/RZ + CNOT ring), optional
              trainable feature map (per-qubit input scale and offset learned with the circuit), readout either
              <Z_0> ('z0') or a trainable linear combination of all <Z_q> ('allz'); early stopping on validation AUC.
HybridV9      CNN features -> trainable projection to n values -> tanh -> angles -> n-qubit circuit -> <Z_q> ->
              linear readout; optional 'fusion' adds the classical projection (tanh) to the readout input.
TwinV9        the matched classical twin: circuit replaced by a dense n -> n tanh layer, same everything else.
noisy_expect_z  Pauli-trajectory simulation of single-qubit depolarizing noise after every gate (probability p per
              touched qubit; Monte-Carlo average over trajectories) plus finite-shot sampling of <Z_q>.
circuit_depth  gate depth after as-soon-as-possible scheduling (each gate occupies the qubits it acts on).
"""
from __future__ import annotations

import time

import numpy as np

from src.quantum.models import Adam
from src.quantum.simulator import Circuit, ZZFeatureMap, _rot, fidelity_kernel, vqc_circuit


def circuit_depth(circ: Circuit) -> dict:
    level = np.zeros(circ.n, int)
    n1 = n2 = 0
    for op in circ.ops:
        qs = [op[1]] if op[0] in ("RX", "RY", "RZ", "H") else [op[1], op[2]]
        d = level[qs].max() + 1
        level[qs] = d
        n1 += len(qs) == 1
        n2 += len(qs) == 2
    return {"depth": int(level.max()), "one_qubit_gates": int(n1), "two_qubit_gates": int(n2),
            "state_vector_bytes": int(16 * 2 ** circ.n)}


def zz_depth(n: int, reps: int) -> dict:
    per = 2 + 3 * (n - 1)   # H, RZ, then CNOT-RZ-CNOT on a linear chain (sequential worst case)
    return {"depth": int(reps * per), "one_qubit_gates": int(reps * (2 * n + (n - 1))), "two_qubit_gates": int(reps * 2 * (n - 1)),
            "state_vector_bytes": int(16 * 2 ** n)}


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def _bce(z, y, w):
    return float(np.sum(w * (np.logaddexp(0, z) - y * z)) / np.sum(w))


class VQCv9:
    def __init__(self, n_qubits=8, n_layers=3, trainable_map=False, readout="z0", seed=42):
        self.circ = vqc_circuit(n_qubits, n_layers, reupload=True)
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(-0.1, 0.1, self.circ.n_weights) * np.pi
        self.a, self.o = np.ones(n_qubits), np.zeros(n_qubits)   # feature-map scale / offset (trainable if enabled)
        self.v = np.zeros(n_qubits)
        self.v[0] = 2.0
        self.b = 0.0
        self.n_qubits, self.n_layers, self.trainable_map, self.readout, self.seed = n_qubits, n_layers, trainable_map, readout, seed

    def _angles(self, A):
        return A * self.a + self.o

    def expect(self, A):
        return self.circ.forward(self._angles(A), self.w)

    def logits_from_expect(self, ex):
        return ex @ self.v + self.b if self.readout == "allz" else ex[:, 0] * self.v[0] + self.b

    def logits(self, A):
        return self.logits_from_expect(self.expect(A))

    def fit(self, A, y, Av, yv, epochs=40, batch=64, lr=0.05, patience=6, log=print, name="VQC"):
        rng = np.random.default_rng(self.seed)
        cw = {1: 0.5 / y.mean(), 0: 0.5 / (1 - y.mean())}
        opt = Adam([self.w.shape, self.a.shape, self.o.shape, self.v.shape, ()], lr)
        best, best_state, wait, hist, t0 = -np.inf, None, 0, [], time.time()
        for ep in range(epochs):
            order = rng.permutation(len(y))
            for s in range(0, len(y), batch):
                i = order[s:s + batch]
                ang = self._angles(A[i])
                ex = self.circ.forward(ang, self.w)
                z = self.logits_from_expect(ex)
                wt = np.where(y[i] == 1, cw[1], cw[0])
                dz = wt * (1 / (1 + np.exp(-z)) - y[i]) / wt.sum()
                if self.readout == "allz":
                    dL = dz[:, None] * self.v[None, :]
                    gv = ex.T @ dz
                else:
                    dL = np.zeros_like(ex)
                    dL[:, 0] = dz * self.v[0]
                    gv = np.zeros_like(self.v)
                    gv[0] = float(np.sum(dz * ex[:, 0]))
                gw, gx = self.circ.adjoint_grad(ang, self.w, dL)
                if self.trainable_map:
                    ga, go = (gx * A[i]).sum(0), gx.sum(0)
                else:
                    ga, go = np.zeros_like(self.a), np.zeros_like(self.o)
                gb = float(np.sum(dz))
                self.w, self.a, self.o, self.v, self.b = opt.step([self.w, self.a, self.o, self.v, self.b], [gw, ga, go, gv, gb])
            zv = self.logits(Av)
            auc = _auc(yv, zv)
            vl = _bce(zv, yv, np.where(yv == 1, cw[1], cw[0]))
            hist.append({"epoch": ep + 1, "val_loss": vl, "val_auc": auc, "seconds": round(time.time() - t0, 1)})
            log(f"[{name}] epoch {ep + 1}: val loss {vl:.4f} val AUC {auc:.4f}")
            if auc > best + 1e-4:
                best, best_state, wait = auc, (self.w.copy(), self.a.copy(), self.o.copy(), self.v.copy(), self.b), 0
            else:
                wait += 1
                if wait >= patience:
                    break
        self.w, self.a, self.o, self.v, self.b = best_state
        return {"history": hist, "best_epoch": int(np.argmax([h["val_auc"] for h in hist]) + 1),
                "train_seconds": round(time.time() - t0, 1), "epochs_max": epochs, "batch": batch, "lr": lr, "patience": patience,
                "early_stopping": "validation ROC-AUC", "gradient": "adjoint (exact; equals parameter-shift)"}

    def n_params(self):
        return int(self.w.size + (2 * self.n_qubits if self.trainable_map else 0) + (self.n_qubits if self.readout == "allz" else 1) + 1)

    def to_arrays(self):
        return {"w": self.w, "a": self.a, "o": self.o, "v": self.v, "b": np.array(self.b),
                "shape": np.array([self.n_qubits, self.n_layers, int(self.trainable_map), int(self.readout == "allz")])}

    @classmethod
    def from_arrays(cls, arr):
        n, L, tm, rz = (int(x) for x in arr["shape"])
        m = cls(n, L, bool(tm), "allz" if rz else "z0")
        m.w, m.a, m.o, m.v = (np.asarray(arr[k], float) for k in ("w", "a", "o", "v"))
        m.b = float(arr["b"])
        return m


class _HBase:
    def __init__(self, d_in, n, fusion=False, seed=42):
        rng = np.random.default_rng(seed)
        self.n, self.fusion, self.seed = n, fusion, seed
        self.W = rng.normal(0, 1 / np.sqrt(d_in), (d_in, n))
        self.bW = np.zeros(n)
        self.v = rng.normal(0, 0.3, 2 * n if fusion else n)
        self.c = 0.0
        self.mu = self.sd = None

    def _pre(self, X, train=False, rng=None):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        if train and rng is not None:
            Z = Z * (rng.random(Z.shape) >= 0.1) / 0.9
        return Z

    def _feat(self, h, q):
        return np.concatenate([q, h], 1) if self.fusion else q

    def logits(self, X):
        h = np.tanh(self._pre(X) @ self.W + self.bW)
        return self._feat(h, self._layer(h)) @ self.v + self.c

    def fit(self, X, y, Xv, yv, epochs=60, batch=64, lr=0.01, l2=1e-3, patience=8, log=print, name="hybrid"):
        X, Xv = np.asarray(X, float), np.asarray(Xv, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        rng = np.random.default_rng(self.seed)
        cw = {1: 0.5 / y.mean(), 0: 0.5 / (1 - y.mean())}
        opt = Adam([p.shape if hasattr(p, "shape") else () for p in self.params()], lr)
        best, best_state, wait, hist, t0 = -np.inf, None, 0, [], time.time()
        for ep in range(epochs):
            order = rng.permutation(len(y))
            for s in range(0, len(y), batch):
                i = order[s:s + batch]
                Z = self._pre(X[i], True, rng)
                h = np.tanh(Z @ self.W + self.bW)
                q = self._layer(h)
                f = self._feat(h, q)
                z = f @ self.v + self.c
                wt = np.where(y[i] == 1, cw[1], cw[0])
                dz = wt * (1 / (1 + np.exp(-z)) - y[i]) / wt.sum()
                gv, gc = f.T @ dz, float(dz.sum())
                df = dz[:, None] * self.v[None, :]
                dq = df[:, :self.n]
                g_layer, dh = self._layer_grad(h, dq)
                if self.fusion:
                    dh = dh + df[:, self.n:]
                dpre = dh * (1 - h ** 2)
                new = opt.step(self.params(), [Z.T @ dpre + l2 * self.W, dpre.sum(0), gv, gc] + g_layer)
                self.set_params(new)
            zv = self.logits(Xv)
            auc = _auc(yv, zv)
            hist.append({"epoch": ep + 1, "val_auc": auc, "val_loss": _bce(zv, yv, np.where(yv == 1, cw[1], cw[0])),
                         "seconds": round(time.time() - t0, 1)})
            log(f"[{name}] epoch {ep + 1}: val AUC {auc:.4f}")
            if auc > best + 1e-4:
                best, best_state, wait = auc, [np.copy(p) for p in self.params()], 0
            else:
                wait += 1
                if wait >= patience:
                    break
        self.set_params(best_state)
        return {"history": hist, "best_epoch": int(np.argmax([h["val_auc"] for h in hist]) + 1),
                "train_seconds": round(time.time() - t0, 1), "epochs_max": epochs, "batch": batch, "lr": lr, "l2": l2,
                "patience": patience, "early_stopping": "validation ROC-AUC"}


class HybridV9(_HBase):
    def __init__(self, d_in, n=8, n_layers=3, fusion=False, seed=42):
        super().__init__(d_in, n, fusion, seed)
        self.circ = vqc_circuit(n, n_layers, reupload=True)
        self.n_layers = n_layers
        self.w = np.random.default_rng(seed + 1).uniform(-0.1, 0.1, self.circ.n_weights) * np.pi

    def _layer(self, h):
        return self.circ.forward(0.5 * np.pi * (1 + h), self.w)

    def _layer_grad(self, h, dq):
        gw, gx = self.circ.adjoint_grad(0.5 * np.pi * (1 + h), self.w, dq)
        return [gw], gx * 0.5 * np.pi

    def params(self):
        return [self.W, self.bW, self.v, self.c, self.w]

    def set_params(self, p):
        self.W, self.bW, self.v, self.c, self.w = p[0], p[1], p[2], float(p[3]), p[4]

    def n_params(self):
        return int(self.W.size + self.bW.size + self.v.size + 1 + self.w.size)

    def to_arrays(self):
        return {"W": self.W, "bW": self.bW, "v": self.v, "c": np.array(self.c), "w": self.w, "mu": self.mu, "sd": self.sd,
                "shape": np.array([self.n, self.n_layers, int(self.fusion)])}

    @classmethod
    def from_arrays(cls, a):
        n, L, fu = (int(x) for x in a["shape"])
        m = cls(a["W"].shape[0], n, L, bool(fu))
        m.W, m.bW, m.v, m.c, m.w, m.mu, m.sd = a["W"], a["bW"], a["v"], float(a["c"]), a["w"], a["mu"], a["sd"]
        return m


class TwinV9(_HBase):
    def __init__(self, d_in, n=8, fusion=False, seed=42):
        super().__init__(d_in, n, fusion, seed)
        rng = np.random.default_rng(seed + 1)
        self.U = rng.normal(0, 1 / np.sqrt(n), (n, n))
        self.bU = np.zeros(n)

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

    def n_params(self):
        return int(self.W.size + self.bW.size + self.v.size + 1 + self.U.size + self.bU.size)

    def to_arrays(self):
        return {"W": self.W, "bW": self.bW, "v": self.v, "c": np.array(self.c), "U": self.U, "bU": self.bU,
                "mu": self.mu, "sd": self.sd, "shape": np.array([self.n, int(self.fusion)])}

    @classmethod
    def from_arrays(cls, a):
        n, fu = (int(x) for x in a["shape"])
        m = cls(a["W"].shape[0], n, bool(fu))
        m.W, m.bW, m.v, m.c, m.U, m.bU, m.mu, m.sd = a["W"], a["bW"], a["v"], float(a["c"]), a["U"], a["bU"], a["mu"], a["sd"]
        return m


# ------------------------------------------------------------------------------------------------ noise-aware simulation
def noisy_expect_z(circ: Circuit, X, w, p: float = 0.0, shots: int | None = None, trajectories: int = 32,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """<Z_q> (B, n) under single-qubit depolarizing noise (probability p after every gate on each qubit it touches;
    X, Y or Z error with equal probability) estimated with `trajectories` Pauli trajectories, then (optionally) with
    finite measurement shots (binomial sampling of each qubit's outcome probability)."""
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    if p <= 0:
        ex = circ.expect_z(circ.state(X, w))
    else:
        B = len(X)
        Xr = np.repeat(X, trajectories, 0)
        psi = np.zeros((len(Xr), 2 ** circ.n), np.complex128)
        psi[:, 0] = 1.0
        for op in circ.ops:
            k = op[0]
            if k in ("RX", "RY", "RZ"):
                psi = circ._apply_1q(psi, op[1], _rot(k, circ.angles(op, Xr, w)))
                qs = [op[1]]
            elif k == "H":
                from src.quantum.simulator import H_GATE
                psi = circ._apply_1q(psi, op[1], H_GATE)
                qs = [op[1]]
            elif k == "CNOT":
                psi = circ._apply_cnot(psi, op[1], op[2])
                qs = [op[1], op[2]]
            else:
                psi = circ._apply_cz(psi, op[1], op[2])
                qs = [op[1], op[2]]
            for q in qs:
                hit = rng.random(len(Xr)) < p
                if hit.any():
                    kind = rng.integers(0, 3, len(Xr))
                    for kk, name in enumerate(("RX", "RY", "RZ")):
                        m = hit & (kind == kk)
                        if m.any():
                            psi[m] = circ._apply_pauli(psi[m], name, q)
        ex = circ.expect_z(psi).reshape(B, trajectories, circ.n).mean(1)
    if shots:
        p1 = np.clip((1 - ex) / 2, 0, 1)
        ex = 1 - 2 * rng.binomial(shots, p1) / shots
    return ex


def noisy_kernel(K: np.ndarray, n_qubits: int, n_gates: int, p: float = 0.0, shots: int | None = None,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Fidelity kernel under a global-depolarizing approximation (survival (1-p)^gates per state, two states) and
    finite shots of the compute-uncompute test (binomial estimate of each entry)."""
    rng = rng or np.random.default_rng(0)
    lam = (1 - p) ** (2 * n_gates) if p > 0 else 1.0
    Kn = lam * K + (1 - lam) / 2 ** n_qubits
    if shots:
        Kn = rng.binomial(shots, np.clip(Kn, 0, 1)) / shots
    return Kn


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    t = np.where(np.asarray(y) == 1, 1.0, -1.0)
    Y = np.outer(t, t)
    return float((K * Y).sum() / (np.linalg.norm(K) * np.linalg.norm(Y) + 1e-12))


class QKernelV9:
    """ZZ-feature-map fidelity-kernel SVM with n qubits; bandwidth and C chosen on validation (with kernel-target
    alignment on training data reported for every bandwidth)."""

    def __init__(self, n_qubits=8, reps=2, bandwidth=1.0, C=1.0):
        self.fm = ZZFeatureMap(n_qubits, reps)
        self.n_qubits, self.reps, self.bandwidth, self.C = n_qubits, reps, bandwidth, C

    def states(self, A):
        return self.fm.states(self.bandwidth * np.asarray(A, float))

    def fit(self, A, y, K=None):
        from sklearn.svm import SVC
        S = self.states(A)
        K = fidelity_kernel(S, S) if K is None else K
        self.svc = SVC(kernel="precomputed", C=self.C, class_weight="balanced").fit(K, y)
        sv = self.svc.support_
        self.sv_inputs, self.sv_states = np.asarray(A, float)[sv], S[sv]
        self.dual, self.intercept = self.svc.dual_coef_.ravel(), float(self.svc.intercept_[0])
        return self

    def decision(self, A, K=None):
        Kx = fidelity_kernel(self.states(A), self.sv_states) if K is None else K
        return Kx @ self.dual + self.intercept

    def to_arrays(self):
        return {"sv_inputs": self.sv_inputs, "dual": self.dual, "intercept": np.array(self.intercept),
                "hyper": np.array([self.n_qubits, self.reps, self.bandwidth, self.C])}

    @classmethod
    def from_arrays(cls, a):
        n, reps, bw, C = a["hyper"]
        m = cls(int(n), int(reps), float(bw), float(C))
        m.sv_inputs, m.dual, m.intercept = np.asarray(a["sv_inputs"]), np.asarray(a["dual"]), float(a["intercept"])
        m.sv_states = m.states(m.sv_inputs)
        return m
