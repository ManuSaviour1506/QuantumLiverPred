"""Q1 variational quantum classifier and Q2 quantum-kernel SVM (exact state-vector SIMULATION, 8 qubits).

Input to both: frozen ImageNet EfficientNet-B0 patch features (1280-d GAP) -> standardise -> PCA to 8 components
(fitted on the training split) -> each component min-max scaled on the training split to [0, pi] (clipped).
Q1  angle encoding RY(x_q), data re-uploading before each of 3 variational layers (RY, RZ per qubit + CNOT ring),
    readout <Z_0>, logit = s * <Z_0> + b. 48 circuit parameters + 2 readout parameters. Class-balanced BCE, Adam,
    exact adjoint gradients (equal to parameter-shift gradients, see tests), early stopping on validation loss.
Q2  ZZ feature map (2 repetitions, linear entanglement), fidelity kernel K = |<psi(x)|psi(x')>|^2, SVC with a
    precomputed kernel. Feature bandwidth and C are chosen on validation AUC.
"""
from __future__ import annotations

import time

import numpy as np

from src.quantum.simulator import ZZFeatureMap, fidelity_kernel, vqc_circuit

N_QUBITS = 8


class PCAEncoder:
    """standardise -> PCA(k) -> min-max to [0, pi] (all fitted on training rows only)."""

    def fit(self, X, k=N_QUBITS):
        X = np.asarray(X, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - self.mu) / self.sd
        U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        self.pmean, self.comp = Z.mean(0), Vt[:k]
        P = (Z - self.pmean) @ self.comp.T
        self.lo, self.hi = P.min(0), P.max(0)
        self.explained = float((S[:k] ** 2).sum() / (S ** 2).sum())
        return self

    def transform(self, X):
        P = ((np.asarray(X, float) - self.mu) / self.sd - self.pmean) @ self.comp.T
        return np.clip((P - self.lo) / (self.hi - self.lo + 1e-12), 0, 1) * np.pi

    def to_arrays(self, pre="enc"):
        return {f"{pre}_mu": self.mu, f"{pre}_sd": self.sd, f"{pre}_pmean": self.pmean, f"{pre}_comp": self.comp,
                f"{pre}_lo": self.lo, f"{pre}_hi": self.hi}

    @classmethod
    def from_arrays(cls, a, pre="enc"):
        e = cls()
        e.mu, e.sd, e.pmean, e.comp, e.lo, e.hi = (np.asarray(a[f"{pre}_{k}"]) for k in ("mu", "sd", "pmean", "comp", "lo", "hi"))
        return e


class Adam:
    def __init__(self, shapes, lr):
        self.lr, self.t = lr, 0
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]

    def step(self, params, grads):
        self.t += 1
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh, vh = self.m[i] / (1 - 0.9 ** self.t), self.v[i] / (1 - 0.999 ** self.t)
            out.append(p - self.lr * mh / (np.sqrt(vh) + 1e-8))
        return out


def _bce(z, y, w):
    return float(np.sum(w * (np.logaddexp(0, z) - y * z)) / np.sum(w))


class VQC:
    def __init__(self, n_qubits=N_QUBITS, n_layers=3, seed=42):
        self.circ = vqc_circuit(n_qubits, n_layers, reupload=True)
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(-0.1, 0.1, self.circ.n_weights) * np.pi
        self.s, self.b = 2.0, 0.0
        self.n_qubits, self.n_layers, self.seed = n_qubits, n_layers, seed

    def logits(self, A):
        return self.s * self.circ.forward(A, self.w)[:, 0] + self.b

    def fit(self, A, y, Av, yv, epochs=40, batch=64, lr=0.05, patience=6, log=print):
        rng = np.random.default_rng(self.seed)
        cw = {1: 0.5 / y.mean(), 0: 0.5 / (1 - y.mean())}
        opt = Adam([self.w.shape, (), ()], lr)
        best, best_state, wait, hist, t0 = np.inf, None, 0, [], time.time()
        for ep in range(epochs):
            order = rng.permutation(len(y))
            for s in range(0, len(y), batch):
                i = order[s:s + batch]
                ex = self.circ.forward(A[i], self.w)
                z = self.s * ex[:, 0] + self.b
                wt = np.where(y[i] == 1, cw[1], cw[0])
                dz = wt * (1 / (1 + np.exp(-z)) - y[i]) / wt.sum()
                dL = np.zeros_like(ex)
                dL[:, 0] = dz * self.s
                gw, _ = self.circ.adjoint_grad(A[i], self.w, dL)
                gs, gb = float(np.sum(dz * ex[:, 0])), float(np.sum(dz))
                self.w, self.s, self.b = opt.step([self.w, self.s, self.b], [gw, gs, gb])
            wtv = np.where(yv == 1, cw[1], cw[0])
            vl = _bce(self.logits(Av), yv, wtv)
            trl = _bce(self.logits(A), y, np.where(y == 1, cw[1], cw[0]))
            hist.append({"epoch": ep + 1, "train_loss": trl, "val_loss": vl, "seconds": round(time.time() - t0, 1)})
            log(f"[VQC] epoch {ep + 1}: train {trl:.4f} val {vl:.4f}")
            if vl < best - 1e-4:
                best, best_state, wait = vl, (self.w.copy(), self.s, self.b), 0
            else:
                wait += 1
                if wait >= patience:
                    break
        self.w, self.s, self.b = best_state
        return {"history": hist, "best_epoch": int(np.argmin([h["val_loss"] for h in hist]) + 1),
                "train_seconds": round(time.time() - t0, 1), "epochs_max": epochs, "batch": batch, "lr": lr,
                "patience": patience, "gradient": "adjoint (exact; equals parameter-shift)"}

    def to_arrays(self):
        return {"w": self.w, "readout": np.array([self.s, self.b]), "shape": np.array([self.n_qubits, self.n_layers])}

    @classmethod
    def from_arrays(cls, a):
        m = cls(int(a["shape"][0]), int(a["shape"][1]))
        m.w = np.asarray(a["w"], float)
        m.s, m.b = (float(v) for v in a["readout"])
        return m


class QKernelSVM:
    def __init__(self, n_qubits=N_QUBITS, reps=2, bandwidth=1.0, C=1.0):
        self.fm = ZZFeatureMap(n_qubits, reps)
        self.n_qubits, self.reps, self.bandwidth, self.C = n_qubits, reps, bandwidth, C

    def states(self, A):
        return self.fm.states(self.bandwidth * np.asarray(A, float))

    def fit(self, A, y):
        from sklearn.svm import SVC
        S = self.states(A)
        K = fidelity_kernel(S, S)
        self.svc = SVC(kernel="precomputed", C=self.C, class_weight="balanced").fit(K, y)
        sv = self.svc.support_
        self.sv_inputs = np.asarray(A, float)[sv]
        self.sv_states = S[sv]
        self.dual = self.svc.dual_coef_.ravel()
        self.intercept = float(self.svc.intercept_[0])
        self.kernel_train_stats = {"mean_offdiag": float((K.sum() - np.trace(K)) / (K.size - len(K))),
                                   "min_eig": float(np.linalg.eigvalsh(K[:300, :300]).min())}
        return self

    def decision(self, A):
        return fidelity_kernel(self.states(A), self.sv_states) @ self.dual + self.intercept

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
