"""Exact state-vector simulator for small parameterised circuits (NumPy). SIMULATION ONLY - no quantum hardware.

A circuit is a list of operations
    ("RX" | "RY" | "RZ", qubit, param)   rotation exp(-i theta P / 2); param is ("x", feature, scale, offset) for a
                                         data-encoding angle theta = scale * x[feature] + offset, or ("w", index) for a
                                         trainable weight, or a float constant
    ("H", qubit)  ("CNOT", control, target)  ("CZ", a, b)
evaluated for a batch of inputs at once: the state is a complex array (batch, 2**n).
Gradients of <Z_q> observables with respect to trainable weights AND data angles are computed with the adjoint
method (exact; Jones & Gacon 2020). parameter_shift_grad() gives the hardware-compatible estimator
(df/dtheta = [f(theta + pi/2) - f(theta - pi/2)] / 2) and is tested to agree with the adjoint gradients.
"""
from __future__ import annotations

import numpy as np

SQ = {"RX", "RY", "RZ", "H"}


def _rot(kind, th):
    """Batch of 2x2 rotation matrices for angles th (B,)."""
    c, s = np.cos(th / 2), np.sin(th / 2)
    B = len(th)
    U = np.zeros((B, 2, 2), np.complex128)
    if kind == "RX":
        U[:, 0, 0] = c
        U[:, 1, 1] = c
        U[:, 0, 1] = -1j * s
        U[:, 1, 0] = -1j * s
    elif kind == "RY":
        U[:, 0, 0] = c
        U[:, 1, 1] = c
        U[:, 0, 1] = -s
        U[:, 1, 0] = s
    elif kind == "RZ":
        U[:, 0, 0] = np.exp(-0.5j * th)
        U[:, 1, 1] = np.exp(0.5j * th)
    return U


H_GATE = np.array([[1, 1], [1, -1]], np.complex128) / np.sqrt(2)


class Circuit:
    def __init__(self, n_qubits: int, ops: list):
        self.n = n_qubits
        self.ops = ops
        self.n_weights = 1 + max([op[2][1] for op in ops if op[0] in ("RX", "RY", "RZ") and isinstance(op[2], tuple)
                                  and op[2][0] == "w"] + [-1])

    # ------------------------------------------------------------------ elementary operations on (B, 2**n)
    def _view(self, psi):
        return psi.reshape((psi.shape[0],) + (2,) * self.n)

    def _apply_1q(self, psi, q, U):
        v = np.moveaxis(self._view(psi), q + 1, -1)
        sh = v.shape
        v = v.reshape(sh[0], -1, 2)
        if U.ndim == 2:
            out = v @ U.T
        else:
            out = np.einsum("bij,bkj->bki", U, v)
        return np.moveaxis(out.reshape(sh), -1, q + 1).reshape(psi.shape)

    def _apply_cnot(self, psi, c, t):
        v = self._view(psi).copy()
        idx = [slice(None)] * (self.n + 1)
        idx[c + 1] = 1
        sub = v[tuple(idx)]
        tt = t + 1 if t < c else t  # axis index after removing control axis (batch axis stays 0)
        v[tuple(idx)] = np.flip(sub, axis=tt).copy()
        return v.reshape(psi.shape)

    def _apply_cz(self, psi, a, b):
        v = self._view(psi).copy()
        idx = [slice(None)] * (self.n + 1)
        idx[a + 1] = 1
        idx[b + 1] = 1
        v[tuple(idx)] *= -1
        return v.reshape(psi.shape)

    def _apply_pauli(self, psi, kind, q):
        v = self._view(psi)
        out = np.empty_like(v)
        i0 = [slice(None)] * (self.n + 1)
        i1 = list(i0)
        i0[q + 1], i1[q + 1] = 0, 1
        i0, i1 = tuple(i0), tuple(i1)
        if kind == "RX":    # X
            out[i0], out[i1] = v[i1], v[i0]
        elif kind == "RY":  # Y
            out[i0], out[i1] = -1j * v[i1], 1j * v[i0]
        else:               # Z
            out[i0], out[i1] = v[i0], -v[i1]
        return out.reshape(psi.shape)

    # ------------------------------------------------------------------ evaluation
    def angles(self, op, X, w):
        p = op[2]
        B = len(X)
        if isinstance(p, tuple) and p[0] == "x":
            _, f, scale, off = p
            return scale * X[:, f] + off
        if isinstance(p, tuple) and p[0] == "w":
            return np.full(B, w[p[1]])
        return np.full(B, float(p))

    def state(self, X, w=None, return_trace=False):
        X = np.asarray(X, float)
        w = np.zeros(self.n_weights) if w is None else np.asarray(w, float)
        psi = np.zeros((len(X), 2 ** self.n), np.complex128)
        psi[:, 0] = 1.0
        for op in self.ops:
            k = op[0]
            if k in ("RX", "RY", "RZ"):
                psi = self._apply_1q(psi, op[1], _rot(k, self.angles(op, X, w)))
            elif k == "H":
                psi = self._apply_1q(psi, op[1], H_GATE)
            elif k == "CNOT":
                psi = self._apply_cnot(psi, op[1], op[2])
            elif k == "CZ":
                psi = self._apply_cz(psi, op[1], op[2])
            else:
                raise ValueError(k)
        return psi

    def z_signs(self, q):
        bits = (np.arange(2 ** self.n) >> (self.n - 1 - q)) & 1
        return 1.0 - 2.0 * bits

    def expect_z(self, psi):
        """<Z_q> for every qubit: (B, n)."""
        prob = np.abs(psi) ** 2
        return np.stack([prob @ self.z_signs(q) for q in range(self.n)], 1)

    def forward(self, X, w):
        return self.expect_z(self.state(X, w))

    def adjoint_grad(self, X, w, dL_dz):
        """Gradients of L = sum_b sum_q dL_dz[b, q] * <Z_q>_b with respect to the weights (n_weights,) and the
        data features (B, n_features). dL_dz: (B, n)."""
        X = np.asarray(X, float)
        w = np.asarray(w, float)
        psi = self.state(X, w)
        obs = sum(dL_dz[:, q:q + 1] * self.z_signs(q)[None, :] for q in range(self.n))
        lam = obs * psi
        gw = np.zeros(len(w))
        gx = np.zeros_like(X)
        for op in reversed(self.ops):
            k = op[0]
            if k in ("RX", "RY", "RZ"):
                th = self.angles(op, X, w)
                gpsi = self._apply_pauli(psi, k, op[1])
                # d<O>/dtheta = Im <lam| G |psi>  (both evaluated after the gate); factor 2 * (-1/2) folded in
                contrib = np.imag(np.sum(np.conj(lam) * gpsi, axis=1)) * 2 * 0.5
                p = op[2]
                if isinstance(p, tuple) and p[0] == "w":
                    gw[p[1]] += contrib.sum()
                elif isinstance(p, tuple) and p[0] == "x":
                    gx[:, p[1]] += contrib * p[2]
                Uinv = np.conj(np.transpose(_rot(k, th), (0, 2, 1)))
                psi = self._apply_1q(psi, op[1], Uinv)
                lam = self._apply_1q(lam, op[1], Uinv)
            elif k == "H":
                psi = self._apply_1q(psi, op[1], H_GATE)
                lam = self._apply_1q(lam, op[1], H_GATE)
            elif k == "CNOT":
                psi = self._apply_cnot(psi, op[1], op[2])
                lam = self._apply_cnot(lam, op[1], op[2])
            elif k == "CZ":
                psi = self._apply_cz(psi, op[1], op[2])
                lam = self._apply_cz(lam, op[1], op[2])
        return gw, gx

    def parameter_shift_grad(self, X, w, dL_dz):
        """Hardware-compatible gradient w.r.t. trainable weights (2 circuit evaluations per weight)."""
        g = np.zeros(len(w))
        for j in range(len(w)):
            wp, wm = np.array(w, float), np.array(w, float)
            wp[j] += np.pi / 2
            wm[j] -= np.pi / 2
            g[j] = 0.5 * np.sum(dL_dz * (self.forward(X, wp) - self.forward(X, wm)))
        return g


# ---------------------------------------------------------------------------------------------- circuit builders
def vqc_circuit(n_qubits: int, n_layers: int, reupload: bool = True) -> Circuit:
    """Angle encoding RY(x_q) (data re-uploaded before every layer) + layers of RY(w) RZ(w) and a CNOT ring."""
    ops, k = [], 0
    for layer in range(n_layers):
        if layer == 0 or reupload:
            ops += [("RY", q, ("x", q, 1.0, 0.0)) for q in range(n_qubits)]
        for q in range(n_qubits):
            ops.append(("RY", q, ("w", k)))
            ops.append(("RZ", q, ("w", k + 1)))
            k += 2
        ops += [("CNOT", q, (q + 1) % n_qubits) for q in range(n_qubits)]
    return Circuit(n_qubits, ops)


class ZZFeatureMap:
    """Havlicek et al. (2019) ZZ feature map: per repetition H on every qubit, RZ(2 x_q), and for neighbouring pairs
    CNOT - RZ(2 (pi - x_a)(pi - x_b)) - CNOT. (RZ differs from the phase gate only by a global phase, which cancels in
    the fidelity kernel.)"""

    def __init__(self, n_qubits: int, reps: int = 2, pairs: str = "linear"):
        self.n, self.reps = n_qubits, reps
        self.pairs = [(q, q + 1) for q in range(n_qubits - 1)] + ([(n_qubits - 1, 0)] if pairs == "ring" else [])
        self.c = Circuit(n_qubits, [])

    def states(self, X):
        X = np.asarray(X, float)
        c = self.c
        psi = np.zeros((len(X), 2 ** self.n), np.complex128)
        psi[:, 0] = 1.0
        for _ in range(self.reps):
            for q in range(self.n):
                psi = c._apply_1q(psi, q, H_GATE)
            for q in range(self.n):
                psi = c._apply_1q(psi, q, _rot("RZ", 2 * X[:, q]))
            for a, b in self.pairs:
                psi = c._apply_cnot(psi, a, b)
                psi = c._apply_1q(psi, b, _rot("RZ", 2 * (np.pi - X[:, a]) * (np.pi - X[:, b])))
                psi = c._apply_cnot(psi, a, b)
        return psi


def fidelity_kernel(SA: np.ndarray, SB: np.ndarray) -> np.ndarray:
    """K[i, j] = |<psi(a_i)|psi(b_j)>|^2 from precomputed states."""
    return np.abs(np.conj(SA) @ SB.T) ** 2
