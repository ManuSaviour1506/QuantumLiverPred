"""Exact batched state-vector simulator for small parameterised circuits (NumPy). CLASSICAL SIMULATION ONLY.

Conventions (identical to V9's src/quantum/simulator.py; checked in tests/test_v10_simulator.py)
  * Qubit 0 is the most significant bit of the basis index: |q0 q1 ... q_{n-1}>.
  * RX(t) = exp(-i t X / 2), RY(t) = exp(-i t Y / 2), RZ(t) = exp(-i t Z / 2) = diag(e^{-it/2}, e^{it/2}).
  * RZZ(t) = exp(-i t Z_a Z_b / 2). CNOT(c, t) flips t when c = 1. CZ(a, b) = diag(1, 1, 1, -1).
  * Observables are Pauli-Z strings given as tuples of qubit indices; () is the identity.

Gate list
  An op is (name, wires, param). name in {RX, RY, RZ, RZZ, H, X, Y, Z, CNOT, CZ}. param is one of
    None                          fixed gate without angle
    ("c", value)                  constant angle
    ("w", k)                      trainable weight k
    ("x", f, scale, offset)       data angle  scale * x[f] + offset
    ("ax", f, ka, ko)             trainable data angle  w[ka] * x[f] + (w[ko] if ko is not None else 0)
    ("pair", fa, fb, c0, c1, c2, bw)
                                  feature-map pair angle 2 * phi with u = bw * x and
                                  phi = c0 * pi^2 - c1 * pi * (u_a + u_b) + c2 * u_a * u_b
                                  (V9 / Qiskit ZZ map: c0 = c1 = c2 = 1)
    ("neg", inner)                minus the inner angle (used to build inverse circuits)

Gradients
  adjoint_grad: exact gradients of sum_b sum_m dL[b, m] <O_m>_b with respect to every weight and every data feature.
  parameter_shift_grad: the hardware-compatible estimator, applied per gate angle (two evaluations per parameterised
  gate) and chained to the weights; valid because every generator here squares to the identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ROT = {"RX", "RY", "RZ"}
PARAM_GATES = ROT | {"RZZ"}
FIXED_1Q = {"H", "X", "Y", "Z"}
TWO_Q = {"CNOT", "CZ", "RZZ"}

_H = np.array([[1, 1], [1, -1]], np.complex128) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]], np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], np.complex128)
_Z = np.array([[1, 0], [0, -1]], np.complex128)
FIXED_MATRIX = {"H": _H, "X": _X, "Y": _Y, "Z": _Z}
PAULI = {"X": _X, "Y": _Y, "Z": _Z}


def rotation_matrices(kind: str, theta: np.ndarray) -> np.ndarray:
    """(B, 2, 2) matrices of RX / RY / RZ for angles theta (B,)."""
    theta = np.asarray(theta, float)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    U = np.zeros((len(theta), 2, 2), np.complex128)
    if kind == "RX":
        U[:, 0, 0], U[:, 1, 1], U[:, 0, 1], U[:, 1, 0] = c, c, -1j * s, -1j * s
    elif kind == "RY":
        U[:, 0, 0], U[:, 1, 1], U[:, 0, 1], U[:, 1, 0] = c, c, -s, s
    elif kind == "RZ":
        U[:, 0, 0], U[:, 1, 1] = np.exp(-0.5j * theta), np.exp(0.5j * theta)
    else:
        raise ValueError(kind)
    return U


def gate_matrix(name: str, theta: float | None = None) -> np.ndarray:
    """Dense matrix of one gate on its own wires (2x2 or 4x4, wire order as given)."""
    if name in FIXED_MATRIX:
        return FIXED_MATRIX[name].copy()
    if name in ROT:
        return rotation_matrices(name, np.array([theta]))[0]
    if name == "CNOT":
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], np.complex128)
    if name == "CZ":
        return np.diag([1, 1, 1, -1]).astype(np.complex128)
    if name == "RZZ":
        return np.diag(np.exp(-0.5j * theta * np.array([1, -1, -1, 1]))).astype(np.complex128)
    raise ValueError(name)


def z_signs(n: int, mask) -> np.ndarray:
    """Eigenvalues (+1 / -1) of the Pauli-Z string on `mask` for every basis state (length 2**n)."""
    idx = np.arange(2 ** n)
    s = np.ones(2 ** n)
    for q in mask:
        s = s * (1.0 - 2.0 * ((idx >> (n - 1 - q)) & 1))
    return s


@dataclass
class Circuit:
    n: int
    ops: list
    n_features: int | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        ks, fs = [-1], [-1]
        for name, wires, p in self.ops:
            for q in wires:
                if not 0 <= q < self.n:
                    raise ValueError(f"wire {q} outside 0..{self.n - 1}")
            ks += _weight_indices(p)
            fs += _feature_indices(p)
        self.n_weights = max(ks) + 1
        if self.n_features is None:
            self.n_features = max(fs) + 1

    # ----------------------------------------------------------------------------------------- angles
    def angle(self, p, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        B = X.shape[0]
        kind = p[0]
        if kind == "c":
            return np.full(B, float(p[1]))
        if kind == "w":
            return np.full(B, float(w[p[1]]))
        if kind == "x":
            return p[2] * X[:, p[1]] + p[3]
        if kind == "ax":
            return w[p[2]] * X[:, p[1]] + (w[p[3]] if p[3] is not None else 0.0)
        if kind == "pair":
            _, fa, fb, c0, c1, c2, bw = p
            ua, ub = bw * X[:, fa], bw * X[:, fb]
            return 2.0 * (c0 * np.pi ** 2 - c1 * np.pi * (ua + ub) + c2 * ua * ub)
        if kind == "neg":
            return -self.angle(p[1], X, w)
        raise ValueError(p)

    def angle_partials(self, p, X: np.ndarray, w: np.ndarray):
        """d angle / d w[k] and d angle / d x[f] as lists of (index, (B,) array)."""
        B = X.shape[0]
        kind = p[0]
        if kind == "c":
            return [], []
        if kind == "w":
            return [(p[1], np.ones(B))], []
        if kind == "x":
            return [], [(p[1], np.full(B, float(p[2])))]
        if kind == "ax":
            dw = [(p[2], X[:, p[1]].copy())] + ([(p[3], np.ones(B))] if p[3] is not None else [])
            return dw, [(p[1], np.full(B, float(w[p[2]])))]
        if kind == "pair":
            _, fa, fb, c0, c1, c2, bw = p
            ua, ub = bw * X[:, fa], bw * X[:, fb]
            return [], [(fa, 2.0 * bw * (-c1 * np.pi + c2 * ub)), (fb, 2.0 * bw * (-c1 * np.pi + c2 * ua))]
        if kind == "neg":
            dw, dx = self.angle_partials(p[1], X, w)
            return [(k, -v) for k, v in dw], [(f, -v) for f, v in dx]
        raise ValueError(p)

    # ------------------------------------------------------------------------------- elementary updates
    def _view(self, psi):
        return psi.reshape((psi.shape[0],) + (2,) * self.n)

    def apply_1q(self, psi, q, U):
        v = np.moveaxis(self._view(psi), q + 1, -1)
        sh = v.shape
        v = v.reshape(sh[0], -1, 2)
        out = v @ U.T if U.ndim == 2 else np.einsum("bij,bkj->bki", U, v)
        return np.moveaxis(out.reshape(sh), -1, q + 1).reshape(psi.shape)

    def apply_cnot(self, psi, c, t):
        v = self._view(psi).copy()
        idx = [slice(None)] * (self.n + 1)
        idx[c + 1] = 1
        tt = t + 1 if t < c else t
        v[tuple(idx)] = np.flip(v[tuple(idx)], axis=tt).copy()
        return v.reshape(psi.shape)

    def apply_cz(self, psi, a, b):
        return psi * z_signs_cz(self.n, a, b)[None, :]

    def apply_rzz(self, psi, a, b, theta):
        s = z_signs(self.n, (a, b))
        return psi * np.exp(-0.5j * theta[:, None] * s[None, :])

    def apply_pauli(self, psi, kind, q):
        return self.apply_1q(psi, q, PAULI[kind])

    def apply_generator(self, psi, name, wires):
        if name == "RZZ":
            return psi * z_signs(self.n, wires)[None, :]
        return self.apply_pauli(psi, name[1], wires[0])

    def apply_op(self, psi, op, theta=None, inverse=False):
        name, wires, p = op
        if name in ROT:
            th = -theta if inverse else theta
            return self.apply_1q(psi, wires[0], rotation_matrices(name, th))
        if name == "RZZ":
            return self.apply_rzz(psi, wires[0], wires[1], -theta if inverse else theta)
        if name in FIXED_1Q:
            return self.apply_1q(psi, wires[0], FIXED_MATRIX[name])
        if name == "CNOT":
            return self.apply_cnot(psi, wires[0], wires[1])
        if name == "CZ":
            return self.apply_cz(psi, wires[0], wires[1])
        raise ValueError(name)

    # ------------------------------------------------------------------------------------- evaluation
    def _prep(self, X, w):
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[1] < self.n_features:
            raise ValueError(f"inputs have {X.shape[1]} features, circuit needs {self.n_features}")
        w = np.zeros(self.n_weights) if w is None else np.asarray(w, float)
        if w.size < self.n_weights:
            raise ValueError(f"{w.size} weights given, circuit needs {self.n_weights}")
        return X, w

    def state(self, X, w=None, shift: tuple | None = None, psi0=None) -> np.ndarray:
        """Final states (B, 2**n). shift = (op_index, delta) adds delta to that gate's angle (parameter shift)."""
        X, w = self._prep(X, w)
        if psi0 is None:
            psi = np.zeros((len(X), 2 ** self.n), np.complex128)
            psi[:, 0] = 1.0
        else:
            psi = np.array(psi0, np.complex128)
        for i, op in enumerate(self.ops):
            th = None
            if op[0] in PARAM_GATES:
                th = self.angle(op[2], X, w)
                if shift is not None and shift[0] == i:
                    th = th + shift[1]
            psi = self.apply_op(psi, op, th)
        return psi

    def expect(self, psi, masks) -> np.ndarray:
        prob = np.abs(psi) ** 2
        return np.stack([prob @ z_signs(self.n, m) for m in masks], 1)

    def probabilities(self, X, w=None) -> np.ndarray:
        return np.abs(self.state(X, w)) ** 2

    def forward(self, X, w=None, masks=None) -> np.ndarray:
        masks = [(q,) for q in range(self.n)] if masks is None else masks
        return self.expect(self.state(X, w), masks)

    def unitary(self, x, w=None) -> np.ndarray:
        """Dense unitary (2**n x 2**n) for one input row x (column j = U|j>). Intended for n <= 8."""
        D = 2 ** self.n
        X = np.repeat(np.atleast_2d(np.asarray(x, float)), D, 0)
        cols = self.state(X, w, psi0=np.eye(D, dtype=np.complex128))
        return cols.T

    # -------------------------------------------------------------------------------------- gradients
    def adjoint_grad(self, X, w, dL_dobs, masks):
        """Exact gradients of L = sum_b sum_m dL_dobs[b, m] <Z_mask_m>_b. Returns (g_w (n_weights,), g_x (B, n_feat))."""
        X, w = self._prep(X, w)
        psi = self.state(X, w)
        O = np.asarray(dL_dobs, float) @ np.stack([z_signs(self.n, m) for m in masks])  # (B, 2**n) diagonal
        lam = O * psi
        gw = np.zeros(self.n_weights)
        gx = np.zeros((len(X), X.shape[1]))
        for op in reversed(self.ops):
            name, wires, p = op
            if name in PARAM_GATES:
                th = self.angle(p, X, w)
                d_theta = np.imag(np.sum(np.conj(lam) * self.apply_generator(psi, name, wires), axis=1))
                dw, dx = self.angle_partials(p, X, w)
                for k, v in dw:
                    gw[k] += float(np.sum(d_theta * v))
                for f, v in dx:
                    gx[:, f] += d_theta * v
                psi = self.apply_op(psi, op, th, inverse=True)
                lam = self.apply_op(lam, op, th, inverse=True)
            else:
                psi = self.apply_op(psi, op)
                lam = self.apply_op(lam, op)
        return gw, gx

    def parameter_shift_grad(self, X, w, dL_dobs, masks):
        """Gradient with respect to the weights by the two-term shift rule on every parameterised gate."""
        X, w = self._prep(X, w)
        dL = np.asarray(dL_dobs, float)
        gw = np.zeros(self.n_weights)
        for i, op in enumerate(self.ops):
            if op[0] not in PARAM_GATES:
                continue
            dw, _ = self.angle_partials(op[2], X, w)
            if not dw:
                continue
            fp = self.expect(self.state(X, w, shift=(i, np.pi / 2)), masks)
            fm = self.expect(self.state(X, w, shift=(i, -np.pi / 2)), masks)
            d_theta = 0.5 * np.sum(dL * (fp - fm), axis=1)
            for k, v in dw:
                gw[k] += float(np.sum(d_theta * v))
        return gw

    # ------------------------------------------------------------------------------------- structure
    def inverse(self) -> "Circuit":
        ops = []
        for name, wires, p in reversed(self.ops):
            ops.append((name, wires, ("neg", p) if name in PARAM_GATES else p))
        return Circuit(self.n, ops, self.n_features, dict(self.meta, inverse=True))

    def compose(self, other: "Circuit") -> "Circuit":
        if other.n != self.n:
            raise ValueError("qubit counts differ")
        return Circuit(self.n, self.ops + other.ops, max(self.n_features, other.n_features), {})


def z_signs_cz(n: int, a: int, b: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    both = ((idx >> (n - 1 - a)) & 1) & ((idx >> (n - 1 - b)) & 1)
    return 1.0 - 2.0 * both


def shift_features(p, offset: int):
    """Same parameter spec with every feature index moved by `offset` (used to build compute-uncompute circuits)."""
    if p is None:
        return None
    kind = p[0]
    if kind == "x":
        return ("x", p[1] + offset, p[2], p[3])
    if kind == "ax":
        return ("ax", p[1] + offset, p[2], p[3])
    if kind == "pair":
        return ("pair", p[1] + offset, p[2] + offset) + tuple(p[3:])
    if kind == "neg":
        return ("neg", shift_features(p[1], offset))
    return p


def _weight_indices(p):
    if p is None:
        return []
    if p[0] == "w":
        return [p[1]]
    if p[0] == "ax":
        return [p[2]] + ([p[3]] if p[3] is not None else [])
    if p[0] == "neg":
        return _weight_indices(p[1])
    return []


def _feature_indices(p):
    if p is None:
        return []
    if p[0] in ("x", "ax"):
        return [p[1]]
    if p[0] == "pair":
        return [p[1], p[2]]
    if p[0] == "neg":
        return _feature_indices(p[1])
    return []
