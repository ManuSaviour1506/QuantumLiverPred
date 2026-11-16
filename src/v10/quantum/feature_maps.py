"""Diagonal Pauli-Z feature maps for quantum kernels.

One repetition applies H on every qubit and then the diagonal unitary

    D(x) = exp( -i [ sum_q s * u_q Z_q  +  sum_(a,b) phi(u_a, u_b) Z_a Z_b ] ),   u = bandwidth * x,
    phi(u_a, u_b) = c0 * pi^2 - c1 * pi * (u_a + u_b) + c2 * u_a * u_b.

This equals the gate sequence H, RZ(2 s u_q), CNOT(a, b) RZ_b(2 phi) CNOT(a, b) used by V9 (src/quantum/simulator.py::
ZZFeatureMap) and by Qiskit's ZZFeatureMap (P gates instead of RZ: a global phase that cancels in the fidelity).

Pair conventions (c0, c1, c2):
  v9 / qiskit_zz     (1, 1, 1)      phi = (pi - u_a)(pi - u_b)
  product(alpha)     (0, 0, alpha)  phi = alpha u_a u_b      (the "scaled pairwise" alternative)
  none               (0, 0, 0)      product state, no entanglement
  the V9 phase can be split into constant (1, 0, 0), linear (0, 1, 0) and bilinear (0, 0, 1) parts; all three act
  through Z_a Z_b and are therefore two-qubit (entangling) terms, including the constant and the linear part.
"""
from __future__ import annotations

import numpy as np

from src.v10.quantum.circuits import entangler_pairs
from src.v10.quantum.simulator import Circuit, z_signs

PAIR_PRESETS = {"v9": (1.0, 1.0, 1.0), "qiskit_zz": (1.0, 1.0, 1.0), "none": (0.0, 0.0, 0.0),
                "v9_constant": (1.0, 0.0, 0.0), "v9_linear": (0.0, 1.0, 0.0), "v9_bilinear": (0.0, 0.0, 1.0),
                "v9_without_bilinear": (1.0, 1.0, 0.0), "v9_without_constant": (0.0, 1.0, 1.0)}


class ZFeatureMap:
    def __init__(self, n: int, reps: int = 2, pair: str | tuple = "v9", alpha: float = 1.0, entanglement: str = "linear",
                 bandwidth: float = 1.0, single_scale: float = 1.0):
        self.n, self.reps, self.entanglement, self.bandwidth, self.single_scale = n, reps, entanglement, bandwidth, single_scale
        if isinstance(pair, str) and pair == "product":
            self.coeffs = (0.0, 0.0, float(alpha))
        elif isinstance(pair, str):
            self.coeffs = PAIR_PRESETS[pair]
        else:
            self.coeffs = tuple(float(c) for c in pair)
        self.pair_name = pair if isinstance(pair, str) else "custom"
        self.pairs = entangler_pairs(n, entanglement if entanglement != "ring" else "ring")
        self._zs = np.stack([z_signs(n, (q,)) for q in range(n)])                 # (n, 2**n)
        self._zz = np.stack([z_signs(n, p) for p in self.pairs]) if self.pairs else np.zeros((0, 2 ** n))

    def describe(self) -> dict:
        return {"qubits": self.n, "reps": self.reps, "pair": self.pair_name, "coefficients_c0_c1_c2": list(self.coeffs),
                "entanglement": self.entanglement, "pairs": [list(p) for p in self.pairs], "bandwidth": self.bandwidth,
                "single_scale": self.single_scale}

    def pair_phases(self, X) -> np.ndarray:
        """phi(u_a, u_b) for every row and pair: (B, n_pairs)."""
        U = self.bandwidth * np.asarray(X, float)
        c0, c1, c2 = self.coeffs
        if not self.pairs:
            return np.zeros((len(U), 0))
        a = np.array([p[0] for p in self.pairs])
        b = np.array([p[1] for p in self.pairs])
        return c0 * np.pi ** 2 - c1 * np.pi * (U[:, a] + U[:, b]) + c2 * U[:, a] * U[:, b]

    def diagonal_phase(self, X) -> np.ndarray:
        """theta(x) of D(x) = exp(-i theta) for every basis state: (B, 2**n)."""
        U = self.bandwidth * np.asarray(X, float)
        th = self.single_scale * U @ self._zs
        if self.pairs:
            th = th + self.pair_phases(X) @ self._zz
        return th

    def states(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        n = self.n
        th = self.diagonal_phase(X)
        D = np.exp(-1j * th)
        psi = np.zeros((len(X), 2 ** n), np.complex128)
        psi[:, 0] = 1.0
        for _ in range(self.reps):
            psi = hadamard_all(psi, n)
            psi = psi * D
        return psi

    def gate_circuit(self, feature_offset: int = 0) -> Circuit:
        """The same map as an explicit gate list (H, RZ, CNOT-RZ-CNOT), for noisy and hardware-style simulation."""
        ops = []
        c0, c1, c2 = self.coeffs
        bw, s = self.bandwidth, self.single_scale
        for _ in range(self.reps):
            for q in range(self.n):
                ops.append(("H", (q,), None))
            for q in range(self.n):
                ops.append(("RZ", (q,), ("x", q + feature_offset, 2.0 * s * bw, 0.0)))
            for a, b in self.pairs:
                ops.append(("CNOT", (a, b), None))
                ops.append(("RZ", (b,), ("pair", a + feature_offset, b + feature_offset, c0, c1, c2, bw)))
                ops.append(("CNOT", (a, b), None))
        return Circuit(self.n, ops, self.n + feature_offset, {"feature_map": self.describe()})

    def compute_uncompute_circuit(self) -> Circuit:
        """U(x_b)^dag U(x_a) acting on |0>, inputs given as [x_a, x_b] (2n features). P(0...0) = |<psi(x_b)|psi(x_a)>|^2."""
        return self.gate_circuit(0).compose(self.gate_circuit(self.n).inverse())


def hadamard_all(psi: np.ndarray, n: int) -> np.ndarray:
    v = psi.reshape((psi.shape[0],) + (2,) * n)
    h = np.array([[1, 1], [1, -1]], np.complex128) / np.sqrt(2)
    for q in range(n):
        v = np.moveaxis(np.tensordot(v, h, axes=([q + 1], [1])), -1, q + 1)
    return v.reshape(psi.shape)


def v9_phase_parts(X, bandwidth: float, pairs) -> dict:
    """Constant, linear and bilinear parts of the V9 pair phase over the given inputs (each (B, n_pairs))."""
    U = bandwidth * np.asarray(X, float)
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    const = np.full((len(U), len(pairs)), np.pi ** 2)
    lin = -np.pi * (U[:, a] + U[:, b])
    bil = U[:, a] * U[:, b]
    return {"constant": const, "linear": lin, "bilinear": bil, "total": const + lin + bil}
