"""Noise models for V10 (classical simulation of noisy circuits; NOT hardware-calibrated unless a calibration is given).

Exact density-matrix simulation (DensityMatrixSimulator), intended for n <= 8 qubits, with separate channels:
  depolarizing_1q(p1)   after every one-qubit gate on its wire:  rho -> (1 - p1) rho + p1/3 (X rho X + Y rho Y + Z rho Z)
                        (Pauli-error convention: p1 = probability that some Pauli error occurs; the "replace by I/2"
                        convention with parameter lambda corresponds to p1 = 3 lambda / 4)
  depolarizing_2q(p2)   after every two-qubit gate on its pair:  rho -> (1 - p2) rho + p2/15 sum_{P != II} P rho P
  amplitude damping     Kraus K0 = [[1, 0], [0, sqrt(1 - g)]], K1 = [[0, sqrt(g)], [0, 0]]
  phase damping         Kraus K0 = [[1, 0], [0, sqrt(1 - l)]], K1 = [[0, 0], [0, sqrt(l)]]  (coherences x sqrt(1 - l))
  thermal relaxation    from T1, T2 and a duration t:  g = 1 - exp(-t / T1),  l = 1 - exp(-t (2 / T2 - 1 / T1))
                        so that populations relax with T1 and coherences decay as exp(-t / T2); requires T2 <= 2 T1
  idle relaxation       when idle=True the circuit is scheduled into moments (as soon as possible); a qubit idle in a
                        moment (or idle for the rest of it) relaxes for that time
  readout assignment    per qubit P(read 1 | 0) = e0, P(read 0 | 1) = e1, applied to the measured distribution:
                        <Z>_measured = (1 - e0 - e1) <Z> + (e1 - e0)
  finite shots          multinomial sampling of measured bit strings; a Z-string estimate from N shots has standard error
                        sqrt((1 - <P>^2) / N) (maximum 1/sqrt(N)); a probability estimate sqrt(p (1 - p) / N).

Monte-Carlo trajectories (TrajectorySimulator) for the Pauli channels only (depolarizing 1q / 2q): every trajectory
draws its own Pauli errors from an independent seed (numpy SeedSequence.spawn); the trajectory mean converges to the
exact density-matrix result with standard error ~ 1/sqrt(trajectories). Used for convergence analysis.

All parameter values used in V10 experiments are hypothetical simulation settings unless stated otherwise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.v10.quantum.simulator import FIXED_MATRIX, PARAM_GATES, PAULI, Circuit, gate_matrix, rotation_matrices, z_signs

_I2 = np.eye(2, dtype=np.complex128)
_PAULIS_1Q = [PAULI["X"], PAULI["Y"], PAULI["Z"]]
_PAULIS_2Q = [np.kron(a, b) for a in [_I2] + _PAULIS_1Q for b in [_I2] + _PAULIS_1Q][1:]


@dataclass
class NoiseModel:
    p1: float = 0.0
    p2: float = 0.0
    readout_e0: float = 0.0
    readout_e1: float = 0.0
    amp_damp: float = 0.0          # per-gate amplitude damping on the gate's wires (used when t1 is None)
    phase_damp: float = 0.0        # per-gate phase damping on the gate's wires (used when t2 is None)
    t1: float | None = None
    t2: float | None = None
    dur_1q: float = 0.0
    dur_2q: float = 0.0
    idle: bool = False
    calibration: str = "hypothetical (not hardware-calibrated)"

    def as_dict(self):
        return asdict(self)

    def relaxation(self, t: float) -> tuple[float, float]:
        """(gamma, lambda) of thermal relaxation for duration t."""
        if self.t1 is None or t <= 0:
            return 0.0, 0.0
        g = 1 - np.exp(-t / self.t1)
        t2 = self.t2 if self.t2 is not None else 2 * self.t1
        if t2 > 2 * self.t1 + 1e-12:
            raise ValueError("T2 must not exceed 2 T1")
        lam = 1 - np.exp(-t * (2 / t2 - 1 / self.t1))
        return float(g), float(max(lam, 0.0))


def amplitude_damping_kraus(g: float):
    return [np.array([[1, 0], [0, np.sqrt(1 - g)]], np.complex128), np.array([[0, np.sqrt(g)], [0, 0]], np.complex128)]


def phase_damping_kraus(lam: float):
    return [np.array([[1, 0], [0, np.sqrt(1 - lam)]], np.complex128), np.array([[0, 0], [0, np.sqrt(lam)]], np.complex128)]


def schedule_moments(circ: Circuit) -> list[list[int]]:
    level = [0] * circ.n
    moments: list[list[int]] = []
    for i, (_, wires, _) in enumerate(circ.ops):
        d = max(level[q] for q in wires)
        if d == len(moments):
            moments.append([])
        moments[d].append(i)
        for q in wires:
            level[q] = d + 1
    return moments


class DensityMatrixSimulator:
    def __init__(self, circ: Circuit, model: NoiseModel):
        self.c, self.m = circ, model
        self.n = circ.n

    # ------------------------------------------------------------------------------------- primitives
    def _apply(self, rho, wires, U):
        """rho -> U rho U^dag on `wires`; U (2^k, 2^k) or batched (B, 2^k, 2^k)."""
        n, B, k = self.n, rho.shape[0], len(wires)
        t = rho.reshape((B,) + (2,) * (2 * n))
        for side in ("ket", "bra"):
            axes = [1 + q for q in wires] if side == "ket" else [1 + n + q for q in wires]
            M = U if side == "ket" else np.conj(U)
            front = list(range(1, 1 + k))
            t = np.moveaxis(t, axes, front)
            sh = t.shape
            t = t.reshape(B, 2 ** k, -1)
            t = np.einsum("ij,bjr->bir", M, t) if M.ndim == 2 else np.einsum("bij,bjr->bir", M, t)
            t = np.moveaxis(t.reshape(sh), front, axes)
        return t.reshape(rho.shape)

    def _kraus(self, rho, wires, kraus):
        return sum(self._apply(rho, wires, K) for K in kraus)

    def _depol1(self, rho, q, p):
        if p <= 0:
            return rho
        return (1 - p) * rho + (p / 3) * sum(self._apply(rho, (q,), P) for P in _PAULIS_1Q)

    def _depol2(self, rho, wires, p):
        if p <= 0:
            return rho
        return (1 - p) * rho + (p / 15) * sum(self._apply(rho, wires, P) for P in _PAULIS_2Q)

    def _relax(self, rho, q, g, lam):
        if g > 0:
            rho = self._kraus(rho, (q,), amplitude_damping_kraus(g))
        if lam > 0:
            rho = self._kraus(rho, (q,), phase_damping_kraus(lam))
        return rho

    def _gate_unitary(self, op, X, w):
        name, wires, p = op
        if name in ("RX", "RY", "RZ"):
            return rotation_matrices(name, self.c.angle(p, X, w))
        if name == "RZZ":
            th = self.c.angle(p, X, w)
            d = np.exp(-0.5j * th[:, None] * np.array([1, -1, -1, 1])[None, :])
            U = np.zeros((len(th), 4, 4), np.complex128)
            U[:, np.arange(4), np.arange(4)] = d
            return U
        if name in FIXED_MATRIX:
            return FIXED_MATRIX[name]
        return gate_matrix(name)

    # ------------------------------------------------------------------------------------- evolution
    def run(self, X, w=None, rho0=None) -> np.ndarray:
        c, m = self.c, self.m
        X, w = c._prep(X, w)
        D = 2 ** self.n
        if rho0 is None:
            rho = np.zeros((len(X), D, D), np.complex128)
            rho[:, 0, 0] = 1.0
        else:
            rho = np.array(rho0, np.complex128)
        for moment in schedule_moments(c):
            mdur = 0.0
            busy = {}
            for i in moment:
                op = c.ops[i]
                name, wires, _ = op
                rho = self._apply(rho, wires, self._gate_unitary(op, X, w))
                if len(wires) == 1:
                    rho = self._depol1(rho, wires[0], m.p1)
                    dur = m.dur_1q
                else:
                    rho = self._depol2(rho, wires, m.p2)
                    dur = m.dur_2q
                mdur = max(mdur, dur)
                if m.t1 is not None:
                    g, lam = m.relaxation(dur)
                else:
                    g, lam = m.amp_damp, m.phase_damp
                for q in wires:
                    rho = self._relax(rho, q, g, lam)
                    busy[q] = dur
            if m.idle and m.t1 is not None:
                for q in range(self.n):
                    t_idle = mdur - busy.get(q, 0.0)
                    if t_idle > 0:
                        g, lam = m.relaxation(t_idle)
                        rho = self._relax(rho, q, g, lam)
        return rho

    def probabilities(self, X, w=None, readout: bool = True) -> np.ndarray:
        rho = self.run(X, w)
        p = np.real(np.einsum("bii->bi", rho))
        p = np.clip(p, 0, None)
        p = p / p.sum(1, keepdims=True)
        return apply_readout_error(p, self.n, self.m.readout_e0, self.m.readout_e1) if readout else p

    def expect(self, X, w, masks, readout: bool = True) -> np.ndarray:
        p = self.probabilities(X, w, readout)
        return np.stack([p @ z_signs(self.n, mk) for mk in masks], 1)


def apply_readout_error(p: np.ndarray, n: int, e0: float, e1: float) -> np.ndarray:
    if e0 == 0 and e1 == 0:
        return p
    M = np.array([[1 - e0, e1], [e0, 1 - e1]])  # rows: measured 0/1, columns: true 0/1
    t = p.reshape((p.shape[0],) + (2,) * n)
    for q in range(n):
        t = np.moveaxis(np.tensordot(t, M, axes=([q + 1], [1])), -1, q + 1)
    return t.reshape(p.shape)


def sample_expectations(p: np.ndarray, n: int, masks, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Z-string estimates from `shots` multinomial samples of each row's measured distribution."""
    signs = np.stack([z_signs(n, mk) for mk in masks], 1)
    out = np.zeros((len(p), len(masks)))
    for b in range(len(p)):
        counts = rng.multinomial(shots, p[b] / p[b].sum())
        out[b] = counts @ signs / shots
    return out


class TrajectorySimulator:
    """State-vector Monte-Carlo trajectories for the depolarizing (Pauli) channels."""

    def __init__(self, circ: Circuit, p1: float, p2: float):
        self.c, self.p1, self.p2 = circ, p1, p2

    def run(self, X, w, masks, trajectories: int, seed: int) -> np.ndarray:
        """Per-trajectory expectations, shape (trajectories, B, n_masks); trajectory t uses its own child seed."""
        c = self.c
        X, w = c._prep(X, w)
        children = np.random.SeedSequence(seed).spawn(trajectories)
        out = np.zeros((trajectories, len(X), len(masks)))
        for t, ss in enumerate(children):
            rng = np.random.default_rng(ss)
            psi = np.zeros((len(X), 2 ** c.n), np.complex128)
            psi[:, 0] = 1.0
            for op in c.ops:
                name, wires, p = op
                th = c.angle(p, X, w) if name in PARAM_GATES else None
                psi = c.apply_op(psi, op, th)
                if len(wires) == 1 and self.p1 > 0:
                    hit = rng.random(len(X)) < self.p1
                    kind = rng.integers(0, 3, len(X))
                    for k, P in enumerate(("X", "Y", "Z")):
                        sel = hit & (kind == k)
                        if sel.any():
                            psi[sel] = c.apply_pauli(psi[sel], P, wires[0])
                elif len(wires) == 2 and self.p2 > 0:
                    hit = rng.random(len(X)) < self.p2
                    kind = rng.integers(1, 16, len(X))
                    for k in range(1, 16):
                        sel = hit & (kind == k)
                        if sel.any():
                            a, b = divmod(k, 4)
                            for P, q in ((a, wires[0]), (b, wires[1])):
                                if P:
                                    psi[sel] = c.apply_pauli(psi[sel], "XYZ"[P - 1], q)
            out[t] = c.expect(psi, masks)
        return out


def noisy_fidelity_kernel(fmap, XA: np.ndarray, XB: np.ndarray, model: NoiseModel, symmetric: bool = False,
                          shots: int | None = None, rng: np.random.Generator | None = None, batch: int = 256) -> np.ndarray:
    """Noisy compute-uncompute kernel: P(0...0) after U(x_b)^dag U(x_a) with noise after every gate and readout error.
    With symmetric=True (XA is XB) every unordered pair is simulated once and mirrored; the diagonal is simulated too,
    because under noise it is below 1. Exact probabilities unless `shots` is given (binomial estimate per entry)."""
    circ = fmap.compute_uncompute_circuit()
    sim = DensityMatrixSimulator(circ, model)
    n = fmap.n
    if symmetric:
        idx = [(i, j) for i in range(len(XA)) for j in range(i, len(XA))]
    else:
        idx = [(i, j) for i in range(len(XA)) for j in range(len(XB))]
    vals = np.zeros(len(idx))
    for s in range(0, len(idx), batch):
        chunk = idx[s:s + batch]
        Z = np.array([np.concatenate([XA[i], XB[j]]) for i, j in chunk])
        p = sim.probabilities(Z, None, readout=True)
        vals[s:s + batch] = p[:, 0]
    if shots is not None:
        rng = rng or np.random.default_rng(0)
        vals = rng.binomial(shots, np.clip(vals, 0, 1)) / shots
    K = np.zeros((len(XA), len(XB)))
    for (i, j), v in zip(idx, vals):
        K[i, j] = v
        if symmetric:
            K[j, i] = v
    return K
