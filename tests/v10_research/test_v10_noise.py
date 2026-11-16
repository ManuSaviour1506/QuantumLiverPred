"""V10 noise: zero-noise equivalence, channel closed forms, separate gate noise, readout, T1/T2, trajectories, shots."""
import sys

import numpy as np

sys.dont_write_bytecode = True

from src.v10.quantum.circuits import build_v10_vqc  # noqa: E402
from src.v10.quantum.noise import (DensityMatrixSimulator, NoiseModel, TrajectorySimulator, apply_readout_error,  # noqa: E402
                                   sample_expectations)
from src.v10.quantum.simulator import Circuit  # noqa: E402

E = np.zeros((1, 0))


def test_zero_noise_equals_pure_state():
    rng = np.random.default_rng(0)
    c = build_v10_vqc(3, 2)
    w = rng.normal(size=c.n_weights)
    X = rng.uniform(0, np.pi, (4, 3))
    rho = DensityMatrixSimulator(c, NoiseModel()).run(X, w)
    psi = c.state(X, w)
    assert np.allclose(rho, np.einsum("bi,bj->bij", psi, psi.conj()), atol=1e-12)


def test_channels_are_trace_preserving_hermitian_and_positive():
    rng = np.random.default_rng(1)
    c = build_v10_vqc(3, 2)
    m = NoiseModel(p1=0.03, p2=0.08, amp_damp=0.02, phase_damp=0.05)
    rho = DensityMatrixSimulator(c, m).run(rng.uniform(0, np.pi, (3, 3)), rng.normal(size=c.n_weights))
    for r in rho:
        assert np.isclose(np.trace(r).real, 1) and np.allclose(r, r.conj().T)
        assert np.linalg.eigvalsh(r).min() > -1e-12


def test_depolarizing_closed_forms_and_separation():
    p = 0.12
    one = Circuit(1, [("RZ", (0,), ("c", 0.0))])
    z = DensityMatrixSimulator(one, NoiseModel(p1=p)).expect(E, None, [(0,)])[0, 0]
    assert np.isclose(z, 1 - 4 * p / 3)
    two = Circuit(2, [("CNOT", (0, 1), None)])
    z2 = DensityMatrixSimulator(two, NoiseModel(p2=p)).expect(E, None, [(0,)])[0, 0]
    assert np.isclose(z2, 1 - 16 * p / 15)
    assert np.isclose(DensityMatrixSimulator(two, NoiseModel(p1=0.3)).expect(E, None, [(0,)])[0, 0], 1)   # p1 ignores 2q gates
    assert np.isclose(DensityMatrixSimulator(one, NoiseModel(p2=0.3)).expect(E, None, [(0,)])[0, 0], 1)   # p2 ignores 1q gates


def test_damping_and_relaxation_closed_forms():
    g, lam = 0.2, 0.3
    flip = Circuit(1, [("X", (0,), None)])
    p = DensityMatrixSimulator(flip, NoiseModel(amp_damp=g)).probabilities(E, None)[0]
    assert np.isclose(p[1], 1 - g)
    plus = Circuit(1, [("H", (0,), None)])
    rho = DensityMatrixSimulator(plus, NoiseModel(phase_damp=lam)).run(E)[0]
    assert np.isclose(2 * rho[0, 1].real, np.sqrt(1 - lam))
    t1, t2, t = 50.0, 70.0, 10.0
    m = NoiseModel(t1=t1, t2=t2, dur_1q=t)
    assert np.isclose(DensityMatrixSimulator(flip, m).probabilities(E, None)[0][1], np.exp(-t / t1))
    assert np.isclose(2 * abs(DensityMatrixSimulator(plus, m).run(E)[0][0, 1]), np.exp(-t / t2))


def test_idle_relaxation_applies_to_waiting_qubits():
    c = Circuit(2, [("X", (1,), None), ("RZ", (0,), ("c", 0.0)), ("RZ", (0,), ("c", 0.0))])
    m_off = NoiseModel(t1=20.0, t2=20.0, dur_1q=5.0, idle=False)
    m_on = NoiseModel(t1=20.0, t2=20.0, dur_1q=5.0, idle=True)
    p_off = DensityMatrixSimulator(c, m_off).probabilities(E, None)[0]
    p_on = DensityMatrixSimulator(c, m_on).probabilities(E, None)[0]
    # qubit 1 (excited) is busy in moment 1 and idle in moment 2: extra decay exp(-5/20) only with idle=True
    assert np.isclose(p_off[1] + p_off[3], np.exp(-5 / 20))
    assert np.isclose(p_on[1] + p_on[3], np.exp(-10 / 20))


def test_readout_error_closed_form():
    rng = np.random.default_rng(2)
    p = rng.dirichlet(np.ones(4), size=5)
    e0, e1 = 0.03, 0.08
    pm = apply_readout_error(p, 2, e0, e1)
    z = p @ np.array([1, 1, -1, -1])
    zm = pm @ np.array([1, 1, -1, -1])
    assert np.allclose(zm, (1 - e0 - e1) * z + (e1 - e0)) and np.allclose(pm.sum(1), 1)


def test_trajectories_converge_to_density_matrix_and_are_reproducible():
    rng = np.random.default_rng(3)
    c = build_v10_vqc(3, 2)
    w = rng.normal(size=c.n_weights)
    X = rng.uniform(0, np.pi, (2, 3))
    masks = [(0,), (1, 2)]
    exact = DensityMatrixSimulator(c, NoiseModel(p1=0.02, p2=0.06)).expect(X, w, masks)
    traj = TrajectorySimulator(c, 0.02, 0.06).run(X, w, masks, trajectories=3000, seed=11)
    mean, se = traj.mean(0), traj.std(0) / np.sqrt(len(traj))
    assert (np.abs(mean - exact) < 4 * se + 1e-9).all()
    again = TrajectorySimulator(c, 0.02, 0.06).run(X, w, masks, trajectories=50, seed=11)
    assert np.allclose(again, traj[:50])
    other = TrajectorySimulator(c, 0.02, 0.06).run(X, w, masks, trajectories=50, seed=12)
    assert not np.allclose(other, again)


def test_shot_sampling_standard_error():
    rng = np.random.default_rng(4)
    p = np.array([[0.7, 0.3]])
    est = np.array([sample_expectations(p, 1, [(0,)], 1024, rng)[0, 0] for _ in range(400)])
    z = 0.4
    assert abs(est.mean() - z) < 4 * np.sqrt((1 - z ** 2) / 1024) / np.sqrt(400)
    assert abs(est.std() - np.sqrt((1 - z ** 2) / 1024)) < 0.006
