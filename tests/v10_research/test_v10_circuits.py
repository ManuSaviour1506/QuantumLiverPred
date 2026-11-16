"""V10 circuits: V9 topology reproduction, Heisenberg readouts, dead and redundant parameters, V10 ansatz accounting."""
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True

from src.v10.quantum.circuits import (build_v9_vqc, build_v10_vqc, dead_by_commutation, effective_readouts,  # noqa: E402
                                      entangler_pairs, gate_counts, propagate_z_masks)
from src.v10.quantum.diagnostics import jacobian_rank, parameter_table  # noqa: E402
from src.v10.quantum.simulator import Circuit  # noqa: E402


def test_v9_trainable_map_matches_v9_model():
    from src.quantum.v9 import VQCv9
    rng = np.random.default_rng(0)
    n, L = 4, 2
    m = VQCv9(n, L, trainable_map=True, readout="allz")
    m.w = rng.normal(size=m.w.size)
    m.a, m.o = rng.uniform(0.5, 1.5, n), rng.uniform(-0.3, 0.3, n)
    c = build_v9_vqc(n, L, trainable_map=True)
    wfull = np.concatenate([m.w, m.a, m.o])
    X = rng.uniform(0, np.pi, (5, n))
    assert np.allclose(c.forward(X, wfull), m.expect(X))


@pytest.mark.parametrize("n", [3, 4, 5, 6])
@pytest.mark.parametrize("kind", ["v9_ring", "linear"])
def test_heisenberg_readouts_match_numerics(n, kind):
    pairs = entangler_pairs(n, kind)
    ent = Circuit(n, [("CNOT", p, None) for p in pairs])
    rng = np.random.default_rng(n)
    psi = rng.normal(size=(3, 2 ** n)) + 1j * rng.normal(size=(3, 2 ** n))
    psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    after = ent.state(np.zeros((3, 0)), psi0=psi)
    masks = [(q,) for q in range(n)] + [(0, 1)]
    eff = propagate_z_masks(masks, [("CNOT", p) for p in pairs])
    assert np.allclose(ent.expect(after, masks), ent.expect(psi, eff), atol=1e-12)
    if kind == "v9_ring":
        assert eff[0] == tuple(range(1, n))                      # <Z_0>  -> Z_1 ... Z_{n-1}
        for q in range(1, n):
            assert eff[q] == tuple(range(0, q + 1))              # <Z_q>  -> Z_0 ... Z_q


@pytest.mark.parametrize("n,L", [(3, 1), (4, 2), (5, 2)])
def test_v9_dead_parameters_are_structural_and_proved(n, L):
    c = build_v9_vqc(n, L)
    allz = [(q,) for q in range(n)]
    last_rz = [2 * n * (L - 1) + 2 * q + 1 for q in range(n)]
    proof = dead_by_commutation(c, allz)
    assert proof["dead_weights"] == last_rz
    tab = parameter_table(c, allz, n_draws=12, batch=6, seed=1)
    assert tab["zero_all_draws"] == last_rz
    z0 = dead_by_commutation(c, [(0,)])
    assert z0["dead_weights"] == sorted(last_rz + [2 * n * (L - 1)])   # plus the last-layer RY on qubit 0
    assert parameter_table(c, [(0,)], n_draws=12, batch=6, seed=2)["zero_all_draws"] == z0["dead_weights"]


def test_shared_map_weights_are_not_reported_dead():
    """With <Z_0> readout the last-layer data gate on qubit 0 commutes, but its scale and offset are shared with earlier
    layers, so they must not be reported dead (regression test for a bug found during Stage 2)."""
    n, L = 4, 3
    c = build_v9_vqc(n, L, trainable_map=True)
    proof = dead_by_commutation(c, [(0,)])
    assert proof["dead_weights"] == sorted([2 * n * (L - 1)] + [2 * n * (L - 1) + 2 * q + 1 for q in range(n)])
    assert parameter_table(c, [(0,)], n_draws=10, batch=6, seed=7)["zero_all_draws"] == proof["dead_weights"]
    r = jacobian_rank(c, [(0,)], n_inputs=40, n_points=2, seed=8)
    assert r["declared"] - len(proof["dead_weights"]) - r["rank_max"] == n      # the n offsets duplicate RY weights


def test_v9_one_layer_z0_closed_form():
    n = 5
    c = build_v9_vqc(n, 1)
    rng = np.random.default_rng(3)
    w = rng.normal(size=c.n_weights)
    X = rng.uniform(0, np.pi, (7, n))
    z0 = c.forward(X, w, [(0,)])[:, 0]
    assert np.allclose(z0, np.prod(np.cos(X[:, 1:] + w[2:2 * n:2]), axis=1))   # RY weight of qubit q is w[2q]
    X2 = X.copy()
    X2[:, 0] += 0.9
    assert np.allclose(c.forward(X2, w, [(0,)])[:, 0], z0)       # independent of x_0


def test_v9_trainable_map_offsets_are_redundant():
    n, L = 3, 2
    c = build_v9_vqc(n, L, trainable_map=True)
    allz = [(q,) for q in range(n)]
    r = jacobian_rank(c, allz, n_inputs=40, n_points=2, seed=4)
    assert r["declared"] == 2 * n * L + 2 * n
    assert r["rank_max"] == r["declared"] - 2 * n             # n dead RZ + n offset directions


@pytest.mark.parametrize("n,L,scale,ent", [(3, 1, False, "ring"), (3, 2, True, "ring"), (4, 2, False, "ring"), (4, 3, True, "ring"),
                                           (4, 2, True, "brick"), (5, 2, False, "linear")])
def test_v10_ansatz_has_no_dead_or_redundant_weights(n, L, scale, ent):
    c = build_v10_vqc(n, L, entangler=ent, trainable_scale=scale)
    allz = [(q,) for q in range(n)]
    assert dead_by_commutation(c, allz)["dead_weights"] == []
    tab = parameter_table(c, allz, n_draws=10, batch=6, seed=5)
    assert tab["zero_all_draws"] == []
    r = jacobian_rank(c, allz, n_inputs=48, n_points=2, seed=6)
    assert r["rank_max"] == r["declared"]


def test_brick_entangler_has_two_qubit_depth_two_per_layer():
    c = build_v10_vqc(8, 3, entangler="brick")
    assert gate_counts(c)["two_qubit_depth"] == 6
    assert gate_counts(build_v10_vqc(8, 3, entangler="ring"))["two_qubit_depth"] == 24


def test_gate_counts():
    c = build_v9_vqc(8, 2)
    g = gate_counts(c)
    assert g["two_qubit_gates"] == 16 and g["one_qubit_gates"] == 48 and g["declared_weights"] == 32
    assert g["state_vector_bytes"] == 16 * 256
    assert effective_readouts(c, [(0,)]) == [tuple(range(1, 8))]
