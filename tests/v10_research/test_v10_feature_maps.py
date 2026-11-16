"""V10 feature maps: diagonal form equals gate form, equals V9, ZZ identity, periodicity, phase decomposition."""
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True

from src.v10.quantum.feature_maps import ZFeatureMap, v9_phase_parts  # noqa: E402
from src.v10.quantum.kernels import fidelity_kernel  # noqa: E402
from src.v10.quantum.simulator import gate_matrix  # noqa: E402


@pytest.mark.parametrize("pair", ["v9", "product", "none", "v9_bilinear"])
@pytest.mark.parametrize("ent", ["linear", "ring", "full"])
def test_diagonal_form_equals_gate_form(pair, ent):
    rng = np.random.default_rng(0)
    fm = ZFeatureMap(4, reps=2, pair=pair, alpha=1.7, entanglement=ent, bandwidth=0.6)
    X = rng.uniform(0, np.pi, (5, 4))
    S1 = fm.states(X)
    S2 = fm.gate_circuit().state(X)
    F = np.abs(np.sum(np.conj(S1) * S2, 1)) ** 2
    assert np.allclose(F, 1, atol=1e-12)          # equal up to a global phase


def test_v9_preset_equals_v9_feature_map():
    from src.quantum.simulator import ZZFeatureMap
    rng = np.random.default_rng(1)
    for n, bw in ((4, 0.05), (6, 0.5), (8, 1.0)):
        X = rng.uniform(0, np.pi, (6, n))
        v9 = ZZFeatureMap(n, 2).states(bw * X)
        v10 = ZFeatureMap(n, 2, "v9", entanglement="linear", bandwidth=bw).states(X)
        assert np.allclose(fidelity_kernel(v9, v9), fidelity_kernel(v10, v10), atol=1e-12)
        F = np.abs(np.sum(np.conj(v9) * v10, 1)) ** 2
        assert np.allclose(F, 1, atol=1e-12)


def test_cnot_rz_cnot_is_zz_rotation():
    phi = 0.83
    cn = gate_matrix("CNOT")
    sandwich = cn @ np.kron(np.eye(2), gate_matrix("RZ", 2 * phi)) @ cn
    assert np.allclose(sandwich, np.diag(np.exp(-1j * phi * np.array([1, -1, -1, 1]))))


def test_pair_phase_period_pi_up_to_global_phase():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 0.4, (4, 3))
    base = ZFeatureMap(3, 2, (0.3, 0.0, 0.0)).states(X)
    shifted = ZFeatureMap(3, 2, (0.3 + 1.0 / np.pi, 0.0, 0.0)).states(X)   # phi -> phi + pi (c0 multiplies pi^2)
    assert np.allclose(np.abs(np.sum(np.conj(base) * shifted, 1)), 1, atol=1e-12)


def test_phase_decomposition_and_product_alpha():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, np.pi, (10, 4))
    fm = ZFeatureMap(4, 2, "v9", bandwidth=0.25)
    parts = v9_phase_parts(X, 0.25, fm.pairs)
    assert np.allclose(parts["total"], fm.pair_phases(X))
    pr = ZFeatureMap(4, 2, "product", alpha=2.5, bandwidth=0.25)
    U = 0.25 * X
    assert np.allclose(pr.pair_phases(X)[:, 0], 2.5 * U[:, 0] * U[:, 1])


def test_none_pair_gives_product_state():
    from src.v10.quantum.diagnostics import meyer_wallach
    rng = np.random.default_rng(4)
    S = ZFeatureMap(4, 2, "none", bandwidth=1.0).states(rng.uniform(0, np.pi, (5, 4)))
    assert np.allclose(meyer_wallach(S, 4), 0, atol=1e-12)
