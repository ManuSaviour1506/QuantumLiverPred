"""V10 kernels: validity, compute-uncompute equivalence, alignment and spectrum properties, finite shots, SVM affine maps."""
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True

from src.v10.quantum import kernels as KK  # noqa: E402
from src.v10.quantum.feature_maps import ZFeatureMap  # noqa: E402
from src.v10.quantum.noise import NoiseModel, noisy_fidelity_kernel  # noqa: E402


def test_fidelity_kernel_is_valid():
    rng = np.random.default_rng(0)
    S = ZFeatureMap(4, 2, "v9", bandwidth=0.5).states(rng.uniform(0, np.pi, (15, 4)))
    K = KK.fidelity_kernel(S, S)
    assert np.allclose(K, K.T) and np.allclose(np.diag(K), 1)
    assert (K >= -1e-12).all() and (K <= 1 + 1e-12).all() and np.linalg.eigvalsh(K).min() > -1e-10


def test_compute_uncompute_equals_fidelity_without_noise():
    rng = np.random.default_rng(1)
    fm = ZFeatureMap(3, 2, "v9", bandwidth=0.7)
    XA, XB = rng.uniform(0, np.pi, (4, 3)), rng.uniform(0, np.pi, (3, 3))
    K = KK.fidelity_kernel(fm.states(XA), fm.states(XB))
    Kc = noisy_fidelity_kernel(fm, XA, XB, NoiseModel())
    assert np.allclose(K, Kc, atol=1e-12)


def test_alignment_properties():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 40)
    t = np.where(y == 1, 1.0, -1.0)
    assert np.isclose(KK.alignment(np.outer(t, t), y, True), 1.0)
    K = np.exp(-rng.uniform(0, 1, (40, 40)))
    K = 0.5 * (K + K.T)
    shifted = K + 5.0
    assert np.isclose(KK.alignment(K, y, True), KK.alignment(shifted, y, True))     # centring removes c 11^T
    assert not np.isclose(KK.alignment(K, y, False), KK.alignment(shifted, y, False))


def test_spectrum_extremes():
    v = np.random.default_rng(3).normal(size=20)
    r1 = KK.spectrum(np.outer(v, v))
    assert r1["numerical_rank"] == 1 and np.isclose(r1["effective_rank_entropy"], 1) and np.isclose(r1["top_eigenvalue_share"], 1)
    rI = KK.spectrum(np.eye(20))
    assert np.isclose(rI["effective_rank_entropy"], 20) and np.isclose(rI["participation_ratio"], 20)


def test_bloch_vectors_and_projected_kernel():
    rng = np.random.default_rng(4)
    n = 3
    S = ZFeatureMap(n, 2, "v9", bandwidth=0.8).states(rng.uniform(0, np.pi, (6, n)))
    R = KK.bloch_vectors(S, n)
    X_ = np.array([[0, 1], [1, 0]])
    Y_ = np.array([[0, -1j], [1j, 0]])
    Z_ = np.diag([1, -1])
    for q in range(n):
        for k, P in enumerate((X_, Y_, Z_)):
            O = np.array([[1.0]])
            for j in range(n):
                O = np.kron(O, P if j == q else np.eye(2))
            ref = np.real(np.einsum("bi,ij,bj->b", S.conj(), O, S))
            assert np.allclose(R[:, q, k], ref)
    Kp = KK.projected_kernel(R, R, 1.3)
    assert np.allclose(np.diag(Kp), 1) and np.linalg.eigvalsh(Kp).min() > -1e-10


def test_finite_shot_kernel():
    rng = np.random.default_rng(5)
    K = np.full((60, 60), 0.5)
    np.fill_diagonal(K, 1.0)
    est = np.stack([KK.sample_kernel(K, 1024, rng) for _ in range(200)])
    assert np.allclose(est, np.transpose(est, (0, 2, 1))) and np.allclose(est[:, np.arange(60), np.arange(60)], 1)
    sd = est[:, 0, 1].std()
    assert abs(sd - 0.015625) < 0.004
    assert np.isclose(KK.bernoulli_se(0.5, 1024), 0.015625)
    assert np.isclose(KK.bernoulli_se(0.993, 1024), np.sqrt(0.993 * 0.007 / 1024))


def test_svm_affine_kernel_maps():
    """Offset c 11^T: no effect on a kernel SVM trained and tested with it. Scale lam: equivalent to C -> lam C."""
    from sklearn.svm import SVC
    rng = np.random.default_rng(6)
    S = ZFeatureMap(3, 2, "v9", bandwidth=0.9).states(rng.uniform(0, np.pi, (60, 3)))
    K = KK.fidelity_kernel(S, S)
    y = (rng.random(60) < 0.5).astype(int)
    tr, te = np.arange(40), np.arange(40, 60)
    base = SVC(kernel="precomputed", C=2.0, tol=1e-10).fit(K[np.ix_(tr, tr)], y[tr]).decision_function(K[np.ix_(te, tr)])
    c = 0.37
    off = SVC(kernel="precomputed", C=2.0, tol=1e-10).fit(K[np.ix_(tr, tr)] + c, y[tr]).decision_function(K[np.ix_(te, tr)] + c)
    assert np.allclose(base, off, atol=1e-6)
    lam = 0.4
    sc = SVC(kernel="precomputed", C=2.0, tol=1e-10).fit(lam * K[np.ix_(tr, tr)], y[tr]).decision_function(lam * K[np.ix_(te, tr)])
    eq = SVC(kernel="precomputed", C=2.0 * lam, tol=1e-10).fit(K[np.ix_(tr, tr)], y[tr]).decision_function(K[np.ix_(te, tr)])
    assert np.allclose(sc, eq, atol=1e-5)
    # clean-trained model scored on a noisy test kernel lam K + c: scores are an increasing affine map of the clean scores
    m = SVC(kernel="precomputed", C=2.0, tol=1e-10).fit(K[np.ix_(tr, tr)], y[tr])
    s_noisy = m.decision_function(lam * K[np.ix_(te, tr)] + c)
    assert np.allclose(s_noisy, lam * (base - m.intercept_[0]) + m.intercept_[0], atol=1e-6)


def test_kernel_concentration_grows_as_bandwidth_shrinks():
    rng = np.random.default_rng(7)
    X = rng.uniform(0, np.pi, (30, 4))
    means = [KK.offdiag_stats(KK.fidelity_kernel(*(2 * [ZFeatureMap(4, 2, "v9", bandwidth=bw).states(X)])))["mean"]
             for bw in (0.005, 0.01, 0.02, 0.05)]
    assert all(a > b for a, b in zip(means, means[1:]))      # smaller bandwidth -> entries closer to 1
    assert means[0] > 0.99
