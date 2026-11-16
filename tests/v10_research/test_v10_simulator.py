"""V10 simulator: gate matrices, conventions, known states, dense-matrix equivalence, gradients, V9 equivalence."""
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True

from src.v10.quantum.simulator import Circuit, gate_matrix, z_signs  # noqa: E402

I2 = np.eye(2)


def kron_all(ms):
    out = np.array([[1.0 + 0j]])
    for m in ms:
        out = np.kron(out, m)
    return out


def dense_op(n, wires, U):
    """Independent reference: embed a 1- or 2-qubit matrix with qubit 0 as the most significant bit."""
    D = 2 ** n
    M = np.zeros((D, D), complex)
    k = len(wires)
    for col in range(D):
        bits = [(col >> (n - 1 - q)) & 1 for q in range(n)]
        sub = 0
        for q in wires:
            sub = 2 * sub + bits[q]
        for r in range(2 ** k):
            amp = U[r, sub]
            if amp == 0:
                continue
            nb = list(bits)
            for j, q in enumerate(wires):
                nb[q] = (r >> (k - 1 - j)) & 1
            M[sum(b << (n - 1 - q) for q, b in enumerate(nb)), col] += amp
    return M


def reference_state(circ, x, w):
    psi = np.zeros(2 ** circ.n, complex)
    psi[0] = 1
    X = np.atleast_2d(x)
    for name, wires, p in circ.ops:
        th = circ.angle(p, X, w)[0] if p is not None else None
        psi = dense_op(circ.n, wires, gate_matrix(name, th)) @ psi
    return psi


@pytest.mark.parametrize("name", ["RX", "RY", "RZ", "RZZ", "H", "X", "Y", "Z", "CNOT", "CZ"])
def test_gate_unitarity(name):
    for th in (0.0, 0.3, -1.7, np.pi):
        U = gate_matrix(name, th)
        assert np.allclose(U.conj().T @ U, np.eye(len(U)), atol=1e-12)


def test_rotation_conventions():
    th = 0.7
    assert np.allclose(gate_matrix("RZ", th), np.diag([np.exp(-0.35j), np.exp(0.35j)]))
    from scipy.linalg import expm
    for name, P in (("RX", np.array([[0, 1], [1, 0]])), ("RY", np.array([[0, -1j], [1j, 0]])), ("RZ", np.diag([1, -1]))):
        assert np.allclose(gate_matrix(name, th), expm(-0.5j * th * P))
    ZZ = np.diag([1, -1, -1, 1])
    assert np.allclose(gate_matrix("RZZ", th), expm(-0.5j * th * ZZ))


def test_known_states_and_qubit_order():
    c = Circuit(3, [("X", (0,), None)])
    psi = c.state(np.zeros((1, 0)))[0]
    assert np.isclose(abs(psi[4]), 1)                      # qubit 0 is the most significant bit
    bell = Circuit(2, [("H", (0,), None), ("CNOT", (0, 1), None)]).state(np.zeros((1, 0)))[0]
    assert np.allclose(bell, np.array([1, 0, 0, 1]) / np.sqrt(2))
    ghz = Circuit(3, [("H", (0,), None), ("CNOT", (0, 1), None), ("CNOT", (1, 2), None)]).state(np.zeros((1, 0)))[0]
    assert np.allclose(ghz, np.eye(8)[0] / np.sqrt(2) + np.eye(8)[7] / np.sqrt(2))
    # CNOT ordering: control 0 set -> target flips; control 1 unset -> nothing happens
    assert np.isclose(abs(Circuit(2, [("X", (0,), None), ("CNOT", (0, 1), None)]).state(np.zeros((1, 0)))[0][3]), 1)
    assert np.isclose(abs(Circuit(2, [("X", (0,), None), ("CNOT", (1, 0), None)]).state(np.zeros((1, 0)))[0][2]), 1)
    # RY(pi/2)|0> gives <Z> = 0, <Z> of RY(theta)|0> = cos(theta)
    c = Circuit(1, [("RY", (0,), ("x", 0, 1.0, 0.0))])
    th = np.array([[0.0], [0.4], [np.pi / 2], [np.pi]])
    assert np.allclose(c.forward(th)[:, 0], np.cos(th[:, 0]))


def random_circuit(seed=0):
    ops = [("H", (0,), None), ("RY", (1,), ("x", 0, 1.3, 0.2)), ("RX", (2,), ("w", 0)), ("CNOT", (0, 2), None),
           ("RZ", (0,), ("ax", 1, 1, 2)), ("RZZ", (1, 2), ("w", 3)), ("CZ", (2, 0), None), ("CNOT", (2, 1), None),
           ("RZ", (1,), ("pair", 0, 2, 1.0, 1.0, 1.0, 0.5)), ("RY", (2,), ("neg", ("w", 4))), ("RX", (0,), ("ax", 2, 5, None))]
    return Circuit(3, ops)


def test_state_matches_dense_reference_and_is_normalised():
    rng = np.random.default_rng(1)
    c = random_circuit()
    w = rng.normal(size=c.n_weights)
    X = rng.uniform(0, np.pi, (4, c.n_features))
    psi = c.state(X, w)
    assert np.allclose(np.linalg.norm(psi, axis=1), 1)
    for i in range(4):
        assert np.allclose(psi[i], reference_state(c, X[i], w), atol=1e-12)
    U = c.unitary(X[0], w)
    assert np.allclose(U.conj().T @ U, np.eye(8), atol=1e-12)
    assert np.allclose(c.inverse().unitary(X[0], w), U.conj().T, atol=1e-12)


def test_z_string_expectations_match_operators():
    rng = np.random.default_rng(2)
    c = random_circuit()
    w = rng.normal(size=c.n_weights)
    X = rng.uniform(0, np.pi, (3, c.n_features))
    psi = c.state(X, w)
    Z = np.diag([1, -1])
    for mask in [(0,), (2,), (0, 1), (0, 1, 2)]:
        O = kron_all([Z if q in mask else I2 for q in range(3)])
        ref = np.real(np.einsum("bi,ij,bj->b", psi.conj(), O, psi))
        assert np.allclose(c.expect(psi, [mask])[:, 0], ref)
        assert np.allclose(np.diag(O), z_signs(3, mask))


def test_adjoint_equals_parameter_shift_and_finite_differences():
    rng = np.random.default_rng(3)
    c = random_circuit()
    w = rng.normal(size=c.n_weights)
    X = rng.uniform(0, np.pi, (5, c.n_features))
    masks = [(0,), (1, 2), (0, 1, 2)]
    dL = rng.normal(size=(5, len(masks)))
    gw, gx = c.adjoint_grad(X, w, dL, masks)
    assert np.allclose(gw, c.parameter_shift_grad(X, w, dL, masks), atol=1e-10)
    f = lambda X_, w_: np.sum(dL * c.expect(c.state(X_, w_), masks))  # noqa: E731
    eps = 1e-6
    num_w = np.array([(f(X, w + eps * e) - f(X, w - eps * e)) / (2 * eps) for e in np.eye(len(w))])
    assert np.allclose(gw, num_w, atol=1e-6)
    num_x = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Xp, Xm = X.copy(), X.copy()
            Xp[i, j] += eps
            Xm[i, j] -= eps
            num_x[i, j] = (f(Xp, w) - f(Xm, w)) / (2 * eps)
    assert np.allclose(gx, num_x, atol=1e-6)


def test_v10_matches_v9_simulator_on_the_v9_circuit():
    from src.quantum.simulator import vqc_circuit
    from src.v10.quantum.circuits import build_v9_vqc
    rng = np.random.default_rng(4)
    for n, L in ((3, 1), (4, 2), (5, 3)):
        v9 = vqc_circuit(n, L, reupload=True)
        v10 = build_v9_vqc(n, L)
        w = rng.normal(size=v9.n_weights)
        X = rng.uniform(0, np.pi, (6, n))
        assert np.allclose(v9.state(X, w), v10.state(X, w), atol=1e-12)
        dL = rng.normal(size=(6, n))
        gw9, gx9 = v9.adjoint_grad(X, w, dL)
        gw10, gx10 = v10.adjoint_grad(X, w, dL, [(q,) for q in range(n)])
        assert np.allclose(gw9, gw10) and np.allclose(gx9, gx10)
