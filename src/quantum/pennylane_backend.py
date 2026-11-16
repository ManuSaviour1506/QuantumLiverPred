"""Optional PennyLane implementation of the V8 variational circuit (for cross-checking and for running the same trained
weights through PennyLane devices, e.g. `default.qubit` or a hardware plugin).

Not required by V8: training and inference use the NumPy simulator (src/quantum/simulator.py). PennyLane could not be
installed in the build environment, so tests/test_pennylane_equivalence.py is skipped there; with
`pip install pennylane` it checks that both implementations give the same <Z_q> values.
Wire / bit order: PennyLane wire 0 is the most significant bit, as in the NumPy simulator.
"""
from __future__ import annotations

import numpy as np


def available() -> bool:
    try:
        import pennylane  # noqa: F401
        return True
    except ImportError:
        return False


def vqc_expectations(X: np.ndarray, w: np.ndarray, n_qubits: int, n_layers: int, device: str = "default.qubit") -> np.ndarray:
    """<Z_q> for every input row, same circuit as src.quantum.simulator.vqc_circuit (re-uploading)."""
    import pennylane as qml
    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x, w):
        k = 0
        for _ in range(n_layers):
            for q in range(n_qubits):
                qml.RY(x[q], wires=q)
            for q in range(n_qubits):
                qml.RY(w[k], wires=q)
                qml.RZ(w[k + 1], wires=q)
                k += 2
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return np.array([np.asarray(circuit(x, w), float) for x in np.asarray(X, float)])
