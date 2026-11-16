"""Variational-circuit builders with explicit parameter roles, and Heisenberg-picture analysis of Z readouts.

build_v9_vqc        the exact V9 topology (src/quantum/simulator.py::vqc_circuit plus V9's trainable map), weight layout
                    [circuit weights (2 n L) | scale a (n) | offset o (n)] so trained V9 weights load unchanged.
build_v10_vqc       the V10 ansatz. Per layer: data RY (re-uploaded, optional trainable scale shared across layers),
                    trainable RY and RZ, entangler. Then a final trainable RY layer before Z measurement, so no
                    trainable gate sits immediately before a measurement it commutes with. No per-qubit offset
                    parameter is used: it would duplicate the RY weight that follows the data rotation.
propagate_z_masks   Heisenberg propagation of Pauli-Z strings backwards through CNOT / CZ gates.
effective_readouts  the Z strings a measured observable corresponds to before the trailing entangler.
dead_by_commutation trainable weights that provably cannot affect any measured Z string, by commutation with the
                    back-propagated observables (structural, independent of inputs and weight values).
"""
from __future__ import annotations

from src.v10.quantum.simulator import PARAM_GATES, Circuit, _feature_indices, _weight_indices


def entangler_pairs(n: int, kind: str) -> list[tuple[int, int]]:
    if kind == "none" or n < 2:
        return []
    if kind == "linear":
        return [(q, q + 1) for q in range(n - 1)]
    if kind == "ring":
        if n == 2:
            return [(0, 1)]  # V9's ring would apply CNOT(0,1) then CNOT(1,0); V10 avoids the duplicated pair
        return [(q, (q + 1) % n) for q in range(n)]
    if kind == "v9_ring":
        return [(q, (q + 1) % n) for q in range(n)]
    if kind == "full":
        return [(a, b) for a in range(n) for b in range(a + 1, n)]
    if kind == "brick":
        # even pairs then odd pairs: two-qubit depth 2 per layer, so light cones grow by about 2 qubits per layer
        return [(q, q + 1) for q in range(0, n - 1, 2)] + [(q, q + 1) for q in range(1, n - 1, 2)]
    raise ValueError(kind)


def build_v9_vqc(n: int, layers: int, trainable_map: bool = False) -> Circuit:
    """V9 VQC: per layer RY(data) [re-uploaded], RY(w) RZ(w) on every qubit, CNOT ring 0->1->...->n-1->0."""
    ops, roles, k = [], [], 0
    a0, o0 = 2 * n * layers, 2 * n * layers + n
    for layer in range(layers):
        for q in range(n):
            ops.append(("RY", (q,), ("ax", q, a0 + q, o0 + q) if trainable_map else ("x", q, 1.0, 0.0)))
        for q in range(n):
            ops.append(("RY", (q,), ("w", k)))
            ops.append(("RZ", (q,), ("w", k + 1)))
            roles += [f"L{layer + 1}.RY.q{q}", f"L{layer + 1}.RZ.q{q}"]
            k += 2
        ops += [("CNOT", (c, t), None) for c, t in entangler_pairs(n, "v9_ring")]
    if trainable_map:
        roles += [f"map.scale.q{q}" for q in range(n)] + [f"map.offset.q{q}" for q in range(n)]
    c = Circuit(n, ops, n, {"design": "v9", "layers": layers, "trainable_map": trainable_map, "roles": roles})
    return c


def build_v10_vqc(n: int, layers: int, entangler: str = "ring", trainable_scale: bool = False,
                  rotations: tuple = ("RY", "RZ"), final_rotation: str | None = "RY", reupload: bool = True) -> Circuit:
    ops, roles, k = [], [], 0
    scale0 = None
    if trainable_scale:
        scale0 = 0
        roles += [f"map.scale.q{q}" for q in range(n)]
        k = n
    for layer in range(layers):
        if layer == 0 or reupload:
            for q in range(n):
                ops.append(("RY", (q,), ("ax", q, scale0 + q, None) if trainable_scale else ("x", q, 1.0, 0.0)))
        for q in range(n):
            for r in rotations:
                ops.append((r, (q,), ("w", k)))
                roles.append(f"L{layer + 1}.{r}.q{q}")
                k += 1
        ops += [("CNOT", (c, t), None) for c, t in entangler_pairs(n, entangler)]
    if final_rotation:
        for q in range(n):
            ops.append((final_rotation, (q,), ("w", k)))
            roles.append(f"final.{final_rotation}.q{q}")
            k += 1
    meta = {"design": "v10", "layers": layers, "entangler": entangler, "trainable_scale": trainable_scale,
            "rotations": list(rotations), "final_rotation": final_rotation, "reupload": reupload, "roles": roles}
    return Circuit(n, ops, n, meta)


# --------------------------------------------------------------------------------------- Heisenberg analysis
def propagate_z_masks(masks, gates) -> list[tuple[int, ...]]:
    """Conjugate Pauli-Z strings by a list of CNOT / CZ gates (applied in the given order), i.e. return the strings P'
    with  G_1^dag ... G_m^dag  P  G_m ... G_1 = P'. CNOT(c, t): Z_t -> Z_c Z_t, Z_c -> Z_c. CZ commutes with Z strings."""
    out = []
    for m in masks:
        s = set(m)
        for name, wires in reversed(gates):
            if name == "CNOT":
                c, t = wires
                if t in s:
                    s ^= {c}
            elif name == "CZ":
                pass
            else:
                raise ValueError(f"only CNOT / CZ can be propagated, got {name}")
        out.append(tuple(sorted(s)))
    return out


def trailing_entangler(circ: Circuit) -> tuple[list, int]:
    """CNOT / CZ gates after the last single-qubit gate, and the index where that trailing block starts."""
    i = len(circ.ops)
    while i > 0 and circ.ops[i - 1][0] in ("CNOT", "CZ"):
        i -= 1
    return [(name, wires) for name, wires, _ in circ.ops[i:]], i


def effective_readouts(circ: Circuit, masks) -> list[tuple[int, ...]]:
    gates, _ = trailing_entangler(circ)
    return propagate_z_masks(masks, gates)


def dead_by_commutation(circ: Circuit, masks) -> dict:
    """Weights whose gate commutes with every measured observable after Heisenberg propagation.

    Walk backwards from the trailing entangler. On each qubit, trailing single-qubit gates are passed while they
    commute with all back-propagated strings: RZ always commutes with a Z string; any gate commutes on a qubit that no
    string touches. The first non-commuting gate (or any two-qubit gate) on a qubit stops the walk on that qubit.
    Returns the weight indices of the commuting trainable gates and the data features whose gates commute."""
    eff = effective_readouts(circ, masks)
    touched = set(q for m in eff for q in m)
    _, start = trailing_entangler(circ)
    open_q = set(range(circ.n))
    dead_ops = []
    for i in range(start - 1, -1, -1):
        name, wires, p = circ.ops[i]
        if len(wires) == 2:
            open_q -= set(wires)
        else:
            q = wires[0]
            if q not in open_q:
                continue
            commutes = (name in ("RZ", "Z")) or (q not in touched)
            if commutes:
                dead_ops.append(i)
            else:
                open_q.discard(q)
        if not open_q:
            break
    # a weight is dead only if EVERY gate that uses it is dead (weights such as a shared feature-map scale appear in
    # several layers; one commuting occurrence does not make them dead)
    uses: dict[int, list[int]] = {}
    for i, (name, wires, p) in enumerate(circ.ops):
        if name in PARAM_GATES:
            for k in _weight_indices(p):
                uses.setdefault(k, []).append(i)
    dset = set(dead_ops)
    dead_w = [k for k, ops in uses.items() if all(i in dset for i in ops)]
    data_dead = sorted({f for i in dead_ops for f in _feature_indices(circ.ops[i][2])})
    return {"effective_readouts": eff, "dead_weights": sorted(dead_w), "dead_op_indices": sorted(dead_ops),
            "data_features_with_a_commuting_last_occurrence": data_dead}


def gate_counts(circ: Circuit) -> dict:
    one = sum(1 for op in circ.ops if len(op[1]) == 1)
    two = sum(1 for op in circ.ops if len(op[1]) == 2)
    level = [0] * circ.n
    level2 = [0] * circ.n
    for name, wires, _ in circ.ops:
        d = max(level[q] for q in wires) + 1
        for q in wires:
            level[q] = d
        if len(wires) == 2:
            d2 = max(level2[q] for q in wires) + 1
            for q in wires:
                level2[q] = d2
    return {"qubits": circ.n, "declared_weights": circ.n_weights, "one_qubit_gates": one, "two_qubit_gates": two,
            "depth": max(level), "two_qubit_depth": max(level2), "state_vector_bytes": 16 * 2 ** circ.n,
            "density_matrix_bytes": 16 * 4 ** circ.n}
