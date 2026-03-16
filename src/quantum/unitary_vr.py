"""
This module constructs the quantum circuit U_{v_r} that, for each computational
basis index j, prepares a single target qubit whose |0⟩-amplitude encodes the
Chebyshev term (v_r)_j.

The construction supports:
    • r = 0  : (v_0)_j = 1/2 (constant case),
    • r ≥ 1  : (v_r)_j = T_r(x_j), where
          x_j = 2j / N − 1,  N = 2^n.

Registers used:
    • j      : n-qubit index register (prepared in uniform superposition),
    • th     : (p+2)-qubit fixed-point angle register,
    • tgt    : 1-qubit rotation target.

Core idea:
    The angle θ(j) = arccos(2j/N − 1) is computed entirely classically at circuit
    construction time (via NumPy). Its quantized fixed-point bitstring is then
    embedded into the quantum circuit using a reversible XOR-based lookup unitary:

        |j⟩|t⟩ → |j⟩|t XOR b(j)⟩,

    where b(j) is the fixed-point encoding of θ(j). No trigonometric computation
    is performed on the quantum device.

Circuit steps (r ≥ 1):
    1. Prepare a uniform superposition over j.
    2. Coherently load θ(j) into th via an XOR-loader unitary.
    3. Apply a fixed-point controlled-rotation ladder implementing RX(−r·θ̂(j))
       on the target qubit, using weights 2^1, 2^0, 2^−1, …, 2^−p.
    4. Optionally uncompute th to restore it to |0…0⟩.

This version removes any auxiliary n-qubit workspace register (e.g., for reversible
coordinate mapping), since θ(j) is computed classically and directly loaded from j. 
To improve efficiency during parameter sweeps, the module caches the θ(j) lookup 
table and its associated loader unitary for each pair (n, p), allowing repeated 
circuit constructions with different r values to reuse the same classical 
preprocessing.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from typing import Dict, List, Tuple

# Cache for theta lookup tables / loader gates keyed by (n, p).
# For fixed (n, p), the theta table and its corresponding XOR-loader do not depend on r,
# so we can reuse them across multiple circuit constructions when sweeping r.
_THETA_LOADER_CACHE: Dict[Tuple[int, int], Tuple[List[List[int]], UnitaryGate]] = {}


def get_theta_loader_gate(n: int, p: int) -> UnitaryGate:
    """ Retrieve or construct the cached theta loader unitary gate.

        This routine returns a UnitaryGate that coherently loads the
        fixed-point representation of θ(j) = arccos(2j/2^n − 1) into the
        angle register using a reversible XOR lookup operation

            |j⟩|t⟩ → |j⟩|t XOR b(j)⟩.

        The loader gate acts jointly on the index register j and the
        (p+2)-qubit angle register. Since the lookup table depends only on
        (n, p), the resulting gate is cached and reused across repeated
        circuit constructions.

        Args:
            n: Number of qubits in the index register.
            p: Number of fractional bits used in the fixed-point expansion.

        Returns:
            A Qiskit UnitaryGate implementing the reversible θ(j) lookup
            operation on the combined (j, th) registers.

        Raises:
            None.
    """
    key = (n, p)
    cached = _THETA_LOADER_CACHE.get(key)
    if cached is not None:
        return cached[1]

    N = 2**n
    theta_bits_by_j: List[List[int]] = []
    for jval in range(N):
        xj = 2.0 * jval / N - 1.0
        theta = float(np.arccos(np.clip(xj, -1.0, 1.0)))
        theta_bits_by_j.append(theta_to_fixed_bits(theta, p))

    Uload = build_xor_loader_unitary(theta_bits_by_j)
    load_gate = UnitaryGate(Uload, label=f"U_load_theta(n={n},p={p})")
    _THETA_LOADER_CACHE[key] = (theta_bits_by_j, load_gate)
    return load_gate


def clear_theta_loader_cache() -> None:
    """ Clear the cached theta lookup tables and loader gates.

        This routine removes all stored entries in the internal cache used
        for θ(j) lookup tables and their corresponding XOR-loader unitaries.
        Clearing the cache forces the tables to be recomputed the next time
        a loader gate is requested.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
    """
    _THETA_LOADER_CACHE.clear()


def theta_to_fixed_bits(theta: float, p: int):
    """ Convert an angle into a fixed-point binary representation.

        This routine converts a real-valued angle θ into a fixed-point binary
        expansion used by the controlled rotation ladder. The expansion uses
        weights

            2^1, 2^0, 2^-1, … , 2^-p

        producing a total of (p+2) bits. These bits correspond directly to
        controlled rotations that reconstruct the quantized angle through a
        weighted sum of powers of two.

        Args:
            theta: Real-valued angle to encode.
            p: Number of fractional bits used in the fixed-point expansion.

        Returns:
            A list of length (p+2) containing binary digits representing the
            fixed-point decomposition of θ.

        Raises:
            None.
    """
    bits = []
    rem = float(theta)

    # integer bits: 2^1, 2^0
    for k in [1, 0]:
        w = 2.0**k
        b = int(rem >= w)
        bits.append(b)
        rem -= b * w

    # fractional bits: 2^-1 ... 2^-p
    for k in range(-1, -p - 1, -1):
        w = 2.0**k
        b = int(rem >= w)
        bits.append(b)
        rem -= b * w

    return bits


def build_xor_loader_unitary(bitstrings_by_j):
    """ Construct a reversible XOR loader unitary for angle bitstrings.

        This routine builds a permutation unitary implementing the mapping

            |j⟩|t⟩ → |j⟩|t XOR b(j)⟩

        where b(j) is the precomputed fixed-point bitstring encoding the
        quantized value θ(j). The unitary acts jointly on the index register
        j and the angle register th and performs a coherent lookup of the
        angle representation.

        The returned matrix is suitable for wrapping as a Qiskit UnitaryGate.

        Args:
            bitstrings_by_j: List of bitstrings where entry j contains the
                fixed-point representation of θ(j). All bitstrings must have
                the same length m = p+2, and the number of entries must be
                N = 2^n for some integer n.

        Returns:
            A dense unitary matrix U implementing the reversible XOR lookup on
            (j, th), with dimension 2^(n+m).

        Raises:
            None.
    """
    N = len(bitstrings_by_j)
    n = int(np.log2(N))
    m = len(bitstrings_by_j[0])

    dim = 2 ** (n + m)
    U = np.zeros((dim, dim), dtype=complex)

    for j in range(N):
        b = bitstrings_by_j[j]
        # Interpret th bits in little-endian order (bit 0 is least significant).
        b_int = sum((bit << q) for q, bit in enumerate(b))

        for t in range(2**m):
            t2 = t ^ b_int
            in_idx = j + N * t
            out_idx = j + N * t2
            U[out_idx, in_idx] = 1.0

    return U


def U_vr_circuit_no_x(
    n: int,
    p: int,
    r: int,
    qiskit_matches_paper_rx: bool = True,
    uncompute_theta: bool = True,
    *,
    j,
    th,
    tgt,
    qc: QuantumCircuit,
) -> QuantumCircuit:
    """ Construct the quantum circuit implementing U_{v_r} without workspace registers.

        This routine builds a quantum circuit that prepares, for each basis
        state |j⟩ of an n-qubit index register, a target qubit whose |0⟩
        amplitude encodes the Chebyshev polynomial value

            (v_r)_j = T_r(x_j),   where  x_j = 2j / N − 1  and  N = 2^n.

        The circuit uses three logical registers:

            j      : n-qubit index register prepared in uniform superposition,
            th     : (p+2)-qubit fixed-point register holding θ(j),
            tgt    : single target qubit receiving the rotation.

        For r ≥ 1 the circuit performs:

            1. Prepare a uniform superposition over j using Hadamard gates.
            2. Load the fixed-point representation of θ(j) into register th
               via a reversible XOR lookup unitary.
            3. Apply a ladder of bit-controlled RX rotations implementing the
               fixed-point expansion of −r·θ(j) on the target qubit.
            4. Optionally uncompute the θ register to restore it to |0…0⟩.

        For r = 0 a constant rotation is applied corresponding to the
        Chebyshev base case.

        Args:
            n: Number of qubits in the index register.
            p: Number of fractional bits used in the fixed-point expansion.
            r: Chebyshev polynomial order.
            qiskit_matches_paper_rx: If True, adjusts rotation angles so
                Qiskit's RX convention matches the paper convention.
            uncompute_theta: Whether to uncompute the θ register after the
                controlled rotations.
            j: Quantum register containing the n index qubits.
            th: Quantum register storing the (p+2) fixed-point angle bits.
            tgt: Single-qubit target register receiving the rotation.
            qc: QuantumCircuit object to which the operations are appended.

        Returns:
            The modified QuantumCircuit containing the U_{v_r} construction.

        Raises:
            ValueError: If n < 2 or p < 1, or if the provided registers do not
                match the required sizes.
    """
    if n < 2:
        raise ValueError("n must be >= 2.")
    if p < 1:
        raise ValueError("p must be >= 1.")

    m = p + 2
    if len(j) != n:
        raise ValueError(f"len(j) must be {n}")
    if len(th) != m:
        raise ValueError(f"len(th) must be {m}")
    if len(tgt) != 1:
        raise ValueError("tgt must contain exactly 1 qubit")

    # r=0 constant case
    if r == 0:
        qc.h(j)
        qc.rx(-2 * np.pi / 3, tgt[0])
        return qc

    # 1) Uniform superposition over j
    qc.h(j)

    # 2) XOR loader on (j, th)
    load_gate = get_theta_loader_gate(n, p)
    qc.append(load_gate, list(j) + list(th))

    # 3) Bit-controlled RX rotations
    ks = [1, 0] + list(range(-1, -p - 1, -1))
    for bit_index, kpow in enumerate(ks):
        angle_paper = -r * (2.0**kpow)
        phi = 2.0 * angle_paper if qiskit_matches_paper_rx else angle_paper
        qc.crx(phi, th[bit_index], tgt[0])

    # 4) Optional uncompute
    if uncompute_theta:
        qc.append(load_gate, list(j) + list(th))

    return qc
    
# -----------------------------------------------------------------------------
# Example usage (commented out)
# -----------------------------------------------------------------------------
# from qiskit import QuantumRegister
#
# n = 3
# p = 2
# m = p + 2
#
# j = QuantumRegister(n, "j")
# th = QuantumRegister(m, "theta")
# tgt = QuantumRegister(1, "tgt")
#
# qc = QuantumCircuit(j, th, tgt)
# U_vr_circuit_no_x(n=n, p=p, r=2, j=j, th=th, tgt=tgt, qc=qc)
# print(qc)