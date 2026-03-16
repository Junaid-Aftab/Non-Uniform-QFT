"""
This module implements classical preprocessing, fixed-point encodings, reversible
angle-loading unitaries, and Qiskit circuit constructors for the operators
U_{q,r} and U_{u_r} appearing in a fixed-point quantum signal-loading scheme.

The module supports:
    • classical preprocessing of arrays t and s into the derived quantities
          y_j, φ_j, and θ_j,
    • unsigned and sign-magnitude fixed-point encodings of those quantities,
    • reversible XOR-based lookup loaders that coherently map
          |j⟩|t⟩ → |j⟩|t XOR b(j)⟩,
      where b(j) is a precomputed fixed-point bitstring,
    • construction of fixed-q block circuits U_{q,r},
    • construction of LCU-combined circuits U_{u_r},
    • caching of loader gates, bitstring tables, coefficient vectors,
      PREP states, and phase gates.

Classical quantities:
    For N = 2^n and input arrays t and s, the module defines
        y_j = t_j - s_j / N,
    then wraps each value into the interval
        [-1/(2N), 1/(2N)].
    From these adjusted offsets it computes
        φ_j = π N y_j,
        θ_j = arccos(2 N y_j),
    where 2 N y_j is clipped to [-1, 1] before applying arccos.

Fixed-point conventions:
    • Unsigned fixed-point values are stored in little-endian order
          [2^-f, ..., 2^-1, 2^0, 2^1, ..., 2^(i-1)].
    • Signed values use sign-magnitude form
          [sign] + magnitude_bits.
    • The φ-register therefore contains one sign bit followed by unsigned
      magnitude bits.
    • The θ-register contains only unsigned bits.

Registers used by U_{q,r}:
    • j      : n-qubit index register, with N = 2^n,
    • phi    : sign-magnitude fixed-point register storing φ_j,
    • theta  : unsigned fixed-point register storing θ_j,
    • tgt    : single target qubit acted on by controlled RX and RZ ladders.

Core idea:
    All numerical quantities are computed classically at circuit-construction
    time using NumPy and SciPy. The resulting fixed-point bitstrings are then
    embedded into the quantum circuit through reversible classical lookup
    unitaries. No online arithmetic, angle synthesis, or trigonometric
    evaluation is performed on the quantum device beyond loading precomputed
    bit patterns and applying controlled rotations.

Loader synthesis:
    The module provides two realizations of the XOR loaders:
    • "dense": construct the full permutation unitary matrix and wrap it as a
      UnitaryGate,
    • "mcx"  : synthesize the loader as a sequence of multi-controlled X gates
      conditioned on the computational basis state |j⟩.

Structure of U_{q,r}:
    1. Apply Hadamards to the index register j to create a uniform
       superposition over basis states.
    2. If q ≠ 0, coherently load θ(j) into the theta register.
    3. Apply a controlled RX ladder from theta bits onto tgt, with weights
       2^-p, ..., 2^-1, 2^0, 2^1 and an overall factor proportional to −q.
    4. Optionally uncompute the theta register by reapplying the same loader.
    5. Coherently load φ(j) into the phi register.
    6. Apply a signed controlled RZ ladder from the phi magnitude bits onto
       tgt, together with a sign-controlled correction using the phi sign bit.
    7. Optionally uncompute the phi register by reapplying the same loader.

RX and RZ conventions:
    The circuit builder includes switches
        qiskit_matches_paper_rx
        qiskit_matches_paper_rz
    that optionally double rotation parameters so the implemented gates match
    Qiskit's angle conventions relative to the paper formulas.

Coefficient construction:
    The module defines the coefficients α_{q,r} via a specialized Bessel-based
    formula using scipy.special.jv, together with the modified coefficients
    α'_{q,r} obtained by special-case rescaling when q = 0 and/or r = 0.

Structure of U_{u_r}:
    1. Construct the coefficient vector
           [α'_{0,r}, α'_{1,r}, ..., α'_{K-1,r}].
    2. Form the LCU PREP state from coefficient magnitudes.
    3. Identify active q values with non-negligible coefficients.
    4. Apply the corresponding controlled-U_q blocks using an L-qubit
       selection register q, where K = 2^L.
    5. Apply a diagonal phase gate encoding the coefficient phases.
    6. Uncompute the selection register with PREP†.

Caching behavior:
    The implementation caches:
    • phi and theta bitstring tables,
    • synthesized loader gates,
    • coefficient vectors α'_{q,r},
    • PREP states and their normalization constants,
    • diagonal phase gates.

Implementation notes:
    • The helper for U_{u_r} assumes K is a power of two.
    • Theta loading is skipped entirely when q = 0 because the RX contribution
      vanishes in that case.
    • The module exposes helper functions for clearing caches and for
      constructing reusable gate objects separately from the final circuits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
from scipy.special import jv

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import MCXGate, RZGate, StatePreparation, UnitaryGate


LoaderMode = Literal["dense", "mcx"]


# -----------------------------------------------------------------------------
# Fixed-point helpers
# -----------------------------------------------------------------------------

def unsigned_to_fixed_bits(x: float, frac_bits: int, int_bits: int = 2) -> List[int]:
    """Convert a nonnegative real number into unsigned fixed-point bits.

    This routine encodes a nonnegative real number into an unsigned fixed-point
    bitstring stored in little-endian fixed-point order:
        [2^-frac_bits, ..., 2^-1, 2^0, 2^1, ..., 2^(int_bits-1)].

    Args:
        x: Nonnegative real number to encode.
        frac_bits: Number of fractional bits.
        int_bits: Number of integer bits.

    Returns:
        A list of length frac_bits + int_bits containing the fixed-point
        representation of x in little-endian order.

    Raises:
        ValueError: If x is negative.
    """
    if x < 0:
        raise ValueError("unsigned_to_fixed_bits expects x >= 0.")

    rem = float(x)
    bits_be: List[int] = []

    for k in range(int_bits - 1, -1, -1):
        w = 2.0 ** k
        b = int(rem >= w)
        bits_be.append(b)
        rem -= b * w

    for k in range(-1, -frac_bits - 1, -1):
        w = 2.0 ** k
        b = int(rem >= w)
        bits_be.append(b)
        rem -= b * w

    int_part_be = bits_be[:int_bits]
    frac_part_be = bits_be[int_bits:]

    frac_part_le = list(reversed(frac_part_be))
    int_part_le = list(reversed(int_part_be))

    return frac_part_le + int_part_le


def signed_to_fixed_bits_signmag(
    x: float,
    frac_bits: int,
    int_bits: int = 2,
) -> List[int]:
    """Convert a real number into sign-magnitude fixed-point bits.

    This routine encodes a real number using a sign bit followed by the
    unsigned fixed-point representation of its magnitude. The layout is
    [sign] + magnitude_bits, where magnitude_bits follow the same ordering as
    unsigned_to_fixed_bits(...).

    Args:
        x: Real number to encode.
        frac_bits: Number of fractional bits.
        int_bits: Number of integer bits used for the magnitude.

    Returns:
        A bit list of length 1 + int_bits + frac_bits representing x in
        sign-magnitude fixed-point form.

    Raises:
        None.
    """
    sign = 1 if x < 0 else 0
    mag = abs(float(x))
    mag_bits = unsigned_to_fixed_bits(mag, frac_bits=frac_bits, int_bits=int_bits)
    return [sign] + mag_bits


def build_xor_loader_unitary(bitstrings_by_j: List[List[int]]) -> np.ndarray:
    """Construct a reversible XOR loader unitary for angle bitstrings.

    This routine builds a dense permutation unitary implementing the mapping
    |j⟩|t⟩ -> |j⟩|t XOR b(j)⟩, where b(j) is the precomputed fixed-point
    bitstring associated with basis index j.

    Args:
        bitstrings_by_j: List of bitstrings indexed by j.

    Returns:
        A dense unitary matrix implementing the reversible XOR lookup.

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
        b_int = sum((bit << q) for q, bit in enumerate(b))

        for tval in range(2**m):
            t2 = tval ^ b_int
            in_idx = j + N * tval
            out_idx = j + N * t2
            U[out_idx, in_idx] = 1.0

    return U


# -----------------------------------------------------------------------------
# Classical preprocessing
# -----------------------------------------------------------------------------

def y_from_t_and_s(t: np.ndarray, s: np.ndarray, N: int) -> np.ndarray:
    """Compute the adjusted offset y = t - s / N.

    This routine computes the array y = t - s / N and then wraps the result
    into the interval [-1 / (2N), 1 / (2N)] using elementwise adjustments.

    Args:
        t: Array of node values.
        s: Array of grid indices or offsets.
        N: Problem size used in the normalization.

    Returns:
        A NumPy array containing the adjusted values of y.

    Raises:
        None.
    """
    y = np.asarray(t, dtype=float) - (np.asarray(s, dtype=float) / float(N))
    half = 0.5 / float(N)
    y = np.where(y > half, y - 1.0 / float(N), y)
    y = np.where(y < -half, y + 1.0 / float(N), y)
    return y


def phi_from_y(y: np.ndarray, N: int) -> np.ndarray:
    """Compute phi = pi N y.

    This routine scales the input array y by pi times N to produce the phase
    array phi.

    Args:
        y: Input array of adjusted offsets.
        N: Problem size used in the scaling.

    Returns:
        A NumPy array containing phi = pi N y.

    Raises:
        None.
    """
    return np.pi * float(N) * np.asarray(y, dtype=float)


def theta_from_y(y: np.ndarray, N: int) -> np.ndarray:
    """Compute theta = arccos(2 N y) with clipping.

    This routine forms the quantity 2 N y, clips it to the interval [-1, 1],
    and applies arccos elementwise to obtain theta.

    Args:
        y: Input array of adjusted offsets.
        N: Problem size used in the scaling.

    Returns:
        A NumPy array containing theta values.

    Raises:
        None.
    """
    z = 2.0 * float(N) * np.asarray(y, dtype=float)
    z = np.clip(z, -1.0, 1.0)
    return np.arccos(z)


# -----------------------------------------------------------------------------
# Caches
# -----------------------------------------------------------------------------

_LOADER_CACHE: Dict[Tuple[int, int, int, str, LoaderMode, int], Gate] = {}
_ALPHA_VEC_CACHE: Dict[Tuple[int, int], np.ndarray] = {}
_PREP_CACHE: Dict[
    Tuple[bytes, Tuple[int, ...]],
    Tuple[StatePreparation, float, np.ndarray],
] = {}
_PHASE_GATE_CACHE: Dict[Tuple[bytes, Tuple[int, ...]], UnitaryGate] = {}
_PHI_BITSTRINGS_CACHE: Dict[Tuple[int, int, int, int, int], List[List[int]]] = {}
_THETA_BITSTRINGS_CACHE: Dict[Tuple[int, int, int, int, int], List[List[int]]] = {}


# -----------------------------------------------------------------------------
# Cache helpers
# -----------------------------------------------------------------------------

def _hash_array(a: np.ndarray) -> int:
    """Return a process-local hash for array data.

    This routine converts an array into a hashable representation based on its
    shape, dtype, and raw bytes, and returns the resulting Python hash value.

    Args:
        a: Input array to hash.

    Returns:
        An integer hash derived from the array metadata and contents.

    Raises:
        None.
    """
    a = np.asarray(a)
    return hash((a.shape, str(a.dtype), a.tobytes()))


def _hash_bitstrings(bitstrings_by_j: List[List[int]]) -> int:
    """Return a cache hash for a table of bitstrings.

    This routine converts a nested bitstring table into an immutable tuple form
    and returns a hash suitable for use in cache keys.

    Args:
        bitstrings_by_j: List of bitstrings indexed by j.

    Returns:
        An integer hash derived from the bitstring table contents.

    Raises:
        None.
    """
    flat = tuple(tuple(int(b) for b in row) for row in bitstrings_by_j)
    return hash(flat)


def _coeff_cache_key(a: np.ndarray) -> Tuple[bytes, Tuple[int, ...]]:
    """Construct a stable cache key for a small dense array.

    This routine converts an array to contiguous storage and returns a cache key
    consisting of the raw bytes and the array shape.

    Args:
        a: Input array for which to build a cache key.

    Returns:
        A tuple containing the array byte representation and shape.

    Raises:
        None.
    """
    arr = np.ascontiguousarray(np.asarray(a))
    return (arr.tobytes(), arr.shape)


def _normalize_loader_mode(loader_mode: str) -> LoaderMode:
    """Validate and normalize the loader implementation selector.

    This routine lowercases the requested loader mode, verifies that it is one
    of the supported implementations, and returns the normalized value.

    Args:
        loader_mode: Requested loader implementation mode.

    Returns:
        The normalized loader mode string.

    Raises:
        ValueError: If loader_mode is not one of the supported values.
    """
    mode = str(loader_mode).lower()
    if mode not in {"dense", "mcx"}:
        raise ValueError("loader_mode must be either 'dense' or 'mcx'.")
    return mode  # type: ignore[return-value]


def _phi_bitstrings_cache_key(
    n: int,
    m_frac: int,
    int_bits: int,
    t: np.ndarray,
    s: np.ndarray,
) -> Tuple[int, int, int, int, int]:
    """Construct the cache key for phi bitstring tables.

    This routine forms a cache key from the precision parameters and hashed
    input arrays used to compute phi bitstrings.

    Args:
        n: Number of index qubits, with N = 2^n.
        m_frac: Number of fractional bits used for phi encoding.
        int_bits: Number of integer bits used in the phi encoding.
        t: Input array of node values.
        s: Input array of grid indices or offsets.

    Returns:
        A tuple suitable for indexing the phi bitstring cache.

    Raises:
        None.
    """
    return (
        n,
        m_frac,
        int_bits,
        _hash_array(np.asarray(t, dtype=float)),
        _hash_array(np.asarray(s, dtype=int)),
    )


def _theta_bitstrings_cache_key(
    n: int,
    p_frac: int,
    int_bits: int,
    t: np.ndarray,
    s: np.ndarray,
) -> Tuple[int, int, int, int, int]:
    """Construct the cache key for theta bitstring tables.

    This routine forms a cache key from the precision parameters and hashed
    input arrays used to compute theta bitstrings.

    Args:
        n: Number of index qubits, with N = 2^n.
        p_frac: Number of fractional bits used for theta encoding.
        int_bits: Number of integer bits used in the theta encoding.
        t: Input array of node values.
        s: Input array of grid indices or offsets.

    Returns:
        A tuple suitable for indexing the theta bitstring cache.

    Raises:
        None.
    """
    return (
        n,
        p_frac,
        int_bits,
        _hash_array(np.asarray(t, dtype=float)),
        _hash_array(np.asarray(s, dtype=int)),
    )


# -----------------------------------------------------------------------------
# Bitstring-table helpers
# -----------------------------------------------------------------------------

def _get_phi_bitstrings(
    n: int,
    m_frac: int,
    t: np.ndarray,
    s: np.ndarray,
    *,
    int_bits: int = 2,
) -> List[List[int]]:
    """Return cached sign-magnitude phi bitstrings.

    This routine validates the input lengths, computes y and phi from the input
    arrays, converts phi values into sign-magnitude fixed-point bitstrings, and
    caches the resulting table.

    Args:
        n: Number of index qubits, with N = 2^n.
        m_frac: Number of fractional bits used for phi encoding.
        t: Input array of node values.
        s: Input array of grid indices or offsets.
        int_bits: Number of integer bits used in the phi encoding.

    Returns:
        A cached or newly constructed list of sign-magnitude phi bitstrings.

    Raises:
        ValueError: If t or s does not have length N.
    """
    key = _phi_bitstrings_cache_key(n, m_frac, int_bits, t, s)
    cached = _PHI_BITSTRINGS_CACHE.get(key)
    if cached is not None:
        return cached

    N = 2**n
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=int)
    if len(t) != N or len(s) != N:
        raise ValueError(f"t and s must both have length {N}.")

    y = y_from_t_and_s(t, s, N)
    phi = phi_from_y(y, N)

    out = [
        signed_to_fixed_bits_signmag(phi[j], frac_bits=m_frac, int_bits=int_bits)
        for j in range(N)
    ]
    _PHI_BITSTRINGS_CACHE[key] = out
    return out


def _get_theta_bitstrings(
    n: int,
    p_frac: int,
    t: np.ndarray,
    s: np.ndarray,
    *,
    int_bits: int = 2,
) -> List[List[int]]:
    """Return cached theta bitstrings.

    This routine validates the input lengths, computes y and theta from the
    input arrays, converts theta values into unsigned fixed-point bitstrings,
    and caches the resulting table.

    Args:
        n: Number of index qubits, with N = 2^n.
        p_frac: Number of fractional bits used for theta encoding.
        t: Input array of node values.
        s: Input array of grid indices or offsets.
        int_bits: Number of integer bits used in the theta encoding.

    Returns:
        A cached or newly constructed list of theta bitstrings.

    Raises:
        ValueError: If t or s does not have length N.
    """
    key = _theta_bitstrings_cache_key(n, p_frac, int_bits, t, s)
    cached = _THETA_BITSTRINGS_CACHE.get(key)
    if cached is not None:
        return cached

    N = 2**n
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=int)
    if len(t) != N or len(s) != N:
        raise ValueError(f"t and s must both have length {N}.")

    y = y_from_t_and_s(t, s, N)
    th = theta_from_y(y, N)

    out = [
        unsigned_to_fixed_bits(float(th[j]), frac_bits=p_frac, int_bits=int_bits)
        for j in range(N)
    ]
    _THETA_BITSTRINGS_CACHE[key] = out
    return out


# -----------------------------------------------------------------------------
# Loader synthesis helpers
# -----------------------------------------------------------------------------

def _append_controlled_on_bitstring_xor_load(
    qc: QuantumCircuit,
    jreg,
    treg,
    bitstrings_by_j: List[List[int]],
) -> None:
    """Append a reversible XOR load controlled by the index register.

    This routine appends gates implementing the transformation
    |j⟩|t⟩ -> |j⟩|t XOR b(j)⟩ using multi-controlled X gates, without
    constructing a dense unitary matrix.

    Args:
        qc: QuantumCircuit to which the loader operations are appended.
        jreg: Register encoding the index j.
        treg: Target register receiving the XOR-loaded bitstring.
        bitstrings_by_j: List of bitstrings indexed by j.

    Returns:
        None.

    Raises:
        ValueError: If jreg is empty.
        ValueError: If bitstrings_by_j does not have length 2**len(jreg).
    """
    n = len(jreg)
    if n == 0:
        raise ValueError("jreg must be non-empty.")
    if len(bitstrings_by_j) != 2**n:
        raise ValueError("bitstrings_by_j must have length 2**len(jreg).")

    mcx_gate = MCXGate(n)

    for j, bits in enumerate(bitstrings_by_j):
        active_targets = [idx for idx, bit in enumerate(bits) if bit]
        if not active_targets:
            continue

        selector = [(j >> k) & 1 for k in range(n)]

        for k, bit in enumerate(selector):
            if bit == 0:
                qc.x(jreg[k])

        controls = list(jreg)
        for tgt_idx in active_targets:
            qc.append(mcx_gate, controls + [treg[tgt_idx]])

        for k, bit in enumerate(selector):
            if bit == 0:
                qc.x(jreg[k])


def _xor_loader_gate_from_bitstrings_dense(
    *,
    n: int,
    out_len: int,
    bitstrings_by_j: List[List[int]],
    label: str,
) -> UnitaryGate:
    """Build a dense XOR loader gate from a classical bitstring table.

    This routine validates the bitstring table dimensions, constructs the dense
    reversible XOR-loader unitary, and wraps it as a UnitaryGate.

    Args:
        n: Number of index qubits.
        out_len: Length of the output target register.
        bitstrings_by_j: List of bitstrings indexed by j.
        label: Gate label used for the generated unitary.

    Returns:
        A UnitaryGate implementing the reversible XOR loader.

    Raises:
        ValueError: If bitstrings_by_j does not have length 2**n.
        ValueError: If the bitstrings do not all have length out_len.
    """
    if len(bitstrings_by_j) != 2**n:
        raise ValueError("bitstrings_by_j must have length 2**n.")
    if any(len(bits) != out_len for bits in bitstrings_by_j):
        raise ValueError("All bitstrings must have identical length out_len.")

    Uload = build_xor_loader_unitary(bitstrings_by_j)
    return UnitaryGate(Uload, label=label)


def _xor_loader_gate_from_bitstrings_mcx(
    *,
    n: int,
    out_len: int,
    bitstrings_by_j: List[List[int]],
    label: str,
) -> Gate:
    """Build an MCX-based XOR loader gate from a classical bitstring table.

    This routine validates the bitstring table dimensions, constructs a
    temporary subcircuit using multi-controlled X gates, and converts that
    subcircuit into a reusable Gate object.

    Args:
        n: Number of index qubits.
        out_len: Length of the output target register.
        bitstrings_by_j: List of bitstrings indexed by j.
        label: Gate label used for the generated subcircuit.

    Returns:
        A Gate implementing the reversible XOR loader.

    Raises:
        ValueError: If bitstrings_by_j does not have length 2**n.
        ValueError: If the bitstrings do not all have length out_len.
    """
    if len(bitstrings_by_j) != 2**n:
        raise ValueError("bitstrings_by_j must have length 2**n.")
    if any(len(bits) != out_len for bits in bitstrings_by_j):
        raise ValueError("All bitstrings must have identical length out_len.")

    jtmp = QuantumRegister(n, "j")
    treg = QuantumRegister(out_len, "t")
    qc = QuantumCircuit(jtmp, treg, name=label)
    _append_controlled_on_bitstring_xor_load(qc, jtmp, treg, bitstrings_by_j)
    return qc.to_gate(label=label)


def _xor_loader_gate_from_bitstrings(
    *,
    n: int,
    out_len: int,
    bitstrings_by_j: List[List[int]],
    label: str,
    loader_mode: LoaderMode,
) -> Gate:
    """Build an XOR loader gate using the requested synthesis method.

    This routine dispatches to either the dense-unitary loader constructor or
    the MCX-based loader constructor based on the selected loader mode.

    Args:
        n: Number of index qubits.
        out_len: Length of the output target register.
        bitstrings_by_j: List of bitstrings indexed by j.
        label: Gate label used for the generated loader.
        loader_mode: Loader synthesis mode, either ``"dense"`` or ``"mcx"``.

    Returns:
        A Gate implementing the reversible XOR loader.

    Raises:
        ValueError: If loader_mode is unsupported.
    """
    if loader_mode == "dense":
        return _xor_loader_gate_from_bitstrings_dense(
            n=n,
            out_len=out_len,
            bitstrings_by_j=bitstrings_by_j,
            label=label,
        )
    if loader_mode == "mcx":
        return _xor_loader_gate_from_bitstrings_mcx(
            n=n,
            out_len=out_len,
            bitstrings_by_j=bitstrings_by_j,
            label=label,
        )
    raise ValueError("loader_mode must be either 'dense' or 'mcx'.")


# -----------------------------------------------------------------------------
# Public loader helpers
# -----------------------------------------------------------------------------

def get_phi_loader_gate(
    n: int,
    m_frac: int,
    t: np.ndarray,
    s: np.ndarray,
    *,
    int_bits: int = 2,
    loader_mode: str = "dense",
) -> Gate:
    """Retrieve or construct the cached phi loader gate.

    This routine obtains the cached sign-magnitude phi bitstrings, constructs a
    reversible XOR loader gate using the requested loader implementation, and
    stores the gate in the loader cache.

    Args:
        n: Number of index qubits, with N = 2^n.
        m_frac: Number of fractional bits used for phi encoding.
        t: Input array of node values.
        s: Input array of grid indices or offsets.
        int_bits: Number of integer bits used in the phi encoding.
        loader_mode: Loader synthesis mode, either ``"dense"`` or ``"mcx"``.

    Returns:
        A cached or newly constructed Gate implementing the phi loader.

    Raises:
        ValueError: If loader_mode is unsupported.
    """
    mode = _normalize_loader_mode(loader_mode)
    phi_bits_by_j = _get_phi_bitstrings(n, m_frac, t, s, int_bits=int_bits)

    key = (n, m_frac, int_bits, "phi", mode, _hash_bitstrings(phi_bits_by_j))
    cached = _LOADER_CACHE.get(key)
    if cached is not None:
        return cached

    load_gate = _xor_loader_gate_from_bitstrings(
        n=n,
        out_len=1 + int_bits + m_frac,
        bitstrings_by_j=phi_bits_by_j,
        label=f"U_load_phi(n={n},m={m_frac})",
        loader_mode=mode,
    )
    _LOADER_CACHE[key] = load_gate
    return load_gate


def get_theta_loader_gate_from_t_s(
    n: int,
    p_frac: int,
    t: np.ndarray,
    s: np.ndarray,
    *,
    int_bits: int = 2,
    loader_mode: str = "dense",
) -> Gate:
    """Retrieve or construct the cached theta loader gate.

    This routine obtains the cached theta bitstrings, constructs a reversible
    XOR loader gate using the requested loader implementation, and stores the
    gate in the loader cache.

    Args:
        n: Number of index qubits, with N = 2^n.
        p_frac: Number of fractional bits used for theta encoding.
        t: Input array of node values.
        s: Input array of grid indices or offsets.
        int_bits: Number of integer bits used in the theta encoding.
        loader_mode: Loader synthesis mode, either ``"dense"`` or ``"mcx"``.

    Returns:
        A cached or newly constructed Gate implementing the theta loader.

    Raises:
        ValueError: If loader_mode is unsupported.
    """
    mode = _normalize_loader_mode(loader_mode)
    theta_bits_by_j = _get_theta_bitstrings(n, p_frac, t, s, int_bits=int_bits)

    key = (n, p_frac, int_bits, "theta", mode, _hash_bitstrings(theta_bits_by_j))
    cached = _LOADER_CACHE.get(key)
    if cached is not None:
        return cached

    load_gate = _xor_loader_gate_from_bitstrings(
        n=n,
        out_len=int_bits + p_frac,
        bitstrings_by_j=theta_bits_by_j,
        label=f"U_load_theta_ts(n={n},p={p_frac})",
        loader_mode=mode,
    )
    _LOADER_CACHE[key] = load_gate
    return load_gate


def clear_loader_cache() -> None:
    """Clear all cached gates, tables, and precomputed objects.

    This routine empties all module-level caches used for loader gates,
    bitstring tables, coefficient vectors, state-preparation data, and phase
    gates.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    _LOADER_CACHE.clear()
    _ALPHA_VEC_CACHE.clear()
    _PREP_CACHE.clear()
    _PHASE_GATE_CACHE.clear()
    _PHI_BITSTRINGS_CACHE.clear()
    _THETA_BITSTRINGS_CACHE.clear()


# -----------------------------------------------------------------------------
# Coefficients alpha_{q,r} and alpha'_{q,r}
# -----------------------------------------------------------------------------

def alpha_qr(q: int, r: int) -> complex:
    """Compute alpha_{q,r} from the specialized coefficient formula.

    This routine evaluates alpha_{q,r} from Equation (130) specialized to
    gamma = 1 / 2. It returns zero when the parity constraint on q and r is not
    satisfied.

    Args:
        q: Coefficient index q.
        r: Coefficient index r.

    Returns:
        The complex coefficient alpha_{q,r}.

    Raises:
        None.
    """
    if (abs(q - r) % 2) != 0:
        return 0.0 + 0.0j

    a = (q + r) // 2
    b = (r - q) // 2
    return 4.0 * (1j ** r) * jv(a, -np.pi / 4.0) * jv(b, -np.pi / 4.0)


def alpha_qr_prime(q: int, r: int) -> complex:
    """Compute the modified coefficient alpha'_{q,r}.

    This routine rescales alpha_{q,r} according to the special cases for q = 0
    and r = 0.

    Args:
        q: Coefficient index q.
        r: Coefficient index r.

    Returns:
        The complex coefficient alpha'_{q,r}.

    Raises:
        None.
    """
    a = alpha_qr(q, r)
    if q == 0 and r == 0:
        return 0.25 * a
    if q == 0 or r == 0:
        return 0.5 * a
    return a


def alpha_vec_prime(r: int, K: int) -> np.ndarray:
    """Return the vector [alpha'_{0,r}, ..., alpha'_{K-1,r}] with caching.

    This routine builds the modified coefficient vector for a fixed r, stores it
    in the cache, and returns a copy of the cached array.

    Args:
        r: Coefficient index r.
        K: Number of coefficients to include.

    Returns:
        A NumPy array containing [alpha'_{0,r}, ..., alpha'_{K-1,r}].

    Raises:
        None.
    """
    key = (r, K)
    cached = _ALPHA_VEC_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    arr = np.array([alpha_qr_prime(q, r) for q in range(K)], dtype=complex)
    _ALPHA_VEC_CACHE[key] = arr
    return arr.copy()


# -----------------------------------------------------------------------------
# Register specification
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class UqrRegisters:
    """Store the register sizes implied by the fixed-point precision choices.

    This dataclass records the fixed-point precision parameters and provides
    derived register lengths for the phi and theta registers.

    Args:
        n: Number of index qubits.
        m_frac: Number of fractional bits for phi.
        p_frac: Number of fractional bits for theta.
        int_bits_phi: Number of integer bits for phi.
        int_bits_theta: Number of integer bits for theta.

    Returns:
        A frozen dataclass instance describing the register layout.

    Raises:
        None.
    """
    n: int
    m_frac: int
    p_frac: int
    int_bits_phi: int = 2
    int_bits_theta: int = 2

    @property
    def phi_len(self) -> int:
        """Return the total length of the phi register.

        This routine computes the phi-register length as one sign bit plus the
        integer and fractional magnitude bits.

        Args:
            None.

        Returns:
            The total number of qubits required for the phi register.

        Raises:
            None.
        """
        return 1 + self.int_bits_phi + self.m_frac

    @property
    def theta_len(self) -> int:
        """Return the total length of the theta register.

        This routine computes the theta-register length as the total number of
        integer and fractional bits used for theta encoding.

        Args:
            None.

        Returns:
            The total number of qubits required for the theta register.

        Raises:
            None.
        """
        return self.int_bits_theta + self.p_frac


# -----------------------------------------------------------------------------
# Circuit builders: U_{q,r} and LCU-combined U_{u_r}
# -----------------------------------------------------------------------------

def U_qr_circuit_no_x(
    *,
    n: int,
    m: int,
    p: int,
    q: int,
    t: np.ndarray,
    s: np.ndarray,
    qiskit_matches_paper_rx: bool = True,
    qiskit_matches_paper_rz: bool = True,
    uncompute_angles: bool = True,
    int_bits_phi: int = 2,
    int_bits_theta: int = 2,
    loader_mode: str = "dense",
    j,
    phi_reg,
    theta_reg,
    tgt,
    qc: QuantumCircuit,
) -> QuantumCircuit:
    """Construct the quantum circuit implementing U_{q,r} for a fixed q.

    This routine validates the supplied registers and precision parameters,
    prepares the index register in superposition, loads theta and phi values
    using the requested XOR-loader realization, applies the controlled RX and
    RZ operations required for the U_{q,r} block, and optionally uncomputes the
    angle registers.

    Args:
        n: Number of index qubits.
        m: Number of fractional phi bits.
        p: Number of fractional theta bits.
        q: Fixed coefficient index q.
        t: Input array of node values.
        s: Input array of grid indices or offsets.
        qiskit_matches_paper_rx: Whether RX angles are doubled to match Qiskit
            conventions.
        qiskit_matches_paper_rz: Whether RZ angles are doubled to match Qiskit
            conventions.
        uncompute_angles: Whether the loaded angle registers are uncomputed.
        int_bits_phi: Number of integer bits for phi.
        int_bits_theta: Number of integer bits for theta.
        loader_mode: Loader synthesis mode, either ``"dense"`` or ``"mcx"``.
        j: Index register.
        phi_reg: Phi-angle register.
        theta_reg: Theta-angle register.
        tgt: Single-qubit target register.
        qc: QuantumCircuit to which the operations are appended.

    Returns:
        The modified QuantumCircuit implementing U_{q,r}.

    Raises:
        ValueError: If n, m, or p is less than 1.
        ValueError: If the supplied register sizes do not match the expected
            layout.
    """
    mode = _normalize_loader_mode(loader_mode)

    if n < 1:
        raise ValueError("n must be >= 1.")
    if m < 1:
        raise ValueError("m must be >= 1.")
    if p < 1:
        raise ValueError("p must be >= 1.")
    if len(j) != n:
        raise ValueError(f"len(j) must be {n}")

    reg_spec = UqrRegisters(
        n=n,
        m_frac=m,
        p_frac=p,
        int_bits_phi=int_bits_phi,
        int_bits_theta=int_bits_theta,
    )

    if len(phi_reg) != reg_spec.phi_len:
        raise ValueError(f"len(phi_reg) must be {reg_spec.phi_len}")
    if len(theta_reg) != reg_spec.theta_len:
        raise ValueError(f"len(theta_reg) must be {reg_spec.theta_len}")
    if len(tgt) != 1:
        raise ValueError("tgt must contain exactly 1 qubit")

    qc.h(j)

    if q != 0:
        theta_gate = get_theta_loader_gate_from_t_s(
            n=n,
            p_frac=p,
            t=t,
            s=s,
            int_bits=int_bits_theta,
            loader_mode=mode,
        )
        qc.append(theta_gate, list(j) + list(theta_reg))

        ks_theta = list(range(-p, 0)) + list(range(0, int_bits_theta))
        for bit_index, kpow in enumerate(ks_theta):
            angle_paper = -float(q) * (2.0 ** kpow)
            lam = 2.0 * angle_paper if qiskit_matches_paper_rx else angle_paper
            qc.crx(lam, theta_reg[bit_index], tgt[0])

        if uncompute_angles:
            qc.append(theta_gate, list(j) + list(theta_reg))

    phi_gate = get_phi_loader_gate(
        n=n,
        m_frac=m,
        t=t,
        s=s,
        int_bits=int_bits_phi,
        loader_mode=mode,
    )
    qc.append(phi_gate, list(j) + list(phi_reg))

    sign_qubit = phi_reg[0]
    mag_bits = phi_reg[1:]

    ks_phi = list(range(-m, 0)) + list(range(0, int_bits_phi))
    for bit_index, kpow in enumerate(ks_phi):
        angle_paper = 2.0 ** kpow
        lam = 2.0 * angle_paper if qiskit_matches_paper_rz else angle_paper
        qc.crz(lam, mag_bits[bit_index], tgt[0])
        qc.append(
            RZGate(-2.0 * lam).control(2),
            [sign_qubit, mag_bits[bit_index], tgt[0]],
        )

    if uncompute_angles:
        qc.append(phi_gate, list(j) + list(phi_reg))

    return qc


def _prep_gate_for_coeffs(
    coeffs: np.ndarray,
) -> Tuple[StatePreparation, float, np.ndarray]:
    """Build the PREP gate for LCU from a coefficient vector.

    This routine computes coefficient magnitudes and phases, forms the
    normalized amplitude vector for the PREP state, caches the resulting
    StatePreparation object together with its normalization factor and phases,
    and returns the cached tuple.

    Args:
        coeffs: Complex coefficient vector used in the LCU construction.

    Returns:
        A tuple containing the PREP gate, the normalization factor, and the
        coefficient phases.

    Raises:
        ValueError: If all coefficients are zero.
    """
    coeffs = np.asarray(coeffs, dtype=complex)
    cache_key = _coeff_cache_key(coeffs)
    cached = _PREP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mags = np.abs(coeffs)
    alpha = float(np.sum(mags))
    if alpha == 0.0:
        raise ValueError("All coefficients are zero.")

    amps = np.sqrt(mags / alpha)
    phases = np.angle(coeffs)
    prep = StatePreparation(amps, label="PREP")

    out = (prep, alpha, phases)
    _PREP_CACHE[cache_key] = out
    return out


def _diag_phase_gate(phases: np.ndarray) -> UnitaryGate:
    """Construct a cached diagonal phase gate from a phase vector.

    This routine builds the diagonal unitary diag(exp(i * phases[q])) from the
    supplied phase vector, caches the resulting gate, and returns it.

    Args:
        phases: Real-valued phase vector indexed by q.

    Returns:
        A UnitaryGate implementing the diagonal phase operation.

    Raises:
        None.
    """
    phases = np.asarray(phases, dtype=float)
    cache_key = _coeff_cache_key(phases)
    cached = _PHASE_GATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    diag = np.exp(1j * phases)
    gate = UnitaryGate(np.diag(diag), label="PHASE_diag")
    _PHASE_GATE_CACHE[cache_key] = gate
    return gate


def _uq_gate_cache_key(
    *,
    n: int,
    m: int,
    p: int,
    q: int,
    t: np.ndarray,
    s: np.ndarray,
    qiskit_matches_paper_rx: bool,
    qiskit_matches_paper_rz: bool,
    uncompute_angles: bool,
    int_bits_phi: int,
    int_bits_theta: int,
    loader_mode: str,
) -> Tuple:
    """Construct the cache key for a U_q gate configuration.

    This routine hashes the array inputs and combines them with the circuit and
    precision parameters to produce a cache key for a particular U_q
    configuration.

    Args:
        n: Number of index qubits.
        m: Number of fractional phi bits.
        p: Number of fractional theta bits.
        q: Fixed coefficient index q.
        t: Input array of node values.
        s: Input array of grid indices or offsets.
        qiskit_matches_paper_rx: Whether RX angles match Qiskit conventions.
        qiskit_matches_paper_rz: Whether RZ angles match Qiskit conventions.
        uncompute_angles: Whether angle registers are uncomputed.
        int_bits_phi: Number of integer bits for phi.
        int_bits_theta: Number of integer bits for theta.
        loader_mode: Loader synthesis mode.

    Returns:
        A tuple suitable for identifying a unique U_q gate configuration.

    Raises:
        None.
    """
    return (
        n,
        m,
        p,
        q,
        _hash_array(np.asarray(t, dtype=float)),
        _hash_array(np.asarray(s, dtype=int)),
        qiskit_matches_paper_rx,
        qiskit_matches_paper_rz,
        uncompute_angles,
        int_bits_phi,
        int_bits_theta,
        _normalize_loader_mode(loader_mode),
    )


def _build_uq_gate(
    *,
    n: int,
    m: int,
    p: int,
    q: int,
    t: np.ndarray,
    s: np.ndarray,
    qiskit_matches_paper_rx: bool,
    qiskit_matches_paper_rz: bool,
    uncompute_angles: bool,
    int_bits_phi: int,
    int_bits_theta: int,
    loader_mode: str,
) -> Gate:
    """Build a fresh U_q gate on demand.

    This routine allocates fresh registers for the U_q subcircuit, invokes the
    fixed-q circuit constructor, and converts the resulting subcircuit into a
    reusable Gate object.

    Args:
        n: Number of index qubits.
        m: Number of fractional phi bits.
        p: Number of fractional theta bits.
        q: Fixed coefficient index q.
        t: Input array of node values.
        s: Input array of grid indices or offsets.
        qiskit_matches_paper_rx: Whether RX angles match Qiskit conventions.
        qiskit_matches_paper_rz: Whether RZ angles match Qiskit conventions.
        uncompute_angles: Whether angle registers are uncomputed.
        int_bits_phi: Number of integer bits for phi.
        int_bits_theta: Number of integer bits for theta.
        loader_mode: Loader synthesis mode.

    Returns:
        A Gate implementing the requested U_q block.

    Raises:
        ValueError: Propagated from the underlying U_q circuit constructor if
            the input parameters are invalid.
    """
    jreg = QuantumRegister(n, "j")
    phi_reg = QuantumRegister(1 + int_bits_phi + m, "phi")
    theta_reg = QuantumRegister(int_bits_theta + p, "theta")
    tgt = QuantumRegister(1, "tgt")
    sub = QuantumCircuit(jreg, phi_reg, theta_reg, tgt, name=f"U_q={q}")

    U_qr_circuit_no_x(
        n=n,
        m=m,
        p=p,
        q=q,
        t=t,
        s=s,
        qiskit_matches_paper_rx=qiskit_matches_paper_rx,
        qiskit_matches_paper_rz=qiskit_matches_paper_rz,
        uncompute_angles=uncompute_angles,
        int_bits_phi=int_bits_phi,
        int_bits_theta=int_bits_theta,
        loader_mode=loader_mode,
        j=jreg,
        phi_reg=phi_reg,
        theta_reg=theta_reg,
        tgt=tgt,
        qc=sub,
    )

    return sub.to_gate(label=f"U_q={q}")


def _build_controlled_uq_gate(
    *,
    num_controls: int,
    n: int,
    m: int,
    p: int,
    q: int,
    t: np.ndarray,
    s: np.ndarray,
    qiskit_matches_paper_rx: bool,
    qiskit_matches_paper_rz: bool,
    uncompute_angles: bool,
    int_bits_phi: int,
    int_bits_theta: int,
    loader_mode: str,
) -> Gate:
    """Build a fresh controlled-U_q gate on demand.

    This routine builds the underlying U_q gate for the requested parameters
    and then adds the specified number of control qubits.

    Args:
        num_controls: Number of control qubits.
        n: Number of index qubits.
        m: Number of fractional phi bits.
        p: Number of fractional theta bits.
        q: Fixed coefficient index q.
        t: Input array of node values.
        s: Input array of grid indices or offsets.
        qiskit_matches_paper_rx: Whether RX angles match Qiskit conventions.
        qiskit_matches_paper_rz: Whether RZ angles match Qiskit conventions.
        uncompute_angles: Whether angle registers are uncomputed.
        int_bits_phi: Number of integer bits for phi.
        int_bits_theta: Number of integer bits for theta.
        loader_mode: Loader synthesis mode.

    Returns:
        A controlled Gate implementing the requested U_q block.

    Raises:
        ValueError: Propagated from the underlying U_q gate builder if the
            input parameters are invalid.
    """
    base_gate = _build_uq_gate(
        n=n,
        m=m,
        p=p,
        q=q,
        t=t,
        s=s,
        qiskit_matches_paper_rx=qiskit_matches_paper_rx,
        qiskit_matches_paper_rz=qiskit_matches_paper_rz,
        uncompute_angles=uncompute_angles,
        int_bits_phi=int_bits_phi,
        int_bits_theta=int_bits_theta,
        loader_mode=loader_mode,
    )
    return base_gate.control(num_controls)


def U_ur_lcu_circuit(
    *,
    n: int,
    m: int,
    p: int,
    r: int,
    K: int,
    t: np.ndarray,
    s: np.ndarray,
    qiskit_matches_paper_rx: bool = True,
    qiskit_matches_paper_rz: bool = True,
    uncompute_angles: bool = True,
    loader_mode: str = "dense",
) -> Tuple[QuantumCircuit, float]:
    """Construct the LCU-based circuit implementing U_{u_r}.

    This routine forms the modified coefficient vector alpha'_{q,r}, prepares
    the PREP and phase gates for the LCU decomposition, identifies the active
    q values, assembles the controlled U_q blocks, and returns the resulting
    circuit together with the normalization factor alpha_ur.

    Args:
        n: Number of index qubits.
        m: Number of fractional phi bits.
        p: Number of fractional theta bits.
        r: Fixed coefficient index r.
        K: Number of LCU terms. This implementation assumes K is a power of two.
        t: Input array of node values.
        s: Input array of grid indices or offsets.
        qiskit_matches_paper_rx: Whether RX angles are doubled to match Qiskit
            conventions.
        qiskit_matches_paper_rz: Whether RZ angles are doubled to match Qiskit
            conventions.
        uncompute_angles: Whether loaded angle registers are uncomputed.
        loader_mode: Loader synthesis mode, either ``"dense"`` or ``"mcx"``.

    Returns:
        A tuple containing the constructed QuantumCircuit and the normalization
        factor alpha_ur.

    Raises:
        ValueError: If K is less than 1.
        ValueError: If K is not a power of two.
    """
    mode = _normalize_loader_mode(loader_mode)

    if K < 1:
        raise ValueError("K must be >= 1.")

    L = int(np.ceil(np.log2(K)))
    if 2**L != K:
        raise ValueError("This helper assumes K is a power of two for simplicity.")

    coeffs = alpha_vec_prime(r=r, K=K)
    prep, alpha_ur, phases = _prep_gate_for_coeffs(coeffs)
    phase_gate = _diag_phase_gate(phases)

    tol = 1e-12
    active_qs = [q for q, coeff in enumerate(coeffs) if abs(coeff) > tol]

    qreg = QuantumRegister(L, "q")
    jreg = QuantumRegister(n, "j")
    phi_reg = QuantumRegister(1 + 2 + m, "phi")
    theta_reg = QuantumRegister(2 + p, "theta")
    tgt = QuantumRegister(1, "tgt")

    qc = QuantumCircuit(qreg, jreg, phi_reg, theta_reg, tgt, name=f"U_ur_LCU(r={r})")
    qc.append(prep, qreg)

    for q in active_qs:
        bits = [(q >> k) & 1 for k in range(L)]

        for k, bit in enumerate(bits):
            if bit == 0:
                qc.x(qreg[k])

        cUq = _build_controlled_uq_gate(
            num_controls=L,
            n=n,
            m=m,
            p=p,
            q=q,
            t=t,
            s=s,
            qiskit_matches_paper_rx=qiskit_matches_paper_rx,
            qiskit_matches_paper_rz=qiskit_matches_paper_rz,
            uncompute_angles=uncompute_angles,
            int_bits_phi=2,
            int_bits_theta=2,
            loader_mode=mode,
        )
        qc.append(
            cUq,
            list(qreg) + list(jreg) + list(phi_reg) + list(theta_reg) + list(tgt),
        )

        for k, bit in enumerate(bits):
            if bit == 0:
                qc.x(qreg[k])

    qc.append(phase_gate, qreg)
    qc.append(prep.inverse(), qreg)

    return qc, alpha_ur