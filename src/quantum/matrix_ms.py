r"""
This module implements an exact block encoding of the matrix M_s used in the
NUQFT factorization, specialized to the structure of M_s itself.

For N = 2^n and an integer vector
    s = (s_0, s_1, ..., s_{N-1}),   0 <= s_j <= N - 1,
the matrix M_s is defined by
    (M_s)_{i,j} =
        1,  if i = s_j,
        0,  otherwise.

Equivalently, M_s maps |j⟩ to |s_j⟩. Each column therefore contains exactly one
nonzero entry, while the row sparsity is determined by the multiplicity vector
c_s, where c_s[i] counts the number of occurrences of i in s.

Important clarification:
    This module does not implement the full generic sparse block lemma circuit
    from the paper. In particular, it does not explicitly use the generic
    sparse-access oracle triplet
        • O_c : column oracle,
        • O_r : row oracle,
        • O_A : entry oracle,
    nor does it reproduce the full ancilla structure of the lemma.

    Instead, this module implements a direct specialization to M_s that exploits
    the fact that every column has exactly one nonzero entry. The resulting
    circuit has the same target block and the same normalization
        alpha = sqrt(||c_s||_inf),
    but it is a simpler M_s-specific construction rather than the literal
    oracle-based sparse block lemma circuit.

Construction used here:
    For each column index j, define
        • row_j  := s_j,
        • rank_j := #{ j' < j : s_{j'} = s_j }.

    The map
        j -> (rank_j, row_j)
    is injective. The module constructs a unitary W_R satisfying
        |0^(n+1)⟩_a |j⟩ -> |rank_j⟩_a |row_j⟩.

    It also uses a prefix-uniform state preparation G_{d_r}, where
        d_r := ||c_s||_inf,
    such that
        G_{d_r} |0^(n+1)⟩
            = (1 / sqrt(d_r)) * sum_{k=0}^{d_r - 1} |k⟩.

    The final block encoding is
        U_Ms := (G_{d_r}^\dagger ⊗ I) W_R.

    Post-selecting the ancilla register on |0^(n+1)⟩ yields
        (⟨0^(n+1)|_a ⊗ I) U_Ms (|0^(n+1)⟩_a ⊗ I)
            = M_s / sqrt(d_r).

Ancilla usage:
    The construction uses
        • n qubits for the system register,
        • n+1 qubits for the ancilla register.
    Hence the block encoding acts on a total of 2n+1 qubits.

Register conventions:
    • System register j has n qubits.
    • Ancilla register a has n+1 qubits.
    • Gates in this module act on ordered registers
          a[:] + j[:]
      unless explicitly stated otherwise.

Implementation approach:
    The right isometry W_R is implemented as an exact dense permutation unitary
    wrapped as a Qiskit UnitaryGate. The prefix-uniform state G_{d_r} is
    implemented with Qiskit's StatePreparation.

    This approach is intended for small and medium simulation sizes where dense
    unitaries are acceptable and transparent for debugging.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import StatePreparation, UnitaryGate


# -----------------------------------------------------------------------------
# Module-level caches
# -----------------------------------------------------------------------------

_WR_GATE_CACHE: Dict[Tuple[int, ...], Gate] = {}
_GDR_GATE_CACHE: Dict[Tuple[int, int], Gate] = {}
_UMS_GATE_CACHE: Dict[Tuple[int, ...], Gate] = {}


# -----------------------------------------------------------------------------
# Validation and classical preprocessing
# -----------------------------------------------------------------------------

def _is_power_of_two(N: int) -> bool:
    """Check whether a positive integer is a power of two.

    Args:
        N: Positive integer to test.

    Returns:
        True if N is a power of two and False otherwise.

    Raises:
        None.
    """
    return N > 0 and (N & (N - 1) == 0)


def _normalize_s(s: Sequence[int]) -> np.ndarray:
    """Validate and normalize the sampling-index vector s.

    Args:
        s: Sequence of integer row indices defining M_s.

    Returns:
        A one-dimensional NumPy integer array containing the validated entries
        of s.

    Raises:
        ValueError: If s is empty.
        ValueError: If len(s) is not a power of two.
        ValueError: If s is not one-dimensional.
        ValueError: If an entry of s lies outside {0, ..., N - 1}.
    """
    s_arr = np.asarray(s, dtype=int)

    if s_arr.ndim != 1:
        raise ValueError("s must be a one-dimensional sequence of integers.")
    if s_arr.size == 0:
        raise ValueError("s must be nonempty.")

    N = int(s_arr.size)
    if not _is_power_of_two(N):
        raise ValueError("len(s) must be a power of two so that N = 2^n.")

    if np.any(s_arr < 0) or np.any(s_arr >= N):
        raise ValueError("All entries of s must satisfy 0 <= s_j < len(s).")

    return s_arr


def infer_n_from_s(s: Sequence[int]) -> int:
    """Infer the number of system qubits n from the length of s.

    Args:
        s: Sequence defining M_s with length N = 2^n.

    Returns:
        The integer n such that len(s) = 2^n.

    Raises:
        ValueError: If s is invalid.
    """
    s_arr = _normalize_s(s)
    return int(np.log2(len(s_arr)))


def counts_vector_from_s(s: Sequence[int]) -> np.ndarray:
    """Compute the row-multiplicity vector c_s associated with s.

    Args:
        s: Sequence defining M_s.

    Returns:
        A length-N integer array c such that c[i] = #{j : s_j = i}.

    Raises:
        ValueError: If s is invalid.
    """
    s_arr = _normalize_s(s)
    N = int(len(s_arr))
    return np.bincount(s_arr, minlength=N)


def row_sparsity_from_s(s: Sequence[int]) -> int:
    """Return the row sparsity d_r = ||c_s||_inf of M_s.

    Args:
        s: Sequence defining M_s.

    Returns:
        The maximum row multiplicity.

    Raises:
        ValueError: If s is invalid.
    """
    c = counts_vector_from_s(s)
    return int(np.max(c))


def alpha_ms_from_s(s: Sequence[int]) -> float:
    """Return the block-encoding normalization alpha = sqrt(||c_s||_inf).

    Args:
        s: Sequence defining M_s.

    Returns:
        The scalar alpha = sqrt(row_sparsity_from_s(s)).

    Raises:
        ValueError: If s is invalid.
    """
    return float(np.sqrt(row_sparsity_from_s(s)))


def inverse_buckets_from_s(s: Sequence[int]) -> List[List[int]]:
    """Construct the inverse bucket structure associated with s.

    For each row i, this routine builds the ordered list
        buckets[i] = [j : s_j = i].

    Args:
        s: Sequence defining M_s.

    Returns:
        A length-N Python list whose i-th entry contains the list of column
        indices j such that s_j = i, ordered increasingly in j.

    Raises:
        ValueError: If s is invalid.
    """
    s_arr = _normalize_s(s)
    N = int(len(s_arr))

    buckets: List[List[int]] = [[] for _ in range(N)]
    for j, sj in enumerate(s_arr.tolist()):
        buckets[int(sj)].append(int(j))

    return buckets


def occurrence_ranks_from_s(s: Sequence[int]) -> np.ndarray:
    """Compute the occurrence rank of each column within its target row.

    For each column index j, define
        rank_j = #{j' < j : s_{j'} = s_j}.

    This quantity records the position of column j inside the ordered bucket
    associated with the row label s_j.

    Example:
        If s = [1, 1, 2, 1, 2], then the returned ranks are
            [0, 1, 0, 2, 1].

    Args:
        s: Sequence defining M_s.

    Returns:
        A length-N integer array whose j-th entry is the occurrence rank of
        column j among all columns mapping to the same row.

    Raises:
        ValueError: If s is invalid.
    """
    s_arr = _normalize_s(s)
    N = int(len(s_arr))
    seen = np.zeros(N, dtype=int)
    ranks = np.zeros(N, dtype=int)

    for j, row in enumerate(s_arr.tolist()):
        row = int(row)
        ranks[j] = seen[row]
        seen[row] += 1

    return ranks


def reference_ms_matrix(s: Sequence[int]) -> np.ndarray:
    """Construct the dense classical matrix M_s.

    Args:
        s: Sequence defining M_s.

    Returns:
        A dense N x N complex NumPy array with entries
            (M_s)_{i,j} = 1  iff  i = s_j.

    Raises:
        ValueError: If s is invalid.
    """
    s_arr = _normalize_s(s)
    N = int(len(s_arr))
    M = np.zeros((N, N), dtype=complex)
    for j, i in enumerate(s_arr.tolist()):
        M[int(i), int(j)] = 1.0
    return M


# -----------------------------------------------------------------------------
# State-preparation helper
# -----------------------------------------------------------------------------

def get_uniform_prefix_prep_gate(
    d: int,
    num_qubits: int,
    *,
    label: Optional[str] = None,
) -> Gate:
    """Retrieve or construct a state-preparation gate for a uniform prefix state.

    The returned object prepares
        (1 / sqrt(d)) * sum_{k=0}^{d-1} |k>
    on a num_qubits-qubit register.

    Args:
        d: Number of basis states included in the uniform prefix.
        num_qubits: Number of qubits in the target register.
        label: Optional gate label. If omitted, a default label is used.

    Returns:
        A cached or newly constructed gate that prepares the uniform prefix
        state over the first d computational basis states.

    Raises:
        ValueError: If d < 1.
        ValueError: If d > 2^num_qubits.
    """
    if d < 1:
        raise ValueError("d must be >= 1.")
    if d > 2 ** num_qubits:
        raise ValueError("d must satisfy d <= 2^num_qubits.")

    key = (int(d), int(num_qubits))
    cached = _GDR_GATE_CACHE.get(key)
    if cached is not None:
        return cached

    state = np.zeros(2 ** num_qubits, dtype=complex)
    state[:d] = 1.0 / np.sqrt(float(d))

    gate = StatePreparation(state)
    gate.label = label or f"G_{d}"

    _GDR_GATE_CACHE[key] = gate
    return gate


# -----------------------------------------------------------------------------
# Right-isometry oracle
# -----------------------------------------------------------------------------

def build_ms_right_oracle_unitary(s: Sequence[int]) -> np.ndarray:
    """Construct the dense unitary W_R used by the M_s-specific block encoding.

    The unitary acts on registers ordered as
        [a, j],
    where
        • a is an (n+1)-qubit ancilla register,
        • j is an n-qubit system register.

    Its defining action on the input subspace with ancilla initialized to
    |0^(n+1)⟩ is
        |0^(n+1)⟩|j⟩ -> |rank_j⟩|s_j⟩,
    where rank_j is the occurrence rank of column j among all columns sharing
    the same target row s_j.

    Since the map j -> (rank_j, s_j) is injective, this defines an isometry on
    the ancilla-zero subspace. The routine extends that isometry to a full
    permutation unitary on the entire ancilla-plus-system Hilbert space.

    Basis-indexing convention:
        The total Hilbert space is ordered as ancilla ⊗ system, with basis
        states indexed by
            total_index = anc_state + (2^(n+1)) * sys_state.

    Args:
        s: Sequence defining M_s.

    Returns:
        A dense permutation unitary implementing the required right isometry
        extension W_R.

    Raises:
        ValueError: If s is invalid.
    """
    s_arr = _normalize_s(s)
    N = int(len(s_arr))
    anc_dim = 2 * N
    dim = anc_dim * N

    ranks = occurrence_ranks_from_s(s_arr)

    perm = [-1] * dim
    used_outputs = set()

    for j in range(N):
        in_idx = anc_dim * j
        out_idx = int(ranks[j]) + anc_dim * int(s_arr[j])
        perm[in_idx] = out_idx
        used_outputs.add(out_idx)

    remaining_inputs = [x for x in range(dim) if perm[x] == -1]
    remaining_outputs = [y for y in range(dim) if y not in used_outputs]

    for x, y in zip(remaining_inputs, remaining_outputs):
        perm[x] = y

    U = np.zeros((dim, dim), dtype=complex)
    for in_idx, out_idx in enumerate(perm):
        U[out_idx, in_idx] = 1.0

    return U


def get_ms_right_oracle_gate(
    s: Sequence[int],
    *,
    label: Optional[str] = None,
) -> Gate:
    """Retrieve or construct the cached right-isometry gate W_R.

    The returned gate acts on registers ordered as
        a[:] + j[:],
    where len(a) = n+1 and len(j) = n.

    On the ancilla-zero input subspace, it satisfies
        |0^(n+1)⟩|j⟩ -> |rank_j⟩|s_j⟩.

    Args:
        s: Sequence defining M_s.
        label: Optional gate label. If omitted, a default label is used.

    Returns:
        A cached or newly constructed UnitaryGate implementing W_R.

    Raises:
        ValueError: If s is invalid.
    """
    s_arr = _normalize_s(s)
    key = tuple(int(x) for x in s_arr.tolist())

    cached = _WR_GATE_CACHE.get(key)
    if cached is not None:
        return cached

    n = infer_n_from_s(s_arr)
    U = build_ms_right_oracle_unitary(s_arr)
    gate = UnitaryGate(U, label=label or f"W_R_Ms(n={n})")

    _WR_GATE_CACHE[key] = gate
    return gate


# -----------------------------------------------------------------------------
# Block encoding of M_s
# -----------------------------------------------------------------------------

def get_ms_sparse_block_encoding_gate(
    s: Sequence[int],
    *,
    label: Optional[str] = None,
) -> Gate:
    r"""Retrieve or construct the M_s-specific exact block encoding gate.

    This routine implements a direct block encoding specialized to the matrix
    M_s. It should be viewed as an adaptation inspired by the sparse-access
    viewpoint, but not as the literal generic sparse block lemma circuit from
    the paper.

    Let
        d_r = ||c_s||_inf.
    Define
        W_R : |0^(n+1)⟩|j⟩ -> |rank_j⟩|s_j⟩,
    and let G_{d_r} prepare the uniform prefix state
        G_{d_r}|0^(n+1)⟩
            = (1 / sqrt(d_r)) * sum_{k=0}^{d_r-1} |k⟩.

    The returned gate is
        U_Ms = (G_{d_r}^\dagger ⊗ I) W_R.

    Acting on registers ordered as
        a[:] + j[:],
    this unitary satisfies
        (⟨0^(n+1)|_a ⊗ I_j) U_Ms (|0^(n+1)⟩_a ⊗ I_j)
            = M_s / sqrt(d_r).

    Args:
        s: Sequence defining M_s.
        label: Optional gate label. If omitted, a default label is used.

    Returns:
        A Qiskit Gate implementing the exact M_s-specific block encoding.

    Raises:
        ValueError: If s is invalid.
    """
    s_arr = _normalize_s(s)
    key = tuple(int(x) for x in s_arr.tolist())

    cached = _UMS_GATE_CACHE.get(key)
    if cached is not None:
        return cached

    n = infer_n_from_s(s_arr)
    d_r = row_sparsity_from_s(s_arr)

    qc = QuantumCircuit((n + 1) + n, name=label or f"U_Ms_sparse(n={n})")

    anc = list(range(0, n + 1))
    sys = list(range(n + 1, (n + 1) + n))

    Wr = get_ms_right_oracle_gate(s_arr)
    Gdr = get_uniform_prefix_prep_gate(d=d_r, num_qubits=n + 1)

    qc.append(Wr, anc + sys)
    qc.append(Gdr.inverse(), anc)

    gate = qc.to_gate(label=label or f"U_Ms_sparse(n={n})")
    _UMS_GATE_CACHE[key] = gate
    return gate


def U_ms_circuit_no_x(
    *,
    s: Sequence[int],
    j,
    a,
    qc: QuantumCircuit,
    label: Optional[str] = None,
) -> QuantumCircuit:
    """Append the exact M_s block encoding to an existing quantum circuit.

    This routine appends the cached gate returned by
    get_ms_sparse_block_encoding_gate(...) to the supplied QuantumCircuit.
    The circuit uses
        • j : n-qubit system register,
        • a : (n+1)-qubit ancilla register.

    The registers are used in the order
        a[:] + j[:],
    and the resulting unitary U satisfies
        (⟨0^(n+1)|_a ⊗ I_j) U (|0^(n+1)⟩_a ⊗ I_j)
            = M_s / sqrt(||c_s||_inf).

    Args:
        s: Sequence defining M_s.
        j: Quantum register containing the n system qubits.
        a: Quantum register containing the n+1 ancilla qubits.
        qc: QuantumCircuit object to which the operations are appended.
        label: Optional custom label for the cached gate.

    Returns:
        The modified QuantumCircuit containing the M_s block encoding.

    Raises:
        ValueError: If the provided registers do not have the required sizes.
        ValueError: If s is invalid.
    """
    n = infer_n_from_s(s)

    if len(j) != n:
        raise ValueError(f"len(j) must be {n}")
    if len(a) != n + 1:
        raise ValueError(f"len(a) must be {n + 1}")

    gate = get_ms_sparse_block_encoding_gate(s, label=label)
    qc.append(gate, list(a) + list(j))
    return qc


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def ms_block_encoding_summary(s: Sequence[int]) -> dict:
    """Return metadata describing the M_s block encoding instance.

    Args:
        s: Sequence defining M_s.

    Returns:
        A dictionary containing:
            - n
            - N
            - counts
            - row_sparsity
            - column_sparsity
            - alpha
            - ancilla_qubits
            - occurrence_ranks

    Raises:
        ValueError: If s is invalid.
    """
    s_arr = _normalize_s(s)
    c = counts_vector_from_s(s_arr)
    n = infer_n_from_s(s_arr)
    ranks = occurrence_ranks_from_s(s_arr)

    return {
        "n": n,
        "N": int(len(s_arr)),
        "counts": c,
        "row_sparsity": int(np.max(c)),
        "column_sparsity": 1,
        "alpha": float(np.sqrt(np.max(c))),
        "ancilla_qubits": n + 1,
        "occurrence_ranks": ranks,
    }


def clear_ms_cache() -> None:
    """Clear all module-level caches used by this module.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    _WR_GATE_CACHE.clear()
    _GDR_GATE_CACHE.clear()
    _UMS_GATE_CACHE.clear()