"""
Arccos from signed-magnitude input (fully quantum)
--------------------------------------------------
This module builds a fully-quantum circuit that computes arccos(x) from a
signed-magnitude fixed-point input.

The input x is provided as a signed-magnitude fixed-point number:
  - magReg (m qubits) encodes the magnitude |x| as an unsigned fixed-point
    value |x| = u / 2^(m-1), so |x| ∈ [0, 1).
  - signQ (1 qubit) encodes the sign: |0⟩ means x ≥ 0, |1⟩ means x < 0.

The circuit outputs an unsigned fixed-point approximation of arccos(x):
  - thetaReg (w = p + 3 qubits) stores an UNSIGNED integer approximating
    arccos(x) · 2^p. The represented angle is approximately
    int(thetaReg) / 2^p. Since arccos(x) ∈ [0, π], the output is always
    nonnegative; the extra 3 bits provide headroom to avoid modular wrap.

Method:
  1) Map magReg into the CORDIC input register tReg at width n = p + 1 via
     a power-of-two shift, implementing t = u · 2^(p-m) (with truncation if
     p < m).
  2) Run a fully-quantum CORDIC arcsin core to produce direction bits dReg
     encoding asin(|x|). Optionally uncompute CORDIC work registers (leaving
     dReg intact) to clean ancillas.
  3) Accumulate arccos(x) from π/2 and the CORDIC micro-rotation constants
     using ripple-carry constant adders, implementing:
         arccos(x) = π/2 - asin(x)
                  = π/2 - asin(|x|)   if signQ = 0
                    π/2 + asin(|x|)   if signQ = 1.

Notes:
  - Fully quantum: no measurements are used.
  - Uses CDKMRippleCarryAdder(kind="fixed") to add precomputed constants
    into thetaReg.
  - The accumulation step flips signQ during constant-add selection (and
    flips it back) to match the tested sign convention described in the
    implementation comments.
"""

from __future__ import annotations

from functools import cache
from typing import Union, Sequence, Optional, List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import CDKMRippleCarryAdder


# =============================================================================
# Minimal utilities
# =============================================================================

def _rname(register: Union[QuantumRegister, Sequence]) -> str:
    """Return the name of the underlying QuantumRegister.

    Args:
        register: A QuantumRegister or a sequence of qubits belonging to one.

    Returns:
        The register name as a string.

    Raises:
        None.
    """
    return register[0]._register.name


@cache
def fib(i: int) -> int:
    """Compute the i-th Fibonacci number.

    This sequence is used by the multiplication routine to determine
    shift-and-add schedules.

    Args:
        i: Index of the Fibonacci number (nonnegative integer).

    Returns:
        The i-th Fibonacci number.

    Raises:
        None.
    """
    # Fibonacci numbers as used by the provided multiplication routine.
    return int(
        np.round(
            1 / (5 ** 0.5)
            * (
                ((1 + 5 ** 0.5) / 2) ** (i + 1)
                - ((1 - 5 ** 0.5) / 2) ** (i + 1)
            )
        )
    )


def _int_to_twos_comp(val: int, width: int) -> int:
    """Convert a signed integer to its two's-complement representation.

    The result is returned as an unsigned integer in the range [0, 2^width).

    Args:
        val: Signed integer value.
        width: Bit width of the two's-complement representation.

    Returns:
        Unsigned integer encoding val modulo 2^width.

    Raises:
        None.
    """
    return val % (1 << width)


def _xor_load_constant(
    circ: QuantumCircuit,
    reg: QuantumRegister,
    const_unsigned: int
) -> None:
    """Load or unload a classical constant into a quantum register using XOR.

    This routine maps |0⟩ → |const⟩ (and vice versa) by applying X gates to
    qubits corresponding to 1-bits in the little-endian binary expansion.

    Args:
        circ: QuantumCircuit to modify.
        reg: Target quantum register.
        const_unsigned: Unsigned integer constant to load.

    Returns:
        None.

    Raises:
        None.
    """
    for i in range(len(reg)):
        if (const_unsigned >> i) & 1:
            circ.x(reg[i])


# =============================================================================
# Core arithmetic primitives (logic unchanged)
# =============================================================================

def additionGate(
    xReg: Union[QuantumRegister, Sequence],
    yReg: Union[QuantumRegister, Sequence],
) -> Gate:
    """Construct an in-place quantum addition gate.

    Implements the mapping (x, y) → (x + y, y) using a two's-complement–friendly
    ripple-carry construction.

    Args:
        xReg: Quantum register holding the addend and output sum.
        yReg: Quantum register holding the addend and preserved output.

    Returns:
        A Gate implementing in-place addition on xReg.

    Raises:
        None.
    """
    n = min(len(xReg), len(yReg))
    circuit = QuantumCircuit(
        xReg, yReg,
        name=f"({_rname(xReg)}+{_rname(yReg)},{_rname(yReg)})"
    )

    # Step 1
    for i in range(1, n):
        circuit.cx(yReg[i], xReg[i])

    # Step 2
    for i in range(n - 2, 0, -1):
        circuit.cx(yReg[i], yReg[i + 1])

    # Step 3
    for i in range(n - 1):
        circuit.ccx(yReg[i], xReg[i], yReg[i + 1])

    # Step 4
    for i in range(n - 1, 0, -1):
        circuit.cx(yReg[i], xReg[i])
        circuit.ccx(yReg[i - 1], xReg[i - 1], yReg[i])

    # Step 5
    for i in range(1, n - 1):
        circuit.cx(yReg[i], yReg[i + 1])

    # Step 6
    for i in range(n):
        circuit.cx(yReg[i], xReg[i])

    return circuit.to_gate()


def shiftAdditionGate(xReg: QuantumRegister, yReg: QuantumRegister, rshift: int = 1) -> Gate:
    """Construct an in-place shifted addition gate.

    Implements the mapping (x, y) → (x + (y >> rshift), y).

    Args:
        xReg: Quantum register receiving the shifted sum.
        yReg: Quantum register whose value is right-shifted and added.
        rshift: Number of bits to right-shift yReg.

    Returns:
        A Gate implementing the shifted addition.

    Raises:
        None.
    """
    circuit = QuantumCircuit(
        xReg, yReg,
        name=f"({_rname(xReg)}+{_rname(yReg)}/{1 << rshift},{_rname(yReg)})"
    )

    circuit.append(
        additionGate(xReg, yReg),
        xReg[:] + yReg[rshift:] + yReg[:rshift],
    )
    circuit.append(
        additionGate(xReg[-rshift:], yReg[-1:] + yReg[: rshift - 1]).inverse(),
        xReg[-rshift:] + yReg[-1:] + yReg[: rshift - 1],
    )
    circuit.append(
        additionGate(xReg[-rshift:], yReg[:rshift]),
        xReg[-rshift:] + yReg[:rshift],
    )

    return circuit.to_gate()


def multGate(
    xReg: QuantumRegister,
    aux: QuantumRegister,
    m: int,
    gate: bool = True
) -> Union[Gate, QuantumCircuit]:
    """Multiply a register by (1 + 2^{-m}) in place.

    The multiplication is implemented via a sequence of controlled shift-add
    operations using an auxiliary register, following the provided
    Fibonacci-based schedule.

    Args:
        xReg: Quantum register holding the multiplicand and result.
        aux: Auxiliary quantum register used during computation.
        m: Shift parameter defining the factor (1 + 2^{-m}).
        gate: If True, return a Gate; otherwise return a QuantumCircuit.

    Returns:
        A Gate or QuantumCircuit implementing the multiplication.

    Raises:
        None.
    """
    n = len(xReg)
    circuit = QuantumCircuit(xReg, aux, name=f"{_rname(xReg)}(1+1/{1 << m})")

    numIter = 2 * int(0.5 * np.sqrt(5) * n / ((1 + np.sqrt(5)) / 2) ** (m))

    if m >= n:
        return circuit.to_gate() if gate else circuit

    circuit.append(additionGate(aux, xReg), aux[:] + xReg[:])

    for i in range(numIter, -1, -1):
        if m * fib(i) >= n:
            continue

        negative = (fib(i) % 2 == 1)
        if i % 2 == 0:
            circuit.append(
                shiftAdditionGate(xReg, aux, m * fib(i)) if negative
                else shiftAdditionGate(xReg, aux, m * fib(i)).inverse(),
                xReg[:] + aux[:],
            )
        else:
            circuit.append(
                shiftAdditionGate(aux, xReg, m * fib(i)) if negative
                else shiftAdditionGate(aux, xReg, m * fib(i)).inverse(),
                aux[:] + xReg[:],
            )

    circuit.append(additionGate(aux, xReg).inverse(), aux[:] + xReg[:])

    return circuit.to_gate() if gate else circuit


# =============================================================================
# CORDIC arcsin core + cleanup (logic unchanged)
# =============================================================================

def quantumCORDIC(
    tReg: QuantumRegister,
    xReg: QuantumRegister,
    yReg: QuantumRegister,
    multReg: QuantumRegister,
    dReg: QuantumRegister,
    gate: bool = True,
) -> Union[Gate, QuantumCircuit]:
    """Execute the fully-quantum CORDIC routine for arcsin.

    The routine computes direction bits dReg encoding asin(t / 2^(n-2)) while
    updating internal working registers.

    Args:
        tReg: Input register encoding the scaled argument.
        xReg: CORDIC x-register.
        yReg: CORDIC y-register.
        multReg: Auxiliary register used for multiplication.
        dReg: Register storing CORDIC direction bits.
        gate: If True, return a Gate; otherwise return a QuantumCircuit.

    Returns:
        A Gate or QuantumCircuit implementing the CORDIC arcsin core.

    Raises:
        None.
    """
    n = len(tReg)
    circuit = QuantumCircuit(tReg, xReg, yReg, multReg, dReg, name="asin(t)")

    # Set x = 1 in two's complement fixed point for range=[-2,2)
    circuit.x(xReg[-2])

    for i in range(1, n):
        # Infer next rotation direction
        circuit.append(additionGate(tReg, yReg).inverse(), tReg[:] + yReg[:])
        circuit.ccx(xReg[-1], yReg[-1], dReg[-i])
        circuit.ccx(xReg[-1], tReg[-1], dReg[-i])
        circuit.cx(xReg[-1], dReg[-i])
        circuit.cx(tReg[-1], dReg[-i])
        circuit.append(additionGate(tReg, yReg), tReg[:] + yReg[:])

        # Reflect depending on direction
        for j in range(n):
            circuit.cx(dReg[-i], yReg[j])

        # Do rotation block twice
        for _ in range(2):
            circuit.append(shiftAdditionGate(xReg, yReg, i).inverse(), xReg[:] + yReg[:])
            circuit.append(multGate(yReg, multReg, 2 * i), yReg[:] + multReg[:])
            circuit.append(shiftAdditionGate(yReg, xReg, i), yReg[:] + xReg[:])

        # Undo reflection
        for j in range(n):
            circuit.cx(dReg[-i], yReg[j])

        # Compensate for imperfect rotation
        circuit.append(multGate(tReg, multReg, 2 * i), tReg[:] + multReg[:])

    return circuit.to_gate() if gate else circuit


def invRepairCORDIC(
    tReg: QuantumRegister,
    xReg: QuantumRegister,
    yReg: QuantumRegister,
    multReg: QuantumRegister,
    dReg: QuantumRegister,
    gate: bool = True,
) -> Union[Gate, QuantumCircuit]:
    """Construct the inverse repair routine for CORDIC ancillas.

    This circuit uncomputes CORDIC work registers while preserving the
    direction bits dReg.

    Args:
        tReg: Input register used by the CORDIC routine.
        xReg: CORDIC x-register.
        yReg: CORDIC y-register.
        multReg: Auxiliary multiplication register.
        dReg: Register storing CORDIC direction bits.
        gate: If True, return a Gate; otherwise return a QuantumCircuit.

    Returns:
        A Gate or QuantumCircuit implementing the inverse repair.

    Raises:
        None.
    """
    n = len(tReg)
    circuit = QuantumCircuit(tReg, xReg, yReg, multReg, dReg, name="cleanAux+Input")

    circuit.x(xReg[-2])

    for i in range(1, n):
        for j in range(n):
            circuit.cx(dReg[-i], yReg[j])

        for _ in range(2):
            circuit.append(shiftAdditionGate(xReg, yReg, i).inverse(), xReg[:] + yReg[:])
            circuit.append(multGate(yReg, multReg, 2 * i), yReg[:] + multReg[:])
            circuit.append(shiftAdditionGate(yReg, xReg, i), yReg[:] + xReg[:])

        for j in range(n):
            circuit.cx(dReg[-i], yReg[j])

        circuit.append(multGate(tReg, multReg, 2 * i), tReg[:] + multReg[:])

    return circuit.to_gate() if gate else circuit


def repairCORDIC(
    tReg: QuantumRegister,
    xReg: QuantumRegister,
    yReg: QuantumRegister,
    multReg: QuantumRegister,
    dReg: QuantumRegister,
    gate: bool = True,
) -> Union[Gate, QuantumCircuit]:
    """Uncompute CORDIC ancillas after arcsin computation.

    This applies the adjoint of the inverse repair routine, returning all
    CORDIC work registers (except dReg) to |0⟩.

    Args:
        tReg: Input register used by the CORDIC routine.
        xReg: CORDIC x-register.
        yReg: CORDIC y-register.
        multReg: Auxiliary multiplication register.
        dReg: Register storing CORDIC direction bits.
        gate: If True, return a Gate; otherwise return a QuantumCircuit.

    Returns:
        A Gate or QuantumCircuit implementing the repair operation.

    Raises:
        None.
    """
    circuit = QuantumCircuit(tReg, xReg, yReg, multReg, dReg, name="repair")
    circuit.append(
        invRepairCORDIC(tReg, xReg, yReg, multReg, dReg, gate=True).inverse(),
        tReg[:] + xReg[:] + yReg[:] + multReg[:] + dReg[:],
    )
    return circuit.to_gate() if gate else circuit


# =============================================================================
# Ripple-carry constant addition into thetaReg (fully quantum)
# =============================================================================

def _rca_add_constant(
    circ: QuantumCircuit,
    thetaReg: QuantumRegister,
    cReg: QuantumRegister,
    helpQ: QuantumRegister,
    const_unsigned: int,
    ctrls: Optional[Union["qiskit.circuit.Qubit", List["qiskit.circuit.Qubit"]]] = None,
) -> None:
    """Add a classical constant into a quantum register using a ripple-carry adder.

    Performs in-place modular addition:
        thetaReg ← thetaReg + const_unsigned  (mod 2^w)

    Optionally applies the addition under one or more control qubits.

    Args:
        circ: QuantumCircuit to modify.
        thetaReg: Target register receiving the sum.
        cReg: Scratch register used to hold the constant.
        helpQ: Single-qubit helper required by the CDKM adder.
        const_unsigned: Unsigned integer constant to add.
        ctrls: Optional control qubit or list of control qubits.

    Returns:
        None.

    Raises:
        ValueError: If cReg does not match the width of thetaReg, or if helpQ
            is not 1 qubit.
    """
    w = len(thetaReg)
    if len(cReg) != w:
        raise ValueError("cReg must have same width as thetaReg.")
    if len(helpQ) != 1:
        raise ValueError("helpQ must be a 1-qubit register.")

    # Normalize control list
    ctrl_list: Optional[List] = None
    if ctrls is None:
        ctrl_list = None
    elif isinstance(ctrls, list):
        ctrl_list = ctrls
    else:
        ctrl_list = [ctrls]

    # Load constant into cReg
    _xor_load_constant(circ, cReg, const_unsigned)

    adder_gate = CDKMRippleCarryAdder(w, kind="fixed").to_gate(label="CDKM_add")

    if ctrl_list is None:
        circ.append(adder_gate, cReg[:] + thetaReg[:] + helpQ[:])
    else:
        circ.append(adder_gate.control(len(ctrl_list)), ctrl_list + cReg[:] + thetaReg[:] + helpQ[:])

    # Unload constant from cReg
    _xor_load_constant(circ, cReg, const_unsigned)


def _controlled_on_zero(circ: QuantumCircuit, ctrl_qubit, body_fn) -> None:
    """Execute an operation conditioned on a control qubit being |0⟩.

    This is implemented by temporarily flipping the control qubit, executing
    the body, and flipping it back.

    Args:
        circ: QuantumCircuit to modify.
        ctrl_qubit: Control qubit tested for the |0⟩ state.
        body_fn: Callable that appends operations to circ.

    Returns:
        None.

    Raises:
        None.
    """
    circ.x(ctrl_qubit)
    body_fn()
    circ.x(ctrl_qubit)


# =============================================================================
# Main builder: arccos(x) for signed-magnitude input
# =============================================================================

def build_arccos_signedmag_circuit(
    m: int,
    p: int,
    clean_cordic_ancillas: bool = True,
) -> QuantumCircuit:
    """Construct a fully-quantum circuit computing arccos(x) from signed-magnitude input.

    The input x is represented by a magnitude register and a sign qubit:
      - magReg (m qubits): |x| = u / 2^(m-1), with |x| ∈ [0, 1).
      - signQ (1 qubit): |0⟩ for x ≥ 0, |1⟩ for x < 0.

    The output is an unsigned fixed-point approximation of arccos(x):
      - thetaReg (w = p + 3 qubits) encodes arccos(x) · 2^p.

    The circuit uses a fully-quantum CORDIC arcsin core to obtain direction
    bits for asin(|x|), then conditionally accumulates micro-rotation
    constants to form arccos(x).

    Args:
        m: Number of qubits in the magnitude register.
        p: Number of fractional bits in the output scaling.
        clean_cordic_ancillas: Whether to uncompute CORDIC work registers.

    Returns:
        A QuantumCircuit on all required registers implementing arccos(x).

    Raises:
        None.
    """
    n = p + 1          # CORDIC width
    w = p + 3          # output width headroom so [0, π] does not wrap mod 2^w

    magReg = QuantumRegister(m, name="mag")   # |x|
    signQ = QuantumRegister(1, name="sgn")    # 0 => +, 1 => -

    tReg = QuantumRegister(n, name="t")
    xReg = QuantumRegister(n, name="x")
    yReg = QuantumRegister(n, name="y")
    multReg = QuantumRegister(n, name="mult")
    dReg = QuantumRegister(p, name="d")

    thetaReg = QuantumRegister(w, name="theta")
    cReg = QuantumRegister(w, name="c")      # constant scratch
    helpQ = QuantumRegister(1, name="h")     # CDKM helper

    circ = QuantumCircuit(
        magReg, signQ, tReg, xReg, yReg, multReg, dReg, thetaReg, cReg, helpQ,
        name="arccos_smag"
    )

    # -------------------------------------------------------------------------
    # 1) Map magReg into tReg (nonnegative; no sign-extension)
    #    Want: t/2^(p-1) == u/2^(m-1)  =>  t = u * 2^(p-m)
    # -------------------------------------------------------------------------
    shift = p - m  # can be negative (then precision is lost)

    if shift >= 0:
        # left shift (zero-fill)
        for i in range(m):
            j = i + shift
            if j < n:
                circ.cx(magReg[i], tReg[j])
    else:
        # right shift by r (drop low bits)
        r = -shift
        for i in range(r, m):
            j = i - r
            if j < n:
                circ.cx(magReg[i], tReg[j])

    # -------------------------------------------------------------------------
    # 2) Run CORDIC arcsin to compute direction bits dReg encoding asin(|x|)
    # -------------------------------------------------------------------------
    circ.append(
        quantumCORDIC(tReg, xReg, yReg, multReg, dReg, gate=True),
        tReg[:] + xReg[:] + yReg[:] + multReg[:] + dReg[:],
    )

    if clean_cordic_ancillas:
        circ.append(
            repairCORDIC(tReg, xReg, yReg, multReg, dReg, gate=True),
            tReg[:] + xReg[:] + yReg[:] + multReg[:] + dReg[:],
        )

    # -------------------------------------------------------------------------
    # 3) Convert direction bits into arccos(x) with sign-qubit selection
    # -------------------------------------------------------------------------
    scale = 1 << p
    pi_over_2_int = int(np.round((np.pi / 2) * scale))
    pi_over_2_u = pi_over_2_int % (1 << w)
    _xor_load_constant(circ, thetaReg, pi_over_2_u)  # thetaReg <- π/2

    s = signQ[0]
    circ.x(s)  # swap branches for correct mapping of sign=1 meaning "negative"

    for i in range(1, p + 1):
        alpha = 2.0 * np.arctan(2.0 ** (-i))
        alpha_int = int(np.round(alpha * scale))

        plus_alpha_u = _int_to_twos_comp(+alpha_int, w)
        minus_alpha_u = _int_to_twos_comp(-alpha_int, w)
        plus_2alpha_u = _int_to_twos_comp(+2 * alpha_int, w)
        minus_2alpha_u = _int_to_twos_comp(-2 * alpha_int, w)

        di = dReg[-i]  # iteration i stored at dReg[-i]

        # "sign==1" branch (after flip): theta += (-alpha), and if d=1 then += (+2alpha)
        _rca_add_constant(circ, thetaReg, cReg, helpQ, minus_alpha_u, ctrls=s)
        _rca_add_constant(circ, thetaReg, cReg, helpQ, plus_2alpha_u, ctrls=[s, di])

        # "sign==0" branch (after flip): theta += (+alpha), and if d=1 then += (-2alpha)
        def _sign0_ops():
            _rca_add_constant(circ, thetaReg, cReg, helpQ, plus_alpha_u, ctrls=s)
            _rca_add_constant(circ, thetaReg, cReg, helpQ, minus_2alpha_u, ctrls=[s, di])

        _controlled_on_zero(circ, s, _sign0_ops)

    circ.x(s)  # restore sign qubit

    return circ