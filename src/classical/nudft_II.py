"""
Baseline implementations of the nonuniform discrete Fourier transform.

This module provides a direct, dense implementation of the Type-II NUDFT.
It is intended for correctness checks, validation, and small-scale numerical
experiments rather than high-performance computation.
"""

from __future__ import annotations
import numpy as np


def nudft_type2_dense(
    t: np.ndarray,
    x: np.ndarray,
    K: int | None = None,
) -> np.ndarray:
    """
    Compute the Type-II nonuniform discrete Fourier transform (NUDFT).

    This routine evaluates the transform
        X_k = sum_{j=0}^{N-1} x_j * exp(-2π i k t_j),  for k = 0, ..., K-1
    using a dense O(NK) matrix–vector multiplication. It is primarily intended
    for reference computations and small to medium problem sizes.

    As a special case, if the sample locations correspond exactly to the uniform
    grid t_j = j / N and K == N, the transform reduces to the standard discrete
    Fourier transform. In this case, the result is computed using the FFT to
    avoid discrepancies between numerically different evaluation pathways.

    Args:
        t: One-dimensional array of length N containing sample locations in [0, 1).
        x: One-dimensional array of length N containing complex-valued samples.
        K: Number of output Fourier modes. If None, K is set to N.

    Returns:
        A one-dimensional NumPy array of length K containing the Type-II NUDFT
        coefficients.

    Raises:
        ValueError: If t and x do not have the same length.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=complex)

    if t.size != x.size:
        raise ValueError(f"t and x must have the same length, got {t.size} and {x.size}")

    N = t.size
    K = N if K is None else int(K)

    # Exact uniform-grid detection (no tolerance); deterministic for nodes_uniform().
    if K == N:
        tu = np.arange(N, dtype=float) / N
        if np.array_equal(t, tu):
            return np.fft.fft(x)

    k = np.arange(K, dtype=float)
    phase = np.exp(-2j * np.pi * np.outer(k, t))
    return phase @ x