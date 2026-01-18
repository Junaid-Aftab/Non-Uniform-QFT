"""
Baseline implementations of nonuniform discrete Fourier transform.

This module provides a direct, dense implementation of the Type-II NUDFT,
intended for correctness checks and small-scale experiments rather than
high-performance use.
"""

from __future__ import annotations
import numpy as np

def nudft_type2_dense(t: np.ndarray, x: np.ndarray, K: int | None = None) -> np.ndarray:
    """
    Type-II NUDFT:
      X_k = sum_{j=0}^{N-1} x_j * exp(-2π i k t_j),  k=0..K-1
    Dense O(NK), intended for small/medium sizes.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=complex)

    if t.size != x.size:
        raise ValueError(f"t and x must have the same length, got {t.size} and {x.size}")

    N = t.size
    K = N if K is None else int(K)
    k = np.arange(K, dtype=float)
    
    phase = np.exp(-2j * np.pi * np.outer(k, t))
    return phase @ x