"""
Signal-generation utilities on a periodic domain.

This module defines methods for generating constructing complex-valued test
signals commonly used in Fourier and signal-processing experiments.
"""

from __future__ import annotations
import numpy as np
import scipy.fft as sfft


def signal_random_complex(
    N: int,
    rng: np.random.Generator | None = None,
    normalize: bool = True) -> np.ndarray:
    """
    Generate a random complex-valued signal of length N.

    Sampling model:
    - Real part: a ~ Normal(0, 1) elementwise
    - Imag part: b ~ Normal(0, 1) elementwise
    - x = a + i b
    
    If normalize=True, rescale so ||x||_2 = 1 (unless x is all zeros, which is extremely unlikely).
    """
    # If caller does not supply an RNG, create a fresh unseeded generator
    rng = np.random.default_rng() if rng is None else rng

    # Complex Gaussian samples: x = a + i b, where a,b ~ N(0,1)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)

    # Optional normalization to unit L2 norm
    if normalize:
        nrm = np.linalg.norm(x)
        if nrm != 0:
            x = x / nrm

    return x


def signal_smooth_from_nodes(t: np.ndarray) -> np.ndarray:
    """
    Construct a deterministic smooth complex signal evaluated at sample locations t.

    Returns a sum of two complex exponentials:
        exp(2πi * 3 * t) + 0.2 * exp(-2πi * 7 * t)

    Interpretable as two Fourier modes (frequencies +3 and -7) with different amplitudes.
    """
    # Ensure t is a float NumPy array
    t = np.asarray(t, dtype=float)

    # Complex oscillations at specific frequencies
    return np.exp(2j * np.pi * 3 * t) + 0.2 * np.exp(-2j * np.pi * 7 * t)


def signal_sparse_spectrum(
    N: int,
    sparsity: int = 5,
    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate a length-N complex signal whose Fourier spectrum is sparse.

    Steps:
    1) Create a Fourier-domain vector X with mostly zeros
    2) Choose 'sparsity' random frequency indices ks
    3) Assign complex Gaussian coefficients at those ks
    4) Convert to time domain via inverse FFT

    The factor sqrt(N) is a normalization choice (common in signal processing conventions).
    """
    # If caller does not supply an RNG, create a fresh unseeded generator
    rng = np.random.default_rng() if rng is None else rng

    # Fourier-domain coefficients (frequency bins), initially all zero
    X = np.zeros(N, dtype=complex)

    # Pick random distinct frequency indices for the nonzeros
    ks = rng.choice(N, size=min(sparsity, N), replace=False)

    # Fill those frequency bins with random complex amplitudes
    X[ks] = rng.standard_normal(len(ks)) + 1j * rng.standard_normal(len(ks))

    # Transform to time domain; multiply by sqrt(N) for chosen normalization
    x = sfft.ifft(X) * np.sqrt(N)

    return x