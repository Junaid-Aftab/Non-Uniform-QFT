"""
Signal-generation utilities on a periodic domain.

This module provides routines for constructing complex-valued test signals on a
uniform periodic grid. The generated signals are commonly used in Fourier
analysis, sparse spectral experiments, and numerical signal-processing studies.
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

    The signal entries are drawn independently from a complex Gaussian
    distribution with independent real and imaginary parts:
        a_j ~ Normal(0, 1),
        b_j ~ Normal(0, 1),
        x_j = a_j + i b_j.

    If requested, the signal is rescaled to have unit ℓ₂ norm.

    Args:
        N: Length of the signal.
        rng: Optional NumPy random number generator. If None, a new default
            generator is created.
        normalize: Whether to rescale the output so that ||x||₂ = 1.

    Returns:
        A one-dimensional NumPy array of length N containing a complex-valued
        random signal.
    """
    rng = np.random.default_rng() if rng is None else rng

    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)

    if normalize:
        nrm = np.linalg.norm(x)
        if nrm != 0:
            x = x / nrm

    return x


def signal_smooth_from_nodes(t: np.ndarray) -> np.ndarray:
    """
    Construct a deterministic smooth complex signal evaluated at sample nodes.

    The signal is defined as a superposition of two complex exponentials:
        exp(2πi · 3 · t) + 0.2 · exp(−2πi · 7 · t).

    This corresponds to two Fourier modes with frequencies +3 and −7 and
    different amplitudes, yielding a smooth periodic signal.

    Args:
        t: One-dimensional array of sample locations at which the signal is
            evaluated.

    Returns:
        A NumPy array of complex values representing the signal evaluated at t.
    """
    t = np.asarray(t, dtype=float)

    return np.exp(2j * np.pi * 3 * t) + 0.2 * np.exp(-2j * np.pi * 7 * t)


def signal_sparse_spectrum(
    N: int,
    sparsity: int = 5,
    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate a length-N complex signal with a sparse Fourier spectrum.

    The construction proceeds as follows:
    1) Initialize a Fourier-domain vector with all entries equal to zero.
    2) Select a specified number of distinct frequency indices uniformly at random.
    3) Assign complex Gaussian coefficients to those frequencies.
    4) Apply an inverse FFT to obtain the time-domain signal.

    A factor of sqrt(N) is applied after the inverse FFT as a normalization
    convention commonly used in signal processing.

    Args:
        N: Length of the signal.
        sparsity: Number of nonzero Fourier coefficients.
        rng: Optional NumPy random number generator. If None, a new default
            generator is created.

    Returns:
        A one-dimensional NumPy array of length N containing a complex-valued
        signal with sparse frequency content.
    """
    rng = np.random.default_rng() if rng is None else rng

    X = np.zeros(N, dtype=complex)

    ks = rng.choice(N, size=min(sparsity, N), replace=False)

    X[ks] = rng.standard_normal(len(ks)) + 1j * rng.standard_normal(len(ks))

    x = sfft.ifft(X) * np.sqrt(N)

    return x