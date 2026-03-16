"""
Python implementation of a 1D nonuniform-to-uniform Fourier transform using the
low-rank matrix factorization approach of Antolín and Townsend.

This module is a focused port/adaptation of the Type-II NUFFT machinery in
FastTransforms.jl (file: `src/nufft.jl`), but with the direction changed to
compute the transform

    f_k = sum_{j=0}^{N-1} c_j * exp(-2*pi*i*x_j*k),  k = 0,...,N-1,

i.e., mapping values given on nonuniform nodes x_j (the c_j) to uniform Fourier
modes k.

Implementation idea
-------------------
We reuse the same precomputed low-rank factors U and V (rank K chosen from a
tolerance parameter) and apply (approximately) the transpose of the original
Type-II low-rank NUFFT operator:

1) elementwise multiplication by U (per rank)
2) scatter-add onto an FFT grid using t = round(N*x) mod N
3) an FFT along the uniform dimension
4) elementwise multiplication by V (per rank)
5) summation over the rank dimension
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

try:
    from scipy.special import jv, lambertw
except Exception as _exc:  # pragma: no cover
    jv = None  # type: ignore
    lambertw = None  # type: ignore
    _SCIPY_IMPORT_ERROR = _exc
else:
    _SCIPY_IMPORT_ERROR = None


ArrayLike = Union[np.ndarray, list, tuple]


def _require_scipy() -> None:
    """ Raise an informative error if SciPy is not available.

    This module relies on special functions (Bessel J and Lambert W) from
    `scipy.special`. If SciPy is missing or failed to import, this helper raises
    an ImportError with installation guidance.

    Args:
        None.

    Returns:
        None.

    Raises:
        ImportError: If SciPy is not available in the current environment.
    """
    if _SCIPY_IMPORT_ERROR is not None:
        raise ImportError(
            "SciPy is required for this module (scipy.special.jv, lambertw). "
            "Install it via `pip install scipy`."
        ) from _SCIPY_IMPORT_ERROR


def assign_closest_equispaced_gridpoint(x: np.ndarray) -> np.ndarray:
    """ Assign each nonuniform node to the closest equispaced grid point.

    Given real-valued nodes x_j, this routine computes the nearest integer grid
    index s_j = round(N * x_j), where N = len(x). No modulo is applied.

    Args:
        x: One-dimensional array of length N containing real-valued nodes.

    Returns:
        A one-dimensional integer NumPy array of shape (N,) with entries
        s_j = round(N * x_j).

    Raises:
        ValueError: If x is not a one-dimensional array.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    n = x.size
    return np.rint(n * x).astype(np.int64)


def assign_closest_equispaced_fftpoint(x: np.ndarray) -> np.ndarray:
    """ Assign each node to the closest FFT grid index, wrapped modulo N.

    This routine computes the nearest integer grid index and wraps it into the
    FFT grid index set {0, ..., N-1} via modulo arithmetic:
        t_j = round(N * x_j) mod N.

    Args:
        x: One-dimensional array of length N containing real-valued nodes.

    Returns:
        A one-dimensional integer NumPy array of shape (N,) with entries
        t_j = round(N * x_j) mod N.

    Raises:
        None.
    """
    n = x.size
    t = (np.rint(n * x).astype(np.int64) % n)

    return t


def perturbation_parameter(x: np.ndarray, s: np.ndarray) -> float:
    """ Compute the perturbation parameter gamma used to select the low-rank rank.

    The perturbation parameter is defined as
        gamma = max_j |N * x_j - s_j|,
    where s_j = round(N * x_j) (without modulo). This quantity measures the
    maximum deviation from the nearest equispaced grid point.

    Args:
        x: One-dimensional array of length N containing real-valued nodes.
        s: One-dimensional integer array of length N with entries round(N * x_j).

    Returns:
        The perturbation parameter gamma as a Python float.

    Raises:
        ValueError: If x and s are not 1D arrays of the same length.
    """
    x = np.asarray(x)
    s = np.asarray(s)
    if x.ndim != 1 or s.ndim != 1 or x.size != s.size:
        raise ValueError("x and s must be 1D arrays of the same length")
    n = x.size
    return float(np.max(np.abs(n * x - s)))


def find_k(gamma: float, eps: float) -> int:
    """ Choose the rank K for the low-rank NUFFT factorization.

    This routine implements the rank selection rule used in the referenced
    low-rank approximation approach. For sufficiently small perturbation
    (gamma <= eps), it returns K = 1; otherwise it uses a Lambert-W-based
    expression to choose K to meet a target tolerance.

    Args:
        gamma: Perturbation parameter (nonnegative).
        eps: Target accuracy (positive).

    Returns:
        The selected rank K as a positive integer.

    Raises:
        ImportError: If SciPy is not available.
        ValueError: If gamma < 0 or eps <= 0.
    """
    _require_scipy()
    gamma = float(gamma)
    eps = float(eps)
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if gamma <= eps:
        return 1

    val = 5.0 * gamma * np.exp(
        np.real(lambertw(np.log(140.0 / eps) / (5.0 * gamma)))
    )
    return int(np.ceil(val))


def chebyshev_polynomials(n: int, x: np.ndarray) -> np.ndarray:
    """ Evaluate Chebyshev polynomials T_0, ..., T_n at given points.

    This routine computes the first-kind Chebyshev polynomials using the
    three-term recurrence:
        T_0(x) = 1,
        T_1(x) = x,
        T_{k+1}(x) = 2 x T_k(x) - T_{k-1}(x).

    Args:
        n: Maximum polynomial degree (must be >= 0).
        x: One-dimensional array of evaluation points (shape (N,)).

    Returns:
        A NumPy array of shape (N, n+1) where entry (j, k) equals T_k(x[j]).

    Raises:
        ValueError: If n < 0 or if x is not a one-dimensional array.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")

    N = x.size
    T = np.empty((N, n + 1), dtype=x.dtype)
    T[:, 0] = 1.0
    if n >= 1:
        T[:, 1] = x
        two_x = 2.0 * x
        for k in range(1, n):
            T[:, k + 1] = two_x * T[:, k] - T[:, k - 1]
    return T


def bessel_coeffs(
    K: int, gamma: float, dtype: np.dtype = np.complex128
) -> np.ndarray:
    """ Compute the Chebyshev coefficient matrix used in the low-rank approximation.

    This routine forms the (K, K) coefficient matrix built from Bessel J
    functions, following the coefficient formulas used by the low-rank NUFFT
    construction.

    Args:
        K: Target rank (must be >= 1).
        gamma: Perturbation parameter.
        dtype: NumPy dtype for the returned array.

    Returns:
        A complex NumPy array of shape (K, K) containing the coefficient matrix.

    Raises:
        ImportError: If SciPy is not available.
        ValueError: If K < 1.
    """
    _require_scipy()
    if K < 1:
        raise ValueError("K must be >= 1")
    gamma = float(gamma)

    cfs = np.zeros((K, K), dtype=dtype)
    arg = -gamma * np.pi / 2.0

    for p in range(K):
        q0 = p % 2
        for q in range(q0, K, 2):
            v1 = (p + q) / 2.0
            v2 = (q - p) / 2.0
            cfs[p, q] = 4.0 * (1j**q) * jv(v1, arg) * jv(v2, arg)

    cfs[0, :] /= 2.0
    cfs[:, 0] /= 2.0
    return cfs


def construct_u(x: np.ndarray, K: int) -> np.ndarray:
    """ Construct the matrix U used in the low-rank NUFFT factorization.

    For nodes x_j, this routine computes the nearest grid indices s_j and the
    perturbations e_j = N*x_j - s_j. It then builds U using an oscillatory phase
    factor and Chebyshev expansions parameterized by rank K.

    Args:
        x: One-dimensional array of length N containing real-valued nodes.
        K: Rank of the low-rank factorization.

    Returns:
        A complex NumPy array U of shape (N, K).

    Raises:
        ValueError: If x is not a one-dimensional array.
        ImportError: If SciPy is not available.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    N = x.size

    s = assign_closest_equispaced_gridpoint(x)
    er = N * x - s
    gamma = float(np.max(np.abs(er)))

    if gamma == 0.0:
        scaled = er.astype(float)
    else:
        scaled = (er / gamma).astype(float)

    phase = np.exp(-1j * (np.pi * er))
    T = chebyshev_polynomials(K - 1, scaled)  # (N, K) real
    B = bessel_coeffs(K, gamma, dtype=np.complex128)  # (K, K) complex

    return (phase[:, None] * (T @ B)).astype(np.complex128)


def construct_v(N: int, K: int) -> np.ndarray:
    """ Construct the matrix V used in the low-rank NUFFT factorization.

    This routine evaluates Chebyshev polynomials on the mapped uniform grid
    xi_k = 2*k/N - 1 for k = 0, ..., N-1.

    Args:
        N: Transform length (must be positive).
        K: Rank of the low-rank factorization (must be >= 1).

    Returns:
        A complex NumPy array V of shape (N, K).

    Raises:
        ValueError: If N <= 0 or K < 1.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    if K < 1:
        raise ValueError("K must be >= 1")

    omega = np.arange(N, dtype=float)
    xi = omega * (2.0 / N) - 1.0
    return chebyshev_polynomials(K - 1, xi).astype(np.complex128)


@dataclass(frozen=True)
class NUFFT2Plan:
    """
    Precomputed plan for the 1D nonuniform-to-uniform Fourier transform.

    The plan applies the transform
        f_k = sum_{j=0}^{N-1} c_j * exp(-2*pi*i*x_j*k),  k = 0,...,N-1,
    using a low-rank factorization with rank K and a scatter-add onto an FFT
    grid.

    Args:
        x: One-dimensional array of real-valued nonuniform nodes (shape (N,)).
        eps: Target accuracy used to select the rank K.
        U: Complex matrix of shape (N, K) defining the node-side factors.
        V: Complex matrix of shape (N, K) defining the mode-side factors.
        t: Integer array of shape (N,) with entries round(N*x_j) mod N used for
            scatter-add onto the FFT grid.
        N: Transform length.
        K: Low-rank factorization rank.

    Attributes:
        x: Real-valued nonuniform nodes (shape (N,)).
        eps: Target accuracy used to select the rank K.
        U: Complex matrix of shape (N, K).
        V: Complex matrix of shape (N, K).
        t: Integer scatter indices (shape (N,)).
        N: Transform length.
        K: Low-rank factorization rank.
    """

    x: np.ndarray
    eps: float
    U: np.ndarray
    V: np.ndarray
    t: np.ndarray
    N: int
    K: int

    def __call__(self, c: ArrayLike) -> np.ndarray:
        """ Apply the planned nonuniform-to-uniform transform.

        The input may be a single vector of samples c_j or a matrix whose columns
        are independent sample vectors. The output has matching dimensionality,
        with Fourier modes along the first dimension.

        Args:
            c: Nonuniform samples c_j. Either a one-dimensional array of shape
                (N,) or a two-dimensional array of shape (N, M).

        Returns:
            A complex NumPy array of Fourier modes. Shape (N,) if c is 1D, or
            (N, M) if c is 2D.

        Raises:
            ValueError: If c is not 1D or 2D, or if its leading dimension does
                not match the planned length N.
        """
        c_arr = np.asarray(c)

        if c_arr.ndim == 1:
            if c_arr.size != self.N:
                raise ValueError(f"c must have length {self.N}")

            out = np.zeros(self.N, dtype=np.complex128)
            for p in range(self.K):
                g = np.zeros(self.N, dtype=np.complex128)
                # scatter-add: g[t_j] += c_j * U[j,p]
                np.add.at(g, self.t, c_arr * self.U[:, p])
                h = np.fft.fft(g)
                out += self.V[:, p] * h
            return out

        if c_arr.ndim == 2:
            if c_arr.shape[0] != self.N:
                raise ValueError(f"c must have shape ({self.N}, M)")
            M = c_arr.shape[1]

            out = np.zeros((self.N, M), dtype=np.complex128)
            for p in range(self.K):
                g = np.zeros((self.N, M), dtype=np.complex128)
                # scatter-add for each column: g[t_j,:] += c[j,:] * U[j,p]
                np.add.at(g, self.t, c_arr * self.U[:, p][:, None])
                h = np.fft.fft(g, axis=0)
                out += self.V[:, p][:, None] * h
            return out

        raise ValueError("c must be a 1D or 2D array")


def plan_nufft2(x: ArrayLike, eps: float, K: Optional[int] = None) -> NUFFT2Plan:
    """ Precompute a plan for the 1D nonuniform-to-uniform Fourier transform.

    If K is provided, it overrides the rank selection rule and is used directly.
    Otherwise, K is chosen from (gamma, eps) via find_k.

    Args:
        x: One-dimensional array-like of real-valued nonuniform nodes (shape (N,)).
        eps: Target accuracy used to select the rank K (must be positive).
        K: Optional manual rank override (must be >= 1). If provided, eps is not
            used for rank selection.

    Returns:
        A NUFFT2Plan instance that can be called on nonuniform samples c.

    Raises:
        ImportError: If SciPy is not available.
        ValueError: If x is not a 1D array, if eps is not positive, or if K is
            provided but less than 1.
    """
    _require_scipy()

    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError("x must be a 1D array")

    eps_f = float(eps)
    if eps_f <= 0:
        raise ValueError("eps must be positive")

    N = x_arr.size
    t = assign_closest_equispaced_fftpoint(x_arr)

    if K is None:
        gamma = perturbation_parameter(
            x_arr, assign_closest_equispaced_gridpoint(x_arr)
        )
        K_sel = find_k(gamma, eps_f)
    else:
        K_sel = int(K)
        if K_sel < 1:
            raise ValueError("K must be >= 1")

    U = construct_u(x_arr, K_sel)
    V = construct_v(N, K_sel)

    return NUFFT2Plan(x=x_arr, eps=eps_f, U=U, V=V, t=t, N=N, K=K_sel)


def nufft2(c: ArrayLike, x: ArrayLike, eps: float, K: Optional[int] = None) -> np.ndarray:
    """ Compute the 1D nonuniform-to-uniform Fourier transform using low-rank factors.

    This convenience wrapper constructs a NUFFT2Plan via plan_nufft2 and applies
    it to the provided nonuniform samples.

    Args:
        c: Nonuniform samples c_j. Either a one-dimensional array of shape (N,)
            or a two-dimensional array of shape (N, M).
        x: One-dimensional array-like of real-valued nonuniform nodes (shape (N,)).
        eps: Target accuracy used to select the rank K (must be positive).
        K: Optional manual rank override (must be >= 1). If provided, eps is not
            used for rank selection.

    Returns:
        A complex NumPy array of Fourier modes. Shape (N,) if c is 1D, or
        (N, M) if c is 2D.

    Raises:
        ImportError: If SciPy is not available.
        ValueError: If inputs violate the requirements enforced by plan_nufft2
            or by NUFFT2Plan.__call__.
    """
    return plan_nufft2(x, eps, K=K)(c)