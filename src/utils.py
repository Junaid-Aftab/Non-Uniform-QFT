# ---------------------------------------------------------------------------
# Import Packages
# ---------------------------------------------------------------------------
from __future__ import annotations
import time
import numpy as np

# ---------------------------------------------------------------------------
# Error metrics for comparing numerical arrays
# ---------------------------------------------------------------------------
def rel_l2(a, b) -> float:
    """
    Compute the relative L2 (Euclidean) error between arrays a and b.

    Formula:
        ||a - b||_2 / ||b||_2

    Returns NaN if the ||b||_2 is zero.
    """
    # Convert inputs to NumPy arrays
    a = np.asarray(a)
    b = np.asarray(b)

    # L2 norm of the difference
    num = np.linalg.norm(a - b)

    # L2 norm of the reference array
    den = np.linalg.norm(b)

    # Avoid division by zero
    return float(num / den) if den != 0 else float("nan")


def max_abs(a, b) -> float:
    """
    Compute the maximum absolute difference between arrays a and b.

    Formula:
        max(|a_i - b_i|)
    """
    # Convert inputs to NumPy arrays
    a = np.asarray(a)
    b = np.asarray(b)

    # Maximum absolute elementwise difference
    return float(np.max(np.abs(a - b)))

# ---------------------------------------------------------------------------
# Timing utility for benchmarking code blocks
# ---------------------------------------------------------------------------
class Timer:
    """
    Context manager for measuring elapsed execution time.
    """

    def __enter__(self):
        # Record start time using a high-resolution counter
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        # Compute elapsed time on exit
        self.dt = time.perf_counter() - self.t0