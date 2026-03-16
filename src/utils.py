"""
Utility functions for numerical comparison and performance measurement.

This module provides common error metrics for comparing numerical arrays
(such as relative L2 error and maximum absolute difference), along with a
lightweight timing context manager for benchmarking code execution.
"""

from __future__ import annotations
import time
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Error metrics for comparing numerical arrays
# ---------------------------------------------------------------------------
def rel_l2(a, b) -> float:
    """
    Compute the relative L2 (Euclidean) error between two arrays.

    The relative error is defined as

        ||a - b||_2 / ||b||_2

    where ||·||_2 denotes the Euclidean norm.

    Args:
        a: Numerical array representing the computed result.
        b: Numerical array representing the reference result.

    Returns:
        The relative L2 error as a float. If the reference norm ||b||_2
        is zero, the function returns NaN.

    Raises:
        None.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b)

    return float(num / den) if den != 0 else float("nan")


def max_abs(a, b) -> float:
    """
    Compute the maximum absolute difference between two arrays.

    The metric is defined as

        max_i |a_i - b_i|

    and measures the largest pointwise discrepancy between the arrays.

    Args:
        a: Numerical array representing the computed result.
        b: Numerical array representing the reference result.

    Returns:
        The maximum absolute difference between corresponding elements
        of the two arrays.

    Raises:
        None.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    return float(np.max(np.abs(a - b)))


# ---------------------------------------------------------------------------
# Timing utility for benchmarking code blocks
# ---------------------------------------------------------------------------
class Timer:
    """
    Context manager for measuring execution time.

    This utility provides a lightweight mechanism for benchmarking
    sections of code using Python's ``with`` statement. The elapsed
    wall-clock time is recorded using ``time.perf_counter``.

    Example:
        with Timer() as t:
            run_algorithm()

        print(t.dt)

    Args:
        None.

    Returns:
        An object whose ``dt`` attribute stores the elapsed time
        (in seconds) after exiting the context block.

    Raises:
        None.
    """

    def __enter__(self):
        """Start the timer when entering the context."""
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        """Stop the timer and record the elapsed duration."""
        self.dt = time.perf_counter() - self.t0


# =============================================================================
# Plot style + EPS transparency warning avoidance
# =============================================================================
def _apply_publication_rcparams() -> None:
    """
    Apply Matplotlib configuration suitable for publication-quality figures.

    This routine updates global Matplotlib ``rcParams`` to enforce settings
    commonly used in academic figures, including higher resolution output,
    readable font sizes, and disabled transparency for compatibility with
    EPS exports.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "legend.frameon": True,
            "axes.grid": False,
            "savefig.transparent": False,
            "patch.force_edgecolor": True,
        }
    )


def _force_opaque(ax: plt.Axes) -> None:
    """
    Ensure that all plot elements in an axis are fully opaque.

    This function iterates through the graphical objects associated
    with a Matplotlib axis (lines, collections, patches, text objects,
    and legend frames) and forces their alpha values to 1.0.

    The routine is primarily used to avoid transparency warnings when
    saving figures in formats such as EPS that do not support alpha
    transparency.

    Args:
        ax: A Matplotlib ``Axes`` object whose elements will be modified.

    Returns:
        None.

    Raises:
        None.
    """
    for ln in ax.lines:
        try:
            ln.set_alpha(1.0)
        except Exception:
            pass

    for col in ax.collections:
        try:
            col.set_alpha(1.0)
        except Exception:
            pass

    for patch in ax.patches:
        try:
            patch.set_alpha(1.0)
        except Exception:
            pass

    for txt in ax.texts:
        try:
            txt.set_alpha(1.0)
        except Exception:
            pass

    leg = ax.get_legend()
    if leg is not None and leg.get_frame() is not None:
        try:
            leg.get_frame().set_alpha(1.0)
        except Exception:
            pass