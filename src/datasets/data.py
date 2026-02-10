"""
Sampling and signal-generation utilities on a periodic domain.

This module provides routines for generating sample nodes on the interval
[0, 1) under a variety of distributions. The constructions are intended for
experiments on periodic domains, including uniform grids, perturbed grids,
random samples, clustered configurations, and node sets with controlled
minimum separation.
"""

from __future__ import annotations
import numpy as np


def nodes_uniform(N: int) -> np.ndarray:
    """
    Generate N equally spaced sample nodes on the interval [0, 1).

    The nodes are defined by
        t_j = j / N,  for j = 0, …, N − 1.

    This corresponds to the canonical uniform grid on a periodic unit interval,
    where the endpoints 0 and 1 are identified and 1 is therefore excluded.

    Args:
        N: Number of sample nodes.

    Returns:
        A one-dimensional NumPy array of length N containing uniformly spaced
        nodes in [0, 1).
    """
    return np.arange(N, dtype=float) / N


def nodes_perturbed_uniform(
    N: int,
    jitter: float,
    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate a nearly uniform grid by perturbing a uniform node set.

    Starting from the uniform grid t_j = j / N, each node is independently
    perturbed according to
        t_j ← t_j + (jitter / N) · U(−1, 1),
    and then wrapped periodically back into the interval [0, 1).

    The parameter jitter is measured in units of the grid spacing 1 / N. Values
    of jitter ≤ 0.5 typically ensure perturbations remain within half a grid
    cell.

    Args:
        N: Number of sample nodes.
        jitter: Magnitude of the random perturbation relative to the grid spacing.
        rng: Optional NumPy random number generator. If None, a new default
            generator is created.

    Returns:
        A one-dimensional NumPy array of length N containing perturbed nodes in
        [0, 1).
    """
    rng = np.random.default_rng() if rng is None else rng

    t = nodes_uniform(N)

    t = t + (jitter / N) * rng.uniform(-1.0, 1.0, size=N)
    return np.mod(t, 1.0)


def nodes_random(N: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate N independent uniformly distributed random nodes in [0, 1).

    Args:
        N: Number of sample nodes.
        rng: Optional NumPy random number generator. If None, a new default
            generator is created.

    Returns:
        A one-dimensional NumPy array of length N containing i.i.d. uniform
        samples in [0, 1).
    """
    rng = np.random.default_rng() if rng is None else rng
    return rng.random(N)


def nodes_clustered(
    N: int,
    n_clusters: int,
    cluster_std: float,
    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate N sample nodes arranged into clusters on a periodic domain.

    The construction proceeds as follows:
    1) Sample cluster centers uniformly in [0, 1).
    2) Distribute the N nodes as evenly as possible among the clusters.
    3) Sample points around each center using Gaussian noise with standard
       deviation cluster_std.
    4) Wrap all samples periodically into the interval [0, 1).

    Args:
        N: Total number of sample nodes.
        n_clusters: Number of clusters.
        cluster_std: Standard deviation of the Gaussian noise around each
            cluster center.
        rng: Optional NumPy random number generator. If None, a new default
            generator is created.

    Returns:
        A one-dimensional NumPy array of length N containing clustered nodes in
        [0, 1).
    """
    rng = np.random.default_rng() if rng is None else rng

    centers = rng.random(n_clusters)

    counts = np.full(n_clusters, N // n_clusters, dtype=int)
    counts[: N % n_clusters] += 1

    chunks: list[np.ndarray] = []
    for c, m in zip(centers, counts):
        chunks.append(c + cluster_std * rng.standard_normal(m))

    t = np.concatenate(chunks)
    return np.mod(t, 1.0)


def nearest_grid_indices(t: np.ndarray, N: int) -> np.ndarray:
    """
    Map sample nodes to their nearest indices on a uniform grid.

    Each node t_j ∈ [0, 1) is mapped to the integer index
        s_j = round(N · t_j) mod N,
    which lies in the set {0, …, N − 1}.

    Args:
        t: Array of sample nodes in [0, 1).
        N: Number of points in the reference uniform grid.

    Returns:
        A NumPy array of integers in {0, …, N − 1} giving the nearest grid
        indices corresponding to t.
    """
    t = np.asarray(t, dtype=float)
    return np.rint(N * t).astype(int) % N


def nodes_near_colliding(
    N: int,
    min_sep: float = 1e-4,
    rng: np.random.Generator | None = None,
    max_tries: int = 20000) -> np.ndarray:
    """
    Generate N sample nodes with a prescribed minimum circular separation.

    The nodes lie in [0, 1) and are sampled using rejection sampling to enforce
    a minimum pairwise separation min_sep on the unit circle. Sampling is
    aborted after a maximum number of proposals.

    The circular distance between two points cand and u is defined as
        d = min(|cand − u|, 1 − |cand − u|),
    implemented via modular arithmetic.

    Args:
        N: Number of sample nodes.
        min_sep: Minimum allowed circular separation between any two nodes.
        rng: Optional NumPy random number generator. If None, a new default
            generator is created.
        max_tries: Maximum number of candidate samples before giving up.

    Returns:
        A one-dimensional NumPy array of length N containing nodes in [0, 1)
        satisfying the minimum separation constraint.

    Raises:
        RuntimeError: If fewer than N nodes can be placed within max_tries
        proposals.
    """
    rng = np.random.default_rng() if rng is None else rng

    accepted: list[float] = []
    tries = 0

    while len(accepted) < N and tries < max_tries:
        cand = float(rng.random())

        ok = True
        for u in accepted:
            d = abs(((cand - u + 0.5) % 1.0) - 0.5)
            if d < min_sep:
                ok = False
                break

        if ok:
            accepted.append(cand)

        tries += 1

    if len(accepted) < N:
        raise RuntimeError(
            "Failed to sample near-colliding nodes; decrease N or min_sep."
        )

    return np.array(accepted, dtype=float)