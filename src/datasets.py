"""
Sampling and signal-generation utilities on a periodic domain.

This module defines methods for generating sample nodes on the interval [0, 1)
under various distributions (uniform, perturbed, random, clustered, and
near-colliding).
"""

from __future__ import annotations
import numpy as np

def nodes_uniform(N: int) -> np.ndarray:
    """
    Sample N equally spaced nodes on [0, 1):
        t_j = j / N,  j = 0, ..., N-1

    Interpretation:
    - This is the canonical uniform grid on a unit circle (periodic domain).
    - 1.0 is excluded (since 0.0 and 1.0 represent the same point on a circle).
    """
    return np.arange(N, dtype=float) / N

def nodes_perturbed_uniform(
    N: int,
    jitter: float,
    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample a "nearly uniform" grid by jittering each uniform node, then wrapping periodically.
    
        t_j = j/N + (jitter/N) * Uniform(-1, 1)
        
    jitter is measured in units of the grid spacing (1/N):
    Typical jitter <= 0.5 to keep perturbations smaller than half a grid cell.
    """
    # Use provided RNG for reproducibility; otherwise create a new generator
    rng = np.random.default_rng() if rng is None else rng

    # Start from the uniform grid
    t = nodes_uniform(N)

    # Add bounded random perturbations and wrap back into [0, 1)
    t = t + (jitter / N) * rng.uniform(-1.0, 1.0, size=N)
    return np.mod(t, 1.0)

def nodes_random(N: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample N i.i.d. uniformly sampled random nodes in [0, 1).
    """
    rng = np.random.default_rng() if rng is None else rng
    return rng.random(N)

def nodes_clustered(
    N: int,
    n_clusters: int,
    cluster_std: float,
    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample N nodes arranged into clusters on [0, 1) (periodic wrap).

    Steps:
    1) Sample cluster centers uniformly in [0, 1)
    2) Assign roughly equal counts to each cluster
    3) Sample points around each center with Gaussian noise (std = cluster_std)
    4) Wrap into [0, 1)
    """
    rng = np.random.default_rng() if rng is None else rng

    # Random cluster centers
    centers = rng.random(n_clusters)

    # Distribute N points across clusters as evenly as possible
    counts = np.full(n_clusters, N // n_clusters, dtype=int)
    counts[: N % n_clusters] += 1

    # Sample points around each center
    chunks: list[np.ndarray] = []
    for c, m in zip(centers, counts):
        chunks.append(c + cluster_std * rng.standard_normal(m))

    # Concatenate and wrap into [0, 1)
    t = np.concatenate(chunks)
    return np.mod(t, 1.0)

def nearest_grid_indices(t: np.ndarray, N: int) -> np.ndarray:
    """
    Map nodes t in [0, 1) to nearest uniform grid indices:

        s_j = round(N * t_j) mod N

    Returns integers in {0, ..., N-1}.
    """
    t = np.asarray(t, dtype=float)
    return np.rint(N * t).astype(int) % N 

def nodes_near_colliding(
    N: int,
    min_sep: float = 1e-4,
    rng: np.random.Generator | None = None,
    max_tries: int = 20000) -> np.ndarray:
    """
    Return N nodes in [0, 1) with a minimum circular separation min_sep.

    Uses rejection sampling with a maximum number of proposals (max_tries).
    Raises RuntimeError if it cannot place N points.

    Circular distance between cand and u on [0, 1) is:
        d = min(|cand-u|, 1-|cand-u|)
    implemented via modular arithmetic.
    """
    rng = np.random.default_rng() if rng is None else rng

    accepted: list[float] = []
    tries = 0

    while len(accepted) < N and tries < max_tries:
        cand = float(rng.random())

        # Check if candidate is far enough from all previously accepted points
        ok = True
        for u in accepted:
            # shortest distance on a circle
            d = abs(((cand - u + 0.5) % 1.0) - 0.5)
            if d < min_sep:
                ok = False
                break

        if ok:
            accepted.append(cand)

        tries += 1

    if len(accepted) < N:
        raise RuntimeError("Failed to sample near-colliding nodes; decrease N or min_sep.")

    return np.array(accepted, dtype=float)