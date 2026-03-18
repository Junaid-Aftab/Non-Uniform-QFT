# Non-Uniform Quantum Fourier Transform (NUQFT)

## Overview

The Non-Uniform Quantum Fourier Transform (NUQFT) extends the Quantum Fourier Transform (QFT) to data sampled at non-uniform intervals. Many practical signal processing and scientific computing tasks involve irregular sampling grids, where the standard Discrete Fourier Transform (DFT) cannot be applied directly without modification.

This repository provides a numerical and algorithmic implementation of the NUQFT algorithm, including:

- Classical implementations of the Non-Uniform Discrete Fourier Transform (NUDFT)
- Simulations of the quantum unitary primitives required for NUQFT
- Numerical experiments validating the low-rank structure of the NUDFT kernel
- Experiments analyzing precision scaling and conditioning effects

If you use this repository, please cite the following paper:

```bibtex
@misc{aftab2026nonuniformquantumfouriertransform,
  title={Non-Uniform Quantum Fourier Transform},
  author={Junaid Aftab and Yuehaw Khoo and Haizhao Yang},
  year={2026},
  eprint={2602.13472},
  archivePrefix={arXiv},
  primaryClass={quant-ph},
  url={https://arxiv.org/abs/2602.13472}
}
```


# Non-Uniform Discrete Fourier Transform (NUDFT)

## What is the NUDFT

The classical Discrete Fourier Transform (DFT) assumes uniformly spaced samples, an assumption often violated in practice due to irregular measurement locations or adaptive methods. The Non-Uniform Discrete Fourier Transform (NUDFT) extends the DFT to handle non-uniform sampling.
Given samples $x_j = f(t_j)$ taken at non-uniform points $t_j$ on $\mathbb T$, the NUDFT computes Fourier coefficients

$$
X_k = \sum_{j=0}^{N-1} x_j e^{-i \omega_k t_j}.
$$

The frequencies $\omega_k$ may also be non-uniform.

Equivalently, the transform can be written as a matrix–vector multiplication $\vec X = F_{NU} \vec x$ where the matrix entries are $(F_{NU})_{k,j} = e^{-i \omega_k t_j}$. Unlike the classical FFT, which exploits strong structure in the uniform Fourier matrix, the NUDFT matrix does not directly admit an exact fast transform. However, the oscillatory kernel $e^{-i\omega t}$ has smooth structure that can be exploited using approximation techniques.

# Low-Rank Factorization of the NUDFT Matrix

For the Type–II NUDFT, where the frequencies are uniform but the spatial samples are non-uniform, the transform matrix can be expressed as $(F_{II})_{j,k} = e^{-2\pi i t_j k}$.

If the non-uniform grid points are close to a uniform grid, one can write $t_j = \frac{j}{N} + \delta_j$, where $\delta_j$ represents a small perturbation. Substituting this into the exponential yields

$$
e^{-2\pi i t_j k}
=
e^{-2\pi i (t_j - j/N)k}
e^{-2\pi i jk/N}.
$$

This leads to the matrix decomposition

$F_{II} = A \circ F$

where $F$ is the standard DFT matrix  and  $A$ is a modulation matrix with entries

$$
A_{j,k} = e^{-2\pi i (t_j - j/N)k}
$$

and $\circ$ denotes the elementwise (Hadamard) product.

The key observation is that the function $e^{-i x y}$ admits an accurate low-rank approximation using truncated Chebyshev polynomial expansions. As a result, the matrix $A$ can be approximated as a rank-$K$ matrix

$$
A \approx \sum_{r=0}^{K-1} u_r v_r^{T}.
$$

Substituting this into the decomposition gives

$$
F_{II} \approx \sum_{r=0}^{K-1} (u_r v_r^{T}) \circ F = \sum_{r=0}^{K-1} D_{u_r} F D_{v_r}.
$$

Here $D_{u_r}$ is a diagonal matrix containing the vector $u_r$, $D_{v_r}$ is a diagonal matrix containing the vector $v_r$  and  $F$ is the standard discrete Fourier transform matrix.












