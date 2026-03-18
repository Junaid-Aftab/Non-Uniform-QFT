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

Equivalently, the transform can be written as a matrix–vector multiplication $\vec X = F_{NU} \vec x$

where the matrix entries are

$$
(F_{NU})_{k,j} = e^{-i \omega_k t_j}.
$$

Unlike the classical FFT, which exploits strong structure in the uniform Fourier matrix, the NUDFT matrix does not directly admit an exact fast transform. However, the oscillatory kernel $e^{-i\omega t}$ has smooth structure that can be exploited using approximation techniques.

---

# Low-Rank Factorization of the NUDFT Matrix

For the Type–II NUDFT, where the frequencies are uniform but the spatial samples are non-uniform, the transform matrix can be expressed as

$$
(F_{II})_{j,k} = e^{-2\pi i t_j k}.
$$

If the non-uniform grid points are close to a uniform grid, one can write

$$
t_j = \frac{j}{N} + \delta_j
$$

where $\delta_j$ represents a small perturbation.

Substituting this into the exponential yields

$$
e^{-2\pi i t_j k}
=
e^{-2\pi i (t_j - j/N)k}
e^{-2\pi i jk/N}.
$$

This leads to the matrix decomposition

$$
F_{II} = A \circ F
$$

where

- $F$ is the standard DFT matrix  
- $A$ is a modulation matrix with entries

$$
A_{j,k} = e^{-2\pi i (t_j - j/N)k}
$$

and $\circ$ denotes the elementwise (Hadamard) product.

The key observation is that the function

$$
e^{-i x y}
$$

admits an accurate low-rank approximation using truncated Chebyshev polynomial expansions. As a result, the matrix $A$ can be approximated as a rank-$K$ matrix

$$
A \approx \sum_{r=0}^{K-1} u_r v_r^{T}.
$$

Substituting this into the decomposition gives

$$
F_{II} \approx \sum_{r=0}^{K-1} (u_r v_r^{T}) \circ F.
$$

Using the identity

$$
(u_r v_r^{T}) \circ F = D_{u_r} F D_{v_r},
$$

the NUDFT matrix can be approximated by

$$
F_{II} \approx \sum_{r=0}^{K-1} D_{u_r} F D_{v_r}.
$$

Here

- $D_{u_r}$ is a diagonal matrix containing the vector $u_r$  
- $D_{v_r}$ is a diagonal matrix containing the vector $v_r$  
- $F$ is the standard discrete Fourier transform matrix  

The rank $K$ grows only polylogarithmically with the desired approximation accuracy, making this representation efficient in practice.

---

# Quantum Algorithm for the NUQFT

The Non-Uniform Quantum Fourier Transform (NUQFT) translates the above low-rank factorization into a quantum circuit.

The goal is to construct a unitary that block-encodes the NUDFT matrix so that it can be applied to a quantum state.

The algorithm proceeds through the following steps.

---

## Step 1: Preparation of coefficient states

The vectors $v_r$ are defined using Chebyshev polynomials evaluated on a uniform grid

$$
(v_r)_j = T_r\!\left(\frac{2j}{N} - 1\right).
$$

A quantum circuit prepares a superposition state whose amplitudes encode these coefficients

$$
U_{v_r} |0\rangle =
\sum_j v_r(j) |j\rangle.
$$

This circuit uses

- Hadamard gates to create a uniform index superposition  
- quantum arithmetic to compute grid coordinates  
- controlled rotations to implement Chebyshev polynomial evaluations  

---

## Step 2: Preparation of sampling-dependent coefficients

The vectors $u_r$ depend on the non-uniform sampling points $t_j$

$$
(u_r)_j =
\sum_q \alpha'_{qr}
e^{-i\pi N(t_j - s_j/N)}
T_q\!\left(2N(t_j - s_j/N)\right).
$$

Their construction requires oracle access to the sampling grid

$$
O_t : |i\rangle |0\rangle \rightarrow |i\rangle |t_i\rangle.
$$

Using this oracle, quantum arithmetic circuits compute

$$
t_j - \frac{s_j}{N}
$$

and evaluate the Chebyshev polynomial terms in superposition.

---

## Step 3: Block encoding of diagonal matrices

Once the states encoding $u_r$ and $v_r$ are constructed, they are used to build block encodings of the diagonal matrices

$$
D_{u_r}, \quad D_{v_r}.
$$

These operators apply amplitude modulations conditioned on the index register.

---

## Step 4: Application of the quantum Fourier transform

The standard Quantum Fourier Transform implements the matrix $F$

$$
|x\rangle \rightarrow
\frac{1}{\sqrt{N}}
\sum_y e^{-2\pi i xy/N} |y\rangle.
$$

This circuit requires $O(n^2)$ gates for $N = 2^n$.

---

## Step 5: Assembly using the LCU framework

Finally, the Linear Combination of Unitaries (LCU) method is used to combine the block encodings

$$
F_{II} \approx \sum_{r=0}^{K-1} D_{u_r} F D_{v_r}.
$$

LCU prepares an ancilla state encoding the coefficients of the sum and applies a controlled selection of the corresponding unitary operations.

The resulting circuit provides an approximate block encoding of the NUDFT matrix.

---

# Summary

The NUQFT algorithm combines three key ingredients

1. a low-rank approximation of the NUDFT kernel  
2. quantum circuits implementing diagonal coefficient operators  
3. the Quantum Fourier Transform combined using the LCU framework  

This construction yields a quantum algorithm whose complexity scales polylogarithmically in the desired precision and polynomially in the number of qubits, while incorporating the structure of non-uniform sampling grids.












