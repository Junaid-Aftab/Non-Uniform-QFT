# Non-Uniform Quantum Fourier Transform (NUQFT)

## Overview

The Non-Uniform Quantum Fourier Transform (NUQFT) extends the standard Quantum Fourier Transform (QFT) to settings where sampling points are not uniformly space. Many practical signal processing, imaging, and numerical simulation problems involve irregular sampling grids, making classical and quantum Fourier methods designed for uniform grids insufficient.

This repository contains a numerical and algorithmic implementation of the our NUQFT algorithm, including:

- Classical reference implementations of the Non-Uniform Discrete Fourier Transform (NUDFT)
- Low-rank factorization routines used to construct efficient NUQFT circuits
- Quantum-oriented components that simulate the unitary primitives required by the algorithm
- Experiments validating theoretical predictions such as low-rank structure and precision scaling

If you use this repository in academic work, please cite the following paper:

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




## Problem Statement

The classical Discrete Fourier Transform (DFT) assumes signals are sampled at uniform locations. However, real-world data often arises from non-uniform sampling, due to:

- irregular measurement grids  
- adaptive numerical methods  
- physical constraints in sensing systems  

To address this, the Non-Uniform Discrete Fourier Transform (NUDFT) computes Fourier coefficients from non-uniform spatial samples.

Mathematically, given samples $x_j = f(t_j)$
taken at non-uniform points $t_j$, the NUDFT computes

$$
X_k = \sum_{j=0}^{N-1} x_j e^{-i \omega_k t_j}
$$

where the frequencies $\omega_k$ may also be non-uniform.

While efficient classical algorithms exist for this transform, a fully quantum analogue had not been systematically developed. This project implements and evaluates a quantum algorithm that approximates the NUDFT using low-rank structure and block-encoding techniques.


# What This Project Does

This repository provides a computational framework for studying the NUQFT algorithm, including:

### 1. Classical Baselines

Reference implementations of:

- Type-II NUDFT
- Low-rank approximations used by fast NUFFT algorithms

These serve as benchmarks for validating the quantum formulation.

### 2. Quantum Circuit Components

Prototype implementations of the quantum primitives required by the NUQFT algorithm:

- construction of diagonal unitary operators
- block encodings of matrices
- Chebyshev polynomial transformations
- controlled rotations derived from trigonometric functions

### 3. Numerical Experiments

Notebook experiments validate theoretical predictions including:

- low-rank approximation accuracy
- precision dependence on geometric parameters
- correctness of intermediate unitary operators

### 4. Visualization and Analysis

Generated figures replicate the numerical experiments described in the associated paper.

---

# Methodology

The NUQFT algorithm is based on three key ideas.

## 1. Low-Rank Factorization of the NUDFT

The NUDFT matrix can be approximated by a low-rank decomposition

$$
F_{II} \approx \sum_{r=0}^{K-1} D_{u_r} \, F \, D_{v_r}
$$

where:

- $F$ is the standard DFT matrix
- $D_{u_r}$ and $D_{v_r}$ are diagonal matrices derived from Chebyshev expansions

This decomposition converts the non-uniform Fourier transform into a sum of structured operations involving standard Fourier transforms.

---

## 2. Quantum Block-Encoding

Each matrix term in the decomposition is embedded inside a unitary operator using block encoding. This enables quantum circuits to simulate matrix multiplication while preserving reversibility.

Block encoding is a central technique used in modern quantum linear algebra algorithms.

---

## 3. Linear Combination of Unitaries (LCU)

The sum

$$
\sum_r D_{u_r} F D_{v_r}
$$

is implemented using the Linear Combination of Unitaries (LCU) framework, which constructs a quantum circuit representing weighted sums of unitary operators.

---

## Resulting Complexity

The resulting NUQFT circuit achieves:

- Polylogarithmic scaling with precision
- Quadratic dependence on number of qubits
- Logarithmic dependence on geometry parameters

These properties arise from the efficient low-rank structure and block-encoding formulation.

---

# Sample Results

The experiments in this repository replicate several theoretical results from the paper.

## Low-Rank Scaling

The NUDFT kernel admits a rapidly convergent low-rank approximation, meaning only a small number of terms $K$ are needed to approximate the transform with high precision.

Observed behavior:

$$
K = O\left(\frac{\log(1/\epsilon)}{\log\log(1/\epsilon)}\right)
$$

Numerical experiments confirm this theoretical prediction.

Example result locations:

















## Project Description
The NUQFT generalizes the standard Quantum Fourier Transform by relaxing the assumption of uniformly spaced inputs in signal or frequency domains. This repository focuses on the numerical implementation and evaluation of the proposed NUQFT algorithm, including simulations, validation experiments, and performance analysis.

## Repository Structure

```text
Non-Uniform-QFT/
├── .gitignore
├── src/
│   ├── utils.py
│   ├── datasets/
│   │   ├── .gitkeep
│   │   ├── data.py
│   │   └── signals.py
│   ├── classical/
│   │   ├── nudft_II.py
│   │   └── nufft_II_lowrank.py
│   └── quantum/
│       ├── _init_path.py
│       ├── arccos_cordic.py
│       ├── matrix_ms.py
│       ├── nuqft.py
│       ├── unitary_ur.py
│       └── unitary_vr.py
└── tests/
    ├── _init_path.py
    ├── arccos_cordic.ipynb
    ├── matrix_ms.ipynb
    ├── nufft2.ipynb
    ├── precision_vs_kappa.ipynb
    ├── uneg.ipynb
    ├── unitary_ur.ipynb
    ├── unitary_vr.ipynb
    └── figures/
        ├── A1_lowrank_scaling/
        ├── A2_precision_vs_kappa/
        ├── B1_arccos_cordic/
        ├── B3_unitary_vr/
        └── B4_unitary_ur/
```

### Directory Notes

- `src/classical/` contains classical numerical implementations, including NUDFT-II and low-rank NUFFT-II routines.
- `src/quantum/` contains the quantum-oriented components of the project, including unitary constructions, matrix routines, and the main `nuqft.py` module.
- `src/datasets/` contains helper code for data generation and signal construction.
- `tests/` is notebook-driven and serves as the main experimentation and validation area.
- `tests/figures/` stores generated plots and exported results associated with the notebooks.