# Non-Uniform Quantum Fourier Transform (NUQFT)

## Overview

The Non-Uniform Quantum Fourier Transform (NUQFT) generalizes the Quantum Fourier Transform (QFT) to cases where sampling points are not uniformly spaced. This repository provides a numerical and algorithmic implementation of our NUQFT algorithm, including:

- Classical implementations of the Non-Uniform Discrete Fourier Transform (NUDFT)
- Simulation of the algorithm’s quantum unitary primitives
- Experiments validating low-rank structure and precision scaling

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


## Background

The classical Discrete Fourier Transform (DFT) assumes signals are sampled at uniform locations. However, real-world data often arises from non-uniform sampling due to irregular measurement grids, adaptive numerical methods, and similar settings. The Non-Uniform Discrete Fourier Transform (NUDFT) computes Fourier coefficients from non-uniform spatial samples.

Mathematically, given samples \( x_j = f(t_j) \) taken at non-uniform points \( t_j \), the NUDFT computes

$$
X_k = \sum_{j=0}^{N-1} x_j e^{-i \omega_k t_j}
$$

where the frequencies \( \omega_k \) may also be non-uniform.

While efficient classical algorithms exist for this transform, a fully quantum analogue has not been systematically developed. This project implements and evaluates a quantum algorithm that approximates the NUDFT using low-rank structure and block-encoding techniques.












