# Non-Uniform Quantum Fourier Transform (NUQFT)

## Overview

The Non-Uniform Quantum Fourier Transform (NUQFT) generalizes the Quantum Fourier Transform (QFT) to data sampled on non-uniform grids. Many signal processing and scientific computing applications involve irregular sampling, where the standard Discrete Fourier Transform (DFT) cannot be applied directly.

This repository implements our NUQFT algorithm and includes:

- **Classical baselines for non-uniform Fourier transforms:** includes a dense reference implementation of the Type-II Non-Uniform Discrete Fourier Transform (NUDFT) used primarily for correctness verification and small-scale experiments.

- **Low-rank NUFFT approximation:** implements a classical nonuniform-to-uniform Fourier transform using a low-rank matrix factorization approach (adapted from FastTransforms), providing a more efficient approximation of the NUDFT operator.

- **Quantum NUQFT construction via unitary factorization:** contains modules that build the quantum primitives underlying the algorithm (e.g., unitary \(U_r\) and \(V_r\), matrix synthesis routines, and CORDIC-based angle computation) used to represent the non-uniform Fourier transform as quantum-implementable unitary operations.

- **Supporting utilities, datasets, and validation code:** provides signal generation utilities, data handling, and test scripts used to run numerical experiments and validate the behavior of the classical and quantum components.

## Directory Notes

- `src/classical/` contains classical numerical implementations, including NUDFT-II and low-rank NUFFT-II routines.
- `src/quantum/` contains the quantum-oriented components of the project, including unitary constructions, matrix routines, and the main `nuqft.py` module.
- `src/datasets/` contains helper code for data generation and signal construction.
- `tests/` is notebook-driven and serves as the main experimentation and validation area.
- `tests/figures/` stores generated plots and exported results associated with the notebooks.

## Citation

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