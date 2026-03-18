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

## Directory Notes

- `src/classical/` contains classical numerical implementations, including NUDFT-II and low-rank NUFFT-II routines.
- `src/quantum/` contains the quantum-oriented components of the project, including unitary constructions, matrix routines, and the main `nuqft.py` module.
- `src/datasets/` contains helper code for data generation and signal construction.
- `tests/` is notebook-driven and serves as the main experimentation and validation area.
- `tests/figures/` stores generated plots and exported results associated with the notebooks.