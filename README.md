# Non-Uniform Quantum Fourier Transform (NUQFT)

## Overview

The Non-Uniform Quantum Fourier Transform (NUQFT) extends the standard Quantum Fourier Transform (QFT) to settings where sampling points are not uniformly space. Many practical signal processing, imaging, and numerical simulation problems involve irregular sampling grids, making classical and quantum Fourier methods designed for uniform grids insufficient.

This repository contains a numerical and algorithmic implementation of the our NUQFT algorithm, including:

- Classical reference implementations of the Non-Uniform Discrete Fourier Transform (NUDFT)
- Low-rank factorization routines used to construct efficient NUQFT circuits
- Quantum-oriented components that simulate the unitary primitives required by the algorithm
- Experiments validating theoretical predictions such as low-rank structure and precision scaling

The implementation accompanies the research work:

**Non-Uniform Quantum Fourier Transform – Junaid Aftab, Yuehaw Khoo, Haizhao Yang**  
https://arxiv.org/abs/2602.13472

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