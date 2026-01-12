# Non-Uniform Quantum Fourier Transform (NUQFT)

## Project Description

The NUQFT generalizes the standard Quantum Fourier Transform by relaxing the assumption of uniformly spaced inputs in signal or frequency domains. This repository focuses on the numerical implementation and evaluation of the proposed NUQFT algorithm, including simulations, validation experiments, and performance analysis.

## Repository structure
- `src/nuqft/` : reusable implementation code
- `notebooks/` : numbered notebooks (run in order)
- `data/`      : generated datasets (deterministic; small)
- `results/`   : figures/tables produced by experiments
- `tests/`     : unit tests

## Reproducibility
1. Create a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run notebooks in order:
   - `notebooks/00_repo_setup.ipynb`
   - `notebooks/01_problem_setup_and_classical_baselines.ipynb`
   - ...

## Notes
- All random generation should use the fixed seed in `src/nuqft/config.py`.
- Figures/tables should be saved into `results/` with parameter metadata in filenames.
