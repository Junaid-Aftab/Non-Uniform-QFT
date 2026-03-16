# Non-Uniform-QFT

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
```
---

## Installation

Clone the repository:

```bash
git clone https://github.com/<username>/Non-Uniform-QFT.git
cd Non-Uniform-QFT
```

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not present, install common scientific dependencies:

```bash
pip install numpy scipy matplotlib jupyter
```

---

## Usage

### Running Notebooks

Start Jupyter:

```bash
jupyter notebook
```

Open the notebooks inside the `notebooks/` directory to explore examples and experiments.

---

### Running Scripts

Example:

```bash
python experiments/example_simulation.py
```

Scripts typically demonstrate:

- grid construction
- field initialization
- operator application
- numerical experiments

---

## Example Workflow

A typical workflow when experimenting with the codebase:

1. Generate a **non-uniform grid**
2. Initialize a field configuration
3. Apply discretized operators
4. Run numerical evolution or analysis
5. Visualize results

Example (conceptual):

```python
from grid import NonUniformGrid
from fields import ScalarField

grid = NonUniformGrid(size=100, spacing="adaptive")
phi = ScalarField(grid)

phi.initialize_random()
phi.apply_laplacian()

phi.plot()
```

---

## Dependencies

Typical dependencies include:

- numpy
- scipy
- matplotlib
- jupyter

Additional libraries may be required depending on experiments added to the repository.

---

## Development

Contributions and experimentation are encouraged.

Possible areas for extension:

- improved discretization schemes
- adaptive mesh refinement
- additional field types
- gauge field support
- performance optimizations

---

## Limitations

- The repository is **experimental**
- Numerical stability and convergence are still under exploration
- Some modules may be incomplete or subject to change

---

## License

Add a license file to specify usage rights.  
Common choices include:

- MIT License
- BSD License
- GPL License

---

## Citation

If this repository contributes to academic work, consider citing it as:

```
Author. Non-Uniform-QFT. GitHub repository.
```

---

## Acknowledgements

This project draws inspiration from computational approaches to:

- lattice quantum field theory
- adaptive numerical methods
- irregular grid discretization techniques