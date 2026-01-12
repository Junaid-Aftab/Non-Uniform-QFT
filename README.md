# Non-Uniform Quantum Fourier Transform (NUQFT)

## Project Description

This repository contains the numerical computations performed for our **Non-Uniform Quantum Fourier Transform (NUQFT)** project.

The NUQFT generalizes the standard Quantum Fourier Transform by relaxing the assumption of uniformly spaced input in signal or frequency domains. This work focuses on the numerical implementation and evaluation of the proposed NUQFT algorithm, including simulations, validation experiments, and performance analysis.

The codebase is intended to support theoretical investigation and experimental verification of the algorithm, and may serve as a foundation for further research in quantum signal processing and quantum algorithm design.

---

## Mathematical Background

## Mathematical Background

The standard Quantum Fourier Transform (QFT) implements a unitary transformation corresponding to the discrete Fourier transform over uniformly spaced computational basis states. While this structure is central to many quantum algorithms, it limits applicability in scenarios involving non-uniform sampling, irregular frequency grids, or weighted spectral representations.

The **Non-Uniform Quantum Fourier Transform (NUQFT)** extends this framework by allowing transformations defined over non-equidistant input domains. The mathematical formulation adapts classical non-uniform Fourier techniques to the quantum setting while preserving unitarity and compatibility with quantum circuit models.

This repository focuses on the numerical aspects of this formulation, including:
- Discrete representations of non-uniform domains  
- Approximation strategies suitable for quantum implementation  
- Error behavior and numerical stability  

---

## Methodology

The numerical workflow implemented in this repository consists of the following steps:

1. **Problem Setup**  
   Definition of non-uniform sampling points or frequency distributions and construction of the corresponding transformation matrices.

2. **Algorithm Implementation**  
   Numerical realization of the NUQFT algorithm, including matrix-based simulations and, where applicable, circuit-level abstractions.

3. **Simulation and Evaluation**  
   Execution of numerical experiments to validate correctness, analyze approximation error, and compare performance against the standard QFT in relevant regimes.

4. **Analysis**  
   Assessment of numerical accuracy, computational complexity, and scaling behavior with respect to system size and non-uniformity parameters.

---
