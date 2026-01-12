# Non-Uniform Quantum Fourier Transform (NUQFT)

## Project Description

The NUQFT generalizes the standard Quantum Fourier Transform by relaxing the assumption of uniformly spaced inputs in signal or frequency domains. This repository focuses on the numerical implementation and evaluation of the proposed NUQFT algorithm, including simulations, validation experiments, and performance analysis.

## Mathematical Background

This repository focuses on **numerical simulation and validation** of the NUQFT construction rather than direct gate-level quantum compilation. Accordingly, we summarize only the mathematical components that directly inform what is implemented in code.

### Non-Uniform Discrete Fourier Transform (NUDFT)

Given samples \( \{x_j\}_{j=0}^{N-1} \) taken at non-uniform locations  
\( \{t_j\}_{j=0}^{N-1} \subset [0,1) \), the Type–II NUDFT computes
\[
X_k = \sum_{j=0}^{N-1} x_j e^{-2\pi i k t_j}, \qquad k = 0,\dots,N-1.
\]
This can be written in matrix form as
\[
\mathbf{X} = F_{\mathrm{II}} \mathbf{x},
\quad
(F_{\mathrm{II}})_{k,j} = e^{-2\pi i k t_j}.
\]

Unlike the standard DFT, the matrix \(F_{\mathrm{II}}\) is dense and lacks the structure needed for fast FFT-style evaluation. :contentReference[oaicite:0]{index=0}

### Low-Rank Approximation

The NUQFT algorithm is based on the observation that \(F_{\mathrm{II}}\) admits an accurate **low-rank approximation**:
\[
F_{\mathrm{II}} \;\approx\; \sum_{r=0}^{K-1} D_{\mathbf{u}_r}\, F_{\mathrm{DFT}}\, D_{\mathbf{v}_r},
\]
where:
- \(F_{\mathrm{DFT}}\) is the standard DFT matrix,
- \(D_{\mathbf{u}_r}\) and \(D_{\mathbf{v}_r}\) are diagonal matrices,
- the vectors \(\mathbf{u}_r\) and \(\mathbf{v}_r\) depend on the non-uniform sampling points \(\{t_j\}\),
- \(K\) controls the approximation accuracy.

This factorization is obtained by separating the NUDFT kernel into a uniform Fourier term and a smooth correction term, which is then approximated using truncated Chebyshev expansions. For a target accuracy \(\varepsilon\), the required rank satisfies
\[
K = O\!\left(\frac{\log(1/\varepsilon)}{\log\log(1/\varepsilon)}\right).
\] :contentReference[oaicite:1]{index=1}

### Relevance to This Codebase

The numerical components implemented in this repository include:
- Construction of the diagonal vectors \(\mathbf{u}_r\) and \(\mathbf{v}_r\),
- Assembly of the low-rank approximation
  \(\sum_r D_{\mathbf{u}_r} F_{\mathrm{DFT}} D_{\mathbf{v}_r}\),
- Verification of approximation accuracy against the exact NUDFT,
- Empirical analysis of truncation rank \(K\), error decay, and runtime scaling.

Quantum-specific techniques such as block encodings, QSP, and LCU motivate the structure of the decomposition but are **not explicitly simulated** at the gate level. Instead, the code mirrors the linear-algebraic structure that these quantum subroutines would implement.


