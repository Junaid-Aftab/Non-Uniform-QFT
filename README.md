# Non-Uniform Quantum Fourier Transform (NUQFT)

## Project Description

The NUQFT generalizes the standard Quantum Fourier Transform by relaxing the assumption of uniformly spaced inputs in signal or frequency domains. This repository focuses on the numerical implementation and evaluation of the proposed NUQFT algorithm, including simulations, validation experiments, and performance analysis.

## Mathematical Background

## Mathematical Background

### Non-Uniform Discrete Fourier Transform (NUDFT)

Given samples $\{x_j\}_{j=0}^{N-1}$ at non-uniform locations $\{t_j\}_{j=0}^{N-1}\subset[0,1)$, the Type–II NUDFT is

![NUDFT equation](https://latex.codecogs.com/svg.image?X_k%20%3D%20%5Csum_%7Bj%3D0%7D%5E%7BN-1%7D%20x_j%20e%5E%7B-2%5Cpi%20i%20k%20t_j%7D%2C%20%5Cqquad%20k%3D0%2C%5Cldots%2C%20N-1.)

In matrix form,

![Matrix form](https://latex.codecogs.com/svg.image?%5Cmathbf%7BX%7D%20%3D%20F_%7B%5Cmathrm%7BII%7D%7D%5Cmathbf%7Bx%7D%2C%20%5Cad%20(F_%7B%5Cmathrm%7BII%7D%7D)_%7Bk%2Cj%7D%20%3D%20e%5E%7B-2%5Cpi%20i%20k%20t_j%7D.)

### Low-Rank Approximation

The NUQFT construction uses a low-rank approximation

![Low-rank approx](https://latex.codecogs.com/svg.image?F_%7B%5Cmathrm%7BII%7D%7D%20%5Capprox%20%5Csum_%7Br%3D0%7D%5E%7BK-1%7D%20D_%7B%5Cmathbf%7Bu%7D_r%7D%20F_%7B%5Cmathrm%7BDFT%7D%7D%20D_%7B%5Cmathbf%7Bv%7D_r%7D.)

with truncation rank scaling

![K scaling](https://latex.codecogs.com/svg.image?K%20%3D%20O%5C!%5Cleft(%5Cfrac%7B%5Clog(1%2F%5Cvarepsilon)%7D%7B%5Clog%5Clog(1%2F%5Cvarepsilon)%7D%5Cright).)


