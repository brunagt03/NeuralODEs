# Euler-Flow Activations in ResNets

This repository contains the core framework for benchmarking **Euler-flow velocity fields** against traditional network baselines within a continuous-depth Residual Network (ResNet) architecture. 

The framework evaluates the hypothesis that embedding area-preserving fluid dynamics directly into neural networks provides a superior inductive bias for resolving complex, spatially structured 2D classification manifolds.

---

## Introduction

Traditional architectures like Residual Networks can be interpreted as a forward-Euler discretization of an Ordinary Differential Equation (Neural ODE):

$$ \dot{x}(t) = v(x(t), \theta(t)) \implies x_{k+1} = x_k + \Delta t \cdot v(x_k, \theta_k) $$

Instead of using standard flat activations (such as ReLU) to parameterize the velocity field $v$, this project uses velocity fields derived natively from an incompressible fluid's stream function $\psi$.
---

## Repository Architecture

The project is modularized into three primary execution layers:

### 1. Model & Experiment Suite (`main.py`)
*   **`FlowResNet`**: Constructs the continuous-depth pipeline by stacking sequential `SimpleResNetBlock` modules.
*   **Parameter Matching System**: Contains utility functions (`count_params` and `matched_width`) that calculate and align the exact parameter counts of baseline architectures to match the flow-based networks, ensuring mathematically rigorous evaluation.
*   **Unified Optimization Loops**: 
    *   `train_fixed_steps_with_curve`: Runs rigorous optimization for exactly 60,000 steps to monitor convergence velocity.
    *   `train_to_convergence`: Implements validation-driven early stopping to verify structural data efficiency.

### 2. Geometric Activation Layer (`flow.py`)
Implements three distinct divergence-free fluid configurations along with standard network baselines:

| Activation Type | 
| :--- | :--- |
| **Cellular** | $\psi(x,y) = A \cos(k x) \cos(l y)$ | 
| **Vortex** | $\psi(x,y) = \gamma \exp(-\frac{\|(x,y)-(c_x,c_y)\|^2}{\sigma^2})$ | 
| **Shear** | $u(y) = A_h \sin(k_h y + \phi_h)$, $v(x) = A_v \sin(k_v x + \phi_v)$ | 
| **ReLU** | Baseline piecewise-linear perceptron layer | 
| **Siren** | $\sin(\omega_0 \cdot Wx + b)$ | 


### 2. Geometric Activation Layer (`flow.py`)
Implements three distinct divergence-free fluid configurations along with standard network baselines:

| Activation Type | Underlying Mathematics |
| :--- | :--- |
| **Cellular** | $\psi(x,y) = A \cos(k x) \cos(l y)$ |
| **Vortex** | $\psi(x,y) = \gamma \exp(-\frac{\|(x,y)-(c_x,c_y)\|^2}{\sigma^2})$ | 
| **Shear** | $u(y) = A_h \sin(k_h y + \phi_h)$, $v(x) = A_v \sin(k_v x + \phi_v)$ |
| **ReLU** | Baseline piecewise-linear perceptron layer | 
| **Siren** | $\sin(\omega_0 \cdot Wx + b)$ |

### 3. Task definitions (`generators.py`)
Three distinct tasks are supported, defined by the `TaskType` enum:

1.  **Checkerboard (`checkerboard`)**: A classic grid-based task. 
2.  **Radial (`radial`)**: Concentric rings centered at (0.5, 0.5). 
3.  **Spiral (`spiral`)**: Alternating sectors that spiral outward from the center.
---

## Configured Experiments

The main pipeline executes four comprehensive benchmarks across all task types:

1.  **Learning Curves (`run_learning_curve`)**: Trains parameter-matched models on a fixed training set size ($N=2000$) for exactly 60,000 steps across 5 unique seeds to isolate optimization speeds and performance variance.
2.  **Decision Boundaries (`run_boundary_comparison`)**: Generates 2D spatial prediction grids using the median performing seed to visually map the topological expressivity and spatial confidence metrics ($P(\text{class}=1)$).

---
