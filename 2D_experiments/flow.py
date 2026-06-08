"""
Flow Activations and Baselines
================================
All activations receive a (batch, 2) tensor of 2D coordinates and return
a (batch, out_dim) feature tensor. The Euler ResNet block then projects
this back to (batch, 2) and applies the step x ← x + dt·v.

Flow activations enforce divergence-free velocity fields derived from a
stream function ψ:
    u = -∂ψ/∂y,   v = +∂ψ/∂x   =>   ∇·(u,v) = 0

This constraint means the flows are area-preserving (they cannot collapse
all points to a single attractor), which is physically meaningful for
a classification task over a 2D domain.

Baselines:
  - ReLU:  Standard piecewise-linear activation. No spectral bias, no
           divergence-free constraint. Parameter-matched.
  - Siren: sin(ω₀·Wx+b). Sinusoidal like cellular/shear but without the
           geometric 2D structure. Tests whether the advantage of flow
           activations is purely spectral or structural.
"""

import numpy as np
import torch
import torch.nn as nn
from enum import Enum


class ActivationType(str, Enum):
    CELLULAR = "cellular"
    VORTEX   = "vortex"
    SHEAR    = "shear"
    RELU     = "relu"
    SIREN    = "siren"


# ---------------------------------------------------------------------------
# Flow activations (divergence-free)
# ---------------------------------------------------------------------------

class CellularFlowActivation(nn.Module):
    """
    Stream function: ψ(x,y) = A · cos(k·x) · cos(l·y)
    u = -∂ψ/∂y =  A·l · cos(k·x) · sin(l·y)   (sign fixed from original)
    v = +∂ψ/∂x = -A·k · sin(k·x) · cos(l·y)

    The amplitude A = 1/(k²+l²) is chosen so the flow speed doesn't blow
    up as wavenumbers grow. This gives a natural cellular / standing-wave
    pattern — ideal for periodic tasks like checkerboard and sinusoidal.
    """
    def __init__(self, width: int = 3):
        super().__init__()
        self.k = nn.Parameter(torch.rand(width) * 3 + 1.0)
        self.l = nn.Parameter(torch.rand(width) * 3 + 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x[:, 0:1]
        yc = x[:, 1:2]
        A = 1.0 / (self.k**2 + self.l**2 + 1e-5)
        u = A * self.l * torch.cos(self.k * xc) * torch.sin(self.l * yc)
        v = -A * self.k * torch.sin(self.k * xc) * torch.cos(self.l * yc)
        return torch.cat([u, v], dim=1)   # (B, 2·width)

    @property
    def out_features(self) -> int:
        return self.k.shape[0] * 2


class VortexFlowActivation(nn.Module):
    """
    Stream function: ψ(x,y) = γ · exp(-‖(x,y)−(cx,cy)‖² / σ²)
    This is a Gaussian vortex centred at (cx, cy) with circulation γ.
    u = -∂ψ/∂y = γ · exp(…) · (2·dy / σ²)
    v = +∂ψ/∂x = γ · exp(…) · (−2·dx / σ²)

    Naturally suited to radial / rotational tasks. Multiple vortices
    (controlled by `width`) can combine to represent complex flow fields.
    """
    def __init__(self, width: int = 3):
        super().__init__()
        self.cx    = nn.Parameter(torch.randn(width) * 0.3 + 0.5)
        self.cy    = nn.Parameter(torch.randn(width) * 0.3 + 0.5)
        self.gamma = nn.Parameter(torch.randn(width))
        self.sigma = nn.Parameter(torch.ones(width) * 0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x[:, 0:1]
        yc = x[:, 1:2]
        dx = xc - self.cx
        dy = yc - self.cy
        var = self.sigma**2 + 1e-5
        exp = torch.exp(-(dx**2 + dy**2) / var)
        u = self.gamma * exp * (2 * dy / var)
        v = self.gamma * exp * (-2 * dx / var)
        return torch.cat([u, v], dim=1)   # (B, 2·width)

    @property
    def out_features(self) -> int:
        return self.cx.shape[0] * 2


class ShearFlowActivation(nn.Module):
    """
    Axis-aligned shear flows:
      u(y) = A_h · sin(k_h · y + φ_h)   (horizontal transport, depends on y)
      v(x) = A_v · sin(k_v · x + φ_v)   (vertical transport, depends on x)

    These are strictly divergence-free (∂u/∂x = 0, ∂v/∂y = 0) and naturally
    suited to axis-aligned periodic tasks. Expected to underperform on spiral
    and radial tasks, which provides the inductive-bias contrast story.
    """
    def __init__(self, width: int = 3):
        super().__init__()
        self.k_h   = nn.Parameter(torch.rand(width) * 3 + 1.0)
        self.phi_h = nn.Parameter(torch.rand(width) * 2 * np.pi)
        self.A_h   = nn.Parameter(torch.ones(width))
        self.k_v   = nn.Parameter(torch.rand(width) * 3 + 1.0)
        self.phi_v = nn.Parameter(torch.rand(width) * 2 * np.pi)
        self.A_v   = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x[:, 0:1]
        yc = x[:, 1:2]
        u = self.A_h * torch.sin(self.k_h * yc + self.phi_h)
        v = self.A_v * torch.sin(self.k_v * xc + self.phi_v)
        return torch.cat([u, v], dim=1)   # (B, 2·width)

    @property
    def out_features(self) -> int:
        return self.k_h.shape[0] * 2


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class ReLUActivation(nn.Module):
    """
    Standard ReLU block: Linear(2→width) + ReLU.
    Width is adjusted by the caller to match parameter counts with flow blocks.
    Output dimension is `width` (not 2·width), so the projection layer
    must account for this.
    """
    def __init__(self, width: int = 6):
        super().__init__()
        self.linear = nn.Linear(2, width)
        self.act    = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))

    @property
    def out_features(self) -> int:
        return self.linear.out_features


class SirenActivation(nn.Module):
    """
    Sinusoidal representation network (Siren) layer:
      h = sin(ω₀ · W·x + b)
    Uses the recommended weight initialisation: W ~ U(-√6/n_in, √6/n_in)
    for intermediate layers, which preserves the distribution of activations.
    """
    def __init__(self, width: int = 6, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear  = nn.Linear(2, width)
        # Siren weight init: U(-√6/n_in, √6/n_in)
        with torch.no_grad():
            n_in = self.linear.weight.shape[1]
            bound = np.sqrt(6.0 / n_in) / omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))

    @property
    def out_features(self) -> int:
        return self.linear.out_features


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_activation(act_type: ActivationType, width: int) -> nn.Module:
    """
    Instantiate an activation module. For flow activations, `width` is the
    number of basis functions (output is 2·width). For ReLU and Siren,
    width directly determines output dimension.
    """
    if act_type == ActivationType.CELLULAR:
        return CellularFlowActivation(width)
    elif act_type == ActivationType.VORTEX:
        return VortexFlowActivation(width)
    elif act_type == ActivationType.SHEAR:
        return ShearFlowActivation(width)
    elif act_type == ActivationType.RELU:
        return ReLUActivation(width)
    elif act_type == ActivationType.SIREN:
        return SirenActivation(width)
    else:
        raise ValueError(f"Unknown activation: {act_type}")
