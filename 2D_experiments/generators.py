import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple
from enum import Enum


class TaskType(str, Enum):
    CHECKERBOARD = "checkerboard"
    RADIAL       = "radial"
    SPIRAL       = "spiral"


@dataclass
class DataConfig:
    task:     TaskType
    n:        int    # Spatial frequency / number of periods or sectors
    N:        int    # Number of points to sample
    seed:     int = 0


# ---------------------------------------------------------------------------
# Label functions (numpy, operate on arrays for speed)
# ---------------------------------------------------------------------------

def label_checkerboard(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Classic checkerboard. Tests axis-aligned periodic structure."""
    i = np.floor(n * x).astype(int)
    j = np.floor(n * y).astype(int)
    return ((i + j) % 2).astype(np.float32)



def label_radial(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """
    Concentric binary rings centred at (0.5, 0.5).
    Strongly favors vortex-like activations with rotational symmetry.
    n controls number of rings across the unit circle.
    """
    cx, cy = 0.5, 0.5
    r_max = 0.5 * np.sqrt(2)  # corner-to-centre distance
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    return (np.floor(n * r / r_max) % 2).astype(np.float32)


def label_spiral(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """
    Spiral sectors based on angle + radius.
    The label alternates as you sweep around the origin, with the sector
    boundary rotating with radius. Hardest for shear activations; tests
    whether the network can compose rotation and translation.
    n controls the number of full rotations of the spiral.
    """
    cx, cy = 0.5, 0.5
    dx, dy = x - cx, y - cy
    angle = np.arctan2(dy, dx)                          # ∈ [-π, π]
    r = np.sqrt(dx**2 + dy**2) + 1e-8
    # Wind the angle by radius so the boundary spirals outward
    wound = (angle / (2 * np.pi) + n * r) % 1.0
    return (wound > 0.5).astype(np.float32)


_LABEL_FN = {
    TaskType.CHECKERBOARD: label_checkerboard,
    TaskType.RADIAL:       label_radial,
    TaskType.SPIRAL:       label_spiral,
}


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

def generate_dataset(cfg: DataConfig) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Returns:
        X_tensor : (N, 2) float32 — coordinates in [0,1]²
        Y_tensor : (N, 1) float32 — binary labels
        Y_np     : (N,)   int     — numpy labels for plotting
    """
    rng = np.random.default_rng(cfg.seed)
    X = rng.random((cfg.N, 2)).astype(np.float32)
    fn = _LABEL_FN[cfg.task]
    Y = fn(X[:, 0], X[:, 1], cfg.n)

    X_tensor = torch.from_numpy(X)
    Y_tensor = torch.from_numpy(Y).view(-1, 1)
    return X_tensor, Y_tensor, Y.astype(int)


def generate_grid(resolution: int = 256, task: TaskType = TaskType.RADIAL,
                  n: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dense uniform grid for decision-boundary visualisation.
    Returns xs, ys (meshgrid) and Z (label grid).
    """
    lin = np.linspace(0, 1, resolution)
    xs, ys = np.meshgrid(lin, lin)
    fn = _LABEL_FN[task]
    Z = fn(xs.ravel(), ys.ravel(), n).reshape(resolution, resolution)
    return xs, ys, Z


# ---------------------------------------------------------------------------
# Sample-efficiency sweep helper
# ---------------------------------------------------------------------------

SAMPLE_SIZES = [100, 250, 500, 1000, 2000, 5000]


def generate_nested_datasets(task: TaskType, n: int, max_N: int = 5000,
                              seed: int = 0):
    """
    Draw max_N points once, then return nested subsets for sample-efficiency
    experiments. All subsets share the same points so results are comparable.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((max_N, 2)).astype(np.float32)
    fn = _LABEL_FN[task]
    Y = fn(X[:, 0], X[:, 1], n)

    subsets = {}
    for N in SAMPLE_SIZES:
        if N > max_N:
            break
        X_t = torch.from_numpy(X[:N])
        Y_t = torch.from_numpy(Y[:N]).view(-1, 1)
        subsets[N] = (X_t, Y_t, Y[:N].astype(int))
    return subsets
