"""stflip: bpy-free ST-FLIP liquid solver (NumPy / CuPy).

Implementation of "Spatiotemporal FLIP for Fast Free-Surface and Two-Phase
Simulation With Very Large Time Steps", Braun et al., ACM TOG 45(4), 2026.
"""

from .backend import (
    Backend,
    cuda_available,
    cuda_device_name,
    cuda_diagnostics,
    get_backend,
)
from .solver import FrameStats, Params, STFLIPSolver

__version__ = "0.3.0"

__all__ = [
    "Backend",
    "FrameStats",
    "Params",
    "STFLIPSolver",
    "cuda_available",
    "cuda_device_name",
    "cuda_diagnostics",
    "get_backend",
    "__version__",
]
