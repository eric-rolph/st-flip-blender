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
from . import surface_tension, whitewater
from .solver import FrameStats, Params, STFLIPSolver
from .surface import SurfaceReconstruction, reconstruct_surface
from .velocity import SolidBodyRotation, UniformVelocity

__version__ = "0.23.1"

__all__ = [
    "Backend",
    "FrameStats",
    "Params",
    "SolidBodyRotation",
    "STFLIPSolver",
    "SurfaceReconstruction",
    "UniformVelocity",
    "cuda_available",
    "cuda_device_name",
    "cuda_diagnostics",
    "get_backend",
    "reconstruct_surface",
    "surface_tension",
    "whitewater",
    "__version__",
]
