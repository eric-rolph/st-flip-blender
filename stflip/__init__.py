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
from . import benchmarks, paper_validation, surface_tension, tiles, whitewater
from .pressure import PressureSolveError
from .solver import FrameStats, Params, STFLIPSolver
from .surface import SurfaceReconstruction, reconstruct_surface
from .velocity import SolidBodyRotation, UniformVelocity

__version__ = "0.43.0"

__all__ = [
    "Backend",
    "FrameStats",
    "Params",
    "PressureSolveError",
    "SolidBodyRotation",
    "STFLIPSolver",
    "SurfaceReconstruction",
    "UniformVelocity",
    "cuda_available",
    "cuda_device_name",
    "cuda_diagnostics",
    "benchmarks",
    "get_backend",
    "paper_validation",
    "reconstruct_surface",
    "surface_tension",
    "tiles",
    "whitewater",
    "__version__",
]
