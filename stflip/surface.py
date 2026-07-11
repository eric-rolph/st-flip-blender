"""Backend-neutral Appendix-B particle surface reconstruction.

The paper reconstructs a render-only implicit field directly from synchronized
particles.  This module implements that dense reference path without Blender or
SciPy: a cropped union-of-spheres density, the fixed self-quotient feature mask,
and explicit feature-preserving level-set mean-curvature flow.

``SurfaceReconstruction.origin`` is the world-space position of density sample
``density[0, 0, 0]`` (not a voxel corner).  The 0.5 isocontour is the intended
render surface.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

import numpy as np


VOXEL_SIZE_DX = 0.5
SPHERE_RADIUS_DX = 0.5
GAUSSIAN_SIGMA_DX = 2.0
GAUSSIAN_TRUNCATE = 4.0
FEATURE_THRESHOLD = 2.0
FEATURE_SLOPE = 5.0
FEATURE_EPSILON = 1e-10
SPHERE_RAMP_WIDTH_VOXELS = 1.0
SURFACE_ISOVALUE = 0.5
NORMAL_EPSILON = 1e-6
DEFAULT_MCF_ITERATIONS = 30
DEFAULT_MAX_VOXELS = 32_000_000
DEFAULT_PARTICLE_CHUNK_SIZE = 8_192


@dataclass(frozen=True, slots=True)
class SurfaceReconstruction:
    """A cropped implicit render field and audit diagnostics.

    ``density`` remains on the selected NumPy/CuPy backend and is always
    float32. ``origin`` and ``voxel_size`` are host scalars in world units.
    """

    density: Any
    origin: tuple[float, float, float]
    voxel_size: float
    diagnostics: dict[str, Any]


def _array_module(value, array_module=None):
    if array_module is not None:
        return array_module
    if type(value).__module__.split(".", 1)[0] == "cupy":
        import cupy

        return cupy
    return np


def _scalar(value) -> float:
    item = value.item() if hasattr(value, "item") else value
    return float(item)


def _validate_real(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_integer(value, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validate_anchor(value) -> tuple[float, float, float]:
    try:
        anchor = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("grid_anchor must contain three finite values") from exc
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        raise ValueError("grid_anchor must contain three finite values")
    return tuple(float(component) for component in anchor)


def _validate_positions(positions, xp):
    try:
        if bool(xp.iscomplexobj(positions)):
            raise ValueError
        values = xp.asarray(positions, dtype=xp.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "positions must have shape (N, 3) and contain finite values"
        ) from exc
    if values.ndim != 2 or values.shape[1:] != (3,):
        raise ValueError(
            "positions must have shape (N, 3) and contain finite values"
        )
    if not bool(xp.all(xp.isfinite(values)).item()):
        raise ValueError(
            "positions must have shape (N, 3) and contain finite values"
        )
    return xp.ascontiguousarray(values, dtype=xp.float32)


def _validate_field(value, name: str, xp, *, bounded: bool = False):
    try:
        if bool(xp.iscomplexobj(value)):
            raise ValueError
        field = xp.asarray(value, dtype=xp.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite three-dimensional field") from exc
    if field.ndim != 3 or not bool(xp.all(xp.isfinite(field)).item()):
        raise ValueError(f"{name} must be a finite three-dimensional field")
    if bounded and field.size:
        outside = xp.any((field < 0.0) | (field > 1.0))
        if bool(outside.item()):
            raise ValueError(f"{name} values must lie in [0, 1]")
    return xp.ascontiguousarray(field, dtype=xp.float32)


def sphere_ramp(distance, radius: float, voxel_size: float, *, array_module=None):
    """One-voxel linear antialiasing ramp for a sphere indicator.

    The ramp spans ``radius - voxel_size/2`` through
    ``radius + voxel_size/2`` and therefore equals exactly 0.5 at the sphere
    radius.  Taking the maximum over particles produces a union of spheres
    rather than an overlap-dependent sum.
    """

    xp = _array_module(distance, array_module)
    radius = _validate_real(radius, "radius", positive=True)
    voxel_size = _validate_real(voxel_size, "voxel_size", positive=True)
    values = xp.asarray(distance, dtype=xp.float32)
    return xp.clip(
        xp.asarray(SURFACE_ISOVALUE, dtype=xp.float32)
        + (xp.asarray(radius, dtype=xp.float32) - values)
        / xp.asarray(
            SPHERE_RAMP_WIDTH_VOXELS * voxel_size, dtype=xp.float32),
        0.0,
        1.0,
    ).astype(xp.float32)


def _gaussian_kernel(xp, sigma_voxels: float):
    sigma_voxels = _validate_real(
        sigma_voxels, "sigma_voxels", positive=True)
    radius = max(1, int(math.ceil(GAUSSIAN_TRUNCATE * sigma_voxels)))
    offsets = xp.arange(-radius, radius + 1, dtype=xp.float32)
    sigma = xp.asarray(sigma_voxels, dtype=xp.float32)
    kernel = xp.exp(-0.5 * (offsets / sigma) ** 2).astype(xp.float32)
    kernel /= kernel.sum()
    return kernel.astype(xp.float32), radius


def _convolve_axis(values, kernel, radius: int, axis: int, xp):
    padding = [(0, 0)] * 3
    padding[axis] = (radius, radius)
    padded = xp.pad(values, padding, mode="constant", constant_values=0.0)
    result = xp.zeros_like(values, dtype=xp.float32)
    for offset in range(2 * radius + 1):
        source = [slice(None)] * 3
        source[axis] = slice(offset, offset + values.shape[axis])
        result += kernel[offset] * padded[tuple(source)]
    return result.astype(xp.float32)


def gaussian_blur(density, *, sigma_voxels: float = 4.0,
                  array_module=None):
    """Separable, zero-extended Gaussian blur without SciPy.

    The discrete kernel is normalized and truncated at four standard
    deviations. Appendix B's fixed ``sigma=2*dx`` becomes four voxels on the
    twice-resolution reconstruction grid.
    """

    xp = _array_module(density, array_module)
    values = _validate_field(density, "density", xp)
    if values.size == 0:
        return values.copy()
    kernel, radius = _gaussian_kernel(xp, sigma_voxels)
    result = values
    for axis in range(3):
        result = _convolve_axis(result, kernel, radius, axis, xp)
    return result.astype(xp.float32)


def feature_mask(density, blurred_density, *, array_module=None):
    """Return Appendix B's fixed self-quotient feature mask.

    ``g = 1 - smoothstep(2, 5, density/(blurred_density + 1e-10))``.
    The mask is computed once from the unsmoothed density and held fixed for
    every mean-curvature-flow iteration.
    """

    xp = _array_module(density, array_module)
    source = _validate_field(density, "density", xp)
    blurred = _validate_field(blurred_density, "blurred_density", xp)
    if source.shape != blurred.shape:
        raise ValueError("density and blurred_density must have identical shapes")
    quotient = source / (
        blurred + xp.asarray(FEATURE_EPSILON, dtype=xp.float32))
    t = xp.clip(
        (quotient - FEATURE_THRESHOLD) / (FEATURE_SLOPE - FEATURE_THRESHOLD),
        0.0,
        1.0,
    ).astype(xp.float32)
    smooth = t * t * (3.0 - 2.0 * t)
    return (1.0 - smooth).astype(xp.float32)


def _central_difference(values, axis: int, xp):
    padding = [(0, 0)] * 3
    padding[axis] = (1, 1)
    padded = xp.pad(values, padding, mode="edge")
    before = [slice(None)] * 3
    after = [slice(None)] * 3
    before[axis] = slice(0, values.shape[axis])
    after[axis] = slice(2, values.shape[axis] + 2)
    return (0.5 * (padded[tuple(after)] - padded[tuple(before)])).astype(
        xp.float32)


def _pseudo_time_step(mask) -> float:
    if mask.size == 0:
        return 0.0
    maximum = _scalar(mask.max())
    return 0.0 if maximum <= 0.0 else 1.0 / (6.0 * maximum)


def mean_curvature_flow(density, mask, *, iterations: int = DEFAULT_MCF_ITERATIONS,
                        array_module=None):
    """Apply fixed-mask level-set mean-curvature flow in voxel coordinates."""

    xp = _array_module(density, array_module)
    values = _validate_field(density, "density", xp, bounded=True).copy()
    fixed_mask = _validate_field(mask, "mask", xp, bounded=True)
    if values.shape != fixed_mask.shape:
        raise ValueError("density and mask must have identical shapes")
    iterations = _validate_integer(iterations, "iterations", minimum=0)
    if values.size == 0 or iterations == 0:
        return values
    dtau = _pseudo_time_step(fixed_mask)
    if dtau == 0.0:
        return values
    dtau_device = xp.asarray(dtau, dtype=xp.float32)
    epsilon = xp.asarray(NORMAL_EPSILON, dtype=xp.float32)
    for _ in range(iterations):
        gradient = tuple(
            _central_difference(values, axis, xp) for axis in range(3))
        magnitude = xp.sqrt(sum(component * component for component in gradient))
        denominator = xp.maximum(magnitude, epsilon)
        normals = tuple(component / denominator for component in gradient)
        curvature = sum(
            _central_difference(normals[axis], axis, xp)
            for axis in range(3)
        )
        values = xp.clip(
            values + dtau_device * fixed_mask * magnitude * curvature,
            0.0,
            1.0,
        ).astype(xp.float32)
    return values


def _default_padding(iterations: int) -> int:
    # Explicit curvature flow has diffusion distance O(sqrt(k*dtau)). Add two
    # guard samples beyond that band; the compact sphere ramp itself is already
    # included in the crop bounds.
    return int(math.ceil(math.sqrt(iterations / 3.0))) + 2


def _host_bounds(positions, xp) -> tuple[np.ndarray, np.ndarray]:
    """Transfer only six crop scalars from a GPU backend."""
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    if xp is not np:
        minimum = xp.asnumpy(minimum)
        maximum = xp.asnumpy(maximum)
    return (
        np.asarray(minimum, dtype=np.float64),
        np.asarray(maximum, dtype=np.float64),
    )


def _maximum_at(xp, target, indices, values) -> None:
    """Duplicate-index maximum scatter shared by NumPy and modern CuPy."""
    operation = getattr(xp.maximum, "at", None)
    if operation is not None:
        operation(target, indices, values)
        return
    # Compatibility fallback for older CuPy releases. It has the same atomic
    # duplicate-index semantics as ``maximum.at``.
    import cupyx

    cupyx.scatter_max(target, indices, values)


def _splat_union_of_spheres(
    density,
    positions,
    origin,
    voxel_size: float,
    sphere_radius: float,
    particle_chunk_size: int,
    xp,
) -> None:
    """Chunked compact-stencil max scatter for union-of-spheres density."""
    support = sphere_radius + 0.5 * voxel_size
    stencil_radius = int(math.ceil(support / voxel_size))
    axis_offsets = xp.arange(
        -stencil_radius, stencil_radius + 1, dtype=xp.int32)
    oi, oj, ok = xp.meshgrid(
        axis_offsets, axis_offsets, axis_offsets, indexing="ij")
    offsets = xp.stack((oi.ravel(), oj.ravel(), ok.ravel()), axis=1)
    origin_device = xp.asarray(origin, dtype=xp.float32)
    h = xp.asarray(voxel_size, dtype=xp.float32)
    shape_device = xp.asarray(density.shape, dtype=xp.int32)
    flat_density = density.ravel()
    ny, nz = int(density.shape[1]), int(density.shape[2])

    for start in range(0, int(positions.shape[0]), particle_chunk_size):
        particles = positions[start:start + particle_chunk_size]
        base = xp.floor((particles - origin_device) / h).astype(xp.int32)
        indices = base[:, None, :] + offsets[None, :, :]
        valid = xp.all((indices >= 0) & (indices < shape_device), axis=2)
        samples = origin_device + indices.astype(xp.float32) * h
        delta = samples - particles[:, None, :]
        distance = xp.sqrt((delta * delta).sum(axis=2))
        contribution = sphere_ramp(
            distance, sphere_radius, voxel_size, array_module=xp)
        clipped = xp.clip(indices, 0, shape_device - 1)
        flat = (
            (clipped[:, :, 0] * ny + clipped[:, :, 1]) * nz
            + clipped[:, :, 2]
        ).astype(xp.int32)
        # Invalid candidates are safely mapped to zero with a zero value. This
        # avoids a variable-length boolean gather and keeps GPU memory bounded
        # and launch count independent of the number of particles in a chunk.
        flat = xp.where(valid, flat, 0).ravel()
        values = xp.where(valid, contribution, 0.0).astype(xp.float32).ravel()
        _maximum_at(xp, flat_density, flat, values)


def _diagnostics_base(*, particle_count: int, voxel_size: float,
                      sphere_radius: float, iterations: int, padding: int,
                      max_voxels: int, particle_chunk_size: int,
                      xp) -> dict[str, Any]:
    return {
        "particle_count": particle_count,
        "voxel_size_dx": VOXEL_SIZE_DX,
        "voxel_size": voxel_size,
        "sphere_radius_dx": SPHERE_RADIUS_DX,
        "sphere_radius": sphere_radius,
        "sphere_ramp_width_voxels": SPHERE_RAMP_WIDTH_VOXELS,
        "gaussian_sigma_dx": GAUSSIAN_SIGMA_DX,
        "gaussian_sigma_voxels": GAUSSIAN_SIGMA_DX / VOXEL_SIZE_DX,
        "gaussian_truncate": GAUSSIAN_TRUNCATE,
        "feature_threshold": FEATURE_THRESHOLD,
        "feature_slope": FEATURE_SLOPE,
        "feature_epsilon": FEATURE_EPSILON,
        "iterations_requested": iterations,
        "padding_voxels": padding,
        "max_voxels": max_voxels,
        "particle_chunk_size": particle_chunk_size,
        "array_module": getattr(xp, "__name__", type(xp).__name__),
    }


def reconstruct_surface(
    positions,
    dx: float,
    *,
    iterations: int = DEFAULT_MCF_ITERATIONS,
    padding_voxels: int | None = None,
    max_voxels: int = DEFAULT_MAX_VOXELS,
    particle_chunk_size: int = DEFAULT_PARTICLE_CHUNK_SIZE,
    grid_anchor=(0.0, 0.0, 0.0),
    array_module=None,
) -> SurfaceReconstruction:
    """Reconstruct the cropped Appendix-B implicit field from particles.

    The dense reference implementation is deterministic for a fixed backend
    and particle order. It checks the crop against ``max_voxels`` before any
    three-dimensional allocation.
    """

    xp = _array_module(positions, array_module)
    positions = _validate_positions(positions, xp)
    dx = _validate_real(dx, "dx", positive=True)
    voxel_size = VOXEL_SIZE_DX * dx
    sphere_radius = SPHERE_RADIUS_DX * dx
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("dx is too small to define a positive float voxel size")
    iterations = _validate_integer(iterations, "iterations", minimum=0)
    max_voxels = _validate_integer(max_voxels, "max_voxels", minimum=1)
    particle_chunk_size = _validate_integer(
        particle_chunk_size, "particle_chunk_size", minimum=1)
    anchor = _validate_anchor(grid_anchor)
    if padding_voxels is None:
        padding = _default_padding(iterations)
    else:
        padding = _validate_integer(
            padding_voxels, "padding_voxels", minimum=0)

    count = int(positions.shape[0])
    diagnostics = _diagnostics_base(
        particle_count=count,
        voxel_size=voxel_size,
        sphere_radius=sphere_radius,
        iterations=iterations,
        padding=padding,
        max_voxels=max_voxels,
        particle_chunk_size=particle_chunk_size,
        xp=xp,
    )
    if count == 0:
        diagnostics.update({
            "empty": True,
            "grid_shape": (0, 0, 0),
            "voxel_count": 0,
            "iterations_completed": 0,
            "pseudo_time_step": 0.0,
            "feature_mask_min": None,
            "feature_mask_max": None,
        })
        return SurfaceReconstruction(
            density=xp.zeros((0, 0, 0), dtype=xp.float32),
            origin=anchor,
            voxel_size=voxel_size,
            diagnostics=diagnostics,
        )

    position_min, position_max = _host_bounds(positions, xp)
    anchor_array = np.asarray(anchor, dtype=np.float64)
    # The linear ramp is nonzero through radius + h/2.
    support = sphere_radius + 0.5 * voxel_size
    lower = [
        math.floor((float(position_min[axis]) - support - anchor[axis])
                   / voxel_size) - padding
        for axis in range(3)
    ]
    upper = [
        math.ceil((float(position_max[axis]) + support - anchor[axis])
                  / voxel_size) + padding
        for axis in range(3)
    ]
    shape = tuple(upper[axis] - lower[axis] + 1 for axis in range(3))
    voxel_count = math.prod(shape)
    if voxel_count > max_voxels:
        raise MemoryError(
            f"surface crop requires {voxel_count} voxels with shape {shape}, "
            f"exceeding max_voxels={max_voxels}"
        )
    origin_array = anchor_array + np.asarray(lower, dtype=np.float64) * voxel_size
    origin = tuple(float(value) for value in origin_array)
    density = xp.zeros(shape, dtype=xp.float32)

    _splat_union_of_spheres(
        density,
        positions,
        origin,
        voxel_size,
        sphere_radius,
        particle_chunk_size,
        xp,
    )

    blurred = gaussian_blur(
        density,
        sigma_voxels=GAUSSIAN_SIGMA_DX / VOXEL_SIZE_DX,
        array_module=xp,
    )
    mask = feature_mask(density, blurred, array_module=xp)
    dtau = _pseudo_time_step(mask)
    result = mean_curvature_flow(
        density, mask, iterations=iterations, array_module=xp)
    diagnostics.update({
        "empty": False,
        "grid_shape": shape,
        "voxel_count": voxel_count,
        "iterations_completed": iterations,
        "pseudo_time_step": dtau,
        "feature_mask_min": _scalar(mask.min()),
        "feature_mask_max": _scalar(mask.max()),
        "initial_density_min": _scalar(density.min()),
        "initial_density_max": _scalar(density.max()),
        "final_density_min": _scalar(result.min()),
        "final_density_max": _scalar(result.max()),
        "estimated_dense_working_set_bytes": voxel_count * 4 * 12,
    })
    return SurfaceReconstruction(
        density=result.astype(xp.float32),
        origin=origin,
        voxel_size=voxel_size,
        diagnostics=diagnostics,
    )


__all__ = [
    "DEFAULT_MAX_VOXELS",
    "DEFAULT_MCF_ITERATIONS",
    "DEFAULT_PARTICLE_CHUNK_SIZE",
    "FEATURE_EPSILON",
    "FEATURE_SLOPE",
    "FEATURE_THRESHOLD",
    "GAUSSIAN_SIGMA_DX",
    "GAUSSIAN_TRUNCATE",
    "NORMAL_EPSILON",
    "SPHERE_RADIUS_DX",
    "SPHERE_RAMP_WIDTH_VOXELS",
    "SurfaceReconstruction",
    "SURFACE_ISOVALUE",
    "VOXEL_SIZE_DX",
    "feature_mask",
    "gaussian_blur",
    "mean_curvature_flow",
    "reconstruct_surface",
    "sphere_ramp",
]
