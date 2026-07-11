"""Fractional solid apertures for Cartesian MAC grids.

Signed-distance values are positive in the open-fluid region and negative
inside solids.  The helpers in this module are deliberately independent of
Blender and use only operations shared by NumPy and CuPy.
"""

from __future__ import annotations

import math

import numpy as np


def _array_module(array_module):
    return np if array_module is None else array_module


def _numeric_array(xp, value, name: str):
    array = xp.asarray(value)
    try:
        numeric = bool(xp.issubdtype(array.dtype, xp.number))
    except (AttributeError, TypeError):
        numeric = False
    if not numeric:
        raise TypeError(f"{name} must contain numeric values")
    # Inspecting device values would introduce an otherwise unnecessary GPU
    # synchronization. Shape and dtype checks remain safe for either backend;
    # strict value validation is performed for host NumPy arrays.
    if xp is np and not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _broadcast_numeric(xp, named_values):
    arrays = tuple(
        _numeric_array(xp, value, name) for name, value in named_values
    )
    try:
        return xp.broadcast_arrays(*arrays)
    except ValueError as exc:
        names = ", ".join(name for name, _value in named_values)
        raise ValueError(f"{names} must have broadcast-compatible shapes") from exc


def _safe_ratio(xp, numerator, denominator):
    safe_denominator = xp.where(denominator != 0, denominator, 1)
    return numerator / safe_denominator


def _triangle_open_fraction_arrays(xp, phi0, phi1, phi2):
    """Area fraction where a linearly interpolated triangle SDF is positive."""
    values = xp.stack((phi0, phi1, phi2), axis=-1)
    ordered = xp.sort(values, axis=-1)
    low = ordered[..., 0]
    middle = ordered[..., 1]
    high = ordered[..., 2]
    positive_count = (ordered > 0).sum(axis=-1)

    # With one positive vertex, the open region is the similar triangle at
    # that vertex. With two, subtract the analogous closed triangle.
    one_positive = (
        _safe_ratio(xp, high, high - low)
        * _safe_ratio(xp, high, high - middle)
    )
    two_positive = 1.0 - (
        _safe_ratio(xp, -low, middle - low)
        * _safe_ratio(xp, -low, high - low)
    )
    fraction = xp.where(
        positive_count == 0,
        0.0,
        xp.where(
            positive_count == 1,
            one_positive,
            xp.where(positive_count == 2, two_positive, 1.0),
        ),
    )
    return xp.clip(fraction, 0.0, 1.0)


def triangle_open_fraction(
    phi0,
    phi1,
    phi2,
    array_module=None,
):
    """Return the positive-SDF area fraction of one or many triangles.

    Inputs may be scalars or broadcast-compatible arrays. The SDF is assumed
    to vary linearly over each triangle.
    """
    xp = _array_module(array_module)
    phi0, phi1, phi2 = _broadcast_numeric(
        xp,
        (("phi0", phi0), ("phi1", phi1), ("phi2", phi2)),
    )
    return _triangle_open_fraction_arrays(xp, phi0, phi1, phi2)


def _square_open_fraction_arrays(xp, phi00, phi10, phi01, phi11):
    # Join the four corners to their bilinearly interpolated center. This
    # symmetric four-triangle construction avoids choosing a privileged
    # diagonal while remaining exact for planar signed-distance fields.
    center = (phi00 + phi10 + phi01 + phi11) * 0.25
    return 0.25 * (
        _triangle_open_fraction_arrays(xp, phi00, phi10, center)
        + _triangle_open_fraction_arrays(xp, phi10, phi11, center)
        + _triangle_open_fraction_arrays(xp, phi11, phi01, center)
        + _triangle_open_fraction_arrays(xp, phi01, phi00, center)
    )


def square_open_fraction(
    phi00,
    phi10,
    phi01,
    phi11,
    array_module=None,
):
    """Return the positive-SDF area fraction of axis-aligned unit squares.

    ``phi00``, ``phi10``, ``phi01``, and ``phi11`` are the signed-distance
    values at the square corners. Inputs may be broadcast-compatible arrays.
    """
    xp = _array_module(array_module)
    phi00, phi10, phi01, phi11 = _broadcast_numeric(
        xp,
        (
            ("phi00", phi00),
            ("phi10", phi10),
            ("phi01", phi01),
            ("phi11", phi11),
        ),
    )
    return _square_open_fraction_arrays(xp, phi00, phi10, phi01, phi11)


def _validated_node_sdf(node_sdf, array_module=None):
    xp = _array_module(array_module)
    sdf = _numeric_array(xp, node_sdf, "node_sdf")
    if sdf.ndim != 3:
        raise ValueError(
            f"node_sdf must be a 3D array, received shape {sdf.shape}"
        )
    if any(size < 2 for size in sdf.shape):
        raise ValueError(
            "node_sdf needs at least two nodes along every axis; "
            f"received shape {sdf.shape}"
        )
    return xp, sdf


def face_apertures_from_node_sdf(node_sdf, array_module=None):
    """Compute open fractions for all faces of a MAC grid.

    A node grid of shape ``(nx + 1, ny + 1, nz + 1)`` produces ``u``, ``v``,
    and ``w`` apertures with shapes ``(nx + 1, ny, nz)``,
    ``(nx, ny + 1, nz)``, and ``(nx, ny, nz + 1)`` respectively.
    """
    xp, sdf = _validated_node_sdf(node_sdf, array_module)

    alpha_u = _square_open_fraction_arrays(
        xp,
        sdf[:, :-1, :-1],
        sdf[:, 1:, :-1],
        sdf[:, :-1, 1:],
        sdf[:, 1:, 1:],
    )
    alpha_v = _square_open_fraction_arrays(
        xp,
        sdf[:-1, :, :-1],
        sdf[1:, :, :-1],
        sdf[:-1, :, 1:],
        sdf[1:, :, 1:],
    )
    alpha_w = _square_open_fraction_arrays(
        xp,
        sdf[:-1, :-1, :],
        sdf[1:, :-1, :],
        sdf[:-1, 1:, :],
        sdf[1:, 1:, :],
    )
    return alpha_u, alpha_v, alpha_w


def solid_cells_from_node_sdf(node_sdf, array_module=None):
    """Classify cells that are completely covered by solid geometry.

    A cell is removed from the pressure system only when all eight corner
    samples are strictly inside the solid.  Partially cut cells remain active
    so their fractional face apertures do not become false pressure-Dirichlet
    boundaries. A node grid of shape ``(nx + 1, ny + 1, nz + 1)`` returns a
    boolean ``(nx, ny, nz)`` mask.
    """
    xp, sdf = _validated_node_sdf(node_sdf, array_module)
    corners = (
        sdf[:-1, :-1, :-1],
        sdf[1:, :-1, :-1],
        sdf[:-1, 1:, :-1],
        sdf[1:, 1:, :-1],
        sdf[:-1, :-1, 1:],
        sdf[1:, :-1, 1:],
        sdf[:-1, 1:, 1:],
        sdf[1:, 1:, 1:],
    )
    maximum = corners[0]
    # CuPy 14 does not implement ``maximum.reduce``; pairwise ufunc calls
    # preserve the shared NumPy/CuPy code path without stacking eight grids.
    for corner in corners[1:]:
        maximum = xp.maximum(maximum, corner)
    return maximum < 0.0


def _validate_face_shapes(u, v, w, alpha_u, alpha_v, alpha_w):
    named = {
        "u": u,
        "v": v,
        "w": w,
        "alpha_u": alpha_u,
        "alpha_v": alpha_v,
        "alpha_w": alpha_w,
    }
    for name, array in named.items():
        if array.ndim != 3:
            raise ValueError(
                f"{name} must be a 3D face array, received shape {array.shape}"
            )

    if u.shape != alpha_u.shape:
        raise ValueError("u and alpha_u must have identical shapes")
    if v.shape != alpha_v.shape:
        raise ValueError("v and alpha_v must have identical shapes")
    if w.shape != alpha_w.shape:
        raise ValueError("w and alpha_w must have identical shapes")
    if u.shape[0] < 2 or u.shape[1] < 1 or u.shape[2] < 1:
        raise ValueError(f"u has no valid cell grid: shape {u.shape}")

    cell_shape = (u.shape[0] - 1, u.shape[1], u.shape[2])
    expected_v = (cell_shape[0], cell_shape[1] + 1, cell_shape[2])
    expected_w = (cell_shape[0], cell_shape[1], cell_shape[2] + 1)
    if v.shape != expected_v:
        raise ValueError(f"v must have shape {expected_v}, received {v.shape}")
    if w.shape != expected_w:
        raise ValueError(f"w must have shape {expected_w}, received {w.shape}")


def weighted_divergence(
    u,
    v,
    w,
    alpha_u,
    alpha_v,
    alpha_w,
    dx,
    array_module=None,
):
    """Return ``div(alpha * velocity)`` at MAC-grid cell centers."""
    try:
        dx_value = float(dx)
    except (TypeError, ValueError) as exc:
        raise TypeError("dx must be a finite positive scalar") from exc
    if not math.isfinite(dx_value) or dx_value <= 0.0:
        raise ValueError("dx must be a finite positive scalar")

    xp = _array_module(array_module)
    u = _numeric_array(xp, u, "u")
    v = _numeric_array(xp, v, "v")
    w = _numeric_array(xp, w, "w")
    alpha_u = _numeric_array(xp, alpha_u, "alpha_u")
    alpha_v = _numeric_array(xp, alpha_v, "alpha_v")
    alpha_w = _numeric_array(xp, alpha_w, "alpha_w")
    _validate_face_shapes(u, v, w, alpha_u, alpha_v, alpha_w)

    if xp is np:
        for name, alpha in (
            ("alpha_u", alpha_u),
            ("alpha_v", alpha_v),
            ("alpha_w", alpha_w),
        ):
            if bool(np.any((alpha < 0.0) | (alpha > 1.0))):
                raise ValueError(f"{name} values must lie in [0, 1]")

    return (
        alpha_u[1:, :, :] * u[1:, :, :]
        - alpha_u[:-1, :, :] * u[:-1, :, :]
        + alpha_v[:, 1:, :] * v[:, 1:, :]
        - alpha_v[:, :-1, :] * v[:, :-1, :]
        + alpha_w[:, :, 1:] * w[:, :, 1:]
        - alpha_w[:, :, :-1] * w[:, :, :-1]
    ) / dx_value


__all__ = [
    "face_apertures_from_node_sdf",
    "solid_cells_from_node_sdf",
    "square_open_fraction",
    "triangle_open_fraction",
    "weighted_divergence",
]
