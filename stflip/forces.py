"""Art-directable body forces / guides (turbulence, vortex, wind).

Force fields let an artist shape a liquid beyond gravity + inflow velocity —
the guides Mantaflow exposes and the volume/curve forces FLIP Fluids exposes.
Each force contributes a cell-centred acceleration field a(x, t) that the
solver adds to the grid velocity like gravity (u <- u + dt * a); the pressure
projection then keeps the flow incompressible.

Three field types:
  - DIRECTIONAL: a uniform push (wind).
  - VORTEX: a swirl about an axis through a centre, a = s * axis x (x - c),
    with a smooth radial falloff.
  - TURBULENCE: curl of a smooth animated vector potential, so the force is
    divergence-free (adds swirly detail without fighting the pressure solve).

Solver-local cell-centre coordinates are ``(index + 0.5 + origin) * dx`` where
``origin`` is the sparse-window cell offset (0 for the dense solve).  Array-
module agnostic (NumPy or CuPy); randomness for turbulence is drawn once on the
host from the force seed so runs are reproducible.
"""

from __future__ import annotations

import math

import numpy as np

from .surface_tension import smooth_phase


def _solver_local_coords(xp, shape, dx, origin):
    nx, ny, nz = shape
    ax = (xp.arange(nx, dtype=xp.float32) + 0.5 + float(origin[0])) * dx
    ay = (xp.arange(ny, dtype=xp.float32) + 0.5 + float(origin[1])) * dx
    az = (xp.arange(nz, dtype=xp.float32) + 0.5 + float(origin[2])) * dx
    return xp.meshgrid(ax, ay, az, indexing="ij")


def directional_accel(xp, shape, dx, direction, strength):
    """Uniform acceleration ``strength * normalized(direction)`` (wind)."""
    d = np.asarray(direction, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return None
    d = (d / n) * float(strength)
    out = xp.zeros(tuple(shape) + (3,), dtype=xp.float32)
    out[..., 0] = xp.float32(d[0])
    out[..., 1] = xp.float32(d[1])
    out[..., 2] = xp.float32(d[2])
    return out


def vortex_accel(xp, shape, dx, center, axis, strength, radius, origin):
    """Swirl about ``axis`` through ``center`` with a smooth radial falloff.

    ``a = strength * (axis x r_perp) * falloff(|r_perp| / radius)`` where
    r_perp is the component of (x - center) perpendicular to the axis."""
    ax_ = np.asarray(axis, dtype=np.float64)
    na = float(np.linalg.norm(ax_))
    if na < 1e-12 or strength == 0.0:
        return None
    ax_ = ax_ / na
    X, Y, Z = _solver_local_coords(xp, shape, dx, origin)
    rx = X - float(center[0])
    ry = Y - float(center[1])
    rz = Z - float(center[2])
    # r_perp = r - (r.axis) axis
    dot = rx * ax_[0] + ry * ax_[1] + rz * ax_[2]
    px = rx - dot * ax_[0]
    py = ry - dot * ax_[1]
    pz = rz - dot * ax_[2]
    # swirl direction = axis x r_perp
    sx = ax_[1] * pz - ax_[2] * py
    sy = ax_[2] * px - ax_[0] * pz
    sz = ax_[0] * py - ax_[1] * px
    r2 = px * px + py * py + pz * pz
    rad = max(float(radius), 1e-6)
    falloff = xp.exp(-r2 / (rad * rad))
    s = xp.float32(strength) * falloff
    out = xp.empty(tuple(shape) + (3,), dtype=xp.float32)
    out[..., 0] = s * sx
    out[..., 1] = s * sy
    out[..., 2] = s * sz
    return out


def turbulence_accel(xp, shape, dx, strength, scale, time, seed, origin,
                     octaves=3):
    """Divergence-free turbulence: curl of a smooth animated vector potential.

    ``scale`` is the coarsest spatial wavelength (world units); each octave
    halves it.  ``time`` animates the field.  Returns a cell-centred (…, 3)
    acceleration whose magnitude is ~``strength``."""
    if strength == 0.0:
        return None
    X, Y, Z = _solver_local_coords(xp, shape, dx, origin)
    rng = np.random.default_rng(int(seed))
    psi = [xp.zeros(tuple(shape), dtype=xp.float32) for _ in range(3)]
    base_f = 2.0 * math.pi / max(float(scale), 1e-6)
    for octave in range(max(int(octaves), 1)):
        f = base_f * (2 ** octave)
        amp = float(strength) / f * (0.6 ** octave)
        for comp in range(3):
            k = rng.standard_normal(3)
            k = k / (np.linalg.norm(k) + 1e-12)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            arg = (f * (k[0] * X + k[1] * Y + k[2] * Z)
                   + phase + float(time) * (octave + 1) * 0.7)
            psi[comp] = psi[comp] + xp.float32(amp) * xp.sin(arg)
    # a = curl(psi) via central differences (approximately divergence-free)
    dpz_dy = xp.gradient(psi[2], dx, axis=1)
    dpy_dz = xp.gradient(psi[1], dx, axis=2)
    dpx_dz = xp.gradient(psi[0], dx, axis=2)
    dpz_dx = xp.gradient(psi[2], dx, axis=0)
    dpy_dx = xp.gradient(psi[1], dx, axis=0)
    dpx_dy = xp.gradient(psi[0], dx, axis=1)
    out = xp.empty(tuple(shape) + (3,), dtype=xp.float32)
    out[..., 0] = dpz_dy - dpy_dz
    out[..., 1] = dpx_dz - dpz_dx
    out[..., 2] = dpy_dx - dpx_dy
    return out


def confinement_accel(xp, dx, grids, strength, clamp=98.1, smooth_iters=1):
    """Vorticity confinement (Fedkiw et al. 2001): ``a = eps*dx*(N x omega)``.

    Re-energizes swirls that transfers and coarse grids smooth away.  This
    INJECTS energy: it counteracts the LOOK of dissipation and is not a
    conservation fix, so it must stay OFF in validation and decision runs
    (roadmap ENER-M1, Decision 6c).  The acceleration is restricted to
    liquid cells (``c_phi >= 0.5``) and magnitude-clamped per cell (default
    ten gravities) so any slider value stays unconditionally safe.
    """

    if strength == 0.0:
        return None
    u = 0.5 * (grids["u"][1:, :, :] + grids["u"][:-1, :, :])
    v = 0.5 * (grids["v"][:, 1:, :] + grids["v"][:, :-1, :])
    w = 0.5 * (grids["w"][:, :, 1:] + grids["w"][:, :, :-1])
    ox = xp.gradient(w, dx, axis=1) - xp.gradient(v, dx, axis=2)
    oy = xp.gradient(u, dx, axis=2) - xp.gradient(w, dx, axis=0)
    oz = xp.gradient(v, dx, axis=0) - xp.gradient(u, dx, axis=1)
    if smooth_iters > 0:
        # One binomial pass per component tames Monte-Carlo-noise-driven
        # vorticity-magnitude gradients before they are differentiated again.
        ox = smooth_phase(xp, ox, smooth_iters)
        oy = smooth_phase(xp, oy, smooth_iters)
        oz = smooth_phase(xp, oz, smooth_iters)
    mag = xp.sqrt(ox * ox + oy * oy + oz * oz)
    gx = xp.gradient(mag, dx, axis=0)
    gy = xp.gradient(mag, dx, axis=1)
    gz = xp.gradient(mag, dx, axis=2)
    norm = xp.maximum(xp.sqrt(gx * gx + gy * gy + gz * gz), 1e-12)
    nhat_x, nhat_y, nhat_z = gx / norm, gy / norm, gz / norm
    ax = strength * dx * (nhat_y * oz - nhat_z * oy)
    ay = strength * dx * (nhat_z * ox - nhat_x * oz)
    az = strength * dx * (nhat_x * oy - nhat_y * ox)
    liquid = grids["c_phi"] >= 0.5
    out = xp.stack([ax, ay, az], axis=-1) * liquid[..., None]
    amag = xp.sqrt((out * out).sum(axis=-1))
    scale = xp.minimum(1.0, clamp / xp.maximum(amag, 1e-12))
    return (out * scale[..., None]).astype(xp.float32)
