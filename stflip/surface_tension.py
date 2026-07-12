"""Continuum Surface Force (CSF) surface tension (ST-FLIP Sec 3.9).

The paper applies a standard CSF model [Brackbill et al. 1992] with interface
curvature estimated from a cubic B-spline-smoothed reconstruction of the phase
field.  We reuse the cell-centred phase field ``phi_c`` (the liquid fraction /
space-time phase accumulator that P2G already produces), smooth it with a few
separable binomial passes (a discrete cubic-B-spline approximation), and form
the body force

    F = sigma * kappa * grad(phi_s),    kappa = -div( grad(phi_s) / |grad(phi_s)| )

which the solver adds to the grid velocity as an acceleration dt * F / rho.
``grad(phi_s)`` doubles as the smeared surface delta, so F concentrates on the
diffuse interface exactly where it should.

The module is bpy-free and array-module agnostic (NumPy or CuPy).
"""

from __future__ import annotations

# Separable binomial [1 4 6 4 1]/16 stencil: one pass is a cubic B-spline
# smoothing step; repeated passes widen the effective interface support.
_B = (1.0, 4.0, 6.0, 4.0, 16.0)


def _smooth1(xp, a, axis):
    """One [1 4 6 4]/16 binomial pass along ``axis`` with clamped edges."""
    s = 6.0 * a
    s = s + 4.0 * (_shift(xp, a, axis, +1) + _shift(xp, a, axis, -1))
    s = s + 1.0 * (_shift(xp, a, axis, +2) + _shift(xp, a, axis, -2))
    return s / 16.0


def _shift(xp, a, axis, off):
    """Shift ``a`` by ``off`` along ``axis`` with edge (clamp) padding."""
    if off == 0:
        return a
    sl = [slice(None)] * a.ndim
    out = xp.empty_like(a)
    n = a.shape[axis]
    if off > 0:
        src = slice(0, max(n - off, 0))
        dst = slice(off, n)
        edge = slice(0, 1)
    else:
        src = slice(-off, n)
        dst = slice(0, n + off)
        edge = slice(n - 1, n)
    # Interior copy.
    idx_dst = list(sl)
    idx_dst[axis] = dst
    idx_src = list(sl)
    idx_src[axis] = src
    out[tuple(idx_dst)] = a[tuple(idx_src)]
    # Clamp the exposed border to the nearest edge value.
    idx_fill = list(sl)
    idx_edge = list(sl)
    idx_edge[axis] = edge
    idx_fill[axis] = slice(0, off) if off > 0 else slice(n + off, n)
    out[tuple(idx_fill)] = a[tuple(idx_edge)]
    return out


def smooth_phase(xp, phi, iters):
    """Cubic-B-spline-style smoothing of the phase field (``iters`` passes)."""
    s = phi
    for _ in range(max(int(iters), 0)):
        s = _smooth1(xp, s, 0)
        s = _smooth1(xp, s, 1)
        s = _smooth1(xp, s, 2)
    return s


def cell_force(xp, phi_c, dx, sigma, iters=2, eps=1e-6):
    """Cell-centred CSF body force F = sigma * kappa * grad(phi_s).

    ``phi_c`` is the cell-centred liquid fraction in [0, 1].  Returns an
    (nx, ny, nz, 3) force-density field (units sigma / length^2).
    """
    phi_s = smooth_phase(xp, phi_c, iters)
    gx, gy, gz = xp.gradient(phi_s, dx)
    mag = xp.sqrt(gx * gx + gy * gy + gz * gz)
    inv = 1.0 / xp.maximum(mag, eps)
    nx_, ny_, nz_ = gx * inv, gy * inv, gz * inv
    # kappa = -div(n_hat)
    dnx = xp.gradient(nx_, dx, axis=0)
    dny = xp.gradient(ny_, dx, axis=1)
    dnz = xp.gradient(nz_, dx, axis=2)
    kappa = -(dnx + dny + dnz)
    # Only meaningful on the diffuse interface, where |grad phi| is non-trivial.
    f = sigma * kappa
    F = xp.stack([f * gx, f * gy, f * gz], axis=-1)
    return F.astype(xp.float32)
