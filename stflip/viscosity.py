"""Implicit viscosity (Stam-style decoupled Laplacian diffusion).

ST-FLIP's whole point is very large time steps, so viscosity must be solved
*implicitly* — an explicit update u += dt*nu*lap(u) is stable only for
dt < dx^2/(6*nu), which collapses the time step for thick fluids and would
forfeit the method's advantage.  Instead we solve the unconditionally stable
backward-Euler diffusion per velocity component on its MAC face grid,

    (I - dt * nu * L) u^{n+1} = u*,

where L is the 6-point Laplacian and nu is the kinematic viscosity (dx^2/s).
Fully-blocked solid faces are Dirichlet (no-slip: held at the solid velocity);
domain and free-surface faces are natural (Neumann).  The operator is SPD, so
a Jacobi-preconditioned CG converges quickly.  This is the same "simple
viscosity" model production Blender tools expose as a thickness slider; it uses
the decoupled Laplacian rather than the full div(mu(grad u + grad u^T)) stress,
which is the standard graphics approximation.

Array-module agnostic (NumPy or CuPy); no cupy.linalg (plain reductions only).
"""

from __future__ import annotations

import math


def _laplacian(xp, p):
    """6-point Laplacian sum_nb (p_nb - p_i); missing neighbours contribute 0."""
    lap = xp.zeros_like(p)
    lap[:-1] += p[1:] - p[:-1]
    lap[1:] += p[:-1] - p[1:]
    lap[:, :-1] += p[:, 1:] - p[:, :-1]
    lap[:, 1:] += p[:, :-1] - p[:, 1:]
    lap[:, :, :-1] += p[:, :, 1:] - p[:, :, :-1]
    lap[:, :, 1:] += p[:, :, :-1] - p[:, :, 1:]
    return lap


def _neighbor_count(xp, shape):
    """In-grid neighbour count per node (6 interior, fewer on the boundary)."""
    c = xp.full(shape, 6.0, dtype=xp.float32)
    c[0] -= 1.0
    c[-1] -= 1.0
    c[:, 0] -= 1.0
    c[:, -1] -= 1.0
    c[:, :, 0] -= 1.0
    c[:, :, -1] -= 1.0
    return c


def diffuse_component(xp, u, coef, fixed, fixed_value, tol=1e-5,
                      max_iter=200):
    """Solve (I - coef*L) x = u on the free faces of one MAC component.

    ``coef = dt * nu / dx^2``.  ``fixed`` is a boolean array of Dirichlet faces
    (solids) held at ``fixed_value``; every other face is a free DOF with a
    natural boundary.  Returns the diffused component.
    """
    if coef <= 0.0:
        return u
    free = ~fixed
    # Initial guess pins the Dirichlet faces at the solid velocity so their
    # contribution enters the free-face equations through the first residual.
    x = xp.where(fixed, fixed_value, u).astype(xp.float32)
    b = xp.where(free, u, 0.0).astype(xp.float32)

    def A(p):
        # p is zero on fixed faces; the Laplacian then treats each fixed
        # neighbour as a Dirichlet wall (it subtracts p_i, adds nothing back).
        return xp.where(free, p - coef * _laplacian(xp, p), 0.0)

    diag = 1.0 + coef * _neighbor_count(xp, u.shape)
    inv_diag = xp.where(free, 1.0 / diag, 0.0)

    r = xp.where(free, b - A(x), 0.0)
    b_norm = math.sqrt(float((b * b).sum()))
    if b_norm < 1e-30:
        return x
    z = inv_diag * r
    s = z.copy()
    sigma = (z * r).sum()
    for _ in range(1, max_iter + 1):
        As = A(s)
        sAs = (s * As).sum()
        ok = xp.abs(sAs) > 1e-30
        alpha = xp.where(ok, sigma / xp.where(ok, sAs, 1.0), 0.0)
        x = x + alpha * s
        r = r - alpha * As
        if math.sqrt(float((r * r).sum())) / b_norm <= tol:
            break
        z = inv_diag * r
        sigma_new = (z * r).sum()
        ok = xp.abs(sigma) > 1e-30
        beta = xp.where(ok, sigma_new / xp.where(ok, sigma, 1.0), 0.0)
        sigma = sigma_new
        s = z + beta * s
    return x
