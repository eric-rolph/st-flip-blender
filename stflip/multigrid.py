"""Matrix-free geometric multigrid V-cycle preconditioner for the PPE.

The pressure Poisson operator solved in :mod:`stflip.pressure` is a symmetric,
variable-coefficient 7-point Laplacian restricted to the liquid mask.  On large
grids a Jacobi-preconditioned CG needs many iterations because Jacobi only damps
high-frequency error; the low-frequency error decays slowly.  A geometric
multigrid V-cycle used as the CG preconditioner attacks every frequency band on
its own grid level, so the outer CG converges in far fewer iterations.

Everything here is matrix-free and reuses ``pressure.apply_laplacian`` /
``pressure.diagonal`` for the fine and every coarse level, so the coarse
operator is a genuine rediscretization of the same variable-coefficient stencil
rather than an assembled matrix.  No ``cupy.linalg``/cuBLAS is used (the CuPy
Windows wheels do not bundle it): only elementwise ops, strided slicing, and
plain reductions, so the same code runs on CPU (NumPy) and GPU (CuPy).

Coarsening is geometric by a factor of two per axis and is applied only while
every grid dimension is even and at least ``2 * min_size``; otherwise the
hierarchy stops (semi-coarsening is intentionally not attempted).  A coarse face
coefficient is the mean of the four fine face coefficients on the shared coarse
boundary plane divided by four, which for a constant coefficient reproduces the
exact ``beta / H**2`` rediscretization on the coarse spacing ``H = 2 h``.

The V-cycle is only ever a *preconditioner*: the outer CG in :func:`solve`
checks the true residual, so the returned pressure is correct to ``tol``
regardless of how good the coarse hierarchy is.  A poor hierarchy costs
iterations, never accuracy.
"""

from __future__ import annotations

import math

from . import pressure


def _coarsen_faces(xp, k, axis, ncx, ncy, ncz):
    """Coarsen one MAC face-coefficient array by a factor of two per axis.

    ``k`` holds face coefficients normal to ``axis`` with one extra sample along
    that axis.  A coarse face at coarse plane ``p`` sits on fine plane ``2 p`` and
    spans a 2x2 patch of fine faces in the two tangential directions; its
    coefficient is ``mean(4 fine) / 4`` (parallel-conductance average times the
    ``(h / H)**2`` spacing rescale).
    """
    if axis == 0:
        faces = k[0::2, :, :]                      # (ncx+1, 2*ncy, 2*ncz)
        blocks = faces.reshape(ncx + 1, ncy, 2, ncz, 2)
        return blocks.sum(axis=(2, 4)) / 16.0
    if axis == 1:
        faces = k[:, 0::2, :]                       # (2*ncx, ncy+1, 2*ncz)
        blocks = faces.reshape(ncx, 2, ncy + 1, ncz, 2)
        return blocks.sum(axis=(1, 4)) / 16.0
    faces = k[:, :, 0::2]                           # (2*ncx, 2*ncy, ncz+1)
    blocks = faces.reshape(ncx, 2, ncy, 2, ncz + 1)
    return blocks.sum(axis=(1, 3)) / 16.0


def _coarsen_mask(xp, solvable, ncx, ncy, ncz):
    """A coarse cell is active if any of its eight fine children is solvable."""
    blocks = solvable.reshape(ncx, 2, ncy, 2, ncz, 2)
    return blocks.any(axis=(1, 3, 5))


def _restrict(xp, r, ncx, ncy, ncz):
    """Full-weighting restriction: mean of the eight fine children per cell."""
    blocks = r.reshape(ncx, 2, ncy, 2, ncz, 2)
    return blocks.sum(axis=(1, 3, 5)) / 8.0


def _prolong(xp, e, nfx, nfy, nfz):
    """Piecewise-constant prolongation: copy each coarse value to 8 children."""
    out = xp.empty((nfx, nfy, nfz), dtype=e.dtype)
    out[0::2, 0::2, 0::2] = e
    out[1::2, 0::2, 0::2] = e
    out[0::2, 1::2, 0::2] = e
    out[1::2, 1::2, 0::2] = e
    out[0::2, 0::2, 1::2] = e
    out[1::2, 0::2, 1::2] = e
    out[0::2, 1::2, 1::2] = e
    out[1::2, 1::2, 1::2] = e
    return out


class _Level:
    __slots__ = ("kx", "ky", "kz", "solvable", "inv_diag", "shape")

    def __init__(self, xp, kx, ky, kz, solvable):
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.solvable = solvable
        diag = pressure.diagonal(xp, kx, ky, kz, solvable)
        self.solvable = solvable & (diag > 0.0)
        self.inv_diag = xp.where(
            self.solvable, 1.0 / xp.maximum(diag, 1e-30), 0.0)
        self.shape = solvable.shape


def build_hierarchy(xp, kx, ky, kz, solvable, *, min_size=4, max_levels=8):
    """Build coarse levels by factor-two geometric coarsening.

    Returns a list ``[fine, ..., coarsest]``.  Coarsening stops when a further
    level would be smaller than ``min_size`` on any axis, when a dimension is
    odd (only full coarsening is supported), or at ``max_levels``.
    """
    levels = [_Level(xp, kx, ky, kz, solvable)]
    while len(levels) < max_levels:
        nx, ny, nz = levels[-1].shape
        if (nx % 2 or ny % 2 or nz % 2
                or nx // 2 < min_size or ny // 2 < min_size
                or nz // 2 < min_size):
            break
        ncx, ncy, ncz = nx // 2, ny // 2, nz // 2
        top = levels[-1]
        ckx = _coarsen_faces(xp, top.kx, 0, ncx, ncy, ncz)
        cky = _coarsen_faces(xp, top.ky, 1, ncx, ncy, ncz)
        ckz = _coarsen_faces(xp, top.kz, 2, ncx, ncy, ncz)
        cmask = _coarsen_mask(xp, top.solvable, ncx, ncy, ncz)
        # Faces that touch an inactive coarse cell carry no conductance, which
        # keeps the coarse operator consistent with the coarse mask.
        ckx = ckx * _face_gate(xp, cmask, 0)
        cky = cky * _face_gate(xp, cmask, 1)
        ckz = ckz * _face_gate(xp, cmask, 2)
        levels.append(_Level(xp, ckx, cky, ckz, cmask))
    return levels


def _face_gate(xp, mask, axis):
    """1.0 on interior faces between two active cells, else 0.0 (exterior faces
    keep the single-cell activity so Dirichlet p=0 contacts survive coarsening).
    """
    m = mask.astype(xp.float32)
    if axis == 0:
        nx, ny, nz = mask.shape
        gate = xp.zeros((nx + 1, ny, nz), dtype=xp.float32)
        gate[1:-1, :, :] = m[1:, :, :] * m[:-1, :, :]
        gate[0, :, :] = m[0, :, :]
        gate[-1, :, :] = m[-1, :, :]
        return gate
    if axis == 1:
        nx, ny, nz = mask.shape
        gate = xp.zeros((nx, ny + 1, nz), dtype=xp.float32)
        gate[:, 1:-1, :] = m[:, 1:, :] * m[:, :-1, :]
        gate[:, 0, :] = m[:, 0, :]
        gate[:, -1, :] = m[:, -1, :]
        return gate
    nx, ny, nz = mask.shape
    gate = xp.zeros((nx, ny, nz + 1), dtype=xp.float32)
    gate[:, :, 1:-1] = m[:, :, 1:] * m[:, :, :-1]
    gate[:, :, 0] = m[:, :, 0]
    gate[:, :, -1] = m[:, :, -1]
    return gate


def _smooth(xp, level, p, rhs, sweeps, omega):
    """Damped-Jacobi smoothing: p <- p + omega D^-1 (rhs - A p) on active cells."""
    for _ in range(sweeps):
        ap = pressure.apply_laplacian(xp, p, level.kx, level.ky, level.kz,
                                      level.solvable)
        p = p + omega * level.inv_diag * (rhs - ap)
    return p


def vcycle(xp, levels, rhs, *, index=0, nu_pre=2, nu_post=2, nu_coarse=20,
           omega=0.8):
    """One symmetric V-cycle; returns the correction z approximating A^-1 rhs."""
    level = levels[index]
    if index == len(levels) - 1:
        # Coarsest grid: a handful of extra smoothing sweeps stands in for an
        # exact solve.  Approximate is fine for a preconditioner.
        return _smooth(xp, level, xp.zeros_like(rhs), rhs, nu_coarse, omega)

    p = _smooth(xp, level, xp.zeros_like(rhs), rhs, nu_pre, omega)
    residual = rhs - pressure.apply_laplacian(
        xp, p, level.kx, level.ky, level.kz, level.solvable)
    residual = residual * level.solvable

    ncx, ncy, ncz = levels[index + 1].shape
    coarse_rhs = _restrict(xp, residual, ncx, ncy, ncz) * levels[index + 1].solvable
    coarse_e = vcycle(xp, levels, coarse_rhs, index=index + 1,
                      nu_pre=nu_pre, nu_post=nu_post, nu_coarse=nu_coarse,
                      omega=omega)

    nx, ny, nz = level.shape
    p = p + _prolong(xp, coarse_e, nx, ny, nz) * level.solvable
    p = _smooth(xp, level, p, rhs, nu_post, omega)
    return p * level.solvable


def solve(xp, rhs, kx, ky, kz, liquid, tol=1e-4, max_iter=400, check_every=8,
          *, min_size=4, max_levels=8, nu_pre=2, nu_post=2, nu_coarse=20,
          omega=0.8):
    """Multigrid-preconditioned CG.  Returns (p, iterations, rel_residual).

    Drop-in replacement for :func:`stflip.pressure.solve`: same operator, same
    convergence contract (the real residual is checked every ``check_every``
    iterations), but the preconditioner is a geometric V-cycle instead of the
    diagonal.  Falls back to the diagonal preconditioner when the grid is too
    small to coarsen at all, so tiny domains behave exactly like Jacobi-PCG.
    """
    diag = pressure.diagonal(xp, kx, ky, kz, liquid)
    solvable = liquid & (diag > 0.0)
    rhs = rhs * solvable

    p = xp.zeros_like(rhs)
    r = rhs.copy()
    b_norm = math.sqrt(float((r * r).sum()))
    if b_norm < 1e-30:
        return p, 0, 0.0

    levels = build_hierarchy(
        xp, kx, ky, kz, solvable, min_size=min_size, max_levels=max_levels)
    if len(levels) == 1:
        # Nothing to coarsen: defer to the diagonal-preconditioned CG so small
        # grids keep their existing (already fast) behaviour.
        return pressure.solve(
            xp, rhs, kx, ky, kz, liquid, tol=tol, max_iter=max_iter,
            check_every=check_every)

    def precondition(vec):
        return vcycle(xp, levels, vec * solvable, nu_pre=nu_pre,
                      nu_post=nu_post, nu_coarse=nu_coarse, omega=omega)

    z = precondition(r)
    s = z.copy()
    sigma = (z * r).sum()

    rel = 1.0
    it = 0
    for it in range(1, max_iter + 1):
        As = pressure.apply_laplacian(xp, s, kx, ky, kz, solvable)
        sAs = (s * As).sum()
        ok = xp.abs(sAs) > 1e-30
        alpha = xp.where(ok, sigma / xp.where(ok, sAs, 1.0), 0.0)
        p = p + alpha * s
        r = r - alpha * As
        z = precondition(r)
        sigma_new = (z * r).sum()
        ok = xp.abs(sigma) > 1e-30
        beta = xp.where(ok, sigma_new / xp.where(ok, sigma, 1.0), 0.0)
        sigma = sigma_new
        s = z + beta * s
        if it % check_every == 0 or it == max_iter:
            rel = math.sqrt(float((r * r).sum())) / b_norm
            if rel <= tol or not math.isfinite(rel):
                break

    return p * solvable, it, rel
