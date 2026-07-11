"""Variable-coefficient pressure projection (ST-FLIP Sections 3.6-3.7).

Solves  sum_f k_f (p_c - p_nb) / dx^2 = -(div u*)_c  on liquid cells with a
matrix-free Jacobi-preconditioned conjugate gradient.  Face coefficients
k_f = dt * alpha_f / max(rho(phi_f), eps_rho) combine solid-geometry face
apertures with density values from the P2G weight-accumulator phase field, so
no liquid-surface reconstruction is needed.

Air cells (phi_c < 0.5) are Dirichlet p = 0 and are simply excluded from the
system; because their pressure value is zero, masking p by the liquid mask
implements the boundary condition while keeping the operator symmetric.
Dirichlet air/outlet contact anchors a component and makes it positive
definite; a sealed all-liquid component retains the usual constant-pressure
nullspace and is positive semidefinite. Solid faces carry k = 0 and drop out
naturally. Nonzero exterior-face coefficients represent open p=0 boundaries
at half-cell distance, and therefore contribute twice their face coefficient.
"""

from __future__ import annotations


def apply_laplacian(xp, p, kx, ky, kz, liquid):
    """A p = sum_f k_f (p_c - p_nb), restricted to liquid rows.

    p, liquid: (nx, ny, nz); kx: (nx+1, ny, nz); ky: (nx, ny+1, nz);
    kz: (nx, ny, nz+1). Nonzero exterior coefficients impose p=0 at the
    boundary face, half a cell from the adjacent pressure sample.
    """
    pm = p * liquid
    out = xp.zeros_like(p)

    # x-axis internal faces: between cells i-1 and i  -> kx[1:-1]
    fx = kx[1:-1, :, :]
    d = pm[1:, :, :] - pm[:-1, :, :]
    out[1:, :, :] += fx * d
    out[:-1, :, :] -= fx * d

    fy = ky[:, 1:-1, :]
    d = pm[:, 1:, :] - pm[:, :-1, :]
    out[:, 1:, :] += fy * d
    out[:, :-1, :] -= fy * d

    fz = kz[:, :, 1:-1]
    d = pm[:, :, 1:] - pm[:, :, :-1]
    out[:, :, 1:] += fz * d
    out[:, :, :-1] -= fz * d

    # Exterior Dirichlet p=0 is half a cell from the boundary cell centre,
    # hence 2*k rather than the full-cell internal-face coefficient k.
    out[0, :, :] += 2.0 * kx[0, :, :] * pm[0, :, :]
    out[-1, :, :] += 2.0 * kx[-1, :, :] * pm[-1, :, :]
    out[:, 0, :] += 2.0 * ky[:, 0, :] * pm[:, 0, :]
    out[:, -1, :] += 2.0 * ky[:, -1, :] * pm[:, -1, :]
    out[:, :, 0] += 2.0 * kz[:, :, 0] * pm[:, :, 0]
    out[:, :, -1] += 2.0 * kz[:, :, -1] * pm[:, :, -1]

    return out * liquid


def diagonal(xp, kx, ky, kz, liquid):
    """Diagonal of the operator above (sum of incident face coefficients)."""
    diag = (
        kx[1:, :, :] + kx[:-1, :, :]
        + ky[:, 1:, :] + ky[:, :-1, :]
        + kz[:, :, 1:] + kz[:, :, :-1]
    )
    # The sum above includes each exterior coefficient once. Add it once more
    # to match the half-cell (2*k) terms in apply_laplacian.
    diag[0, :, :] += kx[0, :, :]
    diag[-1, :, :] += kx[-1, :, :]
    diag[:, 0, :] += ky[:, 0, :]
    diag[:, -1, :] += ky[:, -1, :]
    diag[:, :, 0] += kz[:, :, 0]
    diag[:, :, -1] += kz[:, :, -1]
    return diag * liquid


def solve(xp, rhs, kx, ky, kz, liquid, tol=1e-4, max_iter=400,
          check_every=8):
    """Jacobi-preconditioned CG.  Returns (p, iterations, rel_residual).

    All scalars (sigma, alpha, beta) stay as 0-d device arrays: converting
    them to Python floats every iteration would force a blocking GPU sync
    three times per iteration and make the solve latency-bound.  Only the
    convergence check transfers to host, every `check_every` iterations.
    Plain reductions are used throughout (no cupy.linalg/cuBLAS, which the
    CuPy Windows wheels do not bundle).
    """
    import math

    diag = diagonal(xp, kx, ky, kz, liquid)
    # Cells with an empty row (isolated by solids) cannot be solved for.
    solvable = liquid & (diag > 0.0)
    rhs = rhs * solvable
    inv_diag = xp.where(solvable, 1.0 / xp.maximum(diag, 1e-30), 0.0)

    p = xp.zeros_like(rhs)
    r = rhs.copy()
    b_norm = math.sqrt(float((r * r).sum()))
    if b_norm < 1e-30:
        return p, 0, 0.0

    z = inv_diag * r
    s = z.copy()
    sigma = (z * r).sum()  # 0-d device scalar

    rel = 1.0
    it = 0
    for it in range(1, max_iter + 1):
        As = apply_laplacian(xp, s, kx, ky, kz, solvable)
        sAs = (s * As).sum()
        # Guard breakdown (sAs ~ 0 at exact convergence) without a sync.
        ok = xp.abs(sAs) > 1e-30
        alpha = xp.where(ok, sigma / xp.where(ok, sAs, 1.0), 0.0)
        p = p + alpha * s
        r = r - alpha * As
        z = inv_diag * r
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
