"""Variable-coefficient pressure projection (ST-FLIP Sections 3.6-3.7).

Solves  sum_f k_f (p_c - p_nb) / dx^2 = -(div u*)_c  on liquid cells with a
matrix-free Jacobi-preconditioned conjugate gradient.  Face coefficients
k_f = dt * alpha_f / max(rho(phi_f), eps_rho) come directly from the P2G
weight-accumulator phase field, so no surface reconstruction is needed.

Air cells (phi_c < 0.5) are Dirichlet p = 0 and are simply excluded from the
system; because their pressure value is zero, masking p by the liquid mask
implements the boundary condition while keeping the operator symmetric
positive definite.  Solid faces carry k = 0 and drop out naturally.
"""

from __future__ import annotations


def apply_laplacian(xp, p, kx, ky, kz, liquid):
    """A p = sum_f k_f (p_c - p_nb), restricted to liquid rows.

    p, liquid: (nx, ny, nz); kx: (nx+1, ny, nz); ky: (nx, ny+1, nz);
    kz: (nx, ny, nz+1).  Boundary faces of the domain must have k = 0.
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

    return out * liquid


def diagonal(xp, kx, ky, kz, liquid):
    """Diagonal of the operator above (sum of incident face coefficients)."""
    diag = (
        kx[1:, :, :] + kx[:-1, :, :]
        + ky[:, 1:, :] + ky[:, :-1, :]
        + kz[:, :, 1:] + kz[:, :, :-1]
    )
    return diag * liquid


def _dot(xp, a, b) -> float:
    """Plain reduction dot product (avoids cuBLAS, which CuPy's Windows
    wheels do not bundle)."""
    return float((a * b).sum())


def solve(xp, rhs, kx, ky, kz, liquid, tol=1e-4, max_iter=400):
    """Jacobi-preconditioned CG.  Returns (p, iterations, rel_residual)."""
    import math

    diag = diagonal(xp, kx, ky, kz, liquid)
    # Cells with an empty row (isolated by solids) cannot be solved for.
    solvable = liquid & (diag > 0.0)
    rhs = rhs * solvable
    inv_diag = xp.where(solvable, 1.0 / xp.maximum(diag, 1e-30), 0.0)

    p = xp.zeros_like(rhs)
    r = rhs.copy()
    b_norm = math.sqrt(_dot(xp, r, r))
    if b_norm < 1e-30:
        return p, 0, 0.0

    z = inv_diag * r
    s = z.copy()
    sigma = _dot(xp, z, r)

    rel = 1.0
    it = 0
    for it in range(1, max_iter + 1):
        As = apply_laplacian(xp, s, kx, ky, kz, solvable)
        sAs = _dot(xp, s, As)
        if abs(sAs) < 1e-30:
            break
        alpha = sigma / sAs
        p = p + alpha * s
        r = r - alpha * As
        rel = math.sqrt(_dot(xp, r, r)) / b_norm
        if rel <= tol:
            break
        z = inv_diag * r
        sigma_new = _dot(xp, z, r)
        beta = sigma_new / sigma
        sigma = sigma_new
        s = z + beta * s

    return p * solvable, it, rel
