"""Semi-implicit capillary stabilizer (roadmap CAP-M1).

An interface-concentrated Helmholtz solve per MAC velocity component that
damps the stiff capillary-wave feedback limiting explicit CSF surface
tension to dt < O(dx^(3/2)) (Brackbill et al. 1992).  The explicit CSF kick
stays as the predictor; this operator is the corrector that lets the
capillary clamp be RELAXED by a bounded user factor (activation and clamp
scaling are CAP-M2; recommended per-scene scales are CAP-M3).  With a
lumped operator on a noisy P2G phase field there is NO unconditional
stability proof, so the clamp is only ever scaled, never removed.

System per component d in (u, v, w):

    (R_d + A_d) u_d = R_d * u_hat_d

``R_d`` is the diagonal of eps-floored face densities (SPD).  ``A_d`` is a
symmetric graph Laplacian over that component's face lattice with edge
coefficients ``a_e = dt**2 * sigma * delta_e * gate_e / dx**2`` where
``delta_e`` is the interface delta function |grad phi_s| averaged to the
edge and ``gate_e`` is nonzero only when BOTH coupled faces are usable
fluid DOFs.  The validity gate is the load-bearing physics choice: at the
insertion point invalid faces hold pre-extrapolation garbage, and without
the gate a shell of invalid air-side faces drags valid free-surface
velocities toward stale values -- spurious drag masquerading as damping.

``apply_laplacian``/``diagonal`` from the pressure module are shape-generic
over any 3D lattice with one-larger-per-axis coefficient arrays, so the
face lattice reuses them directly; zeroing the outermost coefficient layers
turns their exterior half-cell Dirichlet terms into the natural (Neumann)
boundary a velocity solve needs.  Away from the interface every edge
coefficient is zero, so with the initial guess x0 = u_hat the residual is
supported only in the band: the solve is the exact identity elsewhere and
CG converges in a few dozen iterations regardless of domain size.
"""

from __future__ import annotations

import math

from .pressure import apply_laplacian, diagonal


def face_delta(xp, grad_mag_cells, axis):
    """Average the cell-centred interface delta onto one face lattice.

    Boundary faces copy the adjacent cell so a wall-touching interface
    keeps its stabilizer instead of losing it to zero padding.
    """

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    mag = grad_mag_cells
    shape = list(mag.shape)
    shape[axis] += 1
    out = xp.zeros(tuple(shape), dtype=mag.dtype)
    inner = [slice(None)] * 3
    lo = [slice(None)] * 3
    hi = [slice(None)] * 3
    inner[axis] = slice(1, -1)
    lo[axis] = slice(None, -1)
    hi[axis] = slice(1, None)
    out[tuple(inner)] = 0.5 * (mag[tuple(lo)] + mag[tuple(hi)])
    first = [slice(None)] * 3
    first[axis] = 0
    last = [slice(None)] * 3
    last[axis] = -1
    cell_first = [slice(None)] * 3
    cell_first[axis] = 0
    cell_last = [slice(None)] * 3
    cell_last[axis] = -1
    out[tuple(first)] = mag[tuple(cell_first)]
    out[tuple(last)] = mag[tuple(cell_last)]
    return out


def edge_coefficients(xp, delta_dof, dof_mask, dt, sigma, dx):
    """Gated edge coefficients for one component's face-lattice Laplacian.

    Returns (kx, ky, kz), each one larger than ``delta_dof`` along its
    axis.  Interior edge e between DOFs a and b carries
    ``dt**2 * sigma / dx**2 * 0.5 * (delta_a + delta_b)`` and is zeroed
    unless BOTH a and b are usable DOFs.  The outermost layers stay zero
    (natural boundary).
    """

    scale = float(dt) * float(dt) * float(sigma) / (float(dx) * float(dx))
    if scale < 0.0:
        raise ValueError("sigma and dt must not be negative")
    gate = dof_mask.astype(delta_dof.dtype)
    coefs = []
    for axis in range(3):
        shape = list(delta_dof.shape)
        shape[axis] += 1
        k = xp.zeros(tuple(shape), dtype=delta_dof.dtype)
        inner = [slice(None)] * 3
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        inner[axis] = slice(1, -1)
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        pair_gate = gate[tuple(lo)] * gate[tuple(hi)]
        k[tuple(inner)] = (
            scale * 0.5 * (delta_dof[tuple(lo)] + delta_dof[tuple(hi)])
            * pair_gate)
        coefs.append(k)
    return tuple(coefs)


def stabilize_component(xp, u_hat, rho_face, kx, ky, kz, dof_mask,
                        tol=1e-4, max_iter=200, check_every=8):
    """Solve ``(R + A) u = R * u_hat`` on the gated DOFs by Jacobi-CG.

    Non-DOF entries pass through as ``u_hat`` unchanged.  Returns
    ``(u, iterations, rel_residual)``; the relative residual is measured
    against the initial residual ``-A u_hat`` (with ``x0 = u_hat``), which
    is supported only on the interface band.
    """

    solvable = dof_mask
    rho = xp.maximum(rho_face, 1e-30)
    diag = rho + diagonal(xp, kx, ky, kz, solvable)
    inv_diag = xp.where(solvable, 1.0 / diag, 0.0)

    x = u_hat.copy()
    # r = R*u_hat - (R + A) x0 = -A u_hat on the solvable rows.
    r = (-apply_laplacian(xp, x, kx, ky, kz, solvable)) * solvable
    b_norm = math.sqrt(float((r * r).sum()))
    if b_norm < 1e-30:
        return xp.where(solvable, x, u_hat), 0, 0.0

    z = inv_diag * r
    s = z.copy()
    sigma_dot = (z * r).sum()  # 0-d device scalar

    rel = 1.0
    it = 0
    for it in range(1, max_iter + 1):
        As = (rho * s * solvable
              + apply_laplacian(xp, s, kx, ky, kz, solvable))
        sAs = (s * As).sum()
        ok = xp.abs(sAs) > 1e-30
        alpha = xp.where(ok, sigma_dot / xp.where(ok, sAs, 1.0), 0.0)
        x = x + alpha * s
        r = r - alpha * As
        z = inv_diag * r
        sigma_new = (z * r).sum()
        ok = xp.abs(sigma_dot) > 1e-30
        beta = xp.where(ok, sigma_new / xp.where(ok, sigma_dot, 1.0), 0.0)
        sigma_dot = sigma_new
        s = z + beta * s
        if it % check_every == 0 or it == max_iter:
            rel = math.sqrt(float((r * r).sum())) / b_norm
            if rel <= tol or not math.isfinite(rel):
                break

    return xp.where(solvable, x, u_hat), it, rel


def apply_operator(xp, x, rho_face, kx, ky, kz, dof_mask):
    """``(R + A) x`` on the gated DOFs -- exposed for the SPD tests."""

    return (rho_face * x * dof_mask
            + apply_laplacian(xp, x, kx, ky, kz, dof_mask))
