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

import numpy as np


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


def _axis_occupancy(xp, liquid, box, axis):
    """Host bool projection of ``liquid[box]`` onto ``axis`` (any over others)."""
    x0, x1, y0, y1, z0, z1 = box
    sub = liquid[x0:x1, y0:y1, z0:z1]
    other = tuple(a for a in (0, 1, 2) if a != axis)
    present = sub.any(axis=other)
    return np.asarray(present.get() if hasattr(present, "get") else present,
                      dtype=bool)


def _tighten_box(xp, liquid, box):
    """Shrink ``box`` to the bounding box of the liquid inside it, or None."""
    lo = [box[0], box[2], box[4]]
    hi = [box[1], box[3], box[5]]
    for axis in range(3):
        occ = _axis_occupancy(xp, liquid, (lo[0], hi[0], lo[1], hi[1],
                                           lo[2], hi[2]), axis)
        idx = np.nonzero(occ)[0]
        if idx.size == 0:
            return None
        base = lo[axis]
        lo[axis] = base + int(idx[0])
        hi[axis] = base + int(idx[-1]) + 1
    return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])


def _component_boxes(xp, liquid):
    """Bounding boxes of the axis-separable active components.

    Recursively splits the active bounding box wherever a *complete empty plane*
    (a lattice plane with no liquid) lies between two occupied planes on some
    axis.  A connected region can have no such plane — any path across the plane
    would need a liquid cell on it — so a split never separates coupled cells;
    the pressure systems on the two sides are genuinely independent.  Regions
    that only separate along a non-axis direction stay in one box (conservative).
    Returns a list of tight boxes, or None if nothing is active.
    """
    nx, ny, nz = liquid.shape
    root = _tighten_box(xp, liquid, (0, nx, 0, ny, 0, nz))
    if root is None:
        return None
    leaves = []
    stack = [root]
    while stack:
        box = stack.pop()
        split = None
        for axis in range(3):
            occ = _axis_occupancy(xp, liquid, box, axis)
            idx = np.nonzero(occ)[0]
            gaps = np.nonzero(np.diff(idx) > 1)[0]
            if gaps.size:
                cut = box[2 * axis] + int(idx[gaps[0]]) + 1   # first empty plane
                left = list(box)
                left[2 * axis + 1] = cut
                right = list(box)
                right[2 * axis] = cut
                split = (tuple(left), tuple(right))
                break
        if split is None:
            leaves.append(box)
            continue
        for half in split:
            tightened = _tighten_box(xp, liquid, half)
            if tightened is not None:
                stack.append(tightened)
    return leaves


def _box_volume(box):
    return (box[1] - box[0]) * (box[3] - box[2]) * (box[5] - box[4])


def _box_union(boxes):
    x0 = min(b[0] for b in boxes)
    x1 = max(b[1] for b in boxes)
    y0 = min(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    z0 = min(b[4] for b in boxes)
    z1 = max(b[5] for b in boxes)
    return (x0, x1, y0, y1, z0, z1)


def _crop_component(xp, rhs, kx, ky, kz, liquid, tight):
    """Slice the system to one component's box plus a forced-inactive halo.

    The one-cell halo gives boundary cells their true inactive neighbours, and
    forcing every cell outside the tight box inactive keeps a nearby component
    (which cannot be face-adjacent to this one) from being solved twice.  The
    returned scatter writes only the tight region, so component boxes never
    overlap on write.
    """
    x0, x1, y0, y1, z0, z1 = tight
    nx, ny, nz = liquid.shape
    hx0, hx1 = max(0, x0 - 1), min(nx, x1 + 1)
    hy0, hy1 = max(0, y0 - 1), min(ny, y1 + 1)
    hz0, hz1 = max(0, z0 - 1), min(nz, z1 + 1)
    tx0, tx1 = x0 - hx0, x1 - hx0
    ty0, ty1 = y0 - hy0, y1 - hy0
    tz0, tz1 = z0 - hz0, z1 - hz0
    sub_liquid = liquid[hx0:hx1, hy0:hy1, hz0:hz1].copy()
    keep = xp.zeros_like(sub_liquid)
    keep[tx0:tx1, ty0:ty1, tz0:tz1] = True
    sub_liquid = sub_liquid & keep
    parts = (
        rhs[hx0:hx1, hy0:hy1, hz0:hz1],
        kx[hx0:hx1 + 1, hy0:hy1, hz0:hz1],
        ky[hx0:hx1, hy0:hy1 + 1, hz0:hz1],
        kz[hx0:hx1, hy0:hy1, hz0:hz1 + 1],
        sub_liquid,
    )

    def scatter(sub_p):
        full = xp.zeros((nx, ny, nz), dtype=sub_p.dtype)
        full[x0:x1, y0:y1, z0:z1] = sub_p[tx0:tx1, ty0:ty1, tz0:tz1]
        return full

    return parts, scatter


def crop_boxes(xp, rhs, kx, ky, kz, liquid, *, min_gain=0.7):
    """Sub-problems to solve, one per axis-separable component, or None.

    A single connected region yields one bounding-box crop (Phase 0).  Multiple
    disconnected regions are returned as separate boxes so each is solved on its
    own tight support instead of one box spanning the empty gaps between them —
    exact, because disconnected regions are independent pressure systems.  Falls
    back to None (solve the full grid) when neither splitting nor a single crop
    saves enough work.
    """
    boxes = _component_boxes(xp, liquid)
    if boxes is None:
        return None
    grid = liquid.shape[0] * liquid.shape[1] * liquid.shape[2]
    union = _box_union(boxes)
    union_vol = _box_volume(union)
    total = sum(_box_volume(b) for b in boxes)
    if len(boxes) > 1 and total <= min_gain * union_vol:
        return [_crop_component(xp, rhs, kx, ky, kz, liquid, b) for b in boxes]
    if union_vol <= min_gain * grid:
        return [_crop_component(xp, rhs, kx, ky, kz, liquid, union)]
    return None


def crop_to_active(xp, rhs, kx, ky, kz, liquid, *, min_gain=0.7):
    """Restrict the linear system to the tight box around the active cells.

    Returns ``((rhs, kx, ky, kz, liquid), scatter)`` cropped to the active
    bounding box padded by a one-cell inactive halo, or ``None`` when nothing is
    active or the box is not enough smaller than the full grid to be worth it.

    The cropped system is *the same discretization* as the full grid: the halo
    cells are inactive (``liquid`` False), so ``apply_laplacian``'s exterior
    Dirichlet terms multiply by a zero pressure there and every active cell sees
    its true neighbours through the sliced (unchanged) face coefficients.  Where
    the box meets a real domain boundary the halo is clamped away, preserving the
    genuine ``p = 0`` open-boundary faces.  The set of active cells and their
    residual are therefore unchanged, so the outer CG's ``rel <= tol`` contract
    still holds for the scattered pressure; the only difference from the
    full-grid solve is the floating-point summation order of the reductions,
    which perturbs the result at the float32 rounding level (far below ``tol``).

    This is the safe first step toward a fully tiled sparse grid (see
    docs/design/tiled-sparse-grid.md): it skips work on empty regions without
    changing the discretization or the accuracy contract.
    """
    nx, ny, nz = liquid.shape

    def span(axis):
        other = tuple(a for a in (0, 1, 2) if a != axis)
        present = liquid.any(axis=other)
        if not bool(present.any()):
            return None
        lo = int(xp.argmax(present))
        hi = present.shape[0] - int(xp.argmax(present[::-1]))   # exclusive
        n = liquid.shape[axis]
        return max(0, lo - 1), min(n, hi + 1)

    sx, sy, sz = span(0), span(1), span(2)
    if sx is None or sy is None or sz is None:
        return None
    (ax0, ax1), (ay0, ay1), (az0, az1) = sx, sy, sz
    cropped_cells = (ax1 - ax0) * (ay1 - ay0) * (az1 - az0)
    if cropped_cells > min_gain * (nx * ny * nz):
        return None

    parts = (
        rhs[ax0:ax1, ay0:ay1, az0:az1],
        kx[ax0:ax1 + 1, ay0:ay1, az0:az1],
        ky[ax0:ax1, ay0:ay1 + 1, az0:az1],
        kz[ax0:ax1, ay0:ay1, az0:az1 + 1],
        liquid[ax0:ax1, ay0:ay1, az0:az1],
    )

    def scatter(sub_p):
        full = xp.zeros((nx, ny, nz), dtype=sub_p.dtype)
        full[ax0:ax1, ay0:ay1, az0:az1] = sub_p
        return full

    return parts, scatter


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
    boxes = crop_boxes(xp, rhs, kx, ky, kz, liquid)
    if boxes is not None:
        # Each axis-separable component is an independent system, solved on its
        # own tight support; the scatters write disjoint tight regions.
        p = xp.zeros_like(rhs)
        it_max = 0
        rel_max = 0.0
        for parts, scatter in boxes:
            sub_p, iters, rel = _solve_core(
                xp, *parts, tol=tol, max_iter=max_iter, check_every=check_every)
            p = p + scatter(sub_p)
            it_max = max(it_max, iters)
            rel_max = max(rel_max, rel)
        return p, it_max, rel_max
    return _solve_core(xp, rhs, kx, ky, kz, liquid, tol=tol, max_iter=max_iter,
                       check_every=check_every)


def _solve_core(xp, rhs, kx, ky, kz, liquid, tol=1e-4, max_iter=400,
                check_every=8):
    """Diagonal-preconditioned CG on the full given arrays (no cropping)."""
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
