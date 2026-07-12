"""Rasterise Blender meshes onto the simulation grid.

Inside/outside tests use BVH nearest-point normal sign, which requires
reasonably closed meshes.  Grid samples are only tested near the object's
world bounding box, so sparse objects stay cheap even on large grids.
"""

import math

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


_FAR_SDF = np.float32(1e9)


def domain_grid(domain_obj, resolution: int):
    """Grid dims/dx/origin from the domain object's world bounding box."""
    corners = [domain_obj.matrix_world @ Vector(c) for c in domain_obj.bound_box]
    mins = Vector((min(c[i] for c in corners) for i in range(3)))
    maxs = Vector((max(c[i] for c in corners) for i in range(3)))
    extent = maxs - mins
    longest = max(extent)
    dx = longest / resolution
    dims = tuple(max(4, int(round(extent[i] / dx))) for i in range(3))
    return dims, float(dx), np.array(mins, dtype=np.float64)


def _grid_range(bounds, origin, dx, shape, offset, pad=1):
    """Index range of offset grid samples near evaluated world bounds."""
    lo, hi = bounds
    lo_i = [
        max(0, math.floor((lo[i] - origin[i]) / dx - offset[i]) - pad)
        for i in range(3)
    ]
    hi_i = [
        min(
            shape[i],
            math.floor((hi[i] - origin[i]) / dx - offset[i]) + 1 + pad,
        )
        for i in range(3)
    ]
    return lo_i, hi_i


def _cell_range(bounds, origin, dx, dims, pad=1):
    """Index range of cell-centred samples near the object's world bbox."""
    return _grid_range(bounds, origin, dx, dims, (0.5, 0.5, 0.5), pad)


def _world_bvh(obj, depsgraph):
    """Build an evaluated BVH whose coordinates and distances are world-space.

    A local-space nearest point cannot be converted to an exact world-space
    distance with one scale factor when an object is scaled non-uniformly.  By
    transforming the evaluated vertices first, BVH nearest-point queries use
    the correct anisotropic metric and naturally retain rotation support.

    ``orientation`` corrects polygon normals for reflection transforms (an odd
    number of negative scale axes), whose transformed winding is reversed.
    """
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        matrix = evaluated.matrix_world.copy()
        vertices = [matrix @ vertex.co for vertex in mesh.vertices]
        polygons = [tuple(poly.vertices) for poly in mesh.polygons]
        if not vertices or not polygons:
            return None, 1.0, None
        bvh = BVHTree.FromPolygons(vertices, polygons, all_triangles=False)
        determinant = matrix.to_3x3().determinant()
        orientation = -1.0 if determinant < 0.0 else 1.0
        bounds = (
            tuple(min(vertex[axis] for vertex in vertices) for axis in range(3)),
            tuple(max(vertex[axis] for vertex in vertices) for axis in range(3)),
        )
        return bvh, orientation, bounds
    finally:
        evaluated.to_mesh_clear()


def _signed_distance(point, location, normal, distance, orientation):
    if (point - location).dot(normal) * orientation < 0.0:
        return -distance
    return distance


def _sample_sdf(sdf, bounds, bvh, orientation, origin, dx, offset, pad=4):
    """Accumulate one object's signed distance into an offset grid."""
    lo, hi = _grid_range(bounds, origin, dx, sdf.shape, offset, pad)
    for i in range(lo[0], hi[0]):
        for j in range(lo[1], hi[1]):
            for k in range(lo[2], hi[2]):
                wp = Vector((origin[0] + (i + offset[0]) * dx,
                             origin[1] + (j + offset[1]) * dx,
                             origin[2] + (k + offset[2]) * dx))
                loc, normal, _idx, distance = bvh.find_nearest(wp)
                if loc is None:
                    continue
                value = _signed_distance(
                    wp, loc, normal, distance, orientation)
                if value < sdf[i, j, k]:
                    sdf[i, j, k] = value


def _solid_sdfs_from_objects(objects, depsgraph, origin, dx, grids):
    """Populate ``(array, sample_offset)`` grids from shared world BVHs."""
    for obj in objects:
        bvh, orientation, bounds = _world_bvh(obj, depsgraph)
        if bvh is None:
            continue
        for sdf, offset in grids:
            _sample_sdf(
                sdf, bounds, bvh, orientation, origin, dx, offset)


def mask_from_object(obj, depsgraph, origin, dx, dims) -> np.ndarray:
    """Boolean cell mask: cell centres inside the (closed) mesh."""
    mask = np.zeros(dims, dtype=bool)
    # Vectorized parity fast path; fall back to the BVH loop on any failure
    # (e.g. non-mesh evaluation or the mocked objects in the test suite).
    try:
        tris = _extract_triangles(obj, depsgraph)
    except Exception:
        tris = None
    if tris is not None:
        rng = _fast_subrange(tris, origin, dx, dims, pad=1)
        if rng is None:
            return mask
        lo, hi = rng
        counts = tuple(hi[i] - lo[i] for i in range(3))
        sub_origin = tuple(origin[i] + lo[i] * dx for i in range(3))
        mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = _parity_inside(
            tris, sub_origin, dx, counts, (0.5, 0.5, 0.5))
        return mask
    bvh, orientation, bounds = _world_bvh(obj, depsgraph)
    if bvh is None:
        return mask
    lo, hi = _cell_range(bounds, origin, dx, dims)
    for i in range(lo[0], hi[0]):
        for j in range(lo[1], hi[1]):
            for k in range(lo[2], hi[2]):
                wp = Vector((origin[0] + (i + 0.5) * dx,
                             origin[1] + (j + 0.5) * dx,
                             origin[2] + (k + 0.5) * dx))
                loc, normal, _idx, distance = bvh.find_nearest(wp)
                if loc is None:
                    continue
                if _signed_distance(
                        wp, loc, normal, distance, orientation) < 0.0:
                    mask[i, j, k] = True
    return mask


def sdf_from_objects(objects, depsgraph, origin, dx, dims) -> np.ndarray:
    """Approximate cell-centred signed distance to solid obstacles
    (positive outside).  Cells far from every obstacle stay at +big."""
    cell_sdf = np.full(dims, _FAR_SDF, dtype=np.float32)
    _solid_sdfs_from_objects(
        objects,
        depsgraph,
        origin,
        dx,
        ((cell_sdf, (0.5, 0.5, 0.5)),),
    )
    return cell_sdf


def solid_sdfs_from_objects(
    objects, depsgraph, origin, dx, dims,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cell- and node-centred obstacle signed-distance grids.

    Both grids are sampled from the same evaluated world-space BVH for each
    obstacle.  Cell samples lie at ``origin + (index + 0.5) * dx`` and have
    shape ``dims``.  Node samples lie at ``origin + index * dx`` and have one
    more sample per axis.  Positive values are outside solids.
    """
    cell_sdf = np.full(dims, _FAR_SDF, dtype=np.float32)
    node_shape = tuple(int(axis) + 1 for axis in dims)
    node_sdf = np.full(node_shape, _FAR_SDF, dtype=np.float32)
    slow_objects = []
    for obj in objects:
        try:
            tris = _extract_triangles(obj, depsgraph)
        except Exception:
            tris = None
        if tris is None:
            slow_objects.append(obj)
            continue
        # Band-exact signed distance on the padded sub-lattice, min-combined
        # into the global grids (union of solids).  pad covers the exact band
        # plus one saturated ring, matching the BVH path's pad=4 semantics.
        band = 3
        for grid, offset, extra in (
            (cell_sdf, (0.5, 0.5, 0.5), 0),
            (node_sdf, (0.0, 0.0, 0.0), 1),
        ):
            rng = _fast_subrange(tris, origin, dx, dims, pad=band + 1)
            if rng is None:
                continue
            lo, hi = rng
            hi = [min(hi[i] + extra, dims[i] + extra) for i in range(3)]
            counts = tuple(hi[i] - lo[i] for i in range(3))
            sub_origin = tuple(origin[i] + lo[i] * dx for i in range(3))
            sub = _signed_band_sdf(tris, sub_origin, dx, counts, offset,
                                   band=band)
            view = grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            np.minimum(view, sub, out=view)
    if slow_objects:
        _solid_sdfs_from_objects(
            slow_objects,
            depsgraph,
            origin,
            dx,
            (
                (cell_sdf, (0.5, 0.5, 0.5)),
                (node_sdf, (0.0, 0.0, 0.0)),
            ),
        )
    return cell_sdf, node_sdf


def _dilate(mask: np.ndarray, cells: int) -> np.ndarray:
    """Binary-dilate ``mask`` by ``cells`` in the 6-neighbourhood."""
    out = mask.copy()
    for _ in range(max(int(cells), 0)):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        grown[:, :, 1:] |= out[:, :, :-1]
        grown[:, :, :-1] |= out[:, :, 1:]
        out = grown
    return out


def solid_velocity_from_objects(
    objects, prev_matrices, depsgraph, origin, dx, dims, frame_dt,
    band_cells: int = 2,
):
    """Cell-centred rigid solid velocity for animated moving-wall obstacles.

    ``prev_matrices`` maps object name to the 4x4 world matrix (as a NumPy
    array) captured at the previous output frame.  For each animated obstacle
    the rigid velocity v(x) = (x - M_prev M_cur^-1 x) / frame_dt is stamped
    into the cells inside the obstacle plus a ``band_cells`` halo, so boundary
    faces and the near-solid particle band see the full wall speed.  Returns
    an ``dims + (3,)`` float32 field, or None when nothing moved.
    """
    if not objects or frame_dt <= 0.0:
        return None
    vel = None
    for obj in objects:
        m_prev = prev_matrices.get(obj.name)
        if m_prev is None:
            continue
        m_cur = np.array(obj.matrix_world, dtype=np.float64).reshape(4, 4)
        if np.allclose(m_cur, m_prev, atol=1e-12):
            continue
        mask = mask_from_object(obj, depsgraph, origin, dx, dims)
        if not mask.any():
            continue
        mask = _dilate(mask, band_cells)
        idx = np.argwhere(mask)
        world = np.asarray(origin, dtype=np.float64)[None, :] \
            + (idx.astype(np.float64) + 0.5) * dx
        # Same material point at the previous frame: x_prev = M_prev M_cur^-1 x.
        transform = m_prev @ np.linalg.inv(m_cur)
        prev_pts = world @ transform[:3, :3].T + transform[:3, 3][None, :]
        v = ((world - prev_pts) / frame_dt).astype(np.float32)
        if vel is None:
            vel = np.zeros(tuple(dims) + (3,), dtype=np.float32)
        vel[mask] = v
    return vel


# --------------------------------------------------------------------------
# Vectorized fast path (issue #12).
#
# The BVH loops above cost microseconds of Python per cell, which dominates
# bake setup above resolution ~128 and every frame with animated obstacles.
# The fast path extracts the evaluated triangle soup once and then works in
# NumPy: inside/outside via +x ray-crossing parity (exact for closed meshes),
# and signed distance that is exact in a narrow band around the surface and
# sign-correct-but-saturated elsewhere -- which is all the solver consumes
# (masks, apertures near the surface, push-out and gradients within ~1 cell).
# Any extraction failure falls back to the original BVH loops.

def _extract_triangles(obj, depsgraph):
    """World-space (T, 3, 3) float64 triangle array of the evaluated mesh."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        n_v = len(mesh.vertices)
        n_t = len(mesh.loop_triangles)
        if n_v == 0 or n_t == 0:
            return None
        co = np.empty(n_v * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", co)
        idx = np.empty(n_t * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get("vertices", idx)
        matrix = np.array(evaluated.matrix_world, dtype=np.float64)
        world = co.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
        return world[idx.reshape(-1, 3)]
    finally:
        evaluated.to_mesh_clear()


def _parity_inside(tris, origin, dx, counts, offset,
                   ray_chunk=4096, tri_chunk=16384):
    """Inside mask on the sample lattice origin + (index + offset) * dx.

    Casts one +x ray per (j, k) lattice column, intersects it with every
    triangle via 2D barycentrics in the yz-plane, histograms the crossing
    x-positions into lattice bins, and takes the running-parity cumsum.
    Ray origins carry a fixed sub-cell jitter so edge/vertex grazing hits
    (which would double-count) have measure zero.
    """
    nx, ny, nz = counts
    inside = np.zeros(counts, dtype=bool)
    if len(tris) == 0 or min(counts) <= 0:
        return inside
    jy = 2.718281e-5 * dx
    jz = 3.141592e-5 * dx
    ys = origin[1] + (np.arange(ny) + offset[1]) * dx + jy
    zs = origin[2] + (np.arange(nz) + offset[2]) * dx + jz
    ray_y, ray_z = np.meshgrid(ys, zs, indexing="ij")
    ray_y = ray_y.ravel()
    ray_z = ray_z.ravel()
    n_rays = ray_y.shape[0]
    hist = np.zeros((nx + 1, n_rays), dtype=np.int32)
    for ts in range(0, len(tris), tri_chunk):
        tri = tris[ts:ts + tri_chunk]
        ax, ay, az = tri[:, 0, 0], tri[:, 0, 1], tri[:, 0, 2]
        by_a = tri[:, 1, 1] - ay
        bz_a = tri[:, 1, 2] - az
        cy_a = tri[:, 2, 1] - ay
        cz_a = tri[:, 2, 2] - az
        det = by_a * cz_a - bz_a * cy_a
        ok_t = np.abs(det) > 1e-30
        inv_det = np.where(ok_t, det, 1.0)
        bx_a = tri[:, 1, 0] - ax
        cx_a = tri[:, 2, 0] - ax
        for rs in range(0, n_rays, ray_chunk):
            py = ray_y[rs:rs + ray_chunk, None] - ay[None, :]
            pz = ray_z[rs:rs + ray_chunk, None] - az[None, :]
            u = (py * cz_a[None, :] - pz * cy_a[None, :]) / inv_det[None, :]
            v = (by_a[None, :] * pz - bz_a[None, :] * py) / inv_det[None, :]
            hit = (ok_t[None, :] & (u >= 0.0) & (v >= 0.0)
                   & (u + v <= 1.0))
            if not hit.any():
                continue
            rid, tid = np.nonzero(hit)
            x_hit = (ax[tid] + u[rid, tid] * bx_a[tid]
                     + v[rid, tid] * cx_a[tid])
            # First lattice sample strictly past the crossing.
            i0 = np.floor((x_hit - origin[0]) / dx - offset[0]).astype(
                np.int64) + 1
            i0 = np.clip(i0, 0, nx)
            np.add.at(hist, (i0, rid + rs), 1)
    parity = (np.cumsum(hist[:nx], axis=0) % 2).astype(bool)
    return parity.reshape(nx, ny, nz)


def _point_triangle_distance(points, tris, bound, chunk=512):
    """Min distance from each point to the triangle soup, capped at ``bound``.

    Closest-point-on-triangle (Ericson) evaluated as a vectorized where-
    cascade over (chunk, T) pairs, with a per-chunk triangle bbox prefilter
    so only nearby triangles are tested.
    """
    out = np.full(len(points), bound, dtype=np.float64)
    if len(tris) == 0 or len(points) == 0:
        return out
    tmin = tris.min(axis=1)
    tmax = tris.max(axis=1)
    for s in range(0, len(points), chunk):
        p = points[s:s + chunk]
        sel = (np.all(tmax >= p.min(axis=0) - bound, axis=1)
               & np.all(tmin <= p.max(axis=0) + bound, axis=1))
        if not sel.any():
            continue
        a = tris[sel, 0][None]
        b = tris[sel, 1][None]
        c = tris[sel, 2][None]
        q = p[:, None, :]
        ab = b - a
        ac = c - a
        ap = q - a
        d1 = (ab * ap).sum(-1)
        d2 = (ac * ap).sum(-1)
        bp = q - b
        d3 = (ab * bp).sum(-1)
        d4 = (ac * bp).sum(-1)
        cp = q - c
        d5 = (ab * cp).sum(-1)
        d6 = (ac * cp).sum(-1)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        denom = va + vb + vc
        denom = np.where(np.abs(denom) > 1e-30, denom, 1e-30)
        closest = (a + ab * (vb / denom)[..., None]
                   + ac * (vc / denom)[..., None])

        def _edge(base, edge, t_num, t_den, cond, current):
            t_den = np.where(np.abs(t_den) > 1e-30, t_den, 1e-30)
            t = np.clip(t_num / t_den, 0.0, 1.0)
            return np.where(cond[..., None], base + edge * t[..., None],
                            current)

        closest = _edge(b, c - b, d4 - d3, (d4 - d3) + (d5 - d6),
                        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
                        closest)
        closest = _edge(a, ac, d2, d2 - d6,
                        (vb <= 0) & (d2 >= 0) & (d6 <= 0), closest)
        closest = _edge(a, ab, d1, d1 - d3,
                        (vc <= 0) & (d1 >= 0) & (d3 <= 0), closest)
        closest = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, closest)
        closest = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, closest)
        closest = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, closest)
        dist = np.sqrt(((q - closest) ** 2).sum(-1)).min(axis=1)
        out[s:s + chunk] = np.minimum(out[s:s + chunk], dist)
    return out


def _signed_band_sdf(tris, origin, dx, counts, offset, band=3):
    """Signed distance on a lattice: exact within ``band`` cells of the
    surface, sign-correct but saturated to +/-(band+1)*dx beyond it."""
    inside = _parity_inside(tris, origin, dx, counts, offset)
    surface = np.zeros_like(inside)
    surface[1:] |= inside[1:] != inside[:-1]
    surface[:-1] |= inside[1:] != inside[:-1]
    surface[:, 1:] |= inside[:, 1:] != inside[:, :-1]
    surface[:, :-1] |= inside[:, 1:] != inside[:, :-1]
    surface[:, :, 1:] |= inside[:, :, 1:] != inside[:, :, :-1]
    surface[:, :, :-1] |= inside[:, :, 1:] != inside[:, :, :-1]
    band_mask = _dilate(surface, band)
    sat = (band + 1) * dx
    sdf = np.where(inside, -sat, sat).astype(np.float32)
    idx = np.argwhere(band_mask)
    if len(idx):
        pts = (np.asarray(origin, dtype=np.float64)[None, :]
               + (idx.astype(np.float64) + np.asarray(offset)) * dx)
        dist = _point_triangle_distance(pts, tris, bound=sat)
        signs = np.where(inside[band_mask], -1.0, 1.0)
        sdf[band_mask] = (signs * dist).astype(np.float32)
    return sdf


def _fast_subrange(tris, origin, dx, dims, pad):
    """Padded, clamped lattice index range covering the triangle bounds."""
    lo = [max(0, int((float(tris[..., i].min()) - origin[i]) / dx) - pad)
          for i in range(3)]
    hi = [min(int(dims[i]), int((float(tris[..., i].max()) - origin[i]) / dx)
              + 1 + pad) for i in range(3)]
    if any(hi[i] <= lo[i] for i in range(3)):
        return None
    return lo, hi
