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
    _solid_sdfs_from_objects(
        objects,
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
