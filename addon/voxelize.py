"""Rasterise Blender meshes onto the simulation grid.

Inside/outside tests use BVH nearest-point normal sign, which requires
reasonably closed meshes.  Cells are only tested inside the object's world
bounding box, so sparse objects stay cheap even on large grids.
"""

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


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


def _cell_range(obj, origin, dx, dims, pad=1):
    """Index range of grid cells overlapping the object's world bbox."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    lo_i = [max(0, int((lo[i] - origin[i]) / dx) - pad) for i in range(3)]
    hi_i = [min(dims[i], int((hi[i] - origin[i]) / dx) + 1 + pad) for i in range(3)]
    return lo_i, hi_i


def _bvh_and_transform(obj, depsgraph):
    bvh = BVHTree.FromObject(obj, depsgraph)
    inv = obj.matrix_world.inverted()
    # Average scale factor to convert local distances back to world units.
    scale = obj.matrix_world.to_scale()
    s = (abs(scale[0]) + abs(scale[1]) + abs(scale[2])) / 3.0
    return bvh, inv, max(s, 1e-9)


def mask_from_object(obj, depsgraph, origin, dx, dims) -> np.ndarray:
    """Boolean cell mask: cell centres inside the (closed) mesh."""
    mask = np.zeros(dims, dtype=bool)
    bvh, inv, _s = _bvh_and_transform(obj, depsgraph)
    lo, hi = _cell_range(obj, origin, dx, dims)
    for i in range(lo[0], hi[0]):
        for j in range(lo[1], hi[1]):
            for k in range(lo[2], hi[2]):
                wp = Vector((origin[0] + (i + 0.5) * dx,
                             origin[1] + (j + 0.5) * dx,
                             origin[2] + (k + 0.5) * dx))
                lp = inv @ wp
                loc, normal, _idx, _d = bvh.find_nearest(lp)
                if loc is None:
                    continue
                if (lp - loc).dot(normal) < 0.0:
                    mask[i, j, k] = True
    return mask


def sdf_from_objects(objects, depsgraph, origin, dx, dims) -> np.ndarray:
    """Approximate cell-centred signed distance to solid obstacles
    (positive outside).  Cells far from every obstacle stay at +big."""
    sdf = np.full(dims, 1e9, dtype=np.float32)
    for obj in objects:
        bvh, inv, s = _bvh_and_transform(obj, depsgraph)
        # Distance field is only accurate near the obstacle; pad generously.
        lo, hi = _cell_range(obj, origin, dx, dims, pad=4)
        for i in range(lo[0], hi[0]):
            for j in range(lo[1], hi[1]):
                for k in range(lo[2], hi[2]):
                    wp = Vector((origin[0] + (i + 0.5) * dx,
                                 origin[1] + (j + 0.5) * dx,
                                 origin[2] + (k + 0.5) * dx))
                    lp = inv @ wp
                    loc, normal, _idx, dist = bvh.find_nearest(lp)
                    if loc is None:
                        continue
                    d = dist * s
                    if (lp - loc).dot(normal) < 0.0:
                        d = -d
                    if d < sdf[i, j, k]:
                        sdf[i, j, k] = d
    return sdf
