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
            return None, 1.0
        bvh = BVHTree.FromPolygons(vertices, polygons, all_triangles=False)
        determinant = matrix.to_3x3().determinant()
        orientation = -1.0 if determinant < 0.0 else 1.0
        return bvh, orientation
    finally:
        evaluated.to_mesh_clear()


def _signed_distance(point, location, normal, distance, orientation):
    if (point - location).dot(normal) * orientation < 0.0:
        return -distance
    return distance


def mask_from_object(obj, depsgraph, origin, dx, dims) -> np.ndarray:
    """Boolean cell mask: cell centres inside the (closed) mesh."""
    mask = np.zeros(dims, dtype=bool)
    bvh, orientation = _world_bvh(obj, depsgraph)
    if bvh is None:
        return mask
    lo, hi = _cell_range(obj, origin, dx, dims)
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
    sdf = np.full(dims, 1e9, dtype=np.float32)
    for obj in objects:
        bvh, orientation = _world_bvh(obj, depsgraph)
        if bvh is None:
            continue
        # Distance field is only accurate near the obstacle; pad generously.
        lo, hi = _cell_range(obj, origin, dx, dims, pad=4)
        for i in range(lo[0], hi[0]):
            for j in range(lo[1], hi[1]):
                for k in range(lo[2], hi[2]):
                    wp = Vector((origin[0] + (i + 0.5) * dx,
                                 origin[1] + (j + 0.5) * dx,
                                 origin[2] + (k + 0.5) * dx))
                    loc, normal, _idx, dist = bvh.find_nearest(wp)
                    if loc is None:
                        continue
                    d = _signed_distance(
                        wp, loc, normal, dist, orientation)
                    if d < sdf[i, j, k]:
                        sdf[i, j, k] = d
    return sdf
