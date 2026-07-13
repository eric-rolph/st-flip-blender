"""Obstacle-SDF sampling regressions that do not require Blender."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]


def _load_voxelize(monkeypatch):
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = lambda values: np.asarray(tuple(values), dtype=float)
    bvhtree = types.ModuleType("mathutils.bvhtree")
    bvhtree.BVHTree = object
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)
    monkeypatch.setitem(sys.modules, "mathutils.bvhtree", bvhtree)

    name = "_stflip_voxelize_sdf_test"
    path = ROOT / "addon" / "voxelize.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _object_with_bounds(minimum, maximum):
    return types.SimpleNamespace(
        matrix_world=np.eye(3),
        bound_box=(minimum, maximum),
    )


class _CoordinateDistanceBVH:
    def __init__(self):
        self.points = []

    def find_nearest(self, point):
        point = np.asarray(point, dtype=float)
        self.points.append(tuple(point))
        return point.copy(), np.array((1.0, 0.0, 0.0)), 0, point.sum()


def test_combined_sdfs_use_world_space_cell_and_node_coordinates(monkeypatch):
    voxelize = _load_voxelize(monkeypatch)
    origin = np.array((10.0, 20.0, 30.0))
    dims = (2, 1, 1)
    obj = _object_with_bounds(origin, origin + np.array((4.0, 2.0, 2.0)))
    bvh = _CoordinateDistanceBVH()
    bounds = (origin, origin + np.array((4.0, 2.0, 2.0)))
    monkeypatch.setattr(
        voxelize, "_world_bvh", lambda *_args: (bvh, 1.0, bounds))

    cell_sdf, node_sdf = voxelize.solid_sdfs_from_objects(
        [obj], object(), origin, 2.0, dims)

    assert cell_sdf.shape == dims
    assert node_sdf.shape == (3, 2, 2)
    assert cell_sdf.dtype == node_sdf.dtype == np.float32
    assert np.allclose(cell_sdf[:, 0, 0], (63.0, 65.0))
    for i in range(node_sdf.shape[0]):
        for j in range(node_sdf.shape[1]):
            for k in range(node_sdf.shape[2]):
                assert node_sdf[i, j, k] == 60.0 + 2.0 * (i + j + k)
    assert len(bvh.points) == cell_sdf.size + node_sdf.size
    assert (11.0, 21.0, 31.0) in bvh.points
    assert (10.0, 20.0, 30.0) in bvh.points


def test_combined_sdfs_share_each_world_bvh_and_take_obstacle_minimum(
        monkeypatch):
    voxelize = _load_voxelize(monkeypatch)

    class ConstantBVH:
        def __init__(self, distance, inside=False):
            self.distance = distance
            self.inside = inside
            self.calls = 0

        def find_nearest(self, point):
            self.calls += 1
            point = np.asarray(point, dtype=float)
            location = point + np.array((1.0, 0.0, 0.0)) \
                if self.inside else point.copy()
            return location, np.array((1.0, 0.0, 0.0)), 0, self.distance

    outside = _object_with_bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    outside.bvh = ConstantBVH(0.75)
    inside = _object_with_bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    inside.bvh = ConstantBVH(0.25, inside=True)
    built = []

    def world_bvh(obj, _depsgraph):
        built.append(obj)
        return obj.bvh, 1.0, (obj.bound_box[0], obj.bound_box[1])

    monkeypatch.setattr(voxelize, "_world_bvh", world_bvh)

    cell_sdf, node_sdf = voxelize.solid_sdfs_from_objects(
        [outside, inside], object(), np.zeros(3), 1.0, (1, 1, 1))

    assert built == [outside, inside]
    assert np.all(cell_sdf == np.float32(-0.25))
    assert np.all(node_sdf == np.float32(-0.25))
    samples_per_object = cell_sdf.size + node_sdf.size
    assert outside.bvh.calls == inside.bvh.calls == samples_per_object


def test_legacy_cell_sdf_api_does_not_sample_the_node_grid(monkeypatch):
    voxelize = _load_voxelize(monkeypatch)
    obj = _object_with_bounds((0.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    bvh = _CoordinateDistanceBVH()
    monkeypatch.setattr(
        voxelize,
        "_world_bvh",
        lambda *_args: (bvh, 1.0, (obj.bound_box[0], obj.bound_box[1])),
    )

    sdf = voxelize.sdf_from_objects(
        [obj], object(), np.zeros(3), 1.0, (2, 1, 1))

    assert sdf.shape == (2, 1, 1)
    assert len(bvh.points) == sdf.size
    assert bvh.points == [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5)]


def test_combined_sdfs_for_no_obstacles_have_expected_shapes(monkeypatch):
    voxelize = _load_voxelize(monkeypatch)

    cell_sdf, node_sdf = voxelize.solid_sdfs_from_objects(
        [], object(), np.zeros(3), 0.25, (2, 3, 4))

    assert cell_sdf.shape == (2, 3, 4)
    assert node_sdf.shape == (3, 4, 5)
    assert np.all(cell_sdf == np.float32(1e9))
    assert np.all(node_sdf == np.float32(1e9))


def test_solid_velocity_from_objects_rigid_translation(monkeypatch):
    voxelize = _load_voxelize(monkeypatch)
    dims = (8, 8, 8)
    dx = 0.5
    origin = np.zeros(3)
    frame_dt = 0.1

    inside = np.zeros(dims, dtype=bool)
    inside[3:5, 3:5, 3:5] = True
    monkeypatch.setattr(
        voxelize, "mask_from_object",
        lambda obj, deps, org, d, dm: inside.copy())

    # Obstacle translated by (0.2, 0, -0.1) since the previous frame.
    m_cur = np.eye(4)
    m_cur[:3, 3] = (1.2, 0.0, -0.1)
    m_prev = np.eye(4)
    m_prev[:3, 3] = (1.0, 0.0, 0.0)
    obj = types.SimpleNamespace(name="Piston", matrix_world=m_cur)

    vel = voxelize.solid_velocity_from_objects(
        [obj], {"Piston": m_prev}, None, origin, dx, dims, frame_dt)

    assert vel is not None and vel.shape == dims + (3,)
    expected = np.array([0.2, 0.0, -0.1]) / frame_dt
    # Full velocity inside and in the dilated halo, zero far away.
    assert np.allclose(vel[4, 4, 4], expected, atol=1e-5)
    assert np.allclose(vel[2, 4, 4], expected, atol=1e-5)  # 1-cell halo
    assert np.allclose(vel[0, 0, 0], 0.0)

    # A static obstacle (matrix unchanged) contributes nothing.
    static = types.SimpleNamespace(name="Floor", matrix_world=np.eye(4))
    assert voxelize.solid_velocity_from_objects(
        [static], {"Floor": np.eye(4)}, None, origin, dx, dims, frame_dt,
    ) is None


def _cube_triangles(lo=0.25, hi=0.75):
    """12-triangle axis-aligned cube [lo, hi]^3 as a (12, 3, 3) array."""
    v = np.array([[x, y, z]
                  for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)])
    quads = [
        (0, 1, 3, 2), (4, 6, 7, 5),  # x- / x+
        (0, 4, 5, 1), (2, 3, 7, 6),  # y- / y+
        (0, 2, 6, 4), (1, 5, 7, 3),  # z- / z+
    ]
    tris = []
    for a, b, c, d in quads:
        tris.append([v[a], v[b], v[c]])
        tris.append([v[a], v[c], v[d]])
    return np.asarray(tris, dtype=np.float64)


def test_parity_inside_matches_analytic_cube(monkeypatch):
    voxelize = _load_voxelize(monkeypatch)
    tris = _cube_triangles()
    n = 16
    inside = voxelize._parity_inside(
        tris, (0.0, 0.0, 0.0), 1.0 / n, (n, n, n), (0.5, 0.5, 0.5))
    ax = (np.arange(n) + 0.5) / n
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    expected = ((x > 0.25) & (x < 0.75) & (y > 0.25) & (y < 0.75)
                & (z > 0.25) & (z < 0.75))
    assert np.array_equal(inside, expected)


def test_point_triangle_distance_analytic(monkeypatch):
    voxelize = _load_voxelize(monkeypatch)
    tris = _cube_triangles()
    pts = np.array([
        [0.5, 0.5, 0.9],    # 0.15 above the top face
        [0.5, 0.5, 0.5],    # centre: 0.25 from every face
        [0.9, 0.9, 0.9],    # nearest the (0.75,0.75,0.75) corner
    ])
    d = voxelize._point_triangle_distance(pts, tris, bound=10.0)
    assert abs(d[0] - 0.15) < 1e-9
    assert abs(d[1] - 0.25) < 1e-9
    assert abs(d[2] - np.sqrt(3 * 0.15 ** 2)) < 1e-9


def _brute_parity(voxelize, tris, origin, dx, counts, offset):
    """Reference parity: every ray tested against every triangle, no prefilter."""
    nx, ny, nz = counts
    jy, jz = 2.718281e-5 * dx, 3.141592e-5 * dx
    ys = origin[1] + (np.arange(ny) + offset[1]) * dx + jy
    zs = origin[2] + (np.arange(nz) + offset[2]) * dx + jz
    ry, rz = np.meshgrid(ys, zs, indexing="ij")
    ry, rz = ry.ravel(), rz.ravel()
    a = tris[:, 0]
    by_a, cy_a = tris[:, 1, 1] - a[:, 1], tris[:, 2, 1] - a[:, 1]
    bz_a, cz_a = tris[:, 1, 2] - a[:, 2], tris[:, 2, 2] - a[:, 2]
    det = by_a * cz_a - bz_a * cy_a
    ok = np.abs(det) > 1e-30
    inv = np.where(ok, det, 1.0)
    bx_a, cx_a = tris[:, 1, 0] - a[:, 0], tris[:, 2, 0] - a[:, 0]
    py = ry[:, None] - a[None, :, 1]
    pz = rz[:, None] - a[None, :, 2]
    u = (py * cz_a[None] - pz * cy_a[None]) / inv[None]
    v = (by_a[None] * pz - bz_a[None] * py) / inv[None]
    hit = ok[None] & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
    hist = np.zeros((nx + 1, len(ry)), dtype=np.int32)
    r, t = np.nonzero(hit)
    x_hit = a[t, 0] + u[r, t] * bx_a[t] + v[r, t] * cx_a[t]
    i0 = np.clip(np.floor((x_hit - origin[0]) / dx - offset[0]).astype(np.int64)
                 + 1, 0, nx)
    np.add.at(hist, (i0, r), 1)
    return (np.cumsum(hist[:nx], axis=0) % 2).astype(bool).reshape(nx, ny, nz)


def test_parity_prefilter_matches_bruteforce(monkeypatch):
    """The Morton ray ordering + yz-bbox triangle prefilter must give exactly
    the same inside mask as testing every ray against every triangle."""
    voxelize = _load_voxelize(monkeypatch)
    # A rotated octahedron: closed, non-axis-aligned, triangles with varied
    # yz footprints so the prefilter is genuinely exercised.
    c = np.array([0.5, 0.5, 0.5])
    axes = np.array([[0.35, 0.05, 0.02], [0.03, 0.32, 0.06], [0.04, 0.02, 0.3]])
    v = [c + axes[0], c - axes[0], c + axes[1], c - axes[1],
         c + axes[2], c - axes[2]]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    tris = np.asarray([[v[a], v[b], v[c_]] for a, b, c_ in faces],
                      dtype=np.float64)
    n = 24
    args = (tris, (0.0, 0.0, 0.0), 1.0 / n, (n, n, n), (0.5, 0.5, 0.5))
    fast = voxelize._parity_inside(*args)
    ref = _brute_parity(voxelize, *args)
    assert np.array_equal(fast, ref)
    assert fast.any()          # the shape is actually inside somewhere


def test_morton_order_is_a_permutation(monkeypatch):
    voxelize = _load_voxelize(monkeypatch)
    idx = np.array([[0, 0, 0], [3, 1, 2], [1, 7, 4], [7, 7, 7], [2, 2, 2]])
    order = voxelize._morton_order(idx)
    assert sorted(order.tolist()) == list(range(len(idx)))


def test_morton_reordering_does_not_change_distances(monkeypatch):
    """The Morton curve only reorders which cells are batched together for the
    per-chunk triangle prefilter; it must not change any computed distance."""
    voxelize = _load_voxelize(monkeypatch)
    tris = _cube_triangles()
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.2, 1.2, size=(4000, 3))
    raster = voxelize._point_triangle_distance(pts, tris, bound=10.0, chunk=512)
    order = voxelize._morton_order(
        np.floor(pts * 16).astype(np.int64))
    reordered = voxelize._point_triangle_distance(
        pts[order], tris, bound=10.0, chunk=512)
    restored = np.empty_like(reordered)
    restored[order] = reordered
    np.testing.assert_array_equal(restored, raster)


def test_point_triangle_distance_is_chunk_independent(monkeypatch):
    """The per-chunk bbox prefilter only drops triangles provably beyond the
    bound, so the distance is identical for any chunk size."""
    voxelize = _load_voxelize(monkeypatch)
    tris = _cube_triangles()
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.3, 1.3, size=(3000, 3))
    ref = voxelize._point_triangle_distance(pts, tris, bound=0.2, chunk=512)
    for chunk in (1, 64, 257):
        got = voxelize._point_triangle_distance(pts, tris, bound=0.2, chunk=chunk)
        np.testing.assert_array_equal(got, ref)


def test_signed_band_sdf_cube(monkeypatch):
    voxelize = _load_voxelize(monkeypatch)
    tris = _cube_triangles()
    n = 16
    dx = 1.0 / n
    sdf = voxelize._signed_band_sdf(
        tris, (0.0, 0.0, 0.0), dx, (n, n, n), (0.5, 0.5, 0.5), band=3)
    # Signs: centre inside, corner outside.
    assert sdf[8, 8, 8] < 0 < sdf[0, 0, 0]
    # Band accuracy: cell centre (8, 8, 12) sits at z=0.78125, 0.03125 above
    # the top face and inside the band.
    assert abs(sdf[8, 8, 12] - 0.03125) < 1e-5
    # Just inside the top face: z=0.71875 -> -0.03125.
    assert abs(sdf[8, 8, 11] + 0.03125) < 1e-5
    # The far corner exceeds the band bound and is capped at +sat; the cube
    # centre is still inside the dilated band, so it gets its exact distance
    # to the nearest face (0.75 - 0.53125 = 0.21875).
    sat = 4 * dx
    assert sdf[0, 0, 0] == np.float32(sat)
    assert abs(sdf[8, 8, 8] + 0.21875) < 1e-5


def test_fast_mask_and_sdf_wrappers_use_triangles(monkeypatch):
    """Objects exposing evaluated_get/to_mesh take the vectorized path."""
    voxelize = _load_voxelize(monkeypatch)
    tris = _cube_triangles()

    class _FakeObj:
        def evaluated_get(self, deps):
            return self

        def to_mesh(self):
            return self

        def to_mesh_clear(self):
            pass

        def calc_loop_triangles(self):
            pass

        matrix_world = np.eye(4)

        @property
        def vertices(self):
            verts = np.unique(tris.reshape(-1, 3), axis=0)
            self._verts = verts

            class _V:
                def __len__(_s):
                    return len(verts)

                def foreach_get(_s, name, buf):
                    buf[:] = verts.ravel()
            return _V()

        @property
        def loop_triangles(self):
            verts = np.unique(tris.reshape(-1, 3), axis=0)
            lut = {tuple(v): i for i, v in enumerate(verts)}
            idx = np.array([[lut[tuple(p)] for p in t] for t in tris])

            class _T:
                def __len__(_s):
                    return len(idx)

                def foreach_get(_s, name, buf):
                    buf[:] = idx.ravel()
            return _T()

    obj = _FakeObj()
    n = 16
    dx = 1.0 / n
    mask = voxelize.mask_from_object(obj, None, np.zeros(3), dx, (n, n, n))
    assert mask.sum() == 8 ** 3  # cells with centres in (0.25, 0.75)^3
    cell_sdf, node_sdf = voxelize.solid_sdfs_from_objects(
        [obj], None, np.zeros(3), dx, (n, n, n))
    assert cell_sdf.shape == (n, n, n)
    assert node_sdf.shape == (n + 1, n + 1, n + 1)
    assert cell_sdf[8, 8, 8] < 0 < cell_sdf[0, 0, 0]
    # Node exactly on the top face plane (z = 0.75 -> k = 12): |sdf| ~ 0.
    assert abs(node_sdf[8, 8, 12]) < 1e-4


def test_solid_velocity_deforming_mesh(monkeypatch):
    """Matrix unchanged but vertices moved (armature-style deformation):
    band cells must take the nearest current vertex's displacement / dt."""
    voxelize = _load_voxelize(monkeypatch)
    dims = (8, 8, 8)
    dx = 0.5
    origin = np.zeros(3)
    frame_dt = 0.1

    inside = np.zeros(dims, dtype=bool)
    inside[3:5, 3:5, 3:5] = True
    monkeypatch.setattr(
        voxelize, "mask_from_object",
        lambda obj, deps, org, d, dm: inside.copy())

    verts_prev = np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.5]])
    verts_cur = verts_prev + np.array([0.05, 0.0, 0.1])  # uniform deformation
    monkeypatch.setattr(
        voxelize, "_extract_vertices", lambda obj, deps: verts_cur.copy())

    obj = types.SimpleNamespace(name="Flag", matrix_world=np.eye(4))
    vel = voxelize.solid_velocity_from_objects(
        [obj], {"Flag": np.eye(4)}, None, origin, dx, dims, frame_dt,
        prev_vertices={"Flag": verts_prev})

    assert vel is not None
    expected = np.array([0.05, 0.0, 0.1]) / frame_dt
    assert np.allclose(vel[4, 4, 4], expected, atol=1e-5)
    assert np.allclose(vel[0, 0, 0], 0.0)

    # Unchanged vertices and matrix -> nothing moved.
    assert voxelize.solid_velocity_from_objects(
        [obj], {"Flag": np.eye(4)}, None, origin, dx, dims, frame_dt,
        prev_vertices={"Flag": verts_cur},
    ) is None
