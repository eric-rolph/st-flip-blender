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
