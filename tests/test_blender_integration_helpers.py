"""Blender integration regressions that run without importing real ``bpy``."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_source(monkeypatch, name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_handlers(monkeypatch):
    root = "_stflip_handler_test"
    for name in (root, f"{root}.addon", f"{root}.stflip"):
        monkeypatch.setitem(sys.modules, name, _package(name))

    handlers_module = types.ModuleType("bpy.app.handlers")
    handlers_module.persistent = lambda function: function
    handlers_module.frame_change_post = []
    app_module = types.ModuleType("bpy.app")
    app_module.handlers = handlers_module
    bpy = types.ModuleType("bpy")
    bpy.app = app_module
    bpy.data = types.SimpleNamespace(filepath="", objects={})
    bpy.path = types.SimpleNamespace(abspath=lambda path: path)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "bpy.app", app_module)
    monkeypatch.setitem(sys.modules, "bpy.app.handlers", handlers_module)

    monkeypatch.setitem(
        sys.modules, f"{root}.stflip.cache",
        types.ModuleType(f"{root}.stflip.cache"),
    )
    mesher = types.ModuleType(f"{root}.addon.mesher")
    mesher.PARTICLE_OBJ = "STFLIP Particles"
    monkeypatch.setitem(sys.modules, mesher.__name__, mesher)
    module = _load_source(
        monkeypatch, f"{root}.addon.handlers", "addon/handlers.py")
    return module, bpy


def test_unsaved_relative_cache_is_stable_and_scene_unique(monkeypatch):
    handlers, _bpy = _load_handlers(monkeypatch)

    class Scene:
        def __init__(self, pointer):
            self._pointer = pointer
            self.stflip = types.SimpleNamespace(cache_dir="//stflip_cache")

        def as_pointer(self):
            return self._pointer

    first = Scene(0xABC)
    second = Scene(0xDEF)
    first_path = handlers.resolve_cache_dir(first)

    assert handlers.resolve_cache_dir(first) == first_path
    assert handlers.resolve_cache_dir(second) != first_path
    assert f"{os.getpid()}-abc" in first_path.lower()
    assert os.path.isabs(first_path)


def test_unsaved_absolute_cache_is_respected(monkeypatch, tmp_path):
    handlers, _bpy = _load_handlers(monkeypatch)
    scene = types.SimpleNamespace(
        stflip=types.SimpleNamespace(cache_dir=str(tmp_path)),
    )

    assert handlers.resolve_cache_dir(scene) == os.path.normpath(tmp_path)


def test_cleanup_removes_only_current_process_temporary_caches(
        monkeypatch, tmp_path):
    handlers, _bpy = _load_handlers(monkeypatch)
    monkeypatch.setattr(handlers.tempfile, "gettempdir", lambda: str(tmp_path))
    root = tmp_path / "stflip_cache"
    own = root / f"{os.getpid()}-abc"
    other = root / "999999-def"
    unrelated = root / f"prefix-{os.getpid()}"
    for path in (own, other, unrelated):
        path.mkdir(parents=True)
        (path / "frame.npz").write_bytes(b"cache")

    assert handlers._cleanup_process_temporary_caches() == 1
    assert not own.exists()
    assert other.exists()
    assert unrelated.exists()


def test_cleanup_uses_process_exit_callback_without_blender_quit_handler(
        monkeypatch, tmp_path):
    handlers, bpy = _load_handlers(monkeypatch)
    registered = []
    unregistered = []
    monkeypatch.setattr(
        handlers.atexit, "register", lambda callback: registered.append(callback))
    monkeypatch.setattr(
        handlers.atexit, "unregister",
        lambda callback: unregistered.append(callback),
    )
    monkeypatch.setattr(handlers, "_cleanup_process_temporary_caches", lambda: 0)

    assert not hasattr(bpy.app.handlers, "quit_pre")
    handlers.ensure_registered()
    handlers.ensure_registered()
    assert registered == [handlers._cleanup_process_temporary_caches]

    handlers.unregister()
    assert unregistered == [handlers._cleanup_process_temporary_caches]
    assert handlers._ATEXIT_REGISTERED is False


def test_cache_frame_bounds_reject_invalid_metadata(monkeypatch):
    handlers, _bpy = _load_handlers(monkeypatch)

    assert handlers._frame_bounds({}, 3) == (3, 3)
    assert handlers._frame_bounds(
        {"frame_start": 2, "frame_end_baked": 9}, 3) == (2, 9)
    assert handlers._frame_bounds(
        {"frame_start": "bad", "frame_end_baked": 9}, 3) is None
    assert handlers._frame_bounds(
        {"frame_start": 2, "frame_end_baked": None}, 3) is None
    assert handlers._frame_bounds(
        {"frame_start": True, "frame_end_baked": 9}, 3) is None
    assert handlers._frame_bounds(
        {"frame_start": 9, "frame_end_baked": 2}, 3) is None


def test_frame_playback_does_not_require_domain_binding(monkeypatch):
    handlers, _bpy = _load_handlers(monkeypatch)
    seen = []
    monkeypatch.setattr(
        handlers, "_apply_frame", lambda scene, frame: seen.append(frame))
    scene = types.SimpleNamespace(
        stflip=types.SimpleNamespace(domain=None), frame_current=17)

    handlers.stflip_frame_change(scene)

    assert seen == [17]


def _load_mesher(monkeypatch, bound_particle, canonical_particle):
    bpy = types.ModuleType("bpy")
    scene_objects = (
        {} if bound_particle is None
        else {bound_particle.name: bound_particle}
    )
    scene = types.SimpleNamespace(
        stflip=types.SimpleNamespace(
            particle_object=bound_particle, surface_object=None),
        objects=scene_objects,
    )
    scene.collection = types.SimpleNamespace(
        objects=types.SimpleNamespace(link=lambda obj: scene.objects.update(
            {obj.name: obj})))
    bpy.context = types.SimpleNamespace(scene=scene)
    bpy.data = types.SimpleNamespace(
        objects=types.SimpleNamespace(get=lambda name: canonical_particle),
        meshes=types.SimpleNamespace(
            new=lambda name: (_ for _ in ()).throw(
                AssertionError("unexpected duplicate mesh"))),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    return _load_source(monkeypatch, "_stflip_mesher_test", "addon/mesher.py")


def test_particle_helper_reuses_pointer_bound_renamed_object(monkeypatch):
    bound = types.SimpleNamespace(name="My Renamed Particles", type="MESH")
    canonical = types.SimpleNamespace(name="STFLIP Particles", type="MESH")
    mesher = _load_mesher(monkeypatch, bound, canonical)

    result = mesher.ensure_particle_object()

    assert result is bound


def test_particle_helper_explicit_object_wins_over_scene_binding(monkeypatch):
    bound = types.SimpleNamespace(name="Bound Particles", type="MESH")
    canonical = types.SimpleNamespace(name="STFLIP Particles", type="MESH")
    explicit = types.SimpleNamespace(name="Explicit Particles", type="MESH")
    mesher = _load_mesher(monkeypatch, bound, canonical)
    mesher.bpy.context.scene.objects[explicit.name] = explicit

    result = mesher.ensure_particle_object(existing_obj=explicit)

    assert result is explicit
    assert mesher.bpy.context.scene.stflip.particle_object is explicit


def test_legacy_named_output_from_another_scene_is_not_reused(monkeypatch):
    canonical = types.SimpleNamespace(name="STFLIP Particles", type="MESH")
    mesher = _load_mesher(monkeypatch, None, canonical)

    result = mesher._mesh_output(
        None, "particle_object", "STFLIP Particles")

    assert result is None


def test_legacy_named_output_in_current_scene_is_reused(monkeypatch):
    canonical = types.SimpleNamespace(name="STFLIP Particles", type="MESH")
    mesher = _load_mesher(monkeypatch, None, canonical)
    mesher.bpy.context.scene.objects[canonical.name] = canonical

    result = mesher._mesh_output(
        None, "particle_object", "STFLIP Particles")

    assert result is canonical


def _load_voxelize(monkeypatch, bvh_type):
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = lambda values: np.asarray(tuple(values), dtype=float)
    bvhtree = types.ModuleType("mathutils.bvhtree")
    bvhtree.BVHTree = bvh_type
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)
    monkeypatch.setitem(sys.modules, "mathutils.bvhtree", bvhtree)
    return _load_source(
        monkeypatch, "_stflip_voxelize_test", "addon/voxelize.py")


def test_world_bvh_transforms_evaluated_vertices_before_distance_queries(
        monkeypatch):
    calls = []

    class FakeBVH:
        @classmethod
        def FromPolygons(cls, vertices, polygons, all_triangles=False):
            calls.append((vertices, polygons, all_triangles))
            return object()

    class FakeMatrix:
        linear = np.array(((0.0, -0.5, 0.0),
                           (2.0, 0.0, 0.0),
                           (0.0, 0.0, -3.0)))
        translation = np.array((4.0, 5.0, 6.0))

        def copy(self):
            return self

        def __matmul__(self, point):
            return self.linear @ point + self.translation

        def to_3x3(self):
            return types.SimpleNamespace(
                determinant=lambda: float(np.linalg.det(self.linear)))

    mesh = types.SimpleNamespace(
        vertices=[types.SimpleNamespace(co=np.array((1.0, 0.0, 0.0))),
                  types.SimpleNamespace(co=np.array((0.0, 1.0, 0.0))),
                  types.SimpleNamespace(co=np.array((0.0, 0.0, 1.0)))],
        polygons=[types.SimpleNamespace(vertices=(0, 1, 2))],
    )
    evaluated = types.SimpleNamespace(
        matrix_world=FakeMatrix(),
        to_mesh=lambda: mesh,
        to_mesh_clear=lambda: setattr(evaluated, "cleared", True),
        cleared=False,
    )
    obj = types.SimpleNamespace(evaluated_get=lambda depsgraph: evaluated)
    voxelize = _load_voxelize(monkeypatch, FakeBVH)

    bvh, orientation, bounds = voxelize._world_bvh(obj, object())

    assert bvh is not None
    assert orientation == -1.0
    assert evaluated.cleared is True
    vertices, polygons, all_triangles = calls[0]
    expected = [FakeMatrix() @ vertex.co for vertex in mesh.vertices]
    assert np.allclose(vertices, expected)
    assert np.allclose(bounds[0], np.min(expected, axis=0))
    assert np.allclose(bounds[1], np.max(expected, axis=0))
    assert polygons == [(0, 1, 2)]
    assert all_triangles is False


def test_reflection_orientation_keeps_signed_distance_outward(monkeypatch):
    voxelize = _load_voxelize(monkeypatch, object)
    point = np.array((-2.0, 0.0, 0.0))
    location = np.array((-1.0, 0.0, 0.0))
    winding_normal = np.array((1.0, 0.0, 0.0))

    distance = voxelize._signed_distance(
        point, location, winding_normal, 1.0, orientation=-1.0)

    assert distance == 1.0
