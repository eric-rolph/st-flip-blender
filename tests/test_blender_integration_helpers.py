"""Blender integration regressions that run without importing real ``bpy``."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from stflip import cache as core_cache


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
    handlers_module.load_post = []
    app_module = types.ModuleType("bpy.app")
    app_module.handlers = handlers_module
    bpy = types.ModuleType("bpy")
    bpy.app = app_module
    bpy.data = types.SimpleNamespace(filepath="", objects={}, scenes=[])
    bpy.path = types.SimpleNamespace(abspath=lambda path: path)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "bpy.app", app_module)
    monkeypatch.setitem(sys.modules, "bpy.app.handlers", handlers_module)

    monkeypatch.setitem(sys.modules, f"{root}.stflip.cache", core_cache)
    mesher = types.ModuleType(f"{root}.addon.mesher")
    mesher.PARTICLE_OBJ = "STFLIP Particles"

    def output_is_exclusive(scene, obj):
        mesh = getattr(obj, "data", None)
        for other_scene in bpy.data.scenes:
            if other_scene is scene:
                continue
            candidates = (other_scene.objects.values()
                          if hasattr(other_scene.objects, "values")
                          else other_scene.objects)
            for candidate in candidates:
                if candidate is obj or (
                        mesh is not None
                        and getattr(candidate, "data", None) is mesh):
                    return False
        return True

    mesher.output_is_exclusive = output_is_exclusive
    mesher.update_paper_surface_mesh = lambda obj, *mesh: obj
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


def _configure_saved_file(bpy, tmp_path):
    bpy.data.filepath = str(tmp_path / "project.blend")

    def abspath(path):
        if path.startswith("//"):
            return str(tmp_path / path[2:])
        return path

    bpy.path.abspath = abspath


def _scene(cache_dir="//stflip_cache", *, cache_id="", pointer=1,
           status="", state="IDLE", particle=None, surface=None,
           create_surface=False, surface_method="FAST_PREVIEW"):
    settings = types.SimpleNamespace(
        cache_dir=cache_dir,
        cache_id=cache_id,
        bake_status=status,
        bake_state=state,
        bake_error="",
        particle_object=particle,
        surface_object=surface,
        create_surface=create_surface,
        surface_method=surface_method,
    )
    objects = {}
    for obj in (particle, surface):
        if obj is not None:
            objects[obj.name] = obj
    return types.SimpleNamespace(
        stflip=settings,
        frame_start=1,
        frame_current=1,
        objects=objects,
        as_pointer=lambda: pointer,
    )


def test_saved_default_cache_is_persistent_and_scene_unique(
        monkeypatch, tmp_path):
    handlers, bpy = _load_handlers(monkeypatch)
    _configure_saved_file(bpy, tmp_path)
    first = _scene(pointer=0xABC)
    second = _scene(pointer=0xDEF)
    bpy.data.scenes = [first, second]

    first_path = handlers.resolve_cache_dir(first)
    second_path = handlers.resolve_cache_dir(second)

    assert first_path != second_path
    assert os.path.dirname(first_path) == os.path.normpath(
        str(tmp_path / "stflip_cache"))
    assert os.path.basename(first_path) == first.stflip.cache_id
    assert os.path.basename(second_path) == second.stflip.cache_id
    assert len(first.stflip.cache_id) == 32
    assert len(second.stflip.cache_id) == 32
    assert handlers.resolve_cache_dir(first) == first_path


def test_duplicate_scene_cache_ids_are_repaired_deterministically(
        monkeypatch, tmp_path):
    handlers, bpy = _load_handlers(monkeypatch)
    _configure_saved_file(bpy, tmp_path)
    duplicate = "0123456789abcdef0123456789abcdef"
    first = _scene(cache_id=duplicate, pointer=1)
    second = _scene(cache_id=duplicate, pointer=2)
    bpy.data.scenes = [first, second]

    assert handlers.ensure_scene_cache_id(first) == duplicate
    replacement = handlers.ensure_scene_cache_id(second)

    assert replacement != duplicate
    assert second.stflip.cache_id == replacement
    assert handlers.ensure_scene_cache_id(first) == duplicate
    assert handlers.ensure_scene_cache_id(second) == replacement


def test_scene_cache_id_survives_distinct_rna_wrapper_identity(monkeypatch):
    handlers, bpy = _load_handlers(monkeypatch)
    owner = "0123456789abcdef0123456789abcdef"
    scene = _scene(cache_id=owner, pointer=41)
    wrapper = types.SimpleNamespace(
        stflip=scene.stflip,
        as_pointer=lambda: 41,
    )
    bpy.data.scenes = [wrapper]

    assert wrapper is not scene
    assert handlers.ensure_scene_cache_id(scene) == owner
    assert scene.stflip.cache_id == owner


def test_runtime_cache_owner_survives_copy_inserted_before_source(monkeypatch):
    handlers, bpy = _load_handlers(monkeypatch)
    owner = "0123456789abcdef0123456789abcdef"
    source = _scene(cache_id=owner, pointer=51)
    bpy.data.scenes = [source]
    assert handlers.ensure_scene_cache_id(source) == owner

    copied = _scene(cache_id=owner, pointer=52)
    bpy.data.scenes = [copied, source]

    assert handlers.ensure_scene_cache_id(source) == owner
    copied_id = handlers.ensure_scene_cache_id(copied)
    assert copied_id != owner
    assert source.stflip.cache_id == owner
    assert copied.stflip.cache_id == copied_id


def test_saved_explicit_relative_cache_path_remains_exact(monkeypatch, tmp_path):
    handlers, bpy = _load_handlers(monkeypatch)
    _configure_saved_file(bpy, tmp_path)
    scene = _scene(cache_dir="//shared/custom-cache")
    bpy.data.scenes = [scene]

    assert handlers.resolve_cache_dir(scene) == os.path.normpath(
        str(tmp_path / "shared" / "custom-cache"))


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
    assert bpy.app.handlers.frame_change_post == [handlers.stflip_frame_change]
    assert bpy.app.handlers.load_post == [handlers.stflip_load_post]

    handlers.unregister()
    assert unregistered == [handlers._cleanup_process_temporary_caches]
    assert handlers._ATEXIT_REGISTERED is False
    assert bpy.app.handlers.frame_change_post == []
    assert bpy.app.handlers.load_post == []


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


class _FakeMesh:
    def __init__(self):
        self.clear_count = 0
        self.update_count = 0

    def clear_geometry(self):
        self.clear_count += 1

    def update(self):
        self.update_count += 1


def _particle(name="STFLIP Particles"):
    mesh = _FakeMesh()
    obj = types.SimpleNamespace(
        name=name,
        type="MESH",
        data=mesh,
        tag_count=0,
    )

    def update_tag():
        obj.tag_count += 1

    obj.update_tag = update_tag
    return obj


class _PointValues:
    def __init__(self, count=0):
        self.count = count
        self.writes = {}

    def __len__(self):
        return self.count

    def add(self, count):
        self.count += count

    def foreach_set(self, name, values):
        self.writes[name] = np.asarray(values).copy()


class _PlaybackAttributes(dict):
    def __init__(self, vertices):
        super().__init__()
        self.vertices = vertices

    def new(self, name, data_type, domain):
        attribute = types.SimpleNamespace(
            name=name,
            data_type=data_type,
            domain=domain,
            data=_PointValues(len(self.vertices)),
        )
        self[name] = attribute
        return attribute

    def remove(self, attribute):
        self.pop(attribute.name, None)


class _PlaybackMesh(_FakeMesh):
    def __init__(self, vertex_count):
        super().__init__()
        self.vertices = _PointValues(vertex_count)
        self.attributes = _PlaybackAttributes(self.vertices)

    def clear_geometry(self):
        super().clear_geometry()
        self.vertices.count = 0
        self.attributes.clear()


def _playback_particle(vertex_count=2):
    obj = types.SimpleNamespace(
        name="Playback Particles",
        type="MESH",
        data=_PlaybackMesh(vertex_count),
        tag_count=0,
    )
    obj.update_tag = lambda: setattr(obj, "tag_count", obj.tag_count + 1)
    return obj


class _SurfaceObject(dict):
    def __init__(self, method="PAPER_MCF", name="Playback Surface"):
        super().__init__()
        self.name = name
        self.type = "MESH"
        self.data = _FakeMesh()
        self.tag_count = 0
        self["stflip_surface_method"] = method

    def update_tag(self):
        self.tag_count += 1


def _playback_surface_config():
    return {
        "schema": core_cache.SURFACE_CONFIG_SCHEMA,
        "version": core_cache.SURFACE_CONFIG_VERSION,
        "algorithm": "appendix_b_feature_preserving_mcf_v1",
        "mcf_iterations": 30,
    }


def _playback_fingerprint():
    return core_cache.surface_config_fingerprint(
        _playback_surface_config())


def _playback_metadata(owner, fingerprint=None):
    config = _playback_surface_config()
    if fingerprint is None:
        fingerprint = core_cache.surface_config_fingerprint(config)
    return {
        "frame_start": 2,
        "frame_end_baked": 4,
        core_cache.OWNER_KEY: owner,
        "surface_reconstruction": {
            "schema": core_cache.SURFACE_SCHEMA,
            "version": core_cache.SURFACE_VERSION,
            "mode": "PAPER_MCF",
            "config": config,
            "fingerprint": fingerprint,
        },
    }


def test_frame_playback_updates_bound_paper_surface_at_clamped_frame(
        monkeypatch):
    handlers, bpy = _load_handlers(monkeypatch)
    owner = "0123456789abcdef0123456789abcdef"
    fingerprint = _playback_fingerprint()
    particle = _playback_particle()
    surface = _SurfaceObject("FAST_PREVIEW")
    scene = _scene(
        cache_dir="C:/cache",
        cache_id=owner,
        particle=particle,
        surface=surface,
        create_surface=True,
        surface_method="PAPER_MCF",
    )
    bpy.data.scenes = [scene]
    positions = np.array(
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=np.float32)
    velocities = -positions
    paper_mesh = (
        np.array(((0.0, 0.0, 0.0),), dtype=np.float32),
        np.empty((0, 3), dtype=np.int32),
        np.empty((0, 4), dtype=np.int32),
    )
    reads = []
    updates = []
    monkeypatch.setattr(
        handlers.cache,
        "read_meta",
        lambda path: _playback_metadata(owner, fingerprint),
    )

    def read_frame(path, frame):
        reads.append(("particle", path, frame))
        return positions, velocities

    def read_surface(path, frame, requested_fingerprint, **kwargs):
        reads.append((
            "surface",
            path,
            frame,
            requested_fingerprint,
            np.array_equal(kwargs.get("expected_source_positions"), positions),
        ))
        return paper_mesh

    def update_surface(obj, *mesh):
        updates.append((obj, mesh))
        return obj

    monkeypatch.setattr(handlers.cache, "read_frame", read_frame)
    monkeypatch.setattr(handlers.cache, "read_surface", read_surface)
    monkeypatch.setattr(
        handlers.mesher, "update_paper_surface_mesh", update_surface)

    assert handlers._apply_frame(scene, 99) is True

    cache_dir = handlers.resolve_cache_dir(scene)
    assert reads == [
        ("particle", cache_dir, 4),
        ("surface", cache_dir, 4, fingerprint, True),
    ]
    assert updates == [(surface, paper_mesh)]
    np.testing.assert_array_equal(
        particle.data.vertices.writes["co"], positions.ravel())
    np.testing.assert_array_equal(
        particle.data.attributes["velocity"].data.writes["vector"],
        velocities.ravel(),
    )


@pytest.mark.parametrize(
    "mutation", ["schema", "version", "mode", "config_fingerprint"])
def test_paper_playback_rejects_invalid_metadata_without_hiding_particles(
        monkeypatch, mutation):
    handlers, bpy = _load_handlers(monkeypatch)
    owner = "0123456789abcdef0123456789abcdef"
    metadata = _playback_metadata(owner)
    surface_metadata = metadata["surface_reconstruction"]
    if mutation == "schema":
        surface_metadata["schema"] = "other-surface"
    elif mutation == "version":
        surface_metadata["version"] = core_cache.SURFACE_VERSION + 1
    elif mutation == "mode":
        surface_metadata["mode"] = "FAST_PREVIEW"
    else:
        surface_metadata["config"]["mcf_iterations"] += 1

    particle = _playback_particle()
    surface = _SurfaceObject()
    scene = _scene(
        cache_dir="C:/cache",
        cache_id=owner,
        particle=particle,
        surface=surface,
        create_surface=True,
        surface_method="PAPER_MCF",
    )
    bpy.data.scenes = [scene]
    positions = np.ones((2, 3), dtype=np.float32)
    monkeypatch.setattr(handlers.cache, "read_meta", lambda path: metadata)
    monkeypatch.setattr(
        handlers.cache,
        "read_frame",
        lambda path, frame: (positions, -positions),
    )
    monkeypatch.setattr(
        handlers.cache,
        "read_surface",
        lambda *args, **kwargs: pytest.fail(
            "invalid surface metadata reached the mesh cache"),
    )
    monkeypatch.setattr(
        handlers.mesher,
        "update_paper_surface_mesh",
        lambda *args: pytest.fail("invalid surface metadata was applied"),
    )

    handlers.stflip_frame_change(scene)

    assert particle.data.clear_count == 0
    assert particle.data.update_count == 1
    assert len(particle.data.vertices) == 2
    assert surface.data.clear_count == 1
    assert surface.data.update_count == 1


@pytest.mark.parametrize(
    "failure", ["missing", "corrupt", "mismatched", "invalid_fingerprint"])
def test_bad_paper_cache_clears_only_plain_surface_during_playback(
        monkeypatch, failure):
    handlers, bpy = _load_handlers(monkeypatch)
    owner = "0123456789abcdef0123456789abcdef"
    fingerprint = _playback_fingerprint()
    particle = _playback_particle()
    surface = _SurfaceObject()
    scene = _scene(
        cache_dir="C:/cache",
        cache_id=owner,
        particle=particle,
        surface=surface,
        create_surface=True,
        surface_method="PAPER_MCF",
    )
    bpy.data.scenes = [scene]
    positions = np.ones((2, 3), dtype=np.float32)
    monkeypatch.setattr(
        handlers.cache,
        "read_meta",
        lambda path: _playback_metadata(owner, fingerprint),
    )
    monkeypatch.setattr(
        handlers.cache,
        "read_frame",
        lambda path, frame: (positions, -positions),
    )

    def read_surface(path, frame, requested_fingerprint, **_kwargs):
        if failure == "missing":
            return None
        if failure == "invalid_fingerprint":
            raise handlers.cache.CheckpointError(
                "checkpoint fingerprint is invalid")
        raise handlers.cache.SurfaceCacheError(
            "surface is corrupt" if failure == "corrupt"
            else "surface fingerprint binding does not match filename")

    monkeypatch.setattr(handlers.cache, "read_surface", read_surface)
    monkeypatch.setattr(
        handlers.mesher,
        "update_paper_surface_mesh",
        lambda *args: pytest.fail("invalid paper cache was applied"),
    )

    handlers.stflip_frame_change(scene)

    assert particle.data.clear_count == 0
    assert particle.data.update_count == 1
    assert len(particle.data.vertices) == 2
    assert surface.data.clear_count == 1
    assert surface.data.update_count == 1
    assert surface.tag_count == 1
    assert scene.stflip.bake_state == "IDLE"
    assert scene.stflip.bake_status == ""


def test_fast_preview_frame_playback_does_not_touch_surface_cache(
        monkeypatch):
    handlers, bpy = _load_handlers(monkeypatch)
    owner = "0123456789abcdef0123456789abcdef"
    fingerprint = _playback_fingerprint()
    particle = _playback_particle()
    surface = _SurfaceObject()
    scene = _scene(
        cache_dir="C:/cache",
        cache_id=owner,
        particle=particle,
        surface=surface,
        create_surface=True,
        surface_method="FAST_PREVIEW",
    )
    bpy.data.scenes = [scene]
    values = np.zeros((2, 3), dtype=np.float32)
    monkeypatch.setattr(
        handlers.cache,
        "read_meta",
        lambda path: _playback_metadata(owner, fingerprint),
    )
    monkeypatch.setattr(
        handlers.cache, "read_frame", lambda path, frame: (values, values))
    monkeypatch.setattr(
        handlers.cache,
        "read_surface",
        lambda *args: pytest.fail("preview playback read paper cache"),
    )
    monkeypatch.setattr(
        handlers.mesher,
        "update_paper_surface_mesh",
        lambda *args: pytest.fail("preview playback updated paper surface"),
    )

    assert handlers._apply_frame(scene, 2) is True
    assert particle.data.update_count == 1
    assert surface.data.clear_count == 0


def test_paper_frame_handler_does_not_create_missing_surface_output(
        monkeypatch):
    handlers, bpy = _load_handlers(monkeypatch)
    owner = "0123456789abcdef0123456789abcdef"
    fingerprint = _playback_fingerprint()
    particle = _playback_particle()
    scene = _scene(
        cache_dir="C:/cache",
        cache_id=owner,
        particle=particle,
        create_surface=True,
        surface_method="PAPER_MCF",
    )
    bpy.data.scenes = [scene]
    values = np.zeros((2, 3), dtype=np.float32)
    monkeypatch.setattr(
        handlers.cache,
        "read_meta",
        lambda path: _playback_metadata(owner, fingerprint),
    )
    monkeypatch.setattr(
        handlers.cache, "read_frame", lambda path, frame: (values, values))
    monkeypatch.setattr(
        handlers.cache,
        "read_surface",
        lambda *args: pytest.fail("paper cache read without bound output"),
    )
    monkeypatch.setattr(
        handlers.mesher,
        "update_paper_surface_mesh",
        lambda *args: pytest.fail("frame handler created a surface output"),
    )

    assert handlers._apply_frame(scene, 2) is True
    assert scene.stflip.surface_object is None


def test_paper_frame_handler_never_mutates_shared_surface_mesh(monkeypatch):
    handlers, bpy = _load_handlers(monkeypatch)
    owner = "0123456789abcdef0123456789abcdef"
    fingerprint = _playback_fingerprint()
    particle = _playback_particle()
    shared_surface = _SurfaceObject()
    scene = _scene(
        cache_dir="C:/cache",
        cache_id=owner,
        particle=particle,
        surface=shared_surface,
        create_surface=True,
        surface_method="PAPER_MCF",
        pointer=1,
    )
    other = _scene(surface=shared_surface, pointer=2)
    bpy.data.scenes = [scene, other]
    values = np.zeros((2, 3), dtype=np.float32)
    monkeypatch.setattr(
        handlers.cache,
        "read_meta",
        lambda path: _playback_metadata(owner, fingerprint),
    )
    monkeypatch.setattr(
        handlers.cache, "read_frame", lambda path, frame: (values, values))
    monkeypatch.setattr(
        handlers.cache,
        "read_surface",
        lambda *args: pytest.fail("shared surface cache was read"),
    )
    monkeypatch.setattr(
        handlers.mesher,
        "update_paper_surface_mesh",
        lambda *args: pytest.fail("shared surface mesh was mutated"),
    )

    assert handlers._apply_frame(scene, 2) is True
    assert particle.data.update_count == 1
    assert shared_surface.data.clear_count == 0


def test_cache_invalidation_clears_exclusive_paper_surface_without_particles(
        monkeypatch):
    handlers, bpy = _load_handlers(monkeypatch)
    surface = _SurfaceObject()
    scene = _scene(
        cache_dir="C:/cache",
        state="COMPLETE",
        surface=surface,
        create_surface=True,
        surface_method="PAPER_MCF",
    )
    bpy.data.scenes = [scene]
    monkeypatch.setattr(handlers.cache, "read_meta", lambda path: None)

    assert handlers.reconcile_scene_cache(scene) is False
    assert surface.data.clear_count == 1
    assert surface.data.update_count == 1
    assert surface.tag_count == 1
    assert scene.stflip.bake_state == "FAILED"


def test_clear_scene_output_empties_bound_particle_mesh(monkeypatch):
    handlers, _bpy = _load_handlers(monkeypatch)
    particle = _particle("Renamed Particle Output")
    scene = _scene(cache_dir="C:/cache", particle=particle)

    assert handlers.clear_scene_output(scene) is True
    assert particle.data.clear_count == 1
    assert particle.data.update_count == 1
    assert particle.tag_count == 1


def test_clear_scene_output_preserves_mesh_shared_with_another_scene(
        monkeypatch):
    handlers, bpy = _load_handlers(monkeypatch)
    particle = _particle("Linked Particle Output")
    owner = _scene(cache_dir="C:/owner", particle=particle, pointer=1)
    duplicate = _scene(cache_dir="C:/duplicate", particle=particle, pointer=2)
    bpy.data.scenes = [owner, duplicate]

    assert handlers.clear_scene_output(duplicate) is False
    assert particle.data.clear_count == 0


def test_clear_scene_output_does_not_touch_global_foreign_legacy_object(
        monkeypatch):
    handlers, bpy = _load_handlers(monkeypatch)
    foreign = _particle()
    bpy.data.objects[foreign.name] = foreign
    scene = _scene(cache_dir="C:/cache")

    assert handlers.clear_scene_output(scene) is False
    assert foreign.data.clear_count == 0


def test_frame_application_does_not_drive_another_scenes_named_output(
        monkeypatch):
    handlers, bpy = _load_handlers(monkeypatch)
    foreign = _particle()
    bpy.data.objects[foreign.name] = foreign
    owner = "0123456789abcdef0123456789abcdef"
    scene = _scene(cache_dir="C:/cache", cache_id=owner)
    bpy.data.scenes = [scene]
    empty = np.empty((0, 3), dtype=np.float32)
    monkeypatch.setattr(handlers.cache, "read_meta", lambda path: {
        "frame_start": 1,
        "frame_end_baked": 1,
        handlers.cache.OWNER_KEY: owner,
    })
    monkeypatch.setattr(
        handlers.cache, "read_frame", lambda path, frame: (empty, empty))

    assert handlers._apply_frame(scene, 1) is False
    assert foreign.data.clear_count == 0


def test_frame_application_refuses_copied_scene_shared_mesh(monkeypatch):
    handlers, bpy = _load_handlers(monkeypatch)
    shared = _particle("Copied Particle Output")
    owner_id = "0123456789abcdef0123456789abcdef"
    owner = _scene(
        cache_dir="C:/cache", cache_id=owner_id, particle=shared, pointer=1)
    duplicate = _scene(
        cache_dir="C:/other", cache_id="fedcba9876543210fedcba9876543210",
        particle=shared, pointer=2)
    bpy.data.scenes = [owner, duplicate]
    empty = np.empty((0, 3), dtype=np.float32)
    monkeypatch.setattr(handlers.cache, "read_meta", lambda path: {
        "frame_start": 1,
        "frame_end_baked": 1,
        handlers.cache.OWNER_KEY: owner_id,
    })
    monkeypatch.setattr(
        handlers.cache, "read_frame", lambda path, frame: (empty, empty))

    assert handlers._apply_frame(owner, 1) is False
    assert shared.data.clear_count == 0


def test_reconcile_marks_completed_bake_with_missing_metadata_and_clears_output(
        monkeypatch, tmp_path):
    handlers, bpy = _load_handlers(monkeypatch)
    particle = _particle()
    scene = _scene(
        cache_dir=str(tmp_path),
        state="COMPLETE",
        particle=particle,
    )
    bpy.data.scenes = [scene]

    assert handlers.reconcile_scene_cache(scene) is False
    assert scene.stflip.bake_status == (
        "Cache unavailable: metadata is missing; rebake required")
    assert scene.stflip.bake_state == "FAILED"
    assert scene.stflip.bake_error == scene.stflip.bake_status
    assert scene.stflip.bake_progress == 0.0
    assert particle.data.clear_count == 1


def test_reconcile_marks_missing_committed_frame_and_clears_output(
        monkeypatch, tmp_path):
    handlers, bpy = _load_handlers(monkeypatch)
    particle = _particle()
    owner = "0123456789abcdef0123456789abcdef"
    scene = _scene(
        cache_dir=str(tmp_path),
        cache_id=owner,
        status="Bake complete (1 frames) on CUDA",
        particle=particle,
    )
    bpy.data.scenes = [scene]
    core_cache.write_meta(str(tmp_path), {
        "frame_start": 1,
        "frame_end_baked": 1,
        core_cache.OWNER_KEY: owner,
    })

    assert handlers.reconcile_scene_cache(scene) is False
    assert "frame 1 is missing" in scene.stflip.bake_status
    assert particle.data.clear_count == 1


def test_reconcile_detects_a_hole_in_completed_frame_sequence(
        monkeypatch, tmp_path):
    handlers, bpy = _load_handlers(monkeypatch)
    particle = _particle()
    owner = "0123456789abcdef0123456789abcdef"
    scene = _scene(
        cache_dir=str(tmp_path),
        cache_id=owner,
        state="COMPLETE",
        particle=particle,
    )
    bpy.data.scenes = [scene]
    empty = np.empty((0, 3), dtype=np.float32)
    core_cache.write_frame(str(tmp_path), 1, empty, empty)
    core_cache.write_frame(str(tmp_path), 3, empty, empty)
    core_cache.write_meta(str(tmp_path), {
        "frame_start": 1,
        "frame_end": 3,
        "frame_end_baked": 3,
        core_cache.OWNER_KEY: owner,
    })

    assert handlers.reconcile_scene_cache(scene) is False
    assert "frame 2 is missing" in scene.stflip.bake_status
    assert particle.data.clear_count == 1


def test_reconcile_rejects_foreign_cache_even_without_completed_status(
        monkeypatch, tmp_path):
    handlers, bpy = _load_handlers(monkeypatch)
    particle = _particle()
    scene = _scene(
        cache_dir=str(tmp_path),
        cache_id="0123456789abcdef0123456789abcdef",
        particle=particle,
    )
    bpy.data.scenes = [scene]
    core_cache.write_meta(str(tmp_path), {
        "frame_start": 1,
        "frame_end_baked": 1,
        core_cache.OWNER_KEY: "fedcba9876543210fedcba9876543210",
    })

    assert handlers.reconcile_scene_cache(scene) is False
    assert "belongs to another scene" in scene.stflip.bake_status
    assert particle.data.clear_count == 1


def test_reconcile_accepts_ownerless_legacy_metadata_without_clearing(
        monkeypatch, tmp_path):
    handlers, bpy = _load_handlers(monkeypatch)
    particle = _particle()
    scene = _scene(
        cache_dir=str(tmp_path),
        status="Bake complete (1 frames) on CPU",
        particle=particle,
    )
    bpy.data.scenes = [scene]
    empty = np.empty((0, 3), dtype=np.float32)
    core_cache.write_frame(str(tmp_path), 1, empty, empty)
    core_cache.write_meta(
        str(tmp_path), {"frame_start": 1, "frame_end_baked": 1})
    applied = []
    monkeypatch.setattr(
        handlers, "_apply_frame", lambda target, frame: applied.append(frame))

    assert handlers.reconcile_scene_cache(scene) is True
    assert applied == [1]
    assert scene.stflip.bake_status == "Bake complete (1 frames) on CPU"
    assert scene.stflip.bake_state == "COMPLETE"
    assert scene.stflip.bake_error == ""
    assert scene.stflip.bake_progress == 1.0
    assert particle.data.clear_count == 0


def test_reconcile_rejects_completed_state_with_partial_committed_range(
        monkeypatch, tmp_path):
    handlers, bpy = _load_handlers(monkeypatch)
    particle = _particle()
    owner = "0123456789abcdef0123456789abcdef"
    scene = _scene(
        cache_dir=str(tmp_path),
        cache_id=owner,
        state="COMPLETE",
        particle=particle,
    )
    bpy.data.scenes = [scene]
    empty = np.empty((0, 3), dtype=np.float32)
    core_cache.write_frame(str(tmp_path), 1, empty, empty)
    core_cache.write_meta(str(tmp_path), {
        "frame_start": 1,
        "frame_end": 4,
        "frame_end_baked": 1,
        core_cache.OWNER_KEY: owner,
    })

    assert handlers.reconcile_scene_cache(scene) is False
    assert "incomplete frame range" in scene.stflip.bake_status
    assert scene.stflip.bake_state == "FAILED"
    assert particle.data.clear_count == 1


def test_reconcile_preserves_partial_cancelled_cache_and_updates_progress(
        monkeypatch, tmp_path):
    handlers, bpy = _load_handlers(monkeypatch)
    owner = "0123456789abcdef0123456789abcdef"
    scene = _scene(
        cache_dir=str(tmp_path),
        cache_id=owner,
        status="Bake cancelled after 2 cached frames",
        state="CANCELLED",
    )
    scene.frame_current = 2
    bpy.data.scenes = [scene]
    empty = np.empty((0, 3), dtype=np.float32)
    core_cache.write_frame(str(tmp_path), 1, empty, empty)
    core_cache.write_frame(str(tmp_path), 2, empty, empty)
    core_cache.write_meta(str(tmp_path), {
        "frame_start": 1,
        "frame_end": 4,
        "frame_end_baked": 2,
        core_cache.OWNER_KEY: owner,
    })
    monkeypatch.setattr(handlers, "_apply_frame", lambda scene, frame: True)

    assert handlers.reconcile_scene_cache(scene) is True
    assert scene.stflip.bake_state == "CANCELLED"
    assert scene.stflip.bake_progress == 0.5


def test_reconcile_turns_orphaned_running_bake_into_resumable_cancelled_state(
        monkeypatch, tmp_path):
    handlers, bpy = _load_handlers(monkeypatch)
    owner = "0123456789abcdef0123456789abcdef"
    scene = _scene(
        cache_dir=str(tmp_path),
        cache_id=owner,
        status="Baking frame 2/4",
        state="RUNNING",
    )
    bpy.data.scenes = [scene]
    empty = np.empty((0, 3), dtype=np.float32)
    core_cache.write_frame(str(tmp_path), 1, empty, empty)
    core_cache.write_checkpoint(str(tmp_path), 1, {
        "pos": empty,
        "vel": empty,
        "dt_resid": np.empty((0,), dtype=np.float32),
        "time": 0.0,
        "dt_prev": 1.0 / 24.0,
        "rng_state": np.random.default_rng(7).bit_generator.state,
        "outflow_removed_total": 0,
        "volume_outflow_removed_total": 0,
        "pressure_outflow_removed_total": 0,
    }, fingerprint="f" * 64)
    core_cache.write_meta(str(tmp_path), {
        "frame_start": 1,
        "frame_end": 4,
        "frame_end_baked": 1,
        core_cache.OWNER_KEY: owner,
        "checkpoint": {
            "schema": core_cache.CHECKPOINT_SCHEMA,
            "version": core_cache.CHECKPOINT_VERSION,
            "fingerprint": "f" * 64,
            "latest_frame": 1,
            "state": "RUNNING",
        },
        "bake_lifecycle": {
            "state": "RUNNING",
            "last_committed_frame": 1,
            "error": "",
        },
    })
    monkeypatch.setattr(handlers, "_apply_frame", lambda scene, frame: True)

    assert handlers.reconcile_scene_cache(scene) is True
    assert scene.stflip.bake_state == "CANCELLED"
    assert "Resume Bake" in scene.stflip.bake_status
    assert scene.stflip.bake_progress == 0.25
    metadata = core_cache.read_meta(str(tmp_path))
    assert metadata["checkpoint"]["state"] == "CANCELLED"
    assert metadata["bake_lifecycle"] == {
        "state": "CANCELLED",
        "last_committed_frame": 1,
        "error": "",
    }


def test_reconcile_stale_complete_ui_recovers_extended_running_checkpoint(
        monkeypatch, tmp_path):
    handlers, bpy = _load_handlers(monkeypatch)
    owner = "0123456789abcdef0123456789abcdef"
    fingerprint = "a" * 64
    scene = _scene(
        cache_dir=str(tmp_path),
        cache_id=owner,
        status="Bake complete (1 frames) on CPU",
        state="COMPLETE",
    )
    bpy.data.scenes = [scene]
    empty = np.empty((0, 3), dtype=np.float32)
    core_cache.write_frame(str(tmp_path), 1, empty, empty)
    core_cache.write_checkpoint(str(tmp_path), 1, {
        "pos": empty,
        "vel": empty,
        "dt_resid": np.empty((0,), dtype=np.float32),
        "time": 0.0,
        "dt_prev": 1.0 / 24.0,
        "rng_state": np.random.default_rng(9).bit_generator.state,
        "outflow_removed_total": 0,
        "volume_outflow_removed_total": 0,
        "pressure_outflow_removed_total": 0,
    }, fingerprint=fingerprint)
    core_cache.write_meta(str(tmp_path), {
        "frame_start": 1,
        "frame_end": 4,
        "frame_end_baked": 1,
        core_cache.OWNER_KEY: owner,
        "checkpoint": {
            "schema": core_cache.CHECKPOINT_SCHEMA,
            "version": core_cache.CHECKPOINT_VERSION,
            "fingerprint": fingerprint,
            "latest_frame": 1,
            "state": "RUNNING",
        },
        "bake_lifecycle": {
            "state": "RUNNING",
            "last_committed_frame": 1,
            "error": "",
        },
    })
    monkeypatch.setattr(handlers, "_apply_frame", lambda scene, frame: True)

    assert handlers.reconcile_scene_cache(scene) is True
    assert scene.stflip.bake_state == "CANCELLED"
    assert "Resume Bake" in scene.stflip.bake_status
    assert "rebake required" not in scene.stflip.bake_status


def test_reconcile_cancelled_cache_with_missing_checkpoint_requires_rebake(
        monkeypatch, tmp_path):
    handlers, bpy = _load_handlers(monkeypatch)
    owner = "0123456789abcdef0123456789abcdef"
    fingerprint = "b" * 64
    scene = _scene(
        cache_dir=str(tmp_path),
        cache_id=owner,
        status="Bake cancelled after 1 cached frame",
        state="CANCELLED",
    )
    bpy.data.scenes = [scene]
    empty = np.empty((0, 3), dtype=np.float32)
    core_cache.write_frame(str(tmp_path), 1, empty, empty)
    core_cache.write_meta(str(tmp_path), {
        "frame_start": 1,
        "frame_end": 4,
        "frame_end_baked": 1,
        core_cache.OWNER_KEY: owner,
        "checkpoint": {
            "schema": core_cache.CHECKPOINT_SCHEMA,
            "version": core_cache.CHECKPOINT_VERSION,
            "fingerprint": fingerprint,
            "latest_frame": 1,
            "state": "CANCELLED",
        },
        "bake_lifecycle": {
            "state": "CANCELLED",
            "last_committed_frame": 1,
            "error": "",
        },
    })
    monkeypatch.setattr(handlers, "_apply_frame", lambda scene, frame: True)

    assert handlers.reconcile_scene_cache(scene) is True
    assert scene.stflip.bake_state == "FAILED"
    assert "rebaking is required" in scene.stflip.bake_status
    assert Path(core_cache.frame_path(str(tmp_path), 1)).is_file()
    metadata = core_cache.read_meta(str(tmp_path))
    assert metadata["checkpoint"]["state"] == "FAILED"
    assert metadata["bake_lifecycle"]["state"] == "FAILED"


def test_unregister_does_not_clear_valid_output_or_saved_cache(
        monkeypatch, tmp_path):
    handlers, bpy = _load_handlers(monkeypatch)
    particle = _particle()
    scene = _scene(cache_dir=str(tmp_path), particle=particle)
    bpy.data.scenes = [scene]
    marker = tmp_path / "keep.txt"
    marker.write_text("valid cache", encoding="utf-8")
    monkeypatch.setattr(handlers.atexit, "register", lambda callback: None)
    monkeypatch.setattr(handlers.atexit, "unregister", lambda callback: None)

    handlers.ensure_registered()
    handlers.unregister()

    assert marker.read_text(encoding="utf-8") == "valid cache"
    assert particle.data.clear_count == 0


def _load_mesher(monkeypatch, bound_particle, canonical_particle):
    bpy = types.ModuleType("bpy")
    for obj in (bound_particle, canonical_particle):
        if obj is None:
            continue
        if not hasattr(obj, "data"):
            obj.data = types.SimpleNamespace(
                name=f"{obj.name} Mesh", library=None)
        if not hasattr(obj, "library"):
            obj.library = None

    class Objects(list):
        def get(self, name):
            return next((obj for obj in self if obj.name == name), None)

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
        objects=Objects([
            obj for obj in (bound_particle, canonical_particle)
            if obj is not None
        ]),
        scenes=[scene],
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
    explicit = types.SimpleNamespace(
        name="Explicit Particles",
        type="MESH",
        data=types.SimpleNamespace(name="Explicit Mesh", library=None),
        library=None,
    )
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


def test_copied_scene_output_object_is_rejected_for_particle_and_surface(
        monkeypatch):
    shared = types.SimpleNamespace(name="Copied Output", type="MESH")
    mesher = _load_mesher(monkeypatch, shared, None)
    scene = mesher.bpy.context.scene
    other = types.SimpleNamespace(
        objects={shared.name: shared},
        collection=types.SimpleNamespace(objects={}, children=[]),
    )
    mesher.bpy.data.scenes.append(other)

    assert mesher.output_is_exclusive(scene, shared) is False
    for property_name, fallback in (
        ("particle_object", "STFLIP Particles"),
        ("surface_object", "STFLIP Liquid Surface"),
    ):
        setattr(scene.stflip, property_name, shared)
        assert mesher._mesh_output(
            shared, property_name, fallback) is None


def test_distinct_scene_object_sharing_output_mesh_is_rejected(monkeypatch):
    mesh = types.SimpleNamespace(name="Shared Mesh", library=None)
    bound = types.SimpleNamespace(
        name="Bound Output", type="MESH", data=mesh, library=None)
    other_object = types.SimpleNamespace(
        name="Other Output", type="MESH", data=mesh, library=None)
    mesher = _load_mesher(monkeypatch, bound, None)
    scene = mesher.bpy.context.scene
    other = types.SimpleNamespace(
        objects={other_object.name: other_object},
        collection=types.SimpleNamespace(objects={}, children=[]),
    )
    mesher.bpy.data.scenes.append(other)
    mesher.bpy.data.objects.append(other_object)

    assert mesher.output_is_exclusive(scene, bound) is False
    assert mesher._mesh_output(
        bound, "particle_object", "STFLIP Particles") is None


def test_surface_smoothing_is_explicit_blender_modifier_and_updates_in_place(
        monkeypatch):
    mesher = _load_mesher(monkeypatch, None, None)

    class Modifiers(list):
        def get(self, name):
            return next((item for item in self if item.name == name), None)

        def new(self, name, modifier_type):
            item = types.SimpleNamespace(
                name=name,
                type=modifier_type,
                iterations=0,
                lambda_factor=0.0,
                lambda_border=0.0,
                show_viewport=False,
                show_render=False,
            )
            self.append(item)
            return item

        def move(self, source, destination):
            self.insert(destination, self.pop(source))

    updates = []
    obj = types.SimpleNamespace(
        name="Surface",
        type="MESH",
        modifiers=Modifiers(),
        update_tag=lambda: updates.append(True),
    )

    first = mesher.configure_surface_smoothing(obj, True, 3, 0.25)
    second = mesher.configure_surface_smoothing(obj, False, 7, -0.1)

    assert first is second
    assert len(obj.modifiers) == 1
    assert second.type == "LAPLACIANSMOOTH"
    assert second.iterations == 7
    assert second.lambda_factor == -0.1
    assert second.lambda_border == -0.1
    assert second.show_viewport is False
    assert second.show_render is False
    assert len(updates) == 2


def test_preview_node_group_pins_density_threshold_and_adaptivity(monkeypatch):
    mesher = _load_mesher(monkeypatch, None, None)

    class Socket:
        def __init__(self, name, *, in_out="INPUT"):
            self.name = name
            self.identifier = f"id_{name}"
            self.item_type = "SOCKET"
            self.in_out = in_out
            self.default_value = None
            self.min_value = None

    class Interface:
        def __init__(self):
            self.items_tree = []

        def new_socket(self, *, name, in_out, socket_type):
            socket = Socket(name, in_out=in_out)
            socket.socket_type = socket_type
            self.items_tree.append(socket)
            return socket

    node_sockets = {
        "NodeGroupInput": ((), (
            "Geometry", "Points Object", "Radius", "Voxel Size", "Material")),
        "NodeGroupOutput": (("Geometry",), ()),
        "GeometryNodeObjectInfo": (("Object",), ("Geometry",)),
        "GeometryNodeMeshToPoints": (("Mesh", "Radius"), ("Points",)),
        "GeometryNodePointsToVolume": ((
            "Resolution Mode", "Points", "Radius", "Voxel Size", "Density"),
            ("Volume",)),
        "GeometryNodeVolumeToMesh": ((
            "Resolution Mode", "Volume", "Threshold", "Adaptivity"),
            ("Mesh",)),
        "GeometryNodeSetShadeSmooth": (("Geometry",), ("Geometry",)),
        "GeometryNodeSetMaterial": (("Geometry", "Material"), ("Geometry",)),
    }

    class Nodes(list):
        def new(self, node_type):
            inputs, outputs = node_sockets[node_type]
            node = types.SimpleNamespace(
                bl_idname=node_type,
                inputs={name: Socket(name) for name in inputs},
                outputs={name: Socket(name, in_out="OUTPUT")
                         for name in outputs},
            )
            self.append(node)
            return node

    class Group(dict):
        def __init__(self):
            super().__init__()
            self.interface = Interface()
            self.nodes = Nodes()
            self.links = types.SimpleNamespace(new=lambda *args: None)
            self.is_modifier = False

    group = mesher._populate_node_group(Group())
    points_to_volume = next(
        node for node in group.nodes
        if node.bl_idname == "GeometryNodePointsToVolume")
    volume_to_mesh = next(
        node for node in group.nodes
        if node.bl_idname == "GeometryNodeVolumeToMesh")

    assert mesher.GROUP_SCHEMA_VERSION == 3
    assert group[mesher.GROUP_SCHEMA_KEY] == 3
    assert points_to_volume.inputs["Density"].default_value == 1.0
    assert volume_to_mesh.inputs["Threshold"].default_value == 0.5
    assert volume_to_mesh.inputs["Adaptivity"].default_value == 0.0


def test_openvdb_density_meshing_uses_world_transform_and_explicit_controls(
        monkeypatch):
    mesher = _load_mesher(monkeypatch, None, None)
    calls = {}

    class Transform:
        def postTranslate(self, origin):
            calls["origin"] = origin

    class Grid:
        def copyFromArray(self, values):
            calls["density"] = values.copy()

        def convertToPolygons(self, **kwargs):
            calls["polygon_kwargs"] = kwargs
            return (
                np.array(((1.0, 2.0, 3.0), (2.0, 2.0, 3.0),
                          (1.0, 3.0, 3.0), (1.0, 2.0, 4.0))),
                np.array(((0, 1, 2),), dtype=np.uint32),
                np.array(((0, 1, 3, 2),), dtype=np.uint32),
            )

    grid = Grid()
    transform = Transform()
    openvdb = types.ModuleType("openvdb")
    openvdb.FloatGrid = lambda: grid

    def create_transform(*, voxelSize):
        calls["voxel_size"] = voxelSize
        return transform

    openvdb.createLinearTransform = create_transform
    monkeypatch.setitem(sys.modules, "openvdb", openvdb)
    density = np.zeros((3, 4, 5), dtype=np.float64)
    density[1, 2, 3] = 1.0

    vertices, triangles, quads = mesher.density_field_to_polygons(
        density,
        origin=(10.0, 20.0, 30.0),
        voxel_size=0.25,
        isovalue=0.6,
        adaptivity=0.125,
    )

    assert calls["density"].dtype == np.float32
    assert calls["density"].flags.c_contiguous
    assert calls["voxel_size"] == 0.25
    assert calls["origin"] == (10.0, 20.0, 30.0)
    assert calls["polygon_kwargs"] == {
        "isovalue": 0.6,
        "adaptivity": 0.125,
    }
    assert grid.transform is transform
    assert vertices.shape == (4, 3)
    assert vertices.dtype == np.float32
    assert triangles.shape == (1, 3)
    assert quads.shape == (1, 4)


def test_paper_surface_writes_plain_mesh_disables_gn_and_restores_preview(
        monkeypatch):
    mesher = _load_mesher(monkeypatch, None, None)

    class Modifiers(list):
        def get(self, name):
            return next((item for item in self if item.name == name), None)

        def new(self, name, modifier_type):
            item = Modifier(name, modifier_type)
            self.append(item)
            return item

    class Modifier(dict):
        def __init__(self, name, modifier_type):
            super().__init__()
            self.name = name
            self.type = modifier_type
            self.show_viewport = True
            self.show_render = True
            self.node_group = None

    class Mesh:
        def __init__(self):
            self.materials = []
            self.polygons = []
            self.vertices_written = None
            self.faces_written = None
            self.clear_count = 0
            self.update_count = 0

        def clear_geometry(self):
            self.clear_count += 1

        def from_pydata(self, vertices, edges, faces):
            assert edges == []
            self.vertices_written = vertices
            self.faces_written = faces
            self.polygons = [types.SimpleNamespace(material_index=-1)
                             for _face in faces]

        def update(self):
            self.update_count += 1

    class Object(dict):
        def __init__(self):
            super().__init__()
            self.name = "Paper Surface"
            self.type = "MESH"
            self.data = Mesh()
            self.modifiers = Modifiers()
            self.update_count = 0

        def update_tag(self):
            self.update_count += 1

    obj = Object()
    surface_modifier = obj.modifiers.new(mesher.SURFACE_MODIFIER, "NODES")
    smooth_modifier = obj.modifiers.new(
        mesher.SMOOTH_MODIFIER, "LAPLACIANSMOOTH")
    water = object()
    monkeypatch.setattr(mesher, "ensure_water_material", lambda: water)
    monkeypatch.setattr(mesher, "_ensure_surface_output", lambda existing: obj)

    vertices = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                         (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    result = mesher.ensure_paper_surface_object(
        vertices,
        np.array(((0, 1, 2),), dtype=np.int32),
        np.array(((0, 1, 3, 2),), dtype=np.int32),
        existing_obj=obj,
    )

    assert result is obj
    assert obj.data.clear_count == 1
    assert obj.data.faces_written == [(0, 1, 2), (0, 1, 3, 2)]
    assert obj.data.materials == [water]
    assert all(poly.material_index == 0 for poly in obj.data.polygons)
    assert surface_modifier.show_viewport is False
    assert surface_modifier.show_render is False
    assert smooth_modifier.show_viewport is False
    assert smooth_modifier.show_render is False
    assert obj["stflip_surface_method"] == "PAPER_MCF"

    interface = types.SimpleNamespace(items_tree=[
        types.SimpleNamespace(
            name=name,
            item_type="SOCKET",
            in_out="INPUT",
            identifier=f"id_{name}",
        )
        for name in ("Points Object", "Radius", "Voxel Size", "Material")
    ])
    node_group = types.SimpleNamespace(interface=interface)
    monkeypatch.setattr(mesher, "build_node_group", lambda: node_group)
    particle = object()

    restored = mesher.restore_preview_surface(
        particle, 0.2, 0.5, 0.5, existing_obj=obj)

    assert restored is obj
    assert surface_modifier.show_viewport is True
    assert surface_modifier.show_render is True
    assert surface_modifier.node_group is node_group
    assert surface_modifier["id_Points Object"] is particle
    assert surface_modifier["id_Radius"] == 0.1
    assert surface_modifier["id_Voxel Size"] == 0.1
    assert surface_modifier["id_Material"] is water
    assert obj["stflip_surface_method"] == "FAST_PREVIEW"


def _load_surface_properties(monkeypatch):
    root = "_stflip_surface_properties_test"
    for name in (root, f"{root}.addon", f"{root}.stflip"):
        monkeypatch.setitem(sys.modules, name, _package(name))

    bpy = types.ModuleType("bpy")
    bpy.types = types.SimpleNamespace(
        PropertyGroup=object,
        Object=type("Object", (), {}),
        Scene=type("Scene", (), {}),
    )
    bpy.utils = types.SimpleNamespace(
        register_class=lambda cls: None,
        unregister_class=lambda cls: None,
    )
    props = types.ModuleType("bpy.props")

    def property_factory(kind):
        def define(**kwargs):
            return {"kind": kind, **kwargs}
        return define

    for name in (
        "BoolProperty", "EnumProperty", "FloatProperty",
        "FloatVectorProperty", "IntProperty", "PointerProperty",
        "StringProperty",
    ):
        setattr(props, name, property_factory(name))
    bpy.props = props
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", props)
    experiments = types.ModuleType(f"{root}.stflip.experiments")
    experiments.PROFILE_ENUM_ITEMS = [("CUSTOM", "Custom", "")]
    monkeypatch.setitem(
        sys.modules, f"{root}.stflip.experiments", experiments)
    return _load_source(
        monkeypatch,
        f"{root}.addon.properties",
        "addon/properties.py",
    )


def test_paper_surface_properties_have_safe_defaults_and_ranges(monkeypatch):
    properties = _load_surface_properties(monkeypatch)
    settings = properties.STFLIPSettings.__annotations__

    method = settings["surface_method"]
    assert method["default"] == "FAST_PREVIEW"
    assert [item[0] for item in method["items"]] == [
        "FAST_PREVIEW", "PAPER_MCF"]
    iterations = settings["paper_mcf_iterations"]
    assert (iterations["default"], iterations["min"], iterations["max"]) == (
        30, 1, 100)
    adaptivity = settings["paper_mesh_adaptivity"]
    assert (adaptivity["default"], adaptivity["min"], adaptivity["max"]) == (
        0.0, 0.0, 1.0)
    voxel_cap = settings["paper_max_reconstruction_voxels"]
    assert voxel_cap["default"] == 16_777_216
    assert voxel_cap["min"] < voxel_cap["default"] < voxel_cap["max"]


def _load_surface_panels(monkeypatch, *, rebuilding=False):
    root = "_stflip_surface_panels_test"
    for name in (root, f"{root}.addon"):
        monkeypatch.setitem(sys.modules, name, _package(name))
    bpy = types.ModuleType("bpy")
    bpy.types = types.SimpleNamespace(Panel=object)
    bpy.utils = types.SimpleNamespace(
        register_class=lambda cls: None,
        unregister_class=lambda cls: None,
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    operators = types.ModuleType(f"{root}.addon.operators")
    operators.current_cuda_diagnostics = lambda: {
        "available": False,
        "device": "",
        "free_bytes": None,
        "total_bytes": None,
        "error": "",
    }
    operators.surface_rebuild_running = lambda: rebuilding
    monkeypatch.setitem(sys.modules, f"{root}.addon.operators", operators)
    return _load_source(
        monkeypatch,
        f"{root}.addon.panels",
        "addon/panels.py",
    )


class _RecordingLayout:
    def __init__(self, records=None, states=None, *, parent_enabled=True):
        self.records = records if records is not None else []
        self.states = states if states is not None else []
        self._parent_enabled = bool(parent_enabled)
        self.enabled = True

    @property
    def effective_enabled(self):
        return self._parent_enabled and bool(self.enabled)

    def column(self, **kwargs):
        return _RecordingLayout(
            self.records,
            self.states,
            parent_enabled=self.effective_enabled,
        )

    def row(self, **kwargs):
        return _RecordingLayout(
            self.records,
            self.states,
            parent_enabled=self.effective_enabled,
        )

    def prop(self, _settings, name, **kwargs):
        self.records.append(("prop", name))
        self.states.append(("prop", name, self.effective_enabled))

    def label(self, *, text, **kwargs):
        self.records.append(("label", text))
        self.states.append(("label", text, self.effective_enabled))

    def operator(self, name, **kwargs):
        self.records.append(("operator", name))
        self.states.append(("operator", name, self.effective_enabled))

    def separator(self):
        self.records.append(("separator", ""))


def test_object_panel_allows_empty_force_guides(monkeypatch):
    panels = _load_surface_panels(monkeypatch)
    force = types.SimpleNamespace(
        role="FORCE",
        force_type="VORTEX",
        force_strength=4.0,
        force_radius=2.0,
        force_scale=0.5,
    )
    context = types.SimpleNamespace(
        active_object=types.SimpleNamespace(type="EMPTY", stflip=force),
        scene=types.SimpleNamespace(
            stflip=types.SimpleNamespace(bake_state="IDLE")),
    )
    panel = panels.STFLIP_PT_object()
    layout = _RecordingLayout()
    panel.layout = layout

    panel.draw(context)

    props = {value for kind, value in layout.records if kind == "prop"}
    labels = [value for kind, value in layout.records if kind == "label"]
    assert {"role", "force_type", "force_strength", "force_radius"} <= props
    assert not any("Select a mesh" in label for label in labels)


def test_object_panel_warns_when_empty_has_voxelized_role(monkeypatch):
    panels = _load_surface_panels(monkeypatch)
    context = types.SimpleNamespace(
        active_object=types.SimpleNamespace(
            type="EMPTY",
            stflip=types.SimpleNamespace(role="LIQUID"),
        ),
        scene=types.SimpleNamespace(
            stflip=types.SimpleNamespace(bake_state="IDLE")),
    )
    panel = panels.STFLIP_PT_object()
    layout = _RecordingLayout()
    panel.layout = layout

    panel.draw(context)

    props = {value for kind, value in layout.records if kind == "prop"}
    labels = [value for kind, value in layout.records if kind == "label"]
    assert props == {"role"}
    assert "Empty objects support the Force Field role only." in labels


def test_surface_panel_separates_preview_and_paper_controls(monkeypatch):
    panels = _load_surface_panels(monkeypatch)
    settings = types.SimpleNamespace(
        bake_state="IDLE",
        create_surface=True,
        surface_method="PAPER_MCF",
        surface_smoothing=True,
    )
    context = types.SimpleNamespace(
        scene=types.SimpleNamespace(stflip=settings))
    panel = panels.STFLIP_PT_display()
    paper_layout = _RecordingLayout()
    panel.layout = paper_layout
    panel.draw(context)

    paper_props = {
        value for kind, value in paper_layout.records if kind == "prop"}
    paper_labels = [
        value for kind, value in paper_layout.records if kind == "label"]
    assert {
        "paper_mcf_iterations",
        "paper_mesh_adaptivity",
        "paper_max_reconstruction_voxels",
    } <= paper_props
    assert "surface_smoothing" not in paper_props
    assert "Paper constants: radius 0.5Δx, voxel 0.5Δx" in paper_labels
    assert "Gaussian σ = 2Δx" in paper_labels
    assert "Feature mask: θ = 2, ζ = 5" in paper_labels
    assert "Dense reconstruction uses NumPy or CuPy." in paper_labels
    assert "OpenVDB polygonization uses CPU/RAM only." in paper_labels

    settings.surface_method = "FAST_PREVIEW"
    preview_layout = _RecordingLayout()
    panel.layout = preview_layout
    panel.draw(context)
    preview_props = {
        value for kind, value in preview_layout.records if kind == "prop"}
    assert {
        "particle_radius",
        "surface_voxel",
        "surface_smoothing",
        "surface_smoothing_iterations",
        "surface_smoothing_factor",
    } <= preview_props
    assert "paper_mcf_iterations" not in preview_props


def test_surface_panel_freezes_rebuild_controls_but_leaves_cancel_enabled(
        monkeypatch):
    panels = _load_surface_panels(monkeypatch, rebuilding=True)
    settings = types.SimpleNamespace(
        bake_state="COMPLETE",
        create_surface=True,
        surface_method="PAPER_MCF",
        surface_smoothing=True,
    )
    context = types.SimpleNamespace(
        scene=types.SimpleNamespace(stflip=settings))
    panel = panels.STFLIP_PT_display()
    layout = _RecordingLayout()
    panel.layout = layout

    panel.draw(context)

    enabled_by_control = {
        (kind, name): enabled for kind, name, enabled in layout.states}
    assert enabled_by_control[("prop", "create_surface")] is False
    assert enabled_by_control[("prop", "surface_method")] is False
    assert enabled_by_control[("prop", "paper_mcf_iterations")] is False
    assert enabled_by_control[
        ("operator", "stflip.rebuild_paper_surfaces")] is False
    assert enabled_by_control[("operator", "stflip.refresh_surface")] is False
    assert enabled_by_control[
        ("operator", "stflip.cancel_surface_rebuild")] is True


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
