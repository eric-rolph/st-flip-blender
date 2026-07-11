"""Blender integration regressions that run without importing real ``bpy``."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np

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
           status="", state="IDLE", particle=None):
    settings = types.SimpleNamespace(
        cache_dir=cache_dir,
        cache_id=cache_id,
        bake_status=status,
        bake_state=state,
        bake_error="",
        particle_object=particle,
    )
    objects = {} if particle is None else {particle.name: particle}
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
