"""Frame-change handler: stream cached particle frames into the point cloud."""

import atexit
import os
import shutil
import tempfile
import uuid
from numbers import Integral

import bpy
import numpy as np
from bpy.app.handlers import persistent

from ..stflip import cache
from . import mesher


_ATEXIT_REGISTERED = False
_DEFAULT_CACHE_DIR = "//stflip_cache"
_READ_METADATA = object()
_FALLBACK_CACHE_IDS: dict[int, str] = {}
_CACHE_ID_OWNERS: dict[str, int] = {}


def _scene_key(scene) -> int:
    try:
        return int(scene.as_pointer())
    except (AttributeError, TypeError, ValueError):
        return id(scene)


def _canonical_cache_id(value) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        return uuid.UUID(value.strip()).hex
    except (AttributeError, ValueError):
        return ""


def _read_settings_cache_id(settings) -> str:
    try:
        value = getattr(settings, "cache_id", "")
    except (AttributeError, RuntimeError, TypeError):
        value = ""
    if not value:
        try:
            value = settings.get("cache_id", "")
        except (AttributeError, RuntimeError, TypeError):
            value = ""
    return _canonical_cache_id(value)


def _write_settings_cache_id(scene, value: str) -> None:
    settings = scene.stflip
    try:
        setattr(settings, "cache_id", value)
        if _read_settings_cache_id(settings) == value:
            return
    except (AttributeError, RuntimeError, TypeError):
        pass
    try:
        settings["cache_id"] = value
        if _read_settings_cache_id(settings) == value:
            return
    except (AttributeError, KeyError, RuntimeError, TypeError):
        pass
    # Linked/legacy test doubles may expose neither an RNA property nor ID
    # property writes. Keep resolution stable for the rest of this session.
    _FALLBACK_CACHE_IDS[_scene_key(scene)] = value


def _scene_cache_id(scene) -> str:
    settings = getattr(scene, "stflip", None)
    if settings is None:
        return ""
    # A session fallback must override a copied but read-only RNA value.
    return (_FALLBACK_CACHE_IDS.get(_scene_key(scene), "")
            or _read_settings_cache_id(settings))


def _scenes():
    try:
        return list(bpy.data.scenes)
    except (AttributeError, ReferenceError, TypeError):
        return []


def ensure_scene_cache_id(scene, claimed_ids=None) -> str:
    """Return a persistent UUID unique among the current Blender scenes.

    Blender copies custom properties when a scene is duplicated. The first
    scene in ``bpy.data.scenes`` retains a duplicated ID; later copies receive
    a new one. ``claimed_ids`` lets load-time reconciliation perform that same
    rule in one deterministic pass.
    """
    claimed = set() if claimed_ids is None else claimed_ids
    scene_key = _scene_key(scene)
    current = _scene_cache_id(scene)
    scenes = _scenes()

    # Preserve the runtime owner regardless of bpy.data.scenes ordering.
    # Blender inserts Scene.copy() before its source and copies PropertyGroup
    # values verbatim; a simple "first scene wins" rule can therefore steal a
    # live cache UUID from the source when switching back to it.
    known_owner = _CACHE_ID_OWNERS.get(current) if current else None
    live_scene_keys = {_scene_key(candidate) for candidate in scenes}
    if known_owner is not None and known_owner not in live_scene_keys:
        _CACHE_ID_OWNERS.pop(current, None)
        known_owner = None
    if current and known_owner == scene_key:
        claimed.add(current)
        return current

    if claimed_ids is None:
        if current and known_owner is not None:
            claimed.add(current)
        for candidate in scenes:
            # RNA iteration may return a different Python wrapper for the same
            # Blender Scene datablock, so object identity is not reliable.
            if _scene_key(candidate) == scene_key:
                break
            candidate_id = _scene_cache_id(candidate)
            if candidate_id:
                claimed.add(candidate_id)

    if current and current not in claimed:
        _CACHE_ID_OWNERS[current] = scene_key
        claimed.add(current)
        return current

    excluded = set(claimed)
    excluded.update(filter(None, (_scene_cache_id(item) for item in _scenes())))
    while True:
        current = uuid.uuid4().hex
        if current not in excluded:
            break
    _write_settings_cache_id(scene, current)
    _CACHE_ID_OWNERS[current] = scene_key
    claimed.add(current)
    return current


def _temporary_cache_dir(scene) -> str:
    """Return a stable cache path unique to this process and scene.

    Unsaved files have no directory against which Blender can resolve ``//``.
    Including both the process id and Blender's scene pointer prevents two
    Blender instances -- or two scenes in one instance -- from sharing a bake.
    """
    scene_id = _scene_key(scene)
    return os.path.join(
        tempfile.gettempdir(), "stflip_cache",
        f"{os.getpid()}-{scene_id:x}",
    )


def _cleanup_process_temporary_caches() -> int:
    """Remove only unsaved-scene caches created by this Blender process."""
    root = os.path.join(tempfile.gettempdir(), "stflip_cache")
    if not os.path.isdir(root):
        return 0
    prefix = f"{os.getpid()}-"
    removed = 0
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not name.startswith(prefix) or not os.path.isdir(path):
            continue
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            removed += 1
    try:
        os.rmdir(root)
    except OSError:
        pass
    return removed


def resolve_cache_dir(scene) -> str:
    """Resolve the configured cache directory to an absolute path.

    Relative paths in an unsaved file use a process/scene-specific temporary
    directory. For saved files, the default path gets a persistent per-scene
    ownership subdirectory. Explicit custom paths are respected exactly.
    """
    configured_value = getattr(scene.stflip, "cache_dir", "")
    normalized = str(configured_value or "").replace("\\", "/").rstrip("/")
    is_default = not normalized or normalized == _DEFAULT_CACHE_DIR
    configured = _DEFAULT_CACHE_DIR if is_default else configured_value
    is_relative = configured.startswith("//") or not os.path.isabs(configured)
    if is_relative and not getattr(bpy.data, "filepath", ""):
        return _temporary_cache_dir(scene)

    path = bpy.path.abspath(configured)
    if path.startswith("//") or not os.path.isabs(path):
        return _temporary_cache_dir(scene)
    path = os.path.normpath(path)
    if is_default:
        path = os.path.join(path, ensure_scene_cache_id(scene))
    return path


def scene_cache_ownership(scene, metadata=_READ_METADATA) -> str:
    """Return :mod:`stflip.cache` ownership status for ``scene``'s cache."""
    owner_id = ensure_scene_cache_id(scene)
    if metadata is _READ_METADATA:
        metadata = cache.read_meta(resolve_cache_dir(scene))
    return cache.ownership_status(metadata, owner_id)


def _frame_bounds(meta: dict, default_start: int):
    """Validate cache frame bounds before they reach Blender's handler."""
    lo = meta.get("frame_start", default_start)
    hi = meta.get("frame_end_baked", lo)
    if (isinstance(lo, bool) or isinstance(hi, bool)
            or not isinstance(lo, Integral)
            or not isinstance(hi, Integral)
            or hi < lo):
        return None
    return int(lo), int(hi)



def _apply_whitewater_frame(scene, cache_dir: str, frame: int) -> None:
    """Stream the frame's whitewater cache (if any) into the display object.

    Whitewater is a cosmetic secondary output: any failure here must never
    turn authoritative particle playback into an error."""
    try:
        obj = scene.stflip.whitewater_object
        if obj is None or obj.type != "MESH":
            return
        path = os.path.join(cache_dir, f"stflip_ww_{frame:06d}.npz")
        if os.path.isfile(path):
            with np.load(path) as data:
                pos = np.asarray(data["pos"], dtype=np.float32)
                vel = np.asarray(data["vel"], dtype=np.float32)
                kind = np.asarray(data["kind"], dtype=np.int32)
                life = np.asarray(data["life"], dtype=np.float32)
        else:
            pos = np.zeros((0, 3), dtype=np.float32)
            vel = np.zeros((0, 3), dtype=np.float32)
            kind = np.zeros((0,), dtype=np.int32)
            life = np.zeros((0,), dtype=np.float32)
        me = obj.data
        n = len(pos)
        if len(me.vertices) != n:
            me.clear_geometry()
            me.vertices.add(n)
        me.vertices.foreach_set("co", np.ascontiguousarray(pos).ravel())
        for name, dtype, values in (
            ("velocity", "FLOAT_VECTOR", vel),
            ("ww_kind", "INT", kind),
            ("ww_life", "FLOAT", life),
        ):
            attr = me.attributes.get(name)
            if attr is None or len(attr.data) != n:
                if attr is not None:
                    me.attributes.remove(attr)
                attr = me.attributes.new(name, dtype, "POINT")
            key = "vector" if dtype == "FLOAT_VECTOR" else "value"
            attr.data.foreach_set(key, np.ascontiguousarray(values).ravel())
        me.update()
    except Exception:
        pass


def _apply_frame(scene, frame: int) -> bool:
    cache_dir = resolve_cache_dir(scene)
    meta = cache.read_meta(cache_dir)
    if meta is None:
        return False
    if scene_cache_ownership(scene, meta) not in {
            cache.OWNERSHIP_OWNED, cache.OWNERSHIP_LEGACY}:
        return False
    bounds = _frame_bounds(meta, scene.frame_start)
    if bounds is None:
        return False
    lo, hi = bounds
    f = min(max(frame, lo), hi)
    data = cache.read_frame(cache_dir, f)
    if data is None:
        return False
    pos, vel = data

    # Prefer the pointer binding (survives renames); an in-scene canonical
    # object is the fallback for legacy files. Never drive another scene's
    # globally named output.
    obj = _scene_particle_object(scene)
    if obj is None or obj.type != "MESH":
        return False
    # Legacy files and copied scenes can retain a shared output binding. Frame
    # playback is an in-place mesh mutation; never let one scene overwrite the
    # particles displayed by another scene or another object sharing the mesh.
    if not mesher.output_is_exclusive(scene, obj):
        return False
    me = obj.data
    n = len(pos)
    if len(me.vertices) != n:
        me.clear_geometry()
        me.vertices.add(n)
    me.vertices.foreach_set("co", np.ascontiguousarray(pos, dtype=np.float32).ravel())
    attr = me.attributes.get("velocity")
    if attr is None or len(attr.data) != n:
        if attr is not None:
            me.attributes.remove(attr)
        attr = me.attributes.new("velocity", "FLOAT_VECTOR", "POINT")
    attr.data.foreach_set(
        "vector", np.ascontiguousarray(vel, dtype=np.float32).ravel())
    # Shading attributes (age, source, speed) for age-fade, per-source colour,
    # and speed-driven effects; absent on caches that predate the format.
    extra = cache.read_frame_attributes(cache_dir, f)
    for name, dtype, key in (("age", "FLOAT", "value"),
                             ("speed", "FLOAT", "value"),
                             ("source", "INT", "value")):
        values = extra.get(name)
        if values is None or len(values) != n:
            continue
        a = me.attributes.get(name)
        if a is None or len(a.data) != n or a.data_type != dtype:
            if a is not None:
                me.attributes.remove(a)
            a = me.attributes.new(name, dtype, "POINT")
        np_dtype = np.int32 if dtype == "INT" else np.float32
        a.data.foreach_set(
            key, np.ascontiguousarray(values, dtype=np_dtype).ravel())
    me.update()
    # Paper reconstruction is a derived display cache.  Once particles have
    # loaded successfully, a missing or invalid paper mesh must never turn the
    # authoritative particle frame into a playback failure.
    _apply_paper_surface_frame(scene, cache_dir, meta, f, pos)
    _apply_whitewater_frame(scene, cache_dir, f)
    return True


def _object_in_scene(scene, obj) -> bool:
    objects = getattr(scene, "objects", None)
    if objects is None:
        return True
    try:
        if obj.name in objects:
            return True
    except (AttributeError, ReferenceError, TypeError):
        pass
    try:
        return obj in objects
    except (ReferenceError, TypeError):
        return False


def _scene_particle_object(scene):
    settings = getattr(scene, "stflip", None)
    obj = getattr(settings, "particle_object", None) if settings else None
    if obj is not None and _object_in_scene(scene, obj):
        return obj
    try:
        obj = bpy.data.objects.get(mesher.PARTICLE_OBJ)
    except (AttributeError, ReferenceError):
        return None
    return obj if obj is not None and _object_in_scene(scene, obj) else None


def _scene_bound_surface_object(scene):
    """Return only this scene's existing, exclusive surface binding.

    Frame handlers must not create output objects or fall back to a globally
    named surface: either could mutate a different scene's visible result.
    """
    settings = getattr(scene, "stflip", None)
    obj = getattr(settings, "surface_object", None) if settings else None
    if (obj is None or getattr(obj, "type", None) != "MESH"
            or not _object_in_scene(scene, obj)):
        return None
    return obj if mesher.output_is_exclusive(scene, obj) else None


def _is_plain_paper_surface(obj) -> bool:
    try:
        method = obj.get("stflip_surface_method", "")
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        try:
            method = getattr(obj, "stflip_surface_method", "")
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            return False
    return str(method or "").upper() == "PAPER_MCF"


def _clear_mesh_geometry(obj) -> bool:
    mesh = getattr(obj, "data", None)
    if mesh is None or not hasattr(mesh, "clear_geometry"):
        return False
    try:
        mesh.clear_geometry()
        mesh.update()
        update_tag = getattr(obj, "update_tag", None)
        if update_tag is not None:
            update_tag()
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return False
    return True


def _clear_scene_paper_surface(scene, obj=None) -> bool:
    surface = _scene_bound_surface_object(scene) if obj is None else obj
    if surface is None or not _is_plain_paper_surface(surface):
        return False
    return _clear_mesh_geometry(surface)


def _apply_paper_surface_frame(
    scene,
    cache_dir: str,
    meta: dict,
    frame: int,
    source_positions,
) -> bool:
    """Best-effort playback of one cached Appendix-B surface mesh."""
    settings = getattr(scene, "stflip", None)
    if (settings is None
            or not bool(getattr(settings, "create_surface", False))
            or str(getattr(settings, "surface_method", "")) != "PAPER_MCF"):
        return False

    surface = _scene_bound_surface_object(scene)
    if surface is None:
        return False
    reconstruction = meta.get("surface_reconstruction")
    try:
        fingerprint = cache.validate_surface_metadata(reconstruction)
    except cache.SurfaceCacheError:
        _clear_scene_paper_surface(scene, surface)
        return False

    try:
        paper_mesh = cache.read_surface(
            cache_dir,
            frame,
            fingerprint,
            expected_source_positions=source_positions,
        )
    except (cache.CheckpointError, cache.SurfaceCacheError):
        _clear_scene_paper_surface(scene, surface)
        return False
    if paper_mesh is None:
        _clear_scene_paper_surface(scene, surface)
        return False
    try:
        mesher.update_paper_surface_mesh(surface, *paper_mesh)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        _clear_scene_paper_surface(scene, surface)
        return False
    return True


def _mesh_is_shared_with_other_scene(scene, obj) -> bool:
    return not mesher.output_is_exclusive(scene, obj)


def clear_scene_output(scene) -> bool:
    """Empty scene-exclusive particle and plain paper-surface outputs."""
    obj = _scene_particle_object(scene)
    particle_cleared = False
    if (obj is not None and getattr(obj, "type", None) == "MESH"
            and not _mesh_is_shared_with_other_scene(scene, obj)):
        particle_cleared = _clear_mesh_geometry(obj)

    # Geometry Nodes preview output is driven by the particle mesh and needs no
    # destructive mutation here.  A cached PAPER_MCF result is ordinary mesh
    # geometry, so clear an exclusive bound instance explicitly.
    surface = _scene_bound_surface_object(scene)
    surface_cleared = False
    if surface is not obj:
        surface_cleared = _clear_scene_paper_surface(scene, surface)
    elif _is_plain_paper_surface(surface):
        surface_cleared = particle_cleared
    return particle_cleared or surface_cleared


def _expects_persisted_cache(settings) -> bool:
    if str(getattr(settings, "bake_state", "") or "").upper() == "COMPLETE":
        return True
    status = str(getattr(settings, "bake_status", "") or "").lower()
    return status.startswith(("bake complete", "cache unavailable"))


def _invalidate_scene_cache(scene, reason: str, *, always_mark=False) -> bool:
    settings = scene.stflip
    clear_scene_output(scene)
    if always_mark or _expects_persisted_cache(settings):
        message = f"Cache unavailable: {reason}; rebake required"
        settings.bake_status = message
        for name, value in (("bake_state", "FAILED"),
                            ("bake_error", message),
                            ("bake_progress", 0.0)):
            try:
                setattr(settings, name, value)
            except (AttributeError, RuntimeError, TypeError):
                pass
    return False


def reconcile_scene_cache(scene) -> bool:
    """Reconcile persisted bake state and visible output for one scene.

    Valid owned and ownerless legacy caches retain their status and load the
    current frame. Missing/corrupt/foreign state never leaves stale particle
    geometry visible. A persisted completed bake is marked unavailable so the
    UI cannot claim that a missing bake is still complete.
    """
    settings = getattr(scene, "stflip", None)
    if settings is None:
        return False
    cache_dir = resolve_cache_dir(scene)
    metadata = cache.read_meta(cache_dir)
    ownership = scene_cache_ownership(scene, metadata)
    if ownership == cache.OWNERSHIP_MISSING:
        return _invalidate_scene_cache(scene, "metadata is missing")
    if ownership == cache.OWNERSHIP_FOREIGN:
        return _invalidate_scene_cache(
            scene, "cache belongs to another scene", always_mark=True)
    if ownership == cache.OWNERSHIP_INVALID:
        return _invalidate_scene_cache(
            scene, "cache ownership metadata is invalid", always_mark=True)

    bounds = _frame_bounds(metadata, getattr(scene, "frame_start", 1))
    if bounds is None:
        return _invalidate_scene_cache(scene, "metadata is incompatible")
    lo, hi = bounds
    requested_end = metadata.get("frame_end", hi)
    if (isinstance(requested_end, bool)
            or not isinstance(requested_end, Integral)
            or requested_end < lo):
        return _invalidate_scene_cache(scene, "metadata is incompatible")
    requested_end = int(requested_end)
    if hi > requested_end:
        return _invalidate_scene_cache(scene, "metadata is incompatible")
    checkpoint_meta = metadata.get("checkpoint")
    lifecycle = metadata.get("bake_lifecycle")
    durable_state = (
        lifecycle.get("state") if isinstance(lifecycle, dict)
        else checkpoint_meta.get("state")
        if isinstance(checkpoint_meta, dict) else None
    )
    # The .blend can retain a stale COMPLETE RNA value when a later resume
    # extends metadata on disk and Blender exits before the file is saved.
    # Durable RUNNING/CANCELLED/FAILED metadata is authoritative in that case;
    # it must reach the restart recovery below instead of being discarded as a
    # supposedly incomplete completed bake.
    if (_expects_persisted_cache(settings) and hi < requested_end
            and durable_state not in {"RUNNING", "CANCELLED", "FAILED"}):
        return _invalidate_scene_cache(
            scene, "completed bake has an incomplete frame range")
    on_disk = {
        frame for frame in cache.baked_frames(cache_dir)
        if lo <= frame <= hi
    }
    if len(on_disk) != hi - lo + 1:
        missing = next(frame for frame in range(lo, hi + 1)
                       if frame not in on_disk)
        return _invalidate_scene_cache(scene, f"frame {missing} is missing")
    current = int(getattr(scene, "frame_current", lo))
    frame = min(max(current, lo), hi)
    if cache.read_frame(cache_dir, frame) is None:
        return _invalidate_scene_cache(scene, f"frame {frame} is missing")

    checkpoint_valid = False
    if (isinstance(checkpoint_meta, dict)
            and checkpoint_meta.get("schema") == cache.CHECKPOINT_SCHEMA
            and checkpoint_meta.get("version") == cache.CHECKPOINT_VERSION
            and checkpoint_meta.get("latest_frame") == hi
            and isinstance(checkpoint_meta.get("fingerprint"), str)):
        try:
            checkpoint_valid = cache.read_checkpoint(
                cache_dir,
                hi,
                expected_fingerprint=checkpoint_meta["fingerprint"],
            ) is not None
        except cache.CheckpointError:
            checkpoint_valid = False

    # A deleted output object does not make the on-disk cache invalid. If an
    # output is present, refresh it immediately so the saved .blend never
    # displays geometry from a different cached frame after loading.
    _apply_frame(scene, current)
    requested_count = max(1, requested_end - lo + 1)
    committed_count = max(0, min(hi, requested_end) - lo + 1)
    try:
        settings.bake_progress = committed_count / requested_count
        if hi >= requested_end:
            previous_status = str(getattr(settings, "bake_status", "") or "")
            settings.bake_state = "COMPLETE"
            settings.bake_error = ""
            settings.bake_progress = 1.0
            if not previous_status.lower().startswith("bake complete"):
                settings.bake_status = (
                    f"Bake complete ({requested_count} cached frames)")
            if (isinstance(checkpoint_meta, dict)
                    and checkpoint_meta.get("state") == "RUNNING"):
                checkpoint_meta["state"] = (
                    "COMPLETE" if checkpoint_valid else "FAILED")
                metadata["bake_lifecycle"] = {
                    "state": "COMPLETE",
                    "last_committed_frame": hi,
                    "error": "" if checkpoint_valid else (
                        "latest solver checkpoint is missing or corrupt"),
                }
                cache.write_meta(cache_dir, metadata)
        else:
            ui_state = str(getattr(settings, "bake_state", "") or "").upper()
            if ui_state == "RUNNING" or durable_state == "RUNNING":
                if checkpoint_valid:
                    target_state = "CANCELLED"
                    message = (
                        f"Bake interrupted after {committed_count} cached "
                        "frames; Resume Bake to continue")
                    error = ""
                else:
                    target_state = "FAILED"
                    message = (
                        "Bake interrupted and its latest solver checkpoint is "
                        "missing or corrupt; rebake required")
                    error = message
                settings.bake_state = target_state
                settings.bake_status = message
                settings.bake_error = error
                if isinstance(checkpoint_meta, dict):
                    checkpoint_meta["state"] = target_state
                metadata["bake_lifecycle"] = {
                    "state": target_state,
                    "last_committed_frame": hi,
                    "error": error,
                }
                cache.write_meta(cache_dir, metadata)
            elif durable_state in {"CANCELLED", "FAILED"}:
                if checkpoint_valid:
                    settings.bake_state = durable_state
                    durable_error = (
                        str(lifecycle.get("error", ""))
                        if isinstance(lifecycle, dict) else "")
                    settings.bake_error = durable_error
                    if durable_state == "CANCELLED":
                        settings.bake_status = (
                            f"Bake stopped after {committed_count} cached "
                            "frames; Resume Bake to continue")
                    elif durable_error:
                        settings.bake_status = f"Bake failed: {durable_error}"
                    else:
                        settings.bake_status = (
                            f"Bake failed after {committed_count} cached "
                            "frames; Resume Bake can retry")
                else:
                    message = (
                        "Latest solver checkpoint is missing or corrupt; "
                        "cached frames remain playable but rebaking is required"
                    )
                    settings.bake_state = "FAILED"
                    settings.bake_status = message
                    settings.bake_error = message
                    if isinstance(checkpoint_meta, dict):
                        checkpoint_meta["state"] = "FAILED"
                    metadata["bake_lifecycle"] = {
                        "state": "FAILED",
                        "last_committed_frame": hi,
                        "error": message,
                    }
                    cache.write_meta(cache_dir, metadata)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    return True


def reconcile_scene_caches() -> int:
    """Assign unique scene IDs, then reconcile every ST-FLIP scene."""
    scenes = _scenes()
    claimed: set[str] = set()
    for scene in scenes:
        if getattr(scene, "stflip", None) is not None:
            ensure_scene_cache_id(scene, claimed)
    return sum(
        reconcile_scene_cache(scene)
        for scene in scenes
        if getattr(scene, "stflip", None) is not None
    )


@persistent
def stflip_frame_change(scene, depsgraph=None):
    if scene is None or not getattr(scene, "stflip", None):
        return
    if not _apply_frame(scene, scene.frame_current):
        clear_scene_output(scene)


@persistent
def stflip_load_post(_unused=None):
    _FALLBACK_CACHE_IDS.clear()
    _CACHE_ID_OWNERS.clear()
    reconcile_scene_caches()


def ensure_registered():
    global _ATEXIT_REGISTERED
    handlers = bpy.app.handlers.frame_change_post
    if not any(getattr(h, "__name__", "") == "stflip_frame_change"
               for h in handlers):
        handlers.append(stflip_frame_change)
    load_handlers = getattr(bpy.app.handlers, "load_post", None)
    if load_handlers is not None and not any(
            getattr(h, "__name__", "") == "stflip_load_post"
            for h in load_handlers):
        load_handlers.append(stflip_load_post)
    # Blender exposes no quit-pre handler. Python's process-exit callback is
    # invoked during a normal Blender shutdown and works across supported
    # Blender versions.
    if not _ATEXIT_REGISTERED:
        atexit.register(_cleanup_process_temporary_caches)
        _ATEXIT_REGISTERED = True


def register():
    ensure_registered()
    reconcile_scene_caches()


def unregister():
    global _ATEXIT_REGISTERED
    handlers = bpy.app.handlers.frame_change_post
    for h in list(handlers):
        if getattr(h, "__name__", "") == "stflip_frame_change":
            handlers.remove(h)
    load_handlers = getattr(bpy.app.handlers, "load_post", None)
    if load_handlers is not None:
        for h in list(load_handlers):
            if getattr(h, "__name__", "") == "stflip_load_post":
                load_handlers.remove(h)
    if _ATEXIT_REGISTERED:
        atexit.unregister(_cleanup_process_temporary_caches)
        _ATEXIT_REGISTERED = False
    _CACHE_ID_OWNERS.clear()
    # Do not remove caches or clear output on add-on disable: a user can
    # re-enable the add-on in this Blender process and resume valid playback.
