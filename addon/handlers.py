"""Frame-change handler: stream cached particle frames into the point cloud."""

import atexit
import os
import shutil
import tempfile
from numbers import Integral

import bpy
import numpy as np
from bpy.app.handlers import persistent

from ..stflip import cache
from . import mesher


_ATEXIT_REGISTERED = False


def _temporary_cache_dir(scene) -> str:
    """Return a stable cache path unique to this process and scene.

    Unsaved files have no directory against which Blender can resolve ``//``.
    Including both the process id and Blender's scene pointer prevents two
    Blender instances -- or two scenes in one instance -- from sharing a bake.
    """
    try:
        scene_id = int(scene.as_pointer())
    except (AttributeError, TypeError, ValueError):
        # Useful for light-weight test doubles; real Blender scenes always
        # expose as_pointer().
        scene_id = id(scene)
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
    directory.  An explicitly configured absolute path is still respected.
    """
    configured = scene.stflip.cache_dir or "//stflip_cache"
    is_relative = configured.startswith("//") or not os.path.isabs(configured)
    if is_relative and not getattr(bpy.data, "filepath", ""):
        return _temporary_cache_dir(scene)

    path = bpy.path.abspath(configured)
    if path.startswith("//") or not os.path.isabs(path):
        return _temporary_cache_dir(scene)
    return os.path.normpath(path)


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


def _apply_frame(scene, frame: int) -> bool:
    cache_dir = resolve_cache_dir(scene)
    meta = cache.read_meta(cache_dir)
    if meta is None:
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

    # Prefer the pointer binding (survives renames); name lookup is the
    # fallback for legacy files.
    obj = scene.stflip.particle_object
    if obj is None:
        obj = bpy.data.objects.get(mesher.PARTICLE_OBJ)
    if obj is None or obj.type != "MESH":
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
    me.update()
    return True


@persistent
def stflip_frame_change(scene, depsgraph=None):
    if scene is None or not getattr(scene, "stflip", None):
        return
    _apply_frame(scene, scene.frame_current)


def ensure_registered():
    global _ATEXIT_REGISTERED
    handlers = bpy.app.handlers.frame_change_post
    if not any(getattr(h, "__name__", "") == "stflip_frame_change"
               for h in handlers):
        handlers.append(stflip_frame_change)
    # Blender exposes no quit-pre handler. Python's process-exit callback is
    # invoked during a normal Blender shutdown and works across supported
    # Blender versions.
    if not _ATEXIT_REGISTERED:
        atexit.register(_cleanup_process_temporary_caches)
        _ATEXIT_REGISTERED = True


def register():
    ensure_registered()


def unregister():
    global _ATEXIT_REGISTERED
    handlers = bpy.app.handlers.frame_change_post
    for h in list(handlers):
        if getattr(h, "__name__", "") == "stflip_frame_change":
            handlers.remove(h)
    if _ATEXIT_REGISTERED:
        atexit.unregister(_cleanup_process_temporary_caches)
        _ATEXIT_REGISTERED = False
    # Unsaved-scene caches are explicitly temporary. Once the add-on is
    # disabled there is no playback handler left to consume them.
    _cleanup_process_temporary_caches()
