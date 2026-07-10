"""Frame-change handler: stream cached particle frames into the point cloud."""

import os
import tempfile

import bpy
import numpy as np
from bpy.app.handlers import persistent

from ..stflip import cache
from . import mesher


def resolve_cache_dir(scene) -> str:
    """Absolute cache path; falls back to the system temp directory when the
    .blend file has not been saved (so '//' cannot resolve)."""
    path = bpy.path.abspath(scene.stflip.cache_dir)
    if path.startswith("//") or not os.path.isabs(path):
        path = os.path.join(tempfile.gettempdir(), "stflip_cache")
    return path


def _apply_frame(scene, frame: int) -> bool:
    cache_dir = resolve_cache_dir(scene)
    meta = cache.read_meta(cache_dir)
    if meta is None:
        return False
    lo = meta.get("frame_start", scene.frame_start)
    hi = meta.get("frame_end_baked", lo)
    f = min(max(frame, lo), hi)
    data = cache.read_frame(cache_dir, f)
    if data is None:
        return False
    pos, vel = data

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
    if not scene.stflip.domain:
        return
    _apply_frame(scene, scene.frame_current)


def ensure_registered():
    handlers = bpy.app.handlers.frame_change_post
    if not any(getattr(h, "__name__", "") == "stflip_frame_change"
               for h in handlers):
        handlers.append(stflip_frame_change)


def register():
    ensure_registered()


def unregister():
    handlers = bpy.app.handlers.frame_change_post
    for h in list(handlers):
        if getattr(h, "__name__", "") == "stflip_frame_change":
            handlers.remove(h)
