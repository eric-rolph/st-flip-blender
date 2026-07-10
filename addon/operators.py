"""Operators: quick setup, bake (modal), free bake, GPU support installer."""

import importlib
import os
import subprocess
import sys
import tempfile

import bpy
import numpy as np

from ..stflip import cache
from ..stflip.backend import cuda_available, get_backend
from ..stflip.solver import Params, STFLIPSolver
from . import handlers, mesher, voxelize

# The live solver cannot be stored on Blender ID properties; module state it is.
_BAKE: dict = {}


def _fluid_objects(scene, role: str):
    return [o for o in scene.objects
            if o.type == "MESH" and o.stflip.role == role]


def _resolve_cache_dir(scene) -> str:
    path = bpy.path.abspath(scene.stflip.cache_dir)
    if path.startswith("//") or not os.path.isabs(path):
        # .blend not saved yet: fall back to a temp location.
        path = os.path.join(tempfile.gettempdir(), "stflip_cache")
    return path


class STFLIP_OT_quick_setup(bpy.types.Operator):
    """Create a ready-to-bake dam-break scene (domain, liquid, roles)"""
    bl_idname = "stflip.quick_setup"
    bl_label = "Quick Dam-Break Setup"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene

        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 1))
        domain = context.active_object
        domain.name = "STFLIP Domain"
        domain.display_type = "WIRE"
        domain.hide_render = True

        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(-0.65, 0, 0.55))
        liquid = context.active_object
        liquid.name = "STFLIP Liquid"
        liquid.scale = (0.34, 0.98, 0.54)
        liquid.display_type = "WIRE"
        liquid.hide_render = True
        liquid.stflip.role = "LIQUID"

        scene.stflip.domain = domain
        self.report({"INFO"}, "Dam-break scene created; press Bake")
        return {"FINISHED"}


class STFLIP_OT_bake(bpy.types.Operator):
    """Bake the ST-FLIP simulation for the scene frame range"""
    bl_idname = "stflip.bake"
    bl_label = "Bake Simulation"
    bl_options = {"REGISTER"}

    _timer = None

    def invoke(self, context, event):
        scene = context.scene
        st = scene.stflip
        if _BAKE.get("running"):
            self.report({"WARNING"}, "A bake is already running")
            return {"CANCELLED"}
        if st.domain is None:
            self.report({"ERROR"}, "Set a domain object first")
            return {"CANCELLED"}

        liquids = _fluid_objects(scene, "LIQUID")
        inflows = _fluid_objects(scene, "INFLOW")
        obstacles = _fluid_objects(scene, "OBSTACLE")
        if not liquids and not inflows:
            self.report({"ERROR"}, "Mark at least one mesh as Liquid or Inflow")
            return {"CANCELLED"}

        deps = context.evaluated_depsgraph_get()
        dims, dx, origin = voxelize.domain_grid(st.domain, st.resolution)

        st.bake_status = "Voxelizing scene..."
        solid_sdf = None
        if obstacles:
            solid_sdf = voxelize.sdf_from_objects(obstacles, deps, origin, dx, dims)

        fps = scene.render.fps / scene.render.fps_base
        gravity = tuple(scene.gravity) if scene.use_gravity else (0.0, 0.0, 0.0)
        params = Params(
            resolution=dims, dx=dx, gravity=gravity,
            frame_dt=1.0 / fps, cfl_target=st.cfl_target,
            particles_per_cell=st.particles_per_cell,
            flip_blend=st.flip_blend, st_enabled=st.st_enabled,
            jitter_strength=st.jitter_strength,
            adaptive_gamma=st.adaptive_gamma, eta_phi=st.eta_phi,
        )
        try:
            backend = get_backend(st.backend)
        except Exception as exc:
            self.report({"WARNING"},
                        f"GPU backend unavailable ({exc}); using CPU")
            backend = get_backend("cpu")

        solver = STFLIPSolver(params, backend)
        if solid_sdf is not None:
            solver.set_solid_sdf(solid_sdf)

        not_solid = (solid_sdf > 0.0) if solid_sdf is not None else None
        seeded = 0
        for obj in liquids:
            mask = voxelize.mask_from_object(obj, deps, origin, dx, dims)
            if not_solid is not None:
                mask &= not_solid
            seeded += solver.add_liquid_mask(mask)
        for obj in inflows:
            mask = voxelize.mask_from_object(obj, deps, origin, dx, dims)
            if not_solid is not None:
                mask &= not_solid
            solver.add_inflow(mask, tuple(obj.stflip.inflow_velocity))
        if seeded == 0 and not inflows:
            self.report({"ERROR"}, "No liquid cells inside the domain")
            return {"CANCELLED"}

        cache_dir = _resolve_cache_dir(scene)
        cache.clear(cache_dir)
        meta = {
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "frame_end_baked": scene.frame_start,
            "dx": dx, "dims": list(dims), "origin": origin.tolist(),
            "backend": backend.name, "version": 1,
        }
        cache.write_meta(cache_dir, meta)

        pos, vel = solver.get_render_particles()
        cache.write_frame(cache_dir, scene.frame_start,
                          pos + origin[None, :].astype(np.float32), vel)

        particle_obj = mesher.ensure_particle_object()
        if st.create_surface:
            mesher.ensure_surface_object(
                particle_obj, dx, st.particle_radius, st.surface_voxel)
        handlers.ensure_registered()

        _BAKE.update(solver=solver, origin=origin.astype(np.float32),
                     cache_dir=cache_dir, meta=meta,
                     frame=scene.frame_start, end=scene.frame_end,
                     running=True)

        scene.frame_set(scene.frame_start)
        st.bake_status = f"Baked {seeded} particles seeded..."
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        scene = context.scene
        st = scene.stflip
        if event.type == "ESC":
            return self._finish(context, cancelled=True)
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        b = _BAKE
        if not b.get("running"):
            return self._finish(context, cancelled=True)
        if b["frame"] >= b["end"]:
            return self._finish(context, cancelled=False)

        solver: STFLIPSolver = b["solver"]
        stats = solver.step_frame()
        b["frame"] += 1
        pos, vel = solver.get_render_particles()
        cache.write_frame(b["cache_dir"], b["frame"],
                          pos + b["origin"][None, :], vel)
        b["meta"]["frame_end_baked"] = b["frame"]
        cache.write_meta(b["cache_dir"], b["meta"])

        st.bake_status = (
            f"Frame {b['frame']}/{b['end']}  "
            f"({stats.n_particles} pts, {stats.steps} steps, "
            f"{solver.be.name})")
        scene.frame_set(b["frame"])
        return {"RUNNING_MODAL"}

    def _finish(self, context, cancelled: bool):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        st = context.scene.stflip
        st.bake_status = ("Bake cancelled" if cancelled
                          else f"Bake complete ({_BAKE.get('frame', '?')} frames)")
        _BAKE["running"] = False
        _BAKE.pop("solver", None)
        return {"CANCELLED"} if cancelled else {"FINISHED"}


class STFLIP_OT_free_bake(bpy.types.Operator):
    """Delete the bake cache"""
    bl_idname = "stflip.free_bake"
    bl_label = "Free Bake"
    bl_options = {"REGISTER"}

    def execute(self, context):
        if _BAKE.get("running"):
            self.report({"WARNING"}, "Stop the running bake first (Esc)")
            return {"CANCELLED"}
        n = cache.clear(_resolve_cache_dir(context.scene))
        context.scene.stflip.bake_status = ""
        self.report({"INFO"}, f"Removed {n} cache files")
        return {"FINISHED"}


class STFLIP_OT_install_gpu(bpy.types.Operator):
    """Install CuPy (NVIDIA CUDA 12.x) into Blender's user modules.
    Downloads ~100 MB from PyPI; Blender may freeze for a few minutes"""
    bl_idname = "stflip.install_gpu"
    bl_label = "Install GPU Support (CUDA)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        target = bpy.utils.user_resource("SCRIPTS", path="modules", create=True)
        res = None
        for package in ("cupy-cuda13x", "cupy-cuda12x"):
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade",
                   "--target", target, package]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=1800)
            except Exception as exc:
                self.report({"ERROR"}, f"pip failed to run: {exc}")
                return {"CANCELLED"}
            if res.returncode == 0:
                break
        if res is None or res.returncode != 0:
            self.report({"ERROR"},
                        f"pip install failed: {res.stderr[-400:]}")
            return {"CANCELLED"}
        if target not in sys.path:
            sys.path.append(target)
        importlib.invalidate_caches()
        if cuda_available():
            self.report({"INFO"}, "CuPy installed; CUDA GPU detected")
        else:
            self.report({"WARNING"},
                        "CuPy installed but no CUDA device detected")
        return {"FINISHED"}


CLASSES = (
    STFLIP_OT_quick_setup,
    STFLIP_OT_bake,
    STFLIP_OT_free_bake,
    STFLIP_OT_install_gpu,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
