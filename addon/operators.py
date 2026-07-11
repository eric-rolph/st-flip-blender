"""Operators: quick setup, bake (modal), free bake, GPU support installer."""

from __future__ import annotations

import ctypes
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import bpy


CUPY_VERSION = "14.1.1"
GPU_INSTALL_CANDIDATES = (
    {
        "label": "CUDA 13 runtime",
        "slug": "cuda13",
        "requirement": f"cupy-cuda13x[ctk]=={CUPY_VERSION}",
    },
    {
        "label": "CUDA 12 toolkit",
        "slug": "cuda12",
        "requirement": f"cupy-cuda12x=={CUPY_VERSION}",
    },
)

_RUNTIME_DIRNAME = "stflip_cuda_runtime"
_ACTIVE_RUNTIME_FILE = "active.txt"
_INSTALLING_RUNTIME_FILE = ".installing"
_INSTALLING_STALE_SECONDS = 24 * 60 * 60
_GRID_BYTES_PER_CELL = 256
_PARTICLE_BYTES = 160
_CUDA_HOST_GRID_BYTES_PER_CELL = 96
_CUDA_HOST_PARTICLE_BYTES = 48
_MEMORY_HEADROOM_FRACTION = 0.75


def _value(source, *names, default=None):
    for name in names:
        if isinstance(source, dict) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    return default


def normalize_cuda_diagnostics(diagnostic) -> dict:
    """Normalize backend diagnostic mappings, objects, or ``(ok, reason)``."""
    if isinstance(diagnostic, tuple) and len(diagnostic) >= 2:
        available = bool(diagnostic[0])
        reason = str(diagnostic[1] or "")
        prefix = "CUDA preflight passed on "
        return {
            "available": available,
            "device": reason[len(prefix):] if available
            and reason.startswith(prefix) else "",
            "free_bytes": None,
            "total_bytes": None,
            "error": "" if available else reason,
        }
    available = bool(_value(diagnostic, "available", "ok", default=False))
    return {
        "available": available,
        "device": str(_value(
            diagnostic, "device_name", "device", default="") or ""),
        "free_bytes": _value(
            diagnostic, "free_bytes", "memory_free", default=None),
        "total_bytes": _value(
            diagnostic, "total_bytes", "memory_total", default=None),
        "error": "" if available else str(_value(
            diagnostic, "error", "message", "reason", default="") or ""),
    }


def estimate_bake_memory(dims, particles_per_cell: int) -> dict:
    """Return a deliberately conservative worst-case working-set estimate.

    The estimate assumes every domain cell is initially filled.  It includes
    persistent particle/grid state and the large temporary index/weight arrays
    used by P2G, interpolation, pressure projection, and cache export.
    """
    dims = tuple(int(v) for v in dims)
    if len(dims) != 3 or any(v <= 0 for v in dims):
        raise ValueError(f"invalid grid dimensions: {dims!r}")
    ppc = max(1, int(particles_per_cell))
    cells = math.prod(dims)
    particles = cells * ppc
    working = (cells * _GRID_BYTES_PER_CELL
               + particles * _PARTICLE_BYTES)
    cuda_host = (cells * _CUDA_HOST_GRID_BYTES_PER_CELL
                 + particles * _CUDA_HOST_PARTICLE_BYTES)
    return {
        "dims": dims,
        "cells": cells,
        "particles": particles,
        "working_set_bytes": working,
        "cuda_host_bytes": cuda_host,
    }


def _format_bytes(value: int | float) -> str:
    value = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or suffix == "TiB":
            return f"{value:.1f} {suffix}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def memory_guard_reason(estimate: dict, backend_name: str,
                        ram_available: int | None,
                        vram_available: int | None) -> str | None:
    """Explain a predictable OOM, or return ``None`` when headroom is sane."""
    dims = "x".join(str(v) for v in estimate["dims"])
    working = int(estimate["working_set_bytes"])
    if backend_name == "cuda" and vram_available:
        safe_vram = int(vram_available * _MEMORY_HEADROOM_FRACTION)
        if working > safe_vram:
            return (
                f"Grid {dims} may require {_format_bytes(working)} VRAM, "
                f"above the safe {_format_bytes(safe_vram)} of currently "
                "available VRAM. Lower Resolution or Particles / Cell."
            )
    host_need = (int(estimate["cuda_host_bytes"])
                 if backend_name == "cuda" else working)
    if ram_available:
        safe_ram = int(ram_available * _MEMORY_HEADROOM_FRACTION)
        if host_need > safe_ram:
            return (
                f"Grid {dims} may require {_format_bytes(host_need)} RAM, "
                f"above the safe {_format_bytes(safe_ram)} of currently "
                "available RAM. Lower Resolution or Particles / Cell."
            )
    return None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def remove_shadow_numpy(candidate_dir: Path) -> list[str]:
    """Remove pip's target-local NumPy while preserving all other packages.

    Blender ships a tested NumPy build.  ``pip --target`` otherwise installs
    a newer dependency beside CuPy and shadows Blender's copy on restart.
    Candidate directories are isolated, so only direct NumPy artifacts inside
    the supplied directory are eligible for removal.
    """
    candidate_dir = Path(candidate_dir)
    if not candidate_dir.is_dir():
        return []
    removed = []
    for child in candidate_dir.iterdir():
        lower = child.name.lower()
        if lower not in {"numpy", "numpy.libs"} \
                and not (lower.startswith("numpy-")
                         and lower.endswith(".dist-info")):
            continue
        if not _is_within(child, candidate_dir):
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed.append(child.name)
    return removed


def _user_modules_dir(create: bool = False) -> Path | None:
    try:
        value = bpy.utils.user_resource(
            "SCRIPTS", path="modules", create=create)
    except Exception:
        return None
    return Path(value) if value else None


def _runtime_root(create: bool = False) -> Path | None:
    modules = _user_modules_dir(create=create)
    if modules is None:
        return None
    root = modules / _RUNTIME_DIRNAME
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _configured_runtime_path() -> Path | None:
    root = _runtime_root(create=False)
    if root is None:
        return None
    marker = root / _ACTIVE_RUNTIME_FILE
    try:
        candidate = Path(marker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if candidate.is_dir() and _is_within(candidate, root):
        return candidate
    return None


def _activate_runtime_path(candidate: Path | None) -> None:
    if candidate is None:
        return
    prefixes = ("cupy", "cupyx", "cupy_backends", "cuda_pathfinder")
    for name in tuple(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".")
               for prefix in prefixes):
            sys.modules.pop(name, None)
    text = str(candidate)
    if text not in sys.path:
        # NumPy is already loaded by addon.handlers before operators.  Put the
        # isolated runtime first so an obsolete legacy CuPy cannot shadow it.
        sys.path.insert(0, text)
    importlib.invalidate_caches()


_activate_runtime_path(_configured_runtime_path())

import numpy as np  # noqa: E402 - runtime path must be activated first

from ..stflip import cache  # noqa: E402
from ..stflip.backend import (  # noqa: E402
    cuda_device_name,
    cuda_diagnostics,
    get_backend,
)
from ..stflip.solver import Params, STFLIPSolver  # noqa: E402
from ..stflip.velocity import SolidBodyRotation, UniformVelocity  # noqa: E402
from . import handlers, mesher, voxelize  # noqa: E402
from .handlers import resolve_cache_dir  # noqa: E402

# The live solver cannot be stored on Blender ID properties; module state it is.
_BAKE: dict = {}


def _fluid_objects(scene, role: str):
    return [o for o in scene.objects
            if o.type == "MESH" and o.stflip.role == role]


def _finite_vector3(value, label: str, source_name: str) -> np.ndarray:
    """Return a finite float64 vector with a source-specific error."""
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source_name}: {label} must contain three finite values"
        ) from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{source_name}: {label} must contain three finite values"
        )
    return vector


def resolve_liquid_initial_velocity(settings, domain_origin, source_name):
    """Resolve Blender world-space controls to a solver-local field.

    The returned descriptor is constructed from the same normalized field
    passed to the solver, so cache metadata records the values actually used.
    """
    source_name = str(source_name)
    linear = _finite_vector3(
        settings.initial_velocity, "Initial Velocity", source_name)
    origin = _finite_vector3(domain_origin, "Domain Origin", source_name)
    mode = str(settings.initial_velocity_mode)

    if mode == "UNIFORM":
        field = UniformVelocity(tuple(linear))
        return field, {
            "name": source_name,
            "initial_velocity": list(field.value),
            "initial_velocity_mode": mode,
        }
    if mode != "SOLID_BODY":
        raise ValueError(
            f"{source_name}: unknown Initial Velocity Mode {mode!r}")

    center_world = _finite_vector3(
        settings.rotation_center_world, "Rotation Center", source_name)
    axis_authored = _finite_vector3(
        settings.rotation_axis_world, "Rotation Axis", source_name)
    axis_length = float(np.linalg.norm(axis_authored))
    if not np.isfinite(axis_length) or axis_length <= 1e-12:
        raise ValueError(f"{source_name}: Rotation Axis must be non-zero")
    try:
        angular_speed = float(settings.angular_speed)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source_name}: Angular Speed must be finite") from exc
    if not math.isfinite(angular_speed):
        raise ValueError(f"{source_name}: Angular Speed must be finite")

    axis_unit = axis_authored / axis_length
    center_solver = center_world - origin
    angular_velocity = axis_unit * angular_speed
    field = SolidBodyRotation(
        center=tuple(center_solver),
        angular_velocity=tuple(angular_velocity),
        linear_velocity=tuple(linear),
    )
    descriptor = {
        "name": source_name,
        "initial_velocity": list(field.linear_velocity),
        "initial_velocity_mode": mode,
        "solid_body_rotation": {
            "center_world": [float(value) for value in center_world],
            "center_solver_local": list(field.center),
            "axis_world_authored": [float(value) for value in axis_authored],
            "axis_world_unit": [float(value) for value in axis_unit],
            "angular_speed_radians_per_second": angular_speed,
            "angular_velocity_world": list(field.angular_velocity),
        },
    }
    return field, descriptor


def _system_available_memory_bytes() -> int | None:
    """Best-effort available physical RAM without adding a psutil dependency."""
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(
                    ctypes.byref(status)):
                return int(status.available_physical)
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * available_pages
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _cuda_memory_info() -> tuple[int | None, int | None]:
    try:
        import cupy

        free, total = cupy.cuda.runtime.memGetInfo()
        return int(free), int(total)
    except Exception:
        return None, None


def current_cuda_diagnostics(force: bool = False) -> dict:
    """Return normalized compute diagnostics including live memory figures."""
    try:
        raw = cuda_diagnostics(force=force)
    except TypeError:  # Compatibility with older backend implementations.
        raw = cuda_diagnostics()
    except Exception as exc:
        raw = (False, f"CUDA diagnostic failed: {type(exc).__name__}: {exc}")
    result = normalize_cuda_diagnostics(raw)
    if result["available"]:
        if not result["device"]:
            result["device"] = cuda_device_name() or "CUDA device 0"
        free, total = _cuda_memory_info()
        result["free_bytes"] = free
        result["total_bytes"] = total
    return result


def _purge_cuda_imports() -> None:
    """Forget CuPy modules so a newly selected isolated wheel can be loaded."""
    prefixes = ("cupy", "cupyx", "cupy_backends", "cuda_pathfinder")
    for name in tuple(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".")
               for prefix in prefixes):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _safe_remove_runtime_dir(path: Path, root: Path) -> None:
    if path.is_dir() and path != root and _is_within(path, root):
        shutil.rmtree(path, ignore_errors=True)


def cleanup_inactive_cuda_runtimes(active: Path | None = None) -> list[str]:
    """Remove obsolete isolated CUDA installs while preserving the active one."""
    root = _runtime_root(create=False)
    if root is None or not root.is_dir():
        return []
    active = active or _configured_runtime_path()
    active_resolved = active.resolve() if active is not None else None
    removed = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        installing = child / _INSTALLING_RUNTIME_FILE
        if installing.is_file():
            try:
                age = time.time() - installing.stat().st_mtime
            except OSError:
                age = 0.0
            # Another Blender process may be downloading a multi-gigabyte
            # runtime into this directory. Only reap abandoned day-old work.
            if age < _INSTALLING_STALE_SECONDS:
                continue
        try:
            is_active = (active_resolved is not None
                         and child.resolve() == active_resolved)
        except OSError:
            is_active = False
        if is_active:
            continue
        name = child.name
        _safe_remove_runtime_dir(child, root)
        if not child.exists():
            removed.append(name)
    return removed


# On a clean Blender start no old CuPy DLLs are mapped, so this is the safest
# point to reclaim superseded timestamped runtimes from earlier installations.
cleanup_inactive_cuda_runtimes()


_CUDA_PREFLIGHT_PREFIX = "STFLIP_CUDA_PREFLIGHT="
_CUDA_PREFLIGHT_SCRIPT = r"""
import json
try:
    import cupy
    if int(cupy.cuda.runtime.getDeviceCount()) < 1:
        raise RuntimeError("CUDA reported no devices")
    values = cupy.asarray([1.0, 2.0, 3.0, 4.0], dtype=cupy.float32)
    reduced = (values * values + 1.0).sum()
    target = cupy.zeros((3,), dtype=cupy.float32)
    indices = cupy.asarray([0, 1, 1, 2], dtype=cupy.int32)
    updates = cupy.asarray([1.0, 2.0, 3.0, 4.0], dtype=cupy.float32)
    cupy.add.at(target, indices, updates)
    cupy.cuda.get_current_stream().synchronize()
    reduced_host = float(cupy.asnumpy(reduced).item())
    target_host = cupy.asnumpy(target).tolist()
    if abs(reduced_host - 34.0) > 1e-5:
        raise RuntimeError(f"reduction returned {reduced_host}, expected 34")
    if target_host != [1.0, 5.0, 4.0]:
        raise RuntimeError(f"scatter returned {target_host}")
    props = cupy.cuda.runtime.getDeviceProperties(0)
    name = props.get("name", props.get(b"name", "CUDA device 0"))
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    free, total = cupy.cuda.runtime.memGetInfo()
    payload = {"available": True, "device_name": str(name),
               "free_bytes": int(free), "total_bytes": int(total)}
except Exception as exc:
    payload = {"available": False,
               "error": f"{type(exc).__name__}: {exc}"}
print("STFLIP_CUDA_PREFLIGHT=" + json.dumps(payload))
raise SystemExit(0 if payload["available"] else 2)
"""


def _preflight_candidate(candidate_dir: Path) -> dict:
    env = os.environ.copy()
    env["CUPY_COMPILE_WITH_PTX"] = "1"
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (str(candidate_dir) +
                         (os.pathsep + current_pythonpath
                          if current_pythonpath else ""))
    try:
        result = subprocess.run(
            [sys.executable, "-c", _CUDA_PREFLIGHT_SCRIPT],
            capture_output=True, text=True, timeout=180, env=env,
        )
    except Exception as exc:
        return {"available": False,
                "error": f"preflight process failed: {exc}"}
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_CUDA_PREFLIGHT_PREFIX):
            try:
                payload = json.loads(line[len(_CUDA_PREFLIGHT_PREFIX):])
                return normalize_cuda_diagnostics(payload)
            except (json.JSONDecodeError, TypeError, ValueError):
                break
    detail = (result.stderr or result.stdout or
              f"preflight exited with code {result.returncode}")
    return {"available": False, "device": "", "free_bytes": None,
            "total_bytes": None,
            "error": " ".join(detail.strip().split())[-800:]}


def _write_active_runtime(candidate_dir: Path) -> None:
    root = _runtime_root(create=True)
    if root is None or not _is_within(candidate_dir, root):
        raise RuntimeError("Refusing to activate a CUDA runtime outside "
                           "Blender's ST-FLIP modules directory")
    (root / _ACTIVE_RUNTIME_FILE).write_text(
        str(candidate_dir.resolve()), encoding="utf-8")


def _switch_runtime_path(candidate_dir: Path) -> None:
    root = _runtime_root(create=False)
    if root is not None:
        sys.path[:] = [entry for entry in sys.path
                       if not _is_within(Path(entry), root)]
    _purge_cuda_imports()
    _activate_runtime_path(candidate_dir)


def _process_failure_detail(result) -> str:
    detail = result.stderr or result.stdout or (
        f"pip exited with code {result.returncode}")
    return " ".join(detail.strip().split())[-800:]


def _build_solver(params, backend_name: str):
    """Construct and synchronize once so lazy CUDA allocation errors surface."""
    backend = get_backend(backend_name)
    solver = STFLIPSolver(params, backend)
    backend.synchronize()
    return backend, solver


def _measure_output_frame(frame, solver, stats, positions, velocities,
                          compute_wall_s, include_enstrophy):
    """Build one bpy-free diagnostic record from an output-frame snapshot."""
    from ..stflip.metrics import measure_frame

    mac_grids = solver._grids if include_enstrophy and solver._grids else None
    return measure_frame(
        frame=frame,
        simulation_time_s=solver.time,
        params=solver.p,
        stats=stats,
        positions_local=positions,
        velocities=velocities,
        compute_wall_s=compute_wall_s,
        mac_grids=mac_grids,
        array_module=solver.be.xp if mac_grids is not None else None,
    )


def _set_surface_enabled(obj, enabled: bool) -> None:
    if obj is None:
        return
    obj.hide_render = not enabled
    if hasattr(obj, "hide_viewport"):
        obj.hide_viewport = not enabled
    try:
        obj.hide_set(not enabled)
    except (AttributeError, RuntimeError):
        pass
    modifier = getattr(obj, "modifiers", {}).get("STFLIP Surface")
    if modifier is not None:
        modifier.show_viewport = enabled
        modifier.show_render = enabled


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

    def _setup(self, context) -> bool:
        """Voxelize the scene, build the solver, write the first frame.
        Returns False (after self.report) if the bake cannot start."""
        scene = context.scene
        st = scene.stflip
        if _BAKE.get("running"):
            self.report({"WARNING"}, "A bake is already running")
            return False
        if st.domain is None:
            self.report({"ERROR"}, "Set a domain object first")
            return False

        liquids = _fluid_objects(scene, "LIQUID")
        inflows = _fluid_objects(scene, "INFLOW")
        obstacles = _fluid_objects(scene, "OBSTACLE")
        if not liquids and not inflows:
            self.report({"ERROR"}, "Mark at least one mesh as Liquid or Inflow")
            return False

        deps = context.evaluated_depsgraph_get()
        dims, dx, origin = voxelize.domain_grid(st.domain, st.resolution)
        try:
            liquid_velocity_sources = [
                (
                    obj,
                    *resolve_liquid_initial_velocity(
                        obj.stflip, origin, obj.name),
                )
                for obj in liquids
            ]
        except ValueError as exc:
            message = str(exc)
            st.bake_status = f"Bake blocked: {message}"
            self.report({"ERROR"}, message)
            return False
        fps = scene.render.fps / scene.render.fps_base
        gravity = tuple(scene.gravity) if scene.use_gravity else (0.0, 0.0, 0.0)
        params = Params(
            resolution=dims, dx=dx, gravity=gravity,
            frame_dt=1.0 / fps, cfl_target=st.cfl_target,
            particles_per_cell=st.particles_per_cell,
            seed=st.seed,
            flip_blend=st.flip_blend, st_enabled=st.st_enabled,
            jitter_strength=st.jitter_strength,
            adaptive_gamma=st.adaptive_gamma, eta_phi=st.eta_phi,
        )

        cuda_state = ({"available": False, "device": "",
                       "free_bytes": None, "total_bytes": None, "error": ""}
                      if st.backend == "cpu" else current_cuda_diagnostics())
        desired_backend = (
            "cuda" if st.backend != "cpu" and cuda_state["available"]
            else "cpu"
        )
        if st.backend == "cuda" and not cuda_state["available"]:
            reason = cuda_state["error"] or "CUDA compute preflight failed"
            self.report({"WARNING"}, f"{reason}; using CPU")

        estimate = estimate_bake_memory(dims, st.particles_per_cell)
        ram_available = _system_available_memory_bytes()
        memory_reason = memory_guard_reason(
            estimate, desired_backend, ram_available,
            cuda_state["free_bytes"] if desired_backend == "cuda" else None,
        )
        if memory_reason and desired_backend == "cuda":
            cpu_reason = memory_guard_reason(
                estimate, "cpu", ram_available, None)
            if cpu_reason is None:
                self.report({"WARNING"}, f"{memory_reason} Using CPU instead.")
                desired_backend = "cpu"
                memory_reason = None
            else:
                memory_reason = f"{memory_reason} {cpu_reason}"
        if memory_reason:
            st.bake_status = f"Bake blocked: {memory_reason}"
            self.report({"ERROR"}, memory_reason)
            return False

        try:
            backend, solver = _build_solver(params, desired_backend)
        except Exception as exc:
            if desired_backend != "cuda":
                raise
            cpu_reason = memory_guard_reason(
                estimate, "cpu", ram_available, None)
            if cpu_reason:
                message = (
                    f"CUDA solver initialization failed ({exc}), and CPU "
                    f"fallback is unsafe: {cpu_reason}")
                st.bake_status = f"Bake blocked: {message}"
                self.report({"ERROR"}, message)
                return False
            self.report({"WARNING"},
                        f"CUDA solver initialization failed ({exc}); using CPU")
            backend, solver = _build_solver(params, "cpu")

        cuda_device = None
        if backend.name == "cuda":
            cuda_device = (cuda_state["device"] or cuda_device_name()
                           or "CUDA device 0")
            backend_label = f"CUDA ({cuda_device})"
        else:
            backend_label = "CPU (NumPy)"

        st.bake_status = f"Voxelizing scene for {backend_label}..."
        solid_sdf = None
        solid_node_sdf = None
        if obstacles:
            solid_sdf, solid_node_sdf = voxelize.solid_sdfs_from_objects(
                obstacles, deps, origin, dx, dims)
        if solid_sdf is not None:
            solver.set_solid_sdf(solid_sdf, solid_node_sdf)

        not_solid = (solid_sdf > 0.0) if solid_sdf is not None else None
        seeded = 0
        for obj, velocity_field, _descriptor in liquid_velocity_sources:
            mask = voxelize.mask_from_object(obj, deps, origin, dx, dims)
            if not_solid is not None:
                mask &= not_solid
            seeded += solver.add_liquid_mask(
                mask, velocity_field)
        for obj in inflows:
            mask = voxelize.mask_from_object(obj, deps, origin, dx, dims)
            if not_solid is not None:
                mask &= not_solid
            solver.add_inflow(mask, tuple(obj.stflip.inflow_velocity))
        if seeded == 0 and not inflows:
            self.report({"ERROR"}, "No liquid cells inside the domain")
            return False

        cache_dir = resolve_cache_dir(scene)
        cache.clear(cache_dir)
        from ..stflip import __version__ as stflip_version

        meta = {
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "frame_end_baked": scene.frame_start,
            "dx": dx, "dims": list(dims), "origin": origin.tolist(),
            "backend_requested": st.backend,
            "backend": backend.name,
            "cuda_device": cuda_device,
            "addon_version": stflip_version,
            "scene_units": {
                "length_unit": "blender_unit",
                "system": scene.unit_settings.system,
                "scale_length": scene.unit_settings.scale_length,
            },
            "experiment_profile": None,
            "settings": {
                "resolution": st.resolution,
                "grid_dims": list(dims),
                "target_cfl": st.cfl_target,
                "particles_per_cell": st.particles_per_cell,
                "seed": st.seed,
                "spatiotemporal_sampling": st.st_enabled,
                "jitter_strength": st.jitter_strength,
                "adaptive_gamma": st.adaptive_gamma,
                "eta_phi": st.eta_phi,
                "flip_fraction": st.flip_blend,
                "density": params.rho,
                "local_advection_cfl": params.cfl_local,
                "pcg_tolerance": params.pcg_tol,
                "pcg_max_iterations": params.pcg_max_iter,
                "eps_m": params.eps_m,
                "eps_rho_relative": params.eps_rho_rel,
                "gravity": list(gravity),
                "fps": fps,
                "create_surface": st.create_surface,
                "surface_particle_radius_dx": st.particle_radius,
                "surface_voxel_size_dx": st.surface_voxel,
                "collect_metrics": st.collect_metrics,
                "collect_enstrophy": bool(
                    st.collect_metrics and st.collect_enstrophy),
            },
            "liquid_sources": [
                descriptor
                for _obj, _field, descriptor in liquid_velocity_sources
            ],
            "inflow_sources": [
                {"name": obj.name,
                 "velocity": list(obj.stflip.inflow_velocity)}
                for obj in inflows
            ],
            "solid_boundary": {
                **solver.solid_aperture_stats(),
                "obstacle_count": len(obstacles),
            },
            "version": 4,
        }
        from ..stflip.experiments import profile_provenance

        meta["experiment_profile"] = profile_provenance(
            st.experiment_profile, st)
        if st.collect_metrics:
            from ..stflip.metrics import METRICS_SCHEMA, SCHEMA_VERSION

            meta["metrics"] = {
                "schema": METRICS_SCHEMA,
                "version": SCHEMA_VERSION,
                "file": cache.METRICS_NAME,
                "enstrophy_enabled": st.collect_enstrophy,
            }

        pos, vel = solver.get_render_particles()
        cache.write_frame(cache_dir, scene.frame_start,
                          pos + origin[None, :].astype(np.float32), vel)
        if st.collect_metrics:
            record = _measure_output_frame(
                scene.frame_start, solver, None, pos, vel, None, False)
            cache.append_metric(cache_dir, record)
        cache.write_meta(cache_dir, meta)

        particle_obj = mesher.ensure_particle_object(
            existing_obj=st.particle_object)
        st.particle_object = particle_obj
        if st.create_surface:
            st.surface_object = mesher.ensure_surface_object(
                particle_obj, dx, st.particle_radius, st.surface_voxel,
                existing_obj=st.surface_object)
            _set_surface_enabled(st.surface_object, True)
        else:
            stale_surface = st.surface_object
            if stale_surface is None:
                candidate = bpy.data.objects.get(
                    getattr(mesher, "SURFACE_OBJ", "STFLIP Liquid Surface"))
                if candidate is not None and candidate.name in scene.objects:
                    stale_surface = candidate
                    st.surface_object = candidate
            _set_surface_enabled(stale_surface, False)
        handlers.ensure_registered()

        _BAKE.update(solver=solver, origin=origin.astype(np.float32),
                     cache_dir=cache_dir, meta=meta,
                     frame=scene.frame_start, end=scene.frame_end,
                     backend_label=backend_label,
                     collect_metrics=st.collect_metrics,
                     collect_enstrophy=(st.collect_metrics
                                        and st.collect_enstrophy),
                     running=True)
        scene.frame_set(scene.frame_start)
        st.bake_status = (
            f"Baking on {backend_label}: {seeded} particles seeded...")
        return True

    def _bake_next_frame(self, scene) -> bool:
        """Advance one frame; returns True while frames remain."""
        b = _BAKE
        if b["frame"] >= b["end"]:
            return False
        solver: STFLIPSolver = b["solver"]
        compute_started = time.perf_counter()
        stats = solver.step_frame()
        if b.get("collect_metrics"):
            solver.be.synchronize()
            compute_wall_s = time.perf_counter() - compute_started
        else:
            compute_wall_s = None
        b["frame"] += 1
        pos, vel = solver.get_render_particles()
        cache.write_frame(b["cache_dir"], b["frame"],
                          pos + b["origin"][None, :], vel)
        if b.get("collect_metrics"):
            record = _measure_output_frame(
                b["frame"], solver, stats, pos, vel, compute_wall_s,
                b.get("collect_enstrophy", False),
            )
            cache.append_metric(b["cache_dir"], record)
        b["meta"]["frame_end_baked"] = b["frame"]
        cache.write_meta(b["cache_dir"], b["meta"])
        scene.stflip.bake_status = (
            f"Frame {b['frame']}/{b['end']}  "
            f"({stats.n_particles} pts, {stats.steps} steps, "
            f"{b.get('backend_label', solver.be.name)})")
        scene.frame_set(b["frame"])
        return b["frame"] < b["end"]

    def invoke(self, context, event):
        try:
            if not self._setup(context):
                return {"CANCELLED"}
        except Exception as exc:
            _BAKE["running"] = False
            self.report({"ERROR"}, f"Bake setup failed: {exc}")
            return {"CANCELLED"}
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        """Synchronous bake for scripts and headless (blender -b) use."""
        try:
            if not self._setup(context):
                return {"CANCELLED"}
            while self._bake_next_frame(context.scene):
                pass
        except Exception as exc:
            self.report({"ERROR"}, f"Bake failed: {exc}")
            return {"CANCELLED"}
        finally:
            _BAKE["running"] = False
            _BAKE.pop("solver", None)
        backend_label = _BAKE.get("backend_label", "unknown backend")
        context.scene.stflip.bake_status = (
            f"Bake complete ({_BAKE.get('frame', '?')} frames) on "
            f"{backend_label}")
        return {"FINISHED"}

    def modal(self, context, event):
        if event.type == "ESC":
            return self._finish(context, cancelled=True)
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if not _BAKE.get("running"):
            return self._finish(context, cancelled=True)
        # Any exception must still tear down the timer and _BAKE state, or
        # baking is bricked for the rest of the session.
        try:
            more = self._bake_next_frame(context.scene)
        except Exception as exc:
            self.report({"ERROR"}, f"Bake failed: {exc}")
            return self._finish(context, cancelled=True)
        if not more:
            return self._finish(context, cancelled=False)
        return {"RUNNING_MODAL"}

    def _finish(self, context, cancelled: bool):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        st = context.scene.stflip
        backend_label = _BAKE.get("backend_label", "unknown backend")
        st.bake_status = ("Bake cancelled" if cancelled
                          else f"Bake complete ({_BAKE.get('frame', '?')} "
                               f"frames) on {backend_label}")
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
        n = cache.clear(resolve_cache_dir(context.scene))
        context.scene.stflip.bake_status = ""
        self.report({"INFO"}, f"Removed {n} cache files")
        return {"FINISHED"}


class STFLIP_OT_install_gpu(bpy.types.Operator):
    """Install and compute-test a pinned CuPy runtime for this Blender."""
    bl_idname = "stflip.install_gpu"
    bl_label = "Install GPU Support (CUDA)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        st = context.scene.stflip
        root = _runtime_root(create=True)
        if root is None:
            self.report({"ERROR"}, "Blender user modules directory unavailable")
            return {"CANCELLED"}

        attempts = []
        for candidate in GPU_INSTALL_CANDIDATES:
            install_dir = root / (
                f"{candidate['slug']}-{CUPY_VERSION}-{time.time_ns()}")
            install_dir.mkdir(parents=True, exist_ok=False)
            installing_marker = install_dir / _INSTALLING_RUNTIME_FILE
            installing_marker.write_text(
                f"pid={os.getpid()}\n", encoding="utf-8")
            st.bake_status = f"Installing {candidate['label']}..."
            cmd = [
                sys.executable, "-m", "pip", "install",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                "--upgrade", "--upgrade-strategy", "only-if-needed",
                "--target", str(install_dir),
                candidate["requirement"],
            ]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=1800)
            except Exception as exc:
                attempts.append(
                    f"{candidate['label']} pip launch failed: {exc}")
                _safe_remove_runtime_dir(install_dir, root)
                continue
            if result.returncode != 0:
                attempts.append(
                    f"{candidate['label']} install failed: "
                    f"{_process_failure_detail(result)}")
                _safe_remove_runtime_dir(install_dir, root)
                continue

            # Preserve Blender's own tested NumPy.  The isolated CuPy runtime
            # can resolve that existing module, while CUDA component wheels
            # and cuda-pathfinder remain inside this candidate directory.
            removed_numpy = remove_shadow_numpy(install_dir)
            importlib.invalidate_caches()
            preflight = _preflight_candidate(install_dir)
            if not preflight["available"]:
                attempts.append(
                    f"{candidate['label']} compute preflight failed: "
                    f"{preflight['error'] or 'unknown CUDA error'}")
                _safe_remove_runtime_dir(install_dir, root)
                continue

            _write_active_runtime(install_dir)
            installing_marker.unlink(missing_ok=True)
            _switch_runtime_path(install_dir)
            active = current_cuda_diagnostics(force=True)
            if active["available"]:
                cleanup_inactive_cuda_runtimes(install_dir)
            from . import panels
            panels.invalidate_gpu_state()

            preserved = (f"; preserved Blender NumPy {np.__version__}"
                         if removed_numpy else "")
            device = preflight["device"] or "CUDA device 0"
            if active["available"]:
                device = active["device"] or device
                st.bake_status = f"CUDA ready: {device}"
                self.report(
                    {"INFO"},
                    f"Installed {candidate['requirement']}; CUDA compute "
                    f"passed on {device}{preserved}",
                )
            else:
                # A failed CuPy DLL can remain loaded in a Windows process and
                # cannot be unloaded safely.  The clean subprocess proved the
                # selected runtime; the marker activates it on next launch.
                st.bake_status = "CUDA installed; restart Blender to activate"
                self.report(
                    {"WARNING"},
                    f"Installed {candidate['requirement']} and clean-process "
                    f"compute passed on {device}, but this Blender still has "
                    f"stale CUDA modules ({active['error']}). Restart Blender"
                    f"{preserved}.",
                )
            return {"FINISHED"}

        detail = "; ".join(attempts) or "No compatible wheel was attempted"
        message = (
            "CUDA support installation failed. Update the NVIDIA driver, "
            "confirm Internet/PyPI access, then retry. " + detail)
        st.bake_status = message
        self.report({"ERROR"}, message[-1800:])
        return {"CANCELLED"}


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
