"""Operators: quick setup, bake (modal), free bake, GPU support installer."""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import importlib
import json
import math
import ntpath
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
_SETUP_OBJECT_KEY = "stflip_generated_setup"


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


def relative_cache_needs_saved_blend(cache_dir, blend_filepath) -> bool:
    """Whether a relative cache would move after the .blend is first saved."""
    configured = str(cache_dir or "//stflip_cache")
    is_absolute = os.path.isabs(configured) or ntpath.isabs(configured)
    is_relative = configured.startswith("//") or not is_absolute
    return is_relative and not bool(str(blend_filepath or "").strip())


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


def _canonical_json_bytes(value) -> bytes:
    """Encode simulation descriptors without platform-dependent whitespace."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _fingerprint_array(digest, label: str, value, dtype) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest.update(_canonical_json_bytes({
        "label": label,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }))
    digest.update(memoryview(array).cast("B"))


def simulation_fingerprint(
    params,
    dims,
    dx,
    origin,
    backend_name: str,
    sources,
    solid_sdf=None,
    solid_node_sdf=None,
) -> str:
    """Hash every re-voxelized input that can change a resumed trajectory.

    Source order is significant because liquid seeding consumes the NumPy RNG
    in that order. Object names and display settings are deliberately omitted;
    actual masks, resolved velocity descriptors, and outflow modes are not.
    """
    params_payload = {
        name: getattr(params, name)
        for name in sorted(vars(params))
        if not name.startswith("_")
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes({
        "schema": "stflip-bake-fingerprint",
        "version": 1,
        "params": params_payload,
        "dims": [int(v) for v in dims],
        "dx": float(dx),
        "origin": [float(v) for v in origin],
        "backend": str(backend_name),
        "source_count": len(sources),
    }))
    for index, source in enumerate(sources):
        descriptor = {
            key: value for key, value in source.items() if key != "mask"
        }
        digest.update(_canonical_json_bytes({
            "index": index,
            "descriptor": descriptor,
        }))
        _fingerprint_array(
            digest, f"source[{index}].mask", source["mask"], np.uint8)
    if solid_sdf is None:
        digest.update(b"solid_sdf:none")
    else:
        _fingerprint_array(digest, "solid_sdf", solid_sdf, np.float32)
    if solid_node_sdf is None:
        digest.update(b"solid_node_sdf:none")
    else:
        _fingerprint_array(
            digest, "solid_node_sdf", solid_node_sdf, np.float32)
    return digest.hexdigest()


def fingerprint_matches(expected, actual) -> bool:
    """Constant-time comparison for validated SHA-256 bake fingerprints."""
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    if len(expected) != 64 or len(actual) != 64:
        return False
    return hmac.compare_digest(expected, actual)


def validate_resume_metadata(
    metadata,
    expected_fingerprint: str,
    frame_start: int,
    frame_end: int,
) -> int:
    """Validate the atomic commit marker and return its latest frame."""
    if not isinstance(metadata, dict):
        raise ValueError("cache metadata is missing or corrupt")
    stored_start = metadata.get("frame_start")
    latest = metadata.get("frame_end_baked")
    if (isinstance(stored_start, bool) or not isinstance(stored_start, int)
            or isinstance(latest, bool) or not isinstance(latest, int)
            or stored_start != int(frame_start) or latest < stored_start):
        raise ValueError("cache frame commit range is incompatible")
    if int(frame_end) <= latest:
        raise ValueError(
            f"extend Scene End beyond committed frame {latest} before resuming")
    checkpoint = metadata.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("cache has no resumable checkpoint metadata")
    if (checkpoint.get("schema") != cache.CHECKPOINT_SCHEMA
            or checkpoint.get("version") != cache.CHECKPOINT_VERSION):
        raise ValueError("cache checkpoint schema is unsupported")
    if checkpoint.get("latest_frame") != latest:
        raise ValueError("cache checkpoint commit marker is inconsistent")
    if not fingerprint_matches(
            checkpoint.get("fingerprint"), expected_fingerprint):
        raise ValueError(
            "simulation inputs changed since the checkpoint was created")
    state = checkpoint.get("state")
    if state not in {"RUNNING", "COMPLETE", "CANCELLED", "FAILED"}:
        raise ValueError("cache checkpoint lifecycle state is invalid")
    return latest


def _set_bake_lifecycle(settings, state: str, status: str, *,
                        error: str = "", progress: float | None = None) -> None:
    """Update the durable UI-facing bake state as one coherent snapshot."""
    settings.bake_state = state
    settings.bake_status = status
    settings.bake_error = error
    if progress is not None:
        settings.bake_progress = min(1.0, max(0.0, float(progress)))


def _fail_bake(settings, message: str) -> None:
    _set_bake_lifecycle(
        settings, "FAILED", f"Bake failed: {message}", error=message)


def _source_mask(obj, depsgraph, origin, dx, dims, not_solid=None):
    """Voxelize one source and return a validated bool mask and cell count."""
    mask = np.asarray(
        voxelize.mask_from_object(obj, depsgraph, origin, dx, dims),
        dtype=bool,
    )
    if mask.shape != tuple(dims):
        raise ValueError(
            f"{obj.name}: voxel mask shape {mask.shape!r} does not match "
            f"domain grid {tuple(dims)!r}")
    if not_solid is not None:
        mask &= np.asarray(not_solid, dtype=bool)
    return mask, int(np.count_nonzero(mask))


def _solver_params(settings, dims, dx, gravity, fps):
    """Translate Blender controls to the bpy-free solver parameter object."""
    return Params(
        resolution=dims,
        dx=dx,
        gravity=gravity,
        frame_dt=1.0 / fps,
        cfl_target=settings.cfl_target,
        particles_per_cell=settings.particles_per_cell,
        seed=settings.seed,
        flip_blend=settings.flip_blend,
        st_enabled=settings.st_enabled,
        jitter_strength=settings.jitter_strength,
        adaptive_gamma=settings.adaptive_gamma,
        eta_phi=settings.eta_phi,
        rho=settings.density,
        cfl_local=settings.local_cfl,
        pcg_tol=settings.pcg_tolerance,
        pcg_max_iter=settings.pcg_max_iterations,
        eps_rho_rel=settings.density_floor_relative,
    )


def _scene_setup_provenance(scene):
    setup = scene.get("stflip_setup")
    if setup == "WHIRLPOOL_PREVIEW_APPROXIMATE":
        return {
            "kind": setup,
            "exact_reproduction": False,
            "published_constraints": {
                "domain_dimensions_m": [200.0, 200.0, 80.0],
                "outlet_diameter_m": 20.0,
                "outlet_length_m": 10.0,
                "angular_speed_radians_per_second": 0.1,
            },
            "preview_resolution_longest_axis": int(scene.stflip.resolution),
            "limitations": [
                "preview resolution and particle count",
                "initial water fill height is an explicit preview choice",
                "outlet pressure uses the boundary footprint; the authored "
                "10 m pipe length is reference geometry, not a simulated "
                "conduit",
                "Blender Laplacian surfacing is not paper MCF reconstruction",
                "unpublished production-scene details are not inferred",
            ],
        }
    if setup == "HIGH_CFL_JET_LEAK_APPROXIMATE":
        settings = scene.stflip

        def vector_value(obj, attribute):
            try:
                values = getattr(obj, attribute)
                return [float(value) for value in values]
            except (AttributeError, ReferenceError, TypeError, ValueError):
                return None

        def vector_matches(value, expected):
            try:
                return value is not None and bool(np.allclose(value, expected))
            except (TypeError, ValueError):
                return False

        def generated_role(role):
            for obj in getattr(scene, "objects", ()):
                try:
                    if (obj.get(_SETUP_OBJECT_KEY) == "HIGH_CFL_JET_LEAK"
                            and obj.stflip.role == role):
                        return obj
                except (AttributeError, ReferenceError, TypeError):
                    continue
            return None

        domain = settings.domain
        inflow = generated_role("INFLOW")
        plate = generated_role("OBSTACLE")
        outlet = generated_role("OUTFLOW")
        try:
            dimensions = [float(value) for value in domain.dimensions]
            dx = max(dimensions) / int(settings.resolution)
        except (AttributeError, ReferenceError, TypeError, ValueError, ZeroDivisionError):
            dimensions = None
            dx = None
        try:
            fps = float(scene.render.fps) / float(scene.render.fps_base)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            fps = None
        try:
            velocity = np.asarray(
                inflow.stflip.inflow_velocity, dtype=np.float64)
            jet_speed = float(np.linalg.norm(velocity))
            jet_velocity = [float(value) for value in velocity]
        except (AttributeError, ReferenceError, TypeError, ValueError):
            jet_speed = None
            jet_velocity = None
        try:
            inflow_velocity_mode = str(inflow.stflip.inflow_velocity_mode)
            inflow_rotation_center = [
                float(value)
                for value in inflow.stflip.rotation_center_world
            ]
            inflow_rotation_axis = [
                float(value)
                for value in inflow.stflip.rotation_axis_world
            ]
            inflow_angular_speed = float(inflow.stflip.angular_speed)
        except (AttributeError, ReferenceError, TypeError, ValueError):
            inflow_velocity_mode = None
            inflow_rotation_center = None
            inflow_rotation_axis = None
            inflow_angular_speed = None
        try:
            plate_thickness = min(
                abs(float(value)) for value in plate.dimensions)
        except (AttributeError, ReferenceError, TypeError, ValueError):
            plate_thickness = None
        nominal_cells = (
            jet_speed / (fps * dx)
            if (jet_speed is not None and fps and dx) else None
        )
        thickness_dx = (
            plate_thickness / dx
            if plate_thickness is not None and dx else None
        )
        active_frames = None
        try:
            if inflow.stflip.inflow_use_frame_range:
                active_frames = [
                    int(inflow.stflip.inflow_start_frame),
                    int(inflow.stflip.inflow_end_frame),
                ]
        except (AttributeError, ReferenceError, TypeError, ValueError):
            pass
        current_values = {
            "domain_dimensions_m": dimensions,
            "domain_location_m": vector_value(domain, "location"),
            "resolution_longest_axis": int(settings.resolution),
            "frames_per_second": fps,
            "gravity_enabled": bool(getattr(scene, "use_gravity", False)),
            "gravity_meters_per_second2": vector_value(scene, "gravity"),
            "target_cfl": float(settings.cfl_target),
            "local_collision_cfl": float(settings.local_cfl),
            "particles_per_cell": int(settings.particles_per_cell),
            "spatiotemporal_sampling": bool(settings.st_enabled),
            "jitter_strength": float(settings.jitter_strength),
            "adaptive_gamma": bool(settings.adaptive_gamma),
            "interface_steepness": float(settings.eta_phi),
            "flip_fraction": float(settings.flip_blend),
            "inflow_velocity_mode": inflow_velocity_mode,
            "jet_velocity_meters_per_second": jet_velocity,
            "jet_speed_meters_per_second": jet_speed,
            "inflow_rotation_center_world": inflow_rotation_center,
            "inflow_rotation_axis_world": inflow_rotation_axis,
            "inflow_angular_speed_radians_per_second": (
                inflow_angular_speed),
            "inflow_location_m": vector_value(inflow, "location"),
            "inflow_dimensions_m": vector_value(inflow, "dimensions"),
            "inflow_rotation_radians": vector_value(
                inflow, "rotation_euler"),
            "nominal_jet_cells_per_frame": nominal_cells,
            "plate_location_m": vector_value(plate, "location"),
            "plate_dimensions_m": vector_value(plate, "dimensions"),
            "plate_rotation_radians": vector_value(
                plate, "rotation_euler"),
            "plate_thickness_grid_cells": thickness_dx,
            "outflow_mode": (
                None if outlet is None else str(outlet.stflip.outflow_mode)),
            "outflow_location_m": vector_value(outlet, "location"),
            "outflow_dimensions_m": vector_value(outlet, "dimensions"),
            "active_frames_inclusive": active_frames,
        }
        preset_intact = (
            dimensions is not None
            and np.allclose(dimensions, (6.0, 6.0, 6.0))
            and vector_matches(
                current_values["domain_location_m"], (0.0, 0.0, 3.0))
            and int(settings.resolution) == 48
            and fps is not None and math.isclose(fps, 24.0)
            and current_values["gravity_enabled"] is True
            and vector_matches(
                current_values["gravity_meters_per_second2"],
                (0.0, 0.0, -9.81),
            )
            and math.isclose(float(settings.cfl_target), 16.0)
            and math.isclose(float(settings.local_cfl), 1.0)
            and int(settings.particles_per_cell) == 8
            and bool(settings.st_enabled) is True
            and math.isclose(float(settings.jitter_strength), 1.0)
            and bool(settings.adaptive_gamma) is True
            and math.isclose(float(settings.eta_phi), 0.5)
            and math.isclose(
                float(settings.flip_blend), 0.98,
                rel_tol=1e-6, abs_tol=1e-7,
            )
            and inflow_velocity_mode == "UNIFORM"
            and jet_speed is not None and math.isclose(jet_speed, 48.0)
            and vector_matches(jet_velocity, (0.0, 0.0, -48.0))
            and vector_matches(
                current_values["inflow_location_m"], (0.0, 0.0, 5.5))
            and vector_matches(
                current_values["inflow_dimensions_m"], (1.0, 1.0, 0.5))
            and vector_matches(
                current_values["inflow_rotation_radians"], (0.0, 0.0, 0.0))
            and nominal_cells is not None
            and math.isclose(nominal_cells, 16.0, rel_tol=1e-6)
            and vector_matches(
                current_values["plate_location_m"], (0.0, 0.0, 2.0))
            and vector_matches(
                current_values["plate_dimensions_m"], (4.0, 4.0, 0.125))
            and vector_matches(
                current_values["plate_rotation_radians"], (0.0, 0.0, 0.0))
            and thickness_dx is not None
            and math.isclose(thickness_dx, 1.0, rel_tol=1e-6)
            and current_values["outflow_mode"] == "PRESSURE"
            and vector_matches(
                current_values["outflow_location_m"], (0.0, 0.0, 0.0625))
            and vector_matches(
                current_values["outflow_dimensions_m"],
                (5.75, 5.75, 0.125),
            )
            and active_frames == [2, 48]
        )
        return {
            "kind": setup,
            "exact_reproduction": False,
            "preset_intact": bool(preset_intact),
            "paper_figure": 21,
            "published_constraints": {
                "target_cfl": 16.0,
                "obstacle_thickness_grid_cells": 1.0,
                "local_collision_cfl": 1.0,
            },
            "preview_choices": {
                "domain_dimensions_m": [6.0, 6.0, 6.0],
                "resolution_longest_axis": 48,
                "frames_per_second": 24.0,
                "gravity_meters_per_second2": [0.0, 0.0, -9.81],
                "particles_per_cell": 8,
                "spatiotemporal_sampling": True,
                "jitter_strength": 1.0,
                "adaptive_gamma": True,
                "interface_steepness": 0.5,
                "flip_fraction": 0.98,
                "inflow_velocity_mode": "UNIFORM",
                "jet_diameter_m": 1.0,
                "jet_speed_meters_per_second": 48.0,
                "plate_dimensions_m": [4.0, 4.0, 0.125],
                "outflow_mode": "PRESSURE",
                "active_frames_inclusive": [2, 48],
            },
            "current_values": current_values,
            "limitations": [
                "the paper does not publish the exact domain, nozzle, speed, "
                "plate span, duration, or camera parameters",
                "the plate is static; moving and deforming obstacles are not "
                "supported",
                "the refill source is an impinging jet, not a physical "
                "pressure/head-controlled leak model",
                "Blender Laplacian surfacing is not paper MCF reconstruction",
            ],
        }
    return None


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


def _resolve_source_velocity(
    mode,
    linear_velocity,
    rotation_center_world,
    rotation_axis_world,
    angular_speed,
    domain_origin,
    source_name,
    *,
    velocity_label: str,
    mode_label: str,
    velocity_key: str,
    mode_key: str,
):
    """Validate shared liquid/inflow controls and return the actual field.

    Descriptors are built from the normalized field passed to the solver, not
    directly from Blender RNA values.  Cache metadata and bake fingerprints
    therefore describe the float32 values that actually seed particles.
    """
    source_name = str(source_name)
    linear = _finite_vector3(
        linear_velocity, velocity_label, source_name)
    origin = _finite_vector3(domain_origin, "Domain Origin", source_name)
    mode = str(mode)

    if mode == "UNIFORM":
        field = UniformVelocity(tuple(linear))
        return field, {
            "name": source_name,
            velocity_key: list(field.value),
            mode_key: mode,
        }
    if mode != "SOLID_BODY":
        raise ValueError(
            f"{source_name}: unknown {mode_label} {mode!r}")

    center_world = _finite_vector3(
        rotation_center_world, "Rotation Center", source_name)
    axis_authored = _finite_vector3(
        rotation_axis_world, "Rotation Axis", source_name)
    axis_length = float(np.linalg.norm(axis_authored))
    if not np.isfinite(axis_length) or axis_length <= 1e-12:
        raise ValueError(f"{source_name}: Rotation Axis must be non-zero")
    try:
        angular_speed = float(angular_speed)
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
        velocity_key: list(field.linear_velocity),
        mode_key: mode,
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


def resolve_liquid_initial_velocity(settings, domain_origin, source_name):
    """Resolve a liquid source's world-space initial-velocity controls."""
    return _resolve_source_velocity(
        settings.initial_velocity_mode,
        settings.initial_velocity,
        settings.rotation_center_world,
        settings.rotation_axis_world,
        settings.angular_speed,
        domain_origin,
        source_name,
        velocity_label="Initial Velocity",
        mode_label="Initial Velocity Mode",
        velocity_key="initial_velocity",
        mode_key="initial_velocity_mode",
    )


def resolve_inflow_velocity(settings, domain_origin, source_name):
    """Resolve an inflow's world-space controls through the shared validator."""
    return _resolve_source_velocity(
        settings.inflow_velocity_mode,
        settings.inflow_velocity,
        settings.rotation_center_world,
        settings.rotation_axis_world,
        settings.angular_speed,
        domain_origin,
        source_name,
        velocity_label="Inflow Velocity",
        mode_label="Inflow Velocity Mode",
        velocity_key="velocity",
        mode_key="velocity_mode",
    )


def _resolved_velocity_fingerprint(descriptor, velocity_key, mode_key):
    """Return only normalized values that can affect seeded particle state."""
    payload = {
        mode_key: descriptor[mode_key],
        velocity_key: descriptor[velocity_key],
    }
    if descriptor[mode_key] == "SOLID_BODY":
        rotation = descriptor["solid_body_rotation"]
        payload["solid_body_rotation"] = {
            "center_solver_local": rotation["center_solver_local"],
            "angular_velocity_world": rotation["angular_velocity_world"],
        }
    return payload


def _integer_frame(value, label: str, source_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source_name}: {label} must be an integer")
    try:
        frame = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{source_name}: {label} must be an integer") from exc
    if frame != value:
        raise ValueError(f"{source_name}: {label} must be an integer")
    return frame


def resolve_inflow_schedule(
    settings,
    scene_frame_start,
    scene_frame_end,
    frame_dt,
    source_name,
):
    """Translate inclusive *evolved output frames* to solver intervals.

    The cache's first frame is a pre-step snapshot. To make authored frame N
    visible at output N, its refill occurs during the preceding N-1 -> N
    interval. The solver schedule is start-inclusive/end-exclusive; ranges
    with no evolved output (including a range containing only the initial
    snapshot) become the valid inactive interval ``[0, 0)``. The descriptor
    excludes the mutable requested bake end so extending and resuming does not
    alter the fingerprint.
    """
    source_name = str(source_name)
    try:
        dt = float(frame_dt)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source_name}: Frame Duration must be finite") from exc
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"{source_name}: Frame Duration must be positive")
    bake_start = _integer_frame(
        scene_frame_start, "Scene Start Frame", source_name)
    bake_end = _integer_frame(scene_frame_end, "Scene End Frame", source_name)

    if not bool(settings.inflow_use_frame_range):
        return 0.0, None, {
            "active_frame_range": {
                "mode": "UNBOUNDED",
                "authored_inclusive": None,
                "effective_inclusive": None,
                "solver_time_seconds": {
                    "start_inclusive": 0.0,
                    "end_exclusive": None,
                },
            },
        }, None

    start_frame = _integer_frame(
        settings.inflow_start_frame, "Inflow Start Frame", source_name)
    end_frame = _integer_frame(
        settings.inflow_end_frame, "Inflow End Frame", source_name)
    if start_frame > end_frame:
        raise ValueError(
            f"{source_name}: Inflow Start Frame must not exceed End Frame")

    start_time = max(0.0, (start_frame - bake_start - 1) * dt)
    end_time = max(0.0, (end_frame - bake_start) * dt)
    first_evolved_frame = bake_start + 1
    effective_frames = (
        None if end_frame < first_evolved_frame
        else [max(start_frame, first_evolved_frame), end_frame]
    )
    descriptor = {
        "active_frame_range": {
            "mode": "LIMITED",
            "authored_inclusive": [start_frame, end_frame],
            "effective_inclusive": effective_frames,
            "solver_time_seconds": {
                "start_inclusive": start_time,
                "end_exclusive": end_time,
            },
        },
    }

    overlap_start = max(start_frame, first_evolved_frame)
    overlap_end = min(end_frame, bake_end)
    warning = None
    if overlap_start > overlap_end:
        warning = (
            f"Inflow {source_name!r} active frames {start_frame}-{end_frame} "
            f"do not overlap evolved outputs {first_evolved_frame}-"
            f"{bake_end}; the source is deliberately inactive for this bake"
        )
    elif start_frame < bake_start or end_frame > bake_end:
        warning = (
            f"Inflow {source_name!r} active frames {start_frame}-{end_frame} "
            f"extend outside evolved outputs {first_evolved_frame}-"
            f"{bake_end}; this "
            f"bake uses the inclusive overlap {overlap_start}-{overlap_end}"
        )
    return start_time, end_time, descriptor, warning


def _inflow_schedule_overlaps(start_time, end_time, duration) -> bool:
    """Return whether a half-open source interval overlaps this bake."""
    start = float(start_time)
    stop = None if end_time is None else float(end_time)
    span = max(0.0, float(duration))
    return start < span and (stop is None or stop > 0.0) and (
        stop is None or start < stop
    )


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


def _remove_generated_setup_objects(scene) -> int:
    """Replace only objects authored by this add-on's one-click setups."""
    removed = 0
    for obj in list(scene.objects):
        try:
            generated = bool(obj.get(_SETUP_OBJECT_KEY, ""))
        except (AttributeError, ReferenceError, TypeError):
            generated = False
        if not generated:
            continue
        try:
            users_scene = list(obj.users_scene)
        except (AttributeError, ReferenceError, TypeError):
            users_scene = [scene]
        if len(users_scene) <= 1:
            bpy.data.objects.remove(obj, do_unlink=True)
        else:
            try:
                if obj.name in scene.collection.objects:
                    scene.collection.objects.unlink(obj)
            except (AttributeError, ReferenceError, RuntimeError, TypeError):
                continue
        removed += 1
    for key in ("stflip_setup", "stflip_paper_reference"):
        try:
            if key in scene:
                del scene[key]
        except (AttributeError, KeyError, ReferenceError, TypeError):
            pass
    return removed


def _clear_bake_for_new_setup(scene) -> int:
    """Clear this scene's owned bake before replacing authored inputs.

    A one-click setup changes trajectory-defining geometry. Leaving its old
    cache accessible would let the frame handler display stale particles and
    downstream export package the previous setup. Foreign or invalid custom
    cache ownership is never deleted implicitly.
    """
    ensure_cache_id = getattr(handlers, "ensure_scene_cache_id", None)
    if ensure_cache_id is not None:
        ensure_cache_id(scene)
    ownership_check = getattr(handlers, "scene_cache_ownership", None)
    ownership = (
        ownership_check(scene) if ownership_check is not None else "legacy"
    )
    if ownership in {"foreign", "invalid"}:
        raise ValueError(
            "cannot replace the setup while Cache Directory ownership is "
            f"{ownership}; choose a new cache path or use its owning scene"
        )
    removed = cache.clear(resolve_cache_dir(scene))
    clear_output = getattr(handlers, "clear_scene_output", None)
    if clear_output is not None:
        clear_output(scene)
    _set_bake_lifecycle(scene.stflip, "IDLE", "", progress=0.0)
    return removed


def _owned_setup_cache_file_count(scene) -> int:
    """Count files a one-click setup would irreversibly remove.

    This mirrors :func:`stflip.cache.clear` without opening or deleting
    anything. Foreign and invalid cache directories are never offered for
    deletion and remain subject to the stricter execute-time refusal.
    """
    ownership_check = getattr(handlers, "scene_cache_ownership", None)
    ownership = (
        ownership_check(scene) if ownership_check is not None else "legacy"
    )
    if ownership not in {"owned", "legacy", "missing"}:
        return 0
    cache_dir = resolve_cache_dir(scene)
    if not os.path.isdir(cache_dir):
        return 0
    try:
        names = os.listdir(cache_dir)
    except OSError:
        return 0
    return sum(
        1
        for name in names
        if (
            (name.startswith("stflip_") and name.endswith(".npz"))
            or name == cache.META_NAME
            or name == cache.METRICS_NAME
            or name.startswith(".stflip-writing-")
            or name.startswith(".stflip-exporting-")
        )
    )


def _invoke_setup_replace_confirmation(operator, context, event):
    """Confirm the disk mutation that Blender's object Undo cannot restore."""
    file_count = _owned_setup_cache_file_count(context.scene)
    if not file_count:
        return operator.execute(context)
    message = (
        f"This will permanently delete {file_count} cached ST-FLIP file"
        f"{'s' if file_count != 1 else ''}. Blender Undo cannot restore "
        "the bake."
    )
    try:
        return context.window_manager.invoke_confirm(
            operator,
            event,
            title="Replace ST-FLIP Setup?",
            message=message,
            confirm_text="Delete Bake and Replace",
            icon="ERROR",
        )
    except TypeError:
        # Older Blender confirmation signatures still show bl_description,
        # which also states that the cache deletion cannot be undone.
        return context.window_manager.invoke_confirm(operator, event)


class STFLIP_OT_quick_setup(bpy.types.Operator):
    """Create a ready-to-bake dam-break scene (domain, liquid, roles)"""
    bl_idname = "stflip.quick_setup"
    bl_label = "Quick Dam-Break Setup"
    bl_description = (
        "Replace generated setup objects and this scene's owned bake with a "
        "ready-to-bake dam break; deleting cached bake files cannot be undone"
    )
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        return _invoke_setup_replace_confirmation(self, context, event)

    def execute(self, context):
        if _BAKE.get("running"):
            self.report({"WARNING"}, "Cancel the running bake first")
            return {"CANCELLED"}
        scene = context.scene
        try:
            removed_cache_files = _clear_bake_for_new_setup(scene)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _remove_generated_setup_objects(scene)

        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 1))
        domain = context.active_object
        domain.name = "STFLIP Domain"
        domain.display_type = "WIRE"
        domain.hide_render = True
        domain[_SETUP_OBJECT_KEY] = "DAM_BREAK"

        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(-0.65, 0, 0.55))
        liquid = context.active_object
        liquid.name = "STFLIP Liquid"
        liquid.scale = (0.34, 0.98, 0.54)
        liquid.display_type = "WIRE"
        liquid.hide_render = True
        liquid.stflip.role = "LIQUID"
        liquid[_SETUP_OBJECT_KEY] = "DAM_BREAK"

        scene.stflip.domain = domain
        scene.frame_start = 1
        scene.frame_end = 48
        detail = (
            f"; cleared {removed_cache_files} previous bake files"
            if removed_cache_files else ""
        )
        self.report({"INFO"}, f"Dam-break scene created; press Bake{detail}")
        return {"FINISHED"}


class STFLIP_OT_whirlpool_preview(bpy.types.Operator):
    """Create a practical approximation of the paper's whirlpool scene."""
    bl_idname = "stflip.whirlpool_preview"
    bl_label = "Whirlpool Preview (Approx.)"
    bl_description = (
        "Create the paper's published 200 x 200 x 80 m proportions, 20 m "
        "diameter x 10 m outlet, and 0.1 rad/s rotation at preview resolution; "
        "changes scene units/gravity/range and clears its owned bake; this is "
        "not the paper's exact production scene or MCF reconstruction; "
        "deleted cache files cannot be restored by Undo"
    )
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        return _invoke_setup_replace_confirmation(self, context, event)

    def execute(self, context):
        if _BAKE.get("running"):
            self.report({"WARNING"}, "Cancel the running bake first")
            return {"CANCELLED"}
        scene = context.scene
        try:
            removed_cache_files = _clear_bake_for_new_setup(scene)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _remove_generated_setup_objects(scene)

        scene.unit_settings.system = "METRIC"
        scene.unit_settings.scale_length = 1.0
        scene.gravity = (0.0, 0.0, -9.81)
        scene.use_gravity = True

        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 40.0))
        domain = context.active_object
        domain.name = "STFLIP Whirlpool Preview Domain 200x200x80m"
        domain.scale = (100.0, 100.0, 40.0)
        domain.display_type = "WIRE"
        domain.hide_render = True
        domain[_SETUP_OBJECT_KEY] = "WHIRLPOOL"
        domain["stflip_paper_dimensions_m"] = (200.0, 200.0, 80.0)

        # Leave a bottom clearance and an explicit air band above the water.
        # The paper does not publish its initial fill height; keeping several
        # preview cells empty makes the free surface visible without claiming
        # that this inferred height reproduces the production scene.
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 34.5))
        liquid = context.active_object
        liquid.name = "STFLIP Whirlpool Preview Liquid"
        liquid.scale = (99.0, 99.0, 33.5)
        liquid.display_type = "WIRE"
        liquid.hide_render = True
        liquid.stflip.role = "LIQUID"
        liquid[_SETUP_OBJECT_KEY] = "WHIRLPOOL"
        liquid.stflip.initial_velocity_mode = "SOLID_BODY"
        liquid.stflip.initial_velocity = (0.0, 0.0, 0.0)
        liquid.stflip.rotation_center_world = (0.0, 0.0, 0.0)
        liquid.stflip.rotation_axis_world = (0.0, 0.0, 1.0)
        liquid.stflip.angular_speed = 0.1

        # The paper describes a centered bottom pipe. The mesh preserves its
        # published 10 m reference length, while the pressure solver uses only
        # its circular footprint on the exterior boundary (not conduit flow).
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=64, radius=10.0, depth=10.0,
            location=(0.0, 0.0, 5.0),
        )
        outlet = context.active_object
        outlet.name = "STFLIP Whirlpool Preview Outlet D20x10m"
        outlet.display_type = "WIRE"
        outlet.hide_render = True
        outlet.stflip.role = "OUTFLOW"
        outlet.stflip.outflow_mode = "PRESSURE"
        outlet[_SETUP_OBJECT_KEY] = "WHIRLPOOL"
        outlet["stflip_paper_pipe_diameter_m"] = 20.0
        outlet["stflip_paper_pipe_length_m"] = 10.0

        settings = scene.stflip
        settings.domain = domain
        settings.resolution = 48
        settings.particles_per_cell = 4
        settings.cfl_target = 15.0
        settings.create_surface = True
        scene.frame_start = 1
        scene.frame_end = 48
        scene["stflip_setup"] = "WHIRLPOOL_PREVIEW_APPROXIMATE"
        scene["stflip_paper_reference"] = (
            "Whirlpool: 200x200x80 m, D20x10 m bottom outlet, omega=0.1 rad/s"
        )
        _set_bake_lifecycle(
            settings,
            "IDLE",
            "Approx. whirlpool preview created; review settings, then Bake",
            progress=0.0,
        )
        self.report(
            {"INFO"},
            "Approximate whirlpool preview created at 48-cell resolution; "
            "published geometry/rotation retained, production scale and MCF "
            "surface reconstruction are not reproduced"
            + (f"; cleared {removed_cache_files} previous bake files"
               if removed_cache_files else ""),
        )
        return {"FINISHED"}


class STFLIP_OT_high_cfl_jet_leak(bpy.types.Operator):
    """Create a practical approximation of the paper's thin-plate jet."""

    bl_idname = "stflip.high_cfl_jet_leak"
    bl_label = "High-CFL Jet Preview (Approx.)"
    bl_description = (
        "Create a static Figure 21-style water jet at target CFL 16 with a "
        "one-cell plate and pressure outflow; changes units/FPS/gravity/range "
        "and clears the owned bake. Exact parameters are unpublished; deleted "
        "cache files cannot be restored by Undo"
    )
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        return _invoke_setup_replace_confirmation(self, context, event)

    def execute(self, context):
        if _BAKE.get("running"):
            self.report({"WARNING"}, "Cancel the running bake first")
            return {"CANCELLED"}
        scene = context.scene
        try:
            removed_cache_files = _clear_bake_for_new_setup(scene)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _remove_generated_setup_objects(scene)

        # Figure 21 publishes CFL=16 and a plate thickness of one grid cell,
        # but not its world dimensions or jet speed.  These preview choices
        # make one 24 fps frame correspond to a nominal 16-cell jet travel.
        domain_size = 6.0
        resolution = 48
        fps = 24
        dx = domain_size / resolution
        jet_speed = 16.0 * dx * fps

        scene.unit_settings.system = "METRIC"
        scene.unit_settings.scale_length = 1.0
        scene.render.fps = fps
        scene.render.fps_base = 1.0
        scene.gravity = (0.0, 0.0, -9.81)
        scene.use_gravity = True

        bpy.ops.mesh.primitive_cube_add(
            size=2.0, location=(0.0, 0.0, domain_size / 2.0))
        domain = context.active_object
        domain.name = "STFLIP High-CFL Jet Domain (Approx.)"
        domain.scale = (domain_size / 2.0,) * 3
        domain.display_type = "WIRE"
        domain.hide_render = True
        domain[_SETUP_OBJECT_KEY] = "HIGH_CFL_JET_LEAK"
        domain["stflip_preview_dimensions_m"] = (
            domain_size, domain_size, domain_size)
        domain["stflip_preview_dx_m"] = dx

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=64,
            radius=0.5,
            depth=0.5,
            location=(0.0, 0.0, 5.5),
        )
        inflow = context.active_object
        inflow.name = "STFLIP High-CFL Jet Inflow (Approx.)"
        inflow.display_type = "WIRE"
        inflow.hide_render = True
        inflow.stflip.role = "INFLOW"
        inflow.stflip.inflow_velocity_mode = "UNIFORM"
        inflow.stflip.inflow_velocity = (0.0, 0.0, -jet_speed)
        inflow.stflip.inflow_use_frame_range = True
        # Frame 1 is the pre-step cache snapshot. Frame 2 is the first evolved
        # output and still seeds at solver time zero.
        inflow.stflip.inflow_start_frame = 2
        inflow.stflip.inflow_end_frame = 48
        inflow[_SETUP_OBJECT_KEY] = "HIGH_CFL_JET_LEAK"
        inflow["stflip_preview_jet_diameter_m"] = 1.0
        inflow["stflip_preview_jet_speed_meters_per_second"] = jet_speed
        inflow["stflip_preview_nominal_cells_per_frame"] = 16.0

        bpy.ops.mesh.primitive_cube_add(
            size=2.0, location=(0.0, 0.0, 2.0))
        plate = context.active_object
        plate.name = "STFLIP One-Cell Jet Plate (Static, Approx.)"
        plate.scale = (2.0, 2.0, dx / 2.0)
        plate.display_type = "WIRE"
        plate.hide_render = True
        plate.stflip.role = "OBSTACLE"
        plate[_SETUP_OBJECT_KEY] = "HIGH_CFL_JET_LEAK"
        plate["stflip_published_thickness_dx"] = 1.0
        plate["stflip_preview_thickness_m"] = dx
        plate["stflip_static_obstacle"] = True

        # A bottom pressure footprint lets runoff leave without deleting the
        # impact region or pretending that a modeled drain conduit exists.
        bpy.ops.mesh.primitive_cube_add(
            size=2.0, location=(0.0, 0.0, dx / 2.0))
        outlet = context.active_object
        outlet.name = "STFLIP Jet Bottom Pressure Outflow (Approx.)"
        outlet.scale = (
            domain_size / 2.0 - dx,
            domain_size / 2.0 - dx,
            dx / 2.0,
        )
        outlet.display_type = "WIRE"
        outlet.hide_render = True
        outlet.stflip.role = "OUTFLOW"
        outlet.stflip.outflow_mode = "PRESSURE"
        outlet[_SETUP_OBJECT_KEY] = "HIGH_CFL_JET_LEAK"
        outlet["stflip_safe_runoff_outflow"] = True

        settings = scene.stflip
        settings.domain = domain
        settings.resolution = resolution
        settings.cfl_target = 16.0
        settings.local_cfl = 1.0
        settings.particles_per_cell = 8
        settings.st_enabled = True
        settings.jitter_strength = 1.0
        settings.adaptive_gamma = True
        settings.eta_phi = 0.5
        settings.flip_blend = 0.98
        settings.create_surface = True
        scene.frame_start = 1
        scene.frame_end = 48
        scene["stflip_setup"] = "HIGH_CFL_JET_LEAK_APPROXIMATE"
        scene["stflip_paper_reference"] = (
            "Figure 21: target CFL=16, static plate thickness=one grid cell; "
            "other scene parameters are unpublished preview choices"
        )
        _set_bake_lifecycle(
            settings,
            "IDLE",
            "Approx. high-CFL jet preview created; review settings, then Bake",
            progress=0.0,
        )
        self.report(
            {"INFO"},
            "Approximate Figure 21-style jet created at target CFL 16; exact "
            "jet geometry and speed are unpublished and not reproduced"
            + (f"; cleared {removed_cache_files} previous bake files"
               if removed_cache_files else ""),
        )
        return {"FINISHED"}


class STFLIP_OT_bake(bpy.types.Operator):
    """Bake the ST-FLIP simulation for the scene frame range"""
    bl_idname = "stflip.bake"
    bl_label = "Bake Simulation"
    bl_options = {"REGISTER"}

    _timer = None
    _is_resume = False

    def _setup(self, context) -> bool:
        """Voxelize inputs and start a new bake or restore a committed one."""
        scene = context.scene
        st = scene.stflip
        is_resume = bool(self._is_resume)
        if _BAKE.get("running"):
            self.report({"WARNING"}, "A bake is already running")
            return False
        if st.domain is None:
            message = "Set a domain object first"
            _fail_bake(st, message)
            self.report({"ERROR"}, message)
            return False
        if relative_cache_needs_saved_blend(
                st.cache_dir, getattr(bpy.data, "filepath", "")):
            message = (
                "Save the .blend before baking with a relative cache, or "
                "choose an absolute Cache Directory"
            )
            _fail_bake(st, message)
            self.report({"ERROR"}, message)
            return False

        # Geometry, modifiers and animated transforms must all be evaluated at
        # the cache's first frame.  Doing this before obtaining the depsgraph
        # prevents a bake launched from another timeline frame from capturing
        # the wrong source shapes.
        scene.frame_set(scene.frame_start)
        _set_bake_lifecycle(
            st, "RUNNING",
            "Preparing resume..." if is_resume else "Preparing bake...",
            progress=0.0,
        )

        liquids = _fluid_objects(scene, "LIQUID")
        inflows = _fluid_objects(scene, "INFLOW")
        outflows = _fluid_objects(scene, "OUTFLOW")
        obstacles = _fluid_objects(scene, "OBSTACLE")
        if not liquids and not inflows:
            message = "Mark at least one mesh as Liquid or Inflow"
            _fail_bake(st, message)
            self.report({"ERROR"}, message)
            return False

        deps = context.evaluated_depsgraph_get()
        dims, dx, origin = voxelize.domain_grid(st.domain, st.resolution)
        fps = scene.render.fps / scene.render.fps_base
        gravity = tuple(scene.gravity) if scene.use_gravity else (0.0, 0.0, 0.0)
        params = _solver_params(st, dims, dx, gravity, fps)
        try:
            liquid_velocity_sources = [
                (
                    obj,
                    *resolve_liquid_initial_velocity(
                        obj.stflip, origin, obj.name),
                )
                for obj in liquids
            ]
            inflow_velocity_sources = []
            for obj in inflows:
                velocity_field, descriptor = resolve_inflow_velocity(
                    obj.stflip, origin, obj.name)
                start_time, end_time, schedule, warning = (
                    resolve_inflow_schedule(
                        obj.stflip,
                        scene.frame_start,
                        scene.frame_end,
                        params.frame_dt,
                        obj.name,
                    )
                )
                inflow_velocity_sources.append((
                    obj,
                    velocity_field,
                    {**descriptor, **schedule},
                    start_time,
                    end_time,
                    warning,
                ))
        except ValueError as exc:
            message = str(exc)
            _fail_bake(st, message)
            self.report({"ERROR"}, message)
            return False

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
            _fail_bake(st, memory_reason)
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
                _fail_bake(st, message)
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
        liquid_records = []
        fingerprint_sources = []
        for obj, velocity_field, descriptor in liquid_velocity_sources:
            mask, cell_count = _source_mask(
                obj, deps, origin, dx, dims, not_solid)
            descriptor = {**descriptor, "cell_count": cell_count}
            liquid_records.append(descriptor)
            fingerprint_sources.append({
                "role": "LIQUID",
                "velocity": _resolved_velocity_fingerprint(
                    descriptor,
                    "initial_velocity",
                    "initial_velocity_mode",
                ),
                "mask": mask,
            })
            if cell_count == 0:
                self.report(
                    {"WARNING"},
                    f"Liquid {obj.name!r} covers no usable domain cells",
                )
                continue
            if not is_resume:
                seeded += solver.add_liquid_mask(mask, velocity_field)

        inflow_records = []
        usable_inflow_cells = 0
        active_inflow_cells = 0
        requested_duration = max(
            0.0,
            (int(scene.frame_end) - int(scene.frame_start)) * params.frame_dt,
        )
        for (
            obj,
            velocity_field,
            descriptor,
            start_time,
            end_time,
            schedule_warning,
        ) in inflow_velocity_sources:
            mask, cell_count = _source_mask(
                obj, deps, origin, dx, dims, not_solid)
            descriptor = {**descriptor, "cell_count": cell_count}
            inflow_records.append(descriptor)
            fingerprint_sources.append({
                "role": "INFLOW",
                "velocity": _resolved_velocity_fingerprint(
                    descriptor, "velocity", "velocity_mode"),
                "active_frame_range": descriptor["active_frame_range"],
                "mask": mask,
            })
            if schedule_warning:
                self.report({"WARNING"}, schedule_warning)
            if cell_count == 0:
                self.report(
                    {"WARNING"},
                    f"Inflow {obj.name!r} covers no usable domain cells",
                )
                continue
            usable_inflow_cells += cell_count
            solver.add_inflow(
                mask,
                velocity_field,
                start_time=start_time,
                end_time=end_time,
            )
            if _inflow_schedule_overlaps(
                    start_time, end_time, requested_duration):
                active_inflow_cells += cell_count

        outflow_records = []
        for obj in outflows:
            mask, cell_count = _source_mask(
                obj, deps, origin, dx, dims, not_solid)
            mode = str(obj.stflip.outflow_mode)
            if mode not in {"VOLUME", "PRESSURE"}:
                raise ValueError(
                    f"{obj.name}: unsupported Outflow Mode {mode!r}")
            outflow_records.append({
                "name": obj.name,
                "mode": mode,
                "cell_count": cell_count,
            })
            fingerprint_sources.append({
                "role": "OUTFLOW",
                "mode": mode,
                "mask": mask,
            })
            if cell_count == 0:
                self.report(
                    {"WARNING"},
                    f"Outflow {obj.name!r} covers no usable domain cells",
                )
                continue
            solver.add_outflow(mask, mode=mode)

        initial_outflow_cull = (
            solver.cull_outflows() if not is_resume else {})
        seeded = int(solver.pos.shape[0])
        if not is_resume and seeded == 0 and usable_inflow_cells == 0:
            message = "No usable Liquid or Inflow cells exist inside the domain"
            _fail_bake(st, message)
            self.report({"ERROR"}, message)
            return False
        if (not is_resume and seeded == 0 and usable_inflow_cells > 0
                and active_inflow_cells == 0):
            self.report(
                {"WARNING"},
                "All usable inflows are inactive for the requested frame "
                "range; playback remains empty until a later scheduled "
                "output is included",
            )

        # Only clear a previous cache after every source has been validated and
        # integrated into a live solver.  A bad/empty source therefore cannot
        # destroy the user's last usable bake.
        cache_dir = resolve_cache_dir(scene)
        ensure_cache_id = getattr(handlers, "ensure_scene_cache_id", None)
        cache_owner_id = (
            ensure_cache_id(scene) if ensure_cache_id is not None
            else getattr(st, "cache_id", "")
        )
        ownership_check = getattr(handlers, "scene_cache_ownership", None)
        ownership = (
            ownership_check(scene) if ownership_check is not None else "missing"
        )
        refused_ownership = (
            ownership != "owned" if is_resume
            else ownership in {"foreign", "invalid"}
        )
        if refused_ownership:
            message = (
                f"Cache ownership is {ownership}; "
                + ("resume requires the scene that created this checkpoint"
                   if is_resume else
                   "choose a different Cache Directory or use the scene that "
                   "created this cache"))
            _fail_bake(st, message)
            self.report({"ERROR"}, message)
            return False
        fingerprint = simulation_fingerprint(
            params,
            dims,
            dx,
            origin,
            backend.name,
            fingerprint_sources,
            solid_sdf,
            solid_node_sdf,
        )
        existing_meta = cache.read_meta(cache_dir)
        if not is_resume:
            cache.clear(cache_dir)
        from ..stflip import __version__ as stflip_version

        if (is_resume and isinstance(existing_meta, dict)
                and existing_meta.get("addon_version") != stflip_version):
            raise ValueError(
                "checkpoint was created by a different ST-FLIP add-on "
                "version; rebake with the current version")

        setup_provenance = _scene_setup_provenance(scene)
        if (isinstance(setup_provenance, dict)
                and setup_provenance.get("kind")
                == "HIGH_CFL_JET_LEAK_APPROXIMATE"
                and not setup_provenance.get("preset_intact", False)):
            self.report(
                {"WARNING"},
                "High-CFL Jet Preview ratios changed; cache metadata records "
                "current values, but this is no longer the authored preset",
            )

        new_meta = {
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "frame_end_baked": scene.frame_start,
            "dx": dx, "dims": list(dims), "origin": origin.tolist(),
            "backend_requested": st.backend,
            "backend": backend.name,
            "cuda_device": cuda_device,
            "addon_version": stflip_version,
            "cache_owner_id": cache_owner_id,
            "scene_units": {
                "length_unit": "blender_unit",
                "system": scene.unit_settings.system,
                "scale_length": scene.unit_settings.scale_length,
            },
            "scene_setup": setup_provenance,
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
                "surface_geometric_smoothing": st.surface_smoothing,
                "surface_smoothing_iterations": (
                    st.surface_smoothing_iterations),
                "surface_smoothing_factor": st.surface_smoothing_factor,
                "collect_metrics": st.collect_metrics,
                "collect_enstrophy": bool(
                    st.collect_metrics and st.collect_enstrophy),
            },
            "liquid_sources": liquid_records,
            "inflow_sources": inflow_records,
            "outflow_sources": outflow_records,
            "outflow": {
                "source_count": len(outflow_records),
                "initial_cull": initial_outflow_cull,
                **solver.outflow_stats(),
            },
            "solid_boundary": {
                **solver.solid_aperture_stats(),
                "obstacle_count": len(obstacles),
            },
            "checkpoint": {
                "schema": cache.CHECKPOINT_SCHEMA,
                "version": cache.CHECKPOINT_VERSION,
                "fingerprint": fingerprint,
                "latest_frame": scene.frame_start,
                "state": "RUNNING",
            },
            "bake_lifecycle": {
                "state": "RUNNING",
                "last_committed_frame": scene.frame_start,
                "error": "",
            },
            "version": 5,
        }
        from ..stflip.experiments import profile_provenance

        new_meta["experiment_profile"] = profile_provenance(
            st.experiment_profile, st)
        if st.collect_metrics:
            from ..stflip.metrics import METRICS_SCHEMA, SCHEMA_VERSION

            new_meta["metrics"] = {
                "schema": METRICS_SCHEMA,
                "version": SCHEMA_VERSION,
                "file": cache.METRICS_NAME,
                "enstrophy_enabled": st.collect_enstrophy,
            }

        if is_resume:
            latest_frame = validate_resume_metadata(
                existing_meta,
                fingerprint,
                scene.frame_start,
                scene.frame_end,
            )
            if cache.read_frame(cache_dir, latest_frame) is None:
                raise ValueError(
                    f"committed output frame {latest_frame} is missing or corrupt")
            checkpoint_state = cache.read_checkpoint(
                cache_dir,
                latest_frame,
                expected_fingerprint=fingerprint,
            )
            if checkpoint_state is None:
                raise ValueError(
                    f"solver checkpoint for frame {latest_frame} is missing")
            expected_time = (
                latest_frame - int(scene.frame_start)) * params.frame_dt
            if not math.isclose(
                    float(checkpoint_state["time"]), expected_time,
                    rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError(
                    f"solver checkpoint for frame {latest_frame} has an "
                    "incompatible simulation clock")
            solver.restore_state(checkpoint_state)
            seeded = int(solver.pos.shape[0])
            meta = existing_meta
            meta["frame_end"] = scene.frame_end
            meta["checkpoint"]["state"] = "RUNNING"
            meta["bake_lifecycle"] = {
                "state": "RUNNING",
                "last_committed_frame": latest_frame,
                "error": "",
            }
            collect_metrics = isinstance(meta.get("metrics"), dict)
            collect_enstrophy = bool(
                collect_metrics
                and meta["metrics"].get("enstrophy_enabled", False))
            cache.write_meta(cache_dir, meta)
            current_frame = latest_frame
        else:
            meta = new_meta
            current_frame = scene.frame_start
            collect_metrics = bool(st.collect_metrics)
            collect_enstrophy = bool(
                st.collect_metrics and st.collect_enstrophy)
            pos, vel = solver.get_render_particles()
            cache.write_frame(
                cache_dir,
                current_frame,
                pos + origin[None, :].astype(np.float32),
                vel,
            )
            cache.write_checkpoint(
                cache_dir,
                current_frame,
                solver.checkpoint_state(),
                fingerprint=fingerprint,
            )
            if collect_metrics:
                record = _measure_output_frame(
                    current_frame, solver, None, pos, vel, None, False)
                cache.append_metric(cache_dir, record)
            # The metadata is the commit marker: a crash before this write
            # leaves the frame/checkpoint pair outside the committed range.
            cache.write_meta(cache_dir, meta)

        _BAKE.update(
            solver=solver,
            origin=origin.astype(np.float32),
            scene=scene,
            cache_dir=cache_dir,
            meta=meta,
            frame=current_frame,
            end=scene.frame_end,
            backend_label=backend_label,
            collect_metrics=collect_metrics,
            collect_enstrophy=collect_enstrophy,
            running=True,
            cancel_requested=False,
            resumed=is_resume,
        )

        particle_obj = mesher.ensure_particle_object(
            existing_obj=st.particle_object)
        st.particle_object = particle_obj
        if st.create_surface:
            st.surface_object = mesher.ensure_surface_object(
                particle_obj, dx, st.particle_radius, st.surface_voxel,
                existing_obj=st.surface_object)
            mesher.configure_surface_smoothing(
                st.surface_object,
                st.surface_smoothing,
                st.surface_smoothing_iterations,
                st.surface_smoothing_factor,
            )
            _set_surface_enabled(st.surface_object, True)
        else:
            stale_surface = st.surface_object
            if stale_surface is None:
                candidate = bpy.data.objects.get(
                    getattr(mesher, "SURFACE_OBJ", "STFLIP Liquid Surface"))
                if candidate is not None and candidate.name in scene.objects:
                    stale_surface = candidate
                    st.surface_object = candidate
            stale_surface = mesher.scene_exclusive_output(
                scene, stale_surface)
            if stale_surface is None:
                st.surface_object = None
            _set_surface_enabled(stale_surface, False)
        handlers.ensure_registered()

        scene.frame_set(current_frame)
        completed_span = max(0, current_frame - scene.frame_start)
        total_span = max(1, scene.frame_end - scene.frame_start)
        _set_bake_lifecycle(
            st,
            "RUNNING",
            (f"Resuming after frame {current_frame} on {backend_label}..."
             if is_resume else
             f"Baking on {backend_label}: {seeded} particles seeded..."),
            progress=completed_span / total_span,
        )
        return True

    def _bake_next_frame(self, scene=None) -> bool:
        """Advance one frame; returns True while frames remain."""
        b = _BAKE
        scene = b.get("scene", scene)
        if scene is None:
            raise RuntimeError("owning bake scene is no longer available")
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
        cache.write_checkpoint(
            b["cache_dir"],
            b["frame"],
            solver.checkpoint_state(),
            fingerprint=b["meta"]["checkpoint"]["fingerprint"],
        )
        if b.get("collect_metrics"):
            record = _measure_output_frame(
                b["frame"], solver, stats, pos, vel, compute_wall_s,
                b.get("collect_enstrophy", False),
            )
            cache.append_metric(b["cache_dir"], record)
        b["meta"]["frame_end_baked"] = b["frame"]
        b["meta"]["checkpoint"].update({
            "latest_frame": b["frame"],
            "state": "RUNNING",
        })
        b["meta"]["bake_lifecycle"] = {
            "state": "RUNNING",
            "last_committed_frame": b["frame"],
            "error": "",
        }
        if "outflow" in b["meta"]:
            source_count = b["meta"]["outflow"].get("source_count", 0)
            initial_cull = b["meta"]["outflow"].get("initial_cull", {})
            b["meta"]["outflow"] = {
                "source_count": source_count,
                "initial_cull": initial_cull,
                **solver.outflow_stats(),
            }
        cache.write_meta(b["cache_dir"], b["meta"])
        span = max(1, b["end"] - b["meta"]["frame_start"])
        progress = (b["frame"] - b["meta"]["frame_start"]) / span
        removed = int(getattr(stats, "particles_removed", 0))
        _set_bake_lifecycle(
            scene.stflip,
            "RUNNING",
            f"Frame {b['frame']}/{b['end']}  "
            f"({stats.n_particles} pts, {stats.steps} steps, "
            f"{removed} removed, "
            f"{b.get('backend_label', solver.be.name)})",
            progress=progress,
        )
        scene.frame_set(b["frame"])
        return b["frame"] < b["end"]

    def invoke(self, context, event):
        try:
            if not self._setup(context):
                return {"CANCELLED"}
        except Exception as exc:
            self.report({"ERROR"}, f"Bake setup failed: {exc}")
            if _BAKE.get("meta") is not None:
                return self._finish(context, "FAILED", error=str(exc))
            _BAKE.clear()
            _fail_bake(context.scene.stflip, str(exc))
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
            while self._bake_next_frame():
                pass
        except Exception as exc:
            scene = _BAKE.get("scene", context.scene)
            _fail_bake(scene.stflip, str(exc))
            self.report({"ERROR"}, f"Bake failed: {exc}")
            return self._finish(context, "FAILED", error=str(exc))
        return self._finish(context, "COMPLETE")

    def modal(self, context, event):
        if event.type == "ESC":
            return self._finish(context, "CANCELLED")
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if _BAKE.get("cancel_requested"):
            return self._finish(context, "CANCELLED")
        if not _BAKE.get("running"):
            return self._finish(context, "CANCELLED")
        # Any exception must still tear down the timer and _BAKE state, or
        # baking is bricked for the rest of the session.
        try:
            more = self._bake_next_frame()
        except Exception as exc:
            self.report({"ERROR"}, f"Bake failed: {exc}")
            return self._finish(context, "FAILED", error=str(exc))
        if not more:
            return self._finish(context, "COMPLETE")
        return {"RUNNING_MODAL"}

    def _finish(self, context, outcome: str, error: str = ""):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        scene = _BAKE.get("scene", context.scene)
        st = scene.stflip
        backend_label = _BAKE.get("backend_label", "unknown backend")
        meta = _BAKE.get("meta")
        start = int((meta or {}).get("frame_start", scene.frame_start))
        committed = int((meta or {}).get("frame_end_baked", start - 1))
        count = max(0, committed - start + 1)
        if outcome == "COMPLETE" and committed < int(_BAKE.get("end", committed)):
            outcome = "FAILED"
            error = error or "bake ended before the requested frame range"
        if isinstance(meta, dict):
            meta["checkpoint"]["state"] = outcome
            meta["bake_lifecycle"] = {
                "state": outcome,
                "last_committed_frame": committed,
                "error": error if outcome == "FAILED" else "",
            }
            try:
                cache.write_meta(_BAKE["cache_dir"], meta)
            except Exception as exc:
                outcome = "FAILED"
                error = f"could not persist bake lifecycle: {exc}"
        if outcome == "COMPLETE":
            _set_bake_lifecycle(
                st, "COMPLETE",
                f"Bake complete ({count} frames) on {backend_label}",
                progress=1.0,
            )
        elif outcome == "FAILED":
            _fail_bake(st, error or "unknown simulation error")
        else:
            _set_bake_lifecycle(
                st, "CANCELLED",
                f"Bake cancelled after {count} cached frames",
                progress=st.bake_progress,
            )
        result = {"FINISHED"} if outcome == "COMPLETE" else {"CANCELLED"}
        _BAKE.clear()
        return result


class STFLIP_OT_resume_bake(bpy.types.Operator):
    """Resume the latest committed checkpoint after re-validating the scene."""
    bl_idname = "stflip.resume_bake"
    bl_label = "Resume Bake"
    bl_options = {"REGISTER"}
    bl_description = (
        "Re-voxelize simulation inputs, restore the latest committed solver "
        "checkpoint, and continue to an extended Scene End frame"
    )
    _timer = None
    _is_resume = True

    # Do not inherit from the registered Bake operator. Blender's RNA
    # registration treats registered Operator subclasses as runtime types;
    # registering a second RNA type through that inheritance chain can detach
    # the base type's execute callback. Thin wrappers share the implementation
    # while keeping the two Blender operator classes independent.
    def _setup(self, context) -> bool:
        return STFLIP_OT_bake._setup(self, context)

    def _bake_next_frame(self, scene=None) -> bool:
        return STFLIP_OT_bake._bake_next_frame(self, scene)

    def invoke(self, context, event):
        return STFLIP_OT_bake.invoke(self, context, event)

    def execute(self, context):
        return STFLIP_OT_bake.execute(self, context)

    def modal(self, context, event):
        return STFLIP_OT_bake.modal(self, context, event)

    def _finish(self, context, outcome: str, error: str = ""):
        return STFLIP_OT_bake._finish(self, context, outcome, error)


class STFLIP_OT_cancel_bake(bpy.types.Operator):
    """Request cancellation of the active modal bake."""
    bl_idname = "stflip.cancel_bake"
    bl_label = "Cancel Bake"
    bl_options = {"REGISTER"}

    def execute(self, context):
        if not _BAKE.get("running"):
            self.report({"INFO"}, "No bake is running")
            return {"CANCELLED"}
        _BAKE["cancel_requested"] = True
        scene = _BAKE.get("scene", context.scene)
        scene.stflip.bake_status = "Cancelling after the current operation..."
        return {"FINISHED"}


class STFLIP_OT_refresh_surface(bpy.types.Operator):
    """Refresh surfacing controls for an existing valid bake."""
    bl_idname = "stflip.refresh_surface"
    bl_label = "Refresh Surface"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if _BAKE.get("running"):
            self.report({"WARNING"}, "Cancel the running bake first")
            return {"CANCELLED"}
        scene = context.scene
        st = scene.stflip
        ownership_check = getattr(handlers, "scene_cache_ownership", None)
        ownership = (
            ownership_check(scene) if ownership_check is not None else "legacy"
        )
        if ownership in {"foreign", "invalid", "missing"}:
            self.report(
                {"ERROR"},
                f"Cannot refresh surface: cache ownership is {ownership}",
            )
            return {"CANCELLED"}
        meta = cache.read_meta(resolve_cache_dir(scene))
        try:
            dx = float(meta["dx"])
        except (KeyError, TypeError, ValueError):
            self.report({"ERROR"}, "Cache metadata has no valid cell size")
            return {"CANCELLED"}
        if not math.isfinite(dx) or dx <= 0.0:
            self.report({"ERROR"}, "Cache metadata has no valid cell size")
            return {"CANCELLED"}

        reconcile = getattr(handlers, "reconcile_scene_cache", None)
        if reconcile is not None and not reconcile(scene):
            self.report(
                {"ERROR"},
                "Cannot refresh surface: cached particle frame is unavailable",
            )
            return {"CANCELLED"}
        particle_obj = st.particle_object
        if particle_obj is None or getattr(particle_obj, "type", None) != "MESH":
            self.report({"ERROR"}, "Baked particle output is unavailable")
            return {"CANCELLED"}
        if not st.create_surface:
            surface = mesher.scene_exclusive_output(
                scene, st.surface_object)
            if surface is None:
                st.surface_object = None
            _set_surface_enabled(surface, False)
            self.report({"INFO"}, "Surface display disabled")
            return {"FINISHED"}

        st.surface_object = mesher.ensure_surface_object(
            particle_obj,
            dx,
            st.particle_radius,
            st.surface_voxel,
            existing_obj=st.surface_object,
        )
        mesher.configure_surface_smoothing(
            st.surface_object,
            st.surface_smoothing,
            st.surface_smoothing_iterations,
            st.surface_smoothing_factor,
        )
        _set_surface_enabled(st.surface_object, True)
        self.report({"INFO"}, "Surface controls refreshed from baked particles")
        return {"FINISHED"}


class STFLIP_OT_free_bake(bpy.types.Operator):
    """Delete the bake cache"""
    bl_idname = "stflip.free_bake"
    bl_label = "Free Bake"
    bl_options = {"REGISTER"}

    def execute(self, context):
        scene = context.scene
        ensure_cache_id = getattr(handlers, "ensure_scene_cache_id", None)
        if ensure_cache_id is not None:
            ensure_cache_id(scene)
        ownership_check = getattr(handlers, "scene_cache_ownership", None)
        ownership = (
            ownership_check(scene) if ownership_check is not None else "legacy"
        )
        if ownership in {"foreign", "invalid"}:
            self.report(
                {"ERROR"},
                f"Refusing to delete a {ownership} cache",
            )
            return {"CANCELLED"}
        n = cache.clear(resolve_cache_dir(scene))
        clear_output = getattr(handlers, "clear_scene_output", None)
        if clear_output is not None:
            clear_output(scene)
        _set_bake_lifecycle(
            scene.stflip, "IDLE", "", progress=0.0)
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
    STFLIP_OT_whirlpool_preview,
    STFLIP_OT_high_cfl_jet_leak,
    STFLIP_OT_bake,
    STFLIP_OT_resume_bake,
    STFLIP_OT_cancel_bake,
    STFLIP_OT_refresh_surface,
    STFLIP_OT_free_bake,
    STFLIP_OT_install_gpu,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
