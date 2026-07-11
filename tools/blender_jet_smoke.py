"""Installed-add-on smoke for the active high-CFL thin-plate jet.

Run inside Blender after installing the extension. Unlike the broad integration
smoke, this advances the authored 48-cell preset, requires measured high CFL,
and checks that particles do not appear beneath the interior of the one-cell
plate. The runoff outlet is disabled for this diagnostic so it cannot conceal
tunnelling by deleting failed particles.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import bpy
import numpy as np


def _finished(result: set[str], operation: str) -> None:
    if "FINISHED" not in result:
        raise RuntimeError(f"{operation} returned {sorted(result)}")


def run(backend: str = "cuda") -> dict:
    previous = bpy.context.window.scene
    scene = bpy.data.scenes.new("STFLIP Active Jet Smoke")
    cache_dir = Path(tempfile.mkdtemp(prefix="stflip-active-jet-"))
    try:
        bpy.context.window.scene = scene
        _finished(
            bpy.ops.stflip.high_cfl_jet_leak(),
            "high-CFL jet preview setup",
        )
        settings = scene.stflip
        settings.backend = backend
        settings.cache_dir = str(cache_dir)
        settings.collect_metrics = True
        settings.collect_enstrophy = False
        settings.create_surface = False
        scene.frame_start = 1
        scene.frame_end = 4

        plate = next(
            obj for obj in scene.objects if obj.stflip.role == "OBSTACLE")
        outlet = next(
            obj for obj in scene.objects if obj.stflip.role == "OUTFLOW")
        # Do not let a broad bottom sink hide particles that crossed the plate.
        outlet.stflip.role = "NONE"

        _finished(bpy.ops.stflip.bake(), "active high-CFL jet bake")

        from bl_ext.user_default.st_flip.stflip import cache

        metadata = cache.read_meta(str(cache_dir))
        if not isinstance(metadata, dict):
            raise AssertionError("jet bake metadata is missing")
        if metadata.get("backend") != backend:
            raise AssertionError(
                f"requested {backend!r} but bake used "
                f"{metadata.get('backend')!r}"
            )
        frames = cache.committed_frames(str(cache_dir), metadata)
        if frames != [1, 2, 3, 4]:
            raise AssertionError(f"jet committed frames are incomplete: {frames}")
        metrics = cache.read_metrics(str(cache_dir), frames)
        observed = max(
            float(row["particle_cfl_actual_max"])
            for row in metrics
            if row["particle_cfl_actual_max"] is not None
        )
        if observed < 12.0:
            raise AssertionError(
                f"jet did not reach high observed CFL: {observed:.3f}")

        plate_min_z = float(plate.location.z - 0.5 * plate.dimensions.z)
        inner_x = 0.5 * float(plate.dimensions.x) - 0.25
        inner_y = 0.5 * float(plate.dimensions.y) - 0.25
        minimum_z = None
        particle_counts = {}
        for frame in frames[1:]:
            frame_data = cache.read_frame(str(cache_dir), frame)
            if frame_data is None:
                raise AssertionError(f"jet frame {frame} is unreadable")
            positions, velocities = frame_data
            if (not np.all(np.isfinite(positions))
                    or not np.all(np.isfinite(velocities))):
                raise AssertionError(f"jet frame {frame} is invalid")
            particle_counts[frame] = int(positions.shape[0])
            inside = (
                (np.abs(positions[:, 0]) < inner_x)
                & (np.abs(positions[:, 1]) < inner_y)
            )
            if np.any(inside):
                frame_min = float(np.min(positions[inside, 2]))
                minimum_z = (
                    frame_min if minimum_z is None
                    else min(minimum_z, frame_min)
                )
                if frame_min < plate_min_z - 0.25 * float(metadata["dx"]):
                    raise AssertionError(
                        f"jet particles crossed the one-cell plate at frame "
                        f"{frame}: z={frame_min:.4f}, plate={plate_min_z:.4f}"
                    )
        if not any(count > 0 for count in particle_counts.values()):
            raise AssertionError("scheduled jet emitted no particles")

        result = {
            "backend_requested": backend,
            "backend_used": metadata["backend"],
            "frames": frames,
            "observed_particle_cfl_max": observed,
            "particle_counts": particle_counts,
            "plate_thickness_dx": (
                float(plate.dimensions.z) / float(metadata["dx"])),
            "minimum_particle_z_inside_plate_footprint": minimum_z,
            "runoff_outlet_disabled_for_tunnelling_check": True,
        }
        _finished(bpy.ops.stflip.free_bake(), "free active jet bake")
        return result
    finally:
        bpy.context.window.scene = previous
        for obj in list(scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.scenes.remove(scene)
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    print(json.dumps(run(os.environ.get("STFLIP_BACKEND", "cuda")), indent=2))
