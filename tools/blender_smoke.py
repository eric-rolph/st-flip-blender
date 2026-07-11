"""Two-frame Blender integration smoke test for the installed add-on.

Run this inside Blender, either with ``blender --python`` or through a trusted
local automation bridge.  It uses a temporary scene and cache, verifies the
selected compute backend, and restores the user's original scene.
"""

from __future__ import annotations

import importlib
import json
import shutil
import tempfile
import time
import tomllib
from pathlib import Path

import bpy


def _finished(result: set[str], operation: str) -> None:
    if "FINISHED" not in result:
        raise RuntimeError(f"{operation} returned {sorted(result)}")


def _expected_version() -> str:
    manifest_path = Path(__file__).parents[1] / "blender_manifest.toml"
    return tomllib.loads(manifest_path.read_text("utf-8"))["version"]


def _installed_version() -> str:
    module = importlib.import_module("bl_ext.user_default.st_flip")
    manifest_path = Path(module.__file__).with_name("blender_manifest.toml")
    return tomllib.loads(manifest_path.read_text("utf-8"))["version"]


def run(backend: str = "cuda") -> dict:
    """Bake a tiny dam break and return machine-readable validation data."""
    expected_version = _expected_version()
    installed_version = _installed_version()
    if installed_version != expected_version:
        raise AssertionError(
            f"installed add-on {installed_version}, expected {expected_version}"
        )
    window = bpy.context.window
    if window is None:
        raise RuntimeError("smoke test needs a Blender window/context")

    original_scene = window.scene
    scene = bpy.data.scenes.new("STFLIP Smoke Test")
    cache_dir = Path(tempfile.mkdtemp(prefix="stflip_smoke_"))
    # Output helpers retain legacy name fallbacks. Temporarily reserve any
    # outputs in the user's scene so the isolated smoke scene cannot reuse or
    # delete them during cleanup.
    reserved_outputs = []
    for name in ("STFLIP Particles", "STFLIP Liquid Surface"):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            reserved_outputs.append((obj, name))
            obj.name = f"{name} [preserved during smoke test]"
    started = time.perf_counter()
    try:
        window.scene = scene
        scene.frame_start = 1
        scene.frame_end = 2
        _finished(bpy.ops.stflip.quick_setup(), "quick setup")

        # A rotated, sub-cell-aligned obstacle ensures the installed add-on
        # exercises non-binary solid face apertures rather than domain walls
        # alone.
        bpy.ops.mesh.primitive_cube_add(
            size=2.0,
            location=(-0.05, 0.0, 0.42),
            rotation=(0.21, 0.17, 0.29),
        )
        obstacle = bpy.context.active_object
        obstacle.name = "STFLIP Smoke Obstacle"
        obstacle.scale = (0.18, 0.32, 0.28)
        obstacle.display_type = "WIRE"
        obstacle.hide_render = True
        obstacle.stflip.role = "OBSTACLE"
        liquid = next(
            (obj for obj in scene.objects
             if getattr(getattr(obj, "stflip", None), "role", None)
             == "LIQUID"),
            None,
        )
        if liquid is None:
            raise AssertionError("quick setup liquid was not created")
        liquid.stflip.initial_velocity = (4.0, 0.0, 0.0)

        settings = scene.stflip
        settings.experiment_profile = "ENSTROPHY_CFL_10_FLIP_99"
        _finished(
            bpy.ops.stflip.apply_experiment_profile(), "apply profile")
        settings.resolution = 8
        settings.backend = backend
        settings.cache_dir = str(cache_dir)
        settings.create_surface = True

        _finished(bpy.ops.stflip.bake(), "bake")
        meta = json.loads((cache_dir / "stflip_meta.json").read_text("utf-8"))
        if meta["backend"] != backend:
            raise AssertionError(
                f"requested {backend!r}, bake used {meta['backend']!r}"
            )
        if meta.get("version") != 3 or meta.get("settings", {}).get("seed") != 0:
            raise AssertionError("cache metadata lacks the v3 settings snapshot")
        if meta.get("addon_version") != installed_version:
            raise AssertionError("cache metadata add-on version is stale")
        boundary = meta.get("solid_boundary", {})
        if boundary.get("model") != "fractional_node_sdf":
            raise AssertionError("node-SDF solid boundary model was not used")
        if boundary.get("fractional_face_count", 0) <= 0:
            raise AssertionError("smoke obstacle produced no fractional faces")
        provenance = meta.get("experiment_profile", {})
        if provenance.get("matched") != settings.experiment_profile:
            raise AssertionError("applied profile provenance was not preserved")
        from bl_ext.user_default.st_flip.stflip import cache as stflip_cache

        metrics = stflip_cache.read_metrics(
            str(cache_dir), stflip_cache.baked_frames(str(cache_dir)))
        if [row["frame"] for row in metrics] != [1, 2]:
            raise AssertionError("expected one metric record per cached frame")
        if metrics[-1]["compute_wall_s"] is None:
            raise AssertionError("evolved frame lacks synchronized solver timing")
        if metrics[-1]["mac_grid_enstrophy_estimate"] is None:
            raise AssertionError("enstrophy diagnostic was not recorded")
        final_positions, _ = stflip_cache.read_frame(str(cache_dir), 2)
        if float(final_positions[:, 0].max()) <= -0.2:
            raise AssertionError("liquid front did not reach the smoke obstacle")
        from mathutils import Vector

        obstacle_inverse = obstacle.matrix_world.inverted()
        deeply_inside = sum(
            max(abs(value) for value in obstacle_inverse @ Vector(position))
            < 0.6
            for position in final_positions
        )
        if deeply_inside:
            raise AssertionError(
                f"{deeply_inside} particles remained deep inside the obstacle")
        csv_path = cache_dir / "smoke_metrics.csv"
        json_path = cache_dir / "smoke_metrics.json"
        _finished(
            bpy.ops.stflip.export_metrics(
                filepath=str(csv_path), export_format="CSV"),
            "CSV metrics export",
        )
        _finished(
            bpy.ops.stflip.export_metrics(
                filepath=str(json_path), export_format="JSON"),
            "JSON metrics export",
        )

        particles = settings.particle_object
        surface = settings.surface_object
        if particles is None or particles.type != "MESH":
            raise AssertionError("particle output object was not created")
        if len(particles.data.vertices) == 0:
            raise AssertionError("particle output contains no vertices")
        if particles.data.attributes.get("velocity") is None:
            raise AssertionError("particle velocity attribute is missing")
        if surface is None or not any(mod.type == "NODES" for mod in surface.modifiers):
            raise AssertionError("Geometry Nodes surface output was not created")

        return {
            "addon_version": installed_version,
            "backend": meta["backend"],
            "device": meta.get("cuda_device"),
            "frames": meta["frame_end_baked"] - meta["frame_start"] + 1,
            "particles": len(particles.data.vertices),
            "velocity_attribute": True,
            "surface": surface.name,
            "fractional_solid_faces": boundary["fractional_face_count"],
            "particles_deep_inside_obstacle": deeply_inside,
            "metric_frames": len(metrics),
            "metrics_exports": [csv_path.name, json_path.name],
            "elapsed_s": time.perf_counter() - started,
        }
    finally:
        window.scene = original_scene
        for obj in list(scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.scenes.remove(scene)
        for obj, original_name in reserved_outputs:
            if obj.name in bpy.data.objects:
                obj.name = original_name
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
