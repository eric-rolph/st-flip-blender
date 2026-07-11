"""Installed-add-on smoke test for bake, resume, and scene isolation.

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
import numpy as np


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


def _validate_whirlpool_preview(window) -> dict:
    """Create and inspect the paper-constrained preview without baking it."""
    previous = window.scene
    scene = bpy.data.scenes.new("STFLIP Whirlpool Setup Smoke")
    try:
        window.scene = scene
        _finished(bpy.ops.stflip.whirlpool_preview(), "whirlpool preview")
        settings = scene.stflip
        liquid = next(
            obj for obj in scene.objects
            if getattr(getattr(obj, "stflip", None), "role", None) == "LIQUID"
        )
        outlet = next(
            obj for obj in scene.objects
            if getattr(getattr(obj, "stflip", None), "role", None) == "OUTFLOW"
        )
        domain = settings.domain
        if domain is None or not np.allclose(domain.scale, (100.0, 100.0, 40.0)):
            raise AssertionError("whirlpool preview domain is not 200x200x80 m")
        if (liquid.stflip.initial_velocity_mode != "SOLID_BODY"
                or not np.isclose(liquid.stflip.angular_speed, 0.1)
                or not np.allclose(
                    liquid.stflip.rotation_axis_world, (0.0, 0.0, 1.0))):
            raise AssertionError("whirlpool preview rotation is not paper-constrained")
        if (outlet.stflip.outflow_mode != "PRESSURE"
                or not np.isclose(outlet.get(
                    "stflip_paper_pipe_diameter_m"), 20.0)
                or not np.isclose(outlet.get(
                    "stflip_paper_pipe_length_m"), 10.0)):
            raise AssertionError("whirlpool preview outlet dimensions/mode are wrong")
        if (scene.get("stflip_setup") != "WHIRLPOOL_PREVIEW_APPROXIMATE"
                or settings.resolution != 48):
            raise AssertionError("whirlpool preview is not marked approximate")
        return {
            "approximate": True,
            "domain_m": [200.0, 200.0, 80.0],
            "outlet_m": {"diameter": 20.0, "length": 10.0},
            "angular_speed_rad_s": 0.1,
            "preview_resolution": settings.resolution,
        }
    finally:
        window.scene = previous
        for obj in list(scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.scenes.remove(scene)


def _validate_copied_scene_output_isolation(window, source_scene) -> dict:
    """Prove a Blender scene copy receives independent mutable outputs."""
    previous = window.scene
    source_settings = source_scene.stflip
    source_particles = source_settings.particle_object
    source_surface = source_settings.surface_object
    copied = source_scene.copy()
    copied.name = "STFLIP Copied Scene Output Smoke"
    created = []
    ownership_trace = [{
        "stage": "copied",
        "source_id": source_settings.cache_id,
        "copied_id": copied.stflip.cache_id,
        "scene_order": [scene.name for scene in bpy.data.scenes],
    }]
    try:
        window.scene = copied
        copied_settings = copied.stflip
        ownership_trace.append({
            "stage": "activated",
            "source_id": source_settings.cache_id,
            "copied_id": copied_settings.cache_id,
        })
        if (copied_settings.particle_object is not source_particles
                or copied_settings.surface_object is not source_surface):
            raise AssertionError(
                "Blender did not reproduce the expected copied output bindings")

        from bl_ext.user_default.st_flip.addon import mesher

        local_particles = mesher.ensure_particle_object(
            existing_obj=copied_settings.particle_object)
        created.append(local_particles)
        ownership_trace.append({
            "stage": "particles",
            "source_id": source_settings.cache_id,
            "copied_id": copied_settings.cache_id,
        })
        local_surface = mesher.ensure_surface_object(
            local_particles,
            float(json.loads((Path(copied_settings.cache_dir)
                              / "stflip_meta.json").read_text("utf-8"))["dx"]),
            copied_settings.particle_radius,
            copied_settings.surface_voxel,
            existing_obj=copied_settings.surface_object,
        )
        created.append(local_surface)
        ownership_trace.append({
            "stage": "surface",
            "source_id": source_settings.cache_id,
            "copied_id": copied_settings.cache_id,
        })
        if (local_particles is source_particles
                or local_particles.data is source_particles.data
                or local_surface is source_surface
                or local_surface.data is source_surface.data):
            raise AssertionError("copied scene reused a mutable output datablock")
        if (source_particles.name in copied.objects
                or source_surface.name in copied.objects):
            raise AssertionError("copied scene retained stale shared outputs")
        if (source_particles.name not in source_scene.objects
                or source_surface.name not in source_scene.objects):
            raise AssertionError("isolating the copy unlinked the source outputs")
        return {
            "particle_object_distinct": True,
            "particle_mesh_distinct": True,
            "surface_object_distinct": True,
            "surface_mesh_distinct": True,
            "ownership_trace": ownership_trace,
        }
    finally:
        window.scene = previous
        ownership_trace.append({
            "stage": "restored",
            "source_id": source_settings.cache_id,
            "copied_id": copied.stflip.cache_id,
        })
        for obj in reversed(created):
            if obj.name not in bpy.data.objects:
                continue
            mesh = obj.data if getattr(obj, "type", None) == "MESH" else None
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        bpy.data.scenes.remove(copied)


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
        whirlpool_preview = _validate_whirlpool_preview(window)
        window.scene = scene
        _finished(bpy.ops.stflip.quick_setup(), "quick setup")
        scene.frame_start = 1
        scene.frame_end = 2

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

        # Exercise both honest outlet semantics.  These compact source volumes
        # overlap the initial liquid so a two-frame smoke bake observes actual
        # removal rather than merely serializing inactive settings.
        outlets = []
        for index, (mode, x, y) in enumerate((
            ("VOLUME", -0.65, -0.55),
            # Pressure outlets must intersect an exterior domain face.
            ("PRESSURE", -0.88, 0.55),
        ), start=1):
            bpy.ops.mesh.primitive_cube_add(
                size=2.0, location=(x, y, 0.45))
            outlet = bpy.context.active_object
            outlet.name = f"STFLIP Smoke Outflow {index} {mode}"
            outlet.scale = (0.18, 0.18, 0.18)
            outlet.display_type = "WIRE"
            outlet.hide_render = True
            outlet.stflip.role = "OUTFLOW"
            outlet.stflip.outflow_mode = mode
            outlets.append(outlet)
        liquid = next(
            (obj for obj in scene.objects
             if getattr(getattr(obj, "stflip", None), "role", None)
             == "LIQUID"),
            None,
        )
        if liquid is None:
            raise AssertionError("quick setup liquid was not created")
        liquid.stflip.initial_velocity_mode = "SOLID_BODY"
        liquid.stflip.initial_velocity = (4.0, 0.0, 0.0)
        liquid.stflip.rotation_center_world = (0.15, -0.2, 0.3)
        liquid.stflip.rotation_axis_world = (0.0, 0.0, 2.0)
        liquid.stflip.angular_speed = 1.5

        settings = scene.stflip
        settings.experiment_profile = "ENSTROPHY_CFL_10_FLIP_99"
        _finished(
            bpy.ops.stflip.apply_experiment_profile(), "apply profile")
        settings.resolution = 8
        settings.backend = backend
        settings.cache_dir = str(cache_dir)
        settings.create_surface = True
        settings.surface_smoothing = True
        settings.surface_smoothing_iterations = 3
        settings.surface_smoothing_factor = 0.28
        settings.density = 997.0
        settings.local_cfl = 0.75
        settings.pcg_tolerance = 5e-5
        settings.pcg_max_iterations = 275
        settings.density_floor_relative = 2e-3

        _finished(bpy.ops.stflip.bake(), "bake")
        metadata_path = cache_dir / "stflip_meta.json"
        if not metadata_path.is_file():
            from bl_ext.user_default.st_flip.addon.handlers import (
                resolve_cache_dir,
            )

            raise AssertionError(
                "bake returned without cache metadata: "
                f"state={settings.bake_state!r}, "
                f"status={settings.bake_status!r}, "
                f"error={settings.bake_error!r}, "
                f"configured={settings.cache_dir!r}, "
                f"resolved={resolve_cache_dir(scene)!r}, "
                f"files={sorted(path.name for path in cache_dir.iterdir())!r}"
            )
        meta = json.loads(metadata_path.read_text("utf-8"))
        if meta["backend"] != backend:
            raise AssertionError(
                f"requested {backend!r}, bake used {meta['backend']!r}"
            )
        if meta.get("version") != 5 or meta.get("settings", {}).get("seed") != 0:
            raise AssertionError("cache metadata lacks the v5 settings snapshot")
        if meta.get("addon_version") != installed_version:
            raise AssertionError("cache metadata add-on version is stale")
        if not settings.cache_id or meta.get("cache_owner_id") != settings.cache_id:
            raise AssertionError("cache metadata is not owned by the smoke scene")
        advanced = meta.get("settings", {})
        expected_advanced = {
            "density": 997.0,
            "local_advection_cfl": 0.75,
            "pcg_tolerance": 5e-5,
            "pcg_max_iterations": 275,
            "eps_rho_relative": 2e-3,
        }
        for key, expected in expected_advanced.items():
            if not np.isclose(advanced.get(key), expected):
                raise AssertionError(
                    f"advanced setting {key} was not baked: {advanced.get(key)!r}")
        outflow_sources = meta.get("outflow_sources", [])
        modes = {source.get("mode") for source in outflow_sources}
        if modes != {"VOLUME", "PRESSURE"}:
            raise AssertionError(f"outflow modes were not preserved: {modes}")
        if any(source.get("cell_count", 0) <= 0 for source in outflow_sources):
            raise AssertionError("smoke outflow voxelized to no usable cells")
        boundary = meta.get("solid_boundary", {})
        if boundary.get("model") != "fractional_node_sdf":
            raise AssertionError("node-SDF solid boundary model was not used")
        if boundary.get("fractional_face_count", 0) <= 0:
            raise AssertionError("smoke obstacle produced no fractional faces")
        provenance = meta.get("experiment_profile", {})
        if provenance.get("matched") != settings.experiment_profile:
            raise AssertionError("applied profile provenance was not preserved")
        from bl_ext.user_default.st_flip.stflip import cache as stflip_cache

        checkpoint = meta.get("checkpoint", {})
        fingerprint = checkpoint.get("fingerprint")
        if (checkpoint.get("schema") != stflip_cache.CHECKPOINT_SCHEMA
                or checkpoint.get("version") != stflip_cache.CHECKPOINT_VERSION
                or checkpoint.get("latest_frame") != 2
                or checkpoint.get("state") != "COMPLETE"
                or not isinstance(fingerprint, str)
                or len(fingerprint) != 64):
            raise AssertionError("initial bake checkpoint metadata is invalid")
        checkpoint_two_path = Path(stflip_cache.checkpoint_path(
            str(cache_dir), 2))
        checkpoint_two_before = checkpoint_two_path.read_bytes()
        state_two = stflip_cache.read_checkpoint(str(cache_dir), 2)
        frame_two_before = stflip_cache.read_frame(str(cache_dir), 2)

        # Exercise the user-facing long-bake path: extend the requested range,
        # restore frame 2, and continue without replacing committed history.
        scene.frame_end = 3
        _finished(bpy.ops.stflip.resume_bake(), "resume bake")
        meta = json.loads((cache_dir / "stflip_meta.json").read_text("utf-8"))
        resumed_checkpoint = meta.get("checkpoint", {})
        if (meta.get("frame_end_baked") != 3
                or resumed_checkpoint.get("latest_frame") != 3
                or resumed_checkpoint.get("state") != "COMPLETE"
                or resumed_checkpoint.get("fingerprint") != fingerprint):
            raise AssertionError("resumed bake did not commit frame 3")
        if checkpoint_two_path.read_bytes() != checkpoint_two_before:
            raise AssertionError("resume rewrote the prior committed checkpoint")
        frame_two_after = stflip_cache.read_frame(str(cache_dir), 2)
        if not all(np.array_equal(before, after) for before, after in zip(
                frame_two_before, frame_two_after)):
            raise AssertionError("resume changed prior committed output")
        state_three = stflip_cache.read_checkpoint(str(cache_dir), 3)
        if (state_three is None or state_two is None
                or state_three["time"] <= state_two["time"]):
            raise AssertionError("resumed solver clock did not advance")
        if stflip_cache.resumable_frames(str(cache_dir), meta) != [1, 2, 3]:
            raise AssertionError("resumed frame/checkpoint history is incomplete")
        lifecycle = meta.get("bake_lifecycle", {})
        if (lifecycle.get("state") != "COMPLETE"
                or lifecycle.get("last_committed_frame") != 3):
            raise AssertionError("resumed lifecycle was not persisted")

        metrics = stflip_cache.read_metrics(
            str(cache_dir), stflip_cache.baked_frames(str(cache_dir)))
        if [row["frame"] for row in metrics] != [1, 2, 3]:
            raise AssertionError("expected one metric record per cached frame")
        if metrics[-1]["compute_wall_s"] is None:
            raise AssertionError("evolved frame lacks synchronized solver timing")
        if metrics[-1]["mac_grid_enstrophy_estimate"] is None:
            raise AssertionError("enstrophy diagnostic was not recorded")
        initial_positions, initial_velocities = stflip_cache.read_frame(
            str(cache_dir), 1)
        source = next(
            (item for item in meta.get("liquid_sources", [])
             if item.get("name") == liquid.name),
            None,
        )
        if source is None or source.get("initial_velocity_mode") != "SOLID_BODY":
            raise AssertionError("solid-body liquid metadata was not preserved")
        rotation = source.get("solid_body_rotation", {})
        omega = np.asarray(
            rotation.get("angular_velocity_world"), dtype=np.float32)
        center = np.asarray(
            rotation.get("center_world"), dtype=np.float32)
        linear = np.asarray(source.get("initial_velocity"), dtype=np.float32)
        if omega.shape != (3,) or center.shape != (3,) or linear.shape != (3,):
            raise AssertionError("solid-body metadata vectors are malformed")
        expected_velocities = linear + np.cross(
            np.broadcast_to(omega, initial_positions.shape),
            initial_positions.astype(np.float32) - center,
        )
        initial_velocity_max_error = float(np.max(np.abs(
            initial_velocities - expected_velocities)))
        if initial_velocity_max_error > 1e-5:
            raise AssertionError(
                "solid-body initial velocity mismatch: "
                f"max error {initial_velocity_max_error:.3g}"
            )
        final_positions, _ = stflip_cache.read_frame(str(cache_dir), 3)
        if len(final_positions) >= len(initial_positions):
            raise AssertionError("active smoke outflows removed no particles")
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
        smoothing = surface.modifiers.get("STFLIP Geometric Smoothing")
        if (smoothing is None or not smoothing.show_viewport
                or smoothing.iterations != 3
                or not np.isclose(smoothing.lambda_factor, 0.28)):
            raise AssertionError("geometric smoothing controls were not applied")
        settings.surface_smoothing_factor = 0.17
        _finished(bpy.ops.stflip.refresh_surface(), "refresh surface")
        smoothing = surface.modifiers.get("STFLIP Geometric Smoothing")
        if not np.isclose(smoothing.lambda_factor, 0.17):
            raise AssertionError("surface refresh did not update smoothing")
        owner_before_copy = settings.cache_id
        copied_scene_isolation = _validate_copied_scene_output_isolation(
            window, scene)
        ownership_trace = copied_scene_isolation.pop("ownership_trace", [])
        from bl_ext.user_default.st_flip.addon import handlers

        ownership_after_copy = handlers.scene_cache_ownership(scene)
        if (window.scene is not scene
                or settings.cache_id != owner_before_copy
                or ownership_after_copy != "owned"):
            raise AssertionError(
                "copied-scene isolation changed source cache ownership: "
                f"active={window.scene.name!r}, source={scene.name!r}, "
                f"before={owner_before_copy!r}, after={settings.cache_id!r}, "
                f"ownership={ownership_after_copy!r}, "
                f"metadata_owner={meta.get('cache_owner_id')!r}, "
                f"trace={ownership_trace!r}"
            )
        if (settings.bake_state != "COMPLETE"
                or not np.isclose(settings.bake_progress, 1.0)
                or settings.bake_error):
            raise AssertionError("successful bake lifecycle state is incorrect")

        result = {
            "addon_version": installed_version,
            "backend": meta["backend"],
            "device": meta.get("cuda_device"),
            "frames": meta["frame_end_baked"] - meta["frame_start"] + 1,
            "particles": len(particles.data.vertices),
            "particles_initial": len(initial_positions),
            "outflow_modes": sorted(modes),
            "outflow_removed": len(initial_positions) - len(final_positions),
            "velocity_attribute": True,
            "surface": surface.name,
            "surface_smoothing": True,
            "fractional_solid_faces": boundary["fractional_face_count"],
            "initial_velocity_mode": source["initial_velocity_mode"],
            "initial_velocity_max_error": initial_velocity_max_error,
            "particles_deep_inside_obstacle": deeply_inside,
            "metric_frames": len(metrics),
            "resume_continuity": True,
            "copied_scene_output_isolation": copied_scene_isolation,
            "metrics_exports": [csv_path.name, json_path.name],
            "elapsed_s": time.perf_counter() - started,
            "whirlpool_preview": whirlpool_preview,
        }
        _finished(bpy.ops.stflip.free_bake(), "free bake")
        if (cache_dir / "stflip_meta.json").exists():
            raise AssertionError("Free Bake left cache metadata behind")
        if settings.bake_state != "IDLE":
            raise AssertionError("Free Bake did not reset lifecycle state")
        result["free_bake_cleared_output"] = (
            settings.particle_object is particles
            and len(particles.data.vertices) == 0)
        if not result["free_bake_cleared_output"]:
            raise AssertionError("Free Bake left scene output bindings behind")

        return result
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
