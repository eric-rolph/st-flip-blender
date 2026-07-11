"""Installed-add-on smoke for inline Paper MCF baking and exact resume.

Run from a normal Blender process so an active window is available. The test
uses an isolated temporary scene/cache, exercises the requested CPU or CUDA
backend, and removes all generated data before returning.
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
    manifest = Path(__file__).parents[1] / "blender_manifest.toml"
    return tomllib.loads(manifest.read_text("utf-8"))["version"]


def _installed_version() -> str:
    module = importlib.import_module("bl_ext.user_default.st_flip")
    manifest = Path(module.__file__).with_name("blender_manifest.toml")
    return tomllib.loads(manifest.read_text("utf-8"))["version"]


def run(backend: str = "cuda", iterations: int = 30) -> dict:
    if backend not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu' or 'cuda'")
    if not 1 <= int(iterations) <= 100:
        raise ValueError("iterations must be in [1, 100]")
    installed = _installed_version()
    expected = _expected_version()
    if installed != expected:
        raise AssertionError(f"installed add-on {installed}, expected {expected}")

    window = bpy.context.window
    if window is None:
        raise RuntimeError("paper smoke needs a Blender window/context")
    original_scene = window.scene
    scene = bpy.data.scenes.new(f"STFLIP Paper {backend.upper()} Smoke")
    cache_dir = Path(tempfile.mkdtemp(prefix=f"stflip_paper_{backend}_"))
    reserved_outputs = []
    for name in ("STFLIP Particles", "STFLIP Liquid Surface"):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            reserved_outputs.append((obj, name))
            obj.name = f"{name} [preserved during paper smoke]"

    started = time.perf_counter()
    try:
        window.scene = scene
        _finished(bpy.ops.stflip.quick_setup(), "quick setup")
        scene.frame_start = 1
        scene.frame_end = 2
        settings = scene.stflip
        settings.resolution = 8
        settings.backend = backend
        settings.cache_dir = str(cache_dir)
        settings.create_surface = True
        settings.surface_method = "PAPER_MCF"
        settings.paper_mcf_iterations = int(iterations)
        settings.paper_mesh_adaptivity = 0.0
        settings.paper_max_reconstruction_voxels = 262_144

        _finished(bpy.ops.stflip.bake(), "Paper MCF bake")
        from bl_ext.user_default.st_flip.stflip import cache as stflip_cache

        metadata_path = cache_dir / stflip_cache.META_NAME
        meta = json.loads(metadata_path.read_text("utf-8"))
        reconstruction = meta.get("surface_reconstruction", {})
        fingerprint = stflip_cache.validate_surface_metadata(reconstruction)
        config = reconstruction.get("config", {})
        if (settings.bake_state != "COMPLETE"
                or meta.get("backend") != backend
                or reconstruction.get("state") != "COMPLETE"
                or reconstruction.get("latest_frame") != 2
                or config.get("mcf_iterations") != int(iterations)
                or config.get("backend") != backend
                or stflip_cache.surface_frames(
                    str(cache_dir), fingerprint) != [1, 2]):
            raise AssertionError("inline Paper MCF bake metadata is incomplete")

        surface = settings.surface_object
        if surface is None:
            raise AssertionError("inline Paper MCF surface object is missing")
        modifier = surface.modifiers.get("STFLIP Surface")
        if (surface.get("stflip_surface_method") != "PAPER_MCF"
                or len(surface.data.vertices) == 0
                or len(surface.data.polygons) == 0
                or (modifier is not None
                    and (modifier.show_viewport or modifier.show_render))):
            raise AssertionError("inline Paper MCF mesh is not visibly active")

        for frame in (1, 2):
            source = stflip_cache.read_frame(str(cache_dir), frame)
            mesh = stflip_cache.read_surface(
                str(cache_dir),
                frame,
                fingerprint,
                expected_source_positions=source[0],
            )
            if mesh is None or len(mesh[0]) == 0:
                raise AssertionError(f"surface frame {frame} is invalid")

        checkpoint_two = Path(stflip_cache.checkpoint_path(
            str(cache_dir), 2)).read_bytes()
        surface_two = Path(stflip_cache.surface_path(
            str(cache_dir), 2, fingerprint)).read_bytes()
        scene.frame_end = 3
        _finished(bpy.ops.stflip.resume_bake(), "Paper MCF resume")
        resumed = json.loads(metadata_path.read_text("utf-8"))
        resumed_reconstruction = resumed.get("surface_reconstruction", {})
        resumed_fingerprint = stflip_cache.validate_surface_metadata(
            resumed_reconstruction)
        if (resumed.get("frame_end_baked") != 3
                or resumed.get("checkpoint", {}).get("latest_frame") != 3
                or resumed_reconstruction.get("state") != "COMPLETE"
                or resumed_reconstruction.get("latest_frame") != 3
                or resumed_fingerprint != fingerprint
                or stflip_cache.surface_frames(
                    str(cache_dir), fingerprint) != [1, 2, 3]):
            raise AssertionError("Paper MCF resume did not extend both caches")
        if (Path(stflip_cache.checkpoint_path(
                str(cache_dir), 2)).read_bytes() != checkpoint_two
                or Path(stflip_cache.surface_path(
                    str(cache_dir), 2, fingerprint)).read_bytes()
                != surface_two):
            raise AssertionError("resume rewrote a committed frame")

        scene.frame_set(1)
        counts = (len(surface.data.vertices), len(surface.data.polygons))
        scene.frame_set(3)
        scene.frame_set(1)
        if counts != (len(surface.data.vertices), len(surface.data.polygons)):
            raise AssertionError("resumed paper playback is not reproducible")

        result = {
            "addon_version": installed,
            "simulation_backend": resumed["backend"],
            "surface_backend": config["backend"],
            "surface_iterations": config["mcf_iterations"],
            "frames": 3,
            "surface_fingerprint": fingerprint,
            "vertices": len(surface.data.vertices),
            "polygons": len(surface.data.polygons),
            "resume_preserved_prior_frames": True,
            "elapsed_s": time.perf_counter() - started,
        }
        _finished(bpy.ops.stflip.free_bake(), "free bake")
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
