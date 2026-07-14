"""Headless installed-extension smoke for Blender's real Python runtime.

The extension archive must be installed and enabled before invoking this file::

    blender --background --offline-mode --python tools/blender_headless_smoke.py \
      -- --require-openvdb --output blender-smoke.json

Unlike :mod:`tools.blender_smoke`, this entry point deliberately avoids
window-only operators.  It validates the installed extension namespace,
registered ``bpy`` properties, a tiny CPU ST-FLIP step, Appendix-B density
reconstruction, Geometry Nodes creation, and OpenVDB polygonization when the
active Blender build provides its Python module.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tomllib
import traceback
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


EXTENSION_MODULE = "bl_ext.user_default.st_flip"
OPENVDB_MODULES = ("openvdb", "pyopenvdb")


def script_arguments(argv: Sequence[str]) -> list[str]:
    """Return arguments intended for this script, not for Blender itself."""
    values = list(argv)
    if "--" in values:
        return values[values.index("--") + 1:]
    return values[1:]


def load_openvdb(
    importer: Callable[[str], object] = importlib.import_module,
) -> tuple[str | None, object | None]:
    """Load Blender's OpenVDB binding under either supported module name.

    Blender 4.2's official archives expose ``pyopenvdb`` while some newer or
    externally packaged builds expose ``openvdb``.  A non-import loader error
    is intentionally allowed to propagate: a broken binary must not be
    reported as an unavailable optional feature.
    """
    for name in OPENVDB_MODULES:
        try:
            return name, importer(name)
        except ImportError:
            continue
    return None, None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-openvdb",
        action="store_true",
        help="fail when Blender has no usable OpenVDB Python binding",
    )
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    return parser


def _installed_extension():
    module = importlib.import_module(EXTENSION_MODULE)
    module_path = Path(module.__file__).resolve()
    manifest_path = module_path.with_name("blender_manifest.toml")
    if not manifest_path.is_file():
        raise AssertionError(
            f"installed extension manifest is missing beside {module_path}"
        )
    manifest = tomllib.loads(manifest_path.read_text("utf-8"))
    if manifest.get("id") != "st_flip":
        raise AssertionError(f"unexpected installed extension id: {manifest.get('id')!r}")
    return module, module_path, manifest


def _tiny_cpu_solver(stflip_module):
    params = stflip_module.Params(
        resolution=(8, 8, 8),
        dx=1.0 / 8.0,
        gravity=(0.0, 0.0, -1.0),
        frame_dt=1.0 / 24.0,
        cfl_target=4.0,
        cfl_local=1.0,
        particles_per_cell=1,
        seed=17,
        pcg_tol=1e-5,
        pcg_max_iter=120,
    )
    solver = stflip_module.STFLIPSolver(params, "cpu")
    liquid = np.zeros(params.resolution, dtype=bool)
    liquid[1:4, 2:5, 1:4] = True
    seeded = solver.add_liquid_mask(liquid, velocity=(0.35, 0.05, 0.0))
    if seeded != int(liquid.sum()):
        raise AssertionError(f"seeded {seeded} particles, expected {int(liquid.sum())}")

    stats = solver.step_frame()
    positions, velocities = solver.get_render_particles()
    residual = solver.be.to_numpy(solver.dt_resid)
    if positions.shape != velocities.shape or positions.ndim != 2 \
            or positions.shape[1:] != (3,):
        raise AssertionError(
            f"invalid particle output shapes: {positions.shape}, {velocities.shape}"
        )
    if len(positions) != seeded:
        raise AssertionError(f"particle count changed in closed smoke: {seeded} -> {len(positions)}")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise AssertionError("CPU solver produced non-finite particle state")
    residual_max = float(np.max(np.abs(residual))) if residual.size else 0.0
    if residual_max > 0.5 * params.frame_dt + 1e-7:
        raise AssertionError(
            f"temporal residual {residual_max} exceeded Appendix-A bound"
        )
    if stats.steps < 1 or stats.n_particles != seeded:
        raise AssertionError("CPU frame statistics do not match evolved state")
    return solver, positions, velocities, residual_max, stats


def _write_particle_mesh(mesher, positions, velocities):
    obj = mesher.ensure_particle_object()
    mesh = obj.data
    mesh.clear_geometry()
    mesh.vertices.add(len(positions))
    mesh.vertices.foreach_set(
        "co", np.ascontiguousarray(positions, dtype=np.float32).ravel())
    velocity = mesh.attributes.get("velocity")
    if velocity is None:
        velocity = mesh.attributes.new("velocity", "FLOAT_VECTOR", "POINT")
    velocity.data.foreach_set(
        "vector", np.ascontiguousarray(velocities, dtype=np.float32).ravel())
    mesh.update()
    if len(mesh.vertices) != len(positions):
        raise AssertionError("Blender particle mesh did not receive every particle")
    return obj


def run(*, require_openvdb: bool = False) -> dict:
    """Run the installed-extension smoke and return JSON-safe diagnostics."""
    import bpy

    if not bpy.app.background:
        raise RuntimeError("headless smoke must run with Blender --background")
    if tuple(bpy.app.version) < (4, 2, 0):
        raise RuntimeError(f"Blender 4.2+ required, found {bpy.app.version_string}")

    extension, module_path, manifest = _installed_extension()
    if not hasattr(bpy.types.Scene, "stflip"):
        raise AssertionError(
            "installed extension is importable but not enabled/registered"
        )
    settings = bpy.context.scene.stflip
    settings.resolution = 8
    settings.cfl_target = 4.0
    settings.st_enabled = True
    if settings.resolution != 8 or not settings.st_enabled:
        raise AssertionError("registered Scene.stflip properties are not writable")

    stflip_module = importlib.import_module(f"{EXTENSION_MODULE}.stflip")
    solver, positions, velocities, residual_max, stats = _tiny_cpu_solver(
        stflip_module)

    reconstruction = stflip_module.reconstruct_surface(
        positions,
        solver.p.dx,
        iterations=2,
        max_voxels=250_000,
    )
    density = solver.be.to_numpy(reconstruction.density)
    if density.ndim != 3 or density.size == 0 or not np.all(np.isfinite(density)):
        raise AssertionError("Appendix-B reconstruction produced an invalid field")
    if float(density.max()) <= 0.5:
        raise AssertionError("Appendix-B field contains no 0.5 isosurface")

    mesher = importlib.import_module(f"{EXTENSION_MODULE}.addon.mesher")
    particle_object = _write_particle_mesh(mesher, positions, velocities)
    preview_object = mesher.ensure_surface_object(
        particle_object,
        solver.p.dx,
        0.5,
        0.5,
    )
    preview_modifier = preview_object.modifiers.get("STFLIP Surface")
    if preview_modifier is None or preview_modifier.type != "NODES":
        raise AssertionError("installed add-on did not create its Geometry Nodes surface")

    openvdb_name, _openvdb = load_openvdb()
    openvdb_result = {
        "available": openvdb_name is not None,
        "module": openvdb_name,
        "required": bool(require_openvdb),
        "polygons": None,
    }
    if openvdb_name is None:
        if require_openvdb:
            raise RuntimeError(
                "OpenVDB was required but neither openvdb nor pyopenvdb imports"
            )
    else:
        vertices, triangles, quads = mesher.density_field_to_polygons(
            density,
            reconstruction.origin,
            reconstruction.voxel_size,
            isovalue=0.5,
            adaptivity=0.0,
        )
        polygon_count = len(triangles) + len(quads)
        if len(vertices) == 0 or polygon_count == 0:
            raise AssertionError("OpenVDB returned an empty 0.5 isosurface")
        paper_object = mesher.ensure_paper_surface_object(
            vertices,
            triangles,
            quads,
            existing_obj=preview_object,
        )
        if len(paper_object.data.polygons) != polygon_count:
            raise AssertionError("Blender paper mesh does not match OpenVDB polygons")
        openvdb_result["polygons"] = polygon_count

    return {
        "status": "passed",
        "blender": {
            "version": bpy.app.version_string,
            "background": bool(bpy.app.background),
        },
        "extension": {
            "module": EXTENSION_MODULE,
            "path": str(module_path),
            "version": manifest["version"],
            "registered_scene_properties": True,
        },
        "solver": {
            "backend": solver.be.name,
            "particles": len(positions),
            "steps": int(stats.steps),
            "residual_abs_max_s": residual_max,
        },
        "surface": {
            "grid_shape": list(density.shape),
            "mcf_iterations": 2,
            "geometry_nodes": True,
            "openvdb": openvdb_result,
        },
    }


def _emit(result: dict, output: Path | None) -> None:
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if output is not None:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(
        script_arguments(sys.argv) if argv is None else list(argv))
    try:
        result = run(require_openvdb=options.require_openvdb)
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _emit(result, options.output)
        traceback.print_exc()
        return 1
    _emit(result, options.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
