"""Executable, auditable small-scale paper-reference benchmark runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import numpy as np

from .backend import cuda_device_name, get_backend
from .benchmarks import (
    GlugBenchmark,
    KleefsmanBenchmark,
    compare_gauge_series,
    load_gauge_reference,
    water_height_from_particles,
)
from .solver import Params, STFLIPSolver


PAPER_VALIDATION_SCHEMA = "stflip-paper-reference-v1"
DEFAULT_GLUG_SURFACE_TENSION = 0.072


@dataclass(frozen=True, slots=True)
class PaperRunConfig:
    """Shared controls for one compact reference run."""

    longest_resolution: int = 32
    dx: float | None = None
    frames: int = 12
    frame_rate: float = 24.0
    cfl_target: float = 8.0
    particles_per_cell: int = 2
    gas_particles_per_cell: int = 2
    glug_surface_tension: float = DEFAULT_GLUG_SURFACE_TENSION
    seed: int = 0
    backend: str = "cpu"
    pressure_solver: str = "multigrid"

    def __post_init__(self) -> None:
        if (isinstance(self.longest_resolution, bool)
                or not isinstance(self.longest_resolution, int)
                or self.longest_resolution < 8):
            raise ValueError("longest_resolution must be an integer >= 8")
        if self.dx is not None and (
                isinstance(self.dx, bool) or not math.isfinite(self.dx)
                or self.dx <= 0.0):
            raise ValueError("dx must be finite and positive")
        for name in ("frames", "particles_per_cell", "gas_particles_per_cell"):
            value = getattr(self, name)
            minimum = 0 if name == "frames" else 1
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("frame_rate", "cfl_target"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if (isinstance(self.glug_surface_tension, bool)
                or not isinstance(self.glug_surface_tension, (int, float))
                or not math.isfinite(self.glug_surface_tension)
                or self.glug_surface_tension < 0.0):
            raise ValueError("glug_surface_tension must be finite and non-negative")
        if self.backend not in {"cpu", "cuda"}:
            raise ValueError("backend must be cpu or cuda")
        if self.pressure_solver not in {"jacobi", "multigrid"}:
            raise ValueError("pressure_solver must be jacobi or multigrid")


def _grid_for(case, config: PaperRunConfig):
    return case.grid(
        dx=config.dx,
        longest_resolution=None if config.dx is not None
        else config.longest_resolution,
    )


def _params(
    grid,
    config: PaperRunConfig,
    *,
    two_phase: bool,
    surface_tension: float = 0.0,
) -> Params:
    return Params(
        resolution=grid.shape,
        dx=grid.dx,
        gravity=(0.0, 0.0, -9.81),
        frame_dt=1.0 / config.frame_rate,
        cfl_target=config.cfl_target,
        particles_per_cell=config.particles_per_cell,
        seed=config.seed,
        pressure_solver=config.pressure_solver,
        two_phase=two_phase,
        gas_particles_per_cell=config.gas_particles_per_cell,
        surface_tension=surface_tension,
    )


def _grid_metadata(grid) -> dict[str, Any]:
    return {
        "requested_extent_m": grid.requested_extent,
        "effective_extent_m": grid.effective_extent,
        "shape": grid.shape,
        "dx_m": grid.dx,
        "cell_count": int(np.prod(grid.shape, dtype=np.int64)),
    }


def _stats_record(stats) -> dict[str, Any]:
    return {
        "steps": stats.steps,
        "particles": stats.n_particles,
        "max_speed_m_s": stats.max_speed,
        "dt_s": list(stats.dt_values),
        "particle_cfl_actual": list(stats.particle_cfl_actual_values),
        "pressure_iterations": list(stats.pcg_iters),
        "pressure_relative_residuals": list(stats.pcg_rel_residuals),
        "particles_removed": stats.particles_removed,
        "volume_outflow_removed": stats.volume_outflow_removed,
        "pressure_outflow_removed": stats.pressure_outflow_removed,
    }


def _implementation_metadata() -> dict[str, Any]:
    """Identify the exact Python source and runtime used for a run."""

    from . import __version__

    source_root = Path(__file__).resolve().parent
    source_files = sorted(source_root.glob("*.py"), key=lambda path: path.name)
    digest = hashlib.sha256()
    for path in source_files:
        payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return {
        "package": "st-flip-blender",
        "version": __version__,
        "python_source_sha256": digest.hexdigest(),
        "source_files": [path.name for path in source_files],
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
    }


def _solver_metadata(solver, config: PaperRunConfig) -> dict[str, Any]:
    array_library = getattr(solver.be.xp, "__name__", type(solver.be.xp).__name__)
    array_library_version = getattr(solver.be.xp, "__version__", None)
    return {
        "backend_requested": config.backend,
        "backend_used": solver.be.name,
        "backend_runtime": {
            "array_library": str(array_library),
            "array_library_version": (
                str(array_library_version)
                if array_library_version is not None else None
            ),
            "cupy_version": (
                str(array_library_version)
                if solver.be.name == "cuda" and array_library_version is not None
                else None
            ),
            "cuda_device": (
                cuda_device_name() if solver.be.name == "cuda" else None
            ),
        },
        "normalized_params": asdict(solver.p),
    }


def run_kleefsman_benchmark(
    config: PaperRunConfig,
    *,
    reference_csv: str | Path | None = None,
    reference_citation: str | None = None,
) -> dict[str, Any]:
    """Run the published 3-D geometry and record H2/H4 water heights."""

    if reference_csv is None:
        if reference_citation is not None:
            raise ValueError("reference_citation requires reference_csv")
    elif (not isinstance(reference_citation, str)
          or not reference_citation.strip()):
        raise ValueError(
            "reference_citation must be a nonblank string with reference_csv")
    if reference_citation is not None:
        reference_citation = reference_citation.strip()
    case = KleefsmanBenchmark()
    grid = _grid_for(case, config)
    backend = get_backend(config.backend)
    solver = STFLIPSolver(_params(grid, config, two_phase=False), backend)
    solver.set_solid_sdf(
        case.solid_sdf_cells(grid), node_sdf=case.solid_sdf_nodes(grid))
    open_roof = np.zeros(grid.shape, dtype=bool)
    open_roof[:, :, -1] = True
    solver.add_outflow(open_roof, mode="PRESSURE", faces=("z_max",))
    solver.add_liquid_mask(case.liquid_mask(grid))

    times = [0.0]
    gauge_values = {name: [] for name, _xy in case.gauges_xy}
    frame_records = []

    def sample() -> None:
        positions, _velocities = solver.get_render_particles()
        for name, xy in case.gauges_xy:
            gauge_values[name].append(water_height_from_particles(
                positions, xy, dx=grid.dx))

    sample()
    for frame in range(1, config.frames + 1):
        stats = solver.step_frame()
        backend.synchronize()
        times.append(float(solver.time))
        sample()
        frame_records.append({"frame": frame, **_stats_record(stats)})

    artifact: dict[str, Any] = {
        "schema": PAPER_VALIDATION_SCHEMA,
        "case": "kleefsman",
        "claim_scope": (
            "Published geometry and seeded gauge sampling. This becomes "
            "experimental validation only when an attributable reference CSV is supplied."
        ),
        "implementation": _implementation_metadata(),
        "geometry": case.metadata(),
        "grid": _grid_metadata(grid),
        "config": asdict(config),
        "solver": _solver_metadata(solver, config),
        "boundary_conditions": {
            "domain_faces": {
                "x_min": "closed_free_slip_no_through",
                "x_max": "closed_free_slip_no_through",
                "y_min": "closed_free_slip_no_through",
                "y_max": "closed_free_slip_no_through",
                "z_min": "closed_free_slip_no_through",
                "z_max": "atmospheric_pressure_p0_outflow",
            },
            "obstacle": "static_sdf_free_slip_no_through",
            "registered_outflow": solver.outflow_stats(),
        },
        "physics_assumptions": {
            "surface_tension_n_per_m": 0.0,
            "surface_tension_provenance": (
                "The ST-FLIP paper states surface tension was disabled for "
                "all reported experiments except glugging."
            ),
        },
        "samples": {
            "time_s": times,
            "water_height_m": gauge_values,
            "method": {
                "footprint_radius_m": 1.5 * grid.dx,
                "vertical_estimator": (
                    "highest bottom-connected occupied dx layer; one empty "
                    "layer tolerated; detached spray ignored"),
            },
        },
        "frames": frame_records,
    }
    if reference_csv is not None:
        reference = load_gauge_reference(reference_csv)
        artifact["reference"] = {
            "citation": reference_citation,
            "sha256": reference["sha256"],
            "path_name": Path(reference_csv).name,
        }
        artifact["comparison"] = compare_gauge_series(
            times, gauge_values, reference)
    else:
        artifact["reference"] = None
        artifact["comparison"] = None
    return artifact


def _glug_phase_record(solver, case: GlugBenchmark) -> dict[str, Any]:
    positions, phase = solver.get_render_phase_particles()
    positions = np.asarray(positions, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float32)
    liquid = phase >= 0.5
    gas = ~liquid
    scale = case.length_scale
    margin = case.margin_ratio * scale
    height = case.container_height_ratio * scale
    neck = case.connector_length_ratio * scale
    lower = (positions[:, 2] >= margin) & (positions[:, 2] <= margin + height)
    connector = ((positions[:, 2] > margin + height)
                 & (positions[:, 2] < margin + height + neck))
    upper = positions[:, 2] >= margin + height + neck
    liquid_total = int(np.count_nonzero(liquid))
    gas_total = int(np.count_nonzero(gas))
    return {
        "liquid_particles": liquid_total,
        "gas_particles": gas_total,
        "liquid_lower_fraction": (
            float(np.count_nonzero(liquid & lower) / liquid_total)
            if liquid_total else 0.0),
        "liquid_connector_fraction": (
            float(np.count_nonzero(liquid & connector) / liquid_total)
            if liquid_total else 0.0),
        "liquid_upper_fraction": (
            float(np.count_nonzero(liquid & upper) / liquid_total)
            if liquid_total else 0.0),
        "gas_lower_fraction": (
            float(np.count_nonzero(gas & lower) / gas_total)
            if gas_total else 0.0),
        "liquid_center_of_mass_z_m": (
            float(np.mean(positions[liquid, 2])) if liquid_total else None),
    }


def run_glug_benchmark(config: PaperRunConfig) -> dict[str, Any]:
    """Run the paper-constrained two-phase glugging geometry."""

    case = GlugBenchmark()
    grid = _grid_for(case, config)
    backend = get_backend(config.backend)
    solver = STFLIPSolver(
        _params(
            grid,
            config,
            two_phase=True,
            surface_tension=config.glug_surface_tension,
        ),
        backend,
    )
    solver.set_solid_sdf(
        case.solid_sdf_cells(grid), node_sdf=case.solid_sdf_nodes(grid))
    solver.add_liquid_mask(case.liquid_mask(grid))
    solver.fill_gas()

    phase_samples = [{"time_s": 0.0, **_glug_phase_record(solver, case)}]
    frame_records = []
    for frame in range(1, config.frames + 1):
        stats = solver.step_frame()
        backend.synchronize()
        phase_samples.append({
            "time_s": float(solver.time),
            **_glug_phase_record(solver, case),
        })
        frame_records.append({"frame": frame, **_stats_record(stats)})
    return {
        "schema": PAPER_VALIDATION_SCHEMA,
        "case": "glug",
        "claim_scope": (
            "Two-phase regression in the paper-published dimension ratios. "
            "Unpublished wall/layout choices and the surface-tension value are "
            "explicit assumptions; this is not a PF-FLIP-equivalence claim."
        ),
        "implementation": _implementation_metadata(),
        "geometry": case.metadata(),
        "grid": _grid_metadata(grid),
        "config": asdict(config),
        "solver": _solver_metadata(solver, config),
        "boundary_conditions": {
            "connected_cavity_walls": "static_sdf_free_slip_no_through",
            "external_outflow": None,
            "registered_outflow": solver.outflow_stats(),
        },
        "physics_assumptions": {
            "surface_tension_n_per_m": config.glug_surface_tension,
            "surface_tension_provenance": (
                "Implementation assumption approximating a water-air interface; "
                "the ST-FLIP paper states surface tension is enabled for Figure "
                "23 but does not publish its coefficient."
            ),
        },
        "samples": phase_samples,
        "sampling_method": {
            "positions": (
                "all liquid and gas particles re-synchronized to the common "
                "global output time using ST-FLIP Algorithm 1 lines 31-34"
            ),
            "phase_alignment": (
                "phase values are filtered by the same outflow-survivor mask "
                "as the re-synchronized positions"
            ),
        },
        "frames": frame_records,
    }


def write_paper_artifact(path: str | Path, artifact: dict[str, Any]) -> Path:
    """Atomically write one strict JSON benchmark artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    handle, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return output


__all__ = [
    "DEFAULT_GLUG_SURFACE_TENSION",
    "PAPER_VALIDATION_SCHEMA",
    "PaperRunConfig",
    "run_glug_benchmark",
    "run_kleefsman_benchmark",
    "write_paper_artifact",
]
