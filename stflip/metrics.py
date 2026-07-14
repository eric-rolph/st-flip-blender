"""Bpy-free per-frame metrics for ST-FLIP simulations.

The metric schema is intentionally flat so the same records can be stored as
JSON Lines and exported losslessly to CSV.  Length-dependent quantities are
named in *solver units*: the solver currently consumes Blender coordinates
directly and does not promise SI conversion for non-default scene unit scales.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


METRICS_SCHEMA = "stflip.frame_metrics"
# Version 3 (roadmap ENER-M0, the single deliberate frame-record bump):
# adds the angular-momentum estimates and optional phase-weighted masses.
# ``cache.read_metrics`` silently drops rows whose version differs, so
# resumed pre-existing bakes lose their historical metric rows after this
# upgrade -- an accepted, versioned decision (roadmap Decision 2).
SCHEMA_VERSION = 3

# This order is both the strict JSONL schema and the stable CSV column order.
FRAME_FIELD_ORDER = (
    "schema_version",
    "frame",
    "simulation_time_s",
    "target_cfl",
    "particle_count",
    "particles_removed",
    "volume_outflow_removed",
    "pressure_outflow_removed",
    "solver_steps",
    "inactive_time_s",
    "dt_min_s",
    "dt_mean_s",
    "dt_max_s",
    "particle_cfl_estimated_mean",
    "particle_cfl_estimated_max",
    "particle_cfl_actual_mean",
    "particle_cfl_actual_max",
    "pcg_solve_count",
    "pcg_iterations_total",
    "pcg_iterations_max",
    "pcg_relative_residual_max",
    "pcg_converged_all",
    "speed_max_solver_units_per_s",
    "speed_rms_solver_units_per_s",
    "particle_volume_estimate_solver_units3",
    "total_particle_mass_estimate",
    "kinetic_energy_particle_estimate",
    "momentum_x_estimate",
    "momentum_y_estimate",
    "momentum_z_estimate",
    "angular_momentum_x_estimate",
    "angular_momentum_y_estimate",
    "angular_momentum_z_estimate",
    "center_of_mass_local_x_solver_units",
    "center_of_mass_local_y_solver_units",
    "center_of_mass_local_z_solver_units",
    "compute_wall_s",
    "phase_field_threshold",
    "phase_threshold_volume_estimate_solver_units3",
    "phase_threshold_volume_fraction_estimate",
    "mac_grid_enstrophy_estimate",
)

_SCALAR_TYPES = (str, int, float, bool, type(None))


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_values(values: Sequence | None, name: str) -> tuple[float, ...]:
    if values is None:
        return ()
    return tuple(_finite_float(value, name) for value in values)


def _summary(values: tuple[float, ...]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values, dtype=np.float64)), max(values)


def _dt_summary(
    values: tuple[float, ...],
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    return min(values), float(np.mean(values, dtype=np.float64)), max(values)


def _array_namespace(value, array_module=None):
    if array_module is not None:
        return array_module
    if type(value).__module__.split(".", 1)[0] == "cupy":
        import cupy

        return cupy
    return np


def _derivative(xp, values, axis: int, dx: float):
    """First derivative with centered interior and one-sided boundary rows."""
    if values.shape[axis] < 2:
        raise ValueError("MAC-grid dimensions must contain at least two cells")
    result = xp.empty_like(values)
    centre = [slice(None)] * values.ndim
    ahead = [slice(None)] * values.ndim
    behind = [slice(None)] * values.ndim
    centre[axis] = slice(1, -1)
    ahead[axis] = slice(2, None)
    behind[axis] = slice(None, -2)
    result[tuple(centre)] = (
        values[tuple(ahead)] - values[tuple(behind)]
    ) / (2.0 * dx)

    first = [slice(None)] * values.ndim
    second = [slice(None)] * values.ndim
    first[axis] = 0
    second[axis] = 1
    result[tuple(first)] = (
        values[tuple(second)] - values[tuple(first)]
    ) / dx

    last = [slice(None)] * values.ndim
    penultimate = [slice(None)] * values.ndim
    last[axis] = -1
    penultimate[axis] = -2
    result[tuple(last)] = (
        values[tuple(last)] - values[tuple(penultimate)]
    ) / dx
    return result


def estimate_mac_grid_metrics(
    grids: Mapping[str, Any],
    dx: float,
    *,
    phase_threshold: float = 0.5,
    array_module=None,
) -> dict[str, float]:
    """Estimate liquid volume and enstrophy from a staggered MAC grid.

    ``c_phi >= phase_threshold`` is treated as liquid.  This phase threshold is
    not a geometric volume fraction, so the returned volume is explicitly an
    estimate.  Enstrophy uses cell-centred face averages and finite differences
    to approximate ``0.5 * integral(|curl(u)|**2) dV`` over that mask.
    """
    dx = _finite_float(dx, "dx")
    threshold = _finite_float(phase_threshold, "phase_threshold")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("phase_threshold must be between zero and one")

    try:
        u, v, w, phase = (grids[name] for name in ("u", "v", "w", "c_phi"))
    except KeyError as exc:
        raise ValueError(f"missing MAC-grid field {exc.args[0]!r}") from exc
    xp = _array_namespace(phase, array_module)
    phase = xp.asarray(phase)
    if phase.ndim != 3:
        raise ValueError("c_phi must be a three-dimensional cell grid")
    nx, ny, nz = phase.shape
    expected = {
        "u": (nx + 1, ny, nz),
        "v": (nx, ny + 1, nz),
        "w": (nx, ny, nz + 1),
    }
    arrays = {"u": xp.asarray(u), "v": xp.asarray(v), "w": xp.asarray(w)}
    for name, values in arrays.items():
        if values.shape != expected[name]:
            raise ValueError(
                f"{name} has shape {values.shape}, expected {expected[name]}"
            )

    uc = 0.5 * (arrays["u"][:-1] + arrays["u"][1:])
    vc = 0.5 * (arrays["v"][:, :-1] + arrays["v"][:, 1:])
    wc = 0.5 * (arrays["w"][:, :, :-1] + arrays["w"][:, :, 1:])

    omega = _derivative(xp, wc, 1, dx) - _derivative(xp, vc, 2, dx)
    omega_sq = omega * omega
    omega = _derivative(xp, uc, 2, dx) - _derivative(xp, wc, 0, dx)
    omega_sq += omega * omega
    omega = _derivative(xp, vc, 0, dx) - _derivative(xp, uc, 1, dx)
    omega_sq += omega * omega

    liquid = phase >= threshold
    cell_volume = dx**3
    volume = float((liquid.sum() * cell_volume).item())
    enstrophy = float((0.5 * (omega_sq * liquid).sum() * cell_volume).item())
    if not math.isfinite(volume) or not math.isfinite(enstrophy):
        raise ValueError("MAC-grid metric inputs produced a non-finite result")
    return {
        "phase_field_threshold": threshold,
        "phase_threshold_volume_estimate_solver_units3": volume,
        "phase_threshold_volume_fraction_estimate": (
            float(liquid.mean().item())
        ),
        "mac_grid_enstrophy_estimate": enstrophy,
    }


def measure_frame(
    *,
    frame: int,
    simulation_time_s: float,
    params,
    stats,
    positions_local: np.ndarray,
    velocities: np.ndarray,
    phases: np.ndarray | None = None,
    compute_wall_s: float | None = None,
    mac_grids: Mapping[str, Any] | None = None,
    phase_threshold: float = 0.5,
    array_module=None,
) -> dict[str, Any]:
    """Build one strict schema-v3 record from a cached particle snapshot.

    Positions are expected to be the render-resynchronized snapshot
    (``get_render_particles``), not raw jittered positions: temporal jitter
    noise grows with CFL and would otherwise contaminate exactly the
    energy/angular-momentum decay curves this schema exists to gate.  The
    angular momentum is taken about the (mass-weighted, when ``phases`` is
    given) particle centre so a drifting bulk does not read as spin.
    """
    if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)):
        raise TypeError("frame must be an integer")
    simulation_time = _finite_float(simulation_time_s, "simulation_time_s")
    if compute_wall_s is not None:
        compute_wall_s = _finite_float(compute_wall_s, "compute_wall_s")
        if compute_wall_s < 0.0:
            raise ValueError("compute_wall_s must not be negative")

    positions = np.asarray(positions_local)
    velocity = np.asarray(velocities)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("positions_local must have shape (N, 3)")
    if velocity.shape != positions.shape:
        raise ValueError("velocities must have the same (N, 3) shape")
    if not np.issubdtype(positions.dtype, np.number):
        raise TypeError("positions_local must be numeric")
    if not np.issubdtype(velocity.dtype, np.number):
        raise TypeError("velocities must be numeric")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocity)):
        raise ValueError("particle snapshots must contain only finite values")

    dx = _finite_float(params.dx, "params.dx")
    rho = _finite_float(params.rho, "params.rho")
    target_cfl = _finite_float(params.cfl_target, "params.cfl_target")
    ppc = int(params.particles_per_cell)
    if dx <= 0.0 or rho < 0.0 or ppc <= 0:
        raise ValueError("dx and particles_per_cell must be positive; rho >= 0")

    dt_values = _finite_values(
        getattr(stats, "dt_values", None) if stats is not None else None,
        "dt_values",
    )
    dt_min, dt_mean, dt_max = _dt_summary(dt_values)
    inactive_time = _finite_float(
        getattr(stats, "inactive_time_s", 0.0) if stats is not None else 0.0,
        "inactive_time_s",
    )
    if inactive_time < 0.0:
        raise ValueError("inactive_time_s must not be negative")
    estimated_cfl = _finite_values(
        getattr(stats, "particle_cfl_estimated_values", None)
        if stats is not None
        else None,
        "particle_cfl_estimated_values",
    )
    actual_cfl = _finite_values(
        getattr(stats, "particle_cfl_actual_values", None)
        if stats is not None
        else None,
        "particle_cfl_actual_values",
    )
    estimated_mean, estimated_max = _summary(estimated_cfl)
    actual_mean, actual_max = _summary(actual_cfl)

    pcg_iters = tuple(
        int(value)
        for value in (
            getattr(stats, "pcg_iters", ()) if stats is not None else ()
        )
    )
    if any(value < 0 for value in pcg_iters):
        raise ValueError("pcg_iters must not contain negative values")
    pcg_residuals = _finite_values(
        getattr(stats, "pcg_rel_residuals", None) if stats is not None else None,
        "pcg_rel_residuals",
    )
    if pcg_residuals and len(pcg_residuals) != len(pcg_iters):
        raise ValueError("pcg residual and iteration counts must have equal length")
    if pcg_residuals:
        pcg_converged = all(value <= float(params.pcg_tol) for value in pcg_residuals)
        pcg_residual_max = max(pcg_residuals)
    else:
        pcg_converged = None
        pcg_residual_max = None

    count = int(positions.shape[0])
    particle_volume = dx**3 / ppc
    particle_mass = rho * particle_volume
    # Optional per-particle phase tags (liquid > 0.5) make the mass-weighted
    # estimates below correct for two-phase snapshots; without them every
    # particle carries the uniform liquid mass, exactly as in schema v2.
    masses = None
    if phases is not None:
        phase_values = np.asarray(phases)
        if phase_values.shape != (count,):
            raise ValueError("phases must have shape (N,)")
        if not np.issubdtype(phase_values.dtype, np.number):
            raise TypeError("phases must be numeric")
        if count and not np.all(np.isfinite(phase_values)):
            raise ValueError("phases must contain only finite values")
        rho_gas = _finite_float(getattr(params, "rho_gas", 0.0), "params.rho_gas")
        if rho_gas < 0.0:
            raise ValueError("params.rho_gas must not be negative")
        masses = particle_volume * np.where(
            phase_values.astype(np.float64) > 0.5, rho, rho_gas)
    if count:
        speed_sq = np.einsum(
            "ij,ij->i", velocity, velocity, dtype=np.float64
        )
        speed_max = math.sqrt(float(speed_sq.max()))
        speed_rms = math.sqrt(float(speed_sq.mean(dtype=np.float64)))
        speed_sq_sum = float(speed_sq.sum(dtype=np.float64))
        if masses is None:
            total_mass = count * particle_mass
            momentum = particle_mass * velocity.sum(axis=0, dtype=np.float64)
            kinetic_energy = 0.5 * particle_mass * speed_sq_sum
            centre = positions.mean(axis=0, dtype=np.float64)
        else:
            total_mass = float(masses.sum(dtype=np.float64))
            momentum = (masses[:, None] * velocity).sum(
                axis=0, dtype=np.float64)
            kinetic_energy = 0.5 * float(
                (masses * speed_sq).sum(dtype=np.float64))
            if total_mass > 0.0:
                centre = (masses[:, None] * positions).sum(
                    axis=0, dtype=np.float64) / total_mass
            else:
                centre = positions.mean(axis=0, dtype=np.float64)
        relative = positions.astype(np.float64) - centre
        weighted_velocity = (
            particle_mass * velocity.astype(np.float64)
            if masses is None
            else masses[:, None] * velocity.astype(np.float64)
        )
        angular_momentum = np.cross(relative, weighted_velocity).sum(
            axis=0, dtype=np.float64)
    else:
        speed_max = speed_rms = speed_sq_sum = 0.0
        total_mass = 0.0
        kinetic_energy = 0.0
        momentum = np.zeros(3, dtype=np.float64)
        angular_momentum = np.zeros(3, dtype=np.float64)
        centre = (None, None, None)
    removed_counts = {
        name: int(getattr(stats, name, 0) if stats is not None else 0)
        for name in (
            "particles_removed",
            "volume_outflow_removed",
            "pressure_outflow_removed",
        )
    }
    if any(value < 0 for value in removed_counts.values()):
        raise ValueError("outflow removal counts must not be negative")
    if removed_counts["particles_removed"] != (
        removed_counts["volume_outflow_removed"]
        + removed_counts["pressure_outflow_removed"]
    ):
        raise ValueError("outflow removal counts must sum to particles_removed")
    record = {
        "schema_version": SCHEMA_VERSION,
        "frame": int(frame),
        "simulation_time_s": simulation_time,
        "target_cfl": target_cfl,
        "particle_count": count,
        **removed_counts,
        "solver_steps": int(getattr(stats, "steps", 0) if stats is not None else 0),
        "inactive_time_s": inactive_time,
        "dt_min_s": dt_min,
        "dt_mean_s": dt_mean,
        "dt_max_s": dt_max,
        "particle_cfl_estimated_mean": estimated_mean,
        "particle_cfl_estimated_max": estimated_max,
        "particle_cfl_actual_mean": actual_mean,
        "particle_cfl_actual_max": actual_max,
        "pcg_solve_count": len(pcg_iters),
        "pcg_iterations_total": sum(pcg_iters),
        "pcg_iterations_max": max(pcg_iters) if pcg_iters else None,
        "pcg_relative_residual_max": pcg_residual_max,
        "pcg_converged_all": pcg_converged,
        "speed_max_solver_units_per_s": speed_max,
        "speed_rms_solver_units_per_s": speed_rms,
        "particle_volume_estimate_solver_units3": count * particle_volume,
        "total_particle_mass_estimate": total_mass,
        "kinetic_energy_particle_estimate": kinetic_energy,
        "momentum_x_estimate": float(momentum[0]),
        "momentum_y_estimate": float(momentum[1]),
        "momentum_z_estimate": float(momentum[2]),
        "angular_momentum_x_estimate": float(angular_momentum[0]),
        "angular_momentum_y_estimate": float(angular_momentum[1]),
        "angular_momentum_z_estimate": float(angular_momentum[2]),
        "center_of_mass_local_x_solver_units": (
            None if centre[0] is None else float(centre[0])
        ),
        "center_of_mass_local_y_solver_units": (
            None if centre[1] is None else float(centre[1])
        ),
        "center_of_mass_local_z_solver_units": (
            None if centre[2] is None else float(centre[2])
        ),
        "compute_wall_s": compute_wall_s,
        "phase_field_threshold": None,
        "phase_threshold_volume_estimate_solver_units3": None,
        "phase_threshold_volume_fraction_estimate": None,
        "mac_grid_enstrophy_estimate": None,
    }
    if mac_grids is not None:
        record.update(
            estimate_mac_grid_metrics(
                mac_grids,
                dx,
                phase_threshold=phase_threshold,
                array_module=array_module,
            )
        )
    validate_frame_record(record)
    return record


def validate_frame_record(record: Mapping[str, Any]) -> None:
    """Validate exact schema, scalar values, and JSON-finite numeric values."""
    if not isinstance(record, Mapping):
        raise TypeError("metric record must be a mapping")
    if set(record) != set(FRAME_FIELD_ORDER):
        missing = [name for name in FRAME_FIELD_ORDER if name not in record]
        extra = [name for name in record if name not in FRAME_FIELD_ORDER]
        raise ValueError(f"invalid metric fields; missing={missing}, extra={extra}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported metric schema {record['schema_version']!r}")
    frame = record["frame"]
    if isinstance(frame, bool) or not isinstance(frame, int):
        raise TypeError("metric frame must be an integer")
    for name in (
        "particle_count",
        "particles_removed",
        "volume_outflow_removed",
        "pressure_outflow_removed",
        "solver_steps",
    ):
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"metric field {name!r} must be an integer")
        if value < 0:
            raise ValueError(f"metric field {name!r} must not be negative")
    if record["particles_removed"] != (
        record["volume_outflow_removed"]
        + record["pressure_outflow_removed"]
    ):
        raise ValueError("outflow removal counts must sum to particles_removed")
    inactive_time = record["inactive_time_s"]
    if isinstance(inactive_time, bool) or not isinstance(
        inactive_time, (int, float)
    ):
        raise TypeError("metric field 'inactive_time_s' must be numeric")
    if inactive_time < 0.0:
        raise ValueError("metric field 'inactive_time_s' must not be negative")
    for name, value in record.items():
        if not isinstance(value, _SCALAR_TYPES):
            raise TypeError(f"metric field {name!r} must be a JSON scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metric field {name!r} must be finite or null")


# --------------------------------------------------------------------------
# Calm-surface height metrics (roadmap CALM-M1).
#
# These are standalone helpers consumed by the calm-surface validation
# scenarios; they never touch the strict frame-record schema above.


def particle_height_map(
    positions: np.ndarray,
    *,
    dx: float,
    resolution: Sequence[int],
    particles_per_cell: int,
    floor_z: float = 0.0,
) -> np.ndarray:
    """Column water-height map from particle counts.

    ``h(x, y) = dx * column_count(x, y) / particles_per_cell`` resolves
    sub-voxel surface motion: one particle entering or leaving a column moves
    the estimate by ``dx / particles_per_cell``.  A boolean occupancy column
    sum would quantize to whole cells, which hides exactly the low-amplitude
    calm-surface noise this metric exists to measure.
    """

    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("positions must be an (N, 3) array")
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("positions must be finite")
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be finite and positive")
    if isinstance(particles_per_cell, bool) or not isinstance(
            particles_per_cell, int) or particles_per_cell < 1:
        raise ValueError("particles_per_cell must be a positive integer")
    shape = tuple(int(n) for n in resolution)
    if len(shape) != 3 or any(n < 1 for n in shape):
        raise ValueError("resolution must contain three positive integers")
    nx, ny = shape[0], shape[1]
    heights = np.zeros((nx, ny), dtype=np.float64)
    if values.size:
        above = values[values[:, 2] >= floor_z]
        if len(above):
            ix = np.clip((above[:, 0] / dx).astype(np.int64), 0, nx - 1)
            iy = np.clip((above[:, 1] / dx).astype(np.int64), 0, ny - 1)
            np.add.at(heights, (ix, iy), 1.0)
    heights *= dx / float(particles_per_cell)
    return heights


def surface_height_map(
    density: np.ndarray,
    *,
    origin: Sequence[float],
    voxel_size: float,
    iso: float = 0.5,
) -> np.ndarray:
    """Per-column height of the highest ``iso`` crossing of a render density.

    Operates on the Appendix-B reconstruction density (``reconstruct_surface``
    output moved to the host).  The crossing between the highest wet voxel and
    its dry neighbour above is located by linear interpolation, giving
    sub-voxel sensitivity; this is the primary gating variant because it
    measures the surface users actually see.  Columns with no wet voxel
    return NaN.
    """

    field = np.asarray(density, dtype=np.float64)
    if field.ndim != 3:
        raise ValueError("density must be a 3D array")
    anchor = tuple(float(v) for v in origin)
    if len(anchor) != 3 or not all(math.isfinite(v) for v in anchor):
        raise ValueError("origin must contain three finite values")
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size must be finite and positive")
    if not math.isfinite(iso):
        raise ValueError("iso must be finite")
    nx, ny, nz = field.shape
    heights = np.full((nx, ny), np.nan, dtype=np.float64)
    if 0 in field.shape:
        return heights
    wet = field >= iso
    any_wet = wet.any(axis=2)
    if not any_wet.any():
        return heights
    # Highest wet voxel per column: argmax over the reversed z axis.
    top = nz - 1 - np.argmax(wet[:, :, ::-1], axis=2)
    ii, jj = np.nonzero(any_wet)
    kk = top[ii, jj]
    d_wet = field[ii, jj, kk]
    interior = kk < nz - 1
    d_dry = np.where(interior, field[ii, jj, np.minimum(kk + 1, nz - 1)], iso)
    # Fraction of the voxel above the wet centre where density falls to iso.
    delta = d_wet - d_dry
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(delta > 0.0, (d_wet - iso) / delta, 0.5)
    frac = np.clip(frac, 0.0, 1.0)
    # Wet voxels saturated to the domain top report their upper voxel face.
    frac = np.where(interior, frac, 0.5)
    heights[ii, jj] = anchor[2] + (kk + 0.5 + frac) * voxel_size
    return heights


def height_map_stats(
    height_map: np.ndarray,
    *,
    reference_map: np.ndarray | None = None,
) -> dict[str, Any]:
    """Spatial statistics of one height map over its wet columns.

    ``reference_map`` (typically the per-column temporal mean) is subtracted
    before spatial statistics when given, so scenes whose equilibrium surface
    is legitimately non-flat (a stirred pool dips at the vortex core) can be
    measured for NOISE about their own steady shape rather than for shape.
    """

    values = np.asarray(height_map, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("height_map must be a 2D array")
    wet = np.isfinite(values)
    if reference_map is not None:
        reference = np.asarray(reference_map, dtype=np.float64)
        if reference.shape != values.shape:
            raise ValueError("reference_map shape must match height_map")
        wet = wet & np.isfinite(reference)
    count = int(np.count_nonzero(wet))
    if count == 0:
        return {
            "wet_column_count": 0,
            "height_mean": None,
            "height_rms_spatial": None,
        }
    sample = values[wet]
    if reference_map is not None:
        sample = sample - np.asarray(reference_map, dtype=np.float64)[wet]
    mean = float(sample.mean(dtype=np.float64))
    rms = float(np.sqrt(np.mean((sample - mean) ** 2, dtype=np.float64)))
    return {
        "wet_column_count": count,
        "height_mean": mean,
        "height_rms_spatial": rms,
    }


def process_peak_memory_bytes() -> int | None:
    """Best-effort peak resident-set size of this process, in bytes.

    Several roadmap features stack transient memory (particle ids, temporal
    moment grids, reflection snapshots); this probe lets validation runs
    report the stack without adding a dependency.  Returns None when the
    platform offers no cheap peak-RSS source.
    """

    try:
        import resource
    except ImportError:
        resource = None
    if resource is not None:
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # ru_maxrss is kilobytes on Linux and bytes on macOS.
        scale = 1 if sys.platform == "darwin" else 1024
        return int(peak * scale)
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes

        class _MemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(_MemoryCounters)
        # The pseudo-handle is (HANDLE)-1; it must travel as a pointer-sized
        # value or the upper 32 bits are lost on 64-bit Python.
        handle = ctypes.c_void_p(-1)
        # Modern Windows exports the call from kernel32 as K32...; the
        # legacy psapi export is a fallback for older systems.
        for module, name in (
            (ctypes.windll.kernel32, "K32GetProcessMemoryInfo"),
            (ctypes.windll.psapi, "GetProcessMemoryInfo"),
        ):
            query = getattr(module, name, None)
            if query is None:
                continue
            query.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_MemoryCounters),
                ctypes.wintypes.DWORD,
            ]
            query.restype = ctypes.wintypes.BOOL
            if query(handle, ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
    return None
