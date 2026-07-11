"""Bpy-free per-frame metrics for ST-FLIP simulations.

The metric schema is intentionally flat so the same records can be stored as
JSON Lines and exported losslessly to CSV.  Length-dependent quantities are
named in *solver units*: the solver currently consumes Blender coordinates
directly and does not promise SI conversion for non-default scene unit scales.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


METRICS_SCHEMA = "stflip.frame_metrics"
SCHEMA_VERSION = 2

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
    compute_wall_s: float | None = None,
    mac_grids: Mapping[str, Any] | None = None,
    phase_threshold: float = 0.5,
    array_module=None,
) -> dict[str, Any]:
    """Build one strict schema-v2 record from a cached particle snapshot."""
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
    if count:
        speed_sq = np.einsum(
            "ij,ij->i", velocity, velocity, dtype=np.float64
        )
        speed_max = math.sqrt(float(speed_sq.max()))
        speed_rms = math.sqrt(float(speed_sq.mean(dtype=np.float64)))
        velocity_sum = velocity.sum(axis=0, dtype=np.float64)
        centre = positions.mean(axis=0, dtype=np.float64)
        speed_sq_sum = float(speed_sq.sum(dtype=np.float64))
    else:
        speed_max = speed_rms = speed_sq_sum = 0.0
        velocity_sum = np.zeros(3, dtype=np.float64)
        centre = (None, None, None)

    particle_volume = dx**3 / ppc
    particle_mass = rho * particle_volume
    momentum = particle_mass * velocity_sum
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
        "total_particle_mass_estimate": count * particle_mass,
        "kinetic_energy_particle_estimate": 0.5 * particle_mass * speed_sq_sum,
        "momentum_x_estimate": float(momentum[0]),
        "momentum_y_estimate": float(momentum[1]),
        "momentum_z_estimate": float(momentum[2]),
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
