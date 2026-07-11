"""Deterministic, bpy-free validation of ST-FLIP at matched CFL values.

The four-case matrix deliberately keeps the compute backend fixed.  It tests
the spatiotemporal mechanism independently from CUDA acceleration by varying
only temporal sampling (ST-FLIP versus instantaneous P2G) and target CFL
(1 versus 16).  Timing is reported separately from trajectory-quality
comparisons and is never used as a correctness oracle.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import __version__, kernels
from .backend import get_backend
from .metrics import METRICS_SCHEMA, SCHEMA_VERSION, measure_frame
from .solver import Params, STFLIPSolver


VALIDATION_SCHEMA = "stflip.matched_ablation_validation"
VALIDATION_VERSION = 3
MULTI_SEED_SCHEMA = "stflip.multi_seed_matched_validation"
MULTI_SEED_VERSION = 2


@dataclass(frozen=True, slots=True)
class ValidationCase:
    identifier: str
    label: str
    st_enabled: bool
    cfl_target: float


MATCHED_CASES = (
    ValidationCase("st_cfl_1", "ST-FLIP, CFL 1", True, 1.0),
    ValidationCase(
        "instantaneous_cfl_1", "Instantaneous P2G, CFL 1", False, 1.0),
    ValidationCase("st_cfl_16", "ST-FLIP, CFL 16", True, 16.0),
    ValidationCase(
        "instantaneous_cfl_16",
        "Instantaneous P2G, CFL 16",
        False,
        16.0,
    ),
)


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Configuration shared by all four matched cases."""

    resolution: int = 16
    frames: int = 4
    particles_per_cell: int = 2
    seed: int = 0
    backend: str = "cpu"
    high_cfl_threshold: float = 8.0
    frame_rate: float = 24.0
    gravity_z: float = -9.81

    def __post_init__(self) -> None:
        for name in ("resolution", "frames", "particles_per_cell", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.resolution < 4:
            raise ValueError("resolution must be at least 4")
        if self.frames < 1:
            raise ValueError("frames must be positive")
        if self.particles_per_cell < 1:
            raise ValueError("particles_per_cell must be positive")
        if self.seed < 0:
            raise ValueError("seed must not be negative")
        if self.backend not in {"cpu", "cuda"}:
            raise ValueError("backend must be 'cpu' or 'cuda'")
        for name in ("high_cfl_threshold", "frame_rate", "gravity_z"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.high_cfl_threshold <= 0.0:
            raise ValueError("high_cfl_threshold must be positive")
        if self.frame_rate <= 0.0:
            raise ValueError("frame_rate must be positive")


@dataclass(slots=True)
class _FrameOutput:
    positions: np.ndarray
    velocities: np.ndarray
    metrics: dict[str, Any]
    output_sha256: str
    dt_resid_abs_max_s: float
    dt_resid_rms_s: float
    dt_prev_s: float
    dt_max_seen_s: float
    normalized_deposited_mass: np.ndarray
    phase_density: np.ndarray
    temporal_quadrature_state: dict[str, Any]


@dataclass(slots=True)
class _CaseOutput:
    case: ValidationCase
    params: Params
    initial_particle_state_sha256: str
    initial_shared_state_sha256: str
    initial_checkpoint_sha256: str
    m0: float
    frames: list[_FrameOutput]
    timing: dict[str, Any]


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _hash_initial_checkpoint(
    state: Mapping[str, Any],
    m0: float,
    *,
    include_dt_prev: bool = True,
) -> str:
    """Hash trajectory state, including RNG/m0 and optionally prior dt.

    ``dt_prev`` is intentionally initialized from the target CFL, so it must
    remain in each case's full checkpoint fingerprint but is omitted from the
    shared-initialization fingerprint used across the matched CFL matrix.
    """
    digest = hashlib.sha256()
    digest.update(
        b"stflip-initial-checkpoint-v1\0"
        if include_dt_prev
        else b"stflip-shared-initial-state-v1\0"
    )
    for name in ("pos", "vel", "dt_resid"):
        value = np.ascontiguousarray(state[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(json.dumps(
            {"dtype": value.dtype.str, "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii"))
        digest.update(memoryview(value).cast("B"))
    scalar_names = [
        "time",
        "rng_state",
        "outflow_removed_total",
        "volume_outflow_removed_total",
        "pressure_outflow_removed_total",
    ]
    if include_dt_prev:
        scalar_names.append("dt_prev")
    scalar_state = {name: state[name] for name in scalar_names}
    scalar_state["m0"] = float(m0)
    digest.update(json.dumps(
        scalar_state,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"))
    return digest.hexdigest()


def _source_revision() -> str | None:
    """Best-effort Git commit for provenance; packaged installs return None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            check=False,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip().lower()
    if result.returncode or len(value) != 40 \
            or any(character not in "0123456789abcdef" for character in value):
        return None
    return value


def _source_worktree_dirty() -> bool | None:
    """Best-effort dirty flag complementing the recorded Git revision."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            check=False,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    return bool(result.stdout.strip())


def _source_file_hashes() -> dict[str, str]:
    """Hash the bpy-free solver sources that define validation behavior."""
    package_root = Path(__file__).resolve().parent
    result = {}
    for path in sorted(package_root.glob("*.py")):
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        result[path.name] = hashlib.sha256(payload).hexdigest()
    return result


def _dam_break_mask(resolution: int) -> np.ndarray:
    mask = np.zeros((resolution,) * 3, dtype=bool)
    mask[: max(1, resolution // 3), :, : max(1, resolution // 2)] = True
    return mask


def _params(config: ValidationConfig, case: ValidationCase) -> Params:
    n = config.resolution
    return Params(
        resolution=(n, n, n),
        dx=1.0 / n,
        gravity=(0.0, 0.0, float(config.gravity_z)),
        frame_dt=1.0 / float(config.frame_rate),
        cfl_target=case.cfl_target,
        particles_per_cell=config.particles_per_cell,
        st_enabled=case.st_enabled,
        seed=config.seed,
    )


def _run_case(config: ValidationConfig, case: ValidationCase) -> _CaseOutput:
    params = _params(config, case)
    backend = get_backend(config.backend)
    solver = STFLIPSolver(params, backend)
    solver.add_liquid_mask(_dam_break_mask(config.resolution))
    initial = solver.checkpoint_state()
    initial_particle_hash = _hash_arrays(
        initial["pos"], initial["vel"], initial["dt_resid"])
    initial_shared_hash = _hash_initial_checkpoint(
        initial, solver.m0, include_dt_prev=False)
    initial_checkpoint_hash = _hash_initial_checkpoint(initial, solver.m0)

    outputs: list[_FrameOutput] = []
    step_times: list[float] = []
    render_times: list[float] = []
    dt_max_seen = 0.0
    for frame in range(1, config.frames + 1):
        backend.synchronize()
        started = time.perf_counter()
        stats = solver.step_frame()
        backend.synchronize()
        step_wall_s = time.perf_counter() - started

        started = time.perf_counter()
        positions, velocities = solver.get_render_particles()
        backend.synchronize()
        render_wall_s = time.perf_counter() - started
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        velocities = np.ascontiguousarray(velocities, dtype=np.float32)
        residual = np.ascontiguousarray(
            solver.be.to_numpy(solver.dt_resid), dtype=np.float32)
        residual_abs = np.abs(residual)
        residual_max = (
            float(residual_abs.max()) if residual_abs.size else 0.0)
        residual_rms = (
            float(np.sqrt(np.mean(
                residual.astype(np.float64) ** 2, dtype=np.float64)))
            if residual.size else 0.0
        )
        theta = -residual / max(float(solver._dt_prev), 1e-12)
        if case.st_enabled:
            deposition_weights = kernels.w_temporal(np, theta)
            weight_mode = "one_sided_temporal_kernel"
        else:
            deposition_weights = np.ones_like(theta)
            weight_mode = "instantaneous_unit_weight"
        in_slab = theta[(theta >= -0.5) & (theta <= 0.5)]
        occupied_bins = 0
        if in_slab.size:
            occupied_bins = int(np.count_nonzero(np.histogram(
                in_slab, bins=16, range=(-0.5, 0.5))[0]))
        weight_sum = float(deposition_weights.sum(dtype=np.float64))
        weight_sq_sum = float(np.square(
            deposition_weights, dtype=np.float64).sum(dtype=np.float64))
        effective_fraction = (
            (weight_sum * weight_sum)
            / (len(deposition_weights) * weight_sq_sum)
            if weight_sq_sum > 0.0 and len(deposition_weights) else 0.0
        )
        # ``solver._grids`` was deposited at the beginning of the final global
        # substep. CFL-1 and CFL-16 therefore leave it at different within-frame
        # times. Recompute Eq. 8 read-only from the current 4D particle state so
        # every comparison samples the same output time. This diagnostic P2G is
        # intentionally outside both solver and render timing scopes.
        diagnostic_grids = solver._p2g(solver._dt_prev)
        phase_density = np.ascontiguousarray(
            solver.be.to_numpy(diagnostic_grids["c_phi"]), dtype=np.float32)
        normalized_mass = np.ascontiguousarray(
            solver.be.to_numpy(diagnostic_grids["c_m"]) / solver.m0,
            dtype=np.float32,
        )
        metrics = measure_frame(
            frame=frame,
            simulation_time_s=solver.time,
            params=params,
            stats=stats,
            positions_local=positions,
            velocities=velocities,
            compute_wall_s=step_wall_s,
        )
        dt_max_seen = max(dt_max_seen, *(stats.dt_values or (0.0,)))
        outputs.append(_FrameOutput(
            positions=positions,
            velocities=velocities,
            metrics=metrics,
            output_sha256=_hash_arrays(positions, velocities),
            dt_resid_abs_max_s=residual_max,
            dt_resid_rms_s=residual_rms,
            dt_prev_s=float(solver._dt_prev),
            dt_max_seen_s=float(dt_max_seen),
            normalized_deposited_mass=normalized_mass,
            phase_density=phase_density,
            temporal_quadrature_state={
                "deposition_weight_mode": weight_mode,
                "theta_min": float(theta.min()) if theta.size else None,
                "theta_max": float(theta.max()) if theta.size else None,
                "occupied_slab_bins": occupied_bins,
                "slab_bin_count": 16,
                "temporal_weight_mean": (
                    float(deposition_weights.mean(dtype=np.float64))
                    if deposition_weights.size else 0.0),
                "effective_weighted_sample_fraction": float(
                    effective_fraction),
            },
        ))
        step_times.append(step_wall_s)
        render_times.append(render_wall_s)

    return _CaseOutput(
        case=case,
        params=params,
        initial_particle_state_sha256=initial_particle_hash,
        initial_shared_state_sha256=initial_shared_hash,
        initial_checkpoint_sha256=initial_checkpoint_hash,
        m0=float(solver.m0),
        frames=outputs,
        timing={
            "step_wall_s_by_frame": step_times,
            "render_resynchronization_wall_s_by_frame": render_times,
            "step_wall_s_total": float(sum(step_times)),
            "render_resynchronization_wall_s_total": float(sum(render_times)),
            "observed_wall_s_total": float(sum(step_times) + sum(render_times)),
        },
    )


def _occupancy(positions: np.ndarray, params: Params) -> np.ndarray:
    indices = np.floor(positions / params.dx).astype(np.int64)
    upper = np.asarray(params.resolution, dtype=np.int64) - 1
    indices = np.clip(indices, 0, upper)
    result = np.zeros(params.resolution, dtype=bool)
    if indices.size:
        result[tuple(indices.T)] = True
    return result


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _dimensionless_laplacian(values: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(values, dtype=np.float64), 1, mode="edge")
    centre = padded[1:-1, 1:-1, 1:-1]
    return (
        padded[2:, 1:-1, 1:-1]
        + padded[:-2, 1:-1, 1:-1]
        + padded[1:-1, 2:, 1:-1]
        + padded[1:-1, :-2, 1:-1]
        + padded[1:-1, 1:-1, 2:]
        + padded[1:-1, 1:-1, :-2]
        - 6.0 * centre
    )


def _compare_cases(candidate: _CaseOutput, reference: _CaseOutput) -> dict:
    if len(candidate.frames) != len(reference.frames):
        raise ValueError("validation cases must have equal frame counts")
    comparisons = []
    for candidate_frame, reference_frame in zip(
            candidate.frames, reference.frames):
        candidate_pos = candidate_frame.positions
        reference_pos = reference_frame.positions
        if candidate_pos.shape != reference_pos.shape:
            raise ValueError(
                "matched validation requires stable particle correspondence")
        displacement = np.linalg.norm(
            candidate_pos.astype(np.float64)
            - reference_pos.astype(np.float64),
            axis=1,
        )
        velocity_delta = (
            candidate_frame.velocities.astype(np.float64)
            - reference_frame.velocities.astype(np.float64)
        )
        if displacement.size:
            position_rmse = float(np.sqrt(np.mean(displacement**2)))
            position_p95 = float(np.quantile(displacement, 0.95))
            velocity_rmse = float(np.sqrt(np.mean(velocity_delta**2)))
        else:
            position_rmse = position_p95 = velocity_rmse = 0.0
        candidate_occupancy = _occupancy(candidate_pos, candidate.params)
        reference_occupancy = _occupancy(reference_pos, reference.params)
        union = int(np.count_nonzero(
            candidate_occupancy | reference_occupancy))
        intersection = int(np.count_nonzero(
            candidate_occupancy & reference_occupancy))
        occupancy_iou = 1.0 if union == 0 else intersection / union
        candidate_energy = candidate_frame.metrics[
            "kinetic_energy_particle_estimate"]
        reference_energy = reference_frame.metrics[
            "kinetic_energy_particle_estimate"]
        energy_relative_error = (
            abs(candidate_energy - reference_energy) / abs(reference_energy)
            if reference_energy else None
        )
        normalized_mass_delta = (
            candidate_frame.normalized_deposited_mass.astype(np.float64)
            - reference_frame.normalized_deposited_mass.astype(np.float64)
        )
        phase_delta = (
            candidate_frame.phase_density.astype(np.float64)
            - reference_frame.phase_density.astype(np.float64)
        )
        normalized_mass_rmse = float(np.sqrt(np.mean(
            normalized_mass_delta * normalized_mass_delta,
            dtype=np.float64,
        )))
        phase_density_rmse = float(np.sqrt(np.mean(
            phase_delta * phase_delta,
            dtype=np.float64,
        )))
        phase_laplacian_delta = (
            _dimensionless_laplacian(candidate_frame.phase_density)
            - _dimensionless_laplacian(reference_frame.phase_density)
        )
        phase_laplacian_rmse = float(np.sqrt(np.mean(
            phase_laplacian_delta * phase_laplacian_delta,
            dtype=np.float64,
        )))
        candidate_liquid = candidate_frame.phase_density >= 0.5
        reference_liquid = reference_frame.phase_density >= 0.5
        phase_union = int(np.count_nonzero(
            candidate_liquid | reference_liquid))
        phase_intersection = int(np.count_nonzero(
            candidate_liquid & reference_liquid))
        phase_iou = (
            1.0 if phase_union == 0 else phase_intersection / phase_union)
        comparisons.append({
            "frame": candidate_frame.metrics["frame"],
            "normalized_deposited_mass_rmse": normalized_mass_rmse,
            "phase_density_rmse": phase_density_rmse,
            "phase_laplacian_rmse": phase_laplacian_rmse,
            "phase_threshold_iou": float(phase_iou),
            "position_rmse_solver_units": position_rmse,
            "position_rmse_dx": position_rmse / candidate.params.dx,
            "position_p95_dx": position_p95 / candidate.params.dx,
            "velocity_component_rmse_solver_units_per_s": velocity_rmse,
            "occupancy_iou": float(occupancy_iou),
            "kinetic_energy_relative_error": energy_relative_error,
        })
    field_metrics = {
        "coherence_primary_measure": "phase_density_rmse_mean",
        "normalized_deposited_mass_rmse_mean": float(np.mean([
            value["normalized_deposited_mass_rmse"] for value in comparisons
        ], dtype=np.float64)),
        "normalized_deposited_mass_rmse_final": comparisons[-1][
            "normalized_deposited_mass_rmse"],
        "phase_density_rmse_mean": float(np.mean([
            value["phase_density_rmse"] for value in comparisons
        ], dtype=np.float64)),
        "phase_density_rmse_final": comparisons[-1]["phase_density_rmse"],
        "phase_laplacian_rmse_mean": float(np.mean([
            value["phase_laplacian_rmse"] for value in comparisons
        ], dtype=np.float64)),
        "phase_laplacian_rmse_final": comparisons[-1][
            "phase_laplacian_rmse"],
        "phase_threshold_iou_mean": float(np.mean([
            value["phase_threshold_iou"] for value in comparisons
        ], dtype=np.float64)),
        "phase_threshold_iou_final": comparisons[-1]["phase_threshold_iou"],
    }
    trajectory_diagnostics = {
        "interpretation": (
            "Diagnostic only: ST-FLIP intentionally jitters particle sample "
            "times, so particle correspondence is not a primary quality gate."
        ),
        "position_rmse_dx_mean": float(np.mean([
            value["position_rmse_dx"] for value in comparisons
        ], dtype=np.float64)),
        "position_rmse_dx_final": comparisons[-1]["position_rmse_dx"],
        "occupancy_iou_mean": float(np.mean([
            value["occupancy_iou"] for value in comparisons
        ], dtype=np.float64)),
        "occupancy_iou_final": comparisons[-1]["occupancy_iou"],
    }
    return {
        "candidate": candidate.case.identifier,
        "reference": reference.case.identifier,
        "frames": comparisons,
        "field_metrics": field_metrics,
        "trajectory_diagnostics": trajectory_diagnostics,
    }


def temporal_quadrature_coverage(
    sample_count: int = 256,
    bin_count: int = 16,
) -> dict[str, Any]:
    """Deterministically exercise the Eq. 7–8 temporal quadrature domain.

    Stratified midpoint samples cover the complete normalized time slab.  The
    check uses the kernel's exact unit normalization and a resolution-doubling
    convergence test instead of selecting a dataset-dependent quality
    threshold.  The instantaneous ablation is reported as a single-time-point
    control, not as an attempted approximation to the temporal integral.
    """
    if (isinstance(sample_count, bool) or not isinstance(sample_count, int)
            or sample_count < 2 * bin_count):
        raise ValueError("sample_count must be an integer at least 2*bin_count")
    if (isinstance(bin_count, bool) or not isinstance(bin_count, int)
            or bin_count < 2 or sample_count % bin_count):
        raise ValueError(
            "bin_count must divide sample_count and be at least two")

    def estimate(count: int):
        tau = (np.arange(count, dtype=np.float64) + 0.5) / count - 0.5
        weights = kernels.w_temporal(np, tau)
        return tau, weights, float(weights.mean(dtype=np.float64))

    tau, weights, normalization = estimate(sample_count)
    _, _, coarse_normalization = estimate(sample_count // 2)
    histogram = np.histogram(tau, bins=bin_count, range=(-0.5, 0.5))[0]
    weight_sum = float(weights.sum(dtype=np.float64))
    weight_sq_sum = float(np.square(
        weights, dtype=np.float64).sum(dtype=np.float64))
    normalization_error = abs(normalization - 1.0)
    coarse_error = abs(coarse_normalization - 1.0)
    weighted_mean_tau = float(np.mean(weights * tau, dtype=np.float64))
    weighted_second_moment = float(np.mean(
        weights * tau * tau, dtype=np.float64))
    return {
        "equations": "Eq. 7-8 temporal slab quadrature using Eq. 19 W_T",
        "sample_count": sample_count,
        "bin_count": bin_count,
        "occupied_bins": int(np.count_nonzero(histogram)),
        "normalization_estimate": normalization,
        "normalization_error": normalization_error,
        "coarse_sample_count": sample_count // 2,
        "coarse_normalization_error": coarse_error,
        "resolution_doubling_reduces_error": normalization_error < coarse_error,
        "weighted_mean_tau": weighted_mean_tau,
        "weighted_second_moment_tau": weighted_second_moment,
        "effective_weighted_sample_fraction": float(
            (weight_sum * weight_sum) / (sample_count * weight_sq_sum)),
        "instantaneous_control": {
            "sample_tau": 0.0,
            "occupied_bins": 1,
            "interpretation": (
                "Unit-weight instantaneous P2G samples one time and does not "
                "cover the normalized temporal slab."
            ),
        },
        "passed": (
            bool(np.all(histogram > 0))
            and normalization_error < coarse_error
            and weighted_mean_tau > 0.0
        ),
    }


def detect_high_cfl(
    cases: Mapping[str, Mapping[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """Report whether both matched high-target cases achieved high CFL."""
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("high-CFL threshold must be finite and positive")
    maxima = {}
    for identifier in ("st_cfl_16", "instantaneous_cfl_16"):
        try:
            frames = cases[identifier]["frames"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"missing validation case {identifier!r}") from exc
        values = [
            frame["metrics"].get("particle_cfl_actual_max")
            for frame in frames
        ]
        finite = [float(value) for value in values if value is not None]
        maxima[identifier] = max(finite) if finite else 0.0
    reached_by_case = {
        identifier: value >= threshold for identifier, value in maxima.items()
    }
    return {
        "threshold": threshold,
        "actual_max_by_case": maxima,
        "reached_by_case": reached_by_case,
        "reached": all(reached_by_case.values()),
    }


def _public_case(output: _CaseOutput) -> dict[str, Any]:
    return {
        "identifier": output.case.identifier,
        "label": output.case.label,
        "parameters": asdict(output.params),
        "m0": output.m0,
        "initial_particle_state_sha256": (
            output.initial_particle_state_sha256),
        "initial_shared_state_sha256": output.initial_shared_state_sha256,
        "initial_checkpoint_sha256": output.initial_checkpoint_sha256,
        "frames": [
            {
                "metrics": frame.metrics,
                "output_sha256": frame.output_sha256,
                "deposition_sha256": _hash_arrays(
                    frame.normalized_deposited_mass,
                    frame.phase_density,
                ),
                "temporal_state": {
                    "dt_resid_abs_max_s": frame.dt_resid_abs_max_s,
                    "dt_resid_rms_s": frame.dt_resid_rms_s,
                    "dt_prev_s": frame.dt_prev_s,
                    "dt_max_seen_s": frame.dt_max_seen_s,
                    "quadrature": frame.temporal_quadrature_state,
                },
            }
            for frame in output.frames
        ],
    }


def run_matched_validation(config: ValidationConfig | None = None) -> dict:
    """Run all four real solver cases and return a JSON-native artifact."""
    config = ValidationConfig() if config is None else config
    source_hashes = _source_file_hashes()
    source_revision = _source_revision()
    source_worktree_dirty = _source_worktree_dirty()
    outputs = {case.identifier: _run_case(config, case) for case in MATCHED_CASES}
    if _source_file_hashes() != source_hashes:
        raise RuntimeError(
            "bpy-free solver sources changed while validation was running")
    initial_particle_hashes = {
        output.initial_particle_state_sha256 for output in outputs.values()
    }
    initial_shared_hashes = {
        output.initial_shared_state_sha256 for output in outputs.values()
    }
    if len(initial_shared_hashes) != 1:
        raise RuntimeError(
            "matched validation cases did not share particle/RNG/m0 initial "
            "state")
    checkpoint_hashes_by_case = {
        identifier: output.initial_checkpoint_sha256
        for identifier, output in outputs.items()
    }
    checkpoints_match_within_cfl = (
        checkpoint_hashes_by_case["st_cfl_1"]
        == checkpoint_hashes_by_case["instantaneous_cfl_1"]
        and checkpoint_hashes_by_case["st_cfl_16"]
        == checkpoint_hashes_by_case["instantaneous_cfl_16"]
    )
    if not checkpoints_match_within_cfl:
        raise RuntimeError(
            "temporal ablation cases did not share their complete initial "
            "checkpoint at matched CFL")

    st_degradation = _compare_cases(outputs["st_cfl_16"], outputs["st_cfl_1"])
    instantaneous_degradation = _compare_cases(
        outputs["instantaneous_cfl_16"],
        outputs["instantaneous_cfl_1"],
    )
    high_pair = _compare_cases(
        outputs["st_cfl_16"], outputs["instantaneous_cfl_16"])
    common_ablation_reference = _compare_cases(
        outputs["st_cfl_16"], outputs["instantaneous_cfl_1"])
    public_cases = {
        identifier: _public_case(output)
        for identifier, output in outputs.items()
    }
    high_cfl = detect_high_cfl(public_cases, config.high_cfl_threshold)

    st_fields = st_degradation["field_metrics"]
    instantaneous_fields = instantaneous_degradation["field_metrics"]
    primary_name = "phase_density_rmse_mean"
    st_primary_error = st_fields[primary_name]
    instantaneous_primary_error = instantaneous_fields[primary_name]
    primary_improved = st_primary_error < instantaneous_primary_error
    st_interface_iou = st_fields["phase_threshold_iou_mean"]
    instantaneous_interface_iou = instantaneous_fields[
        "phase_threshold_iou_mean"]
    interface_iou_improved = st_interface_iou > instantaneous_interface_iou
    internal_coherence_improved = (
        primary_improved and interface_iou_improved)
    quadrature = temporal_quadrature_coverage()
    residual_checks = {}
    for identifier, output in outputs.items():
        maxima = [frame.dt_resid_abs_max_s for frame in output.frames]
        bounds = [
            0.5 * frame.dt_max_seen_s + 1e-7 for frame in output.frames
        ]
        if output.case.st_enabled:
            passed = all(value <= bound for value, bound in zip(maxima, bounds))
        else:
            passed = all(value <= 1e-9 for value in maxima)
        residual_checks[identifier] = {
            "passed": passed,
            "dt_resid_abs_max_s": max(maxima, default=0.0),
        }

    timing = {
        identifier: output.timing for identifier, output in outputs.items()
    }
    artifact = {
        "schema": VALIDATION_SCHEMA,
        "version": VALIDATION_VERSION,
        "metric_schema": {
            "schema": METRICS_SCHEMA,
            "version": SCHEMA_VERSION,
        },
        "purpose": (
            "Matched ST-FLIP versus instantaneous-P2G validation on one fixed "
            "backend; instantaneous P2G is an ablation, not standard FLIP/GFM."
        ),
        "environment": {
            "addon_version": __version__,
            "source_revision": source_revision,
            "source_worktree_dirty": source_worktree_dirty,
            "source_files_sha256": source_hashes,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "backend": config.backend,
        },
        "scenario": {
            **asdict(config),
            "geometry": "unit-cube dam break; x<floor(N/3), z<floor(N/2)",
            "deposition_evaluation": (
                "Read-only Eq. 8 P2G recomputed from each solver's current "
                "4D particle state at the shared frame-output time; excluded "
                "from timing."
            ),
            "hash_interpretation": (
                "SHA-256 values are bit-level execution fingerprints for "
                "audit/debugging, not a cross-hardware or cross-process "
                "physical-determinism guarantee."
            ),
            "case_order": [case.identifier for case in MATCHED_CASES],
        },
        "cases": public_cases,
        "mechanism": {
            "eq7_8_temporal_quadrature_coverage": quadrature,
        },
        "quality": {
            "interpretation": (
                "This is an internal phase-field coherence surrogate, not a "
                "paper reproduction. It compares each method's high-CFL Eq. "
                "13 pressure/interface field with its own matched CFL-1 "
                "trajectory. Raw Eq. 8 mass and particle correspondence are "
                "reported as diagnostics, not selected as acceptance gates."
            ),
            "st_high_vs_st_low": st_degradation,
            "instantaneous_high_vs_instantaneous_low": (
                instantaneous_degradation),
            "st_high_vs_instantaneous_high": high_pair,
            "st_high_vs_instantaneous_low_diagnostic": (
                common_ablation_reference),
            "primary": {
                "measure": primary_name,
                "basis": (
                    "Bounded Eq. 13 phase-density RMSE; this field drives the "
                    "variable-coefficient pressure/interface treatment, but "
                    "is not the paper's MCF render-surface metric"),
                "st_error": st_primary_error,
                "instantaneous_error": instantaneous_primary_error,
                "st_over_instantaneous_error_ratio": _safe_ratio(
                    st_primary_error, instantaneous_primary_error),
                "st_has_lower_error": primary_improved,
            },
            "secondary_evidence": {
                "phase_laplacian_rmse_mean": {
                    "st_error": st_fields["phase_laplacian_rmse_mean"],
                    "instantaneous_error": instantaneous_fields[
                        "phase_laplacian_rmse_mean"],
                    "st_has_lower_error": (
                        st_fields["phase_laplacian_rmse_mean"]
                        < instantaneous_fields["phase_laplacian_rmse_mean"]),
                },
                "phase_threshold_iou_mean": {
                    "required_for_coherence_gate": True,
                    "st_value": st_interface_iou,
                    "instantaneous_value": instantaneous_interface_iou,
                    "st_has_higher_value": interface_iou_improved,
                },
                "normalized_deposited_mass_rmse_mean": {
                    "acceptance_gate": False,
                    "st_error": st_fields[
                        "normalized_deposited_mass_rmse_mean"],
                    "instantaneous_error": instantaneous_fields[
                        "normalized_deposited_mass_rmse_mean"],
                    "st_has_lower_error": (
                        st_fields["normalized_deposited_mass_rmse_mean"]
                        < instantaneous_fields[
                            "normalized_deposited_mass_rmse_mean"]),
                    "interpretation": (
                        "Unfiltered Eq. 8 mass retains the intended temporal "
                        "Monte Carlo variance and allowed adaptive-gamma "
                        "normalization shift; it is reported prominently but "
                        "is not the paper's T=7 SDF/surface-normal metric."
                    ),
                },
            },
        },
        "timing": timing,
        "acceptance": {
            "same_initial_particle_state": len(initial_particle_hashes) == 1,
            "same_initial_shared_state": len(initial_shared_hashes) == 1,
            "complete_checkpoints_match_within_cfl": (
                checkpoints_match_within_cfl),
            "initial_particle_state_sha256": next(iter(
                initial_particle_hashes)),
            "initial_shared_state_sha256": next(iter(
                initial_shared_hashes)),
            "initial_checkpoint_sha256_by_case": checkpoint_hashes_by_case,
            "high_cfl": high_cfl,
            "temporal_residual_checks": residual_checks,
            "eq7_8_temporal_quadrature_passed": quadrature["passed"],
            "core_checks_passed": all(
                value["passed"] for value in residual_checks.values()
            ) and quadrature["passed"],
            "internal_coherence_improved": internal_coherence_improved,
            "validation_ready": (
                high_cfl["reached"]
                and all(value["passed"] for value in residual_checks.values())
                and quadrature["passed"]
                and internal_coherence_improved
            ),
        },
    }
    # Reject accidental NumPy scalars, NaNs, or infinities before returning an
    # object advertised as a portable JSON artifact.
    return json.loads(json.dumps(artifact, allow_nan=False))


def run_multi_seed_validation(
    config: ValidationConfig,
    seeds=(0, 1, 2),
) -> dict:
    """Run matched validation across seeds and aggregate coherence evidence.

    An internal regression claim should not depend on one favorable seed.
    The aggregate gate requires every run to reach high observed CFL and pass
    mechanism/residual checks. Mean Eq. 13 phase RMSE and threshold-interface
    IoU must both favor ST-FLIP, with a strict majority of seeds agreeing on
    each. Raw deposited mass remains a non-gating variance diagnostic.
    """
    try:
        normalized_seeds = tuple(int(seed) for seed in seeds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("validation seeds must be integers") from exc
    if (not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds)
            or any(seed < 0 for seed in normalized_seeds)):
        raise ValueError(
            "validation seeds must be distinct non-negative integers")

    source_hashes = _source_file_hashes()
    runs = [
        run_matched_validation(replace(config, seed=seed))
        for seed in normalized_seeds
    ]
    if _source_file_hashes() != source_hashes or any(
        run["environment"]["source_files_sha256"] != source_hashes
        for run in runs
    ):
        raise RuntimeError(
            "bpy-free solver sources changed during multi-seed validation")
    st_errors = [run["quality"]["primary"]["st_error"] for run in runs]
    instantaneous_errors = [
        run["quality"]["primary"]["instantaneous_error"] for run in runs
    ]
    st_mean = float(np.mean(st_errors, dtype=np.float64))
    instantaneous_mean = float(np.mean(
        instantaneous_errors, dtype=np.float64))
    wins = sum(
        run["quality"]["primary"]["st_has_lower_error"] for run in runs
    )
    iou_records = [
        run["quality"]["secondary_evidence"]["phase_threshold_iou_mean"]
        for run in runs
    ]
    st_iou_mean = float(np.mean(
        [record["st_value"] for record in iou_records], dtype=np.float64))
    instantaneous_iou_mean = float(np.mean(
        [record["instantaneous_value"] for record in iou_records],
        dtype=np.float64,
    ))
    iou_wins = sum(record["st_has_higher_value"] for record in iou_records)
    raw_mass_records = [
        run["quality"]["secondary_evidence"][
            "normalized_deposited_mass_rmse_mean"]
        for run in runs
    ]
    st_raw_mass_mean = float(np.mean(
        [record["st_error"] for record in raw_mass_records],
        dtype=np.float64,
    ))
    instantaneous_raw_mass_mean = float(np.mean(
        [record["instantaneous_error"] for record in raw_mass_records],
        dtype=np.float64,
    ))
    raw_mass_wins = sum(
        record["st_has_lower_error"] for record in raw_mass_records)
    majority = len(runs) // 2 + 1
    high_cfl_all = all(
        run["acceptance"]["high_cfl"]["reached"] for run in runs)
    core_checks_all = all(
        run["acceptance"]["core_checks_passed"] for run in runs)
    mean_improved = st_mean < instantaneous_mean
    mean_iou_improved = st_iou_mean > instantaneous_iou_mean
    internal_coherence_improved = (
        mean_improved
        and wins >= majority
        and mean_iou_improved
        and iou_wins >= majority
    )
    artifact = {
        "schema": MULTI_SEED_SCHEMA,
        "version": MULTI_SEED_VERSION,
        "purpose": (
            "Multi-seed robustness wrapper for an internal ST-FLIP phase-"
            "field coherence surrogate; the instantaneous-P2G control is not "
            "FLIP/GFM and this is not paper-publication evidence."
        ),
        "seeds": list(normalized_seeds),
        "base_scenario": {
            **asdict(config),
            "seed": None,
        },
        "runs": runs,
        "aggregate": {
            "primary_measure": runs[0]["quality"]["primary"]["measure"],
            "st_error_mean": st_mean,
            "instantaneous_error_mean": instantaneous_mean,
            "st_over_instantaneous_error_ratio": _safe_ratio(
                st_mean, instantaneous_mean),
            "st_wins": int(wins),
            "seed_count": len(runs),
            "strict_majority_required": majority,
            "strict_majority_st_wins": wins >= majority,
            "mean_st_error_is_lower": mean_improved,
            "phase_threshold_iou_st_mean": st_iou_mean,
            "phase_threshold_iou_instantaneous_mean": (
                instantaneous_iou_mean),
            "phase_threshold_iou_st_wins": int(iou_wins),
            "strict_majority_phase_threshold_iou_st_wins": (
                iou_wins >= majority),
            "mean_phase_threshold_iou_is_higher": mean_iou_improved,
            "raw_mass_diagnostic": {
                "acceptance_gate": False,
                "measure": "normalized_deposited_mass_rmse_mean",
                "st_error_mean": st_raw_mass_mean,
                "instantaneous_error_mean": instantaneous_raw_mass_mean,
                "st_over_instantaneous_error_ratio": _safe_ratio(
                    st_raw_mass_mean, instantaneous_raw_mass_mean),
                "st_wins": int(raw_mass_wins),
                "interpretation": (
                    "Raw Eq. 8 mass includes intended Monte Carlo variance; "
                    "this internal run reports it without treating it as the "
                    "paper's T=7 SDF/surface metric."
                ),
            },
            "high_cfl_reached_all": high_cfl_all,
            "core_checks_passed_all": core_checks_all,
            "internal_coherence_improved": internal_coherence_improved,
            "validation_ready": (
                high_cfl_all
                and core_checks_all
                and internal_coherence_improved
            ),
        },
    }
    return json.loads(json.dumps(artifact, allow_nan=False))


def write_validation_artifact(path: str | os.PathLike, artifact: dict) -> str:
    """Atomically write a strict validation JSON artifact."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        artifact,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=".stflip-validation-",
        suffix=".json",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return str(destination)


__all__ = [
    "MATCHED_CASES",
    "MULTI_SEED_SCHEMA",
    "MULTI_SEED_VERSION",
    "VALIDATION_SCHEMA",
    "VALIDATION_VERSION",
    "ValidationCase",
    "ValidationConfig",
    "detect_high_cfl",
    "run_matched_validation",
    "run_multi_seed_validation",
    "temporal_quadrature_coverage",
    "write_validation_artifact",
]
