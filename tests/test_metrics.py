from types import SimpleNamespace

import numpy as np
import pytest

from stflip.metrics import (
    FRAME_FIELD_ORDER,
    SCHEMA_VERSION,
    estimate_mac_grid_metrics,
    measure_frame,
    validate_frame_record,
)


def _params():
    return SimpleNamespace(
        dx=0.5,
        rho=4.0,
        particles_per_cell=2,
        pcg_tol=0.1,
        cfl_target=8.0,
    )


def _stats():
    return SimpleNamespace(
        steps=2,
        inactive_time_s=0.05,
        dt_values=[0.1, 0.2],
        particle_cfl_estimated_values=[1.0, 2.0],
        particle_cfl_actual_values=[1.5, 2.5],
        pcg_iters=[3, 4],
        pcg_rel_residuals=[0.05, 0.2],
        particles_removed=5,
        volume_outflow_removed=3,
        pressure_outflow_removed=2,
    )


def test_particle_metrics_and_solver_diagnostics_have_expected_formulas():
    positions = np.asarray(((0.0, 0.0, 0.0), (2.0, 4.0, 6.0)))
    velocities = np.asarray(((3.0, 4.0, 0.0), (0.0, 0.0, 0.0)))

    record = measure_frame(
        frame=7,
        simulation_time_s=0.3,
        params=_params(),
        stats=_stats(),
        positions_local=positions,
        velocities=velocities,
        compute_wall_s=0.5,
    )

    assert tuple(record) == FRAME_FIELD_ORDER
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["target_cfl"] == 8.0
    assert record["particle_count"] == 2
    assert record["particles_removed"] == 5
    assert record["volume_outflow_removed"] == 3
    assert record["pressure_outflow_removed"] == 2
    assert record["solver_steps"] == 2
    assert record["inactive_time_s"] == pytest.approx(0.05)
    assert record["dt_min_s"] == pytest.approx(0.1)
    assert record["dt_mean_s"] == pytest.approx(0.15)
    assert record["dt_max_s"] == pytest.approx(0.2)
    assert record["particle_cfl_estimated_mean"] == pytest.approx(1.5)
    assert record["particle_cfl_estimated_max"] == pytest.approx(2.0)
    assert record["particle_cfl_actual_mean"] == pytest.approx(2.0)
    assert record["particle_cfl_actual_max"] == pytest.approx(2.5)
    assert record["pcg_solve_count"] == 2
    assert record["pcg_iterations_total"] == 7
    assert record["pcg_iterations_max"] == 4
    assert record["pcg_relative_residual_max"] == pytest.approx(0.2)
    assert record["pcg_converged_all"] is False
    assert record["speed_max_solver_units_per_s"] == pytest.approx(5.0)
    assert record["speed_rms_solver_units_per_s"] == pytest.approx(np.sqrt(12.5))
    assert record["particle_volume_estimate_solver_units3"] == pytest.approx(0.125)
    assert record["total_particle_mass_estimate"] == pytest.approx(0.5)
    assert record["kinetic_energy_particle_estimate"] == pytest.approx(3.125)
    assert record["momentum_x_estimate"] == pytest.approx(0.75)
    assert record["momentum_y_estimate"] == pytest.approx(1.0)
    assert record["momentum_z_estimate"] == pytest.approx(0.0)
    assert record["center_of_mass_local_x_solver_units"] == pytest.approx(1.0)
    assert record["center_of_mass_local_y_solver_units"] == pytest.approx(2.0)
    assert record["center_of_mass_local_z_solver_units"] == pytest.approx(3.0)
    assert record["compute_wall_s"] == pytest.approx(0.5)
    assert record["phase_threshold_volume_estimate_solver_units3"] is None
    assert record["mac_grid_enstrophy_estimate"] is None


def test_empty_initial_frame_uses_zero_estimates_and_null_diagnostics():
    empty = np.empty((0, 3), dtype=np.float32)

    record = measure_frame(
        frame=1,
        simulation_time_s=0.0,
        params=_params(),
        stats=None,
        positions_local=empty,
        velocities=empty,
    )

    assert record["particle_count"] == 0
    assert record["speed_max_solver_units_per_s"] == 0.0
    assert record["total_particle_mass_estimate"] == 0.0
    assert record["kinetic_energy_particle_estimate"] == 0.0
    assert record["center_of_mass_local_x_solver_units"] is None
    assert record["dt_mean_s"] is None
    assert record["inactive_time_s"] == 0.0
    assert record["pcg_converged_all"] is None


def test_mac_grid_estimates_solid_body_rotation_over_phase_mask():
    # u=-y, v=x, w=0 gives curl_z=2 everywhere and |curl|^2=4.
    nx = ny = nz = 2
    u = np.empty((nx + 1, ny, nz), dtype=np.float64)
    for j in range(ny):
        u[:, j, :] = -(j + 0.5)
    v = np.empty((nx, ny + 1, nz), dtype=np.float64)
    for i in range(nx):
        v[i, :, :] = i + 0.5
    w = np.zeros((nx, ny, nz + 1), dtype=np.float64)
    phase = np.zeros((nx, ny, nz), dtype=np.float64)
    phase[0] = 1.0  # Four liquid cells at dx=1.

    estimates = estimate_mac_grid_metrics(
        {"u": u, "v": v, "w": w, "c_phi": phase},
        1.0,
    )

    assert estimates["phase_field_threshold"] == 0.5
    assert estimates["phase_threshold_volume_estimate_solver_units3"] == 4.0
    assert estimates["phase_threshold_volume_fraction_estimate"] == 0.5
    assert estimates["mac_grid_enstrophy_estimate"] == pytest.approx(8.0)


def test_measure_frame_can_include_mac_grid_estimates():
    phase = np.ones((2, 2, 2), dtype=np.float32)
    grids = {
        "u": np.zeros((3, 2, 2), dtype=np.float32),
        "v": np.zeros((2, 3, 2), dtype=np.float32),
        "w": np.zeros((2, 2, 3), dtype=np.float32),
        "c_phi": phase,
    }
    empty = np.empty((0, 3), dtype=np.float32)

    record = measure_frame(
        frame=1,
        simulation_time_s=0.0,
        params=_params(),
        stats=None,
        positions_local=empty,
        velocities=empty,
        mac_grids=grids,
        phase_threshold=0.75,
    )

    assert record["phase_field_threshold"] == 0.75
    assert record["phase_threshold_volume_estimate_solver_units3"] == 1.0
    assert record["phase_threshold_volume_fraction_estimate"] == 1.0
    assert record["mac_grid_enstrophy_estimate"] == 0.0


def test_invalid_particle_or_grid_inputs_are_rejected():
    with pytest.raises(ValueError, match="same .* shape"):
        measure_frame(
            frame=1,
            simulation_time_s=0.0,
            params=_params(),
            stats=None,
            positions_local=np.zeros((2, 3)),
            velocities=np.zeros((3, 3)),
        )
    with pytest.raises(ValueError, match="finite"):
        measure_frame(
            frame=1,
            simulation_time_s=0.0,
            params=_params(),
            stats=None,
            positions_local=np.asarray(((np.nan, 0.0, 0.0),)),
            velocities=np.zeros((1, 3)),
        )
    with pytest.raises(ValueError, match="expected"):
        estimate_mac_grid_metrics(
            {
                "u": np.zeros((2, 2, 2)),
                "v": np.zeros((2, 3, 2)),
                "w": np.zeros((2, 2, 3)),
                "c_phi": np.zeros((2, 2, 2)),
            },
            1.0,
        )


def test_strict_record_validation_rejects_extra_and_nonfinite_values():
    empty = np.empty((0, 3), dtype=np.float32)
    record = measure_frame(
        frame=1,
        simulation_time_s=0.0,
        params=_params(),
        stats=None,
        positions_local=empty,
        velocities=empty,
    )

    with pytest.raises(ValueError, match="extra"):
        validate_frame_record({**record, "nested": {}})
    record["compute_wall_s"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_frame_record(record)


def test_outflow_metric_counts_must_be_nonnegative_and_consistent():
    empty = np.empty((0, 3), dtype=np.float32)
    stats = _stats()
    stats.particles_removed = 4
    with pytest.raises(ValueError, match="sum"):
        measure_frame(
            frame=1,
            simulation_time_s=0.0,
            params=_params(),
            stats=stats,
            positions_local=empty,
            velocities=empty,
        )
    stats.particles_removed = -1
    stats.volume_outflow_removed = -3
    with pytest.raises(ValueError, match="negative"):
        measure_frame(
            frame=1,
            simulation_time_s=0.0,
            params=_params(),
            stats=stats,
            positions_local=empty,
            velocities=empty,
        )


def test_inactive_time_metric_must_be_finite_and_nonnegative():
    empty = np.empty((0, 3), dtype=np.float32)
    stats = _stats()
    stats.inactive_time_s = -0.1
    with pytest.raises(ValueError, match="inactive_time_s"):
        measure_frame(
            frame=1,
            simulation_time_s=0.0,
            params=_params(),
            stats=stats,
            positions_local=empty,
            velocities=empty,
        )
    stats.inactive_time_s = np.inf
    with pytest.raises(ValueError, match="inactive_time_s"):
        measure_frame(
            frame=1,
            simulation_time_s=0.0,
            params=_params(),
            stats=stats,
            positions_local=empty,
            velocities=empty,
        )
