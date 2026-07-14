"""Tests for the ENER-M0 angular-momentum metrics and rotating tank."""

from types import SimpleNamespace

import numpy as np
import pytest

from stflip.metrics import (
    FRAME_FIELD_ORDER,
    SCHEMA_VERSION,
    measure_frame,
    validate_frame_record,
)
from stflip.validation import (
    ROTATING_TANK_SCHEMA,
    RotatingTankConfig,
    run_rotating_tank_validation,
)


def _params(**overrides):
    values = {
        "dx": 0.5,
        "rho": 4.0,
        "particles_per_cell": 2,
        "pcg_tol": 0.1,
        "cfl_target": 8.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _measure(positions, velocities, **kwargs):
    return measure_frame(
        frame=1,
        simulation_time_s=0.0,
        params=kwargs.pop("params", _params()),
        stats=None,
        positions_local=positions,
        velocities=velocities,
        **kwargs,
    )


class TestAngularMomentum:
    def test_rigid_rotation_matches_analytic_value(self):
        # Four unit-offset particles spinning about their own centre carry
        # L_z = sum m * r_perp^2 * omega exactly.
        omega = 2.0
        centre = np.array([3.0, 4.0, 5.0])
        offsets = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        positions = centre + offsets
        velocities = omega * np.stack(
            [-offsets[:, 1], offsets[:, 0], np.zeros(4)], axis=1)
        params = _params()
        record = _measure(positions, velocities, params=params)
        particle_mass = params.rho * params.dx**3 / params.particles_per_cell
        expected = particle_mass * 4 * omega  # r_perp = 1 for every particle
        assert record["angular_momentum_z_estimate"] == pytest.approx(expected)
        assert record["angular_momentum_x_estimate"] == pytest.approx(0.0)
        assert record["angular_momentum_y_estimate"] == pytest.approx(0.0)

    def test_taken_about_centre_so_drift_is_not_spin(self):
        # A uniformly translating pair has zero angular momentum about its
        # own centre even though it spins about the origin.
        positions = np.array([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
        velocities = np.array([[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        record = _measure(positions, velocities)
        assert record["angular_momentum_z_estimate"] == pytest.approx(0.0)

    def test_empty_snapshot_reports_zero(self):
        record = _measure(np.zeros((0, 3)), np.zeros((0, 3)))
        assert record["angular_momentum_x_estimate"] == 0.0
        assert record["angular_momentum_y_estimate"] == 0.0
        assert record["angular_momentum_z_estimate"] == 0.0
        assert record["total_particle_mass_estimate"] == 0.0

    def test_schema_v3_fields_present_and_validating(self):
        record = _measure(np.zeros((1, 3)), np.zeros((1, 3)))
        assert record["schema_version"] == SCHEMA_VERSION == 3
        assert tuple(record) == FRAME_FIELD_ORDER
        validate_frame_record(record)

    def test_schema_v2_record_is_rejected(self):
        # cache.read_metrics silently drops rows failing validation, so a
        # pre-bump row must fail here -- the documented, accepted loss.
        record = _measure(np.zeros((1, 3)), np.zeros((1, 3)))
        stale = dict(record)
        stale["schema_version"] = 2
        with pytest.raises(ValueError):
            validate_frame_record(stale)


class TestPhaseWeightedMasses:
    def test_gas_particles_carry_gas_density(self):
        params = _params(rho=1000.0, rho_gas=1.25, two_phase=True)
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        velocities = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        phases = np.array([1.0, 0.0])
        record = _measure(
            positions, velocities, params=params, phases=phases)
        volume = params.dx**3 / params.particles_per_cell
        expected_mass = volume * (1000.0 + 1.25)
        assert record["total_particle_mass_estimate"] == pytest.approx(
            expected_mass)
        assert record["momentum_z_estimate"] == pytest.approx(expected_mass)
        assert record["kinetic_energy_particle_estimate"] == pytest.approx(
            0.5 * expected_mass)
        # The mass-weighted centre sits essentially on the liquid particle.
        assert record["center_of_mass_local_x_solver_units"] < 0.01

    def test_uniform_when_phases_omitted(self):
        params = _params(rho=1000.0, rho_gas=1.25, two_phase=True)
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        velocities = np.zeros((2, 3))
        record = _measure(positions, velocities, params=params)
        volume = params.dx**3 / params.particles_per_cell
        assert record["total_particle_mass_estimate"] == pytest.approx(
            2 * 1000.0 * volume)

    def test_phase_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            _measure(
                np.zeros((2, 3)), np.zeros((2, 3)),
                phases=np.zeros((3,)))

    def test_nonfinite_phases_rejected(self):
        with pytest.raises(ValueError):
            _measure(
                np.zeros((1, 3)), np.zeros((1, 3)),
                phases=np.array([np.nan]))


class TestRotatingTankConfig:
    def test_rejects_bad_values(self):
        with pytest.raises(ValueError):
            RotatingTankConfig(cfl_targets=())
        with pytest.raises(ValueError):
            RotatingTankConfig(cfl_targets=(0.0,))
        with pytest.raises(ValueError):
            RotatingTankConfig(tank_radius=0.6)
        with pytest.raises(ValueError):
            RotatingTankConfig(angular_speed=-1.0)
        with pytest.raises(ValueError):
            RotatingTankConfig(resolution=4)


@pytest.fixture(scope="module")
def report():
    config = RotatingTankConfig(
        resolution=12,
        frames=2,
        particles_per_cell=2,
        cfl_targets=(1.0, 16.0),
    )
    return run_rotating_tank_validation(config)


class TestRotatingTank:
    def test_report_structure(self, report):
        assert report["schema"] == ROTATING_TANK_SCHEMA
        assert set(report["cases"]) == {"cfl_1", "cfl_16"}
        case = report["cases"]["cfl_16"]
        # Baseline frame 0 plus two stepped frames.
        assert len(case["frames"]) == 3
        assert len(case["angular_momentum_z_retention"]) == 3
        assert case["angular_momentum_z_retention"][0] == pytest.approx(1.0)
        assert report["summary"]["cfl_ceiling_estimate"] > 0.0
        assert "vs_floor" in str(sorted(report["summary"]["cfl_16"]))

    def test_seeded_rotation_carries_positive_spin(self, report):
        for case in report["cases"].values():
            baseline = case["frames"][0]
            assert baseline["angular_momentum_z_estimate"] > 0.0
            assert baseline["kinetic_energy_particle_estimate"] > 0.0

    def test_short_run_retains_most_angular_momentum(self, report):
        # Two frames at 12^3 must not shed the bulk of L_z; this is a
        # sanity floor, not a conservation gate (see the summary note).
        for case in report["cases"].values():
            final = case["angular_momentum_z_retention"][-1]
            assert final is not None
            assert 0.3 < final <= 1.05
