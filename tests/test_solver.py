import numpy as np
import pytest

from stflip import cache
from stflip import (
    FrameStats,
    Params,
    SolidBodyRotation,
    STFLIPSolver,
    UniformVelocity,
    cuda_diagnostics,
)
from stflip.apertures import weighted_divergence
from stflip.metrics import estimate_mac_grid_metrics, measure_frame


def _dam_break(
    n=24,
    cfl=8.0,
    st=True,
    seed=0,
    ppc=8,
    velocity=(0.0, 0.0, 0.0),
):
    p = Params(
        resolution=(n, n, n), dx=1.0 / n, gravity=(0.0, 0.0, -9.81),
        frame_dt=1.0 / 24.0, cfl_target=cfl, particles_per_cell=ppc,
        st_enabled=st, seed=seed,
    )
    s = STFLIPSolver(p, "cpu")
    mask = np.zeros((n, n, n), dtype=bool)
    mask[: n // 3, :, : n // 2] = True
    s.add_liquid_mask(mask, velocity)
    return s


def test_m0_uniform_reference_is_exactly_particles_per_cell():
    # Unit-integral spatial/temporal kernels make the uniform expectation
    # analytically equal to particles-per-cell, without calibration noise.
    assert _dam_break(ppc=2).m0 == 2.0
    assert _dam_break(ppc=8).m0 == 8.0


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("resolution", (4, 0, 4)),
        ("resolution", (4, 4.5, 4)),
        ("dx", 0.0),
        ("dx", np.nan),
        ("gravity", (0.0, np.inf, 0.0)),
        ("rho", -1.0),
        ("frame_dt", 0.0),
        ("cfl_target", np.inf),
        ("particles_per_cell", 1.5),
        ("flip_blend", 1.01),
        ("jitter_strength", -0.01),
        ("eta_phi", 0.0),
        ("eps_m", np.nan),
        ("eps_rho_rel", 0.0),
        ("pcg_tol", 0.0),
        ("pcg_max_iter", 0),
        ("cfl_local", np.inf),
        ("seed", 1.5),
        ("seed", -1),
    ),
)
def test_params_reject_invalid_values(name, value):
    with pytest.raises((TypeError, ValueError)):
        Params(**{name: value})


def test_solid_sdf_inputs_are_shape_and_finite_validated_atomically():
    p = Params(resolution=(3, 3, 3), dx=0.25)
    solver = STFLIPSolver(p, "cpu")
    original = solver.be.to_numpy(solver.sdf).copy()

    with pytest.raises(ValueError, match="sdf_cells must have shape"):
        solver.set_solid_sdf(np.ones((2, 3, 3), dtype=np.float32))
    bad_cells = np.ones(p.resolution, dtype=np.float32)
    bad_cells[1, 1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solver.set_solid_sdf(bad_cells)
    cells = np.ones(p.resolution, dtype=np.float32)
    nodes = np.ones((4, 4, 4), dtype=np.float32)
    nodes[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        solver.set_solid_sdf(cells, nodes)

    np.testing.assert_array_equal(solver.be.to_numpy(solver.sdf), original)
    assert solver._solid_node_sdf is None


def test_phase_field_uses_eq13_transition_scale():
    """Eq. 13 is C(m / (eta_phi * m0)), C(x) = min(sqrt(x), 1)."""
    p = Params(
        resolution=(4, 4, 4), dx=0.25, gravity=(0.0, 0.0, 0.0),
        particles_per_cell=1, st_enabled=False, eta_phi=0.25, seed=11,
    )
    s = STFLIPSolver(p, "cpu")
    mask = np.zeros(p.resolution, dtype=bool)
    mask[1, 1, 1] = True
    s.add_liquid_mask(mask)
    # Keep every nonzero sample in C's unsaturated branch so the transition
    # scale (rather than only the clamp) is exercised.
    s.m0 = 100.0

    grids = s._p2g(p.frame_dt)

    for grid_name in ("u", "v", "w", "c"):
        mass = s.be.to_numpy(grids[f"{grid_name}_m"])
        phase = s.be.to_numpy(grids[f"{grid_name}_phi"])
        expected = np.minimum(np.sqrt(mass / (p.eta_phi * s.m0)), 1.0)
        assert np.any((expected > 0.0) & (expected < 1.0))
        np.testing.assert_allclose(phase, expected, rtol=1e-6, atol=1e-7)


def test_legacy_and_uniform_velocity_seed_identical_particle_state():
    p = Params(
        resolution=(4, 4, 4),
        dx=0.25,
        gravity=(0.0, 0.0, 0.0),
        particles_per_cell=8,
        seed=91,
    )
    mask = np.zeros(p.resolution, dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    legacy = STFLIPSolver(p, "cpu")
    explicit = STFLIPSolver(p, "cpu")

    legacy.add_liquid_mask(mask, (1.25, -0.5, 0.125))
    explicit.add_liquid_mask(mask, UniformVelocity((1.25, -0.5, 0.125)))

    np.testing.assert_array_equal(legacy.pos, explicit.pos)
    np.testing.assert_array_equal(legacy.vel, explicit.vel)
    np.testing.assert_array_equal(legacy.dt_resid, explicit.dt_resid)


def test_solid_body_rotation_samples_actual_jittered_particle_positions():
    p = Params(
        resolution=(4, 4, 4),
        dx=0.25,
        gravity=(0.0, 0.0, 0.0),
        particles_per_cell=8,
        seed=27,
    )
    solver = STFLIPSolver(p, "cpu")
    mask = np.zeros(p.resolution, dtype=bool)
    mask[1, 1, 1] = True
    field = SolidBodyRotation(
        center=(0.5, 0.5, 0.5),
        angular_velocity=(0.0, 0.0, 2.0),
        linear_velocity=(0.25, -0.125, 0.5),
    )

    assert solver.add_liquid_mask(mask, field) == 8

    positions = solver.be.to_numpy(solver.pos)
    velocities = solver.be.to_numpy(solver.vel)
    np.testing.assert_array_equal(velocities, field.sample(positions))
    # Sampling cell centres would give one shared value; actual particle
    # sampling must preserve the velocity variation from spatial jitter.
    assert np.unique(velocities[:, :2], axis=0).shape[0] > 1


def test_inflow_rotation_is_resampled_at_each_new_particle_position():
    p = Params(
        resolution=(3, 3, 3),
        dx=0.25,
        gravity=(0.0, 0.0, 0.0),
        particles_per_cell=1,
        seed=42,
    )
    solver = STFLIPSolver(p, "cpu")
    mask = np.zeros(p.resolution, dtype=bool)
    mask[1, 1, 1] = True
    field = SolidBodyRotation(
        center=(0.25, 0.25, 0.25),
        angular_velocity=(0.0, 0.0, 1.0),
        linear_velocity=(0.1, 0.2, 0.3),
    )
    solver.add_inflow(mask, field)

    solver._seed_inflows()
    first_positions = solver.be.to_numpy(solver.pos).copy()
    np.testing.assert_array_equal(
        solver.be.to_numpy(solver.vel), field.sample(first_positions)
    )

    xp = solver.be.xp
    solver.pos = xp.zeros((0, 3), dtype=xp.float32)
    solver.vel = xp.zeros((0, 3), dtype=xp.float32)
    solver.dt_resid = xp.zeros((0,), dtype=xp.float32)
    solver._seed_inflows()
    second_positions = solver.be.to_numpy(solver.pos)

    assert not np.array_equal(second_positions, first_positions)
    np.testing.assert_array_equal(
        solver.be.to_numpy(solver.vel), field.sample(second_positions)
    )


def test_velocity_inputs_are_validated_even_when_masks_are_empty():
    p = Params(resolution=(2, 2, 2), dx=0.5, particles_per_cell=1)
    solver = STFLIPSolver(p, "cpu")
    empty = np.zeros(p.resolution, dtype=bool)

    with pytest.raises(ValueError, match="three finite values"):
        solver.add_liquid_mask(empty, (np.nan, 0.0, 0.0))
    with pytest.raises(ValueError, match="three finite values"):
        solver.add_inflow(empty, (0.0, np.inf, 0.0))


def test_liquid_and_inflow_masks_must_match_the_solver_grid():
    p = Params(resolution=(2, 3, 4), dx=0.5, particles_per_cell=1)
    solver = STFLIPSolver(p, "cpu")
    outside = np.ones((3, 3, 4), dtype=bool)
    broadcastable = np.ones((1, 3, 4), dtype=bool)

    with pytest.raises(ValueError, match="cell_mask must have solver shape"):
        solver.add_liquid_mask(outside)
    with pytest.raises(ValueError, match="cell_mask must have solver shape"):
        solver.add_inflow(broadcastable)

    assert solver.pos.shape == (0, 3)
    assert not solver._inflows


def test_inflow_owns_its_cell_mask_after_registration():
    p = Params(
        resolution=(2, 2, 2),
        dx=0.5,
        gravity=(0.0, 0.0, 0.0),
        particles_per_cell=1,
        seed=12,
    )
    solver = STFLIPSolver(p, "cpu")
    mask = np.zeros(p.resolution, dtype=bool)
    mask[0, 0, 0] = True
    solver.add_inflow(mask)

    # Mutating caller-owned storage must not retarget a registered inflow.
    mask[:] = False
    mask[1, 1, 1] = True
    solver._seed_inflows()

    seeded_cell = np.floor(solver.pos[0] / p.dx).astype(int)
    np.testing.assert_array_equal(seeded_cell, (0, 0, 0))


def test_overlapping_inflows_use_registration_order_precedence():
    p = Params(
        resolution=(3, 1, 1), dx=1.0,
        gravity=(0.0, 0.0, 0.0), particles_per_cell=1, seed=8,
    )
    solver = STFLIPSolver(p, "cpu")
    first = np.zeros(p.resolution, dtype=bool)
    first[0:2, 0, 0] = True
    second = np.zeros(p.resolution, dtype=bool)
    second[1:3, 0, 0] = True
    solver.add_inflow(first, (1.0, 0.0, 0.0))
    solver.add_inflow(second, (2.0, 0.0, 0.0))

    solver._seed_inflows()

    cells = np.floor(solver.pos[:, 0] / p.dx).astype(int)
    assert cells.tolist() == [0, 1, 2]
    np.testing.assert_array_equal(
        solver.vel,
        np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    # Occupancy from the first call prevents either source from reseeding.
    solver._seed_inflows()
    assert solver.pos.shape[0] == 3


def test_inflow_schedule_is_inclusive_at_start_and_exclusive_at_end():
    p = Params(
        resolution=(2, 2, 2), dx=0.5, gravity=(0.0, 0.0, 0.0),
        frame_dt=0.25, particles_per_cell=1, seed=17,
    )
    solver = STFLIPSolver(p, "cpu")
    mask = np.zeros(p.resolution, dtype=bool)
    mask[0, 0, 0] = True
    solver.add_inflow(
        mask, (1.0, 0.0, 0.0), start_time=0.5, end_time=1.0,
    )

    solver.time = 0.5 - 1e-6
    solver._seed_inflows()
    assert solver.pos.shape[0] == 0

    solver.time = 0.5
    solver._seed_inflows()
    assert solver.pos.shape[0] == 1

    solver.pos = solver.pos[:0]
    solver.vel = solver.vel[:0]
    solver.dt_resid = solver.dt_resid[:0]
    solver.time = 1.0
    solver._seed_inflows()
    assert solver.pos.shape[0] == 0


@pytest.mark.parametrize(
    ("start_time", "end_time"),
    (
        (-1.0, None), (np.nan, None), (True, None),
        (1.0, 0.5), (0.0, np.inf), (0.0, False),
    ),
)
def test_inflow_schedule_rejects_invalid_intervals(start_time, end_time):
    p = Params(resolution=(2, 2, 2), dx=0.5)
    solver = STFLIPSolver(p, "cpu")
    mask = np.ones(p.resolution, dtype=bool)
    with pytest.raises(ValueError, match="inflow .*time"):
        solver.add_inflow(mask, start_time=start_time, end_time=end_time)


def test_zero_duration_inflow_schedule_is_valid_and_inactive():
    p = Params(resolution=(2, 2, 2), dx=0.5, particles_per_cell=1)
    solver = STFLIPSolver(p, "cpu")
    mask = np.ones(p.resolution, dtype=bool)
    solver.add_inflow(mask, start_time=0.5, end_time=0.5)
    solver.time = 0.5

    solver._seed_inflows()

    assert solver.pos.shape[0] == 0


def test_inactive_inflow_filters_before_domain_occupancy_allocation(monkeypatch):
    p = Params(
        resolution=(2, 2, 2), dx=0.5, particles_per_cell=1,
        frame_dt=1.0 / 24.0,
    )
    solver = STFLIPSolver(p, "cpu")
    mask = np.ones(p.resolution, dtype=bool)
    solver.add_liquid_mask(mask)
    solver.add_inflow(mask, start_time=1.0, end_time=2.0)

    def unexpected_scatter(*_args, **_kwargs):
        raise AssertionError("inactive inflow allocated an occupancy grid")

    monkeypatch.setattr(solver.be, "scatter_add", unexpected_scatter)
    solver.time = 0.0
    solver._seed_inflows()


def test_inflow_exclusive_endpoint_tolerates_accumulated_frame_roundoff():
    frame_dt = 1.0 / 24.0
    p = Params(
        resolution=(2, 2, 2), dx=0.5, particles_per_cell=1,
        frame_dt=frame_dt,
    )
    solver = STFLIPSolver(p, "cpu")
    mask = np.ones(p.resolution, dtype=bool)
    solver.add_inflow(mask, start_time=0.0, end_time=0.5)
    solver.time = sum(frame_dt for _ in range(12))

    solver._seed_inflows()

    assert solver.pos.shape[0] == 0


def test_dam_break_runs_and_stays_finite():
    s = _dam_break()
    n0 = s.pos.shape[0]
    assert n0 > 0
    for _ in range(5):
        stats = s.step_frame()
        assert stats.steps >= 1
    assert s.pos.shape[0] == n0  # fixed particle set, no re-seeding
    pos = s.be.to_numpy(s.pos)
    vel = s.be.to_numpy(s.vel)
    assert np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))
    size = s.size
    for ax in range(3):
        assert pos[:, ax].min() >= 0.0
        assert pos[:, ax].max() <= size[ax]
    # The column should have collapsed: fluid spread beyond the initial third.
    assert pos[:, 0].max() > size[0] * 0.5


def test_frame_stats_report_each_substep_without_extra_vmax_samples():
    s = _dam_break(n=8, cfl=0.25, ppc=1)
    s.vel[:] = s.be.xp.asarray((4.0, 0.0, 0.0), dtype=s.be.xp.float32)

    stats = s.step_frame()

    assert stats.steps > 1
    assert len(stats.dt_values) == stats.steps
    assert len(stats.particle_cfl_estimated_values) == stats.steps
    assert len(stats.particle_cfl_actual_values) == stats.steps
    assert len(stats.pcg_iters) == stats.steps
    assert len(stats.pcg_rel_residuals) == stats.steps
    assert np.all(np.isfinite(stats.particle_cfl_estimated_values))
    assert np.all(np.isfinite(stats.particle_cfl_actual_values))
    assert np.all(np.isfinite(stats.pcg_rel_residuals))
    assert stats.particle_cfl_estimated_values[0] == pytest.approx(
        4.0 * stats.dt_values[0] / s.p.dx)
    assert stats.particle_cfl_actual_values[-1] == pytest.approx(
        stats.max_speed * stats.dt_values[-1] / s.p.dx)
    for i in range(1, stats.steps):
        # The post-step reduction from i-1 is reused to estimate step i.
        previous_speed = (
            stats.particle_cfl_actual_values[i - 1] / stats.dt_values[i - 1]
        )
        estimated_speed = (
            stats.particle_cfl_estimated_values[i] / stats.dt_values[i]
        )
        assert estimated_speed == pytest.approx(previous_speed)


def test_actual_particle_cfl_can_exceed_estimate_under_acceleration():
    n = 8
    p = Params(
        resolution=(n, n, n), dx=1.0 / n,
        gravity=(0.0, 0.0, -40.0), frame_dt=0.05,
        cfl_target=0.1, particles_per_cell=1,
        st_enabled=False, seed=17,
    )
    s = STFLIPSolver(p, "cpu")
    mask = np.zeros(p.resolution, dtype=bool)
    mask[2:6, 2:6, 4:6] = True
    s.add_liquid_mask(mask)

    stats = s.step_frame()

    assert stats.steps == 1
    assert stats.particle_cfl_estimated_values == pytest.approx([0.0])
    assert stats.particle_cfl_actual_values[0] > p.cfl_target
    assert stats.particle_cfl_actual_values[0] > (
        stats.particle_cfl_estimated_values[0])


def test_jitter_residual_bound():
    """Appendix A: |dt_resid| <= dt_max / 2 at all times."""
    s = _dam_break(cfl=10.0)
    dt_max = 0.0
    for _ in range(6):
        stats = s.step_frame()
        dt_max = max(dt_max, max(stats.dt_values))
        resid = np.abs(s.be.to_numpy(s.dt_resid))
        assert resid.max() <= 0.5 * dt_max + 1e-7


def test_still_pool_stays_calm():
    n = 20
    p = Params(resolution=(n, n, n), dx=1.0 / n, frame_dt=1.0 / 24.0,
               cfl_target=6.0, seed=1)
    s = STFLIPSolver(p, "cpu")
    mask = np.zeros((n, n, n), dtype=bool)
    mask[:, :, : n // 2] = True
    s.add_liquid_mask(mask)
    for _ in range(4):
        s.step_frame()
    pos = s.be.to_numpy(s.pos)
    # Surface can ripple slightly but the pool must not explode upward.
    assert pos[:, 2].max() < 0.75
    speed = np.linalg.norm(s.be.to_numpy(s.vel), axis=1)
    assert speed.max() < 3.0


def test_instantaneous_p2g_ablation_runs():
    s = _dam_break(st=False)
    for _ in range(2):
        s.step_frame()
    resid = np.abs(s.be.to_numpy(s.dt_resid))
    assert resid.max() < 1e-9  # no jitter -> particles stay synchronised


def test_render_particles_resynchronised_shape():
    s = _dam_break()
    s.step_frame()
    pos, vel = s.get_render_particles()
    assert pos.shape == s.pos.shape and vel.shape == s.vel.shape
    assert np.all(np.isfinite(pos))


def test_advection_local_cfl_is_not_clipped_at_40_substeps():
    class SampleCountingSolver(STFLIPSolver):
        sample_calls = 0

        def _sample_faces(self, grids, pos):
            self.sample_calls += 1
            return self.be.xp.zeros_like(pos)

    p = Params(
        resolution=(4, 4, 4), dx=1.0, gravity=(0.0, 0.0, 0.0),
        cfl_local=1.0,
    )
    s = SampleCountingSolver(p, "cpu")
    s.pos = s.be.xp.asarray([[1.5, 1.5, 1.5]], dtype=s.be.xp.float32)
    # The sampled grid—not the unrelated particle-carried velocity—bounds the
    # advector. ceil(41 cells / cfl_local) requires 41 RK substeps, each with
    # three Ralston RK3 samples.
    s.vel = s.be.xp.asarray([[0.0, 0.0, 0.0]], dtype=s.be.xp.float32)
    grids = {
        "u": np.full((5, 4, 4), 41.0, dtype=np.float32),
        "v": np.zeros((4, 5, 4), dtype=np.float32),
        "w": np.zeros((4, 4, 5), dtype=np.float32),
    }
    dt_act = s.be.xp.asarray([1.0], dtype=s.be.xp.float32)

    s._advect(grids, s.pos.copy(), dt_act)

    assert s.sample_calls == 3 * 41


def test_advection_local_cfl_does_not_use_particle_velocity():
    class SampleCountingSolver(STFLIPSolver):
        sample_calls = 0

        def _sample_faces(self, grids, pos):
            self.sample_calls += 1
            return self.be.xp.zeros_like(pos)

    p = Params(
        resolution=(4, 4, 4), dx=1.0, gravity=(0.0, 0.0, 0.0),
        cfl_local=1.0,
    )
    s = SampleCountingSolver(p, "cpu")
    s.pos = np.asarray([[1.5, 1.5, 1.5]], dtype=np.float32)
    s.vel = np.asarray([[1000.0, 0.0, 0.0]], dtype=np.float32)
    grids = {
        "u": np.full((5, 4, 4), 2.0, dtype=np.float32),
        "v": np.zeros((4, 5, 4), dtype=np.float32),
        "w": np.zeros((4, 4, 5), dtype=np.float32),
    }

    s._advect(grids, s.pos.copy(), np.asarray([1.0], dtype=np.float32))

    assert s.sample_calls == 3 * 2


def _constant_advection_grids(shape, velocity):
    nx, ny, nz = shape
    u_value, v_value, w_value = velocity
    return {
        "u": np.full((nx + 1, ny, nz), u_value, dtype=np.float32),
        "v": np.full((nx, ny + 1, nz), v_value, dtype=np.float32),
        "w": np.full((nx, ny, nz + 1), w_value, dtype=np.float32),
    }


def test_volume_outflow_is_checked_at_every_local_advection_substep():
    p = Params(
        resolution=(6, 2, 2), dx=1.0, gravity=(0.0, 0.0, 0.0),
        cfl_local=1.0,
    )
    solver = STFLIPSolver(p, "cpu")
    sink = np.zeros(p.resolution, dtype=bool)
    sink[3, 0, 0] = True
    solver.add_outflow(sink, "VOLUME")
    positions = np.asarray([[1.5, 0.5, 0.5]], dtype=np.float32)

    result, volume_removed, pressure_removed = solver._advect(
        _constant_advection_grids(p.resolution, (4.0, 0.0, 0.0)),
        positions,
        np.asarray([1.0], dtype=np.float32),
        track_outflows=True,
    )

    # A final-position-only test would see x=5.5 and miss the one-cell sink.
    assert result[0, 0] == pytest.approx(3.5)
    assert volume_removed.tolist() == [True]
    assert pressure_removed.tolist() == [False]


def test_volume_outflow_uses_swept_voxel_traversal_for_diagonal_motion():
    p = Params(
        resolution=(3, 3, 1), dx=1.0, gravity=(0.0, 0.0, 0.0),
        cfl_local=1.0,
    )
    solver = STFLIPSolver(p, "cpu")
    sink = np.zeros(p.resolution, dtype=bool)
    sink[1, 1, 0] = True
    solver.add_outflow(sink, "VOLUME")

    _, removed, _ = solver._advect(
        _constant_advection_grids(p.resolution, (0.3, -0.3, 0.0)),
        np.asarray([[0.9, 1.2, 0.5]], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        track_outflows=True,
    )
    assert removed.tolist() == [True]

    _, nearby_removed, _ = solver._advect(
        _constant_advection_grids(p.resolution, (0.3, -0.1, 0.0)),
        np.asarray([[0.9, 2.2, 0.5]], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        track_outflows=True,
    )
    assert nearby_removed.tolist() == [False]


def test_swept_voxel_traversal_is_warning_free_for_stationary_segments():
    p = Params(resolution=(2, 2, 2), dx=1.0)
    solver = STFLIPSolver(p, "cpu")
    mask = np.zeros(p.resolution, dtype=bool)
    point = np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32)

    with np.errstate(all="raise"):
        hit = solver._segments_hit_mask(mask, point, point)

    assert hit.tolist() == [False]


def test_pressure_outflow_allows_only_its_opened_exterior_face():
    p = Params(
        resolution=(4, 3, 3), dx=1.0, gravity=(0.0, 0.0, 0.0),
        cfl_local=1.0,
    )
    solver = STFLIPSolver(p, "cpu")
    opening = np.zeros(p.resolution, dtype=bool)
    opening[0, 1, 1] = True
    solver.add_outflow(opening, "pressure")
    positions = np.asarray(
        [[0.25, 1.5, 1.5], [0.25, 0.5, 0.5]], dtype=np.float32
    )

    result, volume_removed, pressure_removed = solver._advect(
        _constant_advection_grids(p.resolution, (-2.0, 0.0, 0.0)),
        positions,
        np.ones((2,), dtype=np.float32),
        track_outflows=True,
    )

    assert volume_removed.tolist() == [False, False]
    assert pressure_removed.tolist() == [True, False]
    assert result[0, 0] < 0.0
    assert result[1, 0] >= 0.0
    stats = solver.outflow_stats()
    assert stats["pressure_open_face_count"] == 1
    assert stats["pressure_open_face_counts"] == {
        "x_min": 1,
        "x_max": 0,
        "y_min": 0,
        "y_max": 0,
        "z_min": 0,
        "z_max": 0,
    }


def test_pressure_exit_uses_segment_boundary_intersection_coordinates():
    p = Params(
        resolution=(3, 3, 1), dx=1.0, gravity=(0.0, 0.0, 0.0),
        cfl_local=1.0,
    )
    start = np.asarray([[0.2, 1.6, 0.5]], dtype=np.float32)
    grids = _constant_advection_grids(p.resolution, (-0.4, 0.6, 0.0))

    open_at_intersection = STFLIPSolver(p, "cpu")
    mask = np.zeros(p.resolution, dtype=bool)
    mask[0, 1, 0] = True
    open_at_intersection.add_outflow(mask, "PRESSURE")
    exited_pos, _, exited = open_at_intersection._advect(
        grids, start.copy(), np.asarray([1.0], dtype=np.float32),
        track_outflows=True,
    )
    assert exited.tolist() == [True]
    assert exited_pos[0, 0] < 0.0

    open_only_at_endpoint = STFLIPSolver(p, "cpu")
    mask[:] = False
    mask[0, 2, 0] = True
    open_only_at_endpoint.add_outflow(mask, "PRESSURE")
    clamped_pos, _, exited = open_only_at_endpoint._advect(
        grids, start.copy(), np.asarray([1.0], dtype=np.float32),
        track_outflows=True,
    )
    assert exited.tolist() == [False]
    assert clamped_pos[0, 0] >= 0.0


def test_pressure_outflow_requires_an_exterior_intersection():
    p = Params(resolution=(3, 3, 3), dx=1.0)
    solver = STFLIPSolver(p, "cpu")
    interior = np.zeros(p.resolution, dtype=bool)
    interior[1, 1, 1] = True

    with pytest.raises(ValueError, match="intersect the domain exterior"):
        solver.add_outflow(interior, "PRESSURE")
    with pytest.raises(ValueError, match="VOLUME.*PRESSURE"):
        solver.add_outflow(interior, "DELETE")


def test_cull_outflows_filters_all_state_and_updates_cumulative_stats():
    p = Params(
        resolution=(3, 3, 3), dx=1.0, gravity=(0.0, 0.0, 0.0),
        particles_per_cell=1,
    )
    solver = STFLIPSolver(p, "cpu")
    volume = np.zeros(p.resolution, dtype=bool)
    volume[1, 0, 0] = True
    pressure = np.zeros(p.resolution, dtype=bool)
    pressure[0, 1, 1] = True
    solver.add_outflow(volume, "VOLUME")
    solver.add_outflow(pressure, "PRESSURE")
    solver.pos = np.asarray(
        [[1.5, 0.5, 0.5], [-0.1, 1.5, 1.5], [2.5, 2.5, 2.5]],
        dtype=np.float32,
    )
    solver.vel = np.asarray(
        [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    solver.dt_resid = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

    removed = solver.cull_outflows()

    assert removed == {
        "particles_removed": 2,
        "volume_outflow_removed": 1,
        "pressure_outflow_removed": 1,
    }
    np.testing.assert_array_equal(solver.pos, [[2.5, 2.5, 2.5]])
    np.testing.assert_array_equal(solver.vel, [[30.0, 0.0, 0.0]])
    np.testing.assert_allclose(solver.dt_resid, [0.3])
    stats = solver.outflow_stats()
    assert stats["particles_removed_total"] == 2
    assert stats["volume_outflow_removed_total"] == 1
    assert stats["pressure_outflow_removed_total"] == 1

    assert solver.cull_outflows() == {
        "particles_removed": 0,
        "volume_outflow_removed": 0,
        "pressure_outflow_removed": 0,
    }
    assert solver.outflow_stats()["particles_removed_total"] == 2


def test_render_resync_filters_pressure_exits_without_mutating_solver_state():
    p = Params(
        resolution=(3, 2, 2), dx=1.0, gravity=(0.0, 0.0, 0.0),
        particles_per_cell=1,
    )
    solver = STFLIPSolver(p, "cpu")
    opening = np.zeros(p.resolution, dtype=bool)
    opening[0, 0, 0] = True
    solver.add_outflow(opening, "PRESSURE")
    solver.pos = np.asarray([[0.2, 0.5, 0.5]], dtype=np.float32)
    solver.vel = np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float32)
    solver.dt_resid = np.asarray([0.5], dtype=np.float32)
    solver._grids = _constant_advection_grids(
        p.resolution, (-1.0, 0.0, 0.0)
    )
    raw_state = (
        solver.pos.copy(), solver.vel.copy(), solver.dt_resid.copy()
    )
    before = solver.outflow_stats()

    positions, velocities = solver.get_render_particles()

    assert positions.shape == velocities.shape == (0, 3)
    np.testing.assert_array_equal(solver.pos, raw_state[0])
    np.testing.assert_array_equal(solver.vel, raw_state[1])
    np.testing.assert_array_equal(solver.dt_resid, raw_state[2])
    assert solver.outflow_stats() == before


def test_step_filters_all_particle_arrays_and_reports_volume_removal():
    p = Params(
        resolution=(6, 2, 2), dx=1.0, gravity=(0.0, 0.0, 0.0),
        frame_dt=1.0, cfl_target=4.0, cfl_local=1.0,
        particles_per_cell=1, st_enabled=False, flip_blend=1.0,
    )
    solver = STFLIPSolver(p, "cpu")
    solver.pos = np.asarray(
        [[1.5, 0.5, 0.5], [1.5, 1.5, 1.5]], dtype=np.float32
    )
    solver.vel = np.asarray(
        [[4.0, 11.0, 12.0], [4.0, 21.0, 22.0]], dtype=np.float32
    )
    solver.dt_resid = np.asarray([0.0, 1.5], dtype=np.float32)
    sink = np.zeros(p.resolution, dtype=bool)
    sink[3, 0, 0] = True
    solver.add_outflow(sink, "VOLUME")
    grids = _constant_advection_grids(p.resolution, (4.0, 0.0, 0.0))
    for name in ("u", "v", "w"):
        grids[f"{name}_valid"] = np.ones_like(grids[name], dtype=bool)
        grids[f"{name}_phi"] = np.ones_like(grids[name], dtype=np.float32)
    grids["c_phi"] = np.zeros(p.resolution, dtype=np.float32)
    solver._p2g = lambda _dt_prev: grids
    stats = FrameStats()

    solver._step(1.0, stats)

    assert solver.pos.shape == (1, 3)
    assert solver.vel.shape == (1, 3)
    assert solver.dt_resid.shape == (1,)
    np.testing.assert_array_equal(solver.vel[0], (4.0, 21.0, 22.0))
    assert solver.dt_resid[0] == pytest.approx(0.5)
    assert stats.particles_removed == 1
    assert stats.volume_outflow_removed == 1
    assert stats.pressure_outflow_removed == 0
    assert solver.outflow_stats()["particles_removed_total"] == 1


def test_step_frame_finishes_clock_after_outflow_removes_every_particle():
    p = Params(
        resolution=(2, 2, 2), dx=1.0, gravity=(0.0, 0.0, 0.0),
        frame_dt=1.0, cfl_target=1.0, particles_per_cell=1,
        st_enabled=False,
    )
    solver = STFLIPSolver(p, "cpu")
    solver.pos = np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32)
    solver.vel = np.asarray([[4.0, 0.0, 0.0]], dtype=np.float32)
    solver.dt_resid = np.zeros((1,), dtype=np.float32)
    sink = np.zeros(p.resolution, dtype=bool)
    sink[0, 0, 0] = True
    solver.add_outflow(sink, "VOLUME")

    stats = solver.step_frame()

    assert solver.pos.shape[0] == 0
    assert solver.time == pytest.approx(1.0)
    assert sum(stats.dt_values) + stats.inactive_time_s == pytest.approx(
        p.frame_dt
    )
    assert stats.dt_values == pytest.approx([0.25])
    assert stats.inactive_time_s == pytest.approx(0.75)
    assert stats.steps == 1
    assert len(stats.pcg_iters) == stats.steps
    assert len(stats.particle_cfl_estimated_values) == stats.steps
    assert len(stats.particle_cfl_actual_values) == stats.steps
    assert stats.particles_removed == 1


def test_empty_frame_fast_forwards_without_grid_work():
    p = Params(
        resolution=(8, 8, 8), dx=0.125, frame_dt=0.5,
        gravity=(0.0, 0.0, 0.0), particles_per_cell=1,
    )
    solver = STFLIPSolver(p, "cpu")

    def forbidden_step(dt, stats):
        raise AssertionError("empty frame must not execute a grid step")

    solver._step = forbidden_step
    stats = solver.step_frame()

    assert solver.time == pytest.approx(0.5)
    assert stats.steps == 0
    assert stats.dt_values == []
    assert stats.inactive_time_s == pytest.approx(0.5)


def test_step_without_outflows_uses_untracked_legacy_advection_path():
    p = Params(
        resolution=(2, 2, 2), dx=0.5, gravity=(0.0, 0.0, 0.0),
        particles_per_cell=1, st_enabled=False,
    )
    solver = STFLIPSolver(p, "cpu")
    mask = np.zeros(p.resolution, dtype=bool)
    mask[0, 0, 0] = True
    solver.add_liquid_mask(mask)
    tracking_values = []

    def fake_advect(grids, pos, dt_act, *, track_outflows=False):
        tracking_values.append(track_outflows)
        return pos

    solver._advect = fake_advect
    solver._step(0.1, FrameStats())

    assert tracking_values == [False]


def test_no_outflow_solver_keeps_dense_masks_lazy_and_stats_transfer_free():
    solver = STFLIPSolver(
        Params(resolution=(8, 8, 8), dx=0.125, particles_per_cell=1),
        "cpu",
    )
    assert solver._volume_outflow is None
    assert solver._pressure_outflow is None

    def forbidden_transfer(values):
        raise AssertionError("empty outflow stats must not transfer arrays")

    solver.be.to_numpy = forbidden_transfer
    stats = solver.outflow_stats()

    assert stats["volume_cell_count"] == 0
    assert stats["pressure_cell_count"] == 0
    assert stats["pressure_open_face_count"] == 0


def test_outflow_geometry_stats_are_cached_until_geometry_changes():
    p = Params(resolution=(2, 2, 2), dx=1.0)
    solver = STFLIPSolver(p, "cpu")
    mask = np.zeros(p.resolution, dtype=bool)
    mask[0, 0, 0] = True
    solver.add_outflow(mask, "PRESSURE")
    original = solver.be.to_numpy
    transfers = 0

    def counting_transfer(values):
        nonlocal transfers
        transfers += 1
        return original(values)

    solver.be.to_numpy = counting_transfer
    first = solver.outflow_stats()
    second = solver.outflow_stats()
    assert first == second
    assert transfers == 1

    second_mask = np.zeros(p.resolution, dtype=bool)
    second_mask[-1, -1, -1] = True
    solver.add_outflow(second_mask, "PRESSURE")
    solver.outflow_stats()
    assert transfers == 2


def test_pressure_outflow_projection_uses_half_cell_exterior_dirichlet():
    p = Params(
        resolution=(2, 3, 3), dx=1.0, gravity=(0.0, 0.0, 0.0),
        particles_per_cell=1, st_enabled=False, flip_blend=0.0,
        pcg_tol=1e-8, pcg_max_iter=200,
    )
    solver = STFLIPSolver(p, "cpu")
    opening = np.zeros(p.resolution, dtype=bool)
    opening[0, 1, 1] = True
    solver.add_outflow(opening, "PRESSURE")
    grids = _constant_advection_grids(p.resolution, (0.0, 0.0, 0.0))
    grids["u"][0, 1, 1] = -1.0
    for name in ("u", "v", "w"):
        grids[f"{name}_valid"] = np.ones_like(grids[name], dtype=bool)
        grids[f"{name}_phi"] = np.ones_like(grids[name], dtype=np.float32)
    grids["c_phi"] = np.ones(p.resolution, dtype=np.float32)
    solver._p2g = lambda _dt_prev: grids
    stats = FrameStats()

    solver._step(0.1, stats)

    assert stats.pcg_iters[0] > 0
    assert solver._grids["u"][0, 1, 1] == pytest.approx(0.0, abs=2e-5)


def test_solid_obstacle_blocks_particles():
    n = 24
    s = _dam_break(n=n)
    # Solid floor slab occupying the bottom quarter on the right half.
    sdf = np.full((n, n, n), 1e9, dtype=np.float32)
    dx = 1.0 / n
    for i in range(n):
        for k in range(n):
            x = (i + 0.5) * dx
            z = (k + 0.5) * dx
            if x > 0.5:
                sdf[i, :, k] = z - 0.25 if z < 0.5 else sdf[i, :, k]
    s.set_solid_sdf(sdf)
    for _ in range(4):
        s.step_frame()
    pos = s.be.to_numpy(s.pos)
    inside = (pos[:, 0] > 0.55) & (pos[:, 2] < 0.2)
    assert inside.mean() < 0.02  # essentially no particles deep in the solid


def test_cell_sdf_only_keeps_binary_aperture_fallback():
    p = Params(
        resolution=(2, 2, 2), dx=0.5, gravity=(0.0, 0.0, 0.0),
        particles_per_cell=1,
    )
    s = STFLIPSolver(p, "cpu")
    s.set_solid_sdf(np.ones(p.resolution, dtype=np.float32))

    alpha_u, alpha_v, alpha_w, solid = s._solid_face_apertures()
    assert not np.any(s.be.to_numpy(solid))
    for alpha in (alpha_u, alpha_v, alpha_w):
        host = s.be.to_numpy(alpha)
        assert set(np.unique(host)) <= {0.0, 1.0}

    stats = s.solid_aperture_stats()
    assert stats == {
        "model": "binary_cell_center",
        "total_face_count": 36,
        "blocked_face_count": 24,
        "fractional_face_count": 0,
        "open_face_count": 12,
    }


def test_fractional_node_sdf_closes_domain_faces_and_reports_counts():
    p = Params(
        resolution=(2, 2, 2), dx=0.5, gravity=(0.0, 0.0, 0.0),
        particles_per_cell=1,
    )
    s = STFLIPSolver(p, "cpu")
    cell_sdf = np.ones(p.resolution, dtype=np.float32)
    node_sdf = np.empty((3, 3, 3), dtype=np.float32)
    node_sdf[:, 0, :] = -0.5
    node_sdf[:, 1, :] = 0.5
    node_sdf[:, 2, :] = 1.5
    s.set_solid_sdf(cell_sdf, node_sdf)

    alpha_u, alpha_v, alpha_w, solid = (
        tuple(s.be.to_numpy(value) for value in s._solid_face_apertures())
    )
    assert not np.any(solid)
    assert np.all(alpha_u[[0, -1], :, :] == 0.0)
    assert np.all(alpha_v[:, [0, -1], :] == 0.0)
    assert np.all(alpha_w[:, :, [0, -1]] == 0.0)
    assert np.any((alpha_u > 0.0) & (alpha_u < 1.0))

    stats = s.solid_aperture_stats()
    assert stats["model"] == "fractional_node_sdf"
    assert stats["total_face_count"] == 36
    assert stats["blocked_face_count"] >= 24
    assert stats["fractional_face_count"] > 0
    assert (
        stats["blocked_face_count"]
        + stats["fractional_face_count"]
        + stats["open_face_count"]
        == stats["total_face_count"]
    )


def test_pressure_outflow_restores_only_geometric_exterior_aperture():
    p = Params(
        resolution=(2, 2, 2), dx=0.5, gravity=(0.0, 0.0, 0.0),
        particles_per_cell=1,
    )
    solver = STFLIPSolver(p, "cpu")
    cell_sdf = np.ones(p.resolution, dtype=np.float32)
    node_sdf = np.empty((3, 3, 3), dtype=np.float32)
    node_sdf[:, 0, :] = -0.5
    node_sdf[:, 1, :] = 0.5
    node_sdf[:, 2, :] = 1.5
    solver.set_solid_sdf(cell_sdf, node_sdf)
    opening = np.zeros(p.resolution, dtype=bool)
    opening[0, 0, 0] = True
    solver.add_outflow(opening, "PRESSURE")

    alpha_u, _, _, _ = solver._active_face_apertures()

    assert alpha_u[0, 0, 0] == pytest.approx(0.5)
    assert solver._pressure_face_masks()[0][0, 0, 0]


def test_pressure_outflow_cannot_open_zero_aperture_fractional_face():
    p = Params(
        resolution=(2, 2, 2), dx=0.5, gravity=(0.0, 0.0, 0.0),
        particles_per_cell=1,
    )
    solver = STFLIPSolver(p, "cpu")
    cell_sdf = np.ones(p.resolution, dtype=np.float32)
    node_sdf = np.ones((3, 3, 3), dtype=np.float32)
    node_sdf[0] = -1.0
    solver.set_solid_sdf(cell_sdf, node_sdf)
    opening = np.zeros(p.resolution, dtype=bool)
    opening[0, 0, 0] = True
    solver.add_outflow(opening, "PRESSURE")

    alpha_u, _, _, _ = solver._active_face_apertures()

    assert alpha_u[0, 0, 0] == 0.0
    assert not solver._pressure_face_masks()[0][0, 0, 0]


def test_extrapolation_does_not_cross_zero_aperture_faces():
    p = Params(
        resolution=(2, 2, 2), dx=0.5, gravity=(0.0, 0.0, 0.0),
        particles_per_cell=1,
    )
    s = STFLIPSolver(p, "cpu")
    u = np.zeros((5, 1, 1), dtype=np.float32)
    u[0, 0, 0] = 2.0
    valid = np.zeros_like(u, dtype=bool)
    valid[0, 0, 0] = True
    allowed = np.ones_like(u, dtype=bool)
    allowed[2, 0, 0] = False

    result, result_valid = s._extrapolate(
        u, valid, layers=5, allowed=allowed)

    assert result[1, 0, 0] == pytest.approx(2.0)
    assert result[2, 0, 0] == pytest.approx(0.0)
    assert result[3, 0, 0] == pytest.approx(0.0)
    assert not result_valid[2, 0, 0]
    assert not result_valid[3, 0, 0]


def test_fractional_projection_removes_aperture_weighted_divergence():
    p = Params(
        resolution=(2, 2, 2), dx=0.5, gravity=(0.0, 0.0, 0.0),
        particles_per_cell=1, st_enabled=False, flip_blend=0.0,
        cfl_target=1.0, pcg_tol=1e-8, pcg_max_iter=200,
    )
    s = STFLIPSolver(p, "cpu")
    cell_sdf = np.ones(p.resolution, dtype=np.float32)
    node_sdf = np.empty((3, 3, 3), dtype=np.float32)
    node_sdf[:, 0, :] = -0.5
    node_sdf[:, 1, :] = 0.5
    node_sdf[:, 2, :] = 1.5
    s.set_solid_sdf(cell_sdf, node_sdf)
    alpha_u, alpha_v, alpha_w, _ = s._solid_face_apertures()

    u = np.zeros((3, 2, 2), dtype=np.float32)
    v = np.zeros((2, 3, 2), dtype=np.float32)
    w = np.zeros((2, 2, 3), dtype=np.float32)
    # This internal face has alpha=0.5.  Correct projection requires alpha
    # in the PPE/divergence but not in the pressure acceleration itself.
    u[1, 0, :] = 1.0
    grids = {
        "u": u,
        "v": v,
        "w": w,
        "u_valid": np.ones_like(u, dtype=bool),
        "v_valid": np.ones_like(v, dtype=bool),
        "w_valid": np.ones_like(w, dtype=bool),
        "u_phi": np.ones_like(u),
        "v_phi": np.ones_like(v),
        "w_phi": np.ones_like(w),
        "c_phi": np.ones(p.resolution, dtype=np.float32),
    }
    before = weighted_divergence(
        u, v, w, alpha_u, alpha_v, alpha_w, p.dx)
    assert np.linalg.norm(before.ravel()) > 0.0
    s._p2g = lambda _dt_prev: grids

    frame_stats = FrameStats()
    s._step(0.1, frame_stats)

    after = weighted_divergence(
        s._grids["u"], s._grids["v"], s._grids["w"],
        alpha_u, alpha_v, alpha_w, p.dx,
    )
    assert frame_stats.pcg_iters and frame_stats.pcg_iters[0] > 0
    assert np.linalg.norm(after.ravel()) <= (
        5e-6 * np.linalg.norm(before.ravel()))


def test_no_energy_kick_for_zero_temporal_weight_particles():
    """Regression: an isolated particle whose time sample lands in the
    temporal kernel's zero tail deposits ~no mass, invalidating its own
    faces.  The FLIP delta must then be formed against the extrapolated old
    field (not hard zeros), or the particle receives its neighbours' full
    velocity as a spurious energy kick (~1.98x speed in one step)."""
    n = 16
    p = Params(resolution=(n, n, n), dx=1.0 / n, gravity=(0.0, 0.0, 0.0),
               frame_dt=1.0 / 24.0, cfl_target=8.0, seed=2)
    s = STFLIPSolver(p, "cpu")
    xp = s.be.xp
    mask = np.zeros((n, n, n), dtype=bool)
    mask[4:8, 6:10, 6:10] = True
    s.add_liquid_mask(mask, velocity=(1.0, 0.0, 0.0))
    # Isolated particle 3 cells from the block, same velocity, with a time
    # residual that puts theta at the very edge of the slab (W_T ~ 0).
    s.pos = xp.concatenate(
        [s.pos, xp.asarray([[11.5 / n, 8.0 / n, 8.0 / n]], dtype=xp.float32)])
    s.vel = xp.concatenate(
        [s.vel, xp.asarray([[1.0, 0.0, 0.0]], dtype=xp.float32)])
    dt = p.frame_dt / 4.0
    s._dt_prev = dt
    s.dt_resid = xp.concatenate(
        [s.dt_resid, xp.asarray([0.49995 * dt], dtype=xp.float32)])

    s._step(dt, FrameStats())
    speed = float(np.linalg.norm(s.be.to_numpy(s.vel)[-1]))
    assert speed < 1.3, f"isolated particle gained energy: |v| = {speed:.3f}"


def _checkpoint_solver(*, seed_particles):
    params = Params(
        resolution=(6, 6, 6),
        dx=1.0 / 6.0,
        gravity=(0.0, 0.0, -0.2),
        frame_dt=1.0 / 30.0,
        cfl_target=2.0,
        particles_per_cell=1,
        seed=2026,
    )
    solver = STFLIPSolver(params, "cpu")
    sdf = np.ones(params.resolution, dtype=np.float32)
    sdf[4, 2, 1] = -0.1
    solver.set_solid_sdf(sdf)

    inflow = np.zeros(params.resolution, dtype=bool)
    inflow[1, 4, 3] = True
    solver.add_inflow(inflow, (0.15, -0.05, 0.0))
    outflow = np.zeros(params.resolution, dtype=bool)
    outflow[0, 0, 0] = True
    solver.add_outflow(outflow, "VOLUME")
    if seed_particles:
        liquid = np.zeros(params.resolution, dtype=bool)
        liquid[0, 0, 0] = True
        liquid[2:4, 1:3, 2:4] = True
        solver.add_liquid_mask(liquid, (0.05, 0.0, 0.0))
        assert solver.cull_outflows()["volume_outflow_removed"] == 1
    return solver


def test_checkpoint_restore_matches_uninterrupted_future_frames_exactly():
    uninterrupted = _checkpoint_solver(seed_particles=True)
    uninterrupted.step_frame()
    snapshot = uninterrupted.checkpoint_state()

    restored = _checkpoint_solver(seed_particles=False)
    restored.restore_state(snapshot)

    restored_snapshot = restored.checkpoint_state()
    for name in ("pos", "vel", "dt_resid"):
        np.testing.assert_array_equal(restored_snapshot[name], snapshot[name])
    for name in (
        "time", "dt_prev", "rng_state", "outflow_removed_total",
        "volume_outflow_removed_total", "pressure_outflow_removed_total",
    ):
        assert restored_snapshot[name] == snapshot[name]

    for _ in range(2):
        expected_stats = uninterrupted.step_frame()
        actual_stats = restored.step_frame()
        assert actual_stats == expected_stats
        expected = uninterrupted.checkpoint_state()
        actual = restored.checkpoint_state()
        for name in ("pos", "vel", "dt_resid"):
            np.testing.assert_array_equal(actual[name], expected[name])
        for name in (
            "time", "dt_prev", "rng_state", "outflow_removed_total",
            "volume_outflow_removed_total",
            "pressure_outflow_removed_total",
        ):
            assert actual[name] == expected[name]


def test_partial_cache_resume_extends_without_reseeding(tmp_path):
    cache_dir = str(tmp_path)
    partial = _checkpoint_solver(seed_particles=True)
    pos, vel = partial.get_render_particles()
    cache.write_frame(cache_dir, 1, pos, vel)
    cache.write_checkpoint(cache_dir, 1, partial.checkpoint_state())
    metadata = {
        "frame_start": 1,
        "frame_end": 2,
        "frame_end_baked": 1,
        "checkpoint": {
            "schema": cache.CHECKPOINT_SCHEMA,
            "version": cache.CHECKPOINT_VERSION,
            "fingerprint": "d" * 64,
            "latest_frame": 1,
            "state": "CANCELLED",
        },
    }
    cache.write_meta(cache_dir, metadata)
    assert cache.resumable_frames(cache_dir) == [1]

    uninterrupted = _checkpoint_solver(seed_particles=False)
    uninterrupted.restore_state(partial.checkpoint_state())
    uninterrupted.step_frame()

    resumed = _checkpoint_solver(seed_particles=False)
    resumed.restore_state(cache.read_checkpoint(cache_dir, 1))
    resumed.step_frame()
    pos, vel = resumed.get_render_particles()
    cache.write_frame(cache_dir, 2, pos, vel)
    cache.write_checkpoint(cache_dir, 2, resumed.checkpoint_state())
    metadata["frame_end_baked"] = 2
    metadata["checkpoint"].update(latest_frame=2, state="COMPLETE")
    cache.write_meta(cache_dir, metadata)

    assert cache.resumable_frames(cache_dir) == [1, 2]
    expected = uninterrupted.checkpoint_state()
    actual = resumed.checkpoint_state()
    for name in ("pos", "vel", "dt_resid"):
        np.testing.assert_array_equal(actual[name], expected[name])
    assert actual["rng_state"] == expected["rng_state"]


def test_checkpoint_restore_is_strict_atomic_and_owns_arrays():
    solver = _checkpoint_solver(seed_particles=True)
    solver.step_frame()
    before = solver.checkpoint_state()
    detached = solver.checkpoint_state()
    detached["pos"][:] = 0.0
    np.testing.assert_array_equal(solver.checkpoint_state()["pos"], before["pos"])

    invalid = solver.checkpoint_state()
    invalid["vel"] = invalid["vel"].astype(np.float64)
    with pytest.raises(ValueError, match="vel"):
        solver.restore_state(invalid)
    after = solver.checkpoint_state()
    for name in ("pos", "vel", "dt_resid"):
        np.testing.assert_array_equal(after[name], before[name])
    assert after["rng_state"] == before["rng_state"]

    invalid = solver.checkpoint_state()
    invalid["pressure_outflow_removed_total"] += 1
    with pytest.raises(ValueError, match="inconsistent"):
        solver.restore_state(invalid)

    invalid = solver.checkpoint_state()
    invalid["pos"][0, 0] = np.float32(solver.size[0] + 1.0)
    with pytest.raises(ValueError, match="outside"):
        solver.restore_state(invalid)


@pytest.mark.gpu
def test_gpu_backend_parity():
    cupy = pytest.importorskip("cupy")
    available, reason = cuda_diagnostics(force=True)
    assert available, reason

    initial_field = SolidBodyRotation(
        center=(0.5, 0.5, 0.5),
        angular_velocity=(0.0, 0.0, 0.25),
        linear_velocity=(0.1, 0.0, 0.0),
    )
    s_cpu = _dam_break(n=16, seed=7, velocity=initial_field)
    s_gpu = STFLIPSolver(s_cpu.p, "cuda")
    # A fractional cut near the far x wall exercises node-SDF aperture math,
    # alpha-weighted projection, and setup statistics on both backends without
    # intersecting the two-frame dam-break front.
    node_x = 13.25 - np.arange(17, dtype=np.float32)[:, None, None]
    node_sdf = np.broadcast_to(node_x, (17, 17, 17)).copy()
    cell_x = 13.25 - (
        np.arange(16, dtype=np.float32) + 0.5
    )[:, None, None]
    cell_sdf = np.broadcast_to(cell_x, (16, 16, 16)).copy()
    s_cpu.set_solid_sdf(cell_sdf, node_sdf)
    s_gpu.set_solid_sdf(cell_sdf, node_sdf)
    assert s_cpu.solid_aperture_stats() == s_gpu.solid_aperture_stats()
    assert s_gpu.solid_aperture_stats()["fractional_face_count"] > 0
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[:5, :, :8] = True
    s_gpu.add_liquid_mask(mask, initial_field)
    volume_outflow = np.zeros_like(mask)
    volume_outflow[2, 8, 4] = True
    pressure_outflow = np.zeros_like(mask)
    pressure_outflow[0, 8, 12] = True
    for instance in (s_cpu, s_gpu):
        instance.add_outflow(volume_outflow, "VOLUME")
        instance.add_outflow(pressure_outflow, "PRESSURE")
    assert s_cpu.outflow_stats() == s_gpu.outflow_stats()
    assert s_gpu.outflow_stats()["pressure_open_face_count"] == 1

    # Jitter and velocity-field evaluation both happen in host float32 before
    # upload, so initial state is bitwise identical across solver backends.
    np.testing.assert_array_equal(
        s_gpu.be.to_numpy(s_gpu.pos), s_cpu.be.to_numpy(s_cpu.pos)
    )
    np.testing.assert_array_equal(
        s_gpu.be.to_numpy(s_gpu.vel), s_cpu.be.to_numpy(s_cpu.vel)
    )
    # Checkpoints are host float32 state and must restore directly onto CUDA
    # without changing source/outflow/solid configuration.
    s_gpu.restore_state(s_cpu.checkpoint_state())
    assert isinstance(s_gpu.pos, cupy.ndarray)
    np.testing.assert_array_equal(
        s_gpu.be.to_numpy(s_gpu.dt_resid), s_cpu.be.to_numpy(s_cpu.dt_resid)
    )
    cpu_culled = s_cpu.cull_outflows()
    gpu_culled = s_gpu.cull_outflows()
    assert gpu_culled == cpu_culled
    assert cpu_culled["volume_outflow_removed"] > 0
    np.testing.assert_array_equal(
        s_gpu.be.to_numpy(s_gpu.pos), s_cpu.be.to_numpy(s_cpu.pos)
    )
    np.testing.assert_array_equal(
        s_gpu.be.to_numpy(s_gpu.vel), s_cpu.be.to_numpy(s_cpu.vel)
    )
    np.testing.assert_array_equal(
        s_gpu.be.to_numpy(s_gpu.dt_resid), s_cpu.be.to_numpy(s_cpu.dt_resid)
    )

    # The backend must keep solver state and compute results on-device.
    assert s_gpu.be.xp is cupy
    assert isinstance(s_gpu.pos, cupy.ndarray)
    assert isinstance(s_gpu.vel, cupy.ndarray)
    kernel_values = s_gpu.be.from_numpy(
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    kernel_result = kernel_values * kernel_values + 1.0
    assert isinstance(kernel_result, cupy.ndarray)
    assert float(kernel_result.sum()) == pytest.approx(17.0)
    scatter_result = cupy.zeros((2,), dtype=cupy.float32)
    s_gpu.be.scatter_add(
        scatter_result,
        cupy.asarray([0, 0, 1], dtype=cupy.int32),
        cupy.asarray([1.0, 2.0, 4.0], dtype=cupy.float32),
    )
    s_gpu.be.synchronize()
    np.testing.assert_array_equal(
        s_gpu.be.to_numpy(scatter_result), np.asarray([3.0, 4.0]))

    for _ in range(2):
        cpu_stats = s_cpu.step_frame()
        gpu_stats = s_gpu.step_frame()
        assert gpu_stats.steps == cpu_stats.steps
        assert gpu_stats.particles_removed == cpu_stats.particles_removed
        assert (
            gpu_stats.volume_outflow_removed
            == cpu_stats.volume_outflow_removed
        )
        assert (
            gpu_stats.pressure_outflow_removed
            == cpu_stats.pressure_outflow_removed
        )
        # A nonzero PCG iteration count proves the pressure projection ran.
        assert gpu_stats.pcg_iters and all(i > 0 for i in gpu_stats.pcg_iters)
    s_gpu.be.synchronize()
    assert s_cpu.outflow_stats() == s_gpu.outflow_stats()
    assert s_cpu.outflow_stats()["volume_outflow_removed_total"] > 0

    assert s_gpu._grids
    assert all(isinstance(value, cupy.ndarray)
               for value in s_gpu._grids.values())

    pos_c = s_cpu.be.to_numpy(s_cpu.pos)
    pos_g = s_gpu.be.to_numpy(s_gpu.pos)
    vel_c = s_cpu.be.to_numpy(s_cpu.vel)
    vel_g = s_gpu.be.to_numpy(s_gpu.vel)
    assert np.all(np.isfinite(pos_g))
    np.testing.assert_allclose(pos_g, pos_c, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(vel_g, vel_c, rtol=2e-5, atol=2e-6)
    for grid_name in ("u", "v", "w", "u_phi", "v_phi", "w_phi", "c_phi"):
        np.testing.assert_allclose(
            s_gpu.be.to_numpy(s_gpu._grids[grid_name]),
            s_cpu.be.to_numpy(s_cpu._grids[grid_name]),
            rtol=2e-5,
            atol=2e-6,
        )

    cpu_record = measure_frame(
        frame=2, simulation_time_s=s_cpu.time, params=s_cpu.p,
        stats=cpu_stats, positions_local=pos_c, velocities=vel_c,
        compute_wall_s=None,
    )
    gpu_record = measure_frame(
        frame=2, simulation_time_s=s_gpu.time, params=s_gpu.p,
        stats=gpu_stats, positions_local=pos_g, velocities=vel_g,
        compute_wall_s=None,
    )
    for field in (
        "particles_removed",
        "volume_outflow_removed",
        "pressure_outflow_removed",
        "particle_cfl_estimated_max",
        "particle_cfl_actual_max",
        "speed_max_solver_units_per_s",
        "kinetic_energy_particle_estimate",
    ):
        assert gpu_record[field] == pytest.approx(
            cpu_record[field], rel=3e-5, abs=2e-6)

    cpu_grid_metrics = estimate_mac_grid_metrics(
        s_cpu._grids, s_cpu.p.dx)
    gpu_grid_metrics = estimate_mac_grid_metrics(
        s_gpu._grids, s_gpu.p.dx, array_module=cupy)
    assert gpu_grid_metrics["phase_threshold_volume_fraction_estimate"] \
        == pytest.approx(
            cpu_grid_metrics["phase_threshold_volume_fraction_estimate"])
    assert gpu_grid_metrics["mac_grid_enstrophy_estimate"] == pytest.approx(
        cpu_grid_metrics["mac_grid_enstrophy_estimate"], rel=2e-4, abs=2e-6)


def test_shading_attributes_age_source_speed():
    """Exported attributes: age grows with time, source ids distinguish
    seeding sources, speed matches |velocity|, all aligned to positions."""
    import numpy as np
    from stflip import Params, STFLIPSolver
    n = 16
    s = STFLIPSolver(Params(resolution=(n, n, n), dx=1.0 / n,
                            gravity=(0, 0, -9.81), frame_dt=1 / 24,
                            cfl_target=6.0, seed=0), "cpu")
    a = np.zeros((n, n, n), bool)
    a[:n // 3, :, :n // 2] = True
    b = np.zeros((n, n, n), bool)
    b[2 * n // 3:, :, :n // 2] = True
    s.add_liquid_mask(a)          # source 0
    s.add_liquid_mask(b)          # source 1
    for _ in range(3):
        s.step_frame()
    pos, vel, attrs = s.get_render_particles_ex()
    assert set(attrs) == {"age", "source", "speed"}
    assert len(pos) == len(vel) == len(attrs["age"]) == len(attrs["source"])
    assert len(attrs["speed"]) == len(pos)
    # two sources present
    assert set(np.unique(attrs["source"]).tolist()) == {0, 1}
    # age is positive (particles have existed for 3 frames)
    assert attrs["age"].min() > 0.0
    assert attrs["age"].max() <= 3.0 / 24.0 + 1e-4
    # speed equals |velocity|
    assert np.allclose(attrs["speed"], np.linalg.norm(vel, axis=1), atol=1e-5)


def test_shading_attributes_survive_outflow_and_reconcile():
    import numpy as np
    from stflip import Params, STFLIPSolver
    n = 16
    s = STFLIPSolver(Params(resolution=(n, n, n), dx=1.0 / n,
                            gravity=(0, 0, -9.81), frame_dt=1 / 24,
                            cfl_target=6.0, seed=1), "cpu")
    m = np.zeros((n, n, n), bool)
    m[:, :, n // 2:] = True
    s.add_liquid_mask(m)
    drain = np.zeros((n, n, n), bool)
    drain[:, :, :2] = True
    s.add_outflow(drain, mode="VOLUME")
    for _ in range(4):
        s.step_frame()
    # attrs stay length-consistent with the culled particle set
    assert s.age.shape[0] == s.pos.shape[0]
    assert s.source_id.shape[0] == s.pos.shape[0]
    pos, vel, attrs = s.get_render_particles_ex()
    assert len(attrs["age"]) == len(pos)
