import numpy as np
import pytest

from stflip import FrameStats, Params, STFLIPSolver, cuda_diagnostics
from stflip.apertures import weighted_divergence
from stflip.metrics import estimate_mac_grid_metrics, measure_frame


def _dam_break(n=24, cfl=8.0, st=True, seed=0, ppc=8):
    p = Params(
        resolution=(n, n, n), dx=1.0 / n, gravity=(0.0, 0.0, -9.81),
        frame_dt=1.0 / 24.0, cfl_target=cfl, particles_per_cell=ppc,
        st_enabled=st, seed=seed,
    )
    s = STFLIPSolver(p, "cpu")
    mask = np.zeros((n, n, n), dtype=bool)
    mask[: n // 3, :, : n // 2] = True
    s.add_liquid_mask(mask)
    return s


def test_m0_calibration_close_to_ppc():
    s = _dam_break()
    # Normalised kernels: expected accumulator ~ particles-per-cell.
    assert 0.7 * 8 <= s.m0 <= 1.3 * 8


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
    # ceil(41 cells / cfl_local) requires 41 RK substeps. Each Ralston RK3
    # substep samples the velocity field three times.
    s.vel = s.be.xp.asarray([[41.0, 0.0, 0.0]], dtype=s.be.xp.float32)
    dt_act = s.be.xp.asarray([1.0], dtype=s.be.xp.float32)

    s._advect({}, s.pos.copy(), dt_act)

    assert s.sample_calls == 3 * 41


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


@pytest.mark.gpu
def test_gpu_backend_parity():
    cupy = pytest.importorskip("cupy")
    available, reason = cuda_diagnostics(force=True)
    assert available, reason

    s_cpu = _dam_break(n=16, seed=7)
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
    s_gpu.add_liquid_mask(mask)

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
        # A nonzero PCG iteration count proves the pressure projection ran.
        assert gpu_stats.pcg_iters and all(i > 0 for i in gpu_stats.pcg_iters)
    s_gpu.be.synchronize()

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
