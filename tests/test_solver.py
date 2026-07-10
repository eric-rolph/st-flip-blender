import numpy as np
import pytest

from stflip import Params, STFLIPSolver


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


def test_plain_flip_mode_runs():
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

    from stflip.solver import FrameStats
    s._step(dt, FrameStats())
    speed = float(np.linalg.norm(s.be.to_numpy(s.vel)[-1]))
    assert speed < 1.3, f"isolated particle gained energy: |v| = {speed:.3f}"


@pytest.mark.gpu
def test_gpu_backend_parity():
    cupy = pytest.importorskip("cupy")
    assert cupy.cuda.runtime.getDeviceCount() > 0
    s_cpu = _dam_break(n=16, seed=7)
    s_gpu = STFLIPSolver(s_cpu.p, "cuda")
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[:5, :, :8] = True
    s_gpu.add_liquid_mask(mask)
    for _ in range(2):
        s_cpu.step_frame()
        s_gpu.step_frame()
    pos_g = s_gpu.be.to_numpy(s_gpu.pos)
    assert np.all(np.isfinite(pos_g))
    assert pos_g.shape == s_cpu.be.to_numpy(s_cpu.pos).shape
    # Chaotic dynamics preclude bitwise parity; compare bulk statistics.
    com_c = s_cpu.be.to_numpy(s_cpu.pos).mean(axis=0)
    com_g = pos_g.mean(axis=0)
    assert np.linalg.norm(com_c - com_g) < 0.1
