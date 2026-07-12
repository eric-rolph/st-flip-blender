"""Particle sheeting / anti-clumping (position-only redistribution)."""

import numpy as np

from stflip import Params, STFLIPSolver


def _solver(n=24, sheeting=0.0, gravity=(0, 0, -9.81), seed=0):
    return STFLIPSolver(
        Params(resolution=(n, n, n), dx=1.0 / n, gravity=gravity,
               frame_dt=1 / 24, cfl_target=8.0, seed=seed,
               sheeting=sheeting), "cpu")


def test_sheeting_spreads_overdense_interior_clump():
    n = 24
    dx = 1.0 / n
    s = _solver(n=n, sheeting=0.6, gravity=(0, 0, 0))
    base = np.zeros((n, n, n), bool)
    base[6:18, 6:18, 6:18] = True
    s.add_liquid_mask(base)
    rng = np.random.default_rng(0)
    extra = (np.array([11, 11, 11]) + rng.random((2000, 3))).astype(np.float32) * dx
    s.pos = s.be.from_numpy(np.vstack([s.be.to_numpy(s.pos), extra]))
    s.vel = s.be.from_numpy(np.zeros((s.pos.shape[0], 3), np.float32))
    peak0 = int(s.be.to_numpy(s._cell_counts()).max())
    for _ in range(12):
        s._apply_sheeting()
    peak1 = int(s.be.to_numpy(s._cell_counts()).max())
    assert peak1 < peak0 * 0.25          # the clump is dramatically relieved
    assert np.all(np.isfinite(s.be.to_numpy(s.pos)))


def test_sheeting_does_not_inflate_free_surface():
    """Redistribution must not push surface particles down the density cliff
    into the air; the surface height stays put."""
    def top(sheeting):
        n = 24
        s = _solver(n=n, sheeting=sheeting, seed=1)
        m = np.zeros((n, n, n), bool)
        m[:, :, : n // 2] = True
        s.add_liquid_mask(m)
        for _ in range(8):
            s.step_frame()
        return s.be.to_numpy(s.pos)[:, 2].max()
    assert abs(top(0.5) - top(0.0)) < 0.02


def test_zero_sheeting_is_a_noop():
    def run():
        n = 20
        s = _solver(n=n, sheeting=0.0, seed=2)
        m = np.zeros((n, n, n), bool)
        m[:n // 3, :, : n // 2] = True
        s.add_liquid_mask(m)
        for _ in range(4):
            s.step_frame()
        return s.be.to_numpy(s.pos)
    assert np.array_equal(run(), run())


def test_sheeting_stable_on_dam_break():
    n = 24
    s = _solver(n=n, sheeting=0.5, seed=3)
    m = np.zeros((n, n, n), bool)
    m[:n // 3, :, : n // 2] = True
    n0 = s.add_liquid_mask(m)
    for _ in range(5):
        s.step_frame()
    pos = s.be.to_numpy(s.pos)
    vel = s.be.to_numpy(s.vel)
    assert s.pos.shape[0] == n0                # position-only, no particle loss
    assert np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))
    assert np.linalg.norm(vel, axis=1).max() < 8.0
