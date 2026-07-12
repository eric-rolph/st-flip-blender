"""Sparse active-block production grid."""

import numpy as np

from stflip import Params, STFLIPSolver


def _run(sparse, n=48, frames=4, seed=0):
    p = Params(resolution=(n, n, n), dx=1.0 / n, gravity=(0, 0, -9.81),
               frame_dt=1 / 24, cfl_target=6.0, seed=seed, sparse=sparse,
               block_size=8)
    s = STFLIPSolver(p, "cpu")
    m = np.zeros((n, n, n), bool)
    m[2:14, 2:14, 20:40] = True
    s.add_liquid_mask(m)
    coms = []
    for _ in range(frames):
        s.step_frame()
        pos, _ = s.get_render_particles()
        coms.append(pos.mean(axis=0))
    return s, np.array(coms)


def test_sparse_matches_dense_exactly():
    """Because the window always contains the fluid plus the full velocity
    extrapolation band, the sparse solve is bitwise-identical to the dense one."""
    _sd, com_dense = _run(False)
    _ss, com_sparse = _run(True)
    assert np.abs(com_dense - com_sparse).max() == 0.0


def test_sparse_window_is_smaller_than_domain():
    s, _ = _run(True, n=48)
    assert s._grid_origin is not None
    sub = s._grids["c_m"].shape
    assert np.prod(sub) < 48 ** 3  # empty space is never allocated


def test_sparse_dam_break_stable():
    n = 40
    p = Params(resolution=(n, n, n), dx=1.0 / n, gravity=(0, 0, -9.81),
               frame_dt=1 / 24, cfl_target=8.0, seed=3, sparse=True)
    s = STFLIPSolver(p, "cpu")
    m = np.zeros((n, n, n), bool)
    m[:13, :, :20] = True
    s.add_liquid_mask(m)
    for _ in range(3):
        s.step_frame()
    pos, vel = s.get_render_particles()
    assert np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))
    assert pos[:, 0].max() > 0.5  # column collapsed and spread


def test_sparse_apic_two_phase_combo_runs():
    n = 24
    p = Params(resolution=(n, n, n), dx=1.0 / n, gravity=(0, 0, -9.81),
               frame_dt=1 / 24, cfl_target=4.0, seed=2, sparse=True,
               transfer="apic")
    s = STFLIPSolver(p, "cpu")
    m = np.zeros((n, n, n), bool)
    m[4:12, 4:12, 12:20] = True
    s.add_liquid_mask(m)
    for _ in range(3):
        s.step_frame()
    pos, vel = s.get_render_particles()
    assert np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))
