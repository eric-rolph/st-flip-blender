"""Animated moving-wall boundaries."""

import numpy as np

from stflip import Params, STFLIPSolver


def _piston_scene(n=24, vpiston=2.0, gravity=(0, 0, 0), seed=2):
    dx = 1.0 / n
    p = Params(resolution=(n, n, n), dx=dx, gravity=gravity, frame_dt=1 / 24,
               cfl_target=3.0, seed=seed)
    s = STFLIPSolver(p, "cpu")
    m = np.zeros((n, n, n), bool)
    m[:, :, 6:14] = True
    s.add_liquid_mask(m)
    sdf = np.full((n, n, n), 1e9, np.float32)
    sv = np.zeros((n, n, n, 3), np.float32)
    for k in range(n):
        z = (k + 0.5) * dx
        if z < 6 * dx:  # piston top flush with the liquid base
            sdf[:, :, k] = z - 6 * dx
            sv[:, :, k, 2] = vpiston
    s.set_solid_sdf(sdf, solid_vel=sv)
    return s


def test_moving_piston_lifts_fluid():
    s = _piston_scene()
    z0 = s.be.to_numpy(s.pos)[:, 2].mean()
    for _ in range(4):
        s.step_frame()
    pos = s.be.to_numpy(s.pos)
    vel = s.be.to_numpy(s.vel)
    assert np.all(np.isfinite(pos))
    assert pos[:, 2].mean() > z0 + 0.02, "piston should raise the fluid"
    assert vel[:, 2].mean() > 0.0, "fluid should gain upward velocity"


def test_static_solid_velocity_is_noop():
    """A zero solid-velocity field must reproduce the plain-static result."""
    def run(with_vel):
        n = 24
        dx = 1.0 / n
        p = Params(resolution=(n, n, n), dx=dx, gravity=(0, 0, -9.81),
                   frame_dt=1 / 24, cfl_target=4.0, seed=1)
        s = STFLIPSolver(p, "cpu")
        m = np.zeros((n, n, n), bool)
        m[:n // 2, :, 6:16] = True
        s.add_liquid_mask(m)
        sdf = np.full((n, n, n), 1e9, np.float32)
        for k in range(6):
            sdf[:, :, k] = (k + 0.5) * dx - 6 * dx
        if with_vel:
            s.set_solid_sdf(sdf, solid_vel=np.zeros((n, n, n, 3), np.float32))
        else:
            s.set_solid_sdf(sdf)
        for _ in range(4):
            s.step_frame()
        return s.be.to_numpy(s.pos)
    a = run(False)
    b = run(True)
    assert np.allclose(a, b, atol=1e-5)


def test_retreating_wall_does_not_drag_fluid():
    """A wall moving away from resting fluid must not suck it inward: only the
    penetrating (inward) relative normal velocity may be removed."""
    n = 16
    dx = 1.0 / n
    p = Params(resolution=(n, n, n), dx=dx, gravity=(0, 0, 0), frame_dt=1 / 24,
               cfl_target=3.0, seed=0)
    s = STFLIPSolver(p, "cpu")
    sdf = np.full((n, n, n), 1e9, np.float32)
    sv = np.zeros((n, n, n, 3), np.float32)
    for k in range(n):
        z = (k + 0.5) * dx
        if z < 6 * dx:  # floor retreating downward, away from the fluid above
            sdf[:, :, k] = z - 6 * dx
            sv[:, :, k, 2] = -1.0
    s.set_solid_sdf(sdf, solid_vel=sv)
    s.pos = s.be.from_numpy(np.array([[0.5, 0.5, 6.2 * dx]], dtype=np.float32))
    s.vel = s.be.from_numpy(np.zeros((1, 3), dtype=np.float32))
    s._enforce_solid_velocity()
    vz = float(s.be.to_numpy(s.vel)[0, 2])
    assert vz > -0.1, f"retreating wall dragged the fluid: vz={vz:.3f}"


def test_moving_wall_no_tunneling():
    """Fast piston must not let particles end up deep inside the solid."""
    s = _piston_scene(vpiston=4.0)
    for _ in range(4):
        s.step_frame()
    d = s.be.to_numpy(s._sample_cells(s.sdf, s.pos))
    # essentially no particles more than a cell inside the piston
    assert (d < -1.5 / 24).mean() < 0.02
