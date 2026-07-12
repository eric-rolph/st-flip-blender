"""Art-directable body forces (directional / vortex / turbulence)."""

import numpy as np
import pytest

from stflip import Params, STFLIPSolver
from stflip import forces as F


def test_directional_is_uniform_and_normalized():
    a = F.directional_accel(np, (8, 8, 8), 0.1, (3.0, 0.0, 0.0), 2.0)
    assert np.allclose(a[..., 0], 2.0)          # magnitude = strength
    assert np.allclose(a[..., 1], 0.0)
    assert F.directional_accel(np, (4, 4, 4), 0.1, (0, 0, 0), 1.0) is None


def test_vortex_swirls_about_axis():
    n = 20
    dx = 1.0 / n
    a = F.vortex_accel(np, (n, n, n), dx, (0.5, 0.5, 0.5), (0, 0, 1),
                       5.0, 0.4, np.zeros(3))
    ax, ay, _ = np.meshgrid((np.arange(n) + 0.5) * dx,
                            (np.arange(n) + 0.5) * dx,
                            (np.arange(n) + 0.5) * dx, indexing="ij")
    rx, ry = ax - 0.5, ay - 0.5
    tangential = a[..., 0] * (-ry) + a[..., 1] * rx
    assert (tangential >= -1e-6).mean() > 0.95   # consistent swirl sense
    assert np.abs(a[..., 2]).max() < 1e-5        # no acceleration along axis


def test_turbulence_is_divergence_free_and_reproducible():
    n = 24
    dx = 1.0 / n
    a = F.turbulence_accel(np, (n, n, n), dx, 2.0, 0.5, 0.0, 7, np.zeros(3))
    div = (np.gradient(a[..., 0], dx, axis=0)
           + np.gradient(a[..., 1], dx, axis=1)
           + np.gradient(a[..., 2], dx, axis=2))
    a_rms = np.sqrt((a * a).mean())
    div_rms = np.sqrt((div * div).mean())
    assert div_rms < 0.05 * a_rms                # curl noise -> ~zero divergence
    assert np.all(np.isfinite(a))
    b = F.turbulence_accel(np, (n, n, n), dx, 2.0, 0.5, 0.0, 7, np.zeros(3))
    assert np.array_equal(a, b)                  # same seed -> identical
    c = F.turbulence_accel(np, (n, n, n), dx, 2.0, 0.5, 0.0, 8, np.zeros(3))
    assert not np.array_equal(a, c)              # different seed differs


def _pool(force=None, n=24):
    s = STFLIPSolver(Params(resolution=(n, n, n), dx=1.0 / n,
                            gravity=(0, 0, 0), frame_dt=1 / 24,
                            cfl_target=6.0, seed=0), "cpu")
    m = np.zeros((n, n, n), bool)
    m[:, :, : n // 2] = True
    s.add_liquid_mask(m)
    if force is not None:
        s.add_force(**force)
    for _ in range(3):
        s.step_frame()
    return s


def test_force_induces_motion_and_zero_force_is_noop():
    still = _pool(None)
    turbo = _pool({"force_type": "TURBULENCE", "strength": 6.0, "scale": 0.4,
                   "seed": 3})
    v_still = np.linalg.norm(still.be.to_numpy(still.vel), axis=1).max()
    v_turbo = np.linalg.norm(turbo.be.to_numpy(turbo.vel), axis=1).max()
    assert v_still < 1e-3                         # gravity-free pool stays still
    assert v_turbo > 0.2                          # turbulence stirs it
    assert np.all(np.isfinite(turbo.be.to_numpy(turbo.vel)))


def test_add_force_validates_type():
    s = STFLIPSolver(Params(resolution=(8, 8, 8), dx=1.0 / 8), "cpu")
    with pytest.raises(ValueError):
        s.add_force("MAGNET", 1.0)
