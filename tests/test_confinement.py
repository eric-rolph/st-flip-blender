"""ENER-M1: vorticity confinement look control.

Confinement injects energy by design; these tests pin its safety envelope
(zero-strength is a bitwise no-op, a calm pool stays calm at slider max)
and its purpose (rotating-tank enstrophy strictly increases).
"""

import hashlib

import numpy as np
import pytest

from stflip import forces, metrics
from stflip.backend import get_backend
from stflip.solver import Params, STFLIPSolver
from stflip.velocity import SolidBodyRotation


def _state_hash(solver) -> str:
    digest = hashlib.sha256()
    for arr in (solver.pos, solver.vel, solver.dt_resid):
        host = solver.be.to_numpy(arr)
        digest.update(np.ascontiguousarray(host).tobytes())
    return digest.hexdigest()


def _params(n=12, **overrides):
    base = dict(
        resolution=(n, n, n), dx=1.0 / n, gravity=(0.0, 0.0, -9.81),
        frame_dt=1.0 / 24.0, cfl_target=8.0, particles_per_cell=2,
        st_enabled=True, seed=9,
    )
    base.update(overrides)
    return Params(**base)


def _dam_mask(n):
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[: n // 3, :, : n // 2] = True
    return mask


def _pool_mask(n):
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[:, :, : n // 2] = True
    return mask


def _run(params, setup, frames=2):
    solver = STFLIPSolver(params, get_backend("cpu"))
    setup(solver)
    for _ in range(frames):
        solver.step_frame()
    return solver


class TestConfinementAccel:
    def test_zero_strength_returns_none(self):
        n = 8
        grids = {
            "u": np.zeros((n + 1, n, n), dtype=np.float32),
            "v": np.zeros((n, n + 1, n), dtype=np.float32),
            "w": np.zeros((n, n, n + 1), dtype=np.float32),
            "c_phi": np.ones((n, n, n), dtype=np.float32),
        }
        assert forces.confinement_accel(np, 0.125, grids, 0.0) is None

    def test_zero_velocity_gives_zero_accel(self):
        n = 8
        grids = {
            "u": np.zeros((n + 1, n, n), dtype=np.float32),
            "v": np.zeros((n, n + 1, n), dtype=np.float32),
            "w": np.zeros((n, n, n + 1), dtype=np.float32),
            "c_phi": np.ones((n, n, n), dtype=np.float32),
        }
        accel = forces.confinement_accel(np, 0.125, grids, 10.0)
        assert accel.shape == (n, n, n, 3)
        assert float(np.abs(accel).max()) == 0.0

    def test_clamp_bounds_magnitude(self):
        n = 10
        rng = np.random.default_rng(4)
        grids = {
            "u": rng.normal(0.0, 50.0, (n + 1, n, n)).astype(np.float32),
            "v": rng.normal(0.0, 50.0, (n, n + 1, n)).astype(np.float32),
            "w": rng.normal(0.0, 50.0, (n, n, n + 1)).astype(np.float32),
            "c_phi": np.ones((n, n, n), dtype=np.float32),
        }
        accel = forces.confinement_accel(np, 0.1, grids, 1000.0)
        mag = np.sqrt((accel.astype(np.float64) ** 2).sum(axis=-1))
        assert float(mag.max()) <= 98.1 * (1.0 + 1e-5)

    def test_masked_to_liquid_cells(self):
        n = 10
        rng = np.random.default_rng(4)
        grids = {
            "u": rng.normal(0.0, 5.0, (n + 1, n, n)).astype(np.float32),
            "v": rng.normal(0.0, 5.0, (n, n + 1, n)).astype(np.float32),
            "w": rng.normal(0.0, 5.0, (n, n, n + 1)).astype(np.float32),
            "c_phi": np.zeros((n, n, n), dtype=np.float32),
        }
        grids["c_phi"][:, :, : n // 2] = 1.0
        accel = forces.confinement_accel(np, 0.1, grids, 10.0)
        assert float(np.abs(accel[:, :, n // 2 :]).max()) == 0.0


class TestSolverIntegration:
    def test_zero_strength_is_bitwise_noop(self):
        n = 12
        baseline = _run(_params(n), lambda s: s.add_liquid_mask(_dam_mask(n)))
        forced = _run(_params(n), lambda s: (
            s.add_liquid_mask(_dam_mask(n)),
            s.add_force("CONFINEMENT", 0.0),
        ))
        assert _state_hash(baseline) == _state_hash(forced)

    def test_still_pool_stays_calm_at_slider_max(self):
        n = 12
        solver = _run(_params(n), lambda s: (
            s.add_liquid_mask(_pool_mask(n)),
            s.add_force("CONFINEMENT", 50.0),
        ), frames=3)
        speeds = np.linalg.norm(solver.be.to_numpy(solver.vel), axis=1)
        assert np.all(np.isfinite(speeds))
        # A settled pool has near-zero vorticity, so even a maxed slider
        # must not manufacture bulk motion from projection noise.
        assert float(speeds.max()) < 0.5

    def test_rotating_tank_enstrophy_increases(self):
        n = 12
        field = SolidBodyRotation(
            center=(0.5, 0.5, 0.5), angular_velocity=(0.0, 0.0, 3.0))

        def setup(strength):
            def inner(s):
                mask = np.ones((n,) * 3, dtype=bool)
                s.add_liquid_mask(mask, velocity=field)
                if strength:
                    s.add_force("CONFINEMENT", strength)
            return inner

        enstrophy = {}
        for strength in (0.0, 8.0):
            solver = _run(
                _params(n, gravity=(0.0, 0.0, 0.0)), setup(strength),
                frames=2)
            grids = solver._p2g(solver._dt_prev)
            host = {
                name: solver.be.to_numpy(grids[name])
                for name in ("u", "v", "w", "c_phi")
            }
            estimate = metrics.estimate_mac_grid_metrics(
                host, solver.p.dx)
            enstrophy[strength] = estimate["mac_grid_enstrophy_estimate"]
        assert enstrophy[8.0] > enstrophy[0.0]

    def test_add_force_rejects_unknown_type(self):
        solver = STFLIPSolver(_params(), get_backend("cpu"))
        with pytest.raises(ValueError):
            solver.add_force("SWIRLINESS", 1.0)
