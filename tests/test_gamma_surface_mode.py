"""CALM-M3: surface-scoped jitter attenuation (gamma_mode)."""

import hashlib

import numpy as np
import pytest

from stflip.backend import get_backend
from stflip.solver import Params, STFLIPSolver


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
        st_enabled=True, seed=8,
    )
    base.update(overrides)
    return Params(**base)


def _pool_mask(n):
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[:, :, : n // 2] = True
    return mask


def _run(params, setup, frames=3):
    solver = STFLIPSolver(params, get_backend("cpu"))
    setup(solver)
    for _ in range(frames):
        solver.step_frame()
    return solver


class TestParams:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            Params(resolution=(8, 8, 8), dx=0.125, gamma_mode="vorticity")


class TestSpeedModeUnchanged:
    def test_speed_mode_is_bitwise_default(self):
        n = 12
        default = _run(_params(n), lambda s: s.add_liquid_mask(
            _pool_mask(n)))
        explicit = _run(_params(n, gamma_mode="speed"),
                        lambda s: s.add_liquid_mask(_pool_mask(n)))
        assert _state_hash(default) == _state_hash(explicit)


class TestMechanism:
    def _deep_resid_ratio(self, solver, z_lo, z_hi):
        pos = solver.be.to_numpy(solver.pos)
        resid = np.abs(solver.be.to_numpy(solver.dt_resid))
        deep = (pos[:, 2] > z_lo) & (pos[:, 2] < z_hi)
        assert deep.any()
        dt = float(solver._dt_prev)
        return float(resid[deep].mean()) / dt

    def test_deep_bulk_jitter_restored(self):
        # The acceptance criterion from the roadmap: under surface mode,
        # deep-bulk residuals return to the U(-dt/2, dt/2) stationary mean
        # |resid| ~ dt/4; under the speed gate a calm pool sits near zero.
        n = 12
        surface = _run(
            _params(n, gamma_mode="surface"),
            lambda s: s.add_liquid_mask(_pool_mask(n)), frames=3)
        speed = _run(
            _params(n),
            lambda s: s.add_liquid_mask(_pool_mask(n)), frames=3)
        deep_surface = self._deep_resid_ratio(surface, 0.1, 0.35)
        deep_speed = self._deep_resid_ratio(speed, 0.1, 0.35)
        assert 0.15 <= deep_surface <= 0.35
        assert deep_speed < 0.05

    def test_near_surface_particles_stay_damped(self):
        # The interface band must keep its attenuation: interiorness only
        # fires where c_phi is deep-liquid.
        n = 12
        surface = _run(
            _params(n, gamma_mode="surface"),
            lambda s: s.add_liquid_mask(_pool_mask(n)), frames=3)
        near = self._deep_resid_ratio(surface, 0.44, 0.5)
        deep = self._deep_resid_ratio(surface, 0.1, 0.35)
        assert near < 0.6 * deep

    def test_solid_masked_cells_read_interior(self):
        # A solid block occupying the pool floor depresses c_phi around
        # it; the solid mask must hand those cells interiorness so
        # block-adjacent calm bulk gets full jitter like deep bulk.
        n = 12

        def setup(s):
            s.add_liquid_mask(_pool_mask(n))
            cells = (np.stack(np.meshgrid(
                *(np.arange(n),) * 3, indexing="ij"), axis=-1) + 0.5) / n
            sdf = (cells[..., 2] - 0.15).astype(np.float32)
            s.set_solid_sdf(sdf)  # solid below z = 0.15

        surface = _run(
            _params(n, gamma_mode="surface"), setup, frames=3)
        above_block = self._deep_resid_ratio(surface, 0.17, 0.3)
        assert 0.12 <= above_block <= 0.38


class TestNormComposition:
    def test_m0_plateau_holds_in_surface_mode(self):
        # Full force-free box: interiorness makes gamma = jitter_strength
        # everywhere; the persisted-gamma divisor must keep the deposited
        # mean at exactly m0 = ppc.
        n = 12
        params = _params(
            n, gravity=(0.0, 0.0, 0.0), gamma_mode="surface",
            adaptive_gamma=True, particles_per_cell=8, cfl_target=4.0)
        solver = _run(
            params,
            lambda s: s.add_liquid_mask(np.ones((n,) * 3, dtype=bool)),
            frames=2)
        grids = solver._p2g(solver._dt_prev)
        interior = solver.be.to_numpy(
            grids["c_m"])[2:-2, 2:-2, 2:-2] / solver.m0
        assert float(interior.mean()) == pytest.approx(1.0, rel=0.01)

    def test_checkpoint_resume_bit_identical(self):
        # gamma_prev rides the reserved v3 member; resuming a surface-mode
        # exact-normalization bake must reproduce the uninterrupted run.
        n = 12
        params = _params(n, gamma_mode="surface")
        straight = _run(
            params, lambda s: s.add_liquid_mask(_pool_mask(n)), frames=4)

        first = STFLIPSolver(params, get_backend("cpu"))
        first.add_liquid_mask(_pool_mask(n))
        for _ in range(2):
            first.step_frame()
        state = first.checkpoint_state()
        assert "gamma_prev" in state

        resumed = STFLIPSolver(params, get_backend("cpu"))
        resumed.restore_state(state)
        for _ in range(2):
            resumed.step_frame()
        assert _state_hash(resumed) == _state_hash(straight)

    def test_gamma_prev_tracks_population_under_outflow(self):
        n = 12

        def setup(s):
            s.add_liquid_mask(_pool_mask(n))
            sink = np.zeros((n,) * 3, dtype=bool)
            sink[:, :, :1] = True
            s.add_outflow(sink, mode="VOLUME")

        solver = _run(
            _params(n, gamma_mode="surface"), setup, frames=3)
        assert solver._gamma_prev is not None
        assert solver._gamma_prev.shape[0] == solver.pos.shape[0]
