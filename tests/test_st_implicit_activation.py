"""CAP-M2: semi-implicit capillary stabilizer activation."""

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


def _sphere_mask(n, centre, radius):
    cells = (np.stack(np.meshgrid(
        *(np.arange(n),) * 3, indexing="ij"), axis=-1) + 0.5) / n
    return np.linalg.norm(cells - np.asarray(centre), axis=-1) <= radius


def _params(n=16, **overrides):
    base = dict(
        resolution=(n, n, n), dx=1.0 / n, gravity=(0.0, 0.0, 0.0),
        frame_dt=1.0 / 24.0, cfl_target=8.0, particles_per_cell=2,
        st_enabled=True, seed=12,
    )
    base.update(overrides)
    return Params(**base)


def _run(params, mask, velocity=(0.0, 0.0, 0.0), frames=4):
    solver = STFLIPSolver(params, get_backend("cpu"))
    solver.add_liquid_mask(mask, velocity=velocity)
    stats = None
    for _ in range(frames):
        stats = solver.step_frame()
    return solver, stats


class TestGating:
    def test_rejects_non_bool(self):
        with pytest.raises(TypeError):
            Params(resolution=(8, 8, 8), dx=0.125, st_implicit="on")

    def test_active_stabilizer_changes_the_result(self):
        n = 16
        mask = _sphere_mask(n, (0.5, 0.5, 0.5), 0.2)
        plain, _ = _run(
            _params(n, surface_tension=100.0, st_clamp_scale=8.0), mask)
        stabilized, _ = _run(
            _params(n, surface_tension=100.0, st_clamp_scale=8.0,
                    st_implicit=True), mask)
        assert _state_hash(plain) != _state_hash(stabilized)

    def test_counters_populated_only_when_active(self):
        n = 16
        mask = _sphere_mask(n, (0.5, 0.5, 0.5), 0.2)
        _, stats_off = _run(
            _params(n, surface_tension=100.0), mask, frames=1)
        assert stats_off.st_cg_iters == []
        _, stats_on = _run(
            _params(n, surface_tension=100.0, st_implicit=True), mask,
            frames=1)
        assert len(stats_on.st_cg_iters) == 3 * stats_on.steps
        assert all(rel <= 1e-4 or rel == 0.0
                   for rel in stats_on.st_cg_rel_residuals)


class TestPhysics:
    def test_translating_droplet_no_stabilizer_drag(self):
        # The review's key gate: energy tests cannot catch drag (drag
        # reduces energy and "passes").  The failure mode the validity
        # gate guards is invalid air-side faces pulling the droplet BELOW
        # its translation speed.  Measured context: at scale 8 BOTH modes
        # share a small positive drift from the relaxed clamp's documented
        # accuracy trade (explicit at scale 1 holds 1.001), so the gate is
        # a hard no-slowdown floor plus a bounded upper drift.
        n = 16
        params = _params(
            n, surface_tension=100.0, st_clamp_scale=8.0,
            st_implicit=True)
        solver, _ = _run(
            params, _sphere_mask(n, (0.35, 0.5, 0.5), 0.2),
            velocity=(1.0, 0.0, 0.0), frames=6)
        vel = solver.be.to_numpy(solver.vel)
        com_vx = float(vel[:, 0].mean())
        assert com_vx >= 0.99
        assert com_vx <= 1.05

    def test_stabilizer_prevents_the_explicit_blowup(self):
        # The headline: at scale 8 (dt ~ 3x the Brackbill limit) the
        # explicit feedback grows; the stabilizer must keep peak speeds
        # bounded and strictly below the un-stabilized run's.
        n = 16
        mask = _sphere_mask(n, (0.35, 0.5, 0.5), 0.2)
        explicit, _ = _run(
            _params(n, surface_tension=100.0, st_clamp_scale=8.0),
            mask, velocity=(1.0, 0.0, 0.0), frames=6)
        implicit, _ = _run(
            _params(n, surface_tension=100.0, st_clamp_scale=8.0,
                    st_implicit=True),
            mask, velocity=(1.0, 0.0, 0.0), frames=6)
        vmax_explicit = float(np.linalg.norm(
            explicit.be.to_numpy(explicit.vel), axis=1).max())
        vmax_implicit = float(np.linalg.norm(
            implicit.be.to_numpy(implicit.vel), axis=1).max())
        assert vmax_implicit < 3.0
        assert vmax_implicit < vmax_explicit

    def test_clamp_relaxed_droplet_stays_bounded(self):
        # dt runs ~3x above the Brackbill limit at scale 8; the stabilizer
        # must keep the interface finite with bounded speeds instead of
        # the explicit chatter blow-up.
        n = 16
        params = _params(
            n, surface_tension=100.0, st_clamp_scale=8.0,
            st_implicit=True)
        solver, stats = _run(
            params, _sphere_mask(n, (0.5, 0.5, 0.5), 0.22), frames=6)
        vel = solver.be.to_numpy(solver.vel)
        pos = solver.be.to_numpy(solver.pos)
        assert np.all(np.isfinite(vel))
        assert np.all(np.isfinite(pos))
        speeds = np.linalg.norm(vel, axis=1)
        assert float(speeds.max()) < 3.0
        assert stats.n_particles == solver.pos.shape[0]
