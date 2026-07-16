"""PERF-M1: per-particle local speed bound for sub-stepped advection."""

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
        resolution=(n, n, n), gravity=(0.0, 0.0, -9.81), dx=1.0 / n,
        frame_dt=1.0 / 24.0, cfl_target=8.0, particles_per_cell=2,
        st_enabled=True, seed=7,
    )
    base.update(overrides)
    return Params(**base)


def _pool_mask(n):
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[:, :, : n // 2] = True
    return mask


def _dam_mask(n):
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[: n // 3, :, : n // 2] = True
    return mask


def _run(params, mask, frames=3):
    solver = STFLIPSolver(params, get_backend("cpu"))
    solver.add_liquid_mask(mask)
    for _ in range(frames):
        solver.step_frame()
    return solver


def _random_grids(xp, n, rng, spike=None):
    grids = {
        "u": xp.asarray(
            rng.normal(0.0, 1.0, (n + 1, n, n)).astype(np.float32)),
        "v": xp.asarray(
            rng.normal(0.0, 1.0, (n, n + 1, n)).astype(np.float32)),
        "w": xp.asarray(
            rng.normal(0.0, 1.0, (n, n, n + 1)).astype(np.float32)),
    }
    if spike is not None:
        i, j, k, value = spike
        grids["u"][i, j, k] = value
    return grids


class TestGating:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            Params(resolution=(8, 8, 8), dx=0.125, advection_bound="turbo")

    def test_default_is_bitwise_global(self):
        n = 12
        default = _run(_params(n), _dam_mask(n))
        explicit = _run(
            _params(n, advection_bound="global"), _dam_mask(n))
        assert _state_hash(default) == _state_hash(explicit)

    def test_local_changes_the_result(self):
        n = 12
        plain = _run(_params(n), _dam_mask(n))
        local = _run(_params(n, advection_bound="local"), _dam_mask(n))
        assert _state_hash(plain) != _state_hash(local)


class TestBoundLemma:
    def test_bound_dominates_reachable_samples(self):
        # G-L: for random spiky fields, the cell bound dominates the
        # sampled speed at every point within cfl_local * dx of any
        # point in the cell (the reach of one RK3 sub-step, including
        # its stages).
        n = 12
        solver = STFLIPSolver(
            _params(n, advection_bound="local"), get_backend("cpu"))
        xp = solver.be.xp
        rng = np.random.default_rng(3)
        grids = _random_grids(xp, n, rng, spike=(5, 6, 7, 40.0))
        bound = solver.be.to_numpy(
            solver._advection_speed_bound_grid(grids))
        dx = solver.p.dx
        reach = solver.p.cfl_local * dx
        starts = rng.uniform(0.0, 1.0, (512, 3)).astype(np.float32)
        offsets = rng.uniform(-reach, reach, (512, 3)).astype(np.float32)
        queries = np.clip(starts + offsets, 0.0, np.nextafter(1.0, 0.0))
        sampled = solver.be.to_numpy(
            solver._sample_faces(grids, xp.asarray(queries)))
        speeds = np.linalg.norm(sampled.astype(np.float64), axis=1)
        cells = np.clip((starts / dx).astype(np.int64), 0, n - 1)
        limits = bound[cells[:, 0], cells[:, 1], cells[:, 2]]
        assert np.all(speeds <= limits * (1.0 + 1e-5))

    def test_uniform_field_matches_global(self):
        # A uniform field makes the local and global bounds equal; RK3
        # is exact on a constant field, so both modes land on the same
        # endpoint up to float32 accumulation order (the sub-step
        # PARTITIONS differ: equal splits vs cap-sized steps + tail).
        n = 12
        solver_a = STFLIPSolver(_params(n), get_backend("cpu"))
        solver_b = STFLIPSolver(
            _params(n, advection_bound="local"), get_backend("cpu"))
        xp = solver_a.be.xp
        grids = {
            "u": xp.full((n + 1, n, n), 0.8, dtype=xp.float32),
            "v": xp.full((n, n + 1, n), -0.3, dtype=xp.float32),
            "w": xp.full((n, n, n + 1), 0.5, dtype=xp.float32),
        }
        rng = np.random.default_rng(11)
        pos = xp.asarray(
            rng.uniform(0.2, 0.8, (256, 3)).astype(np.float32))
        dt_act = xp.asarray(
            rng.uniform(0.0, 0.1, (256,)).astype(np.float32))
        out_a = solver_a.be.to_numpy(
            solver_a._advect(grids, pos.copy(), dt_act))
        out_b = solver_b.be.to_numpy(
            solver_b._advect(grids, pos.copy(), dt_act))
        np.testing.assert_allclose(out_a, out_b, rtol=1e-5, atol=1e-6)

    def test_backward_advection_displacement_bounded(self):
        # The un-jitter path passes negative per-particle dt_act; the
        # total travel bound |dt| * vmax must survive the sign flip.
        n = 12
        solver = STFLIPSolver(
            _params(n, advection_bound="local"), get_backend("cpu"))
        xp = solver.be.xp
        rng = np.random.default_rng(5)
        grids = _random_grids(xp, n, rng, spike=(3, 3, 3, 20.0))
        pos = xp.asarray(
            rng.uniform(0.1, 0.9, (512, 3)).astype(np.float32))
        dt_act = xp.asarray(
            rng.uniform(-0.02, 0.02, (512,)).astype(np.float32))
        out = solver.be.to_numpy(solver._advect(grids, pos.copy(), dt_act))
        vmax = float(solver.be.to_numpy(
            solver._grid_velocity_bound(grids)))
        travel = np.linalg.norm(
            out - solver.be.to_numpy(pos), axis=1)
        limit = np.abs(solver.be.to_numpy(dt_act)) * vmax
        assert np.all(travel <= limit + 1e-6)


class TestPhysics:
    def test_still_pool_stays_still(self):
        # G-H: a hydrostatic pool must stay calm under the local bound
        # (a still field has bound ~ 0 everywhere, one sub-step).
        n = 12
        solver = _run(
            _params(n, advection_bound="local"), _pool_mask(n), frames=3)
        speeds = np.linalg.norm(solver.be.to_numpy(solver.vel), axis=1)
        assert float(speeds.max()) < 0.5

    def test_dam_break_stays_close_to_global(self):
        # Smoke-level accuracy: the local mode is a different integrator,
        # not a different fluid.  Bulk statistics stay close over a
        # short chaotic run.
        n = 12
        plain = _run(_params(n), _dam_mask(n), frames=3)
        local = _run(_params(n, advection_bound="local"), _dam_mask(n),
                     frames=3)
        pos_a = plain.be.to_numpy(plain.pos)
        pos_b = local.be.to_numpy(local.pos)
        assert pos_a.shape == pos_b.shape
        centroid_gap = np.linalg.norm(
            pos_a.mean(axis=0) - pos_b.mean(axis=0))
        assert centroid_gap < 0.05
        front_gap = abs(
            float(pos_a[:, 0].max()) - float(pos_b[:, 0].max()))
        assert front_gap < 0.15

    def test_reflection_composes(self):
        n = 12
        solver = _run(
            _params(n, advection_bound="local", reflection=True),
            _dam_mask(n), frames=3)
        vel = solver.be.to_numpy(solver.vel)
        assert np.all(np.isfinite(vel))
        assert float(np.abs(vel).max()) > 0.5

    def test_volume_outflow_still_removes(self):
        n = 12
        params = _params(n, advection_bound="local")
        solver = STFLIPSolver(params, get_backend("cpu"))
        solver.add_liquid_mask(_dam_mask(n))
        sink = np.zeros((n,) * 3, dtype=bool)
        # Directly in the collapsing dam's path so removal provably
        # happens within a few frames at this scale.
        sink[n // 3: n // 3 + 2, :, :2] = True
        solver.add_outflow(sink, mode="VOLUME")
        before = solver.pos.shape[0]
        for _ in range(4):
            solver.step_frame()
        removed = solver._volume_outflow_removed_total
        assert removed > 0
        assert solver.pos.shape[0] == before - removed
        assert solver.dt_resid.shape[0] == solver.pos.shape[0]
        assert solver.particle_id.shape[0] == solver.pos.shape[0]


class TestZeroDt:
    def test_zero_dt_is_not_aliased_and_still_relaxes(self):
        # All-zero dt_act: the local branch must still apply the single
        # relaxation+clamp pass the global branch would and must not
        # hand the caller's array back aliased (review finding).
        n = 12
        solver = STFLIPSolver(
            _params(n, advection_bound="local"), get_backend("cpu"))
        xp = solver.be.xp
        grids = {
            "u": xp.zeros((n + 1, n, n), dtype=xp.float32),
            "v": xp.zeros((n, n + 1, n), dtype=xp.float32),
            "w": xp.zeros((n, n, n + 1), dtype=xp.float32),
        }
        rng = np.random.default_rng(2)
        pos = xp.asarray(
            rng.uniform(0.2, 0.8, (64, 3)).astype(np.float32))
        dt_act = xp.zeros((64,), dtype=xp.float32)
        out = solver._advect(grids, pos, dt_act)
        assert out is not pos
        np.testing.assert_allclose(
            solver.be.to_numpy(out), solver.be.to_numpy(pos))


class TestResume:
    def test_checkpoint_resume_bit_identical(self):
        n = 12
        params = _params(n, advection_bound="local")
        straight = _run(params, _dam_mask(n), frames=4)

        first = STFLIPSolver(params, get_backend("cpu"))
        first.add_liquid_mask(_dam_mask(n))
        for _ in range(2):
            first.step_frame()
        state = first.checkpoint_state()

        resumed = STFLIPSolver(params, get_backend("cpu"))
        resumed.restore_state(state)
        for _ in range(2):
            resumed.step_frame()
        assert _state_hash(resumed) == _state_hash(straight)
