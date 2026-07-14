"""SAMP-M3: temporal_sampling wiring (pseudo | sobol_owen | cp_rot)."""

import hashlib

import numpy as np
import pytest

from stflip import sampling
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
        st_enabled=True, seed=4,
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


def _run(params, mask_fn, frames=3):
    solver = STFLIPSolver(params, get_backend("cpu"))
    solver.add_liquid_mask(mask_fn(params.resolution[0]))
    for _ in range(frames):
        solver.step_frame()
    return solver


class TestParams:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            Params(resolution=(8, 8, 8), dx=0.125,
                   temporal_sampling="halton")

    def test_default_is_pseudo(self):
        assert Params(resolution=(8, 8, 8),
                      dx=0.125).temporal_sampling == "pseudo"


class TestModes:
    def test_sobol_is_deterministic_and_differs_from_pseudo(self):
        n = 12
        sobol_a = _run(
            _params(n, temporal_sampling="sobol_owen"), _dam_mask)
        sobol_b = _run(
            _params(n, temporal_sampling="sobol_owen"), _dam_mask)
        pseudo = _run(_params(n, temporal_sampling="pseudo"), _dam_mask)
        assert _state_hash(sobol_a) == _state_hash(sobol_b)
        assert _state_hash(sobol_a) != _state_hash(pseudo)

    @pytest.mark.parametrize("mode", ["sobol_owen", "cp_rot"])
    def test_drift_bound_holds(self, mode):
        n = 12
        solver = STFLIPSolver(
            _params(n, temporal_sampling=mode), get_backend("cpu"))
        solver.add_liquid_mask(_dam_mask(n))
        dt_max = 0.0
        for _ in range(4):
            stats = solver.step_frame()
            dt_max = max(dt_max, *(stats.dt_values or (0.0,)))
            resid = solver.be.to_numpy(solver.dt_resid)
            assert np.all(np.isfinite(resid))
            assert float(np.abs(resid).max()) <= 0.5 * dt_max + 1e-9

    def test_still_pool_stays_calm_under_sobol(self):
        n = 12
        solver = _run(
            _params(n, temporal_sampling="sobol_owen"), _pool_mask,
            frames=3)
        speeds = np.linalg.norm(solver.be.to_numpy(solver.vel), axis=1)
        assert float(speeds.max()) < 0.5

    def test_sobol_mode_does_not_consume_the_rng(self):
        # The stateless sampler must leave self._rng exactly where a run
        # that never draws temporal jitter leaves it, so whitewater and
        # seeding streams stay aligned.
        n = 12
        sobol = _run(
            _params(n, temporal_sampling="sobol_owen"), _dam_mask)
        no_jitter = _run(_params(n, jitter_strength=0.0), _dam_mask)
        pseudo = _run(_params(n), _dam_mask)
        assert (sobol._rng.bit_generator.state
                == no_jitter._rng.bit_generator.state)
        assert (sobol._rng.bit_generator.state
                != pseudo._rng.bit_generator.state)


class TestCheckpointing:
    def test_sobol_resume_is_bit_identical(self):
        # Stateless deviates + persisted ids + substep counter: resuming
        # mid-bake must reproduce the uninterrupted trajectory exactly.
        n = 12
        params = _params(n, temporal_sampling="sobol_owen")
        straight = _run(params, _dam_mask, frames=4)

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

    def test_pre_v3_checkpoint_refused_for_sobol(self):
        n = 12
        donor = STFLIPSolver(_params(n), get_backend("cpu"))
        donor.add_liquid_mask(_dam_mask(n))
        donor.step_frame()
        state = donor.checkpoint_state()
        for key in ("particle_id", "next_particle_id", "substep_index"):
            state.pop(key, None)

        legacy = STFLIPSolver(_params(n), get_backend("cpu"))
        legacy.restore_state(dict(state))  # pseudo mode synthesizes ids

        sobol = STFLIPSolver(
            _params(n, temporal_sampling="sobol_owen"), get_backend("cpu"))
        with pytest.raises(ValueError, match="stable particle ids"):
            sobol.restore_state(dict(state))


class TestCpRotSampler:
    def test_range_determinism_and_frozen_offsets(self):
        ids = np.arange(256, dtype=np.int64)
        a = sampling.temporal_xi_cp_rot(np, ids, 3, 0)
        b = sampling.temporal_xi_cp_rot(np, ids, 3, 0)
        assert np.array_equal(a, b)
        assert a.dtype == np.float32
        assert float(a.min()) >= -0.5
        assert float(a.max()) <= 0.5
        # The defining CP property: pairwise offsets are constant across
        # steps (mod 1), which is exactly why it is only an A/B arm.
        s0 = sampling.temporal_xi_cp_rot(np, ids[:2], 0, 0).astype(
            np.float64)
        s7 = sampling.temporal_xi_cp_rot(np, ids[:2], 7, 0).astype(
            np.float64)
        d0 = (s0[0] - s0[1]) % 1.0
        d7 = (s7[0] - s7[1]) % 1.0
        assert d0 == pytest.approx(d7, abs=1e-6)
