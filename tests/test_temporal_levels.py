"""TIME-M2: two-time-level slab reconstruction behind temporal_levels."""

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
        st_enabled=True, seed=9,
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


class TestParams:
    @pytest.mark.parametrize("bad", [0, 3, True, "2"])
    def test_rejects_invalid_levels(self, bad):
        with pytest.raises((TypeError, ValueError)):
            Params(resolution=(8, 8, 8), dx=0.125, temporal_levels=bad)

    @pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
    def test_rejects_invalid_reg(self, bad):
        with pytest.raises(ValueError):
            Params(resolution=(8, 8, 8), dx=0.125, temporal_fit_reg=bad)


class TestParity:
    def test_level_one_is_bitwise_default(self):
        n = 12
        default = _run(_params(n), _dam_mask(n))
        explicit = _run(_params(n, temporal_levels=1), _dam_mask(n))
        assert _state_hash(default) == _state_hash(explicit)

    def test_level_two_changes_the_result(self):
        n = 12
        plain = _run(_params(n), _dam_mask(n))
        fitted = _run(_params(n, temporal_levels=2), _dam_mask(n))
        assert _state_hash(plain) != _state_hash(fitted)

    def test_huge_reg_recovers_the_one_sided_mean(self):
        # lam -> inf must degrade to today's estimator: the fit's slope
        # vanishes and level-2 deposits match level-1 exactly on every
        # face (the reconstruction becomes qbar).
        n = 12
        base = _run(_params(n), _pool_mask(n), frames=2)
        huge = _run(
            _params(n, temporal_levels=2, temporal_fit_reg=1e12),
            _pool_mask(n), frames=2)
        g1 = base._p2g(base._dt_prev)
        g2 = huge._p2g(huge._dt_prev)
        for g in ("u", "v", "w"):
            np.testing.assert_allclose(
                base.be.to_numpy(g1[g]), huge.be.to_numpy(g2[g]),
                atol=1e-5)


class TestLinearExactness:
    def test_linear_in_tau_signal_recovered_through_real_p2g(self):
        # Construct v_x = a + s * theta per particle and check the
        # reconstructed face velocity equals the slab-end value
        # a + s * 0.5 at interior faces (reg = 0 -> exact fit).
        n = 12
        params = _params(
            n, gravity=(0.0, 0.0, 0.0), temporal_levels=2,
            temporal_fit_reg=0.0, particles_per_cell=8,
            exact_temporal_norm=False)
        solver = STFLIPSolver(params, get_backend("cpu"))
        solver.add_liquid_mask(np.ones((n,) * 3, dtype=bool))
        rng = np.random.default_rng(3)
        count = solver.pos.shape[0]
        dt_prev = float(solver._dt_prev)
        theta = rng.uniform(-0.5, 0.5, count).astype(np.float32)
        solver.dt_resid = solver.be.from_numpy(
            (-theta * dt_prev).astype(np.float32))
        a, s = 0.7, 2.0
        vel = np.zeros((count, 3), dtype=np.float32)
        vel[:, 0] = a + s * theta
        solver.vel = solver.be.from_numpy(vel)
        grids = solver._p2g(dt_prev)
        u = solver.be.to_numpy(grids["u"])[3:-3, 2:-2, 2:-2]
        assert u.size
        np.testing.assert_allclose(u, a + s * 0.5, atol=5e-3)


class TestPressureSystemInvariance:
    def test_phi_and_validity_identical_across_levels(self):
        # The primary arm's structural guarantee: phase and validity keep
        # their zeroth-moment recipes, so the PPE sees identical
        # coefficients whichever level is active.
        n = 12
        one = _run(_params(n, temporal_levels=1), _dam_mask(n), frames=2)
        # Same trajectory prefix is NOT required for this check; compare
        # the deposit of the SAME particle state instead.
        state = one.checkpoint_state()
        two = STFLIPSolver(
            _params(n, temporal_levels=2), get_backend("cpu"))
        two.restore_state(state)
        g1 = one._p2g(one._dt_prev)
        g2 = two._p2g(two._dt_prev)
        for key in ("c_phi", "u_phi", "v_phi", "w_phi",
                    "u_valid", "v_valid", "w_valid"):
            np.testing.assert_array_equal(
                one.be.to_numpy(g1[key]), two.be.to_numpy(g2[key]))

    def test_two_phase_pcg_iterations_unchanged(self):
        n = 12

        def run(levels):
            params = _params(
                n, temporal_levels=levels, two_phase=True, rho_gas=1.25,
                pressure_solver="multigrid")
            solver = STFLIPSolver(params, get_backend("cpu"))
            solver.add_liquid_mask(_pool_mask(n))
            solver.fill_gas()
            stats = solver.step_frame()
            return stats.pcg_iters

        # The FIRST substep's pressure system is built from the same
        # particle state and identical phi/validity fields, so iteration
        # counts must match exactly.
        assert run(1)[0] == run(2)[0]


class TestPlumbing:
    def test_still_pool_stays_calm(self):
        n = 12
        solver = _run(
            _params(n, temporal_levels=2), _pool_mask(n), frames=3)
        speeds = np.linalg.norm(solver.be.to_numpy(solver.vel), axis=1)
        assert float(speeds.max()) < 0.5

    def test_no_transient_moment_keys_leak(self):
        n = 12
        solver = _run(
            _params(n, temporal_levels=2), _pool_mask(n), frames=1)
        grids = solver._p2g(solver._dt_prev)
        assert not any(k.endswith(("_s1w", "_s2w", "_s1q"))
                       for k in grids)

    def test_checkpoint_resume_bit_identical(self):
        n = 12
        params = _params(n, temporal_levels=2)
        straight = _run(params, _dam_mask(n), frames=4)

        first = STFLIPSolver(params, get_backend("cpu"))
        first.add_liquid_mask(_dam_mask(n))
        for _ in range(2):
            first.step_frame()
        state = first.checkpoint_state()
        assert not any(k.endswith(("_s1w", "_s2w", "_s1q"))
                       for k in state)

        resumed = STFLIPSolver(params, get_backend("cpu"))
        resumed.restore_state(state)
        for _ in range(2):
            resumed.step_frame()
        assert _state_hash(resumed) == _state_hash(straight)

    def test_drift_bound_holds(self):
        n = 12
        solver = STFLIPSolver(
            _params(n, temporal_levels=2), get_backend("cpu"))
        solver.add_liquid_mask(_dam_mask(n))
        dt_max = 0.0
        for _ in range(4):
            stats = solver.step_frame()
            dt_max = max(dt_max, *(stats.dt_values or (0.0,)))
            resid = solver.be.to_numpy(solver.dt_resid)
            assert float(np.abs(resid).max()) <= 0.5 * dt_max + 1e-9

    def test_reflection_composes(self):
        n = 12
        solver = _run(
            _params(n, temporal_levels=2, reflection=True), _dam_mask(n),
            frames=2)
        vel = solver.be.to_numpy(solver.vel)
        assert np.all(np.isfinite(vel))
