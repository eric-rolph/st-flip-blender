"""NORM-M1: exact gamma-conditioned temporal weighting (paper Sec 3.10).

The paper attenuates jitter per particle (gamma_p < 1) but keeps the
gamma = 1 temporal kernel and m0, accepting a bounded error (up to 7.7
percent mean weight, 3.9 percent phi_st).  Params.exact_temporal_norm
divides each particle's deposited weight by the closed-form mean weight
mu(gamma_p), restoring E[w] = 1 at every gamma.
"""

import hashlib

import numpy as np
import pytest

from stflip import kernels
from stflip.backend import get_backend
from stflip.solver import Params, STFLIPSolver

MU_0 = 945.0 / 1024.0


def _state_hash(solver) -> str:
    digest = hashlib.sha256()
    for arr in (solver.pos, solver.vel, solver.dt_resid):
        host = solver.be.to_numpy(arr)
        digest.update(np.ascontiguousarray(host).tobytes())
    return digest.hexdigest()


def _run(params, setup, frames=2):
    solver = STFLIPSolver(params, get_backend("cpu"))
    setup(solver)
    for _ in range(frames):
        solver.step_frame()
    return solver


def _dam_mask(n):
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[: n // 3, :, : n // 2] = True
    return mask


def _pool_mask(n):
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[:, :, : n // 2] = True
    return mask


class TestMeanWeightPolynomial:
    def test_extremes_are_exact(self):
        assert kernels.w_temporal_mean(
            np, np.float32(1.0)) == np.float32(1.0)
        assert kernels.w_temporal_mean(np, np.float64(0.0)) == float(
            kernels.w_temporal(np, np.float64(0.0)))
        assert kernels.w_temporal_mean(np, np.float64(0.0)) == MU_0

    def test_matches_quadrature_of_actual_kernel(self):
        # Midpoint quadrature of the repo's own W_T over the unit slab.
        tau = (np.arange(400_000, dtype=np.float64) + 0.5) / 400_000.0 - 0.5
        for g in (0.0, 0.2, 0.5, 0.8, 1.0):
            quad = float(kernels.w_temporal(np, g * tau).mean())
            poly = float(kernels.w_temporal_mean(np, np.float64(g)))
            assert quad == pytest.approx(poly, abs=1e-8)

    def test_bounded_and_monotonic(self):
        g = np.linspace(0.0, 1.0, 101)
        mu = kernels.w_temporal_mean(np, g)
        assert np.all(mu >= MU_0)
        assert np.all(mu <= 1.0)
        assert np.all(np.diff(mu) > 0.0)


class TestSolverParity:
    def test_gamma_one_is_bitwise_identical(self):
        # Full jitter everywhere: mu(1) == 1.0 exactly, so the exact
        # normalization must not change a single bit.
        n = 12
        base = dict(
            resolution=(n, n, n), dx=1.0 / n, gravity=(0.0, 0.0, -9.81),
            frame_dt=1.0 / 24.0, cfl_target=8.0, particles_per_cell=2,
            st_enabled=True, adaptive_gamma=False, jitter_strength=1.0,
            seed=5,
        )
        hashes = []
        for exact in (False, True):
            solver = _run(
                Params(**base, exact_temporal_norm=exact),
                lambda s: s.add_liquid_mask(_dam_mask(n)), frames=3)
            hashes.append(_state_hash(solver))
        assert hashes[0] == hashes[1]

    def test_attenuated_runs_differ(self):
        # A calm pool with adaptive gamma active is exactly where the
        # conditioning changes deposits.
        n = 12
        base = dict(
            resolution=(n, n, n), dx=1.0 / n, gravity=(0.0, 0.0, -9.81),
            frame_dt=1.0 / 24.0, cfl_target=8.0, particles_per_cell=2,
            st_enabled=True, adaptive_gamma=True, seed=5,
        )
        hashes = []
        for exact in (False, True):
            solver = _run(
                Params(**base, exact_temporal_norm=exact),
                lambda s: s.add_liquid_mask(_pool_mask(n)), frames=2)
            hashes.append(_state_hash(solver))
        assert hashes[0] != hashes[1]

    def test_rejects_non_bool(self):
        with pytest.raises(TypeError):
            Params(resolution=(8, 8, 8), dx=0.125,
                   exact_temporal_norm="yes")


class TestDepositedMassCalibration:
    @pytest.mark.parametrize("strength", [0.0, 0.3, 0.7])
    def test_m0_plateau_at_every_gamma(self, strength):
        # A fully filled, force-free box: particles never move, theta is
        # distributed as gamma * U(-1/2, 1/2), and the interior accumulator
        # mean must equal m0 = ppc regardless of gamma.  Legacy weighting
        # instead scales the deposit by mu(gamma).
        n = 12
        base = dict(
            resolution=(n, n, n), dx=1.0 / n, gravity=(0.0, 0.0, 0.0),
            frame_dt=1.0 / 24.0, cfl_target=4.0, particles_per_cell=8,
            st_enabled=True, adaptive_gamma=False,
            jitter_strength=strength, seed=11,
        )
        full = np.ones((n,) * 3, dtype=bool)
        means = {}
        for exact in (False, True):
            solver = _run(Params(**base, exact_temporal_norm=exact),
                          lambda s: s.add_liquid_mask(full), frames=2)
            grids = solver._p2g(solver._dt_prev)
            interior = solver.be.to_numpy(
                grids["c_m"])[2:-2, 2:-2, 2:-2] / solver.m0
            means[exact] = float(interior.mean())
        mu = float(kernels.w_temporal_mean(np, np.float64(strength)))
        assert means[True] == pytest.approx(1.0, rel=0.01)
        assert means[False] == pytest.approx(mu, rel=0.01)


class TestTwoPhaseInterface:
    def test_interface_phi_bias_removed(self):
        # Calm liquid below fast gas: legacy weighting reads the interface
        # liquid fraction as mu(0) / (mu(0) + 1) ~ 0.48 instead of 0.5,
        # biasing face densities toward gas.
        n = 12
        base = dict(
            resolution=(n, n, n), dx=1.0 / n, gravity=(0.0, 0.0, 0.0),
            frame_dt=1.0 / 24.0, cfl_target=4.0, particles_per_cell=8,
            st_enabled=True, adaptive_gamma=True, two_phase=True,
            rho_gas=1.2, seed=7,
        )

        def setup(s):
            s.add_liquid_mask(_pool_mask(n))
            s.fill_gas()
            gas = s.be.to_numpy(s.phase) < 0.5
            vel = s.be.to_numpy(s.vel)
            vel[gas, 0] = 2.0
            s.vel = s.be.from_numpy(vel.astype(np.float32))

        phi = {}
        for exact in (False, True):
            solver = _run(Params(**base, exact_temporal_norm=exact),
                          setup, frames=1)
            grids = solver._p2g(solver._dt_prev)
            w_phi = solver.be.to_numpy(grids["w_phi"])
            phi[exact] = float(w_phi[2:-2, 2:-2, n // 2].mean())
        # Comparative assertions are robust to the residual stirring the
        # gas stream induces in one frame.
        assert phi[True] > phi[False]
        assert abs(phi[True] - 0.5) < abs(phi[False] - 0.5)
        assert phi[False] < 0.495


class _RecordingSolver(STFLIPSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_calls = []

    def _jitter_gamma(self, dt):
        gamma = super()._jitter_gamma(dt)
        self.gamma_calls.append(
            (float(dt), self.be.to_numpy(gamma).copy()))
        return gamma


class TestRecomputeContract:
    def test_p2g_recompute_matches_draw(self):
        # The exact normalization recomputes gamma at P2G time instead of
        # persisting it; that is only correct while nothing between the
        # draw and the next deposit modifies velocities.  Exercise the
        # riskiest tail features (sheeting, moving solid, gravity) and
        # assert every P2G recompute reproduces the preceding draw bitwise.
        n = 12
        params = Params(
            resolution=(n, n, n), dx=1.0 / n, gravity=(0.0, 0.0, -9.81),
            frame_dt=1.0 / 24.0, cfl_target=8.0, particles_per_cell=2,
            st_enabled=True, adaptive_gamma=True, sheeting=1.0, seed=3,
        )
        solver = _RecordingSolver(params, get_backend("cpu"))
        solver.add_liquid_mask(_dam_mask(n))
        cells = (np.stack(np.meshgrid(
            *(np.arange(n),) * 3, indexing="ij"), axis=-1) + 0.5) / n
        sdf = np.linalg.norm(cells - (0.5, 0.5, 0.2), axis=-1) - 0.15
        vel = np.zeros((n, n, n, 3), dtype=np.float32)
        vel[..., 0] = 0.4
        solver.set_solid_sdf(sdf.astype(np.float32), solid_vel=vel)
        for _ in range(3):
            solver.step_frame()

        calls = solver.gamma_calls
        assert len(calls) >= 4
        # Calls alternate P2G (recompute) and draw within each substep; a
        # draw at index k must match the following substep's P2G at k + 1.
        checked = 0
        for (draw_dt, draw_gamma), (p2g_dt, p2g_gamma) in zip(
                calls[1::2], calls[2::2]):
            assert p2g_dt == draw_dt
            assert np.array_equal(draw_gamma, p2g_gamma)
            checked += 1
        assert checked >= 1
