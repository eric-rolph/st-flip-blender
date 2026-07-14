"""Tests for the CAP-M0 capillary clamp scale and kick limiter."""

import math

import numpy as np
import pytest

from stflip.backend import get_backend
from stflip.solver import Params, STFLIPSolver


def _pool_solver(**overrides):
    n = overrides.pop("resolution", 16)
    values = {
        "resolution": (n, n, n),
        "dx": 1.0 / n,
        "gravity": (0.0, 0.0, -9.81),
        "frame_dt": 1.0 / 24.0,
        "cfl_target": 8.0,
        "particles_per_cell": 2,
        "st_enabled": True,
        "seed": 0,
    }
    values.update(overrides)
    params = Params(**values)
    solver = STFLIPSolver(params, get_backend("cpu"))
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[:, :, : n // 2] = True
    solver.add_liquid_mask(mask)
    return solver


class TestParamsValidation:
    def test_scale_bounds(self):
        Params(resolution=(8, 8, 8), dx=0.125, st_clamp_scale=1.0)
        Params(resolution=(8, 8, 8), dx=0.125, st_clamp_scale=16.0)
        for bad in (0.5, 0.0, 16.5, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                Params(resolution=(8, 8, 8), dx=0.125, st_clamp_scale=bad)

    def test_limiter_bounds(self):
        Params(resolution=(8, 8, 8), dx=0.125, st_max_dv_cells=0.0)
        Params(resolution=(8, 8, 8), dx=0.125, st_max_dv_cells=2.0)
        for bad in (-0.1, float("nan")):
            with pytest.raises(ValueError):
                Params(resolution=(8, 8, 8), dx=0.125, st_max_dv_cells=bad)


class TestClampScale:
    def test_substep_count_matches_quantized_clamp(self):
        # A still pool has near-zero velocity, so the capillary clamp alone
        # sets the substep count: steps == ceil(frame_dt / (scale * dt_cap)).
        sigma = 800.0
        frame_dt = 1.0 / 24.0
        n = 16
        dt_cap = math.sqrt(1000.0 * (1.0 / n) ** 3 / (4.0 * math.pi * sigma))
        expected_base = math.ceil(frame_dt / dt_cap)
        assert expected_base >= 8  # the clamp-bound regime, per the roadmap

        stats_by_scale = {}
        for scale in (1.0, 4.0):
            solver = _pool_solver(
                resolution=n, surface_tension=sigma, st_clamp_scale=scale)
            stats_by_scale[scale] = solver.step_frame()

        assert stats_by_scale[1.0].steps == expected_base
        expected_scaled = math.ceil(frame_dt / (4.0 * dt_cap))
        assert stats_by_scale[4.0].steps == expected_scaled
        assert expected_scaled < expected_base

    def test_scale_only_relaxes_the_capillary_bound(self):
        # With sigma = 0 the clamp never engages, so the scale is inert.
        base = _pool_solver(resolution=12, st_clamp_scale=1.0)
        scaled = _pool_solver(resolution=12, st_clamp_scale=8.0)
        assert base.step_frame().steps == scaled.step_frame().steps


class TestKickLimiter:
    def test_unbinding_limiter_is_bitwise_inert(self):
        # A limiter too large to clip anything must not change one bit of
        # the trajectory: the clip wiring itself is a no-op until it binds.
        results = []
        for limiter in (0.0, 1e9):
            solver = _pool_solver(
                resolution=12, surface_tension=5.0,
                st_max_dv_cells=limiter)
            for _ in range(2):
                solver.step_frame()
            results.append((
                solver.be.to_numpy(solver.pos).copy(),
                solver.be.to_numpy(solver.vel).copy(),
            ))
        np.testing.assert_array_equal(results[0][0], results[1][0])
        np.testing.assert_array_equal(results[0][1], results[1][1])

    def test_binding_limiter_bounds_capillary_velocity(self):
        # Zero gravity and a curved droplet surface: every bit of velocity
        # comes from the CSF kick, so a tightly limited run must end far
        # slower than an unlimited one at the same relaxed clamp.
        def run(limiter):
            n = 12
            params = Params(
                resolution=(n, n, n),
                dx=1.0 / n,
                gravity=(0.0, 0.0, 0.0),
                frame_dt=1.0 / 24.0,
                cfl_target=8.0,
                particles_per_cell=2,
                st_enabled=True,
                seed=0,
                surface_tension=50.0,
                st_clamp_scale=4.0,
                st_max_dv_cells=limiter,
            )
            solver = STFLIPSolver(params, get_backend("cpu"))
            centres = (np.stack(np.meshgrid(
                *(np.arange(n),) * 3, indexing="ij"), axis=-1) + 0.5) / n
            mask = np.linalg.norm(centres - 0.5, axis=-1) <= 0.3
            solver.add_liquid_mask(mask)
            stats = None
            for _ in range(2):
                stats = solver.step_frame()
            return stats.max_speed

        limited = run(0.005)
        unlimited = run(0.0)
        assert limited < unlimited
        assert limited < 0.5 * unlimited


def test_defaults_do_not_change_a_sigma_free_bake():
    # Guard the default path: both new params at their defaults leave a
    # surface-tension-free bake byte-identical to an explicit default run.
    base = _pool_solver(resolution=12)
    explicit = _pool_solver(
        resolution=12, st_clamp_scale=1.0, st_max_dv_cells=0.0)
    base.step_frame()
    explicit.step_frame()
    np.testing.assert_array_equal(
        base.be.to_numpy(base.pos), explicit.be.to_numpy(explicit.pos))
    np.testing.assert_array_equal(
        base.be.to_numpy(base.vel), explicit.be.to_numpy(explicit.vel))
