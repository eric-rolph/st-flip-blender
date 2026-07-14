"""TIME-M1: W_T-weighted linear-in-time moment fit."""

import numpy as np
import pytest

from stflip import kernels, temporal_fit
from stflip.validation import temporal_quadrature_coverage


def _uniform_slab(n=4096):
    tau = (np.arange(n, dtype=np.float64) + 0.5) / n - 0.5
    return tau, kernels.w_temporal(np, tau)


class TestFit:
    def test_exact_linear_recovery(self):
        tau, w = _uniform_slab()
        rng = np.random.default_rng(1)
        w = w * rng.random(tau.shape)  # arbitrary extra spatial weights
        q = 3.0 * tau - 0.7
        moments = temporal_fit.accumulate_moments(np, tau, q, w)
        value, slope = temporal_fit.fit_evaluate(np, moments, lam=0.0)
        assert value == pytest.approx(3.0 * 0.5 - 0.7, abs=1e-9)
        assert slope == pytest.approx(3.0, abs=1e-9)

    def test_large_lam_degrades_to_one_sided_mean(self):
        tau, w = _uniform_slab()
        q = np.sin(6.0 * tau)
        moments = temporal_fit.accumulate_moments(np, tau, q, w)
        value, slope = temporal_fit.fit_evaluate(np, moments, lam=1e12)
        mean = temporal_fit.one_sided_mean(np, moments)
        assert slope == pytest.approx(0.0, abs=1e-9)
        assert value == pytest.approx(float(mean), abs=1e-9)

    def test_zero_variance_gives_zero_slope(self):
        tau = np.full(64, 0.2266)
        w = kernels.w_temporal(np, tau)
        q = np.full(64, 1.5)
        moments = temporal_fit.accumulate_moments(np, tau, q, w)
        value, slope = temporal_fit.fit_evaluate(np, moments, lam=0.01)
        assert slope == pytest.approx(0.0, abs=1e-12)
        assert value == pytest.approx(1.5, abs=1e-9)

    def test_batched_axis(self):
        rng = np.random.default_rng(2)
        tau = rng.uniform(-0.5, 0.5, (10, 128))
        w = kernels.w_temporal(np, tau)
        q = 2.0 * tau
        moments = temporal_fit.accumulate_moments(np, tau, q, w)
        value, slope = temporal_fit.fit_evaluate(np, moments, lam=0.0)
        assert value.shape == (10,)
        assert np.allclose(slope, 2.0, atol=1e-9)

    def test_rejects_negative_lam(self):
        tau, w = _uniform_slab(64)
        moments = temporal_fit.accumulate_moments(np, tau, tau, w)
        with pytest.raises(ValueError):
            temporal_fit.fit_evaluate(np, moments, lam=-0.1)


class TestAnalyticsCrossCheck:
    def test_matches_temporal_quadrature_coverage(self):
        # The one-sided mean's effective evaluation time and effective
        # sample fraction quoted throughout the roadmap (0.2266 / 0.613)
        # must agree with the repo's own diagnostics.
        coverage = temporal_quadrature_coverage(4096, 16)
        assert coverage["weighted_mean_tau"] == pytest.approx(
            0.5 - 35.0 / 128.0, abs=1e-4)
        assert coverage["effective_weighted_sample_fraction"] == (
            pytest.approx(0.613, abs=2e-3))
        tau, w = _uniform_slab()
        moments = temporal_fit.accumulate_moments(np, tau, tau, w)
        tbar = float(moments["S1w"] / moments["S0w"])
        assert tbar == pytest.approx(
            coverage["weighted_mean_tau"], abs=1e-4)


class TestJitterSimulation:
    def test_stationary_distribution_matches_theory(self):
        rng = np.random.default_rng(3)
        thetas = temporal_fit.simulate_jitter_taus(rng, 20_000, 24, 1.0)
        final = thetas[-1]
        # Unclamped stationary law is U(-1/2, 1/2).
        assert abs(float(final.mean())) < 0.01
        assert float(final.std()) == pytest.approx(
            1.0 / np.sqrt(12.0), abs=0.01)
        assert float(np.abs(final).max()) <= 0.5 + 1e-12

    def test_attenuated_gamma_narrows_support(self):
        rng = np.random.default_rng(4)
        thetas = temporal_fit.simulate_jitter_taus(rng, 20_000, 24, 0.4)
        final = thetas[-1]
        assert float(np.abs(final).max()) <= 0.2 + 1e-12
        assert float(final.std()) == pytest.approx(
            0.4 / np.sqrt(12.0), abs=0.01)

    def test_abrupt_dt_change_stays_finite_and_bounded(self):
        rng = np.random.default_rng(5)
        thetas = temporal_fit.simulate_jitter_taus(
            rng, 5_000, 24, 1.0, dt_ratio=2.0)
        assert np.all(np.isfinite(thetas))
        # Residual bound |resid| <= dt_max/2 translates to theta bounded by
        # the previous dt ratio after the switch.
        assert float(np.abs(thetas).max()) <= 1.0 + 1e-12
