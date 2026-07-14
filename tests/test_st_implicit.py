"""CAP-M1: semi-implicit capillary stabilizer library."""

import math

import numpy as np
import pytest

from stflip import st_implicit
from stflip.surface_tension import smoothed_phase_gradient


def _band_problem(n=16, sigma=1.0, dt=0.01, dx=None, seed=0):
    """A flat-interface band problem on one face lattice."""

    dx = dx if dx is not None else 1.0 / n
    rng = np.random.default_rng(seed)
    phi = np.zeros((n, n, n), dtype=np.float64)
    phi[:, :, : n // 2] = 1.0
    _phi_s, _gx, _gy, _gz, mag = smoothed_phase_gradient(np, phi, dx)
    delta = st_implicit.face_delta(np, mag, 2)  # w-component lattice
    mask = np.ones(delta.shape, dtype=bool)
    kx, ky, kz = st_implicit.edge_coefficients(
        np, delta, mask, dt, sigma, dx)
    rho = np.full(delta.shape, 1000.0, dtype=np.float64)
    u_hat = rng.normal(0.0, 1.0, delta.shape)
    return delta, mask, (kx, ky, kz), rho, u_hat, dx


class TestBuildingBlocks:
    def test_face_delta_constant_field_and_shapes(self):
        mag = np.full((4, 5, 6), 2.5)
        for axis, shape in ((0, (5, 5, 6)), (1, (4, 6, 6)), (2, (4, 5, 7))):
            face = st_implicit.face_delta(np, mag, axis)
            assert face.shape == shape
            assert np.allclose(face, 2.5)

    def test_edge_coefficients_gate_and_boundary(self):
        delta = np.ones((4, 4, 4))
        mask = np.ones((4, 4, 4), dtype=bool)
        mask[0, 0, 0] = False
        kx, ky, kz = st_implicit.edge_coefficients(
            np, delta, mask, dt=0.1, sigma=2.0, dx=0.5)
        scale = 0.1 * 0.1 * 2.0 / 0.25
        # Outermost layers are zero (natural boundary).
        assert float(np.abs(kx[0]).max()) == 0.0
        assert float(np.abs(kx[-1]).max()) == 0.0
        # Interior edges carry the averaged delta...
        assert kx[2, 1, 1] == pytest.approx(scale)
        # ...but every edge touching the masked DOF is gated off.
        assert kx[1, 0, 0] == 0.0
        assert ky[0, 1, 0] == 0.0
        assert kz[0, 0, 1] == 0.0


class TestOperator:
    def test_symmetric_positive_definite(self):
        delta, mask, coefs, rho, _u, _dx = _band_problem()
        rng = np.random.default_rng(3)
        x = rng.normal(size=delta.shape) * mask
        y = rng.normal(size=delta.shape) * mask
        kx, ky, kz = coefs
        ax = st_implicit.apply_operator(np, x, rho, kx, ky, kz, mask)
        ay = st_implicit.apply_operator(np, y, rho, kx, ky, kz, mask)
        assert float((x * ay).sum()) == pytest.approx(
            float((ax * y).sum()), rel=1e-10)
        assert float((x * ax).sum()) > 0.0


class TestSolve:
    def test_identity_away_from_interface(self):
        delta, mask, coefs, rho, u_hat, _dx = _band_problem()
        kx, ky, kz = coefs
        u, _it, rel = st_implicit.stabilize_component(
            np, u_hat, rho, kx, ky, kz, mask, tol=1e-6)
        assert rel <= 1e-6
        # Faces with no nonzero incident coefficient are outside the
        # interface band; their velocities must pass through BITWISE.
        incident = (kx[1:] + kx[:-1] + ky[:, 1:] + ky[:, :-1]
                    + kz[:, :, 1:] + kz[:, :, :-1])
        far = incident == 0.0
        assert far.any()
        assert np.array_equal(u[far], u_hat[far])

    def test_consistency_near_explicit_limit(self):
        # At half the Brackbill step the implicit correction is a small
        # multiple of max(a / R): the solve must stay within that bound of
        # the explicit (pass-through) field.
        n = 16
        dx = 1.0 / n
        sigma = 1.0
        rho_val = 1000.0
        dt_b = math.sqrt(rho_val * dx ** 3 / (4.0 * math.pi * sigma))
        delta, mask, coefs, rho, u_hat, _dx = _band_problem(
            n=n, sigma=sigma, dt=0.5 * dt_b)
        kx, ky, kz = coefs
        u, _it, _rel = st_implicit.stabilize_component(
            np, u_hat, rho, kx, ky, kz, mask, tol=1e-8)
        diag_a = (kx[1:] + kx[:-1] + ky[:, 1:] + ky[:, :-1]
                  + kz[:, :, 1:] + kz[:, :, :-1])
        ratio = float((diag_a / rho).max())
        band = diag_a > 0.0
        scale = float(np.abs(u_hat[band]).max())
        diff = float(np.abs(u - u_hat)[band].max())
        assert diff <= 3.0 * ratio * scale

    def test_gated_dofs_do_not_leak(self):
        delta, mask, _coefs, rho, u_hat, dx = _band_problem()
        blocked = mask.copy()
        blocked[:, :8, :] = False
        kx, ky, kz = st_implicit.edge_coefficients(
            np, delta, blocked, dt=0.01, sigma=1.0, dx=dx)
        u_a, _i, _r = st_implicit.stabilize_component(
            np, u_hat, rho, kx, ky, kz, blocked, tol=1e-8)
        perturbed = u_hat.copy()
        perturbed[:, :8, :] += 100.0
        u_b, _i, _r = st_implicit.stabilize_component(
            np, perturbed, rho, kx, ky, kz, blocked, tol=1e-8)
        # Values on the gated-off side pass through (perturbed)...
        assert np.array_equal(u_b[:, :8, :], perturbed[:, :8, :])
        # ...and never influence the solved side.
        assert np.array_equal(u_a[:, 8:, :], u_b[:, 8:, :])

    def test_converges_at_stiff_clamp_scale(self):
        # Sixteen-fold clamp relaxation makes a / R of order ten; CG must
        # still converge and the result must damp, not amplify.
        n = 16
        dx = 1.0 / n
        sigma = 1.0
        rho_val = 1000.0
        dt_b = math.sqrt(rho_val * dx ** 3 / (4.0 * math.pi * sigma))
        delta, mask, _c, rho, u_hat, _dx = _band_problem(
            n=n, sigma=sigma, dt=16.0 * dt_b)
        kx, ky, kz = st_implicit.edge_coefficients(
            np, delta, mask, 16.0 * dt_b, sigma, dx)
        u, it, rel = st_implicit.stabilize_component(
            np, u_hat, rho, kx, ky, kz, mask, tol=1e-5, max_iter=200)
        assert rel <= 1e-5
        assert it < 200
        # Energy in the R-weighted norm must not grow (damping property of
        # (R + A)^-1 R with SPD A).
        before = float((rho * u_hat * u_hat).sum())
        after = float((rho * u * u).sum())
        assert after < before

    def test_zero_sigma_passes_through_bitwise(self):
        delta, mask, _c, rho, u_hat, dx = _band_problem()
        kx, ky, kz = st_implicit.edge_coefficients(
            np, delta, mask, dt=0.1, sigma=0.0, dx=dx)
        u, it, rel = st_implicit.stabilize_component(
            np, u_hat, rho, kx, ky, kz, mask)
        assert it == 0
        assert rel == 0.0
        assert np.array_equal(u, u_hat)
