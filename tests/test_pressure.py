import numpy as np

from stflip import pressure


def _setup(n=16, seed=3):
    rng = np.random.default_rng(seed)
    liquid = np.zeros((n, n, n), dtype=bool)
    liquid[2:-2, 2:-2, 2:-2] = True
    kx = np.ones((n + 1, n, n), dtype=np.float64)
    ky = np.ones((n, n + 1, n), dtype=np.float64)
    kz = np.ones((n, n, n + 1), dtype=np.float64)
    # Closed domain boundaries.
    kx[0] = kx[-1] = 0.0
    ky[:, 0] = ky[:, -1] = 0.0
    kz[:, :, 0] = kz[:, :, -1] = 0.0
    rhs = rng.standard_normal((n, n, n)) * liquid
    rhs -= rhs.mean(where=liquid)  # compatible RHS
    rhs *= liquid
    return rhs, kx, ky, kz, liquid


def test_pcg_converges_and_solves():
    rhs, kx, ky, kz, liquid = _setup()
    p, iters, rel = pressure.solve(np, rhs, kx, ky, kz, liquid,
                                   tol=1e-6, max_iter=2000)
    assert rel < 1e-5
    ap = pressure.apply_laplacian(np, p, kx, ky, kz, liquid)
    err = np.linalg.norm((ap - rhs)[liquid]) / np.linalg.norm(rhs[liquid])
    assert err < 1e-4


def test_operator_symmetry():
    rng = np.random.default_rng(0)
    rhs, kx, ky, kz, liquid = _setup(n=8)
    # Non-binary coefficients exercise the fractional-aperture operator; the
    # same face value must contribute symmetrically to its two incident cells.
    kx *= rng.uniform(0.05, 1.0, size=kx.shape)
    ky *= rng.uniform(0.05, 1.0, size=ky.shape)
    kz *= rng.uniform(0.05, 1.0, size=kz.shape)
    x = rng.standard_normal(liquid.shape) * liquid
    y = rng.standard_normal(liquid.shape) * liquid
    ax = pressure.apply_laplacian(np, x, kx, ky, kz, liquid)
    ay = pressure.apply_laplacian(np, y, kx, ky, kz, liquid)
    assert abs(np.vdot(ax, y) - np.vdot(x, ay)) < 1e-8 * max(
        1.0, abs(np.vdot(ax, y)))


def test_zero_rhs_gives_zero_pressure():
    rhs, kx, ky, kz, liquid = _setup()
    p, iters, rel = pressure.solve(np, rhs * 0.0, kx, ky, kz, liquid)
    assert np.allclose(p, 0.0)
    assert iters == 0


def test_exterior_coefficients_use_half_cell_dirichlet_terms():
    liquid = np.ones((1, 1, 1), dtype=bool)
    p = np.asarray([[[3.0]]])
    kx = np.asarray([[[2.0]], [[5.0]]])
    ky = np.asarray([[[7.0], [11.0]]])
    kz = np.asarray([[[13.0, 17.0]]])

    applied = pressure.apply_laplacian(np, p, kx, ky, kz, liquid)
    diag = pressure.diagonal(np, kx, ky, kz, liquid)
    expected_diagonal = 2.0 * (2.0 + 5.0 + 7.0 + 11.0 + 13.0 + 17.0)

    assert diag[0, 0, 0] == expected_diagonal
    assert applied[0, 0, 0] == expected_diagonal * 3.0


def test_operator_stays_symmetric_with_open_exterior_coefficients():
    rng = np.random.default_rng(17)
    _, kx, ky, kz, liquid = _setup(n=5)
    liquid[:] = True
    kx[0] = rng.uniform(0.1, 1.0, size=kx[0].shape)
    kx[-1] = rng.uniform(0.1, 1.0, size=kx[-1].shape)
    ky[:, 0] = rng.uniform(0.1, 1.0, size=ky[:, 0].shape)
    ky[:, -1] = rng.uniform(0.1, 1.0, size=ky[:, -1].shape)
    kz[:, :, 0] = rng.uniform(0.1, 1.0, size=kz[:, :, 0].shape)
    kz[:, :, -1] = rng.uniform(0.1, 1.0, size=kz[:, :, -1].shape)
    x = rng.standard_normal(liquid.shape)
    y = rng.standard_normal(liquid.shape)

    ax = pressure.apply_laplacian(np, x, kx, ky, kz, liquid)
    ay = pressure.apply_laplacian(np, y, kx, ky, kz, liquid)

    np.testing.assert_allclose(
        np.vdot(ax, y), np.vdot(x, ay), rtol=1e-12, atol=1e-12
    )
