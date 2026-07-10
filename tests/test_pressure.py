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
