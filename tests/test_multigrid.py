"""Geometric multigrid V-cycle PPE preconditioner (stflip/multigrid.py)."""

import numpy as np
import pytest

from stflip import Params, STFLIPSolver, multigrid, pressure


def _problem(shape, *, seed=0, variable=True, masked=False):
    nx, ny, nz = shape
    rng = np.random.default_rng(seed)

    def faces(sh):
        return (rng.uniform(0.5, 2.0, sh).astype(np.float32) if variable
                else np.ones(sh, np.float32))

    kx = faces((nx + 1, ny, nz))
    ky = faces((nx, ny + 1, nz))
    kz = faces((nx, ny, nz + 1))
    liquid = np.ones((nx, ny, nz), bool)
    if masked:
        gx, gy, gz = np.mgrid[0:nx, 0:ny, 0:nz]
        c = np.array(shape) / 2.0
        r2 = (gx - c[0]) ** 2 + (gy - c[1]) ** 2 + (gz - c[2]) ** 2
        liquid = r2 > (min(shape) * 0.18) ** 2
    rhs = (rng.standard_normal((nx, ny, nz)).astype(np.float32)) * liquid
    return rhs, kx, ky, kz, liquid


def _true_rel_residual(p, rhs, kx, ky, kz, liquid):
    solvable = liquid & (pressure.diagonal(np, kx, ky, kz, liquid) > 0.0)
    ap = pressure.apply_laplacian(np, p, kx, ky, kz, solvable)
    b = rhs * solvable
    denom = np.sqrt(float((b * b).sum()))
    if denom < 1e-30:
        return 0.0
    return np.sqrt(float((((b - ap) * solvable) ** 2).sum())) / denom


@pytest.mark.parametrize("variable", [False, True])
def test_multigrid_solution_satisfies_the_operator(variable):
    rhs, kx, ky, kz, liquid = _problem((32, 32, 32), seed=1, variable=variable)
    p, iters, rel = multigrid.solve(
        np, rhs, kx, ky, kz, liquid, tol=1e-5, max_iter=400)
    assert 0 < iters <= 400
    # The reported residual and an independently recomputed one both clear tol.
    assert rel <= 1e-4
    assert _true_rel_residual(p, rhs, kx, ky, kz, liquid) <= 1e-4


def test_multigrid_matches_jacobi_solution():
    rhs, kx, ky, kz, liquid = _problem((32, 32, 32), seed=2, masked=True)
    pj, _, _ = pressure.solve(np, rhs, kx, ky, kz, liquid, tol=1e-7, max_iter=800)
    pm, _, _ = multigrid.solve(np, rhs, kx, ky, kz, liquid, tol=1e-7, max_iter=800)
    # Same SPD system, so the two solutions coincide (up to the shared tol).
    solvable = liquid & (pressure.diagonal(np, kx, ky, kz, liquid) > 0.0)
    scale = np.abs(pj[solvable]).max()
    assert np.abs((pm - pj) * solvable).max() <= 1e-3 * scale


def test_multigrid_iteration_count_is_grid_independent():
    """The whole point: Jacobi-PCG iterations grow with resolution while the
    multigrid-preconditioned count stays roughly flat and far lower."""
    counts = {}
    for n in (32, 64):
        rhs, kx, ky, kz, liquid = _problem((n, n, n), seed=3, variable=False)
        _, jac, _ = pressure.solve(
            np, rhs, kx, ky, kz, liquid, tol=1e-5, max_iter=600)
        _, mg, _ = multigrid.solve(
            np, rhs, kx, ky, kz, liquid, tol=1e-5, max_iter=600)
        counts[n] = (jac, mg)
    # Jacobi roughly doubles from 32 to 64; multigrid barely moves.
    assert counts[64][0] > 1.5 * counts[32][0]
    assert counts[64][1] <= counts[32][1] + 8
    # And multigrid is dramatically cheaper at the larger size.
    assert counts[64][1] < counts[64][0] / 3


def test_small_grid_falls_back_to_jacobi_identically():
    # Below 2*min_size no coarsening is possible; the result must be bit-for-bit
    # the diagonal-preconditioned CG so tiny domains are never disadvantaged.
    rhs, kx, ky, kz, liquid = _problem((6, 6, 6), seed=4)
    pm, itm, relm = multigrid.solve(
        np, rhs, kx, ky, kz, liquid, tol=1e-6, max_iter=400)
    pj, itj, relj = pressure.solve(
        np, rhs, kx, ky, kz, liquid, tol=1e-6, max_iter=400)
    assert np.array_equal(pm, pj)
    assert (itm, relm) == (itj, relj)


@pytest.mark.parametrize(
    ("shape", "expected_levels"),
    [
        ((64, 64, 64), 5),
        ((48, 32, 96), 4),
        ((100, 80, 60), 3),   # 100->50->25 (odd) stops coarsening
        ((50, 50, 50), 2),
        ((7, 7, 7), 1),       # cannot coarsen at all
    ],
)
def test_hierarchy_depth(shape, expected_levels):
    rhs, kx, ky, kz, liquid = _problem(shape, seed=5, variable=False)
    solvable = liquid & (pressure.diagonal(np, kx, ky, kz, liquid) > 0.0)
    levels = multigrid.build_hierarchy(np, kx, ky, kz, solvable)
    assert len(levels) == expected_levels
    # Each level halves the previous one along every axis.
    for coarse, fine in zip(levels[1:], levels[:-1]):
        assert coarse.shape == tuple(d // 2 for d in fine.shape)


def test_empty_and_zero_mask_are_trivial():
    rhs, kx, ky, kz, liquid = _problem((16, 16, 16), seed=6)
    p, iters, rel = multigrid.solve(
        np, np.zeros_like(rhs), kx, ky, kz, liquid)
    assert iters == 0 and rel == 0.0 and not np.any(p)

    dead = np.zeros((16, 16, 16), bool)
    p, iters, rel = multigrid.solve(np, rhs, kx, ky, kz, dead)
    assert not np.any(p)


def _equivalence_solver(pressure_solver):
    params = Params(
        resolution=(16, 16, 16),
        dx=1.0 / 16.0,
        gravity=(0.0, 0.0, -9.81),
        frame_dt=1.0 / 30.0,
        cfl_target=4.0,
        particles_per_cell=8,
        seed=99,
        pressure_solver=pressure_solver,
    )
    solver = STFLIPSolver(params, "cpu")
    mask = np.zeros(params.resolution, dtype=bool)
    mask[2:14, 2:14, 8:15] = True     # a raised block that will collapse
    solver.add_liquid_mask(mask)
    return solver


def test_multigrid_preconditioner_does_not_change_the_trajectory():
    """A preconditioner only changes the CG path, not the projected velocity
    field, so a bake with multigrid must track the Jacobi bake closely."""
    jac = _equivalence_solver("jacobi")
    mg = _equivalence_solver("multigrid")
    for _ in range(6):
        jac.step_frame()
        mg.step_frame()
    pj = jac.be.to_numpy(jac.pos)
    pm = mg.be.to_numpy(mg.pos)
    assert pj.shape == pm.shape
    # Both projections converge to pcg_tol, so positions agree to well within a
    # cell after several frames of accumulated advection.
    assert np.abs(pj - pm).max() < 0.05 * jac.p.dx * jac.p.resolution[0]


def test_params_reject_unknown_pressure_solver():
    with pytest.raises(ValueError, match="pressure_solver"):
        Params(resolution=(8, 8, 8), dx=0.1, pressure_solver="banana")


@pytest.mark.gpu
def test_multigrid_cpu_gpu_parity():
    from stflip.backend import get_backend
    try:
        gpu = get_backend("cuda")
    except Exception:                       # pragma: no cover - no CUDA present
        pytest.skip("CUDA backend unavailable")
    rhs, kx, ky, kz, liquid = _problem((32, 32, 32), seed=7)
    pc, itc, _ = multigrid.solve(np, rhs, kx, ky, kz, liquid, tol=1e-5)
    xp = gpu.xp
    pg, itg, _ = multigrid.solve(
        xp, xp.asarray(rhs), xp.asarray(kx), xp.asarray(ky), xp.asarray(kz),
        xp.asarray(liquid), tol=1e-5)
    assert itc == itg
    np.testing.assert_allclose(gpu.to_numpy(pg), pc, atol=1e-4, rtol=1e-4)
