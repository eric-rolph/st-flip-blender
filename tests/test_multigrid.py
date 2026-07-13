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


def _two_phase_jump_problem(n, *, ratio=800.0, config="bubble", seed=1):
    """A PPE with a rho_l/rho_g coefficient jump like a two-phase solve.

    Face coefficients are k_f = 1/rho_face with a HARMONIC face density, which
    gives a sharp low-conductance barrier at the interface (the classic
    high-contrast case that degrades a naive geometric multigrid). The top
    (gas) boundary is opened to atmospheric p = 0 to make the system SPD.
    """
    gx, gy, gz = np.mgrid[0:n, 0:n, 0:n]
    c = n / 2.0
    if config == "flat":
        liquid_cell = gz < n / 2
    elif config == "film":                 # thin liquid sheet spanning the domain
        liquid_cell = np.abs(gz - c) < 1.5
    else:                                  # gas bubble inside liquid
        liquid_cell = (gx - c) ** 2 + (gy - c) ** 2 + (gz - c) ** 2 > (n * 0.28) ** 2
    k = 1.0 / np.where(liquid_cell, ratio, 1.0).astype(np.float64)

    def face(axis):
        if axis == 0:
            a, b = k[1:, :, :], k[:-1, :, :]
        elif axis == 1:
            a, b = k[:, 1:, :], k[:, :-1, :]
        else:
            a, b = k[:, :, 1:], k[:, :, :-1]
        return (2.0 * a * b / (a + b)).astype(np.float32)

    kx = np.zeros((n + 1, n, n), np.float32)
    ky = np.zeros((n, n + 1, n), np.float32)
    kz = np.zeros((n, n, n + 1), np.float32)
    kx[1:-1] = face(0)
    ky[:, 1:-1] = face(1)
    kz[:, :, 1:-1] = face(2)
    kz[:, :, -1] = k[:, :, -1].astype(np.float32)          # open gas boundary
    rhs = np.random.default_rng(seed).standard_normal((n, n, n)).astype(np.float32)
    return rhs, kx, ky, kz, np.ones((n, n, n), bool)


@pytest.mark.parametrize("ratio", [800.0, 1e4])
@pytest.mark.parametrize("config", ["flat", "bubble", "film"])
def test_multigrid_stays_grid_independent_at_two_phase_density_ratios(config, ratio):
    """Issue #14: the variable-density PPE is severely ill-conditioned at
    production density ratios (rho_l/rho_g ~ 800). Jacobi-PCG iterations balloon
    with resolution; the multigrid-preconditioned count must stay flat and low.
    This guards the coarsening against a future change that silently regresses
    high-contrast robustness.
    """
    counts = {}
    for n in (32, 64):
        rhs, kx, ky, kz, liquid = _two_phase_jump_problem(
            n, ratio=ratio, config=config)
        _, jac, _ = pressure.solve(
            np, rhs, kx, ky, kz, liquid, tol=1e-5, max_iter=3000)
        _, mg, rel = multigrid.solve(
            np, rhs, kx, ky, kz, liquid, tol=1e-5, max_iter=3000)
        assert rel <= 1e-4, (config, ratio, n)          # actually converges
        counts[n] = (jac, mg)
    # Jacobi grows sharply with resolution; multigrid barely moves...
    assert counts[64][0] > 1.5 * counts[32][0]
    assert counts[64][1] <= counts[32][1] + 8
    # ...and is an order of magnitude cheaper at the finer grid.
    assert counts[64][1] < counts[64][0] / 4


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


def _localized_problem(n=48, *, touch_boundary=False, seed=0):
    rng = np.random.default_rng(seed)
    kx = rng.uniform(0.5, 2.0, (n + 1, n, n)).astype(np.float32)
    ky = rng.uniform(0.5, 2.0, (n, n + 1, n)).astype(np.float32)
    kz = rng.uniform(0.5, 2.0, (n, n, n + 1)).astype(np.float32)
    liquid = np.zeros((n, n, n), bool)
    if touch_boundary:
        liquid[0:10, 0:10, 0:10] = True           # active at a real domain corner
    else:
        liquid[20:30, 18:34, 22:33] = True         # interior blob, no boundary
    rhs = (rng.standard_normal((n, n, n)).astype(np.float32)) * liquid
    return rhs, kx, ky, kz, liquid


def test_crop_returns_none_when_not_worthwhile():
    from stflip import pressure
    # Whole grid active -> cropping saves nothing.
    _, kx, ky, kz, _ = _localized_problem(16)
    full = np.ones((16, 16, 16), bool)
    rhs = np.ones((16, 16, 16), np.float32)
    assert pressure.crop_to_active(np, rhs, kx, ky, kz, full) is None
    # Nothing active -> nothing to crop.
    dead = np.zeros((16, 16, 16), bool)
    assert pressure.crop_to_active(np, rhs, kx, ky, kz, dead) is None


@pytest.mark.parametrize("touch_boundary", [False, True])
@pytest.mark.parametrize("solver", ["jacobi", "multigrid"])
def test_active_crop_matches_full_grid_within_tolerance(monkeypatch, solver,
                                                        touch_boundary):
    """Cropping to the active box must not change the discretization: the
    cropped solve agrees with the full-grid solve to the float32 rounding
    level, both for interior and domain-boundary-touching regions."""
    from stflip import pressure
    solve = multigrid.solve if solver == "multigrid" else pressure.solve
    rhs, kx, ky, kz, liquid = _localized_problem(
        48, touch_boundary=touch_boundary, seed=1)

    p_crop, _, rel_crop = solve(np, rhs, kx, ky, kz, liquid, tol=1e-6,
                                max_iter=800)

    monkeypatch.setattr(pressure, "crop_to_active", lambda *a, **k: None)
    p_full, _, _ = solve(np, rhs, kx, ky, kz, liquid, tol=1e-6, max_iter=800)

    solvable = liquid & (pressure.diagonal(np, kx, ky, kz, liquid) > 0.0)
    scale = float(np.abs(p_full[solvable]).max())
    # Agreement far below the solve tolerance, and the cropped solve honours the
    # residual contract on exactly the same active cells.
    assert np.abs((p_crop - p_full) * solvable).max() <= 1e-4 * scale
    assert rel_crop <= 1e-4
    assert not np.any(p_crop[~solvable])            # inactive cells stay zero


def test_active_crop_is_deterministic():
    rhs, kx, ky, kz, liquid = _localized_problem(48, seed=2)
    a = multigrid.solve(np, rhs, kx, ky, kz, liquid, tol=1e-6)[0]
    b = multigrid.solve(np, rhs, kx, ky, kz, liquid, tol=1e-6)[0]
    assert np.array_equal(a, b)


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
