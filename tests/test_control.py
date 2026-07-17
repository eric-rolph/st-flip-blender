"""GOAL-M1: goal-directed control -- CEM, genome, objectives, windows."""

import hashlib

import numpy as np
import pytest

from stflip.backend import get_backend
from stflip.control import (
    CEMResult,
    ForceGene,
    Objective,
    decode_genes,
    genome_dim,
    mass_in_box,
    optimize_forces,
    rollout_score,
)
from stflip.solver import Params, STFLIPSolver


def _state_hash(solver) -> str:
    digest = hashlib.sha256()
    for arr in (solver.pos, solver.vel, solver.dt_resid):
        host = solver.be.to_numpy(arr)
        digest.update(np.ascontiguousarray(host).tobytes())
    return digest.hexdigest()


def _params(n=12, **overrides):
    base = dict(
        resolution=(n, n, n), gravity=(0.0, 0.0, -9.81), dx=1.0 / n,
        frame_dt=1.0 / 24.0, cfl_target=8.0, particles_per_cell=2,
        st_enabled=True, seed=7,
    )
    base.update(overrides)
    return Params(**base)


def _dam_mask(n):
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[: n // 3, :, : n // 2] = True
    return mask


def _dam_solver(n=12, **overrides):
    solver = STFLIPSolver(_params(n, **overrides), get_backend("cpu"))
    solver.add_liquid_mask(_dam_mask(n))
    return solver


class TestGenome:
    def test_dim_and_decode_bounds(self):
        gene = ForceGene(
            force_type="DIRECTIONAL", strength=(0.0, 10.0),
            direction=((-1, 1), (-1, 1), (0, 1)),
            window=((0.0, 0.5), (0.1, 1.0)))
        assert gene.dim == 6
        kwargs = gene.decode(np.zeros(6))
        assert kwargs["strength"] == 0.0
        assert kwargs["t_start"] == 0.0
        assert kwargs["t_end"] == pytest.approx(0.1)
        kwargs = gene.decode(np.ones(6))
        assert kwargs["strength"] == 10.0
        assert np.isclose(np.linalg.norm(kwargs["direction"]), 1.0)

    def test_multi_gene_split(self):
        genes = [ForceGene(strength=(0.0, 1.0)),
                 ForceGene(force_type="VORTEX", strength=(0.0, 2.0),
                           center=((0, 1), (0, 1), (0, 1)),
                           radius=(0.05, 0.5))]
        d = genome_dim(genes)
        assert d == 1 + 5
        decoded = decode_genes(genes, np.full(d, 0.5))
        assert decoded[0]["force_type"] == "DIRECTIONAL"
        assert decoded[1]["force_type"] == "VORTEX"
        assert decoded[1]["radius"] == pytest.approx(0.275)

    def test_degenerate_direction_falls_back(self):
        gene = ForceGene(strength=(1.0, 1.0),
                         direction=((0, 0), (0, 0), (0, 0)))
        kwargs = gene.decode(np.full(gene.dim, 0.5))
        assert kwargs["direction"] == (0.0, 0.0, 1.0)


class TestObjective:
    def test_mass_in_box(self):
        pos = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
        assert mass_in_box(pos, ((0, 0, 0), (0.5, 0.5, 0.5))) == 0.5
        assert mass_in_box(np.zeros((0, 3)), ((0, 0, 0), (1, 1, 1))) == 0.0

    def test_score_combines_targets_and_penalties(self):
        obj = Objective(
            targets=[(2, ((0, 0, 0), (1, 1, 1)), 2.0)],
            keep_dry=[(1, ((0, 0, 0), (1, 1, 1)), 1.0)])
        frames = {1: np.array([[0.5, 0.5, 0.5]]),
                  2: np.array([[0.5, 0.5, 0.5]])}
        assert obj.score(frames) == pytest.approx(2.0 - 1.0)
        assert obj.last_frame == 2
        assert obj.frames() == [1, 2]

    def test_empty_objective_rejected(self):
        with pytest.raises(ValueError):
            Objective().last_frame


class TestCEM:
    def test_converges_on_analytic_sphere(self):
        # Stand-in "rollout": negative distance to a hidden optimum.
        genes = [ForceGene(strength=(0.0, 1.0),
                           direction=((0, 1), (0, 1), (0, 1)))]
        target = np.array([0.7, 0.2, 0.9, 0.4])

        class FakeObjective:
            last_frame = 1

            @staticmethod
            def frames():
                return [1]

            @staticmethod
            def score(_frames):
                raise AssertionError("unused")

        def fake_rollout(_build, _genes, unit, _obj):
            return -float(np.sum((unit - target) ** 2))

        import stflip.control as control
        original = control.rollout_score
        control.rollout_score = fake_rollout
        try:
            result = optimize_forces(
                lambda: None, genes, FakeObjective(),
                generations=20, population=24, seed=1)
        finally:
            control.rollout_score = original
        assert isinstance(result, CEMResult)
        assert result.best_score > -0.01
        assert np.allclose(result.best_unit, target, atol=0.15)

    def test_deterministic_under_seed(self):
        genes = [ForceGene(strength=(0.0, 1.0))]

        def fake_rollout(_build, _genes, unit, _obj):
            return -abs(float(unit[0]) - 0.3)

        import stflip.control as control
        original = control.rollout_score
        control.rollout_score = fake_rollout
        try:
            obj = Objective(targets=[(1, ((0, 0, 0), (1, 1, 1)), 1.0)])
            a = optimize_forces(lambda: None, genes, obj,
                                generations=5, population=8, seed=3)
            b = optimize_forces(lambda: None, genes, obj,
                                generations=5, population=8, seed=3)
        finally:
            control.rollout_score = original
        assert a.best_score == b.best_score
        assert np.array_equal(a.best_unit, b.best_unit)

    def test_patience_stops_early(self):
        genes = [ForceGene(strength=(0.0, 1.0))]

        def fake_rollout(_build, _genes, unit, _obj):
            return 1.0  # constant: never improves after gen 0

        import stflip.control as control
        original = control.rollout_score
        control.rollout_score = fake_rollout
        try:
            obj = Objective(targets=[(1, ((0, 0, 0), (1, 1, 1)), 1.0)])
            result = optimize_forces(
                lambda: None, genes, obj, generations=20,
                population=6, seed=0, patience=3)
        finally:
            control.rollout_score = original
        # gen 0 sets the best; 3 stale generations then stop = 4 total.
        assert len(result.history) == 4

    def test_warm_start_biases_search(self):
        genes = [ForceGene(strength=(0.0, 1.0))]
        seen = []

        def fake_rollout(_build, _genes, unit, _obj):
            seen.append(float(unit[0]))
            return 0.0

        import stflip.control as control
        original = control.rollout_score
        control.rollout_score = fake_rollout
        try:
            obj = Objective(targets=[(1, ((0, 0, 0), (1, 1, 1)), 1.0)])
            optimize_forces(
                lambda: None, genes, obj, generations=1, population=16,
                seed=0, init_mean=[0.9], init_std=0.05)
        finally:
            control.rollout_score = original
        assert abs(np.mean(seen) - 0.9) < 0.1
        with pytest.raises(ValueError):
            optimize_forces(
                lambda: None, genes, obj, generations=1, population=4,
                init_mean=[0.5, 0.5])

    def test_rejects_empty_genome(self):
        with pytest.raises(ValueError):
            optimize_forces(
                lambda: None, [], Objective(
                    targets=[(1, ((0, 0, 0), (1, 1, 1)), 1.0)]))


class TestForceWindows:
    def test_default_window_is_bitwise_unwindowed(self):
        a = _dam_solver()
        a.add_force("DIRECTIONAL", 3.0, direction=(1.0, 0.0, 0.0))
        b = _dam_solver()
        b.add_force("DIRECTIONAL", 3.0, direction=(1.0, 0.0, 0.0),
                    t_start=0.0)
        for _ in range(3):
            a.step_frame()
            b.step_frame()
        assert _state_hash(a) == _state_hash(b)

    def test_never_active_window_is_bitwise_no_force(self):
        a = _dam_solver()
        b = _dam_solver()
        b.add_force("DIRECTIONAL", 3.0, direction=(1.0, 0.0, 0.0),
                    t_start=100.0, t_end=200.0)
        for _ in range(3):
            a.step_frame()
            b.step_frame()
        assert _state_hash(a) == _state_hash(b)

    def test_window_actually_gates(self):
        a = _dam_solver()
        a.add_force("DIRECTIONAL", 3.0, direction=(1.0, 0.0, 0.0))
        b = _dam_solver()
        b.add_force("DIRECTIONAL", 3.0, direction=(1.0, 0.0, 0.0),
                    t_start=0.0, t_end=1.0 / 24.0)
        for _ in range(3):
            a.step_frame()
            b.step_frame()
        assert _state_hash(a) != _state_hash(b)

    def test_invalid_window_rejected(self):
        solver = _dam_solver()
        with pytest.raises(ValueError):
            solver.add_force("DIRECTIONAL", 1.0, t_start=1.0, t_end=1.0)


class TestEndToEnd:
    def test_tiny_dam_steering_improves_objective(self):
        # Mechanism smoke at 12^3 / 6 frames / tiny budget: steering must
        # beat the no-force baseline on mass-in-far-corner (the gate
        # study proper runs at 32^3+ per the design doc).
        n = 12
        box = ((0.7, 0.0, 0.0), (1.0, 1.0, 0.5))
        objective = Objective(targets=[(6, box, 1.0)])
        genes = [ForceGene(
            force_type="DIRECTIONAL", strength=(0.0, 25.0),
            direction=((0.2, 1.0), (-0.3, 0.3), (-0.5, 0.5)))]

        def build():
            return _dam_solver(n)

        baseline = rollout_score(
            build, genes, np.zeros(genome_dim(genes)), objective)
        result = optimize_forces(
            build, genes, objective,
            generations=3, population=5, seed=0)
        assert result.best_score > baseline
        assert result.best_forces[0]["strength"] > 0.0
