"""Goal-directed control over body forces (GOAL-M1).

Derivative-free inverse problems for art direction: declare which force
parameters are free (``ForceGene``), what success means (``Objective``
over particle positions at chosen frames), and let a dependency-free
Cross-Entropy Method search the proxy-resolution rollouts. Design and
pre-registered gates: docs/design/goal-directed-splash.md.

The module is bpy-free and backend-agnostic (NumPy or CuPy solvers);
the optimizer itself always runs on the host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------- genome


@dataclass
class ForceGene:
    """One optimizable force: fixed type, bounded free parameters.

    Every bounded parameter is described by a ``(lo, hi)`` tuple; a
    parameter left as ``None`` keeps the solver default and is not
    optimized. ``direction`` and ``center`` bounds are per-component
    ``((lo, hi), (lo, hi), (lo, hi))``. Directions are re-normalized
    after decoding, so their bounds describe a box the unit vector is
    drawn through, not the vector itself.
    """

    force_type: str = "DIRECTIONAL"
    strength: tuple = (0.0, 1.0)
    direction: tuple | None = None
    center: tuple | None = None
    radius: tuple | None = None
    window: tuple | None = None   # ((t0_lo, t0_hi), (dur_lo, dur_hi))

    def _specs(self):
        specs = [("strength", self.strength)]
        if self.direction is not None:
            specs += [(f"direction{i}", self.direction[i]) for i in range(3)]
        if self.center is not None:
            specs += [(f"center{i}", self.center[i]) for i in range(3)]
        if self.radius is not None:
            specs.append(("radius", self.radius))
        if self.window is not None:
            specs += [("t_start", self.window[0]),
                      ("duration", self.window[1])]
        return specs

    @property
    def dim(self) -> int:
        return len(self._specs())

    def decode(self, unit: np.ndarray) -> dict:
        """Map a [0, 1]^dim slice to add_force kwargs."""
        values = {}
        for (name, (lo, hi)), u in zip(self._specs(), unit):
            values[name] = float(lo) + float(np.clip(u, 0.0, 1.0)) * (
                float(hi) - float(lo))
        kwargs = {"force_type": self.force_type,
                  "strength": values["strength"]}
        if self.direction is not None:
            vec = np.array([values[f"direction{i}"] for i in range(3)])
            norm = float(np.linalg.norm(vec))
            kwargs["direction"] = tuple(
                (vec / norm) if norm > _EPS else (0.0, 0.0, 1.0))
        if self.center is not None:
            kwargs["center"] = tuple(
                values[f"center{i}"] for i in range(3))
        if self.radius is not None:
            kwargs["radius"] = values["radius"]
        if self.window is not None:
            kwargs["t_start"] = values["t_start"]
            kwargs["t_end"] = values["t_start"] + max(
                values["duration"], 1e-6)
        return kwargs


def decode_genes(genes, unit: np.ndarray):
    """Split one flat [0, 1]^d vector into per-gene add_force kwargs."""
    out = []
    offset = 0
    for gene in genes:
        out.append(gene.decode(unit[offset:offset + gene.dim]))
        offset += gene.dim
    return out


def genome_dim(genes) -> int:
    return sum(gene.dim for gene in genes)


# ------------------------------------------------------------ objectives


def mass_in_box(positions: np.ndarray, box) -> float:
    """Fraction of particles inside the AABB ``box = (lo3, hi3)``."""
    if positions.shape[0] == 0:
        return 0.0
    lo, hi = np.asarray(box[0]), np.asarray(box[1])
    inside = np.all((positions >= lo) & (positions < hi), axis=1)
    return float(inside.mean())


@dataclass
class Objective:
    """Weighted goal over particle positions at specific frames.

    ``targets``: list of (frame, box, weight) rewarding mass in ``box``.
    ``keep_dry``: list of (frame, box, weight) penalizing mass in ``box``.
    Frames are 1-indexed; the rollout runs to the largest one.
    """

    targets: list = field(default_factory=list)
    keep_dry: list = field(default_factory=list)

    @property
    def last_frame(self) -> int:
        frames = [f for f, _b, _w in self.targets]
        frames += [f for f, _b, _w in self.keep_dry]
        if not frames:
            raise ValueError("objective needs at least one term")
        return max(frames)

    def frames(self):
        return sorted({f for f, _b, _w in self.targets}
                      | {f for f, _b, _w in self.keep_dry})

    def score(self, positions_by_frame: dict) -> float:
        total = 0.0
        for frame, box, weight in self.targets:
            total += weight * mass_in_box(positions_by_frame[frame], box)
        for frame, box, weight in self.keep_dry:
            total -= weight * mass_in_box(positions_by_frame[frame], box)
        return total


def rollout_score(build_solver, genes, unit, objective) -> float:
    """Build a solver, register decoded forces, run, and score.

    ``build_solver()`` returns a fresh, fully seeded solver (liquid,
    solids, outflows -- everything but the optimized forces).
    """
    solver = build_solver()
    for kwargs in decode_genes(genes, unit):
        solver.add_force(**kwargs)
    wanted = set(objective.frames())
    positions_by_frame = {}
    for frame in range(1, objective.last_frame + 1):
        solver.step_frame()
        if frame in wanted:
            positions_by_frame[frame] = solver.be.to_numpy(solver.pos)
    del solver
    try:
        # Hundreds of short-lived solvers fragment the CuPy pool and
        # progressively double per-rollout cost; release between
        # rollouts (measured: 28 s -> ~13 s per 32^3 rollout).
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    return objective.score(positions_by_frame)


# -------------------------------------------------------------- optimizer


@dataclass
class CEMResult:
    best_unit: np.ndarray
    best_score: float
    history: list          # per-generation dicts
    best_forces: list      # decoded add_force kwargs


def optimize_forces(build_solver, genes, objective, *,
                    generations: int = 12, population: int = 16,
                    elite_frac: float = 0.25, init_std: float = 0.3,
                    std_floor: float = 0.02, seed: int = 0,
                    patience: int = 0, init_mean=None,
                    log=None) -> CEMResult:
    """Cross-Entropy Method over the gene box (docs pre-register gates).

    Deterministic under ``seed``. Scores every candidate with a full
    proxy rollout, refits mean/std to the elite quantile, anneals the
    std toward ``std_floor``, and never evaluates outside [0, 1]^d.
    ``patience`` > 0 stops early after that many consecutive
    generations without a new overall best (0 disables). ``init_mean``
    warm-starts the search distribution (e.g. a coarser stage's best
    unit vector for the refine stage); pair it with a small
    ``init_std``.
    """
    dim = genome_dim(genes)
    if dim == 0:
        raise ValueError("no free parameters to optimize")
    n_elite = max(2, int(round(population * elite_frac)))
    rng = np.random.default_rng(seed)
    if init_mean is not None:
        mean = np.clip(np.asarray(init_mean, dtype=float), 0.0, 1.0)
        if mean.shape != (dim,):
            raise ValueError("init_mean must match the genome dimension")
    else:
        mean = np.full(dim, 0.5)
    std = np.full(dim, float(init_std))
    best_unit = mean.copy()
    best_score = -math.inf
    history = []
    stale = 0
    for gen in range(generations):
        samples = rng.normal(mean, std, size=(population, dim))
        samples = np.clip(samples, 0.0, 1.0)
        if gen > 0:
            samples[0] = best_unit  # elitism: never lose the champion
        scores = np.array([
            rollout_score(build_solver, genes, unit, objective)
            for unit in samples])
        order = np.argsort(scores)[::-1]
        elite = samples[order[:n_elite]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), std_floor)
        if scores[order[0]] > best_score:
            best_score = float(scores[order[0]])
            best_unit = samples[order[0]].copy()
            stale = 0
        else:
            stale += 1
        entry = {"generation": gen,
                 "best": float(scores[order[0]]),
                 "mean": float(scores.mean()),
                 "overall_best": best_score}
        history.append(entry)
        if log is not None:
            log(entry)
        if patience and stale >= patience:
            break
    return CEMResult(
        best_unit=best_unit, best_score=best_score, history=history,
        best_forces=decode_genes(genes, best_unit))
