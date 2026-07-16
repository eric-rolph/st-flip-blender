"""GOAL-M1 gate studies: goal-directed splash via CEM over body forces.

Runs the pre-registered gates from docs/design/goal-directed-splash.md:

  G-CTRL-1  mechanism: 32^3 dam break, one DIRECTIONAL force, optimized
            mass-in-target >= 3x baseline across 3 optimizer seeds.
  G-CTRL-2  unreachable goal: elevated catch basin the uncontrolled
            flow misses (baseline < 0.005) reached at >= 0.05 with the
            keep-dry zone satisfied.
  G-CTRL-3  transfer: 48^3-optimized parameters retain >= 60 percent of
            the objective improvement at 128^3.
  G-CTRL-4  budget: the demo optimization completes in <= 30 min.

Usage: uv run ... python tools/run_goal_splash.py --backend cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stflip.backend import get_backend  # noqa: E402
from stflip.control import (  # noqa: E402
    ForceGene,
    Objective,
    genome_dim,
    optimize_forces,
    rollout_score,
)
from stflip.solver import Params, STFLIPSolver  # noqa: E402

FRAME = 36
TARGET_BOX = ((0.70, 0.25, 0.30), (1.00, 0.75, 0.70))   # elevated basin
DRY_BOX = ((0.70, 0.0, 0.0), (1.00, 1.0, 0.18))         # protected floor
GENES = [ForceGene(
    force_type="DIRECTIONAL", strength=(0.0, 30.0),
    direction=((0.1, 1.0), (-0.4, 0.4), (0.0, 0.9)),
    window=((0.0, 1.0), (0.1, 1.5)))]


def build_solver_factory(n, backend_kind, seed=0):
    def build():
        params = Params(
            resolution=(n, n, n), dx=1.0 / n,
            gravity=(0.0, 0.0, -9.81), frame_dt=1.0 / 24.0,
            cfl_target=8.0, particles_per_cell=2, st_enabled=True,
            seed=seed, pressure_solver="multigrid",
            advection_bound="local")
        solver = STFLIPSolver(params, get_backend(backend_kind))
        mask = np.zeros((n,) * 3, dtype=bool)
        mask[: n // 3, :, : n // 2] = True
        solver.add_liquid_mask(mask)
        return solver
    return build


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--output",
                        default="validation/goal_splash_demo.json")
    args = parser.parse_args()
    started = time.perf_counter()
    objective = Objective(
        targets=[(FRAME, TARGET_BOX, 1.0)],
        keep_dry=[(FRAME, DRY_BOX, 0.5)])
    target_only = Objective(targets=[(FRAME, TARGET_BOX, 1.0)])
    dry_only = Objective(targets=[(FRAME, DRY_BOX, 1.0)])

    # ---- G-CTRL-1 / G-CTRL-2 at 32^3, three optimizer seeds ----------
    build32 = build_solver_factory(32, args.backend)
    zero = np.zeros(genome_dim(GENES))
    baseline32 = rollout_score(build32, GENES, zero, target_only)
    runs = []
    for opt_seed in (0, 1, 2):
        result = optimize_forces(
            build32, GENES, objective, generations=10, population=12,
            seed=opt_seed,
            log=lambda e: print(json.dumps(e), flush=True))
        target_score = rollout_score(
            build32, GENES, result.best_unit, target_only)
        dry_mass = rollout_score(
            build32, GENES, result.best_unit, dry_only)
        runs.append({"opt_seed": opt_seed,
                     "objective": result.best_score,
                     "target_mass": target_score,
                     "dry_mass": dry_mass,
                     "forces": result.best_forces,
                     "history": result.history})
        print(json.dumps({"seed_done": opt_seed,
                          "target": target_score,
                          "baseline": baseline32}), flush=True)
    g1 = all(r["target_mass"] >= max(3.0 * baseline32, 1e-9)
             for r in runs)
    g2 = (baseline32 < 0.005
          and all(r["target_mass"] >= 0.05 for r in runs)
          and all(r["dry_mass"] <= max(2.0 * r["target_mass"], 0.10)
                  for r in runs))

    # ---- G-CTRL-3 transfer: optimize at 48^3, verify at 128^3 --------
    build48 = build_solver_factory(48, args.backend)
    baseline48 = rollout_score(build48, GENES, zero, target_only)
    result48 = optimize_forces(
        build48, GENES, objective, generations=10, population=12,
        seed=0, log=lambda e: print(json.dumps(e), flush=True))
    opt48 = rollout_score(build48, GENES, result48.best_unit,
                          target_only)
    build128 = build_solver_factory(128, args.backend)
    baseline128 = rollout_score(build128, GENES, zero, target_only)
    opt128 = rollout_score(build128, GENES, result48.best_unit,
                           target_only)
    gain_proxy = opt48 - baseline48
    gain_hero = opt128 - baseline128
    retention = gain_hero / gain_proxy if gain_proxy > 1e-9 else 0.0
    g3 = retention >= 0.60
    elapsed = time.perf_counter() - started
    g4 = elapsed <= 1800.0

    artifact = {
        "schema": "stflip.goal_m1_gates", "version": 1,
        "frame": FRAME, "target_box": TARGET_BOX, "dry_box": DRY_BOX,
        "baseline32_target_mass": baseline32,
        "runs32": runs,
        "transfer": {"baseline48": baseline48, "opt48": opt48,
                     "baseline128": baseline128, "opt128": opt128,
                     "retention": retention,
                     "forces": result48.best_forces},
        "elapsed_s": round(elapsed, 1),
        "gates": {"G-CTRL-1": g1, "G-CTRL-2": g2,
                  "G-CTRL-3": g3, "G-CTRL-4": g4},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1), encoding="ascii")
    print(json.dumps(artifact["gates"] | {
        "retention": round(retention, 3),
        "elapsed_s": round(elapsed, 1)}, indent=1))
    print("GOAL-GATES-DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
