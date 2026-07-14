"""TIME-M1 estimator study: is a linear-in-tau slab fit worth building?

The kill point for roadmap initiative TIME.  Simulates the actual Eq. 10-11
jitter process (clamp included, abrupt-dt arm included), builds face-realistic
weighted sample sets (temporal kernel times separable spatial kernel weights),
and compares the ridge-regularised linear fit at the slab end against today's
one-sided W_T mean, both measured against the hypothesized target q(+1/2).

Gate matrix (repaired per the roadmap review):
- (s, gamma) cells are physically consistent -- the temporal signal s and the
  jitter attenuation gamma derive from the same velocity scale, so high-s
  cells run gamma ~ 1 and gamma-attenuated cells only occur at low s.
- GO requires ONE lam satisfying, at the gamma-active cells:
  MSE(fit) <= 0.95 * MSE(mean) at s = 2 and <= 0.80 at s >= 4, AND the
  universal no-regression bound MSE(fit) <= 1.05 * MSE(mean) at ALL cells.
- Gated at the face-realistic sample count (N = 64 raw, effective ~ 25-40
  after spatial weighting); the full N sweep is reported.

Writes a JSON artifact and prints a verdict; a NO-GO here cancels TIME-M2/M3
as designed (documented negative result; this tool and the library stay).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stflip import kernels, temporal_fit  # noqa: E402

S_VALUES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
N_VALUES = (8, 16, 32, 64)
LAM_VALUES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
GATE_N = 64
FACES = 4000
TAU_EVAL = 0.5


def gamma_for_signal(s: float) -> tuple[float, ...]:
    """Physically consistent attenuation levels for a given signal.

    gamma follows local CFL, which is the same velocity scale that creates
    temporal variation s over the slab: quiescent cells (s < 2) can sit
    anywhere in the attenuation range; fast cells (s >= 2) run gamma at or
    near one.
    """

    if s >= 2.0:
        return (0.5, 1.0)
    return (0.0, 0.25, 0.5, 1.0)


def build_cell(rng, s, gamma, n_samples, faces, curvature=0.0,
               dt_ratio=1.0):
    """Sampled estimates and truth for one (s, gamma, N) cell."""

    thetas = temporal_fit.simulate_jitter_taus(
        rng, faces * n_samples, 24, gamma, dt_ratio=dt_ratio)
    tau = thetas[-1].reshape(faces, n_samples)
    w_t = kernels.w_temporal(np, tau)
    offsets = rng.uniform(-1.0, 1.0, (faces, n_samples, 3))
    w_s = np.prod(kernels.w_spatial_1d(np, offsets), axis=-1)
    weights = w_t * w_s
    q = s * tau + curvature * (tau * tau - 1.0 / 12.0)
    truth = s * TAU_EVAL + curvature * (TAU_EVAL * TAU_EVAL - 1.0 / 12.0)
    moments = temporal_fit.accumulate_moments(np, tau, q, weights)
    ess = float(
        (weights.sum(axis=1) ** 2
         / np.maximum((weights ** 2).sum(axis=1), 1e-30)).mean())
    return moments, truth, ess


def mse(values, truth) -> float:
    err = np.asarray(values, dtype=np.float64) - truth
    return float(np.mean(err * err))


def run_study(seed: int = 0, curvature_arm: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    cells = []
    for s in S_VALUES:
        for gamma in gamma_for_signal(s):
            for n in N_VALUES:
                for curved in ((0.0, s) if curvature_arm else (0.0,)):
                    for dt_ratio in (1.0, 1.5):
                        moments, truth, ess = build_cell(
                            rng, s, gamma, n, FACES,
                            curvature=curved, dt_ratio=dt_ratio)
                        mean_mse = mse(temporal_fit.one_sided_mean(
                            np, moments), truth)
                        fit_mses = {}
                        for lam in LAM_VALUES:
                            value, _slope = temporal_fit.fit_evaluate(
                                np, moments, lam, tau_eval=TAU_EVAL)
                            fit_mses[lam] = mse(value, truth)
                        cells.append({
                            "s": s, "gamma": gamma, "n": n,
                            "curvature": curved, "dt_ratio": dt_ratio,
                            "effective_n": ess,
                            "mse_one_sided_mean": mean_mse,
                            "mse_fit_by_lam": fit_mses,
                        })
    verdict = evaluate_gates(cells)
    return {
        "schema": "stflip.time_m1_estimator_study",
        "version": 1,
        "seed": seed,
        "faces_per_cell": FACES,
        "tau_eval": TAU_EVAL,
        "cells": cells,
        "verdict": verdict,
    }


def evaluate_gates(cells) -> dict:
    """Find a single lam passing every roadmap gate at the gated N."""

    gated = [c for c in cells if c["n"] == GATE_N]
    results = {}
    for lam in LAM_VALUES:
        ok_s2 = all(
            c["mse_fit_by_lam"][lam] <= 0.95 * c["mse_one_sided_mean"]
            for c in gated if c["s"] == 2.0)
        ok_s4 = all(
            c["mse_fit_by_lam"][lam] <= 0.80 * c["mse_one_sided_mean"]
            for c in gated if c["s"] >= 4.0)
        no_regress = all(
            c["mse_fit_by_lam"][lam] <= 1.05 * c["mse_one_sided_mean"]
            + 1e-15
            for c in gated)
        results[lam] = {
            "s2_gain": ok_s2,
            "s4_gain": ok_s4,
            "no_regression": no_regress,
            "go": ok_s2 and ok_s4 and no_regress,
        }
    passing = [lam for lam, r in results.items() if r["go"]]
    return {
        "gate_n": GATE_N,
        "by_lam": {str(k): v for k, v in results.items()},
        "passing_lams": passing,
        "go": bool(passing),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", default="validation/temporal_study.json")
    args = parser.parse_args()

    started = time.perf_counter()
    artifact = run_study(seed=args.seed)
    elapsed = time.perf_counter() - started

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1), encoding="ascii")

    verdict = artifact["verdict"]
    print(f"cells: {len(artifact['cells'])}  elapsed: {elapsed:.1f} s")
    gate_cells = [c for c in artifact["cells"] if c["n"] == GATE_N]
    ess = sorted(c["effective_n"] for c in gate_cells)
    print(f"effective N at gate (min/median/max): "
          f"{ess[0]:.0f}/{ess[len(ess) // 2]:.0f}/{ess[-1]:.0f}")
    for lam, r in verdict["by_lam"].items():
        print(f"lam={lam:>5}: s2={r['s2_gain']} s4={r['s4_gain']} "
              f"no_regress={r['no_regression']} -> "
              f"{'GO' if r['go'] else 'no'}")
    print(f"VERDICT: {'GO' if verdict['go'] else 'NO-GO'} "
          f"(passing lams: {verdict['passing_lams']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
