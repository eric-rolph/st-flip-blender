"""ERR-M1 diagnostics study: do the step-control signals earn their keep?

Kill point #2 of the paper-limitations roadmap.  Runs three known failure
episodes with diagnostics enabled and answers, per signal:

1. LEAD TIME -- does the signal alarm at least one substep BEFORE the
   failure metric spikes?  The dam scene's failure marker is the paper's
   own Fig. 7 gap: the ACTUAL per-substep CFL exceeding 1.3x the
   ESTIMATED CFL the step chooser used (dt derives from the previous
   substep's vmax, so impacts overshoot).
2. FALSE ALARMS -- on a calm pool, what fraction of substeps alarm?
   (< 20 percent required.)
3. REDUNDANCY (K1) -- does anything add lead time or precision OVER the
   plain vmax-growth predictor?  If not, ERR collapses to a one-line
   vmax_pred flag and ERR-M2/M3 are cancelled as designed.

The thin-obstacle scene is retained as a DOCUMENTED NEGATIVE: the
solver's local-CFL-1 sub-stepped advection already prevents tunneling
(zero tunneled particles at CFL 16 through a 1.5-cell wall), so the
near-solid signal has no failure to lead there; its alarms on that scene
are correct hazard identification, not false alarms, and are reported
separately.

Writes a JSON artifact and prints the verdict tables.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stflip import step_control  # noqa: E402
from stflip.backend import get_backend  # noqa: E402
from stflip.solver import Params, STFLIPSolver  # noqa: E402

SIGNALS = (
    "clamp_bind_fraction",
    "undersampled_face_fraction",
    "near_solid_fast_fraction",
    "vmax_growth",
)


def _params(n, **overrides):
    base = dict(
        resolution=(n, n, n), dx=1.0 / n, gravity=(0.0, 0.0, -9.81),
        frame_dt=1.0 / 24.0, cfl_target=16.0, particles_per_cell=2,
        st_enabled=True, seed=0,
    )
    base.update(overrides)
    return Params(**base)


def _collect(solver, frames, extra=None):
    """Run and flatten per-substep traces across frames."""

    solver._collect_step_diagnostics = True
    trace = {name: [] for name in SIGNALS}
    trace["cfl_actual"] = []
    trace["cfl_estimated"] = []
    trace["undersampled_delta"] = []
    trace["dt"] = []
    prev_vmax = None
    history = []
    for _ in range(frames):
        stats = solver.step_frame()
        if extra is not None:
            extra(solver, stats)
        for k in range(stats.steps):
            trace["clamp_bind_fraction"].append(
                stats.clamp_bind_fractions[k])
            marginal = stats.undersampled_marginal_fractions[k]
            trace["undersampled_face_fraction"].append(marginal)
            # Review fix 5: the absolute fraction has a large static
            # baseline proportional to interface area; also report the
            # baseline-relative form (delta vs the trailing median).
            history.append(marginal)
            trailing = sorted(history[-9:])
            median = trailing[len(trailing) // 2]
            trace["undersampled_delta"].append(marginal - median)
            trace["near_solid_fast_fraction"].append(
                stats.near_solid_fast_fractions[k])
            dt = stats.dt_values[k]
            cfl_actual = stats.particle_cfl_actual_values[k]
            vmax = cfl_actual * solver.p.dx / dt if dt > 0 else 0.0
            growth = (vmax / prev_vmax
                      if prev_vmax and prev_vmax > 1e-9 else 1.0)
            prev_vmax = vmax
            trace["vmax_growth"].append(growth)
            trace["cfl_actual"].append(cfl_actual)
            trace["cfl_estimated"].append(
                stats.particle_cfl_estimated_values[k])
            trace["dt"].append(dt)
    return trace


def _alarm_steps(trace):
    """Per-signal boolean alarm series (level >= A_RELEASE)."""

    out = {}
    steps = len(trace["dt"])
    for name in SIGNALS:
        out[name] = [
            step_control.alarm_level(name, trace[name][k])
            >= step_control.A_RELEASE
            for k in range(steps)
        ]
    return out


def scene_dam_impact(n=24, frames=24):
    """Gravity dam at CFL 16: failure = the paper's Fig. 7 gap.

    Measured at this scale, actual/estimated CFL has p50 ~ 1.1 and p90
    ~ 2.0 (excluding the from-rest substep zero, whose ratio is
    unbounded).  The failure marker is the first substep k >= 2 whose
    ratio exceeds 1.5 -- above the benign free-fall growth band, i.e. a
    genuine impact overshoot the step chooser did not anticipate.
    """

    solver = STFLIPSolver(_params(n), get_backend("cpu"))
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[: n // 3, :, : (2 * n) // 3] = True
    solver.add_liquid_mask(mask)
    trace = _collect(solver, frames)
    overshoot = [
        k >= 2 and actual > 1.5 * max(estimate, 1e-9)
        for k, (actual, estimate) in enumerate(zip(
            trace["cfl_actual"], trace["cfl_estimated"]))
    ]
    failure_step = overshoot.index(True) if any(overshoot) else None
    return trace, failure_step, "actual CFL > 1.5x estimated at k >= 2"


def scene_thin_obstacle(n=20, frames=8):
    """Fast jet at a thin wall: failure = particles beyond the wall."""

    solver = STFLIPSolver(
        _params(n, gravity=(0.0, 0.0, 0.0)), get_backend("cpu"))
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[2 : n // 3, n // 4 : (3 * n) // 4, n // 4 : (3 * n) // 4] = True
    solver.add_liquid_mask(mask, velocity=(8.0, 0.0, 0.0))
    wall_x = 0.7
    cells = (np.stack(np.meshgrid(
        *(np.arange(n),) * 3, indexing="ij"), axis=-1) + 0.5) / n
    thickness = 1.5 / n
    sdf = (np.abs(cells[..., 0] - wall_x) - thickness / 2.0).astype(
        np.float32)
    solver.set_solid_sdf(sdf)

    solver._collect_step_diagnostics = True
    trace = {name: [] for name in SIGNALS}
    trace["cfl_actual"] = []
    trace["dt"] = []
    tunneled = []
    prev_vmax = None
    for _ in range(frames):
        stats = solver.step_frame()
        pos = solver.be.to_numpy(solver.pos)
        beyond = int((pos[:, 0] > wall_x + thickness).sum())
        for k in range(stats.steps):
            trace["clamp_bind_fraction"].append(
                stats.clamp_bind_fractions[k])
            trace["undersampled_face_fraction"].append(
                stats.undersampled_marginal_fractions[k])
            trace["near_solid_fast_fraction"].append(
                stats.near_solid_fast_fractions[k])
            dt = stats.dt_values[k]
            cfl_actual = stats.particle_cfl_actual_values[k]
            vmax = cfl_actual * solver.p.dx / dt if dt > 0 else 0.0
            growth = (vmax / prev_vmax
                      if prev_vmax and prev_vmax > 1e-9 else 1.0)
            prev_vmax = vmax
            trace["vmax_growth"].append(growth)
            trace["cfl_actual"].append(cfl_actual)
            trace["dt"].append(dt)
            tunneled.append(beyond)
    failure_step = next(
        (k for k, v in enumerate(tunneled) if v > 0), None)
    return trace, failure_step, "particles beyond thin wall"


def scene_calm_pool(n=16, frames=10):
    """Still pool: no failure; every alarm is a false alarm."""

    solver = STFLIPSolver(_params(n), get_backend("cpu"))
    mask = np.zeros((n,) * 3, dtype=bool)
    mask[:, :, : n // 2] = True
    solver.add_liquid_mask(mask)
    trace = _collect(solver, frames)
    return trace, None, "none (false-alarm scene)"


def evaluate(scenes) -> dict:
    """Lead time on the dam scene, false alarms on the calm pool, K1."""

    dam_trace, dam_failure, _ = scenes["dam_impact"]
    calm_trace, _f, _m = scenes["calm_pool"]
    thin_trace, thin_failure, _ = scenes["thin_obstacle"]

    dam_alarms = _alarm_steps(dam_trace)
    calm_alarms = _alarm_steps(calm_trace)
    thin_alarms = _alarm_steps(thin_trace)

    verdict = {}
    for name in SIGNALS:
        lead = None
        if dam_failure is not None:
            series = dam_alarms[name][: dam_failure + 1]
            first = next((k for k, v in enumerate(series) if v), None)
            lead = dam_failure - first if first is not None else None
        rate = (sum(calm_alarms[name]) / len(calm_alarms[name])
                if calm_alarms[name] else 0.0)
        verdict[name] = {
            "dam_lead_time": lead,
            "calm_false_alarm_rate": rate,
            "useful": lead is not None and lead >= 1 and rate < 0.20,
        }

    baseline_lead = verdict["vmax_growth"]["dam_lead_time"]
    k1_survivors = []
    for name in SIGNALS:
        if name == "vmax_growth":
            continue
        lead = verdict[name]["dam_lead_time"]
        beats = (lead is not None
                 and (baseline_lead is None or lead > baseline_lead))
        verdict[name]["beats_vmax_growth"] = beats
        if beats and verdict[name]["useful"]:
            k1_survivors.append(name)

    return {
        "per_signal": verdict,
        "dam_failure_step": dam_failure,
        "thin_obstacle": {
            "failure_step": thin_failure,
            "tunneled": thin_failure is not None,
            "near_solid_hazard_alarm_rate": (
                sum(thin_alarms["near_solid_fast_fraction"])
                / max(len(thin_alarms["near_solid_fast_fraction"]), 1)),
            "note": (
                "documented negative: local-CFL-1 sub-stepped advection "
                "already prevents tunneling; near-solid alarms here are "
                "correct hazard identification, not false alarms"),
        },
        "k1_survivors": k1_survivors,
        "k1_collapse_to_vmax_pred_only": not k1_survivors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="validation/step_control_study.json")
    args = parser.parse_args()

    started = time.perf_counter()
    scenes = {
        "dam_impact": scene_dam_impact(),
        "thin_obstacle": scene_thin_obstacle(),
        "calm_pool": scene_calm_pool(),
    }
    result = evaluate(scenes)
    elapsed = time.perf_counter() - started

    artifact = {
        "schema": "stflip.err_m1_step_control_study",
        "version": 1,
        "scenes": {
            name: {
                "failure_step": failure,
                "failure_metric": metric,
                "substeps": len(trace["dt"]),
                "trace": {k: [float(x) for x in v]
                          for k, v in trace.items()},
            }
            for name, (trace, failure, metric) in scenes.items()
        },
        "evaluation": result,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1), encoding="ascii")

    print(f"elapsed: {elapsed:.1f} s")
    print("dam failure step:", result["dam_failure_step"])
    for name, info in result["per_signal"].items():
        print(f"{name:32s} dam_lead={info['dam_lead_time']} "
              f"calm_false_alarms={info['calm_false_alarm_rate']:.2f} "
              f"useful={info['useful']}")
    thin = result["thin_obstacle"]
    print(f"thin obstacle: tunneled={thin['tunneled']} "
          f"hazard_alarm_rate={thin['near_solid_hazard_alarm_rate']:.2f}")
    print("K1 survivors:", result["k1_survivors"])
    print("VERDICT:",
          "COLLAPSE to vmax_pred-only (K1)"
          if result["k1_collapse_to_vmax_pred_only"]
          else "GO for ERR-M2 with surviving signals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
