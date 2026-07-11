"""Run the matched, bpy-free ST-FLIP validation matrix.

Examples::

    python tools/run_validation.py
    python tools/run_validation.py --scale evidence --require-high-cfl
        --require-coherence-improvement
    python tools/run_validation.py --backend cuda --output validation/cuda.json

The CPU matrix is the mechanism comparison.  A CUDA invocation is useful for
fixed-configuration parity and timing, but must not be presented as evidence
that temporal sampling itself improved the solution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stflip.validation import (  # noqa: E402
    ValidationConfig,
    run_matched_validation,
    run_multi_seed_validation,
    write_validation_artifact,
)


SCALES = {
    "quick": {"resolution": 16, "frames": 4, "particles_per_cell": 2},
    # The paper's standard and this add-on's production default is 8 PPC.
    # Lower counts are useful stress tests but materially under-sample the
    # temporal Monte Carlo estimator and are not publication evidence.
    "evidence": {"resolution": 48, "frames": 10, "particles_per_cell": 8},
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ST-FLIP and instantaneous-P2G at matched CFL 1/16 on one "
            "backend, producing a strict JSON artifact."
        )
    )
    parser.add_argument(
        "--scale", choices=tuple(SCALES), default="quick",
        help="quick developer run or multi-seed evidence-scale dam break",
    )
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--particles-per-cell", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help=(
            "run a multi-seed robustness artifact (for example: "
            "--seeds 0 1 2); overrides --seed"
        ),
    )
    parser.add_argument(
        "--backend", choices=("cpu", "cuda"), default="cpu",
        help=(
            "keep cpu for the method comparison; use cuda separately for "
            "fixed-configuration backend parity/timing"
        ),
    )
    parser.add_argument("--high-cfl-threshold", type=float, default=8.0)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--gravity-z", type=float, default=-9.81)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/stflip_matched_validation.json"),
    )
    parser.add_argument(
        "--require-high-cfl",
        action="store_true",
        help="exit nonzero unless both target-CFL-16 runs achieve the threshold",
    )
    parser.add_argument(
        "--require-coherence-improvement",
        "--require-quality-improvement",
        dest="require_coherence_improvement",
        action="store_true",
        help=(
            "exit nonzero unless the internal Eq. 13 phase-RMSE and "
            "threshold-IoU coherence gate favors ST CFL-16; the older "
            "--require-quality-improvement spelling is retained as an alias"
        ),
    )
    parser.add_argument(
        "--require-validation-ready",
        action="store_true",
        help=(
            "exit nonzero unless core quadrature/residual checks pass, both "
            "high-target cases reach the observed-CFL threshold, and the "
            "internal phase-field coherence gate passes"
        ),
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    selected = SCALES[args.scale]
    config = ValidationConfig(
        resolution=(args.resolution if args.resolution is not None
                    else selected["resolution"]),
        frames=(args.frames if args.frames is not None
                else selected["frames"]),
        particles_per_cell=(
            args.particles_per_cell
            if args.particles_per_cell is not None
            else selected["particles_per_cell"]),
        seed=args.seed,
        backend=args.backend,
        high_cfl_threshold=args.high_cfl_threshold,
        frame_rate=args.frame_rate,
        gravity_z=args.gravity_z,
    )
    artifact = (
        run_multi_seed_validation(config, args.seeds)
        if args.seeds is not None
        else run_matched_validation(config)
    )
    output = write_validation_artifact(args.output, artifact)
    if args.seeds is not None:
        aggregate = artifact["aggregate"]
        summary = {
            "artifact": output,
            "backend": config.backend,
            "seeds": artifact["seeds"],
            "high_cfl_reached_all": aggregate["high_cfl_reached_all"],
            "core_checks_passed_all": aggregate["core_checks_passed_all"],
            "primary_quality_measure": aggregate["primary_measure"],
            "st_wins": aggregate["st_wins"],
            "phase_threshold_iou_st_wins": aggregate[
                "phase_threshold_iou_st_wins"],
            "seed_count": aggregate["seed_count"],
            "mean_st_error_is_lower": aggregate["mean_st_error_is_lower"],
            "mean_phase_threshold_iou_is_higher": aggregate[
                "mean_phase_threshold_iou_is_higher"],
            "raw_mass_diagnostic": aggregate["raw_mass_diagnostic"],
            "validation_ready": aggregate["validation_ready"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.require_high_cfl and not aggregate["high_cfl_reached_all"]:
            return 2
        if (args.require_coherence_improvement
                and not aggregate["internal_coherence_improved"]):
            return 3
        if args.require_validation_ready and not aggregate["validation_ready"]:
            return 4
        return 0

    high_cfl = artifact["acceptance"]["high_cfl"]
    summary = {
        "artifact": output,
        "backend": config.backend,
        "same_initial_shared_state": artifact["acceptance"][
            "same_initial_shared_state"],
        "complete_checkpoints_match_within_cfl": artifact["acceptance"][
            "complete_checkpoints_match_within_cfl"],
        "high_cfl_reached": high_cfl["reached"],
        "actual_high_cfl_max": high_cfl["actual_max_by_case"],
        "primary_quality_measure": artifact["quality"]["primary"]["measure"],
        "st_has_lower_primary_error": artifact["quality"]["primary"][
            "st_has_lower_error"],
        "eq7_8_temporal_quadrature_passed": artifact["acceptance"][
            "eq7_8_temporal_quadrature_passed"],
        "phase_threshold_iou_is_higher": artifact["quality"][
            "secondary_evidence"]["phase_threshold_iou_mean"][
                "st_has_higher_value"],
        "raw_mass_diagnostic": artifact["quality"]["secondary_evidence"][
            "normalized_deposited_mass_rmse_mean"],
        "validation_ready": artifact["acceptance"]["validation_ready"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_high_cfl and not high_cfl["reached"]:
        return 2
    if (args.require_coherence_improvement
            and not artifact["acceptance"][
                "internal_coherence_improved"]):
        return 3
    if (args.require_validation_ready
            and not artifact["acceptance"]["validation_ready"]):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
