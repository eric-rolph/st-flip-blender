"""Run the compact, bpy-free paper-reference benchmark scenes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stflip.paper_validation import (  # noqa: E402
    DEFAULT_GLUG_SURFACE_TENSION,
    PaperRunConfig,
    run_glug_benchmark,
    run_kleefsman_benchmark,
    write_paper_artifact,
)


def _finite_nonnegative(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible Kleefsman-gauge and paper-constrained glug ST-FLIP "
            "reference scenes without claiming unavailable PF-FLIP equivalence."
        ))
    parser.add_argument("--case", choices=("kleefsman", "glug"), required=True)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--dx", type=float, default=None)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--cfl", type=float, default=8.0)
    parser.add_argument("--particles-per-cell", type=int, default=2)
    parser.add_argument("--gas-particles-per-cell", type=int, default=2)
    parser.add_argument(
        "--glug-surface-tension",
        type=_finite_nonnegative,
        default=DEFAULT_GLUG_SURFACE_TENSION,
        help=(
            "assumed glug water-air coefficient in N/m; the paper says it was "
            "enabled but does not publish a value"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--pressure-solver", choices=("jacobi", "multigrid"),
        default="multigrid")
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--reference-citation")
    parser.add_argument("--require-reference", action="store_true")
    parser.add_argument("--max-gauge-rmse", type=_finite_nonnegative)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.case != "kleefsman" and (
            args.reference_csv is not None or args.reference_citation is not None
            or args.require_reference or args.max_gauge_rmse is not None):
        raise SystemExit("reference options apply only to --case kleefsman")
    if (args.case != "glug"
            and args.glug_surface_tension != DEFAULT_GLUG_SURFACE_TENSION):
        raise SystemExit("--glug-surface-tension applies only to --case glug")
    if args.require_reference and args.reference_csv is None:
        raise SystemExit("--require-reference needs --reference-csv")
    if args.max_gauge_rmse is not None and args.reference_csv is None:
        raise SystemExit("--max-gauge-rmse needs --reference-csv")
    if args.reference_csv is None and args.reference_citation is not None:
        raise SystemExit("--reference-citation needs --reference-csv")
    if (args.reference_csv is not None
            and (args.reference_citation is None
                 or not args.reference_citation.strip())):
        raise SystemExit(
            "--reference-csv needs a nonblank --reference-citation")
    config = PaperRunConfig(
        longest_resolution=args.resolution,
        dx=args.dx,
        frames=args.frames,
        frame_rate=args.frame_rate,
        cfl_target=args.cfl,
        particles_per_cell=args.particles_per_cell,
        gas_particles_per_cell=args.gas_particles_per_cell,
        glug_surface_tension=args.glug_surface_tension,
        seed=args.seed,
        backend=args.backend,
        pressure_solver=args.pressure_solver,
    )
    artifact = (
        run_kleefsman_benchmark(
            config,
            reference_csv=args.reference_csv,
            reference_citation=args.reference_citation,
        ) if args.case == "kleefsman" else run_glug_benchmark(config)
    )
    output = write_paper_artifact(args.output, artifact)
    summary = {
        "artifact": str(output),
        "case": args.case,
        "grid": artifact["grid"],
        "reference_compared": artifact.get("comparison") is not None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.max_gauge_rmse is not None:
        failures = {
            gauge: values["rmse_m"]
            for gauge, values in artifact["comparison"].items()
            if values["rmse_m"] > args.max_gauge_rmse
        }
        if failures:
            print(json.dumps({"rmse_failures": failures}, sort_keys=True),
                  file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
