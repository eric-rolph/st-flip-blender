"""Explicit CUDA compute smoke for ST-FLIP.

On ordinary CPU machines this command reports a machine-readable skip and exits
successfully.  GPU validation jobs must pass ``--require-gpu`` so a missing or
broken CUDA runtime is a failure rather than being mistaken for coverage::

    python tools/cuda_smoke.py --require-gpu --output cuda-smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # ``python tools/cuda_smoke.py`` otherwise exposes only ``tools/`` on
    # sys.path, so the repository's source package cannot be imported.
    sys.path.insert(0, str(ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="exit nonzero when CUDA preflight cannot execute real GPU work",
    )
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    return parser


def _execute_cuda_solver_smoke() -> dict:
    from stflip import Params, STFLIPSolver
    from stflip.backend import cuda_device_name

    params = Params(
        resolution=(8, 8, 8),
        dx=1.0 / 8.0,
        gravity=(0.0, 0.0, -1.0),
        frame_dt=1.0 / 24.0,
        cfl_target=4.0,
        cfl_local=1.0,
        particles_per_cell=1,
        seed=23,
        pcg_tol=1e-5,
        pcg_max_iter=120,
    )
    liquid = np.zeros(params.resolution, dtype=bool)
    liquid[1:4, 2:5, 1:4] = True

    solvers = {
        name: STFLIPSolver(params, name) for name in ("cpu", "cuda")
    }
    for solver in solvers.values():
        solver.add_liquid_mask(liquid, velocity=(0.35, 0.05, 0.0))
        solver.step_frame()
        solver.be.synchronize()

    arrays = {}
    for name, solver in solvers.items():
        arrays[name] = tuple(
            np.asarray(solver.be.to_numpy(value), dtype=np.float32)
            for value in (solver.pos, solver.vel, solver.dt_resid)
        )
        if not all(np.all(np.isfinite(value)) for value in arrays[name]):
            raise AssertionError(f"{name} solver produced non-finite state")

    position_error = float(np.max(np.abs(arrays["cuda"][0] - arrays["cpu"][0])))
    velocity_error = float(np.max(np.abs(arrays["cuda"][1] - arrays["cpu"][1])))
    residual_error = float(np.max(np.abs(arrays["cuda"][2] - arrays["cpu"][2])))
    if position_error > 2e-4 or velocity_error > 2e-4 or residual_error > 2e-6:
        raise AssertionError(
            "CPU/CUDA smoke parity exceeded tolerance: "
            f"position={position_error:.3g}, velocity={velocity_error:.3g}, "
            f"residual={residual_error:.3g}"
        )

    return {
        "backend": solvers["cuda"].be.name,
        "device": cuda_device_name() or "CUDA device 0",
        "particles": int(arrays["cuda"][0].shape[0]),
        "position_max_abs_error": position_error,
        "velocity_max_abs_error": velocity_error,
        "residual_max_abs_error": residual_error,
    }


def run_cuda_smoke(
    *,
    require_gpu: bool,
    diagnostics: Callable[..., tuple[bool, str]] | None = None,
    execute: Callable[[], dict] | None = None,
) -> tuple[dict, int]:
    """Run CUDA preflight/solver work, or explicitly skip/fail if unavailable."""
    if diagnostics is None:
        from stflip.backend import cuda_diagnostics

        diagnostics = cuda_diagnostics
    if execute is None:
        execute = _execute_cuda_solver_smoke

    available, reason = diagnostics(force=True)
    if not available:
        return {
            "status": "failed" if require_gpu else "skipped",
            "required": bool(require_gpu),
            "reason": str(reason),
        }, 1 if require_gpu else 0

    details = execute()
    return {
        "status": "passed",
        "required": bool(require_gpu),
        "reason": str(reason),
        **details,
    }, 0


def _emit(result: dict, output: Path | None) -> None:
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if output is not None:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(sys.argv[1:] if argv is None else list(argv))
    try:
        result, code = run_cuda_smoke(require_gpu=options.require_gpu)
    except Exception as exc:
        result = {
            "status": "failed",
            "required": bool(options.require_gpu),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _emit(result, options.output)
        traceback.print_exc()
        return 1
    _emit(result, options.output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
