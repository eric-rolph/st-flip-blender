from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import blender_headless_smoke, cuda_smoke, run_paper_validation


def test_blender_script_arguments_use_only_values_after_separator():
    argv = [
        "blender",
        "--background",
        "--python",
        "tools/blender_headless_smoke.py",
        "--",
        "--require-openvdb",
        "--output",
        "result.json",
    ]

    assert blender_headless_smoke.script_arguments(argv) == [
        "--require-openvdb",
        "--output",
        "result.json",
    ]


def test_blender_script_arguments_support_direct_python_execution():
    assert blender_headless_smoke.script_arguments(
        ["blender_headless_smoke.py", "--output", "result.json"]
    ) == ["--output", "result.json"]


def test_openvdb_loader_falls_back_to_official_blender_module_name():
    expected = SimpleNamespace(name="pyopenvdb")

    def importer(name):
        if name == "openvdb":
            raise ModuleNotFoundError(name)
        assert name == "pyopenvdb"
        return expected

    name, module = blender_headless_smoke.load_openvdb(importer)

    assert name == "pyopenvdb"
    assert module is expected


def test_openvdb_loader_prefers_addon_native_module_name():
    expected = SimpleNamespace(name="openvdb")
    calls = []

    def importer(name):
        calls.append(name)
        return expected

    name, module = blender_headless_smoke.load_openvdb(importer)

    assert name == "openvdb"
    assert module is expected
    assert calls == ["openvdb"]


def test_openvdb_loader_reports_unavailable_without_masking_other_errors():
    def missing(_name):
        raise ModuleNotFoundError

    assert blender_headless_smoke.load_openvdb(missing) == (None, None)

    def broken(_name):
        raise RuntimeError("binary loader failed")

    try:
        blender_headless_smoke.load_openvdb(broken)
    except RuntimeError as exc:
        assert str(exc) == "binary loader failed"
    else:  # pragma: no cover - assertion spelling keeps this pytest-free
        raise AssertionError("non-import OpenVDB failure was swallowed")


def test_cuda_smoke_skips_unavailable_gpu_unless_required():
    optional, optional_code = cuda_smoke.run_cuda_smoke(
        require_gpu=False,
        diagnostics=lambda force: (False, "no NVIDIA device"),
        execute=lambda: (_ for _ in ()).throw(AssertionError("must not execute")),
    )
    required, required_code = cuda_smoke.run_cuda_smoke(
        require_gpu=True,
        diagnostics=lambda force: (False, "no NVIDIA device"),
        execute=lambda: (_ for _ in ()).throw(AssertionError("must not execute")),
    )

    assert optional == {
        "status": "skipped",
        "required": False,
        "reason": "no NVIDIA device",
    }
    assert optional_code == 0
    assert required == {
        "status": "failed",
        "required": True,
        "reason": "no NVIDIA device",
    }
    assert required_code == 1


def test_cuda_smoke_executes_only_after_compute_preflight_passes():
    calls = []

    def execute():
        calls.append("executed")
        return {"backend": "cuda", "device": "Test GPU"}

    result, code = cuda_smoke.run_cuda_smoke(
        require_gpu=True,
        diagnostics=lambda force: (True, "CUDA preflight passed on Test GPU"),
        execute=execute,
    )

    assert code == 0
    assert calls == ["executed"]
    assert result == {
        "status": "passed",
        "required": True,
        "reason": "CUDA preflight passed on Test GPU",
        "backend": "cuda",
        "device": "Test GPU",
    }


def test_cuda_script_runs_from_outside_repository(tmp_path):
    output = tmp_path / "cuda-smoke.json"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            sys.executable,
            str(cuda_smoke.__file__),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text("utf-8"))
    assert result["status"] in {"passed", "skipped"}
    assert result["required"] is False


def test_paper_cli_rejects_nonfinite_or_negative_metric_thresholds(tmp_path):
    base = [
        "--case", "kleefsman",
        "--output", str(tmp_path / "artifact.json"),
        "--reference-csv", str(tmp_path / "reference.csv"),
    ]
    for value in ("nan", "inf", "-0.1"):
        try:
            run_paper_validation.main(
                [*base, "--max-gauge-rmse", value])
        except SystemExit as exc:
            assert exc.code == 2
        else:  # pragma: no cover - keep this test independent of pytest helpers
            raise AssertionError(f"accepted invalid RMSE threshold {value}")


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--reference-citation", "Kleefsman et al."],
         "--reference-citation needs --reference-csv"),
        (["--reference-csv", "reference.csv"],
         "--reference-csv needs a nonblank --reference-citation"),
        (["--reference-csv", "reference.csv", "--reference-citation", "   "],
         "--reference-csv needs a nonblank --reference-citation"),
    ],
)
def test_paper_cli_requires_paired_reference_provenance(
        tmp_path, arguments, message):
    with pytest.raises(SystemExit, match=message):
        run_paper_validation.main([
            "--case", "kleefsman",
            "--frames", "0",
            "--output", str(tmp_path / "artifact.json"),
            *arguments,
        ])
