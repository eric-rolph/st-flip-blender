"""Pure regression tests for Blender add-on GPU setup decisions.

The production module is loaded with a very small fake ``bpy`` surface so
these tests remain runnable in ordinary CPython.  Blender operations are not
executed here; only deterministic helper logic is exercised.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_operators(monkeypatch, tmp_path):
    root = "_stflip_addon_test"
    for name in (root, f"{root}.addon", f"{root}.stflip"):
        monkeypatch.setitem(sys.modules, name, _package(name))

    bpy = types.ModuleType("bpy")
    bpy.types = types.SimpleNamespace(Operator=object)
    bpy.utils = types.SimpleNamespace(
        user_resource=lambda *args, **kwargs: str(tmp_path),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    cache = types.ModuleType(f"{root}.stflip.cache")
    backend = types.ModuleType(f"{root}.stflip.backend")
    backend.cuda_available = lambda: False
    backend.cuda_device_name = lambda: None
    backend.cuda_diagnostics = lambda *args, **kwargs: {
        "available": False,
        "error": "not installed",
    }
    backend.get_backend = lambda name="auto": None
    solver = types.ModuleType(f"{root}.stflip.solver")
    solver.Params = object
    solver.STFLIPSolver = object
    monkeypatch.setitem(sys.modules, cache.__name__, cache)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setitem(sys.modules, solver.__name__, solver)

    for leaf in ("handlers", "mesher", "voxelize"):
        module = types.ModuleType(f"{root}.addon.{leaf}")
        if leaf == "handlers":
            module.resolve_cache_dir = lambda scene: ""
        monkeypatch.setitem(sys.modules, module.__name__, module)

    path = Path(__file__).parents[1] / "addon" / "operators.py"
    name = f"{root}.addon.operators"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gpu_install_candidates_are_pinned_and_ordered(monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)

    requirements = [item["requirement"]
                    for item in operators.GPU_INSTALL_CANDIDATES]
    assert requirements == [
        "cupy-cuda13x[ctk]==14.1.1",
        "cupy-cuda12x==14.1.1",
    ]
    assert len({item["slug"] for item in operators.GPU_INSTALL_CANDIDATES}) == 2


def test_candidate_cleanup_removes_only_shadow_numpy(monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    candidate = tmp_path / "runtime" / "candidate"
    for name in ("numpy", "numpy.libs", "numpy-2.5.1.dist-info", "cupy"):
        (candidate / name).mkdir(parents=True)
        (candidate / name / "sentinel.txt").write_text(name)

    removed = operators.remove_shadow_numpy(candidate)

    assert set(removed) == {"numpy", "numpy.libs", "numpy-2.5.1.dist-info"}
    assert not (candidate / "numpy").exists()
    assert not (candidate / "numpy.libs").exists()
    assert not (candidate / "numpy-2.5.1.dist-info").exists()
    assert (candidate / "cupy" / "sentinel.txt").read_text() == "cupy"


def test_cuda_runtime_cleanup_preserves_active_candidate(monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    root = tmp_path / "stflip_cuda_runtime"
    active = root / "cuda13-active"
    obsolete = root / "cuda12-obsolete"
    installing = root / "cuda13-installing"
    active.mkdir(parents=True)
    obsolete.mkdir()
    installing.mkdir()
    (active / "keep.txt").write_text("active")
    (obsolete / "large-wheel.bin").write_bytes(b"old")
    (installing / operators._INSTALLING_RUNTIME_FILE).write_text("pid=123")

    removed = operators.cleanup_inactive_cuda_runtimes(active)

    assert removed == [obsolete.name]
    assert (active / "keep.txt").read_text() == "active"
    assert not obsolete.exists()
    assert installing.exists()


def test_memory_estimate_allows_128_but_rejects_predictable_512_oom(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    gib = 1024 ** 3

    normal = operators.estimate_bake_memory((128, 128, 128), 8)
    huge = operators.estimate_bake_memory((512, 512, 512), 8)

    assert huge["working_set_bytes"] > normal["working_set_bytes"] * 50
    assert operators.memory_guard_reason(
        normal, backend_name="cpu", ram_available=16 * gib,
        vram_available=None,
    ) is None
    reason = operators.memory_guard_reason(
        huge, backend_name="cpu", ram_available=64 * gib,
        vram_available=None,
    )
    assert reason is not None
    assert "RAM" in reason
    assert "512" in reason


def test_memory_guard_uses_vram_for_cuda_without_blocking_normal_gpu(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)
    gib = 1024 ** 3
    estimate = operators.estimate_bake_memory((128, 128, 128), 8)

    assert operators.memory_guard_reason(
        estimate, backend_name="cuda", ram_available=32 * gib,
        vram_available=16 * gib,
    ) is None
    reason = operators.memory_guard_reason(
        estimate, backend_name="cuda", ram_available=32 * gib,
        vram_available=2 * gib,
    )
    assert reason is not None
    assert "VRAM" in reason


def test_normalize_cuda_diagnostics_accepts_mapping_and_object(
        monkeypatch, tmp_path):
    operators = _load_operators(monkeypatch, tmp_path)

    mapping = operators.normalize_cuda_diagnostics({
        "available": True,
        "device_name": "RTX Test",
        "free_bytes": 12,
        "total_bytes": 20,
    })
    assert mapping == {
        "available": True,
        "device": "RTX Test",
        "free_bytes": 12,
        "total_bytes": 20,
        "error": "",
    }

    diagnostic = types.SimpleNamespace(
        ok=False,
        device="Unavailable GPU",
        memory_free=3,
        memory_total=5,
        message="kernel preflight failed",
    )
    normalized = operators.normalize_cuda_diagnostics(diagnostic)
    assert normalized["available"] is False
    assert normalized["device"] == "Unavailable GPU"
    assert normalized["free_bytes"] == 3
    assert normalized["total_bytes"] == 5
    assert normalized["error"] == "kernel preflight failed"
