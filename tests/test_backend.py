import sys
import types

import numpy as np
import pytest

from stflip import backend


def _fake_cupy(*, device_count=1, allocation_error=None):
    calls = {
        "device_count": 0,
        "allocations": 0,
        "host_transfers": 0,
        "scatter": 0,
        "synchronize": 0,
    }
    module = types.ModuleType("cupy")

    class Runtime:
        @staticmethod
        def getDeviceCount():
            calls["device_count"] += 1
            return device_count

        @staticmethod
        def getDeviceProperties(index):
            assert index == 0
            return {"name": b"Mock CUDA GPU"}

    class Stream:
        @staticmethod
        def synchronize():
            calls["synchronize"] += 1

    class Cuda:
        runtime = Runtime()

        @staticmethod
        def get_current_stream():
            return Stream()

    def asarray(value, dtype=None):
        calls["allocations"] += 1
        if allocation_error is not None:
            raise allocation_error
        return np.asarray(value, dtype=dtype)

    def asnumpy(value):
        calls["host_transfers"] += 1
        return np.asarray(value)

    def scatter_add(target, indices, values):
        calls["scatter"] += 1
        np.add.at(target, indices, values)

    module.cuda = Cuda()
    module.asarray = asarray
    module.asnumpy = asnumpy
    module.zeros = np.zeros
    module.float32 = np.float32
    module.int32 = np.int32
    module.add = types.SimpleNamespace(at=scatter_add)
    return module, calls


def test_cuda_diagnostics_runs_full_compute_preflight(monkeypatch):
    cupy, calls = _fake_cupy()
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(backend, "_CUDA_DIAGNOSTICS_CACHE", None, raising=False)

    available, reason = backend.cuda_diagnostics(force=True)

    assert available is True
    assert "Mock CUDA GPU" in reason
    assert "passed" in reason.lower()
    assert calls["device_count"] == 1
    assert calls["allocations"] >= 3
    assert calls["scatter"] == 1
    assert calls["synchronize"] == 1
    assert calls["host_transfers"] >= 2


def test_cuda_diagnostics_reports_actionable_allocation_failure(monkeypatch):
    cupy, _ = _fake_cupy(allocation_error=MemoryError("mock device OOM"))
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(backend, "_CUDA_DIAGNOSTICS_CACHE", None)

    available, reason = backend.cuda_diagnostics(force=True)

    assert available is False
    assert "allocation" in reason.lower()
    assert "mock device OOM" in reason
    assert "GPU memory" in reason
    assert backend.cuda_available() is False


def test_explicit_cuda_backend_fails_early_with_diagnostic(monkeypatch):
    reason = "CUDA allocation failed (out of memory). Check GPU memory."
    monkeypatch.setattr(
        backend, "cuda_diagnostics", lambda force=False: (False, reason))

    with pytest.raises(RuntimeError, match="CUDA backend unavailable") as exc:
        backend.get_backend("cuda")

    assert reason in str(exc.value)


def test_auto_backend_falls_back_to_cpu_after_failed_preflight(monkeypatch):
    monkeypatch.setattr(
        backend,
        "cuda_diagnostics",
        lambda force=False: (False, "CUDA kernel compilation failed"),
    )

    selected = backend.get_backend("auto")

    assert selected.name == "cpu"
    assert selected.is_gpu is False
