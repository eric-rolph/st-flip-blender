"""Array-module backend abstraction.

The entire solver is written against the array-API-compatible subset shared by
NumPy and CuPy, so the same code runs on CPU (NumPy) and NVIDIA CUDA GPUs
(CuPy). CuPy's experimental ROCm source builds may work on Linux, but AMD GPU
setup is not integrated or tested; the portable fallback is NumPy.

Only two operations differ between the libraries and are wrapped here:
scatter-add (np.add.at vs cupyx.scatter_add) and host transfer.
"""

from __future__ import annotations

import os

import numpy as np

# Newest GPU architectures (e.g. Blackwell sm_120) may be newer than the CuPy
# wheel's NVRTC; PTX forward-compat JIT always works.  CuPy reads this at
# import time, so it must be set before `import cupy` anywhere.
os.environ.setdefault("CUPY_COMPILE_WITH_PTX", "1")

_CUDA_DIAGNOSTICS_CACHE: tuple[bool, str] | None = None


def _scatter_add_for(cupy):
    """Return CuPy's duplicate-index accumulation primitive."""
    if hasattr(cupy.add, "at"):
        return cupy.add.at
    import cupyx

    return cupyx.scatter_add


def _device_name(cupy) -> str:
    """Best-effort name for device zero, suitable for diagnostics/UI."""
    try:
        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props.get("name", props.get(b"name", "CUDA device 0"))
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return "CUDA device 0"


def _cuda_failure(stage: str, error, hint: str) -> tuple[bool, str]:
    if isinstance(error, BaseException):
        detail = f"{type(error).__name__}: {error}"
    else:
        detail = str(error)
    return False, f"CUDA {stage} failed ({detail}). {hint}"


def cuda_diagnostics(force: bool = False) -> tuple[bool, str]:
    """Run a cached CUDA compute preflight and return ``(available, reason)``.

    A device count alone is insufficient: mismatched CuPy/CUDA installations
    often import and enumerate successfully, then fail on the first compiled
    kernel.  This probe exercises allocation, an elementwise kernel, a
    reduction, duplicate-index scatter-add, result validation, and stream
    synchronization.  ``force=True`` reruns it after an installation change.
    """
    global _CUDA_DIAGNOSTICS_CACHE
    if _CUDA_DIAGNOSTICS_CACHE is not None and not force:
        return _CUDA_DIAGNOSTICS_CACHE

    try:
        import cupy
    except Exception as exc:
        result = _cuda_failure(
            "CuPy import", exc,
            "Install the CuPy wheel matching Blender's Python and CUDA "
            "versions (for CUDA 13, use cupy-cuda13x).",
        )
        _CUDA_DIAGNOSTICS_CACHE = result
        return result

    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        result = _cuda_failure(
            "device discovery", exc,
            "Verify that the NVIDIA driver is installed and visible to this "
            "process, then restart Blender.",
        )
        _CUDA_DIAGNOSTICS_CACHE = result
        return result
    if count < 1:
        result = _cuda_failure(
            "device discovery", "no CUDA devices reported",
            "Verify that an NVIDIA GPU and current NVIDIA driver are installed.",
        )
        _CUDA_DIAGNOSTICS_CACHE = result
        return result

    try:
        values = cupy.asarray([1.0, 2.0, 3.0, 4.0], dtype=cupy.float32)
    except Exception as exc:
        result = _cuda_failure(
            "allocation", exc,
            "Check available GPU memory and CuPy/CUDA runtime compatibility.",
        )
        _CUDA_DIAGNOSTICS_CACHE = result
        return result

    try:
        reduced = (values * values + 1.0).sum()
    except Exception as exc:
        result = _cuda_failure(
            "elementwise kernel/reduction", exc,
            "Install a CuPy wheel compatible with the NVIDIA driver and GPU "
            "architecture; restart Blender after changing it.",
        )
        _CUDA_DIAGNOSTICS_CACHE = result
        return result

    try:
        scatter_target = cupy.zeros((3,), dtype=cupy.float32)
        scatter_indices = cupy.asarray([0, 1, 1, 2], dtype=cupy.int32)
        scatter_values = cupy.asarray(
            [1.0, 2.0, 3.0, 4.0], dtype=cupy.float32)
        _scatter_add_for(cupy)(scatter_target, scatter_indices, scatter_values)
    except Exception as exc:
        result = _cuda_failure(
            "duplicate-index scatter", exc,
            "Upgrade CuPy or reinstall the wheel matching the active CUDA "
            "runtime.",
        )
        _CUDA_DIAGNOSTICS_CACHE = result
        return result

    try:
        cupy.cuda.get_current_stream().synchronize()
    except Exception as exc:
        result = _cuda_failure(
            "stream synchronization", exc,
            "Check the NVIDIA driver log and restart Blender before retrying.",
        )
        _CUDA_DIAGNOSTICS_CACHE = result
        return result

    try:
        reduced_host = float(np.asarray(cupy.asnumpy(reduced)).item())
        scatter_host = np.asarray(cupy.asnumpy(scatter_target))
        if not np.isclose(reduced_host, 34.0, rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"reduction returned {reduced_host}, expected 34")
        if not np.allclose(scatter_host, [1.0, 5.0, 4.0],
                           rtol=1e-6, atol=1e-6):
            raise RuntimeError(
                f"scatter returned {scatter_host.tolist()}, expected [1, 5, 4]")
    except Exception as exc:
        result = _cuda_failure(
            "result validation", exc,
            "The CUDA runtime produced incorrect results; reinstall CuPy and "
            "update the NVIDIA driver.",
        )
        _CUDA_DIAGNOSTICS_CACHE = result
        return result

    result = True, f"CUDA preflight passed on {_device_name(cupy)}"
    _CUDA_DIAGNOSTICS_CACHE = result
    return result


class Backend:
    """A thin handle bundling the array module and the few divergent ops."""

    def __init__(self, name: str):
        if name == "cuda":
            import sys
            if ("cupy" in sys.modules
                    and os.environ.get("CUPY_COMPILE_WITH_PTX") != "1"):
                import warnings

                warnings.warn(
                    "cupy was imported before stflip, so "
                    "CUPY_COMPILE_WITH_PTX=1 could not take effect; on very "
                    "new GPU architectures kernel compilation may fail with "
                    "CUDA_ERROR_NO_BINARY_FOR_GPU. Set the variable before "
                    "importing cupy.")
            import cupy  # noqa: F811 - optional dependency

            self.xp = cupy
            # cupy.add.at exists since CuPy 13; cupyx.scatter_add is the
            # deprecated fallback for older versions.
            self._scatter_add = _scatter_add_for(cupy)
            self._is_gpu = True
        elif name == "cpu":
            self.xp = np
            self._scatter_add = np.add.at
            self._is_gpu = False
        else:
            raise ValueError(f"unknown backend {name!r}")
        self.name = name

    @property
    def is_gpu(self) -> bool:
        return self._is_gpu

    def scatter_add(self, target, indices, values) -> None:
        """target[indices] += values with duplicate-index accumulation."""
        self._scatter_add(target, indices, values)

    def to_numpy(self, arr) -> np.ndarray:
        if self._is_gpu:
            return self.xp.asnumpy(arr)
        return np.asarray(arr)

    def from_numpy(self, arr):
        return self.xp.asarray(arr)

    def synchronize(self) -> None:
        if self._is_gpu:
            self.xp.cuda.get_current_stream().synchronize()


def cuda_available() -> bool:
    return cuda_diagnostics()[0]


def cuda_device_name() -> str | None:
    try:
        import cupy

        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return None


def get_backend(name: str = "auto") -> Backend:
    """Resolve 'auto'/'cpu'/'cuda' to a Backend instance."""
    if name == "auto":
        return Backend("cuda") if cuda_available() else Backend("cpu")
    if name == "cuda":
        available, reason = cuda_diagnostics()
        if not available:
            raise RuntimeError(f"CUDA backend unavailable: {reason}")
    return Backend(name)
