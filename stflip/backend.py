"""Array-module backend abstraction.

The entire solver is written against the array-API-compatible subset shared by
NumPy and CuPy, so the same code runs on CPU (NumPy) and NVIDIA CUDA GPUs
(CuPy).  AMD GPUs are supported through CuPy's ROCm builds on Linux; on
Windows the CPU backend is used.

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
            import cupyx

            self.xp = cupy
            # cupy.add.at exists since CuPy 13; cupyx.scatter_add is the
            # deprecated fallback for older versions.
            if hasattr(cupy.add, "at"):
                self._scatter_add = cupy.add.at
            else:
                self._scatter_add = cupyx.scatter_add
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
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


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
    return Backend(name)
