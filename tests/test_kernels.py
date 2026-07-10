import numpy as np

from stflip import kernels


def test_spatial_kernel_normalised():
    r = np.linspace(-1.5, 1.5, 20001)
    w = kernels.w_spatial_1d(np, r)
    integral = np.trapezoid(w, r)
    assert abs(integral - 1.0) < 1e-4
    assert np.all(w >= 0.0)
    assert w[0] == 0.0 and w[-1] == 0.0  # compact support


def test_temporal_kernel_normalised_and_one_sided():
    tau = np.linspace(-0.5, 0.5, 20001)
    w = kernels.w_temporal(np, tau)
    integral = np.trapezoid(w, tau)
    assert abs(integral - 1.0) < 1e-4
    # One-sided: peaks at the forward slab boundary tau = +1/2.
    assert w[-1] == w.max()
    # Zero beyond the slab.
    assert kernels.w_temporal(np, np.array([0.7]))[0] == 0.0


def test_temporal_kernel_weights_recent_samples_more():
    early = kernels.w_temporal(np, np.array([-0.4]))[0]
    late = kernels.w_temporal(np, np.array([0.4]))[0]
    assert late > early


def test_smoothstep():
    assert kernels.smoothstep(np, 0.0, 1.0, np.array([-1.0]))[0] == 0.0
    assert kernels.smoothstep(np, 0.0, 1.0, np.array([2.0]))[0] == 1.0
    mid = kernels.smoothstep(np, 0.0, 1.0, np.array([0.5]))[0]
    assert abs(mid - 0.5) < 1e-6
