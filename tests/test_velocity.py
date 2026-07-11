import numpy as np
import pytest

from stflip import SolidBodyRotation, UniformVelocity
from stflip.velocity import as_velocity_field


def test_uniform_velocity_returns_owned_float32_samples():
    field = UniformVelocity((1.25, -2.0, 0.5))
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [4.0, -3.0, 2.0]], dtype=np.float64
    )

    samples = field.sample(positions)

    np.testing.assert_array_equal(
        samples,
        np.asarray([[1.25, -2.0, 0.5], [1.25, -2.0, 0.5]],
                   dtype=np.float32),
    )
    assert samples.dtype == np.float32
    assert samples.flags.c_contiguous
    assert samples.flags.owndata


def test_solid_body_rotation_is_right_handed_and_adds_linear_velocity():
    field = SolidBodyRotation(
        center=(1.0, -1.0, 0.5),
        angular_velocity=(0.0, 0.0, 2.0),
        linear_velocity=(0.25, -0.5, 1.0),
    )
    positions = np.asarray(
        [
            [1.0, -1.0, 0.5],
            [2.0, -1.0, 0.5],
            [1.0, 1.0, 0.5],
        ],
        dtype=np.float32,
    )

    samples = field.sample(positions)

    np.testing.assert_array_equal(
        samples,
        np.asarray(
            [
                [0.25, -0.5, 1.0],
                [0.25, 1.5, 1.0],
                [-3.75, -0.5, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_empty_position_batches_preserve_vector_shape_and_dtype():
    positions = np.empty((0, 3), dtype=np.float32)
    fields = (
        UniformVelocity(),
        SolidBodyRotation((10.0, -4.0, 2.0), (0.0, 0.0, 0.0)),
    )

    for field in fields:
        samples = field.sample(positions)
        assert samples.shape == (0, 3)
        assert samples.dtype == np.float32
        assert samples.flags.c_contiguous


def test_legacy_vector_normalizes_to_uniform_field():
    field = as_velocity_field([1, 2, 3])

    assert field == UniformVelocity((1.0, 2.0, 3.0))
    assert as_velocity_field(field) is field


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        (),
        (1.0, 2.0),
        (1.0, 2.0, 3.0, 4.0),
        ((1.0, 2.0, 3.0),),
        (np.nan, 0.0, 0.0),
        (np.inf, 0.0, 0.0),
        (1e100, 0.0, 0.0),
        (1.0 + 2.0j, 0.0, 0.0),
        ("x", 0.0, 0.0),
    ],
)
def test_uniform_velocity_rejects_non_vec3_or_non_finite_values(value):
    with pytest.raises(
        ValueError, match="value must contain exactly three finite values"
    ):
        UniformVelocity(value)


def test_rotation_validates_each_parameter():
    with pytest.raises(ValueError, match="center must contain"):
        SolidBodyRotation((0.0, np.nan, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="angular_velocity must contain"):
        SolidBodyRotation((0.0, 0.0, 0.0), (0.0, 1.0))
    with pytest.raises(ValueError, match="linear_velocity must contain"):
        SolidBodyRotation(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, np.inf),
        )


@pytest.mark.parametrize(
    "positions",
    [
        np.zeros(3, dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.asarray([[0.0, np.nan, 0.0]], dtype=np.float32),
        np.asarray([[0.0, np.inf, 0.0]], dtype=np.float32),
        np.asarray([[1e100, 0.0, 0.0]], dtype=np.float64),
        np.asarray([[1.0 + 2.0j, 0.0, 0.0]], dtype=np.complex64),
    ],
)
def test_sampling_rejects_malformed_or_non_finite_positions(positions):
    with pytest.raises(ValueError, match="positions_local must have shape"):
        UniformVelocity().sample(positions)


def test_rotation_rejects_non_finite_sample_results():
    field = SolidBodyRotation(
        center=(-3e38, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 3e38),
    )

    with pytest.raises(ValueError, match="produced non-finite values"):
        field.sample(np.asarray([[3e38, 0.0, 0.0]], dtype=np.float32))
