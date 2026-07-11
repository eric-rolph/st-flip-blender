import numpy as np
import pytest

from stflip.apertures import (
    face_apertures_from_node_sdf,
    solid_cells_from_node_sdf,
    square_open_fraction,
    triangle_open_fraction,
    weighted_divergence,
)


def test_uniform_node_sdf_produces_fully_open_or_solid_grid():
    open_sdf = np.ones((3, 4, 5), dtype=np.float32)
    solid_sdf = -open_sdf

    open_faces = face_apertures_from_node_sdf(open_sdf)
    solid_faces = face_apertures_from_node_sdf(solid_sdf)

    assert [array.shape for array in open_faces] == [
        (3, 3, 4),
        (2, 4, 4),
        (2, 3, 5),
    ]
    for alpha in open_faces:
        assert isinstance(alpha, np.ndarray)
        np.testing.assert_array_equal(alpha, 1.0)
    for alpha in solid_faces:
        np.testing.assert_array_equal(alpha, 0.0)
    np.testing.assert_array_equal(
        solid_cells_from_node_sdf(open_sdf), False
    )
    np.testing.assert_array_equal(
        solid_cells_from_node_sdf(solid_sdf), True
    )


def test_planar_quarter_cut_has_three_quarters_open_area():
    # Across each x-normal face, phi(y, z) = y - 0.25. The open portion is
    # y > 0.25 and therefore occupies exactly 75% of the face.
    node_sdf = np.empty((2, 2, 2), dtype=np.float64)
    node_sdf[:, 0, :] = -0.25
    node_sdf[:, 1, :] = 0.75

    alpha_u, _alpha_v, alpha_w = face_apertures_from_node_sdf(node_sdf)

    np.testing.assert_allclose(alpha_u, 0.75, atol=1e-15)
    np.testing.assert_allclose(alpha_w, 0.75, atol=1e-15)
    assert square_open_fraction(-0.25, 0.75, -0.25, 0.75) \
        == pytest.approx(0.75)
    # Its center is open here, but more importantly any cut cell remains in
    # the pressure system until all eight corners are inside the solid.
    np.testing.assert_array_equal(
        solid_cells_from_node_sdf(node_sdf), False)


def test_partially_cut_cell_with_negative_center_is_not_fully_solid():
    node_sdf = np.full((2, 2, 2), -1.0)
    node_sdf[1, 1, 1] = 0.1

    assert not bool(solid_cells_from_node_sdf(node_sdf)[0, 0, 0])


def test_diagonal_planar_cut_has_half_open_area():
    # phi(x, y) = x + y - 1 cuts each z-normal face along its diagonal.
    node_sdf = np.empty((2, 2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            node_sdf[i, j, :] = i + j - 1.0

    _alpha_u, _alpha_v, alpha_w = face_apertures_from_node_sdf(node_sdf)

    np.testing.assert_allclose(alpha_w, 0.5, atol=1e-15)
    assert square_open_fraction(-1.0, 0.0, 0.0, 1.0) \
        == pytest.approx(0.5)


def test_open_fraction_is_complementary_when_sdf_sign_is_reversed():
    rng = np.random.default_rng(44)
    node_sdf = rng.normal(size=(4, 3, 5))
    positive = face_apertures_from_node_sdf(node_sdf)
    negative = face_apertures_from_node_sdf(-node_sdf)

    for alpha, complement in zip(positive, negative, strict=True):
        np.testing.assert_allclose(alpha + complement, 1.0, atol=2e-15)

    phi0 = np.array([-2.0, -1.0, 3.0])
    phi1 = np.array([1.0, 4.0, -2.0])
    phi2 = np.array([3.0, -2.0, 1.0])
    fraction = triangle_open_fraction(phi0, phi1, phi2)
    complement = triangle_open_fraction(-phi0, -phi1, -phi2)
    np.testing.assert_allclose(fraction + complement, 1.0, atol=2e-15)


@pytest.mark.parametrize(
    "node_sdf, error",
    [
        (np.zeros((2, 2)), "3D array"),
        (np.zeros((2, 1, 2)), "at least two nodes"),
        (np.array([[[0.0, np.nan], [0.0, 0.0]]] * 2), "finite"),
    ],
)
def test_node_sdf_validation_rejects_invalid_inputs(node_sdf, error):
    with pytest.raises(ValueError, match=error):
        face_apertures_from_node_sdf(node_sdf)
    with pytest.raises(ValueError, match=error):
        solid_cells_from_node_sdf(node_sdf)


def test_fraction_helpers_reject_non_numeric_and_incompatible_shapes():
    with pytest.raises(TypeError, match="numeric"):
        triangle_open_fraction("open", 0.0, 1.0)
    with pytest.raises(ValueError, match="broadcast-compatible"):
        square_open_fraction(
            np.zeros((2,)),
            np.zeros((3,)),
            0.0,
            1.0,
        )


def test_weighted_divergence_uses_mac_face_shapes_and_numpy_arrays():
    u = np.zeros((3, 2, 2), dtype=np.float32)
    v = np.zeros((2, 3, 2), dtype=np.float32)
    w = np.zeros((2, 2, 3), dtype=np.float32)
    u[1, :, :] = 1.0
    u[2, :, :] = 3.0
    alpha_u = np.ones_like(u)
    alpha_v = np.ones_like(v)
    alpha_w = np.ones_like(w)

    divergence = weighted_divergence(
        u, v, w, alpha_u, alpha_v, alpha_w, dx=0.5
    )

    assert isinstance(divergence, np.ndarray)
    assert divergence.shape == (2, 2, 2)
    np.testing.assert_array_equal(divergence[0], 2.0)
    np.testing.assert_array_equal(divergence[1], 4.0)


def test_weighted_divergence_rejects_shape_range_and_finite_errors():
    u = np.zeros((3, 2, 2))
    v = np.zeros((2, 3, 2))
    w = np.zeros((2, 2, 3))
    alpha_u = np.ones_like(u)
    alpha_v = np.ones_like(v)
    alpha_w = np.ones_like(w)

    with pytest.raises(ValueError, match="v must have shape"):
        weighted_divergence(
            u,
            v[:-1],
            w,
            alpha_u,
            alpha_v[:-1],
            alpha_w,
            1.0,
        )
    invalid_alpha = alpha_u.copy()
    invalid_alpha[0, 0, 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        weighted_divergence(
            u, v, w, invalid_alpha, alpha_v, alpha_w, 1.0
        )
    invalid_u = u.copy()
    invalid_u[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        weighted_divergence(
            invalid_u, v, w, alpha_u, alpha_v, alpha_w, 1.0
        )
    with pytest.raises(ValueError, match="positive scalar"):
        weighted_divergence(
            u, v, w, alpha_u, alpha_v, alpha_w, 0.0
        )
