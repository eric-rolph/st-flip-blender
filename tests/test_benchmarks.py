import numpy as np
import pytest

from stflip.benchmarks import (
    GlugBenchmark,
    KleefsmanBenchmark,
    benchmark_grid,
    compare_gauge_series,
    load_gauge_reference,
    water_height_from_particles,
)


def test_kleefsman_published_geometry_and_exact_two_centimetre_grid():
    case = KleefsmanBenchmark()
    grid = case.grid(dx=0.02)

    assert grid.shape == (161, 50, 50)
    assert grid.effective_extent == pytest.approx(case.tank)
    assert case.gate_x == pytest.approx(1.992)
    assert dict(case.gauge_x) == {"H2": 0.992, "H4": 2.636}
    assert dict(case.gauges_xy) == {"H2": (0.992, 0.5), "H4": (2.636, 0.5)}
    metadata = case.metadata()
    assert metadata["gauge_sampling_assumption"]["y"] == 0.5
    assert metadata["coordinate_convention"]["transform"] == (
        "identity: x_solver = x_published; y and z unchanged")
    published = metadata["published_coordinates"]
    assert published["initial_water_x_range"] == pytest.approx((1.992, 3.22))
    assert published["gate_x"] == pytest.approx(1.992)
    assert published["obstacle_center"] == pytest.approx(
        (0.744, 0.5, 0.0805))
    assert published["gauge_x"] == pytest.approx(
        {"H2": 0.992, "H4": 2.636})
    assert metadata["solver_coordinates"]["gate_x"] == pytest.approx(1.992)
    assert metadata["solver_coordinates"]["obstacle_center"] == pytest.approx(
        (0.744, 0.5, 0.0805))
    assert metadata["solver_coordinates"]["gauge_x"] == pytest.approx(
        {"H2": 0.992, "H4": 2.636})

    liquid = case.liquid_mask(grid)
    obstacle = case.solid_sdf_cells(grid) < 0.0
    assert liquid.any()
    assert obstacle.any()
    assert not np.any(liquid & obstacle)
    assert case.solid_sdf_nodes(grid).shape == (162, 51, 51)


def test_glug_published_ratios_and_connected_cavity_are_explicit():
    case = GlugBenchmark()
    grid = case.grid(dx=0.025)
    dimensions = case.published_dimensions

    assert dimensions == {
        "L": 1.0,
        "container_square_side": 0.5,
        "connector_radius": 0.05,
        "connector_length": 0.05,
        "initial_upper_water_height": 0.5,
    }
    sdf = case.solid_sdf_cells(grid)
    liquid = case.liquid_mask(grid)
    regions = case.region_masks(grid)
    assert liquid.any()
    assert np.all(sdf[liquid] > 0.0)
    assert np.any((sdf > 0.0) & regions["connector"])
    assert not np.any(liquid & regions["lower"])
    assert case.metadata()["layout_assumptions"]


def test_water_height_uses_bottom_connected_layers_and_ignores_spray():
    dx = 0.1
    z = np.arange(0.05, 0.55, dx)
    column = np.column_stack((np.zeros_like(z), np.zeros_like(z), z))
    one_layer_hole = np.delete(column, 2, axis=0)
    positions = np.vstack((one_layer_hole, [[0.0, 0.0, 1.55]]))

    height = water_height_from_particles(positions, (0.0, 0.0), dx=dx)

    assert height == pytest.approx(0.5)
    assert water_height_from_particles(
        positions, (2.0, 2.0), dx=dx) == 0.0


def test_reference_csv_is_hashed_and_compared_on_reference_timestamps(tmp_path):
    path = tmp_path / "kleefsman.csv"
    path.write_text(
        "time_s,H2_m,H4_m\n0,0,0.55\n0.5,0.2,0.50\n1.0,0.3,0.45\n",
        encoding="utf-8",
    )
    reference = load_gauge_reference(path)

    metrics = compare_gauge_series(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        {"H2": [0.0, 0.1, 0.2, 0.25, 0.3],
         "H4": [0.55, 0.525, 0.5, 0.475, 0.45]},
        reference,
    )

    assert len(reference["sha256"]) == 64
    assert metrics["H2"]["rmse_m"] == pytest.approx(0.0)
    assert metrics["H4"]["mae_m"] == pytest.approx(0.0)
    assert metrics["H2"]["samples"] == 3


def test_gauge_comparison_tolerates_decimal_rounding_at_frame_endpoint():
    metrics = compare_gauge_series(
        [0.0, 1.0 / 24.0],
        {"H2": [0.0, 1.0]},
        {
            "time_s": np.array([0.0, 0.0416666666667]),
            "H2_m": np.array([0.0, 1.0]),
        },
    )

    assert metrics["H2"]["samples"] == 2
    assert metrics["H2"]["rmse_m"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "contents,match",
    [
        ("H2_m\n0\n", "time_s"),
        ("time_s,H2\n0,0\n", "gauge columns"),
        ("time_s,H2_m\n0,0\n0,1\n", "strictly increasing"),
        ("time_s,H2_m\n0,-1\n", "negative"),
        ("time_s,H2_m,H2_m\n0,0,0\n", "unique"),
        ("time_s,H2_m\n0,0,extra\n", "no extra values"),
        ("time_s,H2_m\n0,\n", "complete"),
    ],
)
def test_reference_csv_rejects_ambiguous_or_invalid_data(tmp_path, contents, match):
    path = tmp_path / "bad.csv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_gauge_reference(path)


def test_grid_and_metric_input_validation(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        benchmark_grid((1, 1, 1), longest_resolution=8, dx=0.1)
    with pytest.raises(ValueError, match="three positive"):
        benchmark_grid((1, 1), longest_resolution=8)
    with pytest.raises(ValueError, match="strictly increasing"):
        compare_gauge_series(
            [0.0, 0.0], {"H2": [0.0, 0.1]},
            {"time_s": np.array([0.0, 1.0]), "H2_m": np.array([0.0, 0.1])},
        )
    with pytest.raises(ValueError, match="align with reference"):
        compare_gauge_series(
            [0.0, 1.0], {"H2": [0.0, 0.1]},
            {"time_s": np.array([0.0, 1.0]), "H2_m": np.array([0.0])},
        )
