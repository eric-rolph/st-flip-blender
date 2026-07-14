import numpy as np
import pytest

from stflip.tiles import build_tile_layout


def test_layout_is_sorted_and_expands_a_clamped_tile_halo():
    active = np.zeros((24, 24, 24), dtype=bool)
    active[9, 9, 9] = True

    layout = build_tile_layout(active, tile_size=8, halo_tiles=1)

    assert layout.core_coords.tolist() == [[1, 1, 1]]
    assert layout.tile_count == 27
    assert layout.coords[0].tolist() == [0, 0, 0]
    assert layout.coords[-1].tolist() == [2, 2, 2]
    assert layout.slot((1, 1, 1)) >= 0
    assert layout.slot((-1, 1, 1)) == -1
    assert layout.slot((3, 1, 1)) == -1


def test_pack_unpack_round_trip_preserves_selected_tiles_and_components():
    active = np.zeros((10, 9, 7), dtype=bool)
    active[1, 1, 1] = True
    active[9, 8, 6] = True
    layout = build_tile_layout(active, tile_size=4, halo_tiles=0)
    field = np.arange(active.size * 2, dtype=np.float32).reshape(active.shape + (2,))

    packed = layout.pack(field)
    restored = layout.unpack(packed)

    covered = np.zeros_like(active)
    for coord in layout.coords:
        lo = coord * layout.tile_size
        hi = np.minimum(lo + layout.tile_size, active.shape)
        covered[tuple(slice(a, b) for a, b in zip(lo, hi))] = True
    np.testing.assert_array_equal(restored[covered], field[covered])
    assert np.all(restored[~covered] == 0.0)
    assert packed.shape == (2, 4, 4, 4, 2)


def test_neighbor_slots_distinguish_faces_from_diagonals():
    active = np.ones((16, 16, 16), dtype=bool)
    layout = build_tile_layout(active, tile_size=8, halo_tiles=0)
    origin_slot = layout.slot((0, 0, 0))

    faces = layout.neighbor_slots(include_diagonals=False)[origin_slot]
    all_neighbors = layout.neighbor_slots(include_diagonals=True)[origin_slot]

    assert np.count_nonzero(faces >= 0) == 3
    assert np.count_nonzero(all_neighbors >= 0) == 7
    assert layout.slot((1, 1, 1)) in all_neighbors


def test_halo_exchange_reads_faces_edges_and_corners_without_dense_unpack():
    active = np.ones((8, 8, 8), dtype=bool)
    layout = build_tile_layout(active, tile_size=4, halo_tiles=0)
    x, y, z = np.indices(active.shape)
    dense = (10_000 * x + 100 * y + z).astype(np.int32)
    packed = layout.pack(dense)

    halo = layout.with_halo(packed)
    origin = layout.slot((0, 0, 0))

    np.testing.assert_array_equal(
        halo[origin, 5, 1:5, 1:5], dense[4, 0:4, 0:4])
    np.testing.assert_array_equal(
        halo[origin, 5, 5, 1:5], dense[4, 4, 0:4])
    assert halo[origin, 5, 5, 5] == dense[4, 4, 4]
    assert np.all(halo[origin, 0, :, :] == 0)


def test_halo_exchange_supports_trailing_components_and_validates_shape():
    active = np.ones((4, 4, 4), dtype=bool)
    layout = build_tile_layout(active, tile_size=2, halo_tiles=0)
    vector = np.zeros(active.shape + (3,), dtype=np.float32)
    vector[..., 2] = 7.0

    halo = layout.with_halo(layout.pack(vector))

    assert halo.shape == (8, 4, 4, 4, 3)
    assert np.max(halo[..., 2]) == 7.0
    with pytest.raises(ValueError):
        layout.with_halo(np.zeros((1, 2, 2, 2), dtype=np.float32))


def test_telemetry_exposes_disconnected_tile_gain_and_halo_cost():
    active = np.zeros((64, 64, 64), dtype=bool)
    active[1:3, 1:3, 1:3] = True
    active[61:63, 61:63, 61:63] = True
    layout = build_tile_layout(active, tile_size=8, halo_tiles=1)

    telemetry = layout.telemetry(active)

    assert telemetry.active_cells == 16
    assert telemetry.bounding_box_cells == 62 ** 3
    assert telemetry.core_tiles == 2
    assert telemetry.core_tile_cells == 2 * 8 ** 3
    assert telemetry.bbox_to_core_tile_ratio > 200.0
    assert telemetry.halo_tiles > telemetry.core_tiles
    assert telemetry.halo_tile_fraction < telemetry.bounding_box_fraction
    assert telemetry.as_dict()["active_cells"] == 16


def test_empty_layout_and_boundary_tiles_have_exact_footprints():
    empty = np.zeros((10, 9, 7), dtype=bool)
    layout = build_tile_layout(empty, tile_size=8)
    telemetry = layout.telemetry(empty)

    assert layout.tile_count == 0
    assert layout.pack(np.ones(empty.shape)).shape == (0, 8, 8, 8)
    assert telemetry.core_tile_cells == 0
    assert telemetry.bbox_to_core_tile_ratio is None

    empty[-1, -1, -1] = True
    edge = build_tile_layout(empty, tile_size=8, halo_tiles=0)
    assert edge.telemetry(empty).core_tile_cells == 2 * 1 * 7


@pytest.mark.parametrize(
    "mask,tile_size,halo",
    [
        (np.zeros((2, 2), dtype=bool), 8, 1),
        (np.zeros((2, 2, 2), dtype=np.uint8), 8, 1),
        (np.zeros((2, 2, 2), dtype=bool), 0, 1),
        (np.zeros((2, 2, 2), dtype=bool), 8, -1),
    ],
)
def test_invalid_layout_inputs_are_rejected(mask, tile_size, halo):
    with pytest.raises(ValueError):
        build_tile_layout(mask, tile_size=tile_size, halo_tiles=halo)


def test_pack_and_telemetry_validate_shapes_and_dtypes():
    active = np.zeros((8, 8, 8), dtype=bool)
    active[0, 0, 0] = True
    layout = build_tile_layout(active, tile_size=4)

    with pytest.raises(ValueError):
        layout.pack(np.zeros((8, 8, 7), dtype=np.float32))
    with pytest.raises(ValueError):
        layout.unpack(np.zeros((2, 8, 8, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        layout.telemetry(active.astype(np.uint8))

    different_tile = np.zeros_like(active)
    different_tile[-1, -1, -1] = True
    with pytest.raises(ValueError, match="different core tiles"):
        layout.telemetry(different_tile)
