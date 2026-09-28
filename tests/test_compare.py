import mercantile
import numpy as np
import pandas as pd
import pytest

from vulkanos import compare


def square(mask, r0, c0, size):
    mask[r0 : r0 + size, c0 : c0 + size] = True


def test_pixel_size_equator_zoom18():
    assert compare.pixel_size_m(0, 18) == pytest.approx(0.597, abs=1e-3)


def test_missed_and_detected_areas():
    pred = np.zeros((64, 64), bool)
    osm = np.zeros((64, 64), bool)
    valid = np.ones((64, 64), bool)
    square(pred, 5, 5, 10)  # mapped building
    square(osm, 5, 5, 10)
    square(pred, 40, 40, 10)  # unmapped building
    square(osm, 5, 40, 6)  # OSM building the detector missed

    m = compare.pixel_masks(pred, osm, valid, buffer_px=0)
    assert m["pred"].sum() == 200
    assert m["missed"].sum() == 100
    assert m["osm_detected"].sum() == 100
    assert m["osm"].sum() == 136


def test_buffer_tolerates_misalignment():
    pred = np.zeros((32, 32), bool)
    osm = np.zeros((32, 32), bool)
    square(pred, 10, 10, 8)
    square(osm, 12, 12, 8)  # shifted by 2 px
    valid = np.ones_like(pred)
    assert compare.pixel_masks(pred, osm, valid, 0)["missed"].sum() > 0
    assert compare.pixel_masks(pred, osm, valid, 3)["missed"].sum() == 0


def test_invalid_pixels_ignored():
    pred = np.ones((16, 16), bool)
    osm = np.zeros((16, 16), bool)
    valid = np.zeros((16, 16), bool)
    m = compare.pixel_masks(pred, osm, valid, 0)
    assert m["pred"].sum() == 0 and m["missed"].sum() == 0


def test_remove_small():
    mask = np.zeros((32, 32), bool)
    square(mask, 0, 0, 2)  # 4 px
    square(mask, 10, 10, 5)  # 25 px
    out = compare.remove_small(mask, 10)
    assert out.sum() == 25


def test_objects_matched():
    pred = np.zeros((32, 32), bool)
    osm = np.zeros((32, 32), bool)
    square(pred, 2, 2, 4)
    square(osm, 2, 2, 4)
    square(pred, 20, 20, 4)
    centroids, matched = compare.objects(pred, osm, 1)
    assert len(centroids) == 2
    assert sorted(matched.tolist()) == [False, True]


def test_block_sum():
    mask = np.zeros((16, 16), bool)
    mask[:8, :8] = True
    assert compare.block_sum(mask, 8).tolist() == [[64, 0], [0, 0]]


def test_block_cells_shape_and_resolution():
    chip = mercantile.tile(14.426, 40.821, 17)
    cells = compare.block_cells(chip, 512, 8, 8)
    assert cells.shape == (64, 64)
    assert len(set(cells.ravel())) >= 1


def test_add_scores():
    df = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c"],
            "total_m2": [1000.0, 1000.0, 1000.0],
            "valid_m2": [1000.0, 1000.0, 100.0],
            "pred_m2": [400.0, 50.0, 0.0],
            "osm_m2": [300.0, 50.0, 0.0],
            "missed_m2": [100.0, 0.0, 0.0],
            "osm_detected_m2": [300.0, 25.0, 0.0],
            "pred_count": [4, 1, 0],
            "unmatched_count": [1, 0, 0],
        }
    )
    s = compare.add_scores(df, low_signal_m2=200).set_index("cell_id")
    assert s.loc["a", "completeness"] == pytest.approx(0.75)
    assert s.loc["a", "recall_vs_osm"] == pytest.approx(1.0)
    assert s.loc["a", "gap_score"] == 100
    assert not s.loc["a", "low_signal"]
    assert s.loc["b", "low_signal"]
    assert np.isnan(s.loc["c", "completeness"])
    assert s.loc["c", "no_imagery"]
