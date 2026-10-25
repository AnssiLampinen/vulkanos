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
            "house_count": [5, 1, 0],
            "house_unmatched_count": [2, 0, 0],
            "osm_house_count": [4, 1, 0],
            "osm_house_detected_count": [3, 1, 0],
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
    assert s.loc["a", "house_completeness"] == pytest.approx(0.6)
    assert s.loc["a", "house_recall_vs_osm"] == pytest.approx(0.75)
    assert s.loc["a", "house_gap"] == 2
    assert np.isnan(s.loc["c", "house_completeness"])


def test_split_houses_cuts_narrow_neck():
    pred = np.zeros((40, 60), bool)
    square(pred, 10, 5, 20)  # two 20 px houses
    square(pred, 10, 35, 20)
    pred[18:22, 25:35] = True  # joined by a 4 px wide strip
    labels, n = compare.split_houses(pred, min_sep_px=5)
    assert n == 2
    assert labels[20, 15] != labels[20, 45]
    assert (labels > 0).sum() == pred.sum()
    assert compare.house_edges(labels).any()


def test_split_houses_keeps_rectangle_whole():
    pred = np.zeros((40, 80), bool)
    pred[10:22, 5:75] = True  # long 12 x 70 px block
    labels, n = compare.split_houses(pred, min_sep_px=5)
    assert n == 1
    assert not compare.house_edges(labels).any()


def test_split_houses_small_blob_next_to_big_one():
    pred = np.zeros((40, 40), bool)
    square(pred, 5, 5, 20)
    square(pred, 5, 27, 3)  # tiny blob inside the big one's peak window
    labels, n = compare.split_houses(pred, min_sep_px=10)
    assert n == 2
    assert (labels > 0).sum() == pred.sum()


def test_split_houses_empty():
    labels, n = compare.split_houses(np.zeros((8, 8), bool), 3)
    assert n == 0 and not labels.any()


def test_objects_with_house_labels():
    pred = np.zeros((32, 32), bool)
    osm = np.zeros((32, 32), bool)
    square(pred, 2, 2, 4)
    square(pred, 2, 6, 4)  # touches the first: one blob, two houses
    square(osm, 2, 2, 4)
    labels = np.zeros((32, 32), np.int32)
    labels[2:6, 2:6] = 1
    labels[2:6, 6:10] = 2
    centroids, matched = compare.objects(pred, osm, 0, (labels, 2))
    assert len(centroids) == 2
    assert matched.tolist() == [True, False]


def test_osm_houses_counted_once_and_matched():
    import geopandas as gpd
    from shapely.geometry import box as sbox

    chip = mercantile.tile(14.426, 40.821, 17)
    west, south, east, north = mercantile.xy_bounds(chip)
    px = (east - west) / 512

    def building(r, c, size):
        return sbox(west + c * px, north - (r + size) * px, west + (c + size) * px, north - r * px)

    osm = gpd.GeoDataFrame(
        geometry=[building(10, 10, 10), building(100, 100, 10), building(300, 300, 10)], crs=3857
    )
    by_chip = compare.osm_points_by_chip(osm, chip.z)
    assert sorted(by_chip[(chip.x, chip.y)].tolist()) == [0, 1, 2]

    pred = np.zeros((512, 512), bool)
    square(pred, 10, 10, 10)  # first building detected
    valid = np.ones((512, 512), bool)
    valid[290:320, 290:320] = False  # third building has no imagery
    points, detected = compare.osm_houses(osm, by_chip[(chip.x, chip.y)], chip, pred, valid, 0)
    assert len(points) == 2
    assert detected.tolist() == [True, False]


def test_area_summary_weights_by_buildings():
    df = pd.DataFrame(
        {
            "volcano": ["v", "v", "v"],
            "no_imagery": [False, False, True],
            "pred_m2": [1000.0, 10.0, 500.0],
            "missed_m2": [100.0, 10.0, 500.0],
            "house_count": [10, 1, 5],
            "house_unmatched_count": [1, 1, 5],
        }
    )
    s = compare.area_summary(df).loc["v"]
    assert s["completeness"] == pytest.approx(1 - 110 / 1010)
    assert s["house_completeness"] == pytest.approx(1 - 2 / 11)
    assert s["coverage"] == pytest.approx(2 / 3)
    assert s["houses"] == 11
