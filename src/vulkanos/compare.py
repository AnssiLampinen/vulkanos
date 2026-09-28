"""Compare detected buildings with OSM buildings and aggregate per H3 cell."""

import math
from pathlib import Path

import geopandas as gpd
import h3
import mercantile
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy import ndimage
from shapely.geometry import box

from .detect import load_prob, pred_path
from .imagery import load_chip

EARTH_CIRCUMFERENCE_M = 2 * math.pi * 6378137.0

# Per-cell sums, in m2 unless named *_count.
SUM_FIELDS = [
    "total_m2",  # cell area covered by processed chips
    "valid_m2",  # ... of which has imagery
    "pred_m2",  # detected building area
    "osm_m2",  # OSM building area (on imagery pixels)
    "missed_m2",  # detected area not near any OSM building
    "osm_detected_m2",  # OSM area near a detection
    "pred_count",  # detected objects
    "unmatched_count",  # detected objects with no OSM building nearby
]


def pixel_size_m(lat: float, zoom: int, tile_px: int = 256) -> float:
    """Ground size of one Web Mercator pixel at latitude lat."""
    return EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / (tile_px * 2**zoom)


def disk(radius_px: int) -> np.ndarray:
    r = max(int(radius_px), 0)
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= r * r


def dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0 or not mask.any():
        return mask.copy()
    return ndimage.binary_dilation(mask, structure=disk(radius_px))


def remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0 or min_px <= 1:
        return mask
    sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, n + 1))
    keep = np.concatenate([[False], sizes >= min_px])
    return keep[labels]


def pixel_masks(pred: np.ndarray, osm: np.ndarray, valid: np.ndarray, buffer_px: int) -> dict:
    """Pixel-level comparison masks. All inputs are bool HxW."""
    pred = pred & valid
    osm = osm & valid
    return {
        "valid": valid,
        "pred": pred,
        "osm": osm,
        "missed": pred & ~dilate(osm, buffer_px),
        "osm_detected": osm & dilate(pred, buffer_px),
    }


def objects(pred: np.ndarray, osm: np.ndarray, buffer_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Detected objects. Returns (centroids Nx2 as row/col, matched N bool)."""
    labels, n = ndimage.label(pred, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0, bool)
    idx = np.arange(1, n + 1)
    centroids = np.array(ndimage.center_of_mass(pred, labels, idx)).reshape(-1, 2)
    matched = ndimage.maximum(dilate(osm, buffer_px), labels, idx).astype(bool)
    return centroids, np.atleast_1d(matched)


def block_cells(chip: mercantile.Tile, size_px: int, block_px: int, res: int) -> np.ndarray:
    """H3 cell id of the centre of each block_px x block_px block of a chip."""
    west, south, east, north = mercantile.xy_bounds(chip)
    nb = size_px // block_px
    centres = (np.arange(nb) + 0.5) / nb
    xs = west + centres * (east - west)
    ys = north - centres * (north - south)
    lon = np.degrees(xs / 6378137.0)
    lat = np.degrees(2 * np.arctan(np.exp(ys / 6378137.0)) - math.pi / 2)
    return np.array([[h3.latlng_to_cell(la, lo, res) for lo in lon] for la in lat])


def block_sum(mask: np.ndarray, block_px: int) -> np.ndarray:
    h, w = mask.shape
    return mask.reshape(h // block_px, block_px, w // block_px, block_px).sum(axis=(1, 3))


def rasterize_osm(osm_3857: gpd.GeoDataFrame, chip: mercantile.Tile, size_px: int) -> np.ndarray:
    bounds = mercantile.xy_bounds(chip)
    hits = osm_3857.sindex.query(box(*bounds))
    if len(hits) == 0:
        return np.zeros((size_px, size_px), bool)
    return rasterize(
        osm_3857.geometry.iloc[hits],
        out_shape=(size_px, size_px),
        transform=from_bounds(*bounds, size_px, size_px),
        fill=0,
        default_value=1,
        dtype="uint8",
    ).astype(bool)


def chip_masks(
    chip: mercantile.Tile,
    osm_3857: gpd.GeoDataFrame,
    pred_dir: Path,
    data_dir: Path,
    provider: str,
    zoom: int,
    threshold: float,
    osm_buffer_m: float,
    min_object_m2: float,
) -> tuple[dict, float, int] | None:
    """Comparison masks for one chip: (masks, pixel size m, buffer px), or None
    when the chip has no saved prediction."""
    prob = load_prob(pred_path(pred_dir, chip))
    if prob is None:
        return None
    _, valid = load_chip(data_dir, provider, chip, zoom)
    b = mercantile.bounds(chip)
    px_m = pixel_size_m((b.north + b.south) / 2, zoom)
    buffer_px = math.ceil(osm_buffer_m / px_m)
    pred = remove_small(prob >= threshold, math.ceil(min_object_m2 / (px_m * px_m)))
    masks = pixel_masks(pred, rasterize_osm(osm_3857, chip, prob.shape[0]), valid, buffer_px)
    return masks, px_m, buffer_px


def score_chips(
    chips: list[mercantile.Tile],
    cell_ids: set[str],
    osm: gpd.GeoDataFrame,
    pred_dir: Path,
    data_dir: Path,
    provider: str,
    zoom: int,
    res: int,
    threshold: float,
    osm_buffer_m: float,
    min_object_m2: float,
    block_px: int,
) -> pd.DataFrame:
    """Per-cell sums over all chips with a saved prediction."""
    osm_3857 = osm.to_crs(3857)
    parts = []

    for chip in chips:
        result = chip_masks(
            chip, osm_3857, pred_dir, data_dir, provider, zoom, threshold, osm_buffer_m, min_object_m2
        )
        if result is None:
            continue
        masks, px_m, buffer_px = result
        px_m2 = px_m * px_m
        size = masks["pred"].shape[0]
        cells = block_cells(chip, size, block_px, res)
        blocks = {k: block_sum(m, block_px) for k, m in masks.items()}

        centroids, matched = objects(masks["pred"], masks["osm"], buffer_px)
        pred_count = np.zeros(cells.shape)
        unmatched_count = np.zeros(cells.shape)
        rows, cols = (centroids // block_px).astype(int).T
        np.add.at(pred_count, (rows, cols), 1)
        np.add.at(unmatched_count, (rows, cols), ~matched)

        chip_df = pd.DataFrame(
            {
                "cell_id": cells.ravel(),
                "total_m2": block_px * block_px * px_m2,
                **{f"{k}_m2": blocks[k].ravel() * px_m2 for k in blocks},
                "pred_count": pred_count.ravel(),
                "unmatched_count": unmatched_count.ravel(),
            }
        )
        chip_df = chip_df[chip_df["cell_id"].isin(cell_ids)]
        parts.append(chip_df.groupby("cell_id")[SUM_FIELDS].sum())

    if not parts:
        return pd.DataFrame(columns=["cell_id", *SUM_FIELDS])
    return pd.concat(parts).groupby(level=0)[SUM_FIELDS].sum().rename_axis("cell_id").reset_index()


def add_scores(df: pd.DataFrame, low_signal_m2: float, min_coverage: float = 0.5) -> pd.DataFrame:
    """Derived scores. completeness is NaN where nothing was detected."""
    df = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["completeness"] = 1 - df["missed_m2"] / df["pred_m2"]
        # On cells where OSM is complete, completeness = detector precision
        # and recall_vs_osm = detector recall.
        df["recall_vs_osm"] = df["osm_detected_m2"] / df["osm_m2"]
        df["imagery_coverage"] = df["valid_m2"] / df["total_m2"]
    df["gap_score"] = df["missed_m2"]
    df["low_signal"] = df["pred_m2"] < low_signal_m2
    df["no_imagery"] = df["imagery_coverage"].fillna(0) < min_coverage
    return df
