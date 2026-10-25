import math

import h3
import mercantile
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy import ndimage
from shapely.geometry import box

from data import load_chip
from detect import load_prob, pred_path

# per cell sums
FIELDS = ["total_m2", "valid_m2", "pred_m2", "osm_m2", "missed_m2", "osm_detected_m2", "pred_count", "unmatched_count"]


def pixel_size(lat, zoom):
    # web mercator pixel size in metres
    return 2 * math.pi * 6378137.0 * math.cos(math.radians(lat)) / (256 * 2**zoom)


def dilate(mask, r):
    if r <= 0 or not mask.any():
        return mask
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return ndimage.binary_dilation(mask, structure=x * x + y * y <= r * r)


def remove_small(mask, min_px):
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0 or min_px <= 1:
        return mask
    sizes = ndimage.sum_labels(mask, labels, np.arange(1, n + 1))
    return np.concatenate([[False], sizes >= min_px])[labels]


def rasterize_osm(osm, chip, size):
    # osm has to be in EPSG:3857 here
    bounds = mercantile.xy_bounds(chip)
    hits = osm.sindex.query(box(*bounds))
    if len(hits) == 0:
        return np.zeros((size, size), bool)
    return rasterize(osm.geometry.iloc[hits], out_shape=(size, size), transform=from_bounds(*bounds, size, size),
                     fill=0, default_value=1, dtype="uint8").astype(bool)


def block_cells(chip, size, block, res):
    # h3 cell of the centre of each block x block pixel square
    west, south, east, north = mercantile.xy_bounds(chip)
    n = size // block
    f = (np.arange(n) + 0.5) / n
    lon = np.degrees((west + f * (east - west)) / 6378137.0)
    lat = np.degrees(2 * np.arctan(np.exp((north - f * (north - south)) / 6378137.0)) - math.pi / 2)
    return np.array([[h3.latlng_to_cell(la, lo, res) for lo in lon] for la in lat])


def block_sum(mask, block):
    h, w = mask.shape
    return mask.reshape(h // block, block, w // block, block).sum(axis=(1, 3))


def chip_masks(cfg, chip, osm, pred_dir):
    prob = load_prob(pred_path(pred_dir, chip))
    if prob is None:
        return None
    _, valid = load_chip(cfg, chip)
    b = mercantile.bounds(chip)
    px = pixel_size((b.north + b.south) / 2, cfg["imagery"]["zoom"])
    buf = math.ceil(cfg["compare"]["osm_buffer_m"] / px)
    pred = remove_small(prob >= cfg["detector"]["threshold"], math.ceil(cfg["compare"]["min_object_m2"] / (px * px)))
    pred = pred & valid
    osm = rasterize_osm(osm, chip, prob.shape[0]) & valid
    masks = {
        "valid": valid,
        "pred": pred,
        "osm": osm,
        "missed": pred & ~dilate(osm, buf),  # detected but not in osm
        "osm_detected": osm & dilate(pred, buf),
    }
    return masks, px, buf


def count_objects(pred, osm, buf, block, shape):
    # detected objects counted in the block that holds their centroid
    total = np.zeros(shape)
    unmatched = np.zeros(shape)
    labels, n = ndimage.label(pred, structure=np.ones((3, 3)))
    if n == 0:
        return total, unmatched
    idx = np.arange(1, n + 1)
    rows, cols = (np.array(ndimage.center_of_mass(pred, labels, idx)).reshape(-1, 2) // block).astype(int).T
    matched = np.atleast_1d(ndimage.maximum(dilate(osm, buf), labels, idx).astype(bool))
    np.add.at(total, (rows, cols), 1)
    np.add.at(unmatched, (rows, cols), ~matched)
    return total, unmatched


def score_chips(cfg, chips, cell_ids, osm, pred_dir):
    osm = osm.to_crs(3857)
    block = cfg["compare"]["block_px"]
    parts = []
    for chip in chips:
        res = chip_masks(cfg, chip, osm, pred_dir)
        if res is None:
            continue
        masks, px, buf = res
        cells = block_cells(chip, masks["pred"].shape[0], block, cfg["h3_res"])
        df = pd.DataFrame({"cell_id": cells.ravel(), "total_m2": block * block * px * px})
        for k, m in masks.items():
            df[k + "_m2"] = block_sum(m, block).ravel() * px * px
        total, unmatched = count_objects(masks["pred"], masks["osm"], buf, block, cells.shape)
        df["pred_count"] = total.ravel()
        df["unmatched_count"] = unmatched.ravel()
        parts.append(df[df["cell_id"].isin(cell_ids)].groupby("cell_id")[FIELDS].sum())

    if not parts:
        return pd.DataFrame(columns=["cell_id"] + FIELDS)
    return pd.concat(parts).groupby(level=0)[FIELDS].sum().rename_axis("cell_id").reset_index()


def add_scores(df, low_signal_m2):
    df = df.copy()
    # on cells where osm is complete these are the detector precision and recall
    df["completeness"] = 1 - df["missed_m2"] / df["pred_m2"]
    df["recall_vs_osm"] = df["osm_detected_m2"] / df["osm_m2"]
    df["imagery_coverage"] = df["valid_m2"] / df["total_m2"]
    df["low_signal"] = df["pred_m2"] < low_signal_m2
    df["no_imagery"] = df["imagery_coverage"].fillna(0) < 0.5
    return df


def suggest_cells(scored, n):
    # candidates for labelling: most osm building area
    cols = ["cell_id", "volcano", "dist_km", "osm_m2", "pred_m2", "completeness", "recall_vs_osm"]
    return scored[scored["no_imagery"] == False].sort_values("osm_m2", ascending=False).head(n)[cols]


def detector_metrics(scored, cells):
    # area weighted, osm is the ground truth on complete cells
    sel = scored[scored["cell_id"].isin(cells)]
    g = sel.groupby("volcano")[["pred_m2", "missed_m2", "osm_m2", "osm_detected_m2"]].sum()
    g["cells"] = sel.groupby("volcano").size()
    g.loc["ALL"] = g.sum()
    out = pd.DataFrame(index=g.index)
    out["cells"] = g["cells"].astype(int)
    out["precision"] = 1 - g["missed_m2"] / g["pred_m2"]
    out["recall"] = g["osm_detected_m2"] / g["osm_m2"]
    out["f1"] = 2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"])
    return out.round(3)
