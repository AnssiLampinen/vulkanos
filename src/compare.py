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
FIELDS = ["total_m2", "valid_m2", "pred_m2", "osm_m2", "missed_m2", "osm_detected_m2", "pred_count", "unmatched_count",
          "house_count", "house_unmatched_count", "osm_house_count", "osm_house_detected_count"]


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


def split_houses(pred, min_sep_px):
    # watershed on the distance to the background, so blobs joined by a narrow neck become separate houses
    blobs, n_blobs = ndimage.label(pred, structure=np.ones((3, 3)))
    if n_blobs == 0:
        return np.zeros(pred.shape, np.int32), 0
    dist = ndimage.gaussian_filter(ndimage.distance_transform_edt(pred), 1.0)
    size = 2 * max(int(min_sep_px), 1) + 1
    peaks = pred & (dist == ndimage.maximum_filter(dist, size=size))
    seeds, n = ndimage.label(peaks, structure=np.ones((3, 3)))
    # a small blob next to a big one can lose its peak to the big one, seed it with the whole blob
    seeded = np.atleast_1d(ndimage.maximum(seeds > 0, blobs, np.arange(1, n_blobs + 1))).astype(bool)
    unseeded = np.flatnonzero(~seeded) + 1
    if len(unseeded):
        extra = np.isin(blobs, unseeded)
        seeds[extra] = n + np.searchsorted(unseeded, blobs[extra]) + 1
        n += len(unseeded)
    elevation = np.round((dist.max() - dist) / max(dist.max(), 1e-6) * 65535).astype(np.uint16)
    markers = seeds.astype(np.int32)
    markers[~pred] = -1
    labels = ndimage.watershed_ift(elevation, markers)
    # the background marker floods into building edges, give those pixels the nearest house
    _, (r, c) = ndimage.distance_transform_edt(labels <= 0, return_indices=True)
    labels = labels[r, c]
    labels[~pred] = 0
    present = np.unique(labels[labels > 0])
    lookup = np.zeros(n + 1, np.int32)
    lookup[present] = np.arange(1, len(present) + 1)
    return lookup[labels], len(present)


def house_edges(labels):
    # pixels where two houses touch
    edges = np.zeros(labels.shape, bool)
    left, right = labels[:, :-1], labels[:, 1:]
    edges[:, :-1] |= (left != right) & (left > 0) & (right > 0)
    up, down = labels[:-1, :], labels[1:, :]
    edges[:-1, :] |= (up != down) & (up > 0) & (down > 0)
    return edges


def rasterize_osm(osm, chip, size):
    # osm has to be in EPSG:3857 here
    bounds = mercantile.xy_bounds(chip)
    hits = osm.sindex.query(box(*bounds))
    if len(hits) == 0:
        return np.zeros((size, size), bool)
    return rasterize(osm.geometry.iloc[hits], out_shape=(size, size), transform=from_bounds(*bounds, size, size),
                     fill=0, default_value=1, dtype="uint8").astype(bool)


def rasterize_osm_ids(osm, chip, size):
    # osm row position + 1 for every pixel, 0 is no building
    bounds = mercantile.xy_bounds(chip)
    hits = osm.sindex.query(box(*bounds))
    if len(hits) == 0:
        return np.zeros((size, size), np.int32)
    return rasterize(zip(osm.geometry.iloc[hits], hits + 1), out_shape=(size, size),
                     transform=from_bounds(*bounds, size, size), fill=0, dtype="int32")


def osm_points_by_chip(osm, z):
    # osm rows grouped by the chip holding their representative point, so every building is counted once
    if len(osm) == 0:
        return {}
    pts = osm.geometry.representative_point()
    half = math.pi * 6378137.0
    tx = np.floor((pts.x.to_numpy() + half) / (2 * half) * 2**z).astype(int)
    ty = np.floor((half - pts.y.to_numpy()) / (2 * half) * 2**z).astype(int)
    groups = pd.Series(np.arange(len(pts))).groupby([tx, ty]).groups
    return {k: np.asarray(v) for k, v in groups.items()}


def osm_houses(osm, rows, chip, pred, valid, buf):
    # osm buildings with their point on imagery, returns points as row/col and whether each was detected
    size = pred.shape[0]
    if len(rows) == 0:
        return np.zeros((0, 2)), np.zeros(0, bool)
    west, south, east, north = mercantile.xy_bounds(chip)
    pts = osm.geometry.iloc[rows].representative_point()
    col = np.clip(((pts.x.to_numpy() - west) / (east - west) * size).astype(int), 0, size - 1)
    row = np.clip(((north - pts.y.to_numpy()) / (north - south) * size).astype(int), 0, size - 1)
    keep = valid[row, col]
    row, col, ids = row[keep], col[keep], rows[keep] + 1
    if len(ids) == 0:
        return np.zeros((0, 2)), np.zeros(0, bool)
    # a building hidden under an overlapping one counts as not detected
    detected = ndimage.maximum(dilate(pred, buf), rasterize_osm_ids(osm, chip, size), ids)
    return np.stack([row, col], axis=1), np.atleast_1d(np.asarray(detected, bool))


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


def ragged(blob, px, min_holes):
    # scrub and bare ground give blobs full of holes with pixel-jagged edges, real roofs have clean edges
    _, holes = ndimage.label(ndimage.binary_fill_holes(blob) & ~blob)
    if holes < min_holes:
        return False
    import cv2
    cs, _ = cv2.findContours(np.pad(blob, 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    # outline length against a 2 m simplified outline, buildings stay under ~1.10
    jag = cv2.arcLength(c, True) / max(cv2.arcLength(cv2.approxPolyDP(c, 2.0 / px, True), True), 1)
    return jag > 1.13


def drop_unsure(pred, prob, rgb, min_prob, px, min_holes):
    # false blobs on trees and fields are green and unsure, greenhouses are green but sure
    labels, n = ndimage.label(pred, structure=np.ones((3, 3)))
    if n == 0:
        return pred
    idx = np.arange(1, n + 1)
    green = 2.0 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    is_green = ndimage.mean(green, labels, idx) > 0
    drop = (ndimage.mean(prob, labels, idx) < min_prob) & is_green
    if min_holes:
        for i, sl in enumerate(ndimage.find_objects(labels), 1):
            if is_green[i - 1] and not drop[i - 1]:
                drop[i - 1] = ragged(labels[sl] == i, px, min_holes)
    return np.concatenate([[False], ~drop])[labels]


def chip_masks(cfg, chip, osm, pred_dir):
    prob = load_prob(pred_path(pred_dir, chip))
    if prob is None:
        return None
    rgb, valid = load_chip(cfg, chip)
    b = mercantile.bounds(chip)
    px = pixel_size((b.north + b.south) / 2, cfg["imagery"]["zoom"])
    buf = math.ceil(cfg["compare"]["osm_buffer_m"] / px)
    pred = remove_small(prob >= cfg["detector"]["threshold"], math.ceil(cfg["compare"]["min_object_m2"] / (px * px)))
    pred = pred & valid
    if cfg["compare"]["min_blob_prob"]:
        pred = drop_unsure(pred, prob, rgb, cfg["compare"]["min_blob_prob"], px, cfg["compare"].get("ragged_holes", 0))
    osm = rasterize_osm(osm, chip, prob.shape[0]) & valid
    masks = {
        "valid": valid,
        "pred": pred,
        "osm": osm,
        "missed": pred & ~dilate(osm, buf),  # detected but not in osm
        "osm_detected": osm & dilate(pred, buf),
    }
    return masks, px, buf


def count_objects(pred, osm, buf, block, shape, labels=None):
    # detected objects counted in the block that holds their centroid
    total = np.zeros(shape)
    unmatched = np.zeros(shape)
    if labels is None:
        labels, n = ndimage.label(pred, structure=np.ones((3, 3)))
    else:
        labels, n = labels
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
    osm_by_chip = osm_points_by_chip(osm, chips[0].z) if chips else {}
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

        houses = split_houses(masks["pred"], math.ceil(cfg["compare"]["house_min_sep_m"] / px))
        total, unmatched = count_objects(masks["pred"], masks["osm"], buf, block, cells.shape, houses)
        df["house_count"] = total.ravel()
        df["house_unmatched_count"] = unmatched.ravel()
        points, detected = osm_houses(osm, osm_by_chip.get((chip.x, chip.y), np.zeros(0, int)), chip,
                                      masks["pred"], masks["valid"], buf)
        osm_total = np.zeros(cells.shape)
        osm_detected = np.zeros(cells.shape)
        rows, cols = (points // block).astype(int).T
        np.add.at(osm_total, (rows, cols), 1)
        np.add.at(osm_detected, (rows, cols), detected)
        df["osm_house_count"] = osm_total.ravel()
        df["osm_house_detected_count"] = osm_detected.ravel()
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
    df["house_completeness"] = 1 - df["house_unmatched_count"] / df["house_count"]
    df["house_recall_vs_osm"] = df["osm_house_detected_count"] / df["osm_house_count"]
    df["low_signal"] = df["pred_m2"] < low_signal_m2
    df["no_imagery"] = df["imagery_coverage"].fillna(0) < 0.5
    return df


def area_summary(scored):
    # ratio of sums over cells with imagery, so empty cells don't pull it down
    ok = scored[scored["no_imagery"] == False]
    g = ok.groupby("volcano")
    sums = g[["pred_m2", "missed_m2", "house_count", "house_unmatched_count"]].sum()
    out = pd.DataFrame(index=sums.index)
    out["cells"] = scored.groupby("volcano").size()
    out["coverage"] = g.size() / out["cells"]
    out["completeness"] = 1 - sums["missed_m2"] / sums["pred_m2"]
    out["house_completeness"] = 1 - sums["house_unmatched_count"] / sums["house_count"]
    out["houses"] = sums["house_count"].astype(int)
    out["unmapped_houses"] = sums["house_unmatched_count"].astype(int)
    return out


def suggest_cells(scored, n):
    # candidates for labelling: most osm building area
    cols = ["cell_id", "volcano", "dist_km", "osm_m2", "pred_m2", "completeness", "recall_vs_osm"]
    return scored[scored["no_imagery"] == False].sort_values("osm_m2", ascending=False).head(n)[cols]


def detector_metrics(scored, cells):
    # area weighted, osm is the ground truth on complete cells
    sel = scored[scored["cell_id"].isin(cells)]
    g = sel.groupby("volcano")[["pred_m2", "missed_m2", "osm_m2", "osm_detected_m2", "house_count",
                                "house_unmatched_count", "osm_house_count", "osm_house_detected_count"]].sum()
    g["cells"] = sel.groupby("volcano").size()
    g.loc["ALL"] = g.sum()
    out = pd.DataFrame(index=g.index)
    out["cells"] = g["cells"].astype(int)
    out["precision"] = 1 - g["missed_m2"] / g["pred_m2"]
    out["recall"] = g["osm_detected_m2"] / g["osm_m2"]
    out["f1"] = 2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"])
    out["house_precision"] = 1 - g["house_unmatched_count"] / g["house_count"]
    out["house_recall"] = g["osm_house_detected_count"] / g["osm_house_count"]
    # over 1 means split_houses over-splits
    out["house_count_ratio"] = g["house_count"] / g["osm_house_count"]
    return out.round(3)
