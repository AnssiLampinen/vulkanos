"""Training data from OSM buildings in cells where OSM is believed complete.

Each chip becomes one RGB image plus one label mask per split:
  0 = background, 1 = building, 255 = ignore (not used in loss or metrics).

Pixels are ignored when they lie outside the complete cells, belong to the
other split, have no imagery, or sit on a building edge (OSM footprints are
often shifted a few pixels from the imagery).

The split is spatial: whole H3 cells go to train or val, so validation
measures how the model does on ground it has not seen.

With positives_everywhere, OSM buildings in the other (not complete) cells of
the AOI are also used as "building" training pixels, while their background
stays ignored: it may hide unmapped buildings, so it cannot be trusted as
"not building". Validation is unchanged (complete val cells only).
"""

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import mercantile
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from .compare import block_cells, rasterize_osm
from .imagery import load_chip

IGNORE = 255
SPLITS = ("train", "val")
POSITIVE_ONLY = "pos"  # pixel_split value: cell used for building pixels only (train)


def split_of(cell_id: str, val_fraction: float, seed: int = 0) -> str:
    """Deterministic train/val assignment of a cell."""
    h = int(hashlib.sha1(f"{seed}:{cell_id}".encode()).hexdigest()[:8], 16)
    return "val" if (h % 10_000) / 10_000 < val_fraction else "train"


def edge_band(buildings: np.ndarray, width_px: int) -> np.ndarray:
    """Pixels within width_px of a building boundary (both sides)."""
    if width_px <= 0 or not buildings.any():
        return np.zeros_like(buildings)
    grown = ndimage.binary_dilation(buildings, iterations=width_px)
    shrunk = ndimage.binary_erosion(buildings, iterations=width_px)
    return grown & ~shrunk


def chip_labels(
    buildings: np.ndarray, valid: np.ndarray, pixel_split: np.ndarray, ignore_edge_px: int
) -> dict[str, np.ndarray]:
    """Label mask per split for one chip.

    pixel_split holds "train", "val", POSITIVE_ONLY or "" (unused). POSITIVE_ONLY
    pixels add their buildings to the train mask and leave background ignored.
    """
    base = buildings.astype(np.uint8)
    unusable = ~valid | edge_band(buildings, ignore_edge_px)
    out = {}
    for split in SPLITS:
        mask = base.copy()
        labelled = pixel_split == split
        if split == "train":
            labelled = labelled | ((pixel_split == POSITIVE_ONLY) & buildings)
        mask[unusable | ~labelled] = IGNORE
        out[split] = mask
    return out


def build_dataset(
    sites: list[tuple[str, list[mercantile.Tile], gpd.GeoDataFrame]],
    complete_cells: set[str] | None,
    all_cells: dict[str, set[str]],
    out_dir: Path,
    data_dir: Path,
    provider: str,
    zoom: int,
    res: int,
    block_px: int,
    val_fraction: float,
    ignore_edge_px: int,
    seed: int = 0,
    min_labelled_frac: float = 0.05,
    positives_everywhere: bool = False,
) -> pd.DataFrame:
    """Write images and masks for each (site id, chips, OSM buildings).

    complete_cells=None treats every cell of every site as complete.
    positives_everywhere also uses buildings from the other cells (see module doc).
    Returns the index table, also written to out_dir/index.csv.
    """
    rows = []
    split_rows = []
    for site, chips, osm in sites:
        cells = all_cells[site] if complete_cells is None else all_cells[site] & complete_cells
        cell_split = {c: split_of(c, val_fraction, seed) for c in sorted(cells)}
        split_rows += [{"site": site, "cell_id": c, "split": s} for c, s in cell_split.items()]
        if positives_everywhere:
            cell_split.update({c: POSITIVE_ONLY for c in all_cells[site] - cells})
        osm_3857 = osm.to_crs(3857)

        for chip in chips:
            rgb, valid = load_chip(data_dir, provider, chip, zoom)
            size = rgb.shape[0]
            blocks = block_cells(chip, size, block_px, res)
            block_split = np.array([[cell_split.get(c, "") for c in row] for row in blocks])
            pixel_split = np.repeat(np.repeat(block_split, block_px, axis=0), block_px, axis=1)
            buildings = rasterize_osm(osm_3857, chip, size)
            labels = chip_labels(buildings, valid, pixel_split, ignore_edge_px)

            name = f"{site}_{chip.z}_{chip.x}_{chip.y}.png"
            wrote_image = False
            for split, mask in labels.items():
                labelled = mask != IGNORE
                if labelled.mean() < min_labelled_frac:
                    continue
                if not wrote_image:
                    (out_dir / "images").mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb).save(out_dir / "images" / name)
                    wrote_image = True
                (out_dir / f"masks_{split}").mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(out_dir / f"masks_{split}" / name)
                rows.append(
                    {
                        "site": site,
                        "chip": name,
                        "split": split,
                        "labelled_frac": round(float(labelled.mean()), 4),
                        "building_frac": round(float((mask == 1).sum() / max(labelled.sum(), 1)), 4),
                    }
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.DataFrame(rows)
    index.to_csv(out_dir / "index.csv", index=False)
    splits = pd.DataFrame(split_rows)
    splits.to_csv(out_dir / "cell_split.csv", index=False)
    val_cells = splits.loc[splits["split"] == "val", "cell_id"] if len(splits) else []
    (out_dir / "val_cells.txt").write_text("".join(f"{c}\n" for c in val_cells))
    info = {
        "sites": [site for site, _, _ in sites],
        "complete_cells": "all" if complete_cells is None else len(splits),
        "positives_everywhere": positives_everywhere,
        "val_fraction": val_fraction,
        "ignore_edge_px": ignore_edge_px,
        "seed": seed,
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    return index
