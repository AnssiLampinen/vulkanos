import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

import compare
import data
import detect
import label
import report

STEPS = ["tiles", "detect", "osm", "compare", "report"]


def site_dir(cfg, s):
    d = cfg["data_dir"] / s["id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def pred_dir(cfg, s):
    return site_dir(cfg, s) / "pred" / f"{detect.detector_tag(cfg['detector'])}_z{cfg['imagery']['zoom']}"


def scored_path(cfg, s):
    return site_dir(cfg, s) / f"cells_scored_{detect.detector_tag(cfg['detector'])}.parquet"


def get_cells(cfg, s, radius=None):
    # grid is made on the first run, or again when --radius is given
    path = site_dir(cfg, s) / "cells.parquet"
    if radius or not path.exists():
        radius = radius or cfg["radius_km"]
        cells = data.h3_cells(s, radius, cfg["h3_res"])
        cells.to_parquet(path)
        print(f"[{s['id']}] {len(cells)} cells within {radius} km")
    return gpd.read_parquet(path)


def get_osm(cfg, s, refresh=False):
    d = site_dir(cfg, s)
    path = d / "osm_buildings.parquet"
    # refetch when the grid changed
    if refresh or not path.exists() or path.stat().st_mtime < (d / "cells.parquet").stat().st_mtime:
        cells = gpd.read_parquet(d / "cells.parquet")
        osm = data.fetch_osm(cells.union_all(), cfg["data_dir"] / "osm_cache")
        osm.to_parquet(path)
        print(f"[{s['id']}] {len(osm)} osm buildings")
    return gpd.read_parquet(path)


def run(cfg, s, steps, radius):
    cells = get_cells(cfg, s, radius)
    zoom = cfg["imagery"]["zoom"]
    chips = data.chips(cells, zoom, cfg["imagery"]["chip_tiles"])
    preds = pred_dir(cfg, s)
    tag = detect.detector_tag(cfg["detector"])

    if "tiles" in steps:
        t0 = time.time()
        tiles = [t for c in chips for t in data.children(c, zoom)]
        counts = data.download_tiles(cfg, tiles)
        print(f"[{s['id']}] {len(tiles)} tiles {dict(counts)} ({time.time() - t0:.0f}s)")

    if "detect" in steps:
        t0 = time.time()
        done, skipped = detect.run_detection(cfg, detect.load_detector(cfg["detector"]), chips, preds)
        print(f"[{s['id']}] detected {done} new chips of {len(chips)} with {tag} ({time.time() - t0:.0f}s)")
        if skipped:
            print(f"[{s['id']}] skipped {skipped} chips with tiles not downloaded yet")

    if "osm" in steps or "compare" in steps or "report" in steps:
        osm = get_osm(cfg, s, refresh="osm" in steps)

    if "compare" in steps:
        sums = compare.score_chips(cfg, chips, set(cells["cell_id"]), osm, preds)
        scored = cells.merge(compare.add_scores(sums, cfg["compare"]["low_signal_m2"]), on="cell_id", how="left")
        scored["detector"] = tag
        scored.to_parquet(scored_path(cfg, s))
        ok = scored[(scored["low_signal"] == False) & (scored["no_imagery"] == False)]
        print(f"[{s['id']}] {len(scored)} cells, {len(ok)} with signal, "
              f"median completeness {ok['completeness'].median():.2f}")
        a = compare.area_summary(scored).iloc[0]
        print(f"[{s['id']}] whole area completeness {a['completeness']:.2f} by m2, {a['house_completeness']:.2f} "
              f"by houses, {a['unmapped_houses']} of {a['houses']} houses not in osm")

    if "report" in steps:
        out = cfg["data_dir"] / "out" / f"{s['id']}_{tag}.html"
        report.cell_map(gpd.read_parquet(scored_path(cfg, s)), s, out, report.overlay(cfg, chips, osm, preds))
        print(f"[{s['id']}] map in {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "validate", "label", "dataset", "train"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--site", action="append", help="volcano or test site id, default all volcanoes")
    p.add_argument("--step", action="append", choices=STEPS, help="default all steps")
    p.add_argument("--radius", type=float, help="make the grid again with this radius in km")
    p.add_argument("--checkpoint", help="use a trained unet, e.g. models/r34/best.pt")
    p.add_argument("--cells", type=Path, default=Path("labels/complete_cells.txt"), help="complete cells file")
    p.add_argument("--suggest", type=int, help="validate: list n cells worth labelling")
    p.add_argument("--all-cells", action="store_true", help="dataset: treat every cell as complete")
    p.add_argument("--positives-everywhere", action="store_true",
                   help="dataset: also use osm buildings outside the complete cells")
    p.add_argument("--dataset", type=Path, help="dataset folder, default training.dataset_dir")
    p.add_argument("--name", default="r34", help="train: run name, saved to models/<name>")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["data_dir"] = Path(cfg["data_dir"])
    if args.checkpoint:
        cfg["detector"]["name"] = "unet"
        cfg["detector"]["checkpoint"] = args.checkpoint
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    dataset_dir = args.dataset or Path(cfg["training"]["dataset_dir"])

    all_sites = {s["id"]: s for s in cfg["volcanoes"] + cfg["test_sites"]}
    for i in args.site or []:
        if i not in all_sites:
            sys.exit(f"unknown site {i}")
    sites = [all_sites[i] for i in args.site] if args.site else cfg["volcanoes"]

    if args.cmd == "run":
        for s in sites:
            run(cfg, s, args.step or STEPS, args.radius)

    elif args.cmd == "validate":
        files = [scored_path(cfg, s) for s in sites]
        scored = [pd.read_parquet(f) for f in files if f.exists()]
        if not scored:
            sys.exit("no scores yet, run compare first")
        scored = pd.concat(scored, ignore_index=True)
        if args.suggest:
            print(compare.suggest_cells(scored, args.suggest).to_string(index=False))
        print(compare.detector_metrics(scored, label.read_cells(args.cells)).to_string())

    elif args.cmd == "label":
        page = []
        for s in sites:
            cells = get_cells(cfg, s, args.radius)
            osm = get_osm(cfg, s)
            # scores can be from an older, smaller grid
            if scored_path(cfg, s).exists():
                scores = pd.read_parquet(scored_path(cfg, s), columns=["cell_id", "osm_m2", "pred_m2", "completeness"])
                cells = cells.merge(scores, on="cell_id", how="left")
            preds = pred_dir(cfg, s)
            overlay = None
            if any(preds.glob("*.png")):
                chips = data.chips(cells, cfg["imagery"]["zoom"], cfg["imagery"]["chip_tiles"])
                overlay = report.overlay(cfg, chips, osm, preds)
            print(f"[{s['id']}] {len(cells)} cells, {len(osm)} osm buildings")
            page.append({"id": s["id"], "name": s["name"], "cells": cells, "osm": osm, "overlay": overlay})
        label.serve(page, args.cells, args.port, not args.no_browser)

    elif args.cmd == "dataset":
        from train import make_dataset

        parts = []
        for s in sites:
            cells = get_cells(cfg, s, args.radius)
            chips = data.chips(cells, cfg["imagery"]["zoom"], cfg["imagery"]["chip_tiles"])
            parts.append((s["id"], chips, get_osm(cfg, s), set(cells["cell_id"])))
        complete = None if args.all_cells else set(label.read_cells(args.cells))
        index = make_dataset(cfg, parts, complete, dataset_dir, args.positives_everywhere)
        if len(index):
            print(index.groupby(["site", "split"]).size().unstack(fill_value=0).to_string())
        print(f"dataset in {dataset_dir}")

    elif args.cmd == "train":
        from train import train

        train(cfg, dataset_dir, Path(cfg["training"]["models_dir"]) / args.name)


if __name__ == "__main__":
    main()
