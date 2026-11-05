import argparse
import traceback
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import mercantile
import pandas as pd
import yaml

import compare
import data
import detect
import label
import report
import volcanoes

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
    # computed on every load so a changed summit in volcanoes.yaml shows up without a new grid
    return data.add_volcano_dist(gpd.read_parquet(path), s["volcano"])


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


def prioritise(cfg, scored):
    c = cfg["compare"]
    return compare.add_priority(scored, c.get("detector_fp", 0.029), c.get("concentration", 30), c.get("ok_gap", 0.1))


def run(cfg, s, steps, radius):
    cells = get_cells(cfg, s, radius)
    zoom = cfg["imagery"]["zoom"]
    chips = data.chips(cells, zoom, cfg["imagery"]["chip_tiles"])
    preds = pred_dir(cfg, s)
    tag = detect.detector_tag(cfg["detector"])

    if "tiles" in steps:
        t0 = time.time()
        tiles = [t for c in chips for t in data.children(c, zoom)]
        tiles += list(mercantile.tiles(*data.frame(s, cfg["map_km"]), zooms=report.FRAME_ZOOM))
        counts = data.download_tiles(cfg, tiles)
        print(f"[{s['id']}] {len(tiles)} tiles {dict(counts)} ({time.time() - t0:.0f}s)")

    if "detect" in steps:
        t0 = time.time()
        done, skipped = detect.run_detection(cfg, detect.load_detector(cfg["detector"], cfg.get("device")), chips, preds)
        print(f"[{s['id']}] detected {done} new chips of {len(chips)} with {tag} ({time.time() - t0:.0f}s)")
        if skipped:
            print(f"[{s['id']}] skipped {skipped} chips with tiles not downloaded yet")

    if "osm" in steps or "compare" in steps or "report" in steps:
        osm = get_osm(cfg, s, refresh="osm" in steps)

    if "compare" in steps:
        sums = compare.score_chips(cfg, chips, set(cells["cell_id"]), osm, preds)
        scored = cells.merge(compare.add_scores(sums, cfg["compare"]["low_signal_m2"]), on="cell_id", how="left")
        scored = prioritise(cfg, scored)
        scored["detector"] = tag
        scored.to_parquet(scored_path(cfg, s))
        ok = scored[(scored["low_signal"] == False) & (scored["no_imagery"] == False)]
        print(f"[{s['id']}] {len(scored)} cells, {len(ok)} with signal, "
              f"median completeness {ok['completeness'].median():.2f}")
        r = cfg["compare"].get("readiness", {})
        pct, min_houses = r.get("percentile", 20), r.get("min_houses", 10)
        summary = compare.area_summary(scored, pct, min_houses)
        a = summary.iloc[0]
        print(f"[{s['id']}] whole area completeness {a['completeness']:.2f} by m2, {a['house_completeness']:.2f} "
              f"by houses, {a['unmapped_houses']} of {a['houses']} houses not in osm")
        print(f"[{s['id']}] mapping readiness {a['readiness']:.2f} (p{pct} house completeness over "
              f"{a['readiness_cells']} cells with >= {min_houses} houses, {a['low_count_cells']} cells with fewer left out)")
        out = cfg["data_dir"] / "out" / f"{s['id']}_{tag}_summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.assign(site=s["id"], detector=tag, readiness_percentile=pct, readiness_min_houses=min_houses).to_csv(out)
        print(f"[{s['id']}] summary in {out}")
        print(f"[{s['id']}] cells most in need of mapping:")
        print(compare.top_cells(scored, 10).round(2).to_string(index=False))

    if "report" in steps:
        out = cfg["data_dir"] / "out" / f"{s['id']}_{tag}.html"
        # relative so the map in data/out finds them, forward slashes for windows
        tiles = os.path.relpath(cfg["data_dir"] / "tiles" / cfg["imagery"]["provider"], out.parent)
        tiles = tiles.replace("\\", "/") + "/{z}/{x}/{y}.jpg"
        # priority again so changed detector_fp / concentration / ok_gap show up without running compare
        report.cell_map(prioritise(cfg, gpd.read_parquet(scored_path(cfg, s))), s, out, report.overlay(cfg, chips, osm, preds),
                        report.imagery(cfg, s), report.outlines(cfg, chips, osm, preds), tiles,
                        data.frame(s, cfg["map_km"]))
        print(f"[{s['id']}] map in {out}")


def summary(cfg, sites):
    # one row per site, to compare sites side by side
    rows = []
    for s in sites:
        f = scored_path(cfg, s)
        if not f.exists():
            print(f"[{s['id']}] no scores yet")
            continue
        scored = prioritise(cfg, gpd.read_parquet(f))
        r = cfg["compare"].get("readiness", {})
        a = compare.area_summary(scored, r.get("percentile", 20), r.get("min_houses", 10)).iloc[0].to_dict()
        ok = scored[scored["no_imagery"] == False]
        a.update(site=s["id"], name=s["name"], detector=detect.detector_tag(cfg["detector"]),
                 unmapped_houses_est=int(ok["unmapped_houses_est"].sum()),
                 cells_priority_95=int((ok["mapping_priority"] >= 0.95).sum()),
                 mean_mapping_priority=ok["mapping_priority"].mean())
        rows.append(a)
    if not rows:
        return None
    first = ["site", "name", "detector"]
    out = pd.DataFrame(rows)
    out = out[first + [c for c in out.columns if c not in first]]
    path = cfg["data_dir"] / "out" / f"summary_{detect.detector_tag(cfg['detector'])}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(out.round(3).to_string(index=False))
    print(f"summary in {path}")
    return out


def slug(name):
    # "Volcán Pichincha" -> volcan_pichincha, used as the site id and folder name
    import re
    import unicodedata

    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def csv_sites(path):
    # name,lat,lon per row, the point is taken as the summit
    out = []
    for r in pd.read_csv(path).itertuples():
        sid = slug(r.name)
        v = {"id": sid, "name": r.name, "lat": float(r.lat), "lon": float(r.lon), "source": "csv"}
        out.append({"id": sid, "name": r.name, "lat": v["lat"], "lon": v["lon"], "volcano": v})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "validate", "label", "dataset", "train", "volcanoes", "summary"])
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
    p.add_argument("--device", help="auto (default), cuda, cuda:1, cpu or mps")
    p.add_argument("--workers", type=int, help="threads for tile downloads and chip loading")
    p.add_argument("--detect-batch", type=int, help="chips per gpu batch when detecting, default 16")
    p.add_argument("--sites-csv", type=Path, help="extra sites from a name,lat,lon csv; without --site runs all of them")
    p.add_argument("--stop-on-error", action="store_true", help="run: stop at the first failing site")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["data_dir"] = Path(cfg["data_dir"])
    if args.checkpoint:
        cfg["detector"]["name"] = "unet"
        cfg["detector"]["checkpoint"] = args.checkpoint
    if args.device:
        cfg["device"] = args.device
    if args.workers:
        cfg["imagery"]["workers"] = args.workers
    if args.detect_batch:
        cfg["detector"]["batch_size"] = args.detect_batch
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    dataset_dir = args.dataset or Path(cfg["training"]["dataset_dir"])

    vfile = cfg.get("volcanoes_file", "volcanoes.yaml")
    if args.cmd == "volcanoes":
        volcanoes.refresh_osm(vfile, cfg["data_dir"] / "osm_cache", args.site)
    summits = volcanoes.load(vfile)
    for s in cfg["volcanoes"]:
        v = summits[s["id"]]
        s.update({"name": s.get("name", v["name"]), "lat": v["lat"], "lon": v["lon"], "volcano": v})
    for s in cfg["test_sites"]:
        s["volcano"] = summits[s["volcano"]]

    extra = csv_sites(args.sites_csv) if args.sites_csv else []
    all_sites = {s["id"]: s for s in cfg["volcanoes"] + cfg["test_sites"] + extra}
    for i in args.site or []:
        if i not in all_sites:
            sys.exit(f"unknown site {i}, known: {' '.join(all_sites)}")
    sites = [all_sites[i] for i in args.site] if args.site else (extra or cfg["volcanoes"])

    if args.cmd == "volcanoes":
        for v in summits.values():
            print(f"{v['id']:12} {v['lat']:10.6f} {v['lon']:11.6f}  {v['source']}")

    elif args.cmd == "run":
        failed = []
        for s in sites:
            try:
                run(cfg, s, args.step or STEPS, args.radius)
            except Exception:
                # one bad site (osm timeout, no imagery) shouldn't stop an overnight run
                if args.stop_on_error:
                    raise
                traceback.print_exc()
                failed.append(s["id"])
                print(f"[{s['id']}] FAILED, going on with the next site")
        if len(sites) > 1:
            summary(cfg, sites)
        if failed:
            sys.exit(f"failed sites: {' '.join(failed)}")

    elif args.cmd == "summary":
        summary(cfg, sites)

    elif args.cmd == "validate":
        files = [scored_path(cfg, s) for s in sites]
        scored = [pd.read_parquet(f) for f in files if f.exists()]
        if not scored:
            sys.exit("no scores yet, run compare first")
        scored = pd.concat(scored, ignore_index=True)
        if args.suggest:
            print(compare.suggest_cells(scored, args.suggest).to_string(index=False))
        print(compare.detector_metrics(scored, label.read_cells(args.cells)).to_string())
        print("mapping priority null fitted on these cells:", compare.fit_null(scored, label.read_cells(args.cells)))

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
