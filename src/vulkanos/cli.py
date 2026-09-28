"""Command-line pipeline.

  python -m vulkanos run --config config.yaml [--volcano vesuvius] [--step detect] [--radius-km 2]
  python -m vulkanos run --site ercolano --radius-km 1.5      (quick accuracy check)
  python -m vulkanos validate --suggest 40
  python -m vulkanos validate --cells-file labels/complete_cells.txt
  python -m vulkanos label --site goma --site ercolano     (pick complete cells in the browser)

Training (see train.py):
  python -m vulkanos dataset --site goma --site ercolano --cells-file labels/complete_cells.txt
  python -m vulkanos train --run r34
  python -m vulkanos run --site goma --detector unet --checkpoint models/r34/best.pt --step detect --step compare --step report
  python -m vulkanos validate --site goma --detector unet --checkpoint models/r34/best.pt --cells-file data/train/val_cells.txt
"""

import argparse
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

from . import aoi, compare, dataset, detect, imagery, labeler, osm, report, validate
from .config import Volcano, load_config, volcano_dir, volcanoes

STEPS = ["aoi", "tiles", "detect", "osm", "compare", "report"]


def _cells_path(vdir: Path) -> Path:
    return vdir / "cells.parquet"


def _scored_path(vdir: Path, detector: str) -> Path:
    return vdir / f"cells_scored_{detector}.parquet"


def _pred_dir(vdir: Path, detector: str, zoom: int) -> Path:
    return vdir / "pred" / f"{detector}_z{zoom}"


def run_volcano(cfg: dict, v: Volcano, steps: list[str], radius_km: float) -> None:
    vdir = volcano_dir(cfg, v)
    img = cfg["imagery"]
    det = cfg["detector"]
    tag = detect.detector_tag(det)
    cmp_cfg = cfg["compare"]
    zoom, provider = img["zoom"], img["provider"]

    if "aoi" in steps or not _cells_path(vdir).exists():
        cells = aoi.h3_cells(v, radius_km, cfg["h3_res"])
        cells.to_parquet(_cells_path(vdir))
        print(f"[{v.id}] aoi: {len(cells)} H3 cells within {radius_km} km")
    cells = gpd.read_parquet(_cells_path(vdir))
    chips = aoi.chips(cells, zoom, img["chip_tiles"])

    if "tiles" in steps:
        tiles = [t for c in chips for t in aoi.chip_children(c, zoom)]
        t0 = time.time()
        counts = imagery.download_tiles(tiles, cfg["data_dir"], provider, img.get("workers", 8))
        print(f"[{v.id}] tiles: {len(tiles)} {counts} ({time.time() - t0:.0f}s)")

    if "detect" in steps:
        detector = detect.load_detector(det)
        t0 = time.time()
        n = detect.run_detection(
            detector, chips, cfg["data_dir"], _pred_dir(vdir, tag, zoom), provider, zoom
        )
        print(f"[{v.id}] detect ({tag}): {n} new chips of {len(chips)} ({time.time() - t0:.0f}s)")

    osm_path = vdir / "osm_buildings.parquet"
    if "osm" in steps:
        _fetch_osm(cfg, v, vdir)
    elif "compare" in steps or "report" in steps:
        _ensure_osm(cfg, v, vdir)

    if "compare" in steps:
        t0 = time.time()
        sums = compare.score_chips(
            chips,
            set(cells["cell_id"]),
            gpd.read_parquet(osm_path),
            _pred_dir(vdir, tag, zoom),
            cfg["data_dir"],
            provider,
            zoom,
            cfg["h3_res"],
            det["threshold"],
            cmp_cfg["osm_buffer_m"],
            cmp_cfg["min_object_m2"],
            cmp_cfg["block_px"],
        )
        scored = cells.merge(compare.add_scores(sums, cmp_cfg["low_signal_m2"]), on="cell_id", how="left")
        scored["detector"] = tag
        scored.to_parquet(_scored_path(vdir, tag))
        ok = scored[~scored["low_signal"].fillna(True) & ~scored["no_imagery"].fillna(True)]
        print(
            f"[{v.id}] compare: {len(scored)} cells, {len(ok)} with signal, "
            f"median completeness {ok['completeness'].median():.2f} ({time.time() - t0:.0f}s)"
        )

    if "report" in steps:
        scored = gpd.read_parquet(_scored_path(vdir, tag))
        overlay = report.overlay_image(
            chips,
            gpd.read_parquet(osm_path),
            dict(
                pred_dir=_pred_dir(vdir, tag, zoom),
                data_dir=cfg["data_dir"],
                provider=provider,
                zoom=zoom,
                threshold=det["threshold"],
                osm_buffer_m=cmp_cfg["osm_buffer_m"],
                min_object_m2=cmp_cfg["min_object_m2"],
            ),
        )
        out = report.cell_map(
            scored, v, cfg["data_dir"] / "out" / f"{v.id}_{tag}.html", overlay
        )
        print(f"[{v.id}] report: {out}")


def combine(cfg: dict, vs: list[Volcano], tag: str) -> gpd.GeoDataFrame | None:
    name = detect.detector_tag(cfg["detector"])
    paths = [_scored_path(volcano_dir(cfg, v), name) for v in vs]
    frames = [gpd.read_parquet(p) for p in paths if p.exists()]
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    out = cfg["data_dir"] / "out" / f"{tag}_cells_scored_{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out)
    return combined


DEFAULT_CELLS_FILE = Path("labels/complete_cells.txt")


def _read_cells(path: Path) -> set[str]:
    if not path.exists():
        raise SystemExit(f"cells file not found: {path}")
    return set(labeler.read_cells_file(path))


def _fetch_osm(cfg: dict, v: Volcano, vdir: Path) -> None:
    cells = gpd.read_parquet(_cells_path(vdir))
    buildings = osm.fetch_buildings(cells.union_all(), cfg["data_dir"] / "osm_cache")
    buildings.to_parquet(vdir / "osm_buildings.parquet")
    print(f"[{v.id}] osm: {len(buildings)} building polygons")


def _ensure_osm(cfg: dict, v: Volcano, vdir: Path) -> Path:
    """OSM buildings for the current AOI; refetched when missing or older than the cells."""
    if not _cells_path(vdir).exists():
        raise SystemExit(f"[{v.id}] run the aoi and tiles steps first")
    osm_path = vdir / "osm_buildings.parquet"
    if not osm_path.exists() or osm_path.stat().st_mtime < _cells_path(vdir).stat().st_mtime:
        print(f"[{v.id}] OSM buildings missing or older than the AOI; fetching")
        _fetch_osm(cfg, v, vdir)
    return osm_path


def label_sites(cfg: dict, vs: list[Volcano]) -> list[labeler.Site]:
    img, det, cmp_cfg = cfg["imagery"], cfg["detector"], cfg["compare"]
    tag = detect.detector_tag(det)
    sites = []
    for v in vs:
        vdir = volcano_dir(cfg, v)
        osm_path = _ensure_osm(cfg, v, vdir)
        cells = gpd.read_parquet(_cells_path(vdir))
        if _scored_path(vdir, tag).exists():
            # Scores may be from an older, smaller AOI: attach them, keep all current cells.
            scores = gpd.read_parquet(_scored_path(vdir, tag))
            stats = [c for c in ("osm_m2", "pred_m2", "completeness") if c in scores.columns]
            cells = cells.merge(pd.DataFrame(scores[["cell_id", *stats]]), on="cell_id", how="left")
        osm_gdf = gpd.read_parquet(osm_path)
        overlay = None
        pred_dir = _pred_dir(vdir, tag, img["zoom"])
        if pred_dir.exists() and any(pred_dir.glob("*.png")):
            overlay = report.overlay_image(
                aoi.chips(cells, img["zoom"], img["chip_tiles"]),
                osm_gdf,
                dict(
                    pred_dir=pred_dir,
                    data_dir=cfg["data_dir"],
                    provider=img["provider"],
                    zoom=img["zoom"],
                    threshold=det["threshold"],
                    osm_buffer_m=cmp_cfg["osm_buffer_m"],
                    min_object_m2=cmp_cfg["min_object_m2"],
                ),
            )
        print(f"[{v.id}] {len(cells)} cells, {len(osm_gdf)} OSM buildings, overlay: {tag if overlay else 'none'}")
        sites.append(labeler.Site(v.id, v.name, cells, osm_gdf, overlay))
    return sites


def build_dataset(cfg: dict, vs: list[Volcano], cells_file: Path | None, out_dir: Path) -> None:
    img, tr = cfg["imagery"], cfg["training"]
    sites, all_cells = [], {}
    for v in vs:
        vdir = volcano_dir(cfg, v)
        osm_path = _ensure_osm(cfg, v, vdir)
        cells = gpd.read_parquet(_cells_path(vdir))
        all_cells[v.id] = set(cells["cell_id"])
        sites.append((v.id, aoi.chips(cells, img["zoom"], img["chip_tiles"]), gpd.read_parquet(osm_path)))
    complete = _read_cells(cells_file) if cells_file else None
    t0 = time.time()
    index = dataset.build_dataset(
        sites,
        complete,
        all_cells,
        out_dir,
        cfg["data_dir"],
        img["provider"],
        img["zoom"],
        cfg["h3_res"],
        cfg["compare"]["block_px"],
        tr["val_fraction"],
        tr["ignore_edge_px"],
        tr.get("seed", 0),
    )
    counts = index.groupby(["site", "split"]).size().unstack(fill_value=0) if len(index) else index
    print(counts.to_string() if len(index) else "no chips written")
    print(f"dataset: {out_dir} ({time.time() - t0:.0f}s)")


def main(argv: list[str] | None = None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.yaml")
    area = argparse.ArgumentParser(add_help=False)
    area.add_argument("--volcano", action="append", help="volcano id (repeatable); default all")
    area.add_argument("--site", action="append", help="test site id from test_sites (repeatable)")
    detector = argparse.ArgumentParser(add_help=False)
    detector.add_argument("--detector", help="override detector.name from config")
    detector.add_argument("--checkpoint", help="override detector.checkpoint (for --detector unet)")

    p = argparse.ArgumentParser(prog="vulkanos")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", parents=[common, area, detector], help="run pipeline steps")
    r.add_argument("--step", action="append", choices=STEPS, help="step (repeatable); default all")
    r.add_argument("--radius-km", type=float, help="override radius_km (reruns aoi)")

    val = sub.add_parser("validate", parents=[common, area, detector], help="detector quality on complete-OSM cells")
    val.add_argument("--suggest", type=int, help="list N candidate cells to check visually")
    val.add_argument("--cells-file", type=Path, help="text file, one complete cell_id per line")

    ds = sub.add_parser("dataset", parents=[common, area], help="build training data from OSM labels")
    ds.add_argument("--cells-file", type=Path, help=f"complete cells, one per line (e.g. {DEFAULT_CELLS_FILE})")
    ds.add_argument("--assume-complete", action="store_true", help="treat every cell as complete (quick tests)")
    ds.add_argument("--out", type=Path, help="output folder; default training.dataset_dir")

    lb = sub.add_parser("label", parents=[common, area, detector], help="pick complete cells in the browser")
    lb.add_argument("--cells-file", type=Path, default=DEFAULT_CELLS_FILE)
    lb.add_argument("--port", type=int, default=8765)
    lb.add_argument("--no-browser", action="store_true", help="do not open a browser (e.g. over SSH)")

    t = sub.add_parser("train", parents=[common], help="train a U-Net building segmenter")
    t.add_argument("--run", required=True, help="run name; checkpoints go to models/<run>/")
    t.add_argument("--dataset", type=Path, help="default training.dataset_dir")
    t.add_argument("--encoder")
    t.add_argument("--epochs", type=int)
    t.add_argument("--batch-size", type=int)
    t.add_argument("--crop", type=int)
    t.add_argument("--lr", type=float)
    t.add_argument("--workers", type=int)
    t.add_argument("--max-batches", type=int, help="limit batches per epoch (smoke tests)")
    t.add_argument("--no-pretrained", action="store_true", help="random encoder init (offline)")

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    tr = cfg["training"]

    if args.cmd == "train":
        from .train import train

        train(
            dataset_dir=args.dataset or Path(tr["dataset_dir"]),
            out_dir=Path(tr["models_dir"]) / args.run,
            arch=tr.get("arch", "Unet"),
            encoder=args.encoder or tr["encoder"],
            encoder_weights=None if args.no_pretrained else "imagenet",
            crop=args.crop or tr["crop"],
            batch_size=args.batch_size or tr["batch_size"],
            epochs=args.epochs or tr["epochs"],
            lr=args.lr or tr["lr"],
            workers=tr["workers"] if args.workers is None else args.workers,
            seed=tr.get("seed", 0),
            max_batches=args.max_batches,
        )
        return

    if args.site and args.volcano:
        p.error("use --volcano or --site, not both")
    vs = volcanoes(cfg, args.site, "test_sites") if args.site else volcanoes(cfg, args.volcano)
    tag = "test_sites" if args.site else "volcanoes"

    if args.cmd == "dataset":
        if bool(args.cells_file) == args.assume_complete:
            p.error("give exactly one of --cells-file or --assume-complete")
        build_dataset(cfg, vs, args.cells_file, args.out or Path(tr["dataset_dir"]))
        return

    if args.detector:
        cfg["detector"]["name"] = args.detector
    if args.checkpoint:
        cfg["detector"]["checkpoint"] = args.checkpoint
    if cfg["detector"]["name"] == "unet" and not cfg["detector"].get("checkpoint"):
        p.error("--detector unet needs --checkpoint or detector.checkpoint in config")

    if args.cmd == "label":
        labeler.serve(label_sites(cfg, vs), args.cells_file, args.port, not args.no_browser)
        return

    if args.cmd == "run":
        steps = args.step or STEPS
        if args.radius_km is not None and "aoi" not in steps:
            steps = ["aoi", *steps]
        radius = args.radius_km if args.radius_km is not None else cfg["radius_km"]
        for v in vs:
            run_volcano(cfg, v, steps, radius)
        if "compare" in steps and combine(cfg, vs, tag) is not None:
            print(f"combined scores: {cfg['data_dir'] / 'out'}")
        return

    scored = combine(cfg, vs, tag)
    if scored is None:
        raise SystemExit("No scored cells yet; run the compare step first.")
    if args.suggest:
        print(validate.suggest_cells(scored, args.suggest).to_string(index=False))
    if args.cells_file:
        print(validate.detector_metrics(scored, sorted(_read_cells(args.cells_file))).to_string())
