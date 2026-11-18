"""Plan the rerun of the sites that kept r34 (too few OSM buildings to retrain on) with the weights of the
nearest volcano that did get its own retrained weights. Run on an Aalto machine from ~/vulkanos-team,
normally through scripts/launch_near.sh:

    python scripts/near_plan.py retrain --dry-run   # print what it finds
    python scripts/near_plan.py retrain             # stage 1: data/near/retrain/rNN.csv + hosts.csv
    python scripts/near_plan.py rerun               # stage 2: data/near/rerun/plan.csv, nNN.csv, hosts.csv

What each site got is read from the finished batches, not guessed. For every batch (sites/batches/bNN.csv,
then data/near/retrain/rNN.csv, a later batch replacing an earlier result for the same site) the work
folder is taken from ~/vulkanos-results/logs/<batch>.log ("=== batch <batch> on <host>, <time>, work
folder <W>"), and <W>/models/<site>/site-osm/done says either "fine-tuned from ..." (own weights) or
"only N osm buildings ... keeping ..." (an r34 copy).

A site is a donor when it was fine-tuned, its weights are found, and both detectors saw houses there:
  - r34 found > 0 houses. None at all with >= 300 OSM buildings means the imagery was blank (Esri's
    "Map data not yet available" tiles, PNG and the Solomons), so those weights learned nothing.
    Stage 1 retrains these sites from scratch with the z17 fallback, so they become donors for their
    neighbours instead of sending them to a donor 2000+ km away.
  - its own weights found at least --min-keep (10%) of r34's houses: a fine-tune that collapsed to ~0
    (butajiri_silti_field) is not used.

Stage 2 targets: every r34-copy site, plus fine-tuned sites that still fail the donor test (with --no-bad
they are only listed). Each gets the nearest donor by great-circle distance. Donor weights are found at
<W>/models/<site>/site-osm/best.pt; when <W> was on the batch host's own disk (/l or /tmp) they are copied
with scp to ~/vulkanos-results/weights/<batch>/<site>.pt, else run_batch.sh's ~/vulkanos-results/weights/<site>.pt.
"""
import argparse
import math
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()
HOSTS = "agda,zarquon,fenchurch,hactar,prefect,wonko,zarniwoop,lunkwill,wowbagger"


def slug(name):
    # same as src/main.py slug, kept here so this runs without the pipeline's imports
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


def batch_work(batch, results):
    # host and work folder of a batch, from the line run_batch.sh writes to its log at every start
    log = results / "logs" / f"{batch}.log"
    if not log.exists():
        return None, None
    host = work = None
    for line in log.read_text(errors="replace").splitlines():
        m = re.match(rf"=== batch {batch} on (\S+), .*, work folder (\S+)", line)
        if m:
            host, work = m.group(1), Path(m.group(2))  # the last start wins
    return host, work


def houses(work, site, tag):
    f = work / "data" / site / f"cells_scored_{tag}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f, columns=["house_count", "no_imagery"])
    return int(d.loc[d["no_imagery"] == False, "house_count"].sum())


def scan(args):
    results = Path(args.results).expanduser()
    csvs = sorted((ROOT / "sites" / "batches").glob("b*.csv")) + sorted((ROOT / "data" / "near" / "retrain").glob("r*.csv"))
    sites = {}
    for csv in csvs:
        b = csv.stem
        host, work = batch_work(b, results)
        if work is None and b.startswith("b"):
            work = Path(args.work_root).expanduser() / b  # no log: try the usual home work folder
        for r in pd.read_csv(csv).itertuples():
            s = slug(r.name)
            row = dict(site=s, name=r.name, lat=float(r.lat), lon=float(r.lon),
                       volcano_number=getattr(r, "volcano_number", ""), batch=b, host=host or "",
                       work=str(work or ""), status="missing", osm=None, r34_houses=None, own_houses=None)
            if work is not None:
                done = work / "models" / s / "site-osm" / "done"
                if done.exists():
                    t = done.read_text()
                    m = re.search(r"(\d+) osm buildings", t)
                    row["osm"] = int(m.group(1)) if m else None
                    row["status"] = "trained" if t.startswith("fine-tuned") else "r34_copy"
                    row["r34_houses"] = houses(work, s, "unet-ercolano-r34")
                    row["own_houses"] = houses(work, s, "unet-site-osm")
            if s not in sites or row["status"] != "missing":
                sites[s] = row  # a later batch with a result replaces the earlier one
    df = pd.DataFrame(list(sites.values()))
    if df.empty:
        sys.exit("no batches found in sites/batches")
    tr = df["status"] == "trained"
    r34 = df["r34_houses"].fillna(0)
    own = df["own_houses"].fillna(0)
    df["donor_ok"] = tr & (r34 > 0) & (own > 0) & (own >= args.min_keep * r34)
    df["reason"] = ""
    df.loc[df["status"] == "r34_copy", "reason"] = "r34_copy"
    df.loc[tr & (r34 == 0), "reason"] = "bad_imagery"
    df.loc[tr & (r34 > 0) & ~df["donor_ok"], "reason"] = "collapsed"
    df["weights"] = ""
    return df


def find_weights(row, results, fetch):
    w = Path(row.work) / "models" / row.site / "site-osm" / "best.pt"
    if w.exists():
        return str(w)
    if row.host and not str(row.work).startswith(str(HOME)):
        # the batch ran on that host's own disk: copy the weights to the home folder once
        copy = results / "weights" / row.batch / f"{row.site}.pt"
        if copy.exists():
            return str(copy)
        if fetch:
            copy.parent.mkdir(parents=True, exist_ok=True)
            print(f"  scp {row.host}:{w} -> {copy}")
            r = subprocess.run(["scp", "-q", "-o", "ConnectTimeout=10", f"{row.host}:{w}", str(copy)])
            if r.returncode == 0 and copy.exists():
                return str(copy)
    home_copy = results / "weights" / f"{row.site}.pt"  # run_batch.sh's copy, made when there was room
    if home_copy.exists() and row.batch.startswith("b"):
        return str(home_copy)
    return ""


def overview(df):
    tr = df["status"] == "trained"
    print(f"{len(df)} sites in {df['batch'].nunique()} batches: {tr.sum()} fine-tuned ({df['donor_ok'].sum()} usable "
          f"as donors), {(df['status'] == 'r34_copy').sum()} kept r34, {(df['status'] == 'missing').sum()} with no model")
    for r in ["bad_imagery", "collapsed"]:
        s = df[df["reason"] == r]
        if len(s):
            print(f"  {r}: {' '.join(s['site'])}")
    miss = df[df["status"] == "missing"]
    if len(miss):
        print(f"  no model (failed or not run, left alone): {' '.join(miss['site'])}")


def write_batches(sites, out, hosts, prefix):
    # round robin over the sites sorted by longitude: even batch sizes, one batch per host
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{prefix}*.csv"):
        old.unlink()
    sites = sites.sort_values(["lon", "lat"]).reset_index(drop=True)
    lines = ["person,host"]
    for k, h in enumerate(hosts):
        part = sites.iloc[k::len(hosts)]
        if part.empty:
            continue
        name = f"{prefix}{k + 1:02d}"
        part[["name", "lat", "lon", "volcano_number"]].to_csv(out / f"{name}.csv", index=False)
        lines.append(f"{name},{h}")
    (out / "hosts.csv").write_text("\n".join(lines) + "\n")
    print(f"{len(lines) - 1} batches {prefix}01.. and hosts.csv in {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["retrain", "rerun"])
    p.add_argument("--results", default="~/vulkanos-results", help="where run_batch.sh put logs and weights")
    p.add_argument("--work-root", default="~/vulkanos-work", help="batch work folders when a log is missing")
    p.add_argument("--hosts", default=HOSTS, help="comma list, one batch per host (gallumbits left out, its ssh "
                   "asks a password)")
    p.add_argument("--min-keep", type=float, default=0.1, help="donor needs own-weights houses >= this share of r34's")
    p.add_argument("--no-bad", action="store_true", help="rerun: don't rerun fine-tuned sites that fail the donor test")
    p.add_argument("--no-fetch", action="store_true", help="don't scp weights from other hosts' own disks")
    p.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = p.parse_args()
    results = Path(args.results).expanduser()
    hosts = [h for h in args.hosts.split(",") if h]
    near = ROOT / "data" / "near"

    df = scan(args)
    overview(df)
    if args.stage == "retrain":
        bad = df[df["reason"] == "bad_imagery"]
        print(f"\nstage 1: retrain {len(bad)} sites whose imagery was blank: {' '.join(bad['site'])}")
        if not args.dry_run and len(bad):
            write_batches(bad, near / "retrain", hosts, "r")
        return

    print("\nlooking for donor weights")
    for i in df.index[df["donor_ok"]]:
        df.at[i, "weights"] = find_weights(df.loc[i], results, not args.no_fetch and not args.dry_run)
    lost = df["donor_ok"] & (df["weights"] == "")
    if lost.any():
        print(f"  no weights found for {lost.sum()} fine-tuned sites, not used as donors: {' '.join(df.loc[lost, 'site'])}")
    df.loc[lost, "donor_ok"] = False
    donors = df[df["donor_ok"]]
    if donors.empty:
        sys.exit("no donors with weights found")

    want = ["r34_copy"] + ([] if args.no_bad else ["bad_imagery", "collapsed"])
    targets = df[df["reason"].isin(want)].copy()
    picks = []
    for t in targets.itertuples():
        d = donors.apply(lambda r: km(t.lat, t.lon, r.lat, r.lon), axis=1)
        j = d.idxmin()
        picks.append((donors.at[j, "site"], round(float(d[j]), 1), donors.at[j, "weights"]))
    targets["donor"], targets["donor_km"], targets["donor_weights"] = zip(*picks)
    print(f"\nstage 2: {len(targets)} sites to rerun with the nearest donor's weights, donor distance median "
          f"{targets['donor_km'].median():.0f} km, max {targets['donor_km'].max():.0f} km")
    print(targets[["site", "reason", "donor", "donor_km"]].sort_values("donor_km").to_string(index=False))
    far = targets[targets["donor_km"] > 1000]
    if len(far):
        print(f"\nWARNING: {len(far)} sites have no donor within 1000 km. If they are in PNG / the Solomons, "
              f"run stage 1 first: bash scripts/launch_near.sh retrain")
    if args.dry_run:
        return
    out = near / "rerun"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["site", "name", "lat", "lon", "volcano_number", "batch", "work", "reason", "osm", "r34_houses",
            "own_houses", "donor", "donor_km", "donor_weights"]
    targets[cols].to_csv(out / "plan.csv", index=False)
    df.to_csv(out / "all_sites.csv", index=False)
    print(f"\nplan in {out / 'plan.csv'}")
    write_batches(targets, out, hosts, "n")


if __name__ == "__main__":
    main()
