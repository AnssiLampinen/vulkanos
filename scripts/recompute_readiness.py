# recompute the site summary columns (completeness, houses, old readiness + shrunk / weighted / both) from
# existing cells_scored_<tag>.parquet files, no detection rerun. settings come from config.yaml compare: readiness:
#
#   python scripts/recompute_readiness.py data/batch_results --summary data/batch_results/volcano_summary.csv
#
# with the nearest-site reruns: unet-site-near results replace the unet-site-osm ones of the same site,
# written to a new csv so the old summary stays as it is
#
#   python scripts/recompute_readiness.py data/near_results data/batch_results --tag unet-site-near,unet-site-osm \
#       --summary data/batch_results/volcano_summary.csv --out data/near_results/volcano_summary_near.csv
#
# each root is searched for <site>/cells_scored_<tag>.parquet, both as files and inside results_*.tar.gz
# (no need to unpack them). --tag is a priority list: per site the first tag found wins. a site found more
# than once for the same tag (a batch that was retried) takes the last one in file name order.
# writes <first root>/readiness_<tags>.csv with one row per site and, with --summary, a copy of that csv
# with its columns updated (matched on its site column) to --out, or in place without --out. sites with
# results but no row in --summary get one, name / volcano_number / lat / lon from sites/**/*.csv
import argparse
import io
import re
import sys
import tarfile
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import compare  # noqa: E402

READ = ["volcano", "no_imagery", "house_count", "house_unmatched_count", "house_completeness", "pred_m2", "missed_m2"]
COLS = ["detector", "house_completeness", "completeness", "coverage", "houses", "unmapped_houses", "cells",
        "readiness", "readiness_cells", "low_count_cells", "readiness_shrunk", "readiness_weighted",
        "readiness_both", "shrink_k", "readiness_new_cells", "donor"]
INTS = ["houses", "unmapped_houses", "cells", "readiness_cells", "low_count_cells", "readiness_new_cells"]


def slug(name):
    # same as main.py: "Volcán Pichincha" -> volcan_pichincha
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def site_info(folder):
    # site id -> name, volcano_number, lat, lon from every sites/**/*.csv, first one found wins
    rows = []
    for f in sorted(folder.rglob("*.csv")):
        d = pd.read_csv(f).rename(columns={"volcano_name": "name"})
        if {"name", "lat", "lon"} <= set(d.columns):
            rows.append(d[[c for c in ["name", "volcano_number", "lat", "lon"] if c in d]])
    d = pd.concat(rows, ignore_index=True)
    d["site"] = d["name"].astype(str).map(slug)
    return d.drop_duplicates("site").set_index("site")


def find(roots, tags):
    # {(tag, site): (source name, parquet bytes or path)}, later sources overwrite earlier ones
    pat = re.compile(r"(?:^|/)data/([^/]+)/cells_scored_(.+)\.parquet$")
    donor_pat = re.compile(r"(?:^|/)models/([^/]+)/site-near/donor\.txt$")
    found, donors = {}, {}
    for root in roots:
        for f in sorted(root.rglob("*")):
            p = f.as_posix()
            m = pat.search(p)
            if m and m.group(2) in tags:
                found[(m.group(2), m.group(1))] = (p, f)
            elif f.name.endswith(".tar.gz") and f.is_file():
                print(f"reading {p}", flush=True)
                with tarfile.open(f, "r|gz") as t:  # stream: one pass, no gzip rewinds
                    for mem in t:
                        m, d = pat.search(mem.name), donor_pat.search(mem.name)
                        if m and m.group(2) in tags:
                            found[(m.group(2), m.group(1))] = (f"{p}:{mem.name}", t.extractfile(mem).read())
                        elif d:
                            donors[d.group(1)] = t.extractfile(mem).read().decode().strip()
            elif donor_pat.search(p):
                donors[donor_pat.search(p).group(1)] = f.read_text().strip()
    return found, donors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", type=Path, nargs="+", help="folders with results_*.tar.gz or <site>/cells_scored_<tag>.parquet")
    ap.add_argument("--tag", default="unet-site-osm", help="detector tag, or comma separated tags, first found wins")
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config.yaml")
    ap.add_argument("--summary", type=Path, help="csv with a site column to update the columns of")
    ap.add_argument("--only-summary-sites", action="store_true", help="don't add rows for sites missing from --summary")
    ap.add_argument("--out", type=Path, help="where the updated --summary goes, default: overwrite --summary")
    a = ap.parse_args()

    tags = a.tag.split(",")
    r = yaml.safe_load(a.config.read_text())["compare"].get("readiness", {})
    args = (r.get("percentile", 20), r.get("min_houses", 10), r.get("shrink_c", 0.005), r.get("new_min_houses", 1))
    found, donors = find(a.roots, tags)
    rows = {}
    for tag in reversed(tags):  # first tag last, so it overwrites
        for (t, site), (src, data) in sorted(found.items()):
            if t != tag:
                continue
            scored = pd.read_parquet(io.BytesIO(data) if isinstance(data, bytes) else data, columns=READ)
            s = compare.area_summary(scored, *args)
            if s.empty:
                continue
            row = s.iloc[0].to_dict()
            row.update(site=site, detector=tag, source=src, donor=donors.get(site, "") if tag.endswith("near") else "")
            rows[site] = row
    if not rows:
        sys.exit(f"no cells_scored_{{{a.tag}}}.parquet under {', '.join(map(str, a.roots))}")
    out = pd.DataFrame(rows.values()).set_index("site").sort_index()
    path = a.roots[0] / f"readiness_{'+'.join(tags)}.csv"
    out.to_csv(path)
    print(f"{len(out)} sites ({', '.join(f'{n} {t}' for t, n in out['detector'].value_counts().items())}), in {path}")
    for c in ["readiness", "readiness_shrunk", "readiness_weighted", "readiness_both"]:
        print(f"  {c:20s} median {out[c].median():.3f}  ==0 {int((out[c] == 0).sum()):3d}")

    if a.summary:
        summ = pd.read_csv(a.summary)
        extra = out.index.difference(summ["site"])
        if len(extra) and not a.only_summary_sites:  # sites with results but no row yet, e.g. b11-b17
            info = site_info(Path(__file__).resolve().parents[1] / "sites").reindex(extra)
            add = info.reset_index().rename(columns={"index": "site"})
            for c in ["readiness_percentile", "readiness_min_houses"]:
                if c in summ:
                    add[c] = summ[c].dropna().iloc[0] if summ[c].notna().any() else None
            summ = pd.concat([summ, add], ignore_index=True)
            print(f"  added {len(extra)} sites that had no row: {', '.join(extra)}")
        new = out.reindex(summ["site"])
        missing = summ["site"][new["detector"].isna().to_numpy()].tolist()
        if missing:
            print(f"  {len(missing)} sites of {a.summary.name} have no results, their columns are left empty: "
                  f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
        for c in COLS:  # existing columns keep their place, new ones go at the end
            if c not in new:
                continue
            summ[c] = new[c].to_numpy()
            if c in INTS:
                summ[c] = summ[c].astype("Int64")
        if "map" in summ:  # maps follow the detector, e.g. maps/out/<site>_unet-site-near.html
            has = new["detector"].notna().to_numpy()
            summ.loc[has, "map"] = "maps/out/" + summ.loc[has, "site"] + "_" + new["detector"][has].to_numpy() + ".html"
        for c in ["volcano_number", "readiness_percentile", "readiness_min_houses"]:
            if c in summ:
                summ[c] = pd.to_numeric(summ[c], errors="coerce").round().astype("Int64")
        dest = a.out or a.summary
        dest.parent.mkdir(parents=True, exist_ok=True)
        summ.to_csv(dest, index=False, float_format="%.4f")
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
