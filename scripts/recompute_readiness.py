# recompute the site readiness columns (old readiness + shrunk / weighted / both) from existing
# cells_scored_<tag>.parquet files, no detection rerun. settings come from config.yaml compare: readiness:
#   python scripts/recompute_readiness.py data/batch_results --tag unet-site-osm
#   python scripts/recompute_readiness.py data/batch_results --summary data/batch_results/volcano_summary.csv
# finds <root>/**/<site>/cells_scored_<tag>.parquet, writes <root>/readiness_<tag>.csv and, with
# --summary, adds/updates the readiness columns in that csv (matched on its site column)
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import compare  # noqa: E402

COLS = ["readiness", "readiness_cells", "low_count_cells", "readiness_shrunk", "readiness_weighted",
        "readiness_both", "shrink_k", "readiness_new_cells"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="folder with <site>/cells_scored_<tag>.parquet somewhere below it")
    ap.add_argument("--tag", default="unet-site-osm")
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config.yaml")
    ap.add_argument("--summary", type=Path, help="csv with a site column to add the readiness columns to")
    a = ap.parse_args()

    r = yaml.safe_load(a.config.read_text())["compare"].get("readiness", {})
    args = (r.get("percentile", 20), r.get("min_houses", 10), r.get("shrink_c", 0.005), r.get("new_min_houses", 1))
    rows = []
    for f in sorted(a.root.rglob(f"cells_scored_{a.tag}.parquet")):
        scored = pd.read_parquet(f, columns=["volcano", "no_imagery", "house_count", "house_unmatched_count",
                                             "house_completeness", "pred_m2", "missed_m2"])
        s = compare.area_summary(scored, *args)
        if s.empty:
            continue
        row = s.iloc[0].to_dict()
        row["site"] = f.parent.name
        rows.append(row)
    if not rows:
        sys.exit(f"no cells_scored_{a.tag}.parquet under {a.root}")
    out = pd.DataFrame(rows).drop_duplicates("site", keep="last").set_index("site")
    path = a.root / f"readiness_{a.tag}.csv"
    out.to_csv(path)
    print(f"{len(out)} sites, readiness columns in {path}")
    zero = out["readiness"] == 0
    for c in ["readiness", "readiness_shrunk", "readiness_weighted", "readiness_both"]:
        print(f"  {c:20s} median {out[c].median():.3f}  ==0 {int((out[c] == 0).sum()):3d}  "
              f"==0 where old readiness was 0: {int((out.loc[zero, c] == 0).sum())}")

    if a.summary:
        summ = pd.read_csv(a.summary)
        cols = [c for c in COLS if c in out]
        new = out[cols].reindex(summ["site"])
        for c in cols:  # existing columns keep their place, new ones go at the end
            summ[c] = new[c].to_numpy()
            if c.endswith("cells"):
                summ[c] = summ[c].astype("Int64")
        summ.to_csv(a.summary, index=False, float_format="%.4f")
        print(f"updated {a.summary}")


if __name__ == "__main__":
    main()
