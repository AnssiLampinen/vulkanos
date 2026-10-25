"""Detector quality on cells where OSM is believed complete.

On a cell with complete OSM buildings, OSM is ground truth:
  precision = share of detected area near an OSM building
  recall    = share of OSM area near a detection
The house_* columns are the same counted in houses, and house_count_ratio
(detected houses / OSM buildings) shows whether split_houses over- or
under-splits.
"""

import pandas as pd


def suggest_cells(scored: pd.DataFrame, n: int = 40) -> pd.DataFrame:
    """Candidate cells for a visual completeness check: most OSM building area."""
    cols = ["cell_id", "volcano", "dist_km", "osm_m2", "pred_m2", "completeness", "recall_vs_osm"]
    ok = scored[~scored["no_imagery"]]
    return ok.sort_values("osm_m2", ascending=False).head(n)[cols]


def detector_metrics(scored: pd.DataFrame, complete_cells: list[str]) -> pd.DataFrame:
    """Area-weighted precision / recall / F1 per volcano over the given cells."""
    sel = scored[scored["cell_id"].isin(complete_cells)]
    fields = ["pred_m2", "missed_m2", "osm_m2", "osm_detected_m2"]
    g = sel.groupby("volcano")[fields].sum()
    g["cells"] = sel.groupby("volcano").size()
    g.loc["ALL"] = g.sum()
    out = pd.DataFrame(index=g.index)
    out["cells"] = g["cells"].astype(int)
    out["precision"] = 1 - g["missed_m2"] / g["pred_m2"]
    out["recall"] = g["osm_detected_m2"] / g["osm_m2"]
    out["f1"] = 2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"])
    house_fields = ["house_count", "house_unmatched_count", "osm_house_count", "osm_house_detected_count"]
    if all(f in sel.columns for f in house_fields):
        h = sel.groupby("volcano")[house_fields].sum()
        h.loc["ALL"] = h.sum()
        out["house_precision"] = 1 - h["house_unmatched_count"] / h["house_count"]
        out["house_recall"] = h["osm_house_detected_count"] / h["osm_house_count"]
        out["house_count_ratio"] = h["house_count"] / h["osm_house_count"]
    return out.round(3)
