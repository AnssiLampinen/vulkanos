# Output columns

Two tables come out of a run:

- `data/out/summary_<detector>.csv`: one row per volcano (site). Use this to rank volcanoes.
  `data/out/<site>_<detector>_summary.csv` is the same row for one site, written by `compare`.
- `data/<site>/cells_scored_<detector>.parquet`: one row per H3 cell (resolution 8, about 0.7 km²)
  inside the site radius. The map popups show the same values.

"Detected" means found by the house detector in the imagery, "in OSM" means it overlaps an OSM
building outline after growing that outline by `osm_buffer_m` (2 m), because imagery and OSM are often
a little misaligned. Cells where less than half the area has imagery (`no_imagery`) are left out of
every site-level number.

## Which number ranks the volcanoes

**`readiness`** in the summary csv. It is between 0 and 1, lower is worse mapped, so the volcano with
the lowest `readiness` needs volunteer mappers most. For a "needs mapping" multiplier use
`1 - readiness`.

How it is made: take the cells with imagery and at least `min_houses` (10) detected houses. For each
cell, `house_completeness` is the share of its detected houses that are in OSM. `readiness` is the
20th percentile of those values: 20% of the cells are mapped worse than this, 80% better. A low
percentile is used so the badly mapped part of the area decides the score, not the average, which
well-mapped town centres would pull up. Percentile and `min_houses` are set in `config.yaml` under
`compare: readiness:`.

`readiness` does not grow with the number of people or houses, so it stays separate from the
population factor. `unmapped_houses_est` is the absolute count if you want size to matter.

## Summary csv, one row per site

| column | meaning |
|---|---|
| `site` | site id, folder name in `data/` |
| `name` | site name |
| `detector` | detector used, e.g. `unet-ercolano-r34` |
| `cells` | H3 cells within the radius |
| `coverage` | share of `cells` with imagery |
| `completeness` | `1 - missed_m2 / pred_m2` summed over the site: share of the detected building area that is in OSM |
| `house_completeness` | `1 - house_unmatched_count / house_count` summed over the site: share of detected houses that are in OSM |
| `houses` | detected houses, sum of `house_count` |
| `unmapped_houses` | detected houses not in OSM, sum of `house_unmatched_count` |
| `readiness` | 20th percentile of cell `house_completeness` over cells with >= 10 houses, see above |
| `readiness_cells` | cells that went into `readiness`; under ~20 the score is shaky |
| `low_count_cells` | cells with 1-9 detected houses, left out of `readiness` |
| `unmapped_houses_est` | sum of the cell `unmapped_houses_est`: houses missing from OSM after taking out the detector's false positives |
| `cells_priority_95` | cells with `mapping_priority >= 0.95`, i.e. at least 95% sure OSM misses more than `ok_gap` (10%) of the houses there |
| `mean_mapping_priority` | mean of the cell `mapping_priority` |

The per-site `<site>_<detector>_summary.csv` also has `readiness_percentile` and
`readiness_min_houses`, the settings it was made with.

## Cell table, one row per H3 cell

Identity and position:

| column | meaning |
|---|---|
| `cell_id` | H3 cell id, resolution 8 |
| `volcano` | site id the cell belongs to |
| `dist_km` | distance from the site centre to the cell centre |
| `volcano_dist_km` | distance from the volcano summit to the cell centre |
| `detector` | detector used |
| `geometry` | cell outline (EPSG:4326) |

Sums over the cell (pixels are counted in the cell holding them, objects in the cell holding their centre):

| column | meaning |
|---|---|
| `total_m2` | ground area covered by image chips |
| `valid_m2` | area with real imagery (not Esri's grey "no data" tiles) |
| `pred_m2` | detected building area |
| `osm_m2` | OSM building area |
| `missed_m2` | detected building area that is not in OSM |
| `osm_detected_m2` | OSM building area the detector also found |
| `pred_count` | detected blobs (touching buildings count as one) |
| `unmatched_count` | detected blobs that touch no OSM building |
| `house_count` | detected houses: blobs split into separate houses (watershed, houses at least `house_min_sep_m` = 4 m apart) |
| `house_unmatched_count` | detected houses that touch no OSM building |
| `osm_house_count` | OSM buildings whose point lies on imagery in this cell |
| `osm_house_detected_count` | of those, how many the detector found |

Scores:

| column | formula | meaning |
|---|---|---|
| `completeness` | `1 - missed_m2 / pred_m2` | share of detected building area in OSM |
| `house_completeness` | `1 - house_unmatched_count / house_count` | share of detected houses in OSM, what `readiness` is built from |
| `recall_vs_osm` | `osm_detected_m2 / osm_m2` | share of OSM area the detector found; on fully mapped cells this is the detector's recall |
| `house_recall_vs_osm` | `osm_house_detected_count / osm_house_count` | the same in houses |
| `imagery_coverage` | `valid_m2 / total_m2` | share of the cell with imagery |
| `low_signal` | `pred_m2 < 200` | almost nothing detected, greyed out on the map |
| `no_imagery` | `imagery_coverage < 0.5` | left out of all scores |

Mapping priority: how sure we are that OSM really misses houses in the cell. 2 of 2 houses unmapped is
weak evidence, 300 of 500 is strong, and a raw count can't tell them apart. With `n = house_count`,
`k = house_unmatched_count`, detector false positive rate `fp = detector_fp` (0.029) and
`ok_gap` (0.1), the null hypothesis is "OSM misses at most 10% of the houses here". Under it a detected
house shows up as unmapped with mean rate `p0 = fp + (1 - fp) * ok_gap`. The count is beta-binomial
with that mean and `concentration` (30), because detector errors cluster within a cell.
`detector_fp` and `concentration` were fitted on the Ercolano cells that are fully mapped in OSM
(`python src/main.py validate` prints the fit).

| column | formula | meaning |
|---|---|---|
| `gap_p` | `P(at least k unmapped of n, given the null)`, clipped to >= 1e-15 | p-value, small means OSM clearly misses more than 10% here |
| `gap_significance` | `-log10(gap_p)` | ranking key for cells, higher is surer; 15 is the cap |
| `mapping_priority` | `1 - gap_p` | 0..1 confidence that the cell is under-mapped |
| `unmapped_houses_est` | `max(0, (k - fp * n) / (1 - fp))` | houses really missing from OSM, with the detector's false positives taken out |

These four are empty for `no_imagery` cells.
