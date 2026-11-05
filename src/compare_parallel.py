"""Parallel version of compare.score_chips. compare.py itself is not changed.

Splits the chips into batches and runs the original compare.score_chips on each batch in its own
process, then adds the per-cell sums together. Every column is a sum over chips, so the result is
the same as the serial run.

Use it in place of `python src/main.py`, with the same arguments:

    python src/main_parallel.py run --site ercolano --checkpoint models/ercolano-r34/best.pt --step compare --step report
"""
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

import compare

serial_score_chips = compare.score_chips  # the original, kept before main_parallel swaps it


def _batch(args):
    cfg, chips, cell_ids, osm, pred_dir = args
    return serial_score_chips(cfg, chips, cell_ids, osm, pred_dir)


def score_chips(cfg, chips, cell_ids, osm, pred_dir, workers=None):
    workers = workers or cfg["compare"].get("workers") or os.cpu_count() or 1
    if workers <= 1 or len(chips) < 2:
        return serial_score_chips(cfg, chips, cell_ids, osm, pred_dir)
    # a few batches per worker so a slow batch doesn't leave the other cores idle
    n = min(len(chips), workers * 4)
    batches = [chips[i::n] for i in range(n)]
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_batch, [(cfg, b, cell_ids, osm, pred_dir) for b in batches]))
    parts = [p for p in parts if len(p)]
    if not parts:
        return serial_score_chips(cfg, [], cell_ids, osm, pred_dir)
    return (pd.concat(parts).groupby("cell_id")[compare.FIELDS].sum().reset_index())
