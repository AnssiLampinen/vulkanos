#!/usr/bin/env bash
# Run one numbered batch (sites/batches/bNN.csv) on this machine, with per-site training.
#
#   bash scripts/run_batch.sh b03        # normally started by scripts/launch_all.sh, inside tmux
#
# The home folder is shared by every Aalto machine (NFS, 100 GB quota) and 250 full sites don't fit in
# it, so the work goes to a copy of the repo on this machine's local disk when there is one:
#   1. /l/$USER/vulkanos when /l is writable
#   2. /tmp/vulkanos-$USER when /tmp has at least 60 GB free
#   3. otherwise ~/vulkanos-work/<batch>, and each site's tiles and predictions are deleted once it is
#      done (--clean): about 0.12 GB a site stays (weights, scores, map without imagery)
# VULKANOS_WORK overrides. Every batch uses the .venv of the home clone, made once.
# Small things go to the home folder, under ~/vulkanos-results/:
#   logs/bNN.log              the whole log, read by `launch_all.sh status`
#   results_bNN.tar.gz        summaries, scored cells, training curves, logs
#   weights/<site>.pt         each site's trained weights (~100 MB each), unless NO_WEIGHTS=1
#                             (in case 3 they stay in ~/vulkanos-work/<batch>/models instead)
# Finished sites are skipped on a rerun.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"   # the clone in the home folder
B=${1:-}
if [ -z "$B" ] || [ ! -f "$REPO/sites/batches/$B.csv" ]; then
    echo "usage: bash scripts/run_batch.sh <batch>, batch is one of:"
    ls "$REPO"/sites/batches/b*.csv | xargs -n1 basename | sed 's/\.csv$//' | tr '\n' ' '; echo
    exit 1
fi
OUT="$HOME/vulkanos-results"
mkdir -p "$OUT/logs" "$OUT/weights"
LOG="$OUT/logs/$B.log"
exec > >(tee -a "$LOG") 2>&1

# local disk first, see the top of this file
CLEAN=""
if [ -z "$VULKANOS_WORK" ]; then
    tmp_free=$(df -BG --output=avail /tmp 2>/dev/null | tail -1 | tr -dc 0-9)
    if mkdir -p "/l/$USER" 2>/dev/null && [ -w "/l/$USER" ]; then
        chmod 700 "/l/$USER"
        VULKANOS_WORK="/l/$USER/vulkanos"
    elif [ "${tmp_free:-0}" -ge 60 ]; then
        VULKANOS_WORK="/tmp/vulkanos-$USER"
        mkdir -p "$VULKANOS_WORK" && chmod 700 "$VULKANOS_WORK"
    else
        VULKANOS_WORK="$HOME/vulkanos-work/$B"
        CLEAN=--clean
        NO_WEIGHTS=1
        echo "NOTE: no local disk with room on $(hostname -s) (/tmp has ${tmp_free:-?} GB free), working in the home"
        echo "folder $VULKANOS_WORK and deleting each site's tiles and predictions when it is done"
    fi
fi
W=$VULKANOS_WORK
echo "=== batch $B on $(hostname -s), $(date '+%F %T'), work folder $W"
df -h "$(dirname "$W")" | tail -1 | awk '{print "local disk: " $4 " free of " $2}'

# the local copy is cloned from the home clone, so launch_all.sh pulls from github only once
if [ ! -d "$W/.git" ]; then
    git clone -q "$REPO" "$W"
else
    git -C "$W" pull -q --ff-only || echo "git pull of the local copy failed, running the code as it is"
fi
cd "$W"
# one .venv for all batches, in the home clone (torch is ~6 GB). The first batch to get here makes it,
# the others wait for it. mkdir is atomic, so it works as a lock
export PIP_NO_CACHE_DIR=1
if ! "$REPO/.venv/bin/python" -c "import torch, geopandas, h3, cv2" 2>/dev/null; then
    if mkdir "$REPO/.venv-lock" 2>/dev/null; then
        trap 'rmdir "$REPO/.venv-lock" 2>/dev/null' EXIT
        rm -rf "$REPO/.venv"
        (cd "$REPO" && bash setup.sh)
        rmdir "$REPO/.venv-lock"
    else
        echo "waiting for another batch to finish making $REPO/.venv (remove $REPO/.venv-lock if none is)"
        while [ -d "$REPO/.venv-lock" ]; do sleep 30; done
    fi
fi
[ -e .venv ] || ln -s "$REPO/.venv" .venv
. .venv/bin/activate
# starting weights: from the home clone when it has them, gpu_run.py downloads them otherwise
if [ ! -f models/ercolano-r34/best.pt ] && [ -f "$REPO/models/ercolano-r34/best.pt" ]; then
    mkdir -p models/ercolano-r34
    cp "$REPO/models/ercolano-r34/best.pt" models/ercolano-r34/
fi

SITES=sites/batches/$B.csv
set +e
python scripts/gpu_run.py --sites-csv "$SITES" --radius 10 --skip-done --per-site-train $CLEAN
code=$?
set -e

ids=$(python scripts/gpu_run.py --sites-csv "$SITES" --list-sites)
files=()
for f in data/out/summary_*.csv data/logs; do [ -e "$f" ] && files+=("$f"); done
for s in $ids; do
    for f in data/$s/cells_scored_*.parquet models/$s/site-osm/metrics.jsonl models/$s/site-osm/done; do
        [ -f "$f" ] && files+=("$f")
    done
    if [ -z "$NO_WEIGHTS" ] && [ -f "models/$s/site-osm/best.pt" ] && [ ! -f "$OUT/weights/$s.pt" ]; then
        cp "models/$s/site-osm/best.pt" "$OUT/weights/$s.pt" || echo "could not copy weights of $s (home full?)"
    fi
done
echo
if [ ${#files[@]} -gt 0 ]; then
    tar czf "results_$B.tar.gz" "${files[@]}"
    cp "results_$B.tar.gz" "$OUT/"
    echo "=== batch $B finished $(date '+%F %T'): $OUT/results_$B.tar.gz"
else
    echo "=== batch $B finished $(date '+%F %T') with no results"
fi
[ "$code" = 0 ] && echo "=== all sites ok" || echo "=== some sites failed (listed above). Run launch_all.sh start again to retry only those."
