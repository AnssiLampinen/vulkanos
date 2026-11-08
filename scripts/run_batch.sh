#!/usr/bin/env bash
# Run one numbered batch (sites/batches/bNN.csv) on this machine, with per-site training.
#
#   bash scripts/run_batch.sh b03        # normally started by scripts/launch_all.sh, inside tmux
#
# The home folder is shared by every Aalto machine (NFS, 100 GB quota) and 250 sites don't fit in it,
# so the work happens in a copy of the repo on this machine's local disk, /l/$USER/vulkanos
# (VULKANOS_WORK overrides). That copy has its own .venv, data and models. Only small things go back
# to the home folder, under ~/vulkanos-results/:
#   logs/bNN.log              the whole log, read by `launch_all.sh status`
#   results_bNN.tar.gz        summaries, scored cells, training curves, logs
#   weights/<site>.pt         each site's trained weights (~100 MB each), unless NO_WEIGHTS=1
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

# local disk, falls back to the home folder (fine for a test, not for all 250 sites)
if [ -z "$VULKANOS_WORK" ]; then
    if mkdir -p "/l/$USER" 2>/dev/null && [ -w "/l/$USER" ]; then
        chmod 700 "/l/$USER"
        VULKANOS_WORK="/l/$USER/vulkanos"
    else
        VULKANOS_WORK="$HOME/vulkanos-work/$B"
        echo "WARNING: no /l/$USER on $(hostname -s), working in $VULKANOS_WORK (home quota is 100 GB for everything)"
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
export PIP_NO_CACHE_DIR=1
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c "import torch, geopandas, h3, cv2" 2>/dev/null; then
    rm -rf .venv
    bash setup.sh
fi
. .venv/bin/activate
# starting weights: from the home clone when it has them, gpu_run.py downloads them otherwise
if [ ! -f models/ercolano-r34/best.pt ] && [ -f "$REPO/models/ercolano-r34/best.pt" ]; then
    mkdir -p models/ercolano-r34
    cp "$REPO/models/ercolano-r34/best.pt" models/ercolano-r34/
fi

SITES=sites/batches/$B.csv
set +e
python scripts/gpu_run.py --sites-csv "$SITES" --radius 10 --skip-done --per-site-train
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
