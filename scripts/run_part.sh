#!/usr/bin/env bash
# Run one person's share of the volcanoes on an Aalto GPU machine, start to finish.
#
#   bash scripts/run_part.sh <name>      # anssi adit miro janna khoi hieu duc
#   bash scripts/run_part.sh <name> --site-epochs 5    # extra args go to scripts/gpu_run.py
#
# Every volcano gets its own weights: r34 is fine-tuned on that site's OSM buildings (models/<site>/site-osm)
# and the site is then detected with them. models/ercolano-r34/best.pt itself is never changed.
#
# First run makes .venv (bash setup.sh), later runs reuse it. The work runs inside a tmux session
# called "vulkanos", so you can close the ssh window and it keeps going. Run the same command again
# to look at it (it re-attaches), or `tmux attach -t vulkanos`. Finished sites are skipped on a
# rerun. At the end results_<name>.tar.gz in the repo folder has the summary, scored cells and logs.
set -e
cd "$(dirname "$0")/.."
NAME=$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')
shift || true
EXTRA="$*"
SITES=sites/parts/$NAME.csv
if [ -z "$NAME" ] || [ ! -f "$SITES" ]; then
    echo "usage: bash scripts/run_part.sh <name>, name is one of:"
    ls sites/parts/*.csv | xargs -n1 basename | sed 's/\.csv$//' | grep -v hosts | tr '\n' ' '; echo
    exit 1
fi
HOST=$(tr -d '\r' < sites/parts/hosts.csv | awk -F, -v n="$NAME" '$1 == n {print $2}')
SESSION=vulkanos

# outside tmux: start (or re-attach to) the session and run this script inside it
if [ -z "$TMUX" ]; then
    command -v tmux >/dev/null || { echo "tmux not found"; exit 1; }
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "already running, attaching. Detach again with Ctrl-b then d."
        sleep 2
        exec tmux attach -t "$SESSION"
    fi
    if [ -n "$HOST" ] && [ "$(hostname -s)" != "$HOST" ]; then
        echo "NOTE: you are on $(hostname -s), your machine is $HOST. Fine if $HOST was busy or down."
        sleep 3
    fi
    exec tmux new-session -s "$SESSION" \
        "bash '$PWD/scripts/run_part.sh' '$NAME' $EXTRA; echo; echo 'done. Press Enter to close this window.'; read"
fi

# inside tmux from here
echo "vulkanos part '$NAME' on $(hostname -s), $(wc -l < "$SITES") lines in $SITES (incl. header)"
echo "Leave it running: Ctrl-b then d detaches, then you can log out."
df -h "$HOME" | tail -1 | awk '{print "home disk: " $4 " free of " $2}'
git pull --ff-only || echo "git pull failed, running the code as it is"
export PIP_NO_CACHE_DIR=1   # the torch wheels would fill the home quota twice over
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c "import torch, geopandas, h3" 2>/dev/null; then
    rm -rf .venv
    bash setup.sh
fi
. .venv/bin/activate
mkdir -p data/logs
set +e
python scripts/gpu_run.py --sites-csv "$SITES" --radius 10 --skip-done --per-site-train $EXTRA 2>&1 | tee -a "data/logs/part_$NAME.log"
code=${PIPESTATUS[0]}
set -e

# small results bundle for Anssi: summaries, scored cells (own weights and r34), training logs, logs
# (no maps, tiles or weights, those are big)
ids=$(python scripts/gpu_run.py --sites-csv "$SITES" --list-sites)
files=(data/out/summary_*.csv data/logs)
for s in $ids; do
    for f in data/$s/cells_scored_*.parquet models/$s/site-osm/metrics.jsonl models/$s/site-osm/done; do
        [ -f "$f" ] && files+=("$f")
    done
done
tar czf "results_$NAME.tar.gz" "${files[@]}"
echo
echo "results: $PWD/results_$NAME.tar.gz ($(du -h "results_$NAME.tar.gz" | cut -f1))"
[ "$code" = 0 ] && echo "all sites ok" || echo "some sites failed (listed above). Run the same command again to retry only those."
