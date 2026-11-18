#!/usr/bin/env bash
# Start all numbered batches at once, one per Aalto machine, and check on them.
#
#   bash scripts/launch_all.sh start          # every batch in sites/batches/hosts.csv
#   bash scripts/launch_all.sh start b03 b07  # only these
#   bash scripts/launch_all.sh status         # one line per batch, from the logs in the home folder
#   bash scripts/launch_all.sh stop b03       # kill a batch's tmux session
#
# Run it on any Aalto Linux machine, from the clone in your home folder (e.g. ~/vulkanos-team).
# It ssh's to each machine and starts scripts/run_batch.sh there in a detached tmux session called
# "vulkanos". ssh to the machines must work: log in to the first one with `ssh -A` so your key
# comes along. Use another machine for a batch by editing sites/batches/hosts.csv.
set -e
cd "$(dirname "$0")/.."
REPO=$PWD
CMD=${1:-status}
shift || true
# HOSTS, BATCH_DIR and GPU_ARGS are set by scripts/launch_near.sh for the nearest-site reruns
HOSTS=${HOSTS:-sites/batches/hosts.csv}
BATCH_DIR=${BATCH_DIR:-$REPO/sites/batches}
GPU_ARGS=${GPU_ARGS:---per-site-train}
# tmux session on each machine, launch_near.sh uses its own so old "vulkanos" windows don't block it
SESSION=${SESSION:-vulkanos}
# tr: the csv files may have windows line ends
BATCHES=${*:-$(tail -n +2 "$HOSTS" | tr -d '\r' | cut -d, -f1)}
host_of() { tr -d '\r' < "$HOSTS" | awk -F, -v b="$1" '$1 == b {print $2}'; }
OUT="$HOME/vulkanos-results"
SSH="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

case "$CMD" in
start)
    git pull --ff-only || echo "git pull failed, running the code as it is"
    # one .venv in the home clone, shared by every batch
    if ! .venv/bin/python -c "import torch, geopandas, h3, cv2" 2>/dev/null; then
        rm -rf .venv
        PIP_NO_CACHE_DIR=1 bash setup.sh
    fi
    # download the starting weights once, every batch copies them from here
    if [ ! -f models/ercolano-r34/best.pt ]; then
        mkdir -p models/ercolano-r34
        curl -L --fail -o models/ercolano-r34/best.pt.part \
            https://github.com/AnssiLampinen/vulkanos/raw/models/ercolano-r34/best.pt
        mv models/ercolano-r34/best.pt.part models/ercolano-r34/best.pt
    fi
    for b in $BATCHES; do
        h=$(host_of "$b")
        [ -z "$h" ] && { echo "$b: not in $HOSTS"; continue; }
        printf '%s on %-12s ' "$b" "$h"
        $SSH "$h" "
            if tmux has-session -t $SESSION 2>/dev/null; then
                # a finished batch keeps its window open (press Enter); close it so a retry can start.
                # finished = its log says so, or no run_batch.sh runs here (an older batch's window)
                if tail -3 '$OUT/logs/$b.log' 2>/dev/null | grep -q '^=== batch $b finished' \
                        || ! pgrep -u \$USER -f '^bash /[^ ]*scripts/run_batch[.]sh' >/dev/null; then
                    tmux kill-session -t $SESSION
                else
                    echo 'already running, left alone'; exit 0
                fi
            fi
            gpu=\$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')
            tmux new-session -d -s $SESSION \"BATCH_DIR='$BATCH_DIR' GPU_ARGS='$GPU_ARGS' bash '$REPO/scripts/run_batch.sh' $b; echo; echo done, press Enter; read\"
            echo \"started (gpu before start: \$gpu, users: \$(who | awk '{print \$1}' | sort -u | tr '\n' ' '))\"
        " || echo "ssh failed"
    done
    echo
    echo "check with: bash scripts/launch_all.sh status"
    ;;
status)
    for b in $BATCHES; do
        h=$(host_of "$b")
        log="$OUT/logs/$b.log"
        if [ ! -f "$log" ]; then
            printf '%s %-12s not started\n' "$b" "$h"
            continue
        fi
        n=$(grep -oE '^=== [a-z0-9_]+ detect with (its own|nearest-site) weights ok' "$log" | sort -u | wc -l)
        total=$(($(wc -l < "$BATCH_DIR/$b.csv") - 1))
        last=$(grep '^=== ' "$log" | tail -1 | cut -c1-90)
        printf '%s %-12s %2s/%s sites, %s\n' "$b" "$h" "$n" "$total" "$last"
    done
    ls "$OUT"/results_b*.tar.gz 2>/dev/null | sed 's/^/finished: /'
    ;;
stop)
    for b in $BATCHES; do
        h=$(host_of "$b")
        printf '%s on %-12s ' "$b" "$h"
        $SSH "$h" "tmux kill-session -t $SESSION && echo stopped || echo 'not running'" || echo "ssh failed"
    done
    ;;
*)
    echo "usage: bash scripts/launch_all.sh start|status|stop [batch ...]"
    exit 1
    ;;
esac
