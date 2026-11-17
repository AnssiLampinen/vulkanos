#!/usr/bin/env bash
# Rerun the volcanoes that kept r34 (fewer than 300 OSM buildings, nothing to retrain on) with the weights
# of the nearest volcano that got its own retrained weights. Run on agda from ~/vulkanos-team, after the
# b01..b17 batches are finished:
#
#   bash scripts/launch_near.sh plan        # print what would be rerun and with whose weights, changes nothing
#   bash scripts/launch_near.sh retrain     # stage 1 (optional, ~1 h): retrain the PNG / Solomons sites whose
#                                           #   imagery was blank, with the z17 fallback, so they can be donors
#   bash scripts/launch_near.sh start       # stage 2: plan + start the reruns, one batch nNN per machine
#   bash scripts/launch_near.sh status      # progress of whichever stage was started last
#   bash scripts/launch_near.sh stop
#
# Every batch runs on its machine in a detached tmux session (vulkanos-retrain for stage 1, vulkanos-near
# for stage 2), so it keeps going when you log out. Watch one with: ssh <host> -t tmux attach -t vulkanos-near
# (detach with Ctrl-b then d). Running start again retries failed sites; running batches are left alone.
# Stage 1 and 2 use the same machines, so wait for `status` to show stage 1 finished before `start`.
# The plan comes from the finished batches themselves (models/<site>/site-osm/done), see
# scripts/near_plan.py. Each rerun site gets models/<site>/site-near/best.pt, a link to the donor's
# weights (nothing is copied or changed, models/ercolano-r34 is only read), and new results under the
# detector tag unet-site-near, so the earlier unet-site-osm results stay as they are:
#   ~/vulkanos-work/nNN/data/out/<site>_unet-site-near.html          maps
#   ~/vulkanos-results/results_nNN.tar.gz                            scored cells, summary, donor.txt, logs
#   data/near/rerun/plan.csv                                         site, donor, distance (km)
set -e
cd "$(dirname "$0")/.."
REPO=$PWD
CMD=${1:-status}
shift || true
NEAR=$REPO/data/near
py() { if [ -x .venv/bin/python ]; then .venv/bin/python "$@"; else python3 "$@"; fi; }

stage() {  # retrain | rerun, the one started last
    if [ -f "$NEAR/stage" ]; then cat "$NEAR/stage"; else echo rerun; fi
}
launch() {  # $1 = retrain | rerun, rest = launch_all.sh args
    local s=$1; shift
    if [ "$s" = retrain ]; then
        SESSION=vulkanos-retrain HOSTS=$NEAR/retrain/hosts.csv BATCH_DIR=$NEAR/retrain GPU_ARGS=--per-site-train \
            bash scripts/launch_all.sh "$@"
    else
        SESSION=vulkanos-near HOSTS=$NEAR/rerun/hosts.csv BATCH_DIR=$NEAR/rerun GPU_ARGS="--near-plan $NEAR/rerun/plan.csv" \
            bash scripts/launch_all.sh "$@"
    fi
}

case "$CMD" in
plan)
    py scripts/near_plan.py rerun --dry-run "$@"
    ;;
retrain)
    git pull --ff-only || echo "git pull failed, running the code as it is"
    py scripts/near_plan.py retrain "$@"
    [ -f "$NEAR/retrain/hosts.csv" ] || { echo "nothing to retrain"; exit 0; }
    echo retrain > "$NEAR/stage"
    launch retrain start
    echo "check with: bash scripts/launch_near.sh status. When all are finished: bash scripts/launch_near.sh start"
    ;;
start)
    git pull --ff-only || echo "git pull failed, running the code as it is"
    py scripts/near_plan.py rerun "$@"
    echo rerun > "$NEAR/stage"
    launch rerun start
    echo "check with: bash scripts/launch_near.sh status"
    ;;
status|stop)
    s=$(stage)
    echo "stage: $s"
    launch "$s" "$CMD"
    ;;
*)
    echo "usage: bash scripts/launch_near.sh plan|retrain|start|status|stop"
    exit 1
    ;;
esac
