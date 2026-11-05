#!/usr/bin/env bash
# Train the ercolano detector from OSM alone and compare it with r34 on the hand labelled cells.
# Writes only new folders: models/osm-all, models/osm-suspect, data/train_osm, data/train_suspect.
#   bash run_osm.sh 2>&1 | tee run_osm.log
set -e
cd "$(dirname "$0")"
python src/main.py run --site ercolano --radius 10 --step tiles --step osm
python src/main.py dataset --site ercolano --all-cells --exclude labels/complete_cells.txt --dataset data/train_osm
python src/main.py train --dataset data/train_osm --name osm-all
python src/main.py run --site ercolano --step detect --step compare --checkpoint models/osm-all/last.pt
python src/main.py dataset --site ercolano --all-cells --suspect unet-osm-all --exclude labels/complete_cells.txt \
    --dataset data/train_suspect
python src/main.py train --dataset data/train_suspect --name osm-suspect
python src/main.py run --site ercolano --step detect --step compare --step report --checkpoint models/osm-suspect/last.pt
python src/main.py validate --site ercolano --checkpoint models/osm-suspect/last.pt
if [ -f models/ercolano-r34/best.pt ]; then
    python src/main.py run --site ercolano --step detect --step compare --checkpoint models/ercolano-r34/best.pt
    python src/main.py validate --site ercolano --checkpoint models/ercolano-r34/best.pt
fi
