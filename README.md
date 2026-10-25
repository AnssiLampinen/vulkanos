# vulkanos

How complete are OpenStreetMap buildings around volcanoes? We detect buildings in Esri satellite imagery,
compare them with OSM and give every H3 cell (res 8, about 0.7 km2) a completeness score: the share of
detected building area that is already in OSM.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On an RTX 50xx install torch from the cu128 index first:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

## Running

```
python src/main.py run --site ercolano --radius 1.5
python src/main.py run --site vesuvius --step detect --step compare --step report
```

Steps are tiles, detect, osm, compare and report, all of them by default. The grid is made on the first
run with radius_km from config.yaml, `--radius` makes it again. Everything goes to `data/<site>/` and
the map to `data/out/<site>_<detector>.html`. Without `--site` it runs all the volcanoes.

## Training our own detector

Get tiles and OSM for the training sites:

```
python src/main.py run --site ercolano --site goma --radius 5 --step tiles --step osm
```

Pick the cells where OSM has basically every building. Click cells and press Submit, they get saved
to `labels/complete_cells.txt`. Over ssh use `ssh -L 8765:localhost:8765 <host>` and `--no-browser`.

```
python src/main.py label --site ercolano --site goma
```

Make the dataset (20% of the complete cells go to validation) and train:

```
python src/main.py dataset --site ercolano --site goma
python src/main.py train --name r34
```

`--positives-everywhere` also uses the OSM buildings outside the complete cells, with their background
ignored. `--epochs 5` is good for a quick try, lower `--batch-size` if the GPU runs out of memory.

Then use the model with `--checkpoint`:

```
python src/main.py run --site goma --checkpoint models/r34/best.pt --step detect --step compare --step report
python src/main.py validate --site ercolano --site goma --checkpoint models/r34/best.pt --cells data/train/val_cells.txt
```

Only validate on `val_cells.txt`, the other complete cells were used for training.
