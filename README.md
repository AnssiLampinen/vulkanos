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

## Volcano coordinates

Summits are in `volcanoes.yaml`. Each volcano has the Smithsonian GVP coordinates, an `osm:` line
for the OpenStreetMap summit node and an empty `lat`/`lon` to set by hand. The first one that is set
wins: hand-set, then OSM, then GVP. Fill in the OSM lines (only those lines get rewritten) and print
what is used with

```
python src/main.py volcanoes
```

The volcano grids are centred on the summit, so after changing one make the grid again with
`--radius`. Every cell gets `volcano_dist_km`, the distance to the summit, also for test sites.

## Mapping priority

`compare` ranks cells by how sure we are that OSM misses houses there, not by the raw count: 2 of 2
houses unmapped is weak evidence, 300 of 500 is strong. Each cell gets a one-sided beta-binomial test
against "OSM misses at most `ok_gap` of the houses here", with the detector's false positives
(`detector_fp`, `concentration`) fitted on the complete cells; `validate` prints the fit. Columns:
`gap_p`, `gap_significance` (-log10 p, the ranking key), `mapping_priority` (1 - p, the 0..1
multiplier for the combined ranking) and `unmapped_houses_est`. The map has a Mapping priority layer.

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
