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


What every output column means: [docs/columns.md](docs/columns.md).

## Best pipeline

The most accurate setup so far (2026-11), all on the `priority-filtered` branch:

1. Detector: our U-Net `models/ercolano-r34/best.pt` (the default in config.yaml). On the ercolano
   complete cells it gets area F1 0.91 and house precision 0.97; geobase_unet gets 0.42, yolo 0.13.
2. Shape filter on: `min_blob_prob: 0.8`, `ragged_holes: 5` in config.yaml. Without it scrub and bare
   ground on the slopes show up as many false positive houses.
3. Mapping priority: each cell ranked by how significant its unmapped houses are (see below).

```
python src/main_parallel.py run --site ercolano --step compare --step report
```

`main_parallel.py` is `main.py` with the compare step spread over all CPU cores, same results.
Add `--step tiles --step detect` (or no `--step`) for a new site. The map is
`data/out/<site>_unet-ercolano-r34.html`, the ranking is the Mapping priority layer.

The `mapping-priority` branch alone does not have the shape filter, so it gives more false positives.

## Running

```
python src/main.py run --site ercolano --radius 1.5
python src/main.py run --site vesuvius --step detect --step compare --step report
```

Steps are tiles, detect, osm, compare and report, all of them by default. The grid is made on the first
run with radius_km from config.yaml, `--radius` makes it again. Everything goes to `data/<site>/` and
the map to `data/out/<site>_<detector>.html`. Without `--site` it runs all the volcanoes.

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
