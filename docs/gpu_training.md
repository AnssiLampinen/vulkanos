# Training the building detector on a GPU machine

These steps train a U-Net on OSM buildings from cells where OSM is complete,
then use it as the detector in the pipeline. Tested flow: CPU smoke run; the
GPU numbers below are estimates for an RTX 5070 (12 GB).

## 1. Setup (once)

```bash
git clone <repo-url> vulkanos && cd vulkanos
python3 -m venv .venv && source .venv/bin/activate
# RTX 50xx (Blackwell) needs a CUDA 12.8+ build of PyTorch. Install it first:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

If you see `no kernel image is available for execution on the device`, the
installed PyTorch was built for an older CUDA; reinstall from the cu128 index.

## 2. Get imagery and OSM for the training sites

Tiles and OSM are downloaded on the GPU machine; `data/` does not need copying.

```bash
python -m vulkanos run --site goma --site ercolano --radius-km 1.5 --step tiles --step osm
```

Larger radius gives more training data (3 km is about 4x the chips).

## 3. Choose complete cells

Only cells where OSM has (nearly) every building should be used as labels,
otherwise the model learns that unmapped buildings are background.

Use the browser labeler. It shows the imagery, OSM outlines (yellow) and the
H3 cells; click cells to mark them complete, then press Submit to write
`labels/complete_cells.txt`.

```bash
python -m vulkanos label --site goma --site ercolano --no-browser
```

Over SSH, forward the port from your own machine and open
http://localhost:8765/ in your local browser:

```bash
ssh -L 8765:localhost:8765 <gpu-machine>
```

The server only listens on localhost. Cells saved for other sites stay in the
file. If detections exist (run the `detect` step first), a "Detections vs OSM"
layer is also available. `validate --suggest 40` lists the cells with the most
OSM building area as candidates. For a first try, `--assume-complete` in
step 4 skips labelling.

## 4. Build the dataset

```bash
python -m vulkanos dataset --site goma --site ercolano --cells-file labels/complete_cells.txt
```

Writes `data/train/`: images, `masks_train/`, `masks_val/`, `index.csv`,
`cell_split.csv` and `val_cells.txt`. 20% of the cells are held out for
validation, whole cells at a time.

## 5. Train

Run inside `tmux` or with `nohup` so an SSH disconnect does not stop it.

```bash
# quick check, a few minutes
python -m vulkanos train --run r34-quick --epochs 5
# full run, ~30-45 min
python -m vulkanos train --run r34
```

Per epoch it prints validation precision, recall, F1 and IoU (pixel level, on
held-out cells). Output: `models/<run>/best.pt`, `last.pt`, `metrics.jsonl`.
On out-of-memory errors lower `--batch-size` (e.g. 4).

## 6. Use the model

```bash
python -m vulkanos run --site goma --site ercolano --detector unet --checkpoint models/r34/best.pt \
    --step detect --step compare --step report
python -m vulkanos validate --site goma --site ercolano --detector unet --checkpoint models/r34/best.pt \
    --cells-file data/train/val_cells.txt
```

Only evaluate on `val_cells.txt`: the other complete cells were used for
training, so their scores are inflated. To use the model by default, set
`detector.name: unet` and `detector.checkpoint` in `config.yaml`.

To use a model on another machine, copy `models/<run>/best.pt`.
