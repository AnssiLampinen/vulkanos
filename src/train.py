import hashlib
import json
import random
import time

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from compare import block_cells, rasterize_osm
from data import load_chip

IGNORE = 255
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def split_of(cell, val_fraction, seed=0):
    # whole cells go to train or val, so val is ground the model hasn't seen
    h = int(hashlib.sha1(f"{seed}:{cell}".encode()).hexdigest()[:8], 16)
    return "val" if h % 10000 / 10000 < val_fraction else "train"


def edge_band(buildings, w):
    # osm outlines are often a few px off from the imagery, so ignore around the edges
    if w <= 0 or not buildings.any():
        return np.zeros_like(buildings)
    return ndimage.binary_dilation(buildings, iterations=w) & ~ndimage.binary_erosion(buildings, iterations=w)


def chip_labels(buildings, valid, pixel_split, edge_px):
    # 0 background, 1 building, 255 ignored. "pos" pixels only give their buildings to train,
    # the background there can hide unmapped buildings
    unusable = ~valid | edge_band(buildings, edge_px)
    out = {}
    for split in ["train", "val"]:
        use = pixel_split == split
        if split == "train":
            use = use | ((pixel_split == "pos") & buildings)
        mask = buildings.astype(np.uint8)
        mask[unusable | ~use] = IGNORE
        out[split] = mask
    return out


def make_dataset(cfg, sites, complete, out_dir, positives_everywhere=False):
    # sites are (id, chips, osm, cell ids). complete=None uses every cell
    tr = cfg["training"]
    block = cfg["compare"]["block_px"]
    rows = []
    val_cells = []
    for site, chips, osm, all_cells in sites:
        cells = all_cells if complete is None else all_cells & complete
        cell_split = {c: split_of(c, tr["val_fraction"], tr["seed"]) for c in sorted(cells)}
        val_cells += [c for c, s in cell_split.items() if s == "val"]
        if positives_everywhere:
            cell_split.update({c: "pos" for c in all_cells - cells})
        osm = osm.to_crs(3857)

        for chip in chips:
            rgb, valid = load_chip(cfg, chip)
            size = rgb.shape[0]
            blocks = block_cells(chip, size, block, cfg["h3_res"])
            pixel_split = np.array([[cell_split.get(c, "") for c in row] for row in blocks])
            pixel_split = pixel_split.repeat(block, axis=0).repeat(block, axis=1)
            labels = chip_labels(rasterize_osm(osm, chip, size), valid, pixel_split, tr["ignore_edge_px"])
            # skip masks that are almost all ignored
            labels = {k: m for k, m in labels.items() if (m != IGNORE).mean() >= 0.05}
            if not labels:
                continue

            name = f"{site}_{chip.z}_{chip.x}_{chip.y}.png"
            (out_dir / "images").mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(out_dir / "images" / name)
            for split, mask in labels.items():
                (out_dir / f"masks_{split}").mkdir(exist_ok=True)
                Image.fromarray(mask).save(out_dir / f"masks_{split}" / name)
                rows.append({"site": site, "chip": name, "split": split})

    out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.DataFrame(rows, columns=["site", "chip", "split"])
    index.to_csv(out_dir / "index.csv", index=False)
    (out_dir / "val_cells.txt").write_text("".join(c + "\n" for c in val_cells))
    return index


def make_model(encoder, weights="imagenet"):
    return smp.Unet(encoder_name=encoder, encoder_weights=weights, classes=1)


def normalize(rgb):
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    return (x - MEAN) / STD


class Chips(Dataset):
    def __init__(self, root, names, split, crop, augment):
        self.root = root
        self.names = names
        self.split = split
        self.crop = crop
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        rgb = np.asarray(Image.open(self.root / "images" / self.names[i]).convert("RGB"))
        mask = np.asarray(Image.open(self.root / f"masks_{self.split}" / self.names[i]))
        h, w = mask.shape
        c = min(self.crop, h, w)
        if c < h or c < w:
            r0 = random.randint(0, h - c) if self.augment else (h - c) // 2
            c0 = random.randint(0, w - c) if self.augment else (w - c) // 2
            rgb, mask = rgb[r0:r0 + c, c0:c0 + c], mask[r0:r0 + c, c0:c0 + c]
        if self.augment:
            k = random.randint(0, 3)
            rgb, mask = np.rot90(rgb, k), np.rot90(mask, k)
            if random.random() < 0.5:
                rgb, mask = rgb[:, ::-1], mask[:, ::-1]
            # brightness / contrast
            rgb = np.clip(rgb.astype(np.float32) * random.uniform(0.8, 1.2) + random.uniform(-20, 20), 0, 255)
            rgb = rgb.astype(np.uint8)
        return normalize(rgb), torch.from_numpy(np.ascontiguousarray(mask)).long()


def masked_loss(logits, target):
    # bce + dice on the pixels that aren't ignored
    logits = logits[:, 0].float()
    keep = target != IGNORE
    if not keep.any():
        return logits.sum() * 0
    y = (target == 1).float()
    bce = F.binary_cross_entropy_with_logits(logits[keep], y[keep])
    p = torch.sigmoid(logits) * keep
    y = y * keep
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return bce + dice


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    tp = fp = fn = 0
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        losses.append(masked_loss(logits, y).item())
        keep = y != IGNORE
        pred = (torch.sigmoid(logits[:, 0].float()) >= 0.5) & keep
        truth = (y == 1) & keep
        tp += (pred & truth).sum().item()
        fp += (pred & ~truth).sum().item()
        fn += (~pred & truth).sum().item()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"val_loss": float(np.mean(losses)), "precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-9), "iou": tp / max(tp + fp + fn, 1)}


def train(cfg, dataset_dir, out_dir):
    tr = cfg["training"]
    random.seed(tr["seed"])
    np.random.seed(tr["seed"])
    torch.manual_seed(tr["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"  # bf16, needs an rtx 30xx or newer
    print("device", device)

    index = pd.read_csv(dataset_dir / "index.csv")
    train_ds = Chips(dataset_dir, index[index["split"] == "train"]["chip"].tolist(), "train", tr["crop"], True)
    val_ds = Chips(dataset_dir, index[index["split"] == "val"]["chip"].tolist(), "val", tr["crop"], False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(f"empty split: {len(train_ds)} train, {len(val_ds)} val chips")
    print(f"{len(train_ds)} train, {len(val_ds)} val chips")

    bs = tr["batch_size"]
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=tr["workers"], pin_memory=amp,
                          drop_last=len(train_ds) > bs)
    val_dl = DataLoader(val_ds, bs, num_workers=tr["workers"], pin_memory=amp)

    model = make_model(tr["encoder"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tr["lr"], total_steps=tr["epochs"] * len(train_dl),
                                                pct_start=0.05)

    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "metrics.jsonl", "a")
    best = None
    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        t0 = time.time()
        losses = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = masked_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())

        m = evaluate(model, val_dl, device, amp)
        m.update(epoch=epoch, train_loss=float(np.mean(losses)), seconds=round(time.time() - t0, 1))
        log.write(json.dumps(m) + "\n")
        log.flush()
        print(f"epoch {epoch:3d}  train_loss {m['train_loss']:.3f}  val_loss {m['val_loss']:.3f}  "
              f"P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  IoU {m['iou']:.3f}  ({m['seconds']}s)")
        state = {"model": model.state_dict(), "meta": tr, "epoch": epoch, "metrics": m}
        torch.save(state, out_dir / "last.pt")
        if best is None or m["f1"] > best["f1"]:
            best = m
            torch.save(state, out_dir / "best.pt")
    log.close()
    print(f"best epoch {best['epoch']}: F1 {best['f1']:.3f}  P {best['precision']:.3f}  R {best['recall']:.3f}")
