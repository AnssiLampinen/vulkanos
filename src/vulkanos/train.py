"""Train a U-Net building segmenter on the dataset written by dataset.py.

Uses CUDA with mixed precision when available, CPU otherwise. Writes to
out_dir: best.pt (highest val F1), last.pt, metrics.jsonl, train_config.json.
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .dataset import IGNORE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(arch: str, encoder: str, encoder_weights: str | None):
    import segmentation_models_pytorch as smp

    return smp.create_model(arch, encoder_name=encoder, encoder_weights=encoder_weights, classes=1)


def normalize(rgb: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 to 3xHxW float tensor, ImageNet normalised."""
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


class ChipDataset(Dataset):
    def __init__(self, root: Path, index: pd.DataFrame, split: str, crop: int, augment: bool):
        self.root = root
        self.items = index[index["split"] == split]["chip"].tolist()
        self.split = split
        self.crop = crop
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        name = self.items[i]
        rgb = np.asarray(Image.open(self.root / "images" / name).convert("RGB"))
        mask = np.asarray(Image.open(self.root / f"masks_{self.split}" / name))
        h, w = mask.shape
        c = min(self.crop, h, w)
        if c < h or c < w:
            r0 = random.randint(0, h - c) if self.augment else (h - c) // 2
            c0 = random.randint(0, w - c) if self.augment else (w - c) // 2
            rgb, mask = rgb[r0 : r0 + c, c0 : c0 + c], mask[r0 : r0 + c, c0 : c0 + c]
        if self.augment:
            k = random.randint(0, 3)
            rgb, mask = np.rot90(rgb, k), np.rot90(mask, k)
            if random.random() < 0.5:
                rgb, mask = rgb[:, ::-1], mask[:, ::-1]
            # brightness / contrast jitter
            a = random.uniform(0.8, 1.2)
            b = random.uniform(-20, 20)
            rgb = np.clip(rgb.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        return normalize(rgb), torch.from_numpy(np.ascontiguousarray(mask)).long()


def masked_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + soft Dice over non-ignored pixels."""
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
def evaluate(model, loader, device, amp_dtype, threshold: float = 0.5) -> dict:
    model.eval()
    tp = fp = fn = 0
    losses = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        losses.append(masked_loss(logits, y).item())
        keep = y != IGNORE
        pred = (torch.sigmoid(logits[:, 0].float()) >= threshold) & keep
        truth = (y == 1) & keep
        tp += (pred & truth).sum().item()
        fp += (pred & ~truth).sum().item()
        fn += (~pred & truth).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "val_loss": float(np.mean(losses)) if losses else math.nan,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "iou": tp / max(tp + fp + fn, 1),
    }


def train(
    dataset_dir: Path,
    out_dir: Path,
    arch: str = "Unet",
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    crop: int = 512,
    batch_size: int = 8,
    epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    workers: int = 4,
    seed: int = 0,
    max_batches: int | None = None,
) -> dict:
    """Train and return the best validation metrics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in locals().items()}
    (out_dir / "train_config.json").write_text(json.dumps(config, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"device {device}" + (f" ({torch.cuda.get_device_name()}, {amp_dtype})" if amp_dtype else ""))

    index = pd.read_csv(dataset_dir / "index.csv")
    train_ds = ChipDataset(dataset_dir, index, "train", crop, augment=True)
    val_ds = ChipDataset(dataset_dir, index, "val", crop, augment=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(f"Empty split: {len(train_ds)} train, {len(val_ds)} val chips")
    print(f"chips: {len(train_ds)} train, {len(val_ds)} val")

    loader_args = dict(num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, drop_last=len(train_ds) > batch_size, **loader_args)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, **loader_args)

    model = build_model(arch, encoder, encoder_weights).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = min(len(train_dl), max_batches or len(train_dl))
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup = max(total_steps // 20, 1)

    def lr_factor(step: int) -> float:
        # linear warmup, then cosine decay to 1% of lr
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

    meta = {
        "arch": arch,
        "encoder": encoder,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "crop": crop,
        "dataset": str(dataset_dir),
    }
    best = {"f1": -1.0}
    log = open(out_dir / "metrics.jsonl", "a", encoding="utf-8")
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for i, (x, y) in enumerate(train_dl):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = masked_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        metrics = evaluate(model, val_dl, device, amp_dtype)
        metrics.update(epoch=epoch, train_loss=float(np.mean(losses)), seconds=round(time.time() - t0, 1))
        log.write(json.dumps(metrics) + "\n")
        log.flush()
        print(
            f"epoch {epoch:3d}  train_loss {metrics['train_loss']:.3f}  val_loss {metrics['val_loss']:.3f}  "
            f"P {metrics['precision']:.3f}  R {metrics['recall']:.3f}  F1 {metrics['f1']:.3f}  "
            f"IoU {metrics['iou']:.3f}  ({metrics['seconds']}s)"
        )
        state = {"model": model.state_dict(), "meta": meta, "epoch": epoch, "metrics": metrics}
        torch.save(state, out_dir / "last.pt")
        if metrics["f1"] > best["f1"]:
            best = metrics
            torch.save(state, out_dir / "best.pt")
    log.close()
    print(f"best epoch {best['epoch']}: F1 {best['f1']:.3f}  P {best['precision']:.3f}  R {best['recall']:.3f}")
    return best
