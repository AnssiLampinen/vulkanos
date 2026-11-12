from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from data import load_chip, tiles_done


class Yolo:
    def __init__(self, conf):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        self.model = YOLO(hf_hub_download("keremberke/yolov8m-building-segmentation", "best.pt"))
        self.conf = conf

    def predict(self, rgb):
        import cv2

        h, w = rgb.shape[:2]
        prob = np.zeros((h, w), np.float32)
        # ultralytics wants BGR
        res = self.model.predict(np.ascontiguousarray(rgb[:, :, ::-1]), conf=self.conf, imgsz=max(h, w),
                                 max_det=2000, verbose=False)[0]
        if res.masks is None:
            return prob
        for poly, score in zip(res.masks.xy, res.boxes.conf.cpu().numpy()):
            if len(poly) < 3:
                continue
            layer = np.zeros((h, w), np.uint8)
            cv2.fillPoly(layer, [np.round(poly).astype(np.int32)], 1)
            np.maximum(prob, layer * float(score), out=prob)
        return prob


class GeobaseUnet:
    # onnx unet that takes 256px patches
    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        self.sess = ort.InferenceSession(hf_hub_download("geobase/building-footprint-segmentation", "onnx/model.onnx"))

    def predict(self, rgb):
        h, w = rgb.shape[:2]
        s = 256
        patches = rgb.reshape(h // s, s, w // s, s, 3).swapaxes(1, 2).reshape(-1, s, s, 3)
        out = self.sess.run(None, {self.sess.get_inputs()[0].name: patches.astype(np.float32) / 255.0})[0]
        return out[..., 0].reshape(h // s, w // s, s, s).swapaxes(1, 2).reshape(h, w).astype(np.float32)


class Unet:
    # our own model from train.py
    def __init__(self, checkpoint, device=None, batch_size=16):
        import torch
        from train import amp_dtype, describe, make_model, pick_device

        self.device = pick_device(device)
        self.dtype = amp_dtype(self.device)
        self.batch_size = batch_size if self.device.type != "cpu" else 1
        print("  unet on", describe(self.device))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = make_model(state["meta"]["encoder"], None)
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        # changes when the checkpoint is overwritten, so old predictions get thrown away
        self.version = f"{Path(checkpoint).resolve()}:{Path(checkpoint).stat().st_mtime_ns}"

    def predict(self, rgb):
        return self.predict_batch([rgb])[0]

    def predict_batch(self, rgbs):
        import torch
        from train import normalize

        x = torch.stack([normalize(r) for r in rgbs]).to(self.device, non_blocking=True)
        with torch.no_grad(), torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype is not None):
            logits = self.model(x)
        return list(torch.sigmoid(logits[:, 0].float()).cpu().numpy())


def load_detector(det, device=None):
    if det["name"] == "yolo":
        return Yolo(det["conf"])
    if det["name"] == "geobase_unet":
        return GeobaseUnet()
    if det["name"] == "unet":
        return Unet(det["checkpoint"], device, det.get("batch_size", 16))
    raise ValueError(f"unknown detector {det['name']}")


def detector_tag(det):
    # unets are named after their run folder so different runs don't mix predictions
    if det["name"] == "unet":
        return "unet-" + Path(det["checkpoint"]).parent.name
    return det["name"]


def pred_path(pred_dir, chip):
    return pred_dir / f"{chip.z}_{chip.x}_{chip.y}.png"


def load_prob(path):
    if not path.exists():
        return None
    return np.asarray(Image.open(path), dtype=np.float32) / 255.0


def run_detection(cfg, detector, chips, pred_dir):
    pred_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(detector, "version"):
        stamp = pred_dir / "detector_version.txt"
        if stamp.exists() and stamp.read_text() != detector.version:
            print(f"  checkpoint changed, removing old predictions in {pred_dir}")
            for f in pred_dir.glob("*.png"):
                f.unlink()
        stamp.write_text(detector.version)

    todo = [c for c in chips if not pred_path(pred_dir, c).exists()]
    # otherwise an empty prediction gets saved and never redone
    ready = [c for c in todo if tiles_done(cfg, c)]
    skipped = len(todo) - len(ready)
    bs = getattr(detector, "batch_size", 1)
    done = 0

    def save(chip, prob, valid):
        prob[~valid] = 0
        out = pred_path(pred_dir, chip)
        tmp = out.with_suffix(".part.png")
        Image.fromarray(np.round(prob * 255).astype(np.uint8)).save(tmp)
        tmp.replace(out)

    # chips are read from disk in threads, a few batches at a time so memory stays small
    step = bs * 4
    with ThreadPoolExecutor(cfg["imagery"].get("workers", 8)) as pool:
        for k in range(0, len(ready), step):
            loaded = list(pool.map(lambda c: (c, *load_chip(cfg, c)), ready[k:k + step]))
            for j in range(0, len(loaded), bs):
                batch = loaded[j:j + bs]
                use = [b for b in batch if b[2].any()]
                probs = []
                if use:
                    probs = (detector.predict_batch([b[1] for b in use]) if hasattr(detector, "predict_batch")
                             else [detector.predict(b[1]) for b in use])
                for chip, _, valid in batch:
                    save(chip, probs.pop(0) if valid.any() else np.zeros(valid.shape, np.float32), valid)
            prev = done
            done += len(loaded)
            if done // 500 > prev // 500:
                print(f"  {done}/{len(ready)}")
    return done, skipped
