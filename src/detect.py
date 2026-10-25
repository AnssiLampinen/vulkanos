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
    def __init__(self, checkpoint):
        import torch
        from train import make_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = make_model(state["meta"]["encoder"], None)
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        # changes when the checkpoint is overwritten, so old predictions get thrown away
        self.version = f"{Path(checkpoint).resolve()}:{Path(checkpoint).stat().st_mtime_ns}"

    def predict(self, rgb):
        import torch
        from train import normalize

        x = normalize(rgb)[None].to(self.device)
        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            logits = self.model(x)
        return torch.sigmoid(logits[0, 0].float()).cpu().numpy()


def load_detector(det):
    if det["name"] == "yolo":
        return Yolo(det["conf"])
    if det["name"] == "geobase_unet":
        return GeobaseUnet()
    if det["name"] == "unet":
        return Unet(det["checkpoint"])
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

    done = skipped = 0
    for i, chip in enumerate(chips):
        out = pred_path(pred_dir, chip)
        if out.exists():
            continue
        # otherwise an empty prediction gets saved and never redone
        if not tiles_done(cfg, chip):
            skipped += 1
            continue
        rgb, valid = load_chip(cfg, chip)
        prob = detector.predict(rgb) if valid.any() else np.zeros(valid.shape, np.float32)
        prob[~valid] = 0
        tmp = out.with_suffix(".part.png")
        Image.fromarray(np.round(prob * 255).astype(np.uint8)).save(tmp)
        tmp.replace(out)
        done += 1
        if done % 200 == 0:
            print(f"  {i + 1}/{len(chips)}")
    return done, skipped
