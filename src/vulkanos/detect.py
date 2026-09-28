"""Building detectors behind one interface, plus the chip inference loop.

A detector maps an RGB chip (H x W x 3, uint8) to a building probability map
(H x W, float32 in [0, 1]). Add a new model by writing a class with a
`predict` method and registering a factory in DETECTORS.
"""

from pathlib import Path
from typing import Callable, Protocol

import cv2
import mercantile
import numpy as np
from PIL import Image

from .imagery import load_chip


class Detector(Protocol):
    name: str

    def predict(self, rgb: np.ndarray) -> np.ndarray: ...


class YoloDetector:
    """keremberke/yolov8m-building-segmentation (instance segmentation)."""

    name = "yolo"

    def __init__(self, repo: str = "keremberke/yolov8m-building-segmentation", conf: float = 0.25):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        self.model = YOLO(hf_hub_download(repo, "best.pt"))
        self.conf = conf

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        prob = np.zeros((h, w), np.float32)
        # ultralytics treats numpy input as BGR
        result = self.model.predict(
            np.ascontiguousarray(rgb[:, :, ::-1]),
            conf=self.conf,
            imgsz=max(h, w),
            max_det=2000,
            verbose=False,
        )[0]
        if result.masks is None:
            return prob
        for poly, score in zip(result.masks.xy, result.boxes.conf.cpu().numpy()):
            if len(poly) < 3:
                continue
            layer = np.zeros((h, w), np.uint8)
            cv2.fillPoly(layer, [np.round(poly).astype(np.int32)], 1)
            np.maximum(prob, layer * float(score), out=prob)
        return prob


class GeobaseUnetDetector:
    """geobase/building-footprint-segmentation (ONNX U-Net, 256 px input)."""

    name = "geobase_unet"
    size = 256

    def __init__(self, repo: str = "geobase/building-footprint-segmentation"):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        self.session = ort.InferenceSession(hf_hub_download(repo, "onnx/model.onnx"))
        self.input = self.session.get_inputs()[0].name

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        s = self.size
        if h % s or w % s:
            raise ValueError(f"chip size must be a multiple of {s}")
        patches = (
            rgb.reshape(h // s, s, w // s, s, 3).swapaxes(1, 2).reshape(-1, s, s, 3)
        )
        out = self.session.run(None, {self.input: patches.astype(np.float32) / 255.0})[0]
        return (
            out[..., 0].reshape(h // s, w // s, s, s).swapaxes(1, 2).reshape(h, w)
        ).astype(np.float32)


class UnetDetector:
    """U-Net trained with `python -m vulkanos train` (checkpoint from train.py)."""

    name = "unet"

    def __init__(self, checkpoint: str | Path):
        import torch

        from .train import build_model, normalize

        self.torch = torch
        self.normalize = normalize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = Path(checkpoint)
        state = torch.load(path, map_location="cpu", weights_only=False)
        meta = state["meta"]
        self.model = build_model(meta["arch"], meta["encoder"], encoder_weights=None)
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        # Changes whenever the checkpoint file is replaced, so old predictions are dropped.
        self.version = f"{path.resolve()}:{path.stat().st_mtime_ns}"

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = self.normalize(rgb)[None].to(self.device)
        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            logits = self.model(x)
        return torch.sigmoid(logits[0, 0].float()).cpu().numpy()


DETECTORS: dict[str, Callable[[dict], Detector]] = {
    "yolo": lambda cfg: YoloDetector(conf=cfg.get("conf", 0.25)),
    "geobase_unet": lambda cfg: GeobaseUnetDetector(),
    "unet": lambda cfg: UnetDetector(cfg["checkpoint"]),
}


def detector_tag(cfg: dict) -> str:
    """Name used for prediction folders and output files.

    Trained U-Nets are tagged with their run folder, e.g. unet-r34 for
    models/r34/best.pt, so different training runs do not share caches.
    """
    if cfg["name"] == "unet":
        return f"unet-{Path(cfg['checkpoint']).parent.name}"
    return cfg["name"]


def load_detector(cfg: dict) -> Detector:
    name = cfg["name"]
    if name not in DETECTORS:
        raise ValueError(f"Unknown detector {name!r}; options: {sorted(DETECTORS)}")
    return DETECTORS[name](cfg)


def pred_path(pred_dir: Path, chip: mercantile.Tile) -> Path:
    return pred_dir / f"{chip.z}_{chip.x}_{chip.y}.png"


def save_prob(path: Path, prob: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.png")
    Image.fromarray(np.round(prob * 255).astype(np.uint8)).save(tmp)
    tmp.replace(path)


def load_prob(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.asarray(Image.open(path), dtype=np.float32) / 255.0


def run_detection(
    detector: Detector,
    chips: list[mercantile.Tile],
    data_dir: Path,
    pred_dir: Path,
    provider: str,
    zoom: int,
    log_every: int = 200,
) -> int:
    """Predict every chip that has no saved prediction yet. Returns chips processed."""
    version = getattr(detector, "version", None)
    if version is not None:
        stamp = pred_dir / "detector_version.txt"
        if stamp.exists() and stamp.read_text() != version:
            print(f"  detector changed; clearing old predictions in {pred_dir}")
            for f in pred_dir.glob("*.png"):
                f.unlink()
        pred_dir.mkdir(parents=True, exist_ok=True)
        stamp.write_text(version)
    done = 0
    for i, chip in enumerate(chips):
        out = pred_path(pred_dir, chip)
        if out.exists():
            continue
        rgb, valid = load_chip(data_dir, provider, chip, zoom)
        prob = detector.predict(rgb) if valid.any() else np.zeros(valid.shape, np.float32)
        prob[~valid] = 0
        save_prob(out, prob)
        done += 1
        if done % log_every == 0:
            print(f"  detect {i + 1}/{len(chips)}")
    return done
