"""Download and cache XYZ imagery tiles, and assemble them into chips."""

import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mercantile
import numpy as np
import requests
from PIL import Image

from .aoi import chip_children

TILE_PX = 256

PROVIDERS = {
    "esri": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}

# Tiles with less pixel variation than this are treated as "no imagery" placeholders.
MIN_TILE_STD = 3.0


def tile_path(data_dir: Path, provider: str, t: mercantile.Tile) -> Path:
    return data_dir / "tiles" / provider / str(t.z) / str(t.x) / f"{t.y}.jpg"


def _missing_marker(path: Path) -> Path:
    return path.with_suffix(".missing")


def _download_one(session: requests.Session, url: str, path: Path, retries: int = 3) -> str:
    if path.exists():
        return "cached"
    if _missing_marker(path).exists():
        return "missing"
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                path.parent.mkdir(parents=True, exist_ok=True)
                _missing_marker(path).touch()
                return "missing"
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(path)
            return "downloaded"
        except requests.RequestException:
            time.sleep(2**attempt)
    return "failed"


def download_tiles(
    tiles: list[mercantile.Tile], data_dir: Path, provider: str = "esri", workers: int = 8
) -> dict[str, int]:
    """Download tiles into the cache. Safe to rerun: cached tiles are skipped."""
    template = PROVIDERS[provider]
    session = requests.Session()
    session.headers["User-Agent"] = "vulkanos-research/0.1"
    counts: dict[str, int] = {}

    def job(t: mercantile.Tile) -> str:
        url = template.format(z=t.z, x=t.x, y=t.y)
        return _download_one(session, url, tile_path(data_dir, provider, t))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status in pool.map(job, tiles):
            counts[status] = counts.get(status, 0) + 1
    return counts


def load_tile(data_dir: Path, provider: str, t: mercantile.Tile) -> np.ndarray | None:
    """RGB uint8 array, or None when the tile is missing or a blank placeholder."""
    path = tile_path(data_dir, provider, t)
    if not path.exists():
        return None
    try:
        img = np.asarray(Image.open(io.BytesIO(path.read_bytes())).convert("RGB"))
    except OSError:
        return None
    if img.shape[:2] != (TILE_PX, TILE_PX) or img.std() < MIN_TILE_STD:
        return None
    return img


def load_chip(
    data_dir: Path, provider: str, chip: mercantile.Tile, zoom: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mosaic a chip's tiles. Returns (rgb HxWx3 uint8, valid HxW bool)."""
    children = chip_children(chip, zoom)
    n = int(round(len(children) ** 0.5))
    size = n * TILE_PX
    rgb = np.zeros((size, size, 3), np.uint8)
    valid = np.zeros((size, size), bool)
    for t in children:
        img = load_tile(data_dir, provider, t)
        if img is None:
            continue
        r = (t.y - children[0].y) * TILE_PX
        c = (t.x - children[0].x) * TILE_PX
        rgb[r : r + TILE_PX, c : c + TILE_PX] = img
        valid[r : r + TILE_PX, c : c + TILE_PX] = True
    return rgb, valid
