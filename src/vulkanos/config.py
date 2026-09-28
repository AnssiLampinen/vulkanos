from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Volcano:
    id: str
    name: str
    lat: float
    lon: float
    gvp: int | None = None


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data_dir"] = Path(cfg.get("data_dir", "data"))
    return cfg


def volcanoes(cfg: dict, only: list[str] | None = None, key: str = "volcanoes") -> list[Volcano]:
    """Entries of cfg[key]: "volcanoes", or "test_sites" for quick checks."""
    vs = [Volcano(**v) for v in cfg.get(key, [])]
    if only:
        unknown = set(only) - {v.id for v in vs}
        if unknown:
            raise ValueError(f"Unknown volcano id(s): {sorted(unknown)}")
        vs = [v for v in vs if v.id in only]
    return vs


def volcano_dir(cfg: dict, volcano: Volcano) -> Path:
    d = cfg["data_dir"] / volcano.id
    d.mkdir(parents=True, exist_ok=True)
    return d
