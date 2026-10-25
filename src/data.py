import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import geopandas as gpd
import h3
import mercantile
import numpy as np
import osmnx as ox
import requests
from PIL import Image
from pyproj import Geod
from shapely.geometry import Polygon, box

TILE_URLS = {
    "esri": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}

geod = Geod(ellps="WGS84")


def h3_cells(site, radius_km, res):
    # all cells with their centre inside a geodesic circle around the site
    n = 128
    lons, lats, _ = geod.fwd(np.full(n, site["lon"]), np.full(n, site["lat"]),
                             np.linspace(0, 360, n, endpoint=False), np.full(n, radius_km * 1000.0))
    cells = sorted(h3.geo_to_cells(Polygon(zip(lons, lats)), res))
    centres = [h3.cell_to_latlng(c) for c in cells]
    _, _, dist = geod.inv(np.full(len(cells), site["lon"]), np.full(len(cells), site["lat"]),
                          [c[1] for c in centres], [c[0] for c in centres])
    polys = [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in cells]
    return gpd.GeoDataFrame({"cell_id": cells, "volcano": site["id"], "dist_km": np.round(np.asarray(dist) / 1000, 3)},
                            geometry=polys, crs="EPSG:4326")


def frame(site, km):
    # west, south, east, north of a km x km square around the site
    lons, lats, _ = geod.fwd(np.full(4, site["lon"]), np.full(4, site["lat"]), [0, 90, 180, 270],
                             np.full(4, km * 1000.0 / 2))
    return lons[3], lats[2], lons[1], lats[0]


def chips(cells, zoom, chip_tiles):
    # a chip is the parent tile of chip_tiles x chip_tiles imagery tiles
    area = cells.union_all()
    z = zoom - int(math.log2(chip_tiles))
    return [t for t in mercantile.tiles(*area.bounds, zooms=z) if area.intersects(box(*mercantile.bounds(t)))]


def children(chip, zoom):
    return sorted(mercantile.children(chip, zoom=zoom), key=lambda t: (t.y, t.x))


def tile_path(cfg, t):
    return cfg["data_dir"] / "tiles" / cfg["imagery"]["provider"] / str(t.z) / str(t.x) / f"{t.y}.jpg"


def download_tile(session, url, path):
    missing = path.with_suffix(".missing")
    if path.exists():
        return "cached"
    if missing.exists():
        return "missing"
    for i in range(3):
        try:
            r = session.get(url, timeout=30)
            path.parent.mkdir(parents=True, exist_ok=True)
            if r.status_code == 404:
                missing.touch()
                return "missing"
            r.raise_for_status()
            path.with_suffix(".part").write_bytes(r.content)
            path.with_suffix(".part").replace(path)
            return "downloaded"
        except requests.RequestException:
            time.sleep(2**i)
    return "failed"


def download_tiles(cfg, tiles):
    url = TILE_URLS[cfg["imagery"]["provider"]]
    session = requests.Session()
    session.headers["User-Agent"] = "vulkanos-research/0.1"

    def get(t):
        return download_tile(session, url.format(z=t.z, x=t.x, y=t.y), tile_path(cfg, t))

    with ThreadPoolExecutor(cfg["imagery"]["workers"]) as pool:
        return Counter(pool.map(get, tiles))


def tiles_done(cfg, chip):
    # every tile is either downloaded or known to be missing on the server
    paths = [tile_path(cfg, t) for t in children(chip, cfg["imagery"]["zoom"])]
    return all(p.exists() or p.with_suffix(".missing").exists() for p in paths)


def load_tile(path):
    if not path.exists():
        return None
    try:
        img = np.asarray(Image.open(path).convert("RGB"))
    except OSError:
        return None
    # esri serves flat grey tiles where it has no imagery
    if img.shape[:2] != (256, 256) or img.std() < 3:
        return None
    return img


def load_chip(cfg, chip):
    tiles = children(chip, cfg["imagery"]["zoom"])
    size = int(len(tiles) ** 0.5) * 256
    rgb = np.zeros((size, size, 3), np.uint8)
    valid = np.zeros((size, size), bool)
    for t in tiles:
        img = load_tile(tile_path(cfg, t))
        if img is None:
            continue
        r, c = (t.y - tiles[0].y) * 256, (t.x - tiles[0].x) * 256
        rgb[r:r + 256, c:c + 256] = img
        valid[r:r + 256, c:c + 256] = True
    return rgb, valid


def fetch_osm(area, cache_dir):
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)
    ox.settings.requests_timeout = 300
    try:
        gdf = ox.features_from_polygon(area, tags={"building": True})
    except ox._errors.InsufficientResponseError:
        # no buildings at all
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index()
    gdf = gdf[[c for c in ["element", "id", "building"] if c in gdf.columns] + ["geometry"]].copy()
    gdf["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return gdf
