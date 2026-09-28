"""Area of interest around a volcano: H3 cells and imagery chips covering it."""

import math

import geopandas as gpd
import h3
import mercantile
import numpy as np
from pyproj import Geod
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from .config import Volcano

GEOD = Geod(ellps="WGS84")


def circle(lat: float, lon: float, radius_km: float, n: int = 128) -> Polygon:
    """Geodesic circle as a lon/lat polygon."""
    az = np.linspace(0, 360, n, endpoint=False)
    lons, lats, _ = GEOD.fwd(
        np.full(n, lon), np.full(n, lat), az, np.full(n, radius_km * 1000.0)
    )
    return Polygon(zip(lons, lats))


def cell_polygon(cell: str) -> Polygon:
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def h3_cells(volcano: Volcano, radius_km: float, res: int) -> gpd.GeoDataFrame:
    """H3 cells whose centre lies within radius_km of the volcano."""
    aoi = circle(volcano.lat, volcano.lon, radius_km)
    cells = sorted(h3.geo_to_cells(aoi, res))
    centres = [h3.cell_to_latlng(c) for c in cells]
    _, _, dist = GEOD.inv(
        np.full(len(cells), volcano.lon),
        np.full(len(cells), volcano.lat),
        [lng for _, lng in centres],
        [lat for lat, _ in centres],
    )
    return gpd.GeoDataFrame(
        {
            "cell_id": cells,
            "volcano": volcano.id,
            "dist_km": np.round(np.asarray(dist) / 1000.0, 3),
        },
        geometry=[cell_polygon(c) for c in cells],
        crs="EPSG:4326",
    )


def chip_zoom(zoom: int, chip_tiles: int) -> int:
    k = int(math.log2(chip_tiles))
    if 2**k != chip_tiles:
        raise ValueError("chip_tiles must be a power of two")
    return zoom - k


def chips(cells: gpd.GeoDataFrame, zoom: int, chip_tiles: int) -> list[mercantile.Tile]:
    """Chips (parent tiles of the imagery tiles) that intersect the cells."""
    area = unary_union(cells.geometry.values)
    cz = chip_zoom(zoom, chip_tiles)
    return [
        t
        for t in mercantile.tiles(*area.bounds, zooms=cz)
        if area.intersects(box(*mercantile.bounds(t)))
    ]


def chip_children(chip: mercantile.Tile, zoom: int) -> list[mercantile.Tile]:
    """Imagery tiles of a chip, row-major (top-left first)."""
    tiles = mercantile.children(chip, zoom=zoom)
    return sorted(tiles, key=lambda t: (t.y, t.x))
