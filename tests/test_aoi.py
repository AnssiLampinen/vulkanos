import math

import mercantile
import pytest

from vulkanos import aoi
from vulkanos.config import Volcano

V = Volcano(id="test", name="Test", lat=-7.54, lon=110.446)


def test_h3_cells_within_radius():
    cells = aoi.h3_cells(V, radius_km=3, res=8)
    assert len(cells) > 0
    assert cells["dist_km"].max() <= 3
    # ~28 km2 / 0.74 km2 per cell
    assert 25 < len(cells) < 50
    assert cells.crs.to_epsg() == 4326


def test_chips_cover_cells():
    cells = aoi.h3_cells(V, radius_km=1, res=8)
    chips = aoi.chips(cells, zoom=18, chip_tiles=2)
    assert all(c.z == 17 for c in chips)
    for lon, lat in [(g.centroid.x, g.centroid.y) for g in cells.geometry]:
        assert mercantile.tile(lon, lat, 17) in chips


def test_chip_children_row_major():
    chip = mercantile.Tile(10, 20, 17)
    kids = aoi.chip_children(chip, 18)
    assert kids == [
        mercantile.Tile(20, 40, 18),
        mercantile.Tile(21, 40, 18),
        mercantile.Tile(20, 41, 18),
        mercantile.Tile(21, 41, 18),
    ]


def test_chip_tiles_power_of_two():
    with pytest.raises(ValueError):
        aoi.chip_zoom(18, 3)
    assert aoi.chip_zoom(18, 4) == 18 - int(math.log2(4))
