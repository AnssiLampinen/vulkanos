"""OpenStreetMap building footprints."""

from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import osmnx as ox
from shapely.geometry.base import BaseGeometry


def fetch_buildings(area: BaseGeometry, cache_dir: Path) -> gpd.GeoDataFrame:
    """All OSM building polygons inside area (lon/lat)."""
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)
    ox.settings.requests_timeout = 300
    try:
        gdf = ox.features_from_polygon(area, tags={"building": True})
    except ox._errors.InsufficientResponseError:
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    gdf = gdf.reset_index()
    cols = [c for c in ("element", "id", "building") if c in gdf.columns]
    gdf = gdf[cols + ["geometry"]].copy()
    gdf["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return gdf.set_crs("EPSG:4326", allow_override=True)
