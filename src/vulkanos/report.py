"""Interactive HTML map of cell scores and detection overlay."""

import base64
import io
import math
from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import mercantile
import numpy as np
from branca.element import Element
from PIL import Image

from .compare import chip_masks
from .config import Volcano

# Overlay colours (RGBA)
MATCHED = (26, 150, 65, 170)  # detected, in OSM
UNMAPPED = (215, 25, 28, 200)  # detected, not in OSM
UNDETECTED = (43, 131, 186, 150)  # in OSM, not detected

MAX_OVERLAY_PX = 8192

LEGEND_HTML = """
<div style="position: fixed; bottom: 30px; left: 10px; z-index: 9999; background: white;
            padding: 8px 10px; border-radius: 4px; font: 12px sans-serif; box-shadow: 0 1px 4px #0004;">
  <b>Detections</b><br>
  <span style="color: rgb(26,150,65)">&#9632;</span> detected, in OSM<br>
  <span style="color: rgb(215,25,28)">&#9632;</span> detected, not in OSM<br>
  <span style="color: rgb(43,131,186)">&#9632;</span> in OSM, not detected
</div>
"""


def overlay_image(
    chips: list[mercantile.Tile], osm: gpd.GeoDataFrame, mask_args: dict
) -> tuple[str, list[list[float]]] | None:
    """RGBA PNG (data URL) of detections vs OSM over all chips, plus its
    [[south, west], [north, east]] bounds. Downsampled to MAX_OVERLAY_PX per side."""
    if not chips:
        return None
    osm_3857 = osm.to_crs(3857)
    xs = [c.x for c in chips]
    ys = [c.y for c in chips]
    x0, y0 = min(xs), min(ys)
    nx, ny = max(xs) - x0 + 1, max(ys) - y0 + 1

    rgba = None
    for chip in chips:
        result = chip_masks(chip, osm_3857, **mask_args)
        if result is None:
            continue
        masks = result[0]
        size = masks["pred"].shape[0]
        if rgba is None:
            # Power-of-two step divides the chip size exactly, so every slice is cs x cs.
            ratio = max(nx, ny) * size / MAX_OVERLAY_PX
            step = 2 ** max(0, math.ceil(math.log2(ratio))) if ratio > 1 else 1
            cs = size // step
            rgba = np.zeros((ny * cs, nx * cs, 4), np.uint8)
        layer = np.zeros((cs, cs, 4), np.uint8)
        layer[(masks["osm"] & ~masks["osm_detected"])[::step, ::step]] = UNDETECTED
        layer[(masks["pred"] & ~masks["missed"])[::step, ::step]] = MATCHED
        layer[masks["missed"][::step, ::step]] = UNMAPPED
        r, c = (chip.y - y0) * cs, (chip.x - x0) * cs
        rgba[r : r + cs, c : c + cs] = layer
    if rgba is None:
        return None

    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG", optimize=True)
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    nw = mercantile.bounds(mercantile.Tile(x0, y0, chips[0].z))
    se = mercantile.bounds(mercantile.Tile(x0 + nx - 1, y0 + ny - 1, chips[0].z))
    return url, [[se.south, nw.west], [nw.north, se.east]]

TOOLTIP_FIELDS = [
    "cell_id",
    "dist_km",
    "completeness",
    "gap_score",
    "pred_m2",
    "osm_m2",
    "pred_count",
    "unmatched_count",
    "recall_vs_osm",
    "imagery_coverage",
]


def cell_map(
    scored: gpd.GeoDataFrame,
    volcano: Volcano,
    out: Path,
    overlay: tuple[str, list[list[float]]] | None = None,
) -> Path:
    gdf = scored.copy()
    for c in ["completeness", "gap_score", "pred_m2", "osm_m2", "recall_vs_osm", "imagery_coverage"]:
        gdf[c] = gdf[c].round(2)

    colormap = cm.LinearColormap(["#d7191c", "#fdae61", "#1a9641"], vmin=0, vmax=1)
    colormap.caption = "OSM completeness (share of detected building area present in OSM)"

    def style(feature):
        p = feature["properties"]
        if p["no_imagery"] or p["low_signal"] or p["completeness"] is None:
            return {"fillColor": "#999999", "fillOpacity": 0.15, "weight": 0.2, "color": "#666"}
        return {
            "fillColor": colormap(p["completeness"]),
            "fillOpacity": 0.6,
            "weight": 0.2,
            "color": "#333",
        }

    m = folium.Map(location=[volcano.lat, volcano.lon], zoom_start=12, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Imagery",
        max_zoom=20,
        show=True,
    ).add_to(m)
    # tile.openstreetmap.org blocks pages opened from file:// (no Referer), so
    # OSM data is shown through Carto's CDN instead.
    folium.TileLayer("CartoDB Voyager", name="OSM (Carto Voyager)", max_zoom=20, show=False).add_to(m)
    folium.GeoJson(
        gdf[TOOLTIP_FIELDS + ["low_signal", "no_imagery", "geometry"]],
        name="Completeness",
        style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=TOOLTIP_FIELDS),
    ).add_to(m)
    if overlay is not None:
        url, bounds = overlay
        folium.raster_layers.ImageOverlay(
            url, bounds=bounds, name="Detections vs OSM", interactive=False, zindex=500
        ).add_to(m)
        m.get_root().html.add_child(Element(LEGEND_HTML))
    folium.Marker([volcano.lat, volcano.lon], tooltip=volcano.name).add_to(m)
    colormap.add_to(m)
    folium.LayerControl().add_to(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return out
