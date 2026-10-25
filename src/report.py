import base64
import io

import branca.colormap as cm
import folium
import mercantile
import numpy as np
from branca.element import Element
from PIL import Image

from compare import chip_masks

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

LEGEND = """
<div style="position:fixed; bottom:30px; left:10px; z-index:9999; background:white; padding:6px 8px; font:12px sans-serif">
<span style="color:#1a9641">&#9632;</span> detected, in OSM<br>
<span style="color:#d7191c">&#9632;</span> detected, not in OSM<br>
<span style="color:#2b83ba">&#9632;</span> in OSM, not detected
</div>
"""

FIELDS = ["cell_id", "dist_km", "completeness", "missed_m2", "pred_m2", "osm_m2", "pred_count", "unmatched_count",
          "recall_vs_osm", "imagery_coverage"]


def overlay(cfg, chips, osm, pred_dir, max_px=8192):
    # detections vs osm for the whole area as one png, scaled down to max_px
    osm = osm.to_crs(3857)
    x0, y0 = min(c.x for c in chips), min(c.y for c in chips)
    nx, ny = max(c.x for c in chips) - x0 + 1, max(c.y for c in chips) - y0 + 1

    img = None
    for chip in chips:
        res = chip_masks(cfg, chip, osm, pred_dir)
        if res is None:
            continue
        m = res[0]
        if img is None:
            size = m["pred"].shape[0]
            step = 1
            while max(nx, ny) * size / step > max_px:
                step *= 2
            cs = size // step
            img = np.zeros((ny * cs, nx * cs, 4), np.uint8)
        part = np.zeros((cs, cs, 4), np.uint8)
        part[(m["osm"] & ~m["osm_detected"])[::step, ::step]] = (43, 131, 186, 150)
        part[(m["pred"] & ~m["missed"])[::step, ::step]] = (26, 150, 65, 170)
        part[m["missed"][::step, ::step]] = (215, 25, 28, 200)
        r, c = (chip.y - y0) * cs, (chip.x - x0) * cs
        img[r:r + cs, c:c + cs] = part
    if img is None:
        return None

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    nw = mercantile.bounds(mercantile.Tile(x0, y0, chips[0].z))
    se = mercantile.bounds(mercantile.Tile(x0 + nx - 1, y0 + ny - 1, chips[0].z))
    return url, [[se.south, nw.west], [nw.north, se.east]]


def cell_map(scored, site, out, overlay=None):
    gdf = scored.copy()
    for c in ["completeness", "missed_m2", "pred_m2", "osm_m2", "recall_vs_osm", "imagery_coverage"]:
        gdf[c] = gdf[c].round(2)

    colors = cm.LinearColormap(["#d7191c", "#fdae61", "#1a9641"], vmin=0, vmax=1)
    colors.caption = "OSM completeness (share of detected building area that is in OSM)"

    def style(f):
        p = f["properties"]
        if p["no_imagery"] or p["low_signal"] or p["completeness"] is None:
            return {"fillColor": "#999999", "fillOpacity": 0.15, "weight": 0.2, "color": "#666"}
        return {"fillColor": colors(p["completeness"]), "fillOpacity": 0.6, "weight": 0.2, "color": "#333"}

    m = folium.Map(location=[site["lat"], site["lon"]], zoom_start=12, tiles=None)
    folium.TileLayer(ESRI, attr="Esri World Imagery", name="Imagery", max_zoom=20).add_to(m)
    # tile.openstreetmap.org blocks pages opened from file://, carto doesn't
    folium.TileLayer("CartoDB Voyager", name="OSM", max_zoom=20, show=False).add_to(m)
    folium.GeoJson(gdf[FIELDS + ["low_signal", "no_imagery", "geometry"]], name="Completeness", style_function=style,
                   tooltip=folium.GeoJsonTooltip(fields=FIELDS)).add_to(m)
    if overlay:
        folium.raster_layers.ImageOverlay(overlay[0], bounds=overlay[1], name="Detections vs OSM", interactive=False,
                                          zindex=500).add_to(m)
        m.get_root().html.add_child(Element(LEGEND))
    folium.Marker([site["lat"], site["lon"]], tooltip=site["name"]).add_to(m)
    colors.add_to(m)
    folium.LayerControl().add_to(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
