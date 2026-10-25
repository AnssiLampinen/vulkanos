import base64
import io
import math

import branca.colormap as cm
import folium
import mercantile
import numpy as np
from branca.element import Element
from PIL import Image

from compare import chip_masks, house_edges, split_houses
from data import load_chip

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

LEGEND = """
<div style="position:fixed; bottom:30px; left:10px; z-index:9999; background:white; padding:6px 8px; font:12px sans-serif">
<span style="color:#1a9641">&#9632;</span> detected, in OSM<br>
<span style="color:#d7191c">&#9632;</span> detected, not in OSM<br>
<span style="color:#2b83ba">&#9632;</span> in OSM, not detected<br>
white lines: cuts between houses
</div>
"""

FIELDS = ["cell_id", "dist_km", "completeness", "missed_m2", "pred_m2", "osm_m2", "pred_count", "unmatched_count",
          "house_completeness", "house_count", "house_unmatched_count", "osm_house_count", "recall_vs_osm",
          "imagery_coverage"]


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
        m, px, _ = res
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
        labels, _ = split_houses(m["pred"], math.ceil(cfg["compare"]["house_min_sep_m"] / px))
        # any cut pixel marks the whole step x step block so thin cuts survive downscaling
        part[house_edges(labels).reshape(cs, step, cs, step).any(axis=(1, 3))] = (255, 255, 255, 255)
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


def grid(chips, size, max_px):
    x0, y0 = min(c.x for c in chips), min(c.y for c in chips)
    nx, ny = max(c.x for c in chips) - x0 + 1, max(c.y for c in chips) - y0 + 1
    step = 1
    while max(nx, ny) * size / step > max_px:
        step *= 2
    return x0, y0, nx, ny, step


def to_layer(img, chips, fmt, **kw):
    x0, y0 = min(c.x for c in chips), min(c.y for c in chips)
    x1, y1 = max(c.x for c in chips), max(c.y for c in chips)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format=fmt, **kw)
    url = f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()
    nw = mercantile.bounds(mercantile.Tile(x0, y0, chips[0].z))
    se = mercantile.bounds(mercantile.Tile(x1, y1, chips[0].z))
    return url, [[se.south, nw.west], [nw.north, se.east]]


def imagery(cfg, chips, max_px=8192):
    # the downloaded tiles as one image, so there is a background even when esri doesn't load
    size = 256 * cfg["imagery"]["chip_tiles"]
    x0, y0, nx, ny, step = grid(chips, size, max_px)
    cs = size // step
    img = np.zeros((ny * cs, nx * cs, 4), np.uint8)
    for chip in chips:
        rgb, valid = load_chip(cfg, chip)
        r, c = (chip.y - y0) * cs, (chip.x - x0) * cs
        img[r:r + cs, c:c + cs, :3] = rgb[::step, ::step]
        img[r:r + cs, c:c + cs, 3] = valid[::step, ::step] * 255
    return to_layer(img, chips, "WEBP", quality=80)


def outlines(cfg, chips, osm, pred_dir, max_px=8192):
    # outer edge of the detected buildings, without the cuts between houses
    osm = osm.to_crs(3857)
    img = None
    for chip in chips:
        res = chip_masks(cfg, chip, osm, pred_dir)
        if res is None:
            continue
        pred = res[0]["pred"]
        if img is None:
            size = pred.shape[0]
            x0, y0, nx, ny, step = grid(chips, size, max_px)
            cs = size // step
            img = np.zeros((ny * cs, nx * cs, 4), np.uint8)
        p = np.pad(pred, 1)
        edge = pred & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
        r, c = (chip.y - y0) * cs, (chip.x - x0) * cs
        img[r:r + cs, c:c + cs][edge.reshape(cs, step, cs, step).any(axis=(1, 3))] = (255, 255, 0, 255)
    if img is None:
        return None
    return to_layer(img, chips, "PNG", optimize=True)


def cell_map(scored, site, out, overlay=None, imagery=None, outlines=None):
    gdf = scored.copy()
    for c in ["completeness", "house_completeness", "missed_m2", "pred_m2", "osm_m2", "recall_vs_osm",
              "imagery_coverage"]:
        gdf[c] = gdf[c].round(2)

    colors = cm.LinearColormap(["#d7191c", "#fdae61", "#1a9641"], vmin=0, vmax=1)
    colors.caption = "OSM completeness (share of detected building area that is in OSM)"

    def style_for(field):
        def style(f):
            p = f["properties"]
            if p["no_imagery"] or p["low_signal"] or p[field] is None:
                return {"fillColor": "#999999", "fillOpacity": 0.15, "weight": 0.2, "color": "#666"}
            return {"fillColor": colors(p[field]), "fillOpacity": 0.6, "weight": 0.2, "color": "#333"}
        return style

    m = folium.Map(location=[site["lat"], site["lon"]], zoom_start=12, tiles=None)
    folium.TileLayer(ESRI, attr="Esri World Imagery", name="Imagery", max_zoom=20).add_to(m)
    # tile.openstreetmap.org blocks pages opened from file://, carto doesn't
    folium.TileLayer("CartoDB Voyager", name="OSM", max_zoom=20, show=False).add_to(m)
    if imagery:
        # tile pane so it stays under the cells and detections
        folium.raster_layers.ImageOverlay(imagery[0], bounds=imagery[1], name="Imagery (downloaded tiles)",
                                          interactive=False, pane="tilePane").add_to(m)
    # same colours for both so they can be compared by toggling
    for field, name, show in [("completeness", "Completeness (area)", True),
                              ("house_completeness", "Completeness (houses)", False)]:
        folium.GeoJson(gdf[FIELDS + ["low_signal", "no_imagery", "geometry"]], name=name, style_function=style_for(field),
                       tooltip=folium.GeoJsonTooltip(fields=FIELDS), show=show).add_to(m)
    if overlay:
        folium.raster_layers.ImageOverlay(overlay[0], bounds=overlay[1], name="Detections vs OSM", interactive=False,
                                          zindex=500).add_to(m)
        m.get_root().html.add_child(Element(LEGEND))
    if outlines:
        folium.raster_layers.ImageOverlay(outlines[0], bounds=outlines[1], name="Predicted outlines", interactive=False,
                                          show=False).add_to(m)
    folium.Marker([site["lat"], site["lon"]], tooltip=site["name"]).add_to(m)
    colors.add_to(m)
    folium.LayerControl().add_to(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
