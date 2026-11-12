import re
from pathlib import Path

import osmnx as ox
import yaml
from pyproj import Geod

geod = Geod(ellps="WGS84")

SEARCH_M = 3000  # how far from the gvp point to look for the osm summit


def summit(v):
    # hand-set lat/lon, then osm, then gvp
    for src, d in [("manual", v), ("osm", v.get("osm") or {}), ("gvp", v.get("gvp") or {})]:
        if d.get("lat") is not None and d.get("lon") is not None:
            return float(d["lat"]), float(d["lon"]), src
    raise ValueError(f"{v.get('name')} has no coordinates")


def load(path):
    out = {}
    for vid, v in yaml.safe_load(open(path)).items():
        lat, lon, src = summit(v)
        out[vid] = {"id": vid, "name": v.get("name", vid), "lat": lat, "lon": lon, "source": src}
    return out


def to_float(x):
    try:
        return float(str(x).replace("m", "").strip())
    except (TypeError, ValueError):
        return None


def osm_summit(lat, lon, cache_dir):
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)
    try:
        gdf = ox.features_from_point((lat, lon), tags={"natural": ["volcano", "peak"]}, dist=SEARCH_M)
    except ox._errors.InsufficientResponseError:
        return None
    gdf = gdf[gdf.geom_type == "Point"].reset_index()
    gdf = gdf[gdf["element"] == "node"]
    if gdf.empty:
        return None
    gdf["ele_m"] = gdf["ele"].map(to_float) if "ele" in gdf else None
    _, _, gdf["dist_m"] = geod.inv([lon] * len(gdf), [lat] * len(gdf), gdf.geometry.x, gdf.geometry.y)
    volc = gdf[gdf["natural"] == "volcano"]
    # a volcano node is the summit if there is one, otherwise the highest peak, then the nearest
    pick = volc if len(volc) else gdf
    pick = pick.sort_values(["ele_m", "dist_m"], ascending=[False, True], na_position="last").iloc[0]
    return {"node": int(pick["id"]), "lat": round(pick.geometry.y, 6), "lon": round(pick.geometry.x, 6),
            "ele": pick["ele_m"], "name": pick.get("name"), "dist_m": pick["dist_m"]}


def osm_line(o):
    ele = "null" if o["ele"] is None or o["ele"] != o["ele"] else f"{o['ele']:g}"
    return f"{{node: {o['node']}, lat: {o['lat']}, lon: {o['lon']}, ele: {ele}}}"


def refresh_osm(path, cache_dir, ids=None):
    # rewrites only the osm: line of each entry so comments and hand-set values stay
    path = Path(path)
    text = path.read_text()
    for vid, v in yaml.safe_load(text).items():
        if ids and vid not in ids:
            continue
        g = v.get("gvp") or {}
        lat, lon = (g["lat"], g["lon"]) if g.get("lat") is not None else summit(v)[:2]
        o = osm_summit(lat, lon, cache_dir)
        if o is None:
            print(f"[{vid}] no osm summit within {SEARCH_M} m")
            continue
        print(f"[{vid}] osm node {o['node']} {o['name']!r} at {o['lat']}, {o['lon']}, ele {o['ele']}, "
              f"{o['dist_m']:.0f} m from gvp")
        block = re.compile(rf"(^{re.escape(vid)}:\n(?:[ \t].*\n|\n)*?[ \t]+osm:[ \t]*)\{{.*\}}", re.M)
        text, n = block.subn(lambda m: m.group(1) + osm_line(o), text, count=1)
        if not n:
            print(f"[{vid}] no osm: line to update")
    path.write_text(text)
