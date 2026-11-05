"""Map of the site centres with their grid radius on satellite imagery, to check them before a run.

    python scripts/sites_map.py sites/demo.csv --radius 10    # writes data/out/sites_map.html
"""
import argparse
from pathlib import Path

import folium
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="name,lat,lon")
    p.add_argument("--radius", type=float, default=10, help="km")
    p.add_argument("--out", default=str(ROOT / "data" / "out" / "sites_map.html"))
    args = p.parse_args()

    sites = pd.read_csv(args.csv)
    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri", name="Imagery", max_zoom=19).add_to(m)
    # tile.openstreetmap.org blocks pages opened from file:// and carto now wants an api key, esri works
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", attr="Esri", name="Streets", max_zoom=19, show=False).add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/"
                     "MapServer/tile/{z}/{y}/{x}", attr="Esri", name="Place names", overlay=True).add_to(m)
    for g in pd.unique(sites["name"]):
        r = sites[sites["name"] == g].iloc[0]
        fg = folium.FeatureGroup(name=g).add_to(m)
        gmaps = f"https://www.google.com/maps/@{r.lat},{r.lon},14z/data=!3m1!1e3"
        folium.Marker([r.lat, r.lon], tooltip=g,
                      popup=folium.Popup(f"<b>{g}</b><br>{r.lat:.5f}, {r.lon:.5f}<br>"
                                         f"<a href='{gmaps}' target='_blank'>Google Maps</a>", max_width=250)).add_to(fg)
        folium.Circle([r.lat, r.lon], radius=args.radius * 1000, color="yellow", weight=2, fill=False).add_to(fg)
    m.fit_bounds([[sites.lat.min() - 0.1, sites.lon.min() - 0.1], [sites.lat.max() + 0.1, sites.lon.max() + 0.1]])
    folium.LayerControl(collapsed=False).add_to(m)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    m.save(args.out)
    print(f"map in {args.out}")
    for r in sites.itertuples():
        print(f"{r.name:20} https://www.google.com/maps/@{r.lat},{r.lon},14z/data=!3m1!1e3")


if __name__ == "__main__":
    main()
