import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import geopandas as gpd
import pandas as pd


def read_cells(path):
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def page_data(sites, cells_file):
    frames = []
    for s in sites:
        df = s["cells"][[c for c in ["cell_id", "dist_km", "osm_m2", "pred_m2", "completeness", "geometry"]
                         if c in s["cells"].columns]].copy()
        df["site"] = s["id"]
        frames.append(df.round(2))
    cells = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    data = {
        "file": str(cells_file),
        "sites": [{"id": s["id"], "name": s["name"], "overlay": s["overlay"]} for s in sites],
        "cells": json.loads(cells.to_json(drop_id=True)),
        "osm": {s["id"]: json.loads(s["osm"][["geometry"]].to_json(drop_id=True)) for s in sites},
    }
    return json.dumps(data).encode()


def serve(sites, cells_file, port, open_browser):
    page = (Path(__file__).parent / "label.html").read_bytes()
    data = page_data(sites, cells_file)
    loaded = {c for s in sites for c in s["cells"]["cell_id"]}

    class Handler(BaseHTTPRequestHandler):
        def send(self, body, ctype, status=200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(page, "text/html")
            elif self.path == "/api/data":
                self.send(data, "application/json")
            elif self.path == "/api/selected":
                self.send(json.dumps(read_cells(cells_file)).encode(), "application/json")
            else:
                self.send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/api/save":
                return self.send(b"not found", "text/plain", 404)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            chosen = sorted({c for c in body["cells"] if c in loaded})
            # keep the cells of sites that aren't open right now
            cells = [c for c in read_cells(cells_file) if c not in loaded] + chosen
            cells_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cells_file.with_suffix(".tmp")
            tmp.write_text("".join(c + "\n" for c in cells))
            tmp.replace(cells_file)
            print(f"saved {len(chosen)} cells ({len(cells)} in file) to {cells_file}")
            self.send(json.dumps({"saved": len(chosen), "total": len(cells)}).encode(), "application/json")

        def log_message(self, *args):
            pass

    # localhost only, the page can write files
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}/"
    print(f"labeler at {url}, ctrl+c to stop")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
