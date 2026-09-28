"""Browser tool for choosing the cells used as training labels.

Starts a small web server on localhost. The page shows the satellite imagery,
the OSM building outlines and the H3 cells; click cells to mark them complete
and press Submit to write the cells file.

Over SSH, forward the port first:  ssh -L 8765:localhost:8765 <gpu-machine>
"""

import json
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import geopandas as gpd
import pandas as pd

HEADER = "# Cells where OSM buildings are complete; used as training labels.\n# One H3 cell id per line. Written by: python -m vulkanos label\n"

CELL_FIELDS = ["cell_id", "dist_km", "osm_m2", "pred_m2", "completeness"]


def read_cells_file(path: Path) -> list[str]:
    """Cell ids in the file, ignoring blank lines and # comments."""
    if not path.exists():
        return []
    ids = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    return [c for c in ids if c]


def write_cells_file(path: Path, cells: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(HEADER + "".join(f"{c}\n" for c in cells))
    tmp.replace(path)


@dataclass
class Site:
    id: str
    name: str
    cells: gpd.GeoDataFrame  # cell_id, geometry, optional score columns
    osm: gpd.GeoDataFrame
    overlay: tuple[str, list[list[float]]] | None = None  # detections PNG + bounds


def _page_data(sites: list[Site], cells_file: Path) -> bytes:
    cell_frames = []
    for s in sites:
        df = s.cells[[c for c in CELL_FIELDS if c in s.cells.columns] + ["geometry"]].copy()
        df["site"] = s.id
        for c in ("osm_m2", "pred_m2", "completeness"):
            if c in df.columns:
                df[c] = df[c].round(2)
        cell_frames.append(df)
    cells = gpd.GeoDataFrame(
        pd.concat(cell_frames, ignore_index=True), crs="EPSG:4326"
    )
    data = {
        "file": str(cells_file),
        "sites": [
            {
                "id": s.id,
                "name": s.name,
                "overlay": {"url": s.overlay[0], "bounds": s.overlay[1]} if s.overlay else None,
            }
            for s in sites
        ],
        "cells": json.loads(cells.to_json(drop_id=True)),
        "osm": {s.id: json.loads(s.osm[["geometry"]].to_json(drop_id=True)) for s in sites},
    }
    return json.dumps(data).encode()


def serve(sites: list[Site], cells_file: Path, port: int = 8765, open_browser: bool = True) -> None:
    page = (Path(__file__).parent / "labeler.html").read_bytes()
    loaded = {c for s in sites for c in s.cells["cell_id"]}
    data = _page_data(sites, cells_file)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(page, "text/html; charset=utf-8")
            elif self.path == "/api/data":
                self._send(data, "application/json")
            elif self.path == "/api/selected":
                # Read on every request so a reload shows the file as it is now.
                self._send(json.dumps(read_cells_file(cells_file)).encode(), "application/json")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/api/save":
                return self._send(b"not found", "text/plain", 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                chosen = [str(c) for c in body["cells"] if str(c) in loaded]
            except (ValueError, KeyError, TypeError):
                return self._send(b'{"error": "bad request"}', "application/json", 400)
            # Keep cells from sites that are not loaded in this session.
            others = [c for c in read_cells_file(cells_file) if c not in loaded]
            cells = others + sorted(set(chosen))
            write_cells_file(cells_file, cells)
            print(f"saved {len(chosen)} cells ({len(cells)} in file) to {cells_file}")
            reply = {"saved": len(chosen), "total": len(cells), "file": str(cells_file)}
            self._send(json.dumps(reply).encode(), "application/json")

        def log_message(self, *args):
            pass

    # localhost only: the page can write files, so it must not be reachable from the network.
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}/"
    print(f"labeler running at {url}  (Ctrl+C to stop)")
    print(f"over SSH: ssh -L {port}:localhost:{port} <host>, then open {url} locally")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
