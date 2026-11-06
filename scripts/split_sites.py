"""Split sites/smithsonian_10deg.csv (250 volcanoes within 10 degrees of the equator) between the team.

    python scripts/split_sites.py

Writes sites/parts/<person>.csv (name,lat,lon,volcano_number) and sites/parts/hosts.csv. Every site
costs about the same (fixed 10 km grid), so the parts are balanced by count. Sites are sorted by
region and latitude and dealt out back and forth, so everyone gets a mix of regions instead of one
person getting all the dense Java sites. Duplicate names get the volcano number added, otherwise two
sites would share a data folder.
"""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "sites" / "smithsonian_10deg.csv"
OUT = ROOT / "sites" / "parts"
# person -> Aalto classroom machine, all in Liftarit (T8 C106.2, RTX 5070 12 GB, same as agda)
HOSTS = {
    "anssi": "agda",
    "adit": "beeblebrox",
    "miro": "fenchurch",
    "janna": "hactar",
    "khoi": "prefect",
    "hieu": "wonko",
    "duc": "zarniwoop",
}


def main():
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    counts = Counter(r["volcano_name"] for r in rows)
    for r in rows:
        if counts[r["volcano_name"]] > 1:
            r["volcano_name"] = f"{r['volcano_name']} {r['volcano_number']}"
    # GVP volcano numbers start with a region code
    rows.sort(key=lambda r: (r["volcano_number"][:2], float(r["lat"]), float(r["lon"])))
    people = list(HOSTS)
    parts = {p: [] for p in people}
    for i, r in enumerate(rows):
        k, j = divmod(i, len(people))
        parts[people[j if k % 2 == 0 else len(people) - 1 - j]].append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    for p, rs in parts.items():
        with open(OUT / f"{p}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "lat", "lon", "volcano_number"])
            for r in rs:
                w.writerow([r["volcano_name"], r["lat"], r["lon"], r["volcano_number"]])
        print(f"{p:6} {HOSTS[p]:11} {len(rs)} sites")
    with open(OUT / "hosts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["person", "host"])
        w.writerows(HOSTS.items())


if __name__ == "__main__":
    main()
