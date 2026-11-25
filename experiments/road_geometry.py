"""Compute road geometry for a solution so the report draws real streets.

    python -m experiments.road_geometry --solution output/best_solution.json \
        --osm cache/budapest_osm.json [--download]

Needs the OpenStreetMap highway extract of Budapest (about 100 MB JSON).
Best: the frozen file the GA team built the matrices from
(transport-algorithms ``cache/budapest_city_osm.json``, sha256 in
dataset_road/road_meta.json) -- then every leg matches the cost matrix
exactly.  ``--download`` fetches a fresh copy once from Overpass with the
same query; OSM edits since then can make a few legs differ, and those are
reported (they never change the cost, which always uses the matrices).

Writes route_geometry.json next to the solution; ``python main.py --mode
report --dataset dataset_road`` then picks it up.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

from src.data_loader import load_problem
from src.road_geometry import sha256_file, trip_geometry

OVERPASS = "https://overpass-api.de/api/interpreter"


def download(query: str, dest: Path) -> None:
    """One bounded request; never retried in a loop."""
    url = OVERPASS + "?" + urllib.parse.urlencode({"data": query})
    req = urllib.request.Request(url, headers={
        "User-Agent": "FurnitureTransportCourse/1.1 (single cached academic road extract)"})
    with urllib.request.urlopen(req, timeout=180) as r:
        body = r.read()
    data = json.loads(body)
    if data.get("remark") or not data.get("elements"):
        raise SystemExit("Overpass returned an incomplete extract: " + str(data.get("remark", "")))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--solution", type=Path, default=Path("output/best_solution.json"))
    ap.add_argument("--dataset", type=Path, default=Path("dataset_road"))
    ap.add_argument("--osm", type=Path, default=Path("cache/budapest_osm.json"))
    ap.add_argument("--download", action="store_true",
                    help="fetch the OSM extract once if --osm does not exist")
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="relative km/min difference still counted as matching")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    data = load_problem(a.dataset)
    if not data.road_mode:
        raise SystemExit(f"{a.dataset} is not a road dataset (distance_source != road)")
    meta = json.loads((a.dataset / "road_meta.json").read_text(encoding="utf-8"))

    if not a.osm.exists():
        if not a.download:
            raise SystemExit(f"{a.osm} not found. Copy the GA team's cache/budapest_city_osm.json "
                             "there, or add --download.")
        print("Downloading the OSM highway extract (one request, ~1-2 min)...")
        download(meta["osm_extract_source"]["query"], a.osm)

    digest = sha256_file(a.osm)
    frozen = digest == meta.get("osm_raw_sha256")
    print("OSM extract:", "frozen GA extract (exact)" if frozen
          else f"different extract {digest[:12]} (legs are checked against the matrix)")

    payload = json.loads(a.solution.read_text(encoding="utf-8"))
    sol = payload["best_solution"]
    trips = [{"vehicle_id": t.get("vehicle_id"), "customers": [int(c) for c in t["customers"]]}
             for t in sol["trips"]]
    # Instance customers are a subset of the 250 road-mapped addresses: map
    # local ids to the OSM vertices through the instance's own road_meta.
    coords = [[float(data.lon[i]), float(data.lat[i])] for i in range(len(data.lon))]

    print("Reading the road graph and routing the legs...")
    osm = json.loads(a.osm.read_text(encoding="utf-8"))
    geo = trip_geometry(trips, osm, meta, coords, data.distance, data.duration,
                        a.tolerance, progress=lambda i, n: print(f"  {i}/{n} sources"))
    geo["osm_sha256"] = digest
    geo["osm_is_frozen_extract"] = frozen
    geo["solution"] = str(a.solution)

    out = a.out or a.solution.parent / "route_geometry.json"
    out.write_text(json.dumps(geo) + "\n", encoding="utf-8")
    s = geo["summary"]
    print(f"Saved {out}: {s['legs']} legs - {s['exact']} exact, {s['close']} close, "
          f"{s['differs']} differ, {s['unroutable']} unroutable")


if __name__ == "__main__":
    main()
