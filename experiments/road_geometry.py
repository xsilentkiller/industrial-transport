"""Compute road geometry for a solution so the report draws real streets.

    python -m experiments.road_geometry --solution output/best_solution.json \
        --osm cache/budapest_osm.json [--download]

Needs the OpenStreetMap highway extract of Budapest (about 100 MB JSON).
Best: the frozen file the GA team built the matrices from
(transport-algorithms ``cache/budapest_city_osm.json``, sha256 in
data/road_ga/road_meta.json) -- then every leg matches the cost matrix
exactly.  ``--download`` fetches a fresh copy once from Overpass with the
same query; OSM edits since then can make a few legs differ, and those are
reported (they never change the cost, which always uses the matrices).

Writes route_geometry.json next to the solution; ``python main.py --mode
report --dataset data/road_ga`` then picks it up.
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


def solution_geometry(data, dataset: Path, payload: dict, osm_path: Path,
                      tolerance: float = 0.02, progress=None) -> dict:
    """Street geometry of a saved solution (used by main.py and the CLI below)."""
    meta = json.loads((Path(dataset) / "road_meta.json").read_text(encoding="utf-8"))
    digest = sha256_file(osm_path)
    frozen = digest == meta.get("osm_raw_sha256")
    print("OSM extract:", "the one the matrices were built from (exact)" if frozen
          else f"different extract {digest[:12]} (legs are checked against the matrix)")
    sol = payload["best_solution"]
    trips = [{"vehicle_id": t.get("vehicle_id"), "customers": [int(c) for c in t["customers"]]}
             for t in sol["trips"]]
    # Instance customers are a subset of the road-mapped addresses: map
    # local ids to the OSM vertices through the instance's own road_meta.
    coords = [[float(data.lon[i]), float(data.lat[i])] for i in range(len(data.lon))]
    print("Reading the road graph and routing the legs...")
    osm = json.loads(Path(osm_path).read_text(encoding="utf-8"))
    geo = trip_geometry(trips, osm, meta, coords, data.distance, data.duration,
                        tolerance, progress=progress)
    geo["osm_sha256"] = digest
    geo["osm_is_frozen_extract"] = frozen
    return geo


def save_geometry(geo: dict, out: Path) -> None:
    out.write_text(json.dumps(geo) + "\n", encoding="utf-8")
    s = geo["summary"]
    print(f"Saved {out}: {s['legs']} legs - {s['exact']} exact, {s['close']} close, "
          f"{s['differs']} differ, {s['unroutable']} unroutable")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--solution", type=Path, default=Path("output/best_solution.json"))
    ap.add_argument("--dataset", type=Path, default=Path("data/osm"))
    ap.add_argument("--osm", type=Path, default=Path("cache/budapest_osm.json"))
    ap.add_argument("--download", action="store_true",
                    help="fetch the OSM extract once if --osm does not exist")
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="relative km/min difference still counted as matching")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    data = load_problem(a.dataset)
    if not data.road_mode:
        raise SystemExit(f"{a.dataset} is not a road dataset (distance_source != road)")
    if not a.osm.exists():
        if not a.download:
            raise SystemExit(f"{a.osm} not found: build it with "
                             "python -m experiments.osm_from_geofabrik, or add --download.")
        meta = json.loads((a.dataset / "road_meta.json").read_text(encoding="utf-8"))
        print("Downloading the OSM highway extract (one request, ~1-2 min)...")
        download(meta["osm_extract_source"]["query"], a.osm)

    payload = json.loads(a.solution.read_text(encoding="utf-8"))
    geo = solution_geometry(data, a.dataset, payload, a.osm, a.tolerance,
                            progress=lambda i, n: print(f"  {i}/{n} sources"))
    geo["solution"] = str(a.solution)
    save_geometry(geo, a.out or a.solution.parent / "route_geometry.json")


if __name__ == "__main__":
    main()
