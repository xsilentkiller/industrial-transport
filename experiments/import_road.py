"""Import the GA team's OpenStreetMap road matrices into ``data/road_ga/``.

Source: ``data/shared_pso_250_road.json`` in
https://github.com/huyuxuan028/transport-algorithms (the shared road
benchmark both teams agreed on).  It holds directed road distances
(``distance_matrix_m``) and driving times (``duration_matrix_s``) of the
fastest path between every pair of our 251 locations, computed with Dijkstra
on a local OSM extract of Budapest.

The importer generates nothing.  It checks that the GA file describes exactly
our dataset (same coordinates, same order weights / volumes / service times),
then writes:

    data/road_ga/params.json             data/base/params.json + "distance_source": "road",
                                         16 h workday (back by 24:00, as the GA team)
    data/road_ga/road_distance_km.csv    directed km (same layout as distances.csv)
    data/road_ga/road_duration_min.csv   directed minutes
    data/road_ga/road_meta.json          checksums, OSM node ids, snapped coordinates,
                                         speed assumptions, limitations
    data/road_ga/{vehicles,locations,orders,furniture_clean}.csv  copied from data/base/

Usage:
    python -m experiments.import_road --source ../transport-algorithms/data/shared_pso_250_road.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "base"
OUT = ROOT / "data" / "road_ga"

# Checksum of the shared road file both teams use (GA repo, Nov 2026).
EXPECTED_SHA256 = "6d794dff4a03cebb1f9127796209fa1f263450a780ce76ade9c26244b6a2676c"
COPY = ["vehicles.csv", "locations.csv", "orders.csv", "furniture_clean.csv"]
DECIMALS = 4


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def matrix_csv(matrix: list[list[float]], scale: float) -> str:
    n = len(matrix)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["from_id"] + list(range(n)))
    for i, row in enumerate(matrix):
        w.writerow([i] + [0.0 if i == j else round(v * scale, DECIMALS) for j, v in enumerate(row)])
    return buf.getvalue()


def check_same_dataset(road: dict, dataset: Path) -> None:
    locs = sorted(_rows(dataset / "locations.csv"), key=lambda r: int(r["location_id"]))
    coords = road["coordinates_lonlat"]
    if len(coords) != len(locs):
        raise ValueError(f"road file has {len(coords)} locations, dataset has {len(locs)}")
    for r, (lon, lat) in zip(locs, coords):
        if abs(float(r["lon"]) - lon) > 1e-6 or abs(float(r["lat"]) - lat) > 1e-6:
            raise ValueError(f"location {r['location_id']}: coordinates differ from the road file")

    orders = {int(o["location_id"]): o for o in _rows(dataset / "orders.csv")}
    for c in road["customers"][1:]:
        o = orders[int(c["id"])]
        checks = [
            ("weight", float(o["total_weight_kg"]), c["weight_kg"], 0.02),
            ("volume", float(o["total_volume_m3"]), c["volume_m3"], 0.002),
            ("loading", float(o["loading_time_min"]), c["loading_minutes"], 0.05),
            ("unloading", float(o["packing_time_min"]), c["unloading_minutes"], 0.05),
        ]
        for what, ours, theirs, tol in checks:
            if abs(ours - theirs) > tol:
                raise ValueError(f"customer {c['id']}: {what} {ours} vs road file {theirs}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, required=True,
                    help="GA repo data/shared_pso_250_road.json")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--allow-other-checksum", action="store_true",
                    help="accept a road file that is not the agreed shared one")
    a = ap.parse_args()

    raw = a.source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256 and not a.allow_other_checksum:
        raise SystemExit(f"road file sha256 {digest} is not the shared benchmark file "
                         f"{EXPECTED_SHA256}; pass --allow-other-checksum to import anyway")
    road = json.loads(raw)
    check_same_dataset(road, DATASET)

    dist = road["distance_matrix_m"]
    dur = road["duration_matrix_s"]
    n = len(dist)
    same_vertex = []
    for name, m in (("distance", dist), ("duration", dur)):
        for i, row in enumerate(m):
            if len(row) != n:
                raise ValueError(f"{name} row {i} has {len(row)} values")
            for j, v in enumerate(row):
                if i != j and (v is None or v < 0):
                    raise ValueError(f"{name} {i}->{j} unreachable")
                # Neighbouring addresses can snap to the same road vertex:
                # a 0 km / 0 min leg is real (they share a kerb), not missing.
                if name == "distance" and i < j and v == 0:
                    same_vertex.append([i, j])

    a.out.mkdir(parents=True, exist_ok=True)
    for f in COPY:
        shutil.copyfile(DATASET / f, a.out / f)
    params = json.loads((DATASET / "params.json").read_text(encoding="utf-8"))
    params["distance_source"] = "road"
    # Shared 16-hour workday (GA team's WORKDAY_16H benchmark, and the
    # professor's 16 h limit): every vehicle starts 08:00 and must be back at
    # the depot by 24:00 (960 min).  No separate last-delivery time; first
    # 8 h at the normal wage, the rest at 2x.  On roads 250 customers do not
    # fit the 08:00-23:00 window with this fleet.
    params["hard_day_end"] = "24:00"
    params.pop("last_delivery_by", None)
    (a.out / "params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    (a.out / "road_distance_km.csv").write_text(matrix_csv(dist, 1 / 1000), encoding="utf-8")
    (a.out / "road_duration_min.csv").write_text(matrix_csv(dur, 1 / 60), encoding="utf-8")

    info = road["road_network"]
    meta = {
        "source": "huyuxuan028/transport-algorithms data/shared_pso_250_road.json",
        "source_sha256": digest,
        "units": {"road_distance_km.csv": "km", "road_duration_min.csv": "min"},
        "directed": True,
        "path_objective": info.get("path_objective"),
        "provider": info.get("provider"),
        "map": info.get("map"),
        "osm_raw_sha256": info.get("raw_sha256"),
        "osm_extract_bbox_wsen": info.get("graph_bbox_wsen"),
        "osm_extract_source": info.get("extract_source"),
        "speed_assumptions_kmh": info.get("speed_assumptions_kmh"),
        "limitations": info.get("limitations"),
        "component_resnaps": info.get("component_resnaps"),
        "max_snap_distance_m": info.get("max_snap_distance_m"),
        "same_road_vertex_pairs": same_vertex,
        "osm_node_ids": road.get("osm_node_ids"),
        "snapped_coordinates_lonlat": info.get("snapped_coordinates_lonlat"),
    }
    (a.out / "road_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"imported {n - 1} customers of road data into {a.out} (sha256 {digest[:12]})")


if __name__ == "__main__":
    main()
