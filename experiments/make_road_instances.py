"""Benchmark instances on the real road network, 250 customers.

Road distances exist for the 250 addresses of ``dataset_road/`` (GA team's
OSM matrices).  The benchmark keeps the brief's 250 customers, so every
instance serves all 250 road-mapped addresses and the instances differ in
their orders (what each customer buys: weight, volume, service time).

* customers: all 250 road addresses, in a seeded order (``sample(.., n)``;
  the function also accepts smaller n for quick experiments)
* orders: fresh, with the team's rules (make_instances.make_orders, same seed)
* road_distance_km.csv / road_duration_min.csv: the sub-matrices of those
  addresses -- nothing is re-routed or estimated
* road_meta.json: the OSM vertex of every location, so route geometry works
* same depot, fleet (2 Caddy, 2 Sprinter, 1 MAN TGL), costs and daily window

Sets (disjoint seeds, all 250 customers):

    road_test        100 instances, seeds 5001-5100   (reporting only)
    road_validation   20 instances, seeds 6001-6020   (choosing the method)
    road_tuning        4 instances, seeds 7001-7004   (trying ideas)

Usage:
    python -m experiments.make_road_instances               # road_test
    python -m experiments.make_road_instances --set all
    python -m experiments.make_road_instances --check       # verify vs manifest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
from pathlib import Path

from experiments.make_instances import _csv_text, instance_stats, make_orders, read_csv

ROOT = Path(__file__).resolve().parents[1]
ROAD = ROOT / "dataset_road"
OUT = ROOT / "instances"
N = 250

SETS = {
    "road_test": [(f"road_test_{s}", N, s) for s in range(5001, 5101)],
    "road_validation": [(f"road_val_{s}", N, s) for s in range(6001, 6021)],
    "road_tuning": [(f"road_tune_{s}", N, s) for s in range(7001, 7005)],
}


def _read_matrix(path: Path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return [r[1:] for r in rows[1:]]


def _sub_matrix(m: list[list[str]], ids: list[int]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["from_id"] + list(range(len(ids))))
    for new_i, i in enumerate(ids):
        w.writerow([new_i] + [m[i][j] for j in ids])
    return buf.getvalue()


_CACHE: dict = {}


def _road_source() -> dict:
    if not _CACHE:
        _CACHE["locations"] = sorted(read_csv(ROAD / "locations.csv"), key=lambda r: int(r["location_id"]))
        _CACHE["km"] = _read_matrix(ROAD / "road_distance_km.csv")
        _CACHE["min"] = _read_matrix(ROAD / "road_duration_min.csv")
        _CACHE["meta"] = json.loads((ROAD / "road_meta.json").read_text(encoding="utf-8"))
        _CACHE["furniture"] = read_csv(ROAD / "furniture_clean.csv")
    return _CACHE


def build_instance(name: str, n: int, seed: int, dest: Path) -> dict[str, str]:
    src = _road_source()
    customers = [r for r in src["locations"] if r["type"] == "customer"]
    if n > len(customers):
        raise ValueError(f"{name}: only {len(customers)} road-mapped addresses")
    chosen = random.Random(seed).sample(customers, n)

    fields = list(src["locations"][0].keys()) + ["source_location_id"]
    locations = [{**src["locations"][0], "source_location_id": 0}]
    for new_id, r in enumerate(chosen, start=1):
        locations.append({**r, "location_id": new_id, "source_location_id": r["location_id"]})
    ids = [int(r["source_location_id"]) for r in locations]

    orders = make_orders(src["furniture"], locations, seed)
    params = json.loads((ROAD / "params.json").read_text(encoding="utf-8"))
    params["num_customers"] = n
    params["instance"] = {"name": name, "seed": seed,
                          "generator": "experiments/make_road_instances.py"}
    meta = dict(src["meta"])
    for key in ("osm_node_ids", "snapped_coordinates_lonlat"):
        if meta.get(key):
            meta[key] = [meta[key][i] for i in ids]
    meta["same_road_vertex_pairs"] = None   # only valid for the full dataset
    meta["subset_of"] = "dataset_road (source_location_id in locations.csv)"

    files = {
        "locations.csv": _csv_text(fields, locations),
        "orders.csv": _csv_text(list(orders[0].keys()), orders),
        "road_distance_km.csv": _sub_matrix(src["km"], ids),
        "road_duration_min.csv": _sub_matrix(src["min"], ids),
        "vehicles.csv": (ROAD / "vehicles.csv").read_text(encoding="utf-8"),
        "params.json": json.dumps(params, indent=2, ensure_ascii=False) + "\n",
        "road_meta.json": json.dumps(meta, indent=1) + "\n",
    }
    dest.mkdir(parents=True, exist_ok=True)
    digests = {}
    for fname, text in files.items():
        data = text.encode("utf-8")
        (dest / fname).write_bytes(data)
        digests[fname] = hashlib.sha256(data).hexdigest()
    shutil.copyfile(ROAD / "furniture_clean.csv", dest / "furniture_clean.csv")
    return digests


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", default="road_test", choices=[*SETS, "all"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    names = list(SETS) if a.set == "all" else [a.set]
    manifest_path = a.out / "road_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    road_sha = json.loads((ROAD / "road_meta.json").read_text())["source_sha256"]

    mismatches = 0
    for set_name in names:
        for name, n, seed in SETS[set_name]:
            dest = a.out / set_name / name
            digests = build_instance(name, n, seed, dest)
            entry = {"set": set_name, "customers": n, "seed": seed,
                     "road_source_sha256": road_sha, "sha256": digests,
                     "stats": instance_stats(dest)}
            if a.check:
                old = manifest.get(name)
                if old is None or old["sha256"] != digests:
                    mismatches += 1
                    print(f"MISMATCH {name}")
            else:
                manifest[name] = entry
        print(f"{set_name}: {len(SETS[set_name])} instances in {a.out / set_name}")

    if a.check:
        print("all road instances identical to the manifest" if mismatches == 0
              else f"{mismatches} road instance(s) differ from the manifest")
        raise SystemExit(1 if mismatches else 0)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1) + "\n")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
