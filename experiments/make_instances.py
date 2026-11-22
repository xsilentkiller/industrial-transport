"""Generate benchmark instances with the team's dataset rules.

Every instance is a complete dataset folder (params.json, vehicles.csv,
furniture_clean.csv, locations.csv, orders.csv, distances.csv) that
``load_problem`` reads exactly like ``dataset/``.  Only the random seed
changes; the rules are those of ``dataset/generators/``:

* customers: ``random.Random(seed).sample(places_pool, n)`` (locations.py)
* orders: 1-4 pieces with weights 40/30/20/10, drawn from furniture_clean.csv;
  loading 3 min + 0.05 min/kg, unloading 5 min + 0.05 min/kg (orders.py)
* distances: haversine x 1.3, rounded to metres (Distances.py)
* same depot, fleet, cost parameters and daily window as dataset/params.json

Seed 42 with 250 customers reproduces dataset/ byte for byte (checked by
tests/test_instances.py), which proves the rules are the same.

Sets (disjoint seeds; seed 42 is never used, so the default instance stays
separate from every benchmark set):

    test        100 x 250 customers, seeds 3001-3100   (reporting only)
    validation   20 x 250 customers, seeds 2001-2020   (choosing the method)
    tuning        4 x 250 customers, seeds 1001-1004   (trying ideas)
    scaling       4 per size 50/100/250/500/1000, seeds 4001-4004;
                  fleet repeated ceil(n / 250) times so larger instances stay
                  comparable (5 vehicles per 250 customers)

Usage:
    python -m experiments.make_instances                 # the 100 test instances
    python -m experiments.make_instances --set all
    python -m experiments.make_instances --check         # verify against the manifest

The instance folders (about 60 MB for the test set) are not committed: they
are rebuilt from the seeds.  ``instances/manifest.json`` (committed) records
the SHA-256 of every generated file, so anyone can check they have the exact
same instances.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
OUT = ROOT / "instances"

# ---- the team's generator rules (dataset/generators/) ----------------------
ITEMS_PER_ORDER = [1, 2, 3, 4]
ITEMS_WEIGHTS = [40, 30, 20, 10]
LOAD_BASE_MIN, LOAD_PER_KG_MIN = 3.0, 0.05
UNLOAD_BASE_MIN, UNLOAD_PER_KG_MIN = 5.0, 0.05
WINDING_FACTOR = 1.3
DECIMALS = 3
EARTH_RADIUS_KM = 6371.0088
DEPOT = dict(
    location_id=0, place_id="", type="depot", lat=47.4050, lon=19.1100,
    street="Depot (warehouse)", housenumber="", district=23,
    full_address="Furniture central warehouse, Budapest, Hungary",
)
LOCATION_FIELDS = ["location_id", "place_id", "type", "lat", "lon",
                   "street", "housenumber", "district", "full_address"]

SETS = {
    "tuning": [(f"tune_{s}", 250, s) for s in range(1001, 1005)],
    "validation": [(f"val_{s}", 250, s) for s in range(2001, 2021)],
    "test": [(f"test_{s}", 250, s) for s in range(3001, 3101)],
    "scaling": [(f"scale_n{n}_{s}", n, s)
                for n in (50, 100, 250, 500, 1000) for s in range(4001, 4005)],
}


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _csv_text(fields, rows) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def make_locations(pool: list[dict], n: int, seed: int) -> list[dict]:
    chosen = random.Random(seed).sample(pool, n)
    rows = [dict(DEPOT)]
    for location_id, place in enumerate(chosen, start=1):
        rows.append(dict(
            location_id=location_id, place_id=place["place_id"], type="customer",
            lat=place["lat"], lon=place["lon"], street=place["street"],
            housenumber=place["housenumber"], district=place["district"],
            full_address=place["full_address"],
        ))
    return rows


def make_orders(furniture: list[dict], locations: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    customers = [r for r in locations if r["type"] == "customer"]
    orders = []
    for order_id, customer in enumerate(customers, start=1):
        n_items = rng.choices(ITEMS_PER_ORDER, weights=ITEMS_WEIGHTS)[0]
        picked = rng.sample(furniture, n_items)
        weight = volume = 0.0
        for piece in picked:
            volume += float(piece["length_m"]) * float(piece["width_m"]) * float(piece["height_m"])
            weight += float(piece["weight_kg"])
        orders.append(dict(
            order_id=order_id,
            location_id=customer["location_id"],
            num_items=n_items,
            item_ids=";".join(p["id"] for p in picked),
            total_weight_kg=round(weight, 3),
            total_volume_m3=round(volume, 4),
            loading_time_min=round(LOAD_BASE_MIN + LOAD_PER_KG_MIN * weight, 1),
            packing_time_min=round(UNLOAD_BASE_MIN + UNLOAD_PER_KG_MIN * weight, 1),
        ))
    return orders


def make_distances(locations: list[dict]) -> str:
    locs = sorted(locations, key=lambda r: int(r["location_id"]))
    ids = [int(r["location_id"]) for r in locs]
    coords = [(float(r["lat"]), float(r["lon"])) for r in locs]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["from_id"] + ids)
    for i in range(len(locs)):
        row = [ids[i]]
        for j in range(len(locs)):
            row.append(0.0 if i == j else
                       round(haversine_km(*coords[i], *coords[j]) * WINDING_FACTOR, DECIMALS))
        w.writerow(row)
    return buf.getvalue()


def scaled_vehicles(n_customers: int) -> str:
    """vehicles.csv with the base fleet repeated ceil(n / 250) times."""
    text = (DATASET / "vehicles.csv").read_text(encoding="utf-8")
    copies = max(1, math.ceil(n_customers / 250))
    if copies == 1:
        return text
    rows = list(csv.DictReader(io.StringIO(text)))
    out = []
    for c in range(copies):
        for r in rows:
            out.append({**r, "vehicle_id": str(c * len(rows) + int(r["vehicle_id"]))})
    return _csv_text(list(rows[0].keys()), out)

def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def build_instance(name: str, n: int, seed: int, dest: Path) -> dict[str, str]:
    """Write one instance folder; return {file: sha256}."""
    pool = read_csv(DATASET / "places_pool.csv")
    furniture = read_csv(DATASET / "furniture_clean.csv")
    if n > len(pool):
        raise ValueError(f"{name}: {n} customers but only {len(pool)} places in the pool")

    locations = make_locations(pool, n, seed)
    orders = make_orders(furniture, locations, seed)
    params = json.loads((DATASET / "params.json").read_text(encoding="utf-8"))
    params["num_customers"] = n
    params["instance"] = {"name": name, "seed": seed,
                          "generator": "experiments/make_instances.py"}

    files = {
        "locations.csv": _csv_text(LOCATION_FIELDS, locations),
        "orders.csv": _csv_text(list(orders[0].keys()), orders),
        "distances.csv": make_distances(locations),
        "vehicles.csv": scaled_vehicles(n),
        "params.json": json.dumps(params, indent=2, ensure_ascii=False) + "\n",
    }
    dest.mkdir(parents=True, exist_ok=True)
    digests = {}
    for fname, text in files.items():
        data = text.encode("utf-8")
        (dest / fname).write_bytes(data)
        digests[fname] = hashlib.sha256(data).hexdigest()
    shutil.copyfile(DATASET / "furniture_clean.csv", dest / "furniture_clean.csv")
    return digests


def instance_stats(dest: Path) -> dict:
    """Size of the instance: demand against the fleet's capacity."""
    orders = read_csv(dest / "orders.csv")
    vehicles = read_csv(dest / "vehicles.csv")
    vol = sum(float(o["total_volume_m3"]) for o in orders)
    kg = sum(float(o["total_weight_kg"]) for o in orders)
    service_h = sum(float(o["loading_time_min"]) + float(o["packing_time_min"]) for o in orders) / 60
    usable = sum(0.9 * float(v["max_volume_m3"]) for v in vehicles)
    return {"customers": len(orders), "vehicles": len(vehicles),
            "volume_m3": round(vol, 2), "weight_kg": round(kg, 1),
            "service_h": round(service_h, 2),
            "fleet_rounds_by_volume": round(vol / usable, 2),
            "service_h_per_vehicle": round(service_h / len(vehicles), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", default="test", choices=[*SETS, "all"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true",
                    help="regenerate and compare with instances/manifest.json")
    a = ap.parse_args()

    names = list(SETS) if a.set == "all" else [a.set]
    manifest_path = a.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    mismatches = 0
    for set_name in names:
        for name, n, seed in SETS[set_name]:
            dest = a.out / set_name / name
            digests = build_instance(name, n, seed, dest)
            entry = {"set": set_name, "customers": n, "seed": seed, "sha256": digests,
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
        print("all instances identical to the manifest" if mismatches == 0
              else f"{mismatches} instance(s) differ from the manifest")
        raise SystemExit(1 if mismatches else 0)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1) + "\n")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
