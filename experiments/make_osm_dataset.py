"""Build a road dataset by slicing the routed pool (no routing, seconds).

    python -m experiments.make_osm_dataset --out data/osm               # same 250 customers + orders as data/base/
    python -m experiments.make_osm_dataset --out data/osm_s9 --customers 250 --seed 9

Needs ``cache/road_pool/road_pool.npz`` from experiments/build_road_pool.py.
Writes the same files as data/road_ga/ (so main.py, the report and
experiments/road_geometry.py work unchanged):

    locations.csv  orders.csv  vehicles.csv  furniture_clean.csv  params.json
    road_distance_km.csv  road_duration_min.csv  road_meta.json

Feasibility gate (exit 1 if any check fails, the folder is then marked
unfinished by not writing params.json):
    * src.data_loader.load_problem reads it (finite, >= 0, zero diagonal, orders)
    * every order fits the largest vehicle (weight, volume minus the free share)
    * every customer can be served alone inside the working day:
      depot -> customer -> depot drive + loading + unloading <= day length
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

from experiments.import_road import matrix_csv
from experiments.make_instances import (DEPOT, LOCATION_FIELDS, _csv_text, make_orders,
                                        read_csv, scaled_vehicles)
from src.data_loader import load_problem

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "base"
ROAD = ROOT / "data" / "road_ga"
POOL_DIR = ROOT / "cache" / "road_pool"


def load_pool(pool_dir: Path) -> tuple[dict, dict]:
    npz = pool_dir / "road_pool.npz"
    if not npz.exists():
        raise SystemExit(f"{npz} not found: run python -m experiments.build_road_pool first")
    with np.load(npz) as z:
        pool = {k: z[k] for k in z.files}
    meta = json.loads((pool_dir / "pool_meta.json").read_text(encoding="utf-8"))
    return pool, meta


def pick_locations(pool: dict, same_as: Path | None, n: int | None, seed: int | None):
    """(locations rows, orders rows or None, pool indices with depot first)."""
    index = {str(p): i for i, p in enumerate(pool["place_id"]) if i > 0}
    if same_as is not None:
        locations = sorted(read_csv(same_as / "locations.csv"), key=lambda r: int(r["location_id"]))
        idx = [0]
        bad = []
        for r in locations[1:]:
            i = index.get(str(r["place_id"]))
            if i is None or not pool["usable"][i]:
                bad.append(r["place_id"])
            idx.append(i)
        if bad:
            raise SystemExit(f"{len(bad)} customers of {same_as} are not usable in the pool "
                             f"(see pool_meta.json 'excluded'): place_id {bad[:10]}")
        return locations, read_csv(same_as / "orders.csv"), idx

    places = {r["place_id"]: r for r in read_csv(DATASET / "places_pool.csv")}
    usable = [i for i in range(1, len(pool["place_id"])) if pool["usable"][i]]
    if n > len(usable):
        raise SystemExit(f"only {len(usable)} usable places in the pool, asked for {n}")
    chosen = random.Random(seed).sample(usable, n)
    locations = [dict(DEPOT)]
    for location_id, i in enumerate(chosen, start=1):
        p = places[str(pool["place_id"][i])]
        locations.append(dict(location_id=location_id, place_id=p["place_id"], type="customer",
                              lat=p["lat"], lon=p["lon"], street=p["street"],
                              housenumber=p["housenumber"], district=p["district"],
                              full_address=p["full_address"]))
    return locations, None, [0] + chosen


def feasibility_problems(folder: Path) -> list[str]:
    data = load_problem(folder)
    p = data.params
    problems = []
    big = max(data.vehicles, key=lambda v: v.max_volume_m3)
    room = big.max_volume_m3 * (1 - p.required_free_volume_ratio)
    day = p.day_end_min
    for c in range(1, len(data.lat)):
        if data.weight[c] > big.max_weight_kg or data.volume[c] > room:
            problems.append(f"customer {c}: order does not fit the largest vehicle")
        alone = data.duration[0, c] + data.duration[c, 0] + data.loading[c] + data.unloading[c]
        if day is not None and alone > day:
            problems.append(f"customer {c}: alone needs {alone:.0f} min > day {day:.0f} min")
    return problems


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "osm")
    ap.add_argument("--pool", type=Path, default=POOL_DIR)
    ap.add_argument("--same-as", type=Path, default=None,
                    help="copy customers and orders of this dataset (default: data/base/)")
    ap.add_argument("--customers", type=int, default=None, help="sample this many usable places")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    same_as = None if a.customers else (a.same_as or DATASET)

    pool, pmeta = load_pool(a.pool)
    locations, orders, idx = pick_locations(pool, same_as, a.customers, a.seed)
    n = len(locations) - 1
    if orders is None:
        orders = make_orders(read_csv(DATASET / "furniture_clean.csv"), locations, a.seed)

    sub = np.ix_(idx, idx)
    km, mins = pool["km"][sub], pool["min"][sub]
    vertex = [int(v) for v in pool["vertex"][idx]]
    same_vertex = [[i, j] for i in range(n + 1) for j in range(i + 1, n + 1) if vertex[i] == vertex[j]]

    params = json.loads((ROAD / "params.json").read_text(encoding="utf-8"))
    params["num_customers"] = n
    params["distance_source"] = "road"
    if a.customers:
        params["instance"] = {"seed": a.seed, "generator": "experiments/make_osm_dataset.py"}
    road_meta = {
        "source": "own OSM pool (experiments/build_road_pool.py)",
        "source_sha256": pmeta["pool_sha256"],
        "units": {"road_distance_km.csv": "km", "road_duration_min.csv": "min"},
        "directed": True,
        "path_objective": "estimated travel time; distance measured along the SAME selected path",
        "provider": "local_osm_dijkstra",
        "map": "OpenStreetMap contributors",
        "osm_raw_sha256": pmeta["osm_raw_sha256"],
        "osm_extract_source": pmeta["osm_extract_source"],
        "speed_assumptions_kmh": pmeta["speed_assumptions_kmh"],
        "max_snap_distance_m": float(pool["snap_m"][idx].max()),
        "same_road_vertex_pairs": same_vertex,
        "pool_place_ids": [str(p) for p in pool["place_id"][idx]],
        "osm_node_ids": vertex,
        "snapped_coordinates_lonlat": [[float(x), float(y)] for x, y in pool["snapped_lonlat"][idx]],
    }

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "params.json").unlink(missing_ok=True)
    files = {
        "locations.csv": _csv_text(LOCATION_FIELDS, locations),
        "orders.csv": _csv_text(list(orders[0].keys()), orders),
        "vehicles.csv": scaled_vehicles(n),
        "road_distance_km.csv": matrix_csv(km.tolist(), 1.0),
        "road_duration_min.csv": matrix_csv(mins.tolist(), 1.0),
        "road_meta.json": json.dumps(road_meta, indent=1) + "\n",
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    shutil.copyfile(DATASET / "furniture_clean.csv", out / "furniture_clean.csv")
    # params.json last: an interrupted or infeasible build cannot be loaded.
    tmp = out / "params.json"
    tmp.write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    problems = feasibility_problems(out)
    if problems:
        tmp.unlink()
        print("\n".join(problems[:20]))
        sys.exit(f"{len(problems)} feasibility problem(s); {out} left without params.json")
    print(f"wrote {out}: {n} customers, max snap {road_meta['max_snap_distance_m']:.0f} m, "
          f"{len(same_vertex)} same-vertex pairs; feasibility checks passed")


if __name__ == "__main__":
    main()
