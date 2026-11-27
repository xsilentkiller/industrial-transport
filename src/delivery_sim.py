"""Delivery outcomes on top of an optimised plan.

The optimiser plans the day at 08:00; it cannot know which customers will
not be home or will refuse the furniture.  This module takes the finished
plan (the ``best_solution`` dict saved by main.py) and

* computes the planned arrival / departure time of every stop,
* simulates one day in which each stop is **delivered**, **attempted**
  (customer not home) or **refused**,
* runs a Monte Carlo over many simulated days (how many failures to expect).

A failed stop takes the same time as a delivery and the item simply rides
back to the depot, so the schedule and the real cost never change: the
outcomes only label the stops and fill the re-delivery list (not home,
tomorrow) and the returned-to-stock list (refused).  The optimiser and its
benchmark cost are untouched.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .cost import drive_min
from .data_loader import ProblemData

STATUSES = ("delivered", "attempted", "refused")


@dataclass(frozen=True)
class SimConfig:
    """Probabilities per stop (independent) and the random seed."""

    p_not_home: float = 0.05      # customer not home -> "attempted", re-deliver tomorrow
    p_refused: float = 0.02       # customer refuses -> "refused", back to stock
    seed: int = 42
    monte_carlo_days: int = 1000

    def __post_init__(self) -> None:
        if self.p_not_home < 0 or self.p_refused < 0:
            raise ValueError("delivery failure probabilities must not be negative")
        if self.p_not_home + self.p_refused > 1:
            raise ValueError("p_not_home + p_refused must be at most 1")
        if self.monte_carlo_days < 0:
            raise ValueError("monte_carlo_days must not be negative")
        if self.seed < 0:
            raise ValueError("the simulation seed must not be negative")


# ---------------------------------------------------------------------------
# Customer details (read-only; the dataset is never modified)
# ---------------------------------------------------------------------------
def load_addresses(dataset_dir: str | Path) -> dict[int, dict[str, str]]:
    """location_id -> {"address", "district"} from the dataset's locations.csv.

    Returns {} when the file or its address columns are missing.
    """
    path = Path(dataset_dir) / "locations.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or ())
        if not {"location_id", "full_address"} <= cols:
            return {}
        return {
            int(r["location_id"]): {
                "address": r["full_address"].strip(),
                "district": (r.get("district") or "").strip(),
            }
            for r in reader
            if r.get("location_id", "").strip()
        }


def items_text(data: ProblemData, loc: int) -> str:
    """The items of a customer's order: "Dining Chair ×2, Shoe Rack"."""
    if loc >= len(data.item_ids):
        return ""
    names = {f.item_id: f.name for f in data.furniture}
    counts: dict[str, int] = {}
    for item in data.item_ids[loc]:
        counts[item] = counts.get(item, 0) + 1
    parts = []
    for item, k in counts.items():
        name = names.get(item, item)
        parts.append(f"{name} ×{k}" if k > 1 else name)
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Planned stop times
# ---------------------------------------------------------------------------
def _leg_min(data: ProblemData, k: int, a: int, b: int) -> float:
    road = float(data.duration[a, b]) if data.duration is not None else 0.0
    return drive_min(data, k, float(data.distance[a, b]), road)


def stop_schedule(data: ProblemData, solution: dict) -> list[dict]:
    """Planned arrival and departure of every stop (minutes after day start).

    Each vehicle starts at 0 and runs its trips back to back in the order of
    ``solution["trips"]`` (as ``Solution.day_schedule``).  A trip loads all
    its orders at the depot, then drives stop to stop, unloading each order,
    and drives back.  Trip and stop numbers are 1-based per vehicle.
    """
    rows: list[dict] = []
    for k in range(data.n_vehicles):
        clock = 0.0
        trip_no = 0
        for trip in solution.get("trips", []):
            if int(trip["vehicle_index"]) != k:
                continue
            trip_no += 1
            customers = [int(c) for c in trip["customers"]]
            start = clock
            t = start + float(trip["loading_min"])
            prev = 0
            trip_rows = []
            for n, c in enumerate(customers, start=1):
                arrive = t + _leg_min(data, k, prev, c)
                leave = arrive + float(data.unloading[c])
                order = int(data.order_ids[c]) if data.order_ids is not None else -1
                trip_rows.append({
                    "vehicle_id": data.vehicles[k].vehicle_id,
                    "vehicle_index": k,
                    "vehicle_model": data.vehicles[k].model,
                    "trip": trip_no,
                    "stop": n,
                    "location_id": c,
                    "order_id": order,
                    "arrive_min": arrive,
                    "leave_min": leave,
                    "weight_kg": float(data.weight[c]),
                    "volume_m3": float(data.volume[c]),
                })
                t = leave
                prev = c
            end = t + (_leg_min(data, k, prev, 0) if customers else 0.0)
            for r in trip_rows:
                r["trip_start_min"] = start
                r["trip_end_min"] = end
            rows.extend(trip_rows)
            clock = end
    return rows


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------
def _status(u: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """0 = delivered, 1 = attempted, 2 = refused, from uniform draws."""
    out = np.zeros(u.shape, dtype=np.int8)
    out[u < cfg.p_refused + cfg.p_not_home] = 1
    out[u < cfg.p_refused] = 2
    return out


def draw_outcomes(n: int, cfg: SimConfig, rng: np.random.Generator) -> list[str]:
    """One status per stop: refused if u < p_refused, attempted if
    u < p_refused + p_not_home, otherwise delivered."""
    return [STATUSES[int(x)] for x in _status(rng.random(n), cfg)]


def simulate_day(
    data: ProblemData,
    solution: dict,
    addresses: dict[int, dict[str, str]],
    cfg: SimConfig,
) -> dict:
    """One simulated day: every stop gets an outcome; times stay as planned."""
    stops = stop_schedule(data, solution)
    statuses = draw_outcomes(len(stops), cfg, np.random.default_rng(cfg.seed))
    clock = data.params.clock
    records: list[dict] = []
    returns: dict[tuple[int, int], dict] = {}
    for row, status in zip(stops, statuses):
        loc = row["location_id"]
        info = addresses.get(loc, {})
        vehicle = f"{row['vehicle_id']} {row['vehicle_model']}"
        rec = {
            "vehicle_id": row["vehicle_id"],
            "vehicle": vehicle,
            "trip": row["trip"],
            "stop": row["stop"],
            "arrive": clock(row["arrive_min"]),
            "leave": clock(row["leave_min"]),
            "order_id": "" if row["order_id"] < 0 else row["order_id"],
            "location_id": loc,
            "address": info.get("address") or f"location {loc}",
            "district": info.get("district", ""),
            "items": items_text(data, loc),
            "weight_kg": row["weight_kg"],
            "volume_m3": row["volume_m3"],
            "status": status,
        }
        records.append(rec)
        if status != "delivered":
            key = (row["vehicle_index"], row["trip"])
            ret = returns.setdefault(key, {
                "vehicle_id": row["vehicle_id"],
                "vehicle": vehicle,
                "trip": row["trip"],
                "time": clock(row["trip_end_min"]),
                "orders": [],
            })
            ret["orders"].append({"order_id": rec["order_id"], "items": rec["items"],
                                  "status": status})
    counts = {s: sum(1 for r in records if r["status"] == s) for s in STATUSES}
    counts["total"] = len(records)
    return {
        "config": asdict(cfg),
        "records": records,
        "counts": counts,
        "depot_returns": list(returns.values()),
        "redelivery": [r for r in records if r["status"] == "attempted"],
        "returned_to_stock": [r for r in records if r["status"] == "refused"],
    }


def _spread(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {"mean": 0.0, "p5": 0.0, "p95": 0.0}
    return {
        "mean": float(x.mean()),
        "p5": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
    }


def monte_carlo(vehicle_index_per_stop: np.ndarray, n_vehicles: int, cfg: SimConfig) -> dict:
    """Many simulated days at once: how many failed deliveries to expect."""
    veh = np.asarray(vehicle_index_per_stop, dtype=np.int64)
    days = int(cfg.monte_carlo_days)
    rng = np.random.default_rng(cfg.seed + 1)
    status = _status(rng.random((days, veh.size)), cfg)
    attempted = (status == 1).sum(axis=1)
    refused = (status == 2).sum(axis=1)
    failed = status > 0
    shares = []
    for k in range(n_vehicles):
        cols = veh == k
        shares.append(float(failed[:, cols].any(axis=1).mean()) if days and cols.any() else 0.0)
    return {
        "days": days,
        "stops": int(veh.size),
        "attempted": _spread(attempted),
        "refused": _spread(refused),
        "failed": _spread(attempted + refused),
        "share_days_with_failure_by_vehicle": shares,
    }


def build_simulation(data: ProblemData, solution: dict, dataset_dir, cfg: SimConfig) -> dict:
    """Simulated day + Monte Carlo, ready to store in best_solution.json."""
    day = simulate_day(data, solution, load_addresses(dataset_dir), cfg)
    veh = np.array([r["vehicle_index"] for r in stop_schedule(data, solution)], dtype=np.int64)
    day["monte_carlo"] = monte_carlo(veh, data.n_vehicles, cfg)
    return day
