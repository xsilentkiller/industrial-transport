"""Data loading for the furniture delivery instance.

This is the only module in the project that opens a CSV file.  Everything
downstream receives plain NumPy arrays / small dataclasses through
``ProblemData``.

Files read once
---------------
params.json          global rules
vehicles.csv         fleet specs
locations.csv        depot + customers (lat/lon, ids)
orders.csv           demand, loading/unloading times
distances.csv        251 x 251 distance matrix (km)
furniture_clean.csv  item table used to cross-check order totals

Road mode (``"distance_source": "road"`` in params.json) reads, instead of
distances.csv:

road_distance_km.csv   directed road distance of the fastest path (km)
road_duration_min.csv  directed driving time of the same path (min)

Both come from the GA team's OpenStreetMap road graph (see
experiments/import_road.py).  In road mode every vehicle drives a leg in the
road time; fuel and maintenance are paid on road kilometres.
"""

from __future__ import annotations

import csv
import json
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    currency: str
    distance_unit: str
    time_unit: str
    depot_id: int
    num_customers: int
    shift_hours: float
    overtime_multiplier: float
    fuel_price_per_l: float
    packing_mode: int
    required_free_volume_ratio: float
    min_customers_per_trip: int
    max_customers_per_trip: int
    allow_multiple_trips_per_vehicle: bool
    must_return_to_depot: bool
    loading_counts_as_work_time: bool

    # Packing mode 1: a trip may use the 10 % reserve (90-100 % of the cargo
    # volume) for this PENALTY per m3 above the 90 % line (search only, never
    # part of the real cost).  Above 100 % is physically impossible.
    volume_overflow_eur_per_m3: float = 100.0

    # "euclid" = distances.csv (haversine x 1.3, time = km / avg speed)
    # "road"   = road_distance_km.csv + road_duration_min.csv (directed)
    distance_source: str = "euclid"

    # Daily working window (optional; None = no limit, the original V0 model).
    # Clock times are "HH:MM" strings; every vehicle starts at day_start.
    day_start: str = "08:00"
    night_rate_from: str | None = None   # wage x night_multiplier after this
    night_multiplier: float | None = None
    last_delivery_by: str | None = None  # last customer served by this time
    hard_day_end: str | None = None      # vehicle back at the depot by this time

    @property
    def shift_minutes(self) -> float:
        return float(self.shift_hours) * 60.0

    def minutes_after_start(self, clock: str | None) -> float | None:
        """'22:00' -> minutes after day_start (840 for an 08:00 start)."""
        if clock is None:
            return None
        return _clock_min(clock) - _clock_min(self.day_start)

    @property
    def night_from_min(self) -> float | None:
        return self.minutes_after_start(self.night_rate_from)

    @property
    def last_delivery_min(self) -> float | None:
        return self.minutes_after_start(self.last_delivery_by)

    @property
    def day_end_min(self) -> float | None:
        return self.minutes_after_start(self.hard_day_end)

    def clock(self, minutes_after_start: float) -> str:
        """Minutes after day_start -> 'HH:MM' wall-clock time."""
        total = int(round(_clock_min(self.day_start) + minutes_after_start))
        return f"{total // 60:02d}:{total % 60:02d}"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Params":
        fields = cls.__dataclass_fields__.values()
        required = {
            f.name for f in fields
            if f.default is MISSING and f.default_factory is MISSING
        }
        missing = required.difference(d)
        if missing:
            raise ValueError(f"params.json is missing keys: {sorted(missing)}")
        names = {f.name for f in fields}
        params = cls(**{name: d[name] for name in names if name in d})
        params.validate_day_window()
        return params

    def validate_day_window(self) -> None:
        if (self.night_rate_from is None) != (self.night_multiplier is None):
            raise ValueError("night_rate_from and night_multiplier must be set together")
        checks = [
            ("night_rate_from", self.night_from_min),
            ("last_delivery_by", self.last_delivery_min),
            ("hard_day_end", self.day_end_min),
        ]
        for name, value in checks:
            if value is not None and value < self.shift_minutes:
                raise ValueError(f"{name} must be at or after the end of the normal shift")
        if (
            self.day_end_min is not None
            and self.last_delivery_min is not None
            and self.last_delivery_min > self.day_end_min
        ):
            raise ValueError("last_delivery_by must not be after hard_day_end")


def _clock_min(clock: str) -> float:
    hh, mm = clock.strip().split(":")
    return int(hh) * 60.0 + int(mm)


@dataclass(frozen=True)
class Vehicle:
    vehicle_id: int
    type_id: str
    model: str
    max_weight_kg: float
    max_volume_m3: float
    cargo_length_m: float
    cargo_width_m: float
    cargo_height_m: float
    fuel_l_per_km: float
    top_speed_kmh: float
    avg_speed_kmh: float
    wage_per_h: float
    maintenance_per_1000km: float

    def usable_volume_m3(self, free_ratio: float) -> float:
        """Volume that may actually be loaded (the free-volume rule)."""
        return self.max_volume_m3 * (1.0 - free_ratio)


@dataclass(frozen=True)
class FurnitureItem:
    item_id: str
    name: str
    length_m: float
    width_m: float
    height_m: float
    weight_kg: float

    @property
    def volume_m3(self) -> float:
        return self.length_m * self.width_m * self.height_m


@dataclass
class ProblemData:
    """Plain-array view of one problem instance.

    Index 0 is always the depot.  Customer indices are 1..n_customers and
    coincide with ``location_id`` in the CSV files.
    """

    params: Params
    vehicles: tuple[Vehicle, ...]
    distance: np.ndarray          # (n+1, n+1) km
    weight: np.ndarray            # (n+1,) kg, index 0 = 0
    volume: np.ndarray            # (n+1,) m3, index 0 = 0
    loading: np.ndarray           # (n+1,) min at depot, index 0 = 0
    unloading: np.ndarray         # (n+1,) min at customer, index 0 = 0
    lat: np.ndarray | None = None
    lon: np.ndarray | None = None
    order_ids: np.ndarray | None = None
    item_ids: tuple[tuple[str, ...], ...] = ()
    furniture: tuple[FurnitureItem, ...] = ()
    customer_ids: np.ndarray | None = None
    # Road mode only: (n+1, n+1) directed driving minutes, same for every
    # vehicle.  None = straight-line mode (time = km / vehicle speed).
    duration: np.ndarray | None = None

    # Fast vehicle arrays used by the decoder.  They are derived from
    # ``vehicles`` and ``params`` in __post_init__.
    vehicle_weight_cap: np.ndarray = field(init=False, repr=False)
    vehicle_volume_cap: np.ndarray = field(init=False, repr=False)
    vehicle_usable_volume: np.ndarray = field(init=False, repr=False)
    vehicle_speed: np.ndarray = field(init=False, repr=False)
    vehicle_wage: np.ndarray = field(init=False, repr=False)
    vehicle_fuel_rate: np.ndarray = field(init=False, repr=False)
    vehicle_maint_per_km: np.ndarray = field(init=False, repr=False)
    leg_cost: np.ndarray = field(init=False, repr=False)
    symmetric: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.distance = np.asarray(self.distance, dtype=np.float64)
        self.weight = np.asarray(self.weight, dtype=np.float64)
        self.volume = np.asarray(self.volume, dtype=np.float64)
        self.loading = np.asarray(self.loading, dtype=np.float64)
        self.unloading = np.asarray(self.unloading, dtype=np.float64)

        n = self.distance.shape[0]
        if self.distance.shape != (n, n):
            raise ValueError("distance matrix must be square")
        if self.duration is not None:
            self.duration = np.asarray(self.duration, dtype=np.float64)
            if self.duration.shape != (n, n):
                raise ValueError("duration matrix must match the distance matrix")
        for name in ("weight", "volume", "loading", "unloading"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
        if not self.vehicles:
            raise ValueError("at least one vehicle is required")

        if self.lat is None:
            self.lat = np.zeros(n, dtype=np.float64)
        if self.lon is None:
            self.lon = np.zeros(n, dtype=np.float64)
        if self.order_ids is None:
            self.order_ids = np.full(n, -1, dtype=np.int64)
        if self.customer_ids is None:
            self.customer_ids = np.arange(1, n, dtype=np.int64)

        self.vehicle_weight_cap = np.array(
            [v.max_weight_kg for v in self.vehicles], dtype=np.float64
        )
        self.vehicle_volume_cap = np.array(
            [v.max_volume_m3 for v in self.vehicles], dtype=np.float64
        )
        self.vehicle_usable_volume = self.vehicle_volume_cap * (
            1.0 - self.params.required_free_volume_ratio
        )
        # Capacity is a hard rule: every single order must fit on at least
        # one vehicle, otherwise no legal plan exists at all.
        fits_any = np.any(
            (self.weight[:, None] <= self.vehicle_weight_cap[None, :] + 1e-9)
            & (self.volume[:, None] <= self.vehicle_volume_cap[None, :] + 1e-9),
            axis=1,
        )
        if not fits_any.all():
            bad = [int(i) for i in np.flatnonzero(~fits_any)]
            raise ValueError(
                f"orders of customers {bad[:10]} fit on no vehicle "
                "(weight or volume above every vehicle's capacity)"
            )
        self.vehicle_speed = np.array(
            [v.avg_speed_kmh for v in self.vehicles], dtype=np.float64
        )
        self.vehicle_wage = np.array(
            [v.wage_per_h for v in self.vehicles], dtype=np.float64
        )
        self.vehicle_fuel_rate = np.array(
            [v.fuel_l_per_km for v in self.vehicles], dtype=np.float64
        )
        self.vehicle_maint_per_km = np.array(
            [v.maintenance_per_1000km / 1000.0 for v in self.vehicles],
            dtype=np.float64,
        )

        # Matrix used by routing heuristics (nearest neighbour, 2-opt): plain
        # km in straight-line mode (unchanged behaviour); in road mode the
        # fleet-average EUR of a leg, because a short but slow road can cost
        # more than a longer fast one.
        if self.duration is None:
            self.leg_cost = self.distance
        else:
            per_km = float(np.mean(
                self.vehicle_fuel_rate * self.params.fuel_price_per_l
                + self.vehicle_maint_per_km
            ))
            per_min = float(np.mean(self.vehicle_wage)) / 60.0
            self.leg_cost = self.distance * per_km + self.duration * per_min
        self.symmetric = bool(np.allclose(self.leg_cost, self.leg_cost.T, atol=1e-9))

    # ------------------------------------------------------------------
    @property
    def road_mode(self) -> bool:
        return self.duration is not None

    @property
    def n_customers(self) -> int:
        return int(self.distance.shape[0] - 1)

    @property
    def n_vehicles(self) -> int:
        return len(self.vehicles)

    @property
    def depot_id(self) -> int:
        return int(self.params.depot_id)

    def vehicle_ids(self) -> list[int]:
        return [v.vehicle_id for v in self.vehicles]

    def summary(self) -> str:
        total_w = float(self.weight.sum())
        total_v = float(self.volume.sum())
        total_load = float(self.loading.sum())
        total_unload = float(self.unloading.sum())
        return (
            f"{self.n_customers} customers, {self.n_vehicles} vehicles, "
            f"demand {total_w:.1f} kg / {total_v:.2f} m3, "
            f"service {total_load + total_unload:.1f} min"
        )


# ---------------------------------------------------------------------------
# CSV helpers -- kept private to this module
# ---------------------------------------------------------------------------
def _read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_params(path: Path) -> Params:
    with path.open(encoding="utf-8") as f:
        return Params.from_dict(json.load(f))


def _load_vehicles(path: Path) -> tuple[Vehicle, ...]:
    rows = _read_dict_rows(path)
    vehicles = []
    seen: set[int] = set()
    for r in rows:
        vid = int(r["vehicle_id"])
        if vid in seen:
            raise ValueError(f"duplicate vehicle_id {vid}")
        seen.add(vid)
        vehicles.append(
            Vehicle(
                vehicle_id=vid,
                type_id=r["type_id"],
                model=r["model"],
                max_weight_kg=float(r["max_weight_kg"]),
                max_volume_m3=float(r["max_volume_m3"]),
                cargo_length_m=float(r["cargo_length_m"]),
                cargo_width_m=float(r["cargo_width_m"]),
                cargo_height_m=float(r["cargo_height_m"]),
                fuel_l_per_km=float(r["fuel_l_per_km"]),
                top_speed_kmh=float(r["top_speed_kmh"]),
                avg_speed_kmh=float(r["avg_speed_kmh"]),
                wage_per_h=float(r["wage_per_h"]),
                maintenance_per_1000km=float(r["maintenance_per_1000km"]),
            )
        )
    if not vehicles:
        raise ValueError("vehicles.csv is empty")
    return tuple(vehicles)


def _load_furniture(path: Path) -> tuple[FurnitureItem, ...]:
    rows = _read_dict_rows(path)
    items = []
    seen: set[str] = set()
    for r in rows:
        item_id = r["id"]
        if item_id in seen:
            raise ValueError(f"duplicate furniture id {item_id}")
        seen.add(item_id)
        items.append(
            FurnitureItem(
                item_id=item_id,
                name=r["name"],
                length_m=float(r["length_m"]),
                width_m=float(r["width_m"]),
                height_m=float(r["height_m"]),
                weight_kg=float(r["weight_kg"]),
            )
        )
    return tuple(items)


def _load_locations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_dict_rows(path)
    rows.sort(key=lambda r: int(r["location_id"]))
    ids = [int(r["location_id"]) for r in rows]
    if ids != list(range(len(rows))):
        raise ValueError("locations.csv ids must run 0..N without gaps")
    if not rows or rows[0]["type"] != "depot":
        raise ValueError("locations.csv must start with the depot at id 0")
    lat = np.array([float(r["lat"]) for r in rows], dtype=np.float64)
    lon = np.array([float(r["lon"]) for r in rows], dtype=np.float64)
    return lat, lon


def _load_distances(path: Path, n: int) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_ids = [int(x) for x in header[1:]]
        row_ids: list[int] = []
        values: list[list[float]] = []
        for row in reader:
            row_ids.append(int(row[0]))
            values.append([float(x) for x in row[1:]])
    if col_ids != list(range(n)):
        raise ValueError("distances.csv columns are not 0..N")
    if row_ids != list(range(n)):
        raise ValueError("distances.csv rows are not 0..N")
    d = np.asarray(values, dtype=np.float64)
    if d.shape != (n, n):
        raise ValueError(f"distances.csv shape {d.shape}, expected ({n}, {n})")
    if not np.allclose(d, d.T, atol=1e-9):
        raise ValueError("distance matrix is not symmetric")
    if np.max(np.abs(np.diag(d))) > 1e-12:
        raise ValueError("distance matrix diagonal must be zero")
    if np.any(d[np.triu_indices(n, 1)] <= 0.0):
        raise ValueError("all off-diagonal distances must be positive")
    return d


def _load_directed_matrix(path: Path, n: int, what: str) -> np.ndarray:
    """Directed (asymmetric) matrix with the same layout as distances.csv."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_ids = [int(x) for x in header[1:]]
        row_ids: list[int] = []
        values: list[list[float]] = []
        for row in reader:
            row_ids.append(int(row[0]))
            values.append([float(x) for x in row[1:]])
    if col_ids != list(range(n)) or row_ids != list(range(n)):
        raise ValueError(f"{path.name}: rows/columns are not 0..N")
    m = np.asarray(values, dtype=np.float64)
    if m.shape != (n, n):
        raise ValueError(f"{path.name}: shape {m.shape}, expected ({n}, {n})")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"{path.name}: unreachable pair (non-finite {what})")
    if np.max(np.abs(np.diag(m))) > 1e-12:
        raise ValueError(f"{path.name}: diagonal must be zero")
    # Zero off-diagonal entries are allowed: two addresses can snap to the
    # same road vertex (road_meta.json lists those pairs).
    if np.any(m < 0.0):
        raise ValueError(f"{path.name}: negative {what}")
    return m


def _load_orders(
    path: Path,
    n: int,
    furniture: tuple[FurnitureItem, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[tuple[str, ...], ...]]:
    rows = _read_dict_rows(path)
    by_id = {item.item_id: item for item in furniture}

    weight = np.zeros(n, dtype=np.float64)
    volume = np.zeros(n, dtype=np.float64)
    loading = np.zeros(n, dtype=np.float64)
    unloading = np.zeros(n, dtype=np.float64)
    order_ids = np.full(n, -1, dtype=np.int64)
    item_ids: list[tuple[str, ...]] = [tuple() for _ in range(n)]
    seen: set[int] = set()

    for r in rows:
        loc = int(r["location_id"])
        if not (1 <= loc < n):
            raise ValueError(f"order location_id {loc} is not a customer")
        if loc in seen:
            raise ValueError(f"more than one order for location_id {loc}")
        seen.add(loc)

        ids = tuple(x for x in r["item_ids"].split(";") if x)
        if len(ids) != int(r["num_items"]):
            raise ValueError(f"order {r['order_id']}: num_items does not match item_ids")
        missing = [x for x in ids if x not in by_id]
        if missing:
            raise ValueError(f"order {r['order_id']} references unknown items {missing}")

        calc_w = sum(by_id[x].weight_kg for x in ids)
        calc_v = sum(by_id[x].volume_m3 for x in ids)
        csv_w = float(r["total_weight_kg"])
        csv_v = float(r["total_volume_m3"])
        if abs(calc_w - csv_w) > 0.02:
            raise ValueError(
                f"order {r['order_id']}: weight mismatch ({calc_w:.3f} vs {csv_w:.3f})"
            )
        if abs(calc_v - csv_v) > 0.002:
            raise ValueError(
                f"order {r['order_id']}: volume mismatch ({calc_v:.4f} vs {csv_v:.4f})"
            )

        weight[loc] = csv_w
        volume[loc] = csv_v
        loading[loc] = float(r["loading_time_min"])
        unloading[loc] = float(r["packing_time_min"])
        order_ids[loc] = int(r["order_id"])
        item_ids[loc] = ids

    missing_locs = [i for i in range(1, n) if i not in seen]
    if missing_locs:
        raise ValueError(f"customers without an order: {missing_locs[:10]}...")
    return weight, volume, loading, unloading, order_ids, item_ids


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
def load_problem(dataset_dir: str | Path = "data/base") -> ProblemData:
    """Read the six dataset files once and return plain arrays.

    No other module in the project reads a CSV.
    """
    root = Path(dataset_dir)
    params = _load_params(root / "params.json")
    vehicles = _load_vehicles(root / "vehicles.csv")
    furniture = _load_furniture(root / "furniture_clean.csv")
    lat, lon = _load_locations(root / "locations.csv")
    n = len(lat)

    if n - 1 != params.num_customers:
        raise ValueError(
            f"locations.csv has {n - 1} customers, params.json says {params.num_customers}"
        )
    if params.depot_id != 0:
        raise ValueError("this loader expects depot_id 0 (distance matrix row/column 0)")

    duration = None
    if params.distance_source == "road":
        distance = _load_directed_matrix(root / "road_distance_km.csv", n, "distance")
        duration = _load_directed_matrix(root / "road_duration_min.csv", n, "duration")
    elif params.distance_source == "euclid":
        distance = _load_distances(root / "distances.csv", n)
    else:
        raise ValueError(f"unknown distance_source {params.distance_source!r}")
    weight, volume, loading, unloading, order_ids, item_ids = _load_orders(
        root / "orders.csv", n, furniture
    )

    return ProblemData(
        params=params,
        vehicles=vehicles,
        distance=distance,
        weight=weight,
        volume=volume,
        loading=loading,
        unloading=unloading,
        lat=lat,
        lon=lon,
        order_ids=order_ids,
        item_ids=tuple(item_ids),
        furniture=furniture,
        duration=duration,
    )
