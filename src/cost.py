"""Trip-level cost, time, load and feasibility calculations.

Sign conventions
----------------
* ``base_cost_eur`` = fuel + maintenance + wage for all trip minutes at the
  normal hourly rate (no overtime premium).
* ``overtime_cost`` = the extra 1x premium for minutes beyond the 8 h shift.
* ``cost_eur`` = ``base_cost_eur + overtime_cost`` (standalone trip cost).
* A vehicle that does several trips is charged overtime on its *total* day
  time.  The solution evaluator therefore sums ``base_cost_eur`` over the
  trips and adds the vehicle-day overtime premium once.  For a single-trip
  vehicle the result is exactly ``cost_eur``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_loader import ProblemData


@dataclass(frozen=True)
class PenaltyWeights:
    """EUR penalty per unit of constraint violation.

    The defaults make capacity / max-50 violations overwhelmingly expensive.
    The minimum-customer rule is deliberately *soft* (the user's choice), but
    still expensive enough that the optimizer only uses a short trip when it
    buys a real saving.
    """

    per_kg_over: float = 1_000.0
    per_m3_over: float = 10_000.0
    per_minute_over_shift: float = 0.0  # 0 because vehicle-day overtime is paid
    per_customer_under_min: float = 2_000.0
    per_customer_over_max: float = 1_000_000.0


DEFAULT_WEIGHTS = PenaltyWeights()


@dataclass(frozen=True)
class SegmentStats:
    """Distance and load of one contiguous segment of a giant tour."""

    customers: tuple[int, ...]
    distance_km: float
    weight_kg: float
    volume_m3: float
    loading_min: float
    unloading_min: float

    @property
    def count(self) -> int:
        return len(self.customers)


@dataclass(frozen=True)
class TripMetrics:
    vehicle_index: int
    vehicle_id: int
    customers: tuple[int, ...]

    distance_km: float
    driving_min: float
    loading_min: float
    unloading_min: float
    total_min: float

    weight_kg: float
    volume_m3: float

    fuel_cost: float
    maintenance_cost: float
    wage_base_cost: float
    overtime_min: float
    overtime_cost: float
    base_cost_eur: float
    cost_eur: float

    weight_over: float
    raw_volume_over: float
    volume_over: float
    under_min: int
    over_max: int
    penalty: float

    capacity_ok: bool
    free_volume_ok: bool
    min_customers_ok: bool
    max_customers_ok: bool
    feasible: bool

    # Convenient aliases used in reports / tests.
    @property
    def time_min(self) -> float:
        return self.total_min

    @property
    def time(self) -> float:
        return self.total_min

    @property
    def cost(self) -> float:
        return self.cost_eur

    @property
    def count(self) -> int:
        return len(self.customers)

    @property
    def violations(self) -> dict[str, float]:
        return {
            "weight_over_kg": self.weight_over,
            "raw_volume_over_m3": self.raw_volume_over,
            "free_volume_over_m3": self.volume_over,
            "customers_under_min": float(self.under_min),
            "customers_over_max": float(self.over_max),
        }

    def as_dict(self) -> dict:
        return {
            "vehicle_index": self.vehicle_index,
            "vehicle_id": self.vehicle_id,
            "customers": list(self.customers),
            "count": self.count,
            "distance_km": self.distance_km,
            "driving_min": self.driving_min,
            "loading_min": self.loading_min,
            "unloading_min": self.unloading_min,
            "time_min": self.total_min,
            "weight_kg": self.weight_kg,
            "volume_m3": self.volume_m3,
            "raw_volume_over_m3": self.raw_volume_over,
            "free_volume_over_m3": self.volume_over,
            "fuel_cost": self.fuel_cost,
            "maintenance_cost": self.maintenance_cost,
            "wage_base_cost": self.wage_base_cost,
            "overtime_min": self.overtime_min,
            "overtime_cost": self.overtime_cost,
            "base_cost_eur": self.base_cost_eur,
            "cost_eur": self.cost_eur,
            "penalty": self.penalty,
            "feasible": self.feasible,
            "violations": self.violations,
        }


# ---------------------------------------------------------------------------
# Segment statistics
# ---------------------------------------------------------------------------
def segment_stats(data: ProblemData, customers: tuple[int, ...] | list[int]) -> SegmentStats:
    """Compute distance and load for an ordered customer sequence.

    The trip always starts and ends at the depot.
    """
    cust = tuple(int(c) for c in customers)
    if not cust:
        return SegmentStats((), 0.0, 0.0, 0.0, 0.0, 0.0)

    arr = np.asarray(cust, dtype=np.int64)
    d = data.distance
    distance = float(d[0, arr[0]])
    if len(arr) > 1:
        distance += float(d[arr[:-1], arr[1:]].sum())
    distance += float(d[arr[-1], 0])

    return SegmentStats(
        customers=cust,
        distance_km=distance,
        weight_kg=float(data.weight[arr].sum()),
        volume_m3=float(data.volume[arr].sum()),
        loading_min=float(data.loading[arr].sum()),
        unloading_min=float(data.unloading[arr].sum()),
    )


def segment_stats_from_cumulative(
    customers: tuple[int, ...],
    distance_km: float,
    weight_kg: float,
    volume_m3: float,
    loading_min: float,
    unloading_min: float,
) -> SegmentStats:
    """Build SegmentStats from values already computed by the Split DP."""
    return SegmentStats(
        customers=customers,
        distance_km=float(distance_km),
        weight_kg=float(weight_kg),
        volume_m3=float(volume_m3),
        loading_min=float(loading_min),
        unloading_min=float(unloading_min),
    )


# ---------------------------------------------------------------------------
# Trip evaluation
# ---------------------------------------------------------------------------
def trip_metrics_from_stats(
    data: ProblemData,
    vehicle_index: int,
    stats: SegmentStats,
    weights: PenaltyWeights = DEFAULT_WEIGHTS,
) -> TripMetrics:
    """Evaluate one trip on one specific vehicle."""
    v = data.vehicles[vehicle_index]
    p = data.params

    driving_min = stats.distance_km / v.avg_speed_kmh * 60.0
    loading_min = stats.loading_min if p.loading_counts_as_work_time else 0.0
    unloading_min = stats.unloading_min
    total_min = driving_min + loading_min + unloading_min

    fuel_cost = stats.distance_km * v.fuel_l_per_km * p.fuel_price_per_l
    maintenance_cost = stats.distance_km * v.maintenance_per_1000km / 1000.0
    wage_base_cost = total_min / 60.0 * v.wage_per_h

    overtime_min = max(0.0, total_min - p.shift_minutes)
    overtime_cost = (
        overtime_min / 60.0 * v.wage_per_h * max(0.0, p.overtime_multiplier - 1.0)
    )
    base_cost_eur = fuel_cost + maintenance_cost + wage_base_cost
    cost_eur = base_cost_eur + overtime_cost

    weight_over = max(0.0, stats.weight_kg - v.max_weight_kg)
    usable_volume = v.max_volume_m3 * (1.0 - p.required_free_volume_ratio)
    raw_volume_over = max(0.0, stats.volume_m3 - v.max_volume_m3)
    volume_over = max(0.0, stats.volume_m3 - usable_volume)
    under_min = max(0, p.min_customers_per_trip - stats.count)
    over_max = max(0, stats.count - p.max_customers_per_trip)

    penalty = (
        weights.per_kg_over * weight_over
        + weights.per_m3_over * volume_over
        + weights.per_customer_under_min * under_min
        + weights.per_customer_over_max * over_max
    )

    capacity_ok = weight_over <= 1e-9 and raw_volume_over <= 1e-9
    free_volume_ok = volume_over <= 1e-9
    min_customers_ok = under_min == 0
    max_customers_ok = over_max == 0
    feasible = capacity_ok and free_volume_ok and min_customers_ok and max_customers_ok

    return TripMetrics(
        vehicle_index=vehicle_index,
        vehicle_id=v.vehicle_id,
        customers=stats.customers,
        distance_km=stats.distance_km,
        driving_min=driving_min,
        loading_min=loading_min,
        unloading_min=unloading_min,
        total_min=total_min,
        weight_kg=stats.weight_kg,
        volume_m3=stats.volume_m3,
        fuel_cost=fuel_cost,
        maintenance_cost=maintenance_cost,
        wage_base_cost=wage_base_cost,
        overtime_min=overtime_min,
        overtime_cost=overtime_cost,
        base_cost_eur=base_cost_eur,
        cost_eur=cost_eur,
        weight_over=weight_over,
        raw_volume_over=raw_volume_over,
        volume_over=volume_over,
        under_min=under_min,
        over_max=over_max,
        penalty=penalty,
        capacity_ok=capacity_ok,
        free_volume_ok=free_volume_ok,
        min_customers_ok=min_customers_ok,
        max_customers_ok=max_customers_ok,
        feasible=feasible,
    )


def trip_metrics(
    data: ProblemData,
    vehicle_index: int,
    customers: tuple[int, ...] | list[int],
    weights: PenaltyWeights = DEFAULT_WEIGHTS,
) -> TripMetrics:
    """Convenience wrapper: build stats from customers, then evaluate."""
    return trip_metrics_from_stats(
        data, vehicle_index, segment_stats(data, customers), weights
    )


def day_overtime_extra(
    data: ProblemData,
    vehicle_index: int,
    total_day_min: float,
) -> float:
    """Extra 1x overtime premium for a vehicle-day, in EUR."""
    v = data.vehicles[vehicle_index]
    overtime_min = max(0.0, total_day_min - data.params.shift_minutes)
    return (
        overtime_min
        / 60.0
        * v.wage_per_h
        * max(0.0, data.params.overtime_multiplier - 1.0)
    )
