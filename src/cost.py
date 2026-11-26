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

    Penalties are never real money: they are kept apart from ``cost_eur`` /
    ``true_cost_eur`` and only steer the search.

    * Weight above the payload and volume above 100 % of the cargo space are
      *impossible*: they have no penalty, a trip that breaks them simply
      cannot be put on that vehicle (see :func:`fits_vehicle`).
    * The customer-count rules are *soft*: a trip may have fewer than the
      minimum or more than the maximum customers, but pays a penalty for
      each customer below / above the limit.
    """

    per_minute_over_shift: float = 0.0  # 0 because vehicle-day overtime is paid
    per_customer_under_min: float = 500.0   # soft: short trips discouraged
    per_customer_over_max: float = 100.0    # soft: long trips discouraged
    # Daily window (hard rules, priced so that breaking them never pays):
    per_minute_past_day_end: float = 1_000.0     # back at depot after 23:00
    per_minute_late_delivery: float = 1_000.0    # last customer after 22:00


DEFAULT_WEIGHTS = PenaltyWeights()

# Rounding tolerance of the hard capacity check (kg and m3).
CAPACITY_TOL = 1e-9


def fits_vehicle(data: ProblemData, vehicle_index: int, weight_kg: float, volume_m3: float) -> bool:
    """Hard rule: the load must not exceed the payload or 100 % of the volume."""
    return bool(
        weight_kg <= data.vehicle_weight_cap[vehicle_index] + CAPACITY_TOL
        and volume_m3 <= data.vehicle_volume_cap[vehicle_index] + CAPACITY_TOL
    )


@dataclass(frozen=True)
class SegmentStats:
    """Distance and load of one contiguous segment of a giant tour."""

    customers: tuple[int, ...]
    distance_km: float
    weight_kg: float
    volume_m3: float
    loading_min: float
    unloading_min: float
    # Road mode: driving minutes along the road legs (vehicle independent).
    # Straight-line mode leaves it at 0 and time = km / vehicle speed.
    road_min: float = 0.0

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
    overflow_m3: float
    overflow_cost: float
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
            "overflow_m3": self.overflow_m3,
            "overflow_cost": self.overflow_cost,
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
    distance = leg_sum(data.distance, arr)
    road_min = leg_sum(data.duration, arr) if data.duration is not None else 0.0

    return SegmentStats(
        customers=cust,
        distance_km=distance,
        weight_kg=float(data.weight[arr].sum()),
        volume_m3=float(data.volume[arr].sum()),
        loading_min=float(data.loading[arr].sum()),
        unloading_min=float(data.unloading[arr].sum()),
        road_min=road_min,
    )


def leg_sum(m: np.ndarray, arr: np.ndarray) -> float:
    """depot -> arr[0] -> ... -> arr[-1] -> depot, summed over matrix ``m``."""
    total = float(m[0, arr[0]])
    if len(arr) > 1:
        total += float(m[arr[:-1], arr[1:]].sum())
    return total + float(m[arr[-1], 0])


def drive_min(data: ProblemData, vehicle_index: int, distance_km: float, road_min: float) -> float:
    """Driving minutes of a trip: road time in road mode, km / speed otherwise."""
    if data.duration is not None:
        return float(road_min)
    return float(distance_km) / data.vehicles[vehicle_index].avg_speed_kmh * 60.0


def segment_stats_from_cumulative(
    customers: tuple[int, ...],
    distance_km: float,
    weight_kg: float,
    volume_m3: float,
    loading_min: float,
    unloading_min: float,
    road_min: float = 0.0,
) -> SegmentStats:
    """Build SegmentStats from values already computed by the Split DP."""
    return SegmentStats(
        customers=customers,
        distance_km=float(distance_km),
        weight_kg=float(weight_kg),
        volume_m3=float(volume_m3),
        loading_min=float(loading_min),
        unloading_min=float(unloading_min),
        road_min=float(road_min),
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

    driving_min = drive_min(data, vehicle_index, stats.distance_km, stats.road_min)
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
    weight_over = max(0.0, stats.weight_kg - v.max_weight_kg)
    usable_volume = v.max_volume_m3 * (1.0 - p.required_free_volume_ratio)
    raw_volume_over = max(0.0, stats.volume_m3 - v.max_volume_m3)
    volume_over = max(0.0, stats.volume_m3 - usable_volume)
    # Volume in the 90-100 % band is allowed but paid for (real cost, not an
    # infeasibility).  Volume above 100 % is a capacity violation (penalty).
    overflow_m3 = max(0.0, min(stats.volume_m3, v.max_volume_m3) - usable_volume)
    overflow_cost = p.volume_overflow_eur_per_m3 * overflow_m3

    base_cost_eur = fuel_cost + maintenance_cost + wage_base_cost + overflow_cost
    cost_eur = base_cost_eur + overtime_cost

    under_min = max(0, p.min_customers_per_trip - stats.count)
    over_max = max(0, stats.count - p.max_customers_per_trip)

    # Soft rules only: capacity is a hard rule and never has a price.
    penalty = (
        weights.per_customer_under_min * under_min
        + weights.per_customer_over_max * over_max
    )

    capacity_ok = weight_over <= CAPACITY_TOL and raw_volume_over <= CAPACITY_TOL
    # True when the 10 % reserve is untouched; informational only now.
    free_volume_ok = volume_over <= 1e-9
    min_customers_ok = under_min == 0
    max_customers_ok = over_max == 0
    # Feasible = every hard rule holds.  The count rules are soft: breaking
    # them costs a penalty but the trip is still allowed.
    feasible = capacity_ok

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
        overflow_m3=overflow_m3,
        overflow_cost=overflow_cost,
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


# ---------------------------------------------------------------------------
# Vehicle-day evaluation (the daily working window)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DayMetrics:
    """Cost and time-window check of one vehicle's whole working day.

    Every vehicle starts at ``day_start`` (08:00) and drives its trips back to
    back.  Wages for all minutes at the normal rate are already in the trips'
    ``base_cost_eur``; this object holds only the *extra* premiums on top:

    * shift end (16:00) .. night_rate_from (22:00): overtime_multiplier (2x)
    * after night_rate_from (22:00):                night_multiplier   (3x)

    Two hard rules are priced as penalties (never real money):

    * the last customer must be served by ``last_delivery_by`` (22:00)
    * the vehicle must be back at the depot by ``hard_day_end`` (23:00)
    """

    total_min: float
    overtime_min: float          # minutes paid at the overtime rate
    night_min: float             # minutes paid at the night rate
    overtime_cost: float         # extra premium EUR for overtime minutes
    night_cost: float            # extra premium EUR for night minutes
    last_delivery_min: float     # minutes after day start of the last delivery
    late_delivery_min: float     # minutes past last_delivery_by
    past_day_end_min: float      # minutes past hard_day_end
    penalty: float

    @property
    def extra_cost(self) -> float:
        """Real money on top of the normal wage (overtime + night premiums)."""
        return self.overtime_cost + self.night_cost

    @property
    def beyond_shift_min(self) -> float:
        return self.overtime_min + self.night_min

    @property
    def feasible(self) -> bool:
        return self.penalty <= 1e-9


def return_leg_min(data: ProblemData, vehicle_index: int, last_customer: int) -> float:
    """Driving minutes from a trip's last customer back to the depot."""
    c = int(last_customer)
    road = float(data.duration[c, 0]) if data.duration is not None else 0.0
    return drive_min(data, vehicle_index, float(data.distance[c, 0]), road)


def day_metrics(
    data: ProblemData,
    vehicle_index: int,
    total_day_min: float,
    max_return_min: float = 0.0,
    weights: PenaltyWeights = DEFAULT_WEIGHTS,
) -> DayMetrics:
    """Evaluate one vehicle-day.

    ``max_return_min`` is the longest return leg among the vehicle's trips.
    The trips of a day can run in any order, so the best order puts that trip
    last: the last delivery then happens at ``total_day_min - max_return_min``,
    the earliest possible.
    """
    p = data.params
    v = data.vehicles[vehicle_index]
    T = float(total_day_min)
    wage_min = v.wage_per_h / 60.0
    shift = p.shift_minutes

    night_from = p.night_from_min
    ot_until = T if night_from is None else min(T, night_from)
    overtime_min = max(0.0, ot_until - shift)
    night_min = 0.0 if night_from is None else max(0.0, T - max(night_from, shift))

    overtime_cost = overtime_min * wage_min * max(0.0, p.overtime_multiplier - 1.0)
    night_cost = (
        0.0 if night_from is None
        else night_min * wage_min * max(0.0, float(p.night_multiplier) - 1.0)
    )

    if T <= 0.0:
        last_delivery = 0.0
    else:
        last_delivery = max(0.0, T - max_return_min)
    late = 0.0
    if p.last_delivery_min is not None and T > 0.0:
        late = max(0.0, last_delivery - p.last_delivery_min)
    past_end = 0.0
    if p.day_end_min is not None:
        past_end = max(0.0, T - p.day_end_min)

    penalty = (
        weights.per_minute_late_delivery * late
        + weights.per_minute_past_day_end * past_end
    )
    return DayMetrics(
        total_min=T,
        overtime_min=overtime_min,
        night_min=night_min,
        overtime_cost=overtime_cost,
        night_cost=night_cost,
        last_delivery_min=last_delivery,
        late_delivery_min=late,
        past_day_end_min=past_end,
        penalty=penalty,
    )


def day_overtime_extra(
    data: ProblemData,
    vehicle_index: int,
    total_day_min: float,
) -> float:
    """Extra overtime + night premium for a vehicle-day, in EUR (real money)."""
    return day_metrics(data, vehicle_index, total_day_min).extra_cost


def overtime_lower_bound_min(data: ProblemData, total_work_min: float) -> float:
    """Overtime that no assignment can avoid: work beyond V x normal shift."""
    return max(0.0, total_work_min - data.n_vehicles * data.params.shift_minutes)
