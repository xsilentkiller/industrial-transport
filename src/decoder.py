"""Split decoder: random keys -> giant tour -> trips -> vehicles.

Pipeline
--------
1. ``keys_to_tour`` sorts the random keys into a giant tour.
2. ``greedy_cut`` makes a quick capacity-aware partition (baseline).
3. ``split_dp`` solves the optimal contiguous partition of that same giant
   tour.  Capacity (weight and 100 % volume) is hard: a trip that does not
   fit a vehicle is never put on it.  The minimum / maximum customer counts
   are soft penalties.
4. ``assign_trips`` assigns the resulting trips to the five vehicles,
   charging vehicle-day overtime, and refines the assignment with moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cost import (
    CAPACITY_TOL,
    DEFAULT_WEIGHTS,
    PenaltyWeights,
    SegmentStats,
    TripMetrics,
    day_metrics,
    overtime_lower_bound_min,
    return_leg_min,
    segment_stats,
    trip_metrics_from_stats,
)
from .data_loader import ProblemData
from .kernels import assign_kernel, split_dp_kernel


def _f64(a) -> np.ndarray:
    """Contiguous float64 view/copy for the compiled kernels."""
    return np.ascontiguousarray(a, dtype=np.float64)


def _vehicle_arrays(data: ProblemData) -> dict[str, np.ndarray]:
    return {
        "speeds": _f64(data.vehicle_speed),
        "wages": _f64(data.vehicle_wage),
        "fuel_rates": _f64(data.vehicle_fuel_rate),
        "maint": _f64(data.vehicle_maint_per_km),
        "cap_w": _f64(data.vehicle_weight_cap),
        "cap_v": _f64(data.vehicle_usable_volume),
        "cap_v_full": _f64(data.vehicle_volume_cap),
    }


@dataclass(frozen=True)
class DecoderConfig:
    # None means "take the value from params.json".
    min_customers: int | None = None
    max_customers: int | None = None

    # Soft min: the DP may create a short trip but pays the per-customer
    # penalty.  The hard variant is used only when the soft result is
    # infeasible, as a safety net that guarantees a min-10 partition.
    try_hard_min_variant: bool = True
    enforce_min_customers: bool = False

    # A small fixed cost per trip can be used as a search heuristic to avoid
    # an unnecessary number of depot returns.  It is NOT part of the reported
    # true EUR cost.
    dp_trip_fixed_cost: float = 0.0

    use_greedy_baseline: bool = True
    assignment_refine: bool = True
    assignment_passes: int = 3
    assignment_swaps: bool = False

    penalty: PenaltyWeights = DEFAULT_WEIGHTS

    # Longest trip (in customers) the Split DP tries.  The maximum-customer
    # rule is soft, so the DP must look past it; capacity and time stop real
    # trips far earlier (~20 customers on the current data).  The DP runtime
    # grows linearly with this bound.
    dp_max_trip_len: int = 100

    @property
    def min_count(self) -> int:
        return 10 if self.min_customers is None else int(self.min_customers)

    @property
    def max_count(self) -> int:
        return 50 if self.max_customers is None else int(self.max_customers)


@dataclass
class TripPlan:
    """One trip: vehicle, ordered customers, time and cost."""

    vehicle_index: int
    vehicle_id: int
    customers: tuple[int, ...]
    metrics: TripMetrics

    @property
    def time_min(self) -> float:
        return self.metrics.total_min

    @property
    def cost_eur(self) -> float:
        return self.metrics.cost_eur

    @property
    def base_cost_eur(self) -> float:
        return self.metrics.base_cost_eur

    @property
    def distance_km(self) -> float:
        return self.metrics.distance_km

    @property
    def feasible(self) -> bool:
        return self.metrics.feasible

    def as_dict(self) -> dict:
        d = self.metrics.as_dict()
        d["vehicle_index"] = self.vehicle_index
        d["vehicle_id"] = self.vehicle_id
        d["customers"] = list(self.customers)
        return d


@dataclass
class Solution:
    """A list of trips plus fleet-level totals.

    ``solution.trips`` is the required list of trips.  Iterating over the
    Solution itself also yields the trips.
    """

    trips: list[TripPlan]
    true_cost_eur: float
    penalty: float
    objective: float
    feasible: bool

    total_distance_km: float
    total_time_min: float
    total_weight_kg: float
    total_volume_m3: float

    vehicle_total_min: np.ndarray
    vehicle_overtime_min: np.ndarray
    vehicle_overtime_cost: np.ndarray
    day_overtime_cost: float
    method: str
    # Daily window, per vehicle (minutes after day start).
    vehicle_night_min: np.ndarray | None = None
    vehicle_last_delivery_min: np.ndarray | None = None
    vehicle_day_penalty: np.ndarray | None = None
    # Part of ``penalty`` from soft rules (counts, volume reserve); the rest
    # is hard (daily window).
    soft_penalty: float = 0.0
    # Line items.  cost_breakdown sums to true_cost_eur (company money, for
    # analysis); penalty_breakdown sums to penalty (search only).
    cost_breakdown: dict = field(default_factory=dict)
    penalty_breakdown: dict = field(default_factory=dict)

    def __iter__(self):
        return iter(self.trips)

    def __len__(self) -> int:
        return len(self.trips)

    def __getitem__(self, item):
        return self.trips[item]

    @property
    def n_trips(self) -> int:
        return len(self.trips)

    @property
    def shift_over_min(self) -> float:
        return float(self.vehicle_overtime_min.sum())

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "true_cost_eur": self.true_cost_eur,
            "penalty": self.penalty,
            "soft_penalty": self.soft_penalty,
            "hard_penalty": self.penalty - self.soft_penalty,
            "cost_breakdown": dict(self.cost_breakdown),
            "penalty_breakdown": dict(self.penalty_breakdown),
            "objective": self.objective,
            "feasible": self.feasible,
            "n_trips": self.n_trips,
            "total_distance_km": self.total_distance_km,
            "total_time_min": self.total_time_min,
            "total_weight_kg": self.total_weight_kg,
            "total_volume_m3": self.total_volume_m3,
            "day_overtime_cost": self.day_overtime_cost,
            "shift_over_min": self.shift_over_min,
            "vehicle_total_min": self.vehicle_total_min.tolist(),
            "vehicle_overtime_min": self.vehicle_overtime_min.tolist(),
            "vehicle_overtime_cost": self.vehicle_overtime_cost.tolist(),
            "longest_day_min": self.longest_day_min,
            "vehicle_night_min": _list(self.vehicle_night_min),
            "vehicle_last_delivery_min": _list(self.vehicle_last_delivery_min),
            "vehicle_day_penalty": _list(self.vehicle_day_penalty),
            "trips": [t.as_dict() for t in self.trips],
        }

    @property
    def longest_day_min(self) -> float:
        return float(self.vehicle_total_min.max()) if len(self.vehicle_total_min) else 0.0

    def day_schedule(self, data: ProblemData) -> list[dict]:
        """Per-vehicle working day as wall-clock times (for reports)."""
        p = data.params
        rows = []
        for k in range(len(self.vehicle_total_min)):
            T = float(self.vehicle_total_min[k])
            trips = [t for t in self.trips if t.vehicle_index == k]
            clock = 0.0
            trip_rows = []
            for t in trips:
                trip_rows.append(
                    {"start": p.clock(clock), "end": p.clock(clock + t.time_min),
                     "customers": len(t.customers)}
                )
                clock += t.time_min
            last = (
                None if self.vehicle_last_delivery_min is None or T <= 0
                else float(self.vehicle_last_delivery_min[k])
            )
            rows.append(
                {
                    "vehicle_id": data.vehicles[k].vehicle_id,
                    "model": data.vehicles[k].model,
                    "trips": len(trips),
                    "work_h": T / 60.0,
                    "start": p.clock(0.0) if trips else None,
                    "last_delivery": p.clock(last) if last is not None else None,
                    "finish": p.clock(T) if trips else None,
                    "overtime_h": float(self.vehicle_overtime_min[k]) / 60.0,
                    "night_h": (
                        0.0 if self.vehicle_night_min is None
                        else float(self.vehicle_night_min[k]) / 60.0
                    ),
                    "premium_eur": float(self.vehicle_overtime_cost[k]),
                    "window_penalty": (
                        0.0 if self.vehicle_day_penalty is None
                        else float(self.vehicle_day_penalty[k])
                    ),
                    "trip_times": trip_rows,
                }
            )
        return rows

    def overtime_summary(self, data: ProblemData) -> dict:
        """Overtime used vs the minimum no plan with this much work can avoid."""
        total = float(self.vehicle_total_min.sum())
        lb = overtime_lower_bound_min(data, total)
        used = float(self.vehicle_overtime_min.sum())
        return {
            "total_work_h": total / 60.0,
            "overtime_h": used / 60.0,
            "unavoidable_overtime_h": lb / 60.0,
            "avoidable_overtime_h": max(0.0, used - lb) / 60.0,
            "night_h": (
                0.0 if self.vehicle_night_min is None
                else float(self.vehicle_night_min.sum()) / 60.0
            ),
            "longest_day_h": self.longest_day_min / 60.0,
        }


def _list(arr) -> list | None:
    return None if arr is None else np.asarray(arr, dtype=float).tolist()


# ---------------------------------------------------------------------------
# Random-key encoding
# ---------------------------------------------------------------------------
def keys_to_tour(keys: np.ndarray | list[float]) -> list[int]:
    """Sort random keys ascending; customer ids are 1..n."""
    arr = np.asarray(keys, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("keys must be one-dimensional")
    return (np.argsort(arr, kind="stable") + 1).astype(int).tolist()


def tour_to_keys(tour: list[int] | tuple[int, ...]) -> np.ndarray:
    """Rank-encode a permutation as random keys in (0, 1)."""
    arr = np.asarray(tour, dtype=np.int64)
    n = len(arr)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if sorted(arr.tolist()) != list(range(1, n + 1)):
        raise ValueError("tour must be a permutation of 1..n")
    keys = np.empty(n, dtype=np.float64)
    keys[arr - 1] = (np.arange(n, dtype=np.float64) + 0.5) / n
    return keys


# ---------------------------------------------------------------------------
# Greedy cut
# ---------------------------------------------------------------------------
def _any_vehicle_can_carry(
    data: ProblemData,
    weight_kg: float,
    volume_m3: float,
    count: int,
    max_customers: int,
) -> bool:
    if count > max_customers:
        return False
    return bool(
        np.any(
            (weight_kg <= data.vehicle_weight_cap + 1e-9)
            & (volume_m3 <= data.vehicle_volume_cap + 1e-9)
        )
    )


def greedy_cut(
    data: ProblemData,
    tour: list[int] | tuple[int, ...],
    config: DecoderConfig = DecoderConfig(),
) -> list[SegmentStats]:
    """Sequentially fill a trip until no vehicle can take the next customer."""
    if not tour:
        return []
    max_customers = config.max_count
    d = data.distance
    t = data.duration if data.duration is not None else np.zeros_like(d)

    segments: list[SegmentStats] = []
    current: list[int] = []
    weight = volume = loading = unloading = distance = road = 0.0
    last = 0

    def close() -> None:
        nonlocal current, weight, volume, loading, unloading, distance, road, last
        if current:
            segments.append(
                SegmentStats(
                    customers=tuple(current),
                    distance_km=distance,
                    weight_kg=weight,
                    volume_m3=volume,
                    loading_min=loading,
                    unloading_min=unloading,
                    road_min=road,
                )
            )
        current = []
        weight = volume = loading = unloading = distance = road = 0.0
        last = 0

    for c in tour:
        c = int(c)
        if current:
            trial_dist = distance - d[last, 0] + d[last, c] + d[c, 0]
            trial_road = road - t[last, 0] + t[last, c] + t[c, 0]
            trial_w = weight + data.weight[c]
            trial_v = volume + data.volume[c]
            fits = _any_vehicle_can_carry(
                data, trial_w, trial_v, len(current) + 1, max_customers
            )
            if fits:
                current.append(c)
                weight, volume = trial_w, trial_v
                loading += data.loading[c]
                unloading += data.unloading[c]
                distance = trial_dist
                road = trial_road
                last = c
                continue
            close()

        # Start a new trip.  A single customer is always kept, even if it
        # exceeds every vehicle, so it can be reported as a penalised trip.
        current = [c]
        weight = data.weight[c]
        volume = data.volume[c]
        loading = data.loading[c]
        unloading = data.unloading[c]
        distance = d[0, c] + d[c, 0]
        road = t[0, c] + t[c, 0]
        last = c

    close()
    return segments


# ---------------------------------------------------------------------------
# Split DP
# ---------------------------------------------------------------------------
def split_dp(
    data: ProblemData,
    tour: list[int] | tuple[int, ...],
    config: DecoderConfig = DecoderConfig(),
    enforce_min: bool | None = None,
) -> list[SegmentStats]:
    """Optimal contiguous partition of a giant tour.

    The DP is O(n * max_trip_length * n_vehicles), vectorised over the
    possible end positions of each segment.  It returns the chosen trips.
    """
    n = len(tour)
    if n == 0:
        return []

    if enforce_min is None:
        enforce_min = config.enforce_min_customers

    max_len = min(max(config.max_count, config.dp_max_trip_len), n)
    min_len = max(1, config.min_count) if enforce_min else 1
    if min_len > max_len:
        # Cannot satisfy the minimum; fall back to the soft version.
        min_len = 1

    p = data.params
    w = config.penalty
    tour_arr = np.asarray(tour, dtype=np.int64)

    # Cumulative arrays for O(1) segment statistics.
    seg_w = data.weight[tour_arr]
    seg_v = data.volume[tour_arr]
    seg_load = data.loading[tour_arr]
    seg_unload = data.unloading[tour_arr]
    cum_w = np.concatenate(([0.0], np.cumsum(seg_w)))
    cum_v = np.concatenate(([0.0], np.cumsum(seg_v)))
    cum_load = np.concatenate(([0.0], np.cumsum(seg_load)))
    cum_unload = np.concatenate(([0.0], np.cumsum(seg_unload)))

    d = data.distance
    d0 = d[0, tour_arr]
    dback = d[tour_arr, 0]
    if n > 1:
        dseq = d[tour_arr[:-1], tour_arr[1:]]
        cum_d = np.concatenate(([0.0], np.cumsum(dseq)))
    else:
        cum_d = np.zeros(1, dtype=np.float64)

    road_mode = data.duration is not None
    if road_mode:
        tm = data.duration
        t0 = tm[0, tour_arr]
        tback = tm[tour_arr, 0]
        if n > 1:
            cum_t = np.concatenate(([0.0], np.cumsum(tm[tour_arr[:-1], tour_arr[1:]])))
        else:
            cum_t = np.zeros(1, dtype=np.float64)
    else:
        t0 = tback = cum_t = np.zeros(1, dtype=np.float64)

    # The DP itself runs in compiled code (src/kernels.py).  Per segment and
    # vehicle it prices: fuel + maintenance + wages + trip overtime, the
    # soft volume-reserve and customer-count penalties; a segment that does
    # not fit vehicle k (weight or 100 % volume) is impossible on k.
    vec = _vehicle_arrays(data)
    dp, prev = split_dp_kernel(
        n, min_len, max_len,
        _f64(cum_w), _f64(cum_v), _f64(cum_load), _f64(cum_unload),
        _f64(d0), _f64(dback), _f64(cum_d),
        road_mode, _f64(t0), _f64(tback), _f64(cum_t),
        bool(p.loading_counts_as_work_time),
        vec["speeds"], vec["wages"], vec["fuel_rates"], vec["maint"],
        vec["cap_w"], vec["cap_v"], vec["cap_v_full"],
        float(p.volume_overflow_eur_per_m3),
        float(p.shift_minutes),
        float(max(0.0, p.overtime_multiplier - 1.0)),
        float(p.fuel_price_per_l),
        int(p.min_customers_per_trip), int(p.max_customers_per_trip),
        float(w.per_customer_under_min), float(w.per_customer_over_max),
        float(config.dp_trip_fixed_cost or 0.0),
        float(CAPACITY_TOL),
    )

    if not np.isfinite(dp[n]) or prev[n] < 0:
        # No partition exists (only possible with enforce_min, when the
        # minimum count cannot be met within capacity).  Never fall back to
        # one oversized trip: use the soft version, which always has a
        # solution because every single order fits some vehicle.
        if min_len > 1:
            return split_dp(data, tour, config, enforce_min=False)
        return greedy_cut(data, tour, config)

    # Backtrack and rebuild the segment statistics from the cumulative sums.
    chosen: list[SegmentStats] = []
    j = n
    while j > 0:
        i = int(prev[j])
        if i < 0:
            return greedy_cut(data, tour, config)
        jj = j - 1
        customers = tuple(int(x) for x in tour_arr[i : jj + 1])
        chosen.append(
            SegmentStats(
                customers=customers,
                distance_km=float(d0[i] + (cum_d[jj] - cum_d[i]) + dback[jj]),
                weight_kg=float(cum_w[j] - cum_w[i]),
                volume_m3=float(cum_v[j] - cum_v[i]),
                loading_min=float(cum_load[j] - cum_load[i]),
                unloading_min=float(cum_unload[j] - cum_unload[i]),
                road_min=(
                    float(t0[i] + (cum_t[jj] - cum_t[i]) + tback[jj]) if road_mode else 0.0
                ),
            )
        )
        j = i
    chosen.reverse()
    return chosen


# ---------------------------------------------------------------------------
# Trip-to-vehicle assignment
# ---------------------------------------------------------------------------
def _build_metrics_matrix(
    data: ProblemData,
    segments: list[SegmentStats],
    weights: PenaltyWeights,
) -> list[list[TripMetrics]]:
    return [
        [
            trip_metrics_from_stats(data, k, seg, weights)
            for k in range(data.n_vehicles)
        ]
        for seg in segments
    ]


def assign_trips(
    data: ProblemData,
    segments: list[SegmentStats],
    config: DecoderConfig = DecoderConfig(),
    method: str = "split",
) -> Solution:
    """Assign trips to vehicles, minimising true cost + penalties.

    Greedy construction (most constrained trips first) followed by
    move/swap local search on the assignment.
    """
    if not segments:
        return Solution(
            trips=[],
            true_cost_eur=0.0,
            penalty=0.0,
            objective=0.0,
            feasible=True,
            total_distance_km=0.0,
            total_time_min=0.0,
            total_weight_kg=0.0,
            total_volume_m3=0.0,
            vehicle_total_min=np.zeros(data.n_vehicles),
            vehicle_overtime_min=np.zeros(data.n_vehicles),
            vehicle_overtime_cost=np.zeros(data.n_vehicles),
            day_overtime_cost=0.0,
            method=method,
        )

    weights = config.penalty
    V = data.n_vehicles
    m = len(segments)
    metrics = _build_metrics_matrix(data, segments, weights)

    times = np.array(
        [[metrics[i][k].total_min for k in range(V)] for i in range(m)],
        dtype=np.float64,
    )
    base = np.array(
        [[metrics[i][k].base_cost_eur for k in range(V)] for i in range(m)],
        dtype=np.float64,
    )
    pens = np.array(
        [[metrics[i][k].penalty for k in range(V)] for i in range(m)],
        dtype=np.float64,
    )
    # Hard capacity: trip i may only ever be put on a vehicle it fits.
    fits = np.array(
        [[metrics[i][k].capacity_ok for k in range(V)] for i in range(m)],
        dtype=bool,
    )
    no_vehicle = np.flatnonzero(~fits.any(axis=1))
    if len(no_vehicle):
        i = int(no_vehicle[0])
        raise ValueError(
            f"trip {list(segments[i].customers)} "
            f"({segments[i].weight_kg:.1f} kg, {segments[i].volume_m3:.3f} m3) "
            "fits on no vehicle; it must be split before assignment"
        )
    # Return leg of each trip on each vehicle: decides when the last delivery
    # of a vehicle-day happens (the trip with the longest return goes last).
    rets = np.array(
        [
            [return_leg_min(data, k, segments[i].customers[-1]) for k in range(V)]
            for i in range(m)
        ],
        dtype=np.float64,
    )

    shift_min = data.params.shift_minutes
    shift_w = weights.per_minute_over_shift

    # Most constrained first: large volume, then large weight, then count.
    order = sorted(
        range(m),
        key=lambda i: (
            -segments[i].volume_m3,
            -segments[i].weight_kg,
            -segments[i].count,
        ),
    )

    # Greedy construction (equal-cost ties go to the less-loaded vehicle),
    # then move and swap local search on the assignment.  Each vehicle-day
    # is scored as overtime/night premium + daily-window penalty
    # (cost.day_metrics).  Runs in compiled code (src/kernels.py).
    p = data.params
    night_from = p.night_from_min
    last_by = p.last_delivery_min
    day_end = p.day_end_min
    assign = assign_kernel(
        _f64(times), _f64(base), _f64(pens), np.ascontiguousarray(fits), _f64(rets),
        np.asarray(order, dtype=np.int64),
        _f64([v.wage_per_h / 60.0 for v in data.vehicles]),
        float(shift_min),
        night_from is not None, float(night_from or 0.0),
        float(max(0.0, p.overtime_multiplier - 1.0)),
        float(max(0.0, float(p.night_multiplier) - 1.0)) if night_from is not None else 0.0,
        last_by is not None, float(last_by or 0.0),
        day_end is not None, float(day_end or 0.0),
        float(weights.per_minute_late_delivery), float(weights.per_minute_past_day_end),
        float(shift_w),
        bool(config.assignment_refine), bool(config.assignment_swaps),
        int(config.assignment_passes),
    )

    # Trip order inside each vehicle-day: keep the decoded order, but run the
    # trip with the longest return leg last (earliest possible last delivery).
    trip_order: list[int] = []
    for k in range(V):
        idxs = [int(i) for i in np.flatnonzero(assign == k)]
        if idxs:
            last = max(idxs, key=lambda i: (rets[i, k], -i))
            idxs.remove(last)
            idxs.append(last)
        trip_order.extend(idxs)

    trips: list[TripPlan] = []
    total_distance = total_time = total_weight = total_volume = 0.0
    for i in trip_order:
        k = int(assign[i])
        mm = metrics[i][k]
        trips.append(
            TripPlan(
                vehicle_index=k,
                vehicle_id=data.vehicles[k].vehicle_id,
                customers=segments[i].customers,
                metrics=mm,
            )
        )
        total_distance += mm.distance_km
        total_time += mm.total_min
        total_weight += mm.weight_kg
        total_volume += mm.volume_m3

    vehicle_total_min = np.zeros(V, dtype=np.float64)
    vehicle_max_ret = np.zeros(V, dtype=np.float64)
    for i, k in enumerate(assign):
        vehicle_total_min[k] += times[i, k]
        vehicle_max_ret[k] = max(vehicle_max_ret[k], rets[i, k])

    days = [
        day_metrics(data, k, vehicle_total_min[k], vehicle_max_ret[k], weights)
        for k in range(V)
    ]
    vehicle_overtime_min = np.array([d.beyond_shift_min for d in days])
    vehicle_overtime_cost = np.array([d.extra_cost for d in days])
    day_overtime_cost = float(vehicle_overtime_cost.sum())
    day_penalty = float(sum(d.penalty for d in days))

    # Base cost = fuel + maintenance + all hours at the normal wage rate.
    # The vehicle-day overtime and night premiums are added once here.
    true_cost = sum(
        trip.metrics.base_cost_eur for trip in trips
    ) + day_overtime_cost
    # Penalties are never added to the true cost; only to the search
    # objective.  Soft = customer-count rules (allowed, discouraged).
    # Hard = daily window (priced so high that breaking it never pays).
    soft_penalty = float(sum(trip.metrics.penalty for trip in trips))
    hard_penalty = day_penalty + shift_w * float(
        np.maximum(0.0, vehicle_total_min - shift_min).sum()
    )
    penalty = soft_penalty + hard_penalty
    # Feasible = every hard rule holds (capacity, daily window).  A plan
    # that only breaks soft rules is still a legal plan.
    feasible = hard_penalty <= 1e-7 and all(t.metrics.capacity_ok for t in trips)

    cost_breakdown = {
        "fuel": float(sum(t.metrics.fuel_cost for t in trips)),
        "maintenance": float(sum(t.metrics.maintenance_cost for t in trips)),
        "wages_base": float(sum(t.metrics.wage_base_cost for t in trips)),
        "overtime_premium": float(sum(d.overtime_cost for d in days)),
        "night_premium": float(sum(d.night_cost for d in days)),
    }
    penalty_breakdown = {
        # soft
        "under_min_customers": float(sum(t.metrics.under_min_penalty for t in trips)),
        "over_max_customers": float(sum(t.metrics.over_max_penalty for t in trips)),
        "volume_reserve_used": float(sum(t.metrics.overflow_penalty for t in trips)),
        # hard (priced so high that breaking them never pays)
        "late_last_delivery": float(sum(
            weights.per_minute_late_delivery * d.late_delivery_min for d in days)),
        "back_after_day_end": float(sum(
            weights.per_minute_past_day_end * d.past_day_end_min for d in days)),
        "over_shift": float(shift_w * np.maximum(0.0, vehicle_total_min - shift_min).sum()),
    }

    return Solution(
        trips=trips,
        true_cost_eur=float(true_cost),
        penalty=float(penalty),
        objective=float(true_cost + penalty),
        feasible=feasible,
        soft_penalty=soft_penalty,
        cost_breakdown=cost_breakdown,
        penalty_breakdown=penalty_breakdown,
        total_distance_km=float(total_distance),
        total_time_min=float(total_time),
        total_weight_kg=float(total_weight),
        total_volume_m3=float(total_volume),
        vehicle_total_min=vehicle_total_min,
        vehicle_overtime_min=vehicle_overtime_min,
        vehicle_overtime_cost=vehicle_overtime_cost,
        day_overtime_cost=day_overtime_cost,
        method=method,
        vehicle_night_min=np.array([d.night_min for d in days]),
        vehicle_last_delivery_min=np.array([d.last_delivery_min for d in days]),
        vehicle_day_penalty=np.array([d.penalty for d in days]),
    )


# ---------------------------------------------------------------------------
# Full decoder
# ---------------------------------------------------------------------------
def decode_tour(
    data: ProblemData,
    tour: list[int] | tuple[int, ...],
    config: DecoderConfig = DecoderConfig(),
) -> Solution:
    """Decode one giant tour into the best of the Split / greedy partitions."""
    segments = split_dp(data, tour, config, enforce_min=False)
    best = assign_trips(data, segments, config, method="split-soft")

    # If the soft partition pays a min-customer penalty, try the hard variant.
    if config.try_hard_min_variant and best.penalty > 1e-7:
        hard_segments = split_dp(data, tour, config, enforce_min=True)
        hard = assign_trips(data, hard_segments, config, method="split-hard")
        if hard.objective < best.objective - 1e-9:
            best = hard

    if config.use_greedy_baseline:
        gsegments = greedy_cut(data, tour, config)
        greedy = assign_trips(data, gsegments, config, method="greedy")
        if greedy.objective < best.objective - 1e-9:
            best = greedy

    return best


def decode_keys(
    data: ProblemData,
    keys: np.ndarray | list[float],
    config: DecoderConfig = DecoderConfig(),
) -> Solution:
    return decode_tour(data, keys_to_tour(keys), config)


def make_chunk_solution(
    data: ProblemData,
    tour: list[int] | tuple[int, ...],
    chunk_size: int = 10,
    config: DecoderConfig = DecoderConfig(),
) -> Solution:
    """Deterministic seed: cut the tour into fixed-size chunks.

    A chunk is cut early when the next customer would not fit on any
    vehicle, so the seed never breaks the hard capacity rule.
    """
    segments: list[SegmentStats] = []
    current: list[int] = []
    weight = volume = 0.0
    for c in tour:
        c = int(c)
        w_new = weight + data.weight[c]
        v_new = volume + data.volume[c]
        if current and not _any_vehicle_can_carry(
            data, w_new, v_new, len(current) + 1, chunk_size
        ):
            segments.append(segment_stats(data, current))
            current, w_new, v_new = [], data.weight[c], data.volume[c]
        current.append(c)
        weight, volume = w_new, v_new
    if current:
        segments.append(segment_stats(data, current))
    return assign_trips(data, segments, config, method=f"chunk-{chunk_size}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def check_solution(
    data: ProblemData,
    solution: Solution,
    require_min_customers: bool = False,
    require_max_customers: bool = False,
) -> list[str]:
    """Return a list of human-readable problems; empty means all checks pass.

    Capacity is always checked (hard rule).  The customer-count limits are
    soft, so they are only reported when explicitly required.
    """
    errors: list[str] = []
    seen: list[int] = []
    n = data.n_customers
    p = data.params

    for t, trip in enumerate(solution.trips):
        cust = list(trip.customers)
        seen.extend(cust)
        if len(cust) != len(set(cust)):
            errors.append(f"trip {t}: duplicate customers")
        for c in cust:
            if not (1 <= c <= n):
                errors.append(f"trip {t}: customer {c} outside 1..{n}")
        if require_max_customers and len(cust) > p.max_customers_per_trip:
            errors.append(
                f"trip {t}: {len(cust)} customers > max {p.max_customers_per_trip}"
            )
        if require_min_customers and len(cust) < p.min_customers_per_trip:
            errors.append(
                f"trip {t}: {len(cust)} customers < min {p.min_customers_per_trip}"
            )
        if not trip.metrics.capacity_ok:
            errors.append(
                f"trip {t}: capacity violation "
                f"(+{trip.metrics.weight_over:.3f} kg, "
                f"+{trip.metrics.raw_volume_over:.4f} m3 raw)"
            )
        if trip.vehicle_index < 0 or trip.vehicle_index >= data.n_vehicles:
            errors.append(f"trip {t}: invalid vehicle index {trip.vehicle_index}")

    if solution.vehicle_day_penalty is not None:
        for k, pen in enumerate(solution.vehicle_day_penalty):
            if pen > 1e-7:
                T = float(solution.vehicle_total_min[k])
                last = float(solution.vehicle_last_delivery_min[k])
                errors.append(
                    f"vehicle {data.vehicles[k].vehicle_id}: daily window violated "
                    f"(last delivery {p.clock(last)}, back {p.clock(T)})"
                )

    if len(seen) != n:
        errors.append(f"customer count {len(seen)} != {n}")
    if sorted(seen) != list(range(1, n + 1)):
        missing = sorted(set(range(1, n + 1)) - set(seen))
        dup = sorted({c for c in seen if seen.count(c) > 1})
        if missing:
            errors.append(f"missing customers: {missing[:10]}...")
        if dup:
            errors.append(f"duplicated customers: {dup[:10]}...")
    return errors
