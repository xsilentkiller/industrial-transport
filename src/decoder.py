"""Split decoder: random keys -> giant tour -> trips -> vehicles.

Pipeline
--------
1. ``keys_to_tour`` sorts the random keys into a giant tour.
2. ``greedy_cut`` makes a quick capacity-aware partition (baseline).
3. ``split_dp`` solves the optimal contiguous partition of that same giant
   tour, with a soft minimum-customer penalty and a hard maximum of 50.
4. ``assign_trips`` assigns the resulting trips to the five vehicles,
   charging vehicle-day overtime, and refines the assignment with moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cost import (
    DEFAULT_WEIGHTS,
    PenaltyWeights,
    SegmentStats,
    TripMetrics,
    day_overtime_extra,
    segment_stats,
    trip_metrics_from_stats,
)
from .data_loader import ProblemData


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
            "trips": [t.as_dict() for t in self.trips],
        }


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
            & (volume_m3 <= data.vehicle_usable_volume + 1e-9)
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

    segments: list[SegmentStats] = []
    current: list[int] = []
    weight = volume = loading = unloading = distance = 0.0
    last = 0

    def close() -> None:
        nonlocal current, weight, volume, loading, unloading, distance, last
        if current:
            segments.append(
                SegmentStats(
                    customers=tuple(current),
                    distance_km=distance,
                    weight_kg=weight,
                    volume_m3=volume,
                    loading_min=loading,
                    unloading_min=unloading,
                )
            )
        current = []
        weight = volume = loading = unloading = distance = 0.0
        last = 0

    for c in tour:
        c = int(c)
        if current:
            trial_dist = distance - d[last, 0] + d[last, c] + d[c, 0]
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

    max_len = min(config.max_count, n)
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

    dp = np.full(n + 1, np.inf, dtype=np.float64)
    prev = np.full(n + 1, -1, dtype=np.int64)
    dp[0] = 0.0

    cap_w = data.vehicle_weight_cap
    cap_v = data.vehicle_usable_volume
    speeds = data.vehicle_speed
    wages = data.vehicle_wage
    fuel_rates = data.vehicle_fuel_rate
    maint = data.vehicle_maint_per_km
    shift_min = p.shift_minutes
    ot_extra_rate = max(0.0, p.overtime_multiplier - 1.0)
    fuel_price = p.fuel_price_per_l

    min_customers = p.min_customers_per_trip
    max_customers = p.max_customers_per_trip

    for i in range(n):
        if not np.isfinite(dp[i]):
            continue
        j0 = i + min_len - 1
        j1 = min(n - 1, i + max_len - 1)
        if j0 > j1:
            continue
        js = np.arange(j0, j1 + 1, dtype=np.int64)
        cnt = js - i + 1

        weight = cum_w[js + 1] - cum_w[i]
        volume = cum_v[js + 1] - cum_v[i]
        loading = cum_load[js + 1] - cum_load[i]
        unloading = cum_unload[js + 1] - cum_unload[i]
        service = unloading + (loading if p.loading_counts_as_work_time else 0.0)
        distance = d0[i] + (cum_d[js] - cum_d[i]) + dback[js]

        count_pen = (
            w.per_customer_under_min * np.maximum(0, min_customers - cnt)
            + w.per_customer_over_max * np.maximum(0, cnt - max_customers)
        )

        best = np.full(len(js), np.inf, dtype=np.float64)
        for k in range(data.n_vehicles):
            time = distance * (60.0 / speeds[k]) + service
            base = (
                distance * (fuel_rates[k] * fuel_price + maint[k])
                + wages[k] * (time / 60.0)
            )
            overtime = (
                np.maximum(0.0, time - shift_min)
                * (wages[k] * ot_extra_rate / 60.0)
            )
            cap_pen = (
                w.per_kg_over * np.maximum(0.0, weight - cap_w[k])
                + w.per_m3_over * np.maximum(0.0, volume - cap_v[k])
            )
            total = base + overtime + cap_pen + count_pen
            np.minimum(best, total, out=best)

        if config.dp_trip_fixed_cost:
            best += config.dp_trip_fixed_cost

        cand = dp[i] + best
        target = js + 1
        better = cand < dp[target] - 1e-9
        if np.any(better):
            idx = target[better]
            dp[idx] = cand[better]
            prev[idx] = i

    if not np.isfinite(dp[n]) or prev[n] < 0:
        # No partition exists (normally only when n < min and enforce_min).
        return [segment_stats(data, tour)]

    # Backtrack and rebuild the segment statistics from the cumulative sums.
    chosen: list[SegmentStats] = []
    j = n
    while j > 0:
        i = int(prev[j])
        if i < 0:
            return [segment_stats(data, tour)]
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

    shift_min = data.params.shift_minutes
    shift_w = weights.per_minute_over_shift

    def ot(k: int, t: float) -> float:
        return day_overtime_extra(data, k, t)

    def shift_pen(t: float) -> float:
        return shift_w * max(0.0, t - shift_min)

    # Most constrained first: large volume, then large weight, then count.
    order = sorted(
        range(m),
        key=lambda i: (
            -segments[i].volume_m3,
            -segments[i].weight_kg,
            -segments[i].count,
        ),
    )

    assign = np.full(m, -1, dtype=np.int64)
    vehicle_times = np.zeros(V, dtype=np.float64)

    for i in order:
        best_k = -1
        best_delta = np.inf
        best_time = np.inf
        for k in range(V):
            delta = (
                base[i, k]
                + pens[i, k]
                + ot(k, vehicle_times[k] + times[i, k])
                - ot(k, vehicle_times[k])
                + shift_pen(vehicle_times[k] + times[i, k])
                - shift_pen(vehicle_times[k])
            )
            # Equal-cost assignments go to the less-loaded vehicle, so the
            # fleet is not needlessly concentrated on one van.
            if (
                delta < best_delta - 1e-9
                or (
                    abs(delta - best_delta) <= 1e-9
                    and vehicle_times[k] < best_time - 1e-9
                )
            ):
                best_delta = delta
                best_time = vehicle_times[k]
                best_k = k
        if best_k < 0:
            best_k = 0
        assign[i] = best_k
        vehicle_times[best_k] += times[i, best_k]

    # Move local search: relocate one trip to another vehicle.
    if config.assignment_refine:
        for _ in range(max(1, config.assignment_passes)):
            improved = False
            for i in range(m):
                a = int(assign[i])
                Ta = vehicle_times[a]
                old_max = float(vehicle_times.max())
                best_b = -1
                best_delta = 0.0
                best_new_max = old_max
                for b in range(V):
                    if b == a:
                        continue
                    Tb = vehicle_times[b]
                    old = (
                        base[i, a]
                        + pens[i, a]
                        + ot(a, Ta)
                        + ot(b, Tb)
                        + shift_pen(Ta)
                        + shift_pen(Tb)
                    )
                    new = (
                        base[i, b]
                        + pens[i, b]
                        + ot(a, Ta - times[i, a])
                        + ot(b, Tb + times[i, b])
                        + shift_pen(Ta - times[i, a])
                        + shift_pen(Tb + times[i, b])
                    )
                    delta = new - old
                    trial = vehicle_times.copy()
                    trial[a] -= times[i, a]
                    trial[b] += times[i, b]
                    new_max = float(trial.max())
                    if (
                        delta < best_delta - 1e-9
                        or (
                            abs(delta - best_delta) <= 1e-9
                            and new_max < best_new_max - 1e-9
                        )
                    ):
                        best_delta = delta
                        best_new_max = new_max
                        best_b = b
                if best_b >= 0:
                    vehicle_times[a] -= times[i, a]
                    vehicle_times[best_b] += times[i, best_b]
                    assign[i] = best_b
                    improved = True
            if not improved:
                break

    # Optional swap local search (useful for the final best solution).
    if config.assignment_swaps and m > 1:
        for _ in range(max(1, config.assignment_passes)):
            improved = False
            for i in range(m):
                a = int(assign[i])
                for j in range(i + 1, m):
                    b = int(assign[j])
                    if a == b:
                        continue
                    Ta = vehicle_times[a]
                    Tb = vehicle_times[b]
                    old = (
                        base[i, a]
                        + base[j, b]
                        + pens[i, a]
                        + pens[j, b]
                        + ot(a, Ta)
                        + ot(b, Tb)
                        + shift_pen(Ta)
                        + shift_pen(Tb)
                    )
                    new = (
                        base[i, b]
                        + base[j, a]
                        + pens[i, b]
                        + pens[j, a]
                        + ot(a, Ta - times[i, a] + times[j, a])
                        + ot(b, Tb - times[j, b] + times[i, b])
                        + shift_pen(Ta - times[i, a] + times[j, a])
                        + shift_pen(Tb - times[j, b] + times[i, b])
                    )
                    if new < old - 1e-9:
                        vehicle_times[a] += times[j, a] - times[i, a]
                        vehicle_times[b] += times[i, b] - times[j, b]
                        assign[i], assign[j] = b, a
                        improved = True
                        a = b
            if not improved:
                break

    trips: list[TripPlan] = []
    total_distance = total_time = total_weight = total_volume = 0.0
    for i, k in enumerate(assign):
        mm = metrics[i][int(k)]
        trips.append(
            TripPlan(
                vehicle_index=int(k),
                vehicle_id=data.vehicles[int(k)].vehicle_id,
                customers=segments[i].customers,
                metrics=mm,
            )
        )
        total_distance += mm.distance_km
        total_time += mm.total_min
        total_weight += mm.weight_kg
        total_volume += mm.volume_m3

    vehicle_total_min = np.zeros(V, dtype=np.float64)
    for trip in trips:
        vehicle_total_min[trip.vehicle_index] += trip.metrics.total_min

    vehicle_overtime_min = np.maximum(
        0.0, vehicle_total_min - data.params.shift_minutes
    )
    vehicle_overtime_cost = np.array(
        [day_overtime_extra(data, k, vehicle_total_min[k]) for k in range(V)],
        dtype=np.float64,
    )
    day_overtime_cost = float(vehicle_overtime_cost.sum())

    # Base cost = fuel + maintenance + all hours at the normal wage rate.
    # The vehicle-day overtime premium is added once here, not per trip.
    true_cost = sum(
        trip.metrics.base_cost_eur for trip in trips
    ) + day_overtime_cost
    penalty = sum(trip.metrics.penalty for trip in trips) + shift_w * float(
        vehicle_overtime_min.sum()
    )
    feasible = penalty <= 1e-7

    return Solution(
        trips=trips,
        true_cost_eur=float(true_cost),
        penalty=float(penalty),
        objective=float(true_cost + penalty),
        feasible=feasible,
        total_distance_km=float(total_distance),
        total_time_min=float(total_time),
        total_weight_kg=float(total_weight),
        total_volume_m3=float(total_volume),
        vehicle_total_min=vehicle_total_min,
        vehicle_overtime_min=vehicle_overtime_min,
        vehicle_overtime_cost=vehicle_overtime_cost,
        day_overtime_cost=day_overtime_cost,
        method=method,
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
    """Deterministic feasible seed: cut the tour into fixed-size chunks."""
    segments = [
        segment_stats(data, tour[i : i + chunk_size])
        for i in range(0, len(tour), chunk_size)
    ]
    return assign_trips(data, segments, config, method=f"chunk-{chunk_size}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def check_solution(
    data: ProblemData,
    solution: Solution,
    require_min_customers: bool = False,
) -> list[str]:
    """Return a list of human-readable problems; empty means all checks pass."""
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
        if len(cust) > p.max_customers_per_trip:
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
        if not trip.metrics.free_volume_ok:
            errors.append(
                f"trip {t}: free-volume violation "
                f"(+{trip.metrics.volume_over:.4f} m3 beyond the 10% reserve)"
            )
        if trip.vehicle_index < 0 or trip.vehicle_index >= data.n_vehicles:
            errors.append(f"trip {t}: invalid vehicle index {trip.vehicle_index}")

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
