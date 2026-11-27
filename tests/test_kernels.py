"""The Numba kernels (src/kernels.py) must give exactly the same results as
the pure-Python/NumPy code they replaced.

Each test keeps a copy of the original implementation (``_ref_*``) and
compares it with the compiled version on random inputs.  "Same" means equal
floats and equal routes, not "close": the kernels are a speed-only change.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from src.cost import CAPACITY_TOL, day_metrics
from src.data_loader import load_problem
from src.decoder import DecoderConfig, assign_trips, split_dp
from src.kernels import HAVE_NUMBA
from src.pso import two_opt_order, two_opt_order_directed


# ---------------------------------------------------------------------------
# Reference implementations (the code before the Numba change)
# ---------------------------------------------------------------------------
def _ref_split_customers(data, tour, config, enforce_min):
    """Original vectorised Split DP; returns the chosen trips (customers)."""
    n = len(tour)
    max_len = min(max(config.max_count, config.dp_max_trip_len), n)
    min_len = max(1, config.min_count) if enforce_min else 1
    if min_len > max_len:
        min_len = 1
    p, w = data.params, config.penalty
    ta = np.asarray(tour, dtype=np.int64)
    cum = lambda x: np.concatenate(([0.0], np.cumsum(x)))
    cum_w, cum_v = cum(data.weight[ta]), cum(data.volume[ta])
    cum_load, cum_unload = cum(data.loading[ta]), cum(data.unloading[ta])
    d = data.distance
    d0, dback = d[0, ta], d[ta, 0]
    cum_d = cum(d[ta[:-1], ta[1:]]) if n > 1 else np.zeros(1)
    road = data.duration is not None
    if road:
        tm = data.duration
        t0, tback = tm[0, ta], tm[ta, 0]
        cum_t = cum(tm[ta[:-1], ta[1:]]) if n > 1 else np.zeros(1)
    dp = np.full(n + 1, np.inf)
    prev = np.full(n + 1, -1, dtype=np.int64)
    dp[0] = 0.0
    ot = max(0.0, p.overtime_multiplier - 1.0)
    for i in range(n):
        if not np.isfinite(dp[i]):
            continue
        j0, j1 = i + min_len - 1, min(n - 1, i + max_len - 1)
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
        if road:
            road_drive = t0[i] + (cum_t[js] - cum_t[i]) + tback[js]
        count_pen = (
            w.per_customer_under_min * np.maximum(0, p.min_customers_per_trip - cnt)
            + w.per_customer_over_max * np.maximum(0, cnt - p.max_customers_per_trip)
        )
        best = np.full(len(js), np.inf)
        for k in range(data.n_vehicles):
            time = road_drive + service if road else (
                distance * (60.0 / data.vehicle_speed[k]) + service)
            base = (distance * (data.vehicle_fuel_rate[k] * p.fuel_price_per_l
                                + data.vehicle_maint_per_km[k])
                    + data.vehicle_wage[k] * (time / 60.0))
            overtime = np.maximum(0.0, time - p.shift_minutes) * (
                data.vehicle_wage[k] * ot / 60.0)
            cap_v, cap_full = data.vehicle_usable_volume[k], data.vehicle_volume_cap[k]
            overflow = p.volume_overflow_eur_per_m3 * np.clip(
                volume - cap_v, 0.0, cap_full - cap_v)
            total = base + overtime + overflow + count_pen
            fits = (weight <= data.vehicle_weight_cap[k] + CAPACITY_TOL) & (
                volume <= cap_full + CAPACITY_TOL)
            np.minimum(best, np.where(fits, total, np.inf), out=best)
        if config.dp_trip_fixed_cost:
            best += config.dp_trip_fixed_cost
        cand = dp[i] + best
        target = js + 1
        better = cand < dp[target] - 1e-9
        dp[target[better]] = cand[better]
        prev[target[better]] = i
    if not np.isfinite(dp[n]) or prev[n] < 0:
        return None
    out, j = [], n
    while j > 0:
        i = int(prev[j])
        out.append(tuple(int(x) for x in ta[i:j]))
        j = i
    return out[::-1], float(dp[n])


def _ref_assign(data, segments, config):
    """Original assignment search; returns {customers: vehicle index}."""
    weights = config.penalty
    from src.cost import return_leg_min, trip_metrics_from_stats

    V, m = data.n_vehicles, len(segments)
    met = [[trip_metrics_from_stats(data, k, s, weights) for k in range(V)] for s in segments]
    times = np.array([[met[i][k].total_min for k in range(V)] for i in range(m)])
    base = np.array([[met[i][k].base_cost_eur for k in range(V)] for i in range(m)])
    pens = np.array([[met[i][k].penalty for k in range(V)] for i in range(m)])
    fits = np.array([[met[i][k].capacity_ok for k in range(V)] for i in range(m)])
    rets = np.array([[return_leg_min(data, k, segments[i].customers[-1])
                      for k in range(V)] for i in range(m)])
    shift_min = data.params.shift_minutes
    shift_w = weights.per_minute_over_shift

    def day_obj(k, t, r):
        dm = day_metrics(data, k, t, r, weights)
        return dm.extra_cost + dm.penalty + shift_w * max(0.0, t - shift_min)

    def max_ret(k, exclude=(), add=-1):
        best = rets[add, k] if add >= 0 else 0.0
        for idx in np.flatnonzero(assign == k):
            if idx not in exclude and rets[idx, k] > best:
                best = rets[idx, k]
        return float(best)

    order = sorted(range(m), key=lambda i: (-segments[i].volume_m3,
                                            -segments[i].weight_kg, -segments[i].count))
    assign = np.full(m, -1, dtype=np.int64)
    vt, vr = np.zeros(V), np.zeros(V)
    for i in order:
        best_k, best_delta, best_time = -1, np.inf, np.inf
        for k in range(V):
            if not fits[i, k]:
                continue
            delta = (base[i, k] + pens[i, k]
                     + day_obj(k, vt[k] + times[i, k], max(vr[k], rets[i, k]))
                     - day_obj(k, vt[k], vr[k]))
            if delta < best_delta - 1e-9 or (
                    abs(delta - best_delta) <= 1e-9 and vt[k] < best_time - 1e-9):
                best_delta, best_time, best_k = delta, vt[k], k
        assign[i] = best_k
        vt[best_k] += times[i, best_k]
        vr[best_k] = max(vr[best_k], rets[i, best_k])
    if config.assignment_refine:
        for _ in range(max(1, config.assignment_passes)):
            improved = False
            for i in range(m):
                a = int(assign[i])
                Ta, Ra_new = vt[a], max_ret(a, exclude=(i,))
                best_b, best_delta, best_new_max = -1, 0.0, float(vt.max())
                for b in range(V):
                    if b == a or not fits[i, b]:
                        continue
                    Tb = vt[b]
                    old = (base[i, a] + pens[i, a] + day_obj(a, Ta, vr[a])
                           + day_obj(b, Tb, vr[b]))
                    new = (base[i, b] + pens[i, b] + day_obj(a, Ta - times[i, a], Ra_new)
                           + day_obj(b, Tb + times[i, b], max(vr[b], rets[i, b])))
                    delta = new - old
                    trial = vt.copy()
                    trial[a] -= times[i, a]
                    trial[b] += times[i, b]
                    new_max = float(trial.max())
                    if delta < best_delta - 1e-9 or (
                            abs(delta - best_delta) <= 1e-9 and new_max < best_new_max - 1e-9):
                        best_delta, best_new_max, best_b = delta, new_max, b
                if best_b >= 0:
                    vt[a] -= times[i, a]
                    vt[best_b] += times[i, best_b]
                    assign[i] = best_b
                    vr[a], vr[best_b] = max_ret(a), max_ret(best_b)
                    improved = True
            if not improved:
                break
    if config.assignment_swaps and m > 1:
        for _ in range(max(1, config.assignment_passes)):
            improved = False
            for i in range(m):
                a = int(assign[i])
                for j in range(i + 1, m):
                    b = int(assign[j])
                    if a == b or not (fits[i, b] and fits[j, a]):
                        continue
                    Ta, Tb = vt[a], vt[b]
                    old = (base[i, a] + base[j, b] + pens[i, a] + pens[j, b]
                           + day_obj(a, Ta, vr[a]) + day_obj(b, Tb, vr[b]))
                    new = (base[i, b] + base[j, a] + pens[i, b] + pens[j, a]
                           + day_obj(a, Ta - times[i, a] + times[j, a],
                                     max_ret(a, exclude=(i,), add=j))
                           + day_obj(b, Tb - times[j, b] + times[i, b],
                                     max_ret(b, exclude=(j,), add=i)))
                    if new < old - 1e-9:
                        vt[a] += times[j, a] - times[i, a]
                        vt[b] += times[i, b] - times[j, b]
                        assign[i], assign[j] = b, a
                        vr[a], vr[b] = max_ret(a), max_ret(b)
                        improved = True
                        a = b
            if not improved:
                break
    return {segments[i].customers: int(assign[i]) for i in range(m)}


def _ref_two_opt(order, D, directed):
    order = list(order)
    n = len(order)
    if n < 3:
        return order

    def prefix(seq):
        a = np.asarray(seq, dtype=np.int64)
        return (np.concatenate(([0.0], np.cumsum(D[a[:-1], a[1:]]))),
                np.concatenate(([0.0], np.cumsum(D[a[1:], a[:-1]]))))

    fwd, bwd = prefix(order)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                prev = order[i - 1] if i > 0 else 0
                nxt = order[j + 1] if j + 1 < n else 0
                a, b = order[i], order[j]
                if directed:
                    delta = (D[prev, b] + D[a, nxt] + (bwd[j] - bwd[i])
                             - D[prev, a] - D[b, nxt] - (fwd[j] - fwd[i]))
                else:
                    delta = D[prev, b] + D[a, nxt] - D[prev, a] - D[b, nxt]
                if delta < -1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    if directed:
                        fwd, bwd = prefix(order)
                    improved = True
    return order


# ---------------------------------------------------------------------------
class TestKernelsMatchReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets = {name: load_problem(name) for name in ("data/base", "data/osm")}

    def _tours(self, data, count, seed):
        rng = np.random.default_rng(seed)
        ids = np.arange(1, data.n_customers + 1)
        return [[int(x) for x in rng.permutation(ids)] for _ in range(count)]

    def test_numba_is_available(self):
        # Not required for correctness, but without it runs are ~15x slower.
        self.assertTrue(HAVE_NUMBA, "pip install numba (see requirements.txt)")

    def test_split_dp_matches(self):
        cfg = DecoderConfig()
        for name, data in self.datasets.items():
            for tour in self._tours(data, 8, 1):
                for enforce_min in (False, True):
                    ref = _ref_split_customers(data, tour, cfg, enforce_min)
                    if ref is None:
                        continue
                    got = split_dp(data, tour, cfg, enforce_min=enforce_min)
                    with self.subTest(dataset=name, enforce_min=enforce_min):
                        self.assertEqual([s.customers for s in got], ref[0])

    def test_assignment_matches(self):
        configs = [DecoderConfig(), replace(DecoderConfig(), assignment_swaps=True)]
        for name, data in self.datasets.items():
            for cfg in configs:
                for tour in self._tours(data, 6, 2):
                    segs = split_dp(data, tour, cfg)
                    got = assign_trips(data, segs, cfg)
                    got_map = {t.customers: t.vehicle_index for t in got.trips}
                    with self.subTest(dataset=name, swaps=cfg.assignment_swaps):
                        self.assertEqual(got_map, _ref_assign(data, segs, cfg))

    def test_two_opt_matches(self):
        rng = np.random.default_rng(3)
        for directed in (False, True):
            for _ in range(30):
                n = int(rng.integers(1, 40))
                D = rng.random((n + 1, n + 1)) * 10
                if not directed:
                    D = (D + D.T) / 2
                order = [int(x) for x in rng.permutation(np.arange(1, n + 1))]
                got = (two_opt_order_directed(order, D) if directed
                       else two_opt_order(order, D, symmetric=True))
                with self.subTest(directed=directed, n=n):
                    self.assertEqual(got, _ref_two_opt(order, D, directed))


if __name__ == "__main__":
    unittest.main()
