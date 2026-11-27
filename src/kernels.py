"""Numba-compiled inner loops of the decoder.

These functions are the hot paths of a PSO run (profiled on data/osm):

* ``split_dp_kernel``  - the Split dynamic programme over a giant tour
* ``assign_kernel``    - greedy trip-to-vehicle assignment + move/swap search
* ``two_opt_kernel``   - 2-opt on the stop order of one trip

They are a speed-only change: each one performs exactly the same floating-
point operations in the same order as the pure-Python/NumPy code it
replaced, so results are bit-for-bit identical (checked by
tests/test_kernels.py).  Only plain numbers and NumPy arrays go in and out.

Numba compiles each function on first use and caches the machine code in
``__pycache__`` (``cache=True``), so only the very first run pays the
compile time (a few seconds).  Without Numba installed the same functions
run as plain Python: correct, but much slower.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised implicitly
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        """Fallback: run the function as plain Python."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(fn):
            return fn

        return wrap


# ---------------------------------------------------------------------------
# Split DP
# ---------------------------------------------------------------------------
@njit(cache=True)
def split_dp_kernel(
    n, min_len, max_len,
    cum_w, cum_v, cum_load, cum_unload,
    d0, dback, cum_d,
    road_mode, t0, tback, cum_t,
    loading_counts,
    speeds, wages, fuel_rates, maint,
    cap_w, cap_v, cap_v_full,
    overflow_rate, shift_min, ot_extra_rate, fuel_price,
    min_customers, max_customers, w_under, w_over,
    fixed_cost, cap_tol,
):
    """Forward DP; returns (dp, prev).  dp[j] = best cost of tour[:j]."""
    V = speeds.shape[0]
    dp = np.full(n + 1, np.inf)
    prev = np.full(n + 1, -1, dtype=np.int64)
    dp[0] = 0.0
    for i in range(n):
        if not np.isfinite(dp[i]):
            continue
        j0 = i + min_len - 1
        j1 = min(n - 1, i + max_len - 1)
        if j0 > j1:
            continue
        dpi = dp[i]
        for j in range(j0, j1 + 1):
            cnt = j - i + 1
            weight = cum_w[j + 1] - cum_w[i]
            volume = cum_v[j + 1] - cum_v[i]
            loading = cum_load[j + 1] - cum_load[i]
            unloading = cum_unload[j + 1] - cum_unload[i]
            if loading_counts:
                service = unloading + loading
            else:
                service = unloading + 0.0
            distance = d0[i] + (cum_d[j] - cum_d[i]) + dback[j]
            road_drive = 0.0
            if road_mode:
                road_drive = t0[i] + (cum_t[j] - cum_t[i]) + tback[j]
            count_pen = (
                w_under * max(0, min_customers - cnt)
                + w_over * max(0, cnt - max_customers)
            )
            best = np.inf
            for k in range(V):
                if road_mode:
                    time = road_drive + service
                else:
                    time = distance * (60.0 / speeds[k]) + service
                base = (
                    distance * (fuel_rates[k] * fuel_price + maint[k])
                    + wages[k] * (time / 60.0)
                )
                overtime = max(0.0, time - shift_min) * (wages[k] * ot_extra_rate / 60.0)
                over = volume - cap_v[k]
                hi = cap_v_full[k] - cap_v[k]
                if over < 0.0:
                    over = 0.0
                elif over > hi:
                    over = hi
                overflow_pen = overflow_rate * over
                total = base + overtime + overflow_pen + count_pen
                if not (weight <= cap_w[k] + cap_tol and volume <= cap_v_full[k] + cap_tol):
                    total = np.inf
                if total < best:
                    best = total
            if fixed_cost != 0.0:
                best += fixed_cost
            cand = dpi + best
            if cand < dp[j + 1] - 1e-9:
                dp[j + 1] = cand
                prev[j + 1] = i
    return dp, prev


# ---------------------------------------------------------------------------
# Trip-to-vehicle assignment
# ---------------------------------------------------------------------------
@njit(cache=True)
def day_obj(
    T, r, wage_min, shift,
    has_night, night_from, ot_factor, night_factor,
    has_last, last_delivery_min, has_end, day_end_min,
    w_late, w_end, shift_w,
):
    """Premiums + daily-window penalty of one vehicle-day.

    Same arithmetic as ``cost.day_metrics`` followed by
    ``extra_cost + penalty + shift_w * max(0, T - shift)``.
    """
    if has_night:
        ot_until = min(T, night_from)
    else:
        ot_until = T
    overtime_min = max(0.0, ot_until - shift)
    if has_night:
        night_min = max(0.0, T - max(night_from, shift))
    else:
        night_min = 0.0
    overtime_cost = overtime_min * wage_min * ot_factor
    if has_night:
        night_cost = night_min * wage_min * night_factor
    else:
        night_cost = 0.0
    if T <= 0.0:
        last_delivery = 0.0
    else:
        last_delivery = max(0.0, T - r)
    late = 0.0
    if has_last and T > 0.0:
        late = max(0.0, last_delivery - last_delivery_min)
    past_end = 0.0
    if has_end:
        past_end = max(0.0, T - day_end_min)
    penalty = w_late * late + w_end * past_end
    return (overtime_cost + night_cost) + penalty + shift_w * max(0.0, T - shift)


@njit(cache=True)
def _max_ret(assign, rets, k, ex, add):
    """Longest return leg on vehicle k, skipping trip ``ex`` (-1 = none) and
    adding trip ``add`` (-1 = none)."""
    best = rets[add, k] if add >= 0 else 0.0
    for idx in range(assign.shape[0]):
        if assign[idx] == k and idx != ex and rets[idx, k] > best:
            best = rets[idx, k]
    return best


@njit(cache=True)
def assign_kernel(
    times, base, pens, fits, rets, order,
    wage_min, shift,
    has_night, night_from, ot_factor, night_factor,
    has_last, last_delivery_min, has_end, day_end_min,
    w_late, w_end, shift_w,
    refine, swaps, passes,
):
    """Greedy construction, then move and swap local search.  Returns assign."""
    m, V = times.shape
    assign = np.full(m, -1, dtype=np.int64)
    vt = np.zeros(V)
    vr = np.zeros(V)

    for oi in range(m):
        i = order[oi]
        best_k = -1
        best_delta = np.inf
        best_time = np.inf
        for k in range(V):
            if not fits[i, k]:
                continue
            t_new = vt[k] + times[i, k]
            r_new = max(vr[k], rets[i, k])
            delta = (
                base[i, k]
                + pens[i, k]
                + day_obj(t_new, r_new, wage_min[k], shift, has_night, night_from,
                          ot_factor, night_factor, has_last, last_delivery_min,
                          has_end, day_end_min, w_late, w_end, shift_w)
                - day_obj(vt[k], vr[k], wage_min[k], shift, has_night, night_from,
                          ot_factor, night_factor, has_last, last_delivery_min,
                          has_end, day_end_min, w_late, w_end, shift_w)
            )
            if delta < best_delta - 1e-9 or (
                abs(delta - best_delta) <= 1e-9 and vt[k] < best_time - 1e-9
            ):
                best_delta = delta
                best_time = vt[k]
                best_k = k
        if best_k < 0:
            for k in range(V):
                if fits[i, k]:
                    best_k = k
                    break
        assign[i] = best_k
        vt[best_k] += times[i, best_k]
        vr[best_k] = max(vr[best_k], rets[i, best_k])

    if refine:
        for _ in range(max(1, passes)):
            improved = False
            for i in range(m):
                a = assign[i]
                Ta = vt[a]
                Ra_new = _max_ret(assign, rets, a, i, -1)
                old_max = vt.max()
                best_b = -1
                best_delta = 0.0
                best_new_max = old_max
                for b in range(V):
                    if b == a or not fits[i, b]:
                        continue
                    Tb = vt[b]
                    old = (
                        base[i, a]
                        + pens[i, a]
                        + day_obj(Ta, vr[a], wage_min[a], shift, has_night, night_from,
                                  ot_factor, night_factor, has_last, last_delivery_min,
                                  has_end, day_end_min, w_late, w_end, shift_w)
                        + day_obj(Tb, vr[b], wage_min[b], shift, has_night, night_from,
                                  ot_factor, night_factor, has_last, last_delivery_min,
                                  has_end, day_end_min, w_late, w_end, shift_w)
                    )
                    new = (
                        base[i, b]
                        + pens[i, b]
                        + day_obj(Ta - times[i, a], Ra_new, wage_min[a], shift, has_night,
                                  night_from, ot_factor, night_factor, has_last,
                                  last_delivery_min, has_end, day_end_min,
                                  w_late, w_end, shift_w)
                        + day_obj(Tb + times[i, b], max(vr[b], rets[i, b]), wage_min[b],
                                  shift, has_night, night_from, ot_factor, night_factor,
                                  has_last, last_delivery_min, has_end, day_end_min,
                                  w_late, w_end, shift_w)
                    )
                    delta = new - old
                    trial = vt.copy()
                    trial[a] -= times[i, a]
                    trial[b] += times[i, b]
                    new_max = trial.max()
                    if delta < best_delta - 1e-9 or (
                        abs(delta - best_delta) <= 1e-9 and new_max < best_new_max - 1e-9
                    ):
                        best_delta = delta
                        best_new_max = new_max
                        best_b = b
                if best_b >= 0:
                    vt[a] -= times[i, a]
                    vt[best_b] += times[i, best_b]
                    assign[i] = best_b
                    vr[a] = _max_ret(assign, rets, a, -1, -1)
                    vr[best_b] = _max_ret(assign, rets, best_b, -1, -1)
                    improved = True
            if not improved:
                break

    if swaps and m > 1:
        for _ in range(max(1, passes)):
            improved = False
            for i in range(m):
                a = assign[i]
                for j in range(i + 1, m):
                    b = assign[j]
                    if a == b or not (fits[i, b] and fits[j, a]):
                        continue
                    Ta = vt[a]
                    Tb = vt[b]
                    old = (
                        base[i, a]
                        + base[j, b]
                        + pens[i, a]
                        + pens[j, b]
                        + day_obj(Ta, vr[a], wage_min[a], shift, has_night, night_from,
                                  ot_factor, night_factor, has_last, last_delivery_min,
                                  has_end, day_end_min, w_late, w_end, shift_w)
                        + day_obj(Tb, vr[b], wage_min[b], shift, has_night, night_from,
                                  ot_factor, night_factor, has_last, last_delivery_min,
                                  has_end, day_end_min, w_late, w_end, shift_w)
                    )
                    new = (
                        base[i, b]
                        + base[j, a]
                        + pens[i, b]
                        + pens[j, a]
                        + day_obj(Ta - times[i, a] + times[j, a],
                                  _max_ret(assign, rets, a, i, j), wage_min[a], shift,
                                  has_night, night_from, ot_factor, night_factor,
                                  has_last, last_delivery_min, has_end, day_end_min,
                                  w_late, w_end, shift_w)
                        + day_obj(Tb - times[j, b] + times[i, b],
                                  _max_ret(assign, rets, b, j, i), wage_min[b], shift,
                                  has_night, night_from, ot_factor, night_factor,
                                  has_last, last_delivery_min, has_end, day_end_min,
                                  w_late, w_end, shift_w)
                    )
                    if new < old - 1e-9:
                        vt[a] += times[j, a] - times[i, a]
                        vt[b] += times[i, b] - times[j, b]
                        assign[i] = b
                        assign[j] = a
                        vr[a] = _max_ret(assign, rets, a, -1, -1)
                        vr[b] = _max_ret(assign, rets, b, -1, -1)
                        improved = True
                        a = b
            if not improved:
                break
    return assign


# ---------------------------------------------------------------------------
# 2-opt on one trip (depot -> order -> depot)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _prefix(order, D, fwd, bwd):
    """fwd[k] = sum of forward legs order[0..k]; bwd the same legs reversed."""
    fwd[0] = 0.0
    bwd[0] = 0.0
    for k in range(order.shape[0] - 1):
        fwd[k + 1] = fwd[k] + D[order[k], order[k + 1]]
        bwd[k + 1] = bwd[k] + D[order[k + 1], order[k]]


@njit(cache=True)
def two_opt_directed_kernel(order, D):
    """First-improvement 2-opt for an asymmetric matrix (in place)."""
    n = order.shape[0]
    if n < 3:
        return order
    fwd = np.empty(n)
    bwd = np.empty(n)
    _prefix(order, D, fwd, bwd)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                prev = order[i - 1] if i > 0 else 0
                nxt = order[j + 1] if j + 1 < n else 0
                a = order[i]
                b = order[j]
                inner_old = fwd[j] - fwd[i]
                inner_new = bwd[j] - bwd[i]
                delta = (D[prev, b] + D[a, nxt] + inner_new
                         - D[prev, a] - D[b, nxt] - inner_old)
                if delta < -1e-9:
                    lo = i
                    hi = j
                    while lo < hi:
                        tmp = order[lo]
                        order[lo] = order[hi]
                        order[hi] = tmp
                        lo += 1
                        hi -= 1
                    _prefix(order, D, fwd, bwd)
                    improved = True
    return order


@njit(cache=True)
def two_opt_symmetric_kernel(order, D):
    """First-improvement 2-opt for a symmetric matrix (in place)."""
    n = order.shape[0]
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                prev = order[i - 1] if i > 0 else 0
                nxt = order[j + 1] if j + 1 < n else 0
                a = order[i]
                b = order[j]
                delta = D[prev, b] + D[a, nxt] - D[prev, a] - D[b, nxt]
                if delta < -1e-9:
                    lo = i
                    hi = j
                    while lo < hi:
                        tmp = order[lo]
                        order[lo] = order[hi]
                        order[hi] = tmp
                        lo += 1
                        hi -= 1
                    improved = True
    return order
