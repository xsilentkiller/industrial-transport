"""Hard and soft constraint rules.

Hard (impossible, no penalty can buy them):
* weight above the vehicle payload
* volume above 100 % of the cargo space

Soft (allowed, but each violating customer pays a penalty that is never
added to the true EUR cost):
* fewer than ``min_customers_per_trip`` customers: 500 EUR per missing one
* more than ``max_customers_per_trip`` customers: 100 EUR per extra one
"""

from __future__ import annotations

import unittest

import numpy as np

from src.cost import DEFAULT_WEIGHTS, segment_stats
from src.data_loader import load_problem
from src.decoder import (
    DecoderConfig,
    assign_trips,
    check_solution,
    decode_tour,
    make_chunk_solution,
    split_dp,
)
from tests.test_cost import base_params, caddy, make_problem


def line_distances(n_customers: int, depot_km: float, step_km: float) -> np.ndarray:
    """Depot ``depot_km`` from every customer; customers ``step_km`` apart."""
    n = n_customers + 1
    d = np.zeros((n, n))
    for i in range(1, n):
        d[0, i] = d[i, 0] = depot_km
        for j in range(1, n):
            d[i, j] = abs(i - j) * step_km
    return d


class TestPenaltyValues(unittest.TestCase):
    def test_default_weights(self) -> None:
        self.assertEqual(DEFAULT_WEIGHTS.per_customer_under_min, 500.0)
        self.assertEqual(DEFAULT_WEIGHTS.per_customer_over_max, 100.0)
        # Capacity has no price at all.
        self.assertFalse(hasattr(DEFAULT_WEIGHTS, "per_kg_over"))
        self.assertFalse(hasattr(DEFAULT_WEIGHTS, "per_m3_over"))


class TestHardCapacity(unittest.TestCase):
    def _two_vehicle_problem(self, weight: list[float], volume: list[float]):
        """A very cheap small van and a very expensive large one."""
        small = caddy(vehicle_id=1, wage_per_h=1.0, fuel_l_per_km=0.0,
                      maintenance_per_1000km=0.0)
        large = caddy(vehicle_id=2, type_id="big", max_weight_kg=5000.0,
                      max_volume_m3=40.0, wage_per_h=1000.0)
        n = len(weight) - 1
        return make_problem(
            line_distances(n, 10.0, 1.0),
            weight=weight,
            volume=volume,
            loading=[0.0] + [1.0] * n,
            unloading=[0.0] + [1.0] * n,
            params=base_params(num_customers=n, min_customers_per_trip=1),
            vehicles=(small, large),
        )

    def test_volume_over_100_percent_never_assigned(self) -> None:
        # 2 x 2.0 m3 = 4.0 m3 > 3.1 m3 of the cheap van.
        data = self._two_vehicle_problem([0.0, 10.0, 10.0], [0.0, 2.0, 2.0])
        sol = assign_trips(data, [segment_stats(data, [1, 2])])
        self.assertEqual(sol.trips[0].vehicle_id, 2)  # the expensive one
        self.assertTrue(sol.feasible)
        self.assertEqual(check_solution(data, sol), [])

    def test_weight_over_payload_never_assigned(self) -> None:
        # 2 x 400 kg = 800 kg > 680 kg payload of the cheap van.
        data = self._two_vehicle_problem([0.0, 400.0, 400.0], [0.0, 0.1, 0.1])
        sol = assign_trips(data, [segment_stats(data, [1, 2])])
        self.assertEqual(sol.trips[0].vehicle_id, 2)
        self.assertTrue(sol.feasible)

    def test_trip_that_fits_no_vehicle_is_rejected(self) -> None:
        data = self._two_vehicle_problem(
            [0.0, 3000.0, 3000.0], [0.0, 1.0, 1.0]   # 6000 kg > 5000 kg
        )
        with self.assertRaises(ValueError):
            assign_trips(data, [segment_stats(data, [1, 2])])

    def test_split_never_builds_an_oversized_trip(self) -> None:
        # One van, depot 100 km away: one big trip would be far cheaper than
        # two, but 4 x 1.0 m3 = 4.0 m3 does not fit 3.1 m3.
        n = 4
        data = make_problem(
            line_distances(n, 100.0, 1.0),
            weight=[0.0] + [10.0] * n,
            volume=[0.0] + [1.0] * n,
            loading=[0.0] + [1.0] * n,
            unloading=[0.0] + [1.0] * n,
            params=base_params(num_customers=n, min_customers_per_trip=1),
        )
        segs = split_dp(data, [1, 2, 3, 4])
        self.assertGreaterEqual(len(segs), 2)
        for s in segs:
            self.assertLessEqual(s.volume_m3, 3.1 + 1e-9)
        sol = decode_tour(data, [1, 2, 3, 4])
        self.assertTrue(sol.feasible)
        self.assertTrue(all(t.metrics.capacity_ok for t in sol.trips))

    def test_chunk_seed_respects_capacity(self) -> None:
        n = 6
        data = make_problem(
            line_distances(n, 10.0, 1.0),
            weight=[0.0] + [10.0] * n,
            volume=[0.0] + [1.0] * n,          # only 3 fit in one van
            loading=[0.0] + [1.0] * n,
            unloading=[0.0] + [1.0] * n,
            params=base_params(num_customers=n, min_customers_per_trip=1),
        )
        sol = make_chunk_solution(data, list(range(1, n + 1)), chunk_size=10)
        self.assertTrue(all(t.metrics.capacity_ok for t in sol.trips))

    def test_loader_rejects_an_order_that_fits_no_vehicle(self) -> None:
        with self.assertRaises(ValueError):
            make_problem(
                line_distances(1, 10.0, 1.0),
                weight=[0.0, 10.0],
                volume=[0.0, 5.0],             # > 3.1 m3 of the only van
                loading=[0.0, 1.0],
                unloading=[0.0, 1.0],
                params=base_params(num_customers=1),
            )

    def test_real_data_never_breaks_capacity(self) -> None:
        data = load_problem("data/base")
        rng = np.random.default_rng(7)
        for _ in range(3):
            tour = (rng.permutation(data.n_customers) + 1).tolist()
            sol = decode_tour(data, tour)
            for t in sol.trips:
                k = t.vehicle_index
                self.assertLessEqual(t.metrics.weight_kg, data.vehicle_weight_cap[k] + 1e-9)
                self.assertLessEqual(t.metrics.volume_m3, data.vehicle_volume_cap[k] + 1e-9)


class TestSoftCustomerCounts(unittest.TestCase):
    def _far_depot_problem(self, n: int, max_customers: int, depot_km: float = 100.0):
        """Depot far away and an expensive return: one trip is best."""
        v = caddy(fuel_l_per_km=1.0, maintenance_per_1000km=0.0)
        return make_problem(
            line_distances(n, depot_km, 0.5),
            weight=[0.0] + [1.0] * n,
            volume=[0.0] + [0.01] * n,
            loading=[0.0] + [1.0] * n,
            unloading=[0.0] + [1.0] * n,
            params=base_params(
                num_customers=n,
                min_customers_per_trip=1,
                max_customers_per_trip=max_customers,
            ),
            vehicles=(v,),
        )

    def test_trip_may_exceed_max_and_pays_100_per_customer(self) -> None:
        # 4 customers, max 2: two extra customers cost 2 x 100 = 200 EUR of
        # penalty, while a second depot round trip costs ~200 km of fuel.
        data = self._far_depot_problem(4, max_customers=2)
        sol = decode_tour(data, [1, 2, 3, 4])
        self.assertEqual(sol.n_trips, 1)
        self.assertEqual(sol.trips[0].metrics.over_max, 2)
        self.assertAlmostEqual(sol.soft_penalty, 200.0)
        self.assertTrue(sol.feasible)              # soft rule: still legal
        # The penalty is never part of the real cost.
        self.assertAlmostEqual(sol.objective, sol.true_cost_eur + 200.0)
        self.assertAlmostEqual(
            sol.true_cost_eur, sum(t.metrics.base_cost_eur for t in sol.trips)
            + sol.day_overtime_cost,
        )
        self.assertEqual(check_solution(data, sol), [])
        self.assertTrue(check_solution(data, sol, require_max_customers=True))

    def test_split_searches_past_the_max(self) -> None:
        # 10 extra customers = 1000 EUR penalty; a second depot round trip
        # (2 x 500 km at 1.6 EUR/km fuel) costs more, so one trip wins.
        # The old DP never even tried a trip longer than 50.
        n = 60
        data = self._far_depot_problem(n, max_customers=50, depot_km=500.0)
        segs = split_dp(data, list(range(1, n + 1)), DecoderConfig())
        self.assertEqual(len(segs), 1)            # 60 > 50 is allowed
        self.assertEqual(segs[0].count, 60)

    def test_trip_below_min_is_allowed_with_500_penalty(self) -> None:
        data = make_problem(
            line_distances(3, 10.0, 1.0),
            weight=[0.0, 1.0, 1.0, 1.0],
            volume=[0.0, 0.1, 0.1, 0.1],
            loading=[0.0, 1.0, 1.0, 1.0],
            unloading=[0.0, 1.0, 1.0, 1.0],
            params=base_params(num_customers=3, min_customers_per_trip=10),
        )
        sol = assign_trips(data, [segment_stats(data, [1, 2, 3])])
        self.assertAlmostEqual(sol.soft_penalty, 7 * 500.0)
        self.assertAlmostEqual(sol.penalty, sol.soft_penalty)
        self.assertTrue(sol.feasible)
        self.assertAlmostEqual(sol.objective - sol.true_cost_eur, 3500.0)


if __name__ == "__main__":
    unittest.main()
