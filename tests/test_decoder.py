"""Decoder / Split-DP / assignment checks."""

from __future__ import annotations

import dataclasses
import time
import unittest

import numpy as np

from src.data_loader import load_problem
from src.decoder import (
    DecoderConfig,
    check_solution,
    decode_tour,
    greedy_cut,
    keys_to_tour,
    split_dp,
    tour_to_keys,
)
from src.pso import nearest_neighbor_tour


class TestDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_problem("data/base")

    def assert_permutation(self, customers: list[int], n: int) -> None:
        self.assertEqual(len(customers), n)
        self.assertEqual(len(set(customers)), n)
        self.assertEqual(sorted(customers), list(range(1, n + 1)))

    def test_keys_roundtrip(self) -> None:
        rng = np.random.default_rng(7)
        keys = rng.random(self.data.n_customers)
        tour = keys_to_tour(keys)
        self.assert_permutation(tour, self.data.n_customers)
        self.assertEqual(keys_to_tour(tour_to_keys(tour)), tour)

    def test_first_20_customers(self) -> None:
        tour = list(range(1, 21))
        cfg = DecoderConfig(enforce_min_customers=True, use_greedy_baseline=False)
        sol = decode_tour(self.data, tour, cfg)

        seen = [c for t in sol.trips for c in t.customers]
        self.assert_permutation(seen, 20)
        self.assertGreaterEqual(sol.n_trips, 1)
        for t in sol.trips:
            self.assertGreaterEqual(len(t.customers), 10)
            self.assertLessEqual(len(t.customers), 50)
            self.assertTrue(t.metrics.capacity_ok, t.metrics.violations)
            self.assertTrue(t.metrics.max_customers_ok)
        self.assertTrue(sol.feasible)
        self.assertAlmostEqual(sol.penalty, 0.0)
        print("\nFirst-20 decoder result:")
        for t in sol.trips:
            print(
                f"  vehicle {t.vehicle_id}: {len(t.customers)} customers, "
                f"{t.time_min:.1f} min, EUR {t.cost_eur:.2f}, "
                f"{t.metrics.weight_kg:.1f} kg, {t.metrics.volume_m3:.3f} m3"
            )

    def test_greedy_cut_covers_all(self) -> None:
        tour = nearest_neighbor_tour(self.data)
        segs = greedy_cut(self.data, tour)
        seen = [c for s in segs for c in s.customers]
        self.assert_permutation(seen, self.data.n_customers)
        for s in segs:
            self.assertLessEqual(s.count, 50)
            self.assertTrue(
                np.any(
                    (s.weight_kg <= self.data.vehicle_weight_cap + 1e-9)
                    & (s.volume_m3 <= self.data.vehicle_volume_cap + 1e-9)
                )
            )

    def test_split_dp_covers_all(self) -> None:
        tour = nearest_neighbor_tour(self.data)
        segs = split_dp(self.data, tour)
        seen = [c for s in segs for c in s.customers]
        self.assert_permutation(seen, self.data.n_customers)
        for s in segs:
            self.assertLessEqual(s.count, 50)

    def test_full_250_and_timing(self) -> None:
        # A single nearest-neighbour tour is not expected to fit the daily
        # window (see test_day_window.py); here the decoder is checked on the
        # original model without the window.
        data = dataclasses.replace(
            self.data,
            params=dataclasses.replace(
                self.data.params,
                night_rate_from=None,
                night_multiplier=None,
                last_delivery_by=None,
                hard_day_end=None,
            ),
        )
        tour = nearest_neighbor_tour(data)
        cfg = DecoderConfig(use_greedy_baseline=True)

        sol = decode_tour(data, tour, cfg)
        errors = check_solution(data, sol)
        self.assertEqual(errors, [])
        self.assert_permutation(
            [c for t in sol.trips for c in t.customers], self.data.n_customers
        )
        self.assertTrue(sol.feasible)
        self.assertAlmostEqual(sol.penalty, 0.0)
        self.assertTrue(all(len(t.customers) <= 50 for t in sol.trips))
        self.assertTrue(all(t.metrics.capacity_ok for t in sol.trips))

        # Time the full 250-customer decoder (same fixed tour).
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            decode_tour(self.data, tour, cfg)
            times.append(time.perf_counter() - t0)
        print(
            f"\nDecoder timing on 250 customers: "
            f"mean {np.mean(times) * 1000:.1f} ms, "
            f"median {np.median(times) * 1000:.1f} ms, "
            f"min {np.min(times) * 1000:.1f} ms"
        )
        print(
            f"Solution: {sol.n_trips} trips, EUR {sol.true_cost_eur:.2f}, "
            f"{sol.total_distance_km:.1f} km, {sol.total_time_min / 60:.1f} h, "
            f"overtime {sol.shift_over_min / 60:.1f} h"
        )

    def test_solution_format(self) -> None:
        tour = nearest_neighbor_tour(self.data)
        sol = decode_tour(self.data, tour)
        self.assertGreater(len(sol.trips), 0)
        trip = sol.trips[0]
        self.assertIsInstance(trip.vehicle_id, int)
        self.assertIsInstance(trip.customers, tuple)
        self.assertGreater(trip.time_min, 0.0)
        self.assertGreater(trip.cost_eur, 0.0)
        # The wrapper is iterable like a list of trips.
        self.assertEqual(len(list(sol)), len(sol.trips))


if __name__ == "__main__":
    unittest.main()
