"""Daily working window: 08:00 start, 2x overtime for every minute after the
8 h shift, last delivery by 22:00, back at the depot by 23:00.

Synthetic numbers are chosen so every value can be checked by hand:
wage 60 EUR/h = 1 EUR/min, speed 60 km/h = 1 min/km, no fuel or maintenance.
"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from src.cost import DEFAULT_WEIGHTS, day_metrics, segment_stats
from src.data_loader import load_problem
from src.decoder import DecoderConfig, assign_trips, check_solution, decode_tour
from src.pso import nearest_neighbor_tour
from tests.test_cost import base_params, caddy, make_problem

WINDOW = dict(
    day_start="08:00",
    last_delivery_by="22:00",
    hard_day_end="23:00",
)


def easy_vehicle(vehicle_id: int = 1):
    return caddy(
        vehicle_id=vehicle_id,
        max_weight_kg=10_000.0,
        max_volume_m3=100.0,
        fuel_l_per_km=0.0,
        maintenance_per_1000km=0.0,
        avg_speed_kmh=60.0,
        wage_per_h=60.0,
    )


def window_problem(n_vehicles: int = 1, loading_min: float = 0.0, d=None):
    if d is None:
        d = np.zeros((4, 4))
    n = len(d)
    params = base_params(
        num_customers=n - 1,
        min_customers_per_trip=1,
        required_free_volume_ratio=0.0,
        **WINDOW,
    )
    return make_problem(
        d,
        weight=[0.0] + [1.0] * (n - 1),
        volume=[0.0] + [0.1] * (n - 1),
        loading=[0.0] + [loading_min] * (n - 1),
        unloading=[0.0] * n,
        params=params,
        vehicles=tuple(easy_vehicle(i + 1) for i in range(n_vehicles)),
    )


class TestDayMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.data = window_problem()

    def test_clock_conversion(self) -> None:
        p = self.data.params
        self.assertIsNone(p.night_from_min)          # no separate night rate
        self.assertEqual(p.last_delivery_min, 840.0)  # 22:00 is 14 h after 08:00
        self.assertEqual(p.day_end_min, 900.0)      # 23:00 is 15 h after 08:00
        self.assertEqual(p.clock(905.0), "23:05")

    def test_overtime_only(self) -> None:
        # 10 h day ends 18:00: 2 h at 2x -> premium 120 min x 1 EUR x (2-1).
        dm = day_metrics(self.data, 0, 600.0, max_return_min=30.0)
        self.assertAlmostEqual(dm.overtime_min, 120.0)
        self.assertAlmostEqual(dm.night_min, 0.0)
        self.assertAlmostEqual(dm.extra_cost, 120.0)
        self.assertTrue(dm.feasible)

    def test_after_22_is_still_2x(self) -> None:
        # 14.5 h day, last trip returns in 30 min: last delivery exactly 22:00,
        # back 22:30. All 390 min after 16:00 are overtime at 2x (+1x premium).
        dm = day_metrics(self.data, 0, 870.0, max_return_min=30.0)
        self.assertAlmostEqual(dm.overtime_min, 390.0)
        self.assertAlmostEqual(dm.night_min, 0.0)
        self.assertAlmostEqual(dm.extra_cost, 390.0)
        self.assertAlmostEqual(dm.last_delivery_min, 840.0)
        self.assertTrue(dm.feasible)

    def test_late_delivery_is_penalised(self) -> None:
        # Same day, but the return takes 20 min: last delivery 22:10.
        dm = day_metrics(self.data, 0, 870.0, max_return_min=20.0)
        self.assertAlmostEqual(dm.late_delivery_min, 10.0)
        self.assertAlmostEqual(
            dm.penalty, 10.0 * DEFAULT_WEIGHTS.per_minute_late_delivery
        )
        self.assertFalse(dm.feasible)

    def test_back_after_23_is_penalised(self) -> None:
        # Back at 23:10, last delivery 22:00: only the hard end is broken.
        dm = day_metrics(self.data, 0, 910.0, max_return_min=70.0)
        self.assertAlmostEqual(dm.late_delivery_min, 0.0)
        self.assertAlmostEqual(dm.past_day_end_min, 10.0)
        self.assertAlmostEqual(
            dm.penalty, 10.0 * DEFAULT_WEIGHTS.per_minute_past_day_end
        )

    def test_penalty_never_pays(self) -> None:
        # One minute late costs more than any wage saving it could buy.
        wage_per_min_overtime = 60.0 / 60.0 * 2.0
        self.assertGreater(DEFAULT_WEIGHTS.per_minute_late_delivery, 100 * wage_per_min_overtime)
        self.assertGreater(DEFAULT_WEIGHTS.per_minute_past_day_end, 100 * wage_per_min_overtime)

    def test_invalid_window_rejected(self) -> None:
        with self.assertRaises(ValueError):
            base_params(**{**WINDOW, "last_delivery_by": "23:30"}).validate_day_window()


class TestAssignmentRespectsWindow(unittest.TestCase):
    def test_third_trip_moves_to_second_vehicle(self) -> None:
        # Three 6 h trips. One vehicle would need 18 h (past 23:00), so the
        # assignment must split them over the two vehicles.
        data = window_problem(n_vehicles=2, loading_min=360.0)
        segs = [segment_stats(data, [c]) for c in (1, 2, 3)]
        sol = assign_trips(data, segs, DecoderConfig(), method="test")
        self.assertTrue(sol.feasible)
        self.assertLessEqual(sol.longest_day_min, 900.0)
        self.assertEqual(sorted(np.round(sol.vehicle_total_min)), [360.0, 720.0])
        self.assertEqual(check_solution(data, sol), [])

    def test_longest_return_runs_last(self) -> None:
        # Customer 1 is 10 km from the depot, customer 2 is 60 km. The trip to
        # customer 2 runs last, so the last delivery is 60 min before the end.
        d = np.array(
            [[0, 10, 60, 1], [10, 0, 50, 1], [60, 50, 0, 1], [1, 1, 1, 0]], float
        )
        data = window_problem(n_vehicles=1, loading_min=300.0, d=d)
        segs = [segment_stats(data, [2]), segment_stats(data, [1])]
        sol = assign_trips(data, segs, DecoderConfig(), method="test")
        self.assertEqual(sol.trips[-1].customers, (2,))
        total = 300.0 + 20.0 + 300.0 + 120.0
        self.assertAlmostEqual(sol.vehicle_total_min[0], total)
        self.assertAlmostEqual(sol.vehicle_last_delivery_min[0], total - 60.0)


class TestRealDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_problem("data/base")

    def test_dataset_has_window(self) -> None:
        p = self.data.params
        self.assertIsNone(p.night_multiplier)  # overtime is 2x all the way
        self.assertEqual(p.day_start, "08:00")
        self.assertEqual(p.last_delivery_by, "22:00")
        self.assertEqual(p.hard_day_end, "23:00")

    def test_penalty_matches_checker(self) -> None:
        # A naive nearest-neighbour tour may break the window; whatever the
        # decoder reports must agree with the independent check.
        sol = decode_tour(self.data, nearest_neighbor_tour(self.data), DecoderConfig())
        errors = check_solution(self.data, sol)
        self.assertEqual(sol.feasible, errors == [])

    def test_without_window_matches_original_model(self) -> None:
        params = dataclasses.replace(
            self.data.params,
            night_rate_from=None,
            night_multiplier=None,
            last_delivery_by=None,
            hard_day_end=None,
        )
        data = dataclasses.replace(self.data, params=params)
        sol = decode_tour(data, nearest_neighbor_tour(data), DecoderConfig())
        self.assertTrue(sol.feasible)
        self.assertEqual(check_solution(data, sol), [])


if __name__ == "__main__":
    unittest.main()
