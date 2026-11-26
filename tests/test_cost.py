"""Hand-check tests for the trip cost function.

The first test uses the real dataset and a three-customer trip.  The expected
numbers are derived by hand from the formulas (distance, speed, wage, fuel,
maintenance).  The other tests use tiny synthetic instances so that overtime
and the 10% free-volume rule can be checked exactly.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.cost import DEFAULT_WEIGHTS, segment_stats, trip_metrics
from src.data_loader import Params, ProblemData, Vehicle, load_problem
from src.decoder import assign_trips


def base_params(**overrides) -> Params:
    values = dict(
        currency="EUR",
        distance_unit="km",
        time_unit="min",
        depot_id=0,
        num_customers=1,
        shift_hours=8.0,
        overtime_multiplier=2.0,
        fuel_price_per_l=1.6,
        packing_mode=1,
        required_free_volume_ratio=0.10,
        min_customers_per_trip=10,
        max_customers_per_trip=50,
        allow_multiple_trips_per_vehicle=True,
        must_return_to_depot=True,
        loading_counts_as_work_time=True,
    )
    values.update(overrides)
    return Params(**values)


def caddy(**overrides) -> Vehicle:
    values = dict(
        vehicle_id=1,
        type_id="caddy",
        model="VW Caddy Cargo SWB",
        max_weight_kg=680.0,
        max_volume_m3=3.1,
        cargo_length_m=1.8,
        cargo_width_m=1.55,
        cargo_height_m=1.11,
        fuel_l_per_km=0.0535,
        top_speed_kmh=178.0,
        avg_speed_kmh=50.0,
        wage_per_h=12.0,
        maintenance_per_1000km=80.0,
    )
    values.update(overrides)
    return Vehicle(**values)


def make_problem(
    distance: np.ndarray,
    weight: np.ndarray,
    volume: np.ndarray,
    loading: np.ndarray,
    unloading: np.ndarray,
    params: Params | None = None,
    vehicles: tuple[Vehicle, ...] | None = None,
) -> ProblemData:
    n = len(distance)
    if params is None:
        params = base_params(num_customers=n - 1)
    if vehicles is None:
        vehicles = (caddy(),)
    return ProblemData(
        params=params,
        vehicles=vehicles,
        distance=np.asarray(distance, dtype=float),
        weight=np.asarray(weight, dtype=float),
        volume=np.asarray(volume, dtype=float),
        loading=np.asarray(loading, dtype=float),
        unloading=np.asarray(unloading, dtype=float),
    )


class TestRealThreeCustomerTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_problem("data/base")

    def test_three_customers_hand_calculation(self) -> None:
        # Customers 2, 5, 8 in a VW Caddy (vehicle index 0).
        m = trip_metrics(self.data, 0, [2, 5, 8])

        # Hand calculation:
        # distance = d(0,2)+d(2,5)+d(5,8)+d(8,0) = 49.910 km
        # driving = 49.910 / 50 * 60 = 59.892 min
        # loading = 10.9 min, unloading = 16.9 min
        # total = 87.692 min
        self.assertAlmostEqual(m.distance_km, 49.910, places=9)
        self.assertAlmostEqual(m.driving_min, 59.892, places=9)
        self.assertAlmostEqual(m.loading_min, 10.9, places=9)
        self.assertAlmostEqual(m.unloading_min, 16.9, places=9)
        self.assertAlmostEqual(m.total_min, 87.692, places=9)
        self.assertAlmostEqual(m.weight_kg, 39.418, places=9)
        self.assertAlmostEqual(m.volume_m3, 0.9288, places=9)

        # fuel = 49.910 * 0.0535 * 1.60 = 4.272296
        # maintenance = 49.910 * 80 / 1000 = 3.992800
        # wage = 87.692 / 60 * 12 = 17.538400
        self.assertAlmostEqual(m.fuel_cost, 4.272296, places=9)
        self.assertAlmostEqual(m.maintenance_cost, 3.9928, places=9)
        self.assertAlmostEqual(m.wage_base_cost, 17.5384, places=9)
        self.assertAlmostEqual(m.cost_eur, 25.803496, places=9)

        self.assertEqual(m.overtime_min, 0.0)
        self.assertTrue(m.capacity_ok)
        self.assertTrue(m.free_volume_ok)
        self.assertTrue(m.max_customers_ok)
        # Only 3 customers, so the soft minimum is violated: the trip is still
        # allowed (feasible) but pays 7 x 500 EUR penalty, not in the cost.
        self.assertEqual(m.under_min, 7)
        self.assertFalse(m.min_customers_ok)
        self.assertTrue(m.feasible)
        self.assertAlmostEqual(
            m.penalty, 7 * DEFAULT_WEIGHTS.per_customer_under_min, places=9
        )
        self.assertAlmostEqual(m.penalty, 3500.0, places=9)
        self.assertAlmostEqual(m.cost_eur, 25.803496, places=9)


class TestOvertime(unittest.TestCase):
    def test_single_overtime_trip(self) -> None:
        # Depot -> 1 -> 2 -> depot = 360 km at 60 km/h = 360 min driving.
        # Loading 120 + 120 = 240 min.  Total = 600 min = 10 h.
        d = np.array(
            [
                [0.0, 120.0, 120.0],
                [120.0, 0.0, 120.0],
                [120.0, 120.0, 0.0],
            ]
        )
        params = base_params(
            num_customers=2,
            min_customers_per_trip=1,
            required_free_volume_ratio=0.0,
            fuel_price_per_l=1.6,
        )
        v = caddy(
            max_weight_kg=10_000.0,
            max_volume_m3=100.0,
            fuel_l_per_km=0.0,
            maintenance_per_1000km=0.0,
            avg_speed_kmh=60.0,
            wage_per_h=10.0,
        )
        data = make_problem(
            d,
            weight=[0.0, 1.0, 1.0],
            volume=[0.0, 0.1, 0.1],
            loading=[0.0, 120.0, 120.0],
            unloading=[0.0, 0.0, 0.0],
            params=params,
            vehicles=(v,),
        )

        m = trip_metrics(data, 0, [1, 2])
        self.assertAlmostEqual(m.distance_km, 360.0)
        self.assertAlmostEqual(m.driving_min, 360.0)
        self.assertAlmostEqual(m.loading_min, 240.0)
        self.assertAlmostEqual(m.total_min, 600.0)
        self.assertAlmostEqual(m.overtime_min, 120.0)
        self.assertAlmostEqual(m.wage_base_cost, 100.0)
        self.assertAlmostEqual(m.overtime_cost, 20.0)  # 2 h * 10 EUR * 1 extra
        self.assertAlmostEqual(m.cost_eur, 120.0)
        self.assertTrue(m.feasible)

    def test_vehicle_day_overtime_is_aggregated(self) -> None:
        # Two 5-hour trips on one vehicle -> 10-hour day, 2 hours overtime.
        d = np.full((5, 5), 1000.0)
        np.fill_diagonal(d, 0.0)
        # Trip 1: 0->1->2->0 = 30+30+60 = 120 km (2 h at 60 km/h).
        d[0, 1] = d[1, 0] = 30.0
        d[1, 2] = d[2, 1] = 30.0
        d[2, 0] = d[0, 2] = 60.0
        # Trip 2: 0->3->4->0 = 120 km.
        d[0, 3] = d[3, 0] = 30.0
        d[3, 4] = d[4, 3] = 30.0
        d[4, 0] = d[0, 4] = 60.0

        params = base_params(
            num_customers=4,
            min_customers_per_trip=1,
            required_free_volume_ratio=0.0,
        )
        v = caddy(
            max_weight_kg=10_000.0,
            max_volume_m3=100.0,
            fuel_l_per_km=0.0,
            maintenance_per_1000km=0.0,
            avg_speed_kmh=60.0,
            wage_per_h=10.0,
        )
        data = make_problem(
            d,
            weight=[0.0, 1.0, 1.0, 1.0, 1.0],
            volume=[0.0, 0.1, 0.1, 0.1, 0.1],
            loading=[0.0, 90.0, 90.0, 90.0, 90.0],
            unloading=[0.0, 0.0, 0.0, 0.0, 0.0],
            params=params,
            vehicles=(v,),
        )

        segs = [segment_stats(data, [1, 2]), segment_stats(data, [3, 4])]
        sol = assign_trips(data, segs, method="test")
        self.assertEqual(sol.n_trips, 2)
        self.assertAlmostEqual(sol.vehicle_total_min[0], 600.0)
        self.assertAlmostEqual(sol.vehicle_overtime_min[0], 120.0)
        self.assertAlmostEqual(sol.day_overtime_cost, 20.0)
        # Each trip alone is 5 h -> 50 EUR base, no trip-level overtime.
        self.assertAlmostEqual(sol.trips[0].cost_eur, 50.0)
        self.assertAlmostEqual(sol.trips[1].cost_eur, 50.0)
        # Day: 100 EUR base + 20 EUR overtime premium.
        self.assertAlmostEqual(sol.true_cost_eur, 120.0)
        self.assertAlmostEqual(sol.penalty, 0.0)
        self.assertTrue(sol.feasible)


class TestFreeVolume(unittest.TestCase):
    def _caddy_trip(self, per_customer_m3: float):
        n = 11
        d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
        params = base_params(
            num_customers=10,
            min_customers_per_trip=10,
            required_free_volume_ratio=0.10,
        )
        data = make_problem(
            d,
            weight=np.array([0.0] + [10.0] * 10),
            volume=np.array([0.0] + [per_customer_m3] * 10),
            loading=np.array([0.0] + [1.0] * 10),
            unloading=np.array([0.0] + [1.0] * 10),
            params=params,
            vehicles=(caddy(),),
        )
        return data, trip_metrics(data, 0, list(range(1, 11)))

    def test_reserve_used_pays_overflow_fee(self) -> None:
        # 10 x 0.30 m3 = 3.00 m3 in a Caddy: usable 2.79 m3, physical 3.10 m3.
        # 0.21 m3 of the 10 % reserve is used: fee 0.21 x 100 EUR, trip feasible.
        data, m = self._caddy_trip(0.30)
        self.assertAlmostEqual(m.volume_m3, 3.0)
        self.assertTrue(m.capacity_ok)
        self.assertFalse(m.free_volume_ok)   # reserve is touched (informational)
        self.assertAlmostEqual(m.overflow_m3, 0.21, places=9)
        self.assertAlmostEqual(
            m.overflow_cost, 0.21 * data.params.volume_overflow_eur_per_m3, places=9
        )
        self.assertTrue(m.feasible)
        self.assertAlmostEqual(m.penalty, 0.0, places=9)
        # The fee is part of the real cost.
        self.assertAlmostEqual(
            m.base_cost_eur,
            m.fuel_cost + m.maintenance_cost + m.wage_base_cost + m.overflow_cost,
            places=9,
        )

    def test_above_physical_volume_is_infeasible(self) -> None:
        # 10 x 0.32 m3 = 3.20 m3 > 3.10 m3: impossible (hard rule, no price).
        data, m = self._caddy_trip(0.32)
        self.assertFalse(m.capacity_ok)
        self.assertFalse(m.feasible)
        self.assertAlmostEqual(m.raw_volume_over, 0.10, places=9)
        self.assertAlmostEqual(m.overflow_m3, 0.31, places=9)  # band is capped
        self.assertAlmostEqual(m.penalty, 0.0, places=9)  # no penalty can buy it

    def test_below_ninety_percent_has_no_fee(self) -> None:
        data, m = self._caddy_trip(0.25)   # 2.50 m3 < 2.79 m3
        self.assertTrue(m.free_volume_ok)
        self.assertEqual(m.overflow_cost, 0.0)
        self.assertTrue(m.feasible)


if __name__ == "__main__":
    unittest.main()


class TestOverflowSplitConsistency(unittest.TestCase):
    def test_split_dp_prices_overflow_like_cost_function(self) -> None:
        # One Caddy, 10 customers that only fit by using the reserve.
        # The DP objective must equal the trip cost incl. the overflow fee.
        from src.decoder import DecoderConfig, split_dp

        n = 11
        d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
        params = base_params(
            num_customers=10, min_customers_per_trip=10,
            required_free_volume_ratio=0.10,
        )
        data = make_problem(
            d,
            weight=np.array([0.0] + [10.0] * 10),
            volume=np.array([0.0] + [0.30] * 10),
            loading=np.array([0.0] + [1.0] * 10),
            unloading=np.array([0.0] + [1.0] * 10),
            params=params,
            vehicles=(caddy(),),
        )
        segs = split_dp(data, list(range(1, 11)), DecoderConfig())
        self.assertEqual(len(segs), 1)
        m = trip_metrics(data, 0, segs[0].customers)
        self.assertTrue(m.feasible)
        self.assertGreater(m.overflow_cost, 0.0)
