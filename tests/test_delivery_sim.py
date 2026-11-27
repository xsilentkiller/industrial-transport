"""Delivery outcomes: stop schedule, simulated day, Monte Carlo (src/delivery_sim.py)."""

from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.data_loader import load_problem
from src.decoder import decode_tour
from src.delivery_sim import (
    SimConfig,
    build_simulation,
    items_text,
    load_addresses,
    monte_carlo,
    simulate_day,
    stop_schedule,
)
from src.pso import volume_aware_nearest_neighbor_tour


def base_solution(data):
    return decode_tour(data, volume_aware_nearest_neighbor_tour(data)).as_dict()


class TestStopSchedule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_problem("data/base")
        cls.solution = base_solution(cls.data)
        cls.rows = stop_schedule(cls.data, cls.solution)

    def test_trip_end_matches_trip_time(self):
        trips = {}
        for r in self.rows:
            trips[(r["vehicle_index"], r["trip"])] = (r["trip_start_min"], r["trip_end_min"])
        per_vehicle: dict[int, list[float]] = {}
        for t in self.solution["trips"]:
            per_vehicle.setdefault(t["vehicle_index"], []).append(t["time_min"])
        for k, times in per_vehicle.items():
            for n, time_min in enumerate(times, start=1):
                start, end = trips[(k, n)]
                self.assertAlmostEqual(end - start, time_min, delta=1e-6)

    def test_one_row_per_customer(self):
        n = self.data.n_customers
        self.assertEqual(sorted(r["location_id"] for r in self.rows), list(range(1, n + 1)))
        for r in self.rows:
            self.assertLessEqual(r["arrive_min"], r["leave_min"])
        by_trip: dict[tuple, list[int]] = {}
        for r in self.rows:
            by_trip.setdefault((r["vehicle_index"], r["trip"]), []).append(r["stop"])
        for stops in by_trip.values():
            self.assertEqual(stops, list(range(1, len(stops) + 1)))

    def test_unused_vehicle_has_no_rows(self):
        sol = dict(self.solution)
        sol["trips"] = [dict(t, vehicle_index=4, vehicle_id=self.data.vehicles[4].vehicle_id)
                        for t in self.solution["trips"]]
        rows = stop_schedule(self.data, sol)
        self.assertTrue(rows)
        self.assertTrue(all(r["vehicle_index"] == 4 for r in rows))

    def test_items_text_counts_duplicates(self):
        names = {f.item_id: f.name for f in self.data.furniture}
        item_ids = list(self.data.item_ids)
        item_ids[1] = ("fid002", "fid002", "fid001")
        data = replace(self.data, item_ids=tuple(item_ids))
        self.assertEqual(items_text(data, 1), f"{names['fid002']} ×2, {names['fid001']}")

    def test_load_addresses_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "locations.csv").write_text("location_id,lat,lon\n0,47.4,19.1\n7,47.5,19.0\n",
                                                  encoding="utf-8")
            self.assertEqual(load_addresses(tmp), {})
        self.assertEqual(load_addresses("data/osm")[1],
                         {"address": "Tímár utca 1, 1036 Budapest, Hungary", "district": "3"})


class TestSimulatedDay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_problem("data/base")
        cls.solution = base_solution(cls.data)
        cls.addresses = load_addresses("data/base")
        cls.n = cls.data.n_customers

    def sim(self, p_not_home, p_refused, seed=42):
        cfg = SimConfig(p_not_home=p_not_home, p_refused=p_refused, seed=seed)
        return simulate_day(self.data, self.solution, self.addresses, cfg)

    def test_zero_probability_all_delivered(self):
        day = self.sim(0.0, 0.0)
        self.assertEqual(day["counts"]["delivered"], day["counts"]["total"])
        self.assertEqual(day["counts"]["total"], self.n)
        self.assertEqual(day["redelivery"], [])
        self.assertEqual(day["returned_to_stock"], [])
        self.assertEqual(day["depot_returns"], [])

    def test_all_not_home(self):
        day = self.sim(1.0, 0.0)
        self.assertTrue(all(r["status"] == "attempted" for r in day["records"]))
        self.assertEqual(len(day["redelivery"]), self.n)
        self.assertEqual(len(day["depot_returns"]), len(self.solution["trips"]))

    def test_times_identical_for_any_outcomes(self):
        a, b = self.sim(0.0, 0.0), self.sim(0.5, 0.3)
        self.assertEqual([(r["arrive"], r["leave"]) for r in a["records"]],
                         [(r["arrive"], r["leave"]) for r in b["records"]])

    def test_same_seed_same_result(self):
        self.assertEqual(self.sim(0.3, 0.2, seed=7)["records"], self.sim(0.3, 0.2, seed=7)["records"])

    def test_different_seed_differs(self):
        a = [r["status"] for r in self.sim(0.3, 0.2, seed=42)["records"]]
        b = [r["status"] for r in self.sim(0.3, 0.2, seed=43)["records"]]
        self.assertNotEqual(a, b)

    def test_invalid_config_raises(self):
        for kwargs in ({"p_not_home": -0.1, "p_refused": 0.0},
                       {"p_not_home": 0.8, "p_refused": 0.3},
                       {"monte_carlo_days": -1},
                       {"seed": -1}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                SimConfig(**kwargs)

    def test_monte_carlo_rate(self):
        cfg = SimConfig(p_not_home=0.05, p_refused=0.02, monte_carlo_days=5000)
        sol = dict(self.solution)
        sol["trips"] = [t for t in self.solution["trips"] if t["vehicle_index"] != 0]
        stops = stop_schedule(self.data, sol)
        veh = np.array([r["vehicle_index"] for r in stops])
        mc = monte_carlo(veh, self.data.n_vehicles, cfg)
        self.assertAlmostEqual(mc["failed"]["mean"] / len(stops), 0.07, delta=0.005)
        self.assertEqual(mc, monte_carlo(veh, self.data.n_vehicles, cfg))
        shares = mc["share_days_with_failure_by_vehicle"]
        self.assertEqual(len(shares), self.data.n_vehicles)
        self.assertTrue(all(0.0 <= x <= 1.0 for x in shares))
        self.assertEqual(shares[0], 0.0)

    def test_unknown_address_fallback(self):
        day = simulate_day(self.data, self.solution, {}, SimConfig())
        r = day["records"][0]
        self.assertEqual(r["address"], f"location {r['location_id']}")
        self.assertEqual(r["district"], "")

    def test_solution_untouched(self):
        before = copy.deepcopy(self.solution)
        sim = build_simulation(self.data, self.solution, "data/base", SimConfig(monte_carlo_days=50))
        self.assertEqual(before, self.solution)
        self.assertIn("monte_carlo", sim)
        self.assertEqual(sim["monte_carlo"]["days"], 50)


class TestMainIntegration(unittest.TestCase):
    def run_main(self, out, *extra):
        import contextlib
        import io
        import json

        import main as main_module

        args = ["--mode", "pso", "--dataset", "data/base", "--iterations", "2", "--swarm", "4",
                "--patience", "0", "--no-geometry", "--output-dir", out, *extra]
        with contextlib.redirect_stdout(io.StringIO()):
            main_module.main(args)
        return json.loads(Path(out, "best_solution.json").read_text(encoding="utf-8"))

    def test_main_payload_has_simulation(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            with_sim = self.run_main(a)
            without = self.run_main(b, "--no-sim")
        self.assertEqual(with_sim["delivery_simulation"]["counts"]["total"], 250)
        self.assertNotIn("delivery_simulation", without)
        self.assertEqual(with_sim["best_solution"], without["best_solution"])


if __name__ == "__main__":
    unittest.main()
