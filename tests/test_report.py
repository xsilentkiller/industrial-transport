"""Report generator smoke tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data_loader import load_problem
from src.decoder import DecoderConfig, decode_tour
from src.delivery_sim import SimConfig, build_simulation
from src.report import generate_report


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_problem("data/base")
        cfg = DecoderConfig(enforce_min_customers=True, use_greedy_baseline=False)
        sol = decode_tour(cls.data, list(range(1, 21)), cfg)
        cost = sol.true_cost_eur
        cls.payload = {
            "dataset": "data/base",
            "config": {
                "swarm_size": 1,
                "iterations": 2,
                "seed": 1,
                "local_search_interval": 0,
                "local_search_budget": 0,
                "final_local_search_budget": 0,
            },
            "runtime_sec": 0.1,
            "evaluations": 2,
            "best_feasible_cost": cost,
            "check_errors": [],
            "best_solution": sol.as_dict(),
            "history": [
                {
                    "iteration": 1,
                    "best_score": cost + 10.0,
                    "best_true_cost": cost + 10.0,
                    "best_penalty": 0.0,
                    "best_feasible_cost": cost + 10.0,
                    "evaluations": 1,
                },
                {
                    "iteration": 2,
                    "best_score": cost,
                    "best_true_cost": cost,
                    "best_penalty": 0.0,
                    "best_feasible_cost": cost,
                    "evaluations": 2,
                },
            ],
        }

    def test_generate_report_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            report = generate_report(self.data, self.payload, out)

            self.assertTrue(report.exists())
            for name in (
                "summary.csv",
                "vehicles.csv",
                "trips.csv",
                "history.csv",
                "convergence.png",
                "vehicle_timeline.png",
                "cost_breakdown.png",
                "trip_capacity.png",
                "route_map.png",
            ):
                self.assertTrue((out / name).exists(), name)

            text = report.read_text(encoding="utf-8")
            self.assertIn("Furniture Delivery - PSO Report", text)
            self.assertIn("Trip details", text)
            self.assertIn("Convergence and loss", text)
            self.assertIn("Vehicle-day timeline", text)

            trip_lines = (out / "trips.csv").read_text(
                encoding="utf-8-sig"
            ).splitlines()
            self.assertEqual(len(trip_lines), 1 + self.payload["best_solution"]["n_trips"])
            history_lines = (out / "history.csv").read_text(
                encoding="utf-8-sig"
            ).splitlines()
            self.assertEqual(len(history_lines), 1 + len(self.payload["history"]))
            self.assertEqual(len(list(out.glob("*.png"))), 5)

    DELIVERY_CSVS = ("deliveries.csv", "redelivery.csv", "returned_to_stock.csv")

    def test_delivery_section_and_csvs(self) -> None:
        payload = dict(self.payload)
        payload["delivery_simulation"] = build_simulation(
            self.data, payload["best_solution"], "data/base",
            SimConfig(p_not_home=0.3, p_refused=0.5, monte_carlo_days=100),
        )
        stops = sum(len(t["customers"]) for t in payload["best_solution"]["trips"])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            text = generate_report(self.data, payload, out).read_text(encoding="utf-8")
            self.assertIn("Delivery records (simulated day)", text)
            self.assertIn('data-status="refused"', text)
            self.assertIn('data-status="attempted"', text)
            for name in self.DELIVERY_CSVS:
                self.assertTrue((out / name).exists(), name)
            rows = (out / "deliveries.csv").read_text(encoding="utf-8-sig").splitlines()
            self.assertEqual(len(rows), 1 + stops)
            self.assertEqual(rows[0].split(",")[:3], ["vehicle", "trip", "stop"])

    def test_report_without_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            text = generate_report(self.data, self.payload, out).read_text(encoding="utf-8")
            self.assertNotIn('id="deliveries"', text)
            for name in self.DELIVERY_CSVS:
                self.assertFalse((out / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
