"""Instance generator: same rules as dataset/generators, deterministic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.make_instances import SETS, build_instance
from src.data_loader import load_problem


class TestInstanceGenerator(unittest.TestCase):
    def test_seed_42_reproduces_the_dataset(self) -> None:
        # Proves the generator follows the team's rules exactly.
        with tempfile.TemporaryDirectory() as tmp:
            build_instance("repro", 250, 42, Path(tmp))
            for f in ("locations.csv", "orders.csv", "distances.csv", "vehicles.csv"):
                ours = (Path(tmp) / f).read_bytes().replace(b"\r\n", b"\n")
                team = (Path("dataset") / f).read_bytes().replace(b"\r\n", b"\n")
                self.assertEqual(ours, team, f)

    def test_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.assertEqual(build_instance("x", 250, 3001, Path(a)),
                             build_instance("x", 250, 3001, Path(b)))

    def test_instance_loads_and_differs_from_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_instance("test_3001", 250, 3001, Path(tmp))
            data = load_problem(tmp)
            default = load_problem("dataset")
        self.assertEqual(data.n_customers, 250)
        self.assertEqual(data.n_vehicles, 5)
        self.assertEqual(data.params.hard_day_end, default.params.hard_day_end)
        self.assertFalse((data.distance == default.distance).all())

    def test_large_instances_scale_the_fleet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_instance("big", 500, 4001, Path(tmp))
            data = load_problem(tmp)
        self.assertEqual(data.n_customers, 500)
        self.assertEqual(data.n_vehicles, 10)
        self.assertEqual(len({v.vehicle_id for v in data.vehicles}), 10)

    def test_sets_have_the_planned_sizes_and_disjoint_seeds(self) -> None:
        self.assertEqual(len(SETS["test"]), 100)
        self.assertEqual(len(SETS["validation"]), 20)
        self.assertEqual(len(SETS["tuning"]), 4)
        seeds = {s: {seed for _, _, seed in SETS[s]} for s in SETS}
        self.assertFalse(seeds["test"] & (seeds["validation"] | seeds["tuning"]))
        self.assertNotIn(42, set().union(*seeds.values()))


if __name__ == "__main__":
    unittest.main()
