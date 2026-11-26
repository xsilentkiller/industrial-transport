"""Tests for experiments/tune.py (PSO parameter tuning)."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments import tune


def row(combo, instance, seed, cost, feasible=True, runtime=100.0, iters=50, rnd=1):
    return {
        "round": rnd, "combo": combo, "swarm": 20, "patience": 10, "ls_interval": 5,
        "max_iterations": 300, "instance": instance, "seed": seed,
        "real_cost": cost if feasible else None, "penalty": 0.0, "feasible": feasible,
        "iterations_run": iters, "stopped_early": True, "runtime_s": runtime,
        "evaluations": 1000, "error": "",
    }


class TestScoring(unittest.TestCase):
    def test_gap_per_instance(self) -> None:
        rows = tune.add_gaps([
            row("A", "i1", 1, 100.0), row("B", "i1", 1, 110.0),
            row("A", "i2", 1, 200.0), row("B", "i2", 1, 200.0),
        ])
        self.assertEqual([round(r["gap_pct"], 9) for r in rows], [0.0, 10.0, 0.0, 0.0])

    def test_infeasible_gap_is_inf_and_ranked_last(self) -> None:
        rows = tune.add_gaps([
            row("A", "i1", 1, 100.0), row("B", "i1", 1, None, feasible=False),
        ])
        self.assertTrue(math.isinf(rows[1]["gap_pct"]))
        ranked = tune.rank(tune.summarise(rows, ["A", "B"]))
        self.assertEqual([s["combo"] for s in ranked], ["A", "B"])

    def test_errored_run_counts_as_infeasible(self) -> None:
        bad = row("B", "i1", 1, None, feasible=False)
        bad["error"] = "boom"
        rows = tune.add_gaps([row("A", "i1", 1, 100.0), bad])
        self.assertTrue(math.isinf(rows[1]["gap_pct"]))

    def test_rank_tie_prefers_faster(self) -> None:
        summary = [
            {"combo": "slow", "median_gap": 1.0, "median_runtime": 200.0},
            {"combo": "fast", "median_gap": 1.3, "median_runtime": 100.0},
            {"combo": "far", "median_gap": 3.0, "median_runtime": 10.0},
        ]
        self.assertEqual([s["combo"] for s in tune.rank(summary, tie_pp=0.5)],
                         ["fast", "slow", "far"])

    def test_rows_of_other_grids_are_ignored(self) -> None:
        rows = [row("A", "i1", 1, 100.0), row("OLD", "i1", 1, 50.0)]
        summary = tune.summarise(tune.add_gaps(rows, ["A"]), ["A"])
        self.assertEqual(summary[0]["median_gap"], 0.0)   # OLD did not set best known

    def test_suggested_iterations_rounds_up_to_5(self) -> None:
        self.assertEqual(tune.suggested_iterations([41, 43, 47]), 45)
        self.assertEqual(tune.suggested_iterations([40, 40]), 40)

    def test_keep_count(self) -> None:
        self.assertEqual(tune.keep_count(0.5, 24), 12)
        self.assertEqual(tune.keep_count(0.5, 5), 3)
        self.assertEqual(tune.keep_count(4, 12), 4)
        self.assertEqual(tune.keep_count(4, 2), 2)

    def test_round_seeds(self) -> None:
        self.assertEqual(tune.round_seeds([1, 2, 2], 0), [1])
        self.assertEqual(tune.round_seeds([1, 2, 2], 1), [2, 3])
        self.assertEqual(tune.round_seeds([1, 2, 2], 2), [4, 5])


class TestFiles(unittest.TestCase):
    def test_find_instances_parent_and_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a", "b"):
                (root / name).mkdir()
                (root / name / "params.json").write_text("{}")
            (root / "notes").mkdir()
            found = tune.find_instances([str(root)])
            self.assertEqual([p.name for p in found], ["a", "b"])
            self.assertEqual(tune.find_instances([str(root / "a")])[0].name, "a")

    def test_resume_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.csv"
            tune.append_run(path, row("A", "i1", 1, 100.5))
            tune.append_run(path, row("A", "i1", 2, None, feasible=False))
            rows = tune.load_runs(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["real_cost"], 100.5)
            self.assertIsNone(rows[1]["real_cost"])
            self.assertFalse(rows[1]["feasible"])
            self.assertEqual(tune.done_keys(rows), {("A", "i1", 1), ("A", "i1", 2)})


class TestEndToEnd(unittest.TestCase):
    """Tiny real tuning run: two 25-customer instances, two rounds."""

    def test_end_to_end_tiny(self) -> None:
        from experiments.make_instances import build_instance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, seed in enumerate((11, 12)):
                build_instance(f"t{i}", 25, seed, root / "inst" / f"t{i}")
            argv = [
                "--instances", str(root / "inst"), "--out", str(root / "tuning"),
                "--name", "e2e", "--swarm", "3", "4", "--patience", "3",
                "--ls-interval", "5", "--max-iterations", "15",
                "--baseline-iterations", "6",
                "--seeds-per-round", "1", "1", "--keep", "1", "--workers", "2",
                "--time-limit", "60",
            ]
            out = tune.main(argv)
            for name in ("runs.csv", "summary.csv", "tuning.png", "best_config.json"):
                self.assertTrue((out / name).exists(), name)

            with (out / "runs.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            # round 1: (2 combos + baseline) x 2 instances; round 2: (1 + baseline) x 2
            self.assertEqual(len(rows), 3 * 2 + 2 * 2)
            baseline_rounds = {r["round"] for r in rows if r["combo"] == tune.BASELINE_KEY_PREFIX + "6"}
            self.assertEqual(baseline_rounds, {"1", "2"})

            best = json.loads((out / "best_config.json").read_text(encoding="utf-8"))
            self.assertEqual(best["suggested_iterations"] % 5, 0)
            self.assertIn(best["swarm_size"], (3, 4, 20))

            # Rerunning the same command adds nothing.
            tune.main(argv)
            with (out / "runs.csv").open(encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))), len(rows))


if __name__ == "__main__":
    unittest.main()
