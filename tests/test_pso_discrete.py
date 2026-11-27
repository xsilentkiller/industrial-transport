"""Tests for the discrete PSO with path relinking (src/pso_discrete.py).

All runs use data/osm with small budgets (swarm 6, at most 300 loop
evaluations, a short final polish), so the whole file runs in well under a
minute.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from src.data_loader import load_problem
from src.decoder import check_solution
from src.pso import PSOResult
from src.pso_discrete import (
    DiscretePSO,
    DiscretePSOConfig,
    mutate_tour,
    relink_tour,
)

N = 250
SMALL = DiscretePSOConfig(
    swarm_size=6,
    seed=1,
    max_evaluations=300,
    final_local_search_budget=30,
)


def _is_permutation(tour, n=N) -> bool:
    return len(tour) == n and sorted(int(c) for c in tour) == list(range(1, n + 1))


def _plan_tour(sol) -> list[int]:
    return [c for t in sol.trips for c in t.customers]


class TestMoves(unittest.TestCase):
    """The two particle moves on their own (no decoding)."""

    def test_mutation_keeps_a_permutation(self):
        rng = np.random.default_rng(0)
        tour = list(rng.permutation(N) + 1)
        for _ in range(500):
            tour = mutate_tour(tour, rng, int(rng.integers(1, 5)))
            self.assertTrue(_is_permutation(tour))

    def test_mutation_changes_the_tour(self):
        rng = np.random.default_rng(3)
        tour = list(range(1, N + 1))
        changed = sum(mutate_tour(tour, rng, 2) != tour for _ in range(50))
        self.assertGreater(changed, 40)

    def test_relink_keeps_a_permutation(self):
        rng = np.random.default_rng(1)
        tour = list(rng.permutation(N) + 1)
        guide = list(rng.permutation(N) + 1)
        for _ in range(500):
            tour = relink_tour(tour, guide, rng)
            self.assertTrue(_is_permutation(tour))
            if rng.random() < 0.3:
                guide = list(rng.permutation(N) + 1)

    def test_relink_copies_a_guide_segment_and_keeps_the_rest_in_order(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            tour = list(rng.permutation(N) + 1)
            guide = list(rng.permutation(N) + 1)
            child = relink_tour(tour, guide, rng)
            same = [i for i in range(N) if child[i] == guide[i]]
            self.assertTrue(same, "no position copied from the guide")
            # The copied block is one contiguous run of positions that match
            # the guide; outside that run the child keeps the particle's
            # relative order.  (Other runs can match by chance, so accept
            # the child if any run explains it.)
            runs, start = [], same[0]
            for a, b in zip(same, same[1:] + [None]):
                if b != a + 1:
                    runs.append((start, a))
                    start = b

            def explains(lo, hi):
                seg = set(guide[lo : hi + 1])
                rest_child = [c for k, c in enumerate(child) if not lo <= k <= hi]
                return rest_child == [c for c in tour if c not in seg]

            self.assertTrue(any(explains(lo, hi) for lo, hi in runs))

    def test_relink_toward_itself_is_identity(self):
        rng = np.random.default_rng(2)
        tour = list(rng.permutation(N) + 1)
        self.assertEqual(relink_tour(tour, list(tour), rng), tour)


class TestDiscretePSORun(unittest.TestCase):
    """One shared run on data/osm, checked from several sides."""

    @classmethod
    def setUpClass(cls):
        cls.data = load_problem("data/osm")
        cls.opt = DiscretePSO(cls.data, SMALL)
        cls.res = cls.opt.run()

    def test_returns_a_pso_result(self):
        self.assertIsInstance(self.res, PSOResult)
        self.assertIsNone(self.res.seed_solution)
        self.assertGreater(self.res.iterations_run, 0)
        self.assertGreater(self.res.runtime_sec, 0.0)

    def test_best_plan_serves_every_customer_once(self):
        best = self.res.best_solution
        self.assertTrue(_is_permutation(_plan_tour(best)))
        self.assertEqual(check_solution(self.data, best), [])
        self.assertAlmostEqual(self.res.best_score, best.objective, places=6)

    def test_best_feasible_matches_its_plan(self):
        feas = self.res.best_feasible_solution
        if feas is None:
            self.assertEqual(self.res.best_feasible_score, float("inf"))
        else:
            self.assertTrue(feas.feasible)
            self.assertAlmostEqual(self.res.best_feasible_score, feas.true_cost_eur, places=6)

    def test_particles_are_permutations_after_the_run(self):
        for p in self.opt.particles:
            self.assertTrue(_is_permutation(p.tour))
            self.assertTrue(_is_permutation(p.pbest_tour))

    def test_evaluation_budget_is_respected(self):
        self.assertGreaterEqual(self.res.evaluations, SMALL.max_evaluations)
        self.assertLessEqual(self.res.evaluations, SMALL.max_evaluations + SMALL.swarm_size)
        self.assertEqual(self.res.evaluations, self.opt.evaluations)
        self.assertGreater(self.opt.polish_evaluations, 0)

    def test_history_rows(self):
        hist = self.res.history
        self.assertEqual(len(hist), self.res.iterations_run + 2)
        keys = {"iteration", "best_score", "best_true_cost", "best_penalty",
                "best_feasible_cost", "evaluations", "credits"}
        for row in hist:
            self.assertTrue(keys <= set(row))
        self.assertEqual(hist[0]["iteration"], 0)
        scores = [row["best_score"] for row in hist]
        self.assertTrue(all(b <= a + 1e-9 for a, b in zip(scores, scores[1:])))
        self.assertAlmostEqual(hist[-1]["best_score"], self.res.best_score, places=6)

    def test_credits_sum_to_the_total_improvement(self):
        hist = self.res.history
        credits = hist[-1]["credits"]
        self.assertEqual(set(credits), {"construction", "swarm", "local_search", "final_polish"})
        self.assertEqual(credits["construction"], 0.0)
        self.assertEqual(credits["local_search"], 0.0)
        total = hist[0]["best_score"] - self.res.best_score
        self.assertAlmostEqual(sum(credits.values()), total, delta=1e-6)
        for row in hist:
            self.assertAlmostEqual(
                sum(row["credits"].values()), hist[0]["best_score"] - row["best_score"],
                delta=1e-6,
            )

    def test_swarm_uses_attraction_moves(self):
        self.assertGreater(self.opt.attraction_moves, 0)


class TestDiscretePSOBehaviour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_problem("data/osm")
        cls.cfg = replace(SMALL, max_evaluations=120, final_local_search_budget=10)

    def _run(self, cfg):
        opt = DiscretePSO(self.data, cfg)
        return opt, opt.run()

    def test_same_seed_gives_identical_result(self):
        _, a = self._run(self.cfg)
        _, b = self._run(self.cfg)
        self.assertEqual(a.best_score, b.best_score)
        self.assertEqual(_plan_tour(a.best_solution), _plan_tour(b.best_solution))
        self.assertEqual(a.evaluations, b.evaluations)
        self.assertEqual(
            [r["best_score"] for r in a.history], [r["best_score"] for r in b.history]
        )

    def test_different_seeds_differ(self):
        opt_a, a = self._run(self.cfg)
        opt_b, b = self._run(replace(self.cfg, seed=2))
        tours_a = [p.tour for p in opt_a.particles]
        tours_b = [p.tour for p in opt_b.particles]
        self.assertNotEqual(tours_a, tours_b)

    def test_budget_below_swarm_size_stops_cleanly(self):
        opt, res = self._run(replace(self.cfg, max_evaluations=3))
        self.assertEqual(res.iterations_run, 0)
        self.assertEqual(res.evaluations, self.cfg.swarm_size)
        self.assertEqual(check_solution(self.data, res.best_solution), [])
        self.assertTrue(_is_permutation(_plan_tour(res.best_solution)))
        self.assertEqual(len(res.history), 2)

    def test_freeze_swarm_disables_attraction(self):
        opt, res = self._run(replace(self.cfg, max_evaluations=60, freeze_swarm=True))
        self.assertEqual(opt.attraction_moves, 0)
        self.assertGreater(res.iterations_run, 0)
        self.assertEqual(check_solution(self.data, res.best_solution), [])

    def test_patience_stops_early(self):
        # inertia 0 and frozen swarm: particles never move, so nothing improves.
        cfg = replace(self.cfg, max_evaluations=10_000, patience=2,
                      inertia=0.0, freeze_swarm=True)
        _, res = self._run(cfg)
        self.assertTrue(res.stopped_early)
        self.assertEqual(res.iterations_run, 2)


if __name__ == "__main__":
    unittest.main()
