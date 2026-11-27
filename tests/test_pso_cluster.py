"""Option C: cluster-centre PSO (src/pso_cluster.py)."""

from __future__ import annotations

import unittest

import numpy as np

from src.data_loader import load_problem
from src.decoder import check_solution
from src.pso_cluster import ClusterPSO, ClusterPSOConfig, centres_to_tour


def cfg(**kw) -> ClusterPSOConfig:
    base = dict(swarm_size=6, seed=5, max_evaluations=300, verbose=False)
    base.update(kw)
    return ClusterPSOConfig(**base)


class TestCentresToTour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_problem("data/osm")
        cls.n = cls.data.n_customers
        cls.all = list(range(1, cls.n + 1))

    def test_random_particles_give_permutations(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            k = int(rng.integers(3, 25))
            centres = rng.random((k, 2))
            weights = rng.uniform(0.5, 2.0, k)
            tour = centres_to_tour(self.data, centres, weights)
            self.assertEqual(sorted(tour), self.all)

    def test_empty_cluster_is_fine(self):
        # One centre far outside the city gets no customers.
        centres = np.array([[0.5, 0.5], [0.2, 0.8], [50.0, 50.0]])
        tour = centres_to_tour(self.data, centres, np.ones(3))
        self.assertEqual(sorted(tour), self.all)

    def test_identical_centres_are_fine(self):
        centres = np.full((5, 2), 0.5)
        tour = centres_to_tour(self.data, centres, np.ones(5))
        self.assertEqual(sorted(tour), self.all)


class TestClusterPSO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_problem("data/osm")
        cls.opt = ClusterPSO(cls.data, cfg())
        cls.res = cls.opt.run()

    def test_best_plan_is_valid(self):
        best = self.res.best_solution
        served = sorted(c for t in best.trips for c in t.customers)
        self.assertEqual(served, list(range(1, self.data.n_customers + 1)))
        self.assertEqual(check_solution(self.data, best), [])
        self.assertIsNotNone(self.res.best_feasible_solution)

    def test_deterministic_for_a_seed(self):
        again = ClusterPSO(self.data, cfg()).run()
        self.assertEqual(again.best_score, self.res.best_score)
        other = ClusterPSO(self.data, cfg(seed=6)).run()
        self.assertNotEqual([h["best_score"] for h in other.history],
                            [h["best_score"] for h in self.res.history])

    def test_stops_at_the_budget(self):
        self.assertGreaterEqual(self.res.evaluations, 300)
        self.assertLessEqual(self.res.evaluations, 300 + 6)

    def test_tiny_budget(self):
        res = ClusterPSO(self.data, cfg(max_evaluations=3)).run()
        self.assertEqual(check_solution(self.data, res.best_solution), [])

    def test_credits_sum_to_total_improvement(self):
        h = self.res.history
        credits = h[-1]["credits"]
        self.assertEqual(set(credits), {"construction", "swarm", "local_search", "final_polish"})
        self.assertEqual(credits["local_search"], 0.0)
        self.assertAlmostEqual(sum(credits.values()), h[0]["best_score"] - self.res.best_score,
                               delta=1e-6)

    def test_frozen_swarm_makes_no_attraction_moves(self):
        frozen = ClusterPSO(self.data, cfg(freeze_swarm=True))
        frozen.run()
        self.assertEqual(frozen.attraction_moves, 0)
        self.assertGreater(self.opt.attraction_moves, 0)


if __name__ == "__main__":
    unittest.main()
