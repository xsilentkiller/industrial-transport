"""Tests for the convergence fixes in src/pso.py."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from src.cost import segment_stats
from src.data_loader import load_problem
from src.decoder import decode_tour, keys_to_tour
from src.pso import (
    MemeticPSO,
    PSOConfig,
    improve_solution_trips,
    two_opt_order,
    volume_aware_nearest_neighbor_tour,
)


def naive_two_opt(order, data):
    """Original implementation: recompute the whole trip for every move."""
    order = list(order)
    current = segment_stats(data, order)
    improved = True
    while improved:
        improved = False
        n = len(order)
        for i in range(n - 1):
            for j in range(i + 2, n):
                cand = order[:i] + list(reversed(order[i : j + 1])) + order[j + 1 :]
                stats = segment_stats(data, cand)
                if stats.distance_km < current.distance_km - 1e-9:
                    order, current = cand, stats
                    improved = True
    return order


class TestFastTwoOpt(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_problem("dataset")

    def test_same_result_as_naive(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(20):
            order = [int(c) for c in rng.choice(np.arange(1, 251), size=12, replace=False)]
            self.assertEqual(two_opt_order(order, self.data.distance), naive_two_opt(order, self.data))

    def test_never_longer(self) -> None:
        rng = np.random.default_rng(8)
        order = [int(c) for c in rng.choice(np.arange(1, 251), size=20, replace=False)]
        before = segment_stats(self.data, order).distance_km
        after = segment_stats(self.data, two_opt_order(order, self.data.distance)).distance_km
        self.assertLessEqual(after, before + 1e-9)


class TestCandidateScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_problem("dataset")

    def test_candidates_scored_like_the_incumbent(self) -> None:
        # With polish_candidates, a candidate gets the same trip polish as the
        # incumbent, so the same tour scores the same either way.
        pso = MemeticPSO(self.data, PSOConfig(verbose=False, polish_candidates=True))
        tour = volume_aware_nearest_neighbor_tour(self.data)
        score, _ = pso._evaluate_tour(tour)
        raw = decode_tour(self.data, tour, pso.fast_decoder)
        polished = improve_solution_trips(self.data, raw, pso.fast_decoder, passes=1)
        self.assertAlmostEqual(score, min(raw.objective, polished.objective), places=6)
        self.assertLess(score, raw.objective)  # the polish matters on this tour

    def test_switch_off_restores_v0_scoring(self) -> None:
        pso = MemeticPSO(self.data, PSOConfig(verbose=False, polish_candidates=False))
        tour = keys_to_tour(np.random.default_rng(3).random(self.data.n_customers))
        score, _ = pso._evaluate_tour(tour)
        self.assertAlmostEqual(score, decode_tour(self.data, tour, pso.fast_decoder).objective)

    def test_seeded_particles_stay_near_their_tour(self) -> None:
        cfg = replace(PSOConfig(verbose=False, seed=5), seeded_init=True)
        pso = MemeticPSO(self.data, cfg)
        base = np.linspace(0.0, 1.0, self.data.n_customers)
        moved = pso._perturbed(base)
        rank_shift = np.abs(np.argsort(np.argsort(moved)) - np.arange(self.data.n_customers)).mean()
        self.assertLess(rank_shift / self.data.n_customers, 0.10)


if __name__ == "__main__":
    unittest.main()
