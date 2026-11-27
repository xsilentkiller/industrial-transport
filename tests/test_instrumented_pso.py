"""The current method, measured from the outside (experiments/instrumented_pso.py).

The wrapper must not change what MemeticPSO does: same plan, same history
costs.  It only adds improvement credits, an evaluation budget and the
frozen-swarm ablation.
"""

from __future__ import annotations

import unittest

from experiments.instrumented_pso import InstrumentedMemeticPSO
from src.data_loader import load_problem
from src.pso import MemeticPSO, PSOConfig


def small_cfg(**kw) -> PSOConfig:
    base = dict(swarm_size=4, iterations=3, patience=0, max_seconds=0.0, verbose=False, seed=3)
    base.update(kw)
    return PSOConfig(**base)


class TestInstrumentedPSO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_problem("data/base")
        cls.plain = MemeticPSO(cls.data, small_cfg()).run()
        cls.inst = InstrumentedMemeticPSO(cls.data, small_cfg()).run()

    def test_identical_to_original(self):
        a, b = self.plain, self.inst
        self.assertEqual(a.best_feasible_solution.true_cost_eur, b.best_feasible_solution.true_cost_eur)
        self.assertEqual([t.customers for t in a.best_solution.trips],
                         [t.customers for t in b.best_solution.trips])
        self.assertEqual(a.evaluations, b.evaluations)
        n = len(a.history)
        self.assertEqual([h["best_score"] for h in a.history],
                         [h["best_score"] for h in b.history[:n]])

    def test_credits_sum_to_total_improvement(self):
        h = self.inst.history
        credits = h[-1]["credits"]
        self.assertEqual(set(credits), {"construction", "swarm", "local_search", "final_polish"})
        self.assertAlmostEqual(sum(credits.values()),
                               h[0]["best_score"] - self.inst.best_score, delta=1e-6)

    def test_budget_stops_the_loop(self):
        cfg = small_cfg(iterations=200, swarm_size=4)
        opt = InstrumentedMemeticPSO(self.data, cfg, max_evaluations=300)
        res = opt.run()
        self.assertGreaterEqual(opt.loop_evaluations, 300)
        self.assertLess(res.iterations_run, 200)
        # One iteration is at most swarm + local-search budget evaluations.
        self.assertLess(opt.loop_evaluations, 300 + 4 + cfg.local_search_budget + 2)

    def test_frozen_swarm_has_no_velocity(self):
        res = InstrumentedMemeticPSO(self.data, small_cfg(), freeze_swarm=True).run()
        self.assertTrue(res.best_feasible_solution is not None)
        opt = InstrumentedMemeticPSO(self.data, small_cfg(), freeze_swarm=True)
        self.assertEqual((opt.config.inertia, opt.config.c1, opt.config.c2), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
