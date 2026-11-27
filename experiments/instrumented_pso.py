"""The current method (``MemeticPSO``), measured from the outside.

``src/pso.py`` is not modified.  This subclass only observes it:

* **credits** — every improvement of the global best, in EUR, is credited to
  ``swarm`` (a particle's new position, including scheduled restarts),
  ``local_search`` (the memetic step on the global best inside the loop) or
  ``final_polish``.  ``construction`` stays 0: the start value is
  ``history[0]["best_score"]`` (after the construction heuristics).
  Each history row gains a cumulative ``credits`` dict; one extra row is
  appended after the final polish.
* **evaluation budget** — ``max_evaluations``: the loop stops at the start of
  the iteration after the one that reached it (the original checks a time
  limit there, which this wrapper triggers), then the final polish runs.
* **frozen swarm** — ``freeze_swarm=True`` runs with inertia = c1 = c2 = 0,
  so particles never move (the ablation of the independent evaluation).

With ``max_evaluations=0`` and ``freeze_swarm=False`` the plan, the
evaluations and the history costs are exactly those of ``MemeticPSO``
(tests/test_instrumented_pso.py).
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from src.data_loader import ProblemData
from src.pso import MemeticPSO, PSOConfig, PSOResult

SOURCES = ("construction", "swarm", "local_search", "final_polish")


class InstrumentedMemeticPSO(MemeticPSO):
    def __init__(
        self,
        data: ProblemData,
        config: PSOConfig = PSOConfig(),
        max_evaluations: int = 0,
        freeze_swarm: bool = False,
    ) -> None:
        if freeze_swarm:
            config = replace(config, inertia=0.0, c1=0.0, c2=0.0)
        if max_evaluations > 0 and config.max_seconds <= 0:
            # The loop's only exit check is the time limit; switch it on with
            # a limit that never triggers by itself.
            config = replace(config, max_seconds=1e15)
        super().__init__(data, config)
        self.max_evaluations = int(max_evaluations)
        self.freeze_swarm = bool(freeze_swarm)
        self.credits = {s: 0.0 for s in SOURCES}
        self._iteration = 0
        self._swarm_gain_this_iteration = 0.0
        self._score_at_last_record = None
        self.loop_evaluations = 0

    # -- observation hooks ---------------------------------------------
    def _update_global(self, score, sol, position) -> None:
        before = self.gbest_score
        super()._update_global(score, sol, position)
        if self._iteration >= 1 and self.gbest_score < before:
            gain = float(before - self.gbest_score)
            self.credits["swarm"] += gain
            self._swarm_gain_this_iteration += gain

    def _record(self, iteration: int) -> None:
        super()._record(iteration)
        if iteration >= 1 and self._score_at_last_record is not None:
            total = float(self._score_at_last_record - self.gbest_score)
            rest = total - self._swarm_gain_this_iteration
            if rest > 1e-12:
                self.credits["local_search"] += rest
        self._swarm_gain_this_iteration = 0.0
        self._score_at_last_record = float(self.gbest_score)
        self.history[-1]["credits"] = dict(self.credits)
        if self.max_evaluations > 0 and self.evaluations >= self.max_evaluations:
            self._start = -1e18          # the loop's time check now breaks

    def _final_polish(self) -> None:
        # MemeticPSO adds its polish evaluations to the same counter; the
        # budget is about the loop, so remember where the loop ended.
        self.loop_evaluations = self.evaluations
        before = float(self.gbest_score)
        super()._final_polish()
        gain = before - float(self.gbest_score)
        if gain > 1e-12:
            self.credits["final_polish"] += gain
        last = self.history[-1]["iteration"] if self.history else 0
        super()._record(last + 1)
        self.history[-1]["credits"] = dict(self.credits)

    def run(self) -> PSOResult:
        t0 = perf_counter()
        result = super().run()
        result.runtime_sec = perf_counter() - t0
        return result
