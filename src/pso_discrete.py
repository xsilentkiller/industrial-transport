"""Discrete PSO with path relinking (Option B).

A particle is a giant tour (a permutation of 1..n), scored exactly like
``MemeticPSO._evaluate_tour``: Split DP + trip assignment with the fast
decoder, then one pass of 2-opt inside the trips.  There is no local search
inside the loop, so every improvement of the global best before the final
polish comes from the swarm itself.

One step of a particle:

1. inertia: with probability ``inertia`` apply ``mutation_moves`` random
   swap / insert moves;
2. cognitive pull: with probability ``c1 * r1`` (r1 ~ U(0, 1)) relink toward
   the particle's best tour: copy a random contiguous segment of pbest into
   the same positions and fill the other positions with the remaining
   customers in the particle's current order (order crossover);
3. social pull: the same with probability ``c2 * r2`` toward the global best;
4. evaluate, update pbest and gbest.

``freeze_swarm=True`` skips 2 and 3 (pure mutation), which isolates what the
attraction adds.

The final polish is the same as for the other methods: ``MemeticPSO``'s local
search for ``final_local_search_budget`` moves, trip 2-opt with the final
decoder, and a re-decode with the final decoder.  Its evaluations are counted
in ``polish_evaluations``, not in the loop budget.

Coefficients
------------
Chosen once on instances_osm/road_tuning/road_tune_7001..7004 (never on
data/osm), 2026-11-27: swarm 20, ``max_evaluations=5000``, seed 42,
``mutation_moves=2``, final polish (150 moves) included.  Score = best
objective in EUR after the polish; one run per cell:

    setting (inertia, c1, c2)     7001      7002      7003      7004    median
    A  0.3, 0.6, 0.6           1435.16   1490.53   1530.33   1465.83   1478.18
    B  0.5, 0.3, 0.9           1415.98   1475.44   1538.66   1463.06   1469.25
    C  0.2, 1.0, 1.0           1429.85   1467.37   1506.80   1426.71   1448.61

C has the lowest median (and wins on 7002-7004), so the defaults are
inertia 0.2, c1 1.0, c2 1.0.  The swarm did the work there: of the total
improvement over the best start tour, the loop found 46 / 46 / 99 / 69 EUR
and the final polish 22 / 31 / 0 / 0 EUR.  C lies on the edge of the tested
grid (strongest pull, weakest mutation); stronger pulls were not tried.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from .data_loader import ProblemData
from .decoder import DecoderConfig, Solution, decode_tour
from .pso import (
    MemeticPSO,
    PSOConfig,
    PSOResult,
    improve_solution_trips,
    nearest_neighbor_tour,
    randomized_nearest_neighbor_tour,
    sweep_tour,
    volume_aware_nearest_neighbor_tour,
    volume_aware_sweep_tour,
)

_EPS = 1e-9


@dataclass(frozen=True)
class DiscretePSOConfig:
    swarm_size: int = 20
    seed: int = 42
    max_evaluations: int = 20_000
    patience: int = 0          # iterations without improvement, 0 = off
    max_seconds: float = 0.0   # 0 = off
    freeze_swarm: bool = False
    verbose: bool = False
    # inertia / c1 / c2: chosen on road_tune_7001..7004 (module docstring).
    inertia: float = 0.2       # chance per particle and step of k random swap/insert moves
    c1: float = 1.0            # cognitive pull (toward pbest)
    c2: float = 1.0            # social pull (toward gbest)
    mutation_moves: int = 2
    final_local_search_budget: int = 150


@dataclass
class _Particle:
    tour: list[int]
    score: float
    pbest_tour: list[int]
    pbest_score: float
    pbest_solution: Solution


# ---------------------------------------------------------------------------
# Moves on a giant tour
# ---------------------------------------------------------------------------
def mutate_tour(tour: list[int], rng: np.random.Generator, moves: int) -> list[int]:
    """Return a copy of ``tour`` after ``moves`` random swap / insert moves."""
    out = list(tour)
    n = len(out)
    if n < 2:
        return out
    for _ in range(max(0, int(moves))):
        i, j = (int(x) for x in rng.choice(n, size=2, replace=False))
        if rng.random() < 0.5:   # swap
            out[i], out[j] = out[j], out[i]
        else:                    # insert: move the customer at i to position j
            out.insert(j, out.pop(i))
    return out


def relink_tour(tour: list[int], guide: list[int], rng: np.random.Generator) -> list[int]:
    """Order crossover toward ``guide``.

    A random contiguous segment guide[lo..hi] is copied into the same
    positions; the other positions get the remaining customers in the order
    they have in ``tour``.
    """
    n = len(tour)
    if n == 0:
        return []
    a, b = (int(x) for x in rng.integers(0, n, size=2))
    lo, hi = (a, b) if a <= b else (b, a)
    segment = guide[lo : hi + 1]
    in_seg = set(segment)
    rest = [c for c in tour if c not in in_seg]
    return rest[:lo] + list(segment) + rest[lo:]


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
class DiscretePSO:
    def __init__(self, data: ProblemData, cfg: DiscretePSOConfig = DiscretePSOConfig()) -> None:
        self.data = data
        self.cfg = cfg
        self.config = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n = data.n_customers

        self.fast_decoder = replace(DecoderConfig(), use_greedy_baseline=False, assignment_swaps=True)
        self.final_decoder = replace(DecoderConfig(), use_greedy_baseline=True, assignment_swaps=True)

        self.evaluations = 0
        self.polish_evaluations = 0
        self.attraction_moves = 0
        self.history: list[dict] = []
        self.credits = {"construction": 0.0, "swarm": 0.0, "local_search": 0.0, "final_polish": 0.0}

        self.particles: list[_Particle] = []
        self.gbest_tour: list[int] | None = None
        self.gbest_solution: Solution | None = None
        self.gbest_score = np.inf

        self.best_feasible_solution: Solution | None = None
        self.best_feasible_score = np.inf
        self._best_feasible_objective = np.inf

        self._iteration = 0
        self._last_improvement = 0
        self._start = 0.0

    # ------------------------------------------------------------------
    # Evaluation and bookkeeping
    # ------------------------------------------------------------------
    def _evaluate_tour(self, tour: list[int]) -> tuple[float, Solution]:
        """Same scoring as ``MemeticPSO._evaluate_tour``."""
        sol = decode_tour(self.data, tour, self.fast_decoder)
        self.evaluations += 1
        polished = improve_solution_trips(self.data, sol, self.fast_decoder, passes=1)
        if polished.objective < sol.objective - _EPS:
            sol = polished
        self._update_feasible(sol)
        return sol.objective, sol

    def _update_feasible(self, sol: Solution) -> None:
        if sol.feasible and sol.objective < self._best_feasible_objective - _EPS:
            self.best_feasible_solution = sol
            self.best_feasible_score = sol.true_cost_eur
            self._best_feasible_objective = sol.objective

    def _offer(self, sol: Solution, tour: list[int], source: str | None) -> bool:
        """Make ``sol`` the global best if it is better; credit ``source``."""
        self._update_feasible(sol)
        if sol.objective < self.gbest_score - _EPS:
            if source is not None:
                self.credits[source] += float(self.gbest_score - sol.objective)
            self.gbest_score = float(sol.objective)
            self.gbest_solution = sol
            self.gbest_tour = list(tour)
            self._last_improvement = self._iteration
            return True
        return False

    @staticmethod
    def _solution_tour(sol: Solution) -> list[int]:
        return [c for t in sol.trips for c in t.customers]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _start_tours(self) -> list[list[int]]:
        d = self.data
        fixed = [
            nearest_neighbor_tour(d),
            sweep_tour(d),
            volume_aware_sweep_tour(d),
            volume_aware_nearest_neighbor_tour(d),
        ]
        tours: list[list[int]] = []
        for i in range(self.cfg.swarm_size):
            kind = i % 5
            if kind == 4:
                base = randomized_nearest_neighbor_tour(d, self.rng, k=3)
            else:
                base = list(fixed[kind])
            if i >= 4:
                base = mutate_tour(base, self.rng, int(self.rng.integers(2, 7)))
            tours.append(base)
        return tours

    def _init_swarm(self) -> None:
        self.particles = []
        for tour in self._start_tours():
            score, sol = self._evaluate_tour(tour)
            self.particles.append(_Particle(tour, score, list(tour), score, sol))
        best = min(self.particles, key=lambda p: p.pbest_score)
        self.gbest_score = float(best.pbest_score)
        self.gbest_solution = best.pbest_solution
        self.gbest_tour = list(best.pbest_tour)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _step(self, p: _Particle) -> None:
        cfg = self.cfg
        rng = self.rng
        tour = p.tour
        if rng.random() < cfg.inertia:
            tour = mutate_tour(tour, rng, cfg.mutation_moves)
        if not cfg.freeze_swarm:
            if rng.random() < cfg.c1 * rng.random():
                tour = relink_tour(tour, p.pbest_tour, rng)
                self.attraction_moves += 1
            if rng.random() < cfg.c2 * rng.random():
                tour = relink_tour(tour, self.gbest_tour, rng)
                self.attraction_moves += 1
        p.tour = tour
        score, sol = self._evaluate_tour(tour)
        p.score = score
        if score < p.pbest_score - _EPS:
            p.pbest_score = score
            p.pbest_tour = list(tour)
            p.pbest_solution = sol
        self._offer(sol, tour, "swarm")

    def run(self) -> PSOResult:
        cfg = self.cfg
        self._start = perf_counter()
        self._iteration = 0
        self._init_swarm()
        self._record(0)

        iterations_run = 0
        stopped_early = False
        iteration = 0
        while self.evaluations < cfg.max_evaluations:
            iteration += 1
            self._iteration = iteration
            for p in self.particles:
                self._step(p)
            self._record(iteration)
            iterations_run = iteration
            if cfg.verbose and (iteration == 1 or iteration % 10 == 0):
                self._log(iteration)
            if cfg.patience > 0 and iteration - self._last_improvement >= cfg.patience:
                stopped_early = True
                break
            if cfg.max_seconds > 0 and perf_counter() - self._start > cfg.max_seconds:
                break

        self._final_polish()
        self._record(iterations_run + 1)
        if cfg.verbose:
            self._log(iterations_run + 1)

        return PSOResult(
            best_solution=self.gbest_solution,
            best_score=float(self.gbest_score),
            best_feasible_solution=self.best_feasible_solution,
            best_feasible_score=float(self.best_feasible_score),
            history=self.history,
            runtime_sec=perf_counter() - self._start,
            evaluations=self.evaluations,
            seed_solution=None,
            iterations_run=iterations_run,
            stopped_early=stopped_early,
        )

    # ------------------------------------------------------------------
    # Final polish (identical for every method in the comparison)
    # ------------------------------------------------------------------
    def _final_polish(self) -> None:
        best = self.gbest_solution
        helper = MemeticPSO(self.data, PSOConfig(seed=self.cfg.seed, swarm_size=1, verbose=False))
        tour = self._solution_tour(best)
        new_tour, _, sol = helper._local_search(tour, best, self.cfg.final_local_search_budget)
        self.polish_evaluations += helper.evaluations
        self._offer(sol, new_tour, "final_polish")

        polished = improve_solution_trips(self.data, self.gbest_solution, self.final_decoder, passes=3)
        self.polish_evaluations += 1
        self._offer(polished, self._solution_tour(polished), "final_polish")

        tour = self._solution_tour(self.gbest_solution)
        redecoded = decode_tour(self.data, tour, self.final_decoder)
        self.polish_evaluations += 1
        self._offer(redecoded, tour, "final_polish")

    # ------------------------------------------------------------------
    def _record(self, iteration: int) -> None:
        sol = self.gbest_solution
        self.history.append(
            {
                "iteration": iteration,
                "best_score": float(self.gbest_score),
                "best_true_cost": float(sol.true_cost_eur) if sol else np.nan,
                "best_penalty": float(sol.penalty) if sol else np.nan,
                "best_feasible_cost": float(self.best_feasible_score)
                if self.best_feasible_solution is not None
                else np.nan,
                "evaluations": self.evaluations,
                "credits": dict(self.credits),
            }
        )

    def _log(self, iteration: int) -> None:
        sol = self.gbest_solution
        print(
            f"iter {iteration:4d}  best={self.gbest_score:10.2f}  "
            f"cost={sol.true_cost_eur:10.2f}  penalty={sol.penalty:9.1f}  "
            f"feasible={self.best_feasible_score:10.2f}  evals={self.evaluations:6d}  "
            f"attract={self.attraction_moves:5d}  {perf_counter() - self._start:6.1f}s"
        )
