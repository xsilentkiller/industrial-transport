"""Split-based memetic random-key particle swarm optimisation.

The swarm searches in continuous random-key space.  A key vector is decoded by
sorting it into a giant tour, applying the Split DP, and assigning the trips to
vehicles.  A memetic local search periodically improves the giant tour of the
global best with 2-opt / relocate / swap moves that are guided by nearest
neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter

import numpy as np

from .data_loader import ProblemData
from .cost import segment_stats
from .decoder import (
    DecoderConfig,
    Solution,
    assign_trips,
    decode_tour,
    keys_to_tour,
    make_chunk_solution,
    tour_to_keys,
)


# ---------------------------------------------------------------------------
# Configuration / results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PSOConfig:
    swarm_size: int = 20
    iterations: int = 60
    inertia: float = 0.72
    c1: float = 1.49
    c2: float = 1.49
    v_max: float = 0.25
    seed: int = 42

    local_search_interval: int = 5
    local_search_budget: int = 40
    local_search_neighbors: int = 8
    final_local_search_budget: int = 120

    reinit_interval: int = 30
    reinit_fraction: float = 0.10

    # Convergence fixes (measured in the README section "Convergence fixes";
    # tested in tests/test_convergence.py).
    # 1) Score every candidate like the incumbent: 2-opt inside its trips
    #    before comparing.  Without it, polished incumbents are compared
    #    against unpolished candidates and nothing is ever accepted.
    polish_candidates: bool = True
    # Write the polished order back into the particle (Lamarckian learning).
    # Off: slower and no better (seeds 1-3: 1,197.34 vs 1,195.45 EUR).
    lamarckian: bool = False
    # 2) Start (and restart) particles near good tours instead of uniform
    #    random keys: random tours are 0 % feasible under the daily window.
    seeded_init: bool = True
    seed_noise: tuple[float, float] = (0.01, 0.10)
    # 3) Restart the worst particles when the swarm has collapsed onto the
    #    global best (mean rank distance below this; 0 = off).  Off: with
    #    seeded starts it triggered once in 3 runs and changed nothing.
    diversity_threshold: float = 0.0
    diversity_restart_fraction: float = 0.3

    max_seconds: float = 0.0
    verbose: bool = True
    log_interval: int = 10

    decoder: DecoderConfig = field(default_factory=DecoderConfig)


@dataclass
class PSOResult:
    best_solution: Solution
    best_score: float
    best_feasible_solution: Solution | None
    best_feasible_score: float
    history: list[dict]
    runtime_sec: float
    evaluations: int
    seed_solution: Solution | None


@dataclass
class _Particle:
    position: np.ndarray
    velocity: np.ndarray
    pbest_position: np.ndarray
    pbest_score: float
    pbest_solution: Solution
    current_solution: Solution
    current_score: float


# ---------------------------------------------------------------------------
# Construction heuristics
# ---------------------------------------------------------------------------
def nearest_neighbor_tour(data: ProblemData) -> list[int]:
    n = data.n_customers
    unvisited = np.ones(n + 1, dtype=bool)
    unvisited[0] = False
    tour: list[int] = []
    cur = 0
    for _ in range(n):
        dist = data.leg_cost[cur].copy()
        dist[~unvisited] = np.inf
        nxt = int(np.argmin(dist))
        tour.append(nxt)
        unvisited[nxt] = False
        cur = nxt
    return tour


def randomized_nearest_neighbor_tour(
    data: ProblemData, rng: np.random.Generator, k: int = 3
) -> list[int]:
    n = data.n_customers
    unvisited = np.ones(n + 1, dtype=bool)
    unvisited[0] = False
    tour: list[int] = []
    cur = 0
    for _ in range(n):
        dist = data.leg_cost[cur].copy()
        dist[~unvisited] = np.inf
        candidates = np.argsort(dist)[: max(1, min(k, int(unvisited.sum())))]
        nxt = int(candidates[int(rng.integers(0, len(candidates)))])
        tour.append(nxt)
        unvisited[nxt] = False
        cur = nxt
    return tour


def sweep_tour(data: ProblemData) -> list[int]:
    dx = data.lon[1:] - data.lon[0]
    dy = data.lat[1:] - data.lat[0]
    angles = np.arctan2(dy, dx)
    return (np.argsort(angles) + 1).astype(int).tolist()


def volume_aware_sweep_tour(
    data: ProblemData,
    threshold: float | None = None,
) -> list[int]:
    """Put low-volume customers first, each group sorted by polar angle.

    A Caddy can only take 10 low-volume orders (2.79 m3 / 10 = 0.279 m3 on
    average), so a contiguous low-volume prefix gives the Split DP a chance
    to build cheap Caddy trips.
    """
    n = data.n_customers
    if threshold is None:
        smallest_usable = float(np.min(data.vehicle_usable_volume))
        threshold = smallest_usable / max(1, data.params.min_customers_per_trip)
    dx = data.lon[1:] - data.lon[0]
    dy = data.lat[1:] - data.lat[0]
    angles = np.arctan2(dy, dx)
    ids = np.arange(1, n + 1)
    low = ids[data.volume[1:] <= threshold]
    high = ids[data.volume[1:] > threshold]
    low = low[np.argsort(angles[low - 1])]
    high = high[np.argsort(angles[high - 1])]
    return np.concatenate([low, high]).astype(int).tolist()


def _nearest_neighbor_order(data: ProblemData, subset: np.ndarray | list[int]) -> list[int]:
    unvisited = set(int(x) for x in subset)
    cur = 0
    out: list[int] = []
    while unvisited:
        nxt = min(unvisited, key=lambda c: data.leg_cost[cur, c])
        out.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return out


def volume_aware_nearest_neighbor_tour(
    data: ProblemData,
    threshold: float | None = None,
) -> list[int]:
    """Low-volume customers (sweep order) followed by high-volume customers
    in nearest-neighbour order.  This gives the Split DP a cheap Caddy prefix
    without destroying the compact routing of the larger orders.
    """
    n = data.n_customers
    if threshold is None:
        smallest_usable = float(np.min(data.vehicle_usable_volume))
        threshold = smallest_usable / max(1, data.params.min_customers_per_trip)
    dx = data.lon[1:] - data.lon[0]
    dy = data.lat[1:] - data.lat[0]
    angles = np.arctan2(dy, dx)
    ids = np.arange(1, n + 1)
    low = ids[data.volume[1:] <= threshold]
    high = ids[data.volume[1:] > threshold]
    low_order = low[np.argsort(angles[low - 1])].astype(int).tolist()
    high_order = _nearest_neighbor_order(data, high)
    return low_order + high_order


def two_opt_order_directed(order: list[int], D: np.ndarray) -> list[int]:
    """First-improvement 2-opt for an asymmetric matrix (road legs).

    Reversing order[i..j] also reverses every leg inside the segment, so the
    move is priced with prefix sums of the forward and backward leg costs
    (O(1) per move, O(n) to rebuild the sums after an accepted move).
    """
    order = list(order)
    n = len(order)
    if n < 3:
        return order

    def prefix(seq: list[int]) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(seq, dtype=np.int64)
        fwd = np.concatenate(([0.0], np.cumsum(D[a[:-1], a[1:]])))
        bwd = np.concatenate(([0.0], np.cumsum(D[a[1:], a[:-1]])))
        return fwd, bwd

    fwd, bwd = prefix(order)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                prev = order[i - 1] if i > 0 else 0
                nxt = order[j + 1] if j + 1 < n else 0
                a, b = order[i], order[j]
                inner_old = fwd[j] - fwd[i]
                inner_new = bwd[j] - bwd[i]
                delta = (D[prev, b] + D[a, nxt] + inner_new
                         - D[prev, a] - D[b, nxt] - inner_old)
                if delta < -1e-9:
                    order[i : j + 1] = order[i : j + 1][::-1]
                    fwd, bwd = prefix(order)
                    improved = True
    return order


def two_opt_order(order: list[int], D: np.ndarray, symmetric: bool | None = None) -> list[int]:
    """First-improvement 2-opt on one trip (depot -> order -> depot).

    Each move reverses order[i..j] (at least three customers) and is priced by
    the change of its two end edges only, O(1) per move, instead of
    recomputing the whole trip.  Same scan order and acceptance rule as the
    original implementation, so it returns the same trip.
    """
    if symmetric is None:
        symmetric = bool(np.allclose(D, D.T, atol=1e-9))
    if not symmetric:
        return two_opt_order_directed(order, D)
    order = list(order)
    n = len(order)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                prev = order[i - 1] if i > 0 else 0
                nxt = order[j + 1] if j + 1 < n else 0
                a, b = order[i], order[j]
                delta = D[prev, b] + D[a, nxt] - D[prev, a] - D[b, nxt]
                if delta < -1e-9:
                    order[i : j + 1] = order[i : j + 1][::-1]
                    improved = True
    return order


def improve_solution_trips(
    data: ProblemData,
    sol: Solution,
    config: DecoderConfig,
    passes: int = 2,
) -> Solution:
    """Memetic trip polish: 2-opt inside every trip, then re-assign.

    The customer set of each trip is unchanged, so capacity/count checks are
    unchanged; only the order (and therefore distance) improves.  Re-running
    the assignment can move the improved trips between vehicles.
    """
    best = sol
    D = data.leg_cost
    for _ in range(max(1, passes)):
        segments = []
        for trip in best.trips:
            order = list(trip.customers)
            if len(order) >= 3:
                order = two_opt_order(order, D, data.symmetric)
            segments.append(segment_stats(data, order))
        new = assign_trips(data, segments, config, method="trip-2opt")
        if new.objective < best.objective - 1e-9:
            best = new
        else:
            break
    return best


# ---------------------------------------------------------------------------
# PSO
# ---------------------------------------------------------------------------
class MemeticPSO:
    def __init__(
        self,
        data: ProblemData,
        config: PSOConfig = PSOConfig(),
    ) -> None:
        self.data = data
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.n = data.n_customers

        # Fast path inside the swarm: Split DP only, no greedy baseline.
        # Assignment swaps stay ON: without them the swarm scores tours with a
        # weaker decoder than the seeded incumbent (data/osm: 1,578.50 vs
        # 1,565.95 EUR on the same tour), so no particle could ever beat it
        # and the convergence curve stayed flat.  Swaps also make the decode
        # faster (fewer daily-window violations for the polish to repair):
        # 57 vs 74 ms per evaluation.
        self.fast_decoder = replace(
            config.decoder,
            use_greedy_baseline=False,
            assignment_swaps=True,
        )
        self.final_decoder = replace(
            config.decoder,
            use_greedy_baseline=True,
            assignment_swaps=True,
        )

        self.nearest = self._nearest_lists(config.local_search_neighbors)
        self.evaluations = 0
        self.history: list[dict] = []

        self.gbest_position: np.ndarray | None = None
        self.gbest_solution: Solution | None = None
        self.gbest_score = np.inf

        self.best_feasible_solution: Solution | None = None
        self.best_feasible_score = np.inf
        self._best_feasible_objective = np.inf
        self.seed_solution: Solution | None = None
        self.restarts = 0

        self.particles: list[_Particle] = []
        self._start = 0.0
        self._last_improvement = 0

    # ------------------------------------------------------------------
    # Evaluation and bookkeeping
    # ------------------------------------------------------------------
    def _evaluate_tour(self, tour: list[int]) -> tuple[float, Solution]:
        sol = decode_tour(self.data, tour, self.fast_decoder)
        self.evaluations += 1
        if self.config.polish_candidates:
            polished = improve_solution_trips(
                self.data, sol, self.fast_decoder, passes=1
            )
            if polished.objective < sol.objective - 1e-9:
                sol = polished
        return sol.objective, sol

    def _evaluate_position(self, pos: np.ndarray) -> tuple[float, Solution, np.ndarray]:
        """Decode a particle; with Lamarckian learning, return the keys of the
        polished tour so the particle keeps what the polish found."""
        score, sol = self._evaluate_tour(keys_to_tour(pos))
        if self.config.polish_candidates and self.config.lamarckian and sol.trips:
            pos = tour_to_keys(self._solution_tour(sol))
        return score, sol, pos

    def _update_feasible(self, sol: Solution) -> None:
        # Feasible = all hard rules hold; the plan may still pay a soft
        # (customer-count) penalty, so plans are compared on the objective
        # (true cost + soft penalty).  The reported score stays the true cost.
        if sol.feasible and sol.objective < self._best_feasible_objective - 1e-9:
            self.best_feasible_solution = sol
            self.best_feasible_score = sol.true_cost_eur
            self._best_feasible_objective = sol.objective

    def _update_global(self, score: float, sol: Solution, position: np.ndarray) -> None:
        self._update_feasible(sol)
        if score < self.gbest_score - 1e-9:
            self.gbest_score = score
            self.gbest_solution = sol
            self.gbest_position = position.copy()
            self._last_improvement = self._iteration

    def _nearest_lists(self, k: int) -> dict[int, np.ndarray]:
        n = self.n
        k = max(1, min(k, n - 1))
        out: dict[int, np.ndarray] = {}
        for c in range(1, n + 1):
            dist = self.data.leg_cost[c].copy()
            dist[c] = np.inf
            dist[0] = np.inf
            out[c] = np.argsort(dist)[:k].astype(np.int64)
        return out

    @staticmethod
    def _solution_tour(sol: Solution) -> list[int]:
        return [c for trip in sol.trips for c in trip.customers]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _initial_tours(self) -> list[list[int]]:
        tours = [
            nearest_neighbor_tour(self.data),
            sweep_tour(self.data),
            volume_aware_sweep_tour(self.data),
            volume_aware_nearest_neighbor_tour(self.data),
        ]
        for _ in range(2):
            tours.append(randomized_nearest_neighbor_tour(self.data, self.rng, k=3))
        # A second volume-aware tour with a slightly different threshold.
        smallest_usable = float(np.min(self.data.vehicle_usable_volume))
        tours.append(
            volume_aware_sweep_tour(
                self.data, threshold=smallest_usable / 9.0
            )
        )
        return tours

    def _init_particles(self) -> None:
        n = self.n
        tours = self._initial_tours()
        self.particles = []

        for i in range(self.config.swarm_size):
            if i < len(tours):
                pos = tour_to_keys(tours[i])
                # Keep the four deterministic heuristics exact; add a little
                # noise to the randomised/extra seeds.
                if i >= 4:
                    pos = np.clip(pos + self.rng.normal(0.0, 0.02, n), 0.0, 1.0)
            elif self.config.seeded_init:
                pos = self._perturbed(tour_to_keys(tours[i % len(tours)]))
            else:
                pos = self.rng.random(n)

            vel = self.rng.uniform(-0.05, 0.05, n)
            score, sol, pos = self._evaluate_position(pos)
            self.particles.append(
                _Particle(
                    position=pos,
                    velocity=vel,
                    pbest_position=pos.copy(),
                    pbest_score=score,
                    pbest_solution=sol,
                    current_solution=sol,
                    current_score=score,
                )
            )

        # Initialise the global best from the swarm.
        best_i = int(np.argmin([p.pbest_score for p in self.particles]))
        self.gbest_score = self.particles[best_i].pbest_score
        self.gbest_solution = self.particles[best_i].pbest_solution
        self.gbest_position = self.particles[best_i].pbest_position.copy()
        for p in self.particles:
            self._update_feasible(p.pbest_solution)

        # Deterministic feasible seed: fixed chunks of 10 along a
        # nearest-neighbour tour.  This guarantees that the benchmark has a
        # feasible incumbent even before the swarm finds one.
        seed_tour = nearest_neighbor_tour(self.data)
        seed_sol = make_chunk_solution(
            self.data, seed_tour, chunk_size=10, config=self.fast_decoder
        )
        self.evaluations += 1
        self.seed_solution = seed_sol
        self._update_feasible(seed_sol)

        if seed_sol.objective < self.gbest_score - 1e-9:
            self.gbest_score = seed_sol.objective
            self.gbest_solution = seed_sol
            self.gbest_position = tour_to_keys(seed_tour)

        # Put the feasible seed into the worst particle, so the swarm always
        # contains at least one feasible member.
        worst = int(np.argmax([p.pbest_score for p in self.particles]))
        self._inject(worst, seed_tour, seed_sol)

    def _perturbed(self, keys: np.ndarray) -> np.ndarray:
        """Keys near a good tour: Gaussian noise of a random strength."""
        lo, hi = self.config.seed_noise
        sigma = float(self.rng.uniform(lo, hi))
        return np.clip(keys + self.rng.normal(0.0, sigma, len(keys)), 0.0, 1.0)

    def _diversity(self) -> float:
        """Mean rank distance of the particles to the global best (0..~0.33)."""
        g = np.argsort(np.argsort(self.gbest_position))
        return float(np.mean([
            np.abs(np.argsort(np.argsort(p.position)) - g).mean() / self.n
            for p in self.particles
        ]))

    def _restart_particles(self, indices) -> None:
        """Replace particles: near an elite tour when seeded, else random."""
        elite = sorted(self.particles, key=lambda p: p.pbest_score)[:5]
        for idx in indices:
            if self.config.seeded_init:
                base = elite[int(self.rng.integers(0, len(elite)))].pbest_position
                if self.rng.random() < 0.5:
                    base = self.gbest_position
                pos = self._perturbed(base)
            else:
                pos = self.rng.random(self.n)
            score, sol, pos = self._evaluate_position(pos)
            self.particles[int(idx)] = _Particle(
                position=pos,
                velocity=self.rng.uniform(-0.05, 0.05, self.n),
                pbest_position=pos.copy(),
                pbest_score=score,
                pbest_solution=sol,
                current_solution=sol,
                current_score=score,
            )
            self._update_global(score, sol, pos)

    def _inject(self, idx: int, tour: list[int], sol: Solution) -> None:
        pos = tour_to_keys(tour)
        self.particles[idx] = _Particle(
            position=pos,
            velocity=np.zeros_like(pos),
            pbest_position=pos.copy(),
            pbest_score=sol.objective,
            pbest_solution=sol,
            current_solution=sol,
            current_score=sol.objective,
        )

    def _seed_incumbents(self) -> None:
        """Decode and polish the two strongest deterministic constructions.

        The swarm starts from noisy versions of these tours, but the Split DP
        and the trip polish are cheap enough to run once with the full decoder
        before the first generation.  The result becomes the initial global
        best and is injected into the worst particle.
        """
        seeds = [
            ("nn-polished", nearest_neighbor_tour(self.data)),
            ("volume-nn-polished", volume_aware_nearest_neighbor_tour(self.data)),
        ]

        best_tour: list[int] | None = None
        best_sol: Solution | None = None
        for method, tour in seeds:
            sol = decode_tour(self.data, tour, self.final_decoder)
            sol.method = method
            self.evaluations += 1
            polished = improve_solution_trips(
                self.data, sol, self.final_decoder, passes=3
            )
            polished.method = method
            self.evaluations += 1
            if polished.objective < sol.objective - 1e-9:
                sol = polished
            self._update_feasible(sol)

            if sol.objective < self.gbest_score - 1e-9:
                self.gbest_score = sol.objective
                self.gbest_solution = sol
                self.gbest_position = tour_to_keys(self._solution_tour(sol))

            if best_sol is None or sol.objective < best_sol.objective - 1e-9:
                best_sol = sol
                best_tour = self._solution_tour(sol)

        if best_sol is None or best_tour is None or not self.particles:
            return

        worst = int(np.argmax([p.pbest_score for p in self.particles]))
        if best_sol.objective < self.particles[worst].pbest_score - 1e-9:
            self._inject(worst, best_tour, best_sol)

    # ------------------------------------------------------------------
    # Local search
    # ------------------------------------------------------------------
    def _local_search(
        self,
        tour: list[int],
        sol: Solution,
        budget: int,
    ) -> tuple[list[int], float, Solution]:
        current_tour = list(tour)
        current_score = sol.objective
        current_sol = sol
        best_tour = list(tour)
        best_score = current_score
        best_sol = sol

        n = len(current_tour)
        if n < 2 or budget <= 0:
            return best_tour, best_score, best_sol

        pos = np.empty(n + 1, dtype=np.int64)
        pos[np.asarray(current_tour, dtype=np.int64)] = np.arange(n)

        for _ in range(budget):
            move = int(self.rng.integers(0, 3))
            i = int(self.rng.integers(0, n))
            c = current_tour[i]
            neighbors = self.nearest[c]
            d = int(neighbors[int(self.rng.integers(0, len(neighbors)))])
            j = int(pos[d])
            if i == j:
                continue

            candidate = current_tour.copy()
            if move == 0:  # 2-opt
                lo, hi = (i, j) if i < j else (j, i)
                candidate[lo : hi + 1] = reversed(candidate[lo : hi + 1])
            elif move == 1:  # relocate c next to d
                x = candidate.pop(i)
                jj = j - 1 if j > i else j
                if jj == i:
                    continue
                candidate.insert(jj, x)
            else:  # swap
                candidate[i], candidate[j] = candidate[j], candidate[i]

            if candidate == current_tour:
                continue

            score, new_sol = self._evaluate_tour(candidate)
            if score < current_score - 1e-9:
                current_tour = candidate
                current_score = score
                current_sol = new_sol
                pos = np.empty(n + 1, dtype=np.int64)
                pos[np.asarray(current_tour, dtype=np.int64)] = np.arange(n)
                if score < best_score - 1e-9:
                    best_tour = candidate.copy()
                    best_score = score
                    best_sol = new_sol

        return best_tour, best_score, best_sol

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> PSOResult:
        self._start = perf_counter()
        self._iteration = 0
        self._init_particles()
        self._seed_incumbents()
        # Iteration 0 = the starting point, so the curve shows what the
        # swarm adds on top of the construction heuristics.
        self._record(0)

        n = self.n
        cfg = self.config

        for iteration in range(1, cfg.iterations + 1):
            self._iteration = iteration
            if cfg.max_seconds > 0 and perf_counter() - self._start > cfg.max_seconds:
                break

            for p in self.particles:
                r1 = self.rng.random(n)
                r2 = self.rng.random(n)
                p.velocity = (
                    cfg.inertia * p.velocity
                    + cfg.c1 * r1 * (p.pbest_position - p.position)
                    + cfg.c2 * r2 * (self.gbest_position - p.position)
                )
                np.clip(p.velocity, -cfg.v_max, cfg.v_max, out=p.velocity)
                p.position = np.clip(p.position + p.velocity, 0.0, 1.0)

                score, sol, p.position = self._evaluate_position(p.position)
                p.current_score = score
                p.current_solution = sol
                if score < p.pbest_score - 1e-9:
                    p.pbest_score = score
                    p.pbest_solution = sol
                    p.pbest_position = p.position.copy()
                self._update_global(score, sol, p.position)

            # Memetic step on the global best.
            if cfg.local_search_interval > 0 and iteration % cfg.local_search_interval == 0:
                if self.gbest_solution is not None:
                    tour = self._solution_tour(self.gbest_solution)
                    new_tour, new_score, new_sol = self._local_search(
                        tour, self.gbest_solution, cfg.local_search_budget
                    )
                    trip_polished = improve_solution_trips(
                        self.data, new_sol, self.fast_decoder, passes=1
                    )
                    if trip_polished.objective < new_score - 1e-9:
                        new_sol = trip_polished
                        new_score = trip_polished.objective
                        new_tour = self._solution_tour(trip_polished)
                    if new_score < self.gbest_score - 1e-9:
                        self.gbest_score = new_score
                        self.gbest_solution = new_sol
                        self.gbest_position = tour_to_keys(new_tour)
                        self._last_improvement = iteration
                        self._update_feasible(new_sol)
                        worst = int(np.argmax([p.pbest_score for p in self.particles]))
                        self._inject(worst, new_tour, new_sol)

            # Diversity restart: on a fixed schedule when the search stalls,
            # and (fix 3) whenever the swarm has collapsed onto the best.
            stalled = (
                cfg.reinit_interval > 0
                and iteration % cfg.reinit_interval == 0
                and iteration - self._last_improvement >= cfg.reinit_interval
            )
            collapsed = (
                cfg.diversity_threshold > 0
                and self._diversity() < cfg.diversity_threshold
            )
            if stalled or collapsed:
                frac = cfg.diversity_restart_fraction if collapsed else cfg.reinit_fraction
                k = max(1, int(round(cfg.swarm_size * frac)))
                worst_indices = np.argsort([p.pbest_score for p in self.particles])[-k:]
                self._restart_particles(worst_indices)
                self.restarts += 1

            self._record(iteration)
            if cfg.verbose and (iteration % cfg.log_interval == 0 or iteration == 1):
                self._log(iteration)

        # Final polish: full decoder on the best tour, then a longer local
        # search on the best feasible tour.
        self._final_polish()

        runtime = perf_counter() - self._start
        best_sol = self.gbest_solution
        if best_sol is None:
            # Extremely defensive fallback; should never happen.
            best_sol = decode_tour(
                self.data, nearest_neighbor_tour(self.data), self.final_decoder
            )
            self.gbest_score = best_sol.objective
            self._update_feasible(best_sol)

        return PSOResult(
            best_solution=best_sol,
            best_score=self.gbest_score,
            best_feasible_solution=self.best_feasible_solution,
            best_feasible_score=self.best_feasible_score,
            history=self.history,
            runtime_sec=runtime,
            evaluations=self.evaluations,
            seed_solution=self.seed_solution,
        )

    def _final_polish(self) -> None:
        cfg = self.config
        if self.gbest_solution is None:
            return

        # Longer local search on the best tour.
        tour = self._solution_tour(self.gbest_solution)
        new_tour, new_score, new_sol = self._local_search(
            tour, self.gbest_solution, cfg.final_local_search_budget
        )
        if new_score < self.gbest_score - 1e-9:
            self.gbest_score = new_score
            self.gbest_solution = new_sol
            self.gbest_position = tour_to_keys(new_tour)
            self._update_feasible(new_sol)

        # 2-opt inside each trip, then re-assign; repeat on the best feasible.
        for candidate in (self.gbest_solution, self.best_feasible_solution):
            if candidate is None:
                continue
            polished = improve_solution_trips(
                self.data, candidate, self.final_decoder, passes=3
            )
            self.evaluations += 1
            self._update_feasible(polished)
            if polished.objective < self.gbest_score - 1e-9:
                self.gbest_score = polished.objective
                self.gbest_solution = polished
                self.gbest_position = tour_to_keys(self._solution_tour(polished))

        # Polish every distinct personal best.  This gives the memetic step a
        # chance to improve tours that the swarm did not promote to gbest.
        seen: set[tuple[int, ...]] = set()
        for p in self.particles:
            tour_key = tuple(self._solution_tour(p.pbest_solution))
            if tour_key in seen:
                continue
            seen.add(tour_key)
            polished = improve_solution_trips(
                self.data, p.pbest_solution, self.final_decoder, passes=2
            )
            self.evaluations += 1
            self._update_feasible(polished)
            if polished.objective < self.gbest_score - 1e-9:
                self.gbest_score = polished.objective
                self.gbest_solution = polished
                self.gbest_position = tour_to_keys(self._solution_tour(polished))

        # Re-decode with greedy baseline + assignment swaps.
        for candidate in (self.gbest_solution, self.best_feasible_solution):
            if candidate is None:
                continue
            cand_tour = self._solution_tour(candidate)
            polished = decode_tour(self.data, cand_tour, self.final_decoder)
            self.evaluations += 1
            self._update_feasible(polished)
            if polished.objective < self.gbest_score - 1e-9:
                self.gbest_score = polished.objective
                self.gbest_solution = polished
                self.gbest_position = tour_to_keys(cand_tour)

    # ------------------------------------------------------------------
    def _record(self, iteration: int) -> None:
        sol = self.gbest_solution
        feas = self.best_feasible_solution
        self.history.append(
            {
                "iteration": iteration,
                "best_score": float(self.gbest_score),
                "best_true_cost": float(sol.true_cost_eur) if sol else np.nan,
                "best_penalty": float(sol.penalty) if sol else np.nan,
                "best_feasible_cost": float(self.best_feasible_score)
                if feas is not None
                else np.nan,
                "evaluations": self.evaluations,
            }
        )

    def _log(self, iteration: int) -> None:
        sol = self.gbest_solution
        feas = self.best_feasible_solution
        elapsed = perf_counter() - self._start
        cost = sol.true_cost_eur if sol else np.nan
        pen = sol.penalty if sol else np.nan
        feas_cost = self.best_feasible_score if feas is not None else np.nan
        print(
            f"iter {iteration:4d}  best={self.gbest_score:10.2f}  "
            f"cost={cost:10.2f}  penalty={pen:9.1f}  "
            f"feasible={feas_cost:10.2f}  evals={self.evaluations:6d}  "
            f"{elapsed:6.1f}s"
        )
