"""Cluster-centre PSO (Option C): the swarm decides which customers ride together.

A particle is K trip centres on the map, each with a priority weight:
``position[k] = (x_k, y_k, w_k)``, coordinates normalised to the city's
bounding box (aspect ratio kept, so distances are proportional to km).  With
about 15–20 centres that is ~50 numbers instead of the 250 random keys of the
original method, and a small move of a centre moves a few customers between
trips: a small, meaningful change, which is what PSO is good at.

Decoding a particle (``centres_to_clusters``): **each cluster is a trip**.

1. Every centre slot has the capacity of a vehicle: the slots copy the
   vehicles of the best construction plan's trips (largest trip first), extra
   slots get the smallest vehicle.  So there are Caddy-, Sprinter- and
   truck-sized clusters, like a real fleet.
2. Customers, largest volume first, join the centre with the smallest
   ``w_k * distance(customer, centre)`` that still has room for them; if
   none has room, the nearest.
3. Clusters with fewer customers than the minimum per trip are merged into
   the nearest cluster (by centre) if the merged load fits the largest
   vehicle.
4. Inside a cluster customers are chained by nearest neighbour (road km)
   from the depot.  Each cluster goes through the Split DP on its own (it
   stays one trip unless it fits no vehicle), then the trips of all
   clusters go through the same trip-to-vehicle assignment and in-trip 2-opt
   as every other method: one evaluation.

Why not one giant tour: Split re-cuts a giant tour as if the fleet were
unlimited and ignores the clusters, so the swarm's grouping would be lost
(measured: trips too bulky for the Caddies, Sprinters past 24:00).

There is no local search inside the loop: every improvement of the global
best before the final polish comes from the swarm.  The final polish is the
same as Option B's (``src/pso_discrete.py``): ``MemeticPSO``'s local search
for ``final_local_search_budget`` moves, in-trip 2-opt with the final
decoder, re-decode with the final decoder.

``freeze_swarm=True`` drops the pbest/gbest attraction: particles only drift
by inertia and a small random jitter (the ablation).

The particles are attracted to the swarm's own best position, not to the
overall best plan: that can still be a construction plan, whose trip centres
decode differently under the cluster rules and would mislead the swarm
(measured: with it the swarm never improved; without it, see below).

Coefficients
------------
Chosen once on instances_osm/road_tuning/road_tune_7001..7004 (never on
data/osm), 2026-11-27: swarm 20, ``max_evaluations=5000``, seed 42, final
polish included.  Best objective in EUR (swarm credit / polish credit):

    setting (inertia, c1, c2, v_max)   7001      7002       7003       7004     median
    A  0.72, 1.49, 1.49, 0.10        1494 (0/4) 1458 (93/25) 1456 (203/9) 1496 (0/41) 1476
    B  0.50, 1.00, 1.50, 0.05        1451 (47/0) 1543 (0/34) 1579 (0/87) 1496 (0/41) 1519
    C  0.40, 1.50, 1.50, 0.03        1446 (38/14) 1543 (0/34) 1529 (138/0) 1496 (0/41) 1512

A (the standard constriction values, as the original method) has the lowest
median and the largest swarm credit, so it is the default.  On some days the
swarm does not beat the best construction plan within 5,000 evaluations; the
comparison uses 20,000.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from .data_loader import ProblemData
from .decoder import DecoderConfig, Solution, assign_trips, decode_tour, split_dp
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
SOURCES = ("construction", "swarm", "local_search", "final_polish")


@dataclass(frozen=True)
class ClusterPSOConfig:
    swarm_size: int = 20
    seed: int = 42
    max_evaluations: int = 20_000
    patience: int = 0             # iterations without improvement, 0 = off
    max_seconds: float = 0.0      # 0 = off
    freeze_swarm: bool = False
    verbose: bool = False
    inertia: float = 0.72
    c1: float = 1.49
    c2: float = 1.49
    v_max_frac: float = 0.1       # velocity limit per step, share of the box
    extra_centres: int = 2        # centres beyond the best construction's trips
    jitter: float = 0.01          # random drift of a frozen swarm
    final_local_search_budget: int = 150
    log_interval: int = 10


# ---------------------------------------------------------------------------
# Geometry and decoding
# ---------------------------------------------------------------------------
def _box_coords(data: ProblemData) -> np.ndarray:
    """(n+1, 2) coordinates in a box ~[0, 1]² proportional to km (index 0 = depot)."""
    lat = np.asarray(data.lat, dtype=np.float64)
    lon = np.asarray(data.lon, dtype=np.float64)
    lat0 = float(lat.mean())
    x = (lon - lon.min()) * np.cos(np.radians(lat0)) * 111.32
    y = (lat - lat.min()) * 110.57
    span = max(float(x.max()), float(y.max()), 1e-9)
    return np.column_stack([x / span, y / span])


def centres_to_clusters(
    data: ProblemData,
    centres: np.ndarray,
    weights: np.ndarray,
    coords: np.ndarray | None = None,
    cap_w: np.ndarray | None = None,
    cap_v: np.ndarray | None = None,
) -> list[list[int]]:
    """Ordered customer lists, one per non-empty cluster (see module doc).

    ``cap_w`` / ``cap_v``: capacity of each centre slot; default the largest
    vehicle for every slot.
    """
    xy = _box_coords(data) if coords is None else coords
    n = data.n_customers
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 2)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    k = len(centres)
    cust = xy[1:]
    dist = np.sqrt(((cust[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2))
    score = dist * weights[None, :]

    big_w = float(np.max(data.vehicle_weight_cap))
    big_v = float(np.max(data.vehicle_usable_volume))
    cap_w = np.full(k, big_w) if cap_w is None else np.asarray(cap_w, dtype=np.float64)
    cap_v = np.full(k, big_v) if cap_v is None else np.asarray(cap_v, dtype=np.float64)
    load_w = np.zeros(k)
    load_v = np.zeros(k)
    members: list[list[int]] = [[] for _ in range(k)]
    # Most constrained first (large volume), ties by customer id.
    order = sorted(range(1, n + 1), key=lambda c: (-float(data.volume[c]), c))
    for c in order:
        w, v = float(data.weight[c]), float(data.volume[c])
        prefs = np.argsort(score[c - 1], kind="stable")
        chosen = int(prefs[0])
        for j in prefs:
            j = int(j)
            if load_w[j] + w <= cap_w[j] + 1e-9 and load_v[j] + v <= cap_v[j] + 1e-9:
                chosen = j
                break
        members[chosen].append(c)
        load_w[chosen] += w
        load_v[chosen] += v

    # Merge clusters below the minimum trip size into the nearest cluster
    # (by centre) whose merged load still fits the largest vehicle.
    min_n = int(data.params.min_customers_per_trip)
    merged = True
    while merged:
        merged = False
        for j in sorted(range(k), key=lambda j: (len(members[j]), j)):
            if not 0 < len(members[j]) < min_n:
                continue
            gap = np.sqrt(((centres - centres[j]) ** 2).sum(axis=1))
            gap[j] = np.inf
            for t in np.argsort(gap, kind="stable"):
                t = int(t)
                if (members[t] and load_w[t] + load_w[j] <= big_w + 1e-9
                        and load_v[t] + load_v[j] <= big_v + 1e-9):
                    members[t] += members[j]
                    load_w[t] += load_w[j]
                    load_v[t] += load_v[j]
                    members[j], load_w[j], load_v[j] = [], 0.0, 0.0
                    merged = True
                    break
            if merged:
                break

    depot = xy[0]
    angle = np.arctan2(centres[:, 1] - depot[1], centres[:, 0] - depot[0])
    out: list[list[int]] = []
    for j in np.argsort(angle, kind="stable"):
        left = set(members[int(j)])
        seq: list[int] = []
        cur = 0
        while left:
            cand = np.fromiter(sorted(left), dtype=np.int64)
            nxt = int(cand[np.argmin(data.distance[cur, cand])])
            seq.append(nxt)
            left.remove(nxt)
            cur = nxt
        if seq:
            out.append(seq)
    return out


def centres_to_tour(data, centres, weights, coords=None, cap_w=None, cap_v=None) -> list[int]:
    """The clusters concatenated (angle order): every customer exactly once."""
    return [c for cl in centres_to_clusters(data, centres, weights, coords, cap_w, cap_v)
            for c in cl]


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
@dataclass
class _Particle:
    position: np.ndarray          # (K, 3): x, y, weight
    velocity: np.ndarray
    pbest_position: np.ndarray
    pbest_score: float
    pbest_solution: Solution


class ClusterPSO:
    def __init__(self, data: ProblemData, cfg: ClusterPSOConfig = ClusterPSOConfig()) -> None:
        self.data = data
        self.cfg = cfg
        self.config = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n = data.n_customers
        self.coords = _box_coords(data)

        self.fast_decoder = replace(DecoderConfig(), use_greedy_baseline=False, assignment_swaps=True)
        self.final_decoder = replace(DecoderConfig(), use_greedy_baseline=True, assignment_swaps=True)

        self.evaluations = 0
        self.polish_evaluations = 0
        self.attraction_moves = 0
        self.history: list[dict] = []
        self.credits = {s: 0.0 for s in SOURCES}

        self.particles: list[_Particle] = []
        self.k = 0
        self.cap_w: np.ndarray | None = None
        self.cap_v: np.ndarray | None = None
        self.gbest_position: np.ndarray | None = None
        # The swarm's own best (what particles are attracted to).  It can be
        # worse than the overall best plan, which may still be a construction
        # plan: those plans' centres decode differently and would mislead.
        self.sbest_position: np.ndarray | None = None
        self.sbest_score = np.inf
        self.gbest_solution: Solution | None = None
        self.gbest_score = np.inf
        self.best_feasible_solution: Solution | None = None
        self.best_feasible_score = np.inf
        self._best_feasible_objective = np.inf
        self._iteration = 0
        self._last_improvement = 0
        self._start = 0.0

    # -- evaluation ------------------------------------------------------
    def _evaluate_tour(self, tour: list[int]) -> tuple[float, Solution]:
        """Same scoring as ``MemeticPSO._evaluate_tour``."""
        sol = decode_tour(self.data, tour, self.fast_decoder)
        self.evaluations += 1
        polished = improve_solution_trips(self.data, sol, self.fast_decoder, passes=1)
        if polished.objective < sol.objective - _EPS:
            sol = polished
        self._update_feasible(sol)
        return sol.objective, sol

    def _evaluate_position(self, pos: np.ndarray) -> tuple[float, Solution]:
        """One evaluation: every cluster is a trip (Split only cuts a cluster
        that fits no vehicle), then the shared assignment and in-trip 2-opt."""
        clusters = centres_to_clusters(
            self.data, pos[:, :2], pos[:, 2], self.coords, self.cap_w, self.cap_v)
        segments = [seg for cl in clusters
                    for seg in split_dp(self.data, cl, self.fast_decoder, enforce_min=False)]
        sol = assign_trips(self.data, segments, self.fast_decoder, method="cluster-pso")
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

    def _swarm_best(self, score: float, position: np.ndarray) -> None:
        if score < self.sbest_score - _EPS:
            self.sbest_score = float(score)
            self.sbest_position = position.copy()

    def _offer(self, sol: Solution, position: np.ndarray | None, source: str | None) -> bool:
        self._update_feasible(sol)
        if sol.objective < self.gbest_score - _EPS:
            if source is not None:
                self.credits[source] += float(self.gbest_score - sol.objective)
            self.gbest_score = float(sol.objective)
            self.gbest_solution = sol
            if position is not None:
                self.gbest_position = position.copy()
            self._last_improvement = self._iteration
            return True
        return False

    @staticmethod
    def _solution_tour(sol: Solution) -> list[int]:
        return [c for t in sol.trips for c in t.customers]

    # -- initialisation --------------------------------------------------
    def _start_tours(self) -> list[list[int]]:
        tours = [
            nearest_neighbor_tour(self.data),
            sweep_tour(self.data),
            volume_aware_sweep_tour(self.data),
            volume_aware_nearest_neighbor_tour(self.data),
        ]
        for _ in range(2):
            tours.append(randomized_nearest_neighbor_tour(self.data, self.rng, k=3))
        return tours

    def _plan_centres(self, sol: Solution) -> np.ndarray:
        """(K, 3) centres of a plan's trips (largest volume first), padded
        with random customers' positions; weights 1."""
        trips = sorted(sol.trips, key=lambda t: -t.metrics.volume_m3)
        pts = [self.coords[list(t.customers)].mean(axis=0) for t in trips[: self.k]]
        while len(pts) < self.k:
            c = int(self.rng.integers(1, self.n + 1))
            pts.append(self.coords[c].copy())
        return np.column_stack([np.array(pts), np.ones(self.k)])

    def _clip(self, pos: np.ndarray) -> np.ndarray:
        pos[:, :2] = np.clip(pos[:, :2], -0.1, 1.1)
        pos[:, 2] = np.clip(pos[:, 2], 0.3, 3.0)
        return pos

    def _init_swarm(self) -> None:
        plans = []
        for tour in self._start_tours():
            score, sol = self._evaluate_tour(tour)
            plans.append(sol)
        best = min(plans, key=lambda s: s.objective)
        self.k = max(1, len(best.trips) + self.cfg.extra_centres)
        # Fleet-aware slots: each centre has the capacity of the vehicle that
        # ran the matching trip of the best plan; extra slots the smallest.
        trips = sorted(best.trips, key=lambda t: -t.metrics.volume_m3)[: self.k]
        small = int(np.argmin(self.data.vehicle_usable_volume))
        slots = [t.vehicle_index for t in trips] + [small] * (self.k - len(trips))
        self.cap_w = np.array([float(self.data.vehicle_weight_cap[v]) for v in slots])
        self.cap_v = np.array([float(self.data.vehicle_usable_volume[v]) for v in slots])
        bases = [self._plan_centres(s) for s in plans]
        self._offer(best, bases[plans.index(best)], None)

        for i in range(self.cfg.swarm_size):
            if i < len(bases):
                pos = bases[i].copy()
            else:
                pos = bases[int(self.rng.integers(0, len(bases)))].copy()
                sigma = float(self.rng.uniform(0.02, 0.08))
                pos[:, :2] += self.rng.normal(0.0, sigma, (self.k, 2))
                pos[:, 2] += self.rng.normal(0.0, 0.1, self.k)
            pos = self._clip(pos)
            vel = self.rng.uniform(-0.01, 0.01, pos.shape)
            score, sol = self._evaluate_position(pos)
            self.particles.append(_Particle(pos, vel, pos.copy(), score, sol))
            self._swarm_best(score, pos)
            self._offer(sol, pos, None)

    # -- one step ----------------------------------------------------------
    def _step(self, p: _Particle) -> None:
        cfg = self.cfg
        shape = p.position.shape
        if cfg.freeze_swarm:
            p.velocity = cfg.inertia * p.velocity + self.rng.normal(0.0, cfg.jitter, shape)
        else:
            r1 = self.rng.random(shape)
            r2 = self.rng.random(shape)
            p.velocity = (
                cfg.inertia * p.velocity
                + cfg.c1 * r1 * (p.pbest_position - p.position)
                + cfg.c2 * r2 * (self.sbest_position - p.position)
            )
            self.attraction_moves += 1
        vmax = cfg.v_max_frac
        np.clip(p.velocity, -vmax, vmax, out=p.velocity)
        p.position = self._clip(p.position + p.velocity)
        score, sol = self._evaluate_position(p.position)
        if score < p.pbest_score - _EPS:
            p.pbest_score = score
            p.pbest_solution = sol
            p.pbest_position = p.position.copy()
        self._swarm_best(score, p.position)
        self._offer(sol, p.position, "swarm")

    # -- main loop -----------------------------------------------------------
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
            if cfg.max_seconds > 0 and perf_counter() - self._start > cfg.max_seconds:
                break
            for p in self.particles:
                self._step(p)
            iterations_run = iteration
            self._record(iteration)
            if cfg.verbose and (iteration % cfg.log_interval == 0 or iteration == 1):
                self._log(iteration)
            if cfg.patience > 0 and iteration - self._last_improvement >= cfg.patience:
                stopped_early = True
                break

        self._final_polish()
        self._record(iterations_run + 1)
        return PSOResult(
            best_solution=self.gbest_solution,
            best_score=self.gbest_score,
            best_feasible_solution=self.best_feasible_solution,
            best_feasible_score=self.best_feasible_score,
            history=self.history,
            runtime_sec=perf_counter() - self._start,
            evaluations=self.evaluations,
            seed_solution=None,
            iterations_run=iterations_run,
            stopped_early=stopped_early,
        )

    def _final_polish(self) -> None:
        """Same as Option B: local search, in-trip 2-opt, re-decode."""
        best = self.gbest_solution
        helper = MemeticPSO(self.data, PSOConfig(seed=self.cfg.seed, swarm_size=1, verbose=False))
        tour = self._solution_tour(best)
        _, _, sol = helper._local_search(tour, best, self.cfg.final_local_search_budget)
        self.polish_evaluations += helper.evaluations
        self._offer(sol, None, "final_polish")

        polished = improve_solution_trips(self.data, self.gbest_solution, self.final_decoder, passes=3)
        self.polish_evaluations += 1
        self._offer(polished, None, "final_polish")

        tour = self._solution_tour(self.gbest_solution)
        redecoded = decode_tour(self.data, tour, self.final_decoder)
        self.polish_evaluations += 1
        self._offer(redecoded, None, "final_polish")

    def _record(self, iteration: int) -> None:
        sol = self.gbest_solution
        feas = self.best_feasible_solution
        self.history.append({
            "iteration": iteration,
            "best_score": float(self.gbest_score),
            "best_true_cost": float(sol.true_cost_eur) if sol else np.nan,
            "best_penalty": float(sol.penalty) if sol else np.nan,
            "best_feasible_cost": float(self.best_feasible_score) if feas is not None else np.nan,
            "evaluations": self.evaluations,
            "credits": dict(self.credits),
        })

    def _log(self, iteration: int) -> None:
        print(
            f"iter {iteration:4d}  best={self.gbest_score:10.2f}  "
            f"evals={self.evaluations:6d}  {perf_counter() - self._start:6.1f}s"
        )
