# Making the swarm do the work: current vs discrete PSO (B) vs cluster-centre PSO (C)

Date: 2026-11-27 · Status: draft for approval

## Why

An independent evaluation (docs/evaluation/2026-11-27-project-evaluation.md)
measured that in the current random-key memetic PSO the swarm contributes
almost nothing: on seed 42 swarm moves improved the best plan by 1.03 EUR,
local search on the global best by 135.2 EUR, and a frozen swarm
(inertia = c1 = c2 = 0) was as good as the real one. The project is a PSO
project, so the swarm must do most of the work, and this must be shown with
numbers.

## What the user asked for

- Build **Option C** (cluster-centre PSO): the swarm decides which customers
  ride together.
- Build **Option B** (discrete PSO on routes, path relinking) **in parallel**.
- Keep the problem fixed: **250 customers**, all served, the instructor's
  rules (5 vehicles, 08:00–24:00, 2x overtime after 8 h) and today's cost
  model, unchanged.
- A **side-by-side comparison report: current vs B vs C**.

## Goals

- Two new optimisers next to the current one, sharing the decoder, the cost
  model and the checks.  **The original algorithm is not modified** (user's
  decision): `src/pso.py`, `src/decoder.py`, `src/cost.py`, `src/kernels.py`
  and `main.py` stay byte-identical; the current `MemeticPSO` is the
  baseline.
- A fair comparison: same instances, same seeds, **same evaluation budget**,
  same final polish.
- Proof of the swarm's role for every method: share of the improvement that
  comes from swarm moves, and a frozen-swarm ablation.
- One report (HTML + CSV) comparing the three.

## Non-goals

- No change to the cost model, the constraints, the decoder's results, or
  the instructor's rules.
- No new tuning of the current method (it runs with today's defaults).
- Not replacing the default method in `main.py` yet: the report decides,
  then the user decides.

## Anti-goals

- A "PSO" where local search still does the work: B and C get **no big
  local search inside the loop** (only the per-candidate in-trip 2-opt
  polish every method already uses) and the same short final polish as the
  current method.
- Comparing at unequal budgets or on the instance used to pick settings.

## Approaches (chosen by the user)

- **A. Repair the current random-key PSO** (Lamarckian keys, LS on every
  particle, lbest topology): cheap, but local search would still do most of
  the work. Not built.
- **B. Discrete PSO with path relinking** (built, parallel session).
- **C. Cluster-centre PSO** (built here; the user's main choice).

## Shared contract

All three optimisers live in `src/` and expose the same call:

```python
result = optimiser.run()          # -> PSOResult (src/pso.py), unchanged type
```

with `PSOResult.history` rows carrying, in addition to today's keys,
`improvement_source` counters (see below). Each method is constructed as
`Method(data, cfg)` with its own config dataclass that has at least
`swarm_size, seed, max_evaluations, patience (0 = off), max_seconds (0 = off),
freeze_swarm (bool), verbose`.

**Evaluation** = one call of the shared plan builder (giant tour →
`split_dp` → `assign_trips`, plus the in-trip 2-opt polish), counted by the
same counter in every method, local-search moves included. The run stops at
`max_evaluations`; the final polish (identical for all three: the current
`_final_polish` logic, budget 150) runs after the budget and is reported
separately.

**Improvement source** — every time the best plan improves, the gain in EUR
is credited to one of: `construction` (initial plans), `swarm` (a particle's
new position after a velocity/relinking/centre move), `local_search`
(explicit local search inside the loop), `final_polish`. For the current
method this is measured from the outside: a subclass
`experiments/instrumented_pso.py::InstrumentedMemeticPSO(MemeticPSO)` wraps
its methods to record the credits without changing what they do; its results
must be bit-identical to `MemeticPSO` (fingerprint test).

**Frozen swarm** (`freeze_swarm=True`): particles do not move toward pbest or
gbest (current: inertia = c1 = c2 = 0; B: no relinking, only the random
mutation; C: no attraction, only the random jitter). Everything else equal.

## Option C — cluster-centre PSO (`src/pso_cluster.py`)

- **Particle:** K trip centres in the plane (lat, lon), K = number of trips
  of the best construction plan + 2 (≈ 17–20), plus one priority weight per
  centre: ≈ 3K ≈ 55 numbers instead of 250 keys. Continuous PSO (inertia,
  c1, c2, velocity clamp) on these numbers.
- **Decode:** centres are visited in polar-angle order around the depot;
  each customer is assigned to the nearest centre (road-time distance from
  the centre's nearest customer, weighted by the centre's priority) that
  still has room for its weight and volume in the largest vehicle; customers
  that fit nowhere go to the nearest centre (Split will cut). Inside each
  cluster customers are ordered by nearest neighbour; clusters are
  concatenated in angle order into a giant tour → the shared plan builder
  (`split_dp` + `assign_trips` + in-trip 2-opt). All 250 customers are always
  in the tour.
- **Initialisation:** centres of the trips of the existing construction
  heuristics, plus k-means-style random spreads of customers; noise so the
  swarm is diverse.
- **Why the swarm matters:** moving a centre a little moves a few customers
  between trips, a small, meaningful change; the swarm decides the grouping,
  the hardest part.

## Option B — discrete PSO with path relinking (`src/pso_discrete.py`)

- **Particle:** a giant tour (permutation of the 250 customers).
- **Move:** with inertia w apply k random swap/insert moves (mutation); with
  probability c1·r1 relink toward pbest, with c2·r2 toward gbest — copy a
  random segment of the guide's tour into the particle and repair the rest in
  the particle's order (order crossover); keep the result (no local search).
- **Initialisation:** the current method's construction tours, perturbed.
- Evaluated by the shared plan builder.

## Comparison

- **Instances:** `data/osm` + the 20 `road_validation` instances (seeds
  6001–6020, all 250 customers, built with `experiments.make_road_instances
  --road-dir data/osm --set road_validation --out instances_osm`). No method
  is tuned on them.
- **Seeds:** 1–5 per instance and method.
- **Budget:** 20,000 evaluations per run, no patience, no time limit (equal
  budget; runtime reported).
- **Settings:** current = today's `PSOConfig` defaults; B and C = sensible
  defaults set once on `instances_osm/road_tuning` (not on the comparison
  instances), stated in the report.
- **Ablation:** each method also runs with `freeze_swarm=True` on `data/osm`
  + 5 validation instances, seeds 1–5.
- Script `experiments/compare_methods.py`: parallel, resumable `runs.csv`
  (one row per run), outputs in `comparison/<name>/`.

## Report (`comparison/<name>/report.html` + CSVs)

- Headline table per method: median / mean / best / worst real cost, median
  gap to the best known per instance, feasibility rate, median runtime.
- Head-to-head: wins per instance, Wilcoxon signed-rank (B vs current, C vs
  current, C vs B).
- Charts: cost-gap distribution per method; convergence (best cost vs
  evaluations, median over runs) per method; **share of improvement by
  source** (construction / swarm / local search / final polish) per method;
  frozen vs normal swarm per method.
- Plain-language conclusion and a note on fairness (same budget, unseen
  instances, same final polish).
- Charts follow the project's validated palette (dataviz skill).

## Parallel work

- This session: the instrumented subclass of the current method, Option C
  (`src/pso_cluster.py`, `tests/test_pso_cluster.py`), the comparison script
  and the report.
- A separate session in its own git worktree (branch from the same commit):
  Option B only, against the contract above, with its own tests; files
  `src/pso_discrete.py`, `tests/test_pso_discrete.py`. It does not touch
  other files.
- When B is done, the user is asked before its branch is merged here; then
  the comparison runs.

## Testing

- Current method: bit-identical results before/after instrumentation
  (fingerprint, as for the Numba change); improvement credits sum to
  (start cost − final cost).
- B and C: every decode serves all 250 customers once, capacity respected
  (`check_solution`), deterministic for a seed, stop exactly at the
  evaluation budget, frozen swarm really freezes (no attraction moves),
  credits sum to the total improvement.
- Comparison script: resume, CSV columns, report files on a tiny run.

## Left to the builder

Exact PSO coefficients of B and C (set once on road_tuning, documented),
the number of mutation moves in B, the k-means details in C, chart styling.
