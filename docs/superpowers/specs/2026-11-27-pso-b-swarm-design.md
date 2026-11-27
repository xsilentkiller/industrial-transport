# Option B: swarm-only improvements: short design

**Branch:** `pso-b-swarm` (from `pso-representations`, commit bc3c77c)
**Date:** 2026-11-27

## Intent

The three-way comparison (`comparison/main/report.html`) found:

- **Current method:** median objective 1,408 €. It is cheapest, but local search does 97 % of its improvement.
- **Option B** (discrete PSO with path relinking): median 1,438 € (+1.6 %). The swarm does 91 % of its improvement, and freezing the swarm costs 57 €.

The user's goal is swarm intelligence. So the aim is to close B's gap to the current method using swarm mechanisms only.

## Goals

- **Mean objective:** lower B's mean objective on the tuning days. The target is within about 1 % of the current method.
- **Swarm share:** keep the swarm's share of the improvement above 80 %.

## Non-goals and anti-goals

- **Non-goals:**
  - no local search inside the loop
  - no change to the original algorithm (`src/pso.py`, `decoder.py`, `cost.py`, `kernels.py`, `data_loader.py`, `main.py`)
  - no change to the instructor's rules (16 h day, 2x overtime, 250 customers, 5 vans)
- **Anti-goals:**
  - gains that come from the final polish rather than the swarm
  - tuning on `data/osm` or on the validation days, because the full comparison must stay a fair test on unseen days

## Approach (picked by the user: "go B")

The user picked this approach over two alternatives:
- **Option C:** B beat it on 87 of 105 runs.
- **Option A** (current method with local search off): random keys carry little order information.

Four swarm mechanisms are added to `src/pso_discrete.py`. Each is a new config field. With the default values, B runs exactly as it does today, so the published comparison stays reproducible.

1. **Topology:** `topology = "global" | "ring"` and `neighbourhood = 2` (neighbours on each side).
   - With `ring`, the social guide is the best pbest among particles i−k … i+k.
2. **Inertia schedule:** `inertia_end: float | None = None`.
   - When set, the mutation probability moves linearly from `inertia` to `inertia_end` as the loop budget is used up.
3. **Restarts:** `restart_after = 0` (off) and `restart_moves = 20`.
   - A particle restarts when its pbest has not improved for `restart_after` iterations and it does not hold the global best.
   - It moves to a copy of a random other particle's pbest, mutated by `restart_moves` moves, and its pbest is reset to that tour.
   - Gains from restarts are credited to the swarm.
4. **Coefficients:** a grid over `c1`, `c2`, `mutation_moves` and `swarm_size`, done in experiments only, with no code change.

## Experiment protocol

- **Runs:**
  - Variants are registered as comparison methods named `B_<variant>`, so `experiments/compare_methods.py` runs them. It looks them up in a new `experiments/b_variants.py`.
  - Days: `instances_osm/road_tuning` (7001–7004), seeds 1–3, budget 20,000, about 10 minutes per batch.
  - Baseline: `comparison/tune_b_baseline` (current and B, same days and seeds).
- **Keep rule:** keep a mechanism if its mean objective beats the variant it builds on, paired over the same 12 runs, and the swarm's share stays above 80 %.
- **Stacking:** greedy, in the order topology → inertia schedule → restarts → coefficient grid.
- **Finally:**
  - set B's defaults to the winner, with the table in B's docstring
  - re-run the full comparison once (overnight) as `comparison/main_v2`

## Left to the builder

- exact grid values
- restart source details
- report labels for the variants

## Not in this design

Making B selectable in `main.py` and updating the README. These are separate steps, to ask the user about afterwards.
