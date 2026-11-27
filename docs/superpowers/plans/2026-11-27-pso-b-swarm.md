# Option B swarm improvements: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the ring topology, the inertia schedule and restarts to Option B as opt-in config fields. Run the variants on the tuning days and adopt the winners as B's defaults.

**Architecture:**
- New fields go in `DiscretePSOConfig`. Their defaults reproduce today's B bit for bit.
- Variants are named presets in `experiments/b_variants.py`. `compare_methods.run_job` resolves names it doesn't know from there, so the existing runner, report and Wilcoxon tests are reused.

**Tech stack:** Python, numpy, unittest, the existing `experiments/compare_methods.py`.

**Spec:** `docs/superpowers/specs/2026-11-27-pso-b-swarm-design.md`

## Global constraints

- These stay byte-identical: `src/pso.py`, `decoder.py`, `cost.py`, `kernels.py`, `data_loader.py`, `main.py`.
- No local search inside B's loop. The final polish stays as it is.
- Tuning uses only `instances_osm/road_tuning`, seeds 1–3, budget 20,000.
- The default `DiscretePSOConfig(swarm_size=6, seed=1, max_evaluations=300, final_local_search_budget=30)` on `data/osm` must give `best_score == 1510.8894215600003`, 300 evaluations, 317 attraction moves and 49 iterations.

## Review focus

1. **Ring with a small swarm:** with `neighbourhood >= swarm_size // 2`, the ring must still work, with no duplicate indices and no crash.
2. **Restarts:** must never discard the particle holding the global best. The global best must never get worse.
3. **Inertia schedule:** must not go outside [0, 1], including when the budget is overshot in the last iteration.
4. **Unknown variant names:** must be recorded as a row error, not crash the runner.
5. **Defaults:** the default config must reproduce the fingerprint exactly. Watch for extra rng draws on default paths.

---

### Task 1: Opt-in swarm mechanisms in `src/pso_discrete.py`

**Files:** modify `src/pso_discrete.py`; test in `tests/test_pso_discrete.py`.

**Produces:**
- New `DiscretePSOConfig` fields:
  - `topology: str = "global"`
  - `neighbourhood: int = 2`
  - `inertia_end: float | None = None`
  - `restart_after: int = 0`
  - `restart_moves: int = 20`
- `ring_neighbours(i: int, n: int, k: int) -> list[int]`
- `DiscretePSO._inertia_now() -> float`
- `DiscretePSO.restarts: int`

- [ ] **Step 1:** write the failing tests.
  - `test_default_fingerprint`: the values in Global constraints.
  - `test_ring_neighbours`:
    - `ring_neighbours(0, 6, 2) == [4, 5, 0, 1, 2]`
    - `ring_neighbours(1, 3, 5)` is sorted `[0, 1, 2]`
  - `test_ring_topology_run`: a valid permutation, deterministic.
  - `test_inertia_schedule`: with `inertia=0.6, inertia_end=0.1`, the value is 0.6 at 0 evaluations, 0.35 at half the budget, and 0.1 at and beyond the budget.
  - `test_restarts_happen_and_gbest_monotone`: `restart_after=1`, so `restarts > 0` and the history's `best_score` never rises.
  - `test_unknown_topology_raises`: `ValueError`.
- [ ] **Step 2:** run `python -m unittest tests.test_pso_discrete -v`. Expected: the new tests fail and the fingerprint test passes.
- [ ] **Step 3:** implement. No new rng draws when every field is at its default.
- [ ] **Step 4:** run the same command. Expected: everything passes, including the fingerprint test.
- [ ] **Step 5:** commit: "B: opt-in ring topology, inertia schedule, restarts".

### Task 2: Variant registry for the comparison runner

**Files:** create `experiments/b_variants.py`; modify `experiments/compare_methods.py` (`run_job` lookup); test in `tests/test_compare_methods.py`.

**Produces:**
- `VARIANTS: dict[str, dict]`, mapping a name such as `"B_ring"` to config overrides.
- `run_variant(name, data, seed, budget, freeze)`
- `compare_methods.resolve_method(name)`, which raises `KeyError` for unknown names.

- [ ] **Step 1:** write the failing tests.
  - `test_variant_method_runs`: the tiny instance with `B_ring` gives a row without error.
  - `test_unknown_method_row_error`: `"nope"` gives a `KeyError` in the row's error field.
- [ ] **Step 2:** run them and see them fail.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run `python -m unittest tests.test_compare_methods -v`. Expected: passes.
- [ ] **Step 5:** commit.

### Task 3: Tuning rounds (experiments, no product code)

Each round uses `python -m experiments.compare_methods --name tune_b_r<N> --instances instances_osm/road_tuning --methods <variants> --seeds 1 2 3 --ablation-instances 0`.

1. **Round 1:** `B_discrete`, `B_ring` (k=1, 2, 4).
2. **Round 2:** the round-1 winner with the inertia schedule (start 0.6 / 0.4 → end 0.05 / 0.1).
3. **Round 3:** the round-2 winner with restarts (after 10 / 25 iterations).
4. **Round 4:** a grid over `c1`/`c2` ∈ {1.0, 1.5}, `mutation_moves` ∈ {1, 2, 3} and `swarm_size` ∈ {20, 40}, pruned to at most 8 variants.

Keep rule: from the spec. Record each round's table in the ledger.

### Task 4: Adopt the winner

**Files:** modify `src/pso_discrete.py` (defaults and docstring table); `tests/test_pso_discrete.py` (update the fingerprint to the new defaults and record the old one in the docstring).

- [ ] Update the defaults and the docstring with the tuning table.
- [ ] Run the full suite: `python run_tests.py`.
- [ ] Commit.
- [ ] Hand over the full comparison command (`--name main_v2`, overnight) to the user.
