# PSO Representations (B, C) + Side-by-side Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two PSO variants in which the swarm does the work — B (discrete PSO with path relinking) and C (cluster-centre PSO) — and compare them side by side with the current, untouched `MemeticPSO` at an equal evaluation budget on unseen 250-customer days.

**Architecture:** New files only. `src/pso_discrete.py` (B, built in a parallel session) and `src/pso_cluster.py` (C) each define a config dataclass and an optimiser whose `run()` returns the existing `PSOResult`. The current method is measured through `experiments/instrumented_pso.py` (a subclass that records improvement credits without changing behaviour). `experiments/compare_methods.py` runs all methods in parallel on the validation instances and writes `comparison/<name>/` (runs.csv, summary.csv, charts, report.html).

**Tech Stack:** Python 3.13, NumPy, Numba (existing kernels), Matplotlib, SciPy (Wilcoxon; add to requirements.txt).

**Spec:** `docs/superpowers/specs/2026-11-27-pso-representations-design.md`

## Global Constraints

- `src/pso.py`, `src/decoder.py`, `src/cost.py`, `src/kernels.py`, `src/data_loader.py`, `main.py` stay byte-identical.
- 250 customers, all served; instructor's rules and today's cost model unchanged.
- Evaluation = one `decode_tour(data, tour, fast_decoder)` (+ in-trip `improve_solution_trips(..., passes=1)` polish), with `fast_decoder = replace(DecoderConfig(), use_greedy_baseline=False, assignment_swaps=True)` exactly as `MemeticPSO`. Counted in `self.evaluations`.
- Budget: `max_evaluations = 20_000`; no patience, no time limit in the comparison. Final polish identical for all methods (the logic of `MemeticPSO._final_polish`, `final_local_search_budget = 150`), run after the budget, credited to `final_polish`.
- B and C: no local search inside the loop.
- `PSOResult.history` rows: today's keys + `"credits": {"construction", "swarm", "local_search", "final_polish"}` (EUR, cumulative).
- Config fields every method has: `swarm_size, seed, max_evaluations, patience, max_seconds, freeze_swarm, verbose`.
- Comparison instances: `data/osm` + `instances_osm/road_validation/*` (20, seeds 6001–6020); seeds 1–5.

## Review Focus

1. A cluster that receives no customers (empty centre) → no empty trip, no crash; all 250 customers still in the tour. (Task 2 test.)
2. Two centres at the same point / all customers to one centre → valid tour, Split cuts it. (Task 2 test.)
3. `max_evaluations` smaller than the swarm size → run stops cleanly with a valid best plan. (Tasks 2, B tests.)
4. Comparison resumed after an interruption → no duplicate rows, finished runs skipped. (Task 4 test.)
5. A method that crashes on one instance → the row records the error, the comparison and report still finish. (Task 4 test.)

---

### Task 1: Instrumented current method (no change to `src/pso.py`)

**Files:** Create `experiments/instrumented_pso.py`, `tests/test_instrumented_pso.py`

**Interfaces:**
- Produces: `class InstrumentedMemeticPSO(MemeticPSO)` with `max_evaluations: int` (constructor kwarg, 0 = off: stops the loop at the end of the iteration that reaches it) and `freeze_swarm: bool` (runs with `replace(cfg, inertia=0, c1=0, c2=0)`); history rows gain `"credits"`. `run_method(name, data, seed, max_evaluations, freeze) -> PSOResult` dispatcher lives in Task 4.

- [ ] Step 1: tests — (a) with `max_evaluations=0` and the same config, `InstrumentedMemeticPSO(...).run()` gives the same `best_feasible_solution.true_cost_eur`, trips and `history` costs as `MemeticPSO(...).run()` (data/base, swarm 4, iterations 3); (b) credits in the last history row sum to `history[0].best_score − final best_score` within 1e-6; (c) `max_evaluations=300` stops with `evaluations` < 300 + one iteration's worth.
- [ ] Step 2: run → FAIL (module missing). Step 3: implement by overriding `_update_global` (credit "swarm"), the memetic block via `_local_search` wrapper (credit "local_search"), `_seed_incumbents`/`_init_particles` (credit "construction"), `_final_polish` (credit "final_polish"), `_record` (add credits). Step 4: PASS.

### Task 2: Option C — cluster-centre PSO

**Files:** Create `src/pso_cluster.py`, `tests/test_pso_cluster.py`

**Interfaces:**
- Produces: `ClusterPSOConfig(swarm_size=20, seed=42, max_evaluations=20_000, patience=0, max_seconds=0.0, freeze_swarm=False, verbose=False, inertia=0.72, c1=1.49, c2=1.49, v_max_frac=0.1, extra_centres=2, final_local_search_budget=150)`, `class ClusterPSO: __init__(data, cfg); run() -> PSOResult`; `centres_to_tour(data, centres, weights) -> list[int]`.
- Decode per spec: K centres (lat, lon) + K priority weights; polar-angle order around the depot; each customer (most constrained first: larger volume first) to the centre minimising `weight_k × haversine(customer, centre)` among centres with room (largest vehicle weight/usable volume), else the nearest centre; inside a cluster nearest-neighbour order from the depot; concatenate → shared evaluation.
- Init: centres of the trips of the construction plans (`MemeticPSO`'s construction tours decoded once each), K = trips of the best + `extra_centres`; missing centres at random customers; Gaussian noise; weights 1 ± 0.1.
- Freeze: velocity update without the c1/c2 terms, only inertia × random jitter.

- [ ] Step 1: tests — every decoded tour is a permutation of 1..250 (data/osm, 50 random particles); empty cluster OK; identical centres OK; `check_solution` empty for the run's best; deterministic per seed; stops at the budget (≤ budget + swarm_size); credits sum to total improvement; frozen run makes no attraction moves (c1/c2 terms zero — assert via a counter).
- [ ] Step 2: FAIL. Step 3: implement. Step 4: PASS; quick run on road_tuning to set coefficients once (record values in the module docstring).

### Task 3 (parallel session): Option B — discrete PSO with path relinking

**Files:** Create `src/pso_discrete.py`, `tests/test_pso_discrete.py` (only these).

**Interfaces:** `DiscretePSOConfig(swarm_size=20, seed=42, max_evaluations=20_000, patience=0, max_seconds=0.0, freeze_swarm=False, verbose=False, inertia=0.3, c1=0.6, c2=0.6, final_local_search_budget=150)`, `class DiscretePSO: __init__(data, cfg); run() -> PSOResult` — same contract as above. Move: inertia → k random swap/insert moves; with prob c1·r1 order-crossover segment from pbest, with c2·r2 from gbest; no local search in the loop. Tests mirror Task 2.

### Task 4: Comparison script + report

**Files:** Create `experiments/compare_methods.py`, `tests/test_compare_methods.py`; modify `requirements.txt` (`scipy>=1.11`).

**Interfaces:** `METHODS = {"current", "B_discrete", "C_cluster"}`; `run_job(job) -> dict` (row: method, instance, seed, freeze, real_cost, penalty, feasible, evaluations, runtime_s, credit_construction, credit_swarm, credit_local_search, credit_final_polish, start_cost, error); CLI `--name --instances --seeds --budget --workers --ablation-instances 5 --quick`; outputs `comparison/<name>/runs.csv, summary.csv, curves.json, *.png, report.html`.

- [ ] Step 1: tests — tiny run (2 synthetic 30-customer instances via `experiments.make_instances.build_instance`, budget 200, seeds 1) writes all files; resume skips done rows; a method raising an exception records `error` and the report still builds.
- [ ] Step 2: FAIL. Step 3: implement (ProcessPoolExecutor, resumable CSV, Wilcoxon via scipy, charts per dataviz palette). Step 4: PASS.
- [ ] Step 5: build `road_validation` instances; run the full comparison (`--name main`); open the report and check it.

### Task 5: README section, run_tests registry, full suite

- [ ] README "Comparison: current vs B vs C" (method summaries, fairness, headline results from the run); `run_tests.py` ABOUT entries; `python run_tests.py` all pass; byte-identity check of the untouched files (`git diff --stat HEAD -- src/pso.py src/decoder.py src/cost.py src/kernels.py src/data_loader.py main.py` empty).
