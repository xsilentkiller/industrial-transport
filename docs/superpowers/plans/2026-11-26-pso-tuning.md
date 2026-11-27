# PSO Parameter Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Early stopping (patience) in the PSO plus a resumable, parallel tuning script that prunes weak combinations in rounds and saves the winner.

**Architecture:** `PSOConfig.patience` stops `MemeticPSO.run` after N iterations without improvement. `experiments/tune.py` runs (combination, instance, seed) jobs in a process pool calling `MemeticPSO` directly, appends each result to `runs.csv`, scores runs by gap to the best known cost per instance, keeps the best combinations per round, and writes `summary.csv`, `tuning.png`, `best_config.json`.

**Tech Stack:** Python 3.13, numpy, matplotlib, concurrent.futures; scipy optional (Wilcoxon).

**Spec:** `docs/superpowers/specs/2026-11-26-pso-tuning-design.md`

## Global Constraints

- User instruction: **no commits, no push.** Commit steps are replaced by "leave uncommitted".
- `patience = 0` must leave current PSO behaviour byte-identical (all existing tests pass).
- Default grid: swarm 10 20 30 40 · patience 10 20 40 · ls_interval 1 5 · max iterations 300.
- Baseline: swarm 20, iterations 60, ls_interval 5, patience 0 — never pruned.
- Rounds: seeds per round 1, 2, 2; keep 0.5 then 4. Per-run time limit 600 s. Workers 7.
- Gap = (real_cost / best_known − 1) × 100; infeasible → inf; tie within 0.5 pp → lower median runtime.
- Must scale to any number of instances (user will run 100 later).
- Output: `tuning/<name>/` (default name `run`); `output/` untouched.

## Review Focus

- Interrupted run, then rerun with the same command → finished runs are skipped, not duplicated.
- Rerun with a different grid into the same folder → existing rows for combinations no longer in the grid are ignored, not crashing.
- An instance folder without `params.json` inside `--instances` → skipped with a printed note.
- A run that finds no feasible plan → gap inf, ranked last, does not crash the median/plot.
- A worker exception (e.g. bad instance) → row recorded with `error`, other runs continue.

---

### Task 1: Patience (early stopping) in the PSO

**Files:**
- Modify: `src/pso.py` (`PSOConfig`, `PSOResult`, `MemeticPSO.run`)
- Modify: `main.py` (`--patience` flag, `run_pso` config + payload)
- Test: `tests/test_convergence.py`

**Interfaces:**
- Produces: `PSOConfig.patience: int = 0`; `PSOResult.iterations_run: int`; `PSOResult.stopped_early: bool`.

- [ ] **Step 1: Write failing tests** in `tests/test_convergence.py`, class `TestPatience`, on a 30-customer instance built with `experiments.make_instances.build_instance("tiny", 30, 1, tmp)`:
  - `test_patience_stops_early`: `PSOConfig(swarm_size=4, iterations=200, patience=5, local_search_interval=5, verbose=False)` → `result.stopped_early is True`, `result.iterations_run < 200`, `len(result.history) == result.iterations_run + 1`.
  - `test_patience_zero_runs_all`: same with `patience=0, iterations=8` → `iterations_run == 8`, `stopped_early is False`.
- [ ] **Step 2: Run** `python -m unittest tests.test_convergence -v` → the two new tests fail (AttributeError).
- [ ] **Step 3: Implement.** After `self._record(iteration)` in the loop: `if cfg.patience > 0 and iteration - self._last_improvement >= cfg.patience: stopped = True; break` (log the final iteration first if verbose). Track `iterations_run = iteration`. Final polish still runs. `main.py`: `--patience` (default `pso.patience`), pass into `PSOConfig`, add `"patience"`, `"iterations_run"`, `"stopped_early"` to payload `config`.
- [ ] **Step 4: Run** `python run_tests.py` → all pass.

### Task 2: Tuning core (jobs, scoring, pruning) — `experiments/tune.py`

**Files:**
- Create: `experiments/tune.py`
- Test: `tests/test_tune.py`

**Interfaces:**
- Consumes: `PSOConfig(patience=...)`, `PSOResult.iterations_run/stopped_early` (Task 1); `load_problem`.
- Produces:
  - `Combo(swarm: int, patience: int, ls_interval: int, iterations: int)` frozen dataclass, `.key -> str` like `"s20_p10_ls5_i300"`, `BASELINE = Combo(20, 0, 5, 60)`.
  - `find_instances(paths: list[str]) -> list[Path]` (folder with `params.json`, or parent → its subfolders that have one; sorted; prints skipped).
  - `run_job(combo_key, swarm, patience, ls_interval, iterations, instance, seed, time_limit) -> dict` (one CSV row; catches exceptions into `error`).
  - `load_runs(path) -> list[dict]`, `append_run(path, row)`.
  - `add_gaps(rows) -> list[dict]` (best known per instance over feasible rows).
  - `summarise(rows, combos) -> list[dict]` (per combo: runs, median/mean/min/max gap, median runtime, median iterations).
  - `rank(summary, tie_pp=0.5) -> list[dict]`.
  - `run_tuning(args) -> Path`.
- CSV columns (`runs.csv`): `round, combo, swarm, patience, ls_interval, max_iterations, instance, seed, real_cost, penalty, feasible, iterations_run, stopped_early, runtime_s, evaluations, error`.

- [ ] **Step 1: Write failing tests** `tests/test_tune.py` (pure functions, no PSO):
  - `test_gap_per_instance`: rows A/B on inst1 cost 100/110, inst2 cost 200/200 → gaps 0, 10, 0, 0.
  - `test_infeasible_gap_is_inf_and_ranked_last`.
  - `test_rank_tie_prefers_faster`: medians 1.0 vs 1.3 pp, runtimes 200 vs 100 → faster first.
  - `test_find_instances_parent_and_skip`: tmp parent with two valid folders + one without `params.json` → 2 paths.
  - `test_resume_keys`: `append_run` then `load_runs` round-trips; done-set excludes completed triples.
- [ ] **Step 2: Run** `python -m unittest tests.test_tune -v` → fail (module missing).
- [ ] **Step 3: Implement** the functions above. `run_job` loads the instance via a per-process cache dict, runs `MemeticPSO(data, PSOConfig(swarm_size, iterations, seed, local_search_interval, patience, max_seconds=time_limit, verbose=False)).run()`, real cost = `best_feasible_solution.true_cost_eur` (None → infeasible, cost blank).
- [ ] **Step 4: Run** tests → pass.

### Task 3: Rounds, outputs, CLI

**Files:**
- Modify: `experiments/tune.py` (`main`, `run_tuning`, `write_outputs`, `plot`)
- Test: `tests/test_tune.py`

**Interfaces:**
- CLI: `python -m experiments.tune [--instances PATH ...] [--name run] [--swarm 10 20 30 40] [--patience 10 20 40] [--ls-interval 1 5] [--max-iterations 300] [--seeds-per-round 1 2 2] [--keep 0.5 4] [--workers 7] [--time-limit 600] [--quick] [--no-baseline]`
- `--quick` ≡ `--seeds-per-round 1` (one round, no pruning).
- `--keep` values < 1 are fractions (ceil), ≥ 1 are counts; the last round's winner is rank 1.
- Writes `tuning/<name>/{runs.csv, summary.csv, tuning.png, best_config.json}`; prints a ranked table.
- `best_config.json`: `swarm_size, patience, local_search_interval, suggested_iterations` (median iterations_run of winner rounded **up** to a multiple of 5), `median_gap_pct, baseline_median_gap_pct, wilcoxon_p` (paired by instance+seed on real cost; `null` if scipy missing or < 6 pairs), `instances, seeds, runs`.

- [ ] **Step 1: Write failing test** `test_end_to_end_tiny` (two 25-customer `build_instance` folders, `--swarm 3 4 --patience 3 --ls-interval 5 --max-iterations 15 --seeds-per-round 1 1 --keep 2 1 --workers 2`): all four files exist; `runs.csv` row count = 2 combos×2 inst×1 + baseline 2 + round 2 survivors…; rerunning the same command adds 0 rows; baseline appears in every round; `best_config.json` has `suggested_iterations % 5 == 0`.
- [ ] **Step 2: Run** → fail.
- [ ] **Step 3: Implement** round loop: for round r, survivors × instances × new seeds (`seed = sum(previous seeds)+1 …`) minus done triples → pool; `as_completed` → `append_run` + progress line; then `add_gaps` over all rows of current-grid combos, `summarise`, `rank`, keep. Plot: left = horizontal bars of median gap per combo (min–max whisker, finalists highlighted, baseline in grey); right = box of iterations_run for finalists. Use the validated palette `#2a78d6, #eb6834, #1baf7a`, neutral `#8a8985`.
- [ ] **Step 4: Run** `python -m unittest tests.test_tune -v` and `python run_tests.py` → all pass.
- [ ] **Step 5: Smoke** `python -m experiments.tune --quick --swarm 10 --patience 10 --ls-interval 5 --max-iterations 20 --name smoke` on the 4 tuning instances → files written, table printed. Leave everything uncommitted.
