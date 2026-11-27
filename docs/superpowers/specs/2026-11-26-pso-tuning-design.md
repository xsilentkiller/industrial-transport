# PSO parameter tuning — design

Date: 2026-11-26 · Branch: `hard-capacity-soft-counts`

## Intent

Find the most suitable **swarm size**, **number of iterations**, **patience**
(early stopping) and **local-search interval** for the memetic PSO, by
experiment instead of by guess, and save the result so it can become the
new default and be cited in the project write-up.

## Goals

- A tuning script that loops over all combinations, runs them in parallel,
  drops weak combinations early, prints the result and saves it to files.
- Early stopping inside a PSO run (patience), off by default.
- Works for any number of instances (default: the 4 tuning instances).
- Results comparable across instances of different cost level.
- Interruptible: rerunning the same command resumes.

## Non-goals

- Changing the solver's defaults automatically (the winner is written to
  `best_config.json`; adopting it is a separate, deliberate step).
- Tuning PSO coefficients (inertia, c1, c2, v_max) or local-search budget.
- Touching `output/` or the normal `python main.py` behaviour.

## Anti-goals

- Tuning on `data/osm` (the reported instance) — that would overfit.
- A winner picked on raw EUR averages (expensive instances would dominate).
- A winner decided by one lucky seed.

## Constraints

- 8 CPU cores → 7 parallel workers by default.
- One run ≈ 60–90 ms per decoder call; a run must never exceed a per-run
  time limit (default 600 s).
- Python + numpy + matplotlib (existing deps); scipy is used for the
  Wilcoxon test only if installed (it is), otherwise the test is skipped.

## Approaches considered

1. **Tune max-iterations as a fixed grid factor** — rejected: with early
   stopping the cap is mostly a guess.
2. **Tune swarm × patience × LS interval, iterations capped high and
   measured** — **chosen**: the suitable iteration count is the median of
   iterations actually used by the winner.
3. Early stopping only inside runs, or only as pruning between rounds —
   rejected; the user chose **both**.

## Design

### 1. Early stopping in the PSO (`src/pso.py`, `main.py`)

- `PSOConfig.patience: int = 0` (0 = off, current behaviour).
- In `MemeticPSO.run`, after recording iteration *i*: if `patience > 0` and
  `i - self._last_improvement >= patience`, stop the loop (the final polish
  still runs). `_last_improvement` already exists; it must also be updated
  when the memetic step improves gbest (it is) and starts at 0.
- `PSOResult` gains `iterations_run` and `stopped_early`.
- `main.py`: `--patience` flag; `best_solution.json` config records it.

### 2. Search space (`experiments/tune.py`)

| Factor | Values |
|---|---|
| Swarm size | 10, 20, 30, 40 |
| Patience | 10, 20, 40 |
| LS interval | 1, 5 |
| Max iterations (cap) | 300 |

24 combinations, plus the **baseline** = today's default (swarm 20,
60 iterations, LS interval 5, no patience). The baseline is never pruned.
All values are CLI-overridable (`--swarm 10 20 ...`, `--patience ...`,
`--ls-interval ...`, `--max-iterations`).

### 3. Instances and seeds

- `--instances` takes folders or a parent folder (every subfolder with a
  `params.json`); default `instances_osm/road_tuning` (4 instances).
- Seeds 1, 2, 3, … assigned per round; a (combo, instance, seed) triple is
  run at most once.

### 4. Rounds (successive halving)

| Round | Who runs | New seeds per instance | Keep |
|---|---|---|---|
| 1 | all 24 + baseline | 1 | best half (12) |
| 2 | survivors + baseline | 2 more | best 4 |
| 3 | finalists + baseline | 2 more (5 total) | winner |

Configurable: `--seeds-per-round 1 2 2`, `--keep 0.5 4`.

### 5. Scoring

- Per run: `real_cost` = true EUR cost of the best feasible plan
  (after final polish). Infeasible run → gap = +inf.
- Per instance: `best_known` = lowest real cost of any run on it.
- `gap_pct = (real_cost / best_known − 1) × 100`, recomputed after each
  round over all runs so far.
- Combination score = **median gap** over its runs. Ranking ties within
  0.5 percentage points go to the lower median runtime.
- Final: Wilcoxon signed-rank test, winner vs baseline, paired by
  (instance, seed), on real cost.

### 6. Execution

- `ProcessPoolExecutor(--workers 7)`; each worker loads the instance and
  calls `MemeticPSO(...).run()` directly (no report, no street geometry).
- Each finished run is appended to `runs.csv` immediately (resume key:
  combo + instance + seed). Rerunning the same command skips done runs.
- Progress line per run: `[37/240] s20 p20 ls1  road_tune_7002 seed 2  1,512.40 EUR  84 it  142 s`.

### 7. Outputs (`tuning/<name>/`, default name `run`)

- `runs.csv` — one row per run: round, swarm, patience, ls_interval,
  instance, seed, real_cost, penalty, feasible, gap_pct, iterations_run,
  stopped_early, runtime_s, evaluations.
- `summary.csv` — one row per combination: runs, median/mean/min/max gap,
  median runtime, median iterations, last round reached.
- `tuning.png` — (a) median gap per combination with spread, finalists
  highlighted, baseline marked; (b) iterations used by the finalists.
- `best_config.json` — winner: swarm_size, patience, local_search_interval,
  suggested iterations (median iterations used, rounded up to a multiple of
  5), median gap, gap vs baseline, Wilcoxon p-value, instances, seeds.
- The summary table is also printed at the end.

### 8. Tests

- Patience stops a run early on a tiny instance and `iterations_run` is
  reported; patience 0 runs all iterations (unchanged behaviour).
- Tuning script on 2 tiny synthetic/base instances with tiny settings:
  produces all four files, resume skips finished runs, pruning keeps the
  configured number, baseline is never pruned, gap is 0 for the best run.

## Estimated cost

~240 runs. LS interval 1 adds ~40 decoder calls per iteration, so runs
with it are slower. Rough total ≈ 2.5–3.5 h on 7 workers. `--quick`
(round 1 only, 1 seed) gives a first look in ≈ 40 min.

## Left to the builder

- Exact chart styling (follow the existing report palette).
- Worker memory: load each instance once per worker process (cache).
