# Project evaluation — Furniture Delivery Optimiser

Date: 2026-11-27 · Independent reviewer (fresh context, no access to the
development conversation), judging the repository at commit `322729e` the way
an instructor would. Everything it ran wrote only to a scratch folder.

## What was verified

- `python run_tests.py`: **103 tests, 11 files, all pass, 22.9 s** (the
  README says "57 tests (7 files) in about 4 s").
- Default run (`python main.py --mode pso --no-geometry`, temp folder):
  **1,429.38 EUR**, 15 trips, 777.6 km, 256 iterations (early stop), 17,878
  evaluations, ~88 s. Identical to the committed `output/` (deterministic).
- **Hand recomputation** of the committed plan from the CSV matrices and
  vehicle rates without `src/`: base 1,086.13 + vehicle-day overtime 343.25 =
  **1,429.3835 EUR**, exact match. All customers once, all trips within weight
  and 90 % volume, longest day 959.36 / 960 min.
- **Robustness** (data/osm, default config): seeds 1–5, 42 → 1,434.3 /
  1,545.9 / 1,554.2 / 1,438.1 / 1,533.3 / 1,429.4 EUR (mean ≈ 1,489, range
  **8.7 %**). Seeds 2, 3, 5 stopped after 22–26 iterations, only 1–2 % below
  the starting construction (1,565.95 EUR).
- **Ablation** (instrumented): with seed 42 the swarm improved the global
  best 4 times (worth **1.03 EUR**); local search on the global best 91
  times (worth **135.2 EUR**). Seed 1 similar (0.13 vs 130.8 EUR). With the
  swarm frozen (inertia = c1 = c2 = 0), mean over seeds 1, 2, 3, 42 was
  **1,487.3 EUR** vs **1,491.0 EUR** with the normal swarm: no measurable PSO
  contribution.
- data/road_ga, seed 42: 1,430.34 EUR vs 1,472.06 EUR for the shared
  reference plan (`pso_reference_solution`, not the GA team's best; **no real
  GA comparison exists**, not verified).
- Decoder: ≈ 2.8 ms per decode with Numba, ≈ 521 ms with the JIT disabled
  (README says "65 ms per decode", stale).

## Scores (1 = poor, 5 = excellent)

| # | Criterion | Score | Evidence |
|---|---|---|---|
| 1 | Modelling & realism | **3** | Strong: real cost vs penalty strictly separated (`cost.py:283-294`, `decoder.py:685-698`), vehicle-day overtime charged once, directed road matrices, hard capacity. Weak: in road mode driving time is the same for every vehicle (`cost.py:219-223`) — the MAN TGL drives as fast as a Caddy; no driver breaks, no customer time windows, unlimited parallel loading at the depot; delivery simulation never feeds back into cost (`delivery_sim.py:11-16`); the 10 % volume reserve is a search penalty (`cost.py:277-294`) but README line 228 calls it real cost. |
| 2 | Algorithm design | **2.5** | Sound standard pipeline (random keys → Split → assignment → local search, Prins-style). But the Split DP prices each segment on the cheapest vehicle as if the fleet were unlimited, with trip-level overtime only (`kernels.py:90-110`), so it optimises a proxy; the 5-vehicle limit, day overtime and 24:00 rule are only handled by the assignment. The PSO itself adds almost nothing (ablation). Patience 20 stops half the seeds prematurely. |
| 3 | Correctness & verification | **4** | 103 tests pass; GA validator cost reproduced exactly (`test_road.py:177-191`); Numba kernels checked against the original Python; hand recomputation matches. Gaps: no brute-force check that Split is optimal on small tours; no end-to-end check of a saved plan's totals; kernel tests only prove equality with the old code. |
| 4 | Experimental methodology | **2** | Tuning invalidated by the time limit; defaults hand-set; no held-out test set; baseline comparison with unequal budgets (problems 2–4). |
| 5 | Results | **3** | 1,429 EUR, feasible, independently verified, all overtime unavoidable. But it is the best seed; the median seed is ≈ 1,486 EUR; no comparison with GA results or an equal-budget baseline. |
| 6 | Software & reproducibility | **3** | Clean modules, single CLI, deterministic seeds, data provenance (sha256, pool README, manifests). Hygiene: 21 tracked `__pycache__/*.pyc` despite `.gitignore`; tracked `.superpowers/brainstorm/.last-token`; `output/` tracked and rewritten in most commits; 30 MB `.npz` (`.git` 56 MB); commit messages like "....", "tune.....", "updating..."; `scipy` missing from `requirements.txt` though `tune.py:300` needs it; no pinned versions. |
| 7 | Documentation & presentation | **2** | README detailed but headline numbers stale or contradicting the code (problem 1). Report good but mislabels: data/osm called the "GA team's road network" (`report.py:738`), "last delivery by None" shown. |

**Overall: 3 / 5.** The engineering is solid and the cost evaluation is
trustworthy (exactly reproduced, cross-checked, matches a hand
calculation). What lets it down is the experimental claims and the
documentation: the README does not describe the code that produced
`output/`, the tuning does not support the current defaults, and the "PSO"
part of the method contributes almost nothing measurable. An instructor who
probes any of these will find them.

## Top strengths

1. Real cost and penalty strictly separated, with breakdowns that sum to the
   totals (`test_constraints.py:210`).
2. Independent verification: GA validator cost reproduced to the cent; the
   committed plan matched by hand.
3. Real-road pipeline from OSM with provenance, snap checks and a
   feasibility gate on every dataset.
4. Numba speed-up proven equal to the original code; deterministic runs.
5. The tuning harness itself (successive halving, resume, median gap, tie
   rule) is well built, and its own spec forbids tuning on data/osm.

## Top problems (ranked by how much they would hurt)

1. **README contradicts the code and the committed output.** README: 1,734.94
   EUR, 19 trips, 1,058 km, 150 s, 1,836 evaluations (lines 587-603); actual:
   1,429.38 EUR, 15 trips, 777.6 km, 77 s, 17,878 evaluations. "LS every 5
   iterations, final budget 120" vs code 1 and 150. Penalty table (lines
   245-253: 1,000 EUR/kg, 2,000 EUR per missing customer, 1,000,000 per extra
   customer) vs code (capacity hard with no price; counts 500 and 100,
   `cost.py:42-43`). "Feasible only when min/max checks pass" contradicts
   `cost.py:303`. Test count wrong; "43 % more than straight line" stale.
   *Fix:* regenerate every number from `output/` and the code; correct the
   constraint table. ~2 h.
2. **Defaults are not the tuning result.** Winner in
   `tuning/main/best_config.json` is s10_p40_ls5; code has swarm 30,
   patience 20, LS every iteration, final budget 150. s30_p20_ls1 was dropped
   in round 2 (rank 9). Comment `pso.py:36-38` says "swarm 40", code 30. The
   last two commits only change these defaults and regenerate `output/` on
   data/osm — looks like hand-tuning on the reported instance, which the spec
   lists as an anti-goal. *Fix:* adopt the tuned config, or re-tune with the
   current code and state the process honestly. ~1 h + rerun.
3. **Tuning ran under a binding time limit with the pre-Numba code.** 133 of
   244 runs hit 600 s; median run 2,857 evaluations vs ~17,900 now; patience
   variants identical (s40_p10/p20/p40_ls1 all 4.79 %); the winner was chosen
   on a 1.4 s runtime difference between two time-capped runs (613.8 vs
   615.2 s). *Fix:* re-run `experiments.tune` with the Numba code, an
   evaluation budget instead of wall time, and 5+ seeds from round 1. ~1 h
   setup + compute.
4. **No held-out evaluation and no fair baseline.** The 4 tuning instances
   share data/osm's 250 locations (only the orders differ); road_test and
   road_validation are never built or used (README line 401 still says
   "phase 2"). The Wilcoxon p = 0.015 compares a winner selected among 24
   configurations against a baseline given half the runtime (314 vs 614 s).
   *Fix:* run the final config on 20+ unseen instances with an equal-budget
   baseline (multi-start local search, Clarke-Wright + Split); report mean,
   std, Wilcoxon. 3–4 h.
5. **The PSO component is not doing the work** (instrumentation and
   frozen-swarm ablation above). The method is presented as a PSO. *Fix:*
   present it honestly as a memetic search, or show a measured swarm
   contribution (e.g. lower LS share, or local search on each particle's
   best); add the ablation as a table. 2–3 h.
6. **Seed variance and premature early stopping:** 8.7 % spread; 3 of 6
   seeds stop after ~25 iterations. *Fix:* patience ~60 or stall detection
   on the time budget; report a multi-seed distribution, not seed 42. 1 h +
   compute.
7. **Split DP ignores the fleet and the day limit** (`kernels.py:90-110`).
   *Fix:* document at least; better a per-type usage penalty or a
   fleet-aware Split. 1 h document / ~6 h fix.
8. **Repository hygiene:** tracked `.pyc` and `.superpowers` token, `output/`
   churn, missing scipy. *Fix:* `git rm --cached`, `.gitignore`,
   requirements. 0.5 h.

## Questions the instructor is likely to ask

1. *How much does the swarm contribute vs the local search?* Cannot answer
   today; the ablation says almost nothing.
2. *How did you choose the parameters?* Partly: tuning exists but
   contradicts the defaults and ran under an invalidating time cap.
3. *How robust is 1,429 EUR across seeds?* No data in the repo; the
   reviewer's runs give 1,429–1,554 EUR.
4. *How do you compare with the GA team?* Only an evaluator cross-check; no
   quality comparison on the same instance and budget.
5. *Is the Split really optimal?* Only for a fixed tour and a proxy cost that
   ignores the fleet; no brute-force test.
6. *Why is the volume reserve a penalty and not money? Why are 10/50 soft?*
   The code can justify it; the README says the opposite.
7. *How far is 1,429 EUR from optimal?* Only the unavoidable-overtime bound
   (avoidable overtime 0 h); no lower bound on the full cost.
8. *What happens when deliveries fail?* Only labels and a Monte Carlo of
   counts; cost and schedule unchanged.

## Quick wins (under 1 hour each)

- Update the README benchmark table, default config, test count (103,
  11 files, ~23 s) and decoder timing (~3 ms).
- Fix the constraint/penalty table and the definition of "feasible" to
  match `cost.py`.
- Fix the `pso.py:36-38` comment; adopt the tuned config or state
  "hand-set after tuning, not validated".
- Add a 5-seed results table (min / median / max).
- Relabel "Deterministic feasible seed" (on data/osm it prints penalty
  1,111,109.9, so it is not feasible); fix the report's "GA team's road
  network" and "last delivery by None".
- `git rm -r --cached` the `__pycache__` files and `.superpowers/`; add
  `scipy` to `requirements.txt`.
- One slide with the frozen-swarm ablation: better shown than found.
