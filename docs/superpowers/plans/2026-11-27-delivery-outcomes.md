# Delivery Outcomes + Shared OSM Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Label every stop of the optimised plan delivered / attempted (not home) / refused in a simulated day, measure the risk over many simulated days, list every delivery in the report, and ship the routed 1,500-address OSM pool in git for colleagues.

**Architecture:** A new module `src/delivery_sim.py` works on the saved solution dict (`payload["best_solution"]`), so `main.py` (after the PSO) and `--mode report` use the same data. It computes each stop's planned arrival/departure from the trip order, draws outcomes from a seeded RNG, and runs a vectorised Monte Carlo. `main.py` stores the result under `payload["delivery_simulation"]`; `src/report.py` only displays it (HTML section + 3 CSVs). Times and real cost never change. The OSM pool is built into `data/osm_pool/` (committed) instead of `cache/road_pool/` (ignored).

**Tech Stack:** Python 3.13, NumPy, existing report HTML (no new libraries; a few lines of inline JS for filtering).

**Spec:** `docs/superpowers/specs/2026-11-27-delivery-outcomes-design.md`

## Global Constraints

- Optimiser, objective and PSO defaults untouched; `payload["best_solution"]` identical with and without the simulation.
- A failed stop takes exactly the same time as a delivery; no extra depot time; times and real cost identical for every outcome mix.
- Probabilities: not home `0.05`, refused `0.02`; RNG `numpy.random.default_rng`; same seed → same result.
- Dataset files are never modified. Customers are shown by order ID + address + district; no invented names.
- Instructor's rules unchanged (08:00–24:00, 2× overtime after 8 h, 250 customers, 5 vans).
- Status strings exactly: `"delivered"`, `"attempted"`, `"refused"`.
- Clock strings via `data.params.clock(minutes_after_start)`.
- Nothing is committed without the user's go-ahead (commit steps below run only when the user says so).

## Review Focus

1. A vehicle with no trips (e.g. an unused Caddy) → no records for it, no crash, Monte Carlo still runs. (Task 2 test.)
2. Invalid probabilities (negative, or `p_not_home + p_refused > 1`) → `ValueError` from `SimConfig`, and `main.py` exits with that message. (Task 3 test.)
3. An old `best_solution.json` without `delivery_simulation` given to `--mode report` → report builds with no delivery section and no delivery CSVs. (Task 5 test.)
4. A dataset whose `locations.csv` lacks `full_address`/`district` (or a location missing from it) → address shown as `location <id>`, district `""`. (Task 2 test.)
5. An order with the same item twice → items shown as `Name ×2`; a location with no order (`order_ids == -1`) → order shown as `""`. (Task 2 test.)

---

### Task 1: Ship the 1,500-address OSM pool in git

**Files:**
- Modify: `experiments/build_road_pool.py:53` (`POOL_DIR`), docstring
- Modify: `experiments/make_osm_dataset.py:6,40` (`POOL_DIR`, docstring)
- Modify: `README.md` (section "Own road data from OSM")
- Create (generated): `data/osm_pool/road_pool.npz`, `data/osm_pool/pool_meta.json`, `data/osm_pool/README.md`

**Interfaces:**
- Produces: `data/osm_pool/road_pool.npz` with arrays `place_id, vertex, snap_m, usable, km, min` (index 0 = depot, i = pool row i-1), as written by `build_road_pool.main`.

- [ ] **Step 1: Wait for the running build** (`python -m experiments.build_road_pool --out data\osm_pool`, log `data/osm_pool_build.log`) to print its final line; then move the log to `data/osm_pool/build.log`.
- [ ] **Step 2: Check size and content.** Run:
  `python -c "import numpy as np; z=np.load('data/osm_pool/road_pool.npz'); print({k: z[k].shape for k in z}, int(z['usable'].sum()))"` and `du -h data/osm_pool/*`.
  Expected: `km`/`min` shape `(1501, 1501)`, usable count printed, `road_pool.npz` < 50 MB (GitHub warns at 50 MB, rejects at 100 MB). If ≥ 50 MB: stop and ask the user (options: float32, or Git LFS).
- [ ] **Step 3: Point both scripts at the committed folder.** `POOL_DIR = ROOT / "data" / "osm_pool"` in both files; update the two docstrings that say `cache/road_pool`.
- [ ] **Step 4: Verify the consumer still works.** Run `python -m experiments.make_osm_dataset --help` (loads defaults) and `python -m unittest tests.test_road -v`. Expected: help prints, tests pass.
- [ ] **Step 5: Write `data/osm_pool/README.md`**: what the files are, how they were built (command, OSM file sha256 `d3283df7c5fb…` from Geofabrik, which differs from the Overpass extract `data/osm` was cut from, so `data/osm` is NOT regenerated from it), usable count, licence line "Map data © OpenStreetMap contributors, ODbL".
- [ ] **Step 6: README.md**: in "Own road data from OSM", state that the routed pool is in `data/osm_pool/` (committed), so colleagues skip the ~30 min build; `cache/` keeps only the raw OSM extract.
- [ ] **Step 7: Commit (only on the user's go-ahead)**
  ```bash
  git add data/osm_pool experiments/build_road_pool.py experiments/make_osm_dataset.py README.md
  git commit -m "Ship routed 1,500-address OSM pool in data/osm_pool"
  ```

---

### Task 2: Planned stop schedule and delivery records

**Files:**
- Create: `src/delivery_sim.py`
- Test: `tests/test_delivery_sim.py`

**Interfaces:**
- Produces:
  ```python
  STATUSES = ("delivered", "attempted", "refused")
  def load_addresses(dataset_dir: str | Path) -> dict[int, dict[str, str]]   # location_id -> {"address", "district"}
  def items_text(data: ProblemData, loc: int) -> str                        # "Dining Chair ×2, Shoe Rack"
  def stop_schedule(data: ProblemData, solution: dict) -> list[dict]
  ```
  `stop_schedule` row keys: `vehicle_id, vehicle_index, vehicle_model, trip, stop, location_id, order_id, arrive_min, leave_min, trip_start_min, trip_end_min, weight_kg, volume_m3`. `trip` and `stop` are 1-based per vehicle; vehicles in index order, trips in the order they appear in `solution["trips"]` (same as `Solution.day_schedule`).
  Timing: per vehicle clock starts at 0; trip start = clock; `arrive_1 = start + trip.loading_min + leg(0, c1)`; `leave_i = arrive_i + data.unloading[c_i]`; `arrive_{i+1} = leave_i + leg(c_i, c_{i+1})`; `trip_end = leave_last + leg(c_last, 0)`; next trip starts at `trip_end`. `leg(a, b) = cost.drive_min(data, k, data.distance[a, b], data.duration[a, b] if road else 0.0)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_delivery_sim.py` (fixture: `data = load_problem("data/base")`, solution = `decode_tour(data, volume_aware_nearest_neighbor_tour(data)).as_dict()`):
  - `test_trip_end_matches_trip_time`: for every trip, `trip_end_min - trip_start_min` equals that trip's `time_min` within `1e-6`.
  - `test_one_row_per_customer`: rows cover every customer exactly once (`sorted(location_id) == list(range(1, n+1))`); `arrive_min <= leave_min`; within a trip `stop` is 1..count.
  - `test_unused_vehicle_has_no_rows`: a solution dict whose trips use only vehicle index 4 → every row has `vehicle_index == 4`; no exception.
  - `test_items_text_counts_duplicates`: build a `ProblemData` via `dataclasses.replace(data, item_ids=...)` where location 1 has `("fid002", "fid002", "fid001")` → `items_text(data, 1) == "Dining Chair ×2, Shoe Rack"` (names from `data.furniture`).
  - `test_load_addresses_fallback`: a temp dir with `locations.csv` having only `location_id,lat,lon` → `load_addresses` returns `{}`-compatible lookups so that the record builder (Task 3) shows `location 7`; here assert `load_addresses(tmp) == {}`; and for `data/osm` assert location 1 → `{"address": "Tímár utca 1, 1036 Budapest, Hungary", "district": "3"}`.
- [ ] **Step 2: Run** `python -m unittest tests.test_delivery_sim -v` → FAIL (`ModuleNotFoundError: src.delivery_sim`).
- [ ] **Step 3: Implement** `load_addresses`, `items_text`, `stop_schedule` in `src/delivery_sim.py` (csv module; missing columns → empty dict; first-seen item order).
- [ ] **Step 4: Run** the tests → PASS.
- [ ] **Step 5: Commit (on go-ahead)** `git add src/delivery_sim.py tests/test_delivery_sim.py && git commit -m "Planned stop schedule for delivery records"`

---

### Task 3: Simulated day, Monte Carlo, `build_simulation`

**Files:**
- Modify: `src/delivery_sim.py`
- Test: `tests/test_delivery_sim.py`

**Interfaces:**
- Consumes: Task 2 functions.
- Produces:
  ```python
  @dataclass(frozen=True)
  class SimConfig:
      p_not_home: float = 0.05
      p_refused: float = 0.02
      seed: int = 42
      monte_carlo_days: int = 1000
      # __post_init__: ValueError if a p < 0, p_not_home + p_refused > 1, or monte_carlo_days < 0

  def draw_outcomes(n: int, cfg: SimConfig, rng: np.random.Generator) -> list[str]
  def simulate_day(data, solution: dict, addresses: dict, cfg: SimConfig) -> dict
  def monte_carlo(vehicle_index_per_stop: np.ndarray, n_vehicles: int, cfg: SimConfig) -> dict
  def build_simulation(data, solution: dict, dataset_dir, cfg: SimConfig) -> dict
  ```
  `draw_outcomes`: `u = rng.random(n)`; `"refused"` if `u < p_refused`, `"attempted"` if `u < p_refused + p_not_home`, else `"delivered"`. Day RNG: `default_rng(cfg.seed)`; Monte Carlo RNG: `default_rng(cfg.seed + 1)`, one `(days, n)` uniform matrix, same thresholds.
  `simulate_day` returns `{"config": asdict(cfg), "records": [...], "counts": {"delivered", "attempted", "refused", "total"}, "depot_returns": [...], "redelivery": [...], "returned_to_stock": [...]}`. Record keys: `vehicle_id, vehicle, trip, stop, arrive, leave (HH:MM), order_id ("" if -1), location_id, address ("location <id>" if unknown), district, items, weight_kg, volume_m3, status`. `depot_returns`: one entry per trip with failures `{"vehicle_id", "vehicle", "trip", "time" (trip end HH:MM), "orders": [{"order_id", "items", "status"}]}`. `redelivery` = records with status attempted; `returned_to_stock` = refused.
  `monte_carlo` returns `{"days": N, "attempted": {"mean","p5","p95"}, "refused": {...}, "failed": {...}, "share_days_with_failure_by_vehicle": [float per vehicle index]}` (percentiles via `np.percentile`).
  `build_simulation` = `simulate_day(...)` plus key `"monte_carlo"`.

- [ ] **Step 1: Write the failing tests:**
  - `test_zero_probability_all_delivered`: `SimConfig(0, 0)` → counts delivered == total, empty redelivery/returned lists.
  - `test_all_not_home`: `SimConfig(1.0, 0.0)` → all `"attempted"`, `len(redelivery) == total`, one `depot_returns` entry per trip.
  - `test_times_identical_for_any_outcomes`: arrive/leave strings equal between `SimConfig(0,0)` and `SimConfig(0.5,0.3)`.
  - `test_same_seed_same_result` / `test_different_seed_differs` (seeds 42 vs 43 with p 0.3 / 0.2 → statuses differ).
  - `test_invalid_config_raises`: `SimConfig(-0.1, 0)`, `SimConfig(0.8, 0.3)`, `SimConfig(monte_carlo_days=-1)` → `ValueError`.
  - `test_monte_carlo_rate`: 5,000 days, p 0.05/0.02 on the base solution → `failed["mean"] / n` within `0.07 ± 0.005`; deterministic for a seed; `share_days_with_failure_by_vehicle` has `n_vehicles` entries in [0, 1] and 0.0 for unused vehicles.
  - `test_unknown_address_fallback`: `simulate_day` with `addresses={}` → address `"location <id>"`.
  - `test_solution_untouched`: `copy.deepcopy(solution) == solution` after `build_simulation`.
- [ ] **Step 2: Run** → FAIL (names not defined).
- [ ] **Step 3: Implement** in `src/delivery_sim.py`.
- [ ] **Step 4: Run** `python -m unittest tests.test_delivery_sim -v` → PASS.
- [ ] **Step 5: Commit (on go-ahead)** `git commit -am "Simulated delivery outcomes and Monte Carlo"`

---

### Task 4: `main.py` integration

**Files:**
- Modify: `main.py` (`run_pso` payload ~line 200; argument parser ~line 360)
- Test: `tests/test_delivery_sim.py`

**Interfaces:**
- Consumes: `SimConfig`, `build_simulation` (Task 3).
- Produces: `payload["delivery_simulation"]` (absent with `--no-sim`); CLI flags `--not-home` (0.05), `--refused` (0.02), `--sim-seed` (42), `--sim-days` (1000), `--no-sim`; a printed line `Delivery simulation (seed 42): delivered N, attempted N, refused N; Monte Carlo 1000 days: failed mean X (5-95 %: a-b)`.

- [ ] **Step 1: Failing test** `test_main_payload_has_simulation`: call `main.main(["--mode", "pso", "--dataset", "data/base", "--iterations", "2", "--swarm", "4", "--patience", "0", "--no-geometry", "--output-dir", <tmp>])`; load `<tmp>/best_solution.json`; assert `"delivery_simulation"` with `counts["total"] == 250`. Same with `--no-sim` → key absent and `best_solution` equal to the first run's.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**: parse flags; build `SimConfig` (on `ValueError` → `parser.error(str(e))`); after `payload` is built and before `json.dump`, `payload["delivery_simulation"] = build_simulation(data, payload["best_solution"], args.dataset, sim_cfg)` unless `--no-sim`; print the summary line.
- [ ] **Step 4: Run** the test and `python main.py --help` → PASS, flags listed.
- [ ] **Step 5: Commit (on go-ahead)** `git commit -am "main.py: delivery simulation after the PSO"`

---

### Task 5: Report section and CSV files

**Files:**
- Modify: `src/report.py` (`generate_report` ~line 1248, `_build_html` ~line 864; new `_delivery_section(sim: dict) -> str`)
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `payload["delivery_simulation"]` (Task 3 shape).
- Produces: when present, files `deliveries.csv` (columns: vehicle, trip, stop, arrive, leave, order_id, address, district, items, weight_kg, volume_m3, status), `redelivery.csv`, `returned_to_stock.csv` (same columns) in the report dir; an HTML `<section id="deliveries">` titled **"Delivery records (simulated day)"** after "Trip details", containing: a caption with seed and probabilities; four count cards (Delivered, Attempted (not home), Refused, Re-deliveries tomorrow); a Monte Carlo table (expected / 5 % / 95 % for attempted, refused, failed; share of days with ≥1 failure per van); filter buttons All/Delivered/Attempted/Refused and a search box (inline `<script>`, no external files) over a records table with `data-status` on each `<tr>`; one line per depot return. Status cells coloured (delivered neutral, attempted amber, refused red) using the report's existing CSS variables/palette. Listed in the report's "Files" section.

- [ ] **Step 1: Failing tests** in `tests/test_report.py`:
  - `test_delivery_section_and_csvs`: payload from a base solution + `build_simulation(..., SimConfig(0.3, 0.2))` → `report.html` contains `Delivery records (simulated day)` and `data-status="refused"`; the three CSVs exist; `deliveries.csv` has 250 data rows.
  - `test_report_without_simulation`: same payload without the key → no `id="deliveries"`, none of the three CSVs.
- [ ] **Step 2: Run** `python -m unittest tests.test_report -v` → FAIL.
- [ ] **Step 3: Implement** `_delivery_section` and the CSV writes (via existing `_write_csv`, `_html_table` where it fits; escape all text with `html.escape`).
- [ ] **Step 4: Run** → PASS; open `output/report/report.html` after Task 6's run and check the filter buttons and search work.
- [ ] **Step 5: Commit (on go-ahead)** `git commit -am "Report: delivery records section and CSVs"`

---

### Task 6: Docs, test registry, end-to-end run

**Files:**
- Modify: `README.md` (new section "Delivery outcomes (simulated)" after "Daily working window"; project layout lines for `src/delivery_sim.py`, `tests/test_delivery_sim.py`, `data/osm_pool/`; options list with the new flags)
- Modify: `run_tests.py` (`ABOUT["test_delivery_sim"] = "failed deliveries: schedule, outcomes, Monte Carlo | data/base + data/osm"`)

- [ ] **Step 1:** Update README and `run_tests.py` as above.
- [ ] **Step 2:** Run `python run_tests.py` → `ALL TESTS PASSED` (86 + new).
- [ ] **Step 3:** Run `python main.py` → prints the delivery summary line; `output/report/report.html` has the section; `output/report/deliveries.csv` has 250 rows; real cost identical to the same run with `--no-sim`.
- [ ] **Step 4: Commit (on go-ahead)** `git commit -am "Docs and tests for delivery outcomes"`
