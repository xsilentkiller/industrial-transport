# Furniture Delivery Optimiser

A Python implementation of a **Split-based memetic random-key particle swarm
optimisation (PSO)** for the 250-customer furniture delivery benchmark in
`dataset/`.

The solver reads the six dataset files once, decodes random-key vectors into
vehicle routes through a Split dynamic program, assigns trips to the mixed
fleet, and improves solutions with memetic local search. Infeasible candidates
are never repaired: their true EUR cost and a proportional penalty are stored
separately.

---

## Quick start

Requirements: Python 3.10+, NumPy, and Matplotlib.

```powershell
# Full run: decoder checks + default PSO benchmark + visual report
python main.py --mode all

# Decoder checks only (first 20 customers, full 250, timing)
python main.py --mode decoder

# PSO only (also writes the report)
python main.py --mode pso

# Rebuild the visual report from an existing solution
python main.py --mode report

# Unit tests
python main.py --mode tests
# or
python -m unittest discover -s tests -v
```

Useful PSO options:

```powershell
python main.py --mode pso --swarm 30 --iterations 120 --seed 42
python main.py --mode pso --time-limit 60
python main.py --mode pso --output-dir output

# Report options
python main.py --mode report --input output/best_solution.json
python main.py --mode report --report-dir output/report
```

The best solution is written to `output/best_solution.json`, including the
trip list, per-trip metrics, fleet totals, PSO configuration, runtime, and
search history.

The visual report is written to `output/report/report.html` (or the directory
given by `--report-dir`). It is a single self-contained HTML file with
embedded charts. Standalone PNG charts and CSV tables are written next to it.

---

## Real road network (with the GA team)

`dataset_road/` is the same 250-customer dataset on **real Budapest roads**:
directed road distances and driving times taken from the GA team's
OpenStreetMap graph (`data/shared_pso_250_road.json` in
huyuxuan028/transport-algorithms, sha256 `6d794dff…`). In road mode a leg's
driving time is the road time (same for every vehicle) and fuel/maintenance
are paid on road km. The road benchmark keeps the brief's **250 customers**
and the 5-vehicle fleet, with the **shared 16-hour workday** of the GA team
(start 08:00, back at the depot by 24:00, first 8 h at the normal wage, the
rest at 2x): on roads, 250 customers do not fit the 08:00-23:00 window with
this fleet. All other rules are unchanged.

```powershell
# re-import (only needed if the GA team publishes a new road file)
python -m experiments.import_road --source ..\transport-algorithms\data\shared_pso_250_road.json

# solve on roads
python main.py --mode pso --dataset dataset_road

# road benchmark instances (250 customers, orders vary by seed)
python -m experiments.make_road_instances --set all     # 100 test instances, all 250 customers
python -m experiments.make_road_instances --set all --check

# draw the routes along the streets (needs the OSM extract once)
python -m experiments.road_geometry --solution output/best_solution.json --osm cache/budapest_osm.json --download
python main.py --mode report --dataset dataset_road
```

Our evaluator reproduces the GA validator's cost of the shared reference plan
exactly (1,472.0597 EUR, `tests/test_road.py`). The report writes
`route_map.html`, an interactive street map; with `route_geometry.json` the
lines follow the roads and every leg is checked against the cost matrix.

### Own road data from OSM (download once, route the pool once)

Instead of the GA team's file, the road matrices can be built here for all
1500 addresses of `dataset/places_pool.csv`; any dataset is then a slice of
that pool (no re-routing). Same OSM query, road classes, speeds and Dijkstra
as the GA team. Everything heavy is cached in `cache/` (not committed).

```powershell
# 1a. once: the OSM road extract -> cache/budapest_osm.json, either from Geofabrik (reliable) ...
pip install osmium
curl.exe -L -o cache\geofabrik\hungary-latest.osm.pbf https://download.geofabrik.de/europe/hungary-latest.osm.pbf
python -m experiments.osm_from_geofabrik
#     ... or from Overpass in 4x4 resumable tiles: add --download to step 1b
# 1b. once: route depot + pool on it (resumable, ~2 min on 7 workers)
python -m experiments.build_road_pool
# 2. a road dataset: same customers/orders as dataset/ ...
python -m experiments.make_osm_dataset --out dataset_osm
#    ... or other customers from the usable pool
python -m experiments.make_osm_dataset --out dataset_osm_s9 --customers 250 --seed 9
# 3. use it like dataset_road
python main.py --mode pso --dataset dataset_osm --output-dir output_osm
python -m experiments.make_road_instances --road-dir dataset_osm --set road_tuning --out instances_osm
```

Check against the GA data (Geofabrik extract of 25 Nov 2026): all 251
locations snap to the same OSM vertex; 97.7 % of the km and 99.1 % of the
minutes agree within 0.1 % (mean difference 0.01 %), the rest are OSM edits
since the GA snapshot. A PSO solution on `dataset_osm` draws 269/269 legs
exactly on the cost matrix.

Pool places that snap more than 150 m from a road or have a depot round trip
over 180 min are excluded (`cache/road_pool/pool_meta.json` lists them).
`make_osm_dataset` refuses to finish (no `params.json`) unless every order fits
the largest vehicle and every customer can be served alone inside the day.

---

## Data loading

`src/data_loader.py` is the **only module that opens a CSV file**. It reads the
following files once and returns plain NumPy arrays and small dataclasses via
`ProblemData`:

| File | Contents used |
|---|---|
| `params.json` | shift length, overtime multiplier, fuel price, free-volume ratio, trip limits |
| `vehicles.csv` | capacity, speed, fuel use, wage, maintenance |
| `locations.csv` | depot and customer coordinates |
| `orders.csv` | weight, volume, loading and unloading time per customer |
| `distances.csv` | 251 × 251 symmetric distance matrix in km |
| `furniture_clean.csv` | item table used to validate each order's weight and volume |

Index `0` is the depot; customers are `1..250`, matching `location_id` in the
dataset. `load_problem()` validates the matrix shape/symmetry, the location
numbering, and every order against the furniture table before returning.

---

## Model assumptions

### Fleet and working day

The fleet is fixed: two VW Caddy vans, two Mercedes Sprinter vans, and one MAN
TGL truck. Every vehicle may perform several trips in one day, and every trip
starts and ends at the depot.

| Vehicle | Type | Weight cap | Raw volume | Usable volume (10% free) | Avg. speed | Wage |
|---|---|---:|---:|---:|---:|---:|
| 1, 2 | Caddy | 680 kg | 3.10 m³ | 2.79 m³ | 50 km/h | 12 EUR/h |
| 3, 4 | Sprinter | 1,231 kg | 10.50 m³ | 9.45 m³ | 48 km/h | 13 EUR/h |
| 5 | MAN TGL | 5,480 kg | 42.70 m³ | 38.43 m³ | 40 km/h | 16 EUR/h |

The 8-hour shift applies **per vehicle-day**, not per trip:

- Overtime is allowed and paid at the `overtime_multiplier` (2.0).
- There is **no penalty for exceeding the shift**; overtime is a real cost.
- The overtime premium is charged once on the vehicle's total day time, not
  separately on each trip.
- Loading at the depot and unloading at the customer both count as work time.

### Trip time

For an ordered customer sequence `c1..ck`:

```text
distance = d(0,c1) + d(c1,c2) + ... + d(ck,0)
driving  = distance / avg_speed * 60
service  = sum(loading_i) + sum(unloading_i)
total    = driving + service
```

### Trip cost

```text
fuel         = distance * fuel_l_per_km * fuel_price_per_l
maintenance  = distance * maintenance_per_1000km / 1000
wage_base    = total / 60 * wage_per_h
base_cost    = fuel + maintenance + wage_base

vehicle OT   = max(0, vehicle_day_total - 480 min)
OT premium   = vehicle_OT / 60 * wage_per_h * (overtime_multiplier - 1)

solution cost = sum(trip base_cost) + sum(vehicle OT premium)
```

`TripPlan.cost_eur` is the **standalone** cost of one trip:
`base_cost + trip-level overtime premium`. Because overtime is a vehicle-day
concept, a trip's standalone cost can differ from its contribution to the
solution cost when the same vehicle performs several trips. The solution
evaluator always aggregates overtime correctly over the vehicle-day.

### Constraints and penalties

| Constraint | Handling | Penalty |
|---|---|---:|
| Weight over vehicle capacity | soft, proportional | 1,000 EUR per kg over |
| Volume beyond usable (90%) volume | soft, proportional | 10,000 EUR per m³ over |
| Fewer than 10 customers | soft, proportional | 2,000 EUR per missing customer |
| More than 50 customers | hard in the decoder, also penalised | 1,000,000 EUR per extra customer |
| Shift overrun | allowed, paid as overtime | 0 EUR penalty |

A trip is feasible only when all capacity, free-volume, minimum-count, and
maximum-count checks pass. The **minimum of 10 customers is deliberately
soft**: a short trip is allowed but pays 2,000 EUR per missing customer. When
the soft partition is penalised, the decoder also tries a hard-minimum
partition as a safety net and keeps whichever has the lower objective.
Capacity violations are never repaired;
they simply carry a large proportional penalty, so the search strongly prefers
feasible trips.

### Objective

```text
objective = true_cost_eur + penalty
feasible  = (penalty <= 1e-7)
```

`true_cost_eur` and `penalty` are stored separately on every `Solution` and
`TripMetrics`. `Solution.feasible` is never used to hide the true cost.

---

## Daily working window

Every vehicle starts at 08:00 and drives its trips back to back. The rules are
set in `dataset/params.json` and enforced in `src/cost.py` (`day_metrics`):

| Time | Rule | Wage |
|---|---|---|
| 08:00-16:00 | normal 8 h shift | 1x |
| 16:00-23:00 | overtime | 2x (`overtime_multiplier`) |
| 22:00 | last customer must be served (`last_delivery_by`) | |
| 22:00-23:00 | only the drive back to the depot | 2x |
| 23:00 | vehicle must be back (`hard_day_end`) | |

Overtime is real money and part of the true cost. A late
last delivery or a return after 23:00 is a planning penalty of 1,000 EUR per
minute, reported separately and never part of the true cost. Inside a
vehicle-day, the trip with the longest return leg runs last, so the last
delivery is as early as possible. The trip-to-vehicle assignment prices every
move with these rules.

Each run also reports the longest vehicle-day, each vehicle's start, last
delivery and finish time, and overtime against the unavoidable minimum
(total work minus 8 h per vehicle). Remove the window keys (`last_delivery_by`, `hard_day_end`) from
`params.json` to get the original model without a daily limit.

---

## Benchmark instances

`python -m experiments.make_instances` builds the 100 test instances
(250 customers each) into `instances/test/`; `--set all` also builds the
tuning (4), validation (20) and scaling (4 each at 50, 100, 250, 500 and
1,000 customers) sets. Each instance is a complete dataset folder that the
solver reads like `dataset/` (`python main.py --dataset instances/test/test_3001`).

* Same rules as `dataset/generators/`; only the seed changes. Seed 42 with 250
  customers reproduces `dataset/` byte for byte (`tests/test_instances.py`).
* Disjoint seeds per set: tuning 1001-1004, validation 2001-2020,
  test 3001-3100, scaling 4001-4004 (seed 42 is never used).
* Scaling instances repeat the 5-vehicle fleet once per 250 customers.
* The folders (about 90 MB) are not committed. `instances/manifest.json`
  records the SHA-256 of every file;
  `python -m experiments.make_instances --set all --check` verifies a rebuild.

| Test set (100 instances) | Mean | Min | Max |
|---|---:|---:|---:|
| Furniture volume (m3) | 116.1 | 104.2 | 128.3 |
| Furniture weight (kg) | 5,128 | 4,706 | 5,610 |
| Loading + unloading time (h) | 41.9 | 41.2 | 42.7 |
| Fleet rounds needed by volume | 1.85 | 1.66 | 2.04 |

---

## Convergence fixes

Diagnosis (default instance, daily window): in V0 the best plan came from the
start heuristics and the swarm never improved it. Two causes were found and
fixed; each fix is a setting in `PSOConfig`, measured with
`python -m experiments.convergence_ablation` (seeds 1-3) and plotted with
`experiments/plot_convergence.py`.

| Variant | Mean final cost | Swarm improvements | Runtime | Kept |
|---|---:|---:|---:|---|
| V0 | 1,207.40 EUR | 1.7 | 180 s | |
| Fix 1: polish every candidate (`polish_candidates`) | 1,195.45 EUR | 9.7 | 223 s | yes |
| + Lamarckian write-back (`lamarckian`) | 1,197.34 EUR | 6.7 | 250 s | no |
| Fix 2: start near good tours (`seeded_init`) | 1,191.94 EUR | 18.7 | 227 s | yes |
| Fix 3: diversity restart (`diversity_threshold=0.02`) | 1,191.94 EUR | 18.7 | 226 s | no, never needed |
| Fix 4: `v_max=0.10` | 1,191.15 EUR | 18.7 | 229 s | no, within noise |

* **Fix 1.** The incumbent was polished (2-opt inside each trip) but every
  candidate was scored without the polish, about 10 % worse for the same
  tour, so the memetic step could never accept a move. Candidates are now
  scored the same way. The trip 2-opt was rewritten with O(1) move deltas
  (same results, about 6x faster) to make this affordable.
* **Fix 2.** Under the daily window random tours are 0 % feasible; the swarm
  spent its iterations escaping penalties. Particles now start (and restart)
  as perturbed copies of good tours.

Three seeds on one instance show the direction, not significance; the
validation on 20 instances is part of phase 2.

---

## Solution format

`Solution.trips` is the required list of trips. Each `TripPlan` contains:

| Field | Meaning |
|---|---|
| `vehicle_index` | 0-based index into `ProblemData.vehicles` |
| `vehicle_id` | dataset vehicle id |
| `customers` | ordered tuple of customer ids |
| `time_min` | driving + loading + unloading |
| `cost_eur` | standalone trip cost |
| `base_cost_eur` | fuel + maintenance + all trip minutes at normal wage |
| `distance_km` | trip distance |
| `metrics` | full `TripMetrics` record, including weight, volume, penalty, and feasibility flags |

`Solution` also exposes fleet totals (`true_cost_eur`, `penalty`, `objective`,
`n_trips`, distance, working time, overtime, per-vehicle times/costs) and
`as_dict()` for JSON serialisation. `Solution` is iterable, so
`for trip in solution:` works directly.

---

## Algorithm

1. **Random-key encoding** — a vector of keys is sorted to produce a giant
   tour (`keys_to_tour`); any permutation can be rank-encoded back to keys
   (`tour_to_keys`).
2. **Greedy cut** — a quick capacity-aware partition used as a decoder
   baseline.
3. **Split DP** — computes the optimal contiguous partition of the giant tour
   in `O(n * max_trip_length * n_vehicles)`. The DP includes capacity
   penalties, the soft minimum-customer penalty, and the hard maximum of 50.
   Each segment is costed with the cheapest vehicle after adding that
   vehicle's trip cost and capacity/count penalties.
4. **Vehicle assignment** — trips are assigned to the five vehicles
   most-constrained-first (largest volume/weight), followed by move and swap
   local search. Vehicle-day overtime is evaluated on the aggregated day.
5. **Memetic PSO** — standard global-best PSO in random-key space, with:
   - deterministic seeds: nearest neighbour, polar sweep, volume-aware sweep,
     and volume-aware nearest neighbour;
   - a deterministic chunk-10 feasible incumbent;
   - decoding and 2-opt polishing of the two strongest constructions before
     the first generation;
   - periodic 2-opt / relocate / swap local search on the global best, guided
     by nearest-neighbour lists;
   - trip-level 2-opt polishing and re-assignment;
   - diversity restarts and a final polish of the global best and all distinct
     personal bests.
6. **Final validation** — the saved solution can be checked with
   `check_solution(data, solution)`, which verifies full customer coverage,
   duplicate customers, trip-size limits, weight capacity, free volume, and
   vehicle indices.

The Split DP is an exact partition algorithm **for a fixed giant tour**; the
PSO searches over giant tours. The overall method is a heuristic, not a
certified global optimum.

---

## Project layout

```text
main.py                 CLI: decoder checks, PSO, report, tests
src/data_loader.py      the only CSV reader; returns ProblemData
src/cost.py             trip metrics, cost formulas, penalties, overtime
src/decoder.py          random keys, greedy cut, Split DP, assignment, checks
src/pso.py              PSO, construction heuristics, local search, polish
src/report.py           HTML/PNG/CSV visual report generator
tests/test_cost.py      hand-checked cost, overtime, free-volume tests
tests/test_decoder.py   first-20, full-250, Split/coverage/timing tests
tests/test_report.py    report files, tables, and chart smoke test
dataset/                benchmark instance (read-only)
output/                 best_solution.json
output/report/          report.html, PNG charts, CSV tables
```

---

## Verification and test results

`python -m unittest discover -s tests -v` runs 11 tests. All pass.

Hand-checked cases include:

- **Three-customer trip:** 49.910 km, 87.692 min, 39.418 kg, 0.9288 m³,
  25.803496 EUR, with the expected 7-customer soft-minimum penalty.
- **Overtime trip:** 600 min = 10 h, 120 min overtime, 20 EUR extra premium,
  120 EUR total on a 10 EUR/h vehicle.
- **Vehicle-day aggregation:** two 5-hour trips produce 2 hours of overtime
  once for the day, not per trip.
- **Free-volume violation:** ten 0.30 m³ orders fit the Caddy's raw 3.10 m³
  but exceed the usable 2.79 m³; the 0.21 m³ excess is penalised while true
  cost remains separately available.
- **Report generator:** produces the HTML report, five PNG charts, and four
  CSV tables (summary, vehicles, trips, history) from a saved solution.

Decoder checks:

- First 20 customers: 2 trips of 10 customers, every customer exactly once,
  all capacity and count checks pass, penalty 0.
- Full 250 customers: 14 trips, every customer exactly once, no trip over 50,
  all capacity checks pass, penalty 0.
- Full 250-customer decoder timing: **mean ≈ 24–25 ms** per decode on the
  development machine.

---

## Visual report

`python main.py --mode report` builds `output/report/report.html` from an
existing `output/best_solution.json`. The same report is generated
automatically after `--mode pso` and `--mode all`.

The HTML file is self-contained (charts are embedded), so it can be opened
directly in a browser without a web server. It includes:

| Output | What it shows |
|---|---|
| Summary cards | true cost, penalty, trips, distance, working time, overtime, runtime, evaluations |
| `convergence.png` | PSO loss/convergence curve, penalty, and cumulative decoder calls |
| `vehicle_timeline.png` | vehicle-day Gantt chart with the 8-hour shift and overtime region |
| `cost_breakdown.png` | fuel, maintenance, wages, and overtime premium per vehicle, plus total cost composition |
| `trip_capacity.png` | weight/usable-volume utilisation and customers per trip |
| `route_map.png` | depot-to-customer routes on the Budapest map and trip distance vs cost |
| Tables | summary, per-vehicle aggregates, and per-trip details |
| CSV files | `summary.csv`, `vehicles.csv`, `trips.csv`, `history.csv` for Excel/Python analysis |

The final polished cost is marked separately on the convergence chart because
the trip-level polish runs after the last PSO iteration.

---

## Benchmark result

Dataset: 250 customers, 5 vehicles, total demand **5,265.18 kg / 123.98 m³**,
total loading + unloading service time **2,525.6 min (42.1 h)**.

Default configuration: swarm 30, 120 iterations, inertia 0.72, `c1 = c2 =
1.49`, `v_max = 0.25`, local search every 5 iterations with budget 40, final
local-search budget 120, seed 42.

| Stage | True EUR cost | Trips | Feasible |
|---|---:|---:|---|
| Deterministic chunk-10 seed | 1,666.51 | 25 | yes |
| Nearest-neighbour decoder | 1,344.89 | 14 | yes |
| Polished volume-aware seed | 1,183.33 | 15 | yes |
| **Final memetic PSO** | **1,177.30** | **15** | **yes** |

Final solution details:

| Metric | Value |
|---|---:|
| True cost | **1,177.30 EUR** |
| Penalty | 0.00 EUR |
| Trips | 15 |
| Distance | 762.1 km |
| Working time | 57.92 h |
| Overtime | 18.05 h |
| Vehicle-day overtime cost | 231.43 EUR |
| Runtime | 128.3 s (≈2 min) |
| Decoder evaluations | 4,708 |

Per-vehicle result:

| Vehicle | Type | Trips | Day time | Overtime | Cost |
|---|---|---:|---:|---:|---:|
| 1 | Caddy | 2 | 8.79 h | 0.79 h | 135.98 EUR |
| 2 | Caddy | 3 | 10.41 h | 2.41 h | 179.57 EUR |
| 3 | Sprinter | 4 | 14.56 h | 6.56 h | 340.96 EUR |
| 4 | Sprinter | 5 | 16.29 h | 8.29 h | 368.12 EUR |
| 5 | MAN TGL | 1 | 7.88 h | 0.00 h | 152.67 EUR |

The saved solution was independently rebuilt from `output/best_solution.json`:
`check_solution` reports no errors, all 250 customers appear exactly once, and
the recomputed true cost matches 1,177.30 EUR.

---

## Notes and limitations

- The instance uses straight-line distances × 1.3 (`distances.csv`), not live
  road routing.
- Packing mode is 1 (keep 10% free volume); three-dimensional box placement
  (packing mode 2) is not modelled.
- The fleet is fixed and every vehicle may perform multiple depot-return
  trips.
- The minimum-customer rule is soft by design; the reported benchmark solution
  happens to have 10–36 customers on every trip.
- Results are deterministic for a fixed seed on the same NumPy/BLAS setup, but
  small platform-dependent floating-point differences can change tie-breaks.
