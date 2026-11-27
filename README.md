# Furniture Delivery Optimiser

A Python implementation of a **Split-based memetic random-key particle swarm
optimisation (PSO)** for a 250-customer furniture delivery problem in Budapest.

The same 250 customers and orders exist in three versions under `data/`, which
differ only in how distance and travel time are measured:

| Folder | Distances | Role |
|---|---|---|
| `data/osm/` | **our own real-road matrices** built from OpenStreetMap | **default** for `main.py` |
| `data/road_ga/` | the GA team's real-road matrices | shared reference with the GA team |
| `data/base/` | straight line × 1.3 | original model + source data (address pool, furniture, generators) |

The solver reads the dataset files once, decodes random-key vectors into
vehicle routes through a Split dynamic program, assigns trips to the mixed
fleet, and improves solutions with memetic local search. Infeasible candidates
are never repaired: their true EUR cost and a proportional penalty are stored
separately.

---

## Quick start

Requirements: Python 3.10+, NumPy, Matplotlib and Numba
(`pip install -r requirements.txt`; `osmium` only to rebuild the OSM road
data, see below).  Numba compiles the solver's inner loops (Split DP,
trip-to-vehicle assignment, 2-opt) in `src/kernels.py`: a PSO run is about
18x faster with identical results (`tests/test_kernels.py` compares every
kernel with the original Python code).  The first run after installing
compiles the kernels (a few seconds); the machine code is then cached in
`src/__pycache__/`.  Without Numba the same code runs as plain Python, much
slower.

`main.py` is the single entry point:

```powershell
# Full run on data/osm (the default): decoder checks + PSO + street geometry + report
python main.py
# another dataset
python main.py --dataset data/base        # straight-line
python main.py --dataset data/road_ga     # GA team's road matrices
python main.py --dataset instances_osm/road_tuning/road_tune_7001

python main.py --mode decoder     # decoder checks only (first 20, full 250, timing)
python main.py --mode pso         # PSO + street geometry + report
python main.py --mode report      # rebuild the report from output/best_solution.json
python main.py --mode road-data   # rebuild data/osm from OpenStreetMap (see below)

# All tests at once, with a per-file summary (what each file checks, which data)
python run_tests.py               # add -v to list every test
python main.py --mode tests       # the same
```

Useful options:

```powershell
python main.py --mode pso --swarm 30 --iterations 120 --seed 42
python main.py --mode pso --time-limit 60
python main.py --mode pso --no-geometry           # map with straight lines (no OSM extract needed)
python main.py --not-home 0.05 --refused 0.02 --sim-seed 42 --sim-days 1000   # delivery outcomes (defaults)
python main.py --no-sim                           # skip the delivery-outcome simulation
python main.py --mode report --input output/best_solution.json --report-dir somewhere/report
```

**`output/` always holds the latest run only** (any dataset; the next run
replaces it; `--output-dir` picks another folder):

```text
output/
  best_solution.json     the plan; its "dataset" field says which dataset it was
  route_geometry.json    road datasets only: street geometry of the routes
  report/                report.html, route_map.html, PNG charts, CSV tables
```

`best_solution.json` holds the trip list, per-trip metrics, fleet totals, PSO
configuration, runtime, and search history. `report/report.html` is a single
self-contained HTML file with embedded charts. On a road dataset `main.py`
also computes `route_geometry.json` when `cache/budapest_osm.json` exists, so
the map follows the streets. To keep a result, copy `output/` or run with
`--output-dir` elsewhere.

---

## Real road network (with the GA team)

`data/road_ga/` is the same 250-customer dataset on **real Budapest roads**:
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
# solve on the GA matrices (routes drawn along streets if cache/budapest_osm.json exists)
python main.py --dataset data/road_ga

# road benchmark instances (250 customers, orders vary by seed)
python -m experiments.make_road_instances --set all            # from data/road_ga
python -m experiments.make_road_instances --set all --check    # verify against the manifest

# re-import (only if the GA team publishes a new road file; the old one is gone)
python -m experiments.import_road --source ..\transport-algorithms\data\shared_pso_250_road.json
```

Our evaluator reproduces the GA validator's cost of the shared reference plan
exactly (1,472.0597 EUR, `tests/test_road.py`).

### Own road data from OSM (`data/osm/`, the default)

`data/osm/` is built here, without the GA team's file: the road network is
taken from OpenStreetMap, **all 1500 addresses** of
`data/base/places_pool.csv` are routed once (a "pool"), and a dataset is a
slice of that pool (no re-routing). Same road classes, speeds, one-way rules
and time-shortest Dijkstra as the GA team. **The routed pool is committed in
`data/osm_pool/`** (1,501 × 1,501 directed km and minutes, 30 MB; see its
README), so a new road dataset needs neither the map file nor the routing:
`python -m experiments.make_osm_dataset --out data/osm_s9 --seed 9`. Only the
raw map and the OSM extract are cached in `cache/` (not committed).

```powershell
# once: the Geofabrik Hungary map file (316 MB) and pyosmium
pip install osmium
curl.exe -L -o cache\geofabrik\hungary-latest.osm.pbf https://download.geofabrik.de/europe/hungary-latest.osm.pbf

# OSM extract -> routed pool (~3 min on 7 workers) -> data/osm/ ; cached steps are skipped
python main.py --mode road-data                    # --rebuild-pool to route again
```

The same steps one by one (more options):

```powershell
python -m experiments.osm_from_geofabrik           # map file -> cache/budapest_osm.json (~20 s)
python -m experiments.build_road_pool [--download] # route the pool; --download = Overpass tiles instead
python -m experiments.make_osm_dataset --out data/osm                       # same customers/orders as data/base
python -m experiments.make_osm_dataset --out data/osm_s9 --customers 250 --seed 9   # other customers
python -m experiments.make_road_instances --road-dir data/osm --set road_tuning --out instances_osm
```

Checks (Geofabrik extract of 25 Nov 2026, all in `tests/test_road.py::TestOsmData`):

* all 251 locations snap to the same OSM road vertex as the GA data;
* 97.7 % of the km and 99.1 % of the minutes agree with the GA matrices within
  0.1 % (mean difference 0.01 %); the rest are OSM edits since the GA snapshot;
* every customer fits the largest vehicle and can be served alone in the day;
* a PSO solution draws 269/269 legs exactly on the cost matrix.

Pool places that snap more than 150 m from a road or have a depot round trip
over 180 min are excluded (3 of 1500; listed in `data/osm_pool/pool_meta.json`).
`make_osm_dataset` refuses to finish (no `params.json`) unless every order fits
the largest vehicle and every customer can be served alone inside the day.

`cache/budapest_osm.json` (42 MB) is the exact map snapshot all `data/osm`
numbers come from; it is not in git, so keep a copy of it.

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
| `distances.csv` | straight-line datasets: 251 × 251 symmetric distance matrix in km |
| `road_distance_km.csv`, `road_duration_min.csv` | road datasets (`"distance_source": "road"`): directed 251 × 251 km and minutes |
| `road_meta.json` | road datasets: OSM vertex of every location (for the street map), source, checks |
| `furniture_clean.csv` | item table used to validate each order's weight and volume |

Index `0` is the depot; customers are `1..250`, matching `location_id` in the
dataset. `load_problem()` validates the matrix shape (symmetry for straight-line,
finite and non-negative for road), the location numbering, and every order
against the furniture table before returning.

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
driving  = distance / avg_speed * 60          (straight-line datasets)
driving  = t(0,c1) + ... + t(ck,0)            (road datasets: road minutes, same for every vehicle)
service  = sum(loading_i) + sum(unloading_i)
total    = driving + service
```

On road data `d` and `t` are directed (one-way streets), so the order of a trip matters.

### Trip cost

```text
fuel         = distance * fuel_l_per_km * fuel_price_per_l
maintenance  = distance * maintenance_per_1000km / 1000
wage_base    = total / 60 * wage_per_h
overflow_fee = max(0, min(volume, cap) - 0.9 * cap) * 100 EUR/m3   (using the 10 % reserve)
base_cost    = fuel + maintenance + wage_base + overflow_fee

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
| Volume 90-100 % of the cargo space (the 10 % reserve) | **allowed, paid** as `overflow_fee` (100 EUR/m³, `volume_overflow_eur_per_m3`) | 0 EUR penalty (it is a real cost) |
| Volume over 100 % of the cargo space | physically impossible, infeasible | 10,000 EUR per m³ over |
| Fewer than 10 customers | soft, proportional | 2,000 EUR per missing customer |
| More than 50 customers | hard in the decoder, also penalised | 1,000,000 EUR per extra customer |
| Shift overrun | allowed, paid as overtime | 0 EUR penalty |
| Last delivery late / back after the day end | planning penalty (see Daily working window) | 1,000 EUR per minute |

A trip is feasible only when the weight, raw-volume (100 %), minimum-count and
maximum-count checks pass. The overflow fee is part of the true cost, so the
report's "Penalty" stays 0 when trips use the reserve. The **minimum of 10 customers is deliberately
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
set in `data/base/params.json` and enforced in `src/cost.py` (`day_metrics`):

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

**Road datasets** (`data/osm`, `data/road_ga`) use the shared 16-hour workday
agreed with the GA team instead: start 08:00, back at the depot by **24:00**,
no separate last-delivery time (first 8 h at the normal wage, the rest at 2x).
On real roads the 250 customers do not fit the 08:00-23:00 window with this
fleet.

Each run also reports the longest vehicle-day, each vehicle's start, last
delivery and finish time, and overtime against the unavoidable minimum
(total work minus 8 h per vehicle). Remove the window keys (`last_delivery_by`, `hard_day_end`) from
`params.json` to get the original model without a daily limit.

---

## Delivery outcomes (simulated)

The optimiser plans the day at 08:00 and cannot know which customers will not
be home or will refuse the furniture. After the PSO, `main.py` runs
`src/delivery_sim.py` on the chosen plan (`--no-sim` switches it off):

* **Stop times:** the planned arrival and departure of every stop (each trip
  loads at the depot, drives stop to stop, unloads each order, drives back);
  they add up exactly to the trip times of the plan.
* **Simulated day:** every stop is independently **not home** with
  probability 5 % (status *attempted*) or **refused** with 2 % (status
  *refused*), otherwise *delivered* (`--not-home`, `--refused`, seed
  `--sim-seed`). The driver keeps a failed item, finishes the trip and brings
  it back to the depot. A failed stop takes the same time as a delivery, so
  **times and real cost do not change**: the benchmark number is the plan's.
  Not-home orders go to a re-delivery list (tomorrow), refused orders back to
  stock.
* **Monte Carlo:** 1,000 simulated days (`--sim-days`): expected attempted,
  refused and failed deliveries per day with 5–95 % ranges, and for each van
  the share of days on which it brings at least one item back.

Results are stored in `best_solution.json` (`delivery_simulation`), so
`--mode report` rebuilds them. The report gains a section **Delivery records
(simulated day)** with count cards, the Monte Carlo table, every delivery
(van, trip, stop, arrival, departure, order, address, district, items, kg,
m³, status; filter by status and search) and the items returned to the depot,
plus `deliveries.csv`, `redelivery.csv` and `returned_to_stock.csv`.
Customers are shown by order ID and address (the data has no names).

## Benchmark instances

`python -m experiments.make_instances` builds the 100 test instances
(250 customers each) into `instances/test/`; `--set all` also builds the
tuning (4), validation (20) and scaling (4 each at 50, 100, 250, 500 and
1,000 customers) sets. Each instance is a complete dataset folder that the
solver reads like `data/base/` (`python main.py --dataset instances/test/test_3001`).

* Same rules as `data/base/generators/`; only the seed changes. Seed 42 with 250
  customers reproduces `data/base/` byte for byte (`tests/test_instances.py`).
* Disjoint seeds per set: tuning 1001-1004, validation 2001-2020,
  test 3001-3100, scaling 4001-4004 (seed 42 is never used).
* Scaling instances repeat the 5-vehicle fleet once per 250 customers.
* The folders (about 90 MB) are not committed. `instances/manifest.json`
  records the SHA-256 of every file;
  `python -m experiments.make_instances --set all --check` verifies a rebuild.
  (The straight-line manifest predates the last `data/base/params.json`
  change; rebuild it with `python -m experiments.make_instances --set all`.)

Road instances work the same way: `python -m experiments.make_road_instances`
builds `road_test` (100), `road_validation` (20) and `road_tuning` (4) from
`data/road_ga` (manifest `instances/road_manifest.json`); all 250 customers,
only the orders change with the seed. `instances_osm/road_tuning/` holds the 4
tuning instances built from **our** `data/osm` (committed in full, ready to run:
`python main.py --dataset instances_osm/road_tuning/road_tune_7001`).

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
fixed; each fix is a setting in `PSOConfig` (`src/pso.py`), measured once
over seeds 1-3 and tested in `tests/test_convergence.py`. (The one-off study
scripts and their raw results were removed; they are in the git history.)

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
main.py                          CLI: decoder checks, PSO, report, road-data, tests
run_tests.py                     runs every test once, prints a per-file summary

src/data_loader.py               the only CSV reader; returns ProblemData
src/cost.py                      trip metrics, cost formulas, overflow fee, penalties, overtime
src/decoder.py                   random keys, greedy cut, Split DP, assignment, checks
src/pso.py                       PSO, construction heuristics, local search, polish
src/kernels.py                   Numba-compiled inner loops: Split DP, assignment, 2-opt
src/report.py                    HTML/PNG/CSV report, street/satellite route map
src/delivery_sim.py              stop times, simulated delivery outcomes, Monte Carlo
src/road_geometry.py             OSM road graph and Dijkstra (shared by pool and map)

experiments/osm_from_geofabrik.py   Geofabrik map file -> cache/budapest_osm.json
experiments/build_road_pool.py      route depot + 1500 pool addresses -> data/osm_pool/
experiments/make_osm_dataset.py     slice the pool -> data/osm/ (feasibility gate)
experiments/road_geometry.py        street geometry of a solution (route_geometry.json)
experiments/import_road.py          one-time import of the GA road file -> data/road_ga/
experiments/make_instances.py       straight-line benchmark instances -> instances/
experiments/make_road_instances.py  road benchmark instances -> instances/, instances_osm/

tests/test_cost.py               hand-checked cost, overtime, overflow fee
tests/test_day_window.py         working day, 2x overtime, day-end rules
tests/test_decoder.py            first-20, full-250, Split/coverage/timing
tests/test_convergence.py        PSO fixes (fast 2-opt, scoring, seeding)
tests/test_kernels.py            Numba kernels give exactly the original results
tests/test_delivery_sim.py       stop times, outcomes, Monte Carlo, main.py integration
tests/test_report.py             report files, tables, charts
tests/test_instances.py          straight-line instance generator
tests/test_road.py               road mode, GA cross-check, OSM data, OSM pipeline

data/base/                       straight-line instance + source data (places pool, furniture, generators)
data/road_ga/                    same instance on the GA team's road matrices (shared reference)
data/osm/                        same instance on our own OSM road matrices (main.py default)
data/osm_pool/                   routed OSM pool: 1501 x 1501 km + minutes (depot + 1500 addresses)
instances/                       benchmark manifests (instance folders rebuilt from seeds)
instances_osm/                   4 road instances on data/osm
cache/                           OSM map file, extract and routed pool (local, not committed)
output/                          latest run: best_solution.json, route_geometry.json, report/
```

---

## Verification and test results

`python run_tests.py` runs all 57 tests (7 files) in about 4 s. All pass.

Hand-checked cases include:

- **Three-customer trip:** 49.910 km, 87.692 min, 39.418 kg, 0.9288 m³,
  25.803496 EUR, with the expected 7-customer soft-minimum penalty.
- **Overtime trip:** 600 min = 10 h, 120 min overtime, 20 EUR extra premium,
  120 EUR total on a 10 EUR/h vehicle.
- **Vehicle-day aggregation:** two 5-hour trips produce 2 hours of overtime
  once for the day, not per trip.
- **Volume reserve:** a trip in the 90-100 % volume band pays the overflow fee
  (100 EUR/m³) and stays feasible; below 90 % there is no fee; above 100 % it
  is infeasible. The Split DP prices the fee exactly like the cost function.
- **Report generator:** produces the HTML report, five PNG charts, and four
  CSV tables (summary, vehicles, trips, history) from a saved solution.
- **Road data:** the GA validator's cost is reproduced exactly; our `data/osm`
  matches the GA matrices, every customer is feasible, instances are exact
  slices (`TestOsmData`); the OSM pipeline is tested on a tiny synthetic map.

`python main.py --mode decoder` (on `data/osm`):

- First 20 customers: 2 trips of 10 customers, every customer exactly once,
  all capacity and count checks pass, penalty 0.
- Full 250 customers decoded from a plain nearest-neighbour tour: every
  customer exactly once, no trip over 50. This raw tour is **not** feasible
  (some trips over 100 % volume, one vehicle back after 24:00); that is
  expected: the check shows what the decoder does with a naive tour, and the
  PSO turns it into a feasible plan.
- Decoder timing: about 65 ms per full 250-customer decode on the development
  machine (road data; straight-line data is faster).

---

## Visual report

`python main.py --mode report` builds `output/report/report.html` from an
existing `output/best_solution.json` (use the same `--dataset` as the run).
The same report is generated automatically after `--mode pso` and `--mode all`.

The HTML file is self-contained (charts are embedded), so it can be opened
directly in a browser without a web server. It includes:

| Output | What it shows |
|---|---|
| Summary cards | true cost, penalty, trips, distance, working time, overtime, runtime, evaluations |
| `convergence.png` | PSO loss/convergence curve, penalty, and cumulative decoder calls |
| `vehicle_timeline.png` | vehicle-day Gantt chart with the 8-hour shift and overtime region |
| `cost_breakdown.png` | fuel, maintenance, wages, volume overflow fee and overtime premium per vehicle (bar totals = vehicle cost), plus total cost composition with EUR legend |
| `trip_capacity.png` | weight/usable-volume utilisation and customers per trip |
| `route_map.png` | routes on the Budapest map (along the streets on road data) and trip distance vs cost |
| `route_map.html` | interactive map: street map, satellite, or satellite + labels (Esri tiles, needs internet); one layer per vehicle, popups per trip and stop |
| Tables | summary, per-vehicle aggregates, and per-trip details (incl. overflow fee per trip) |
| CSV files | `summary.csv`, `vehicles.csv`, `trips.csv`, `history.csv` for Excel/Python analysis |

The final polished cost is marked separately on the convergence chart because
the trip-level polish runs after the last PSO iteration.

---

## Benchmark result

Dataset: 250 customers, 5 vehicles, total demand **5,265.18 kg / 123.98 m³**,
total loading + unloading service time **2,525.6 min (42.1 h)**.

Default configuration (`python main.py`): swarm 30, at most 300 iterations,
early stop after 20 iterations without improvement (`--patience 20`), time
limit 600 s (`--time-limit 600`), inertia 0.72, `c1 = c2 = 1.49`,
`v_max = 0.25`, local search every 5 iterations with budget 40, final
local-search budget 120, seed 42; `polish_candidates` and `seeded_init` on.
The tuning results that informed these values are in `tuning/main/`
(`python -m experiments.tune`).

Result on **`data/osm`** (our OSM road data, the default; the run in `output/`):

| Metric | Value |
|---|---:|
| True cost | **1,734.94 EUR** |
| Penalty | 0.00 EUR (feasible, `check_solution`: no errors) |
| Trips | 19 (10-18 customers each) |
| Distance | 1,058.4 km |
| Working time | 74.15 h |
| Overtime | 34.15 h, all unavoidable (avoidable 0.00 h) |
| Vehicle-day overtime cost | 452.42 EUR |
| Volume overflow fee | 1.65 EUR (2 trips use the 10 % reserve) |
| Runtime | about 150 s, 1,836 decoder evaluations |

| Vehicle | Type | Trips | Day | Overtime | Cost |
|---|---|---:|---|---:|---:|
| 1 | Caddy | 3 | 08:00-20:23 | 4.38 h | 234.50 EUR |
| 2 | Caddy | 4 | 08:00-23:27 | 7.45 h | 312.63 EUR |
| 3 | Sprinter | 4 | 08:00-23:36 | 7.60 h | 361.91 EUR |
| 4 | Sprinter | 4 | 08:00-23:56 | 7.94 h | 366.70 EUR |
| 5 | MAN TGL | 4 | 08:00-22:47 | 6.78 h | 459.20 EUR |

Every vehicle is back before 24:00. The street map draws all 269 legs exactly
on the cost matrix.

Same configuration on **`data/base`** (straight line × 1.3, 08:00-23:00 day):
**1,212.78 EUR**, 15 trips, 764.6 km, 18.14 h overtime (all unavoidable),
feasible. Real roads cost about 43 % more: they are longer and slower, so
there is more driving and more overtime.

---

## Notes and limitations

- Road data (`data/osm`, `data/road_ga`) uses static road-class speeds capped
  by OSM `maxspeed`; no live traffic, turn restrictions or truck-specific
  limits. `data/base` uses straight-line distances × 1.3.
- `data/osm` depends on one OSM snapshot (`cache/budapest_osm.json`, not in
  git); a newer download can change a few routes slightly.
- Packing mode is 1 (keep 10% free volume); three-dimensional box placement
  (packing mode 2) is not modelled.
- The fleet is fixed and every vehicle may perform multiple depot-return
  trips.
- The minimum-customer rule is soft by design; the reported benchmark solution
  has 10–18 customers on every trip.
- Results are deterministic for a fixed seed on the same NumPy/BLAS setup, but
  small platform-dependent floating-point differences can change tie-breaks.
