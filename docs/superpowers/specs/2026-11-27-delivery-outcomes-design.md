# Delivery outcomes: failed deliveries, simulated day, delivery records

Date: 2026-11-27 · Status: draft for approval

## Why

Real deliveries fail: a customer is not home, or refuses the furniture. The
project should show how the plan copes with that, and the report should list
every delivery (who, what, when, outcome), as a delivery company would keep
records.

## What the user asked for

1. **Risk measure (Monte Carlo):** evaluate the optimised plan under random
   failures and report what to expect.
2. **Run the day:** one simulated day, stop by stop. A stop is **delivered**,
   **attempted (not home)** or **refused**. The driver keeps a failed item,
   finishes the trip, unloads it at the depot, loads the next trip, continues.
3. **Delivery records** in `report.html`: one row per delivery with time,
   order, customer, items, outcome; counts of delivered / attempted / refused.

Confirmed by the user: not home 5 %, refused 2 %.

## Goals

- Planned arrival time for every stop (the plan does not store it today).
- One simulated day with outcomes, actual times and actual real cost next to
  the planned ones.
- Monte Carlo over many simulated days: expected failures, extra time and
  cost, chance of a van finishing after 24:00.
- Re-delivery list (not home → tomorrow) and returned-to-stock list
  (refused), as CSV files.
- Everything reproducible from a seed; rebuildable with `--mode report`.

## Non-goals

- No change to the optimiser, its objective, or the tuned PSO settings. The
  planned real cost stays the benchmark number.
- No same-day re-delivery and no re-planning during the day (option 3; later,
  together with rush orders).
- No invented customer names: customers are shown by order ID and address
  (the data has no names).
- No change to the instructor's rules (16 h day 08:00–24:00, 2× overtime
  after 8 h, 250 customers, 5 vans).

## Anti-goals

- Results of the optimiser must not change because of this feature
  (`best_solution` identical with and without the simulation).
- The simulated day must not be presented as the plan: the report labels it
  "simulated" with its seed and probabilities.

## Approach

Considered:

- **A. New module `src/delivery_sim.py`, results stored in the payload
  (chosen).** `main.py` calls it after the PSO; the results go into
  `best_solution.json` under `delivery_simulation`; `report.py` only
  displays them. Keeps the optimiser untouched and the report rebuildable.
- B. Inside the optimiser (failures in the objective): changes the tuned
  search and the benchmark; rejected (anti-goal).
- C. Inside `report.py`: mixes calculation with display, and `--mode report`
  would re-simulate; rejected.

## Model

### Stop times (planned)

Each vehicle starts at 08:00 and runs its trips in the plan's order (the same
order as `Solution.day_schedule`). A trip:

1. loads all its orders at the depot: `sum(loading_min)` of its customers;
2. for each stop in order: drives the leg (road duration in road mode,
   otherwise km / vehicle speed, as in `cost.drive_min`), **arrives**,
   unloads (`unloading_min` of that order), **departs**;
3. drives back to the depot.

The sum equals the trip's `time_min` (checked by a test), so planned times
match the existing schedule exactly.

### Outcomes (simulated)

Per stop, independently: not home with p = 0.05, refused with p = 0.02,
otherwise delivered. Drawn with `numpy.random.default_rng(seed)`.

| Outcome | Time at the stop | Item |
|---|---|---|
| Delivered | its unloading time | delivered |
| Attempted (not home) | 5 min wait, no unloading | stays in the van |
| Refused | 5 min, no unloading | stays in the van |

Back at the depot after a trip, every failed item is unloaded using that
order's loading time (handling time in the data) before the next trip is
loaded. Times after a failure shift (earlier or later); the route itself does
not change.

### Actual real cost

Same km (fuel, maintenance unchanged). Wages and the 2× overtime / night
premiums are recomputed from each van's actual working minutes with the
existing `day_metrics`. Reported: planned vs actual real cost and the
difference; actual finish time per van; a warning if a van would end after
24:00.

### Monte Carlo

N simulated days (default 1,000; seeds derived from the base seed). Reported:
mean and 5 %/95 % range of attempted, refused, extra minutes, actual real
cost; share of days on which any van finishes after 24:00 (and the worst
finish time).

## Configuration

`SimConfig` dataclass in `src/delivery_sim.py` (defaults shown), with
`main.py` flags:

| Field | Default | Flag |
|---|---|---|
| `p_not_home` | 0.05 | `--not-home` |
| `p_refused` | 0.02 | `--refused` |
| `attempt_min` | 5.0 | `--attempt-min` |
| `seed` | 42 | `--sim-seed` |
| `monte_carlo_days` | 1000 | `--sim-days` |
| enabled | on | `--no-sim` switches it off |

## Outputs

`best_solution.json` gains `delivery_simulation`: config, per-stop records,
per-vehicle actual day, depot returns, Monte Carlo summary.

`output/report/` gains:

- `deliveries.csv`: one row per stop — vehicle, trip, stop, planned arrival,
  actual arrival, order ID, address, district, items (names), kg, m³,
  status.
- `redelivery.csv`: not-home orders for tomorrow.
- `returned_to_stock.csv`: refused orders.

`report.html` gains a section **"Delivery records (simulated day)"**:

- count cards: delivered / attempted (not home) / refused / re-deliveries
  tomorrow; planned vs actual real cost;
- Monte Carlo table (expected values and 5–95 % ranges, 24:00 risk);
- the records table, sortable/filterable by status with a small inline
  script (works offline, no libraries), and a line per depot return.

Sketch:

```
Delivery records — simulated day (seed 42, not home 5 %, refused 2 %)
  Delivered 233   Attempted (not home) 12   Refused 5   Re-deliveries tomorrow 12
  [All | Delivered | Attempted | Refused]   [search]
  Van       Trip Stop Planned Actual Order Customer (address)          Items           kg   m³   Status
  1 Caddy   1    1    08:52   08:52  #175  Tímár utca 1, 1036 (III.)   Dining Chair ×2 23.0 0.48 Delivered
  ...
  Back at depot 11:24: unloaded #67 (Wardrobe) → re-delivery list
```

## Testing

- Planned stop times add up to each trip's `time_min` and to the vehicle's
  total; first stop / last return consistent with `day_schedule`.
- p = 0 → every stop delivered, actual = planned times and cost exactly.
- p_not_home = 1 → every stop attempted, nothing delivered, all orders on
  the re-delivery list, depot unloading added.
- Same seed → same outcomes; different seed → different outcomes.
- Capacity: a failed item is never added to a later trip's load.
- Monte Carlo: deterministic for a seed; mean failure rate close to 7 % over
  many days.
- Optimiser untouched: `best_solution` identical with `--no-sim`.
- Report: section and CSV files present; `--mode report` rebuilds them from
  the JSON.

## Left to the builder

Exact HTML/CSS of the section (follow the existing report style), CSV
column order, how seeds for Monte Carlo days are derived, rounding.
