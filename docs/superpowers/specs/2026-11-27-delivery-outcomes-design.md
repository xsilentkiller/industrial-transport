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
- One simulated day with an outcome for every stop.
- Monte Carlo over many simulated days: expected number of attempted and
  refused deliveries and of re-deliveries (with ranges).
- Re-delivery list (not home → tomorrow) and returned-to-stock list
  (refused), as CSV files.
- Everything reproducible from a seed; rebuildable with `--mode report`.

## Non-goals

- No change to the optimiser, its objective, or the tuned PSO settings. The
  planned real cost stays the benchmark number.
- No same-day re-delivery and no re-planning during the day (option 3; later,
  together with rush orders).
- No change to times or cost: a failed stop takes the same time as a
  delivery (user's decision: "stop time will count as same as delivery
  time"), and no extra depot time is added.  The dataset is not modified.
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
| Delivered | its unloading time (as planned) | delivered |
| Attempted (not home) | the same time as a delivery | stays in the van |
| Refused | the same time as a delivery | stays in the van |

Times are exactly the planned times for every outcome (user's decision), so
the day's schedule and real cost do not change.  The failed items ride back
to the depot and are listed as returned there when the trip ends (time of the
return = the trip's planned end); no extra unloading time is added.  A failed
item is never put on a later trip.

### Monte Carlo

N simulated days (default 1,000; seeds derived from the base seed). Reported:
mean and 5 %/95 % range of attempted, refused and re-delivery counts; the
share of days with at least one failed delivery per van; the orders most
often affected are not reported (each stop has the same probability).

## Configuration

`SimConfig` dataclass in `src/delivery_sim.py` (defaults shown), with
`main.py` flags:

| Field | Default | Flag |
|---|---|---|
| `p_not_home` | 0.05 | `--not-home` |
| `p_refused` | 0.02 | `--refused` |
| `seed` | 42 | `--sim-seed` |
| `monte_carlo_days` | 1000 | `--sim-days` |
| enabled | on | `--no-sim` switches it off |

## Outputs

`best_solution.json` gains `delivery_simulation`: config, per-stop records,
depot returns, Monte Carlo summary.

`output/report/` gains:

- `deliveries.csv`: one row per stop — vehicle, trip, stop, arrival and
  departure time, order ID, address, district, items (names), kg, m³,
  status.
- `redelivery.csv`: not-home orders for tomorrow.
- `returned_to_stock.csv`: refused orders.

`report.html` gains a section **"Delivery records (simulated day)"**:

- count cards: delivered / attempted (not home) / refused / re-deliveries
  tomorrow;
- Monte Carlo table (expected counts and 5–95 % ranges);
- the records table, sortable/filterable by status with a small inline
  script (works offline, no libraries), and a line per depot return.

Sketch:

```
Delivery records — simulated day (seed 42, not home 5 %, refused 2 %)
  Delivered 233   Attempted (not home) 12   Refused 5   Re-deliveries tomorrow 12
  [All | Delivered | Attempted | Refused]   [search]
  Van       Trip Stop Arrive Leave Order Customer (address)          Items           kg   m³   Status
  1 Caddy   1    1    08:52  09:01 #175  Tímár utca 1, 1036 (III.)   Dining Chair ×2 23.0 0.48 Delivered
  ...
  Back at depot 11:24: returned #67 (Wardrobe) → re-delivery list
```

## Testing

- Planned stop times add up to each trip's `time_min` and to the vehicle's
  total; first stop / last return consistent with `day_schedule`.
- p = 0 → every stop delivered.
- p_not_home = 1 → every stop attempted, nothing delivered, all orders on
  the re-delivery list.
- Times and real cost are identical for every outcome mix.
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
