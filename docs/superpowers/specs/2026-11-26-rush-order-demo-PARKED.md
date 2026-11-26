# Rush-order demo — PARKED (decisions so far, not an approved design)

Date: 2026-11-26 · Status: parked until after PSO tuning. Resume the design
conversation from here; nothing has been built.

## Why

A live demonstration in the presentation that shows how flexible the model is:
a new urgent order arrives during the day and the plan adapts.

## Decided with the user

- **Interactive demo page** (option B of: one command / interactive page /
  scripted scenario).
- **Layout C — "play the day"**: map on top, the vans' day as a timeline
  underneath with a moving clock; press play, the vans move; add a rush order
  at e.g. 11:00 and watch the timeline re-shape. Mockup:
  `.superpowers/brainstorm/94-1795724226/content/rush-layout.html`.
- **Day rules unchanged** (instructor's condition): vans start at 08:00, back
  by 24:00, overtime 2x after 8 h, 5 vans, 250 customers.
- **Vans may leave the depot later** (minutes or hours) — e.g. wait for a rush
  order to be ready.

## Proposed by Claude, not yet confirmed

- Furniture is loaded at the depot: trips already under way are locked (items
  on board); every customer not yet loaded + the rush order is re-planned onto
  vans that become free at different times.
- Each order gets a ready time (normal 08:00; rush = arrival + preparation,
  e.g. 30 min). A trip cannot leave before all its items are ready.
- Waiting at the depot is paid working time (counts for wages, overtime and the
  24:00 limit).
- The rush address is a click on the map, routed on the fly with the OSM graph
  (`cache/budapest_osm.json`, helpers in `experiments/build_road_pool.py`).
- Rush order must be served the same day; if impossible, the page says so.
- Local Python server (stdlib) + one page with Leaflet (same as
  `route_map.html`; needs internet for map tiles).
- Re-plan target ~20–40 s, using the tuned PSO with early stopping.
- With no rush order the solver must behave exactly as today (so tuning results
  stay valid).

## Open questions for when we resume

- Animation speed (one day in ~60 s?), and whether a rush order can be added
  while playing or only when paused.
- How items for the rush order are chosen (catalogue dropdown?).
- What the "what changed" panel lists (cost before/after real cost, penalty,
  van, waiting, customers moved).
- Reset / several rush orders per day / saving a scenario.

## Order of work agreed

1. PSO tuning (`python -m experiments.tune --name main`), adopt the winner.
2. Then resume this design, write the full spec, plan, build.
