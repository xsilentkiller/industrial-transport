# OSM road pool: all 1,500 addresses, routed once

Road distance and driving time between **every pair** of the depot and the
1,500 addresses of `data/base/places_pool.csv`, on the OpenStreetMap road
network of Budapest. Committed so colleagues can build road datasets in
seconds without the map file or the ~4 min routing step.

| File | Content |
|---|---|
| `road_pool.npz` | NumPy arrays, index 0 = depot, index i = row i-1 of `places_pool.csv` |
| `pool_meta.json` | OSM file checksum, road query, graph size, speeds, excluded places |
| `build.log` | output of the build |

Arrays in `road_pool.npz`:

| Array | Shape | Meaning |
|---|---|---|
| `km` | 1501 × 1501 | road distance in km, **directed** (`km[a, b]` = from a to b; one-way streets) |
| `min` | 1501 × 1501 | driving minutes along the same path, directed |
| `place_id` | 1501 | `place_id` of `places_pool.csv` ("" for the depot) |
| `usable` | 1501 | True if the place may be used in a dataset |
| `vertex`, `snap_m`, `lonlat`, `snapped_lonlat` | 1501 | OSM road vertex each address snapped to, snap distance, coordinates |

Usable: **1,497 of 1,500** places. Excluded (snapped too far from a drivable
road, limit 150 m): place 39 (586 m), 367 (155 m), 1263 (290 m).

```python
import numpy as np
z = np.load("data/osm_pool/road_pool.npz")
km, minutes = z["km"], z["min"]      # km[0, 5] = depot -> place row 4
```

Build a 250-customer road dataset from it (no routing, seconds):

```powershell
python -m experiments.make_osm_dataset --out data/osm_s9 --customers 250 --seed 9
```

## How it was built

```powershell
python -m experiments.build_road_pool --out data\osm_pool
```

on 2026-11-27 from `cache/budapest_osm.json` (Geofabrik Hungary extract,
sha256 `d3283df7c5fb…`; the `download` note in `pool_meta.json` is the
script's default text). Same road classes, speeds, one-way rules and
time-shortest Dijkstra as `data/osm/`.

`data/osm/` (the benchmark dataset) was cut from an earlier routing and is
**not** regenerated from this pool, so benchmark results stay comparable. For
its 251 locations this pool agrees with `data/osm/` within 3.4 m and 0.4 s
per leg.

The resumable routing chunks (`parts/`) are not committed (`.gitignore`).

Map data © OpenStreetMap contributors, available under the Open Database
License (ODbL), https://www.openstreetmap.org/copyright.
