"""Route the whole address pool on our own OSM road graph -- once.

    python -m experiments.build_road_pool [--download] [--workers 8] [--max-snap-m 150]

Pipeline (every step is cached under ``cache/`` and skipped when done):

1. OSM extract ``cache/budapest_osm.json``: downloaded once from Overpass
   with the query in dataset_road/road_meta.json (the GA team's query), cut
   into 4 x 4 bbox tiles (``cache/osm_tiles/``, resumable) and merged by id;
   or copied there by hand.  An existing file is never downloaded again.
2. Motor-car graph with the GA team's rules (src/road_geometry.graph_from_osm).
3. Drivable core: vertices reachable from AND back to the depot's vertex.
   Snapping only to the core means every pair of points is routable.
4. Snap the depot + every place of dataset/places_pool.csv to its nearest
   core vertex.
5. Time-minimising Dijkstra (src/road_geometry.paths_from) from every
   point to every point, in parallel, saved in resumable chunks.
6. ``cache/road_pool/road_pool.npz`` (index 0 = depot, i = pool row i-1):
   place_id, vertex, snap_m, usable, km (directed), min (directed);
   ``cache/road_pool/pool_meta.json``: OSM checksum, query, graph size,
   excluded places and why.

A place is *usable* if it snapped within --max-snap-m and its depot round
trip is at most --max-round-trip-min.  experiments/make_osm_dataset.py only
picks usable places, so every dataset built from the pool is routable.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import pickle
import re
import time
import urllib.parse
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.road_geometry import CLASSES, graph_from_osm, paths_from, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
GA_META = ROOT / "dataset_road" / "road_meta.json"
OSM_FILE = ROOT / "cache" / "budapest_osm.json"
POOL_DIR = ROOT / "cache" / "road_pool"
CHUNK = 50
EARTH_M = 6371000.0


# --------------------------------------------------------------------------
# Graph helpers
# --------------------------------------------------------------------------
def drivable_core(graph: dict, root: int) -> set[int]:
    """Vertices reachable from ``root`` and from which ``root`` is reachable."""
    def reach(adj: dict) -> set[int]:
        seen, stack = {root}, [root]
        while stack:
            for nxt in adj.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    forward = {u: [e[0] for e in es] for u, es in graph.items()}
    reverse: dict[int, list[int]] = {u: [] for u in graph}
    for u, vs in forward.items():
        for v in vs:
            reverse[v].append(u)
    return reach(forward) & reach(reverse)


def _haversine_m(lon: np.ndarray, lat: np.ndarray, p_lon: float, p_lat: float) -> np.ndarray:
    lat1, lat2 = np.radians(lat), math.radians(p_lat)
    dl = np.radians(p_lon - lon)
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * math.cos(lat2) * np.sin(dl / 2) ** 2
    return EARTH_M * 2 * np.arcsin(np.minimum(1.0, np.sqrt(h)))


def snap(nodes: dict, candidates: list[int], points_lonlat: list[list[float]]) -> tuple[list[int], list[float]]:
    """Nearest candidate vertex of every point (vectorised haversine)."""
    ids = np.asarray(candidates, dtype=np.int64)
    lon = np.asarray([nodes[v][0] for v in candidates])
    lat = np.asarray([nodes[v][1] for v in candidates])
    vertex, dist = [], []
    for p_lon, p_lat in points_lonlat:
        d = _haversine_m(lon, lat, p_lon, p_lat)
        k = int(np.argmin(d))
        vertex.append(int(ids[k]))
        dist.append(float(d[k]))
    return vertex, dist


def nearest_graph_vertex(nodes: dict, graph: dict, point: list[float]) -> int:
    return snap(nodes, list(graph), [point])[0][0]


# --------------------------------------------------------------------------
# Routing (parallel, chunked)
# --------------------------------------------------------------------------
_GRAPH: dict | None = None


def _init_worker(graph_path: str) -> None:
    global _GRAPH
    with open(graph_path, "rb") as f:
        _GRAPH = pickle.load(f)


def route_rows(graph: dict, sources: list[int], vertices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """km and minutes from each source vertex to every point's vertex."""
    targets = set(vertices)
    km = np.full((len(sources), len(vertices)), np.inf)
    mins = np.full((len(sources), len(vertices)), np.inf)
    for r, src in enumerate(sources):
        found = paths_from(graph, src, targets)
        for c, v in enumerate(vertices):
            hit = found.get(v)
            if hit is not None:
                km[r, c] = hit[1] / 1000.0
                mins[r, c] = hit[2] / 60.0
    return km, mins


def _route_chunk(job: tuple[str, list[int], list[int]]) -> str:
    out, sources, vertices = job
    km, mins = route_rows(_GRAPH, sources, vertices)
    tmp = out + ".tmp.npz"
    np.savez(tmp, km=km, min=mins)
    os.replace(tmp, out)
    return out


def route_all(graph: dict, vertices: list[int], part_dir: Path, workers: int) -> tuple[np.ndarray, np.ndarray]:
    part_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(vertices).encode()).hexdigest()[:12]
    jobs = []
    for start in range(0, len(vertices), CHUNK):
        path = part_dir / f"part_{key}_{start:05d}.npz"
        if not path.exists():
            jobs.append((str(path), vertices[start:start + CHUNK], vertices))
    n_parts = math.ceil(len(vertices) / CHUNK)
    print(f"routing {len(vertices)} points: {n_parts - len(jobs)}/{n_parts} chunks cached, "
          f"{len(jobs)} to do with {workers} worker(s)")
    if jobs:
        graph_path = part_dir / "graph.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        t0 = time.time()
        if workers <= 1:
            _init_worker(str(graph_path))
            done = map(_route_chunk, jobs)
        else:
            pool = Pool(workers, initializer=_init_worker, initargs=(str(graph_path),))
            done = pool.imap_unordered(_route_chunk, jobs)
        for i, _ in enumerate(done, 1):
            el = time.time() - t0
            print(f"  chunk {i}/{len(jobs)} done, {el / 60:.1f} min, "
                  f"~{el / i * (len(jobs) - i) / 60:.1f} min left", flush=True)
        if workers > 1:
            pool.close()
            pool.join()
        graph_path.unlink()
    km = np.vstack([np.load(part_dir / f"part_{key}_{s:05d}.npz")["km"] for s in range(0, len(vertices), CHUNK)])
    mins = np.vstack([np.load(part_dir / f"part_{key}_{s:05d}.npz")["min"] for s in range(0, len(vertices), CHUNK)])
    return km, mins


# --------------------------------------------------------------------------
# Whole pipeline on an in-memory OSM dict (also used by the tests)
# --------------------------------------------------------------------------
def build_pool(osm: dict, points_lonlat: list[list[float]], part_dir: Path,
               workers: int = 1, max_snap_m: float = 150.0,
               max_round_trip_min: float = 180.0) -> dict:
    """Point 0 is the depot.  Returns arrays + per-point usability."""
    nodes, graph = graph_from_osm(osm)
    if not graph:
        raise SystemExit("the OSM extract has no drivable roads")
    depot_vertex = nearest_graph_vertex(nodes, graph, points_lonlat[0])
    core = drivable_core(graph, depot_vertex)
    print(f"graph: {len(graph)} vertices, drivable core around the depot: {len(core)}")
    vertex, snap_m = snap(nodes, sorted(core), points_lonlat)
    km, mins = route_all(graph, vertex, part_dir, workers)

    round_trip = mins[0, :] + mins[:, 0]
    reasons: dict[int, str] = {}
    for i in range(1, len(points_lonlat)):
        if not (np.isfinite(km[i]).all() and np.isfinite(km[:, i]).all()):
            reasons[i] = "unroutable"
        elif snap_m[i] > max_snap_m:
            reasons[i] = f"snapped {snap_m[i]:.0f} m from the road"
        elif round_trip[i] > max_round_trip_min:
            reasons[i] = f"depot round trip {round_trip[i]:.0f} min"
    if not (np.isfinite(km[0]).all() and np.isfinite(km[:, 0]).all()):
        raise SystemExit("depot is not connected to the road graph")
    usable = np.array([i == 0 or i not in reasons for i in range(len(points_lonlat))])
    return {
        "vertex": np.asarray(vertex, dtype=np.int64),
        "snap_m": np.asarray(snap_m),
        "snapped_lonlat": np.asarray([nodes[v] for v in vertex]),
        "usable": usable,
        "km": km,
        "min": mins,
        "reasons": reasons,
        "graph_vertices": len(graph),
        "core_vertices": len(core),
    }


# Same public Overpass mirrors as dataset/generators/places_pool.py.
MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


TILES = 4                       # 4 x 4 tiles over the GA bbox
TILE_DIR = ROOT / "cache" / "osm_tiles"
_BBOX = re.compile(r"\(\s*[-\d.]+\s*,\s*[-\d.]+\s*,\s*[-\d.]+\s*,\s*[-\d.]+\s*\)")


MAX_SPLIT = 2                   # a failing tile is cut 2 x 2, at most twice


def split_box(box: list[float], tiles: int) -> list[tuple[str, list[float]]]:
    """[w, s, e, n] -> [("r_c", sub-box)] on a tiles x tiles grid."""
    w, s, e, n = box
    return [(f"{r}_{c}", [w + (e - w) * c / tiles, s + (n - s) * r / tiles,
                          w + (e - w) * (c + 1) / tiles, s + (n - s) * (r + 1) / tiles])
            for r in range(tiles) for c in range(tiles)]


def box_query(query: str, box: list[float]) -> str:
    """The GA query with its bbox replaced (same road filter).

    ``(._;>;)`` pulls every node of a way even outside its tile, so the union
    of the tiles is exactly what the single whole-bbox query returns.
    """
    w, s, e, n = box
    q = re.sub(r"\[timeout:\d+\]", "[timeout:180]", query)
    return _BBOX.sub(f"({s:.5f},{w:.5f},{n:.5f},{e:.5f})", q, count=1)


def tile_queries(query: str, bbox_wsen: list[float], tiles: int = TILES) -> list[tuple[str, str]]:
    """The GA query cut into tiles: [(name, query)]."""
    return [(f"tile_{rc}", box_query(query, box)) for rc, box in split_box(bbox_wsen, tiles)]


def _fetch(url: str, query: str) -> bytes:
    req = urllib.request.Request(
        url, data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"User-Agent": "FurnitureTransportCourse/1.1 (cached academic road extract)"})
    with urllib.request.urlopen(req, timeout=240) as r:
        body = r.read()
    data = json.loads(body)
    if data.get("remark") or "elements" not in data:
        raise ValueError("incomplete answer: " + str(data.get("remark", ""))[:200])
    return body


def merge_tiles(parts: list[dict]) -> dict:
    """Union of tile answers, each node/way once (ways keep their full node list)."""
    seen: dict[tuple[str, int], dict] = {}
    for part in parts:
        for el in part["elements"]:
            seen.setdefault((el["type"], el["id"]), el)
    return {"elements": list(seen.values())}


def download_tiled(query: str, bbox_wsen: list[float], dest: Path, tile_dir: Path = TILE_DIR) -> None:
    """Fetch missing tiles (one try per mirror, pauses between requests), then merge.

    Each tile is saved as it arrives; a failed run keeps them, so the next run
    only asks for what is still missing.  Never loops beyond 3 tries a tile.
    """
    tile_dir.mkdir(parents=True, exist_ok=True)

    def ensure(name: str, box: list[float], depth: int) -> list[Path]:
        """Files covering ``box``: the tile itself, or its 2 x 2 split."""
        path = tile_dir / f"{name}.json"
        if path.exists():
            return [path]
        split_mark = tile_dir / f"{name}.split"
        if not split_mark.exists():
            for url in MIRRORS:
                try:
                    body = _fetch(url, box_query(query, box))
                    path.write_bytes(body)
                    print(f"  {name}: {len(body) / 1e6:.1f} MB from {url.split('/')[2]}", flush=True)
                    time.sleep(5)
                    return [path]
                except (OSError, ValueError) as ex:
                    print(f"  {name}: {url.split('/')[2]} failed ({ex})", flush=True)
                    time.sleep(20)
            if depth >= MAX_SPLIT:
                raise SystemExit(f"{name} failed on every mirror; rerun later (finished tiles are kept)")
            print(f"  {name}: splitting into 4 smaller tiles", flush=True)
            split_mark.touch()
        files: list[Path] = []
        for rc, sub in split_box(box, 2):
            files += ensure(f"{name}_{rc}", sub, depth + 1)
        return files

    tiles = split_box(bbox_wsen, TILES)
    done = sum((tile_dir / f"tile_{rc}.json").exists() or (tile_dir / f"tile_{rc}.split").exists()
               for rc, _ in tiles)
    print(f"OSM tiles: {done}/{len(tiles)} cached or split, the rest to download")
    files: list[Path] = []
    for rc, box in tiles:
        files += ensure(f"tile_{rc}", box, 0)
    parts = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    merged = merge_tiles(parts)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(merged), encoding="utf-8")
    ways = sum(1 for e in merged["elements"] if e["type"] == "way")
    print(f"merged {len(files)} tiles into {dest}: {len(merged['elements'])} elements, {ways} ways")


def read_points() ->tuple[list[str], list[list[float]]]:
    """Depot (from dataset/locations.csv) + every pool place, as [lon, lat]."""
    with open(DATASET / "locations.csv", newline="", encoding="utf-8") as f:
        depot = next(r for r in csv.DictReader(f) if r["type"] == "depot")
    with open(DATASET / "places_pool.csv", newline="", encoding="utf-8") as f:
        pool = list(csv.DictReader(f))
    place_ids = [""] + [r["place_id"] for r in pool]
    points = [[float(depot["lon"]), float(depot["lat"])]]
    points += [[float(r["lon"]), float(r["lat"])] for r in pool]
    return place_ids, points


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--osm", type=Path, default=OSM_FILE)
    ap.add_argument("--out", type=Path, default=POOL_DIR)
    ap.add_argument("--download", action="store_true",
                    help="fetch the OSM extract once if --osm does not exist")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-snap-m", type=float, default=150.0)
    ap.add_argument("--max-round-trip-min", type=float, default=180.0)
    a = ap.parse_args()

    ga = json.loads(GA_META.read_text(encoding="utf-8"))
    source = ga["osm_extract_source"]
    if a.osm.exists():
        print(f"using cached extract {a.osm}")
    elif a.download:
        download_tiled(source["query"], source["bbox_wsen"], a.osm)
    else:
        raise SystemExit(f"{a.osm} not found: add --download (one Overpass request) "
                         "or copy the GA team's cache/budapest_city_osm.json there")
    osm_sha = sha256_file(a.osm)
    frozen = osm_sha == ga.get("osm_raw_sha256")
    print(f"OSM sha256 {osm_sha[:12]} ({'same as' if frozen else 'differs from'} the GA team's extract)")

    place_ids, points = read_points()
    print("reading the road graph...")
    osm = json.loads(a.osm.read_text(encoding="utf-8"))
    res = build_pool(osm, points, a.out / "parts", a.workers, a.max_snap_m, a.max_round_trip_min)
    del osm

    a.out.mkdir(parents=True, exist_ok=True)
    npz = a.out / "road_pool.npz"
    np.savez_compressed(npz, place_id=np.asarray(place_ids), vertex=res["vertex"],
                        snap_m=res["snap_m"], snapped_lonlat=res["snapped_lonlat"],
                        usable=res["usable"], km=res["km"], min=res["min"],
                        lonlat=np.asarray(points))
    meta = {
        "built": datetime.date.today().isoformat(),
        "points": len(points),
        "usable_places": int(res["usable"].sum()) - 1,
        "excluded": {place_ids[i]: why for i, why in sorted(res["reasons"].items())},
        "max_snap_m": a.max_snap_m,
        "max_round_trip_min": a.max_round_trip_min,
        "osm_file": str(a.osm.relative_to(ROOT) if a.osm.is_relative_to(ROOT) else a.osm),
        "osm_raw_sha256": osm_sha,
        "osm_is_ga_extract": frozen,
        "osm_extract_source": {"query": source["query"], "bbox_wsen": source.get("bbox_wsen"),
                               "download": f"Overpass, {TILES}x{TILES} bbox tiles merged by element id"},
        "graph_vertices": res["graph_vertices"],
        "core_vertices": res["core_vertices"],
        "speed_assumptions_kmh": CLASSES,
        "pool_sha256": sha256_file(npz),
    }
    (a.out / "pool_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"saved {npz}: {meta['usable_places']} usable places, {len(meta['excluded'])} excluded")


if __name__ == "__main__":
    main()
