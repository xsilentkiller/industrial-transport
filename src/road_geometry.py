"""Road geometry of a solution's trips, for drawing routes on the map.

The cost model only needs the road matrices in ``data/road_ga/``.  To *draw*
a route along the streets we also need the road graph itself: this module
rebuilds it from an OpenStreetMap extract with the GA team's rules
(``budapest_osm.graph_from_osm`` in huyuxuan028/transport-algorithms: same
road classes, speeds, maxspeed cap, one-way handling) and runs the same
time-minimising Dijkstra between the same OSM vertices.

Every drawn leg is checked against the matrix the optimiser used: its length
and time must match ``road_distance_km.csv`` / ``road_duration_min.csv``
within a tolerance.  If the OSM extract is the frozen one the matrices were
built from (same SHA-256), legs match exactly; a fresh download can differ
slightly because OSM changes, and any leg outside the tolerance is reported.
Nothing is invented: a leg that cannot be routed is drawn as a dashed
straight line and listed as such.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from pathlib import Path

# Road classes and default speeds (km/h) -- identical to the GA team's graph.
CLASSES = {
    "motorway": 60, "motorway_link": 40, "trunk": 50, "trunk_link": 40,
    "primary": 40, "primary_link": 30, "secondary": 35, "secondary_link": 30,
    "tertiary": 30, "tertiary_link": 25, "residential": 25, "unclassified": 25,
    "living_street": 15, "service": 15,
}
BLOCKED_ACCESS = {"no", "private", "agricultural", "forestry"}


def metres(a: list[float], b: list[float]) -> float:
    """Haversine metres between two [lon, lat] points."""
    lat1, lat2 = math.radians(a[1]), math.radians(b[1])
    dl = math.radians(b[0] - a[0])
    dp = lat2 - lat1
    h = math.sin(dp / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dl / 2) ** 2
    return 6371000 * 2 * math.asin(min(1.0, math.sqrt(h)))


def graph_from_osm(data: dict) -> tuple[dict[int, list[float]], dict[int, list[tuple[int, float, float]]]]:
    """Directed motor-car graph: node -> [(next, metres, seconds)]."""
    nodes = {e["id"]: [e["lon"], e["lat"]] for e in data["elements"] if e["type"] == "node"}
    graph: dict[int, list[tuple[int, float, float]]] = {}
    for e in data["elements"]:
        if e["type"] != "way":
            continue
        t = e.get("tags", {})
        kind = t.get("highway")
        access = next((t[k] for k in ("motorcar", "motor_vehicle", "vehicle", "access") if k in t), "yes")
        if kind not in CLASSES or access in BLOCKED_ACCESS or t.get("area") == "yes":
            continue
        speed = CLASSES[kind]
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(mph)?", t.get("maxspeed", ""))
        if m:
            speed = min(speed, float(m[1]) * (1.609344 if m[2] else 1))
        if speed <= 0:
            continue
        default = "yes" if t.get("junction") == "roundabout" or kind == "motorway" else "no"
        direction = t.get("oneway", default)
        for u, v in zip(e["nodes"], e["nodes"][1:]):
            if u not in nodes or v not in nodes:
                continue
            dist = metres(nodes[u], nodes[v])
            secs = dist / speed * 3.6
            graph.setdefault(u, [])
            graph.setdefault(v, [])
            if direction not in ("-1", "reverse"):
                graph[u].append((v, dist, secs))
            if direction not in ("yes", "1", "true"):
                graph[v].append((u, dist, secs))
    return nodes, graph


def paths_from(graph, source: int, targets: set[int]) -> dict[int, tuple[list[int], float, float]]:
    """Fastest path from ``source`` to each target: (nodes, metres, seconds)."""
    remaining = set(targets) - {source}
    best = {source: 0.0}
    length = {source: 0.0}
    parent: dict[int, int] = {}
    queue = [(0.0, source)]
    while queue and remaining:
        t, u = heapq.heappop(queue)
        if t != best[u]:
            continue
        remaining.discard(u)
        for v, d, s in graph.get(u, ()):
            nt = t + s
            if nt < best.get(v, math.inf):
                best[v], length[v], parent[v] = nt, length[u] + d, u
                heapq.heappush(queue, (nt, v))
    out = {}
    for target in targets:
        if target == source:
            out[target] = ([source], 0.0, 0.0)
        elif target in best:
            path = [target]
            while path[-1] != source:
                path.append(parent[path[-1]])
            out[target] = (path[::-1], length[target], best[target])
    return out


def nearest_vertex(nodes: dict, graph: dict, point: list[float]) -> int:
    return min(graph, key=lambda n: metres(nodes[n], point))


def trip_geometry(
    trips: list[dict],
    osm: dict,
    road_meta: dict,
    coords_lonlat: list[list[float]],
    dist_km,
    dur_min,
    tolerance: float = 0.02,
    progress=None,
) -> dict:
    """Polylines for every trip, checked leg by leg against the road matrices.

    ``trips``: [{"vehicle_id", "customers": [...]}] in location ids (0 = depot).
    Returns {"trips": [{"vehicle_id", "legs": [{"from", "to", "points": [[lat, lon], ...],
    "km", "min", "matrix_km", "matrix_min", "status"}]}], "summary": {...}}.
    """
    nodes, graph = graph_from_osm(osm)
    ids = list(road_meta.get("osm_node_ids") or [])
    vertex = []
    resnapped = 0
    for loc, point in enumerate(coords_lonlat):
        v = ids[loc] if loc < len(ids) else None
        if v not in graph:            # node absent from this extract
            v = nearest_vertex(nodes, graph, point)
            resnapped += 1
        vertex.append(v)

    needed: dict[int, set[int]] = {}
    for trip in trips:
        stops = [0] + list(trip["customers"]) + [0]
        for a, b in zip(stops, stops[1:]):
            needed.setdefault(vertex[a], set()).add(vertex[b])
    routed = {}
    for i, (src, targets) in enumerate(needed.items(), 1):
        routed[src] = paths_from(graph, src, targets)
        if progress and i % 25 == 0:
            progress(i, len(needed))

    out_trips = []
    counts = {"exact": 0, "close": 0, "differs": 0, "unroutable": 0}
    for trip in trips:
        stops = [0] + list(trip["customers"]) + [0]
        legs = []
        for a, b in zip(stops, stops[1:]):
            mk, mm = float(dist_km[a][b]), float(dur_min[a][b])
            hit = routed[vertex[a]].get(vertex[b])
            if hit is None:
                status = "unroutable"
                pts = [[coords_lonlat[a][1], coords_lonlat[a][0]], [coords_lonlat[b][1], coords_lonlat[b][0]]]
                km = mins = None
            else:
                path, metres_, secs = hit
                km, mins = metres_ / 1000.0, secs / 60.0
                pts = [[coords_lonlat[a][1], coords_lonlat[a][0]]]
                pts += [[nodes[n][1], nodes[n][0]] for n in path]
                pts += [[coords_lonlat[b][1], coords_lonlat[b][0]]]
                err = max(abs(km - mk) / max(mk, 0.05), abs(mins - mm) / max(mm, 0.1))
                status = "exact" if err < 1e-3 else "close" if err <= tolerance else "differs"
            counts[status] += 1
            legs.append({"from": a, "to": b, "points": [[round(p[0], 6), round(p[1], 6)] for p in pts],
                         "km": km, "min": mins, "matrix_km": mk, "matrix_min": mm, "status": status})
        out_trips.append({"vehicle_id": trip["vehicle_id"], "customers": list(trip["customers"]), "legs": legs})

    return {
        "trips": out_trips,
        "summary": {**counts, "legs": sum(counts.values()), "tolerance": tolerance,
                    "resnapped_locations": resnapped,
                    "graph_nodes": len(graph)},
    }


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_geometry(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
