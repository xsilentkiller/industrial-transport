"""OSM road extract from the Geofabrik Hungary file (when Overpass is down).

    curl -L -o cache/geofabrik/hungary-latest.osm.pbf https://download.geofabrik.de/europe/hungary-latest.osm.pbf
    python -m experiments.osm_from_geofabrik
    python -m experiments.build_road_pool          # finds cache/budapest_osm.json, no download

Reproduces the GA team's Overpass query (data/road_ga/road_meta.json):

    way[highway][highway!~"^(footway|path|cycleway|steps|pedestrian|construction|
        proposed|track|bridleway)$"](47.36,18.91,47.64,19.26); (._;>;); out body;

= every way with such a highway tag that has a node inside the bbox, plus
all nodes of those ways (also outside the bbox), written in Overpass JSON
(``{"elements": [nodes..., ways...]}``) so graph_from_osm reads it unchanged.
Only difference: Overpass also keeps a way that crosses the bbox edge
without a node inside it -- at the edge of the bbox, far from the depot.

Needs ``pip install osmium`` (only this script).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import osmium

ROOT = Path(__file__).resolve().parents[1]
PBF = ROOT / "cache" / "geofabrik" / "hungary-latest.osm.pbf"
OUT = ROOT / "cache" / "budapest_osm.json"
GA_META = ROOT / "data" / "road_ga" / "road_meta.json"


def query_rules(query: str) -> tuple[set[str], list[float]]:
    """Excluded highway values and [w, s, e, n] from the GA Overpass query."""
    excluded = set(re.search(r'highway!~"\^\(([^)]*)\)\$"', query)[1].split("|"))
    s, w, n, e = map(float, re.search(r"\]\(([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\)", query).groups())
    return excluded, [w, s, e, n]


def extract(pbf: Path, excluded: set[str], bbox_wsen: list[float]) -> dict:
    w, s, e, n = bbox_wsen
    ways, nodes = [], {}
    fp = (osmium.FileProcessor(str(pbf))
          .with_locations()
          .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
          .with_filter(osmium.filter.KeyFilter("highway")))
    for way in fp:
        if way.tags.get("highway") in excluded:
            continue
        refs, inside = [], False
        for nd in way.nodes:
            loc = nd.location
            if not loc.valid():
                continue
            refs.append((nd.ref, loc.lat, loc.lon))
            if s <= loc.lat <= n and w <= loc.lon <= e:
                inside = True
        if not inside:
            continue
        for ref, lat, lon in refs:
            nodes[ref] = (lat, lon)
        ways.append({"type": "way", "id": way.id, "nodes": [r for r, _, _ in refs],
                     "tags": {t.k: t.v for t in way.tags}})
    elements = [{"type": "node", "id": i, "lat": round(la, 7), "lon": round(lo, 7)}
                for i, (la, lo) in nodes.items()]
    return {"elements": elements + ways}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pbf", type=Path, default=PBF)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} already exists; delete it first to rebuild")
    query = json.loads(GA_META.read_text(encoding="utf-8"))["osm_extract_source"]["query"]
    excluded, bbox = query_rules(query)
    print(f"reading {a.pbf.name}: highway ways in bbox {bbox}, excluding {sorted(excluded)}")
    data = extract(a.pbf, excluded, bbox)
    ways = sum(1 for el in data["elements"] if el["type"] == "way")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(a.out)
    print(f"wrote {a.out}: {ways} ways, {len(data['elements']) - ways} nodes "
          f"({a.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
