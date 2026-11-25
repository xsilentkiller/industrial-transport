"""Road mode: GA team's OSM road matrices (data/road_ga/) and our own
OSM road data (data/osm/), plus the OSM pipeline on a tiny synthetic map."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.cost import day_metrics, leg_sum, return_leg_min, segment_stats, trip_metrics
from src.data_loader import load_problem
from src.decoder import DecoderConfig, decode_tour, greedy_cut, split_dp
from src.pso import nearest_neighbor_tour, two_opt_order

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def naive_directed_two_opt(order, D):
    """Recompute the whole directed trip for every reversal."""
    def cost(o):
        return leg_sum(D, np.asarray(o, dtype=np.int64))
    order = list(order)
    best = cost(order)
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 2, len(order)):
                cand = order[:i] + order[i : j + 1][::-1] + order[j + 1 :]
                c = cost(cand)
                if c < best - 1e-9:
                    order, best = cand, c
                    improved = True
    return order


class TestRoadData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.road = load_problem(ROOT / "data" / "road_ga")
        cls.euclid = load_problem(ROOT / "data" / "base")

    def test_loaded_as_road(self) -> None:
        self.assertTrue(self.road.road_mode)
        self.assertFalse(self.euclid.road_mode)
        self.assertFalse(self.road.symmetric)          # one-way streets
        self.assertEqual(self.road.duration.shape, (251, 251))

    def test_same_orders_and_locations(self) -> None:
        for name in ("weight", "volume", "loading", "unloading", "lat", "lon"):
            np.testing.assert_array_equal(getattr(self.road, name), getattr(self.euclid, name))

    def test_driving_time_is_road_time(self) -> None:
        trip = [5, 17, 42, 99, 120, 7, 230, 64, 11, 150]
        stats = segment_stats(self.road, trip)
        expected = leg_sum(self.road.duration, np.asarray(trip))
        self.assertAlmostEqual(stats.road_min, expected, places=9)
        for k in range(self.road.n_vehicles):          # same for every vehicle
            m = trip_metrics(self.road, k, trip)
            self.assertAlmostEqual(m.driving_min, expected, places=9)
        self.assertAlmostEqual(return_leg_min(self.road, 0, 64), self.road.duration[64, 0])

    def test_euclid_unchanged(self) -> None:
        m = trip_metrics(self.euclid, 2, [5, 17, 42])
        self.assertAlmostEqual(m.driving_min, m.distance_km / 48.0 * 60.0, places=9)

    def test_split_and_greedy_road_minutes(self) -> None:
        tour = nearest_neighbor_tour(self.road)[:80]
        for seg in split_dp(self.road, tour, DecoderConfig()) + greedy_cut(self.road, tour):
            ref = segment_stats(self.road, seg.customers)
            self.assertAlmostEqual(seg.road_min, ref.road_min, places=6)
            self.assertAlmostEqual(seg.distance_km, ref.distance_km, places=6)

    def test_directed_two_opt_matches_naive(self) -> None:
        rng = np.random.default_rng(3)
        D = self.road.leg_cost
        for _ in range(10):
            order = [int(c) for c in rng.choice(np.arange(1, 251), size=11, replace=False)]
            self.assertEqual(two_opt_order(order, D, symmetric=False),
                             naive_directed_two_opt(order, D))

    def test_decoded_solution_is_consistent(self) -> None:
        sol = decode_tour(self.road, nearest_neighbor_tour(self.road))
        served = sorted(c for t in sol.trips for c in t.customers)
        self.assertEqual(served, list(range(1, 251)))


class TestOsmData(unittest.TestCase):
    """Our own OSM road data (data/osm/, built by main.py --mode road-data)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.osm = load_problem(ROOT / "data" / "osm")
        cls.ga = load_problem(ROOT / "data" / "road_ga")
        cls.base = load_problem(ROOT / "data" / "base")
        cls.meta = json.loads((ROOT / "data" / "osm" / "road_meta.json").read_text(encoding="utf-8"))
        cls.ga_meta = json.loads((ROOT / "data" / "road_ga" / "road_meta.json").read_text(encoding="utf-8"))

    def test_loaded_as_road(self) -> None:
        self.assertTrue(self.osm.road_mode)
        self.assertFalse(self.osm.symmetric)           # one-way streets
        self.assertEqual(self.osm.distance.shape, (251, 251))
        self.assertEqual(self.osm.duration.shape, (251, 251))

    def test_same_customers_and_orders_as_base(self) -> None:
        for name in ("weight", "volume", "loading", "unloading", "lat", "lon"):
            np.testing.assert_array_equal(getattr(self.osm, name), getattr(self.base, name))

    def test_same_road_vertices_as_ga(self) -> None:
        self.assertEqual(self.meta["osm_node_ids"], self.ga_meta["osm_node_ids"])

    def test_matrices_agree_with_ga(self) -> None:
        off = ~np.eye(251, dtype=bool)
        for ours, theirs in ((self.osm.distance, self.ga.distance),
                             (self.osm.duration, self.ga.duration)):
            rel = np.abs(ours - theirs)[off] / np.maximum(theirs[off], 0.05)
            self.assertLess(rel.mean(), 0.001)          # mean difference < 0.1 %
            self.assertGreater(np.mean(rel < 1e-3), 0.95)   # >95 % of pairs within 0.1 %
            self.assertAlmostEqual(ours.sum() / theirs.sum(), 1.0, places=3)

    def test_every_customer_feasible(self) -> None:
        from experiments.make_osm_dataset import feasibility_problems
        self.assertEqual(feasibility_problems(ROOT / "data" / "osm"), [])

    def test_meta_supports_street_geometry(self) -> None:
        self.assertIn("own OSM pool", self.meta["source"])
        self.assertEqual(len(self.meta["osm_node_ids"]), 251)
        self.assertEqual(len(self.meta["snapped_coordinates_lonlat"]), 251)
        self.assertIn("query", self.meta["osm_extract_source"])
        self.assertLessEqual(self.meta["max_snap_distance_m"], 150.0)

    def test_decoder_on_osm_roads(self) -> None:
        sol = decode_tour(self.osm, nearest_neighbor_tour(self.osm))
        served = sorted(c for t in sol.trips for c in t.customers)
        self.assertEqual(served, list(range(1, 251)))
        for t in sol.trips:
            ref = segment_stats(self.osm, t.customers)
            self.assertAlmostEqual(t.metrics.distance_km, ref.distance_km, places=6)
            self.assertAlmostEqual(t.metrics.driving_min,
                                   leg_sum(self.osm.duration, np.asarray([0, *t.customers, 0])),
                                   places=6)

    def test_osm_instances_are_slices(self) -> None:
        import csv
        folders = sorted((ROOT / "instances_osm" / "road_tuning").glob("road_tune_*"))
        if not folders:
            self.skipTest("instances_osm not present")
        for d in folders:
            inst = load_problem(d)
            with open(d / "locations.csv", newline="", encoding="utf-8") as f:
                idx = np.asarray([int(r["source_location_id"]) for r in csv.DictReader(f)])
            np.testing.assert_allclose(inst.distance, self.osm.distance[np.ix_(idx, idx)])
            np.testing.assert_allclose(inst.duration, self.osm.duration[np.ix_(idx, idx)])


class TestCrossCheckWithGA(unittest.TestCase):
    """Our evaluator must give the GA validator's cost for the same road plan."""

    def test_reference_plan_cost(self) -> None:
        data = load_problem(ROOT / "data" / "road_ga")
        ref = json.loads((HERE / "data" / "ga_road_reference.json").read_text())
        base = 0.0
        day: dict[int, float] = {}
        for t in ref["transports"]:
            m = trip_metrics(data, t["vehicle"], t["customers"])
            self.assertAlmostEqual(m.distance_km, t["distance_km"], places=3)
            base += m.base_cost_eur
            day[t["vehicle"]] = day.get(t["vehicle"], 0.0) + m.total_min
        overtime = sum(day_metrics(data, k, T).extra_cost for k, T in day.items())
        self.assertAlmostEqual(base + overtime, ref["ga_cost"]["total_eur"], places=2)


if __name__ == "__main__":
    unittest.main()


class TestRoadGeometry(unittest.TestCase):
    """Drawing routes along a (tiny, synthetic) OSM graph."""

    def setUp(self) -> None:
        from src.road_geometry import graph_from_osm, paths_from
        # 3 x 3 street grid, ~110 m blocks; the middle row is one-way eastbound.
        nodes = [{"type": "node", "id": 10 * r + c, "lon": 19.0 + 0.0015 * c, "lat": 47.5 + 0.001 * r}
                 for r in range(3) for c in range(3)]
        ways = []
        for r in range(3):
            tags = {"highway": "residential"}
            if r == 1:
                tags["oneway"] = "yes"
            ways.append({"type": "way", "id": 100 + r, "nodes": [10 * r, 10 * r + 1, 10 * r + 2], "tags": tags})
        for c in range(3):
            ways.append({"type": "way", "id": 200 + c, "nodes": [c, 10 + c, 20 + c],
                         "tags": {"highway": "tertiary"}})
        self.osm = {"elements": nodes + ways}
        self.loc_nodes = [0, 12, 20, 2]           # depot + 3 customers
        _, graph = graph_from_osm(self.osm)
        n = len(self.loc_nodes)
        self.km = [[0.0] * n for _ in range(n)]
        self.mins = [[0.0] * n for _ in range(n)]
        for i, a in enumerate(self.loc_nodes):
            res = paths_from(graph, a, set(self.loc_nodes))
            for j, b in enumerate(self.loc_nodes):
                self.km[i][j] = res[b][1] / 1000
                self.mins[i][j] = res[b][2] / 60
        coord = {e["id"]: [e["lon"], e["lat"]] for e in nodes}
        self.coords = [coord[v] for v in self.loc_nodes]

    def test_one_way_is_directed(self) -> None:
        # 10 -> 12 is direct on the one-way street; 12 -> 10 needs a detour.
        from src.road_geometry import graph_from_osm, paths_from
        _, g = graph_from_osm(self.osm)
        fwd = paths_from(g, 10, {12})[12][1]
        back = paths_from(g, 12, {10})[10][1]
        self.assertGreater(back, fwd + 1)

    def test_legs_match_matrix_exactly(self) -> None:
        from src.road_geometry import trip_geometry
        meta = {"osm_node_ids": self.loc_nodes}
        geo = trip_geometry([{"vehicle_id": 1, "customers": [1, 2, 3]}], self.osm, meta,
                            self.coords, self.km, self.mins)
        self.assertEqual(geo["summary"]["legs"], 4)
        self.assertEqual(geo["summary"]["exact"], 4)
        first = geo["trips"][0]["legs"][0]
        self.assertGreaterEqual(len(first["points"]), 3)   # follows the streets

    def test_mismatch_is_reported(self) -> None:
        from src.road_geometry import trip_geometry
        km = [row[:] for row in self.km]
        km[0][1] *= 1.5
        geo = trip_geometry([{"vehicle_id": 1, "customers": [1, 2, 3]}], self.osm,
                            {"osm_node_ids": self.loc_nodes}, self.coords, km, self.mins)
        self.assertEqual(geo["summary"]["differs"], 1)


class TestRoadInstances(unittest.TestCase):
    def test_sub_matrix_and_meta(self) -> None:
        import csv
        import tempfile
        from experiments.make_road_instances import build_instance
        full = load_problem(ROOT / "data" / "road_ga")
        full_meta = json.loads((ROOT / "data" / "road_ga" / "road_meta.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "x"
            build_instance("x", 50, 5001, dest)
            inst = load_problem(dest)
            with open(dest / "locations.csv", newline="", encoding="utf-8") as f:
                src = [int(r["source_location_id"]) for r in csv.DictReader(f)]
            meta = json.loads((dest / "road_meta.json").read_text())
        self.assertEqual(inst.n_customers, 50)
        self.assertTrue(inst.road_mode)
        idx = np.asarray(src)
        np.testing.assert_allclose(inst.distance, full.distance[np.ix_(idx, idx)])
        np.testing.assert_allclose(inst.duration, full.duration[np.ix_(idx, idx)])
        np.testing.assert_array_equal(inst.lat, full.lat[idx])
        self.assertEqual(meta["osm_node_ids"], [full_meta["osm_node_ids"][i] for i in src])


class TestRoadPool(unittest.TestCase):
    """Own pipeline: snap the pool to the drivable core and route every pair."""

    def test_pool_matrix_and_filter(self) -> None:
        import tempfile

        from experiments.build_road_pool import build_pool, drivable_core
        from src.road_geometry import graph_from_osm, paths_from

        # 3 x 3 grid (middle row one-way east) + a separate 2-node road island.
        nodes = [{"type": "node", "id": 10 * r + c, "lon": 19.0 + 0.0015 * c, "lat": 47.5 + 0.001 * r}
                 for r in range(3) for c in range(3)]
        nodes += [{"type": "node", "id": 900, "lon": 19.0300, "lat": 47.5},
                  {"type": "node", "id": 901, "lon": 19.0310, "lat": 47.5}]
        ways = []
        for r in range(3):
            tags = {"highway": "residential", **({"oneway": "yes"} if r == 1 else {})}
            ways.append({"type": "way", "id": 100 + r, "nodes": [10 * r, 10 * r + 1, 10 * r + 2], "tags": tags})
        for c in range(3):
            ways.append({"type": "way", "id": 200 + c, "nodes": [c, 10 + c, 20 + c],
                         "tags": {"highway": "residential"}})
        ways.append({"type": "way", "id": 300, "nodes": [900, 901], "tags": {"highway": "residential"}})
        osm = {"elements": nodes + ways}
        coord = {e["id"]: [e["lon"], e["lat"]] for e in nodes}

        # depot at node 0, customers at nodes 12 and 10, one on the island.
        points = [coord[0], coord[12], coord[10], [19.0305, 47.5]]
        _, graph = graph_from_osm(osm)
        self.assertNotIn(900, drivable_core(graph, 0))
        with tempfile.TemporaryDirectory() as tmp:
            res = build_pool(osm, points, Path(tmp), workers=1, max_snap_m=150)

        self.assertEqual(list(res["vertex"][:3]), [0, 12, 10])
        self.assertTrue(np.isfinite(res["km"]).all())
        for i, a in enumerate([0, 12, 10]):
            found = paths_from(graph, a, {0, 12, 10})
            for j, b in enumerate([0, 12, 10]):
                self.assertAlmostEqual(res["km"][i, j], found[b][1] / 1000)
                self.assertAlmostEqual(res["min"][i, j], found[b][2] / 60)
        # middle row is one-way east: 10 -> 12 is 2 blocks, 12 -> 10 must detour
        self.assertGreater(res["min"][1, 2], res["min"][2, 1] * 1.5)
        # The island point snaps to the core, but ~2 km away -> not usable.
        self.assertEqual(list(res["usable"]), [True, True, True, False])
        self.assertIn("snapped", res["reasons"][3])
