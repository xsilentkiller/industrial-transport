"""Furniture delivery optimiser -- the one entry point.

Results go to ONE folder, output/, which always holds the LATEST run (any
dataset; best_solution.json records which dataset it was):

    best_solution.json     the plan and its costs
    route_geometry.json    street geometry (road datasets, needs cache/budapest_osm.json)
    report/                report.html, route_map.html (street / satellite), charts, CSVs

    python main.py                                   # decoder checks + PSO + street geometry + report on data/osm/
    python main.py --mode pso                        # PSO + street geometry + report (data/osm)
    python main.py --mode report
    python main.py --dataset data/base                 # the straight-line dataset instead
    python main.py --mode road-data                  # OSM extract -> routed pool -> data/osm/
    python main.py --mode tests
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unittest
from pathlib import Path

import numpy as np

from src.data_loader import ProblemData, load_problem
from src.decoder import DecoderConfig, check_solution, decode_tour
from src.pso import MemeticPSO, PSOConfig, nearest_neighbor_tour
from src.delivery_sim import SimConfig, build_simulation
from src.report import generate_report, load_payload
from src.road_geometry import load_geometry


def _permutation_ok(customers: list[int], n: int) -> bool:
    return len(customers) == n and sorted(customers) == list(range(1, n + 1))


def print_solution_summary(data: ProblemData, sol) -> None:
    print("  REAL COST (company money, for analysis)")
    for name, value in sol.cost_breakdown.items():
        print(f"    {name:20s}: {value:10.2f} EUR")
    print(f"    {'= real cost':20s}: {sol.true_cost_eur:10.2f} EUR")
    print("  PENALTY (search only, not money)")
    for name, value in sol.penalty_breakdown.items():
        print(f"    {name:20s}: {value:10.2f}")
    print(
        f"    {'= penalty':20s}: {sol.penalty:10.2f}  "
        f"(soft {sol.soft_penalty:.2f}, hard {sol.penalty - sol.soft_penalty:.2f})\n"
        f"  objective     : {sol.objective:10.2f}  (real cost + penalty, what the search minimises)\n"
        f"  feasible      : {sol.feasible}  "
        f"({'all hard rules hold' if sol.feasible else 'a hard rule is broken'})\n"
        f"  trips         : {sol.n_trips}\n"
        f"  distance      : {sol.total_distance_km:10.1f} km\n"
        f"  working time  : {sol.total_time_min / 60:10.2f} h\n"
        f"  overtime      : {sol.shift_over_min / 60:10.2f} h\n"
        f"  day OT cost   : {sol.day_overtime_cost:10.2f} EUR\n"
        f"  load          : {sol.total_weight_kg:10.1f} kg / "
        f"{sol.total_volume_m3:8.2f} m3\n"
        f"  method        : {sol.method}"
    )
    for k, v in enumerate(data.vehicles):
        trips = [t for t in sol.trips if t.vehicle_index == k]
        if not trips:
            continue
        base = sum(t.metrics.base_cost_eur for t in trips)
        ot = sol.vehicle_overtime_cost[k]
        print(
            f"    vehicle {v.vehicle_id} ({v.type_id:8s}): "
            f"{len(trips):2d} trips, "
            f"{sol.vehicle_total_min[k] / 60:6.2f} h, "
            f"OT {sol.vehicle_overtime_min[k] / 60:5.2f} h, "
            f"EUR {base + ot:8.2f}"
            + _day_window_text(data, sol, k)
        )
    ots = sol.overtime_summary(data)
    print(
        f"  longest day   : {ots['longest_day_h']:10.2f} h\n"
        f"  overtime used : {ots['overtime_h']:10.2f} h "
        f"(unavoidable {ots['unavoidable_overtime_h']:.2f} h, "
        f"avoidable {ots['avoidable_overtime_h']:.2f} h)"
    )


def _day_window_text(data: ProblemData, sol, k: int) -> str:
    if sol.vehicle_last_delivery_min is None:
        return ""
    p = data.params
    return (
        f", {p.day_start}-{p.clock(sol.vehicle_total_min[k])}"
        f" (last delivery {p.clock(sol.vehicle_last_delivery_min[k])})"
    )


def run_decoder_checks(data: ProblemData) -> None:
    print("=" * 78)
    print("1) First 20 customers -- decoder sanity check")
    print("=" * 78)
    tour20 = list(range(1, 21))
    cfg20 = DecoderConfig(enforce_min_customers=True, use_greedy_baseline=False)
    sol20 = decode_tour(data, tour20, cfg20)
    seen20 = [c for t in sol20.trips for c in t.customers]
    print(f"  every customer exactly once: {_permutation_ok(seen20, 20)}")
    print(f"  trips: {sol20.n_trips}")
    for t in sol20.trips:
        print(
            f"    vehicle {t.vehicle_id}: {len(t.customers):2d} customers, "
            f"{t.time_min:7.2f} min, EUR {t.cost_eur:8.2f}, "
            f"{t.metrics.weight_kg:7.1f} kg, {t.metrics.volume_m3:6.3f} m3, "
            f"capacity_ok={t.metrics.capacity_ok}, "
            f"min_ok={t.metrics.min_customers_ok}"
        )
    print(f"  all trips 10-50 customers: {all(10 <= len(t.customers) <= 50 for t in sol20.trips)}")
    print(f"  all capacity checks pass : {all(t.metrics.capacity_ok for t in sol20.trips)}")
    print(f"  solution penalty         : {sol20.penalty:.2f}")
    print()

    print("=" * 78)
    print("2) Full 250 customers -- coverage, limits, and timing")
    print("=" * 78)
    tour = nearest_neighbor_tour(data)
    cfg = DecoderConfig()
    sol = decode_tour(data, tour, cfg)
    errors = check_solution(data, sol)
    print(f"  every customer exactly once: {_permutation_ok([c for t in sol.trips for c in t.customers], data.n_customers)}")
    print(f"  no trip over 50 customers  : {all(len(t.customers) <= 50 for t in sol.trips)}")
    print(f"  all capacity checks pass   : {all(t.metrics.capacity_ok for t in sol.trips)}")
    print(f"  checker errors             : {errors if errors else 'none'}")
    print_solution_summary(data, sol)

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        decode_tour(data, tour, cfg)
        times.append(time.perf_counter() - t0)
    print(
        f"  decoder time (20 runs)     : "
        f"mean {np.mean(times) * 1000:.1f} ms, "
        f"median {np.median(times) * 1000:.1f} ms, "
        f"min {np.min(times) * 1000:.1f} ms, "
        f"max {np.max(times) * 1000:.1f} ms"
    )
    print()


def run_pso(data: ProblemData, args: argparse.Namespace) -> None:
    print("=" * 78)
    print("3) Split-based memetic random-key PSO")
    print("=" * 78)
    cfg = PSOConfig(
        swarm_size=args.swarm,
        iterations=args.iterations,
        seed=args.seed,
        local_search_interval=args.ls_interval,
        local_search_budget=args.ls_budget,
        final_local_search_budget=args.final_ls_budget,
        max_seconds=args.time_limit,
        patience=args.patience,
        verbose=True,
        log_interval=max(1, args.log_interval),
        decoder=DecoderConfig(),
    )
    result = MemeticPSO(data, cfg).run()

    print()
    print(
        f"PSO finished in {result.runtime_sec:.1f} s, {result.evaluations} decoder calls, "
        f"{result.iterations_run} iterations"
        + (" (early stop)" if result.stopped_early else "")
    )
    if result.seed_solution is not None:
        print(
            f"Deterministic feasible seed: EUR {result.seed_solution.true_cost_eur:.2f}, "
            f"penalty {result.seed_solution.penalty:.1f}"
        )
    if result.best_feasible_solution is not None:
        print("\nBest FEASIBLE solution (benchmark number):")
        print_solution_summary(data, result.best_feasible_solution)
    else:
        print("\nNo feasible solution found (a hard rule is broken in every candidate).")

    print("\nBest overall (may include penalty):")
    print_solution_summary(data, result.best_solution)

    out_dir = run_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The folder holds only the latest run: drop the previous run's geometry.
    (out_dir / "route_geometry.json").unlink(missing_ok=True)
    chosen = (
        result.best_feasible_solution
        if result.best_feasible_solution is not None
        else result.best_solution
    )
    check_errors = check_solution(data, chosen)
    print(
        "\nFinal check_solution: "
        + ("no errors" if not check_errors else "; ".join(check_errors))
    )
    payload = {
        "dataset": str(args.dataset),
        "config": {
            "swarm_size": cfg.swarm_size,
            "iterations": cfg.iterations,
            "seed": cfg.seed,
            "local_search_interval": cfg.local_search_interval,
            "local_search_budget": cfg.local_search_budget,
            "final_local_search_budget": cfg.final_local_search_budget,
            "patience": cfg.patience,
            "iterations_run": result.iterations_run,
            "stopped_early": result.stopped_early,
        },
        "runtime_sec": result.runtime_sec,
        "evaluations": result.evaluations,
        "best_feasible_cost": result.best_feasible_score
        if result.best_feasible_solution is not None
        else None,
        "check_errors": check_errors,
        "best_solution": chosen.as_dict(),
        "day_window": {
            "day_start": data.params.day_start,
            "overtime_multiplier": data.params.overtime_multiplier,
            "last_delivery_by": data.params.last_delivery_by,
            "hard_day_end": data.params.hard_day_end,
        },
        "day_schedule": chosen.day_schedule(data),
        "overtime_summary": chosen.overtime_summary(data),
        "history": result.history,
    }
    if args.sim_config is not None:
        sim = build_simulation(data, payload["best_solution"], args.dataset, args.sim_config)
        payload["delivery_simulation"] = sim
        c, mc = sim["counts"], sim["monte_carlo"]
        print(
            f"\nDelivery simulation (seed {args.sim_config.seed}): delivered {c['delivered']}, "
            f"attempted {c['attempted']}, refused {c['refused']}; "
            f"Monte Carlo {mc['days']} days: failed mean {mc['failed']['mean']:.1f} "
            f"(5-95 %: {mc['failed']['p5']:.0f}-{mc['failed']['p95']:.0f})"
        )
    path = out_dir / "best_solution.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {path}")

    report_dir = Path(args.report_dir) if args.report_dir else out_dir / "report"
    report_path = generate_report(
        data, payload, report_dir, _geometry_for(data, args, out_dir, payload)
    )
    print(f"Saved report {report_path}")


def run_dir(args: argparse.Namespace) -> Path:
    """The single result folder: it always holds the latest run (any dataset)."""
    return Path(args.output_dir)


def run_report(data: ProblemData, args: argparse.Namespace) -> int:
    input_path = (
        Path(args.input)
        if args.input
        else run_dir(args) / "best_solution.json"
    )
    if not input_path.exists():
        print(
            f"Input not found: {input_path}. "
            f"Run 'python main.py --mode pso --dataset {args.dataset}' first."
        )
        return 1
    payload = load_payload(input_path)
    report_dir = (
        Path(args.report_dir)
        if args.report_dir
        else input_path.parent / "report"
    )
    report_path = generate_report(
        data, payload, report_dir, _geometry_for(data, args, input_path.parent, payload)
    )
    print(f"Saved report {report_path}")
    return 0


def _same_trips(geometry: dict, payload: dict) -> bool:
    ours = sorted(tuple(t["customers"]) for t in payload["best_solution"]["trips"])
    theirs = sorted(tuple(t["customers"]) for t in geometry.get("trips", []))
    return ours == theirs


def _geometry_for(data: ProblemData, args: argparse.Namespace,
                  folder: Path, payload: dict) -> dict | None:
    """Street geometry of this solution: reuse route_geometry.json if it fits,
    else compute it (road datasets with the OSM extract), else straight lines."""
    path = folder / "route_geometry.json"
    geometry = load_geometry(path)
    if geometry is not None and _same_trips(geometry, payload):
        return geometry
    if not data.road_mode or args.no_geometry:
        return None
    osm = Path(args.osm)
    if not osm.exists():
        print(f"No {osm}: the map draws straight lines "
              "(python main.py --mode road-data builds the extract).")
        return None
    from experiments.road_geometry import save_geometry, solution_geometry
    print("\nComputing street geometry for the route map...")
    geometry = solution_geometry(data, Path(args.dataset), payload, osm)
    save_geometry(geometry, path)
    return geometry


def run_road_data(args: argparse.Namespace) -> int:
    """OSM extract -> routed address pool -> road dataset (each step cached)."""
    from experiments import build_road_pool, make_osm_dataset, osm_from_geofabrik
    osm = Path(args.osm)
    pbf = osm_from_geofabrik.PBF
    if not osm.exists():
        if not pbf.exists():
            print(f"Missing {pbf}. Download it once with:\n  curl.exe -L -o {pbf} "
                  "https://download.geofabrik.de/europe/hungary-latest.osm.pbf\n"
                  "(or run python -m experiments.build_road_pool --download for Overpass)")
            return 1
        osm_from_geofabrik.main(["--pbf", str(pbf), "--out", str(osm)])
    pool = build_road_pool.POOL_DIR / "road_pool.npz"
    if pool.exists() and not args.rebuild_pool:
        print(f"using routed pool {pool} (--rebuild-pool to route again)")
    else:
        build_road_pool.main(["--osm", str(osm)])
    out = args.dataset
    make_osm_dataset.main(["--out", out])
    print(f"\nNext: python main.py --mode pso --dataset {out}")
    return 0


def run_tests() -> int:
    """All tests once, with a per-file summary (same as python run_tests.py)."""
    from run_tests import run_all
    return run_all(verbose=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=["all", "decoder", "pso", "report", "road-data", "tests"],
        default="all",
    )
    parser.add_argument("--dataset", default="data/osm",
                        help="dataset folder (default data/osm: own OSM road data; "
                             "road-data: the folder to create)")
    parser.add_argument("--output-dir", default="output",
                        help="result folder; holds the latest run and is overwritten by the next")
    parser.add_argument("--osm", default="cache/budapest_osm.json",
                        help="OSM road extract for street geometry and road-data")
    parser.add_argument("--no-geometry", action="store_true",
                        help="do not compute street geometry (map draws straight lines)")
    parser.add_argument("--rebuild-pool", action="store_true",
                        help="road-data: route the address pool again")
    parser.add_argument(
        "--input",
        default=None,
        help="Solution JSON for --mode report (default: <output-dir>/best_solution.json)",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Report directory (default: <output-dir>/report or <input>/report)",
    )
    # PSO defaults come from PSOConfig in src/pso.py: change them there only.
    pso = PSOConfig()
    parser.add_argument("--swarm", type=int, default=pso.swarm_size)
    parser.add_argument("--iterations", type=int, default=pso.iterations)
    parser.add_argument("--seed", type=int, default=pso.seed)
    parser.add_argument("--ls-interval", type=int, default=pso.local_search_interval)
    parser.add_argument("--ls-budget", type=int, default=pso.local_search_budget)
    parser.add_argument("--final-ls-budget", type=int, default=pso.final_local_search_budget)
    parser.add_argument("--time-limit", type=float, default=pso.max_seconds)
    parser.add_argument("--patience", type=int, default=pso.patience,
                        help="stop after this many iterations without improvement "
                             "(0 = off; --iterations is then an upper limit)")
    parser.add_argument("--log-interval", type=int, default=pso.log_interval)
    # Delivery outcomes after the PSO (src/delivery_sim.py); never changes the plan.
    sim = SimConfig()
    parser.add_argument("--not-home", type=float, default=sim.p_not_home,
                        help="chance a customer is not home (attempted, re-deliver tomorrow)")
    parser.add_argument("--refused", type=float, default=sim.p_refused,
                        help="chance a customer refuses the order (returned to stock)")
    parser.add_argument("--sim-seed", type=int, default=sim.seed)
    parser.add_argument("--sim-days", type=int, default=sim.monte_carlo_days,
                        help="simulated days for the Monte Carlo risk estimate")
    parser.add_argument("--no-sim", action="store_true",
                        help="skip the delivery-outcome simulation")
    args = parser.parse_args(argv)
    args.sim_config = None
    if not args.no_sim:
        try:
            args.sim_config = SimConfig(args.not_home, args.refused, args.sim_seed, args.sim_days)
        except ValueError as e:
            parser.error(str(e))

    if args.mode == "tests":
        return run_tests()
    if args.mode == "road-data":
        return run_road_data(args)

    data = load_problem(args.dataset)
    print(f"Loaded: {data.summary()}")
    print()

    if args.mode in ("decoder", "all"):
        run_decoder_checks(data)
    if args.mode in ("pso", "all"):
        run_pso(data, args)
    if args.mode == "report":
        return run_report(data, args)
    return 0

if __name__ == "__main__":
    sys.exit(main())
