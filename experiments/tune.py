"""Tune the PSO: swarm size, patience (early stopping) and local-search interval.

    python -m experiments.tune                         # full tuning on the 4 tuning instances
    python -m experiments.tune --quick                 # round 1 only, 1 seed (first look)
    python -m experiments.tune --instances instances_osm/road_test --name test100

Every combination of the grid (default: swarm 10 20 30 40 x patience 10 20 40
x local-search interval 1 5, iterations capped at 300) is run on every
instance.  Weak combinations are dropped in rounds (successive halving):

    round 1: all combinations, 1 seed per instance    -> keep the best half
    round 2: survivors, 2 more seeds                  -> keep the best 4
    round 3: finalists, 2 more seeds (5 in total)     -> winner

The baseline (today's default: swarm 20, 60 iterations, LS interval 5, no
patience) runs in every round and is never dropped, so the winner can be
compared with it.

Score of a run = gap to the best real cost found on that instance by any run
of this grid, in percent, so every instance counts equally whatever its cost
level.  A combination's score is its median gap; combinations within 0.5
percentage points of the leader count as tied and the faster one wins.
Runs without a feasible plan (or that crashed) get an infinite gap.

The suitable number of iterations is measured, not tuned: the median of the
iterations the winner actually used before early stopping, rounded up to a
multiple of 5.

Outputs in ``<out>/<name>/`` (default ``tuning/run/``):
    runs.csv          one row per run, appended as soon as it finishes
    summary.csv       one row per combination
    tuning.png        median gap per combination + iterations used by finalists
    best_config.json  the winner, ready to become the PSOConfig default

Interrupt at any time: rerunning the same command skips finished runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIELDS = [
    "round", "combo", "swarm", "patience", "ls_interval", "max_iterations",
    "instance", "seed", "real_cost", "penalty", "feasible", "iterations_run",
    "stopped_early", "runtime_s", "evaluations", "error",
]
INT_FIELDS = {"round", "swarm", "patience", "ls_interval", "max_iterations",
              "seed", "iterations_run", "evaluations"}
FLOAT_FIELDS = {"real_cost", "penalty", "runtime_s"}
BOOL_FIELDS = {"feasible", "stopped_early"}
BASELINE_KEY_PREFIX = "baseline_s20_p0_ls5_i"


# ---------------------------------------------------------------------------
# Combinations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Combo:
    swarm: int
    patience: int
    ls_interval: int
    iterations: int
    baseline: bool = False

    @property
    def key(self) -> str:
        if self.baseline:
            return f"{BASELINE_KEY_PREFIX}{self.iterations}"
        return f"s{self.swarm}_p{self.patience}_ls{self.ls_interval}_i{self.iterations}"


def make_grid(swarms, patiences, ls_intervals, max_iterations) -> list[Combo]:
    return [
        Combo(s, p, ls, max_iterations)
        for s in swarms for p in patiences for ls in ls_intervals
    ]


def baseline_combo(iterations: int = 60) -> Combo:
    return Combo(20, 0, 5, iterations, baseline=True)


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------
def find_instances(paths: list[str]) -> list[Path]:
    """Instance folders (with params.json): given directly or as a parent."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if (p / "params.json").exists():
            found.append(p)
            continue
        if not p.is_dir():
            print(f"skipped {p}: not a folder")
            continue
        for child in sorted(x for x in p.iterdir() if x.is_dir()):
            if (child / "params.json").exists():
                found.append(child)
            else:
                print(f"skipped {child}: no params.json")
    names = [p.name for p in found]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"instance folder names must be unique, duplicated: {dup}")
    return sorted(found, key=lambda x: x.name)


# ---------------------------------------------------------------------------
# One run (executed in a worker process)
# ---------------------------------------------------------------------------
_DATA_CACHE: dict[str, object] = {}


def run_job(job: dict) -> dict:
    """Run one PSO and return its runs.csv row (errors are recorded, not raised)."""
    from src.data_loader import load_problem
    from src.pso import MemeticPSO, PSOConfig

    row = {k: job.get(k) for k in FIELDS}
    row.update(real_cost=None, penalty=None, feasible=False, iterations_run=0,
               stopped_early=False, runtime_s=0.0, evaluations=0, error="")
    start = perf_counter()
    try:
        path = job["instance_path"]
        if path not in _DATA_CACHE:
            _DATA_CACHE[path] = load_problem(path)
        data = _DATA_CACHE[path]
        cfg = PSOConfig(
            swarm_size=job["swarm"],
            iterations=job["max_iterations"],
            seed=job["seed"],
            local_search_interval=job["ls_interval"],
            patience=job["patience"],
            max_seconds=job["time_limit"],
            verbose=False,
        )
        result = MemeticPSO(data, cfg).run()
        sol = result.best_feasible_solution
        row.update(
            iterations_run=result.iterations_run,
            stopped_early=result.stopped_early,
            runtime_s=result.runtime_sec,
            evaluations=result.evaluations,
        )
        if sol is not None:
            row.update(real_cost=sol.true_cost_eur, penalty=sol.penalty, feasible=True)
    except Exception as exc:  # recorded so the other runs continue
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["runtime_s"] = perf_counter() - start
    return row


# ---------------------------------------------------------------------------
# runs.csv
# ---------------------------------------------------------------------------
def _parse(name: str, value: str):
    if value in ("", None):
        return None if name not in BOOL_FIELDS else False
    if name in INT_FIELDS:
        return int(float(value))
    if name in FLOAT_FIELDS:
        return float(value)
    if name in BOOL_FIELDS:
        return value.strip().lower() in ("true", "1", "yes")
    return value


def load_runs(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [{k: _parse(k, v) for k, v in r.items() if k in FIELDS}
                for r in csv.DictReader(f)]


def append_run(path: Path, row: dict) -> None:
    path = Path(path)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in FIELDS})


def done_keys(rows: list[dict]) -> set[tuple[str, str, int]]:
    return {(r["combo"], r["instance"], int(r["seed"])) for r in rows}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _ok(r: dict) -> bool:
    return bool(r.get("feasible")) and r.get("real_cost") is not None and not r.get("error")


def add_gaps(rows: list[dict], combos: list[str] | None = None) -> list[dict]:
    """Gap (%) of every run to the best real cost on its instance.

    Only rows of ``combos`` (if given) are used, so rows of an older grid in
    the same folder neither set the best known cost nor appear in the result.
    """
    if combos is not None:
        keep = set(combos)
        rows = [r for r in rows if r["combo"] in keep]
    best: dict[str, float] = {}
    for r in rows:
        if _ok(r):
            best[r["instance"]] = min(best.get(r["instance"], math.inf), r["real_cost"])
    out = []
    for r in rows:
        r = dict(r)
        if _ok(r) and r["instance"] in best:
            r["gap_pct"] = (r["real_cost"] / best[r["instance"]] - 1.0) * 100.0
        else:
            r["gap_pct"] = math.inf
        out.append(r)
    return out


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else math.inf


def summarise(rows: list[dict], combos: list[str]) -> list[dict]:
    """One row per combination (rows must already carry gap_pct)."""
    out = []
    for key in combos:
        rs = [r for r in rows if r["combo"] == key]
        gaps = [r["gap_pct"] for r in rs]
        finite = [g for g in gaps if math.isfinite(g)]
        out.append({
            "combo": key,
            "runs": len(rs),
            "infeasible_runs": sum(1 for g in gaps if not math.isfinite(g)),
            "median_gap": _median(gaps),
            "mean_gap": float(np.mean(finite)) if finite and len(finite) == len(gaps) else math.inf,
            "min_gap": min(gaps) if gaps else math.inf,
            "max_gap": max(gaps) if gaps else math.inf,
            "median_runtime": _median([r["runtime_s"] or 0.0 for r in rs]),
            "median_iterations": _median([r["iterations_run"] or 0 for r in rs]),
            "early_stops": sum(1 for r in rs if r.get("stopped_early")),
        })
    return out


def rank(summary: list[dict], tie_pp: float = 0.5) -> list[dict]:
    """Lowest median gap first; within ``tie_pp`` of the leader, faster first."""
    rest = sorted(summary, key=lambda s: (s["median_gap"], s["median_runtime"]))
    ranked: list[dict] = []
    while rest:
        lead = rest[0]["median_gap"]
        tied = [s for s in rest
                if s["median_gap"] - lead <= tie_pp + 1e-12
                or (math.isinf(lead) and math.isinf(s["median_gap"]))]
        tied.sort(key=lambda s: (s["median_runtime"], s["median_gap"]))
        ranked.extend(tied)
        ids = {id(s) for s in tied}
        rest = [s for s in rest if id(s) not in ids]
    return ranked


def keep_count(spec: float, n: int) -> int:
    """< 1: fraction (rounded up); >= 1: count; never more than n, at least 1."""
    k = math.ceil(spec * n) if spec < 1 else int(spec)
    return max(1, min(n, k))


def round_seeds(seeds_per_round: list[int], r: int) -> list[int]:
    start = 1 + sum(seeds_per_round[:r])
    return list(range(start, start + seeds_per_round[r]))


def suggested_iterations(iterations: list[int]) -> int:
    if not iterations:
        return 0
    return int(math.ceil(float(np.median(iterations)) / 5.0) * 5)


def wilcoxon_vs_baseline(rows: list[dict], winner: str, baseline: str) -> float | None:
    """Paired (instance, seed) Wilcoxon signed-rank p-value on real cost."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    a = {(r["instance"], r["seed"]): r["real_cost"] for r in rows
         if r["combo"] == winner and _ok(r)}
    b = {(r["instance"], r["seed"]): r["real_cost"] for r in rows
         if r["combo"] == baseline and _ok(r)}
    pairs = sorted(set(a) & set(b))
    if len(pairs) < 6 or winner == baseline:
        return None
    diffs = [a[k] - b[k] for k in pairs]
    if all(abs(d) < 1e-9 for d in diffs):
        return 1.0
    try:
        return float(wilcoxon([a[k] for k in pairs], [b[k] for k in pairs]).pvalue)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def _fmt_gap(g: float) -> str:
    return "infeasible" if not math.isfinite(g) else f"{g:.2f}"


def write_summary(path: Path, ranked: list[dict], status: dict[str, str]) -> None:
    fields = ["rank", "combo", "status", "runs", "infeasible_runs", "median_gap",
              "mean_gap", "min_gap", "max_gap", "median_runtime",
              "median_iterations", "early_stops"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, s in enumerate(ranked, 1):
            w.writerow({
                "rank": i, "status": status.get(s["combo"], ""),
                **{k: (round(v, 4) if isinstance(v, float) and math.isfinite(v)
                       else ("inf" if isinstance(v, float) else v))
                   for k, v in s.items()},
            })


def print_table(ranked: list[dict], status: dict[str, str]) -> None:
    print()
    print(f"{'#':>3}  {'combination':28s} {'status':14s} {'runs':>4}  "
          f"{'median gap %':>12}  {'max gap %':>10}  {'iters':>6}  {'time s':>7}")
    for i, s in enumerate(ranked, 1):
        print(f"{i:>3}  {s['combo']:28s} {status.get(s['combo'], ''):14s} {s['runs']:>4}  "
              f"{_fmt_gap(s['median_gap']):>12}  {_fmt_gap(s['max_gap']):>10}  "
              f"{s['median_iterations']:>6.0f}  {s['median_runtime']:>7.0f}")


def plot(path: Path, ranked: list[dict], rows: list[dict], status: dict[str, str],
         finalists: list[str], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    FINAL, PRUNED, BASE = "#2a78d6", "#b9b8b2", "#eb6834"

    finite = [s["max_gap"] for s in ranked if math.isfinite(s["max_gap"])]
    cap = (max(finite) if finite else 1.0) * 1.15 + 0.1
    n = len(ranked)
    fig = plt.figure(figsize=(14, max(4.5, 0.32 * n + 2.0)), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=8.5)

    ys = list(range(n))[::-1]
    for y, s in zip(ys, ranked):
        key = s["combo"]
        color = BASE if key.startswith(BASELINE_KEY_PREFIX) else (
            FINAL if key in finalists else PRUNED)
        med = min(s["median_gap"], cap)
        lo, hi = min(s["min_gap"], cap), min(s["max_gap"], cap)
        ax.plot([lo, hi], [y, y], color=color, lw=1.2, alpha=0.8)
        ax.plot([med], [y], "o", color=color, ms=7, mec=SURFACE, mew=1.5)
        label = "infeasible" if not math.isfinite(s["median_gap"]) else f"{s['median_gap']:.2f}"
        ax.annotate(label, (hi, y), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_yticks(ys, [f"{s['combo']}  ({status.get(s['combo'], '')})" for s in ranked],
                  fontsize=8.5, color=INK)
    ax.set_xlabel("Gap to best known cost per instance (%)  ·  dot = median, line = min-max",
                  color=MUTED, fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(left=min(0, ax.get_xlim()[0]))
    ax.set_title("Median gap per combination (lower is better)", color=INK,
                 fontsize=11, loc="left")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=FINAL, label="finalist"),
        Line2D([], [], marker="o", ls="", color=PRUNED, label="dropped"),
        Line2D([], [], marker="o", ls="", color=BASE, label="baseline (today's default)"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.07 - 2.0 / max(8, n * 3)),
        ncol=3, fontsize=8.5, frameon=False)
    ax.set_xlim(right=cap * 1.12)

    # Iterations actually used by the finalists (early stopping).
    show = [k for k in finalists if not k.startswith(BASELINE_KEY_PREFIX)]
    data = [[r["iterations_run"] for r in rows if r["combo"] == k] for k in show]
    if show and any(data):
        bp = ax2.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.55,
                         medianprops=dict(color=INK, lw=1.5))
        for box in bp["boxes"]:
            box.set_facecolor(FINAL)
            box.set_alpha(0.35)
            box.set_edgecolor(FINAL)
        ax2.set_yticks(range(1, len(show) + 1), show, fontsize=8.5, color=INK)
        for i, d in enumerate(data, 1):
            if d:
                ax2.annotate(f"median {np.median(d):.0f}", (max(d), i), xytext=(6, 0),
                             textcoords="offset points", va="center", fontsize=8, color=INK)
    ax2.set_xlabel("Iterations used before early stop", color=MUTED, fontsize=9)
    ax2.grid(axis="x", color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    ax2.set_title("Iterations used by the finalists", color=INK, fontsize=11, loc="left")

    fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK)
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tuning loop
# ---------------------------------------------------------------------------
def _run_jobs(jobs: list[dict], runs_path: Path, workers: int, done_before: int,
              total: int) -> None:
    def report(r: dict, i: int) -> None:
        cost = "error: " + r["error"] if r["error"] else (
            f"{r['real_cost']:,.2f} EUR" if r["feasible"] else "no feasible plan")
        early = " (early stop)" if r["stopped_early"] else ""
        print(f"[{done_before + i}/{total}] {r['combo']:28s} {r['instance']:16s} "
              f"seed {r['seed']:<3d} {cost}  {r['iterations_run']} it{early}  "
              f"{r['runtime_s']:.0f} s", flush=True)

    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            r = run_job(job)
            append_run(runs_path, r)
            report(r, i)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            append_run(runs_path, r)
            report(r, i)


def run_tuning(args: argparse.Namespace) -> Path:
    instances = find_instances(args.instances)
    if not instances:
        raise SystemExit("no instances found")
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.csv"

    grid = make_grid(args.swarm, args.patience, args.ls_interval, args.max_iterations)
    base = None if args.no_baseline else baseline_combo(args.baseline_iterations)
    combos = {c.key: c for c in grid}
    if base is not None:
        combos[base.key] = base
    seeds_per_round = list(args.seeds_per_round)
    keeps = list(args.keep)
    if len(keeps) < len(seeds_per_round) - 1:
        raise SystemExit("--keep needs one value per round except the last")

    rounds = len(seeds_per_round)
    alive = [c.key for c in grid]
    status = {k: "" for k in combos}
    print(f"Tuning on {len(instances)} instance(s): "
          f"{', '.join(p.name for p in instances[:6])}{' ...' if len(instances) > 6 else ''}")
    print(f"{len(grid)} combinations{' + baseline' if base else ''}, "
          f"{rounds} round(s), seeds per round {seeds_per_round}, workers {args.workers}")
    print(f"Results: {out}")

    ranked: list[dict] = []
    all_keys = list(combos)
    for r in range(rounds):
        seeds = round_seeds(seeds_per_round, r)
        running = alive + ([base.key] if base is not None else [])
        done = done_keys(load_runs(runs_path))
        jobs = []
        for key in running:
            c = combos[key]
            for inst in instances:
                for seed in seeds:
                    if (key, inst.name, seed) in done:
                        continue
                    jobs.append({
                        "round": r + 1, "combo": key, "swarm": c.swarm,
                        "patience": c.patience, "ls_interval": c.ls_interval,
                        "max_iterations": c.iterations, "instance": inst.name,
                        "instance_path": str(inst), "seed": seed,
                        "time_limit": args.time_limit,
                    })
        total = len(running) * len(instances) * len(seeds)
        print(f"\nRound {r + 1}/{rounds}: {len(running)} combination(s) x "
              f"{len(instances)} instance(s) x seeds {seeds} = {total} runs "
              f"({total - len(jobs)} already done)")
        _run_jobs(jobs, runs_path, args.workers, total - len(jobs), total)

        # Score with the rows of rounds 1..r+1 only: on a resumed run, rows of
        # later rounds must not change an earlier round's dropping decision.
        rows = add_gaps([x for x in load_runs(runs_path) if (x["round"] or 0) <= r + 1],
                        all_keys)
        ranked = rank(summarise(rows, all_keys), args.tie)
        if r < rounds - 1:
            candidates = [s["combo"] for s in ranked if s["combo"] in alive]
            k = keep_count(keeps[r], len(candidates))
            survivors, dropped = candidates[:k], candidates[k:]
            for key in dropped:
                status[key] = f"dropped r{r + 1}"
            alive = survivors
            print(f"Round {r + 1} done: keeping {len(survivors)}, dropping {len(dropped)}")
        for key in alive:
            status[key] = "finalist" if r == rounds - 1 else f"alive r{r + 1}"
        if base is not None:
            status[base.key] = "baseline"
        _write_outputs(out, rows, ranked, status, alive, base, instances,
                       seeds_per_round[: r + 1], args, final=(r == rounds - 1))
    return out


def _write_outputs(out: Path, rows, ranked, status, alive, base, instances,
                   seeds_per_round, args, final: bool) -> None:
    write_summary(out / "summary.csv", ranked, status)
    finalists = alive + ([base.key] if base is not None else [])
    plot(out / "tuning.png", ranked, rows, status, finalists,
         f"PSO tuning: {len(instances)} instance(s), "
         f"{sum(seeds_per_round)} seed(s) for the finalists"
         + ("" if final else " (in progress)"))

    winner = next((s for s in ranked if s["combo"] in alive), None)
    if winner is None:
        return
    key = winner["combo"]
    w_rows = [r for r in rows if r["combo"] == key]
    w_iters = [r["iterations_run"] for r in w_rows if r.get("iterations_run")]
    cap_hits = sum(1 for r in w_rows
                   if not r.get("stopped_early") and r["iterations_run"] >= r["max_iterations"])
    base_summary = next((s for s in ranked if base is not None and s["combo"] == base.key), None)
    sample = w_rows[0] if w_rows else {}
    base_rank = next((i for i, s in enumerate(ranked) if base is not None
                      and s["combo"] == base.key), None)
    win_rank = next(i for i, s in enumerate(ranked) if s["combo"] == key)
    baseline_better = base_rank is not None and base_rank < win_rank
    best = {
        "final": final,
        "winner": key,
        "baseline_ranks_higher": baseline_better,
        "swarm_size": sample.get("swarm"),
        "patience": sample.get("patience"),
        "local_search_interval": sample.get("ls_interval"),
        "max_iterations": sample.get("max_iterations"),
        "suggested_iterations": suggested_iterations(w_iters),
        "median_iterations_used": float(np.median(w_iters)) if w_iters else None,
        "runs_hitting_iteration_cap": cap_hits,
        "median_gap_pct": winner["median_gap"],
        "max_gap_pct": winner["max_gap"] if math.isfinite(winner["max_gap"]) else None,
        "median_runtime_s": winner["median_runtime"],
        "baseline": base.key if base is not None else None,
        "baseline_median_gap_pct": (base_summary["median_gap"] if base_summary
                                    and math.isfinite(base_summary["median_gap"]) else None),
        "baseline_median_runtime_s": base_summary["median_runtime"] if base_summary else None,
        "wilcoxon_p_vs_baseline": (wilcoxon_vs_baseline(rows, key, base.key)
                                   if base is not None else None),
        "instances": [p.name for p in instances],
        "seeds": list(range(1, sum(seeds_per_round) + 1)),
        "runs": len(rows),
        "time_limit_s": args.time_limit,
    }
    (out / "best_config.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    if final:
        print_table(ranked, status)
        print(f"\nBest tested combination: {key}  (median gap "
              f"{_fmt_gap(winner['median_gap'])} %, suggested iterations "
              f"{best['suggested_iterations']})")
        if baseline_better:
            print("WARNING: the baseline (today's default) ranks above every tested "
                  "combination - keep the current default, or widen the grid.")
        if best["baseline_median_gap_pct"] is not None:
            print(f"Baseline median gap: {best['baseline_median_gap_pct']:.2f} %"
                  + (f", Wilcoxon p = {best['wilcoxon_p_vs_baseline']:.4f}"
                     if best["wilcoxon_p_vs_baseline"] is not None else ""))
        if cap_hits:
            print(f"Note: {cap_hits} winner run(s) hit the {best['max_iterations']}-iteration "
                  "cap without stopping early; consider a higher --max-iterations.")
        print(f"Saved {out / 'runs.csv'}, {out / 'summary.csv'}, "
              f"{out / 'tuning.png'}, {out / 'best_config.json'}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Tune PSO swarm size, patience and local-search interval.")
    ap.add_argument("--instances", nargs="+", default=["instances_osm/road_tuning"],
                    help="instance folders, or parent folders of instance folders")
    ap.add_argument("--out", default="tuning", help="parent folder for results")
    ap.add_argument("--name", default="run", help="result subfolder name")
    ap.add_argument("--swarm", nargs="+", type=int, default=[10, 20, 30, 40])
    ap.add_argument("--patience", nargs="+", type=int, default=[10, 20, 40])
    ap.add_argument("--ls-interval", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--max-iterations", type=int, default=300,
                    help="safety cap; early stopping normally ends runs before it")
    ap.add_argument("--baseline-iterations", type=int, default=60)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--seeds-per-round", nargs="+", type=int, default=[1, 2, 2])
    ap.add_argument("--keep", nargs="+", type=float, default=[0.5, 4],
                    help="per round except the last: fraction (<1) or count (>=1)")
    ap.add_argument("--tie", type=float, default=0.5,
                    help="median gaps within this many percentage points count as tied")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--time-limit", type=float, default=600.0, help="seconds per run")
    ap.add_argument("--quick", action="store_true",
                    help="one round with one seed (first look, no dropping)")
    return ap


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.seeds_per_round = [1]
        args.keep = []
    return run_tuning(args)


if __name__ == "__main__":
    main()
