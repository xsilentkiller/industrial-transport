"""Convergence ablation: run PSO variants on several seeds and compare.

Usage:
    python -m experiments.convergence_ablation --variants v0 fix1 --seeds 1 2 3
    python -m experiments.convergence_ablation --variants all --seeds 1 2 3 --jobs 2

Each run records the best plan per iteration, the iteration of the last
improvement found by the swarm loop, the cost before and after the final
polish, and the runtime.  Results go to results/convergence/<variant>_s<seed>.json
and a summary table is printed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_problem  # noqa: E402
from src.pso import MemeticPSO, PSOConfig  # noqa: E402

# Each variant adds one change to the previous one, so the table reads as an
# ablation: what each fix adds on top of the others.
VARIANTS: dict[str, dict] = {
    "v0": dict(polish_candidates=False, lamarckian=False),
    "fix1-score": dict(polish_candidates=True, lamarckian=False),
    "fix1-lamarck": dict(polish_candidates=True, lamarckian=True),
    # Fixes 2-4 build on fix1-score (Lamarckian write-back was rejected:
    # slower and no better on seeds 1-3).
    "fix2-seeded": dict(polish_candidates=True, lamarckian=False, seeded_init=True),
    "fix3-diversity": dict(polish_candidates=True, lamarckian=False, seeded_init=True,
                           diversity_threshold=0.02),
    "fix4-vmax": dict(polish_candidates=True, lamarckian=False, seeded_init=True,
                      diversity_threshold=0.02, v_max=0.10),
}


def run_one(args: tuple[str, int, str]) -> dict:
    variant, seed, dataset = args
    data = load_problem(dataset)
    cfg = replace(PSOConfig(verbose=False, seed=seed), **VARIANTS[variant])
    pso = MemeticPSO(data, cfg)
    t = time.perf_counter()
    res = pso.run()
    runtime = time.perf_counter() - t
    curve = [h["best_score"] for h in res.history]
    start = curve[0]
    last_impr = max(
        (h["iteration"] for h, prev in zip(res.history[1:], curve) if h["best_score"] < prev - 1e-9),
        default=1,
    )
    best = res.best_feasible_solution or res.best_solution
    return {
        "variant": variant,
        "seed": seed,
        "cost_iter1": start,
        "cost_before_final_polish": curve[-1],
        "final_cost": best.true_cost_eur,
        "feasible": best.feasible,
        "last_improvement_iter": last_impr,
        "improvements": sum(1 for a, b in zip(curve, curve[1:]) if b < a - 1e-9),
        "runtime_s": runtime,
        "restarts": pso.restarts,
        "evaluations": res.evaluations,
        "curve": curve,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--dataset", default=str(ROOT / "dataset"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "results" / "convergence"))
    a = ap.parse_args()
    variants = list(VARIANTS) if a.variants == ["all"] else a.variants
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [(v, s, a.dataset) for v in variants for s in a.seeds]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        results = list(ex.map(run_one, jobs))
    for r in results:
        (out / f"{r['variant']}_s{r['seed']}.json").write_text(json.dumps(r, indent=1))

    print(f"{'variant':14s} {'seed':>4s} {'iter 1':>9s} {'loop end':>9s} {'final':>9s} "
          f"{'impr.':>5s} {'last':>5s} {'time s':>7s}")
    for r in results:
        print(f"{r['variant']:14s} {r['seed']:4d} {r['cost_iter1']:9.2f} "
              f"{r['cost_before_final_polish']:9.2f} {r['final_cost']:9.2f} "
              f"{r['improvements']:5d} {r['last_improvement_iter']:5d} {r['runtime_s']:7.0f}")
    print()
    for v in variants:
        rs = [r for r in results if r["variant"] == v]
        mean = sum(r["final_cost"] for r in rs) / len(rs)
        print(f"{v:14s} mean final {mean:9.2f} EUR over {len(rs)} seeds, "
              f"mean swarm improvements {sum(r['improvements'] for r in rs) / len(rs):.1f}, "
              f"mean time {sum(r['runtime_s'] for r in rs) / len(rs):.0f} s")


if __name__ == "__main__":
    main()
