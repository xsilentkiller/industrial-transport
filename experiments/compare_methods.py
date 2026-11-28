"""Side-by-side comparison: current PSO vs Option B (discrete) vs Option C (cluster).

    python -m experiments.compare_methods --name main
    python -m experiments.compare_methods --name quick --quick

Every method plans the same 250-customer days with the same cost model, the
same evaluation budget (plans built and scored inside the loop) and its final
polish after the budget.  Instances default to ``data/osm`` plus the 20
``instances_osm/road_validation`` days (never used to choose any setting),
seeds 1-5.  The ablation re-runs each method with a frozen swarm (no
attraction) on the first few instances.

Per run, the improvement of the best plan is credited to its source:
``swarm`` (a particle's new position), ``local_search`` (local search inside
the loop; only the current method has it) and ``final_polish``.  The start
value is the best construction plan.

Results go to ``comparison/<name>/``: ``runs.csv`` (one row per run,
resumable), ``curves.json`` (best value vs evaluations), ``summary.csv``,
the charts and ``report.html``.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import statistics as st
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("swarm", "local_search", "final_polish")
COLORS = {"current": "#2a78d6", "B_discrete": "#eb6834", "C_cluster": "#1baf7a"}
SOURCE_COLORS = {"swarm": "#2a78d6", "local_search": "#eb6834", "final_polish": "#eda100"}
LABELS = {
    "current": "Current (random-key memetic PSO)",
    "B_discrete": "B: discrete PSO (path relinking)",
    "C_cluster": "C: cluster-centre PSO",
}
FIELDS = [
    "method", "instance", "seed", "freeze", "real_cost", "penalty", "objective",
    "feasible", "evaluations", "iterations", "runtime_s", "start_cost",
    "credit_swarm", "credit_local_search", "credit_final_polish", "error",
]


# ---------------------------------------------------------------------------
# Methods: name -> run(data, seed, budget, freeze) -> (PSOResult, loop evaluations)
# ---------------------------------------------------------------------------
def _run_current(data, seed, budget, freeze):
    from experiments.instrumented_pso import InstrumentedMemeticPSO
    from src.pso import PSOConfig

    # Today's defaults; only the stopping rule is the shared budget.
    cfg = PSOConfig(seed=seed, verbose=False, patience=0, max_seconds=0.0, iterations=10**7)
    opt = InstrumentedMemeticPSO(data, cfg, max_evaluations=budget, freeze_swarm=freeze)
    res = opt.run()
    return res, opt.loop_evaluations


def _run_b(data, seed, budget, freeze):
    from src.pso_discrete import DiscretePSO, DiscretePSOConfig

    res = DiscretePSO(data, DiscretePSOConfig(seed=seed, max_evaluations=budget,
                                              freeze_swarm=freeze)).run()
    return res, res.evaluations


def _run_c(data, seed, budget, freeze):
    from src.pso_cluster import ClusterPSO, ClusterPSOConfig

    res = ClusterPSO(data, ClusterPSOConfig(seed=seed, max_evaluations=budget,
                                            freeze_swarm=freeze)).run()
    return res, res.evaluations


METHODS = {"current": _run_current, "B_discrete": _run_b, "C_cluster": _run_c}


def resolve_method(name: str):
    """A built-in method, or a named Option B preset from experiments/b_variants.py."""
    if name in METHODS:
        return METHODS[name]
    from functools import partial

    from experiments.b_variants import VARIANTS, run_variant

    if name in VARIANTS:
        return partial(run_variant, name)
    raise KeyError(f"unknown method {name!r}")


def method_signature(name: str) -> str | None:
    """The configuration a method name stands for (seed and budget left out).

    Stored per comparison folder in methods.json, so a folder is never
    resumed with runs of a method whose defaults have changed since.
    None for names without a known config (they are not checked).
    """
    if name == "current":
        from src.pso import PSOConfig

        return repr(PSOConfig(seed=0, verbose=False, patience=0, max_seconds=0.0, iterations=10**7))
    if name == "B_discrete":
        from src.pso_discrete import DiscretePSOConfig

        return repr(DiscretePSOConfig(seed=0, max_evaluations=0))
    if name == "C_cluster":
        from src.pso_cluster import ClusterPSOConfig

        return repr(ClusterPSOConfig(seed=0, max_evaluations=0))
    from experiments.b_variants import VARIANTS

    if name in VARIANTS:
        from src.pso_discrete import DiscretePSOConfig

        return repr(DiscretePSOConfig(seed=0, max_evaluations=0, **VARIANTS[name]))
    return None


def check_signatures(folder: Path, methods: list[str], rows: list[dict]) -> None:
    """Refuse to resume a folder whose runs were made with other configs."""
    path = folder / "methods.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if stored is None and rows:
        raise SystemExit(f"{folder} has runs but no methods.json, so their configs cannot be "
                         f"checked; use a new --name")
    stored = stored or {}
    done_methods = {r["method"] for r in rows}
    for m in methods:
        sig = method_signature(m)
        if sig is None:
            continue
        if m in stored and stored[m] != sig:
            raise SystemExit(f"method {m!r} has a different config than the runs in {folder}; "
                             f"use a new --name")
        if m not in stored and m in done_methods:
            raise SystemExit(f"{folder} has runs of {m!r} with an unknown config; use a new --name")
        stored[m] = sig
    path.write_text(json.dumps(stored, indent=1), encoding="utf-8")


def run_key(method: str, instance: str, seed: int, freeze: bool) -> str:
    return f"{method}|{instance}|{seed}|{int(bool(freeze))}"


def run_job(job: dict) -> tuple[dict, list]:
    """One run; never raises (errors go into the row)."""
    from src.data_loader import load_problem

    row = {f: "" for f in FIELDS}
    row.update(method=job["method"], instance=job["instance"], seed=job["seed"],
               freeze=int(job["freeze"]))
    curve: list = []
    t0 = perf_counter()
    try:
        data = load_problem(job["path"])
        res, loop_evals = resolve_method(job["method"])(data, job["seed"], job["budget"], job["freeze"])
        best = res.best_feasible_solution or res.best_solution
        credits = res.history[-1].get("credits", {})
        row.update(
            real_cost=round(float(best.true_cost_eur), 4),
            penalty=round(float(best.penalty), 4),
            objective=round(float(best.objective), 4),
            feasible=int(bool(best.feasible)),
            evaluations=int(loop_evals),
            iterations=int(res.iterations_run),
            start_cost=round(float(res.history[0]["best_score"]), 4),
            **{f"credit_{s}": round(float(credits.get(s, 0.0)), 4) for s in SOURCES},
        )
        curve = [[int(h["evaluations"]), float(h["best_score"])] for h in res.history]
    except Exception as exc:  # recorded, the comparison goes on
        row["error"] = f"{type(exc).__name__}: {exc}"
        if os.environ.get("COMPARE_TRACE"):
            traceback.print_exc()
    row["runtime_s"] = round(perf_counter() - t0, 2)
    return row, curve


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def find_instances(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for p in paths:
        p = Path(p)
        if (p / "params.json").exists():
            found.append(p)
        elif p.is_dir():
            found.extend(sorted(q for q in p.iterdir() if (q / "params.json").exists()))
    return found


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def ok_rows(rows: list[dict], freeze: int) -> list[dict]:
    return [r for r in rows if not r["error"] and int(r["freeze"]) == freeze]


def best_known(rows: list[dict]) -> dict[str, float]:
    best: dict[str, float] = {}
    for r in ok_rows(rows, 0):
        best[r["instance"]] = min(best.get(r["instance"], np.inf), _f(r["objective"]))
    return best


def gap(r: dict, best: dict[str, float]) -> float:
    b = best.get(r["instance"])
    return 100.0 * (_f(r["objective"]) / b - 1.0) if b else float("nan")


def summarise(rows: list[dict], methods: list[str]) -> list[dict]:
    best = best_known(rows)
    out = []
    for m in methods:
        rs = [r for r in ok_rows(rows, 0) if r["method"] == m]
        errors = sum(1 for r in rows if r["method"] == m and r["error"])
        if not rs:
            out.append({"method": m, "runs": 0, "errors": errors})
            continue
        cost = [_f(r["real_cost"]) for r in rs]
        gaps = [gap(r, best) for r in rs]
        total = [max(_f(r["start_cost"]) - _f(r["objective"]), 0.0) for r in rs]
        shares = {}
        for s in SOURCES:
            num = sum(_f(r[f"credit_{s}"]) for r in rs)
            den = sum(total)
            shares[s] = 100.0 * num / den if den > 0 else 0.0
        out.append({
            "method": m, "runs": len(rs), "errors": errors,
            "feasible_pct": round(100.0 * sum(int(r["feasible"]) for r in rs) / len(rs), 1),
            "median_cost": round(st.median(cost), 2), "mean_cost": round(st.mean(cost), 2),
            "best_cost": round(min(cost), 2), "worst_cost": round(max(cost), 2),
            "median_gap_pct": round(st.median(gaps), 2), "worst_gap_pct": round(max(gaps), 2),
            "median_runtime_s": round(st.median(_f(r["runtime_s"]) for r in rs), 1),
            "median_improvement": round(st.median(total), 2),
            **{f"share_{s}_pct": round(shares[s], 1) for s in SOURCES},
        })
    return out


def head_to_head(rows: list[dict], a: str, b: str) -> dict:
    ra = {(r["instance"], r["seed"]): _f(r["objective"]) for r in ok_rows(rows, 0) if r["method"] == a}
    rb = {(r["instance"], r["seed"]): _f(r["objective"]) for r in ok_rows(rows, 0) if r["method"] == b}
    keys = sorted(set(ra) & set(rb))
    if not keys:
        return {"a": a, "b": b, "pairs": 0}
    x = np.array([ra[k] for k in keys])
    y = np.array([rb[k] for k in keys])
    p = None
    try:
        from scipy.stats import wilcoxon

        if np.any(np.abs(x - y) > 1e-9):
            p = float(wilcoxon(x, y).pvalue)
    except Exception:
        p = None
    return {"a": a, "b": b, "pairs": len(keys), "a_wins": int((x < y - 1e-6).sum()),
            "b_wins": int((y < x - 1e-6).sum()), "ties": int((np.abs(x - y) <= 1e-6).sum()),
            "mean_diff": float((x - y).mean()), "p": p}


def ablation(rows: list[dict], methods: list[str]) -> list[dict]:
    out = []
    for m in methods:
        frozen = {(r["instance"], r["seed"]): _f(r["objective"])
                  for r in ok_rows(rows, 1) if r["method"] == m}
        normal = {(r["instance"], r["seed"]): _f(r["objective"])
                  for r in ok_rows(rows, 0) if r["method"] == m}
        keys = sorted(set(frozen) & set(normal))
        if not keys:
            continue
        f = [frozen[k] for k in keys]
        n = [normal[k] for k in keys]
        out.append({"method": m, "pairs": len(keys), "normal_mean": st.mean(n),
                    "frozen_mean": st.mean(f), "swarm_benefit": st.mean(f) - st.mean(n),
                    "normal_better": sum(1 for a, b in zip(n, f) if a < b - 1e-6)})
    return out


# ---------------------------------------------------------------------------
# Charts (matplotlib PNG, embedded in the report)
# ---------------------------------------------------------------------------
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c8ccd2")
    ax.spines["bottom"].set_color("#c8ccd2")
    ax.tick_params(colors="#55595f", labelsize=9)
    ax.grid(axis="x", color="#eceef1", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_gaps(rows, methods, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = best_known(rows)
    fig, ax = plt.subplots(figsize=(8.5, 0.9 + 0.9 * len(methods)))
    for i, m in enumerate(methods):
        g = [gap(r, best) for r in ok_rows(rows, 0) if r["method"] == m]
        if not g:
            continue
        y = len(methods) - 1 - i
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(g))
        ax.scatter(g, y + jitter, s=16, color=COLORS.get(m, "#888"), alpha=0.55,
                   edgecolor="white", linewidth=0.6, zorder=3)
        med = st.median(g)
        ax.plot([med, med], [y - 0.3, y + 0.3], color="#1f2329", linewidth=2, zorder=4)
        ax.text(med, y + 0.36, f"median {med:.2f} %", ha="center", fontsize=8.5, color="#1f2329")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([LABELS.get(m, m) for m in reversed(methods)], fontsize=9)
    ax.set_xlabel("Gap to the best plan found for that day (%), lower is better", fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_convergence(curves, rows, methods, budget, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = best_known(rows)
    grid = np.linspace(0, budget, 101)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for m in methods:
        series = []
        for r in ok_rows(rows, 0):
            if r["method"] != m:
                continue
            c = curves.get(run_key(m, r["instance"], int(r["seed"]), False))
            b = best.get(r["instance"])
            if not c or not b:
                continue
            ev = np.array([p[0] for p in c], dtype=float)
            val = np.array([p[1] for p in c], dtype=float)
            val = np.minimum.accumulate(val)
            idx = np.searchsorted(ev, grid, side="right") - 1
            g = np.where(idx >= 0, 100.0 * (val[np.clip(idx, 0, None)] / b - 1.0), np.nan)
            series.append(g)
        if not series:
            continue
        arr = np.array(series)
        valid = ~np.all(np.isnan(arr), axis=0)
        med = np.full(arr.shape[1], np.nan)
        med[valid] = np.nanmedian(arr[:, valid], axis=0)
        ax.plot(grid, med, color=COLORS.get(m, "#888"), linewidth=2, label=LABELS.get(m, m))
        last = med[~np.isnan(med)][-1] if np.any(~np.isnan(med)) else None
        if last is not None:
            ax.text(budget * 1.01, last, f"{last:.1f} %", fontsize=8.5, va="center", color="#1f2329")
    ax.set_xlabel("Plan evaluations inside the loop (equal budget)", fontsize=9)
    ax.set_ylabel("Median gap to the day's best (%)", fontsize=9)
    ax.set_xlim(0, budget * 1.1)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#eceef1", linewidth=0.8)
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_sources(summary, methods, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [s for s in summary if s.get("runs")]
    fig, ax = plt.subplots(figsize=(8.5, 0.9 + 0.75 * len(rows)))
    names = {"swarm": "Swarm moves", "local_search": "Local search in the loop",
             "final_polish": "Final polish"}
    for i, s in enumerate(rows):
        y = len(rows) - 1 - i
        left = 0.0
        for src in SOURCES:
            w = s[f"share_{src}_pct"]
            if w > 0:
                ax.barh(y, w, left=left, height=0.55, color=SOURCE_COLORS[src],
                        edgecolor="white", linewidth=2, label=names[src] if i == 0 else None)
                if w >= 8:
                    ax.text(left + w / 2, y, f"{w:.0f} %", ha="center", va="center",
                            fontsize=8.5, color="white")
            left += w
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS.get(s["method"], s["method"]) for s in reversed(rows)], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the improvement over the best construction plan (%)", fontsize=9)
    _style(ax)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=8.5, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_ablation(abl, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 0.9 + 0.9 * max(1, len(abl))))
    for i, a in enumerate(abl):
        y = len(abl) - 1 - i
        c = COLORS.get(a["method"], "#888")
        ax.barh(y + 0.17, a["normal_mean"], height=0.3, color=c, edgecolor="white",
                linewidth=2, label="Swarm on" if i == 0 else None)
        ax.barh(y - 0.17, a["frozen_mean"], height=0.3, color="white", edgecolor=c,
                hatch="///", linewidth=1.2, label="Swarm frozen" if i == 0 else None)
        ax.text(a["normal_mean"], y + 0.17, f" {a['normal_mean']:,.0f}", va="center", fontsize=8.5)
        ax.text(a["frozen_mean"], y - 0.17, f" {a['frozen_mean']:,.0f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(abl)))
    ax.set_yticklabels([LABELS.get(a["method"], a["method"]) for a in reversed(abl)], fontsize=9)
    lo = min([min(a["normal_mean"], a["frozen_mean"]) for a in abl], default=0)
    hi = max([max(a["normal_mean"], a["frozen_mean"]) for a in abl], default=1)
    ax.set_xlim(lo * 0.97, hi * 1.03)
    ax.set_xlabel("Mean objective (EUR) on the ablation days, lower is better", fontsize=9)
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _img(path: Path) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"<img alt='{html.escape(path.stem)}' src='data:image/png;base64,{data}'>"


def _table(headers, rows, num=()) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(
            f"<td class='{'num' if i in num else ''}'>{html.escape(str(c))}</td>"
            for i, c in enumerate(r)) + "</tr>"
        for r in rows)
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def write_report(folder: Path, rows: list[dict], curves: dict, methods: list[str], args) -> Path:
    summary = summarise(rows, methods)
    with (folder / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        keys = sorted({k for s in summary for k in s}, key=lambda k: (k != "method", k))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    pairs = [head_to_head(rows, a, b) for i, a in enumerate(methods) for b in methods[i + 1:]]
    abl = ablation(rows, methods)

    charts = {
        "gaps": folder / "gaps.png", "convergence": folder / "convergence.png",
        "sources": folder / "improvement_sources.png", "ablation": folder / "ablation.png",
    }
    plot_gaps(rows, methods, charts["gaps"])
    plot_convergence(curves, rows, methods, args.budget, charts["convergence"])
    plot_sources(summary, methods, charts["sources"])
    if abl:
        plot_ablation(abl, charts["ablation"])

    def fmt(x, nd=2):
        return f"{x:,.{nd}f}" if isinstance(x, (int, float)) and x == x else "–"

    sum_rows = []
    for s in summary:
        if not s.get("runs"):
            sum_rows.append([LABELS.get(s["method"], s["method"]), 0, "–", "–", "–", "–", "–",
                             "–", "–", "–", s.get("errors", 0)])
            continue
        sum_rows.append([
            LABELS.get(s["method"], s["method"]), s["runs"], f"{s['feasible_pct']:.0f} %",
            fmt(s["median_cost"]), fmt(s["best_cost"]), fmt(s["worst_cost"]),
            f"{s['median_gap_pct']:.2f} %", f"{s['worst_gap_pct']:.2f} %",
            f"{s['share_swarm_pct']:.0f} %", f"{s['median_runtime_s']:.0f} s", s["errors"],
        ])
    h2h_rows = [[LABELS.get(p["a"], p["a"]), LABELS.get(p["b"], p["b"]), p["pairs"],
                 p.get("a_wins", "–"), p.get("b_wins", "–"), p.get("ties", "–"),
                 fmt(p.get("mean_diff", float("nan"))),
                 "–" if p.get("p") is None else f"{p['p']:.4f}"] for p in pairs]
    abl_rows = [[LABELS.get(a["method"], a["method"]), a["pairs"], fmt(a["normal_mean"]),
                 fmt(a["frozen_mean"]), fmt(a["swarm_benefit"]),
                 f"{a['normal_better']} / {a['pairs']}"] for a in abl]

    ok = [s for s in summary if s.get("runs")]
    winner = min(ok, key=lambda s: s["median_gap_pct"]) if ok else None
    swarmiest = max(ok, key=lambda s: s["share_swarm_pct"]) if ok else None
    conclusion = []
    if winner:
        conclusion.append(
            f"Lowest median gap: <b>{html.escape(LABELS.get(winner['method'], winner['method']))}</b> "
            f"({winner['median_gap_pct']:.2f} %, median real cost {winner['median_cost']:,.2f} EUR).")
    if swarmiest:
        conclusion.append(
            f"Largest share of the improvement made by the swarm itself: "
            f"<b>{html.escape(LABELS.get(swarmiest['method'], swarmiest['method']))}</b> "
            f"({swarmiest['share_swarm_pct']:.0f} %).")
    for a in abl:
        verdict = ("the swarm helps" if a["swarm_benefit"] > 1e-6 and a["normal_better"] > a["pairs"] / 2
                   else "no clear swarm effect")
        conclusion.append(
            f"{html.escape(LABELS.get(a['method'], a['method']))}: freezing the swarm changes the mean "
            f"objective by {a['swarm_benefit']:+,.2f} EUR ({verdict}).")

    instances = sorted({r["instance"] for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    css = """
body{margin:0;background:#f6f7f9;color:#1f2329;font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:28px 22px 48px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:0 0 10px}
.sub{color:#5b616b;margin:0}
section{background:#fff;border:1px solid #e3e6ea;border-radius:12px;padding:18px 20px;margin:16px 0}
.tw{overflow-x:auto;border:1px solid #e3e6ea;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px}
th{background:#f3f5f7;color:#454b54;text-align:left;font-weight:600;padding:7px 10px;white-space:nowrap}
td{padding:6px 10px;border-top:1px solid #eef0f2;white-space:nowrap}td.num{text-align:right;font-variant-numeric:tabular-nums}
img{max-width:100%;display:block}.note{color:#5b616b;font-size:13px}
ul.c li{margin:4px 0}
"""
    body = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PSO comparison: current vs B vs C</title><style>{css}</style></head><body><div class="wrap">
<h1>PSO comparison: current vs B vs C</h1>
<p class="sub">{len(instances)} day(s) &middot; seeds {', '.join(map(str, seeds))} &middot;
budget {args.budget:,} plan evaluations per run &middot; generated {datetime.now():%Y-%m-%d %H:%M}</p>

<section><h2>Conclusion</h2><ul class="c">{''.join(f'<li>{c}</li>' for c in conclusion)}</ul>
<p class="note">Fair by construction: every method plans the same 250-customer days with the
same cost model and the instructor's rules, gets the same number of plan evaluations inside
its loop, and is judged on days that were never used to choose its settings. The current
method runs with today's defaults (it alone has local search inside the loop); B and C have
none, so their improvement comes from the swarm and the final polish.</p></section>

<section><h2>Results side by side</h2>
{_table(["Method", "Runs", "Feasible", "Median real cost (EUR)", "Best", "Worst",
         "Median gap", "Worst gap", "Swarm share", "Median runtime", "Errors"],
        sum_rows, num=range(1, 11))}
<p class="note">Gap = how far a run's objective (real cost + penalty) is above the best plan
any method found for that day. Swarm share = part of the improvement over the best
construction plan that came from swarm moves.</p>
{_img(charts["gaps"])}</section>

<section><h2>Head to head (same day, same seed)</h2>
{_table(["Method A", "Method B", "Pairs", "A better", "B better", "Ties",
         "Mean A − B (EUR)", "Wilcoxon p"], h2h_rows, num=range(2, 8))}
<p class="note">Wilcoxon signed-rank test on the paired objectives; p &lt; 0.05 means the
difference is unlikely to be chance.</p></section>

<section><h2>Convergence</h2>{_img(charts["convergence"])}
<p class="note">Median over all runs of each method of the best plan's gap to the day's best,
by plan evaluations used. The last step of each line is the final polish.</p></section>

<section><h2>Where the improvement comes from</h2>{_img(charts["sources"])}
<p class="note">Credits of every improvement of the best plan after the construction
heuristics: swarm moves, local search inside the loop (current method only), final
polish.</p></section>

<section><h2>Does the swarm matter? (frozen-swarm ablation)</h2>
{_table(["Method", "Pairs", "Swarm on (mean EUR)", "Swarm frozen (mean EUR)",
         "Swarm benefit (EUR)", "Swarm on better"], abl_rows, num=range(1, 6))
 if abl_rows else "<p class='note'>No ablation runs.</p>"}
{_img(charts["ablation"]) if abl else ""}
<p class="note">Frozen = particles get no pull toward their own or the swarm's best (current:
inertia = c1 = c2 = 0; B: mutation only; C: random drift only). If freezing changes nothing,
the swarm is not doing the work.</p></section>

<section><h2>Files</h2><p class="note">runs.csv (one row per run), summary.csv,
curves.json, gaps.png, convergence.png, improvement_sources.png, ablation.png.
Days: {html.escape(', '.join(instances))}.</p></section>
</div></body></html>"""
    path = folder / "report.html"
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="main")
    ap.add_argument("--out", default="comparison")
    ap.add_argument("--instances", nargs="+",
                    default=["data/osm", "instances_osm/road_validation"])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--budget", type=int, default=20_000)
    ap.add_argument("--ablation-instances", type=int, default=6,
                    help="frozen-swarm runs on the first N instances (0 = none)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true",
                    help="data/osm only, seeds 1-2, budget 3000, ablation on it")
    return ap


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.instances, args.seeds, args.budget = ["data/osm"], [1, 2], 3000
        args.ablation_instances = 1
    folder = Path(args.out) / args.name
    folder.mkdir(parents=True, exist_ok=True)
    instances = find_instances(args.instances)
    if not instances:
        raise SystemExit(f"no instances found in {args.instances}")
    names = {p: (p.name if p.parent.name != "data" else str(p).replace("\\", "/")) for p in instances}

    jobs = []
    for freeze, insts in ((False, instances), (True, instances[: args.ablation_instances])):
        for p in insts:
            for seed in args.seeds:
                for m in args.methods:
                    jobs.append({"method": m, "instance": names[p], "path": str(p), "seed": seed,
                                 "freeze": freeze, "budget": args.budget})

    runs_path = folder / "runs.csv"
    curves_path = folder / "curves.json"
    check_signatures(folder, args.methods, load_rows(runs_path))
    curves = json.loads(curves_path.read_text(encoding="utf-8")) if curves_path.exists() else {}
    done = {run_key(r["method"], r["instance"], int(r["seed"]), bool(int(r["freeze"])))
            for r in load_rows(runs_path)}
    todo = [j for j in jobs if run_key(j["method"], j["instance"], j["seed"], j["freeze"]) not in done]
    print(f"{len(instances)} instance(s), methods {args.methods}, seeds {args.seeds}, "
          f"budget {args.budget}: {len(jobs)} runs ({len(jobs) - len(todo)} already done)")

    def store(job, row, curve, k, n):
        append_row(runs_path, row)
        curves[run_key(job["method"], job["instance"], job["seed"], job["freeze"])] = curve
        curves_path.write_text(json.dumps(curves), encoding="utf-8")
        tag = " frozen" if job["freeze"] else ""
        res = row["error"] or f"{_f(row['real_cost']):,.2f} EUR (obj {_f(row['objective']):,.2f})"
        print(f"[{k}/{n}] {job['method']:<10} {job['instance']:<16} seed {job['seed']}{tag}: "
              f"{res}  {row['runtime_s']} s", flush=True)

    if args.workers <= 1:
        for k, job in enumerate(todo, 1):
            row, curve = run_job(job)
            store(job, row, curve, k, len(todo))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_job, j): j for j in todo}
            for k, fut in enumerate(as_completed(futs), 1):
                row, curve = fut.result()
                store(futs[fut], row, curve, k, len(todo))

    report = write_report(folder, load_rows(runs_path), curves, args.methods, args)
    print(f"Saved {report}")
    return folder


if __name__ == "__main__":
    main()
