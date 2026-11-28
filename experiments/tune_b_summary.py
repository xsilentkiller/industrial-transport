"""Summarise the Option B tuning rounds (comparison/tune_b/runs.csv).

    python -m experiments.tune_b_summary [--ref B_discrete] [methods ...]

For every method: runs, mean objective, swarm share of the improvement, and
the paired difference to the reference method and to the current method on
the same (day, seed) pairs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if not r["error"] and r["freeze"] == "0"]


def table(rows: list[dict], ref: str, methods: list[str] | None = None) -> list[dict]:
    by = {}
    for r in rows:
        by.setdefault(r["method"], {})[(r["instance"], r["seed"])] = r
    methods = methods or sorted(by, key=lambda m: (m != "current", m != ref, m))
    out = []
    for m in methods:
        runs = by.get(m, {})
        if not runs:
            continue
        obj = {k: float(r["objective"]) for k, r in runs.items()}
        sw = sum(float(r["credit_swarm"] or 0) for r in runs.values())
        tot = sum(float(r["credit_swarm"] or 0) + float(r["credit_local_search"] or 0)
                  + float(r["credit_final_polish"] or 0) for r in runs.values())
        row = {"method": m, "runs": len(obj), "mean": sum(obj.values()) / len(obj),
               "swarm_pct": 100 * sw / tot if tot else 0.0}
        for name, other in (("vs_ref", ref), ("vs_current", "current")):
            o = by.get(other, {})
            keys = [k for k in obj if k in o]
            if keys and other != m:
                d = [obj[k] - float(o[k]["objective"]) for k in keys]
                row[name] = sum(d) / len(d)
                row[name + "_wins"] = f"{sum(x < -1e-9 for x in d)}/{len(d)}"
        out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("methods", nargs="*")
    ap.add_argument("--ref", default="B_discrete")
    ap.add_argument("--runs", default="comparison/tune_b/runs.csv")
    a = ap.parse_args(argv)
    print(f"{'method':<22}{'runs':>5}{'mean':>10}{'swarm%':>8}{'vs ref':>9}{'wins':>7}{'vs cur':>9}{'wins':>7}")
    for r in table(load(Path(a.runs)), a.ref, a.methods or None):
        print(f"{r['method']:<22}{r['runs']:>5}{r['mean']:>10.1f}{r['swarm_pct']:>8.0f}"
              f"{r.get('vs_ref', float('nan')):>9.1f}{r.get('vs_ref_wins', '-'):>7}"
              f"{r.get('vs_current', float('nan')):>9.1f}{r.get('vs_current_wins', '-'):>7}")


if __name__ == "__main__":
    main()
