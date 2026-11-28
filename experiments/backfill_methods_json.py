"""One-off: write methods.json for comparison folders made before it existed.

Those folders were run before the 2026-11-28 tuning, so their ``B_discrete``
rows are the old B (preset ``B_original``); the other names are unchanged.

    python -m experiments.backfill_methods_json comparison/main comparison/tune_b ...
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from experiments import compare_methods as cm
from experiments.b_variants import VARIANTS
from src.pso_discrete import DiscretePSOConfig

LEGACY = {"B_discrete": repr(DiscretePSOConfig(seed=0, max_evaluations=0, **VARIANTS["B_original"]))}


def backfill(folder: Path) -> dict:
    path = folder / "methods.json"
    if path.exists():
        raise SystemExit(f"{path} already exists")
    with open(folder / "runs.csv", encoding="utf-8") as f:
        methods = sorted({r["method"] for r in csv.DictReader(f)})
    sigs = {m: LEGACY.get(m) or cm.method_signature(m) for m in methods}
    sigs = {m: s for m, s in sigs.items() if s is not None}
    path.write_text(json.dumps(sigs, indent=1), encoding="utf-8")
    return sigs


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(arg, sorted(backfill(Path(arg))))
