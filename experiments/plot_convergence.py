"""Plot mean best cost per iteration for each variant in results/convergence."""
import json, sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "results" / "convergence"
labels = {
    "v0": "V0 (as submitted)",
    "fix1-score": "+ fix 1: score candidates like the best",
    "fix2-seeded": "+ fix 2: start near good tours",
}
curves = defaultdict(list)
for f in src.glob("*_s[123].json"):
    r = json.loads(f.read_text())
    if r["variant"] in labels:
        curves[r["variant"]].append(r["curve"])
fig, ax = plt.subplots(figsize=(8, 4.5))
for v, lab in labels.items():
    c = np.array(curves[v]).mean(axis=0)
    ax.plot(np.arange(1, len(c) + 1), c, lw=2, label=f"{lab}  (end {c[-1]:,.1f})")
ax.set_xlabel("Iteration")
ax.set_ylabel("Best plan cost, EUR (mean of seeds 1-3)")
ax.set_title("The swarm now improves the plan instead of standing still")
ax.grid(alpha=0.3)
ax.legend(fontsize=8.5)
fig.tight_layout()
out = ROOT / "results" / "convergence_curves.png"
fig.savefig(out, dpi=160)
print("saved", out)
