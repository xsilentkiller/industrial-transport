"""Named Option B presets for the tuning rounds.

``compare_methods`` runs any name listed here as a method, e.g.

    python -m experiments.compare_methods --name tune_b_r1 \
        --instances instances_osm/road_tuning --seeds 1 2 3 \
        --ablation-instances 0 --methods B_discrete B_ring1 B_ring2 B_ring4

Each entry is a dict of ``DiscretePSOConfig`` overrides; everything else
(seed, budget, freeze) comes from the comparison job.
"""

from __future__ import annotations

VARIANTS: dict[str, dict] = {
    # Round 1: topology
    "B_ring": {"topology": "ring", "neighbourhood": 2},
    "B_ring1": {"topology": "ring", "neighbourhood": 1},
    "B_ring2": {"topology": "ring", "neighbourhood": 2},
    "B_ring4": {"topology": "ring", "neighbourhood": 4},
}


def run_variant(name: str, data, seed: int, budget: int, freeze: bool):
    """Run the preset ``name``; returns (PSOResult, loop evaluations)."""
    from src.pso_discrete import DiscretePSO, DiscretePSOConfig

    overrides = VARIANTS[name]
    cfg = DiscretePSOConfig(seed=seed, max_evaluations=budget, freeze_swarm=freeze, **overrides)
    res = DiscretePSO(data, cfg).run()
    return res, res.evaluations
