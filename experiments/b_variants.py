"""Named Option B presets for the tuning rounds.

``compare_methods`` runs any name listed here as a method, e.g.

    python -m experiments.compare_methods --name tune_b_r1 \
        --instances instances_osm/road_tuning --seeds 1 2 3 \
        --ablation-instances 0 --methods B_discrete B_ring1 B_ring2 B_ring4

Each entry is a dict of ``DiscretePSOConfig`` overrides; everything else
(seed, budget, freeze) comes from the comparison job.  The tuning presets
pin the pre-tuning coefficients (swarm 20, c 1.0, moves 2) explicitly, so
they reproduce the runs in comparison/tune_b after the defaults changed.
"""

from __future__ import annotations

VARIANTS: dict[str, dict] = {
    # B before the 2026-11-28 tuning (the B of comparison/main)
    "B_original": {"swarm_size": 20, "c1": 1.0, "c2": 1.0, "mutation_moves": 2},
    # Round 1: topology
    "B_ring": {"swarm_size": 20, "c1": 1.0, "c2": 1.0, "mutation_moves": 2, "topology": "ring", "neighbourhood": 2},
    "B_ring1": {"swarm_size": 20, "c1": 1.0, "c2": 1.0, "mutation_moves": 2, "topology": "ring", "neighbourhood": 1},
    "B_ring2": {"swarm_size": 20, "c1": 1.0, "c2": 1.0, "mutation_moves": 2, "topology": "ring", "neighbourhood": 2},
    "B_ring4": {"swarm_size": 20, "c1": 1.0, "c2": 1.0, "mutation_moves": 2, "topology": "ring", "neighbourhood": 4},
    # Round 2: inertia schedule and restarts (global topology)
    "B_inert6to05": {"swarm_size": 20, "c1": 1.0, "c2": 1.0, "mutation_moves": 2, "inertia": 0.6, "inertia_end": 0.05},
    "B_inert4to1": {"swarm_size": 20, "c1": 1.0, "c2": 1.0, "mutation_moves": 2, "inertia": 0.4, "inertia_end": 0.1},
    "B_rs20": {"swarm_size": 20, "c1": 1.0, "c2": 1.0, "mutation_moves": 2, "restart_after": 20},
    "B_rs50": {"swarm_size": 20, "c1": 1.0, "c2": 1.0, "mutation_moves": 2, "restart_after": 50},
    "B_ring4_rs50": {"swarm_size": 20, "c1": 1.0, "c2": 1.0, "mutation_moves": 2, "topology": "ring", "neighbourhood": 4, "restart_after": 50},
    # Round 3: coefficients (more intensification)
    "B_c15": {"swarm_size": 20, "c1": 1.5, "c2": 1.5, "mutation_moves": 2},
    "B_c2": {"swarm_size": 20, "c1": 2.0, "c2": 2.0, "mutation_moves": 2},
    "B_m1": {"swarm_size": 20, "c1": 1.0, "c2": 1.0, "mutation_moves": 1},
    "B_i1": {"swarm_size": 20, "c1": 1.0, "c2": 1.0, "mutation_moves": 2, "inertia": 0.1},
    "B_s10": {"swarm_size": 10, "c1": 1.0, "c2": 1.0, "mutation_moves": 2},
    "B_s40": {"swarm_size": 40, "c1": 1.0, "c2": 1.0, "mutation_moves": 2},
    # Round 4: combine the round-3 winners
    "B_c2m1": {"swarm_size": 20, "c1": 2.0, "c2": 2.0, "mutation_moves": 1},
    "B_c3m1": {"swarm_size": 20, "c1": 3.0, "c2": 3.0, "mutation_moves": 1},
    "B_c2m1i4": {"swarm_size": 20, "c1": 2.0, "c2": 2.0, "mutation_moves": 1, "inertia": 0.4},
    "B_c2m1s40": {"swarm_size": 40, "c1": 2.0, "c2": 2.0, "mutation_moves": 1},
    "B_c2m1sched": {"swarm_size": 20, "c1": 2.0, "c2": 2.0, "mutation_moves": 1, "inertia": 0.4, "inertia_end": 0.1},
    "B_c3m1s40": {"swarm_size": 40, "c1": 3.0, "c2": 3.0, "mutation_moves": 1},
}


def run_variant(name: str, data, seed: int, budget: int, freeze: bool):
    """Run the preset ``name``; returns (PSOResult, loop evaluations)."""
    from src.pso_discrete import DiscretePSO, DiscretePSOConfig

    overrides = VARIANTS[name]
    cfg = DiscretePSOConfig(seed=seed, max_evaluations=budget, freeze_swarm=freeze, **overrides)
    res = DiscretePSO(data, cfg).run()
    return res, res.evaluations
