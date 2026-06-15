"""Generate the raw-data products required by ``figure_5.ipynb``.

This mirrors the Figure-5 generation of ``data_generation.ipynb`` but with one key change:
the empirical directed-evolution panels use a single **best-variant trajectory sweep** per
landscape, so the strategy-performance heat map (final slice) and the directed-evolution
lines (selected cells) are two views of the *same* simulation — guaranteeing they agree.

Shared parameters for the empirical DE panels (heat map AND lines use these identical values):
  generations = 150, popsize = 1200, per-site mutation = 0.01, best-variant (max) metric.
ParD3 is treated as a true N=3 landscape (5x5 strategy grid, its own N3A20 lookup); the other
three empirical landscapes are N=4 (7x7 grid, N4A20 lookup).

Run from the repository root::

    python scripts/generate_figure5_raw_data.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from tqdm.auto import tqdm

from slide.data_generation import (
    EMPIRICAL_NAMES,
    RAW_FILENAMES,
    generate_empirical_decay_curves,
    generate_empirical_strategy_trajectory_sweep,
    generate_nk_decay_curves,
    generate_nk_strategy_sweep,
    load_empirical_landscape,
    nk_grid_pairs,
    strategy_grid,
    uniform_start_locs,
)
from slide.utils import get_raw_data_dir, raw_path, save_raw


def _exists(key: str) -> bool:
    """Return whether the raw product for ``key`` already exists on disk."""
    return raw_path(RAW_FILENAMES[key]).exists()

# ----------------------------------------------------------------------------
# Shared directed-evolution parameters for the empirical panels (heat map + lines).
# ----------------------------------------------------------------------------
# One sim at the SLIDE calibration mutation (0.1/N) and popsize, run out to a long horizon.
# The notebook shows the heat map as the HEATMAP_GEN (25) slice — the calibration point that
# matches the published figure + the look-up — while the DE lines use the full DE_GENERATIONS.
DE_GENERATIONS = 150
HEATMAP_GEN = 25
DE_POPSIZE = 1200
DE_STARTS = int(os.environ.get("DE_STARTS", "100"))
DE_REPS = int(os.environ.get("DE_REPS", "300"))
DE_BATCH = int(os.environ.get("DE_BATCH", "0"))  # replicates per fused vmap; 0 = all at once
# 5A lookup: NK landscapes averaged per (N,K) point. 1 => noisy argmax / no trend; 50 => smooth.
NK_GRID_LANDSCAPES = int(os.environ.get("NK_GRID_LANDSCAPES", "50"))
SEED = 42


def generate_nk_strategy_grid() -> None:
    """Generate the Figure 5A NK directed-evolution strategy lookup grid (final-fitness)."""

    n_range = (10, 50)
    num_grid_samples = 10
    num_alleles = 2
    nk_pairs = nk_grid_pairs(n_range, num_grid_samples)

    mutation_rate = 0.1
    popsize = 1200
    num_reps = 25
    num_steps = 25
    strategy_grid_size = 7
    thresholds, base_chances, splits = strategy_grid(strategy_grid_size)

    num_landscapes_per_pair = NK_GRID_LANDSCAPES
    grid = []
    for n_sites, k in tqdm(nk_pairs, desc="NK strategy grid (5A)"):
        pair = generate_nk_strategy_sweep(
            n_sites=n_sites, num_alleles=num_alleles, k_values=[k], mutation_rate=mutation_rate,
            popsize=popsize, num_landscapes=num_landscapes_per_pair, num_reps=num_reps, num_steps=num_steps,
            strategy_grid_size=strategy_grid_size, outer_reps=1, seed=SEED,
        )
        # generate_nk_strategy_sweep returns (outer=1, k=1, reps, base, split); transpose to
        # (split, base, reps) for the "split_base_reps" layout. (reshape here would scramble axes.)
        grid.append(np.asarray(pair)[0, 0].transpose(2, 1, 0))

    save_raw(
        {
            "data": np.asarray(grid),
            "params": {
                "N_range": n_range, "num_grid_samples": num_grid_samples, "nk_pairs": nk_pairs,
                "A": num_alleles, "mutation_rate": mutation_rate, "popsize": popsize,
                "num_landscapes_per_pair": num_landscapes_per_pair, "num_reps": num_reps, "M": num_steps,
                "strategy_grid_size": strategy_grid_size, "outer_reps": 1,
                "thresholds": np.asarray(thresholds), "base_chances": np.asarray(base_chances),
                "splits": splits, "seed": SEED,
            },
            "metadata": {"description": "NK directed-evolution strategy lookup grid.", "paper_reference": "Figure 5A", "output_key": "nk_strategy_grid", "filename": RAW_FILENAMES["nk_strategy_grid"]},
        },
        RAW_FILENAMES["nk_strategy_grid"],
    )


def generate_nk_lookup(*, n_sites: int, k_values: list[int], strategy_grid_size: int, decay_key: str, strategy_key: str, num_landscapes: int = 125) -> None:
    """Generate the NK decay + strategy lookup products for one alphabet/site count.

    Parameters:
    - n_sites: int
        Number of NK sites (4 for GB1/TrpB/TEV, 3 for ParD3).
    - k_values: list[int]
        NK ``K`` values to simulate.
    - strategy_grid_size: int
        Strategy grid size (7 for N=4, 5 for N=3).
    - decay_key, strategy_key: str
        Registry keys for the decay and strategy raw products.
    - num_landscapes: int
        Number of random NK landscapes per K value for the strategy sweep.
    """

    num_alleles = 20
    mutation_rate = 0.1
    popsize = 1200
    num_steps = 25
    thresholds, base_chances, splits = strategy_grid(strategy_grid_size)

    decay = generate_nk_decay_curves(
        n_sites=n_sites, num_alleles=num_alleles, k_values=k_values, mutation_rate=mutation_rate,
        popsize=popsize, num_starts=10000, num_reps=10, num_steps=num_steps, seed=SEED,
    )
    save_raw(
        {
            "data": decay,
            "params": {"N": n_sites, "A": num_alleles, "K_values": k_values, "mutation_rate": mutation_rate, "popsize": popsize, "num_starts": 10000, "num_reps": 10, "M": num_steps, "seed": SEED},
            "metadata": {"description": f"N{n_sites}A20 NK no-selection decay curves.", "paper_reference": "Figure 5", "output_key": decay_key, "filename": RAW_FILENAMES[decay_key]},
        },
        RAW_FILENAMES[decay_key],
    )

    strategy = generate_nk_strategy_sweep(
        n_sites=n_sites, num_alleles=num_alleles, k_values=k_values, mutation_rate=mutation_rate,
        popsize=popsize, num_landscapes=num_landscapes, num_reps=10, num_steps=num_steps,
        strategy_grid_size=strategy_grid_size, outer_reps=10, seed=SEED,
    )
    save_raw(
        {
            "data": strategy,
            "params": {"N": n_sites, "A": num_alleles, "K_values": k_values, "mutation_rate": mutation_rate, "popsize": popsize, "num_landscapes": num_landscapes, "num_reps": 10, "M": num_steps, "strategy_grid_size": strategy_grid_size, "outer_reps": 10, "thresholds": np.asarray(thresholds), "base_chances": np.asarray(base_chances), "splits": splits, "seed": SEED},
            "metadata": {"description": f"N{n_sites}A20 NK directed-evolution strategy sweep.", "paper_reference": "Figure 5", "output_key": strategy_key, "filename": RAW_FILENAMES[strategy_key]},
        },
        RAW_FILENAMES[strategy_key],
    )


def generate_empirical_decay() -> None:
    """Generate the empirical uniform-start decay curves (for the rho_2 estimate / decay panel)."""

    landscapes = {name: load_empirical_landscape(name) for name in EMPIRICAL_NAMES}
    for name, landscape in tqdm(landscapes.items(), desc="empirical decay (D-G)"):
        popsize = 60 if name == "ParD3" else 2500
        starts_count = 8000 if name == "ParD3" else 10000
        per_site = 0.1 / landscape.ndim
        starts = uniform_start_locs(landscape, num_starts=starts_count, seed=SEED)
        decay = generate_empirical_decay_curves(landscape, mutation_rate=per_site, popsize=popsize, starts=starts, num_reps=10, num_steps=25, seed=SEED)
        save_raw(
            {
                "data": decay,
                "params": {"name": name, "mutation_rate": 0.1, "per_site_mutation_rate": per_site, "popsize": popsize, "starts_count": starts_count, "start_policy": "uniform", "num_reps": 10, "M": 25, "seed": SEED},
                "metadata": {"description": f"{name} uniform-start empirical decay curves.", "paper_reference": "Figure 4", "output_key": f"empirical_decay_{name}_uniform", "filename": RAW_FILENAMES[f"empirical_decay_{name}_uniform"]},
            },
            RAW_FILENAMES[f"empirical_decay_{name}_uniform"],
        )


def generate_empirical_trajectory_sweeps() -> None:
    """Generate the unified best-variant trajectory sweep per empirical landscape (heat map + lines)."""

    landscapes = {name: load_empirical_landscape(name) for name in EMPIRICAL_NAMES}
    for name, landscape in tqdm(landscapes.items(), desc="empirical trajectory sweep (D-G)"):
        if _exists(f"empirical_strategy_traj_{name}"):
            continue  # per-landscape skip so we only regenerate what's missing
        grid_size = 5 if landscape.ndim == 3 else 7
        # ParD3 is the small 3-site landscape: keep its original popsize 60 (matches the published
        # sweep, where high-split strategies starve and Baseline wins); 1200 for the 4-site landscapes.
        popsize = 60 if landscape.ndim == 3 else DE_POPSIZE
        per_site_mutation = 0.1 / landscape.ndim  # calibration mutation rate (0.1/N), matches the look-up sweep
        starts = uniform_start_locs(landscape, num_starts=DE_STARTS, seed=SEED)
        trajectories, thresholds, base_chances, splits = generate_empirical_strategy_trajectory_sweep(
            landscape, starts, mutation_rate=per_site_mutation, popsize=popsize, num_reps=DE_REPS,
            num_steps=DE_GENERATIONS, strategy_grid_size=grid_size, seed=SEED, batch_size=DE_BATCH,
        )
        save_raw(
            {
                "data": trajectories,
                "params": {
                    "name": name, "mutation_rate": per_site_mutation, "popsize": popsize,
                    "num_starts": DE_STARTS, "num_reps": DE_REPS, "M": DE_GENERATIONS,
                    "heatmap_gen": HEATMAP_GEN, "strategy_grid_size": grid_size, "thresholds": thresholds,
                    "base_chances": base_chances, "splits": splits, "seed": SEED,
                },
                "metadata": {"description": f"{name} unified best-variant trajectory sweep (heat map final slice + DE lines).", "paper_reference": "Figure 5D-G", "output_key": f"empirical_strategy_traj_{name}", "filename": RAW_FILENAMES[f"empirical_strategy_traj_{name}"]},
            },
            RAW_FILENAMES[f"empirical_strategy_traj_{name}"],
        )


def main() -> None:
    """Generate all raw-data products required by ``figure_5.ipynb``."""

    print(f"Writing raw data to: {get_raw_data_dir()}")
    start = time.perf_counter()
    # Idempotent: skip products that already exist so re-runs only generate what's missing.
    if not _exists("nk_strategy_grid"):
        generate_nk_strategy_grid()
    if not (_exists("nk_decay_N4_A20") and _exists("nk_strategy_N4_A20")):
        generate_nk_lookup(n_sites=4, k_values=[1, 2, 3], strategy_grid_size=7, decay_key="nk_decay_N4_A20", strategy_key="nk_strategy_N4_A20")
    if not (_exists("nk_decay_N3_A20") and _exists("nk_strategy_N3_A20")):
        generate_nk_lookup(n_sites=3, k_values=[1, 2], strategy_grid_size=5, decay_key="nk_decay_N3_A20", strategy_key="nk_strategy_N3_A20")
    if not all(_exists(f"empirical_decay_{name}_uniform") for name in EMPIRICAL_NAMES):
        generate_empirical_decay()
    if not all(_exists(f"empirical_strategy_traj_{name}") for name in EMPIRICAL_NAMES):
        generate_empirical_trajectory_sweeps()
    print(f"Done in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
