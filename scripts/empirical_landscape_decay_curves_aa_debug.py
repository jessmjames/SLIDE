"""
Debug script: AA-space simulation structured identically to empirical_landscape_decay_curves_codon_fast.py.

Purpose: verify that the codon pipeline gives the same rho as the plain AA pipeline.
Differences from the codon script:
  - fitness function: build_empirical_landscape_function  (no codon mapping)
  - mutation function: build_mutation_function(..., num_options=20)  (A=20 AA states)
  - starting locations: all AA positions directly (no aa_starts_to_codon conversion)
  - output files: decay_curves_{landscape}_aa_debug_m0.1_all_starts.pkl

Everything else (batching, p, rng seeds, generate_decay_curve structure) is identical
to the codon script so any difference in rho must come from the codon machinery.
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import pickle
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from direvo_functions import (
    build_empirical_landscape_function,
    build_mutation_function,
    build_selection_function,
    run_directed_evolution,
)
import selection_function_library as slct
import tqdm
from slide_config import get_slide_data_dir

slide_data_dir = str(get_slide_data_dir())
landscape_dir = os.path.join(parent_dir, 'landscape_arrays')


# ---------------------------------------------------------------------------
# Helpers  (identical to codon script)
# ---------------------------------------------------------------------------

def all_start_locs(ld):
    return np.indices(ld.shape).reshape(len(ld.shape), -1).T


def generate_decay_curve(start_array, fitness_function, mutation_function,
                         p=2500, batch_size=3000):
    """
    Identical structure to generate_decay_curve in empirical_landscape_decay_curves_codon_fast.py.
    """
    start_array = np.array(start_array)
    total_starts, ndim = start_array.shape

    sel_params = {'threshold': 0.0, 'base_chance': 1.0}
    selection_function = build_selection_function(slct.base_chance_threshold_select, sel_params)

    _r1, _r2, r3 = jr.split(jr.PRNGKey(42), 3)
    rng_seeds = jr.split(r3, 10)

    def run_from_start(start):
        i_pop = jnp.broadcast_to(start[None], (int(p), ndim))
        rep_results = jax.vmap(
            lambda r: run_directed_evolution(
                r, i_pop, selection_function, mutation_function,
                fitness_function=fitness_function, num_steps=25
            )[1]
        )(rng_seeds)
        return rep_results['fitness'].mean(axis=-1)   # (10, 25)

    vmapped_run = jax.jit(jax.vmap(run_from_start))

    pad = (-total_starts) % batch_size
    if pad:
        padded = np.concatenate([start_array, start_array[:pad]], axis=0)
    else:
        padded = start_array
    batched = padded.reshape(-1, batch_size, ndim)

    results = []
    for i, starts in enumerate(tqdm.tqdm(batched)):
        batch_result = vmapped_run(jnp.array(starts))
        if pad and i == len(batched) - 1:
            batch_result = batch_result[:batch_size - pad]
        results.append(np.array(batch_result))

    combined = np.concatenate(results, axis=0)   # (total_starts, 10, 25)
    return combined.reshape(-1, 100, 10, 25)


# ---------------------------------------------------------------------------
# Load landscapes
# ---------------------------------------------------------------------------

print("Loading landscapes...")
with open(os.path.join(landscape_dir, 'GB1_landscape_array.pkl'), 'rb') as f:
    GB1 = pickle.load(f)
with open(os.path.join(landscape_dir, 'TrpB_landscape_array.pkl'), 'rb') as f:
    TrpB = pickle.load(f)
with open(os.path.join(landscape_dir, 'TEV_landscape_array.pkl'), 'rb') as f:
    TEV = pickle.load(f)
with open(os.path.join(landscape_dir, 'E3_landscape_array.pkl'), 'rb') as f:
    E3 = pickle.load(f)

landscapes = {
    'gb1':   (GB1,  GB1.ndim),
    'trpb':  (TrpB, TrpB.ndim),
    'tev':   (TEV,  TEV.ndim),
    'pard3': (E3,   E3.ndim),
}

BASE_MUT_RATE = 0.1

# ---------------------------------------------------------------------------
# Run AA-space simulation for all landscapes
# ---------------------------------------------------------------------------

for ld_name, (ld, n_aa) in landscapes.items():
    print(f"\n  Landscape: {ld_name.upper()}  ({n_aa} AA sites)")

    ld_jnp    = jnp.array(ld)
    site_rate = BASE_MUT_RATE / n_aa          # per-AA-site rate (same total as codon script)
    fit_fn    = build_empirical_landscape_function(ld_jnp)
    starts    = all_start_locs(ld)            # (total_starts, n_aa)  — no codon conversion

    mut_fn = build_mutation_function(site_rate, num_options=20)

    # ParD3 uses smaller population to match existing all-starts script
    p = 60 if ld_name == 'pard3' else 2500

    results = generate_decay_curve(starts, fit_fn, mut_fn, p=p, batch_size=3000)

    out_path = os.path.join(
        slide_data_dir,
        f"decay_curves_{ld_name}_aa_debug_m0.1_all_starts.pkl"
    )
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Saved {results.shape} → {out_path}")

print("\nDone.")
