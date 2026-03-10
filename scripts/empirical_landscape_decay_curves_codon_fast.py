"""
Fast nucleotide-space decay curve generation for 4 mutation models.

All models operate in nucleotide (codon-triplet) space: each amino-acid site
is represented as 3 nucleotides [U,C,A,G], so a 4-AA sequence → 12 nts,
a 3-AA sequence → 9 nts.  The fitness function maps nt sequences back to
amino-acid fitness via the codon look-up table.

The AA-uniform baseline (non-codon) is in empirical_landscape_decay_curves_all_starts_fast.py.

Mutation models (all 4-state nucleotide, A=4)
----------------------------------------------
1. nuc_uniform     : uniform nucleotide mutations
2. nuc_h_sapiens_sym : symmetric human nucleotide transition matrix
3. nuc_h_sapiens   : asymmetric human nucleotide transition matrix
4. nuc_e_coli      : asymmetric E. coli nucleotide transition matrix

Mutation rate: m / n_nuc  (same expected mutations per sequence per step)
  GB1   (4 AA → 12 nt): 0.1/12 ≈ 0.0083 per nt site
  TrpB/TEV/ParD3 (3 AA → 9 nt): 0.1/9 ≈ 0.0111 per nt site

Output shape: (-1, 100, 10, 25)
  Compatible with ruggedness_figures_data_processing.ipynb (.mean(axis=2).reshape(-1,25))

Output files (SLIDE_data/):
  decay_curves_{landscape}_{model}_m0.1_all_starts.pkl
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
    get_pre_defined_landscape_function_with_codon,
    build_mutation_function,
    build_custom_mutation_function,
    build_selection_function,
    run_directed_evolution,
    INVERSE_CODON_MAPPER,
)
import selection_function_library as slct
import tqdm
from slide_config import get_slide_data_dir

slide_data_dir = str(get_slide_data_dir())
landscape_dir = os.path.join(parent_dir, 'landscape_arrays')
matrix_dir = os.path.join(parent_dir, 'other_data')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def all_start_locs(ld):
    return np.indices(ld.shape).reshape(len(ld.shape), -1).T


def aa_starts_to_codon(aa_starts):
    """Convert AA index starts → nucleotide (codon triplet) starts.

    aa_starts : (N, n_aa)    AA indices (0-19)
    returns   : (N, n_aa*3)  nucleotide indices (0-3)
    """
    triplets = np.array(INVERSE_CODON_MAPPER)[aa_starts]   # (N, n_aa, 3)
    return triplets.reshape(aa_starts.shape[0], -1).astype(np.int32)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def generate_decay_curve(start_array, fitness_function, mutation_function,
                         p=2500, batch_size=500):
    """
    Run directed evolution from every row in start_array.

    Parameters
    ----------
    start_array      : (total_starts, ndim) array of starting locations
    fitness_function : JAX-jitted fitness function
    mutation_function: JAX mutation function
    p                : population size
    batch_size       : starts to vmap over simultaneously (tune for GPU memory)

    Returns
    -------
    np.ndarray of shape (-1, 100, 10, 25)
    """
    start_array = np.array(start_array)
    total_starts, ndim = start_array.shape

    sel_params = {'threshold': 0.0, 'base_chance': 1.0}
    selection_function = build_selection_function(slct.base_chance_threshold_select, sel_params)

    _r1, _r2, r3 = jr.split(jr.PRNGKey(42), 3)
    rng_seeds = jr.split(r3, 10)

    # Build and JIT once — reused across all batches.
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

    # Pad so total_starts is divisible by batch_size.
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
    return combined.reshape(-1, 100, 10, 25)      # (-1, 100, 10, 25)


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

# (landscape_array, n_aa_sites, n_nuc_sites) — infer n_aa from actual array shape
landscapes = {
    'gb1':   (GB1,  GB1.ndim,  GB1.ndim  * 3),   # (20,20,20,20) → 4 AA, 12 nt
    'trpb':  (TrpB, TrpB.ndim, TrpB.ndim * 3),   # (20,20,20,20) → 4 AA, 12 nt
    'tev':   (TEV,  TEV.ndim,  TEV.ndim  * 3),   # (20,20,20,20) → 4 AA, 12 nt
    'pard3': (E3,   E3.ndim,   E3.ndim   * 3),   # (20,20,20)    → 3 AA,  9 nt
}

# ---------------------------------------------------------------------------
# Load nucleotide transition matrices (all 4×4)
# ---------------------------------------------------------------------------

print("Loading mutation matrices...")
h_sapiens_raw = np.load(os.path.join(matrix_dir, 'normed_h_sapiens_matrix.npy'))
e_coli_raw    = np.load(os.path.join(matrix_dir, 'normed_e_coli_matrix.npy'))

# Symmetric version: average with transpose, renormalise rows.
h_sapiens_sym = (h_sapiens_raw + h_sapiens_raw.T) / 2
h_sapiens_sym = h_sapiens_sym / h_sapiens_sym.sum(axis=1, keepdims=True)

# ---------------------------------------------------------------------------
# Mutation models  (label, build_fn_kwargs)
# All operate in nucleotide space (A=4); rate is set per landscape below.
# ---------------------------------------------------------------------------

BASE_MUT_RATE = 0.1   # total expected mutations per sequence per step

MUTATION_MODELS = [
    ('nuc_uniform',      'uniform',  None),
    ('nuc_h_sapiens_sym','custom',   h_sapiens_sym),
    ('nuc_h_sapiens',    'custom',   h_sapiens_raw),
    ('nuc_e_coli',       'custom',   e_coli_raw),
]

# ---------------------------------------------------------------------------
# Run all models × all landscapes
# ---------------------------------------------------------------------------

for model_name, mut_type, mut_matrix in MUTATION_MODELS:
    print(f"\n{'='*60}")
    print(f"Mutation model: {model_name}")
    print(f"{'='*60}")

    for ld_name, (ld, n_aa, n_nuc) in landscapes.items():
        print(f"\n  Landscape: {ld_name.upper()}  ({n_aa} AA → {n_nuc} nt)")

        ld_jnp     = jnp.array(ld)
        site_rate  = BASE_MUT_RATE / n_nuc       # per-nucleotide-site rate
        fit_fn     = get_pre_defined_landscape_function_with_codon(ld_jnp)
        aa_starts  = all_start_locs(ld)
        starts     = aa_starts_to_codon(aa_starts)   # (total_starts, n_nuc)

        if mut_type == 'uniform':
            mut_fn = build_mutation_function(site_rate, num_options=4)
        else:
            mut_fn = build_custom_mutation_function(site_rate, mut_matrix, A=4)

        results = generate_decay_curve(starts, fit_fn, mut_fn, batch_size=500)

        out_path = os.path.join(
            slide_data_dir,
            f"decay_curves_{ld_name}_{model_name}_m0.1_all_starts.pkl"
        )
        with open(out_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"  Saved {results.shape} → {out_path}")

print("\nDone.")
