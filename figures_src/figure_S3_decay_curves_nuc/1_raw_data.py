"""
Figure S3 — Step 1: nucleotide-space decay curve simulations (75 steps).

Runs all-starts directed evolution for each landscape × mutation model
combination in nucleotide (codon-triplet) space.

Output
------
  SLIDE_data/decay_curves_{landscape}_{model}_m0.1_all_starts_75steps.pkl

  landscapes : gb1, trpb, tev, pard3
  models     : nuc_uniform, nuc_h_sapiens_sym, nuc_h_sapiens, nuc_e_coli

  Each file shape: (-1, 100, 10, 75)

Usage
-----
  python figures_src/figure_S3_decay_curves_nuc/1_raw_data.py

Note: requires a GPU (28 GB recommended). Originally from
scripts/empirical_landscape_decay_curves_codon_fast_75steps.py.

Shared raw data: also used by Figure 6 (graph_mutation_model.pdf).
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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
landscape_dir  = os.path.join(parent_dir, 'landscape_arrays')
matrix_dir     = os.path.join(parent_dir, 'other_data')

NUM_STEPS  = 75
BATCH_SIZE = 1000
BASE_MUT_RATE = 0.1


def all_start_locs(ld):
    return np.indices(ld.shape).reshape(len(ld.shape), -1).T


def aa_starts_to_codon(aa_starts):
    triplets = np.array(INVERSE_CODON_MAPPER)[aa_starts]
    return triplets.reshape(aa_starts.shape[0], -1).astype(np.int32)


def generate_decay_curve(start_array, fitness_function, mutation_function,
                         p=2500, batch_size=BATCH_SIZE):
    start_array = np.array(start_array)
    total_starts, ndim = start_array.shape

    sel_params         = {'threshold': 0.0, 'base_chance': 1.0}
    selection_function = build_selection_function(slct.base_chance_threshold_select, sel_params)
    rng_seeds          = jr.split(jr.split(jr.PRNGKey(42), 3)[2], 10)

    def run_from_start(start):
        i_pop = jnp.broadcast_to(start[None], (int(p), ndim))
        rep_results = jax.vmap(
            lambda r: run_directed_evolution(r, i_pop, selection_function,
                                              mutation_function,
                                              fitness_function=fitness_function,
                                              num_steps=NUM_STEPS)[1]
        )(rng_seeds)
        return rep_results['fitness'].mean(axis=-1)

    vmapped_run = jax.jit(jax.vmap(run_from_start))

    pad = (-total_starts) % batch_size
    padded  = np.concatenate([start_array, start_array[:pad]], axis=0) if pad else start_array
    batched = padded.reshape(-1, batch_size, ndim)

    results = []
    for i, starts in enumerate(tqdm.tqdm(batched)):
        batch_result = vmapped_run(jnp.array(starts))
        if pad and i == len(batched) - 1:
            batch_result = batch_result[:batch_size - pad]
        results.append(np.array(batch_result))

    combined = np.concatenate(results, axis=0)
    return combined.reshape(-1, 100, 10, NUM_STEPS)


print('Loading landscapes...')
landscapes = {}
for name, fname in [('gb1', 'GB1_landscape_array.pkl'), ('trpb', 'TrpB_landscape_array.pkl'),
                    ('tev', 'TEV_landscape_array.pkl'),  ('pard3', 'E3_landscape_array.pkl')]:
    with open(os.path.join(landscape_dir, fname), 'rb') as f:
        ld = pickle.load(f)
    landscapes[name] = (ld, ld.ndim, ld.ndim * 3)

print('Loading mutation matrices...')
h_sapiens_raw = np.load(os.path.join(matrix_dir, 'normed_h_sapiens_matrix.npy'))
e_coli_raw    = np.load(os.path.join(matrix_dir, 'normed_e_coli_matrix.npy'))
h_sapiens_sym = (h_sapiens_raw + h_sapiens_raw.T) / 2
h_sapiens_sym = h_sapiens_sym / h_sapiens_sym.sum(axis=1, keepdims=True)

MUTATION_MODELS = [
    ('nuc_uniform',       'uniform', None),
    ('nuc_h_sapiens_sym', 'custom',  h_sapiens_sym),
    ('nuc_h_sapiens',     'custom',  h_sapiens_raw),
    ('nuc_e_coli',        'custom',  e_coli_raw),
]

for model_name, mut_type, mut_matrix in MUTATION_MODELS:
    print(f'\n{"="*60}\nMutation model: {model_name}\n{"="*60}')
    for ld_name, (ld, n_aa, n_nuc) in landscapes.items():
        out_path = os.path.join(slide_data_dir,
                                f'decay_curves_{ld_name}_{model_name}_m0.1_all_starts_75steps.pkl')
        if os.path.exists(out_path):
            print(f'  {ld_name}: already exists, skipping')
            continue

        print(f'  {ld_name} ({n_aa} AA → {n_nuc} nt)')
        ld_jnp    = jnp.array(ld)
        site_rate = BASE_MUT_RATE / n_nuc
        fit_fn    = get_pre_defined_landscape_function_with_codon(ld_jnp)
        starts    = aa_starts_to_codon(all_start_locs(ld))
        mut_fn    = (build_mutation_function(site_rate, num_options=4)
                     if mut_type == 'uniform'
                     else build_custom_mutation_function(site_rate, mut_matrix, A=4))

        results = generate_decay_curve(starts, fit_fn, mut_fn)
        with open(out_path, 'wb') as f:
            pickle.dump(results, f)
        print(f'  Saved {results.shape} → {out_path}')

print('\nDone.')
