"""
Regenerate ParD3 nuc_uniform decay curves with 20 seeds (instead of 10).
Output shape: (-1, 100, 20, 25)
Output file:  SLIDE_data/decay_curves_pard3_nuc_uniform_m0.1_all_starts_20seeds.pkl
"""

import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import pickle
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import tqdm
from direvo_functions import (
    get_pre_defined_landscape_function_with_codon,
    build_mutation_function,
    build_selection_function,
    run_directed_evolution,
    INVERSE_CODON_MAPPER,
)
import selection_function_library as slct
from slide_config import get_slide_data_dir

slide_data_dir = str(get_slide_data_dir())
landscape_dir  = os.path.join(parent_dir, 'landscape_arrays')

N_SEEDS    = 20
BATCH_SIZE = 500
P          = 2500
MUT_RATE   = 0.1

with open(os.path.join(landscape_dir, 'E3_landscape_array.pkl'), 'rb') as f:
    E3 = pickle.load(f)

n_aa  = E3.ndim       # 3
n_nuc = n_aa * 3      # 9
site_rate = MUT_RATE / n_nuc

ld_jnp  = jnp.array(E3)
fit_fn  = get_pre_defined_landscape_function_with_codon(ld_jnp)
mut_fn  = build_mutation_function(site_rate, num_options=4)

sel_params = {'threshold': 0.0, 'base_chance': 1.0}
sel_fn     = build_selection_function(slct.base_chance_threshold_select, sel_params)

_r1, _r2, r3 = jr.split(jr.PRNGKey(42), 3)
rng_seeds = jr.split(r3, N_SEEDS)

# All AA starting points → convert to codon starts
aa_starts = np.indices(E3.shape).reshape(n_aa, -1).T        # (8000, 3)
triplets  = np.array(INVERSE_CODON_MAPPER)[aa_starts]        # (8000, 3, 3)
starts    = triplets.reshape(aa_starts.shape[0], -1).astype(np.int32)   # (8000, 9)

def run_from_start(start):
    i_pop = jnp.broadcast_to(start[None], (P, n_nuc))
    rep_results = jax.vmap(
        lambda r: run_directed_evolution(
            r, i_pop, sel_fn, mut_fn,
            fitness_function=fit_fn, num_steps=25
        )[1]
    )(rng_seeds)
    return rep_results['fitness'].mean(axis=-1)   # (N_SEEDS, 25)

vmapped_run = jax.jit(jax.vmap(run_from_start))

total_starts = starts.shape[0]
pad = (-total_starts) % BATCH_SIZE
padded  = np.concatenate([starts, starts[:pad]], axis=0) if pad else starts
batched = padded.reshape(-1, BATCH_SIZE, n_nuc)

results = []
for i, batch in enumerate(tqdm.tqdm(batched, desc='ParD3 nuc_uniform 20seeds')):
    out = vmapped_run(jnp.array(batch))
    if pad and i == len(batched) - 1:
        out = out[:BATCH_SIZE - pad]
    results.append(np.array(out))

combined = np.concatenate(results, axis=0)          # (8000, 20, 25)
combined = combined.reshape(-1, 100, N_SEEDS, 25)   # (-1, 100, 20, 25)

out_path = os.path.join(slide_data_dir,
    'decay_curves_pard3_nuc_uniform_m0.1_all_starts_20seeds.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(combined, f)
print(f'Saved {combined.shape} → {out_path}')
