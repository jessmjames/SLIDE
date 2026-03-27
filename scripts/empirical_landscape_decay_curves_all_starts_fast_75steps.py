"""
AA-uniform decay curves, 75 mutational steps.
Same as empirical_landscape_decay_curves_all_starts_fast.py but num_steps=75
and batch_size bumped aggressively (28 GB GPU).
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
from direvo_functions import *
import selection_function_library as slct
import tqdm
from slide_config import get_slide_data_dir

slide_data_dir = str(get_slide_data_dir())
landscape_dir  = os.path.join(parent_dir, 'landscape_arrays')

NUM_STEPS  = 75
BATCH_SIZE = 1000   # push it — 28 GB GPU

def all_start_locs(ld):
    return np.indices(ld.shape).reshape(len(ld.shape), -1).T


def generate_decay_curve(ld, m, p=2500, batch_size=BATCH_SIZE):
    ld = jnp.array(ld)
    all_starts   = all_start_locs(ld)
    total_starts = all_starts.shape[0]
    ndim         = all_starts.shape[1]

    sel_params         = {'threshold': 0.0, 'base_chance': 1.0}
    fitness_function   = build_empirical_landscape_function(ld)
    mutation_function  = build_mutation_function(m, 20)
    selection_function = build_selection_function(slct.base_chance_threshold_select, sel_params)

    _r1, _r2, r3 = jr.split(jr.PRNGKey(42), 3)
    rng_seeds    = jr.split(r3, 10)

    def run_from_start(start):
        i_pop = jnp.broadcast_to(start[None], (int(p), ndim))
        rep_results = jax.vmap(
            lambda r: run_directed_evolution(
                r, i_pop, selection_function, mutation_function,
                fitness_function=fitness_function, num_steps=NUM_STEPS
            )[1]
        )(rng_seeds)
        return rep_results['fitness'].mean(axis=-1)   # (10, NUM_STEPS)

    vmapped_run = jax.jit(jax.vmap(run_from_start))

    pad = (-total_starts) % batch_size
    if pad:
        padded_starts = np.concatenate([all_starts, all_starts[:pad]], axis=0)
    else:
        padded_starts = all_starts
    batched_starts = padded_starts.reshape(-1, batch_size, ndim)

    results = []
    for i, starts in enumerate(tqdm.tqdm(batched_starts)):
        batch_result = vmapped_run(jnp.array(starts))
        if pad and i == len(batched_starts) - 1:
            batch_result = batch_result[:batch_size - pad]
        results.append(np.array(batch_result))

    combined = np.concatenate(results, axis=0)          # (total_starts, 10, NUM_STEPS)
    return combined.reshape(-1, 100, 10, NUM_STEPS)      # (-1, 100, 10, NUM_STEPS)


with open(os.path.join(landscape_dir, 'GB1_landscape_array.pkl'), 'rb') as f:
    GB1 = pickle.load(f)
with open(os.path.join(landscape_dir, 'TrpB_landscape_array.pkl'), 'rb') as f:
    TrpB = pickle.load(f)
with open(os.path.join(landscape_dir, 'TEV_landscape_array.pkl'), 'rb') as f:
    TEV = pickle.load(f)
with open(os.path.join(landscape_dir, 'E3_landscape_array.pkl'), 'rb') as f:
    E3 = pickle.load(f)


print('Generating GB1...')
results = generate_decay_curve(ld=GB1, m=0.1/4)
with open(os.path.join(slide_data_dir, 'decay_curves_gb1_aa_uniform_m0.1_all_starts_75steps.pkl'), 'wb') as f:
    pickle.dump(results, f)
print(f'  saved {results.shape}')

print('Generating TrpB...')
results = generate_decay_curve(ld=TrpB, m=0.1/4)
with open(os.path.join(slide_data_dir, 'decay_curves_trpb_aa_uniform_m0.1_all_starts_75steps.pkl'), 'wb') as f:
    pickle.dump(results, f)
print(f'  saved {results.shape}')

print('Generating TEV...')
results = generate_decay_curve(ld=TEV, m=0.1/4)
with open(os.path.join(slide_data_dir, 'decay_curves_tev_aa_uniform_m0.1_all_starts_75steps.pkl'), 'wb') as f:
    pickle.dump(results, f)
print(f'  saved {results.shape}')

print('Generating ParD3...')
results = generate_decay_curve(ld=E3, m=0.1/3, p=60)
with open(os.path.join(slide_data_dir, 'decay_curves_pard3_aa_uniform_m0.1_all_starts_75steps.pkl'), 'wb') as f:
    pickle.dump(results, f)
print(f'  saved {results.shape}')

print('Done.')
