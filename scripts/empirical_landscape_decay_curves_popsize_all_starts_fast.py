import sys
import os

# Get the parent directory of this script
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

slide_data_dir = "/home/jess/Documents/SLIDE_data"

def all_start_locs(ld):
    return np.indices(ld.shape).reshape(len(ld.shape), -1).T



def generate_decay_curve(ld, m, p=2500, batch_size=2000):
    ld = jnp.array(ld)
    all_starts = all_start_locs(ld)  # shape (total_starts, ndim)
    total_starts = all_starts.shape[0]
    ndim = all_starts.shape[1]

    print(all_starts[0])
    print(all_starts[-1])
    print(all_starts.size)

    # Build functions once
    sel_params = {'threshold': 0.0, 'base_chance': 1.0}
    fitness_function   = build_empirical_landscape_function(ld)
    mutation_function  = build_mutation_function(m, 20)
    selection_function = build_selection_function(slct.base_chance_threshold_select, sel_params)

    # RNG seeds for replicates
    #_r1, _r2, r3 = jr.split(jr.PRNGKey(42), 3)
    #rng_seeds = jr.split(r3, 10)

    base_key = jr.PRNGKey(42)
    start_keys = jr.split(base_key, total_starts)

    # Per-start function
    """
    def run_from_start(start):
        i_pop = jnp.broadcast_to(start[None], (int(p), ndim))
        rep_results = jax.vmap(
            lambda r: run_directed_evolution(
                r, i_pop, selection_function, mutation_function,
                fitness_function=fitness_function, num_steps=25
            )[1]
        )(rng_seeds)
        return rep_results['fitness'].mean(axis=-1)   # (10, 25)
    """
    
    def run_from_start(start_key, start):

        rep_keys = jr.split(start_key, 10)

        i_pop = jnp.broadcast_to(start[None], (int(p), ndim))

        rep_results = jax.vmap(
            lambda r: run_directed_evolution(
                r, i_pop, selection_function, mutation_function,
                fitness_function=fitness_function, num_steps=25
            )[1]
        )(rep_keys)

        return rep_results['fitness'].mean(axis=-1)

    vmapped_run = jax.jit(jax.vmap(run_from_start))  # compile once

    results = []
    # Slice starts into batches
    for i in tqdm.tqdm(range(0, total_starts, batch_size)):
        batch_starts = all_starts[i:i+batch_size]
        batch_keys = start_keys[i:i+batch_size]

        batch_result = vmapped_run(jnp.array(batch_keys), jnp.array(batch_starts))
        #batch_starts = all_starts[i:i+batch_size]
        #batch_result = vmapped_run(jnp.array(batch_starts))
        results.append(batch_result)

    combined = np.array(jnp.concatenate(results, axis=0))  # (total_starts, 10, 25)
    return combined.reshape(-1, 100, 10, 25)

## Loading landscapes

landscape_dir = os.path.join(parent_dir, 'landscape_arrays')

with open(os.path.join(landscape_dir, 'GB1_landscape_array.pkl'), 'rb') as f:
    GB1 = pickle.load(f)

with open(os.path.join(landscape_dir, 'TrpB_landscape_array.pkl'), 'rb') as f:
    TrpB = pickle.load(f)

with open(os.path.join(landscape_dir, 'TEV_landscape_array.pkl'), 'rb') as f:
    TEV = pickle.load(f)

with open(os.path.join(landscape_dir, 'E3_landscape_array.pkl'), 'rb') as f:
    E3 = pickle.load(f)


GB1_results = []
TrpB_results = []
TEV_results = []
ParD3_results = []

popsizes = np.logspace(np.log10(5), np.log10(1000), num=10)[::-1]
print(popsizes)

for popsize in popsizes:

    print(popsize)

    # GB1

    extra_run_str = "2" # Allows saving result to a different place, if desired. 
    sub_folder = "empirical_decay_curves/"
    sub_folder = ""
    print('Generating curves for GB1...')
    ld = GB1

    m = 0.1
    results = generate_decay_curve(ld = ld, m = m/4, p=popsize)
    GB1_results.append(results)

    # TrpB

    print('Generating curves for TrpB...')
    ld = TrpB

    m = 0.1
    results = generate_decay_curve(ld = ld, m = m/4, p=popsize)
    TrpB_results.append(results)

    # TEV

    print('Generating curves for TEV...')
    ld = TEV

    m = 0.1
    results = generate_decay_curve(ld = ld, m = m/4,p=popsize)
    TEV_results.append(results)

    # ParD3

    print('Generating curves for ParD3...')
    ld = E3

    m = 0.1
    results = generate_decay_curve(ld = ld, m = m/3, p=popsize/20)
    ParD3_results.append(results)


with open(os.path.join(slide_data_dir, f"decay_curves_GB1_m0.1_multistart_all_starts_multi_popsize4.pkl"), "wb") as f:
    pickle.dump(np.array(GB1_results[::-1]), f)

with open(os.path.join(slide_data_dir, f"decay_curves_TrpB_m0.1_multistart_all_starts_multi_popsize4.pkl"), "wb") as f:
    pickle.dump(np.array(TrpB_results[::-1]), f)

with open(os.path.join(slide_data_dir, f"decay_curves_TEV_m0.1_multistart_all_starts_multi_popsize4.pkl"), "wb") as f:
    pickle.dump(np.array(TEV_results[::-1]), f)

with open(os.path.join(slide_data_dir, f"decay_curves_ParD3_m0.1_multistart_all_starts_multi_popsize4.pkl"), "wb") as f:
    pickle.dump(np.array(ParD3_results[::-1]), f)