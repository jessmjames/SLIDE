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
from slide_config import get_slide_data_dir

slide_data_dir = str(get_slide_data_dir())

def all_start_locs(ld):
    return np.indices(ld.shape).reshape(len(ld.shape), -1).T


def generate_decay_curve(ld, m, p=2500, batch_size=1000):
    """
    Refactored version of generate_decay_curve that builds JIT/vmapped functions
    once and reuses them across all batches (no recompilation per batch).

    batch_size controls how many starting locations are vmapped at once.
    Reduce it if you run out of GPU memory; increase it for more parallelism.

    Output shape: (-1, 100, 10, 25) i.e. (n_chunks, 100, num_reps, num_steps).
    This matches the shape produced by the original script and is compatible
    with downstream notebook processing (cell 52: .mean(axis=2).reshape(-1,25)).
    """
    ld = jnp.array(ld)

    all_starts = all_start_locs(ld)          # shape (total_starts, ndim)
    total_starts = all_starts.shape[0]
    ndim = all_starts.shape[1]

    print(all_starts[0])
    print(all_starts[-1])
    print(all_starts.size)

    # Build all functions once — avoids re-tracing / re-compiling on every batch.
    sel_params = {'threshold': 0.0, 'base_chance': 1.0}
    fitness_function   = build_empirical_landscape_function(ld)
    mutation_function  = build_mutation_function(m, 20)
    selection_function = build_selection_function(slct.base_chance_threshold_select, sel_params)

    # Derive RNG seeds the same way directedEvolution(jr.PRNGKey(42), num_reps=10) does:
    #   r1, r2, r3 = jr.split(jr.PRNGKey(42), 3)
    #   rng_seeds  = jr.split(r3, num_reps)
    _r1, _r2, r3 = jr.split(jr.PRNGKey(42), 3)
    rng_seeds = jr.split(r3, 10)

    # Define per-start function with stable captures so JIT compiles once.
    def run_from_start(start):
        i_pop = jnp.broadcast_to(start[None], (int(p), ndim))
        rep_results = jax.vmap(
            lambda r: run_directed_evolution(
                r, i_pop, selection_function, mutation_function,
                fitness_function=fitness_function, num_steps=25
            )[1]
        )(rng_seeds)
        return rep_results['fitness'].mean(axis=-1)   # (num_reps, num_steps) = (10, 25)

    vmapped_run = jax.jit(jax.vmap(run_from_start))  # compiled once, reused every batch

    # Pad starts so total is divisible by batch_size, then batch.
    pad = (-total_starts) % batch_size
    if pad:
        padded_starts = np.concatenate([all_starts, all_starts[:pad]], axis=0)
    else:
        padded_starts = all_starts
    batched_starts = padded_starts.reshape(-1, batch_size, ndim)

    results = []
    for i, starts in enumerate(tqdm.tqdm(batched_starts)):
        batch_result = vmapped_run(jnp.array(starts))     # (batch_size, 10, 25)
        if pad and i == len(batched_starts) - 1:
            batch_result = batch_result[:batch_size - pad]
        results.append(batch_result)

    # Concatenate → (total_starts, 10, 25), then reshape to (-1, 100, 10, 25)
    # to match the original script's output shape for downstream compatibility.
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


# GB1

print('Generating curves for GB1...')
results = generate_decay_curve(ld=GB1, m=0.1/4)
with open(os.path.join(slide_data_dir, 'decay_curves_gb1_m0.1_all_starts.pkl'), 'wb') as f:
    pickle.dump(results, f)

# TrpB

print('Generating curves for TrpB...')
results = generate_decay_curve(ld=TrpB, m=0.1/4)
with open(os.path.join(slide_data_dir, 'decay_curves_trpb_m0.1_all_starts.pkl'), 'wb') as f:
    pickle.dump(results, f)

# TEV

print('Generating curves for TEV...')
results = generate_decay_curve(ld=TEV, m=0.1/4)
with open(os.path.join(slide_data_dir, 'decay_curves_tev_m0.1_all_starts.pkl'), 'wb') as f:
    pickle.dump(results, f)

# ParD3

print('Generating curves for ParD3...')
results = generate_decay_curve(ld=E3, m=0.1/3, p=60)
with open(os.path.join(slide_data_dir, 'decay_curves_pard3_m0.1_all_starts.pkl'), 'wb') as f:
    pickle.dump(results, f)
