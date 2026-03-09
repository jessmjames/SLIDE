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

# Define the SLIDE_data path
slide_data_dir = "/home/jess/Documents/SLIDE_data"
def directedEvolution(rng,
                      selection_strategy, 
                      selection_params, 
                      empirical = False, 
                      N = None, 
                      K = None, 
                      landscape = None, 
                      popsize=100, 
                      mut_chance=0.01, 
                      num_steps=50, 
                      num_reps=10, 
                      define_i_pop=None, 
                      pre_optimisation_steps=0,
                      average=True):
    
 
    r1,r2,r3=jr.split(rng,3)

    # Get initial population.
    if define_i_pop == None:
        i_pop = jnp.array([jr.randint(r1, (N,), 0, 2)]*popsize)
    else:
        i_pop = define_i_pop
 
    # Function for evaluating fitness.
    if empirical:
        fitness_function = build_empirical_landscape_function(landscape)
        mutation_function = build_mutation_function(mut_chance, 20)
    else:
        fitness_function = build_NK_landscape_function(r2, N, K)
        mutation_function = build_mutation_function(mut_chance, 2)
 
    # Define selection function.
    selection_function = build_selection_function(
        selection_strategy, selection_params)
    
    if pre_optimisation_steps!= 0:

        pre_op_selection_function=build_selection_function(slct.base_chance_threshold_select, {'base_chance':0.0, 'threshold':0.95})


        pre_op = run_directed_evolution(r3, i_pop=i_pop, 
                               selection_function=pre_op_selection_function, 
                               mutation_function=mutation_function, 
                               fitness_function=fitness_function, 
                               num_steps=pre_optimisation_steps)[1]
 
        i_pop = pre_op['pop'][-1]

    # Bringing it all together.
    vmapped_run = jax.jit(jax.vmap(lambda r: run_directed_evolution(
        r, i_pop, selection_function, mutation_function, fitness_function=fitness_function, num_steps=num_steps)[1]))
    
    # The array of seeds we will take as input.
    rng_seeds = jr.split(r3, num_reps)
    results = vmapped_run(rng_seeds)
 
    return results

def percentile_start_locs(ld, num=10, perc = 98):
    percentile_value = np.percentile(ld.flatten(), perc)
    diffs = np.abs(ld.flatten() - percentile_value)
    closest_indices_flat = np.argpartition(diffs, num)[:num]
    closest_indices_multi = np.unravel_index(closest_indices_flat, ld.shape)
    indexes = np.array(list(zip(*closest_indices_multi)))
    return indexes

def uniform_start_locs(ld, num=10000):
    flat_ld = ld.flatten()
    flat_indexes = np.round(np.linspace(0, flat_ld.shape[0]-1, num)).astype(int)
    indexes = np.array([np.unravel_index(i, ld.shape) for i in flat_indexes])
    return indexes

def all_start_locs(ld):
    return np.indices(ld.shape).reshape(len(ld.shape), -1).T

def get_WT_starts(arr, n_points=10, percentile=99):
    flat = arr.ravel()
    threshold = np.percentile(flat, percentile)
    top_mask = flat >= threshold
    top_indices = np.nonzero(top_mask)[0]
    distances = flat[top_indices] - threshold
    closest = np.argsort(distances)[:n_points]
    flat_indices = top_indices[closest]
    indices_4d = np.column_stack(np.unravel_index(flat_indices, arr.shape))

    return indices_4d

## Loading landscapes

with open('../landscape_arrays/GB1_landscape_array.pkl', 'rb') as f:
    GB1 = pickle.load(f)

with open('../landscape_arrays/TrpB_landscape_array.pkl', 'rb') as f:
    TrpB = pickle.load(f)

with open('../landscape_arrays/TEV_landscape_array.pkl', 'rb') as f:
    TEV = pickle.load(f)

with open('../landscape_arrays/E3_landscape_array.pkl', 'rb') as f:
    E3 = pickle.load(f)

def generate_decay_curve(landscape, mut, popsize=2500, num_steps=25, num_reps=10, seed=42):

    landscape = jnp.array(landscape)

    results = []
    if landscape.shape == (20,20,20,20):
        all_starts = all_start_locs(landscape).reshape(1600,100,4)
    else:
        all_starts = all_start_locs(landscape).reshape(80,100,3)

    print(all_starts[0][0])
    print(all_starts[-1][-1])
    print(len(all_starts.flatten()))

    for starts in tqdm.tqdm(all_starts):
        def single_rep(start):

            params = {'threshold': 0.0, 'base_chance' : 1.0}
            run = directedEvolution(jr.PRNGKey(seed),
                                    selection_strategy=slct.base_chance_threshold_select,
                                    selection_params = params,
                                    popsize=int(popsize),
                                    mut_chance=mut,
                                    num_steps=num_steps,
                                    num_reps=num_reps,
                                    pre_optimisation_steps=0,
                                    define_i_pop=jnp.array([start]*int(popsize)),
                                    empirical=True,
                                    landscape=landscape,
                                    average=False)

            #split_results.append(run['fitness'].max(axis=2).mean(axis=0)[-1])
            return run['fitness'].mean(axis=-1)

        results.append(jax.vmap(single_rep)(starts))

    return(results)

# PARAMETERS
MUT = 0.1
NUM_STEPS = 25
POPSIZE = 2500
NUM_REPS = 10
SEED = 10

# GB1

print('Generating curves for GB1...')
ld = GB1
results = generate_decay_curve(landscape = ld, mut = MUT/4, popsize=POPSIZE, num_reps=NUM_REPS, num_steps=NUM_STEPS, seed=SEED)
with open(os.path.join(slide_data_dir, f"decay_curves_gb1_m0.1_all_starts_IK_seed{SEED}.pkl"), "wb") as f:
    pickle.dump(np.array(results), f)

# TrpB

print('Generating curves for TrpB...')
ld = TrpB
results = generate_decay_curve(landscape = ld, mut = MUT/4, popsize=POPSIZE, num_reps=NUM_REPS, num_steps=NUM_STEPS, seed=SEED)
with open(os.path.join(slide_data_dir, f"decay_curves_trpb_m0.1_all_starts_IK_seed{SEED}.pkl"), "wb") as f:
    pickle.dump(np.array(results), f)

# TEV

print('Generating curves for TEV...')
ld = TEV
results = generate_decay_curve(landscape = ld, mut = MUT/4, popsize=POPSIZE, num_reps=NUM_REPS, num_steps=NUM_STEPS, seed=SEED)
with open(os.path.join(slide_data_dir, f"decay_curves_tev_m0.1_all_starts_IK_seed{SEED}.pkl"), "wb") as f:
    pickle.dump(np.array(results), f)

# ParD3

print('Generating curves for ParD3...')
ld = E3
results = generate_decay_curve(landscape = ld, mut = MUT/3, popsize=int(POPSIZE/20), num_reps=NUM_REPS, num_steps=NUM_STEPS, seed=SEED)
with open(os.path.join(slide_data_dir, f"decay_curves_pard3_m0.1_all_starts_IK_seed{SEED}.pkl"), "wb") as f:
    pickle.dump(np.array(results), f)