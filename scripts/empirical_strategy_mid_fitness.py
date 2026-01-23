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

def get_WT_starts(arr, n_points=10, percentile=99):
    # Flatten array
    flat = arr.ravel()

    # Compute percentile threshold (start of top 1%)
    threshold = np.percentile(flat, percentile)

    # Indices of values in the top percentile
    top_mask = flat >= threshold
    top_indices = np.nonzero(top_mask)[0]

    # Distance from the threshold
    distances = flat[top_indices] - threshold

    # Get indices of the n_points closest to threshold
    closest = np.argsort(distances)[:n_points]

    # Convert back to flat indices
    flat_indices = top_indices[closest]

    # Convert to 4D indices
    indices_4d = np.column_stack(np.unravel_index(flat_indices, arr.shape))

    return indices_4d
 
 
def directed_evo(s_rng,
                      rng_rep,
                      selection_strategy,
                      selection_params,
                      fitness_function=None,
                      num_alleles=2,
                      N = None,
                      popsize=100,
                      mut_chance=0.1,
                      num_steps=50,
                      num_reps=10,
                      define_i_pop=None):
   
 
    # Get initial population.
    if define_i_pop == None:
        i_pop = jnp.array([jr.randint(rng_rep, (N,), 0, num_alleles)]*popsize)
    else:
        i_pop = define_i_pop[None,:] * jnp.ones((popsize, N), dtype=jnp.int32)
 
    mutation_function = build_mutation_function(mut_chance, num_alleles)
 
    # Define selection function.
    selection_function = build_selection_function(
        selection_strategy, selection_params)
 
    # Bringing it all together.
    vmapped_run = jax.jit(jax.vmap(lambda r: run_directed_evolution(
        r, i_pop, selection_function, mutation_function, fitness_function=fitness_function, num_steps=num_steps)[1]))
   
    # The array of seeds we will take as input.
    rng_seeds = jr.split(s_rng, num_reps)
    results = vmapped_run(rng_seeds)
 
    return results
 
 
def get_single_sweep_empirical(base_chances, thresholds, splits, popsize, start, num_alleles = 2, num_reps = 10):
 
    def to_return(rng, landscape_arr):
        fitness_function = build_empirical_landscape_function(landscape_arr)
 
        N = len(landscape_arr.shape)
        def do_single_split(rng, split_size):
           
            def do_threshold(rng, base_chance, threshold):
                params = {'threshold': threshold, 'base_chance': base_chance}
                r1, r2 = jr.split(rng, 2)
                run = directed_evo(r1,
                                   r2,
                                   N=N,
                                   selection_strategy=slct.base_chance_threshold_select,
                                   fitness_function=fitness_function,
                                   num_alleles=num_alleles,
                                   define_i_pop=start[None,:] * jnp.ones((popsize, N), dtype=jnp.int32),
                                   selection_params=params,
                                   popsize=int(popsize/split_size),
                                   mut_chance=0.1/N,
                                   num_steps=25,
                                   num_reps=split_size
                                   )
                return run['fitness'][:,:,-1].max()
           
            vmap_threshold = jax.vmap(do_threshold, in_axes=(None, 0, 0))
 
            # Gets results, a fitness for each base chance.
            results = vmap_threshold(rng, base_chances, thresholds)
 
            return results
       
        rngs = jr.split(rng, num_reps)
 
        vmapped_do_split = jax.vmap(do_single_split, in_axes=(0, None))
 
        results_list = []
        for split_size in splits:
            results = vmapped_do_split(rngs, split_size)
            results_list.append(results)
 
        results = jnp.array(results_list)
        results = jnp.moveaxis(results, 0, -1)  # Move the split dimension to the last axis
        return results
   
    return to_return

with open('../landscape_arrays/GB1_landscape_array.pkl', 'rb') as f:
    GB1 = pickle.load(f)
 
GB1_starts = get_WT_starts(GB1)

thresholds, base_chances = base_chance_threshold_fixed_prop([0,0.19], 0.2, 7)
splits = [24,20,16,12,8,4,1]
m=0.1
p=1200

start_results = []
for start in GB1_starts:
    sweepy_empirical = get_single_sweep_empirical(base_chances, thresholds, splits, p, start, num_alleles=2, num_reps=10)
    start_results.append([sweepy_empirical(i, GB1) for i in jr.split(jr.PRNGKey(42), 10)])

with open(os.path.join(slide_data_dir, "GB1_strategy_sweep_mid_fitness.pkl"), "wb") as f:
    pickle.dump(np.array(start_results), f)