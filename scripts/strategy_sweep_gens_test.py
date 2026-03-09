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

def directedEvolution(s_rng,
                      rng_rep,
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
                      average=True):
   
 
    # Get initial population.
    if define_i_pop == None:
        i_pop = jnp.array([jr.randint(rng_rep, (N,), 0, 2)]*popsize)
    else:
        i_pop = define_i_pop
 
    # Function for evaluating fitness.
    if empirical:
        fitness_function = build_empirical_landscape_function(landscape)
        mutation_function = build_mutation_function(mut_chance, 20)
    else:
        fitness_function = build_NK_landscape_function(rng_rep, N, K)
        mutation_function = build_mutation_function(mut_chance, 2)
 
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
 

 
def get_multi_sweep_NK(base_chances, thresholds, splits, popsize, num_alleles = 2, num_reps = 10, num_landscapes = 12):
    def to_return(rng, N, K):
        def single_land(rng ):
 
            fitness_function = build_NK_landscape_function(rng, N, K)
            init_pop = None
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
                                    define_i_pop=init_pop,
                                    selection_params=params,
                                    popsize=int(popsize/split_size),
                                    mut_chance=0.1/N,
                                    num_steps=1000,
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
       
        rngs = jr.split(rng, num_landscapes)
        vmapped_single_land = jax.vmap(single_land, in_axes=(0))
 
        results = vmapped_single_land(rngs)
 
        mean_results = jnp.mean(results, axis=0)  # Average over landscapes
        return mean_results
 
    return jax.jit(to_return, static_argnums=(1,))
 

thresholds, base_chances = base_chance_threshold_fixed_prop([0,0.19], 0.2, 7)
splits = [24,20,16,12,8,4,1]
m=0.1
p=1200
sweepy_NK = get_multi_sweep_NK(base_chances, thresholds, splits, p, num_alleles=20, num_reps=10, num_landscapes=10)
results=[[sweepy_NK(ii, 10, i).mean(axis=0) for i in [1,4,8]] for ii in jr.split(jr.PRNGKey(42), 10)]

with open(os.path.join(slide_data_dir, "N10_strategy_sweep_1000_gens.pkl"), "wb") as f:
    pickle.dump(np.array(results), f)