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

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"

def directedEvolution(rng,
                      rng_other,
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
    rng_others = jr.split(rng_other, 12)

    # Get initial population.
    if define_i_pop == None:
        i_pop = jnp.array([jr.randint(rng_others[0], (N,), 0, 2)]*500)
    else:
        i_pop = define_i_pop
 
    # Function for evaluating fitness.
    if empirical:
        fitness_function = build_empirical_landscape_function(landscape)
        mutation_function = build_mutation_function(mut_chance, 20)
    else:
        fitness_function = build_NK_landscape_function(rng_others[1], N, K)
        mutation_function = build_mutation_function(mut_chance, 2)
 
    # Define selection function.
    selection_function = build_selection_function(
        selection_strategy, selection_params)
    
    if pre_optimisation_steps!= 0:

        pre_op_selection_function=build_selection_function(slct.base_chance_threshold_select, {'base_chance':0.0, 'threshold':0.95})
        mutation_function_pre_opt= build_mutation_function(0.6/N, 2)

        pre_op = run_directed_evolution(rng_others[2], i_pop=i_pop, 
                               selection_function=pre_op_selection_function, 
                               mutation_function=mutation_function_pre_opt, 
                               fitness_function=fitness_function, 
                               num_steps=pre_optimisation_steps)[1]
 
        i_pop = pre_op['pop'][-1][0][None,:]*jnp.ones(popsize)[:,None]

    # Bringing it all together.
    vmapped_run = jax.jit(jax.vmap(lambda r: run_directed_evolution(
        r, i_pop, selection_function, mutation_function, fitness_function=fitness_function, num_steps=num_steps)[1]))
    
    # The array of seeds we will take as input.
    rng_seeds = jr.split(r3, num_reps)
    results = vmapped_run(rng_seeds)
 
    return results

mut_rates=np.linspace(0.01,2,25)

reps = 25
total_iterations = len(mut_rates)

all_results = []

with tqdm.tqdm(total=total_iterations, desc="Overall Progress") as pbar:
    for i,m in enumerate(mut_rates):
                        
        rep_rngs = jr.split(jr.PRNGKey(42),reps)

        def single_rep(rng_rep):
    
            params = {'threshold': 0.0, 'base_chance' : 1.0}
            run = directedEvolution(rng_rep,
                                    jr.PRNGKey(10),
                                    N = 25,
                                    K=15,
                                    selection_strategy=slct.base_chance_threshold_select,
                                    selection_params = params,
                                    popsize=2000,
                                    mut_chance=m/25,
                                    num_steps=25,
                                    num_reps=20,
                                    pre_optimisation_steps=50,
                                    average=False)
    
            #split_results.append(run['fitness'].max(axis=2).mean(axis=0)[-1])
            return run['fitness'].mean(axis=-1)
    
        repeat_results = jax.vmap(single_rep)(rep_rngs)

        mut_results = np.array(repeat_results)
        pbar.update(1)

        all_results.append(mut_results)

all_results = np.array(all_results)

file_path = os.path.join(slide_data_dir, "mutation_rate_accuracy.pkl")
with open(file_path, "wb") as f:
    pickle.dump(all_results, f)