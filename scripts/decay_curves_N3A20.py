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
        mutation_function = build_mutation_function(mut_chance, 20)
 
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

Ns = np.array([3]*3)
Ks = np.arange(3)
NKs = list(zip(Ns,Ks))

def uniform_start_locs(num=10000):
    flat_indexes = np.round(np.linspace(0, 160000/20-1, num)).astype(int)
    indexes = np.array([np.unravel_index(i, (20,20,20)) for i in flat_indexes])
    return indexes

m=0.1
p=1200

all_starts = np.array(uniform_start_locs(100))
all_starts = all_starts.reshape(10,10,3)

NK_results = []

for nks in NKs:

    results = []

    for starts_subset in tqdm.tqdm(all_starts):

        def single_rep(start):

            params = {'threshold': 0.0, 'base_chance' : 1.0}
            run = directedEvolution(jr.PRNGKey(0),
                                    N = int(nks[0]),
                                    K=int(nks[1]),
                                    selection_strategy=slct.base_chance_threshold_select,
                                    selection_params = params,
                                    popsize=int(p),
                                    mut_chance=m/int(nks[0]),
                                    define_i_pop=jnp.array([start]*int(p)),
                                    num_steps=25,
                                    num_reps=10,
                                    pre_optimisation_steps=0,
                                    average=False)

            #split_results.append(run['fitness'].max(axis=2).mean(axis=0)[-1])
            return run['fitness'].mean(axis=-1)

        results.append(jax.vmap(single_rep)(starts_subset))

    NK_results.append(results)

print(np.array(NK_results).shape)

with open(os.path.join(slide_data_dir, "N3A20_decay_curves.pkl"), "wb") as f:
    pickle.dump(np.array(results), f)