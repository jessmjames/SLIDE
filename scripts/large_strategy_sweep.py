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

def NK_grid(N_range, num_samples=10):
    N = jnp.linspace(N_range[0], N_range[1], num=num_samples)
    K = jnp.array([jnp.linspace(1, i, num_samples)
                  for i in N]).reshape(num_samples, num_samples)
    N = jnp.repeat(N, num_samples).reshape(num_samples, num_samples)
    return N, K

N_grid, K_grid = NK_grid([10, 50])

Ns, Ks = N_grid.flatten(), K_grid.flatten()
Ns = jnp.flip(Ns)
Ks = jnp.flip(Ks)

thresholds, base_chances = base_chance_threshold_fixed_prop([0,0.19], 0.2, 7)
splits = [24,20,16,12,8,4,1]
m=0.1
p=1200
NKs = list(zip(Ns,Ks))

reps = 300

sweep_results = np.zeros((len(NKs),len(splits),len(base_chances), reps))
total_iterations = len(NKs) * len(splits) * len(base_chances)

with tqdm.tqdm(total=total_iterations, desc="Overall Progress") as pbar:
    for i, nks in enumerate(NKs):
        for ii, s in enumerate(splits):
            for iii, (bc, th) in enumerate(zip(base_chances, thresholds)):
                    
                rep_rngs = jr.split(jr.PRNGKey(42),reps)

                def single_rep(rng_rep):

                    split_rngs = jr.split(rng_rep, s)
                
                    def single_s(s_rng):
                        params = {'threshold': th, 'base_chance' : bc}
                        run = directedEvolution(s_rng,
                                                rng_rep,
                                                N = int(nks[0]),
                                                K=int(nks[1]),
                                                selection_strategy=slct.base_chance_threshold_select,
                                                selection_params = params,
                                                popsize=int(p/s),
                                                mut_chance=m/int(nks[0]),
                                                num_steps=25,
                                                num_reps=1,
                                                average=True)
                
                        #split_results.append(run['fitness'].max(axis=2).mean(axis=0)[-1])
                        return run['fitness'][:,:,-1].max(axis=1).mean()
                
                    split_results = jax.vmap(single_s)(split_rngs)

                    return jnp.array(split_results).max()
            
                repeat_results = jax.vmap(single_rep)(rep_rngs)
        
                sweep_results[i,ii,iii] = np.array(repeat_results)
        
                pbar.update(1)

file_path = os.path.join(slide_data_dir, "large_strategy_sweep.pkl")
with open(file_path, "wb") as f:
    pickle.dump(sweep_results, f)