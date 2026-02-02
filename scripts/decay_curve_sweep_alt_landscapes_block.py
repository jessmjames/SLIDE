import sys
import os

# Get the parent directory of this script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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

slide_data_dir = "/home/jess/Documents/SLIDE_data"


def directedEvolution(
    s_rng,
    rng_rep,
    selection_strategy,
    selection_params,
    landscape_type=None,
    N=None,
    K=None,
    landscape=None,
    popsize=100,
    mut_chance=0.01,
    num_steps=50,
    num_reps=10,
    define_i_pop=None,
    average=True,
    noise_scale=0.2,
    num_blocks=5,
    pre_optimisation_steps=10,
):

    # Get initial population.
    if define_i_pop == None:
        i_pop = jnp.array([jr.randint(rng_rep, (N,), 0, 2)] * popsize)
    else:
        i_pop = define_i_pop

    # Function for evaluating fitness.
    # Just examples, these have extra parameters you might want to set.
    
    if landscape_type is None:
        raise ValueError("landscape_type must be specified!")
    elif landscape_type == "additive":
        fitness_function = build_additive_landscape_function(rng_rep, N, scale = 1.3)
    elif landscape_type == "NK":
        fitness_function = build_NK_landscape_function(rng_rep, N, K)
    elif landscape_type == "house_of_cards":
        fitness_function = build_house_of_cards_landscape_function(rng_rep, N, scale = 0.3)
    elif landscape_type == "rough_mount_fuji":
        fitness_function = build_rough_mount_fuji_landscape_function(
            rng_rep, N, slope_scale = 1.2, noise_scale = noise_scale
        )
    elif landscape_type == "stochastic_block":
        fitness_function = build_stochastic_block_landscape_function(
            rng_rep, N, num_blocks = num_blocks
        )
    else:
        raise ValueError(f"Unknown landscape_type: {landscape_type}")
    
    
    mutation_function = build_mutation_function(mut_chance, 2)

    # Define selection function.
    selection_function = build_selection_function(selection_strategy, selection_params)

    if pre_optimisation_steps != 0:

        pre_op_selection_function = build_selection_function(
            slct.base_chance_threshold_select, {"base_chance": 0.0, "threshold": 0.95}
        )

        pre_op = run_directed_evolution(
            rng_rep,
            i_pop=i_pop,
            selection_function=pre_op_selection_function,
            mutation_function=mutation_function,
            fitness_function=fitness_function,
            num_steps=pre_optimisation_steps,
        )[1]

        i_pop = pre_op["pop"][-1]

    # Bringing it all together.
    vmapped_run = jax.jit(
        jax.vmap(
            lambda r: run_directed_evolution(
                r,
                i_pop,
                selection_function,
                mutation_function,
                fitness_function=fitness_function,
                num_steps=num_steps,
            )[1]
        )
    )

    # The array of seeds we will take as input.
    rng_seeds = jr.split(s_rng, num_reps)
    results = vmapped_run(rng_seeds)

    return results


NKs = [[25,20], [25,10],[25,1] ]
noise_vals = [0,2.5,5]
num_blocks_list = [25,3,1]

# m=0.1
m = 0.5
p = 2500

reps = 25

sweep_results = np.zeros((3, 25, 1, 25))
total_iterations = len(NKs)

with tqdm.tqdm(total=total_iterations, desc="Overall Progress") as pbar:
    for i,num_blocks in enumerate(num_blocks_list):

        rep_rngs = jr.split(jr.PRNGKey(42), reps)
        s_rng=jr.PRNGKey(0)

        def single_rep(rng_rep):

            params = {"threshold": 0.0, "base_chance": 1.0}
            run = directedEvolution(
                s_rng,
                rng_rep,
                N=25,
                K=5,
                selection_strategy=slct.base_chance_threshold_select,
                landscape_type="stochastic_block",
                selection_params=params,
                popsize=int(p),
                mut_chance=m / 25,
                num_steps=25,
                num_reps=1,
                average=False,
                pre_optimisation_steps=50,
                num_blocks=num_blocks
            )

            # split_results.append(run['fitness'].max(axis=2).mean(axis=0)[-1])
            return run["fitness"].mean(axis=-1)

        repeat_results = jax.vmap(single_rep)(rep_rngs)

        sweep_results[i, :, :, :] = np.array(repeat_results)

        pbar.update(1)


file_path = os.path.join(slide_data_dir, "alt_landscapes_decay_curve_sweep_blocks.pkl")
with open(file_path, "wb") as f:
    pickle.dump(sweep_results, f)
