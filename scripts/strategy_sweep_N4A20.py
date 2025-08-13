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
 
 
def get_single_sweep_empirical(base_chances, thresholds, splits, popsize, num_alleles = 2, num_reps = 10):
 
    def to_return(rng, landscape_arr, init_pop = None):
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
                                   define_i_pop=init_pop,
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
       
        rngs = jr.split(rng, num_landscapes)
        vmapped_single_land = jax.vmap(single_land, in_axes=(0))
 
        results = vmapped_single_land(rngs)
 
        mean_results = jnp.mean(results, axis=0)  # Average over landscapes
        return mean_results
 
    return jax.jit(to_return, static_argnums=(1,))
 
def generate_range_cube(shape):
    base_cube = jnp.zeros(shape, dtype=jnp.int32)
    for i, d_size in enumerate(shape):
        base_cube = base_cube.at[tuple([slice(None) if j != i else jnp.arange(1,d_size) for j in range(len(shape))])].add(1)
    return base_cube
 
def collapse_range(array):
    range_cube = generate_range_cube(array.shape)
    vals = []
    for i in range(range_cube.max()+ 1):
        vals.append((array * (range_cube == i)).sum())
    return np.array(vals)
 
def get_decay_curve(landscape, start, max_mut = 2.0, num_vals = 20):
    muts = np.linspace(0, max_mut, num_vals)
    shape = landscape.shape
    N = len(shape)
    A = shape[0]
    range_cube = generate_range_cube(shape)
    landy_fft = np.fft.fftn(landscape, norm="ortho")
    base_start = np.zeros(shape, dtype=np.int32)
    base_start[tuple(start)] = 1
    start_fft = np.fft.ifftn(base_start, norm="ortho")
    raw_terms = collapse_range(landy_fft * start_fft)
    range_vals = np.arange(0, range_cube.max() + 1) * A / (N * (A - 1))
    exp_vaks = np.exp(-1.0 * muts[:, None] * range_vals[None, :])
    decay_curve = np.sum(raw_terms[None, :] * exp_vaks, axis=1)
    return muts, decay_curve, raw_terms
 
def fftn_jax(x, axes=None):
    if axes is None:
        axes = tuple(range(x.ndim))  # Default: FFT along all dimensions
   
    # Apply 1D FFT sequentially over specified axes
    for ax in axes:
        x = jax.numpy.fft.fft(x, axis=ax)
   
    return x
 
def get_landscape_spectrum(landscape, norm = True, remove_constant = True, on_gpu = False):
    if on_gpu:
        landy_fft = fftn_jax(landscape)
    else:
        landy_fft = np.fft.fftn(landscape, norm="ortho")
    specturm = landy_fft * np.conj(landy_fft)
    collapsed_spectrum  = collapse_range(specturm)
    if remove_constant:
        collapsed_spectrum = collapsed_spectrum[1:]
    if norm:
        collapsed_spectrum = collapsed_spectrum / (collapsed_spectrum * collapsed_spectrum).sum()**0.5
    return jnp.real(collapsed_spectrum)
 
 
def shuffle_with_mask(rng, values, mask):
    vals_shape = values.shape
    flat_values = values.reshape(-1)
    flat_mask = mask.reshape(-1)
    n_true = flat_mask.sum()
    idx_true = jnp.where(flat_mask, size=n_true, fill_value=-1)[0]
    shuffled_values = flat_values.at[idx_true].set(jax.random.permutation(rng, flat_values[idx_true]))
    return shuffled_values.reshape(vals_shape)
 
def spectral_shuffle(rng, landscape, flip_dirs = True):
    """
    Shuffle the landscape in the spectral domain.
    """
    # Get the spectrum of the landscape
    spectral_landscape = fftn_jax(landscape)
   
    num_dims = len(landscape.shape)
 
    range_cube = generate_range_cube(landscape.shape)
 
    r1, r2 = jr.split(rng, 2)
    rng_keys = jr.split(r1, num_dims + 1)
 
    for i in range(num_dims + 1):
        # Create a mask for the current dimension
        mask = (range_cube == i)
       
        # Shuffle the values in the landscape where the mask is True
        spectral_landscape = shuffle_with_mask(rng_keys[i], spectral_landscape, mask)
 
    if flip_dirs:
        random_flip = jr.bernoulli(r2, p=0.5, shape=spectral_landscape.shape)*2 - 1
        random_flip = random_flip.at[jnp.where(range_cube == 0)].set(1)  # Ensure the constant term is not flipped
        spectral_landscape = spectral_landscape * random_flip
    # Inverse FFT to get the shuffled landscape
    new_landscape = jnp.real(fftn_jax(spectral_landscape))
    normed_new_landscape = new_landscape / jnp.prod(jnp.array(new_landscape.shape))
    return normed_new_landscape
 
 
def NK_grid(N_range, num_samples=10):
    N = jnp.linspace(N_range[0], N_range[1], num=num_samples)
    K = jnp.array([jnp.linspace(1, i, num_samples)
                  for i in N]).reshape(num_samples, num_samples)
    N = jnp.repeat(N, num_samples).reshape(num_samples, num_samples)
    return N, K

thresholds, base_chances = base_chance_threshold_fixed_prop([0,0.19], 0.2, 7)
splits = [24,20,16,12,8,4,1]
m=0.1
p=1200
sweepy_NK = get_multi_sweep_NK(base_chances, thresholds, splits, p, num_alleles=20, num_reps=10, num_landscapes=125)
results=[[sweepy_NK(ii, 4, i).mean(axis=0) for i in [1,2,3]] for ii in jr.split(jr.PRNGKey(42), 10)]

with open(os.path.join(slide_data_dir, "N4A20_strategy_sweep.pkl"), "wb") as f:
    pickle.dump(np.array(results), f)