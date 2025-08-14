# Package imports
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from direvo_functions import *
import selection_function_library as slct
import os
import tqdm
from scipy.optimize import curve_fit
import scipy.optimize
import pandas as pd

# Limit memory usage to 50% (adjust as needed)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

# The DE function

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

### Ruggedness measurement functions

def get_lin_coeffs(landscape_arr):
    num_dims = len(landscape_arr.shape)
    const_term = landscape_arr.mean()
    lin_coeffs = []
    thing_r = list(range(num_dims))
    for i in thing_r:
        a_to_m_through = thing_r[:i] + thing_r[i+1:]
        #lin_coeffs.append( (landscape_arr - const_term).mean(axis = a_to_m_through) )
        lin_coeffs.append((landscape_arr - const_term).mean(axis=tuple(a_to_m_through)))

    return const_term, lin_coeffs

def get_lin_landscape(const_term, lin_coeffs):
    final_shape = [len(l) for l in lin_coeffs]
    final_result = jnp.zeros(final_shape)
    thing_r = list(range(len(lin_coeffs)))
    for i in thing_r:
        axis_shape = thing_r[:i] + thing_r[i+1:]
        final_result += jnp.expand_dims(jnp.array(lin_coeffs[i]), axis = axis_shape)
    return final_result + const_term

def roughness_to_slope(landscape_arr):
    # This is the slope vs rugedness metric.
    const_term, lin_coeffs = get_lin_coeffs(landscape_arr)
    mean_slope = jnp.abs(jnp.array(lin_coeffs)).sum(axis = -1).mean()
    lin_landy = get_lin_landscape(const_term, lin_coeffs)
    error_land = (landscape_arr - lin_landy)
    roughness = error_land.std()
    return roughness/mean_slope

def landscape_r2(landscape_arr):
    # this is the other metric,
    const_term, lin_coeffs = get_lin_coeffs(landscape_arr)
    lin_landy = get_lin_landscape(const_term, lin_coeffs)
    error_land = (landscape_arr - lin_landy)
    return 1 - (error_land.var() / landscape_arr.var())

def get_convergence_rate(rng, N, K, num_reps = 5):
    
    params = {'threshold': 0.0, 'base_chance' : 1.0}
    run = directedEvolution(rng,
        N = N,
        K=K,
        selection_strategy=slct.base_chance_threshold_select,
        selection_params = params,
        popsize=2500,
        mut_chance=0.1/N,
        num_steps=25,
        num_reps=num_reps,
        pre_optimisation_steps=20,
        average=False)['fitness'].mean(axis=-1)
    
    

    decay_rates = [get_single_decay_rate(run[i]) for i in range(num_reps)]
    mean_decay_rate = np.mean(np.array(decay_rates))
    extra_info = {"decay_info" : run, "all_decays" : decay_rates}
    return mean_decay_rate, extra_info

## Seb NK landscape generation functions

def get_array_from_fun(func, shape):
    to_see = np.stack(np.indices( shape), axis = -1).reshape( jnp.prod(jnp.array(shape)), len(shape))
    values = func(to_see)
    return values.reshape(shape)

def get_nk_l_o_shape(rng, N, K, shape):
    return get_array_from_fun(build_NK_landscape_function(rng, N, K), shape)

def local_epistasis(landscape, point):
    shape = landscape.shape
    N = len(shape)
    simple_sign_episasis = 0
    reciprocal_sign_epistasis = 0
    no_epistasis = 0
    base_fitness = landscape[tuple(point)]
    for mut_loc_1 in range(N):
        for mut_loc_2 in range(N):
            if mut_loc_1 == mut_loc_2:
                continue
            for mut_1 in range(shape[mut_loc_1]):
                for mut_2 in range(shape[mut_loc_2]):
                    if (mut_1 == point[mut_loc_1]) or (mut_2 == point[mut_loc_2]):
                        continue
                    #point_1 = point.at[mut_loc_1].set(mut_1)
                    #point_2 = point.at[mut_loc_2].set(mut_2)
                    #point_12 = point_1.at[mut_loc_2].set(mut_2)
                    point_1 = point.copy()
                    point_1[mut_loc_1] = mut_1

                    point_2 = point.copy()
                    point_2[mut_loc_2] = mut_2

                    point_12 = point_1.copy()
                    point_12[mut_loc_2] = mut_2

                    fit_1 = landscape[tuple(point_1)]
                    fit_2 = landscape[tuple(point_2)]
                    fit_12 = landscape[tuple(point_12)]
                    delta_1a = fit_1 - base_fitness
                    delta_1b = fit_12 - fit_2
                    delta_2a = fit_2 - base_fitness
                    delta_2b = fit_12 - fit_1
                    sign_match_1 = (delta_1a * delta_1b) >= 0
                    sign_match_2 = (delta_2a * delta_2b) >= 0
                    if sign_match_1 and sign_match_2:
                        no_epistasis += 1
                    elif sign_match_1 or sign_match_2:
                        simple_sign_episasis += 1
                    else:
                        reciprocal_sign_epistasis += 1
    to_return = {
        "simple_sign_episasis": simple_sign_episasis,
        "reciprocal_sign_epistasis": reciprocal_sign_epistasis,
        "no_epistasis": no_epistasis
    }
    return to_return

def generate_range_cube(shape):
    base_cube = jnp.zeros(shape, dtype=jnp.int32)
    for i, d_size in enumerate(shape):
        base_cube = base_cube.at[tuple([slice(None) if j != i else jnp.arange(1,d_size) for j in range(len(shape))])].add(1)
    return base_cube
 
def find_acc_path_length(landscape, starting_point):
    shape = landscape.shape
    N = len(shape)
    range_cube = generate_range_cube(shape)
    # rotate the landscape so starting point is at 0
    rotated_landsape = jnp.roll(landscape, -starting_point, axis = tuple(range(N)))
    num_paths = jnp.zeros(shape, dtype = jnp.int32)
    num_paths = num_paths.at[tuple([0]*N)].set(1)
    for step in range(N):
        for i, d_size in enumerate(shape):
            move_from_slice = tuple([slice(None) if j != i else slice(None, 1) for j in range(N)])
            move_to_slice = tuple([slice(None) if j != i else slice(1, None) for j in range(N)])
            path_accesable = (rotated_landsape <= rotated_landsape[move_from_slice]) * (range_cube == (step + 1))
            new_paths = jnp.zeros_like(num_paths).at[move_to_slice].set(num_paths[move_from_slice]) * path_accesable
            num_paths = num_paths + new_paths
    rotated_num_paths = jnp.roll(num_paths, starting_point, axis = tuple(range(N)))
    return num_paths
 
# get the index of the argmin
def get_argmax_index(landscape):
    return jnp.unravel_index(landscape.argmax(), landscape.shape)
 
def max_possible_paths(shape):
    N = len(shape)
    return jax.scipy.special.factorial(N) * jnp.prod(jnp.array(shape) - 1)
 
def get_mean_paths_to_max(landscape, norm = True, extra_slack = 0):
    max_loc = jnp.array(get_argmax_index(landscape))
    paths = find_acc_path_length(landscape, max_loc)
    range_cube = generate_range_cube(shape=landscape.shape)
    reversed_range_cube = range_cube.max() - range_cube
    opposite_side_slice = tuple([slice(1,None) for _ in range(len(landscape.shape))])
    if extra_slack > 0:
        valid_entries = reversed_range_cube <= extra_slack
        mean_paths = (paths * valid_entries).sum() / valid_entries.sum()
        return mean_paths
    if norm:
        return paths[opposite_side_slice].mean() / max_possible_paths(landscape.shape)
    else:
        return paths[opposite_side_slice].mean()
    
def find_local_max(landscape_array, limit_fit_quant = 0.5):
    shape = landscape_array.shape
    local_max_array = jnp.zeros(shape)
    limit_fit = jnp.quantile(landscape_array, limit_fit_quant)
    for i, d_size in enumerate(shape):
        for shift_a in range(1, d_size):
            rolled_array = jnp.roll(landscape_array, shift_a, axis=i)
            local_max_array = local_max_array + (landscape_array <= rolled_array)
    local_max_array = local_max_array + (landscape_array <= limit_fit)
    is_weak_local_max = (local_max_array == 0)
    return is_weak_local_max
 
 
def find_distance_to_set(set_array):
    shape = set_array.shape
    N = len(shape)
    current_distances = jnp.ones(shape)*jnp.inf
    current_distances = current_distances.at[set_array].set(0)
    for step in range(N):
        for i, d_size in enumerate(shape):
            for shift_a in range(1, d_size):
                rolled_array = jnp.roll(current_distances, shift_a, axis=i)
                current_distances = jnp.minimum(current_distances, rolled_array + 1)
    return current_distances
 
def find_distance_to_closest_max(landscape_arrays):
    return find_distance_to_set(find_local_max(landscape_arrays)).mean()/len(landscape_arrays.shape)

### Landscape spectra ###

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
 
def get_landscape_spectrum(landscape, norm = False, remove_constant = True, on_gpu = False):
    if on_gpu:
        landy_fft = fftn_jax(landscape)
    else:
        landy_fft = np.fft.fftn(landscape, norm="ortho")
    specturm = landy_fft * np.conj(landy_fft)
    collapsed_spectrum  = collapse_range(specturm)
    collapsed_spectrum = np.real(collapsed_spectrum)
    if remove_constant:
        collapsed_spectrum = collapsed_spectrum[1:]
    if norm:
        collapsed_spectrum = collapsed_spectrum / (collapsed_spectrum * collapsed_spectrum).sum()**0.5
    return collapsed_spectrum

def get_exp_matrix(
    N: int,
    A: int,
    mutations: np.array,
    is_squared: bool = True,
    fix_b0: bool = False,
) -> np.array:
    eigrange = range(N+1) if fix_b0 == False else range(1, N+1)
    eigenvalues = [A*i for i in eigrange]
    factor = 2 if is_squared is True else 1
    if fix_b0:
        return np.array([[np.exp(-mut/(N*(A-1))*l*factor)-1 for l in eigenvalues] for mut in mutations[1:]])
    else:
        return np.array([[np.exp(-mut/(N*(A-1))*l*factor) for l in eigenvalues] for mut in mutations])


## Measure decay rate function.
def get_fourier_coeffs(
    mean_fitness: np.array,
    mutations: np.array,
    N: int,
    A: int,
    is_squared: bool = False,
    fix_b0: bool = False,
    method: str = "nnls",
    alpha: str = 0.1,
) -> tuple[np.array, np.array]:
    """
    Inputs:
    mean_fitness: vector with mean fitness values
    mutations: vector with number of mutations, starting with 0
    N: length of gene
    A: number of alleles (e.g., 20 if amino acids)
    is_squared: flag to set true if mean fitness squared is provided instead.
    method: pick on of ls, ls_constrained, nnls. Results may vary. Latter to enforce positiveness of the coeffs, although positiveness is only true for the squared estimate.
    Outputs:
    (fourier_coeffs, exponentials): tuple[np.array, np.array]
    fourier_coeffs: Vector of length N+1 with Fourier coeffs from low to high frequency, with the first element corresponding to the constant.
    exponentials: len(mutations)xN+1 matrix to extract fit via exponentials*weights
    """
    exponentials = get_exp_matrix(N=N, A=A, mutations=mutations, is_squared=is_squared, fix_b0=fix_b0)
    if fix_b0:
        mean_fitness_0 = mean_fitness[0]
        mean_fitness = mean_fitness[1:] - mean_fitness_0
    if method == "ls":
        fourier_coeffs, residuals, rank, s = np.linalg.lstsq(exponentials, mean_fitness, rcond=None)
    elif method == "ls_constrained":
        res = scipy.optimize.lsq_linear(exponentials, mean_fitness, bounds=(0, np.inf))
        fourier_coeffs = res.x
    elif method == "nnls":
        fourier_coeffs, rnorm = scipy.optimize.nnls(exponentials, mean_fitness)
    elif method == "nnls_reg":
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        m, p = exponentials.shape
        A_aug = np.vstack([exponentials, np.sqrt(alpha) * np.eye(p)])
        b_aug = np.concatenate([mean_fitness, np.zeros(p)])
        fourier_coeffs, _ = scipy.optimize.nnls(A_aug, b_aug)
    else:
        raise ValueError("Method unavailable.")
    if fix_b0:
        b0 = mean_fitness_0 - np.sum(np.abs(fourier_coeffs))
        fourier_coeffs = np.concatenate((np.array([b0]), fourier_coeffs))
        exponentials = get_exp_matrix(N=N, A=A, mutations=mutations, is_squared=is_squared, fix_b0=False)
    return (fourier_coeffs, exponentials)

def get_rho(fourier_coeffs: np.array) -> float:
    fourier_coeffs = np.abs(fourier_coeffs)
    return (np.argmax(np.abs(fourier_coeffs[1:])) + 1) / fourier_coeffs[1:].shape[0]

def get_fourier_decay(fourier_coeffs: np.array, A: int, N: int, is_squared: bool = False) -> float:
    assert len(fourier_coeffs) == N+1
    ind_max = np.argmax(np.abs(fourier_coeffs[1:])) + 1
    decay_rate = (A*ind_max)/(N*(A-1)) * (2 if is_squared else 1)
    return decay_rate
