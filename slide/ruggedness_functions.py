from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pickle
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from slide.direvo_functions import *
from slide import selection_function_library as slct
import os
import tqdm
from scipy.optimize import curve_fit
import scipy.optimize
import pandas as pd

Array = jax.Array | np.ndarray
Shape = tuple[int, ...] | list[int]
SelectionShape = Callable[[jax.Array, Mapping[str, float]], jax.Array]

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'

def _legacy_directed_evolution(rng: jax.Array, selection_strategy: SelectionShape, selection_params: Mapping[str, float], empirical: bool=False, N: int | None=None, K: int | None=None, landscape: Array | None=None, popsize: int=100, mut_chance: float=0.01, num_steps: int=50, num_reps: int=10, define_i_pop: Array | None=None, pre_optimisation_steps: int=0, average: bool=True) -> dict[str, jax.Array]:
    """Run the historical directed-evolution wrapper used by old analyses.

    Parameters:
    - rng: jax.Array
        JAX PRNG key for landscape generation and replicate runs.
    - selection_strategy: SelectionShape
        Selection probability function.
    - selection_params: Mapping[str, float]
        Parameters for the selection strategy.
    - empirical: bool
        Whether to use an empirical landscape rather than a generated NK landscape.
    - N: int | None
        Number of genotype sites for generated NK landscapes.
    - K: int | None
        NK interaction parameter for generated NK landscapes.
    - landscape: Array | None
        Empirical landscape used when ``empirical`` is true.
    - popsize: int
        Population size.
    - mut_chance: float
        Per-site mutation probability.
    - num_steps: int
        Number of directed-evolution generations.
    - num_reps: int
        Number of replicate trajectories.
    - define_i_pop: Array | None
        Optional initial population.
    - pre_optimisation_steps: int
        Number of preliminary selection generations.
    - average: bool
        Historical flag retained for compatibility.

    Returns:
    - dict[str, jax.Array]
        Replicate fitness and population histories.
    """
    r1, r2, r3 = jr.split(rng, 3)
    if define_i_pop == None:
        i_pop = jnp.array([jr.randint(r1, (N,), 0, 2)] * popsize)
    else:
        i_pop = define_i_pop
    if empirical:
        fitness_function = build_empirical_landscape_function(landscape)
        mutation_function = build_mutation_function(mut_chance, 20)
    else:
        fitness_function = build_NK_landscape_function(r2, N, K)
        mutation_function = build_mutation_function(mut_chance, 2)
    selection_function = build_selection_function(selection_strategy, selection_params)
    if pre_optimisation_steps != 0:
        pre_op_selection_function = build_selection_function(slct.base_chance_threshold_select, {'base_chance': 0.0, 'threshold': 0.95})
        pre_op = run_directed_evolution(r3, i_pop=i_pop, selection_function=pre_op_selection_function, mutation_function=mutation_function, fitness_function=fitness_function, num_steps=pre_optimisation_steps)[1]
        i_pop = pre_op['pop'][-1]
    vmapped_run = jax.jit(jax.vmap(lambda r: run_directed_evolution(r, i_pop, selection_function, mutation_function, fitness_function=fitness_function, num_steps=num_steps)[1]))
    rng_seeds = jr.split(r3, num_reps)
    results = vmapped_run(rng_seeds)
    return results

def get_lin_coeffs(landscape_arr: Array) -> tuple[jax.Array | np.ndarray, list[jax.Array | np.ndarray]]:
    """Estimate additive linear coefficients for a landscape.

    Parameters:
    - landscape_arr: Array
        N-dimensional fitness landscape.

    Returns:
    - tuple[jax.Array | np.ndarray, list[jax.Array]]
        Mean fitness and one marginal coefficient vector per landscape axis.
    """
    num_dims = len(landscape_arr.shape)
    const_term = landscape_arr.mean()
    lin_coeffs = []
    thing_r = list(range(num_dims))
    for i in thing_r:
        a_to_m_through = thing_r[:i] + thing_r[i + 1:]
        lin_coeffs.append((landscape_arr - const_term).mean(axis=tuple(a_to_m_through)))
    return (const_term, lin_coeffs)

def get_lin_landscape(const_term: float | jax.Array | np.ndarray, lin_coeffs: Sequence[Array]) -> jax.Array:
    """Reconstruct an additive landscape from linear coefficients.

    Parameters:
    - const_term: float | jax.Array
        Constant fitness offset.
    - lin_coeffs: Sequence[Array]
        One marginal coefficient vector per landscape axis.

    Returns:
    - jax.Array
        Additive landscape with shape inferred from the coefficients.
    """
    final_shape = [len(l) for l in lin_coeffs]
    final_result = jnp.zeros(final_shape)
    thing_r = list(range(len(lin_coeffs)))
    for i in thing_r:
        axis_shape = thing_r[:i] + thing_r[i + 1:]
        final_result += jnp.expand_dims(jnp.array(lin_coeffs[i]), axis=axis_shape)
    return final_result + const_term

def roughness_to_slope_old(landscape_arr: Array) -> jax.Array:
    """Compute residual roughness relative to additive slope magnitude.

    Parameters:
    - landscape_arr: Array
        N-dimensional fitness landscape.

    Returns:
    - jax.Array
        Ratio of additive-model residual standard deviation to mean slope.
    """
    const_term, lin_coeffs = get_lin_coeffs(landscape_arr)
    mean_slope = jnp.abs(jnp.array(lin_coeffs)).sum(axis=-1).mean()
    lin_landy = get_lin_landscape(const_term, lin_coeffs)
    error_land = landscape_arr - lin_landy
    roughness = error_land.std()
    return roughness / mean_slope

def landscape_r2_old(landscape_arr: Array) -> jax.Array:
    """Compute the additive linear model R-squared for a landscape.

    Parameters:
    - landscape_arr: Array
        N-dimensional fitness landscape.

    Returns:
    - jax.Array
        Fraction of landscape variance explained by additive terms.
    """
    const_term, lin_coeffs = get_lin_coeffs(landscape_arr)
    lin_landy = get_lin_landscape(const_term, lin_coeffs)
    error_land = landscape_arr - lin_landy
    return 1 - error_land.var() / landscape_arr.var()

def get_convergence_rate(rng: jax.Array, N: int, K: int, num_reps: int=5) -> tuple[float, dict[str, object]]:
    """Estimate a convergence decay rate on generated NK landscapes.

    Parameters:
    - rng: jax.Array
        JAX PRNG key.
    - N: int
        Number of genotype sites.
    - K: int
        NK interaction parameter.
    - num_reps: int
        Number of replicate trajectories.

    Returns:
    - tuple[float, dict[str, object]]
        Mean fitted decay rate and intermediate decay information.
    """
    params = {'threshold': 0.0, 'base_chance': 1.0}
    run = _legacy_directed_evolution(rng, N=N, K=K, selection_strategy=slct.base_chance_threshold_select, selection_params=params, popsize=2500, mut_chance=0.1 / N, num_steps=25, num_reps=num_reps, pre_optimisation_steps=20, average=False)['fitness'].mean(axis=-1)
    decay_rates = [get_single_decay_rate(run[i]) for i in range(num_reps)]
    mean_decay_rate = np.mean(np.array(decay_rates))
    extra_info = {'decay_info': run, 'all_decays': decay_rates}
    return (mean_decay_rate, extra_info)

def get_array_from_fun(func: Callable[[Array], Array], shape: Shape) -> Array:
    """Evaluate a vectorized landscape function on every coordinate in a shape.

    Parameters:
    - func: Callable[[Array], Array]
        Function mapping coordinates to values.
    - shape: Shape
        Output landscape shape.

    Returns:
    - jax.Array | np.ndarray
        Function values reshaped to ``shape``.
    """
    to_see = np.stack(np.indices(shape), axis=-1).reshape(jnp.prod(jnp.array(shape)), len(shape))
    values = func(to_see)
    return values.reshape(shape)

def get_nk_l_o_shape(rng: jax.Array, N: int, K: int, shape: Shape) -> Array:
    """Generate an NK landscape array with an explicit output shape.

    Parameters:
    - rng: jax.Array
        JAX PRNG key.
    - N: int
        Number of genotype sites.
    - K: int
        NK interaction parameter.
    - shape: Shape
        Shape of the output landscape array.

    Returns:
    - jax.Array | np.ndarray
        Generated NK fitness landscape.
    """
    return get_array_from_fun(build_NK_landscape_function(rng, N, K), shape)

def local_epistasis_old(landscape: Array, point: np.ndarray) -> dict[str, int]:
    """Count pairwise sign-epistasis classes around one genotype.

    Parameters:
    - landscape: Array
        N-dimensional fitness landscape.
    - point: np.ndarray
        Genotype coordinate used as the reference point.

    Returns:
    - dict[str, int]
        Counts of simple sign, reciprocal sign, and no-epistasis cases.
    """
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
                    if mut_1 == point[mut_loc_1] or mut_2 == point[mut_loc_2]:
                        continue
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
                    sign_match_1 = delta_1a * delta_1b >= 0
                    sign_match_2 = delta_2a * delta_2b >= 0
                    if sign_match_1 and sign_match_2:
                        no_epistasis += 1
                    elif sign_match_1 or sign_match_2:
                        simple_sign_episasis += 1
                    else:
                        reciprocal_sign_epistasis += 1
    to_return = {'simple_sign_episasis': simple_sign_episasis, 'reciprocal_sign_epistasis': reciprocal_sign_epistasis, 'no_epistasis': no_epistasis}
    return to_return

def generate_range_cube(shape: Shape) -> jax.Array:
    """Return the mutational distance layer for each landscape coordinate.

    Parameters:
    - shape: Shape
        Landscape shape.

    Returns:
    - jax.Array
        Integer array assigning each coordinate to a distance layer.
    """
    base_cube = jnp.zeros(shape, dtype=jnp.int32)
    for i, d_size in enumerate(shape):
        base_cube = base_cube.at[tuple([slice(None) if j != i else jnp.arange(1, d_size) for j in range(len(shape))])].add(1)
    return base_cube

def _roll_landscape_to_origin(landscape: Array, starting_point: Array) -> jax.Array:
    """Roll a landscape so ``starting_point`` is placed at the origin.

    Parameters:
    - landscape: Array
        N-dimensional fitness landscape.
    - starting_point: Array
        Coordinate shifted to the origin.

    Returns:
    - jax.Array
        Landscape equivalent to ``jnp.roll(landscape, -starting_point)``.
    """
    landscape_array = jnp.asarray(landscape)
    point_array = jnp.asarray(starting_point, dtype=jnp.int32)
    coordinate_grids = jnp.indices(landscape_array.shape, dtype=jnp.int32)
    source_indices = tuple(
        (coordinate_grids[axis] + point_array[axis]) % landscape_array.shape[axis]
        for axis in range(landscape_array.ndim)
    )
    return landscape_array[source_indices]

def find_acc_path_length(landscape: Array, starting_point: Array) -> jax.Array:
    """Count accessible monotonic paths from a starting point.

    Parameters:
    - landscape: Array
        N-dimensional fitness landscape.
    - starting_point: Array
        Genotype coordinate used as the path origin.

    Returns:
    - jax.Array
        Path counts indexed by landscape coordinate.
    """
    shape = landscape.shape
    N = len(shape)
    range_cube = generate_range_cube(shape)
    rotated_landsape = _roll_landscape_to_origin(landscape, starting_point)
    num_paths = jnp.zeros(shape, dtype=jnp.int32)
    num_paths = num_paths.at[tuple([0] * N)].set(1)
    for step in range(N):
        for i, d_size in enumerate(shape):
            move_from_slice = tuple([slice(None) if j != i else slice(None, 1) for j in range(N)])
            move_to_slice = tuple([slice(None) if j != i else slice(1, None) for j in range(N)])
            path_accesable = (rotated_landsape <= rotated_landsape[move_from_slice]) * (range_cube == step + 1)
            new_paths = jnp.zeros_like(num_paths).at[move_to_slice].set(num_paths[move_from_slice]) * path_accesable
            num_paths = num_paths + new_paths
    rotated_num_paths = jnp.roll(num_paths, starting_point, axis=tuple(range(N)))
    return num_paths

def get_argmax_index(landscape: Array) -> tuple[jax.Array, ...]:
    """Return the coordinate of the maximum landscape value.

    Parameters:
    - landscape: Array
        N-dimensional fitness landscape.

    Returns:
    - tuple[jax.Array, ...]
        Coordinate returned by ``jax.numpy.unravel_index``.
    """
    return jnp.unravel_index(landscape.argmax(), landscape.shape)

def max_possible_paths(shape: Shape) -> jax.Array:
    """Compute the maximum possible monotonic paths across a landscape shape.

    Parameters:
    - shape: Shape
        Landscape shape.

    Returns:
    - jax.Array
        Maximum path count under the combinatorial approximation used here.
    """
    N = len(shape)
    return jax.scipy.special.factorial(N) * jnp.prod(jnp.array(shape) - 1)

def get_mean_paths_to_max_old(landscape: Array, norm: bool=True, extra_slack: int=0) -> jax.Array:
    """Compute the mean accessible path count to the global maximum.

    Parameters:
    - landscape: Array
        N-dimensional fitness landscape.
    - norm: bool
        Whether to normalize by the maximum possible path count.
    - extra_slack: int
        Number of terminal distance layers to include instead of only the opposite
        corner region.

    Returns:
    - jax.Array
        Mean accessible path count or normalized mean path count.
    """
    max_loc = jnp.array(get_argmax_index(landscape))
    paths = find_acc_path_length(landscape, max_loc)
    range_cube = generate_range_cube(shape=landscape.shape)
    reversed_range_cube = range_cube.max() - range_cube
    opposite_side_slice = tuple([slice(1, None) for _ in range(len(landscape.shape))])
    if extra_slack > 0:
        valid_entries = reversed_range_cube <= extra_slack
        mean_paths = (paths * valid_entries).sum() / valid_entries.sum()
        return mean_paths
    if norm:
        return paths[opposite_side_slice].mean() / max_possible_paths(landscape.shape)
    else:
        return paths[opposite_side_slice].mean()

def find_local_max(landscape_array: Array, limit_fit_quant: float=0.5) -> jax.Array:
    """Identify weak local maxima above a fitness quantile threshold.

    Parameters:
    - landscape_array: Array
        N-dimensional fitness landscape.
    - limit_fit_quant: float
        Quantile below which points are excluded from local maxima.

    Returns:
    - jax.Array
        Boolean array marking weak local maxima.
    """
    shape = landscape_array.shape
    local_max_array = jnp.zeros(shape)
    limit_fit = jnp.quantile(landscape_array, limit_fit_quant)
    for i, d_size in enumerate(shape):
        for shift_a in range(1, d_size):
            rolled_array = jnp.roll(landscape_array, shift_a, axis=i)
            local_max_array = local_max_array + (landscape_array <= rolled_array)
    local_max_array = local_max_array + (landscape_array <= limit_fit)
    is_weak_local_max = local_max_array == 0
    return is_weak_local_max

def find_distance_to_set(set_array: Array) -> jax.Array:
    """Compute Hamming-like distance to the nearest true entry in a mask.

    Parameters:
    - set_array: Array
        Boolean mask over genotype coordinates.

    Returns:
    - jax.Array
        Distance from each coordinate to the nearest true mask entry.
    """
    shape = set_array.shape
    N = len(shape)
    current_distances = jnp.where(set_array, 0.0, jnp.inf)
    for step in range(N):
        for i, d_size in enumerate(shape):
            for shift_a in range(1, d_size):
                rolled_array = jnp.roll(current_distances, shift_a, axis=i)
                current_distances = jnp.minimum(current_distances, rolled_array + 1)
    return current_distances

def find_distance_to_closest_max_old(landscape_arrays: Array) -> jax.Array:
    """Compute mean normalized distance to the closest local maximum.

    Parameters:
    - landscape_arrays: Array
        N-dimensional fitness landscape.

    Returns:
    - jax.Array
        Mean distance to local maxima divided by the number of sites.
    """
    return find_distance_to_set(find_local_max(landscape_arrays)).mean() / len(landscape_arrays.shape)

def generate_range_cube(shape: Shape) -> jax.Array:
    """Return the mutational distance layer for each landscape coordinate.

    Parameters:
    - shape: Shape
        Landscape shape.

    Returns:
    - jax.Array
        Integer array assigning each coordinate to a distance layer.
    """
    base_cube = jnp.zeros(shape, dtype=jnp.int32)
    for i, d_size in enumerate(shape):
        base_cube = base_cube.at[tuple([slice(None) if j != i else jnp.arange(1, d_size) for j in range(len(shape))])].add(1)
    return base_cube

def collapse_range(array: Array) -> np.ndarray:
    """Sum array values by mutational distance layer.

    Parameters:
    - array: Array
        N-dimensional values to collapse by range layer.

    Returns:
    - np.ndarray
        One value per mutational distance layer.
    """
    range_cube = generate_range_cube(array.shape)
    vals = []
    for i in range(range_cube.max() + 1):
        vals.append((array * (range_cube == i)).sum())
    return np.array(vals)

def get_decay_curve(landscape: np.ndarray, start: np.ndarray, max_mut: float=2.0, num_vals: int=20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the theoretical Fourier decay curve from one starting genotype.

    Parameters:
    - landscape: np.ndarray
        N-dimensional fitness landscape.
    - start: np.ndarray
        Starting genotype coordinate.
    - max_mut: float
        Maximum mutation distance to evaluate.
    - num_vals: int
        Number of mutation values in the curve.

    Returns:
    - tuple[np.ndarray, np.ndarray, np.ndarray]
        Mutation values, decay curve, and collapsed Fourier terms.
    """
    muts = np.linspace(0, max_mut, num_vals)
    shape = landscape.shape
    N = len(shape)
    A = shape[0]
    range_cube = generate_range_cube(shape)
    landy_fft = np.fft.fftn(landscape, norm='ortho')
    base_start = np.zeros(shape, dtype=np.int32)
    base_start[tuple(start)] = 1
    start_fft = np.fft.ifftn(base_start, norm='ortho')
    raw_terms = collapse_range(landy_fft * start_fft)
    range_vals = np.arange(0, range_cube.max() + 1) * A / (N * (A - 1))
    exp_vaks = np.exp(-1.0 * muts[:, None] * range_vals[None, :])
    decay_curve = np.sum(raw_terms[None, :] * exp_vaks, axis=1)
    return (muts, decay_curve, raw_terms)

def fftn_jax(x: jax.Array, axes: Sequence[int] | None=None) -> jax.Array:
    """Apply a JAX FFT across selected axes.

    Parameters:
    - x: jax.Array
        Input array.
    - axes: Sequence[int] | None
        Axes over which to apply FFT; all axes are used when omitted.

    Returns:
    - jax.Array
        FFT-transformed array.
    """
    if axes is None:
        axes = tuple(range(x.ndim))
    for ax in axes:
        x = jax.numpy.fft.fft(x, axis=ax)
    return x

def get_landscape_spectrum(landscape: Array, norm: bool=False, remove_constant: bool=True, on_gpu: bool=False) -> np.ndarray:
    """Collapse landscape Fourier power by mutational distance layer.

    Parameters:
    - landscape: Array
        N-dimensional fitness landscape.
    - norm: bool
        Whether to L2-normalize the collapsed spectrum.
    - remove_constant: bool
        Whether to drop the zero-frequency coefficient.
    - on_gpu: bool
        Whether to use the JAX FFT path.

    Returns:
    - np.ndarray
        Collapsed real-valued spectrum.
    """
    if on_gpu:
        landy_fft = fftn_jax(landscape)
    else:
        landy_fft = np.fft.fftn(landscape, norm='ortho')
    specturm = landy_fft * np.conj(landy_fft)
    collapsed_spectrum = collapse_range(specturm)
    collapsed_spectrum = np.real(collapsed_spectrum)
    if remove_constant:
        collapsed_spectrum = collapsed_spectrum[1:]
    if norm:
        collapsed_spectrum = collapsed_spectrum / (collapsed_spectrum * collapsed_spectrum).sum() ** 0.5
    return collapsed_spectrum

def get_exp_matrix(N: int, A: int, mutations: np.ndarray, is_squared: bool=True, fix_b0: bool=False) -> np.ndarray:
    """Build the exponential design matrix for Fourier decay fitting.

    Parameters:
    - N: int
        Number of genotype sites.
    - A: int
        Number of alleles per site.
    - mutations: np.ndarray
        Mutation values at which fitness means were measured.
    - is_squared: bool
        Whether the decay model is for squared fitness.
    - fix_b0: bool
        Whether to fit coefficients relative to the zero-mutation value.

    Returns:
    - np.ndarray
        Exponential design matrix.
    """
    eigrange = range(N + 1) if fix_b0 == False else range(1, N + 1)
    eigenvalues = [A * i for i in eigrange]
    factor = 2 if is_squared is True else 1
    if fix_b0:
        return np.array([[np.exp(-mut / (N * (A - 1)) * l * factor) - 1 for l in eigenvalues] for mut in mutations[1:]])
    else:
        return np.array([[np.exp(-mut / (N * (A - 1)) * l * factor) for l in eigenvalues] for mut in mutations])

def get_fourier_coeffs(mean_fitness: np.ndarray, mutations: np.ndarray, N: int, A: int, is_squared: bool=False, fix_b0: bool=False, method: str='nnls', alpha: float=0.1) -> tuple[np.ndarray, np.ndarray]:
    """Fit Fourier coefficients from mean-fitness decay measurements.

    Parameters:
    - mean_fitness: np.ndarray
        Mean fitness values across mutation distances.
    - mutations: np.ndarray
        Mutation values, starting at zero.
    - N: int
        Number of genotype sites.
    - A: int
        Number of alleles per site.
    - is_squared: bool
        Whether mean squared fitness is being fitted.
    - fix_b0: bool
        Whether to fit coefficients relative to the zero-mutation value.
    - method: str
        Fitting method: ``ls``, ``ls_constrained``, ``nnls``, or ``nnls_reg``.
    - alpha: float
        Ridge penalty used by ``nnls_reg``.

    Returns:
    - tuple[np.ndarray, np.ndarray]
        Fitted Fourier coefficients and the exponential design matrix.
    """
    exponentials = get_exp_matrix(N=N, A=A, mutations=mutations, is_squared=is_squared, fix_b0=fix_b0)
    if fix_b0:
        mean_fitness_0 = mean_fitness[0]
        mean_fitness = mean_fitness[1:] - mean_fitness_0
    if method == 'ls':
        fourier_coeffs, residuals, rank, s = np.linalg.lstsq(exponentials, mean_fitness, rcond=None)
    elif method == 'ls_constrained':
        res = scipy.optimize.lsq_linear(exponentials, mean_fitness, bounds=(0, np.inf))
        fourier_coeffs = res.x
    elif method == 'nnls':
        fourier_coeffs, rnorm = scipy.optimize.nnls(exponentials, mean_fitness)
    elif method == 'nnls_reg':
        if alpha < 0:
            raise ValueError('alpha must be >= 0')
        m, p = exponentials.shape
        A_aug = np.vstack([exponentials, np.sqrt(alpha) * np.eye(p)])
        b_aug = np.concatenate([mean_fitness, np.zeros(p)])
        fourier_coeffs, _ = scipy.optimize.nnls(A_aug, b_aug)
    else:
        raise ValueError('Method unavailable.')
    if fix_b0:
        b0 = mean_fitness_0 - np.sum(np.abs(fourier_coeffs))
        fourier_coeffs = np.concatenate((np.array([b0]), fourier_coeffs))
        exponentials = get_exp_matrix(N=N, A=A, mutations=mutations, is_squared=is_squared, fix_b0=False)
    return (fourier_coeffs, exponentials)

def get_rho(fourier_coeffs: np.ndarray) -> float:
    """Return the normalized dominant non-constant Fourier frequency.

    Parameters:
    - fourier_coeffs: np.ndarray
        Fourier coefficients ordered from low to high frequency.

    Returns:
    - float
        Dominant frequency index divided by the number of non-constant bands.
    """
    fourier_coeffs = np.abs(fourier_coeffs)
    return (np.argmax(np.abs(fourier_coeffs[1:])) + 1) / fourier_coeffs[1:].shape[0]

def get_fourier_decay(fourier_coeffs: np.ndarray, A: int, N: int, is_squared: bool=False) -> float:
    """Return the decay rate implied by the dominant Fourier coefficient.

    Parameters:
    - fourier_coeffs: np.ndarray
        Fourier coefficients ordered from low to high frequency.
    - A: int
        Number of alleles per site.
    - N: int
        Number of genotype sites.
    - is_squared: bool
        Whether the coefficients describe squared fitness.

    Returns:
    - float
        Dominant Fourier decay rate.
    """
    assert len(fourier_coeffs) == N + 1
    ind_max = np.argmax(np.abs(fourier_coeffs[1:])) + 1
    decay_rate = A * ind_max / (N * (A - 1)) * (2 if is_squared else 1)
    return decay_rate

def get_spectral_entropy_old(landscape: Array, remove_constant: bool=True, on_gpu: bool=False) -> float:
    """Compute normalized entropy of the collapsed Fourier spectrum.

    Parameters:
    - landscape: Array
        N-dimensional fitness landscape.
    - remove_constant: bool
        Whether to omit the zero-frequency coefficient.
    - on_gpu: bool
        Whether to use the JAX FFT path.

    Returns:
    - float
        Normalized spectral entropy.
    """
    spectrum = get_landscape_spectrum(landscape, norm=True, remove_constant=remove_constant, on_gpu=on_gpu)
    p = spectrum / sum(spectrum)
    spectral_entropy = -np.sum(p * np.log(p + 1e-10)) / np.log(len(spectrum))
    return spectral_entropy

def get_dirichlet_metric_old(landscape: Array, on_gpu: bool=False) -> float:
    """Compute the normalized Dirichlet metric from the Fourier spectrum.

    Parameters:
    - landscape: Array
        N-dimensional fitness landscape.
    - on_gpu: bool
        Whether to use the JAX FFT path.

    Returns:
    - float
        Weighted spectral roughness metric.
    """
    spectrum = get_landscape_spectrum(landscape, norm=True, remove_constant=True, on_gpu=on_gpu)
    A = landscape.shape[0]
    N = len(landscape.shape)
    d = N * (A - 1)
    indices = np.arange(1, N + 1)
    return np.sum(A * indices * spectrum) / np.sum(spectrum) / d

def _jax_lin_coeffs(landscape_arr: Array) -> tuple[jax.Array, list[jax.Array]]:
    """Estimate additive coefficients using JAX operations.

    Parameters:
    - landscape_arr: Array
        N-dimensional fitness landscape.

    Returns:
    - tuple[jax.Array, list[jax.Array]]
        Mean fitness and one marginal coefficient vector per landscape axis.
    """
    landscape_array = jnp.asarray(landscape_arr)
    const_term = landscape_array.mean()
    axes = tuple(range(landscape_array.ndim))
    lin_coeffs = [
        (landscape_array - const_term).mean(axis=tuple(axis for axis in axes if axis != coefficient_axis))
        for coefficient_axis in axes
    ]
    return const_term, lin_coeffs

def _jax_lin_landscape(const_term: jax.Array, lin_coeffs: Sequence[Array]) -> jax.Array:
    """Reconstruct an additive landscape from JAX coefficient arrays.

    Parameters:
    - const_term: jax.Array
        Constant fitness offset.
    - lin_coeffs: Sequence[Array]
        One marginal coefficient vector per landscape axis.

    Returns:
    - jax.Array
        Additive landscape with shape inferred from the coefficients.
    """
    coeff_arrays = [jnp.asarray(coefficient) for coefficient in lin_coeffs]
    final_shape = tuple(int(coefficient.shape[0]) for coefficient in coeff_arrays)
    final_result = jnp.zeros(final_shape, dtype=jnp.result_type(const_term, *coeff_arrays))
    axes = tuple(range(len(coeff_arrays)))
    for coefficient_axis, coefficient in enumerate(coeff_arrays):
        final_result = final_result + jnp.expand_dims(
            coefficient,
            axis=tuple(axis for axis in axes if axis != coefficient_axis),
        )
    return final_result + const_term

def roughness_to_slope(landscape_arr: Array) -> jax.Array:
    """Compute residual roughness relative to additive slope magnitude.

    Parameters:
    - landscape_arr: Array
        N-dimensional fitness landscape.

    Returns:
    - jax.Array
        Ratio of additive-model residual standard deviation to mean slope.
    """
    landscape_array = jnp.asarray(landscape_arr)
    const_term, lin_coeffs = _jax_lin_coeffs(landscape_array)
    mean_slope = jnp.abs(jnp.stack(lin_coeffs)).sum(axis=-1).mean()
    lin_landy = _jax_lin_landscape(const_term, lin_coeffs)
    error_land = landscape_array - lin_landy
    roughness = error_land.std()
    return roughness / mean_slope

def landscape_r2(landscape_arr: Array) -> jax.Array:
    """Compute the additive linear model R-squared for a landscape.

    Parameters:
    - landscape_arr: Array
        N-dimensional fitness landscape.

    Returns:
    - jax.Array
        Fraction of landscape variance explained by additive terms.
    """
    landscape_array = jnp.asarray(landscape_arr)
    const_term, lin_coeffs = _jax_lin_coeffs(landscape_array)
    lin_landy = _jax_lin_landscape(const_term, lin_coeffs)
    error_land = landscape_array - lin_landy
    return 1 - error_land.var() / landscape_array.var()

def local_epistasis(landscape: Array, point: np.ndarray) -> dict[str, jax.Array]:
    """Count pairwise sign-epistasis classes around one genotype.

    Parameters:
    - landscape: Array
        N-dimensional fitness landscape.
    - point: np.ndarray
        Genotype coordinate used as the reference point.

    Returns:
    - dict[str, jax.Array]
        Counts of simple sign, reciprocal sign, and no-epistasis cases.
    """
    landscape_array = jnp.asarray(landscape)
    point_array = np.asarray(point, dtype=int)
    shape = landscape_array.shape
    simple_sign_episasis = jnp.array(0, dtype=jnp.int32)
    reciprocal_sign_epistasis = jnp.array(0, dtype=jnp.int32)
    no_epistasis = jnp.array(0, dtype=jnp.int32)
    base_fitness = landscape_array[tuple(point_array.tolist())]

    for mut_loc_1 in range(landscape_array.ndim):
        for mut_loc_2 in range(landscape_array.ndim):
            if mut_loc_1 == mut_loc_2:
                continue
            for mut_1 in range(shape[mut_loc_1]):
                for mut_2 in range(shape[mut_loc_2]):
                    if mut_1 == point_array[mut_loc_1] or mut_2 == point_array[mut_loc_2]:
                        continue
                    point_1 = point_array.copy()
                    point_1[mut_loc_1] = mut_1
                    point_2 = point_array.copy()
                    point_2[mut_loc_2] = mut_2
                    point_12 = point_1.copy()
                    point_12[mut_loc_2] = mut_2
                    fit_1 = landscape_array[tuple(point_1.tolist())]
                    fit_2 = landscape_array[tuple(point_2.tolist())]
                    fit_12 = landscape_array[tuple(point_12.tolist())]
                    delta_1a = fit_1 - base_fitness
                    delta_1b = fit_12 - fit_2
                    delta_2a = fit_2 - base_fitness
                    delta_2b = fit_12 - fit_1
                    sign_match_1 = delta_1a * delta_1b >= 0
                    sign_match_2 = delta_2a * delta_2b >= 0
                    no_epistasis = no_epistasis + jnp.asarray(sign_match_1 & sign_match_2, dtype=jnp.int32)
                    simple_sign_episasis = simple_sign_episasis + jnp.asarray(sign_match_1 ^ sign_match_2, dtype=jnp.int32)
                    reciprocal_sign_epistasis = reciprocal_sign_epistasis + jnp.asarray(
                        (~sign_match_1) & (~sign_match_2),
                        dtype=jnp.int32,
                    )

    return {
        'simple_sign_episasis': simple_sign_episasis,
        'reciprocal_sign_epistasis': reciprocal_sign_epistasis,
        'no_epistasis': no_epistasis,
    }

def get_mean_paths_to_max(landscape: Array, norm: bool=True, extra_slack: int=0) -> jax.Array:
    """Compute the mean accessible path count to the global maximum.

    Parameters:
    - landscape: Array
        N-dimensional fitness landscape.
    - norm: bool
        Whether to normalize by the maximum possible path count.
    - extra_slack: int
        Number of terminal distance layers to include instead of only the opposite
        corner region.

    Returns:
    - jax.Array
        Mean accessible path count or normalized mean path count.
    """
    landscape_array = jnp.asarray(landscape)
    max_loc = jnp.asarray(get_argmax_index(landscape_array), dtype=jnp.int32)
    paths = find_acc_path_length(landscape_array, max_loc)
    range_cube = generate_range_cube(shape=landscape_array.shape)
    reversed_range_cube = range_cube.max() - range_cube
    opposite_side_slice = tuple([slice(1, None) for _ in range(landscape_array.ndim)])
    if extra_slack > 0:
        valid_entries = reversed_range_cube <= extra_slack
        mean_paths = (paths * valid_entries).sum() / valid_entries.sum()
        return mean_paths
    if norm:
        return paths[opposite_side_slice].mean() / max_possible_paths(landscape_array.shape)
    return paths[opposite_side_slice].mean()

def find_distance_to_closest_max(landscape_arrays: Array) -> jax.Array:
    """Compute mean normalized distance to the closest local maximum.

    Parameters:
    - landscape_arrays: Array
        N-dimensional fitness landscape.

    Returns:
    - jax.Array
        Mean distance to local maxima divided by the number of sites.
    """
    landscape_array = jnp.asarray(landscape_arrays)
    return find_distance_to_set(find_local_max(landscape_array)).mean() / landscape_array.ndim

def _collapse_range_jax(array: Array) -> jax.Array:
    """Sum array values by mutational distance layer using JAX operations.

    Parameters:
    - array: Array
        N-dimensional values to collapse by range layer.

    Returns:
    - jax.Array
        One value per mutational distance layer.
    """
    array_jax = jnp.asarray(array)
    range_cube = generate_range_cube(array_jax.shape)
    return jnp.stack([
        (array_jax * (range_cube == distance_layer)).sum()
        for distance_layer in range(array_jax.ndim + 1)
    ])

def _get_landscape_spectrum_jax(
    landscape: Array,
    norm: bool=False,
    remove_constant: bool=True,
) -> jax.Array:
    """Collapse landscape Fourier power by mutational distance layer with JAX.

    Parameters:
    - landscape: Array
        N-dimensional fitness landscape.
    - norm: bool
        Whether to L2-normalize the collapsed spectrum.
    - remove_constant: bool
        Whether to drop the zero-frequency coefficient.

    Returns:
    - jax.Array
        Collapsed real-valued spectrum.
    """
    landscape_array = jnp.asarray(landscape)
    landy_fft = fftn_jax(landscape_array)
    spectrum = landy_fft * jnp.conj(landy_fft)
    collapsed_spectrum = jnp.real(_collapse_range_jax(spectrum))
    if remove_constant:
        collapsed_spectrum = collapsed_spectrum[1:]
    if norm:
        collapsed_spectrum = collapsed_spectrum / (collapsed_spectrum * collapsed_spectrum).sum() ** 0.5
    return collapsed_spectrum

def get_spectral_entropy(landscape: Array, remove_constant: bool=True, on_gpu: bool=False) -> jax.Array:
    """Compute normalized entropy of the collapsed Fourier spectrum.

    Parameters:
    - landscape: Array
        N-dimensional fitness landscape.
    - remove_constant: bool
        Whether to omit the zero-frequency coefficient.
    - on_gpu: bool
        Retained for API compatibility; the public implementation uses JAX.

    Returns:
    - jax.Array
        Normalized spectral entropy.
    """
    spectrum = _get_landscape_spectrum_jax(landscape, norm=True, remove_constant=remove_constant)
    p = spectrum / spectrum.sum()
    spectral_entropy = -jnp.sum(p * jnp.log(p + 1e-10)) / jnp.log(len(spectrum))
    return spectral_entropy

def get_dirichlet_metric(landscape: Array, on_gpu: bool=False) -> jax.Array:
    """Compute the normalized Dirichlet metric from the Fourier spectrum.

    Parameters:
    - landscape: Array
        N-dimensional fitness landscape.
    - on_gpu: bool
        Retained for API compatibility; the public implementation uses JAX.

    Returns:
    - jax.Array
        Weighted spectral roughness metric.
    """
    landscape_array = jnp.asarray(landscape)
    spectrum = _get_landscape_spectrum_jax(landscape_array, norm=True, remove_constant=True)
    num_alleles = landscape_array.shape[0]
    num_sites = landscape_array.ndim
    max_distance = num_sites * (num_alleles - 1)
    indices = jnp.arange(1, num_sites + 1)
    return jnp.sum(num_alleles * indices * spectrum) / spectrum.sum() / max_distance
