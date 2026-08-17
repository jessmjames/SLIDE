from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.optimize import curve_fit
Array = jax.Array | np.ndarray
Numeric = int | float | np.number
Shape = tuple[int, ...] | list[int]
FitnessFunction = Callable[[jax.Array], jax.Array]
MutationFunction = Callable[[jax.Array, jax.Array], jax.Array]
SelectionShape = Callable[[jax.Array, Mapping[str, float]], jax.Array]
SelectionFunction = Callable[[jax.Array, jax.Array], jax.Array]
ExtraInfoFn = Callable[..., jax.Array]

default_extra_info: dict[str, ExtraInfoFn] = {'fitness': lambda *args, **kwargs: kwargs['fitnesses'], 'pop': lambda *args, **kwargs: kwargs['pop']}

def run_directed_evolution(rng: jax.Array, i_pop: Array, selection_function: SelectionFunction, mutation_function: MutationFunction, fitness_function: FitnessFunction, num_options: int=2, fitness_noise: float=0.0001, num_steps: int=30, extra_function_dict: Mapping[str, ExtraInfoFn]=default_extra_info) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Run mutation-selection evolution for a fixed number of generations.

    Parameters:
    - rng: jax.Array
        JAX PRNG key used to seed each generation.
    - i_pop: Array
        Initial population with shape ``(popsize, n_sites)``.
    - selection_function: SelectionFunction
        Callable that samples selected population indices from fitness values.
    - mutation_function: MutationFunction
        Callable that mutates a population.
    - fitness_function: FitnessFunction
        Callable that maps a population to fitness values.
    - num_options: int
        Number of allelic states per site.
    - fitness_noise: float
        Relative Gaussian noise applied to fitness values before selection.
    - num_steps: int
        Number of mutation-selection generations.
    - extra_function_dict: Mapping[str, ExtraInfoFn]
        Per-generation summary functions keyed by output name.

    Returns:
    - tuple[jax.Array, dict[str, jax.Array]]
        Final population and scanned per-generation history arrays.
    """

    def single_iteration(rng: jax.Array, population: jax.Array, selection_function: SelectionFunction, mutation_function: MutationFunction, fitness_function: FitnessFunction, num_options: int, fitness_noise: float=0.0001, extra_function_dict: Mapping[str, ExtraInfoFn] | None=None) -> tuple[jax.Array, dict[str, jax.Array]]:
        r1, r2, r3 = jr.split(rng, 3)
        population_fitness = fitness_function(population)
        fitness_noise = population_fitness * jr.normal(r1, population_fitness.shape) * fitness_noise
        population_fitness = population_fitness + fitness_noise
        selected_population = selection_function(r2, population_fitness)
        resamped_pop = population[selected_population]
        mutated_pop = mutation_function(r3, resamped_pop)
        current_info_dict = {'fitnesses': population_fitness, 'pop': population}
        extra_info = {key: func(**current_info_dict) for key, func in extra_function_dict.items()}
        return (mutated_pop, extra_info)

    def multiple_iterations(population: jax.Array, local_rng: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        new_pop, extra_info = single_iteration(local_rng, population, selection_function, mutation_function, fitness_function, num_options=num_options, fitness_noise=fitness_noise, extra_function_dict=extra_function_dict)
        return (new_pop, extra_info)
    rng_array = jr.split(rng, num_steps)
    return jax.lax.scan(multiple_iterations, i_pop, rng_array)

def run_diffusion(rng: jax.Array, i_pop: Array, mutation_function: MutationFunction, fitness_function: FitnessFunction, fitness_noise: float=0.0001, num_steps: int=30, extra_function_dict: Mapping[str, ExtraInfoFn]=default_extra_info) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Run mutation-only diffusion for a fixed number of generations.

    Parameters:
    - rng: jax.Array
        JAX PRNG key used to seed each generation.
    - i_pop: Array
        Initial population with shape ``(popsize, n_sites)``.
    - mutation_function: MutationFunction
        Callable that mutates a population.
    - fitness_function: FitnessFunction
        Callable that maps a population to fitness values.
    - fitness_noise: float
        Relative Gaussian noise applied to recorded fitness values.
    - num_steps: int
        Number of mutation-only generations.
    - extra_function_dict: Mapping[str, ExtraInfoFn]
        Per-generation summary functions keyed by output name.

    Returns:
    - tuple[jax.Array, dict[str, jax.Array]]
        Final population and scanned per-generation history arrays.
    """

    def single_iteration(rng: jax.Array, population: jax.Array, mutation_function: MutationFunction, fitness_function: FitnessFunction, fitness_noise: float=0.0001, extra_function_dict: Mapping[str, ExtraInfoFn] | None=None) -> tuple[jax.Array, dict[str, jax.Array]]:
        r1, r2 = jr.split(rng, 2)
        population_fitness = fitness_function(population)
        noise = population_fitness * jr.normal(r1, population_fitness.shape) * fitness_noise
        population_fitness = population_fitness + noise
        mutated_pop = mutation_function(r2, population)
        current_info_dict = {'fitnesses': population_fitness, 'pop': population}
        extra_info = {key: func(**current_info_dict) for key, func in extra_function_dict.items()}
        return (mutated_pop, extra_info)

    def multiple_iterations(population: jax.Array, local_rng: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        new_pop, extra_info = single_iteration(local_rng, population, mutation_function, fitness_function, fitness_noise=fitness_noise, extra_function_dict=extra_function_dict)
        return (new_pop, extra_info)
    rng_array = jr.split(rng, num_steps)
    return jax.lax.scan(multiple_iterations, i_pop, rng_array)

def build_empirical_landscape_function(landscape: Array) -> FitnessFunction:
    """Build a vectorized fitness lookup for an empirical landscape.

    Parameters:
    - landscape: Array
        N-dimensional fitness array indexed by genotype coordinates.

    Returns:
    - FitnessFunction
        Jitted function mapping a population of genotype coordinates to fitnesses.
    """

    def get_fitness(i: jax.Array) -> jax.Array:
        return landscape[tuple(i)]
    return jax.jit(jax.vmap(get_fitness))
GB1_acid_start = jnp.array([3, 17, 0, 3], dtype=jnp.int32)
GB1_codon_start = jnp.array([3, 0, 0, 3, 0, 2, 3, 0, 3, 3, 0, 0], dtype=jnp.int32)
CODON_MAPPER = jnp.array([[[8, 18, 9, 7], [8, 18, 9, 7], [4, 18, -1, -1], [4, 18, -1, 10]], [[4, 1, 11, 13], [4, 1, 11, 13], [4, 1, 14, 13], [4, 1, 14, 13]], [[5, 19, 15, 18], [5, 19, 15, 18], [5, 19, 12, 13], [6, 19, 12, 13]], [[3, 2, 17, 0], [3, 2, 17, 0], [3, 2, 16, 0], [3, 2, 16, 0]]], dtype=jnp.int32)
INVERSE_CODON_MAPPER = jnp.array([[3, 0, 3], [1, 0, 1], [3, 0, 1], [3, 0, 0], [0, 2, 0], [2, 0, 0], [2, 3, 0], [0, 0, 3], [0, 0, 0], [0, 0, 2], [0, 3, 3], [1, 0, 2], [2, 2, 2], [1, 0, 3], [1, 2, 2], [2, 0, 2], [3, 2, 2], [3, 0, 2], [0, 0, 1], [2, 0, 1]], dtype=jnp.int32)
INVERSE_CODON_MAPER = INVERSE_CODON_MAPPER

def inverse_codon(codon: Array) -> jax.Array:
    """Map amino-acid indices to representative nucleotide codons.

    Parameters:
    - codon: Array
        Amino-acid index or array of indices.

    Returns:
    - jax.Array
        Codon triplets, with invalid indices mapped to ``[-1, -1, -1]``.
    """
    codon = jnp.asarray(codon)
    max_index = INVERSE_CODON_MAPPER.shape[0] - 1
    valid = (codon >= 0) & (codon <= max_index)
    safe_codon = jnp.clip(codon, 0, max_index)
    return jnp.where(valid[..., None], INVERSE_CODON_MAPPER[safe_codon], -1)

def get_pre_defined_landscape_function_with_codon(landscape: Array) -> FitnessFunction:
    """Build a codon-level lookup for an amino-acid landscape.

    Parameters:
    - landscape: Array
        Amino-acid fitness landscape with one axis per amino-acid site.

    Returns:
    - FitnessFunction
        Jitted function mapping codon-encoded populations to fitness values.
    """
    n = len(landscape.shape)
    min_fitness = jnp.min(landscape)
    buffered_landscape = jnp.pad(landscape, [(0, 1)] * n, constant_values=min_fitness)

    def get_codon(i: jax.Array) -> jax.Array:
        return CODON_MAPPER[tuple(i)]
    vmapped_get_codons = jax.jit(jax.vmap(get_codon))

    def get_fitties(params: jax.Array) -> jax.Array:
        parries_reshaped = jnp.reshape(params, (-1, 3))
        codon_set = vmapped_get_codons(parries_reshaped)
        return buffered_landscape[tuple(codon_set)]
    return jax.jit(jax.vmap(get_fitties))

def get_pd_landscape_function_codon_masked(landscape: Array, mask: Array, replacement: Array) -> FitnessFunction:
    """Build a masked codon-level lookup for an amino-acid landscape.

    Parameters:
    - landscape: Array
        Amino-acid fitness landscape with one axis per amino-acid site.
    - mask: Array
        Boolean mask indicating codon positions to replace before lookup.
    - replacement: Array
        Replacement codon values used where ``mask`` is true.

    Returns:
    - FitnessFunction
        Jitted function mapping masked codon-encoded populations to fitness values.
    """
    n = len(landscape.shape)
    min_fitness = jnp.min(landscape)
    buffered_landscape = jnp.pad(landscape, [(0, 1)] * n, constant_values=min_fitness)

    def get_codon(i: jax.Array) -> jax.Array:
        return CODON_MAPPER[tuple(i)]
    vmapped_get_codons = jax.jit(jax.vmap(get_codon))

    def get_fitties(params: jax.Array) -> jax.Array:
        new_params = jnp.where(mask, replacement, params)
        parries_reshaped = jnp.reshape(new_params, (-1, 3))
        codon_set = vmapped_get_codons(parries_reshaped)
        return buffered_landscape[tuple(codon_set)]
    return jax.jit(jax.vmap(get_fitties))

def convert_landscape_function_to_codon(landscape_function: FitnessFunction, stop_codon_strategy: str='min_fitness', stop_codon_value: float | None=None, stop_codon_index: int=20, sample_size: int=256, sample_extra: float=0.0, rng: jax.Array=jr.PRNGKey(0)) -> FitnessFunction:
    """Wrap an amino-acid fitness function for codon-encoded populations.

    Parameters:
    - landscape_function: FitnessFunction
        Function that accepts amino-acid encoded populations.
    - stop_codon_strategy: str
        Strategy for handling stop codons.
    - stop_codon_value: float | None
        Fitness value used by the ``fill_fitness`` stop-codon strategy.
    - stop_codon_index: int
        Amino-acid index used when treating stop codons as an extra amino acid.
    - sample_size: int
        Number of random samples used by sampling-based stop-codon strategies.
    - sample_extra: float
        Offset added to sampled minimum fitness values.
    - rng: jax.Array
        JAX PRNG key for sampling-based stop-codon handling.

    Returns:
    - FitnessFunction
        Function mapping codon-encoded populations to fitness values.
    """
    valid_strategies = {'min_fitness', 'fill_fitness', 'another_amino_acid', 'sample_min_fitness'}
    if stop_codon_strategy not in valid_strategies:
        raise ValueError(f'stop_codon_strategy must be one of {sorted(valid_strategies)} (got {stop_codon_strategy}).')
    if stop_codon_strategy == 'fill_fitness' and stop_codon_value is None:
        raise ValueError('stop_codon_value must be provided for fill_fitness.')

    def convert_gene(gene: jax.Array) -> jax.Array:
        reshaped = gene.reshape(-1, 3)
        codon_set = CODON_MAPPER[tuple(reshaped.T)]
        return codon_set
    vmapped_convert_gene = jax.jit(jax.vmap(convert_gene))

    def codon_landscape_function(pop: jax.Array) -> jax.Array:
        codon_genes = vmapped_convert_gene(pop)
        return landscape_function(codon_genes)
    return codon_landscape_function

def build_NK_landscape_function(rng: jax.Array, N: int, K: int, fitness_distribution: Callable[[jax.Array], jax.Array]=jr.normal) -> FitnessFunction:
    """Build a vectorized random NK landscape fitness function.

    Parameters:
    - rng: jax.Array
        JAX PRNG key used to generate interactions and site contributions.
    - N: int
        Number of sites in the genotype.
    - K: int
        Number of interacting partner sites per site.
    - fitness_distribution: Callable[[jax.Array], jax.Array]
        Random distribution used for site-level fitness contributions.

    Returns:
    - FitnessFunction
        Jitted function mapping NK genotype populations to fitness values.
    """
    r1, r2, r3 = jr.split(rng, 3)
    base_row = 1 * (jnp.arange(N - 1) < K)

    def permutate_rows(rng: jax.Array, i: jax.Array) -> jax.Array:
        perm_row = jr.permutation(rng, base_row)
        return jnp.insert(perm_row, i, 1.0)
    permutate_rows = jax.vmap(permutate_rows)
    interaction_matrix = permutate_rows(jr.split(r1, N), jnp.arange(N))
    vector_foldin = jax.vmap(lambda base_rng, data: jr.fold_in(base_rng, data))
    fitness_distribution = jax.vmap(fitness_distribution)

    def get_fitness(gene: jax.Array) -> jax.Array:
        individual_site_fitness = vector_foldin(jr.split(r2, N), gene)
        interaction_fitness = interaction_matrix @ individual_site_fitness + jr.split(r3, N)
        return jnp.sum(fitness_distribution(interaction_fitness))
    return jax.jit(jax.vmap(get_fitness))

def build_mutation_function(mutation_chance: float, num_options: int=2) -> MutationFunction:
    """Build a uniform per-site mutation function.

    Parameters:
    - mutation_chance: float
        Probability that each site mutates in one generation.
    - num_options: int
        Number of possible allelic states per site.

    Returns:
    - MutationFunction
        Function that mutates a population array.
    """

    def mutation_function(rng: jax.Array, pop: jax.Array) -> jax.Array:
        r1, r2 = jr.split(rng, 2)
        pshape = pop.shape
        has_mutation = jr.bernoulli(r1, mutation_chance, pshape)
        mut_delta = jr.randint(r2, pshape, 1, num_options)
        return (pop + has_mutation * mut_delta) % num_options
    return mutation_function

def build_custom_mutation_function(mutation_chance: float, transition_matrix: Array, A: int | None=None) -> MutationFunction:
    """Build a per-site mutation function from a transition matrix.

    Parameters:
    - mutation_chance: float
        Probability that each site mutates in one generation.
    - transition_matrix: Array
        Row-stochastic transition probabilities between allelic states.
    - A: int | None
        Number of allelic states, inferred from ``transition_matrix`` when omitted.

    Returns:
    - MutationFunction
        Function that mutates a population according to transition probabilities.
    """
    if A is None:
        A = transition_matrix.shape[0]
    transition_matrix = jnp.array(transition_matrix)

    def mutation_function(rng: jax.Array, pop: jax.Array) -> jax.Array:
        r1, r2 = jr.split(rng, 2)
        pshape = pop.shape
        has_mutation = jr.bernoulli(r1, mutation_chance, pshape)
        flat_pop = pop.flatten()
        pop_size = flat_pop.shape[0]

        def mutate_site(rng: jax.Array, aa: jax.Array) -> jax.Array:
            return jr.choice(rng, jnp.arange(A), p=transition_matrix[aa])
        flat_mutated_pop = jax.vmap(mutate_site)(jr.split(r2, pop_size), flat_pop)
        mutated_pop = flat_mutated_pop.reshape(pshape)
        return jnp.where(has_mutation, mutated_pop, pop)
    return mutation_function

def build_selection_function(selection_function_shape: SelectionShape, params: Mapping[str, float]) -> SelectionFunction:
    """Build a stochastic selection sampler from a probability shape function.

    Parameters:
    - selection_function_shape: SelectionShape
        Callable mapping fitness values and parameters to selection probabilities.
    - params: Mapping[str, float]
        Parameters passed to ``selection_function_shape``.

    Returns:
    - SelectionFunction
        Function that samples selected population indices from fitness values.
    """

    def selection_function(rng: jax.Array, fitnesses: jax.Array, state: int=0) -> jax.Array:
        psize = fitnesses.shape[-1]
        selection_prob = selection_function_shape(fitnesses, params)
        selected = jnp.ones((psize,)) * jr.bernoulli(rng, p=selection_prob, shape=(psize,))
        return jr.choice(rng, jnp.arange(psize), (psize,), p=selected)
    return selection_function

def param_sampler(*ranges: tuple[float, float], rng: jax.Array, num_samples: int=10) -> jax.Array:
    """Sample shuffled parameter values from one or more numeric ranges.

    Parameters:
    - *ranges: tuple[float, float]
        Inclusive lower and upper bounds for each sampled parameter.
    - rng: jax.Array
        JAX PRNG key used to shuffle sampled values.
    - num_samples: int
        Number of samples per parameter.

    Returns:
    - jax.Array
        Array of sampled parameters with one row per input range.
    """
    samples = jnp.array([jnp.linspace(range[0], range[1], num=num_samples) for range in ranges])
    return jr.permutation(rng, samples, axis=1, independent=True)

def grid_sampler(x: tuple[float, float], y: tuple[float, float], num_samples: int=10) -> jax.Array:
    """Create a flattened two-dimensional parameter grid.

    Parameters:
    - x: tuple[float, float]
        Lower and upper bounds for the first parameter.
    - y: tuple[float, float]
        Lower and upper bounds for the second parameter.
    - num_samples: int
        Number of samples along each axis.

    Returns:
    - jax.Array
        Array containing flattened grid coordinates for ``x`` and ``y``.
    """
    x = jnp.linspace(x[0], x[1], num=num_samples)
    y = jnp.linspace(y[0], y[1], num=num_samples)
    X, Y = jnp.meshgrid(x, y)
    return jnp.array([X.flatten(), Y.flatten()])

def base_chance_threshold_fixed_prop(base_chance_range: tuple[float, float], proportion: float, num_samples: int=10) -> jax.Array:
    """Create threshold/base-chance samples for a fixed selected proportion.

    Parameters:
    - base_chance_range: tuple[float, float]
        Lower and upper bounds for baseline selection chance.
    - proportion: float
        Target selected fraction used to derive thresholds.
    - num_samples: int
        Number of samples to generate.

    Returns:
    - jax.Array
        Two-row array containing thresholds and base chances.
    """

    def base_chance_threshold_integral(base_chance: jax.Array, proportion: float) -> jax.Array:
        return (1 - proportion) / (1 - base_chance)
    base_chance_range_relevant = [base_chance_range[0], min(proportion, base_chance_range[1])]
    base_chance_samples = jnp.linspace(base_chance_range_relevant[0], base_chance_range_relevant[1], num=num_samples)
    threshold_samples = base_chance_threshold_integral(base_chance_samples, proportion)
    return jnp.array([threshold_samples, base_chance_samples])

def model_function(x: np.ndarray, *params: float, mut: float | np.ndarray=0.1) -> np.ndarray:
    """Evaluate the exponential decay model used by ``get_single_decay_rate``.

    Parameters:
    - x: np.ndarray
        Step positions at which to evaluate the model.
    - *params: float
        Decay-rate parameters followed by the fitted asymptote.
    - mut: float | np.ndarray
        Mutation scale used in the exponent.

    Returns:
    - np.ndarray
        Model values at ``x``.
    """
    num_params = 1
    constant = params[-1]
    params = params[:-1]
    mut_curves = np.exp(-1.0 * mut * x[:, None] * np.array(params)[None, :])
    weights = np.linspace(0.1, 0.9, num_params)
    weights = np.ones(num_params)
    weights = weights / weights.sum()
    sum_curves = np.sum(mut_curves * weights[None, :], axis=1)
    return sum_curves * (1 - constant) + constant

def get_single_decay_rate(decay_data: np.ndarray, mut: float | np.ndarray | None=0.1, num_steps: int=25) -> tuple[float, float]:
    """Fit a single exponential decay rate to normalized decay data.

    Parameters:
    - decay_data: np.ndarray
        One-dimensional decay trajectory.
    - mut: float | np.ndarray | None
        Mutation scale or explicit step positions.
    - num_steps: int
        Number of steps when ``mut`` is a scalar.

    Returns:
    - tuple[float, float]
        Fitted decay rate and fitted asymptote.
    """
    num_params = 1
    decay_data = decay_data / decay_data[0]
    if isinstance(mut, (int, float, complex)) or jnp.ndim(mut) == 0:
        steps = np.linspace(0, num_steps - 1, num_steps)
    else:
        steps = mut
    if mut is None:
        mut = np.arange(len(decay_data))
    asymptote_guess = decay_data[-3:].mean() / decay_data[0]
    lower_bound = min(0.0, asymptote_guess - 0.2)
    upper_bound = max(1.1, asymptote_guess + 0.2)
    init_guess = np.concatenate([np.linspace(0.1, 0.9, num_params), [asymptote_guess]])
    lbounds = [0.0] * num_params + [lower_bound]
    ubounds = [2.0] * num_params + [upper_bound]
    model = lambda x, *params: model_function(x, *params, mut=mut)
    params, _ = curve_fit(model, steps, decay_data, p0=init_guess, maxfev=9000, ftol=0.0001, xtol=1e-05, bounds=(lbounds, ubounds))
    mean_params = np.mean(params[:-1])
    fitted_constant = params[-1]
    return (mean_params, fitted_constant)

def model_function_IK(x: np.ndarray, *params: float, mut: float | np.ndarray=0.1, y0: float=1) -> np.ndarray:
    """Evaluate the IK exponential decay model with a configurable initial value.

    Parameters:
    - x: np.ndarray
        Step positions at which to evaluate the model.
    - *params: float
        Decay-rate parameters followed by the fitted asymptote.
    - mut: float | np.ndarray
        Mutation scale used in the exponent.
    - y0: float
        Initial model value.

    Returns:
    - np.ndarray
        Model values at ``x``.
    """
    num_params = 1
    constant = params[-1]
    params = params[:-1]
    mut_curves = np.exp(-1.0 * mut * x[:, None] * np.array(params)[None, :])
    weights = np.linspace(0.1, 0.9, num_params)
    weights = np.ones(num_params)
    weights = weights / weights.sum()
    sum_curves = np.sum(mut_curves * weights[None, :], axis=1)
    return sum_curves * (y0 - constant) + constant

def get_single_decay_rate_IK(decay_data: np.ndarray, mut: float | np.ndarray | None=0.1, num_steps: int=25) -> tuple[float, float]:
    """Fit the IK exponential decay model to a decay trajectory.

    Parameters:
    - decay_data: np.ndarray
        One-dimensional decay trajectory.
    - mut: float | np.ndarray | None
        Mutation scale or explicit step positions.
    - num_steps: int
        Number of steps when ``mut`` is a scalar.

    Returns:
    - tuple[float, float]
        Fitted decay rate and fitted asymptote.
    """
    num_params = 1
    decay_data = decay_data
    if isinstance(mut, (int, float, complex)) or jnp.ndim(mut) == 0:
        steps = np.linspace(0, num_steps - 1, num_steps)
    else:
        steps = mut
    if mut is None:
        mut = np.arange(len(decay_data))
    asymptote_guess = decay_data[-int(np.round(0.2 * len(decay_data))):].mean()
    lower_bound = 0.8 * asymptote_guess
    upper_bound = 1.2 * asymptote_guess
    nd = 2
    inds_a = np.arange(0, 2 * nd, 1, dtype=int)
    mid_ab = int(np.round(num_steps / 2))
    inds_b = np.arange(mid_ab - nd, mid_ab + nd + 1, 1, dtype=int)
    y_a = decay_data[inds_a].mean()
    y_b = decay_data[inds_b].mean()
    if y_a > asymptote_guess and y_b > asymptote_guess and (y_a > y_b):
        init_guess_rho = -np.log((y_b - asymptote_guess) / (y_a - asymptote_guess)) / mid_ab / mut
        if init_guess_rho < 0.1 or init_guess_rho > 2.5:
            init_guess_rho = 0.6
        init_guess = np.concatenate([[init_guess_rho], [asymptote_guess]])
    else:
        init_guess = np.concatenate([np.linspace(0.1, 0.9, num_params), [asymptote_guess]])
    lbounds = [0.1] * num_params + [lower_bound]
    ubounds = [3] * num_params + [upper_bound]
    decay_start = decay_data[0]
    model = lambda x, *params: model_function_IK(x, *params, mut=mut, y0=decay_start)
    params, _ = curve_fit(model, steps, decay_data, p0=init_guess, maxfev=9000, ftol=0.0001, xtol=1e-05, bounds=(lbounds, ubounds))
    mean_params = np.mean(params[:-1])
    fitted_constant = params[-1]
    return (mean_params, fitted_constant)

def model_function_IK_v2(x: np.ndarray, *params: float, mut: float | np.ndarray=0.1, fix_amplitude: bool=False, F0: float | None=1) -> np.ndarray:
    """Evaluate a one-rate exponential model with optional fixed amplitude.

    Parameters:
    - x: np.ndarray
        Step positions at which to evaluate the model.
    - *params: float
        ``rho`` plus amplitude and asymptote, or ``rho`` plus asymptote when the
        amplitude is fixed.
    - mut: float | np.ndarray
        Mutation scale used in the exponent.
    - fix_amplitude: bool
        Whether to derive the amplitude from ``F0`` and the asymptote.
    - F0: float | None
        Initial value used when fixing the amplitude.

    Returns:
    - np.ndarray
        Model values at ``x``.
    """
    rho = params[0]
    c = params[-1]
    C = params[1] if fix_amplitude == False else F0 - c
    return C * np.exp(-1.0 * mut * x * rho) + c

def get_single_decay_rate_IK_v2(decay_data: np.ndarray, mut: float=0.1, num_steps: int=25, fix_amplitude: bool=False) -> tuple[float, float, float]:
    """Fit the IK v2 exponential decay model to a decay trajectory.

    Parameters:
    - decay_data: np.ndarray
        One-dimensional decay trajectory.
    - mut: float
        Mutation scale used in the exponent.
    - num_steps: int
        Number of sampled generations.
    - fix_amplitude: bool
        Whether to fit only ``rho`` and asymptote.

    Returns:
    - tuple[float, float, float]
        Fitted ``rho``, amplitude, and asymptote.
    """
    if fix_amplitude:
        F0 = decay_data[0]
    else:
        F0 = None
    steps = np.arange(0, num_steps)
    relb = 0.9
    asymptote_guess = decay_data[-int(np.round(0.2 * len(decay_data))):].mean()
    lb_asymptote = (1 - np.sign(asymptote_guess) * relb) * asymptote_guess
    ub_asymptote = (1 + np.sign(asymptote_guess) * relb) * asymptote_guess
    if not fix_amplitude:
        amplitude_guess = decay_data[0] - asymptote_guess
        lb_amplitude = (1 - np.sign(amplitude_guess) * relb) * amplitude_guess
        ub_amplitude = (1 + np.sign(amplitude_guess) * relb) * amplitude_guess + 0.1
    rho_guess = 0.5
    lb_rho = 0.1
    ub_rho = 5
    mid_ab = int(np.round(num_steps / 2))
    y_a = decay_data[0]
    y_b = decay_data[mid_ab]
    if y_a > asymptote_guess and y_b > asymptote_guess and (y_a > y_b):
        rho_guess = -np.log((y_b - asymptote_guess) / (y_a - asymptote_guess)) / mid_ab / mut
        if rho_guess < lb_rho or rho_guess > ub_rho:
            rho_guess = 0.5 * (lb_rho + ub_rho)
    if fix_amplitude:
        init_guess = np.array([rho_guess, asymptote_guess])
        lbounds = [lb_rho, lb_asymptote]
        ubounds = [ub_rho, ub_asymptote]
    else:
        init_guess = np.array([rho_guess, amplitude_guess, asymptote_guess])
        lbounds = [lb_rho, lb_amplitude, lb_asymptote]
        ubounds = [ub_rho, ub_amplitude, ub_asymptote]
    model = lambda x, *params: model_function_IK_v2(x, *params, mut=mut, fix_amplitude=fix_amplitude, F0=F0)
    params, _ = curve_fit(model, steps, decay_data, p0=init_guess, maxfev=9000, ftol=0.0001, xtol=1e-05, bounds=(lbounds, ubounds))
    rho = params[0]
    c = params[-1]
    C = params[1] if fix_amplitude == False else F0 - c
    return (rho, C, c)
