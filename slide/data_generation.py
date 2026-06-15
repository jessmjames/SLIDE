"""Reusable raw-data generation helpers for the SLIDE notebook pipeline.

The executable orchestration lives in ``data_generation.ipynb``. This module
keeps reusable simulation kernels, start samplers, product filename constants,
and small registry helpers used by that notebook and by ``data_processing.ipynb``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from . import selection_function_library as slct
from .direvo_functions import (
    base_chance_threshold_fixed_prop,
    build_NK_landscape_function,
    build_empirical_landscape_function,
    build_mutation_function,
    build_selection_function,
    run_diffusion,
    run_directed_evolution,
)
from .utils import get_landscape_arrays_dir, load_pickle, raw_path


EMPIRICAL_NAMES: tuple[str, ...] = ("GB1", "TrpB", "TEV", "ParD3")
GENERATION_STEPS: tuple[int, ...] = (5, 25, 50, 75, 100, 500, 1000)

EMPIRICAL_LANDSCAPE_FILES: dict[str, str] = {
    "GB1": "GB1_landscape_array.pkl",
    "TrpB": "TrpB_landscape_array.pkl",
    "TEV": "TEV_landscape_array.pkl",
    "ParD3": "E3_landscape_array.pkl",
    "E3": "E3_landscape_array.pkl",
}


def load_empirical_landscape(name: str) -> np.ndarray:
    """Load an empirical landscape array by short name.

    Parameters:
    - name: str
        Landscape key, one of ``GB1``, ``TrpB``, ``TEV``, ``ParD3``, or the file-level alias ``E3``.

    Returns:
    - np.ndarray
        The empirical fitness landscape as a NumPy array.
    """

    return np.asarray(load_pickle(get_landscape_arrays_dir() / EMPIRICAL_LANDSCAPE_FILES[name]))


def all_start_locs(landscape: np.ndarray) -> np.ndarray:
    """Return every genotype coordinate in an empirical landscape.

    Parameters:
    - landscape: np.ndarray
        N-dimensional empirical fitness array.

    Returns:
    - np.ndarray
        Integer array of shape ``(landscape.size, landscape.ndim)`` containing all coordinates in row-major order.
    """

    return np.column_stack(np.unravel_index(np.arange(landscape.size), landscape.shape)).astype(np.int32)


def uniform_start_locs(
    landscape: np.ndarray,
    *,
    num_starts: int = 10000,
    seed: int = 42,
    replace: bool = False,
) -> np.ndarray:
    """Sample starting genotypes uniformly from an empirical landscape.

    Parameters:
    - landscape: np.ndarray
        N-dimensional empirical fitness array.
    - num_starts: int
        Number of starting coordinates to sample.
    - seed: int
        NumPy random seed for reproducible sampling.
    - replace: bool
        Whether the same coordinate may be sampled more than once.

    Returns:
    - np.ndarray
        Integer coordinate array with shape ``(num_starts, landscape.ndim)``.
    """

    rng = np.random.default_rng(seed)
    flat_indices = rng.choice(landscape.size, size=num_starts, replace=replace)
    return np.column_stack(np.unravel_index(flat_indices, landscape.shape)).astype(np.int32)


def evenly_spaced_start_locs(landscape: np.ndarray, *, num_starts: int = 10000) -> np.ndarray:
    """Choose deterministic, evenly spaced starting coordinates.

    Parameters:
    - landscape: np.ndarray
        N-dimensional empirical fitness array.
    - num_starts: int
        Number of coordinates to return.

    Returns:
    - np.ndarray
        Integer coordinate array with approximately even coverage of flattened landscape indices.
    """

    flat_indices = np.round(np.linspace(0, landscape.size - 1, num_starts)).astype(int)
    return np.column_stack(np.unravel_index(flat_indices, landscape.shape)).astype(np.int32)


def percentile_start_locs(landscape: np.ndarray, *, num_starts: int = 10, percentile: float = 99) -> np.ndarray:
    """Choose starts closest to a high-fitness percentile threshold.

    Parameters:
    - landscape: np.ndarray
        N-dimensional empirical fitness array.
    - num_starts: int
        Number of starting coordinates to return.
    - percentile: float
        Fitness percentile used as the target threshold.

    Returns:
    - np.ndarray
        Integer coordinate array of starts near the requested percentile.
    """

    flat = landscape.ravel()
    threshold = np.percentile(flat, percentile)
    top_indices = np.nonzero(flat >= threshold)[0]
    distances = flat[top_indices] - threshold
    closest = np.argsort(distances)[:num_starts]
    flat_indices = top_indices[closest]
    return np.column_stack(np.unravel_index(flat_indices, landscape.shape)).astype(np.int32)


def repeated_population(start: np.ndarray, popsize: int) -> jnp.ndarray:
    """Create a clonal population from one starting genotype.

    Parameters:
    - start: np.ndarray
        One genotype coordinate.
    - popsize: int
        Number of population members.

    Returns:
    - jnp.ndarray
        JAX integer array of shape ``(popsize, len(start))``.
    """

    start_array = jnp.asarray(start, dtype=jnp.int32)
    return jnp.tile(start_array[None, :], (int(popsize), 1))


def random_start(rng_key: jax.Array, *, n_sites: int, num_alleles: int) -> np.ndarray:
    """Sample one random NK starting genotype.

    Parameters:
    - rng_key: jax.Array
        JAX random key used to sample the genotype.
    - n_sites: int
        Number of genotype sites.
    - num_alleles: int
        Number of allelic states per site.

    Returns:
    - np.ndarray
        Integer coordinate array with shape ``(n_sites,)``.
    """

    return np.asarray(jr.randint(rng_key, (n_sites,), 0, num_alleles), dtype=np.int32)


def _fitness_history(*, fitnesses: jax.Array, pop: jax.Array) -> jax.Array:
    """Return only per-generation fitness values from diffusion history.

    Parameters:
    - fitnesses: jax.Array
        Fitness values recorded for the current population.
    - pop: jax.Array
        Current population array supplied by the diffusion loop.

    Returns:
    - jax.Array
        Fitness values for the current population.
    """

    return fitnesses


_FITNESS_ONLY_HISTORY: dict[str, Callable[..., jax.Array]] = {"fitness": _fitness_history}


def run_empirical_diffusion_replicates(
    rng: jax.Array,
    landscape: np.ndarray,
    start: np.ndarray,
    *,
    popsize: int,
    mutation_rate: float,
    num_reps: int,
    num_steps: int,
) -> dict[str, jax.Array]:
    """Run mutation-only diffusion replicates on an empirical landscape.

    Parameters:
    - rng: jax.Array
        JAX random key used to seed replicate trajectories.
    - landscape: np.ndarray
        Empirical fitness landscape array.
    - start: np.ndarray
        Starting genotype coordinate.
    - popsize: int
        Population size for each replicate.
    - mutation_rate: float
        Per-site mutation probability used by the mutation function.
    - num_reps: int
        Number of independent replicate trajectories.
    - num_steps: int
        Number of mutation-only generations.

    Returns:
    - dict[str, jax.Array]
        A history dictionary with ``fitness`` and ``pop`` arrays. The leading dimension indexes replicates.
    """

    fitness_function = build_empirical_landscape_function(jnp.asarray(landscape))
    mutation_function = build_mutation_function(mutation_rate, landscape.shape[0])
    initial_population = repeated_population(start, popsize)
    rng_seeds = jr.split(rng, num_reps)
    vmapped = jax.jit(
        jax.vmap(
            lambda r: run_diffusion(
                r,
                initial_population,
                mutation_function,
                fitness_function=fitness_function,
                num_steps=num_steps,
            )[1]
        )
    )
    return vmapped(rng_seeds)


def generate_empirical_decay_curves(
    landscape: np.ndarray,
    *,
    mutation_rate: float,
    popsize: int,
    starts: np.ndarray,
    num_reps: int = 10,
    num_steps: int = 25,
    seed: int = 42,
    batch_size: int = 100,
) -> np.ndarray:
    """Generate empirical no-selection fitness decay curves for many starts.

    Parameters:
    - landscape: np.ndarray
        Empirical fitness landscape array.
    - mutation_rate: float
        Per-site mutation probability.
    - popsize: int
        Population size for each starting genotype.
    - starts: np.ndarray
        Integer coordinate array of starting genotypes.
    - num_reps: int
        Number of replicate diffusion trajectories per start.
    - num_steps: int
        Number of mutation-only generations.
    - seed: int
        Master JAX seed.
    - batch_size: int
        Number of starts to process per vectorized chunk.

    Returns:
    - np.ndarray
        NumPy array with mean fitness trajectories, indexed by start, replicate, and generation.
    """

    rng_seeds = jr.split(jr.PRNGKey(seed), len(starts))

    def run_start(args: tuple[jnp.ndarray, jax.Array]) -> jnp.ndarray:
        start, rng = args
        run = run_empirical_diffusion_replicates(
            rng,
            landscape,
            start,
            popsize=popsize,
            mutation_rate=mutation_rate,
            num_reps=num_reps,
            num_steps=num_steps,
        )
        return run["fitness"].mean(axis=-1)

    chunks = []
    num_chunks = max(1, int(np.ceil(len(starts) / batch_size)))
    for start_chunk, rng_chunk in zip(np.array_split(starts, num_chunks), np.array_split(rng_seeds, num_chunks)):
        chunks.append(jax.vmap(run_start)((jnp.asarray(start_chunk), jnp.asarray(rng_chunk))))
    return np.asarray(jnp.concatenate(chunks, axis=0))


def run_nk_diffusion_replicates(
    rng: jax.Array,
    *,
    n_sites: int,
    k: int,
    num_alleles: int,
    start: np.ndarray,
    popsize: int,
    mutation_rate: float,
    num_reps: int,
    num_steps: int,
) -> dict[str, jax.Array]:
    """Run mutation-only diffusion replicates on one NK landscape.

    Parameters:
    - rng: jax.Array
        JAX random key used for the NK landscape and replicate seeds.
    - n_sites: int
        Number of sites in the NK landscape.
    - k: int
        NK epistatic interaction parameter.
    - num_alleles: int
        Number of alleles per site.
    - start: np.ndarray
        Starting genotype coordinate.
    - popsize: int
        Population size for each replicate.
    - mutation_rate: float
        Per-site mutation probability.
    - num_reps: int
        Number of replicate trajectories.
    - num_steps: int
        Number of mutation-only generations.

    Returns:
    - dict[str, jax.Array]
        A history dictionary with ``fitness`` and ``pop`` arrays. The leading dimension indexes replicates.
    """

    fitness_function = build_NK_landscape_function(rng, n_sites, k, fitness_distribution=jr.normal)
    mutation_function = build_mutation_function(mutation_rate, num_alleles)
    initial_population = repeated_population(start, popsize)
    vmapped = jax.jit(
        jax.vmap(
            lambda r: run_diffusion(
                r,
                initial_population,
                mutation_function,
                fitness_function=fitness_function,
                num_steps=num_steps,
            )[1]
        )
    )
    return vmapped(jr.split(rng, num_reps))


def run_nk_start_averaged_diffusion(
    *,
    rng_key: jax.Array,
    trajectory_rng_key: jax.Array | None = None,
    n_sites: int,
    k: int,
    num_alleles: int,
    starts: np.ndarray,
    popsize: int,
    mutation_rate_per_site: float,
    num_reps_per_start: int,
    num_steps: int,
) -> np.ndarray:
    """Run start-resolved diffusion on one NK landscape with JAX batching.

    Parameters:
    - rng_key: jax.Array
        JAX random key used to build the NK landscape.
    - trajectory_rng_key: jax.Array | None
        Optional JAX random key used to derive replicate trajectory keys. If ``None``, ``rng_key`` is used.
    - n_sites: int
        Number of NK genotype sites.
    - k: int
        NK epistatic interaction parameter.
    - num_alleles: int
        Number of allelic states per site.
    - starts: np.ndarray
        Starting genotype coordinates with shape ``(num_starts, n_sites)``.
    - popsize: int
        Number of clonal population members per starting genotype.
    - mutation_rate_per_site: float
        Per-site mutation probability.
    - num_reps_per_start: int
        Number of independent diffusion replicates for each start.
    - num_steps: int
        Number of mutation-only generations.

    Returns:
    - np.ndarray
        Start-level mean fitness trajectories with shape ``(num_starts, num_steps)``.
    """

    starts_array = jnp.asarray(starts, dtype=jnp.int32)
    num_starts = int(starts_array.shape[0])
    fitness_function = build_NK_landscape_function(rng_key, n_sites, k)
    mutation_function = build_mutation_function(mutation_rate_per_site, num_alleles)
    initial_populations = jnp.repeat(starts_array[:, None, :], int(popsize), axis=1)
    if trajectory_rng_key is None:
        trajectory_rng_key = rng_key
    replicate_keys = jr.split(
        jr.fold_in(trajectory_rng_key, 10_000),
        num_starts * int(num_reps_per_start),
    ).reshape(num_starts, int(num_reps_per_start), 2)

    def run_one_replicate(initial_population: jax.Array, replicate_key: jax.Array) -> jax.Array:
        history = run_diffusion(
            replicate_key,
            initial_population,
            mutation_function,
            fitness_function=fitness_function,
            num_steps=num_steps,
            extra_function_dict=_FITNESS_ONLY_HISTORY,
        )[1]
        return history["fitness"].mean(axis=-1)

    run_one_start = jax.vmap(run_one_replicate, in_axes=(None, 0))
    run_all_starts = jax.jit(jax.vmap(run_one_start, in_axes=(0, 0)))
    replicate_curves = run_all_starts(initial_populations, replicate_keys)
    return np.asarray(replicate_curves.mean(axis=1), dtype=float)


def nk_uniform_start_locs(*, n_sites: int, num_alleles: int, num_starts: int) -> np.ndarray:
    """Return deterministic starting coordinates for an NK genotype space.

    Parameters:
    - n_sites: int
        Number of genotype sites.
    - num_alleles: int
        Number of alleles per site.
    - num_starts: int
        Number of starts to return.

    Returns:
    - np.ndarray
        Integer coordinate array with shape ``(num_starts, n_sites)``.
    """

    total = num_alleles**n_sites
    flat_indices = np.round(np.linspace(0, total - 1, num_starts)).astype(int)
    return np.column_stack(np.unravel_index(flat_indices, (num_alleles,) * n_sites)).astype(np.int32)


@partial(jax.jit, static_argnames=("n_sites", "num_alleles", "k", "popsize", "mutation_rate", "num_reps", "num_steps"))
def _nk_decay_for_starts(
    start_keys: jax.Array,
    start_coords: jax.Array,
    *,
    n_sites: int,
    num_alleles: int,
    k: int,
    popsize: int,
    mutation_rate: float,
    num_reps: int,
    num_steps: int,
) -> jax.Array:
    """Mean per-replicate NK diffusion decay for a batch of starts in one fused kernel.

    Each start uses its own NK landscape seeded by its ``start_key`` (matching the scalar
    path), and the same key seeds the replicate trajectories. Compiles once per
    ``(n_sites, k, ...)`` and is reused across batches.

    Parameters:
    - start_keys: jax.Array
        Per-start keys, shape ``(batch, 2)``; each seeds both the landscape and the replicates.
    - start_coords: jax.Array
        Per-start genotype coordinates, shape ``(batch, n_sites)``.
    - n_sites, num_alleles, k, popsize, mutation_rate, num_reps, num_steps
        Static simulation parameters (``mutation_rate`` is the per-site rate).

    Returns:
    - jax.Array
        Mean fitness trajectories, shape ``(batch, num_reps, num_steps)``.
    """

    mutation_function = build_mutation_function(mutation_rate, num_alleles)

    def one_start(start_key: jax.Array, start_coord: jax.Array) -> jax.Array:
        interaction_matrix, site_rng, offset_rng = nk_landscape_arrays(start_key, n_sites, k)

        def fitness_function(population: jax.Array) -> jax.Array:
            return nk_population_fitness(population, interaction_matrix, site_rng, offset_rng)

        initial_population = repeated_population(start_coord, popsize)
        rep_histories = jax.vmap(
            lambda r: run_diffusion(r, initial_population, mutation_function, fitness_function=fitness_function, num_steps=num_steps)[1]
        )(jr.split(start_key, num_reps))
        return rep_histories["fitness"].mean(axis=-1)

    return jax.vmap(one_start)(start_keys, start_coords)


def generate_nk_decay_curves(
    *,
    n_sites: int,
    num_alleles: int,
    k_values: Sequence[int],
    mutation_rate: float,
    popsize: int,
    num_starts: int,
    num_reps: int = 10,
    num_steps: int = 25,
    seed: int = 42,
    batch_size: int = 200,
) -> np.ndarray:
    """Generate no-selection NK fitness decay curves across K values.

    Parameters:
    - n_sites: int
        Number of NK sites.
    - num_alleles: int
        Number of alleles per site.
    - k_values: Sequence[int]
        NK ``K`` values to simulate.
    - mutation_rate: float
        Total mutation rate, divided by ``n_sites`` internally.
    - popsize: int
        Population size for each start.
    - num_starts: int
        Number of deterministic starts in genotype space.
    - num_reps: int
        Number of replicate trajectories per start.
    - num_steps: int
        Number of mutation-only generations.
    - seed: int
        Master JAX seed.
    - batch_size: int
        Number of starts processed per device call. The per-step diffusion history is large,
        so this defaults to 200; ``<= 0`` processes all starts at once.

    Returns:
    - np.ndarray
        NumPy array indexed by K value, start, replicate, and generation.
    """

    starts = jnp.asarray(nk_uniform_start_locs(n_sites=n_sites, num_alleles=num_alleles, num_starts=num_starts))
    master_keys = jr.split(jr.PRNGKey(seed), len(k_values))
    num_chunks = 1 if batch_size <= 0 or batch_size >= num_starts else int(np.ceil(num_starts / batch_size))
    results = []
    for k, key in zip(k_values, master_keys):
        start_keys = jr.split(key, num_starts)
        chunks = [
            np.asarray(
                _nk_decay_for_starts(
                    key_chunk,
                    coord_chunk,
                    n_sites=n_sites,
                    num_alleles=num_alleles,
                    k=int(k),
                    popsize=popsize,
                    mutation_rate=mutation_rate / n_sites,
                    num_reps=num_reps,
                    num_steps=num_steps,
                )
            )
            for key_chunk, coord_chunk in zip(np.array_split(start_keys, num_chunks), np.array_split(starts, num_chunks))
        ]
        results.append(np.concatenate(chunks, axis=0))
    return np.asarray(results)


def strategy_grid(num_options: int) -> tuple[jnp.ndarray, jnp.ndarray, list[int]]:
    """Construct the base-chance/threshold/splitting grid for DE sweeps.

    Parameters:
    - num_options: int
        Number of base-chance and split options. The paper uses 5 or 7 depending on the sweep.

    Returns:
    - tuple[jnp.ndarray, jnp.ndarray, list[int]]
        Threshold samples, base-chance samples, and population split sizes.
    """

    thresholds, base_chances = base_chance_threshold_fixed_prop([0, 0.19], 0.2, num_options)
    if num_options == 5:
        splits = [20, 15, 10, 5, 1]
    elif num_options == 7:
        splits = [24, 20, 16, 12, 8, 4, 1]
    else:
        splits = list(np.round(np.linspace(24, 1, num_options)).astype(int))
    return thresholds, base_chances, splits


def nk_landscape_arrays(rng: jax.Array, n_sites: int, k: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build the traced arrays that define one NK landscape.

    These reproduce the internal state of ``build_NK_landscape_function`` so the
    fitness map can be evaluated as a pure function of traced arrays, which lets a
    single ``jax.jit`` kernel be reused across many landscapes without recompiling.

    Parameters:
    - rng: jax.Array
        JAX PRNG key identifying the landscape (same key as ``build_NK_landscape_function``).
    - n_sites: int
        Number of genotype sites.
    - k: int
        Number of interacting partner sites per site.

    Returns:
    - tuple[jax.Array, jax.Array, jax.Array]
        The interaction matrix, the per-site fitness key, and the per-site offset key.
    """

    r1, r2, r3 = jr.split(rng, 3)
    base_row = 1 * (jnp.arange(n_sites - 1) < k)

    def permutate_rows(perm_rng: jax.Array, index: jax.Array) -> jax.Array:
        return jnp.insert(jr.permutation(perm_rng, base_row), index, 1.0)

    interaction_matrix = jax.vmap(permutate_rows)(jr.split(r1, n_sites), jnp.arange(n_sites))
    return interaction_matrix, r2, r3


def nk_population_fitness(
    population: jax.Array,
    interaction_matrix: jax.Array,
    site_rng: jax.Array,
    offset_rng: jax.Array,
    *,
    fitness_distribution: Callable[[jax.Array], jax.Array] = jr.normal,
) -> jax.Array:
    """Evaluate NK fitness for a population from its traced landscape arrays.

    Parameters:
    - population: jax.Array
        Integer genotype array of shape ``(popsize, n_sites)``.
    - interaction_matrix: jax.Array
        Site-interaction matrix produced by :func:`nk_landscape_arrays`.
    - site_rng: jax.Array
        Per-site fitness key produced by :func:`nk_landscape_arrays`.
    - offset_rng: jax.Array
        Per-site offset key produced by :func:`nk_landscape_arrays`.
    - fitness_distribution: Callable[[jax.Array], jax.Array]
        Random distribution used for site-level fitness contributions.

    Returns:
    - jax.Array
        Fitness value per population member, shape ``(popsize,)``.
    """

    n_sites = interaction_matrix.shape[0]

    def gene_fitness(gene: jax.Array) -> jax.Array:
        site_fitness = jax.vmap(lambda base_rng, data: jr.fold_in(base_rng, data))(jr.split(site_rng, n_sites), gene)
        interaction_fitness = interaction_matrix @ site_fitness + jr.split(offset_rng, n_sites)
        return jnp.sum(jax.vmap(fitness_distribution)(interaction_fitness))

    return jax.vmap(gene_fitness)(population)


def _max_final_fitness(
    rng: jax.Array,
    initial_population: jax.Array,
    selection_function: "Callable",
    mutation_function: "Callable",
    fitness_function: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    split_size: int,
    num_steps: int,
) -> jax.Array:
    """Run ``split_size`` directed-evolution subpopulations and return the best final fitness."""

    subpop_histories = jax.vmap(
        lambda r: run_directed_evolution(
            r,
            initial_population,
            selection_function,
            mutation_function,
            fitness_function=fitness_function,
            num_steps=num_steps,
        )[1]
    )(jr.split(rng, split_size))
    return subpop_histories["fitness"][:, :, -1].max()


@partial(jax.jit, static_argnames=("n_sites", "num_alleles", "popsize", "split_size", "mutation_rate", "num_steps"))
def _nk_strategy_grid_scores(
    rep_keys: jax.Array,
    base_chances: jax.Array,
    thresholds: jax.Array,
    interaction_matrix: jax.Array,
    site_rng: jax.Array,
    offset_rng: jax.Array,
    *,
    n_sites: int,
    num_alleles: int,
    popsize: int,
    split_size: int,
    mutation_rate: float,
    num_steps: int,
) -> jax.Array:
    """Score every (replicate, strategy) cell of one NK landscape in a single fused kernel.

    Replaces the per-cell ``jax.jit`` loop: the kernel compiles once per ``(n_sites,
    split_size)`` and is reused across landscapes because the landscape enters only
    through traced arrays. Numerics match the scalar path: each replicate key seeds the
    random start and the subpopulation split, and is shared across all strategies in that
    replicate.

    Parameters:
    - rep_keys: jax.Array
        Replicate keys, shape ``(num_reps, 2)``.
    - base_chances: jax.Array
        Baseline selection probabilities, shape ``(grid,)``.
    - thresholds: jax.Array
        Rank thresholds paired elementwise with ``base_chances``, shape ``(grid,)``.
    - interaction_matrix, site_rng, offset_rng: jax.Array
        Traced landscape arrays from :func:`nk_landscape_arrays`.
    - n_sites, num_alleles, popsize, split_size, mutation_rate, num_steps
        Static simulation parameters (see :func:`generate_nk_strategy_sweep`).

    Returns:
    - jax.Array
        Maximum final fitness per cell, shape ``(num_reps, grid)``.
    """

    sub_pop = int(popsize / split_size)
    mutation_function = build_mutation_function(mutation_rate, num_alleles)

    def fitness_function(population: jax.Array) -> jax.Array:
        return nk_population_fitness(population, interaction_matrix, site_rng, offset_rng)

    def run_one(rng: jax.Array, base_chance: jax.Array, threshold: jax.Array) -> jax.Array:
        selection_function = build_selection_function(slct.base_chance_threshold_select, {"threshold": threshold, "base_chance": base_chance})
        initial_population = jnp.tile(jr.randint(rng, (1, n_sites), 0, num_alleles), (sub_pop, 1))
        return _max_final_fitness(rng, initial_population, selection_function, mutation_function, fitness_function, split_size=split_size, num_steps=num_steps)

    over_strategies = jax.vmap(run_one, in_axes=(None, 0, 0))
    over_replicates = jax.vmap(over_strategies, in_axes=(0, None, None))
    return over_replicates(rep_keys, base_chances, thresholds)


@partial(jax.jit, static_argnames=("n_sites", "num_alleles", "popsize", "split_size", "mutation_rate", "num_steps"))
def _empirical_strategy_grid_scores(
    rep_keys: jax.Array,
    base_chances: jax.Array,
    thresholds: jax.Array,
    start: jax.Array,
    landscape: jax.Array,
    *,
    n_sites: int,
    num_alleles: int,
    popsize: int,
    split_size: int,
    mutation_rate: float,
    num_steps: int,
) -> jax.Array:
    """Score every (replicate, strategy) cell for one empirical start in a single fused kernel.

    Parameters:
    - rep_keys: jax.Array
        Replicate keys, shape ``(num_reps, 2)``.
    - base_chances, thresholds: jax.Array
        Strategy grid arrays paired elementwise, shape ``(grid,)``.
    - start: jax.Array
        Starting genotype coordinate, shape ``(n_sites,)``.
    - landscape: jax.Array
        Empirical fitness landscape array.
    - n_sites, num_alleles, popsize, split_size, mutation_rate, num_steps
        Static simulation parameters (see :func:`generate_empirical_strategy_sweep`).

    Returns:
    - jax.Array
        Maximum final fitness per cell, shape ``(num_reps, grid)``.
    """

    sub_pop = int(popsize / split_size)
    mutation_function = build_mutation_function(mutation_rate, num_alleles)
    fitness_function = build_empirical_landscape_function(landscape)
    initial_population = repeated_population(start, sub_pop)

    def run_one(rng: jax.Array, base_chance: jax.Array, threshold: jax.Array) -> jax.Array:
        selection_function = build_selection_function(slct.base_chance_threshold_select, {"threshold": threshold, "base_chance": base_chance})
        return _max_final_fitness(rng, initial_population, selection_function, mutation_function, fitness_function, split_size=split_size, num_steps=num_steps)

    over_strategies = jax.vmap(run_one, in_axes=(None, 0, 0))
    over_replicates = jax.vmap(over_strategies, in_axes=(0, None, None))
    return over_replicates(rep_keys, base_chances, thresholds)


def _grid_scores_in_batches(score_fn: Callable[[jax.Array], jax.Array], rep_keys: jax.Array, batch_size: int) -> np.ndarray:
    """Evaluate a per-replicate grid scorer, chunking replicates to bound device memory.

    Parameters:
    - score_fn: Callable[[jax.Array], jax.Array]
        Function mapping a batch of replicate keys to a ``(batch, grid)`` score array.
    - rep_keys: jax.Array
        All replicate keys, shape ``(num_reps, 2)``.
    - batch_size: int
        Maximum replicates evaluated per device call; ``<= 0`` runs them all at once.

    Returns:
    - np.ndarray
        Concatenated scores, shape ``(num_reps, grid)``.
    """

    num_reps = rep_keys.shape[0]
    if batch_size is None or batch_size <= 0 or batch_size >= num_reps:
        return np.asarray(score_fn(rep_keys))
    num_chunks = int(np.ceil(num_reps / batch_size))
    return np.concatenate([np.asarray(score_fn(chunk)) for chunk in np.array_split(rep_keys, num_chunks)], axis=0)


@partial(jax.jit, static_argnames=("n_sites", "num_alleles", "split", "popsize", "mutation_rate", "num_steps"))
def best_variant_traces_nk(
    rep_keys: jax.Array,
    start_coords: jax.Array,
    base_chance: jax.Array,
    threshold: jax.Array,
    interaction_matrix: jax.Array,
    site_rng: jax.Array,
    offset_rng: jax.Array,
    *,
    n_sites: int,
    num_alleles: int,
    split: int,
    popsize: int,
    mutation_rate: float,
    num_steps: int,
) -> jax.Array:
    """Best-variant NK directed-evolution trajectories for a batch of replicates.

    For each replicate the population is split into ``split`` subpopulations (each seeded by
    the replicate key) and evolved; the trajectory is the **maximum fitness across all
    subpopulations and members at each generation** — the best variant found, which is the
    quantity a directed-evolution campaign keeps and which matches the strategy-sweep metric.
    One fused kernel over all replicates of one fixed strategy.

    Parameters:
    - rep_keys: jax.Array
        Per-replicate keys, shape ``(num_reps, 2)``; each seeds its subpopulation split.
    - start_coords: jax.Array
        Per-replicate starting genotype coordinates, shape ``(num_reps, n_sites)``.
    - base_chance, threshold: jax.Array
        Scalar selection parameters for this strategy.
    - interaction_matrix, site_rng, offset_rng: jax.Array
        Traced NK landscape arrays from :func:`nk_landscape_arrays`.
    - n_sites, num_alleles, split, popsize, mutation_rate, num_steps
        Static simulation parameters.

    Returns:
    - jax.Array
        Best-variant fitness trajectory per replicate, shape ``(num_reps, num_steps)``.
    """

    sub_pop = int(popsize / split)
    selection_function = build_selection_function(slct.base_chance_threshold_select, {"threshold": threshold, "base_chance": base_chance})
    mutation_function = build_mutation_function(mutation_rate, num_alleles)

    def fitness_function(population: jax.Array) -> jax.Array:
        return nk_population_fitness(population, interaction_matrix, site_rng, offset_rng)

    def one_replicate(rng: jax.Array, start_coord: jax.Array) -> jax.Array:
        initial_population = repeated_population(start_coord, sub_pop)
        subpop_best = jax.vmap(
            lambda split_key: run_directed_evolution(split_key, initial_population, selection_function, mutation_function, fitness_function=fitness_function, num_steps=num_steps)[1]["fitness"].max(axis=-1)
        )(jr.split(rng, split))
        return subpop_best.max(axis=0)

    return jax.vmap(one_replicate)(rep_keys, start_coords)


@partial(jax.jit, static_argnames=("n_sites", "num_alleles", "split", "popsize", "mutation_rate", "num_steps"))
def best_variant_traces_empirical(
    rep_keys: jax.Array,
    start_coord: jax.Array,
    base_chance: jax.Array,
    threshold: jax.Array,
    landscape: jax.Array,
    *,
    n_sites: int,
    num_alleles: int,
    split: int,
    popsize: int,
    mutation_rate: float,
    num_steps: int,
) -> jax.Array:
    """Best-variant empirical directed-evolution trajectories for a batch of replicates.

    The trajectory is the maximum fitness across all subpopulations and members at each
    generation (the best variant found), matching the strategy-sweep metric.

    Parameters:
    - rep_keys: jax.Array
        Per-replicate keys, shape ``(num_reps, 2)``.
    - start_coord: jax.Array
        Fixed starting genotype coordinate shared by all replicates, shape ``(n_sites,)``.
    - base_chance, threshold: jax.Array
        Scalar selection parameters for this strategy.
    - landscape: jax.Array
        Empirical fitness landscape array.
    - n_sites, num_alleles, split, popsize, mutation_rate, num_steps
        Static simulation parameters.

    Returns:
    - jax.Array
        Best-variant fitness trajectory per replicate, shape ``(num_reps, num_steps)``.
    """

    sub_pop = int(popsize / split)
    selection_function = build_selection_function(slct.base_chance_threshold_select, {"threshold": threshold, "base_chance": base_chance})
    mutation_function = build_mutation_function(mutation_rate, num_alleles)
    fitness_function = build_empirical_landscape_function(landscape)
    initial_population = repeated_population(start_coord, sub_pop)

    def one_replicate(rng: jax.Array) -> jax.Array:
        subpop_best = jax.vmap(
            lambda split_key: run_directed_evolution(split_key, initial_population, selection_function, mutation_function, fitness_function=fitness_function, num_steps=num_steps)[1]["fitness"].max(axis=-1)
        )(jr.split(rng, split))
        return subpop_best.max(axis=0)

    return jax.vmap(one_replicate)(rep_keys)


@partial(jax.jit, static_argnames=("n_sites", "num_alleles", "k", "popsize", "split_size", "num_reps", "mutation_rate", "num_steps"))
def _nk_landscape_batch_scores(
    landscape_keys: jax.Array,
    base_chances: jax.Array,
    thresholds: jax.Array,
    *,
    n_sites: int,
    num_alleles: int,
    k: int,
    popsize: int,
    split_size: int,
    num_reps: int,
    mutation_rate: float,
    num_steps: int,
) -> jax.Array:
    """Max-final-fitness per (landscape, replicate, base-chance) for one (N,K) and split size.

    Vmaps over a batch of NK landscapes (each seeded by its key, which also seeds its
    replicates and random starts), so a whole (N,K) lookup point is one fused kernel.

    Returns:
    - jax.Array
        Shape ``(num_landscapes, num_reps, grid)``.
    """
    sub_pop = int(popsize / split_size)
    mutation_function = build_mutation_function(mutation_rate, num_alleles)

    def one_landscape(land_key: jax.Array) -> jax.Array:
        interaction_matrix, site_rng, offset_rng = nk_landscape_arrays(land_key, n_sites, k)

        def fitness_function(population: jax.Array) -> jax.Array:
            return nk_population_fitness(population, interaction_matrix, site_rng, offset_rng)

        def run_one(rng: jax.Array, base_chance: jax.Array, threshold: jax.Array) -> jax.Array:
            selection_function = build_selection_function(slct.base_chance_threshold_select, {"threshold": threshold, "base_chance": base_chance})
            initial_population = jnp.tile(jr.randint(rng, (1, n_sites), 0, num_alleles), (sub_pop, 1))
            return _max_final_fitness(rng, initial_population, selection_function, mutation_function, fitness_function, split_size=split_size, num_steps=num_steps)

        over_strategies = jax.vmap(run_one, in_axes=(None, 0, 0))
        over_replicates = jax.vmap(over_strategies, in_axes=(0, None, None))
        return over_replicates(jr.split(land_key, num_reps), base_chances, thresholds)

    return jax.vmap(one_landscape)(landscape_keys)


def generate_nk_strategy_space_point(
    rng: jax.Array,
    *,
    n_sites: int,
    num_alleles: int,
    k: int,
    popsize: int,
    num_reps: int,
    num_landscapes: int,
    strategy_grid_size: int,
    mutation_rate: float,
    num_steps: int,
    batch_size: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Landscape-averaged NK strategy space for one (N,K) lookup point (Figure 5A).

    Vmaps over landscapes (chunked by ``batch_size``) instead of looping them in Python, so the
    Figure 5A grid is fast. Output axes are ``(split, base, reps)`` for the ``split_base_reps``
    layout (a transpose, not a reshape).

    Parameters:
    - rng: jax.Array
        Key for this (N,K) point; split into the landscape keys.
    - mutation_rate: float
        Total mutation rate, divided by ``n_sites`` internally.
    - batch_size: int
        Landscapes per fused vmap call; ``<= 0`` runs them all at once.

    Returns:
    - tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(split, base, reps)`` strategy space, plus ``thresholds``, ``base_chances``, ``splits``.
    """
    thresholds, base_chances, splits = strategy_grid(strategy_grid_size)
    base_chances_j = jnp.asarray(base_chances)
    thresholds_j = jnp.asarray(thresholds)
    per_site_mutation = mutation_rate / n_sites
    landscape_keys = np.asarray(jr.split(rng, num_landscapes))
    num_chunks = 1 if batch_size <= 0 or batch_size >= num_landscapes else int(np.ceil(num_landscapes / batch_size))

    split_results = []
    for split_size in splits:
        scores_sum = None
        for key_chunk in np.array_split(landscape_keys, num_chunks):
            scores = np.asarray(_nk_landscape_batch_scores(
                jnp.asarray(key_chunk),
                base_chances_j,
                thresholds_j,
                n_sites=n_sites,
                num_alleles=num_alleles,
                k=int(k),
                popsize=popsize,
                split_size=int(split_size),
                num_reps=num_reps,
                mutation_rate=per_site_mutation,
                num_steps=num_steps,
            )).sum(axis=0)  # (num_reps, grid), summed over the landscape chunk
            scores_sum = scores if scores_sum is None else scores_sum + scores
        split_results.append(scores_sum / num_landscapes)  # landscape-averaged (num_reps, grid)

    space = np.asarray(split_results).transpose(0, 2, 1)  # (num_splits, num_reps, base) -> (split, base, reps)
    return space, np.asarray(thresholds), np.asarray(base_chances), np.asarray(splits)


def generate_nk_strategy_sweep(
    *,
    n_sites: int,
    num_alleles: int,
    k_values: Sequence[int],
    mutation_rate: float,
    popsize: int,
    num_landscapes: int,
    num_reps: int,
    num_steps: int,
    strategy_grid_size: int,
    outer_reps: int = 10,
    seed: int = 42,
    batch_size: int = 0,
) -> np.ndarray:
    """Run NK directed-evolution strategy sweeps.

    Parameters:
    - n_sites: int
        Number of NK sites.
    - num_alleles: int
        Number of alleles per site.
    - k_values: Sequence[int]
        NK ``K`` values to simulate.
    - mutation_rate: float
        Total mutation rate, divided by ``n_sites`` internally.
    - popsize: int
        Total population size.
    - num_landscapes: int
        Number of random NK landscapes per K value.
    - num_reps: int
        Replicates per strategy.
    - num_steps: int
        Directed-evolution generations.
    - strategy_grid_size: int
        Number of base-chance/splitting options.
    - outer_reps: int
        Number of outer repeats.
    - seed: int
        Master JAX seed.
    - batch_size: int
        Maximum replicates evaluated per device call. ``<= 0`` evaluates all replicates
        at once; set a smaller value to reduce peak device memory.

    Returns:
    - np.ndarray
        NumPy array of strategy performance scores.
    """

    thresholds, base_chances, splits = strategy_grid(strategy_grid_size)
    base_chances = jnp.asarray(base_chances)
    thresholds = jnp.asarray(thresholds)
    master_keys = jr.split(jr.PRNGKey(seed), outer_reps)
    all_results = []
    for outer_key in master_keys:
        k_results = []
        for k in k_values:
            landscape_keys = jr.split(outer_key, num_landscapes)
            landscape_results = []
            for landscape_key in landscape_keys:
                interaction_matrix, site_rng, offset_rng = nk_landscape_arrays(landscape_key, n_sites, int(k))
                rep_keys = jr.split(landscape_key, num_reps)
                split_results = []
                for split_size in splits:
                    score_fn = lambda keys, split_size=int(split_size): _nk_strategy_grid_scores(
                        keys,
                        base_chances,
                        thresholds,
                        interaction_matrix,
                        site_rng,
                        offset_rng,
                        n_sites=n_sites,
                        num_alleles=num_alleles,
                        popsize=popsize,
                        split_size=split_size,
                        mutation_rate=mutation_rate / n_sites,
                        num_steps=num_steps,
                    )
                    split_results.append(_grid_scores_in_batches(score_fn, rep_keys, batch_size))
                landscape_results.append(np.moveaxis(np.asarray(split_results), 0, -1))
            k_results.append(np.asarray(landscape_results).mean(axis=0))
        all_results.append(k_results)
    return np.asarray(all_results)


@partial(jax.jit, static_argnames=("n_sites", "num_alleles", "popsize", "split_size", "mutation_rate", "num_steps"))
def _empirical_cell_trajectories(
    rep_keys: jax.Array,
    base_chances: jax.Array,
    thresholds: jax.Array,
    start: jax.Array,
    landscape: jax.Array,
    *,
    n_sites: int,
    num_alleles: int,
    popsize: int,
    split_size: int,
    mutation_rate: float,
    num_steps: int,
) -> jax.Array:
    """Best-variant trajectory for one split size across all base chances, vmapped over replicates.

    Returns the full over-generations best-variant trajectory (max over subpopulations and
    members) for every cell, so a single sweep yields both the strategy-performance heat map
    (final slice) and the directed-evolution lines (selected cells).

    Parameters:
    - rep_keys: jax.Array
        Per-replicate keys, shape ``(num_reps, 2)``.
    - base_chances, thresholds: jax.Array
        Strategy grid arrays paired elementwise, shape ``(grid,)``.
    - start: jax.Array
        Starting genotype coordinate, shape ``(n_sites,)``.
    - landscape: jax.Array
        Empirical fitness landscape array.
    - n_sites, num_alleles, popsize, split_size, mutation_rate, num_steps
        Static simulation parameters.

    Returns:
    - jax.Array
        Best-variant trajectories, shape ``(num_reps, grid, num_steps)``.
    """

    sub_pop = int(popsize / split_size)
    mutation_function = build_mutation_function(mutation_rate, num_alleles)
    fitness_function = build_empirical_landscape_function(landscape)
    initial_population = repeated_population(start, sub_pop)

    def run_one(rng: jax.Array, base_chance: jax.Array, threshold: jax.Array) -> jax.Array:
        selection_function = build_selection_function(slct.base_chance_threshold_select, {"threshold": threshold, "base_chance": base_chance})
        subpop_best = jax.vmap(
            lambda split_key: run_directed_evolution(split_key, initial_population, selection_function, mutation_function, fitness_function=fitness_function, num_steps=num_steps)[1]["fitness"].max(axis=-1)
        )(jr.split(rng, split_size))
        return subpop_best.max(axis=0)

    over_strategies = jax.vmap(run_one, in_axes=(None, 0, 0))
    over_replicates = jax.vmap(over_strategies, in_axes=(0, None, None))
    return over_replicates(rep_keys, base_chances, thresholds)


def generate_empirical_strategy_trajectory_sweep(
    landscape: np.ndarray,
    starts: np.ndarray,
    *,
    mutation_rate: float,
    popsize: int,
    num_reps: int,
    num_steps: int,
    strategy_grid_size: int,
    seed: int = 42,
    batch_size: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run an empirical strategy sweep that keeps the best-variant trajectory of every cell.

    The heat map (final-fitness per strategy) and the directed-evolution lines (selected
    strategies) are both derived from this one array, guaranteeing they are different views of
    the same simulation.

    ``batch_size`` chunks the fused replicate vmap (the ``reps x grid`` device batch): ``<= 0``
    runs all replicates in one vmap (fastest, most memory); set a smaller value to cap the
    device batch when pushing the replicate count high. The result is identical either way.

    Parameters:
    - landscape: np.ndarray
        Empirical fitness landscape array.
    - starts: np.ndarray
        Starting genotype coordinates, shape ``(num_starts, n_sites)``.
    - mutation_rate: float
        Per-site mutation probability (shared by heat map and lines).
    - popsize: int
        Total population size (shared by heat map and lines).
    - num_reps: int
        Replicate directed-evolution campaigns per start, averaged over.
    - num_steps: int
        Directed-evolution generations.
    - strategy_grid_size: int
        Number of base-chance/splitting options.
    - seed: int
        Master JAX seed.

    Returns:
    - tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``trajectories`` with shape ``(num_starts, num_splits, grid, num_steps)`` (mean over
        replicates), and the ``thresholds``, ``base_chances``, ``splits`` grid arrays.
    """

    thresholds, base_chances, splits = strategy_grid(strategy_grid_size)
    base_chances_j = jnp.asarray(base_chances)
    thresholds_j = jnp.asarray(thresholds)
    landscape_j = jnp.asarray(landscape)
    n_sites = landscape_j.ndim
    num_alleles = landscape_j.shape[0]
    master_keys = jr.split(jr.PRNGKey(seed), len(starts))

    num_chunks = 1 if batch_size <= 0 or batch_size >= num_reps else int(np.ceil(num_reps / batch_size))
    all_start_results = []
    for start, start_key in zip(starts, master_keys):
        start_j = jnp.asarray(start)
        rep_keys = np.asarray(jr.split(start_key, num_reps))
        split_results = []
        for split_size in splits:
            # Sum best-variant trajectories over replicate chunks, then divide by num_reps to get
            # the mean. Chunking caps the fused vmap (reps x grid) for device memory; the result
            # is identical to running all replicates at once.
            rep_sum = None
            for rep_chunk in np.array_split(rep_keys, num_chunks):
                cell = np.asarray(_empirical_cell_trajectories(
                    jnp.asarray(rep_chunk),
                    base_chances_j,
                    thresholds_j,
                    start_j,
                    landscape_j,
                    n_sites=n_sites,
                    num_alleles=num_alleles,
                    popsize=popsize,
                    split_size=int(split_size),
                    mutation_rate=mutation_rate,
                    num_steps=num_steps,
                )).sum(axis=0)
                rep_sum = cell if rep_sum is None else rep_sum + cell
            split_results.append(rep_sum / num_reps)
        all_start_results.append(np.asarray(split_results))
    return np.asarray(all_start_results), np.asarray(thresholds), np.asarray(base_chances), np.asarray(splits)


def generate_empirical_strategy_sweep(
    landscape: np.ndarray,
    starts: np.ndarray,
    *,
    mutation_rate: float,
    popsize: int,
    num_reps: int,
    num_steps: int,
    strategy_grid_size: int,
    outer_reps: int = 10,
    seed: int = 42,
    batch_size: int = 0,
) -> np.ndarray:
    """Run empirical directed-evolution strategy sweeps.

    Parameters:
    - landscape: np.ndarray
        Empirical fitness landscape array.
    - starts: np.ndarray
        Starting genotype coordinates.
    - mutation_rate: float
        Per-site mutation probability.
    - popsize: int
        Total population size.
    - num_reps: int
        Replicates per strategy.
    - num_steps: int
        Directed-evolution generations.
    - strategy_grid_size: int
        Number of base-chance/splitting options.
    - outer_reps: int
        Number of outer repeats per starting genotype.
    - seed: int
        Master JAX seed.
    - batch_size: int
        Maximum replicates evaluated per device call. ``<= 0`` evaluates all replicates
        at once; set a smaller value to reduce peak device memory.

    Returns:
    - np.ndarray
        NumPy array of empirical strategy performance scores.
    """

    thresholds, base_chances, splits = strategy_grid(strategy_grid_size)
    base_chances = jnp.asarray(base_chances)
    thresholds = jnp.asarray(thresholds)
    landscape = jnp.asarray(landscape)
    n_sites = landscape.ndim
    num_alleles = landscape.shape[0]
    master_keys = jr.split(jr.PRNGKey(seed), len(starts))
    all_start_results = []
    for start, start_key in zip(starts, master_keys):
        start = jnp.asarray(start)
        outer_results = []
        for outer_key in jr.split(start_key, outer_reps):
            rep_keys = jr.split(outer_key, num_reps)
            split_results = []
            for split_size in splits:
                score_fn = lambda keys, split_size=int(split_size): _empirical_strategy_grid_scores(
                    keys,
                    base_chances,
                    thresholds,
                    start,
                    landscape,
                    n_sites=n_sites,
                    num_alleles=num_alleles,
                    popsize=popsize,
                    split_size=split_size,
                    mutation_rate=mutation_rate,
                    num_steps=num_steps,
                )
                split_results.append(_grid_scores_in_batches(score_fn, rep_keys, batch_size))
            outer_results.append(np.moveaxis(np.asarray(split_results), 0, -1))
        all_start_results.append(outer_results)
    return np.asarray(all_start_results)


RAW_FILENAMES: dict[str, str] = {
    "nk_decay_grid": "nk_decay_grid_raw_data.pkl",
    "nk_strategy_grid": "nk_strategy_grid_raw_data.pkl",
    "nk_popsize_accuracy": "nk_popsize_accuracy_raw_data.pkl",
    "nk_mutation_accuracy": "nk_mutation_accuracy_raw_data.pkl",
    "nk_heterogeneity": "nk_heterogeneity_raw_data.pkl",
    "nk_decay_N4_A20": "nk_decay_N4_A20_raw_data.pkl",
    "nk_strategy_N4_A20": "nk_strategy_N4_A20_raw_data.pkl",
    "nk_decay_N3_A20": "nk_decay_N3_A20_raw_data.pkl",
    "nk_strategy_N3_A20": "nk_strategy_N3_A20_raw_data.pkl",
}

for _steps in GENERATION_STEPS:
    RAW_FILENAMES[f"nk_strategy_N4_A20_steps{_steps}"] = f"nk_strategy_N4_A20_steps{_steps}_raw_data.pkl"

for _name in EMPIRICAL_NAMES:
    RAW_FILENAMES[f"empirical_decay_{_name}_uniform"] = f"empirical_decay_{_name}_uniform_raw_data.pkl"
    RAW_FILENAMES[f"empirical_decay_{_name}_all"] = f"empirical_decay_{_name}_all_raw_data.pkl"
    RAW_FILENAMES[f"empirical_decay_{_name}_popsize"] = f"empirical_decay_{_name}_popsize_raw_data.pkl"
    RAW_FILENAMES[f"empirical_strategy_{_name}_uniform"] = f"empirical_strategy_{_name}_uniform_raw_data.pkl"
    # Unified best-variant trajectory sweep: heat map (final slice) and DE lines share this one product.
    RAW_FILENAMES[f"empirical_strategy_traj_{_name}"] = f"empirical_strategy_traj_{_name}_raw_data.pkl"


def expected_raw_outputs() -> dict[str, Path]:
    """Return the expected raw output paths for the retained pipeline products.

    Returns:
    - dict[str, Path]
        Mapping from raw product key to its path under ``raw_data``.
    """

    return {key: raw_path(filename) for key, filename in RAW_FILENAMES.items()}


def missing_raw_outputs() -> dict[str, Path]:
    """Return expected raw output paths that do not currently exist.

    Returns:
    - dict[str, Path]
        Mapping from missing raw product key to its path under ``raw_data``.
    """

    return {key: path for key, path in expected_raw_outputs().items() if not path.exists()}


def ordered_unique_pairs(pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Deduplicate NK pairs while preserving their first-seen order.

    Parameters:
    - pairs: Sequence[tuple[int, int]]
        Ordered NK ``(N, K)`` pairs that may contain duplicates.

    Returns:
    - list[tuple[int, int]]
        Integer ``(N, K)`` pairs with duplicates removed.
    """

    return list(dict.fromkeys((int(n_sites), int(k)) for n_sites, k in pairs))


def nk_grid_pairs(n_range: tuple[int, int] = (10, 50), num_samples: int = 10, K_start: int = 1) -> list[tuple[int, int]]:
    """Construct the NK ``(N, K)`` grid used by Figures 3 and 5.

    Parameters:
    - n_range: tuple[int, int]
        Inclusive range of N values sampled linearly.
    - num_samples: int
        Number of N samples and number of K samples per N.
    - K_start: int
        First K value sampled for each N.

    Returns:
    - list[tuple[int, int]]
        List of ``(N, K)`` pairs in the historical reversed order used by the original scripts and processing code.
    """

    n_values = np.linspace(n_range[0], n_range[1], num=num_samples).astype(int)
    pairs = []
    for n_sites in n_values:
        k_values = np.linspace(K_start, n_sites + K_start - 1, num_samples).astype(int)
        for k in k_values:
            pairs.append((int(n_sites), int(k)))
    return list(reversed(pairs))
