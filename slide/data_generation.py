"""Reusable raw-data generation helpers for the SLIDE notebook pipeline.

The executable orchestration lives in ``data_generation.ipynb``. This module
keeps reusable simulation kernels, start samplers, product filename constants,
and small registry helpers used by that notebook and by ``data_processing.ipynb``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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

    Returns:
    - np.ndarray
        NumPy array indexed by K value, start, replicate, and generation.
    """

    starts = nk_uniform_start_locs(n_sites=n_sites, num_alleles=num_alleles, num_starts=num_starts)
    master_keys = jr.split(jr.PRNGKey(seed), len(k_values))
    results = []
    for k, key in zip(k_values, master_keys):
        start_keys = jr.split(key, len(starts))
        k_results = []
        for start, start_key in zip(starts, start_keys):
            run = run_nk_diffusion_replicates(
                start_key,
                n_sites=n_sites,
                k=int(k),
                num_alleles=num_alleles,
                start=start,
                popsize=popsize,
                mutation_rate=mutation_rate / n_sites,
                num_reps=num_reps,
                num_steps=num_steps,
            )
            k_results.append(run["fitness"].mean(axis=-1))
        results.append(k_results)
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


def _run_strategy(
    rng: jax.Array,
    fitness_function: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    n_sites: int,
    num_alleles: int,
    start: np.ndarray | None,
    split_size: int,
    base_chance: float,
    threshold: float,
    popsize: int,
    mutation_rate: float,
    num_steps: int,
) -> jax.Array:
    """Run one directed-evolution strategy configuration.

    Parameters:
    - rng: jax.Array
        JAX random key.
    - fitness_function: Callable[[jnp.ndarray], jnp.ndarray]
        JAX-compatible fitness lookup function.
    - n_sites: int
        Number of genotype sites.
    - num_alleles: int
        Number of alleles per site.
    - start: np.ndarray | None
        Optional starting genotype; if absent, one random start is used.
    - split_size: int
        Number of split subpopulations.
    - base_chance: float
        Baseline selection probability.
    - threshold: float
        Rank threshold for guaranteed selection.
    - popsize: int
        Total population size before splitting.
    - mutation_rate: float
        Per-site mutation probability.
    - num_steps: int
        Number of directed-evolution generations.

    Returns:
    - jax.Array
        Maximum final fitness observed across split subpopulations.
    """

    params = {"threshold": threshold, "base_chance": base_chance}
    selection_function = build_selection_function(slct.base_chance_threshold_select, params)
    mutation_function = build_mutation_function(mutation_rate, num_alleles)
    if start is None:
        initial_population = jnp.tile(jr.randint(rng, (1, n_sites), 0, num_alleles), (int(popsize / split_size), 1))
    else:
        initial_population = repeated_population(start, int(popsize / split_size))
    vmapped = jax.jit(
        jax.vmap(
            lambda r: run_directed_evolution(
                r,
                initial_population,
                selection_function,
                mutation_function,
                fitness_function=fitness_function,
                num_steps=num_steps,
            )[1]
        )
    )
    return vmapped(jr.split(rng, split_size))["fitness"][:, :, -1].max()


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

    Returns:
    - np.ndarray
        NumPy array of strategy performance scores.
    """

    thresholds, base_chances, splits = strategy_grid(strategy_grid_size)
    master_keys = jr.split(jr.PRNGKey(seed), outer_reps)
    all_results = []
    for outer_key in master_keys:
        k_results = []
        for k in k_values:
            landscape_keys = jr.split(outer_key, num_landscapes)
            landscape_results = []
            for landscape_key in landscape_keys:
                fitness_function = build_NK_landscape_function(landscape_key, n_sites, int(k))
                rep_keys = jr.split(landscape_key, num_reps)
                split_results = []
                for split_size in splits:
                    grid_results = [
                        _run_strategy(
                            rep_key,
                            fitness_function,
                            n_sites=n_sites,
                            num_alleles=num_alleles,
                            start=None,
                            split_size=int(split_size),
                            base_chance=float(base_chance),
                            threshold=float(threshold),
                            popsize=popsize,
                            mutation_rate=mutation_rate / n_sites,
                            num_steps=num_steps,
                        )
                        for rep_key in rep_keys
                        for base_chance, threshold in zip(base_chances, thresholds)
                    ]
                    split_results.append(np.asarray(grid_results).reshape(num_reps, strategy_grid_size))
                landscape_results.append(np.moveaxis(np.asarray(split_results), 0, -1))
            k_results.append(np.asarray(landscape_results).mean(axis=0))
        all_results.append(k_results)
    return np.asarray(all_results)


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

    Returns:
    - np.ndarray
        NumPy array of empirical strategy performance scores.
    """

    thresholds, base_chances, splits = strategy_grid(strategy_grid_size)
    fitness_function = build_empirical_landscape_function(jnp.asarray(landscape))
    master_keys = jr.split(jr.PRNGKey(seed), len(starts))
    all_start_results = []
    for start, start_key in zip(starts, master_keys):
        outer_results = []
        for outer_key in jr.split(start_key, outer_reps):
            rep_keys = jr.split(outer_key, num_reps)
            split_results = []
            for split_size in splits:
                grid_results = [
                    _run_strategy(
                        rep_key,
                        fitness_function,
                        n_sites=landscape.ndim,
                        num_alleles=landscape.shape[0],
                        start=start,
                        split_size=int(split_size),
                        base_chance=float(base_chance),
                        threshold=float(threshold),
                        popsize=popsize,
                        mutation_rate=mutation_rate,
                        num_steps=num_steps,
                    )
                    for rep_key in rep_keys
                    for base_chance, threshold in zip(base_chances, thresholds)
                ]
                split_results.append(np.asarray(grid_results).reshape(num_reps, strategy_grid_size))
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
}

for _steps in GENERATION_STEPS:
    RAW_FILENAMES[f"nk_strategy_N4_A20_steps{_steps}"] = f"nk_strategy_N4_A20_steps{_steps}_raw_data.pkl"

for _name in EMPIRICAL_NAMES:
    RAW_FILENAMES[f"empirical_decay_{_name}_uniform"] = f"empirical_decay_{_name}_uniform_raw_data.pkl"
    RAW_FILENAMES[f"empirical_decay_{_name}_all"] = f"empirical_decay_{_name}_all_raw_data.pkl"
    RAW_FILENAMES[f"empirical_decay_{_name}_popsize"] = f"empirical_decay_{_name}_popsize_raw_data.pkl"
    RAW_FILENAMES[f"empirical_strategy_{_name}_uniform"] = f"empirical_strategy_{_name}_uniform_raw_data.pkl"


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
