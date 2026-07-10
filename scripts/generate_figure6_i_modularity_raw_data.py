"""Generate stochastic raw data for Figure 6I modularity analysis."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from slide.direvo_functions import (  # noqa: E402
    build_empirical_landscape_function,
    build_mutation_function,
    run_diffusion,
)
from slide.ruggedness_functions import get_nk_l_o_shape  # noqa: E402
from slide.utils import get_raw_data_dir, load_pickle, save_pickle  # noqa: E402

N_SITES: int = 4
NUM_ALLELES: int = 20
K_VALUES: tuple[int, ...] = (0, 1, 2, 3)
LAMBDA_VALUES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
MODULES: tuple[tuple[int, ...], ...] = ((0, 1), (2, 3))
NUM_LANDSCAPES: int = 20
NUM_LOCAL_STARTS: int = 20
NUM_POPULATION_REPLICATES: int = 10
POPULATION_SIZE: int = 2_500
TOTAL_MUTATION_RATE: float = 0.1
NUM_DECAY_STEPS: int = 75
GLOBAL_START_BATCH_SIZE: int = 100
RANDOM_SEED: int = 42


def parse_csv_numbers(value: str, number_type: type[int] | type[float]) -> tuple[int, ...] | tuple[float, ...]:
    """Parse comma-separated numeric values.

    Parameters:
    - value: str
        Comma-separated values.
    - number_type: type[int] | type[float]
        Numeric conversion type.

    Returns:
    - tuple[int, ...] | tuple[float, ...]
        Parsed values.
    """
    parsed = tuple(number_type(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("At least one value is required.")
    return parsed


def generate_nk_landscapes(
    n_sites: int, num_alleles: int, k_values: Sequence[int], num_landscapes: int, seed: int
) -> dict[str, object]:
    """Generate paired NK landscapes used by Figure 6I.

    Parameters:
    - n_sites: int
        Number of sites.
    - num_alleles: int
        Alleles per site.
    - k_values: Sequence[int]
        NK interaction values.
    - num_landscapes: int
        Replicates per K.
    - seed: int
        JAX seed.

    Returns:
    - dict[str, object]
        Raw landscape payload.
    """
    shape = (num_alleles,) * n_sites
    landscapes = np.empty((len(k_values), num_landscapes, *shape), dtype=np.float32)
    base_key = jr.PRNGKey(seed)
    for k_index, k_value in enumerate(k_values):
        for replicate in range(num_landscapes):
            key = jr.fold_in(jr.fold_in(base_key, int(k_value)), replicate)
            landscapes[k_index, replicate] = np.asarray(
                get_nk_l_o_shape(key, n_sites, int(k_value), shape), dtype=np.float32
            )
    return {
        "data": {"landscapes": landscapes},
        "params": {"N": n_sites, "A": num_alleles, "K_values": tuple(k_values),
                   "num_landscapes": num_landscapes, "seed": seed},
        "metadata": {"paper_reference": "Figure 6I"},
    }


def modular_projection(landscape: np.ndarray) -> np.ndarray:
    """Project a four-site landscape onto modules (0,1) and (2,3).

    Parameters:
    - landscape: np.ndarray
        Four-dimensional fitness landscape.

    Returns:
    - np.ndarray
        Additive two-module projection.
    """
    values = np.asarray(landscape, dtype=np.float64)
    if values.ndim != N_SITES:
        raise ValueError(f"Expected {N_SITES} dimensions, received {values.ndim}.")
    first = values.mean(axis=MODULES[1], keepdims=True)
    second = values.mean(axis=MODULES[0], keepdims=True)
    return first + second - values.mean()


def modular_landscape(landscape: np.ndarray, lambda_value: float) -> tuple[np.ndarray, dict[str, float]]:
    """Interpolate toward modularity while preserving mean and variance.

    Parameters:
    - landscape: np.ndarray
        Original NK landscape.
    - lambda_value: float
        Modular projection strength in [0, 1].

    Returns:
    - tuple[np.ndarray, dict[str, float]]
        Rescaled landscape and scaling statistics.
    """
    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lambda_value must lie in [0, 1].")
    original = np.asarray(landscape, dtype=np.float64)
    candidate = (1.0 - lambda_value) * original + lambda_value * modular_projection(original)
    original_mean, original_std = float(original.mean()), float(original.std())
    candidate_mean, candidate_std = float(candidate.mean()), float(candidate.std())
    if candidate_std <= 0.0:
        raise ValueError("Modular candidate has zero variance.")
    result = original_mean + (candidate - candidate_mean) * (original_std / candidate_std)
    if lambda_value == 0.0 and not np.allclose(result, original, rtol=1e-6, atol=1e-7):
        raise AssertionError("lambda=0 does not reproduce the original landscape.")
    if not np.isclose(result.mean(), original_mean, rtol=1e-6, atol=1e-7):
        raise AssertionError("Landscape mean was not preserved.")
    if not np.isclose(result.std(), original_std, rtol=1e-6, atol=1e-7):
        raise AssertionError("Landscape standard deviation was not preserved.")
    return result.astype(np.float32), {
        "original_mean": original_mean, "original_std": original_std,
        "candidate_mean": candidate_mean, "candidate_std": candidate_std,
        "scale_factor": original_std / candidate_std,
    }


def fitness_history_only(*, fitnesses: jax.Array, pop: jax.Array) -> jax.Array:
    """Return recorded fitness values.

    Parameters:
    - fitnesses: jax.Array
        Current fitness values.
    - pop: jax.Array
        Current population, unused.

    Returns:
    - jax.Array
        Fitness values.
    """
    del pop
    return fitnesses


def linear_ids_to_starts(linear_ids: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Convert flat genotype IDs to coordinates.

    Parameters:
    - linear_ids: np.ndarray
        Flat IDs.
    - shape: tuple[int, ...]
        Landscape shape.

    Returns:
    - np.ndarray
        Genotype coordinates.
    """
    return np.column_stack(np.unravel_index(linear_ids, shape)).astype(np.int32)


def build_runner(
    landscape: np.ndarray, popsize: int, mutation_rate_per_site: float,
    num_replicates: int, num_steps: int, seed: int,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Build the jitted per-start stochastic F_mu runner.

    Parameters:
    - landscape: np.ndarray
        Fitness landscape.
    - popsize: int
        Population size.
    - mutation_rate_per_site: float
        Per-site mutation probability.
    - num_replicates: int
        Population replicates per start.
    - num_steps: int
        Generations.
    - seed: int
        Simulation seed.

    Returns:
    - Callable[[jax.Array, jax.Array], jax.Array]
        Batched F_mu runner.
    """
    fitness_function = build_empirical_landscape_function(jnp.asarray(landscape))
    mutation_function = build_mutation_function(mutation_rate_per_site, landscape.shape[0])
    base_key = jr.PRNGKey(seed)

    def run_start(start: jax.Array, start_id: jax.Array) -> jax.Array:
        keys = jr.split(jr.fold_in(base_key, start_id.astype(jnp.uint32)), num_replicates)
        population = jnp.broadcast_to(start[None], (popsize, start.shape[0]))

        def run_replicate(key: jax.Array) -> jax.Array:
            history = run_diffusion(
                key, population, mutation_function, fitness_function=fitness_function,
                num_steps=num_steps, extra_function_dict={"fitness": fitness_history_only},
            )[1]
            return history["fitness"].mean(axis=-1)

        return jax.vmap(run_replicate)(keys).mean(axis=0)

    return jax.jit(jax.vmap(run_start, in_axes=(0, 0)))


def stochastic_curves(
    landscape: np.ndarray, local_starts: np.ndarray, local_ids: np.ndarray,
    seed: int, args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute all-start global and sampled local G_mu curves.

    Parameters:
    - landscape: np.ndarray
        Modified landscape.
    - local_starts: np.ndarray
        Shared local coordinates.
    - local_ids: np.ndarray
        Shared local flat IDs.
    - seed: int
        Simulation seed.
    - args: argparse.Namespace
        Simulation configuration.

    Returns:
    - tuple[np.ndarray, np.ndarray]
        Global curve and local curves.
    """
    runner = build_runner(
        landscape, args.population_size, args.total_mutation_rate / N_SITES,
        args.num_population_replicates, args.num_decay_steps, seed,
    )
    local_f = np.asarray(runner(jnp.asarray(local_starts), jnp.asarray(local_ids)), dtype=np.float64)
    global_sum = np.zeros(args.num_decay_steps, dtype=np.float64)
    for start in range(0, landscape.size, args.global_start_batch_size):
        stop = min(start + args.global_start_batch_size, landscape.size)
        ids = np.arange(start, stop, dtype=np.uint32)
        coordinates = linear_ids_to_starts(ids, landscape.shape)
        f_mu = np.asarray(runner(jnp.asarray(coordinates), jnp.asarray(ids)), dtype=np.float64)
        global_sum += np.square(f_mu).sum(axis=0)
    return global_sum / landscape.size, np.square(local_f)


def generate_raw(landscape_payload: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    """Generate the complete stochastic Figure 6I raw payload.

    Parameters:
    - landscape_payload: dict[str, object]
        Base NK landscapes.
    - args: argparse.Namespace
        Generation configuration.

    Returns:
    - dict[str, object]
        Raw G_mu payload.
    """
    k_values = tuple(int(value) for value in args.k_values)
    lambdas = tuple(float(value) for value in args.lambda_values)
    available_k_values = tuple(int(value) for value in landscape_payload["params"]["K_values"])
    missing_k_values = sorted(set(k_values) - set(available_k_values))
    if missing_k_values:
        raise ValueError(f"Base landscape cache is missing K values {missing_k_values}.")
    k_indices = [available_k_values.index(value) for value in k_values]
    base = np.asarray(landscape_payload["data"]["landscapes"], dtype=np.float32)[
        k_indices, :args.num_landscapes
    ]
    curve_shape = (len(k_values), len(lambdas), args.num_landscapes, args.num_decay_steps)
    local_shape = curve_shape[:-1] + (args.num_local_starts, args.num_decay_steps)
    global_values = np.empty(curve_shape, dtype=np.float32)
    local_values = np.empty(local_shape, dtype=np.float32)
    local_coordinates = np.empty((len(k_values), args.num_landscapes, args.num_local_starts, N_SITES), dtype=np.int16)
    selection_seeds = np.empty((len(k_values), args.num_landscapes), dtype=np.uint32)
    simulation_seeds = np.empty(curve_shape[:-1], dtype=np.uint32)
    scaling = np.empty(curve_shape[:-1] + (5,), dtype=np.float64)
    all_ids = np.arange(NUM_ALLELES ** N_SITES, dtype=np.uint32)
    total_conditions = len(k_values) * args.num_landscapes * len(lambdas)
    with tqdm.tqdm(total=total_conditions, desc="Figure 6I modularity", unit="condition") as progress:
        for k_index, k_value in enumerate(k_values):
            for landscape_index in range(args.num_landscapes):
                pair_index = k_index * args.num_landscapes + landscape_index
                selection_seed = args.seed + 1_000_000 + pair_index
                selection_seeds[k_index, landscape_index] = selection_seed
                rng = np.random.default_rng(selection_seed)
                selected_ids = rng.choice(all_ids, size=args.num_local_starts, replace=False).astype(np.uint32)
                starts = linear_ids_to_starts(selected_ids, base[k_index, landscape_index].shape)
                local_coordinates[k_index, landscape_index] = starts
                for lambda_index, lambda_value in enumerate(lambdas):
                    progress.set_postfix_str(
                        f"K={k_value}, landscape={landscape_index + 1}/{args.num_landscapes}, "
                        f"lambda={lambda_value:.2f}",
                        refresh=True,
                    )
                    modified, stats = modular_landscape(base[k_index, landscape_index], lambda_value)
                    simulation_seed = args.seed + 2_000_000 + pair_index * len(lambdas) + lambda_index
                    simulation_seeds[k_index, lambda_index, landscape_index] = simulation_seed
                    global_curve, local_curves = stochastic_curves(
                        modified, starts, selected_ids, simulation_seed, args
                    )
                    global_values[k_index, lambda_index, landscape_index] = global_curve
                    local_values[k_index, lambda_index, landscape_index] = local_curves
                    scaling[k_index, lambda_index, landscape_index] = tuple(stats.values())
                    progress.update()
    if not np.isfinite(global_values).all() or not np.isfinite(local_values).all():
        raise FloatingPointError("Generated G_mu arrays contain non-finite values.")
    return {
        "data": {"g_mu_global": global_values, "g_mu_local": local_values,
                 "local_start_coordinates": local_coordinates,
                 "selection_seeds": selection_seeds, "simulation_seeds": simulation_seeds,
                 "scaling_statistics": scaling},
        "params": {"N": N_SITES, "A": NUM_ALLELES, "K_values": k_values,
                   "lambda_values": lambdas, "modules": MODULES,
                   "num_landscapes": args.num_landscapes, "num_local_starts": args.num_local_starts,
                   "num_population_replicates": args.num_population_replicates,
                   "population_size": args.population_size, "total_mutation_rate": args.total_mutation_rate,
                   "num_decay_steps": args.num_decay_steps,
                   "global_start_batch_size": args.global_start_batch_size, "seed": args.seed,
                   "local_starts_shared_across_lambda": True,
                   "scaling_fields": ("original_mean", "original_std", "candidate_mean", "candidate_std", "scale_factor")},
        "metadata": {"paper_reference": "Figure 6I", "description": "Stochastic modularity G_mu curves."},
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
    - argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=get_raw_data_dir())
    parser.add_argument("--overwrite-raw", action="store_true")
    parser.add_argument("--overwrite-landscapes", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--k-values", type=lambda value: parse_csv_numbers(value, int), default=K_VALUES)
    parser.add_argument("--lambda-values", type=lambda value: parse_csv_numbers(value, float), default=LAMBDA_VALUES)
    parser.add_argument("--num-landscapes", type=int, default=NUM_LANDSCAPES)
    parser.add_argument("--num-local-starts", type=int, default=NUM_LOCAL_STARTS)
    parser.add_argument("--num-population-replicates", type=int, default=NUM_POPULATION_REPLICATES)
    parser.add_argument("--population-size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--total-mutation-rate", type=float, default=TOTAL_MUTATION_RATE)
    parser.add_argument("--num-decay-steps", type=int, default=NUM_DECAY_STEPS)
    parser.add_argument("--global-start-batch-size", type=int, default=GLOBAL_START_BATCH_SIZE)
    return parser


def main() -> None:
    """Generate base landscapes when needed and write Figure 6I raw data."""
    args = build_parser().parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    landscape_path = args.raw_dir / "figure6_new_nk_landscapes.pkl"
    output_path = args.raw_dir / "figure6_i_modularity_gmu_global_local_raw.pkl"
    if output_path.exists() and not args.overwrite_raw:
        print(f"Skipping existing raw payload: {output_path}")
        return
    if landscape_path.exists() and not args.overwrite_landscapes:
        landscapes = load_pickle(landscape_path)
    else:
        landscapes = generate_nk_landscapes(N_SITES, NUM_ALLELES, args.k_values, args.num_landscapes, args.seed)
        save_pickle(landscapes, landscape_path)
    available = np.asarray(landscapes["data"]["landscapes"])
    if available.shape[0] < len(args.k_values) or available.shape[1] < args.num_landscapes:
        raise ValueError("Base landscape cache does not contain the requested K/landscape counts.")
    payload = generate_raw(landscapes, args)
    save_pickle(payload, output_path)
    print(f"Saved Figure 6I raw payload: {output_path}")


if __name__ == "__main__":
    main()
