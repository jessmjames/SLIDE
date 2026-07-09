"""Generate stochastic raw Figure 6 G-H G_mu curves.

This standalone script mirrors the G-H raw-data path expected by
``figure_6_new_new.ipynb``. For each perturbed synthetic NK landscape, it
estimates fitted-decay inputs using the same procedure as the other stochastic
decay notebooks: run mutation-only population replicates per start, average
replicates to F_mu, square to start-level G_mu, and average start-level G_mu
over all starts for the global curve.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from slide.direvo_functions import (  # noqa: E402
    build_empirical_landscape_function,
    build_mutation_function,
    run_diffusion,
)
from slide.ruggedness_functions import get_dirichlet_metric, get_nk_l_o_shape  # noqa: E402
from slide.utils import get_processed_data_dir, get_raw_data_dir, load_pickle, save_pickle  # noqa: E402


N_SITES: int = 4
NUM_ALLELES: int = 20
K_VALUES: tuple[int, ...] = (0, 1, 2, 3)
FRACTIONS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)
NUM_LANDSCAPES: int = 20
NUM_LOCAL_STARTS: int = 20
NUM_POPULATION_REPLICATES: int = 10
POPULATION_SIZE: int = 2_500
TOTAL_MUTATION_RATE: float = 0.1
NUM_DECAY_STEPS: int = 75
PERTURBATIONS: tuple[str, ...] = ("lethal", "neutral")
RANDOM_SEED: int = 42


def generate_figure6_new_landscapes(
    n_sites: int,
    num_alleles: int,
    k_values: tuple[int, ...],
    num_landscapes: int,
    seed: int,
) -> dict[str, object]:
    """Generate paired NK landscapes for the lethal and neutral analyses.

    Parameters:
    - n_sites: int
        Number of genotype sites.
    - num_alleles: int
        Number of alleles per site.
    - k_values: tuple[int, ...]
        NK epistatic interaction values.
    - num_landscapes: int
        Independent landscapes generated for every K value.
    - seed: int
        Base JAX random seed.

    Returns:
    - dict[str, object]
        Raw landscape array and complete generation metadata.
    """
    shape = (num_alleles,) * n_sites
    landscapes = np.empty((len(k_values), num_landscapes, *shape), dtype=np.float32)
    base_key = jr.PRNGKey(seed)
    for k_index, k_value in enumerate(k_values):
        for replicate in range(num_landscapes):
            landscape_key = jr.fold_in(jr.fold_in(base_key, k_value), replicate)
            landscapes[k_index, replicate] = np.asarray(
                get_nk_l_o_shape(landscape_key, n_sites, k_value, shape),
                dtype=np.float32,
            )
    return {
        "data": {"landscapes": landscapes},
        "params": {
            "N": n_sites,
            "A": num_alleles,
            "K_values": k_values,
            "num_landscapes": num_landscapes,
            "seed": seed,
        },
        "metadata": {
            "paper_reference": "Figure 6 new",
            "description": "Paired N=4, A=20 NK landscapes for lethal-node and neutral-ridge analyses.",
        },
    }


def process_figure6_new_landscapes(
    raw_payload: dict[str, object],
    fractions: tuple[float, ...],
    perturbation_seed: int,
) -> dict[str, object]:
    """Compute analytical rho_2 after nested lethal and neutral perturbations.

    Parameters:
    - raw_payload: dict[str, object]
        Paired NK landscapes and generation parameters.
    - fractions: tuple[float, ...]
        Fractions of flattened values overwritten at each perturbation level.
    - perturbation_seed: int
        Seed controlling node order, ridge site, reference allele, and allele order.

    Returns:
    - dict[str, object]
        Replicate-level rho_2 values, summaries, perturbation choices, and metadata.
    """
    landscapes = np.asarray(raw_payload["data"]["landscapes"], dtype=np.float32)  # type: ignore[index]
    params = raw_payload["params"]  # type: ignore[index]
    if not isinstance(params, dict):
        raise TypeError("raw_payload['params'] must be a dictionary.")
    n_sites = int(params["N"])
    num_alleles = int(params["A"])
    k_values = tuple(int(value) for value in params["K_values"])  # type: ignore[union-attr]
    num_landscapes = int(params["num_landscapes"])
    fraction_array = np.asarray(fractions, dtype=float)
    if np.any(np.diff(fraction_array) < 0) or fraction_array[0] != 0.0:
        raise ValueError("Fractions must be increasing and begin at zero.")

    output_shape = (len(k_values), len(fractions), num_landscapes)
    lethal_rho2 = np.empty(output_shape, dtype=float)
    neutral_rho2 = np.empty(output_shape, dtype=float)
    ridge_sites = np.empty((len(k_values), num_landscapes), dtype=np.int16)
    reference_alleles = np.empty((len(k_values), num_landscapes), dtype=np.int16)
    alternative_allele_orders = np.empty((len(k_values), num_landscapes, num_alleles - 1), dtype=np.int16)
    total_nodes = num_alleles**n_sites
    lethal_counts = np.rint(fraction_array * total_nodes).astype(int)
    neutral_counts = np.rint(fraction_array * num_alleles).astype(int)

    for k_index, k_value in enumerate(k_values):
        for replicate in range(num_landscapes):
            landscape = landscapes[k_index, replicate]
            rng = np.random.default_rng(perturbation_seed + 10_000 * k_value + replicate)
            lethal_order = rng.permutation(total_nodes)
            ridge_site = int(rng.integers(0, n_sites))
            reference_allele = int(rng.integers(0, num_alleles))
            alternative_alleles = np.delete(np.arange(num_alleles), reference_allele)
            rng.shuffle(alternative_alleles)
            ridge_sites[k_index, replicate] = ridge_site
            reference_alleles[k_index, replicate] = reference_allele
            alternative_allele_orders[k_index, replicate] = alternative_alleles

            base_rho2 = float(get_dirichlet_metric(landscape))
            lethal_rho2[k_index, 0, replicate] = base_rho2
            neutral_rho2[k_index, 0, replicate] = base_rho2
            landscape_minimum = float(landscape.min())
            reference_index: list[slice | int] = [slice(None)] * n_sites
            reference_index[ridge_site] = reference_allele
            reference_slice = landscape[tuple(reference_index)].copy()

            for fraction_index in range(1, len(fractions)):
                lethal_landscape = landscape.copy()
                lethal_landscape.reshape(-1)[lethal_order[: lethal_counts[fraction_index]]] = landscape_minimum
                lethal_rho2[k_index, fraction_index, replicate] = float(get_dirichlet_metric(lethal_landscape))

                neutral_landscape = landscape.copy()
                for allele in alternative_alleles[: neutral_counts[fraction_index]]:
                    target_index: list[slice | int] = [slice(None)] * n_sites
                    target_index[ridge_site] = int(allele)
                    neutral_landscape[tuple(target_index)] = reference_slice
                neutral_rho2[k_index, fraction_index, replicate] = float(get_dirichlet_metric(neutral_landscape))

    return {
        "data": {
            "fractions": fraction_array,
            "lethal_rho2": lethal_rho2,
            "neutral_rho2": neutral_rho2,
            "lethal_mean": lethal_rho2.mean(axis=2),
            "lethal_std": lethal_rho2.std(axis=2, ddof=1),
            "neutral_mean": neutral_rho2.mean(axis=2),
            "neutral_std": neutral_rho2.std(axis=2, ddof=1),
            "ridge_sites": ridge_sites,
            "reference_alleles": reference_alleles,
            "alternative_allele_orders": alternative_allele_orders,
            "lethal_counts": lethal_counts,
            "neutral_counts": neutral_counts,
        },
        "params": {
            "N": n_sites,
            "A": num_alleles,
            "K_values": k_values,
            "num_landscapes": num_landscapes,
            "fractions": fractions,
            "perturbation_seed": perturbation_seed,
        },
        "metadata": {
            "paper_reference": "Figure 6 new panels G-H",
            "standard_deviation_ddof": 1,
            "lethal_definition": "Nested random nodes forced to each landscape's global minimum.",
            "neutral_definition": "Nested alternative-allele slices copied from one reference allele at one focal site.",
            "neutral_fraction_definition": "Fraction of genotype values overwritten.",
        },
    }


def reconstruct_perturbed_landscape(
    raw_payload: dict[str, object],
    analytical_payload: dict[str, object],
    perturbation: str,
    k_index: int,
    fraction_index: int,
    landscape_index: int,
) -> np.ndarray:
    """Reconstruct a panel G/H perturbed landscape.

    Parameters:
    - raw_payload: dict[str, object]
        Original NK landscapes.
    - analytical_payload: dict[str, object]
        Existing analytical perturbation results.
    - perturbation: str
        Lethal or neutral perturbation name.
    - k_index: int
        K-value index.
    - fraction_index: int
        Perturbation-fraction index.
    - landscape_index: int
        Landscape-replicate index.

    Returns:
    - np.ndarray
        Reconstructed perturbed landscape.
    """
    if perturbation not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation {perturbation!r}.")
    raw_data = raw_payload["data"]
    analytical_data = analytical_payload["data"]
    analytical_params = analytical_payload["params"]
    if not isinstance(raw_data, dict) or not isinstance(analytical_data, dict) or not isinstance(analytical_params, dict):
        raise TypeError("Payload data and params entries must be dictionaries.")

    landscape = np.asarray(raw_data["landscapes"], dtype=np.float32)[k_index, landscape_index]
    k_value = int(analytical_params["K_values"][k_index])  # type: ignore[index]
    rng = np.random.default_rng(int(analytical_params["perturbation_seed"]) + 10_000 * k_value + landscape_index)
    lethal_order = rng.permutation(landscape.size)
    ridge_site = int(rng.integers(0, landscape.ndim))
    reference_allele = int(rng.integers(0, landscape.shape[ridge_site]))
    alternative_alleles = np.delete(np.arange(landscape.shape[ridge_site]), reference_allele)
    rng.shuffle(alternative_alleles)

    perturbed = landscape.copy()
    if perturbation == "lethal":
        count = int(np.asarray(analytical_data["lethal_counts"])[fraction_index])
        perturbed.reshape(-1)[lethal_order[:count]] = float(landscape.min())
    else:
        count = int(np.asarray(analytical_data["neutral_counts"])[fraction_index])
        reference_index: list[slice | int] = [slice(None)] * landscape.ndim
        reference_index[ridge_site] = reference_allele
        reference_slice = landscape[tuple(reference_index)].copy()
        for allele in alternative_alleles[:count]:
            target_index: list[slice | int] = [slice(None)] * landscape.ndim
            target_index[ridge_site] = int(allele)
            perturbed[tuple(target_index)] = reference_slice
    return perturbed


def fitness_history_only(*, fitnesses: jax.Array, pop: jax.Array) -> jax.Array:
    """Return per-generation population fitness values for diffusion histories.

    Parameters:
    - fitnesses: jax.Array
        Fitness values recorded for the current population.
    - pop: jax.Array
        Current population, unused here.

    Returns:
    - jax.Array
        Fitness values for the current population.
    """
    return fitnesses


def linear_ids_to_starts(linear_ids: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Convert flat genotype IDs into coordinate starts.

    Parameters:
    - linear_ids: np.ndarray
        One-dimensional flat genotype IDs.
    - shape: tuple[int, ...]
        Landscape genotype shape.

    Returns:
    - np.ndarray
        Coordinate array with shape ``(len(linear_ids), len(shape))``.
    """
    return np.column_stack(np.unravel_index(linear_ids, shape)).astype(np.int32)


def build_start_fmu_runner(
    landscape: np.ndarray,
    popsize: int,
    mutation_rate_per_site: float,
    num_replicates_per_start: int,
    num_steps: int,
    seed: int,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Build a jitted runner returning one F_mu curve per start.

    Parameters:
    - landscape: np.ndarray
        Perturbed landscape used for fitness lookup.
    - popsize: int
        Population size for each replicate trajectory.
    - mutation_rate_per_site: float
        Per-site mutation probability.
    - num_replicates_per_start: int
        Number of stochastic replicates averaged into F_mu.
    - num_steps: int
        Number of mutation-only generations.
    - seed: int
        Base seed for stochastic replicate trajectories.

    Returns:
    - Callable[[jax.Array, jax.Array], jax.Array]
        Jitted callable mapping starts and flat IDs to F_mu curves.
    """
    fitness_function = build_empirical_landscape_function(jnp.asarray(landscape))
    mutation_function = build_mutation_function(mutation_rate_per_site, landscape.shape[0])
    base_key = jr.PRNGKey(seed)
    history_functions = {"fitness": fitness_history_only}

    def run_one_start(start: jax.Array, start_id: jax.Array) -> jax.Array:
        start_key = jr.fold_in(base_key, start_id.astype(jnp.uint32))
        replicate_keys = jr.split(start_key, num_replicates_per_start)
        initial_population = jnp.broadcast_to(start[None], (popsize, start.shape[0]))

        def run_one_replicate(replicate_key: jax.Array) -> jax.Array:
            history = run_diffusion(
                replicate_key,
                initial_population,
                mutation_function,
                fitness_function=fitness_function,
                num_steps=num_steps,
                extra_function_dict=history_functions,
            )[1]
            return history["fitness"].mean(axis=-1)

        return jax.vmap(run_one_replicate)(replicate_keys).mean(axis=0)

    return jax.jit(jax.vmap(run_one_start, in_axes=(0, 0)))


def compute_stochastic_g_mu_curves(
    landscape: np.ndarray,
    local_coordinates: np.ndarray,
    local_flat_ids: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute stochastic global and local G_mu curves for one landscape.

    Parameters:
    - landscape: np.ndarray
        Perturbed fitness landscape.
    - local_coordinates: np.ndarray
        Uniformly sampled local start coordinates.
    - local_flat_ids: np.ndarray
        Flat IDs matching ``local_coordinates``.
    - seed: int
        Base trajectory seed.
    - args: argparse.Namespace
        Parsed runtime parameters.

    Returns:
    - tuple[np.ndarray, np.ndarray]
        Global all-start G_mu curve and local one-start G_mu curves.
    """
    runner = build_start_fmu_runner(
        landscape,
        int(args.population_size),
        float(args.total_mutation_rate / N_SITES),
        int(args.num_population_replicates),
        int(args.num_decay_steps),
        seed,
    )
    local_f_mu = np.asarray(
        runner(jnp.asarray(local_coordinates, dtype=jnp.int32), jnp.asarray(local_flat_ids, dtype=jnp.uint32)),
        dtype=np.float64,
    )
    local_g_mu = np.square(local_f_mu)

    global_sum = np.zeros(int(args.num_decay_steps), dtype=np.float64)
    total_starts = int(landscape.size)
    for chunk_start in range(0, total_starts, int(args.global_start_batch_size)):
        chunk_stop = min(chunk_start + int(args.global_start_batch_size), total_starts)
        flat_ids = np.arange(chunk_start, chunk_stop, dtype=np.uint32)
        starts = linear_ids_to_starts(flat_ids, landscape.shape)
        f_mu = np.asarray(
            runner(jnp.asarray(starts, dtype=jnp.int32), jnp.asarray(flat_ids, dtype=jnp.uint32)),
            dtype=np.float64,
        )
        global_sum += np.square(f_mu).sum(axis=0)
    return global_sum / total_starts, local_g_mu


def generate_figure6_decay_raw(
    raw_payload: dict[str, object],
    analytical_payload: dict[str, object],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Generate stochastic global and local G_mu curves for panels G/H.

    Parameters:
    - raw_payload: dict[str, object]
        Original NK landscapes.
    - analytical_payload: dict[str, object]
        Existing analytical perturbation results.
    - seed: int
        Master local-start selection and trajectory seed.
    - args: argparse.Namespace
        Parsed runtime parameters.

    Returns:
    - dict[str, object]
        Compact global/local G_mu curves, local starts, seeds, parameters, and metadata.
    """
    analytical_data = analytical_payload["data"]
    analytical_params = analytical_payload["params"]
    if not isinstance(analytical_data, dict) or not isinstance(analytical_params, dict):
        raise TypeError("analytical_payload data and params entries must be dictionaries.")
    fractions = np.asarray(analytical_data["fractions"], dtype=float)
    k_values = tuple(int(value) for value in analytical_params["K_values"])  # type: ignore[union-attr]
    num_landscapes = int(analytical_params["num_landscapes"])
    curve_shape = (len(PERTURBATIONS), len(k_values), len(fractions), num_landscapes, int(args.num_decay_steps))
    local_shape = curve_shape[:-1] + (int(args.num_local_starts), int(args.num_decay_steps))
    start_shape = curve_shape[:-1] + (int(args.num_local_starts), N_SITES)
    g_mu_global = np.empty(curve_shape, dtype=np.float32)
    g_mu_local = np.empty(local_shape, dtype=np.float32)
    local_start_coordinates = np.empty(start_shape, dtype=np.int16)
    selection_seeds = np.empty(curve_shape[:-1], dtype=np.uint32)
    simulation_seeds = np.empty(curve_shape[:-1], dtype=np.uint32)
    all_flat_indices = np.arange(NUM_ALLELES**N_SITES, dtype=np.int64)

    for perturbation_index, perturbation in enumerate(PERTURBATIONS):
        for k_index, k_value in enumerate(k_values):
            for fraction_index in range(len(fractions)):
                for landscape_index in range(num_landscapes):
                    print(
                        f"{perturbation} K={k_value} fraction={fractions[fraction_index]:.2f} "
                        f"landscape={landscape_index + 1}/{num_landscapes}",
                        flush=True,
                    )
                    perturbed = reconstruct_perturbed_landscape(
                        raw_payload,
                        analytical_payload,
                        perturbation,
                        k_index,
                        fraction_index,
                        landscape_index,
                    )
                    stream_index = (
                        ((perturbation_index * len(k_values) + k_index) * len(fractions) + fraction_index)
                        * num_landscapes
                        + landscape_index
                    )
                    selection_seed = seed + 2 * stream_index
                    simulation_seed = selection_seed + 1
                    output_index = (perturbation_index, k_index, fraction_index, landscape_index)
                    selection_seeds[output_index] = selection_seed
                    simulation_seeds[output_index] = simulation_seed
                    selection_rng = np.random.default_rng(selection_seed)
                    local_flat = selection_rng.choice(
                        all_flat_indices,
                        size=int(args.num_local_starts),
                        replace=False,
                    ).astype(np.uint32)
                    local_coordinates = linear_ids_to_starts(local_flat, perturbed.shape).astype(np.int16)
                    global_curve, local_curves = compute_stochastic_g_mu_curves(
                        perturbed,
                        local_coordinates,
                        local_flat,
                        simulation_seed,
                        args,
                    )
                    g_mu_global[output_index] = global_curve
                    g_mu_local[output_index] = local_curves
                    local_start_coordinates[output_index] = local_coordinates

    if not np.all(np.isfinite(g_mu_global)) or not np.all(np.isfinite(g_mu_local)):
        raise AssertionError("Raw G_mu curves contain non-finite values.")
    return {
        "data": {
            "g_mu_global": g_mu_global,
            "g_mu_local": g_mu_local,
            "local_start_coordinates": local_start_coordinates,
            "selection_seeds": selection_seeds,
            "simulation_seeds": simulation_seeds,
        },
        "params": {
            "N": N_SITES,
            "A": NUM_ALLELES,
            "K_values": k_values,
            "fractions": tuple(float(value) for value in fractions),
            "num_landscapes": num_landscapes,
            "perturbations": PERTURBATIONS,
            "num_local_starts": int(args.num_local_starts),
            "num_population_replicates": int(args.num_population_replicates),
            "population_size": int(args.population_size),
            "total_mutation_rate": float(args.total_mutation_rate),
            "mutation_rate_per_site": float(args.total_mutation_rate / N_SITES),
            "num_decay_steps": int(args.num_decay_steps),
            "global_start_batch_size": int(args.global_start_batch_size),
            "seed": seed,
        },
        "metadata": {
            "paper_reference": "Figure 6 new panels G-H fitted decay rates",
            "global_definition": (
                "For every genotype start, average stochastic population replicates "
                "to F_mu, square to start-level G_mu, then average over all starts."
            ),
            "local_definition": (
                "For each sampled local start, average stochastic population replicates "
                "to F_mu and square to one-start G_mu."
            ),
            "trajectory_axes": ("perturbation", "K", "fraction", "landscape", "generation"),
            "local_axes": ("perturbation", "K", "fraction", "landscape", "local_start", "generation"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Parameters:
    - None

    Returns:
    - argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=get_raw_data_dir(), help="Directory containing/writing raw data pickles.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=get_processed_data_dir(),
        help="Directory containing/writing processed analytical perturbation pickles.",
    )
    parser.add_argument("--overwrite-raw", action="store_true", help="Overwrite the G-H raw G_mu output pickle.")
    parser.add_argument("--overwrite-landscapes", action="store_true", help="Regenerate the NK landscape raw cache.")
    parser.add_argument("--overwrite-analytical", action="store_true", help="Regenerate the analytical lethal/neutral rho_2 cache.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Base seed matching figure_6_new_new.ipynb.")
    parser.add_argument("--population-size", type=int, default=POPULATION_SIZE, help="Population size per stochastic replicate.")
    parser.add_argument(
        "--num-population-replicates",
        type=int,
        default=NUM_POPULATION_REPLICATES,
        help="Stochastic replicate trajectories averaged into each F_mu.",
    )
    parser.add_argument("--num-local-starts", type=int, default=NUM_LOCAL_STARTS, help="Local one-start samples per landscape.")
    parser.add_argument("--num-decay-steps", type=int, default=NUM_DECAY_STEPS, help="Number of generations recorded.")
    parser.add_argument("--total-mutation-rate", type=float, default=TOTAL_MUTATION_RATE, help="Total mutations per genotype per step.")
    parser.add_argument("--global-start-batch-size", type=int, default=100, help="Number of all-start genotypes processed per JAX batch.")
    return parser


def main() -> None:
    """Run raw Figure 6 G-H stochastic data generation.

    Parameters:
    - None

    Returns:
    - None
        Writes the notebook-compatible raw G-H pickle.
    """
    args = build_parser().parse_args()
    raw_path = args.raw_dir / "figure6_new_nk_landscapes.pkl"
    analytical_path = args.processed_dir / "figure6_new_lethal_neutral_rho2.pkl"
    decay_raw_path = args.raw_dir / "figure6_new_new_lethal_neutral_gmu_global_local_raw.pkl"

    if raw_path.exists() and not args.overwrite_landscapes:
        figure6_new_raw = load_pickle(raw_path)
        print(f"Loaded NK landscape cache: {raw_path}")
    else:
        figure6_new_raw = generate_figure6_new_landscapes(N_SITES, NUM_ALLELES, K_VALUES, NUM_LANDSCAPES, int(args.seed))
        save_pickle(figure6_new_raw, raw_path)
        print(f"Saved NK landscape cache: {raw_path}")

    if analytical_path.exists() and not args.overwrite_analytical:
        figure6_new_payload = load_pickle(analytical_path)
        print(f"Loaded analytical perturbation cache: {analytical_path}")
    else:
        figure6_new_payload = process_figure6_new_landscapes(figure6_new_raw, FRACTIONS, int(args.seed) + 1)
        save_pickle(figure6_new_payload, analytical_path)
        print(f"Saved analytical perturbation cache: {analytical_path}")

    if decay_raw_path.exists() and not args.overwrite_raw:
        print(f"Skipping existing G-H raw payload: {decay_raw_path}")
        return

    figure6_decay_raw = generate_figure6_decay_raw(figure6_new_raw, figure6_new_payload, int(args.seed) + 2, args)
    save_pickle(figure6_decay_raw, decay_raw_path)

    raw_global = np.asarray(figure6_decay_raw["data"]["g_mu_global"])  # type: ignore[index]
    raw_local = np.asarray(figure6_decay_raw["data"]["g_mu_local"])  # type: ignore[index]
    expected_global_shape = (2, 4, 6, 20, int(args.num_decay_steps))
    expected_local_shape = (2, 4, 6, 20, int(args.num_local_starts), int(args.num_decay_steps))
    if raw_global.shape != expected_global_shape:
        raise AssertionError(f"Expected global shape {expected_global_shape}, received {raw_global.shape}.")
    if raw_local.shape != expected_local_shape:
        raise AssertionError(f"Expected local shape {expected_local_shape}, received {raw_local.shape}.")
    print(f"Saved G-H stochastic raw payload: {decay_raw_path}")


if __name__ == "__main__":
    main()
