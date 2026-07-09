"""Generate raw Figure 6 G-H global/local G_mu decay curves.

This standalone script mirrors the G-H raw-data path in
``figure_6_new_new.ipynb`` without modifying the notebook. It writes the raw
pickle that the notebook already loads:

``raw_data/figure6_new_new_lethal_neutral_gmu_global_local_raw.pkl``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from slide.ruggedness_functions import get_dirichlet_metric, get_nk_l_o_shape
from slide.utils import get_processed_data_dir, get_raw_data_dir, load_pickle, save_pickle


N_SITES: int = 4
NUM_ALLELES: int = 20
K_VALUES: tuple[int, ...] = (0, 1, 2, 3)
FRACTIONS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)
NUM_LANDSCAPES: int = 20
NUM_LOCAL_STARTS: int = 20
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

    expected_shape = (len(k_values), num_landscapes, *((num_alleles,) * n_sites))
    if landscapes.shape != expected_shape:
        raise ValueError(f"Expected landscape shape {expected_shape}, received {landscapes.shape}.")

    output_shape = (len(k_values), len(fractions), num_landscapes)
    lethal_rho2 = np.empty(output_shape, dtype=float)
    neutral_rho2 = np.empty(output_shape, dtype=float)
    ridge_sites = np.empty((len(k_values), num_landscapes), dtype=np.int16)
    reference_alleles = np.empty((len(k_values), num_landscapes), dtype=np.int16)
    alternative_allele_orders = np.empty((len(k_values), num_landscapes, num_alleles - 1), dtype=np.int16)
    total_nodes = num_alleles**n_sites
    lethal_counts = np.rint(fraction_array * total_nodes).astype(int)
    neutral_counts = np.rint(fraction_array * num_alleles).astype(int)
    if np.any(neutral_counts > num_alleles - 1):
        raise ValueError("Neutral fractions request more alternative alleles than are available.")

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
                    if not np.array_equal(neutral_landscape[tuple(target_index)], reference_slice):
                        raise AssertionError("Neutral allele slice was not copied exactly.")
                neutral_rho2[k_index, fraction_index, replicate] = float(get_dirichlet_metric(neutral_landscape))

    if not np.array_equal(lethal_rho2[:, 0, :], neutral_rho2[:, 0, :]):
        raise AssertionError("Zero-fraction lethal and neutral rho_2 values differ.")
    if not np.all(np.isfinite(lethal_rho2)) or not np.all(np.isfinite(neutral_rho2)):
        raise FloatingPointError("Processed rho_2 arrays contain non-finite values.")
    if np.any(np.diff(lethal_counts) < 0) or np.any(np.diff(neutral_counts) < 0):
        raise AssertionError("Perturbation sets are not nested.")

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


def apply_kernel_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Apply one row-stochastic kernel to one function axis.

    Parameters:
    - values: np.ndarray
        Genotype-indexed function values.
    - kernel: np.ndarray
        Single-site row-stochastic kernel.
    - axis: int
        Axis receiving the kernel.

    Returns:
    - np.ndarray
        Kernel-transformed function values.
    """
    transformed = np.tensordot(kernel, values, axes=([1], [axis]))
    return np.moveaxis(transformed, 0, axis)


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

    if ridge_site != int(np.asarray(analytical_data["ridge_sites"])[k_index, landscape_index]):
        raise AssertionError("Reconstructed ridge site differs from analytical metadata.")
    if reference_allele != int(np.asarray(analytical_data["reference_alleles"])[k_index, landscape_index]):
        raise AssertionError("Reconstructed reference allele differs from metadata.")
    if not np.array_equal(
        alternative_alleles,
        np.asarray(analytical_data["alternative_allele_orders"])[k_index, landscape_index],
    ):
        raise AssertionError("Reconstructed allele order differs from metadata.")

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


def uniform_amino_acid_kernel(num_alleles: int, mutation_rate: float) -> np.ndarray:
    """Build the one-site uniform mutation transition used by G/H.

    Parameters:
    - num_alleles: int
        Number of allelic states per site.
    - mutation_rate: float
        Per-site probability of mutating in one generation.

    Returns:
    - np.ndarray
        Row-stochastic transition matrix with uniform non-self mutations.
    """
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must lie in [0, 1].")
    if num_alleles < 2:
        raise ValueError("num_alleles must be at least two.")
    kernel = np.full((num_alleles, num_alleles), mutation_rate / (num_alleles - 1), dtype=np.float64)
    np.fill_diagonal(kernel, 1.0 - mutation_rate)
    return kernel


def compute_uniform_g_mu_curves(
    landscape: np.ndarray,
    local_coordinates: np.ndarray,
    kernel: np.ndarray,
    num_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute deterministic global and local squared-decay curves.

    Parameters:
    - landscape: np.ndarray
        Perturbed fitness landscape.
    - local_coordinates: np.ndarray
        Uniformly sampled local start coordinates.
    - kernel: np.ndarray
        One-site row-stochastic mutation transition.
    - num_steps: int
        Number of mutation-only generations.

    Returns:
    - tuple[np.ndarray, np.ndarray]
        Global all-start G_mu curve and local one-start G_mu curves.
    """
    expected_fitness = np.asarray(landscape, dtype=np.float64)
    local_indices = tuple(np.asarray(local_coordinates, dtype=np.intp).T)
    global_curve = np.empty(num_steps, dtype=np.float64)
    local_curves = np.empty((local_coordinates.shape[0], num_steps), dtype=np.float64)
    for step_index in range(num_steps):
        squared = np.square(expected_fitness)
        global_curve[step_index] = float(squared.mean())
        local_curves[:, step_index] = squared[local_indices]
        for axis in range(expected_fitness.ndim):
            expected_fitness = apply_kernel_axis(expected_fitness, kernel, axis)
    return global_curve, local_curves


def generate_figure6_decay_raw(
    raw_payload: dict[str, object],
    analytical_payload: dict[str, object],
    seed: int,
) -> dict[str, object]:
    """Generate deterministic global and local G_mu curves for panels G/H.

    Parameters:
    - raw_payload: dict[str, object]
        Original NK landscapes.
    - analytical_payload: dict[str, object]
        Existing analytical perturbation results.
    - seed: int
        Master local-start selection seed.

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
    curve_shape = (len(PERTURBATIONS), len(k_values), len(fractions), num_landscapes, NUM_DECAY_STEPS)
    local_shape = curve_shape[:-1] + (NUM_LOCAL_STARTS, NUM_DECAY_STEPS)
    start_shape = curve_shape[:-1] + (NUM_LOCAL_STARTS, N_SITES)
    g_mu_global = np.empty(curve_shape, dtype=np.float32)
    g_mu_local = np.empty(local_shape, dtype=np.float32)
    local_start_coordinates = np.empty(start_shape, dtype=np.int16)
    selection_seeds = np.empty(curve_shape[:-1], dtype=np.uint32)
    kernel = uniform_amino_acid_kernel(NUM_ALLELES, TOTAL_MUTATION_RATE / N_SITES)
    all_flat_indices = np.arange(NUM_ALLELES**N_SITES, dtype=np.int64)

    for perturbation_index, perturbation in enumerate(PERTURBATIONS):
        for k_index, k_value in enumerate(k_values):
            print(f"Generating {perturbation} K={k_value}")
            for fraction_index in range(len(fractions)):
                for landscape_index in range(num_landscapes):
                    perturbed = reconstruct_perturbed_landscape(
                        raw_payload,
                        analytical_payload,
                        perturbation,
                        k_index,
                        fraction_index,
                        landscape_index,
                    )
                    expected_rho2 = float(
                        np.asarray(analytical_data[f"{perturbation}_rho2"])[k_index, fraction_index, landscape_index]
                    )
                    if not np.isclose(float(get_dirichlet_metric(perturbed)), expected_rho2, rtol=1e-5, atol=1e-6):
                        raise AssertionError("Reconstructed perturbation differs from analytical result.")
                    stream_index = (
                        ((perturbation_index * len(k_values) + k_index) * len(fractions) + fraction_index)
                        * num_landscapes
                        + landscape_index
                    )
                    selection_seed = seed + stream_index
                    output_index = (perturbation_index, k_index, fraction_index, landscape_index)
                    selection_seeds[output_index] = selection_seed
                    selection_rng = np.random.default_rng(selection_seed)
                    local_flat = selection_rng.choice(all_flat_indices, size=NUM_LOCAL_STARTS, replace=False)
                    local_coordinates = np.column_stack(np.unravel_index(local_flat, perturbed.shape)).astype(np.int16)
                    global_curve, local_curves = compute_uniform_g_mu_curves(
                        perturbed,
                        local_coordinates,
                        kernel,
                        NUM_DECAY_STEPS,
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
            "amino_acid_mutation_kernel": kernel,
        },
        "params": {
            "N": N_SITES,
            "A": NUM_ALLELES,
            "K_values": k_values,
            "fractions": tuple(float(value) for value in fractions),
            "num_landscapes": num_landscapes,
            "perturbations": PERTURBATIONS,
            "num_local_starts": NUM_LOCAL_STARTS,
            "total_mutation_rate": TOTAL_MUTATION_RATE,
            "mutation_rate_per_site": TOTAL_MUTATION_RATE / N_SITES,
            "num_decay_steps": NUM_DECAY_STEPS,
            "seed": seed,
        },
        "metadata": {
            "paper_reference": "Figure 6 new panels G-H fitted decay rates",
            "global_definition": "G_mu averaged exactly over all genotype starts.",
            "local_definition": "Uniformly sampled genotype starts without replacement.",
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
    raw_dir = get_raw_data_dir()
    processed_dir = get_processed_data_dir()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=raw_dir, help="Directory containing/writing raw data pickles.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=processed_dir,
        help="Directory containing/writing processed analytical perturbation pickles.",
    )
    parser.add_argument("--overwrite-raw", action="store_true", help="Overwrite the G-H raw G_mu output pickle.")
    parser.add_argument("--overwrite-landscapes", action="store_true", help="Regenerate the NK landscape raw cache.")
    parser.add_argument(
        "--overwrite-analytical",
        action="store_true",
        help="Regenerate the analytical lethal/neutral rho_2 cache.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Base seed matching figure_6_new_new.ipynb.")
    return parser


def main() -> None:
    """Run raw Figure 6 G-H data generation.

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
        figure6_new_raw = generate_figure6_new_landscapes(N_SITES, NUM_ALLELES, K_VALUES, NUM_LANDSCAPES, args.seed)
        save_pickle(figure6_new_raw, raw_path)
        print(f"Saved NK landscape cache: {raw_path}")

    if analytical_path.exists() and not args.overwrite_analytical:
        figure6_new_payload = load_pickle(analytical_path)
        print(f"Loaded analytical perturbation cache: {analytical_path}")
    else:
        figure6_new_payload = process_figure6_new_landscapes(figure6_new_raw, FRACTIONS, args.seed + 1)
        save_pickle(figure6_new_payload, analytical_path)
        print(f"Saved analytical perturbation cache: {analytical_path}")

    if decay_raw_path.exists() and not args.overwrite_raw:
        print(f"Skipping existing G-H raw payload: {decay_raw_path}")
        return

    figure6_decay_raw = generate_figure6_decay_raw(figure6_new_raw, figure6_new_payload, args.seed + 2)
    save_pickle(figure6_decay_raw, decay_raw_path)

    raw_global = np.asarray(figure6_decay_raw["data"]["g_mu_global"])  # type: ignore[index]
    raw_local = np.asarray(figure6_decay_raw["data"]["g_mu_local"])  # type: ignore[index]
    expected_global_shape = (2, 4, 6, 20, 75)
    expected_local_shape = (2, 4, 6, 20, 20, 75)
    if raw_global.shape != expected_global_shape:
        raise AssertionError(f"Expected global shape {expected_global_shape}, received {raw_global.shape}.")
    if raw_local.shape != expected_local_shape:
        raise AssertionError(f"Expected local shape {expected_local_shape}, received {raw_local.shape}.")
    print(f"Saved G-H raw payload: {decay_raw_path}")


if __name__ == "__main__":
    main()
