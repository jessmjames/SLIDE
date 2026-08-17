"""Generate full-nucleotide start-subsampled G_mu products for Figure 6 D-F.

This script streams over the full nucleotide genotype space, computes one
start-level F_mu curve by averaging replicate trajectories from the same start,
squares that start-level curve, and accumulates compact G_mu prefix averages for
deterministic random orderings of the start space.
"""

from __future__ import annotations

import argparse
import pickle
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
    FitnessFunction,
    MutationFunction,
    build_custom_mutation_function,
    build_mutation_function,
    build_selection_function,
    get_pre_defined_landscape_function_with_codon,
    run_directed_evolution,
)
from slide.selection_function_library import base_chance_threshold_select  # noqa: E402
from slide.utils import get_raw_data_dir  # noqa: E402


LANDSCAPE_FILES: dict[str, str] = {
    "gb1": "GB1_landscape_array.pkl",
    "trpb": "TrpB_landscape_array.pkl",
    "tev": "TEV_landscape_array.pkl",
    "pard3": "E3_landscape_array.pkl",
}

MUTATION_MATRIX_FILES: dict[str, str | None] = {
    "nuc_uniform": None,
    "nuc_e_coli_weighted": "normed_e_coli_matrix.npy",
    "nuc_e_coli_directed": "normed_e_coli_matrix.npy",
    "nuc_a_thaliana_weighted": "normed_a_thaliana_matrix.npy",
    "nuc_a_thaliana_directed": "normed_a_thaliana_matrix.npy",
    "nuc_human_directed": "normed_human_codon_matrix.npy",
}


def parse_csv_choices(value: str, allowed_values: Sequence[str]) -> list[str]:
    """Parse comma-separated choices and expand the value ``all``.

    Parameters
    ----------
    value:
        Comma-separated command-line value.
    allowed_values:
        Valid names in their desired default order.

    Returns
    -------
    list[str]
        Parsed names.
    """
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    if requested == ["all"]:
        return list(allowed_values)
    invalid = sorted(set(requested) - set(allowed_values))
    if invalid:
        raise ValueError(f"Unknown choices {invalid}. Allowed values: {list(allowed_values)}")
    return requested


def parse_counts(value: str, total_starts: int) -> np.ndarray:
    """Parse, clip, and deduplicate start-count prefixes.

    Parameters
    ----------
    value:
        Comma-separated positive integer counts.
    total_starts:
        Full number of nucleotide starts for the landscape.

    Returns
    -------
    np.ndarray
        Sorted count array including ``total_starts``.
    """
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("At least one start count is required.")
    if any(count <= 0 for count in parsed):
        raise ValueError("All start counts must be positive.")
    clipped = [min(count, total_starts) for count in parsed]
    clipped.append(total_starts)
    return np.array(sorted(set(clipped)), dtype=np.int64)


def load_landscape(landscape_name: str) -> np.ndarray:
    """Load an empirical amino-acid landscape array.

    Parameters
    ----------
    landscape_name:
        Short landscape key.

    Returns
    -------
    np.ndarray
        Amino-acid fitness landscape.
    """
    path = REPO_ROOT / "landscape_arrays" / LANDSCAPE_FILES[landscape_name]
    with path.open("rb") as handle:
        landscape = pickle.load(handle)
    return np.asarray(landscape)


def load_mutation_matrix(model_name: str) -> np.ndarray | None:
    """Load and validate a nucleotide transition matrix.

    Parameters
    ----------
    model_name:
        Mutation model key.

    Returns
    -------
    np.ndarray | None
        Row-stochastic 4 by 4 transition matrix, or ``None`` for uniform
        nucleotide mutations.
    """
    matrix_file = MUTATION_MATRIX_FILES[model_name]
    if matrix_file is None:
        return None
    matrix = np.asarray(np.load(REPO_ROOT / "other_data" / matrix_file), dtype=np.float64)
    if model_name in {"nuc_e_coli_weighted", "nuc_a_thaliana_weighted"}:
        matrix = symmetric_sinkhorn_kernel(matrix)
    if matrix.shape != (4, 4):
        raise ValueError(f"{model_name} matrix has shape {matrix.shape}, expected (4, 4).")
    if np.any(matrix < 0.0):
        raise ValueError(f"{model_name} matrix contains negative entries.")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError(f"{model_name} matrix rows do not sum to 1.")
    return matrix


def symmetric_sinkhorn_kernel(kernel: np.ndarray, tolerance: float = 1e-13) -> np.ndarray:
    """Create a symmetric doubly-stochastic kernel by diagonal scaling.

    Parameters
    ----------
    kernel:
        Non-negative square base kernel.
    tolerance:
        Maximum permitted row-sum error.

    Returns
    -------
    np.ndarray
        Symmetric doubly-stochastic kernel preserving the input zero pattern.
    """
    symmetric = 0.5 * (np.asarray(kernel, dtype=np.float64) + np.asarray(kernel, dtype=np.float64).T)
    scale = np.ones(symmetric.shape[0], dtype=np.float64)
    for _ in range(100_000):
        row_sums = scale * (symmetric @ scale)
        if np.max(np.abs(row_sums - 1.0)) < tolerance:
            break
        scale *= np.sqrt(1.0 / row_sums)
    else:
        raise RuntimeError("Symmetric Sinkhorn scaling did not converge.")
    return scale[:, None] * symmetric * scale[None, :]


def linear_ids_to_nucleotide_starts(linear_ids: np.ndarray, nucleotide_sites: int) -> np.ndarray:
    """Convert linear genotype IDs into base-four nucleotide coordinates.

    Parameters
    ----------
    linear_ids:
        One-dimensional integer genotype IDs.
    nucleotide_sites:
        Number of nucleotide positions.

    Returns
    -------
    np.ndarray
        Coordinates with shape ``(len(linear_ids), nucleotide_sites)`` and
        values in ``0..3``.
    """
    ids = np.asarray(linear_ids, dtype=np.uint64)
    starts = np.empty((len(ids), nucleotide_sites), dtype=np.int32)
    for position in range(nucleotide_sites):
        shift = 2 * (nucleotide_sites - 1 - position)
        starts[:, position] = ((ids >> shift) & np.uint64(3)).astype(np.int32)
    return starts


def build_ordering_parameters(
    num_orderings: int,
    total_starts: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build affine pseudo-permutation parameters for genotype IDs.

    Parameters
    ----------
    num_orderings:
        Number of independent orderings.
    total_starts:
        Full genotype-space size. Must be a power of two for odd multipliers to
        be bijective modulo ``total_starts``.
    seed:
        Random seed for deterministic multiplier and offset generation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Odd multipliers, offsets, and ordering seeds.
    """
    if total_starts <= 0 or total_starts & (total_starts - 1) != 0:
        raise ValueError("total_starts must be a power of two.")
    rng = np.random.default_rng(seed)
    ordering_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=num_orderings, dtype=np.uint64)
    multipliers = rng.integers(1, total_starts, size=num_orderings, dtype=np.uint64) | np.uint64(1)
    offsets = rng.integers(0, total_starts, size=num_orderings, dtype=np.uint64)
    return multipliers, offsets, ordering_seeds


def build_start_fmu_runner(
    fitness_function: FitnessFunction,
    mutation_function: MutationFunction,
    popsize: int,
    num_replicates_per_start: int,
    num_steps: int,
    seed: int,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Build a jitted function that returns one F_mu curve per start.

    Parameters
    ----------
    fitness_function:
        JAX codon-level fitness function.
    mutation_function:
        JAX nucleotide mutation function.
    popsize:
        Population size per trajectory.
    num_replicates_per_start:
        Number of replicate trajectories averaged for each start.
    num_steps:
        Number of generations recorded.
    seed:
        Base PRNG seed.

    Returns
    -------
    Callable[[jax.Array, jax.Array], jax.Array]
        Jitted callable mapping ``(starts, linear_ids)`` to F_mu curves with
        shape ``(num_starts, num_steps)``.
    """
    selection_params = {"threshold": 0.0, "base_chance": 1.0}
    selection_function = build_selection_function(base_chance_threshold_select, selection_params)
    base_key = jr.PRNGKey(seed)

    def run_one_start(start: jax.Array, linear_id: jax.Array) -> jax.Array:
        start_key = jr.fold_in(base_key, linear_id.astype(jnp.uint32))
        replicate_keys = jr.split(start_key, num_replicates_per_start)
        initial_population = jnp.broadcast_to(start[None], (popsize, start.shape[0]))

        def run_one_replicate(replicate_key: jax.Array) -> jax.Array:
            replicate_result = run_directed_evolution(
                replicate_key,
                initial_population,
                selection_function,
                mutation_function,
                fitness_function=fitness_function,
                num_options=4,
                num_steps=num_steps,
            )[1]
            return replicate_result["fitness"].mean(axis=-1)

        replicate_curves = jax.vmap(run_one_replicate)(replicate_keys)
        return replicate_curves.mean(axis=0)

    return jax.jit(jax.vmap(run_one_start, in_axes=(0, 0)))


def accumulate_g_mu_prefixes(
    sum_g_mu: np.ndarray,
    included_counts: np.ndarray,
    linear_ids: np.ndarray,
    start_g_mu: np.ndarray,
    multipliers: np.ndarray,
    offsets: np.ndarray,
    counts: np.ndarray,
    total_starts: int,
) -> None:
    """Accumulate start-level squared curves into ordering/count prefixes.

    Parameters
    ----------
    sum_g_mu:
        Running sums with shape ``(num_orderings, num_counts, num_steps)``.
    included_counts:
        Running start counts with shape ``(num_orderings, num_counts)``.
    linear_ids:
        Linear nucleotide genotype IDs for the current chunk or batch.
    start_g_mu:
        Squared start-level curves with shape ``(len(linear_ids), num_steps)``.
    multipliers:
        Odd affine permutation multipliers.
    offsets:
        Affine permutation offsets.
    counts:
        Prefix counts to accumulate.
    total_starts:
        Full genotype-space size.

    Returns
    -------
    None
        Arrays are updated in place.
    """
    ids = np.asarray(linear_ids, dtype=np.uint64)
    total = np.uint64(total_starts)
    for ordering_index, (multiplier, offset) in enumerate(zip(multipliers, offsets, strict=True)):
        ranks = (ids * multiplier + offset) % total
        for count_index, count in enumerate(counts):
            mask = ranks < np.uint64(count)
            num_included = int(mask.sum())
            if num_included == 0:
                continue
            sum_g_mu[ordering_index, count_index] += start_g_mu[mask].sum(axis=0)
            included_counts[ordering_index, count_index] += num_included


def save_payload(
    output_path: Path,
    g_mu: np.ndarray,
    sum_g_mu: np.ndarray,
    included_counts: np.ndarray,
    counts: np.ndarray,
    params: dict[str, object],
) -> None:
    """Save a metadata-rich full-nucleotide G_mu payload.

    Parameters
    ----------
    output_path:
        Destination pickle path.
    g_mu:
        Prefix-averaged G_mu curves.
    sum_g_mu:
        Raw accumulated G_mu sums.
    included_counts:
        Number of starts included in each prefix.
    counts:
        Requested prefix counts.
    params:
        Generation parameters and model metadata.

    Returns
    -------
    None
        Writes ``output_path``.
    """
    payload = {
        "data": {
            "g_mu": g_mu,
            "sum_g_mu": sum_g_mu,
            "included_counts": included_counts,
            "counts": counts,
        },
        "params": params,
        "metadata": {
            "description": "Full nucleotide-space start-subsampled G_mu curves for Figure 6 D-F.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_landscape_model(
    landscape_name: str,
    model_name: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """Generate and save one landscape/model full-nucleotide G_mu payload.

    Parameters
    ----------
    landscape_name:
        Short landscape key.
    model_name:
        Mutation model key.
    output_dir:
        Directory for raw-derived output products.
    args:
        Parsed command-line arguments.

    Returns
    -------
    pathlib.Path
        Saved output path.
    """
    landscape = load_landscape(landscape_name)
    nucleotide_sites = int(landscape.ndim * 3)
    total_starts = int(4**nucleotide_sites)
    counts = parse_counts(args.counts, total_starts)
    output_path = output_dir / f"figure6_full_nuc_gmu_{landscape_name}_{model_name}_raw.pkl"
    if output_path.exists() and not args.overwrite:
        print(f"Skipping existing product: {output_path}")
        return output_path

    mutation_matrix = load_mutation_matrix(model_name)
    site_rate = float(args.total_mutation_rate / nucleotide_sites)
    fitness_function = get_pre_defined_landscape_function_with_codon(jnp.asarray(landscape))
    if mutation_matrix is None:
        mutation_function = build_mutation_function(site_rate, num_options=4)
    else:
        mutation_function = build_custom_mutation_function(site_rate, mutation_matrix, A=4)

    popsize = int(args.pard3_popsize if landscape_name == "pard3" else args.empirical_popsize)
    f_mu_runner = build_start_fmu_runner(
        fitness_function,
        mutation_function,
        popsize,
        int(args.num_replicates_per_start),
        int(args.num_steps),
        int(args.seed),
    )
    multipliers, offsets, ordering_seeds = build_ordering_parameters(
        int(args.num_orderings),
        total_starts,
        int(args.seed),
    )
    sum_g_mu = np.zeros((args.num_orderings, len(counts), args.num_steps), dtype=np.float64)
    included_counts = np.zeros((args.num_orderings, len(counts)), dtype=np.int64)

    print(
        f"\n{landscape_name.upper()} {model_name}: "
        f"{total_starts:,} starts, {nucleotide_sites} nt sites, counts={counts.tolist()}"
    )
    chunk_starts = range(0, total_starts, int(args.chunk_size))
    for chunk_start in tqdm.tqdm(chunk_starts, total=(total_starts + args.chunk_size - 1) // args.chunk_size):
        chunk_stop = min(chunk_start + int(args.chunk_size), total_starts)
        chunk_ids = np.arange(chunk_start, chunk_stop, dtype=np.uint64)
        for batch_start in range(0, len(chunk_ids), int(args.batch_size)):
            batch_ids = chunk_ids[batch_start : batch_start + int(args.batch_size)]
            batch_starts = linear_ids_to_nucleotide_starts(batch_ids, nucleotide_sites)
            f_mu = np.asarray(
                f_mu_runner(jnp.asarray(batch_starts), jnp.asarray(batch_ids, dtype=jnp.uint32)),
                dtype=np.float64,
            )
            start_g_mu = f_mu**2
            accumulate_g_mu_prefixes(
                sum_g_mu,
                included_counts,
                batch_ids,
                start_g_mu,
                multipliers,
                offsets,
                counts,
                total_starts,
            )

    expected_counts = np.broadcast_to(counts[None, :], included_counts.shape)
    if not np.array_equal(included_counts, expected_counts):
        raise RuntimeError(
            f"Included counts mismatch for {landscape_name}/{model_name}: "
            f"expected {expected_counts}, observed {included_counts}"
        )
    with np.errstate(invalid="raise", divide="raise"):
        g_mu = sum_g_mu / included_counts[:, :, None]
    if g_mu.shape != (args.num_orderings, len(counts), args.num_steps):
        raise RuntimeError(f"Unexpected g_mu shape: {g_mu.shape}")
    if not np.isfinite(g_mu).all():
        raise RuntimeError("g_mu contains non-finite values.")

    params: dict[str, object] = {
        "landscape": landscape_name,
        "model": model_name,
        "nucleotide_sites": nucleotide_sites,
        "num_total_starts": total_starts,
        "num_replicates_per_start": int(args.num_replicates_per_start),
        "num_orderings": int(args.num_orderings),
        "num_steps": int(args.num_steps),
        "total_mutation_rate": float(args.total_mutation_rate),
        "per_site_mutation_rate": site_rate,
        "chunk_size": int(args.chunk_size),
        "batch_size": int(args.batch_size),
        "popsize": popsize,
        "mutation_matrix": mutation_matrix,
        "ordering_seeds": ordering_seeds,
        "ordering_multipliers": multipliers,
        "ordering_offsets": offsets,
    }
    save_payload(output_path, g_mu, sum_g_mu, included_counts, counts, params)
    print(f"Saved {g_mu.shape} -> {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landscapes", default="pard3", help="Comma-separated landscape keys or 'all'.")
    parser.add_argument(
        "--models",
        default="nuc_uniform,nuc_e_coli_weighted,nuc_e_coli_directed,nuc_a_thaliana_weighted,nuc_a_thaliana_directed",
        help="Comma-separated mutation model keys.",
    )
    parser.add_argument("--output-dir", type=Path, default=get_raw_data_dir(), help="Output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing payloads.")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Maximum starts per outer stream chunk.")
    parser.add_argument("--batch-size", type=int, default=8, help="Starts evaluated together in one JAX call.")
    parser.add_argument("--num-orderings", type=int, default=20, help="Number of random start orderings.")
    parser.add_argument("--num-replicates-per-start", type=int, default=10, help="Replicate trajectories per start.")
    parser.add_argument("--num-steps", type=int, default=75, help="Number of generations recorded.")
    parser.add_argument("--total-mutation-rate", type=float, default=0.1, help="Total mutations per genome per step.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--counts", default="1,10,100,1000,10000,100000", help="Comma-separated prefix counts.")
    parser.add_argument("--pard3-popsize", type=int, default=60, help="Population size for ParD3.")
    parser.add_argument("--empirical-popsize", type=int, default=2500, help="Population size for other landscapes.")
    return parser


def main() -> None:
    """Run full-nucleotide G_mu generation from command-line arguments.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Writes requested payload files.
    """
    parser = build_parser()
    args = parser.parse_args()
    landscapes = parse_csv_choices(args.landscapes, list(LANDSCAPE_FILES))
    models = parse_csv_choices(args.models, list(MUTATION_MATRIX_FILES))
    if args.chunk_size <= 0 or args.batch_size <= 0:
        raise ValueError("chunk-size and batch-size must be positive.")
    if args.num_orderings <= 0 or args.num_replicates_per_start <= 0 or args.num_steps <= 0:
        raise ValueError("num-orderings, num-replicates-per-start, and num-steps must be positive.")

    for landscape_name in landscapes:
        for model_name in models:
            run_landscape_model(landscape_name, model_name, args.output_dir, args)


if __name__ == "__main__":
    main()
