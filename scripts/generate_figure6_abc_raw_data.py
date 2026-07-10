"""Generate raw Figure 6 A-C local G_mu trajectory payloads.

This standalone script mirrors the raw-data generation used by
``figure_6_new_new.ipynb`` for panels A-C. It writes replicate-level
mutation-only fitness trajectories for nucleotide NK landscapes, one pickle per
mutation model, without modifying any figure notebook or processed data file.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import jax.random as jr
import numpy as np
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from slide.data_generation import (  # noqa: E402
    nk_grid_pairs,
    ordered_unique_pairs,
    random_start,
    run_nk_start_averaged_diffusion,
)
from slide.utils import get_raw_data_dir, load_pickle, save_pickle  # noqa: E402


N_VALUES: tuple[int, ...] = (10, 14, 18, 23, 27, 32, 36, 41, 45, 50)
NUM_ALLELES: int = 4
NUM_K_VALUES_PER_N: int = 10
NUM_LANDSCAPES_PER_PAIR: int = 25
NUM_STARTS_PER_LANDSCAPE: int = 25
NUM_POPULATION_REPLICATES: int = 5
POPULATION_SIZE: int = 2_500
NUM_GENERATIONS: int = 25
TOTAL_MUTATION_RATE: float = 0.5
RANDOM_SEED: int = 42

MODEL_KEYS: tuple[str, ...] = (
    "nuc_uniform",
    "nuc_e_coli_weighted",
    "nuc_e_coli_directed",
    "nuc_a_thaliana_weighted",
    "nuc_a_thaliana_directed",
)

MUTATION_MATRIX_FILES: dict[str, str | None] = {
    "nuc_uniform": None,
    "nuc_e_coli_weighted": "normed_e_coli_matrix.npy",
    "nuc_e_coli_directed": "normed_e_coli_matrix.npy",
    "nuc_a_thaliana_weighted": "normed_a_thaliana_matrix.npy",
    "nuc_a_thaliana_directed": "normed_a_thaliana_matrix.npy",
}

DEFAULT_MODELS: tuple[str, ...] = (
    "nuc_a_thaliana_weighted",
    "nuc_a_thaliana_directed",
)


def parse_csv_choices(value: str, allowed_values: Sequence[str]) -> list[str]:
    """Parse comma-separated model choices.

    Parameters:
    - value: str
        Comma-separated command-line value, or ``all``.
    - allowed_values: Sequence[str]
        Valid choices in their default order.

    Returns:
    - list[str]
        Parsed model names.
    """
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if requested == ["all"]:
        return list(allowed_values)
    invalid = sorted(set(requested) - set(allowed_values))
    if invalid:
        raise ValueError(f"Unknown models {invalid}. Allowed models: {list(allowed_values)}")
    if not requested:
        raise ValueError("At least one model is required.")
    return requested


def symmetric_sinkhorn_kernel(kernel: np.ndarray, tolerance: float = 1e-13) -> np.ndarray:
    """Create a symmetric doubly-stochastic kernel by diagonal scaling.

    Parameters:
    - kernel: np.ndarray
        Non-negative square base kernel.
    - tolerance: float
        Maximum permitted row-sum error.

    Returns:
    - np.ndarray
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


def load_mutation_kernel(model: str) -> np.ndarray | None:
    """Load a nucleotide mutation kernel for a Figure 6 A-C model.

    Parameters:
    - model: str
        Mutation model key.

    Returns:
    - np.ndarray | None
        Row-stochastic 4 by 4 transition matrix, or ``None`` for uniform
        mutation.
    """
    matrix_file = MUTATION_MATRIX_FILES[model]
    if matrix_file is None:
        return None
    kernel = np.asarray(np.load(REPO_ROOT / "other_data" / matrix_file), dtype=np.float64)
    if model.endswith("_weighted"):
        kernel = symmetric_sinkhorn_kernel(kernel)
    if kernel.shape != (NUM_ALLELES, NUM_ALLELES):
        raise ValueError(f"{model} kernel has shape {kernel.shape}.")
    if np.any(kernel < 0.0) or not np.allclose(kernel.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError(f"{model} kernel is not row-stochastic.")
    return kernel


def generate_model_raw_payload(
    model: str,
    raw_dir: Path,
    overwrite: bool,
    seed: int,
) -> Path:
    """Generate one raw Figure 6 A-C trajectory payload.

    Parameters:
    - model: str
        Mutation model key.
    - raw_dir: Path
        Directory receiving the raw pickle.
    - overwrite: bool
        Whether to overwrite an existing raw pickle.
    - seed: int
        Base JAX random seed.

    Returns:
    - Path
        Path to the generated or reused raw pickle.
    """
    output_path = raw_dir / f"figure6_ecoli_nk_{model}_raw.pkl"
    if output_path.exists() and not overwrite:
        existing_payload = load_pickle(output_path)
        trajectories = np.asarray(existing_payload["data"]["fitness_trajectories"], dtype=np.float32)
        expected_shape = (
            len(ordered_unique_pairs(nk_grid_pairs((10, 50), NUM_K_VALUES_PER_N, K_start=0))),
            NUM_LANDSCAPES_PER_PAIR,
            NUM_STARTS_PER_LANDSCAPE,
            NUM_POPULATION_REPLICATES,
            NUM_GENERATIONS,
        )
        if trajectories.shape != expected_shape:
            raise ValueError(f"{output_path} has shape {trajectories.shape}; expected {expected_shape}.")
        print(f"Reusing existing raw payload: {output_path}")
        return output_path

    raw_pairs = nk_grid_pairs((10, 50), NUM_K_VALUES_PER_N, K_start=0)
    nk_pairs = ordered_unique_pairs(raw_pairs)
    if len(nk_pairs) != 100 or any(k_value >= n_sites for n_sites, k_value in nk_pairs):
        raise AssertionError("Expected 100 valid Figure 6 A-C NK pairs with K < N.")

    kernel = load_mutation_kernel(model)
    pair_keys = jr.split(jr.PRNGKey(seed), len(nk_pairs))
    trajectories = np.empty(
        (
            len(nk_pairs),
            NUM_LANDSCAPES_PER_PAIR,
            NUM_STARTS_PER_LANDSCAPE,
            NUM_POPULATION_REPLICATES,
            NUM_GENERATIONS,
        ),
        dtype=np.float32,
    )
    starts_padded = np.full(
        (
            len(nk_pairs),
            NUM_LANDSCAPES_PER_PAIR,
            NUM_STARTS_PER_LANDSCAPE,
            max(N_VALUES),
        ),
        -1,
        dtype=np.int8,
    )
    landscape_keys_saved = np.empty(
        (len(nk_pairs), NUM_LANDSCAPES_PER_PAIR, 2),
        dtype=np.uint32,
    )

    progress = tqdm.tqdm(
        enumerate(zip(pair_keys, nk_pairs, strict=True)),
        total=len(nk_pairs),
        desc=f"{model} NK pairs",
    )
    for pair_index, (pair_key, pair) in progress:
        n_sites, k_value = pair
        landscape_keys = jr.split(pair_key, NUM_LANDSCAPES_PER_PAIR)
        for landscape_index, landscape_key in enumerate(landscape_keys):
            start_keys = jr.split(
                jr.fold_in(landscape_key, 100_000 + landscape_index),
                NUM_STARTS_PER_LANDSCAPE,
            )
            starts = np.asarray(
                [
                    random_start(key, n_sites=n_sites, num_alleles=NUM_ALLELES)
                    for key in start_keys
                ],
                dtype=np.int32,
            )
            starts_padded[pair_index, landscape_index, :, :n_sites] = starts
            landscape_keys_saved[pair_index, landscape_index] = np.asarray(landscape_key)
            trajectories[pair_index, landscape_index] = run_nk_start_averaged_diffusion(
                rng_key=landscape_key,
                trajectory_rng_key=landscape_key,
                n_sites=n_sites,
                k=k_value,
                num_alleles=NUM_ALLELES,
                starts=starts,
                popsize=POPULATION_SIZE,
                mutation_rate_per_site=TOTAL_MUTATION_RATE / n_sites,
                num_reps_per_start=NUM_POPULATION_REPLICATES,
                num_steps=NUM_GENERATIONS,
                mutation_matrix=kernel,
                return_replicates=True,
            )

    payload = {
        "data": {
            "fitness_trajectories": trajectories,
            "starts_padded": starts_padded,
            "landscape_keys": landscape_keys_saved,
            "mutation_kernel": kernel,
        },
        "params": {
            "model": model,
            "N_values": N_VALUES,
            "A": NUM_ALLELES,
            "nk_pairs": nk_pairs,
            "num_landscapes": NUM_LANDSCAPES_PER_PAIR,
            "num_starts": NUM_STARTS_PER_LANDSCAPE,
            "num_population_replicates": NUM_POPULATION_REPLICATES,
            "population_size": POPULATION_SIZE,
            "M": NUM_GENERATIONS,
            "total_mutation_rate": TOTAL_MUTATION_RATE,
            "mutation_rate_per_site": "total_mutation_rate / N",
            "seed": seed,
        },
        "metadata": {
            "paper_reference": "Figure 6A-C",
            "description": f"Replicate-level local G_mu support for {model}.",
            "format": "Matches figure_6_new_new.ipynb A-C raw payloads.",
        },
    }
    save_pickle(payload, output_path)
    print(f"Saved raw payload: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters:
    - None
        Arguments are read from ``sys.argv``.

    Returns:
    - argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate raw Figure 6 A-C NK local G_mu trajectory payloads.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=(
            "Comma-separated model keys, or 'all'. Defaults to the missing "
            "A. thaliana weighted and directed models."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=get_raw_data_dir(),
        help="Directory receiving raw output pickles.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing raw payloads.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Base seed matching figure_6_new_new.ipynb.",
    )
    return parser.parse_args()


def main() -> None:
    """Run raw data generation.

    Parameters:
    - None
        Runtime settings are supplied by command-line arguments.

    Returns:
    - None
        Raw pickle files are written to disk.
    """
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    models = parse_csv_choices(args.models, MODEL_KEYS)
    for model in models:
        generate_model_raw_payload(
            model=model,
            raw_dir=args.raw_dir,
            overwrite=bool(args.overwrite),
            seed=int(args.seed),
        )


if __name__ == "__main__":
    main()
