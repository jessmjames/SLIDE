import os
import sys
import pickle
import numpy as np
import jax.numpy as jnp

# Ensure repo root is on sys.path before local imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from slide_config import get_slide_data_dir
from direvo_functions import get_single_decay_rate

# Resolve SLIDE_data path via env var / local untracked config / sensible default
slide_data_dir = str(get_slide_data_dir())

# Which codon sweeps to process (label must match the sweep filename suffix)
CODON_SWEEPS = ["e_coli", "a_thaliana", "human"]

# Codon table sweeps (nucleotide-level, codon-mapped fitness)
CODON_TABLE_SWEEPS = ["e_coli", "a_thaliana", "human"]

# Avoid overwriting existing outputs by default
OVERWRITE = True


def NK_grid(N_range, num_samples=10):
    N = jnp.linspace(N_range[0], N_range[1], num=num_samples)
    K = jnp.array([jnp.linspace(1, i, num_samples) for i in N]).reshape(
        num_samples, num_samples
    )
    N = jnp.repeat(N, num_samples).reshape(num_samples, num_samples)
    return N, K


def compute_accuracy(decay_data, mut=0.5):
    reshaped_decay_curves = decay_data.reshape(100, -1, 25)
    normalized_decay_curves = (
        reshaped_decay_curves / reshaped_decay_curves[:, :, 0][:, :, np.newaxis]
    )

    decay_rates = np.zeros(
        (normalized_decay_curves.shape[0], normalized_decay_curves.shape[1])
    )

    for i in range(normalized_decay_curves.shape[0]):
        for ii in range(normalized_decay_curves.shape[1]):
            decay_rates[i, ii] = get_single_decay_rate(
                normalized_decay_curves[i, ii, :], mut=mut
            )[0]

    N_grid, K_grid = NK_grid([10, 50])
    Ns, Ks = N_grid.flatten(), K_grid.flatten()
    Ns = jnp.flip(Ns)
    Ks = jnp.flip(Ks)
    NKs = list(zip(Ns, Ks))

    k_plus_one_over_ns = np.clip(
        (np.array(NKs)[:, 1] + 1) / np.array(NKs)[:, 0], 0, 1
    )

    return k_plus_one_over_ns, decay_rates


def main():
    for label in CODON_SWEEPS:
        sweep_path = os.path.join(
            slide_data_dir, f"large_decay_curve_sweep_codon_{label}.pkl"
        )
        if not os.path.exists(sweep_path):
            print(f"Missing sweep file: {sweep_path}")
            continue

        with open(sweep_path, "rb") as f:
            decay_data = pickle.load(f)

        k_plus_one_over_ns, decay_rates = compute_accuracy(decay_data, mut=0.5)

        out_path = os.path.join(
            "plot_data", f"ruggedness_accuracy_codon_{label}.pkl"
        )
        if (not OVERWRITE) and os.path.exists(out_path):
            print(f"Skipping existing output: {out_path}")
            continue

        os.makedirs("plot_data", exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump((k_plus_one_over_ns, decay_rates), f)
        print(f"Wrote: {out_path}")

    for label in CODON_TABLE_SWEEPS:
        sweep_path = os.path.join(
            slide_data_dir,
            "codon_table_sweeps",
            f"large_decay_curve_sweep_codon_table_{label}.pkl",
        )
        if not os.path.exists(sweep_path):
            print(f"Missing sweep file: {sweep_path}")
            continue

        with open(sweep_path, "rb") as f:
            decay_data = pickle.load(f)

        k_plus_one_over_ns, decay_rates = compute_accuracy(decay_data, mut=0.5 / 3.0)

        out_path = os.path.join(
            "plot_data", f"ruggedness_accuracy_codon_table_{label}.pkl"
        )
        if (not OVERWRITE) and os.path.exists(out_path):
            print(f"Skipping existing output: {out_path}")
            continue

        os.makedirs("plot_data", exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump((k_plus_one_over_ns, decay_rates), f)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
