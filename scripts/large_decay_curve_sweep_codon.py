import sys
import os

# Get the parent directory of this script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from slide_config import get_slide_data_dir

import pickle
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from direvo_functions import *
import selection_function_library as slct
import tqdm

# Resolve SLIDE_data path via env var / local untracked config / sensible default
slide_data_dir = str(get_slide_data_dir())

# --- Mutation kernel settings -------------------------------------------------
# Normalized codon kernels (rows already sum to 1).
CODON_KERNELS = [
    ("e_coli", os.path.join("other_data", "normed_e_coli_matrix.npy")),
    ("a_thaliana", os.path.join("other_data", "normed_a_thaliana_matrix.npy")),
    ("human", os.path.join("other_data", "normed_human_codon_matrix.npy")),
]

# Raw codon tables (not normalized). These are normalized before use.
CODON_TABLE_KERNELS = [
    ("e_coli", os.path.join("other_data", "e_coli_matrix.npy")),
    ("a_thaliana", os.path.join("other_data", "a_thaliana_matrix.npy")),
    ("human", os.path.join("other_data", "human_codon_matrix.npy")),
]

# If True, uses the codon transition matrices for mutation; otherwise falls back to uniform mutation.
USE_CODON_KERNEL = True
RUN_CODON_TABLES = True

# For codon-table mutations, model nucleotides explicitly (3 per codon).
CODON_TABLE_N_MULTIPLIER = 3
# Optional override for m when using codon tables (set to None to use m).
CODON_TABLE_M = None

# Avoid overwriting existing sweep outputs by default.
OVERWRITE = False


def directedEvolution(
    rng,
    selection_strategy,
    selection_params,
    empirical=False,
    N=None,
    K=None,
    landscape=None,
    popsize=100,
    mut_chance=0.01,
    num_steps=50,
    num_reps=10,
    define_i_pop=None,
    pre_optimisation_steps=0,
    average=True,
    num_options=2,
    mutation_matrix=None,
    use_codon_fitness=False,
    codon_landscape_n=None,
    codon_landscape_k=None,
):

    r1, r2, r3 = jr.split(rng, 3)

    # Get initial population.
    if define_i_pop is None:
        i_pop = jnp.array([jr.randint(r1, (N,), 0, num_options)] * popsize)
    else:
        i_pop = define_i_pop

    # Function for evaluating fitness.
    if empirical:
        fitness_function = build_empirical_landscape_function(landscape)
    else:
        if use_codon_fitness:
            if codon_landscape_n is None or codon_landscape_k is None:
                raise ValueError("codon_landscape_n and codon_landscape_k are required when use_codon_fitness=True")
            base_fitness = build_NK_landscape_function(r2, codon_landscape_n, codon_landscape_k)
            fitness_function = convert_landscape_function_to_codon(base_fitness)
        else:
            fitness_function = build_NK_landscape_function(r2, N, K)

    # Mutation function (codon kernel or uniform)
    if mutation_matrix is None:
        mutation_function = build_mutation_function(mut_chance, num_options)
    else:
        mutation_function = build_custom_mutation_function(mut_chance, mutation_matrix, A=num_options)

    # Define selection function.
    selection_function = build_selection_function(selection_strategy, selection_params)

    if pre_optimisation_steps != 0:

        pre_op_selection_function = build_selection_function(
            slct.base_chance_threshold_select, {"base_chance": 0.0, "threshold": 0.95}
        )

        pre_op = run_directed_evolution(
            r3,
            i_pop=i_pop,
            selection_function=pre_op_selection_function,
            mutation_function=mutation_function,
            fitness_function=fitness_function,
            num_steps=pre_optimisation_steps,
        )[1]

        i_pop = pre_op["pop"][-1]

    # Bringing it all together.
    vmapped_run = jax.jit(
        jax.vmap(
            lambda r: run_directed_evolution(
                r,
                i_pop,
                selection_function,
                mutation_function,
                fitness_function=fitness_function,
                num_steps=num_steps,
            )[1]
        )
    )

    # The array of seeds we will take as input.
    rng_seeds = jr.split(r3, num_reps)
    results = vmapped_run(rng_seeds)

    return results


def NK_grid(N_range, num_samples=10):
    N = jnp.linspace(N_range[0], N_range[1], num=num_samples)
    K = jnp.array([jnp.linspace(1, i, num_samples) for i in N]).reshape(
        num_samples, num_samples
    )
    N = jnp.repeat(N, num_samples).reshape(num_samples, num_samples)
    return N, K


N_grid, K_grid = NK_grid([10, 50])

Ns, Ks = N_grid.flatten(), K_grid.flatten()
Ns = jnp.flip(Ns)
Ks = jnp.flip(Ks)
NKs = jnp.array(list(zip(Ns, Ks)))

# m=0.1
m = 0.5
p = 2500

reps = 25

def normalize_kernel(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Mutation kernel must be a square matrix.")
    row_sums = matrix.sum(axis=1)
    deficit = 1.0 - row_sums
    out = matrix.copy()
    diag_idx = np.arange(out.shape[0])
    out[diag_idx, diag_idx] = out[diag_idx, diag_idx] + deficit
    return out


def run_single_kernel(
    label: str,
    transition_matrix: np.ndarray | None,
    num_options: int,
    output_dir: str,
    name_prefix: str,
    n_multiplier: int = 1,
    m_value: float | None = None,
    use_codon_fitness: bool = False,
) -> None:
    out_name = f"{name_prefix}_{label}.pkl" if label else f"{name_prefix}.pkl"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, out_name)
    if (not OVERWRITE) and os.path.exists(file_path):
        print(f"Skipping existing output: {file_path}")
        return

    sweep_results = np.zeros((100, 25, 10, 25))
    total_iterations = len(NKs)

    with tqdm.tqdm(total=total_iterations, desc=f"Overall Progress ({label})") as pbar:
        for i, nks in enumerate(NKs):

            rep_rngs = jr.split(jr.PRNGKey(42), reps)

            def single_rep(rng_rep):

                base_n = int(nks[0])
                base_k = int(nks[1])
                effective_n = base_n * n_multiplier
                effective_m = m if m_value is None else m_value

                params = {"threshold": 0.0, "base_chance": 1.0}
                run = directedEvolution(
                    rng_rep,
                    N=effective_n,
                    K=base_k,
                    selection_strategy=slct.base_chance_threshold_select,
                    selection_params=params,
                    popsize=int(p),
                    mut_chance=effective_m / effective_n,
                    num_steps=25,
                    num_reps=10,
                    pre_optimisation_steps=50,
                    average=False,
                    num_options=num_options,
                    mutation_matrix=transition_matrix,
                    use_codon_fitness=use_codon_fitness,
                    codon_landscape_n=base_n,
                    codon_landscape_k=base_k,
                )

                return run["fitness"].mean(axis=-1)

            repeat_results = jax.vmap(single_rep)(rep_rngs)

            sweep_results[i, :, :, :] = np.array(repeat_results)

            pbar.update(1)

    with open(file_path, "wb") as f:
        pickle.dump(sweep_results, f)
    print(f"Wrote: {file_path}")


codon_kernel_out_dir = slide_data_dir
codon_table_out_dir = os.path.join(slide_data_dir, "codon_table_sweeps")

if USE_CODON_KERNEL:
    for label, path in CODON_KERNELS:
        kernel = np.load(path)
        num_options = int(kernel.shape[0])
        run_single_kernel(
            label,
            kernel,
            num_options,
            output_dir=codon_kernel_out_dir,
            name_prefix="large_decay_curve_sweep_codon",
        )

if RUN_CODON_TABLES:
    for label, path in CODON_TABLE_KERNELS:
        kernel = normalize_kernel(np.load(path))
        num_options = int(kernel.shape[0])
        run_single_kernel(
            label,
            kernel,
            num_options,
            output_dir=codon_table_out_dir,
            name_prefix="large_decay_curve_sweep_codon_table",
            n_multiplier=CODON_TABLE_N_MULTIPLIER,
            m_value=CODON_TABLE_M,
            use_codon_fitness=True,
        )

if (not USE_CODON_KERNEL) and (not RUN_CODON_TABLES):
    run_single_kernel(
        "",
        None,
        num_options=2,
        output_dir=codon_kernel_out_dir,
        name_prefix="large_decay_curve_sweep_codon",
    )
