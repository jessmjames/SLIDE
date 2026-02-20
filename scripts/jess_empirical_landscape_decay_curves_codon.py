import sys
import os

# Get the parent directory of this script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import pickle
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from direvo_functions import *
import selection_function_library as slct
import tqdm

# Define the SLIDE_data path
slide_data_dir = "/home/seb/code/phd/SLIDE/SLIDE_data"

def directedEvolution(rng,
                      selection_strategy, 
                      selection_params, 
                      empirical = False, 
                      N = None, 
                      K = None, 
                      landscape = None, 
                      popsize=100, 
                      mutation_function=None,
                      num_steps=50, 
                      num_reps=10, 
                      define_i_pop=None, 
                      pre_optimisation_steps=0,
                      average=True):
    
 
    r1,r2,r3=jr.split(rng,3)

    # Get initial population.
    if define_i_pop == None:
        i_pop = jnp.array([jr.randint(r1, (N,), 0, 2)]*popsize)
    else:
        i_pop = define_i_pop
 
    # Function for evaluating fitness.
    if empirical:
        fitness_function = landscape  # Already a function for codon version
    else:
        fitness_function = build_NK_landscape_function(r2, N, K)
 
    # Define selection function.
    selection_function = build_selection_function(
        selection_strategy, selection_params)
    
    if pre_optimisation_steps!= 0:

        pre_op_selection_function=build_selection_function(slct.base_chance_threshold_select, {'base_chance':0.0, 'threshold':0.95})


        pre_op = run_directed_evolution(r3, i_pop=i_pop, 
                               selection_function=pre_op_selection_function, 
                               mutation_function=mutation_function, 
                               fitness_function=fitness_function, 
                               num_steps=pre_optimisation_steps)[1]
 
        i_pop = pre_op['pop'][-1]


    vmapped_run = jax.jit(jax.vmap(lambda r: run_directed_evolution(
        r, i_pop, selection_function, mutation_function, fitness_function=fitness_function, num_steps=num_steps)[1]))
    
    # The array of seeds we will take as input.
    rng_seeds = jr.split(r3, num_reps)
    results = vmapped_run(rng_seeds)
 
    return results

def uniform_start_locs_aa(ld, num=10000):
    """Generate uniform start locations in AA space"""
    flat_ld = ld.flatten()
    flat_indexes = np.round(np.linspace(0, flat_ld.shape[0]-1, num)).astype(int)
    indexes = np.array([np.unravel_index(i, ld.shape) for i in flat_indexes])
    return indexes

def convert_aa_starts_to_codon(aa_starts):
    """Convert amino acid indices to codon (nucleotide) sequences using vectorized operations"""
    # aa_starts shape: (num_starts, n_aa)
    # INVERSE_CODON_MAPPER[aa_starts] gives shape: (num_starts, n_aa, 3)
    codon_triplets = np.array(INVERSE_CODON_MAPPER)[aa_starts]
    # Reshape to flatten: (num_starts, n_aa * 3)
    codon_starts = codon_triplets.reshape(aa_starts.shape[0], -1)
    return jnp.array(codon_starts, dtype=jnp.int32)

def generate_decay_curve_codon(ld, mutation_function, m, p=2500):
    """
    Generate decay curves for codon-based simulations.
    
    Parameters:
    - ld: Landscape array (amino acid based)
    - mutation_function: Pre-built mutation function
    - m: Mutation rate (per nucleotide)
    - p: Population size
    """
    # Create codon-based fitness function
    fitness_function = get_pre_defined_landscape_function_with_codon(ld)
    
    results = []
    
    # Generate starting locations in AA space
    aa_starts = uniform_start_locs_aa(ld, num=10000)
    
    # Convert to codon space
    codon_starts = convert_aa_starts_to_codon(aa_starts)
    
    # Reshape for batch processing
    if ld.shape == (20,20,20,20):
        # GB1: 4 AA -> 12 nucleotides
        codon_starts = codon_starts.reshape(20, 500, 12)
    else:
        # TrpB/TEV/ParD3: 3 AA -> 9 nucleotides  
        codon_starts = codon_starts.reshape(20, 500, 9)

    for starts in tqdm.tqdm(codon_starts):
        print(f"Processing batch, shape: {starts.shape}")
        
        def single_rep(start):
            params = {'threshold': 0.0, 'base_chance' : 1.0}
            run = directedEvolution(jr.PRNGKey(42),
                                    selection_strategy=slct.base_chance_threshold_select,
                                    selection_params = params,
                                    popsize=int(p),
                                    mutation_function=mutation_function,
                                    num_steps=25,
                                    num_reps=10,
                                    pre_optimisation_steps=0,
                                    define_i_pop=jnp.array([start]*int(p)),
                                    empirical=True,
                                    landscape=fitness_function,
                                    average=False)

            return run['fitness'].mean(axis=-1)

        results.append(jax.jit(jax.vmap(single_rep))(starts))

    return results

## Load landscapes
print("Loading landscapes...")

with open('landscape_arrays/GB1_landscape_array.pkl', 'rb') as f:
    GB1 = pickle.load(f)

with open('landscape_arrays/TrpB_landscape_array.pkl', 'rb') as f:
    TrpB = pickle.load(f)

with open('landscape_arrays/TEV_landscape_array.pkl', 'rb') as f:
    TEV = pickle.load(f)

with open('landscape_arrays/E3_landscape_array.pkl', 'rb') as f:
    E3 = pickle.load(f)

## Load and prepare mutation matrices
print("Loading mutation matrices...")

# Load the 4x4 nucleotide transition matrices
h_sapiens_matrix = np.load('other_data/normed_h_sapiens_matrix.npy')
human_codon_matrix = np.load('other_data/normed_human_codon_matrix.npy')
e_coli_matrix = np.load('other_data/normed_e_coli_matrix.npy')

# Create symmetric h_sapiens matrix
h_sapiens_sym = (h_sapiens_matrix + h_sapiens_matrix.T) / 2
# Renormalize rows to sum to 1
h_sapiens_sym = h_sapiens_sym / h_sapiens_sym.sum(axis=1, keepdims=True)

print(f"h_sapiens matrix shape: {h_sapiens_matrix.shape}")
print(f"human_codon matrix shape: {human_codon_matrix.shape}")
print(f"e_coli matrix shape: {e_coli_matrix.shape}")

## Define mutation models
mutation_models = {
    'codon_uniform': ('uniform', None),
    'codon_h_sapiens_sym': ('custom', h_sapiens_sym),
    'codon_human': ('custom', human_codon_matrix),
    'codon_ecoli': ('custom', e_coli_matrix),
}

## Define landscapes
landscapes = {
    'gb1': (GB1, 4, 12),      # 4 AA sites -> 12 nucleotides
    'trpb': (TrpB, 3, 9),     # 3 AA sites -> 9 nucleotides
    'tev': (TEV, 3, 9),       # 3 AA sites -> 9 nucleotides
    'pard3': (E3, 3, 9),      # 3 AA sites -> 9 nucleotides
}

## Run simulations
m = 0.1  # Base mutation rate

for model_name, (mut_type, mut_matrix) in mutation_models.items():
    print(f"\n{'='*60}")
    print(f"Running mutation model: {model_name}")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = os.path.join(slide_data_dir, "empirical_decay_curves", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    for landscape_name, (ld, n_aa, n_nuc) in landscapes.items():
        print(f"\nGenerating curves for {landscape_name.upper()} with {model_name}...")
        
        # Adjust mutation rate for sequence length
        adjusted_m = m / n_nuc
        
        # Build mutation function
        if mut_type == 'uniform':
            mutation_function = build_mutation_function(adjusted_m, num_options=4)
        else:
            mutation_function = build_custom_mutation_function(adjusted_m, mut_matrix, A=4)
        
        # Generate decay curves
        results = generate_decay_curve_codon(ld, mutation_function, adjusted_m)
        
        # Save results
        output_path = os.path.join(output_dir, f"decay_curves_{landscape_name}_m0.1.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(np.array(results), f)
        
        print(f"Saved to: {output_path}")

print("\n" + "="*60)
print("All codon-based decay curves generated successfully!")
print("="*60)
