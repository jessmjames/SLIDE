import numpy as np
import pandas as pd
import os

# 1. Define the raw data from Table 1
# Columns: [AT>GC, GC>AT, AT>TA, GC>TA, AT>CG, GC>CG]
data = {
    "H. sapiens":      [0.221, 0.408, 0.067, 0.110, 0.078, 0.125],
    "D. melanogaster": [0.190, 0.305, 0.121, 0.155, 0.115, 0.115],
    "C. elegans":      [0.105, 0.205, 0.220, 0.281, 0.113, 0.077],
    "A. thaliana":     [0.118, 0.588, 0.059, 0.094, 0.071, 0.071],
    "S. cerevisiae":   [0.103, 0.286, 0.060, 0.312, 0.057, 0.182],
    "E. coli":         [0.162, 0.317, 0.195, 0.115, 0.155, 0.056]
}

import numpy as np

def create_substitution_matrix_aligned(values):
    """
    Constructs a 4x4 mutation matrix aligned with CODON_MAPPER.
    Order: 0:T, 1:C, 2:A, 3:G
    """
    at_gc, gc_at, at_ta, gc_ta, at_cg, gc_cg = values
    
    mat = np.zeros((4, 4))
    
    # Row 0: T (Matches A symmetry)
    mat[0, 1] = at_gc  # T -> C (AT -> GC transition)
    mat[0, 2] = at_ta  # T -> A (AT -> TA transversion)
    mat[0, 3] = at_cg  # T -> G (AT -> CG transversion)
    
    # Row 1: C (Matches G symmetry)
    mat[1, 0] = gc_at  # C -> T (GC -> AT transition)
    mat[1, 2] = gc_ta  # C -> A (GC -> TA transversion)
    mat[1, 3] = gc_cg  # C -> G (GC -> CG transversion)
    
    # Row 2: A
    mat[2, 0] = at_ta  # A -> T (AT -> TA transversion)
    mat[2, 1] = at_cg  # A -> C (AT -> CG transversion)
    mat[2, 3] = at_gc  # A -> G (AT -> GC transition)
    
    # Row 3: G
    mat[3, 0] = gc_at  # G -> T (GC -> AT transition) - Wait, G->T is GC->TA
    # Let's do this carefully base by base:
    
    # 0=T, 1=C, 2=A, 3=G
    
    # --- From T (0) ---
    mat[0, 1] = at_gc # T -> C
    mat[0, 2] = at_ta # T -> A
    mat[0, 3] = at_cg # T -> G
    
    # --- From C (1) ---
    mat[1, 0] = gc_at # C -> T
    mat[1, 2] = gc_ta # C -> A
    mat[1, 3] = gc_cg # C -> G
    
    # --- From A (2) ---
    mat[2, 0] = at_ta # A -> T
    mat[2, 1] = at_cg # A -> C
    mat[2, 3] = at_gc # A -> G
    
    # --- From G (3) ---
    mat[3, 0] = gc_ta # G -> T
    mat[3, 1] = gc_cg # G -> C
    mat[3, 2] = gc_at # G -> A
    
    return mat

# Create output directory if it doesn't exist
output_dir = "other_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Loop through and save matrices
all_matrices = {}

for species, values in data.items():
    # Generate the 4x4 matrix
    mat = create_substitution_matrix_aligned(values)
    
    # Normalize by row (so each row sums to 1.0)
    # Note: If a row is all zeros, this will throw a warning. 
    # But here, every row has at least 3 values from the spectrum.
    normed_mat = mat / mat.sum(axis=1, keepdims=True)
    
    # Save files with sanitized names
    name_slug = species.lower().replace(". ", "_").replace(" ", "_")
    np.save(f"{output_dir}/{name_slug}_matrix.npy", mat)
    np.save(f"{output_dir}/normed_{name_slug}_matrix.npy", normed_mat)
    
    all_matrices[species] = normed_mat

print(f"Processed {len(all_matrices)} species. Files saved to '{output_dir}/'.")

# Example: View the matrix for E. coli
print("\nNormalized matrix for E. coli (A, C, G, T):")
print(all_matrices["E. coli"])