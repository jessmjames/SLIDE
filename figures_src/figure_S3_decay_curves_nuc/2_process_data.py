"""
Figure S3 — Step 2: compute spectral rho reference values.

Computes the spectral Dirichlet rho for each landscape × mutation model
using the generalised Fourier transform in nucleotide space.

Input
-----
  landscape_arrays/{GB1,TrpB,TEV,E3}_landscape_array.pkl
  other_data/normed_{h_sapiens,e_coli}_matrix.npy

Output
------
  plot_data/spectral_rho_comparison.pkl

Usage
-----
  python figures_src/figure_S3_decay_curves_nuc/2_process_data.py

Originally from scripts/compute_spectral_rho_models.py.

Shared output: plot_data/spectral_rho_comparison.pkl is also used by Figure 6.
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
script     = os.path.join(parent_dir, 'scripts', 'compute_spectral_rho_models.py')

result = subprocess.run([sys.executable, script], cwd=parent_dir)
sys.exit(result.returncode)
