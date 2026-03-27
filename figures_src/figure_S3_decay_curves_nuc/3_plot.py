"""
Figure S3 — Step 3: plot 75-step nuc decay curves.

Input
-----
  SLIDE_data/decay_curves_{landscape}_{model}_m0.1_all_starts_75steps.pkl
  plot_data/spectral_rho_comparison.pkl
  landscape_arrays/{GB1,TrpB,TEV,E3}_landscape_array.pkl
  other_data/normed_e_coli_matrix.npy

Output
------
  figures/decay_curves_nuc_75steps.pdf

Usage
-----
  python figures_src/figure_S3_decay_curves_nuc/3_plot.py

Originally from scripts/plot_decay_curves_nuc.py.
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
script     = os.path.join(parent_dir, 'scripts', 'plot_decay_curves_nuc.py')

result = subprocess.run([sys.executable, script], cwd=parent_dir)
sys.exit(result.returncode)
