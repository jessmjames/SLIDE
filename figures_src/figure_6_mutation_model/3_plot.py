"""
Figure 6 — Step 3: plot rho accuracy over trajectory sampling for nuc models.

Input
-----
  plot_data/trajectory_subsampling_{model}_75steps.pkl
    models: nuc_uniform, nuc_h_sapiens_sym, nuc_e_coli, aa_uniform
  plot_data/spectral_rho_comparison.pkl

Output
------
  figures/graph_mutation_model.pdf
  figures/accuracy_over_sampling_aa_uniform_75steps.pdf
  figures/accuracy_panel_{d,e,f}_75steps.pdf
  figures/accuracy_panel_legend_75steps.pdf

Usage
-----
  python figures_src/figure_6_mutation_model/3_plot.py
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
script     = os.path.join(parent_dir, 'scripts', 'plot_mutation_model_comparison.py')

result = subprocess.run([sys.executable, script, '--steps', '75'], cwd=parent_dir)
sys.exit(result.returncode)
