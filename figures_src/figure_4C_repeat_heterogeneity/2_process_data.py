"""
Figure 4C — Step 2: generate heterogeneity_data.pkl (shared with Fig 4B).

This figure uses the same plot_data/heterogeneity_data.pkl as Figure 4B.
Run figures_src/figure_4B_landscape_heterogeneity/2_process_data.py to generate it.

Input
-----
  (delegated — see figure_4B_landscape_heterogeneity/2_process_data.py)

Output
------
  plot_data/heterogeneity_data.pkl
    (NK_rhos, empirical_rhos): lists of rho distributions per landscape
"""
import os, sys, subprocess

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
script = os.path.join(parent_dir, 'figures_src', 'figure_4B_landscape_heterogeneity', '2_process_data.py')
subprocess.run([sys.executable, script], check=True)
