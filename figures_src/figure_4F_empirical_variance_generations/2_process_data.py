"""
Figure 4F — Step 2: generate estimation_variance.pkl (shared with Fig 4E).

This figure uses the same plot_data/estimation_variance.pkl as Figure 4E.
Run figures_src/figure_4E_empirical_variance_popsize/2_process_data.py to generate it.

Input
-----
  (delegated — see figure_4E_empirical_variance_popsize/2_process_data.py)

Output
------
  plot_data/estimation_variance.pkl
    (array_results1, array_results2): array_results2 used here (bootstrap rho
    over number of generations sampled, one array per landscape).
"""
import os, sys, subprocess

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
script = os.path.join(parent_dir, 'figures_src', 'figure_4E_empirical_variance_popsize', '2_process_data.py')
subprocess.run([sys.executable, script], check=True)
