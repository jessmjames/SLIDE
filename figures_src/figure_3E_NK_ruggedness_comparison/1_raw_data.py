"""
Figure 3E — Step 1: generate NK landscape metrics across K values.

Raw data is computed inline in ruggedness_figures_data_processing.ipynb (cell 26)
by sweeping N=12, K in [0, 11] with 50 random landscape replicates each:

  For each (N, K) pair and replicate, the following are computed:
    - roughness_to_slope()         — roughness-to-slope ratio
    - landscape_r2()               — Fourier R² metric
    - get_mean_paths_to_max()      — mean number of paths to global max
    - find_distance_to_closest_max() — distance to closest local max
    - local_epistasis()            — local epistasis count
    - get_convergence_rate()       — rho (decay rate) from simulation

No external simulation files are required — all metrics are computed analytically
or via short JAX simulations on NK landscapes generated on-the-fly.

To regenerate NK_ruggedness_metric_comparison.pkl, run cells 26–31 of
ruggedness_figures_data_processing.ipynb.
"""

import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
out = os.path.join(parent_dir, 'plot_data', 'NK_ruggedness_metric_comparison.pkl')

if os.path.exists(out):
    print(f"Already exists: {out}")
else:
    print("NK_ruggedness_metric_comparison.pkl not found.")
    print("Run cells 26–31 of ruggedness_figures_data_processing.ipynb to generate it.")
