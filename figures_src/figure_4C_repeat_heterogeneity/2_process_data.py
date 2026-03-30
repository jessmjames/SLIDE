"""
Figure 4C — Step 2: compute NK and empirical rho distributions.

Fits decay rates across many starting points for NK and empirical landscapes,
producing rho distributions for the violin plot.

Input
-----
  SLIDE_data/ decay curve arrays (shared with Fig 4B)

Output
------
  plot_data/heterogeneity_data.pkl
    (NK_rhos, empirical_rhos): lists of rho distributions per landscape

Note: not yet extracted into a standalone script. Currently lives in
ruggedness_figures_data_processing_IK.ipynb cells 48–50.
Run that notebook section to regenerate heterogeneity_data.pkl.
"""

import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
out = os.path.join(parent_dir, 'plot_data', 'heterogeneity_data.pkl')

if os.path.exists(out):
    print(f"Already exists: {out}")
else:
    print("heterogeneity_data.pkl not found.")
    print("Run cells 48–50 of ruggedness_figures_data_processing_IK.ipynb to generate it.")
