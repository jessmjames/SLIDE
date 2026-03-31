"""
Figure 4B — Step 1: raw data requirements.

This figure requires per-landscape empirical decay curves (all starting
points) and NK heterogeneity simulation data.

Raw data files required (in SLIDE_data/)
-----------------------------------------
  decay_curves_gb1_m0.1_all_starts.pkl
  decay_curves_trpb_m0.1_all_starts.pkl
  decay_curves_tev_m0.1_all_starts.pkl
  decay_curves_pard3_m0.1_all_starts.pkl
  N4A20_heterogeneity.pkl

These are produced by simulation scripts (see ruggedness_figures_data_processing.ipynb
cells 38, 49 for how they are loaded).
"""

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from slide_config import get_slide_data_dir
slide_data_dir = get_slide_data_dir()

required = [
    os.path.join(slide_data_dir, 'decay_curves_gb1_m0.1_all_starts.pkl'),
    os.path.join(slide_data_dir, 'decay_curves_trpb_m0.1_all_starts.pkl'),
    os.path.join(slide_data_dir, 'decay_curves_tev_m0.1_all_starts.pkl'),
    os.path.join(slide_data_dir, 'decay_curves_pard3_m0.1_all_starts.pkl'),
    os.path.join(slide_data_dir, 'N4A20_heterogeneity.pkl'),
]

for path in required:
    status = 'OK' if os.path.exists(path) else 'MISSING'
    print(f'[{status}] {path}')
