"""
Figure 4D — Step 1: raw data requirements.

This figure uses the same per-landscape all-starts empirical decay curves
as Figure 4B, plus the Fourier spectra from Figure 4A (for the dotted
reference lines showing weighted-average frequency).

Raw data files required (in SLIDE_data/)
-----------------------------------------
  decay_curves_gb1_m0.1_all_starts.pkl
  decay_curves_trpb_m0.1_all_starts.pkl
  decay_curves_tev_m0.1_all_starts.pkl
  decay_curves_pard3_m0.1_all_starts.pkl

Also requires (from landscape_arrays/)
---------------------------------------
  landscape_arrays/GB1_landscape_array.pkl
  landscape_arrays/E3_landscape_array.pkl      (ParD3)
  landscape_arrays/TEV_landscape_array.pkl
  landscape_arrays/TrpB_landscape_array.pkl

These are used by 2_process_data.py (Fig 4A) to produce
  plot_data/fourier_spectra_empirical.pkl
which 3_plot.py loads for the dotted reference lines.
"""

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from slide_config import get_slide_data_dir
slide_data_dir = get_slide_data_dir()

required_sim = [
    os.path.join(slide_data_dir, 'decay_curves_gb1_m0.1_all_starts.pkl'),
    os.path.join(slide_data_dir, 'decay_curves_trpb_m0.1_all_starts.pkl'),
    os.path.join(slide_data_dir, 'decay_curves_tev_m0.1_all_starts.pkl'),
    os.path.join(slide_data_dir, 'decay_curves_pard3_m0.1_all_starts.pkl'),
]
required_static = [
    os.path.join(parent_dir, 'landscape_arrays', 'GB1_landscape_array.pkl'),
    os.path.join(parent_dir, 'landscape_arrays', 'E3_landscape_array.pkl'),
    os.path.join(parent_dir, 'landscape_arrays', 'TEV_landscape_array.pkl'),
    os.path.join(parent_dir, 'landscape_arrays', 'TrpB_landscape_array.pkl'),
]

for path in required_sim + required_static:
    status = 'OK' if os.path.exists(path) else 'MISSING'
    print(f'[{status}] {path}')
