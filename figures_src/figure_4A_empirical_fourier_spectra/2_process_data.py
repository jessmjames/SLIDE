"""
Figure 4A — Step 2: compute Fourier spectra of empirical landscapes.

Loads each empirical landscape array and computes its Fourier spectrum
using get_landscape_spectrum() from ruggedness_functions.py.

Input
-----
  landscape_arrays/GB1_landscape_array.pkl
  landscape_arrays/E3_landscape_array.pkl      (ParD3)
  landscape_arrays/TEV_landscape_array.pkl
  landscape_arrays/TrpB_landscape_array.pkl

Output
------
  plot_data/fourier_spectra_empirical.pkl
    tuple: (full_spectrum_gb1, full_spectrum_trpb, full_spectrum_tev,
            full_spectrum_pard3)

Note
----
  Originally from ruggedness_figures_data_processing.ipynb cells 34–35, 47.
  Requires GPU/JAX for on_gpu=True; set on_gpu=False for CPU-only runs
  (slower).
"""

import os
import sys
import pickle

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from ruggedness_functions import get_landscape_spectrum

landscape_arrays_dir = os.path.join(parent_dir, 'landscape_arrays')
plot_data_dir        = os.path.join(parent_dir, 'plot_data')
os.makedirs(plot_data_dir, exist_ok=True)

out = os.path.join(plot_data_dir, 'fourier_spectra_empirical.pkl')

if os.path.exists(out):
    print(f'Already exists: {out}')
    sys.exit(0)

with open(os.path.join(landscape_arrays_dir, 'GB1_landscape_array.pkl'), 'rb') as f:
    GB1 = pickle.load(f)

with open(os.path.join(landscape_arrays_dir, 'E3_landscape_array.pkl'), 'rb') as f:
    ParD3 = pickle.load(f)

with open(os.path.join(landscape_arrays_dir, 'TEV_landscape_array.pkl'), 'rb') as f:
    TEV = pickle.load(f)

with open(os.path.join(landscape_arrays_dir, 'TrpB_landscape_array.pkl'), 'rb') as f:
    TrpB = pickle.load(f)

full_spectrum_gb1   = get_landscape_spectrum(GB1,   remove_constant=False, on_gpu=True, norm=False)
full_spectrum_trpb  = get_landscape_spectrum(TrpB,  remove_constant=False, on_gpu=True, norm=False)
full_spectrum_tev   = get_landscape_spectrum(TEV,   remove_constant=False, on_gpu=True, norm=False)
full_spectrum_pard3 = get_landscape_spectrum(ParD3, remove_constant=False, on_gpu=True, norm=False)

with open(out, 'wb') as f:
    pickle.dump((full_spectrum_gb1, full_spectrum_trpb, full_spectrum_tev, full_spectrum_pard3), f)

print(f'Saved → {out}')
