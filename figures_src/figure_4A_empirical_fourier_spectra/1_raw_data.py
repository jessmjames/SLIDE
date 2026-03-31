"""
Figure 4A — Step 1: raw data requirements.

This figure uses Fourier spectra computed directly from four empirical
fitness landscape arrays stored in landscape_arrays/.

Raw landscape arrays required
------------------------------
  landscape_arrays/GB1_landscape_array.pkl
  landscape_arrays/E3_landscape_array.pkl      (ParD3)
  landscape_arrays/TEV_landscape_array.pkl
  landscape_arrays/TrpB_landscape_array.pkl

These files are static inputs tracked in the repository.
No simulation runs are required for this figure.
"""

import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

required = [
    os.path.join(parent_dir, 'landscape_arrays', 'GB1_landscape_array.pkl'),
    os.path.join(parent_dir, 'landscape_arrays', 'E3_landscape_array.pkl'),
    os.path.join(parent_dir, 'landscape_arrays', 'TEV_landscape_array.pkl'),
    os.path.join(parent_dir, 'landscape_arrays', 'TrpB_landscape_array.pkl'),
]

for path in required:
    status = 'OK' if os.path.exists(path) else 'MISSING'
    print(f'[{status}] {path}')
