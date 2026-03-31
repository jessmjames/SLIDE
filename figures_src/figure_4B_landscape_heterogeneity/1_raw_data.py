"""
Figure 4B — Step 1: run decay curve and heterogeneity simulations.

Runs two simulation scripts in sequence:
  1. empirical_landscape_decay_curves_all_starts.py
       → SLIDE_data/decay_curves_{gb1,trpb,tev,pard3}_m0.1_all_starts.pkl
  2. landscape_heterogeneity.py
       → SLIDE_data/N4A20_heterogeneity2.pkl

Note: both scripts require a GPU.

Filename note: landscape_heterogeneity.py writes N4A20_heterogeneity2.pkl
(with a trailing "2"), not N4A20_heterogeneity.pkl as the earlier stub
documented. Downstream loading code should use the "2" variant.

Shared raw data: the decay curve outputs (decay_curves_*_all_starts.pkl)
are also used by Figures 4C and 4D.

Usage
-----
  python figures_src/figure_4B_landscape_heterogeneity/1_raw_data.py
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

decay_script = os.path.join(parent_dir, 'scripts',
                             'empirical_landscape_decay_curves_all_starts.py')
hetero_script = os.path.join(parent_dir, 'scripts',
                              'landscape_heterogeneity.py')

for script in [decay_script, hetero_script]:
    print(f'\nRunning {os.path.basename(script)}...')
    result = subprocess.run([sys.executable, script], cwd=parent_dir)
    if result.returncode != 0:
        sys.exit(result.returncode)

print('\nDone.')
