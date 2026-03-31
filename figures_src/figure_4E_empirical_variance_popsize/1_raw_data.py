"""
Figure 4E — Step 1: generate empirical landscape decay curves over population sizes.

Runs all-starts directed evolution for all four empirical landscapes across
a range of population sizes.

Output
------
  SLIDE_data/decay_curves_gb1_m0.1_multistart_10000_uniform_multi_popsize.pkl
  SLIDE_data/decay_curves_trpb_m0.1_multistart_10000_uniform_multi_popsize.pkl
  SLIDE_data/decay_curves_tev_m0.1_multistart_10000_uniform_multi_popsize.pkl
  SLIDE_data/decay_curves_pard3_m0.1_multistart_10000_uniform_multi_popsize.pkl

Note: requires a GPU.
Shared raw data: also used by Figure 4F.

Usage
-----
  python figures_src/figure_4E_empirical_variance_popsize/1_raw_data.py
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

script = os.path.join(parent_dir, 'scripts', 'empirical_landscape_decay_curves_popsize.py')

print(f'Running {os.path.basename(script)}...')
result = subprocess.run([sys.executable, script], cwd=parent_dir)
if result.returncode != 0:
    sys.exit(result.returncode)

print('Done.')
