"""
Figure 4F — Step 1: generate empirical landscape decay curves over population sizes.

This figure shares its raw data with Figure 4E. Delegates to
figures_src/figure_4E_empirical_variance_popsize/1_raw_data.py.

Output
------
  SLIDE_data/decay_curves_{gb1,trpb,tev,pard3}_m0.1_multistart_10000_uniform_multi_popsize.pkl

Note: requires a GPU.

Usage
-----
  python figures_src/figure_4F_empirical_variance_generations/1_raw_data.py
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

script = os.path.join(parent_dir, 'figures_src',
                      'figure_4E_empirical_variance_popsize', '1_raw_data.py')

print(f'Delegating to figure_4E {os.path.basename(script)}...')
subprocess.run([sys.executable, script], cwd=parent_dir, check=True)

print('Done.')
