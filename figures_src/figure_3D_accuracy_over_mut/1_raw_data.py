"""
Figure 3D — Step 1: generate decay curve sweep over mutation rates.

Runs a sweep over 25 mutation rates (0.01 to 2.0) on a fixed NK
landscape, with 500 replicates each.

Output
------
  SLIDE_data/mutation_rate_accuracy.pkl  — raw decay curves, shape (25, ?, 25)

Note: requires a GPU.

Usage
-----
  python figures_src/figure_3D_accuracy_over_mut/1_raw_data.py
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

script = os.path.join(parent_dir, 'scripts', 'mutation_rate_accuracy.py')

print(f'Running {os.path.basename(script)}...')
result = subprocess.run([sys.executable, script], cwd=parent_dir)
if result.returncode != 0:
    sys.exit(result.returncode)

print('Done.')
