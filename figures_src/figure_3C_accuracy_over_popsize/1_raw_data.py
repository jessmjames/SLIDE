"""
Figure 3C — Step 1: generate decay curve sweep over population sizes.

Runs a sweep over 25 population sizes (100 to 2500) on a fixed NK
landscape, with 500 replicates each.

Output
------
  SLIDE_data/popsize_accuracy.pkl  — raw decay curves, shape (25, ?, 25)

Note: requires a GPU.

Usage
-----
  python figures_src/figure_3C_accuracy_over_popsize/1_raw_data.py
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

script = os.path.join(parent_dir, 'scripts', 'popsize_accuracy.py')

print(f'Running {os.path.basename(script)}...')
result = subprocess.run([sys.executable, script], cwd=parent_dir)
if result.returncode != 0:
    sys.exit(result.returncode)

print('Done.')
