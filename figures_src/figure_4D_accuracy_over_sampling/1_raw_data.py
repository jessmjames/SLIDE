"""
Figure 4D — Step 1: generate raw simulation data.

This figure shares its simulation raw data with Figure 4B (the all-starts
empirical decay curves). Delegates to
figures_src/figure_4B_landscape_heterogeneity/1_raw_data.py, which runs:

  1. empirical_landscape_decay_curves_all_starts.py
       → SLIDE_data/decay_curves_{gb1,trpb,tev,pard3}_m0.1_all_starts.pkl
  2. landscape_heterogeneity.py
       → SLIDE_data/N4A20_heterogeneity2.pkl

Note: both scripts require a GPU. Only the decay curve outputs are consumed
by Figure 4D; the heterogeneity pkl is generated as a side-effect of running
the shared 4B pipeline.

Figure 4D also requires Fourier spectra for the dotted reference lines; those
are produced by figures_src/figure_4A_.../2_process_data.py, not here.

Usage
-----
  python figures_src/figure_4D_accuracy_over_sampling/1_raw_data.py
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

script = os.path.join(parent_dir, 'figures_src',
                      'figure_4B_landscape_heterogeneity', '1_raw_data.py')

print('Delegating to Figure 4B raw data script...')
result = subprocess.run([sys.executable, script], cwd=parent_dir, check=True)
