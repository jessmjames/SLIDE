"""
Figure 6 — Step 1: decay curve simulations for all mutation models (75 steps).

Runs all-starts directed evolution for all four landscapes under four
nucleotide mutation models plus the AA-uniform baseline.

Output
------
  SLIDE_data/decay_curves_{landscape}_{model}_m0.1_all_starts_75steps.pkl

  landscapes : gb1, trpb, tev, pard3
  nuc models : nuc_uniform, nuc_h_sapiens_sym, nuc_h_sapiens, nuc_e_coli
  aa model   : aa_uniform

Usage
-----
  python figures_src/figure_6_mutation_model/1_raw_data.py

Note: requires a GPU. Calls the two existing simulation scripts.

Shared raw data: nuc model outputs are also used by Figure S3.
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Nuc models (shared with S3)
nuc_script = os.path.join(parent_dir, 'scripts',
                           'empirical_landscape_decay_curves_codon_fast_75steps.py')
# AA-uniform baseline
aa_script  = os.path.join(parent_dir, 'scripts',
                           'empirical_landscape_decay_curves_all_starts_fast_75steps.py')

for script in [nuc_script, aa_script]:
    print(f'\nRunning {os.path.basename(script)}...')
    result = subprocess.run([sys.executable, script], cwd=parent_dir)
    if result.returncode != 0:
        sys.exit(result.returncode)

print('\nDone.')
