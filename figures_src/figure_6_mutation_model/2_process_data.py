"""
Figure 6 — Step 2: bootstrap trajectory subsampling for each mutation model.

Reads decay curve pkls from SLIDE_data/, bootstrap-resamples trajectory subsets,
fits IK decay rates, saves subsampling accuracy data to plot_data/.

Also runs spectral rho computation (shared with S3).

Input
-----
  SLIDE_data/decay_curves_{landscape}_{model}_m0.1_all_starts_75steps.pkl

Output
------
  plot_data/trajectory_subsampling_{model}_75steps.pkl
    models: nuc_uniform, nuc_h_sapiens_sym, nuc_e_coli, aa_uniform

  plot_data/spectral_rho_comparison.pkl  (shared with S3)

Usage
-----
  python figures_src/figure_6_mutation_model/2_process_data.py
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
subsample_script = os.path.join(parent_dir, 'scripts', 'compute_trajectory_subsampling.py')
spectral_script  = os.path.join(parent_dir, 'scripts', 'compute_spectral_rho_models.py')

MODELS = ['nuc_uniform', 'nuc_h_sapiens_sym', 'nuc_e_coli', 'aa_uniform']

for model in MODELS:
    print(f'\nSubsampling: {model}')
    result = subprocess.run([sys.executable, subsample_script, model, '--steps', '75'],
                            cwd=parent_dir)
    if result.returncode != 0:
        sys.exit(result.returncode)

print('\nComputing spectral rho...')
result = subprocess.run([sys.executable, spectral_script], cwd=parent_dir)
sys.exit(result.returncode)
