"""
Triple panel: ρ prediction accuracy vs true (K+1)/N for three mutation models
(E. coli, A. thaliana, Human), matching the notebook fill-between style.

Requires: plot_data/ruggedness_accuracy_codon_{e_coli,a_thaliana,human}.pkl
Output:   figures/rho_accuracy_mutation_models.pdf
"""

import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import pickle
import numpy as np
import matplotlib.pyplot as plt

plot_data_dir = os.path.join(parent_dir, 'plot_data')
figures_dir   = os.path.join(parent_dir, 'figures')

MODELS = [
    ('E. coli',      'ruggedness_accuracy_codon_e_coli'),
    ('A. thaliana',  'ruggedness_accuracy_codon_a_thaliana'),
    ('Human',        'ruggedness_accuracy_codon_human'),
]

fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), dpi=300, sharey=True)

for ax, (title, fname) in zip(axes, MODELS):
    with open(os.path.join(plot_data_dir, fname + '.pkl'), 'rb') as f:
        true_rho, estimated_samples = pickle.load(f)

    # estimated_samples: (100,) array of arrays, each length 250
    estimated_mean_per_nk = np.array([np.mean(s) for s in estimated_samples])

    # Sort by true rho, group into 10 bins of 10 NK combos each
    order        = np.argsort(true_rho)
    sorted_true  = true_rho[order]
    sorted_est   = estimated_mean_per_nk[order]

    grouped_true = sorted_true.reshape(10, -1)
    grouped_est  = sorted_est.reshape(10, -1)

    mean_true  = grouped_true.mean(axis=1)
    mean_decay = grouped_est.mean(axis=1)
    std_decay  = grouped_est.std(axis=1)

    ax.plot(mean_true, mean_decay, 'o-', lw=1.5, label=r'Mean estimated $\rho$')
    ax.fill_between(mean_true, mean_decay - std_decay, mean_decay + std_decay,
                    alpha=0.3, edgecolor=None)
    ax.plot(mean_true, mean_true, c='red', alpha=0.5, ls='--', label=r'$(K+1)/N$')

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r'$(K+1)/N$', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel(r'Estimated $\rho$', fontsize=9)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2,
           fontsize=8, bbox_to_anchor=(0.5, -0.1), frameon=False)

plt.tight_layout()
out = os.path.join(figures_dir, 'rho_accuracy_mutation_models.pdf')
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved → {out}')
