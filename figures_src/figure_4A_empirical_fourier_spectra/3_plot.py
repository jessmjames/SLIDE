"""
Figure 4A — Step 3: plot empirical landscape Fourier spectra.

Input
-----
  plot_data/fourier_spectra_empirical.pkl
    tuple: (spectrum_gb1, spectrum_trpb, spectrum_tev, spectrum_pard3)
    Each spectrum is a 1-D array of Fourier power coefficients (index 0 is
    the constant/DC term).

Output
------
  figures/empirical_fourier_spectra.pdf

Usage
-----
  python figures_src/figure_4A_empirical_fourier_spectra/3_plot.py

Originally from ruggedness_figures_plots.ipynb cells 30–31.
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

import pickle
import numpy as np
import matplotlib.pyplot as plt

plot_data_dir = os.path.join(parent_dir, 'plot_data')
figures_dir   = os.path.join(parent_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Style settings (from notebook cells 2–3)
c2 = 'tab:blue'
c1 = 'tab:orange'
c3 = '#f55f74'
c4 = 'tab:green'

titlesize = 10
labelsize = 8
ticksize  = 6
legendsize = 8
dpi = 350
plt.rcParams["font.family"] = "DejaVu Sans"

with open(os.path.join(plot_data_dir, 'fourier_spectra_empirical.pkl'), 'rb') as f:
    spectra = pickle.load(f)

# ---- Plotting code from notebook cell 31 (exact copy) ----
colours = [c1,c2,c4,c3]
labels = ['GB1', 'TrpB', 'TEV','ParD3']
markers = ['o','s','^','D']

plt.figure(figsize=(3.5,3), dpi=300)
A=20
N=4
fourier_vals = []
d = N*(A-1)

for n,i in enumerate(spectra):
    i = i[1:]
    i_norm = (i - i.min()) / (i.max() - i.min())
    indexes = np.arange(len(i))+1
    weighted_avg = np.sum(A * indexes * i) / np.sum(i) / d
    #K = (decay_rate_measurements[n]*N)-1
    plt.plot(list(range(1,len(i_norm)+1)), i_norm, label=labels[n], c=colours[n], marker=markers[n],markersize=4)
    #plt.axvline(x=(K + 1)*(A-1)/A, c=colours[n], linestyle='--', alpha=0.5)
    plt.axvline(x=weighted_avg*4*(A-1)/A, c=colours[n], linestyle='--', alpha=0.5)
    fourier_vals.append(weighted_avg)

plt.legend(fontsize=legendsize, loc='upper right')
plt.title('Empirical landscape fourier spectra', fontsize=titlesize)
plt.xlabel(r'Frequency index $i$', fontsize=labelsize)
plt.ylabel(r'Power spectral coefficient $b_i$', fontsize=labelsize)
plt.xticks(([1,2,3,4]))
plt.tick_params(axis='both', which='major', labelsize=ticksize)
# ---- End of notebook cell 31 ----

out_path = os.path.join(figures_dir, 'empirical_fourier_spectra.pdf')
plt.savefig(out_path, dpi=dpi)
print(f'Saved → {out_path}')
