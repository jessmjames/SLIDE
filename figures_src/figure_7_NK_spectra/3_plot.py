"""
Figure 7 — Step 3: plot NK power spectra, fitness decay, and spectrum estimation.

Input
-----
  plot_data/fourier_analysis.pkl
    dict with keys:
      'N_used'    : int  — gene length N
      'A_used'    : int  — number of alleles A
      'Ks_used'   : list — K values swept
      'nk_builts' : list — NK landscape arrays, one per K

Output
------
  figures/NK_spectra.pdf

Usage
-----
  python figures_src/figure_7_NK_spectra/3_plot.py

Originally from ruggedness_figures_plots.ipynb cell 82.
Helper functions (get_exp_matrix, get_fourier_coeffs, get_single_decay_rate)
are copied verbatim from cell 80 of the same notebook.
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

import pickle
import numpy as np
import scipy.optimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from matplotlib import rcParams

from ruggedness_functions import get_landscape_spectrum

plot_data_dir = os.path.join(parent_dir, 'plot_data')
figures_dir   = os.path.join(parent_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper functions — copied verbatim from notebook cell 80
# ---------------------------------------------------------------------------

def get_exp_matrix(
    N: int,
    A: int,
    mutations: np.array,
    is_squared: bool = True,
    fix_b0: bool = False,
) -> np.array:
    eigrange = range(N+1) if fix_b0 == False else range(1, N+1)
    eigenvalues = [A*i for i in eigrange]
    factor = 2 if is_squared is True else 1
    if fix_b0:
        return np.array([[np.exp(-mut/(N*(A-1))*l*factor)-1 for l in eigenvalues] for mut in mutations[1:]])
    else:
        return np.array([[np.exp(-mut/(N*(A-1))*l*factor) for l in eigenvalues] for mut in mutations])


## Measure decay rate function.
def get_fourier_coeffs(
    mean_fitness: np.array,
    mutations: np.array,
    N: int,
    A: int,
    is_squared: bool = False,
    fix_b0: bool = False,
    method: str = "nnls",
    alpha: str = 0.1,
) -> tuple[np.array, np.array]:
    """
    Inputs:
    mean_fitness: vector with mean fitness values
    mutations: vector with number of mutations, starting with 0
    N: length of gene
    A: number of alleles (e.g., 20 if amino acids)
    is_squared: flag to set true if mean fitness squared is provided instead.
    method: pick on of ls, ls_constrained, nnls. Results may vary. Latter to enforce positiveness of the coeffs, although positiveness is only true for the squared estimate.
    Outputs:
    (fourier_coeffs, exponentials): tuple[np.array, np.array]
    fourier_coeffs: Vector of length N+1 with Fourier coeffs from low to high frequency, with the first element corresponding to the constant.
    exponentials: len(mutations)xN+1 matrix to extract fit via exponentials*weights
    """
    exponentials = get_exp_matrix(N=N, A=A, mutations=mutations, is_squared=is_squared, fix_b0=fix_b0)
    if fix_b0:
        mean_fitness_0 = mean_fitness[0]
        mean_fitness = mean_fitness[1:] - mean_fitness_0
    if method == "ls":
        fourier_coeffs, residuals, rank, s = np.linalg.lstsq(exponentials, mean_fitness, rcond=None)
    elif method == "ls_constrained":
        res = scipy.optimize.lsq_linear(exponentials, mean_fitness, bounds=(0, np.inf))
        fourier_coeffs = res.x
    elif method == "nnls":
        fourier_coeffs, rnorm = scipy.optimize.nnls(exponentials, mean_fitness)
    elif method == "nnls_reg":
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        m, p = exponentials.shape
        A_aug = np.vstack([exponentials, np.sqrt(alpha) * np.eye(p)])
        b_aug = np.concatenate([mean_fitness, np.zeros(p)])
        fourier_coeffs, _ = scipy.optimize.nnls(A_aug, b_aug)
    else:
        raise ValueError("Method unavailable.")
    if fix_b0:
        b0 = mean_fitness_0 - np.sum(np.abs(fourier_coeffs))
        fourier_coeffs = np.concatenate((np.array([b0]), fourier_coeffs))
        exponentials = get_exp_matrix(N=N, A=A, mutations=mutations, is_squared=is_squared, fix_b0=False)
    return (fourier_coeffs, exponentials)

def model_function(x,*params):

    """
    This is the what we are fitting to (sum of exponentials).
    It assumes a decay rate of 0.5 mutations per step.

    x = steps
    params = the output of the fitting function (get_single_decay_rate).
    """

    mut = 0.5
    num_params = 1
    constant = params[-1]
    params = params[:-1]
    mut_curves = np.exp(-1.0*mut*x[:,None]*np.array(params)[None,:])
    weights = np.linspace(0.1, 0.9, num_params)
    weights = np.ones(num_params)
    weights = weights / weights.sum()
    sum_curves = np.sum(mut_curves * weights[None,:], axis = 1)
    return sum_curves * (1 - constant) + constant

## Measure decay rate function.

def get_single_decay_rate(decay_data, mut = 0.5, num_steps = 25):

    num_params = 1
    decay_data = decay_data/decay_data[0]

    import jax.numpy as jnp
    if isinstance(mut, (int, float, complex)) or jnp.ndim(mut) == 0:
        steps = np.linspace(0,num_steps-1,num_steps)
    else:
        steps = mut

    if mut is None:
        mut = np.arange(len(decay_data))  # Default steps

    init_guess = np.linspace(0.1, 0.9, num_params)
    init_guess = np.concat([init_guess,[0.0]])
    lbounds = [0.0]*num_params + [-0.4]
    ubounds = [2.0]*num_params + [0.4]

    params, _ = curve_fit(model_function, steps, decay_data,p0=init_guess, maxfev= 9000, ftol = 1e-4, xtol = 1e-5, bounds = (lbounds, ubounds))

    mean_params = np.mean(params[:-1])
    fitted_constant = params[-1]  # The second returned parameter

    return mean_params, fitted_constant  # Return full params for plotting

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with open(os.path.join(plot_data_dir, 'fourier_analysis.pkl'), 'rb') as f:
    nk_data = pickle.load(f)

# ---------------------------------------------------------------------------
# Plotting — copied verbatim from ruggedness_figures_plots.ipynb cell 82
# ---------------------------------------------------------------------------

dpi = 350

# Set globally
rcParams['font.family'] = 'Open Sans'

N = nk_data['N_used']
K_vec = nk_data['Ks_used']
A = nk_data['A_used']
NK_landscapes = nk_data['nk_builts']
NK_spectra = []
NK_max = [(K+1)*(A-1)/A for K in K_vec]
for f in NK_landscapes:
    NK_spectra.append(get_landscape_spectrum(f, norm = True, remove_constant = False, on_gpu = True))

mut = 0.5
mutations = np.arange(start=0, stop=5.5, step=mut)
num_steps = len(mutations)
exponentials = get_exp_matrix(N=N, A=A, mutations=mutations, is_squared=True)
fitness_decay = []
fitness_decay_terms = []
fitted_decay = []
fitted_rho = []
for i, K in enumerate(K_vec):
    fitness_decay.append(np.dot(exponentials, NK_spectra[i]))
    tmp = 1 # fitness_decay[i][0]
    fitness_decay[i] = fitness_decay[i] /tmp
    NK_spectra[i] = NK_spectra[i] / tmp
    fitness_decay_terms_K = np.zeros(exponentials.shape)
    for j in range(exponentials.shape[1]):
        fitness_decay_terms_K[:, j] = exponentials[:, j] * NK_spectra[i][j]
    fitness_decay_terms.append(fitness_decay_terms_K)

    estim_data = get_single_decay_rate(fitness_decay[i], mut = mut, num_steps = num_steps)
    # fitted_decay.append(np.array([np.exp(-mutations*(estim_data[0]))*(1 - estim_data[-1]) + estim_data[-1]]))
    fitted_decay.append(np.array([np.exp(-mutations*(estim_data[0]))*(1 - estim_data[-1])]))
    fitted_rho.append(estim_data[0])

markers = ['o', 's', 'D', '^', 'v', '<', '>', 'x', '+', '*']

linestyles = ['-', '--', '-.', ':']

fig = plt.figure(figsize=(18, 5))
gs = gridspec.GridSpec(2, 3, figure=fig)

axx = fig.add_subplot(gs[:, 0])
for i, nk in enumerate(K_vec):
    line, = axx.plot(
        range(N + 1), NK_spectra[i],
        label=f"$K={nk}$",
        marker=markers[i],
        markersize=6,
        linewidth=1.5
    )
    color = line.get_color()
    axx.axvline(NK_max[i], color=color, linestyle=':')
    axx.axvline(fitted_rho[i]*N*(A-1)/2/A, color=color, linestyle='--')
    # axx.axvline(fitted_rho_NEW[i]*N*(A-1)/2/A, color=color, linestyle='-.')
    line, = axx.plot(
        range(N + 1), NK_spectra[i],
        color=color,
        markersize=6,
        linewidth=1.5
    )
axx.legend()
axx.set_xlim([0, N])
axx.set_ylim([0, 1])
axx.set_xlabel("Frequency index $i$ (-)")
axx.set_ylabel(f"Coefficients $b_i$ (-)")
axx.set_title(f"Power spectra (N={N}, A={A})", fontsize=10, fontweight='bold')
axx.text(
    -0.1, 1, "a",            # x, y in axes fraction
    transform=axx.transAxes,
    fontsize=20,
    va='bottom', ha='right'
)

# middle plot
for axi, (sel, ii) in enumerate(zip([1,3],[2,5])):
    axx = axx = fig.add_subplot(gs[axi, 1])

    axx.plot(
        mutations, fitness_decay[sel]-fitness_decay_terms[sel][0, 0],
        label=f"$G_{{\\mu}}-b_0$", color='k', linestyle='-', markersize=6, linewidth=2.2,
    )
    axx.plot(
        mutations, fitted_decay[sel][0,:],
        label=f"$G_{{\\mu,\\rho}}-c$", color='black', linestyle=':', markersize=6, linewidth=2.2,
    )
    for j, (ls, mk) in zip(range(exponentials.shape[1]), itertools.product(linestyles, markers)):
        if j>0 and NK_spectra[sel][j]>1e-4:
            if j == ii:
                axx.plot(
                    mutations,
                    fitness_decay_terms[sel][:, j], # + fitness_decay[sel] - fitness_decay_terms[sel][0, j],
                    label=f"$b_{{{j}}}\\mathrm{{e}}^{{\\frac{{-2\\mu\\lambda_{i}}}{{d}}}}$",
                    linestyle='-',
                    marker=mk, markersize=6, linewidth=2.2,
                )
            else:
                axx.plot(
                    mutations,
                    fitness_decay_terms[sel][:, j], # + fitness_decay[sel] - fitness_decay_terms[sel][0, j],
                    #label=f"$b_{{{j}}}\\mathrm{{e}}^{{\\frac{{-2\\mu\\lambda_{i}}}{{d}}}}$",
                    linestyle='--',
                    marker=mk, markersize=4, linewidth=1.5,
                )
    axx.legend(ncol=1)
    axx.set_title(f"Fitness decay ($K={K_vec[sel]}$)", fontsize=10, fontweight='bold')
    axx.set_xlabel(f"Mutations $\\mu$ (-)", )
    axx.set_ylabel(f"Decay (-)" )
    axx.set_xlim([0,mutations.max()])
    axx.set_ylim([0,(fitness_decay[sel]-fitness_decay_terms[sel][0, 0]).max()])

    letters = ["b", "c"]
    axx.text(
        -0.1, 1, letters[axi],
        transform=axx.transAxes,
        fontsize=20,
        va='bottom', ha='right'
    )

# right plot
for i, sel in enumerate([1,3]):
    decay = fitness_decay[sel]
    np.random.seed(2122)
    dnoise = 0.05
    noise = np.random.uniform(low=-dnoise,high=dnoise,size=fitness_decay[sel].shape)
    decay_noisy = fitness_decay[sel] + noise

    spectrum, _ = get_fourier_coeffs(mean_fitness=decay, mutations=mutations, N=N, A=A, is_squared=True, method="ls_constrained", fix_b0=True)
    spectrum_noisy, _ = get_fourier_coeffs(mean_fitness=decay_noisy, mutations=mutations, N=N, A=A, is_squared=True, method="nnls", fix_b0=True)
    spectrum_reg, _ = get_fourier_coeffs(mean_fitness=decay_noisy, mutations=mutations, N=N, A=A, is_squared=True, method="nnls_reg",alpha=1e-3, fix_b0=True)

    axx = fig.add_subplot(gs[i, 2])
    axx.plot(range(N + 1), NK_spectra[sel], label=f"True", markersize=7, linewidth=2, marker=markers[0])
    axx.plot(range(N + 1), spectrum, label=f"Estimated", markersize=5, linewidth=1.5, marker=markers[1], linestyle="--")
    axx.plot(range(N + 1), spectrum_noisy, label=f"Noisy", markersize=6, linewidth=1.5, marker=markers[2], linestyle="-")
    axx.plot(range(N + 1), spectrum_reg, label=f"Regularised", markersize=6, linewidth=1.5, marker=markers[3], linestyle=":")
    axx.legend()
    axx.set_xlim([0, N])
    axx.set_ylim([0, max(1,spectrum_noisy.max())])
    axx.set_xlabel("Frequency index $i$ (-)")
    axx.set_ylabel(f"Coefficients $b_i$ (-)")
    axx.set_title(f"Spectrum estimation ($K={K_vec[sel]}$)", fontsize=10, fontweight='bold')

    letters = ["d", "e"]
    axx.text(
        -0.1, 1, letters[i],
        transform=axx.transAxes,
        fontsize=20,
        va='bottom', ha='right'
    )

fig.subplots_adjust(hspace=0.5, wspace=0.25)
out_path = os.path.join(figures_dir, 'NK_spectra.pdf')
plt.savefig(out_path, dpi=dpi)
print(f'Saved -> {out_path}')
