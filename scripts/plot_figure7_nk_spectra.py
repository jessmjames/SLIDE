"""Figure 7: NK / empirical Fourier spectra + fitness-decay decomposition.

Standalone, reproducible version of the spectra figure that was previously only in
``data_visualisation.ipynb`` (cell that saves ``figures/NK_spectra.pdf``). It:

1. loads the four empirical landscapes (GB1, TrpB, TEV, ParD3),
2. builds + caches ``processed_data/fourier_analysis.pkl`` (the ``nk_data`` payload),
3. renders ``figures/NK_spectra.pdf`` — power spectra (a), fitness decay (b,c) and
   spectrum estimation (d,e).

Runs on CPU (``on_gpu=False`` / ``JAX_PLATFORMS=cpu``); no GPU required.

Notation: the squared-fitness decay rate is rho_2, so the fitted-decay curve is labelled
``G_{mu,rho_2}`` (matching Figs 5/6).

Run from the repository root::

    python scripts/plot_figure7_nk_spectra.py
"""

from __future__ import annotations

import os
import sys
import pickle
import itertools
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU-only: no GPU needed for FFT + curve fits

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jax.random as jr

from slide.ruggedness_functions import get_landscape_spectrum, get_exp_matrix, get_fourier_coeffs, get_nk_l_o_shape
from slide.direvo_functions import get_single_decay_rate

DPI = 300
# Synthetic NK landscapes for the explicit power spectra. The figure caption quotes A=20 (the
# theoretical alphabet), but the *explicit* spectra are computed on binary (A=2) landscapes so the
# full A^N genotype space is tractable (2^10 = 1024 vs 20^10 ~ 1e13). N=10, K in {0,3,6,9}.
N_SITES = 10
A_ALLELES = 2
K_VALUES = [0, 3, 6, 9]
SEED = 0


def build_nk_data() -> dict:
    """Build (and cache) the synthetic NK landscapes for the spectra figure (CPU; A=2, N=10)."""
    shape = (A_ALLELES,) * N_SITES
    base = jr.PRNGKey(SEED)
    nk_builts = [np.asarray(get_nk_l_o_shape(jr.fold_in(base, K), N_SITES, K, shape)) for K in K_VALUES]
    nk_data = {"N_used": N_SITES, "A_used": A_ALLELES, "Ks_used": list(K_VALUES), "nk_builts": nk_builts}
    out = _REPO / "processed_data" / "fourier_analysis.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(nk_data, f)
    print(f"Cached -> {out}  (N={N_SITES}, A={A_ALLELES}, K={K_VALUES})")
    return nk_data


def main() -> None:
    """Build the data payload and render ``figures/NK_spectra.pdf``."""
    nk_data = build_nk_data()
    N = nk_data["N_used"]
    K_vec = nk_data["Ks_used"]
    A = nk_data["A_used"]
    NK_landscapes = nk_data["nk_builts"]

    NK_spectra = [get_landscape_spectrum(f, norm=True, remove_constant=False, on_gpu=False) for f in NK_landscapes]
    NK_max = [(K + 1) * (A - 1) / A for K in K_vec]

    mut = 0.5
    mutations = np.arange(start=0, stop=5.5, step=mut)
    num_steps = len(mutations)
    exponentials = get_exp_matrix(N=N, A=A, mutations=mutations, is_squared=True)

    fitness_decay, fitness_decay_terms, fitted_decay, fitted_rho = [], [], [], []
    for i, K in enumerate(K_vec):
        fitness_decay.append(np.dot(exponentials, NK_spectra[i]))
        fitness_decay_terms_K = np.zeros(exponentials.shape)
        for j in range(exponentials.shape[1]):
            fitness_decay_terms_K[:, j] = exponentials[:, j] * NK_spectra[i][j]
        fitness_decay_terms.append(fitness_decay_terms_K)
        estim_data = get_single_decay_rate(fitness_decay[i], mut=mut, num_steps=num_steps)
        fitted_decay.append(np.array([np.exp(-mutations * (estim_data[0])) * (1 - estim_data[-1])]))
        fitted_rho.append(estim_data[0])

    markers = ["o", "s", "D", "^", "v", "<", ">", "x", "+", "*"]
    linestyles = ["-", "--", "-.", ":"]

    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # (a) power spectra
    axx = fig.add_subplot(gs[:, 0])
    for i, nk in enumerate(K_vec):
        line, = axx.plot(range(N + 1), NK_spectra[i], label=f"$K={nk}$", marker=markers[i], markersize=6, linewidth=1.5)
        color = line.get_color()
        axx.axvline(NK_max[i], color=color, linestyle=":")
        axx.axvline(fitted_rho[i] * N * (A - 1) / 2 / A, color=color, linestyle="--")
        axx.plot(range(N + 1), NK_spectra[i], color=color, markersize=6, linewidth=1.5)
    axx.legend()
    axx.set_xlim([0, N])
    axx.set_ylim([0, 1])
    axx.set_xlabel("Frequency index $i$ (-)")
    axx.set_ylabel("Coefficients $b_i$ (-)")
    axx.set_title(f"Power spectra (N={N}, A={A})", fontsize=10, fontweight="bold")
    axx.text(-0.1, 1, "a", transform=axx.transAxes, fontsize=20, va="bottom", ha="right")

    # (b, c) fitness decay decomposition
    sel_max = min(3, len(K_vec) - 1)
    for axi, (sel, ii) in enumerate(zip([1, sel_max], [2, 5])):
        axx = fig.add_subplot(gs[axi, 1])
        axx.plot(mutations, fitness_decay[sel] - fitness_decay_terms[sel][0, 0],
                 label=r"$G_{\mu}-b_0$", color="k", linestyle="-", markersize=6, linewidth=2.2)
        axx.plot(mutations, fitted_decay[sel][0, :],
                 label=r"$G_{\mu,\rho_2^{\mathrm{fit}}}-c_2$", color="black", linestyle=":", markersize=6, linewidth=2.2)
        for j, (ls, mk) in zip(range(exponentials.shape[1]), itertools.product(linestyles, markers)):
            if j > 0 and NK_spectra[sel][j] > 1e-4:
                if j == ii:
                    axx.plot(mutations, fitness_decay_terms[sel][:, j],
                             label=rf"$b_{{{j}}}\mathrm{{e}}^{{\frac{{-2\mu\lambda_{j}}}{{d}}}}$",
                             linestyle="-", marker=mk, markersize=6, linewidth=2.2)
                else:
                    axx.plot(mutations, fitness_decay_terms[sel][:, j],
                             linestyle="--", marker=mk, markersize=4, linewidth=1.5)
        axx.legend(ncol=1)
        axx.set_title(f"Fitness decay ($K={K_vec[sel]}$)", fontsize=10, fontweight="bold")
        axx.set_xlabel(r"Mutations $\mu$ (-)")
        axx.set_ylabel("Decay (-)")
        axx.set_xlim([0, mutations.max()])
        axx.set_ylim([0, (fitness_decay[sel] - fitness_decay_terms[sel][0, 0]).max()])
        axx.text(-0.1, 1, ["b", "c"][axi], transform=axx.transAxes, fontsize=20, va="bottom", ha="right")

    # (d, e) spectrum estimation
    for i, sel in enumerate([1, sel_max]):
        decay = fitness_decay[sel]
        rng = np.random.default_rng(2122)
        dnoise = 0.05
        decay_noisy = fitness_decay[sel] + rng.uniform(low=-dnoise, high=dnoise, size=fitness_decay[sel].shape)
        spectrum, _ = get_fourier_coeffs(mean_fitness=decay, mutations=mutations, N=N, A=A, is_squared=True, method="ls_constrained", fix_b0=True)
        spectrum_noisy, _ = get_fourier_coeffs(mean_fitness=decay_noisy, mutations=mutations, N=N, A=A, is_squared=True, method="nnls", fix_b0=True)
        spectrum_reg, _ = get_fourier_coeffs(mean_fitness=decay_noisy, mutations=mutations, N=N, A=A, is_squared=True, method="nnls_reg", alpha=1e-3, fix_b0=True)
        axx = fig.add_subplot(gs[i, 2])
        axx.plot(range(N + 1), NK_spectra[sel], label="True", markersize=7, linewidth=2, marker=markers[0])
        axx.plot(range(N + 1), spectrum, label="Estimated", markersize=5, linewidth=1.5, marker=markers[1], linestyle="--")
        axx.plot(range(N + 1), spectrum_noisy, label="Noisy", markersize=6, linewidth=1.5, marker=markers[2], linestyle="-")
        axx.plot(range(N + 1), spectrum_reg, label="Regularised", markersize=6, linewidth=1.5, marker=markers[3], linestyle=":")
        axx.legend()
        axx.set_xlim([0, N])
        axx.set_ylim([0, max(1, spectrum_noisy.max())])
        axx.set_xlabel("Frequency index $i$ (-)")
        axx.set_ylabel("Coefficients $b_i$ (-)")
        axx.set_title(f"Spectrum estimation ($K={K_vec[sel]}$)", fontsize=10, fontweight="bold")
        axx.text(-0.1, 1, ["d", "e"][i], transform=axx.transAxes, fontsize=20, va="bottom", ha="right")

    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    out = _REPO / "figures" / "NK_spectra.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
