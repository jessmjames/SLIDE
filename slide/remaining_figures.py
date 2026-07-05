"""Shared data processing and plotting for the post-Figure-5 notebooks.

The functions in this module keep the notebooks compact while preserving the
raw-data, processed-data, and plotting stages used by Figures 3--5.
"""

from __future__ import annotations

from collections.abc import Callable
import pickle
from pathlib import Path

import jax.random as jr
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from slide.direvo_functions import get_single_decay_rate, get_single_decay_rate_IK_v2, model_function_IK_v2
from slide.ruggedness_functions import get_exp_matrix, get_fourier_coeffs, get_landscape_spectrum, get_nk_l_o_shape
from slide.utils import (
    FIGURE_LABEL_SIZE,
    FIGURE_LEGEND_SIZE,
    FIGURE_TICK_SIZE,
    FIGURE_TITLE_SIZE,
    PANEL_LETTER_SIZE,
    get_figures_dir,
    get_processed_data_dir,
    get_raw_data_dir,
    load_pickle,
    save_pickle,
)

SAVE_TYPES = ("pdf", "png", "eps")
PANEL_DPI = 350
LANDSCAPE_NAMES = ("GB1", "TrpB", "TEV", "ParD3")
LANDSCAPE_KEYS = ("gb1", "trpb", "tev", "pard3")
LANDSCAPE_COLORS = ("tab:orange", "tab:blue", "tab:green", "tab:red")
LANDSCAPE_MARKERS = ("o", "s", "^", "D")
MUTATION_MODELS = ("nuc_uniform", "nuc_h_sapiens_sym", "nuc_e_coli")


def save_figure(fig: Figure, stem: str, *, bbox_inches: str = "tight") -> None:
    """Save a figure in all manuscript output formats.

    Parameters:
    - fig: Figure
        Matplotlib figure to save.
    - stem: str
        Filename stem without an extension.
    - bbox_inches: str
        Bounding-box mode passed to Matplotlib.

    Returns:
    - None
        Files are written beneath the configured figures directory.
    """
    root = get_figures_dir()
    for suffix in SAVE_TYPES:
        destination = root / suffix
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination / f"{stem}.{suffix}", dpi=PANEL_DPI, bbox_inches=bbox_inches)


def add_panel_letter(ax: Axes, letter: str) -> None:
    """Add a bold manuscript panel letter to an axis.

    Parameters:
    - ax: Axes
        Axis receiving the annotation.
    - letter: str
        Panel letter.

    Returns:
    - None
        The annotation is added directly to ``ax``.
    """
    ax.text(-0.14, 1.10, letter, transform=ax.transAxes, fontsize=PANEL_LETTER_SIZE,
            fontweight="bold", va="top", ha="left")


def load_figure6_sampling_payload() -> dict[str, object]:
    """Load the existing 75-generation Figure 6 sampling products.

    Returns:
    - dict[str, object]
        Sampling distributions and spectral reference values.
    """
    processed = get_processed_data_dir()
    data = {
        model: load_pickle(processed / f"trajectory_subsampling_{model}_75steps.pkl")
        for model in MUTATION_MODELS
    }
    spectral = load_pickle(processed / "spectral_rho_comparison.pkl")
    return {
        "data": data,
        "spectral": spectral,
        "params": {"M": 75, "bootstrap_replicates": 1000},
        "metadata": {"paper_reference": "Figure 6D-F"},
    }


def process_figure6_sampling_payload(
    raw_by_model: dict[str, list[np.ndarray]],
    spectral: dict[str, dict[str, float]],
    *,
    bootstrap_replicates: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """Bootstrap fitted ruggedness over increasing numbers of starting points.

    Parameters:
    - raw_by_model: dict[str, list[np.ndarray]]
        Raw all-start decay arrays, ordered by empirical landscape for each model.
    - spectral: dict[str, dict[str, float]]
        Analytical ruggedness references by landscape and mutation model.
    - bootstrap_replicates: int
        Number of bootstrap estimates at every sampling depth.
    - seed: int
        NumPy random seed.

    Returns:
    - dict[str, object]
        Figure 6 sampling distributions and metadata.
    """
    rng = np.random.default_rng(seed)
    max_starts = (160_000, 160_000, 160_000, 8_000)
    processed: dict[str, list[list[np.ndarray]]] = {}
    for model, landscape_arrays in raw_by_model.items():
        model_results: list[list[np.ndarray]] = []
        for raw, maximum in zip(landscape_arrays, max_starts, strict=True):
            start_curves = np.asarray(raw, dtype=float).mean(axis=2).reshape(-1, 75)
            sample_counts = np.round(np.logspace(0, np.log10(maximum), 11)).astype(int)
            landscape_results = []
            for count in sample_counts:
                estimates = []
                for _ in range(bootstrap_replicates):
                    indices = rng.choice(start_curves.shape[0], size=int(count), replace=True)
                    curve = np.square(start_curves[indices]).mean(axis=0)
                    curve /= max(float(curve[0]), 1e-10)
                    estimates.append(float(get_single_decay_rate_IK_v2(curve, mut=0.1, num_steps=75)[0] / 2))
                landscape_results.append(np.asarray(estimates))
            model_results.append(landscape_results)
        processed[model] = model_results
    return {
        "data": processed,
        "spectral": spectral,
        "params": {"M": 75, "bootstrap_replicates": bootstrap_replicates, "seed": seed},
        "metadata": {"paper_reference": "Figure 6D-F"},
    }
def plot_sampling_accuracy(ax: Axes, payload: dict[str, object], model: str) -> None:
    """Plot fitted ruggedness against the number of sampled starting points.

    Parameters:
    - ax: Axes
        Axis receiving the panel.
    - payload: dict[str, object]
        Figure 6 sampling payload.
    - model: str
        Mutation-model key.

    Returns:
    - None
        Lines and uncertainty bands are added to ``ax``.
    """
    titles = {
        "nuc_uniform": "Uniform mutation",
        "nuc_h_sapiens_sym": "H. sapiens (symmetric)",
        "nuc_e_coli": "E. coli (asymmetric)",
    }
    rho_labels = {
        "nuc_uniform": r"$\rho_2^{\mathrm{fit}}$",
        "nuc_h_sapiens_sym": r"$\widetilde{\rho}_2^{\mathrm{fit}}$",
        "nuc_e_coli": r"$\overline{\rho}_2^{\mathrm{fit}}$",
    }
    reference_labels = {
        "nuc_uniform": r"$\rho_2$",
        "nuc_h_sapiens_sym": r"$\widetilde{\rho}_2$",
        "nuc_e_coli": r"$\overline{\rho}_2$",
    }
    max_starts = (160_000, 160_000, 160_000, 8_000)
    model_data = payload["data"][model]  # type: ignore[index]
    spectral = payload["spectral"]  # type: ignore[assignment]
    for index, (name, maximum, color, marker) in enumerate(
        zip(LANDSCAPE_NAMES, max_starts, LANDSCAPE_COLORS, LANDSCAPE_MARKERS, strict=True)
    ):
        starts = np.round(np.logspace(0, np.log10(maximum), 11)).astype(int)
        values = model_data[index]
        means = np.asarray([np.mean(item) for item in values], dtype=float)
        stds = np.asarray([np.std(item) for item in values], dtype=float)
        ax.plot(starts, means, color=color, marker=marker, markersize=3.5,
                markevery=2, linewidth=1.3, label=name)
        ax.fill_between(starts, means - stds, means + stds, color=color, alpha=0.15, linewidth=0)
        reference_key = model if model != "nuc_e_coli" else "nuc_e_coli_sym"
        reference = spectral.get(name, {}).get(reference_key, np.nan)
        if np.isfinite(reference):
            ax.axhline(reference, color=color, linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_xscale("log")
    ax.set_title(titles[model], fontsize=FIGURE_TITLE_SIZE)
    ax.set_xlabel("Number of starting points", fontsize=FIGURE_LABEL_SIZE)
    ax.set_ylabel(rho_labels[model], fontsize=FIGURE_LABEL_SIZE)
    ax.set_xlim(1, 160_000)
    ax.set_ylim(0, 2)
    ax.tick_params(axis="both", labelsize=FIGURE_TICK_SIZE)
    reference_handle = mlines.Line2D(
        [], [], color="black", linewidth=1.0, linestyle="--", label=reference_labels[model]
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(reference_handle)
    labels.append(reference_labels[model])
    ax.legend(
        handles=handles,
        labels=labels,
        fontsize=FIGURE_LEGEND_SIZE,
        frameon=True,
        loc="upper right",
        ncol=1,
    )


def build_figure7_raw_payload() -> dict[str, object]:
    """Generate the tractable binary NK landscapes used by Figure 7.

    Returns:
    - dict[str, object]
        Raw synthetic landscapes and complete generation parameters.
    """
    n_sites, num_alleles, seed = 10, 2, 0
    k_values = (0, 3, 6, 9)
    shape = (num_alleles,) * n_sites
    base_key = jr.PRNGKey(seed)
    landscapes = [
        np.asarray(get_nk_l_o_shape(jr.fold_in(base_key, k_value), n_sites, k_value, shape))
        for k_value in k_values
    ]
    return {
        "data": {"landscapes": landscapes},
        "params": {"N": n_sites, "A": num_alleles, "K_values": k_values, "seed": seed},
        "metadata": {
            "paper_reference": "Figure 7",
            "description": "Binary explicit NK landscapes; A=2 is required for tractable 2^10 enumeration.",
        },
    }


def process_figure7_payload(raw_payload: dict[str, object]) -> dict[str, object]:
    """Compute Figure 7 spectra, decay decompositions, and spectrum estimates.

    Parameters:
    - raw_payload: dict[str, object]
        Raw Figure 7 NK landscapes.

    Returns:
    - dict[str, object]
        Fully processed arrays needed by panels A--E.
    """
    params = raw_payload["params"]
    n_sites = int(params["N"])  # type: ignore[index]
    num_alleles = int(params["A"])  # type: ignore[index]
    k_values = tuple(int(value) for value in params["K_values"])  # type: ignore[index]
    landscapes = raw_payload["data"]["landscapes"]  # type: ignore[index]
    spectra = [get_landscape_spectrum(item, norm=True, remove_constant=False, on_gpu=False) for item in landscapes]
    mutations = np.arange(0, 5.5, 0.5)
    exponentials = get_exp_matrix(N=n_sites, A=num_alleles, mutations=mutations, is_squared=True)
    decays, terms, fitted_decays, fitted_rhos = [], [], [], []
    estimates = {}
    for index, spectrum in enumerate(spectra):
        decay = np.dot(exponentials, spectrum)
        component_terms = exponentials * np.asarray(spectrum)[None, :]
        fit = get_single_decay_rate(decay, mut=0.5, num_steps=len(mutations))
        fitted = np.exp(-mutations * fit[0]) * (1 - fit[-1])
        decays.append(decay)
        terms.append(component_terms)
        fitted_decays.append(fitted)
        fitted_rhos.append(float(fit[0]))
        if index in (1, 3):
            rng = np.random.default_rng(2122)
            noisy = decay + rng.uniform(-0.05, 0.05, size=decay.shape)
            estimated, _ = get_fourier_coeffs(decay, mutations, n_sites, num_alleles, True,
                                               method="ls_constrained", fix_b0=True)
            noisy_estimate, _ = get_fourier_coeffs(noisy, mutations, n_sites, num_alleles, True,
                                                    method="nnls", fix_b0=True)
            regularised, _ = get_fourier_coeffs(noisy, mutations, n_sites, num_alleles, True,
                                                 method="nnls_reg", alpha=1e-3, fix_b0=True)
            estimates[index] = {
                "estimated": estimated,
                "noisy": noisy_estimate,
                "regularised": regularised,
            }
    return {
        "data": {
            "spectra": spectra, "mutations": mutations, "decays": decays,
            "terms": terms, "fitted_decays": fitted_decays, "fitted_rhos": fitted_rhos,
            "estimates": estimates,
        },
        "params": {"N": n_sites, "A": num_alleles, "K_values": k_values, "noise": 0.05},
        "metadata": {"paper_reference": "Figure 7"},
    }


def plot_figure7_panel(ax: Axes, payload: dict[str, object], panel: str) -> None:
    """Plot one Figure 7 panel.

    Parameters:
    - ax: Axes
        Axis receiving the panel.
    - payload: dict[str, object]
        Processed Figure 7 payload.
    - panel: str
        Panel letter A--E.

    Returns:
    - None
        Panel artists are added to ``ax``.
    """
    data = payload["data"]
    params = payload["params"]
    spectra = data["spectra"]  # type: ignore[index]
    mutations = np.asarray(data["mutations"])  # type: ignore[index]
    k_values = tuple(params["K_values"])  # type: ignore[arg-type]
    markers = ("o", "s", "D", "^")
    if panel == "A":
        for index, (k_value, spectrum) in enumerate(zip(k_values, spectra, strict=True)):
            line, = ax.plot(range(len(spectrum)), spectrum, marker=markers[index], markersize=4,
                            linewidth=1.2, label=f"K = {k_value}")
            maximum = (k_value + 1) * (int(params["A"]) - 1) / int(params["A"])
            fitted_frequency = data["fitted_rhos"][index] * int(params["N"]) * (int(params["A"]) - 1) / (2 * int(params["A"]))  # type: ignore[index]
            ax.axvline(maximum, color=line.get_color(), linestyle=":", linewidth=0.9)
            ax.axvline(fitted_frequency, color=line.get_color(), linestyle="--", linewidth=0.9)
        ax.set_title(f"Power spectra (N={params['N']}, A={params['A']})", fontsize=FIGURE_TITLE_SIZE)
        ax.set_xlabel(r"Frequency index $i$", fontsize=FIGURE_LABEL_SIZE)
        ax.set_ylabel(r"Coefficients $b_i$", fontsize=FIGURE_LABEL_SIZE)
        ax.set_ylim(0, 1)
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([
            mlines.Line2D([], [], color="black", linestyle=":", linewidth=1.0,
                          label=r"Analytical $\rho_2 \times d/A$"),
            mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.0,
                          label=r"Fitted $\rho_2^{\mathrm{fit}} \times d/A$"),
        ])
        labels.extend([
            r"Analytical $\rho_2 \times d/A$",
            r"Fitted $\rho_2^{\mathrm{fit}} \times d/A$",
        ])
        ax.legend(handles=handles, labels=labels, fontsize=FIGURE_LEGEND_SIZE)
    elif panel in ("B", "C"):
        selected = 1 if panel == "B" else 3
        decay = np.asarray(data["decays"][selected])  # type: ignore[index]
        terms = np.asarray(data["terms"][selected])  # type: ignore[index]
        fitted = np.asarray(data["fitted_decays"][selected])  # type: ignore[index]
        ax.plot(mutations, decay - terms[0, 0], color="black", linewidth=1.8, label=r"$G_\mu-b_0$")
        ax.plot(mutations, fitted, color="black", linestyle=":", linewidth=1.8,
                label=r"$G_{\mu,\rho_2^{\mathrm{fit}}}-c_2$")
        dominant = np.argsort(np.asarray(spectra[selected]))[-5:]
        for component in dominant:
            if component == 0:
                continue
            ax.plot(mutations, terms[:, component], marker=markers[component % len(markers)],
                    markersize=3, linewidth=1.0, linestyle="--",
                    label=rf"$b_{{{component}}}\mathrm{{e}}^{{-2\mu\lambda_{{{component}}}/d}}$")
        ax.set_title(
            f"Fitness decay (N={params['N']}, K={k_values[selected]})",
            fontsize=FIGURE_TITLE_SIZE,
        )
        ax.set_xlabel(r"Mutations $\mu$", fontsize=FIGURE_LABEL_SIZE)
        ax.set_ylabel(r"$G_\mu-b_0/A^N$", fontsize=FIGURE_LABEL_SIZE)
        ax.legend(fontsize=FIGURE_LEGEND_SIZE, ncol=2)
    else:
        selected = 1 if panel == "D" else 3
        estimates = data["estimates"][selected]  # type: ignore[index]
        x_values = range(len(spectra[selected]))
        ax.plot(x_values, spectra[selected], marker="o", linewidth=1.4, label="True")
        ax.plot(x_values, estimates["estimated"], marker="s", linestyle="--", linewidth=1.1, label="Estimated")
        ax.plot(x_values, estimates["noisy"], marker="D", linewidth=1.1, label="Noisy")
        ax.plot(x_values, estimates["regularised"], marker="^", linestyle=":", linewidth=1.1, label="Regularised")
        ax.set_title(
            f"Spectrum estimation (N={params['N']}, K={k_values[selected]})",
            fontsize=FIGURE_TITLE_SIZE,
        )
        ax.set_xlabel(r"Frequency index $i$", fontsize=FIGURE_LABEL_SIZE)
        ax.set_ylabel(r"Coefficients $b_i$", fontsize=FIGURE_LABEL_SIZE)
        ax.legend(fontsize=FIGURE_LEGEND_SIZE)
    ax.tick_params(labelsize=FIGURE_TICK_SIZE)
    ax.grid(True, alpha=0.16)


def load_figure_s2_payload() -> dict[str, object]:
    """Load and bin the cached Figure S2 biased-mutation estimates.

    Returns:
    - dict[str, object]
        Binned means and standard deviations for three mutation models.
    """
    processed = get_processed_data_dir()
    model_files = {
        "E. coli": "ruggedness_accuracy_codon_e_coli.pkl",
        "A. thaliana": "ruggedness_accuracy_codon_a_thaliana.pkl",
        "Human": "ruggedness_accuracy_codon_human.pkl",
    }
    data = {}
    for label, filename in model_files.items():
        true_rho, estimates = load_pickle(processed / filename)
        estimates = np.asarray(estimates, dtype=float)
        order = np.argsort(true_rho)
        grouped_true = np.asarray(true_rho, dtype=float)[order].reshape(10, -1)
        grouped_estimated = estimates.mean(axis=1)[order].reshape(10, -1)
        data[label] = {
            "true": grouped_true.mean(axis=1),
            "mean": grouped_estimated.mean(axis=1),
            "std": grouped_estimated.std(axis=1),
        }
    return {
        "data": data,
        "params": {"estimates_per_pair": 250, "M": 25, "mutation_rate": 0.5},
        "metadata": {"paper_reference": "Figure S2", "caption_note": "Repository data contain 250, not 300, estimates."},
    }


def process_figure_s2_sweeps(raw_by_label: dict[str, np.ndarray]) -> dict[str, object]:
    """Fit and bin the three biased-mutation NK decay sweeps for Figure S2.

    Parameters:
    - raw_by_label: dict[str, np.ndarray]
        Raw sweep arrays keyed by display label.

    Returns:
    - dict[str, object]
        Binned means and standard deviations for plotting.
    """
    n_values = np.repeat(np.linspace(10, 50, 10), 10)
    k_values = np.concatenate([np.linspace(1, n_value, 10) for n_value in np.linspace(10, 50, 10)])
    true_rho = np.clip((k_values + 1) / n_values, 0, 1)[::-1]
    data = {}
    for label, raw in raw_by_label.items():
        curves = np.asarray(raw, dtype=float).reshape(100, -1, 25)
        curves /= np.maximum(curves[:, :, :1], 1e-10)
        estimates = np.zeros(curves.shape[:2], dtype=float)
        for pair_index in range(curves.shape[0]):
            for estimate_index in range(curves.shape[1]):
                estimates[pair_index, estimate_index] = get_single_decay_rate(
                    curves[pair_index, estimate_index], mut=0.5, num_steps=25
                )[0]
        order = np.argsort(true_rho)
        grouped_true = true_rho[order].reshape(10, -1)
        grouped_estimated = estimates.mean(axis=1)[order].reshape(10, -1)
        data[label] = {
            "true": grouped_true.mean(axis=1),
            "mean": grouped_estimated.mean(axis=1),
            "std": grouped_estimated.std(axis=1),
        }
    return {
        "data": data,
        "params": {"estimates_per_pair": 250, "M": 25, "mutation_rate": 0.5},
        "metadata": {"paper_reference": "Figure S2", "caption_note": "Repository data contain 250 estimates."},
    }
def plot_s2_panel(ax: Axes, payload: dict[str, object], model_label: str) -> None:
    """Plot one biased-mutation ruggedness-accuracy panel.

    Parameters:
    - ax: Axes
        Axis receiving the plot.
    - payload: dict[str, object]
        Processed Figure S2 payload.
    - model_label: str
        Display label identifying the mutation model.

    Returns:
    - None
        Panel artists are added directly to ``ax``.
    """
    panel = payload["data"][model_label]  # type: ignore[index]
    ax.plot(panel["true"], panel["mean"], "o-", linewidth=1.4,
            label=r"Mean $\bar{\rho}_2^{\mathrm{fit}}$")
    ax.fill_between(panel["true"], panel["mean"] - panel["std"], panel["mean"] + panel["std"], alpha=0.25)
    ax.plot(panel["true"], panel["true"], color="red", linestyle="--", alpha=0.55, label=r"$\rho_{NK}$")
    ax.set_title(model_label, fontsize=FIGURE_TITLE_SIZE)
    ax.set_xlabel(r"$\rho_{NK}$", fontsize=FIGURE_LABEL_SIZE)
    ax.set_ylabel(r"$\bar{\rho}_2^{\mathrm{fit}}$", fontsize=FIGURE_LABEL_SIZE)
    ax.tick_params(labelsize=FIGURE_TICK_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.18)


def figure_s3_required_paths(slide_data_dir: Path) -> list[Path]:
    """Return the twelve expected 75-step raw paths for Figure S3.

    Parameters:
    - slide_data_dir: Path
        Directory containing the colleague-provided all-start simulations.

    Returns:
    - list[Path]
        Required raw-data paths in panel order.
    """
    return [
        slide_data_dir / f"decay_curves_{landscape}_{model}_m0.1_all_starts_75steps.pkl"
        for model in MUTATION_MODELS
        for landscape in LANDSCAPE_KEYS
    ]


def process_figure_s3_payload(slide_data_dir: Path) -> dict[str, object]:
    """Process the twelve empirical decay products used by Figure S3.

    Parameters:
    - slide_data_dir: Path
        Directory containing 75-generation raw curves.

    Returns:
    - dict[str, object]
        Observed, fitted, and idealised curves for all twelve panels.

    Raises:
    - FileNotFoundError
        If any required colleague-provided raw product is absent.
    """
    required = figure_s3_required_paths(slide_data_dir)
    missing = [path for path in required if not path.exists()]
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Figure S3 requires the following 75-step raw files:\n{listing}")
    processed_dir = get_processed_data_dir()
    spectral = load_pickle(processed_dir / "spectral_rho_comparison.pkl")
    constants_path = processed_dir / "true_constants_nuc.pkl"
    constants = load_pickle(constants_path) if constants_path.exists() else {}
    panels = {}
    for model in MUTATION_MODELS:
        for landscape_key, landscape_name in zip(LANDSCAPE_KEYS, LANDSCAPE_NAMES, strict=True):
            path = slide_data_dir / f"decay_curves_{landscape_key}_{model}_m0.1_all_starts_75steps.pkl"
            with path.open("rb") as handle:
                raw = np.asarray(pickle.load(handle), dtype=float)
            start_curves = raw.mean(axis=2).reshape(-1, 75)
            observed = np.square(start_curves).mean(axis=0)
            scale = max(float(observed[0]), 1e-10)
            normalized = observed / scale
            fitted_rate, fitted_amplitude, fitted_constant = get_single_decay_rate_IK_v2(
                normalized, mut=0.1, num_steps=75
            )
            fitted = model_function_IK_v2(
                np.arange(75), fitted_rate, fitted_amplitude * scale,
                fitted_constant * scale, mut=0.1,
            )
            constant_key = (landscape_name, model)
            true_amplitude, true_constant = constants.get(constant_key, (1.0, 0.0))
            spectral_key = model if model != "nuc_e_coli" else "nuc_e_coli_sym"
            rho = spectral.get(landscape_name, {}).get(spectral_key, fitted_rate / 2)
            idealised = model_function_IK_v2(
                np.arange(75), rho * 2, true_amplitude * scale, true_constant * scale, mut=0.1
            )
            panels[(model, landscape_name)] = {
                "observed": observed, "fitted": fitted, "idealised": idealised,
                "true_constant": true_constant * scale, "fitted_constant": fitted_constant * scale,
            }
    return {
        "data": panels,
        "params": {"M": 75, "mutation_rate": 0.1, "models": MUTATION_MODELS},
        "metadata": {"paper_reference": "Figure S3"},
    }


def plot_s3_panel(ax: Axes, payload: dict[str, object], model: str, landscape: str) -> None:
    """Plot one empirical mutation-model decay panel.

    Parameters:
    - ax: Axes
        Axis receiving the panel.
    - payload: dict[str, object]
        Processed Figure S3 payload.
    - model: str
        Mutation-model key.
    - landscape: str
        Empirical-landscape display name.

    Returns:
    - None
        Panel artists are added directly to ``ax``.
    """
    panel = payload["data"][(model, landscape)]  # type: ignore[index]
    generations = np.arange(len(panel["observed"]))
    ax.plot(generations, panel["observed"], "k.", markersize=2.2, alpha=0.5, label="Data")
    ax.plot(generations, panel["idealised"], color="tab:blue", linewidth=1.3, label="Idealised")
    ax.plot(generations, panel["fitted"], color="tab:orange", linestyle="--", linewidth=1.3, label="Fitted")
    ax.axhline(panel["true_constant"], color="tab:blue", linestyle=":", linewidth=0.9, label=r"True $c$")
    ax.axhline(panel["fitted_constant"], color="tab:orange", linestyle="-.", linewidth=0.9, label=r"Fitted $c$")
    ax.set_title(landscape, fontsize=FIGURE_TITLE_SIZE)
    ax.set_xlabel(r"Generations $M$", fontsize=FIGURE_LABEL_SIZE)
    ax.set_ylabel(r"$G_\mu$", fontsize=FIGURE_LABEL_SIZE)
    ax.tick_params(labelsize=FIGURE_TICK_SIZE)
    ax.spines[["top", "right"]].set_visible(False)


def process_figure_s4_payload(raw_payloads: dict[int, dict[str, object]]) -> dict[str, object]:
    """Reduce the M=25, 50, and 100 strategy grids to lookup summaries.

    Parameters:
    - raw_payloads: dict[int, dict[str, object]]
        Raw strategy grids keyed by generation count.

    Returns:
    - dict[str, object]
        Per-ruggedness-bin means and standard deviations for each horizon.
    """
    processed = {}
    for generations, payload in raw_payloads.items():
        data = np.asarray(payload["data"], dtype=float)
        params = payload["params"]
        base_chances = np.asarray(params["base_chances"], dtype=float)  # type: ignore[index]
        splits = np.asarray(params["splits"], dtype=int)  # type: ignore[index]
        pairs = np.asarray(params["nk_pairs"], dtype=int)  # type: ignore[index]
        rho = (pairs[:, 1] + 1) / pairs[:, 0]
        optimal_base, optimal_split = [], []
        for item in data:
            surface = item.mean(axis=-1)
            row, column = np.unravel_index(np.nanargmax(surface), surface.shape)
            optimal_base.append(base_chances[column])
            optimal_split.append(splits[row])
        rounded = np.round(rho, 1)
        grouped = np.unique(rounded)
        processed[generations] = {
            "rho": grouped,
            "base_mean": np.asarray([np.mean(np.asarray(optimal_base)[rounded == value]) for value in grouped]),
            "base_std": np.asarray([np.std(np.asarray(optimal_base)[rounded == value]) for value in grouped]),
            "split_mean": np.asarray([np.mean(np.asarray(optimal_split)[rounded == value]) for value in grouped]),
            "split_std": np.asarray([np.std(np.asarray(optimal_split)[rounded == value]) for value in grouped]),
        }
    return {
        "data": processed,
        "params": {"generation_counts": tuple(sorted(raw_payloads))},
        "metadata": {"paper_reference": "Figure S4"},
    }


def plot_s4_panel(ax: Axes, payload: dict[str, object], generations: int) -> Axes:
    """Plot one strategy-lookup horizon with a twin splitting axis.

    Parameters:
    - ax: Axes
        Primary base-chance axis.
    - payload: dict[str, object]
        Processed Figure S4 payload.
    - generations: int
        Directed-evolution horizon.

    Returns:
    - Axes
        Twin axis used for splitting.
    """
    panel = payload["data"][generations]  # type: ignore[index]
    twin = ax.twinx()
    ax.errorbar(panel["rho"], panel["base_mean"], yerr=panel["base_std"], fmt="o--",
                capsize=2.5, color="tab:orange", label="Optimal base chance")
    twin.errorbar(panel["rho"], panel["split_mean"], yerr=panel["split_std"], fmt="s--",
                  capsize=2.5, color="tab:blue", label="Optimal splitting")
    ax.set_title(f"M = {generations} generations", fontsize=FIGURE_TITLE_SIZE)
    ax.set_xlabel(r"$\rho_{NK}$", fontsize=FIGURE_LABEL_SIZE)
    ax.set_ylabel(r"Base chance $b$", fontsize=FIGURE_LABEL_SIZE)
    twin.set_ylabel(r"Splitting $s$", fontsize=FIGURE_LABEL_SIZE)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.02, 0.21)
    twin.set_ylim(0, 26)
    ax.tick_params(labelsize=FIGURE_TICK_SIZE)
    twin.tick_params(labelsize=FIGURE_TICK_SIZE)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    return twin


def load_or_process(
    path: Path,
    builder: Callable[[], dict[str, object]],
    *,
    overwrite: bool,
    plot_only: bool,
) -> dict[str, object]:
    """Load a processed payload or build it under notebook flag control.

    Parameters:
    - path: Path
        Processed payload destination.
    - builder: callable
        Zero-argument payload builder.
    - overwrite: bool
        Whether an existing payload may be replaced.
    - plot_only: bool
        Whether processing is forbidden.

    Returns:
    - dict[str, object]
        Loaded or newly built payload.
    """
    if path.exists() and (plot_only or not overwrite):
        return load_pickle(path)
    if plot_only:
        raise FileNotFoundError(f"PLOT_ONLY=True requires processed payload {path}")
    payload = builder()
    save_pickle(payload, path)
    return payload


def standard_paths() -> tuple[Path, Path, Path]:
    """Return configured raw, processed, and figure directories.

    Returns:
    - tuple[Path, Path, Path]
        Raw-data, processed-data, and figure output directories.
    """
    return get_raw_data_dir(), get_processed_data_dir(), get_figures_dir()
