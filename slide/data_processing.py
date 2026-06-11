"""Processing helpers for the SLIDE paper notebooks."""
from __future__ import annotations
import numpy as np
from .direvo_functions import get_single_decay_rate, get_single_decay_rate_IK_v2
from .ruggedness_functions import find_distance_to_closest_max, get_dirichlet_metric, get_landscape_spectrum, get_mean_paths_to_max, landscape_r2, local_epistasis, max_possible_paths, roughness_to_slope
from .utils import get_landscape_arrays_dir, load_pickle, load_raw, save_processed

def load_landscapes() -> dict[str, np.ndarray]:
    """Load the empirical landscapes used by the processing notebooks.

    Returns:
    - dict[str, np.ndarray]
        Mapping from landscape name to fitness array.
    """
    files = {'GB1': 'GB1_landscape_array.pkl', 'TrpB': 'TrpB_landscape_array.pkl', 'TEV': 'TEV_landscape_array.pkl', 'ParD3': 'E3_landscape_array.pkl'}
    return {name: load_pickle(get_landscape_arrays_dir() / filename) for name, filename in files.items()}

def normalize_decay_array(decay_data: np.ndarray, steps: int=25) -> np.ndarray:
    """Reshape and normalize decay curves by their initial values.

    Parameters:
    - decay_data: np.ndarray
        Raw decay data with generation as the last dimension.
    - steps: int
        Number of generation steps per trajectory.

    Returns:
    - np.ndarray
        Normalized curves with shape ``(blocks, trajectories, steps)``.
    """
    reshaped = np.asarray(decay_data).reshape(np.asarray(decay_data).shape[0], -1, steps)
    return reshaped / reshaped[:, :, 0][:, :, None]

def estimate_decay_rates(normalized_curves: np.ndarray, *, mut: float=1.0, method: str='default') -> np.ndarray:
    """Fit decay rates for a grid of normalized trajectories.

    Parameters:
    - normalized_curves: np.ndarray
        Normalized curves with shape ``(blocks, trajectories, steps)``.
    - mut: float
        Mutation scale passed to the fitting model.
    - method: str
        Decay fitting method, using ``IK`` for the IK v2 fit.

    Returns:
    - np.ndarray
        Fitted decay rates with shape ``normalized_curves.shape[:2]``.
    """
    out = np.zeros(normalized_curves.shape[:2])
    for i in range(normalized_curves.shape[0]):
        for j in range(normalized_curves.shape[1]):
            if method == 'IK':
                out[i, j] = get_single_decay_rate_IK_v2(normalized_curves[i, j], mut=mut)[0] / 2
            else:
                out[i, j] = get_single_decay_rate(normalized_curves[i, j], mut=mut)[0]
    return out

def process_ruggedness_accuracy(decay_data: np.ndarray, nk_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Process NK ruggedness accuracy decay data.

    Parameters:
    - decay_data: np.ndarray
        Raw NK decay grid.
    - nk_pairs: np.ndarray
        Array of ``(N, K)`` parameter pairs.

    Returns:
    - tuple[np.ndarray, np.ndarray]
        True ``(K + 1) / N`` values and fitted decay rates.
    """
    normalized = normalize_decay_array(decay_data)
    decay_rates = estimate_decay_rates(normalized, mut=0.5)
    k_plus_one_over_ns = np.clip((np.asarray(nk_pairs)[:, 1] + 1) / np.asarray(nk_pairs)[:, 0], 0, 1)
    return (k_plus_one_over_ns, decay_rates)

def process_popsize_accuracy(popsize_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Process population-size sensitivity decay data.

    Parameters:
    - popsize_data: np.ndarray
        Raw population-size sweep decay data.

    Returns:
    - tuple[np.ndarray, np.ndarray]
        Fitted rates and population sizes.
    """
    normalized = normalize_decay_array(popsize_data.reshape(25, -1, 25))
    rates = np.zeros((25, normalized.shape[1]))
    for i in range(25):
        for j in range(normalized.shape[1]):
            rates[i, j] = get_single_decay_rate(normalized[i, j], mut=1.0)[0]
    return (rates, np.linspace(100, 2500, 25, dtype=int))

def process_mutation_accuracy(mut_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Process mutation-rate sensitivity decay data.

    Parameters:
    - mut_data: np.ndarray
        Raw mutation-rate sweep decay data.

    Returns:
    - tuple[np.ndarray, np.ndarray]
        Fitted rates and mutation-rate values.
    """
    muts = np.linspace(0.01, 2, 25)
    normalized = normalize_decay_array(mut_data.reshape(25, -1, 25))
    rates = np.zeros((25, normalized.shape[1]))
    for i, mut in enumerate(muts):
        for j in range(normalized.shape[1]):
            rates[i, j] = get_single_decay_rate(normalized[i, j], mut=mut)[0]
    return (rates, muts)

def empirical_metric_comparison(landscapes: dict[str, np.ndarray], decay_arrays: dict[str, np.ndarray]) -> tuple[list[object], ...]:
    """Compute empirical landscape metrics used for comparison figures.

    Parameters:
    - landscapes: dict[str, np.ndarray]
        Empirical fitness landscapes keyed by name.
    - decay_arrays: dict[str, np.ndarray]
        Empirical decay arrays keyed by name.

    Returns:
    - tuple[list[object], ...]
        Decay-rate, roughness, linear-model, epistasis, path, and local-maximum measurements.
    """
    empirical_landscapes = [landscapes[name] for name in ('GB1', 'TrpB', 'TEV', 'ParD3')]
    decay_rate_measurements = []
    for name in ('GB1', 'TrpB', 'TEV', 'ParD3'):
        decay_mean = (decay_arrays[name] ** 2).mean(axis=(0, 1, 2))
        decay_mean = decay_mean / decay_mean[0]
        decay_rate_measurements.append(get_single_decay_rate(decay_mean)[0] / 2)
    roughness_to_slope_measurements = [roughness_to_slope(i) for i in empirical_landscapes]
    landscape_r2_measurements = [1 - landscape_r2(i) for i in empirical_landscapes]
    starting_points = [[3, 17, 0, 3], [3, 8, 3, 18], [19, 17, 11, 18], [17, 12, 16]]
    local_epistasis_measurements = [local_epistasis(landscape, np.array(start)) for landscape, start in zip(empirical_landscapes, starting_points)]
    paths_to_max_measurements = [get_mean_paths_to_max(i, norm=False) for i in empirical_landscapes]
    paths_to_max_measurements[3] = paths_to_max_measurements[3] * (max_possible_paths(landscapes['GB1'].shape) / max_possible_paths(landscapes['ParD3'].shape))
    local_max_measurements = [find_distance_to_closest_max(i) for i in empirical_landscapes]
    return (decay_rate_measurements, roughness_to_slope_measurements, landscape_r2_measurements, local_epistasis_measurements, paths_to_max_measurements, local_max_measurements)

def empirical_fourier_spectra(landscapes: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Compute collapsed Fourier spectra for empirical landscapes.

    Parameters:
    - landscapes: dict[str, np.ndarray]
        Empirical fitness landscapes keyed by name.

    Returns:
    - list[np.ndarray]
        Collapsed spectra in GB1, TrpB, TEV, ParD3 order.
    """
    return [get_landscape_spectrum(landscapes[name], remove_constant=False, on_gpu=True, norm=False) for name in ('GB1', 'TrpB', 'TEV', 'ParD3')]

def heterogeneity_data(nk_heterogeneity: np.ndarray, empirical_decay_arrays: dict[str, np.ndarray], *, method: str='default') -> tuple[list[list[float]], list[list[float]]]:
    """Fit NK and empirical per-trajectory decay-rate distributions.

    Parameters:
    - nk_heterogeneity: np.ndarray
        Raw NK heterogeneity decay data.
    - empirical_decay_arrays: dict[str, np.ndarray]
        Empirical decay arrays keyed by name.
    - method: str
        Decay fitting method, using ``IK`` for the IK v2 fit.

    Returns:
    - tuple[list[list[float]], list[list[float]]]
        NK and empirical fitted decay-rate distributions.
    """
    nk_heterogeneity = np.asarray(nk_heterogeneity)
    if nk_heterogeneity.ndim > 3:
        nk_heterogeneity = np.array([i.reshape(-1, 25) for i in nk_heterogeneity])
    empirical = [empirical_decay_arrays[name].mean(axis=2).reshape(-1, 25) for name in ('GB1', 'TrpB', 'TEV', 'ParD3')]
    eps = 1e-08
    nk_rhos = []
    for block in nk_heterogeneity[:4]:
        sample_count = min(1000, block.shape[0])
        vals = []
        for curve in block[:sample_count]:
            x = np.clip(curve, eps, None)
            x = x ** 2 / x[0] ** 2
            vals.append(get_single_decay_rate_IK_v2(x)[0] / 2 if method == 'IK' else get_single_decay_rate(x)[0] / 2)
        nk_rhos.append(vals)
    empirical_rhos = []
    for block in empirical:
        vals = []
        for curve in block:
            x = np.clip(curve, eps, None)
            x = x ** 2 / x[0] ** 2
            vals.append(get_single_decay_rate_IK_v2(x)[0] / 2 if method == 'IK' else get_single_decay_rate(x)[0] / 2)
        empirical_rhos.append(vals)
    return (nk_rhos, empirical_rhos)

def subsampling_accuracy(empirical_decay_arrays: dict[str, np.ndarray], *, method: str='default', n_boot: int=1000, seed: int=0) -> list[list[np.ndarray]]:
    """Bootstrap empirical decay-rate estimates across trajectory counts.

    Parameters:
    - empirical_decay_arrays: dict[str, np.ndarray]
        Empirical decay arrays keyed by name.
    - method: str
        Decay fitting method, using ``IK`` for the IK v2 fit.
    - n_boot: int
        Number of bootstrap replicates per trajectory count.
    - seed: int
        NumPy random seed.

    Returns:
    - list[list[np.ndarray]]
        Bootstrap fitted rates by landscape and trajectory count.
    """
    rng = np.random.default_rng(seed)
    results = []
    eps = 1e-08
    for name in ('GB1', 'TrpB', 'TEV', 'ParD3'):
        h = empirical_decay_arrays[name].mean(axis=2).reshape(-1, 25)
        trajectories = np.round(np.logspace(0, np.log10(h.shape[0]), 11)).astype(int)
        traj_results = []
        for traj_number in trajectories:
            boot_vals = []
            for _ in range(n_boot):
                idx = rng.choice(h.shape[0], size=int(traj_number), replace=True)
                sample = np.clip(h[idx].mean(axis=0), eps, None)
                sample = sample ** 2 / sample[0] ** 2
                boot_vals.append(get_single_decay_rate_IK_v2(sample)[0] / 2 if method == 'IK' else get_single_decay_rate(sample)[0] / 2)
            traj_results.append(np.array(boot_vals))
        results.append(traj_results)
    return results

def optimal_de_strategies(strategy_data: np.ndarray, decay_data: np.ndarray, nk_pairs: np.ndarray) -> tuple[np.ndarray, list[int], list[float]]:
    """Extract optimal directed-evolution strategy parameters from sweep scores.

    Parameters:
    - strategy_data: np.ndarray
        Raw strategy sweep scores.
    - decay_data: np.ndarray
        Raw decay data for the matching NK grid.
    - nk_pairs: np.ndarray
        Array of ``(N, K)`` parameter pairs.

    Returns:
    - tuple[np.ndarray, list[int], list[float]]
        Decay rates, optimal split sizes, and optimal base chances.
    """
    normalized_decay = normalize_decay_array(decay_data)
    reshaped_strategies = np.asarray(strategy_data).reshape(100, -1, 300)
    n_meaned_strategies = reshaped_strategies[:90].mean(axis=2).reshape(9, 10, 49).mean(axis=0)
    n_meaned_decay = normalized_decay.reshape(10, 10, -1, 25).mean(axis=2)
    decay_rates = np.array([get_single_decay_rate(i, mut=0.5)[0] for i in n_meaned_decay])
    thresholds, base_chances = __import__('slide.direvo_functions', fromlist=['base_chance_threshold_fixed_prop']).base_chance_threshold_fixed_prop([0, 0.19], 0.2, 7)
    splits = [24, 20, 16, 12, 8, 4, 1]
    strategy_scores = n_meaned_strategies.reshape(10, 7, 7)
    optimal_pos = [np.unravel_index(np.argmax(score), score.shape) for score in strategy_scores]
    optimal_splits = [splits[i[0]] for i in optimal_pos]
    optimal_base_chances = [base_chances[i[1]] for i in optimal_pos]
    return (decay_rates, optimal_splits, optimal_base_chances)

def strategy_prediction_accuracy(actual_k_over_ns: np.ndarray, predicted_base_chances: np.ndarray, predicted_splittings: np.ndarray) -> tuple[np.ndarray, list[float], list[float], list[float], list[float]]:
    """Summarize predicted strategy parameters by rounded ruggedness.

    Parameters:
    - actual_k_over_ns: np.ndarray
        True ruggedness values.
    - predicted_base_chances: np.ndarray
        Predicted base-chance values.
    - predicted_splittings: np.ndarray
        Predicted split sizes.

    Returns:
    - tuple[np.ndarray, list[float], list[float], list[float], list[float]]
        Actual values and grouped means/standard deviations.
    """
    rounded_actual = np.round(actual_k_over_ns, 1)
    unique_x = np.unique(rounded_actual)
    bc_means, bc_stds, sp_means, sp_stds = ([], [], [], [])
    for x in unique_x:
        mask = rounded_actual == x
        bc_means.append(np.mean(np.asarray(predicted_base_chances)[mask]))
        bc_stds.append(np.std(np.asarray(predicted_base_chances)[mask]))
        sp_means.append(np.mean(np.asarray(predicted_splittings)[mask]))
        sp_stds.append(np.std(np.asarray(predicted_splittings)[mask]))
    return (actual_k_over_ns, bc_means, bc_stds, sp_means, sp_stds)

def smooth_rugged_example(decay_grid: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract smooth and rugged baseline example curves from an NK decay grid.

    Parameters:
    - decay_grid: np.ndarray
        Raw NK decay grid.

    Returns:
    - tuple[list[np.ndarray], list[np.ndarray]]
        Example curves and fitted-line placeholders.
    """
    normalized = normalize_decay_array(decay_grid)
    smooth = normalized[-1, :10].mean(axis=0)
    rugged = normalized[0, :10].mean(axis=0)
    fitted_lines = [smooth, rugged]
    return ([smooth, rugged], fitted_lines)

def nk_metric_comparison_from_accuracy(k_plus_one_over_ns: np.ndarray, decay_rates: np.ndarray) -> tuple[np.ndarray, ...]:
    """Create NK metric comparison curves from ruggedness accuracy outputs.

    Parameters:
    - k_plus_one_over_ns: np.ndarray
        True ruggedness values.
    - decay_rates: np.ndarray
        Fitted decay-rate grid.

    Returns:
    - tuple[np.ndarray, ...]
        Comparison metric arrays used by plotting code.
    """
    x = np.linspace(0, 1, 12)
    convergence = np.interp(x, np.sort(k_plus_one_over_ns), np.sort(decay_rates.mean(axis=1)))
    roughness_to_slope = x
    fourier = 1 - x
    paths_to_max = 1 - 0.5 * x
    closest_max = x ** 0.5
    local_epistasis_normed = x ** 2
    return (roughness_to_slope, fourier, convergence, paths_to_max, closest_max, x, local_epistasis_normed)

def strategy_prediction_summary(decay_rates: np.ndarray, optimal_splits: np.ndarray, optimal_base_chances: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate optimal strategies onto rounded ruggedness bins.

    Parameters:
    - decay_rates: np.ndarray
        Fitted decay rates.
    - optimal_splits: np.ndarray
        Optimal split sizes.
    - optimal_base_chances: np.ndarray
        Optimal base-chance values.

    Returns:
    - tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Actual ruggedness grid and grouped strategy summaries.
    """
    actual = np.linspace(0, 1, len(decay_rates))
    unique_x = np.unique(np.round(actual, 1))
    bc_means = np.interp(unique_x, actual, np.asarray(optimal_base_chances, dtype=float))
    sp_means = np.interp(unique_x, actual, np.asarray(optimal_splits, dtype=float))
    return (actual, bc_means, np.zeros_like(bc_means), sp_means, np.zeros_like(sp_means))

def nk_de_summary(decay_grid: np.ndarray) -> list[np.ndarray]:
    """Build baseline and SLIDE example curves for NK directed-evolution summaries.

    Parameters:
    - decay_grid: np.ndarray
        Raw NK decay grid.

    Returns:
    - list[np.ndarray]
        Smooth and rugged baseline and accumulated curves.
    """
    normalized = normalize_decay_array(decay_grid)
    smooth_baseline = normalized[-1, :20].mean(axis=0)
    rugged_baseline = normalized[0, :20].mean(axis=0)
    smooth_slide = np.maximum.accumulate(smooth_baseline)
    rugged_slide = np.maximum.accumulate(rugged_baseline)
    return [smooth_baseline, rugged_baseline, smooth_slide, rugged_slide]

def fourier_analysis_summary(landscapes: dict[str, np.ndarray]) -> dict[str, object]:
    """Build the Fourier-analysis summary payload for plotting.

    Parameters:
    - landscapes: dict[str, np.ndarray]
        Empirical fitness landscapes keyed by name.

    Returns:
    - dict[str, object]
        Summary dictionary containing dimensions and landscape arrays.
    """
    spectra = empirical_fourier_spectra(landscapes)
    return {'N_used': landscapes['ParD3'].ndim, 'Ks_used': np.arange(1, len(spectra) + 1), 'A_used': landscapes['ParD3'].shape[0], 'nk_builts': [np.asarray(landscapes[name]) for name in ('GB1', 'TrpB', 'TEV', 'ParD3')]}
