"""
Figure 3E — Step 2: build NK landscapes, compute ruggedness metrics, normalise.

For N=12 and K in 0..11, generates 50 random NK landscape replicates per K
value and computes 7 ruggedness metrics for each:

  1. Roughness-to-slope ratio
  2. Fourier (landscape R²)
  3. Convergence rate (decay rate of directed-evolution trajectories)
  4. Mean accessible paths to global maximum
  5. Inverse distance to closest local maximum (transformed: 1 - 1/mean)
  6. k_over_ns  (= (K+1)/N, used as the x-axis in the figure)
  7. Local epistasis (simple-sign + reciprocal-sign counts)

All per-K means are min-max normalised to [0, 1] before saving.

WARNING: This script is CPU/GPU intensive.  Building 12 × 50 = 600 NK
landscapes analytically and running directed-evolution simulations for the
convergence-rate metric takes several minutes (expect 10–30 min depending on
hardware and JAX backend).

Input
-----
  None — all landscapes are generated analytically from fixed random seeds.

Output
------
  plot_data/NK_ruggedness_metric_comparison.pkl
    Tuple:
      (NK_roughness_to_slope, NK_fourier, convergence_rates,
       NK_paths_to_max, NK_closest_max, k_over_ns, NK_le_normed)
    All arrays shape (12,) — one normalised value per K in [0, 11].

Originally extracted from cells 26–31 of
ruggedness_figures_data_processing.ipynb.
"""

import os
import sys
import pickle
import numpy as np
import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

import jax.random as jr
from ruggedness_functions import (
    get_nk_l_o_shape,
    roughness_to_slope,
    landscape_r2,
    get_mean_paths_to_max,
    find_distance_to_closest_max,
    local_epistasis,
    get_convergence_rate,
)

plot_data_dir = os.path.join(parent_dir, 'plot_data')
os.makedirs(plot_data_dir, exist_ok=True)

out = os.path.join(plot_data_dir, 'NK_ruggedness_metric_comparison.pkl')

if os.path.exists(out):
    print(f'Already exists: {out}')
    sys.exit(0)

# ---------------------------------------------------------------------------
# Cell 26 — build landscapes and compute per-replicate metrics
# ---------------------------------------------------------------------------

test_NKs = np.array([
    [12, 0], [12, 1], [12, 2],  [12, 3],  [12, 4],  [12, 5],
    [12, 6], [12, 7], [12, 8],  [12, 9],  [12, 10], [12, 11],
])
shape = (2,) * int(test_NKs[0][0])

NK_roughness_to_slope = []
NK_fourier = []
convergence_rates = []
convergence_rates_extra = []
NK_paths_to_max = []
NK_closest_max = []
NK_local_epistasis = []

rng = jr.PRNGKey(42)
rng_list = jr.split(rng, 50)

for N, K in test_NKs:
    print(f'K={K}')
    for r in tqdm.tqdm(rng_list):
        complete_landscape = get_nk_l_o_shape(r, N, K, shape)
        NK_roughness_to_slope.append(roughness_to_slope(complete_landscape))
        NK_fourier.append(landscape_r2(complete_landscape))
        NK_paths_to_max.append(get_mean_paths_to_max(complete_landscape, norm=False, extra_slack=0))
        NK_closest_max.append(1 / find_distance_to_closest_max(complete_landscape))
        NK_local_epistasis.append(local_epistasis(complete_landscape, [0] * 12))
        cr, extr = get_convergence_rate(r, N, K, num_reps=100)
        convergence_rates.append(cr)
        convergence_rates_extra.append(extr)

# ---------------------------------------------------------------------------
# Cell 27 — reshape to (12, 50)
# ---------------------------------------------------------------------------

NK_roughness_to_slope_r = np.array(NK_roughness_to_slope).reshape(12, 50)
NK_fourier_r             = np.array(NK_fourier).reshape(12, 50)
convergence_rates_r      = np.array(convergence_rates).reshape(12, 50)
NK_paths_to_max_r        = np.array(NK_paths_to_max).reshape(12, 50)
NK_closest_max_r         = np.array(NK_closest_max).reshape(12, 50)
NK_local_epistasis_r     = np.array([
    i['simple_sign_episasis'] + i['reciprocal_sign_epistasis']
    for i in NK_local_epistasis
]).reshape(12, 50)

# ---------------------------------------------------------------------------
# Cell 28 — average over 50 replicates per K
# ---------------------------------------------------------------------------

mean_NK_roughness_to_slope = [np.array(i).mean() for i in NK_roughness_to_slope_r]
mean_NK_fourier            = [np.array(i).mean() for i in NK_fourier_r]
mean_convergence_rates     = [np.array(i).mean() for i in convergence_rates_r]
mean_NK_paths_to_max       = [np.array(i).mean() for i in NK_paths_to_max_r]
mean_NK_closest_max        = [np.array(i).mean() for i in NK_closest_max_r]
mean_NK_local_epistasis    = [np.array(i).mean() for i in NK_local_epistasis_r]

# ---------------------------------------------------------------------------
# Cell 29 — inverse transform for closest-max metric
# ---------------------------------------------------------------------------

mean_NK_closest_max = 1 - 1 / np.array(mean_NK_closest_max)

# ---------------------------------------------------------------------------
# Cell 30 — min-max normalise all metrics
# ---------------------------------------------------------------------------

def norm_data(data):
    return np.array((data - min(data)) / (max(data) - min(data)))

mean_NK_roughness_to_slope = norm_data(np.array(mean_NK_roughness_to_slope))
mean_NK_fourier            = norm_data(np.array(mean_NK_fourier))
mean_NK_convergence_rates  = norm_data(np.array(mean_convergence_rates))
mean_NK_paths_to_max       = norm_data(np.array(mean_NK_paths_to_max))
mean_NK_closest_max        = norm_data(np.array(mean_NK_closest_max))
mean_NK_le_normed          = norm_data(np.array(mean_NK_local_epistasis))

k_over_ns = (test_NKs[:, 1] + 1) / test_NKs[:, 0]

# ---------------------------------------------------------------------------
# Cell 31 — save
# ---------------------------------------------------------------------------

with open(out, 'wb') as f:
    pickle.dump((
        mean_NK_roughness_to_slope,
        mean_NK_fourier,
        mean_NK_convergence_rates,
        mean_NK_paths_to_max,
        mean_NK_closest_max,
        k_over_ns,
        mean_NK_le_normed,
    ), f)

print(f'Saved → {out}')
