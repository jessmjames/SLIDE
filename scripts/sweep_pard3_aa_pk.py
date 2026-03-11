"""
Sweep (p, K) combinations for ParD3 AA uniform to test whether p*K
(total simulation budget) determines the IK rho bias.

Configs:
  - Constant p*K = 600: (600,1),(300,2),(120,5),(60,10),(30,20)
  - Fixed p=60, vary K:  (60,1),(60,2),(60,5),(60,10),(60,20)
  - Fixed K=10, vary p:  (30,10),(60,10),(120,10)

For each config: generate full ParD3 decay curves, average over K seeds,
fit IK rho from all 8000 starting points (no bootstrap), compare to spectral.

Output: figures/sweep_pard3_aa_pk.pdf
"""

import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import pickle
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import tqdm

from direvo_functions import (
    build_empirical_landscape_function, build_mutation_function,
    build_selection_function, run_directed_evolution,
    get_single_decay_rate_IK_v2,
)
import selection_function_library as slct
from ruggedness_functions import get_dirichlet_metric

landscape_dir = os.path.join(parent_dir, 'landscape_arrays')
figures_dir   = os.path.join(parent_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load ParD3
# ---------------------------------------------------------------------------
with open(os.path.join(landscape_dir, 'E3_landscape_array.pkl'), 'rb') as f:
    E3 = pickle.load(f)
E3_np  = np.array(E3, dtype=np.float32)
E3_jnp = jnp.array(E3_np)

spectral_rho = get_dirichlet_metric(E3_np)
print(f"Spectral rho (AA uniform, ParD3): {spectral_rho:.4f}")

# All starting points: (8000, 3)
all_starts = np.indices(E3_np.shape).reshape(E3_np.ndim, -1).T.astype(np.int32)
total_starts, ndim = all_starts.shape

# Fixed functions
sel_params        = {'threshold': 0.0, 'base_chance': 1.0}
selection_fn      = build_selection_function(slct.base_chance_threshold_select, sel_params)
fitness_fn        = build_empirical_landscape_function(E3_jnp)
mut_fn            = build_mutation_function(0.1 / ndim, num_options=20)  # 0.1/3 per site

# ---------------------------------------------------------------------------
# Sweep configs
# ---------------------------------------------------------------------------
CONFIGS = sorted(set([
    # constant p*K = 600
    (600, 1), (300, 2), (120, 5), (60, 10), (30, 20),
    # fixed p=60, vary K
    (60, 1), (60, 2), (60, 5), (60, 20),
    # fixed K=10, vary p
    (30, 10), (120, 10),
]))

print(f"\nConfigs ({len(CONFIGS)} total):")
for p, K in CONFIGS:
    print(f"  p={p:4d}  K={K:2d}  p*K={p*K:5d}")

# ---------------------------------------------------------------------------
# Run each config
# ---------------------------------------------------------------------------

N_BOOT = 500

def run_config(p, K):
    """Return (rho_mean, rho_std) via bootstrap over starting points."""
    _r1, _r2, r3 = jr.split(jr.PRNGKey(42), 3)
    rng_seeds = jr.split(r3, K)

    def run_from_start(start):
        i_pop = jnp.broadcast_to(start[None], (int(p), ndim))
        rep_results = jax.vmap(
            lambda r: run_directed_evolution(
                r, i_pop, selection_fn, mut_fn,
                fitness_function=fitness_fn, num_steps=25
            )[1]
        )(rng_seeds)
        return rep_results['fitness'].mean(axis=-1)   # (K, 25)

    batch_size = max(1, min(200, 12000 // (p * K)))
    vmapped_run = jax.jit(jax.vmap(run_from_start))

    pad = (-total_starts) % batch_size
    padded  = np.concatenate([all_starts, all_starts[:pad]], axis=0) if pad else all_starts
    batched = padded.reshape(-1, batch_size, ndim)

    chunks = []
    for i, starts in enumerate(batched):
        res = vmapped_run(jnp.array(starts))           # (batch, K, 25)
        if pad and i == len(batched) - 1:
            res = res[:batch_size - pad]
        chunks.append(np.array(res))

    data = np.concatenate(chunks, axis=0)              # (total_starts, K, 25)
    h    = data.mean(axis=1)                           # (total_starts, 25) mean over seeds

    # Bootstrap over starting points
    rng = np.random.default_rng(0)
    boot_rhos = []
    for _ in range(N_BOOT):
        idx   = rng.choice(total_starts, size=total_starts, replace=True)
        curve = (h[idx] ** 2).mean(axis=0)
        curve = curve / curve[0]
        rho   = get_single_decay_rate_IK_v2(curve)[0] / 2
        boot_rhos.append(rho)

    return float(np.mean(boot_rhos)), float(np.std(boot_rhos))


results = {}   # (p, K) -> (rho_mean, rho_std)
for p, K in tqdm.tqdm(CONFIGS, desc='configs'):
    print(f"\n  Running p={p}, K={K}  (p*K={p*K})...", flush=True)
    rho_mean, rho_std = run_config(p, K)
    results[(p, K)] = (rho_mean, rho_std)
    bias = (rho_mean - spectral_rho) / spectral_rho * 100
    print(f"  rho={rho_mean:.4f} ± {rho_std:.4f}  bias={bias:+.1f}%")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150, sharey=True)

def plot_group(ax, configs, x_vals, x_key, xlabel, title, color, label):
    configs = sorted(configs, key=lambda c: x_key(c))
    xs    = [x_key(c) for c in configs]
    means = [results[c][0] for c in configs]
    stds  = [results[c][1] for c in configs]
    ax.errorbar(xs, means, yerr=stds, fmt='o-', color=color, capsize=4, label=label)
    ax.axhline(spectral_rho, ls='--', color='k', lw=1, label='Spectral')
    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(fontsize=8)

# --- Panel 1: constant p*K=600, x=K ---
ax = axes[0]
ax.set_ylabel(r'IK $\rho$ estimate')
fixed_pk = [(p, K) for (p, K) in CONFIGS if p * K == 600]
plot_group(ax, fixed_pk, None, lambda c: c[1], 'K (seeds per start)',
           'Fixed p·K = 600\n(vary split)', '#4477AA', 'p*K=600')
for p, K in fixed_pk:
    ax.annotate(f'p={p}', (K, results[(p,K)][0]), textcoords='offset points',
                xytext=(4, 2), fontsize=7)

# --- Panel 2: fixed p=60, vary K ---
ax = axes[1]
fixed_p = [(p, K) for (p, K) in CONFIGS if p == 60]
plot_group(ax, fixed_p, None, lambda c: c[1], 'K (seeds per start)',
           'Fixed p = 60\n(vary K)', '#EE6677', 'p=60')

# --- Panel 3: fixed K=10, vary p ---
ax = axes[2]
fixed_k = [(p, K) for (p, K) in CONFIGS if K == 10]
plot_group(ax, fixed_k, None, lambda c: c[0], 'p (population size)',
           'Fixed K = 10\n(vary p)', '#228833', 'K=10')

fig.suptitle('ParD3 AA uniform — IK rho bias vs simulation budget', fontsize=11)
plt.tight_layout()

out = os.path.join(figures_dir, 'sweep_pard3_aa_pk.pdf')
plt.savefig(out, bbox_inches='tight')
print(f"\nSaved → {out}")
