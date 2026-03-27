# Figure Pipelines

This document describes how to regenerate all figures from scratch.
All scripts are run from the repo root. `SLIDE_data/` is the external directory
resolved by `slide_config.get_slide_data_dir()`.

---

## Master Figure Map

| Figure | Output file(s) | Source | Branch | Notes |
|--------|---------------|--------|--------|-------|
| 2A | `figures/smooth_landscape_3D.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 2B | `figures/rugged_landscape_3D.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 2C | `figures/basis_vectors.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 2D | `figures/smooth_fourier_3D.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 2E | `figures/rugged_fourier_3D.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 3A | `figures/decay_curves_example.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 3B | `figures/accuracy_over_K.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 3C | `figures/accuracy_over_popsize.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 3D | `figures/accuracy_over_mut.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 3E | `figures/NK_ruggedness_metric_comparison.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 4A | `figures/empirical_fourier_spectra.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 4B | `figures/landscape_heterogeneity.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 4C | `figures/repeat_heterogeneity.pdf` | `ruggedness_figures_plots_IK.ipynb` | main | ⚠️ TODO: convert to violin plot |
| 4D | `figures/accuracy_over_sampling.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 4E | `figures/empirical_variance_over_popsize.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 4F | `figures/empirical_variance_over_generations.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 4G | `figures/empirical_ruggedness_metric_comparison_IK.pdf` | `ruggedness_figures_plots_IK.ipynb` | main | |
| 5A | `figures/strategy_prediction.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 5B | `figures/N45K1_DE_fitness.pdf`, `figures/N45K1_DE_strategy_space.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 5C | `figures/N45K25_DE_fitness.pdf`, `figures/N45K25_DE_strategy_space.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 5D | `figures/GB1_decay.pdf`, `figures/GB1_strategy_space.pdf`, `figures/GB1_DE.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 5E | `figures/TrpB_decay.pdf`, `figures/TrpB_strategy_space.pdf`, `figures/TrpB_DE.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 5F | `figures/TEV_decay.pdf`, `figures/TEV_strategy_space.pdf`, `figures/TEV_DE.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 5G | `figures/ParD3_decay.pdf`, `figures/ParD3_strategy_space.pdf`, `figures/TEV_DE.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| 6 | `figures/graph_mutation_model.pdf` | `scripts/plot_mutation_model_comparison.py` | seb_final_plots | ⚠️ `graph_mutation_model.pdf` not yet generated — script currently outputs `accuracy_over_sampling_nuc_models.pdf`. Rename/output needs implementing. |
| 7 | `figures/NK_spectra.pdf` | `ruggedness_figures_plots.ipynb` | main | |
| S1 | `figures/alt_landscapes.pdf` | `ruggedness_figures_data_processing.ipynb` | rebuttals | See detailed pipeline below |
| S2 | `figures/rho_accuracy_mutation_models.pdf` | `scripts/plot_rho_accuracy_mutation_models.py` | seb_final_plots (untracked) | See detailed pipeline below |
| S3 | `figures/decay_curves_nuc_75steps.pdf` | `scripts/plot_decay_curves_nuc.py` | seb_final_plots (untracked) | See detailed pipeline below |
| S4 | `figures/strategy_prediction_25gens.pdf`, `_50gens.pdf`, `_100gens.pdf` | `ruggedness_figures_data_processing_IK.ipynb` | main | ⚠️ Plotting code not found on any branch — data is loaded in the notebook but no savefig calls exist. Needs implementing or locating. |

---

## Long-term goal: rationalised codebase structure

Each figure should eventually have its own folder with exactly 3 files:

```
figures_src/
  figure_S2_rho_accuracy/
    1_raw_data.py       # runs simulations → SLIDE_data/
    2_process_data.py   # SLIDE_data/ → plot_data/*.pkl
    3_plot.py           # plot_data/*.pkl → figures/*.pdf
```

Where multiple figures share raw data, `1_raw_data.py` should clearly note
which other figures depend on it.

---

## Figures 2A–2E, 3A–3E, 4A–4B, 4D–4F, 5A–5G, 7

**Source:** `ruggedness_figures_plots.ipynb` on `main`

All generated in a single notebook. No standalone scripts yet.
Input data requirements TBD when rationalising into per-figure folders.

---

## Figures 4C, 4G

**Source:** `ruggedness_figures_plots_IK.ipynb` on `main`

- **4C** (`repeat_heterogeneity.pdf`) — ⚠️ needs converting to a violin plot
- **4G** (`empirical_ruggedness_metric_comparison_IK.pdf`)

---

## Figure S4

**Output:** `figures/strategy_prediction_25gens.pdf`, `_50gens.pdf`, `_100gens.pdf`
**Source:** `ruggedness_figures_data_processing_IK.ipynb` on `main`
❓ Confirm — plotting may actually live in `ruggedness_figures_plots_IK.ipynb` instead.

---

## Figure S1: `alt_landscapes.pdf`

**Source:** `ruggedness_figures_data_processing.ipynb` on `rebuttals` branch

Compares NK, Rough Mount Fuji, and Stochastic Block landscape models at
low/mid/high ruggedness levels, with strategy space plots and decay curves for each.

### Upstream simulation scripts (all on `rebuttals`/`even_messier`):

```bash
python scripts/decay_curve_sweep_alt_landscapes_NK.py
python scripts/decay_curve_sweep_alt_landscapes_RMF.py
python scripts/decay_curve_sweep_alt_landscapes_block.py
python scripts/strategy_sweep_alt_landscapes_NK.py
python scripts/strategy_sweep_alt_landscapes_RMF.py
python scripts/strategy_sweep_alt_landscapes_block.py
```

Produces in `SLIDE_data/`:
- `landscape_comparsion_strategy_sweep_{NK,RMF,block}.pkl`
- `alt_landscapes_decay_curve_sweep_{NK,RMF,blocks}.pkl`

### Plot

Run the relevant cell in `ruggedness_figures_data_processing.ipynb` (on `rebuttals`):
```python
plt.savefig('figures/alt_landscapes.pdf', dpi=300)
```

---

## Figure S4: `strategy_prediction_{g}gens.pdf`

**Output:** `figures/strategy_prediction_25gens.pdf`, `_50gens.pdf`, `_100gens.pdf`

⚠️ Plotting code not found on any branch. The generation-varying strategy data
is loaded in `ruggedness_figures_data_processing_IK.ipynb` (main) but no savefig
calls exist yet. Needs implementing.

---

## Figure 6: `graph_mutation_model.pdf`

**Script:** `scripts/plot_mutation_model_comparison.py` (`seb_final_plots` working tree)

⚠️ The script currently saves as `accuracy_over_sampling_nuc_models.pdf`.
`graph_mutation_model` appears in the script only as a comment for panel sizing.
The output filename in the script needs updating to `graph_mutation_model.pdf`.

The related figure `accuracy_over_sampling_nuc_models_75steps.pdf` shares the
same upstream data but is a different panel layout (nuc models only, 75 steps).

**Pipeline** (shared with S3 — see S3 below for full simulation detail):

```bash
# 1. Run nuc simulations
python scripts/empirical_landscape_decay_curves_codon_fast_75steps.py

# 2. Bootstrap subsampling for each model
python scripts/compute_trajectory_subsampling.py nuc_uniform       --steps 75
python scripts/compute_trajectory_subsampling.py nuc_h_sapiens_sym --steps 75
python scripts/compute_trajectory_subsampling.py nuc_e_coli        --steps 75
python scripts/compute_trajectory_subsampling.py aa_uniform        --steps 75

# 3. Spectral rho reference
python scripts/compute_spectral_rho_models.py

# 4. Plot
python scripts/plot_mutation_model_comparison.py --steps 75
```

---

## Figure S2: `rho_accuracy_mutation_models.pdf`

**Script:** `scripts/plot_rho_accuracy_mutation_models.py` (untracked; working tree only)

Reads `plot_data/ruggedness_accuracy_codon_{e_coli,a_thaliana,human}.pkl`.
These pkls already exist locally (generated Feb 2026). To regenerate from scratch:

### Step 1 — NK landscape sweep → `SLIDE_data/large_decay_curve_sweep_codon_{label}.pkl`

```bash
git checkout rebut_codon -- scripts/large_decay_curve_sweep_codon.py
python scripts/large_decay_curve_sweep_codon.py
```

Sweeps NK landscapes (N∈[10,50], K∈[1,N], 100 NK combos × 250 reps) under
codon-mapped mutation models for `e_coli`, `a_thaliana`, `human`.

⚠️ Script only exists on `rebut_codon`/`even_messier` — never merged to `main`.

### Step 2 — Fit rho accuracy → `plot_data/ruggedness_accuracy_codon_{label}.pkl`

```bash
git checkout rebut_codon -- scripts/ruggedness_accuracy_from_codon_sweeps.py
python scripts/ruggedness_accuracy_from_codon_sweeps.py
```

⚠️ Same — only on `rebut_codon`/`even_messier`.

### Step 3 — Plot

```bash
python scripts/plot_rho_accuracy_mutation_models.py
```

---

## Figure S3: `decay_curves_nuc_75steps.pdf`

**Script:** `scripts/plot_decay_curves_nuc.py` (untracked; working tree only)

### Step 1 — Nuc-space simulations → `SLIDE_data/`

```bash
python scripts/empirical_landscape_decay_curves_codon_fast_75steps.py
```

Produces for each of `{gb1, trpb, tev, pard3}` × `{nuc_uniform, nuc_h_sapiens_sym, nuc_h_sapiens, nuc_e_coli}`:
```
SLIDE_data/decay_curves_{landscape}_{model}_m0.1_all_starts_75steps.pkl
```
Shape `(-1, 100, 10, 75)` — all AA starting positions in codon space, pop=2500, 10 seeds, 75 steps.

### Step 2 — Spectral rho reference → `plot_data/spectral_rho_comparison.pkl`

```bash
python scripts/compute_spectral_rho_models.py
```

### Step 3 — Plot

```bash
python scripts/plot_decay_curves_nuc.py
```

Caches `plot_data/true_constants_nuc.pkl` on first run (avoids repeating 64^4 computation).

---

## Static input data

| File | Location | Used by |
|------|----------|---------|
| GB1 landscape | `landscape_arrays/GB1_landscape_array.pkl` | S2, S3, Fig 6 |
| TrpB landscape | `landscape_arrays/TrpB_landscape_array.pkl` | S2, S3, Fig 6 |
| TEV landscape | `landscape_arrays/TEV_landscape_array.pkl` | S2, S3, Fig 6 |
| ParD3 landscape | `landscape_arrays/E3_landscape_array.pkl` | S2, S3, Fig 6 |
| H. sapiens mutation matrix | `other_data/normed_h_sapiens_matrix.npy` | S3, Fig 6 |
| E. coli mutation matrix | `other_data/normed_e_coli_matrix.npy` | S2, S3, Fig 6 |
| A. thaliana mutation matrix | `other_data/normed_a_thaliana_matrix.npy` | S2 |
