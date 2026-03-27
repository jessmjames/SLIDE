# SLIDE — Project Notes for Claude

## Figure Generation Pipelines

This document describes how to regenerate the key figures from scratch.
All scripts are run from the repo root unless stated otherwise.
`SLIDE_data/` refers to the external data directory resolved by `slide_config.get_slide_data_dir()`.

---

## `figures/decay_curves_nuc_75steps.pdf`

**Plotting script:** `scripts/plot_decay_curves_nuc.py` (untracked; working tree only)

```bash
python scripts/plot_decay_curves_nuc.py
```

### Step 1 — Run nucleotide-space simulations → `SLIDE_data/`

```bash
python scripts/empirical_landscape_decay_curves_codon_fast_75steps.py
```

Produces for each landscape in `{gb1, trpb, tev, pard3}` and each model in
`{nuc_uniform, nuc_h_sapiens_sym, nuc_h_sapiens, nuc_e_coli}`:

```
SLIDE_data/decay_curves_{landscape}_{model}_m0.1_all_starts_75steps.pkl
```

Shape: `(-1, 100, 10, 75)` — all AA starting positions converted to codon space,
pop=2500, 10 random seeds, 75 steps.

Uses:
- `landscape_arrays/{GB1,TrpB,TEV,E3}_landscape_array.pkl`
- `other_data/normed_h_sapiens_matrix.npy`, `other_data/normed_e_coli_matrix.npy`

### Step 2 — Compute spectral rho reference → `plot_data/spectral_rho_comparison.pkl`

```bash
python scripts/compute_spectral_rho_models.py
```

### Step 3 — Plot

```bash
python scripts/plot_decay_curves_nuc.py
```

Reads the decay curve pkls from `SLIDE_data/`, `plot_data/spectral_rho_comparison.pkl`,
landscape arrays, and `other_data/normed_e_coli_matrix.npy`.
Caches `plot_data/true_constants_nuc.pkl` on first run (avoids repeating 64^4 computation).

---

## `figures/accuracy_over_sampling_nuc_models_75steps.pdf`

**Plotting script:** `scripts/plot_mutation_model_comparison.py` (modified on `seb_final_plots`; `--steps 75` flag added in working tree)

```bash
python scripts/plot_mutation_model_comparison.py --steps 75
```

### Step 1 — Run simulations

Nuc models (same output as Figure 1 above — skip if already done):
```bash
python scripts/empirical_landscape_decay_curves_codon_fast_75steps.py
```

AA-uniform baseline:
```bash
python scripts/empirical_landscape_decay_curves_all_starts_fast_75steps.py
```

Produces:
```
SLIDE_data/decay_curves_{landscape}_aa_uniform_m0.1_all_starts_75steps.pkl
```

### Step 2 — Bootstrap subsampling → `plot_data/trajectory_subsampling_{model}_75steps.pkl`

Run for each mutation model:
```bash
python scripts/compute_trajectory_subsampling.py nuc_uniform      --steps 75
python scripts/compute_trajectory_subsampling.py nuc_h_sapiens_sym --steps 75
python scripts/compute_trajectory_subsampling.py nuc_e_coli        --steps 75
python scripts/compute_trajectory_subsampling.py aa_uniform        --steps 75
```

Each script reads the 4 landscape decay-curve pkls from `SLIDE_data/`, bootstrap-resamples
trajectory subsets (N_BOOT=1000), fits IK decay rates via `get_single_decay_rate_IK_v2`,
and saves the result.

### Step 3 — Spectral rho reference (same as Figure 1 Step 2 — skip if done):

```bash
python scripts/compute_spectral_rho_models.py
```

### Step 4 — Plot

```bash
python scripts/plot_mutation_model_comparison.py --steps 75
```

Also produces: `accuracy_over_sampling_aa_uniform_75steps.pdf`,
`accuracy_panel_{d,e,f}_75steps.pdf`, `accuracy_panel_legend_75steps.pdf`.

---

## `figures/rho_accuracy_mutation_models.pdf`

**Plotting script:** `scripts/plot_rho_accuracy_mutation_models.py` (untracked; working tree only)

```bash
python scripts/plot_rho_accuracy_mutation_models.py
```

Reads:
```
plot_data/ruggedness_accuracy_codon_{e_coli,a_thaliana,human}.pkl
```

These pkls already exist locally (generated Feb 2026). To regenerate from scratch,
follow Steps 1–2 below. **Both scripts only exist on the `rebut_codon`/`even_messier`
branches — they were never merged into `main` or `seb_final_plots`.**

### Step 1 — NK landscape sweep → `SLIDE_data/large_decay_curve_sweep_codon_{label}.pkl`

Recover and run:
```bash
git checkout rebut_codon -- scripts/large_decay_curve_sweep_codon.py
python scripts/large_decay_curve_sweep_codon.py
```

Sweeps NK landscapes (N∈[10,50], K∈[1,N], 100 NK combos × 250 reps) under codon-mapped
mutation models for `e_coli`, `a_thaliana`, and `human`.

### Step 2 — Fit rho accuracy → `plot_data/ruggedness_accuracy_codon_{label}.pkl`

Recover and run:
```bash
git checkout rebut_codon -- scripts/ruggedness_accuracy_from_codon_sweeps.py
python scripts/ruggedness_accuracy_from_codon_sweeps.py
```

Reads the sweep pkls, fits decay rates via `get_single_decay_rate`, computes true
`(K+1)/N` for each NK combo. Each output pkl contains `(k_plus_one_over_ns, decay_rates)`
over 100 NK combos.

### Step 3 — Plot

```bash
python scripts/plot_rho_accuracy_mutation_models.py
```

---

## Static input data locations

| File | Location | Used by |
|------|----------|---------|
| GB1 landscape | `landscape_arrays/GB1_landscape_array.pkl` | all figures |
| TrpB landscape | `landscape_arrays/TrpB_landscape_array.pkl` | all figures |
| TEV landscape | `landscape_arrays/TEV_landscape_array.pkl` | all figures |
| ParD3 landscape | `landscape_arrays/E3_landscape_array.pkl` | all figures |
| H. sapiens mutation matrix | `other_data/normed_h_sapiens_matrix.npy` | Figs 1, 2 |
| E. coli mutation matrix | `other_data/normed_e_coli_matrix.npy` | Figs 1, 2, 3 |
| A. thaliana mutation matrix | `other_data/normed_a_thaliana_matrix.npy` | Fig 3 |
