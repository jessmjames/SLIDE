# Rationalisation Checklist

Goal: every figure gets a folder in `figures_src/` with three files:
- `1_raw_data.py` — runs simulations → `SLIDE_data/`
- `2_process_data.py` — `SLIDE_data/` → `plot_data/*.pkl`
- `3_plot.py` — `plot_data/*.pkl` → `figures/*.pdf`

See `docs/figure_pipelines.md` for full pipeline details per figure.

---

## Quick fixes (do first)

- [ ] **Fig 6** — change `savefig` output in `scripts/plot_mutation_model_comparison.py` from `accuracy_over_sampling_nuc_models.pdf` to `graph_mutation_model.pdf`
- [ ] **S4** — add `savefig` calls to cells 88–89 of `ruggedness_figures_data_processing_IK.ipynb`
- [ ] **4C** — convert `repeat_heterogeneity.pdf` to a violin plot in `ruggedness_figures_plots_IK.ipynb`

---

## Phase 1: figures with existing standalone scripts

These are closest to the target structure — just need organising into folders.

- [ ] **S2** `rho_accuracy_mutation_models.pdf`
  - [ ] Merge `large_decay_curve_sweep_codon.py` and `ruggedness_accuracy_from_codon_sweeps.py` from `rebut_codon` into main
  - [ ] Create `figures_src/figure_S2_rho_accuracy/1_raw_data.py`
  - [ ] Create `figures_src/figure_S2_rho_accuracy/2_process_data.py`
  - [ ] Create `figures_src/figure_S2_rho_accuracy/3_plot.py` (from `scripts/plot_rho_accuracy_mutation_models.py`)

- [ ] **S3** `decay_curves_nuc_75steps.pdf`
  - [ ] Create `figures_src/figure_S3_decay_curves_nuc/1_raw_data.py` (from `empirical_landscape_decay_curves_codon_fast_75steps.py`)
  - [ ] Create `figures_src/figure_S3_decay_curves_nuc/2_process_data.py` (spectral rho computation)
  - [ ] Create `figures_src/figure_S3_decay_curves_nuc/3_plot.py` (from `scripts/plot_decay_curves_nuc.py`)

- [ ] **Fig 6** `graph_mutation_model.pdf`
  - [ ] Create `figures_src/figure_6_mutation_model/1_raw_data.py` (from `empirical_landscape_decay_curves_codon_fast_75steps.py` + aa variant)
  - [ ] Create `figures_src/figure_6_mutation_model/2_process_data.py` (from `compute_trajectory_subsampling.py` × 4 models)
  - [ ] Create `figures_src/figure_6_mutation_model/3_plot.py` (from `scripts/plot_mutation_model_comparison.py`)

---

## Phase 2: notebook-only figures (extract into scripts)

For each figure: identify the relevant notebook cells, trace data dependencies,
write the three scripts.

- [ ] **Fig 2A–2E** (3D landscape visualisations) — `ruggedness_figures_plots.ipynb`
- [ ] **Fig 3A–3E** (NK accuracy plots) — `ruggedness_figures_plots.ipynb`
- [ ] **Fig 4A–4B, 4D–4F** (empirical accuracy/variance) — `ruggedness_figures_plots.ipynb`
- [ ] **Fig 4C** (repeat heterogeneity — violin) — `ruggedness_figures_plots_IK.ipynb`
- [ ] **Fig 4G** (empirical ruggedness comparison IK) — `ruggedness_figures_plots_IK.ipynb`
- [ ] **Fig 5A–5G** (strategy prediction + DE) — `ruggedness_figures_plots.ipynb`
- [ ] **Fig 7** (NK spectra) — `ruggedness_figures_plots.ipynb`
- [ ] **S4** (strategy prediction over generations) — `ruggedness_figures_data_processing_IK.ipynb` cells 83–89

---

## Phase 3: figures with scattered/incomplete code

- [ ] **S1** `alt_landscapes.pdf`
  - [ ] Merge `decay_curve_sweep_alt_landscapes_*.py` and `strategy_sweep_alt_landscapes_*.py` from `rebuttals` into main
  - [ ] Extract plotting cell from `ruggedness_figures_data_processing.ipynb` (`rebuttals`)
  - [ ] Create full 3-file structure in `figures_src/figure_S1_alt_landscapes/`
